//! `org.apache.lucene.monitor.MonitorQuerySerializer`: a `MonitorQuery` as the
//! bytes a persistent query index stores, and back.
//!
//! The layout is `MonitorQuerySerializer.fromParser`'s: the id and the query
//! string as Lucene strings (a vint byte length, then UTF-8), then the
//! metadata entry count as a little-endian int (`DataOutput.writeInt`) and each key and value as
//! strings. The query itself is not stored: it is parsed again from its
//! string by the parser the serializer was built with.

use std::collections::BTreeMap;
use std::sync::Arc;

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::monitor::index::MonitorQuery;
use crate::query::Clause;

/// The parser a serializer rebuilds queries with: a query string to a query,
/// or the parser's error text.
pub type QueryParserFn = Arc<dyn Fn(&str) -> Result<Clause, String> + Send + Sync>;

/// `MonitorQuerySerializer`.
#[derive(Clone)]
pub struct MonitorQuerySerializer {
    parser: QueryParserFn,
}

impl std::fmt::Debug for MonitorQuerySerializer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MonitorQuerySerializer")
    }
}

impl MonitorQuerySerializer {
    /// `MonitorQuerySerializer.fromParser(parser)`.
    pub fn from_parser(parser: QueryParserFn) -> Self {
        MonitorQuerySerializer { parser }
    }

    /// `serialize(query)`. A query with no string form cannot be stored:
    /// Java's `writeString(null)` throws.
    pub fn serialize(&self, query: &MonitorQuery) -> Result<Vec<u8>, String> {
        let Some(text) = &query.query_string else {
            return Err(
                "java.lang.NullPointerException: a MonitorQuery needs a query string to be serialized"
                    .into(),
            );
        };
        let mut out = Vec::new();
        out.write_string(&query.id);
        out.write_string(text);
        let count = i32::try_from(query.metadata.len()).map_err(|_| "too much metadata")?;
        out.write_i32(count);
        for (k, v) in &query.metadata {
            out.write_string(k);
            out.write_string(v);
        }
        Ok(out)
    }

    /// `deserialize(binaryValue)`: the query re-parsed from its string.
    pub fn deserialize(&self, bytes: &[u8]) -> Result<MonitorQuery, String> {
        let corrupt = |e: lucene_store::Error| e.to_string();
        let mut input = SliceInput::new(bytes);
        let id = input.read_string().map_err(corrupt)?;
        let text = input.read_string().map_err(corrupt)?;
        let count = input.read_i32().map_err(corrupt)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..count.max(0) {
            let k = input.read_string().map_err(corrupt)?;
            let v = input.read_string().map_err(corrupt)?;
            metadata.insert(k, v);
        }
        let query = (self.parser)(&text)?;
        Ok(MonitorQuery {
            id,
            query,
            query_string: Some(text),
            metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serializer() -> MonitorQuerySerializer {
        MonitorQuerySerializer::from_parser(Arc::new(|t: &str| {
            if t == "bad" {
                Err("cannot parse".to_string())
            } else {
                Ok(Clause::Term(crate::query::TermQuery::new(
                    "f",
                    t.as_bytes().to_vec(),
                )))
            }
        }))
    }

    fn mq(text: Option<&str>) -> MonitorQuery {
        MonitorQuery {
            id: "i".into(),
            query: Clause::Term(crate::query::TermQuery::new("f", b"x".to_vec())),
            query_string: text.map(str::to_string),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn a_query_without_text_cannot_be_serialized() {
        let e = serializer().serialize(&mq(None)).unwrap_err();
        assert!(e.starts_with("java.lang.NullPointerException"), "{e}");
    }

    #[test]
    fn truncated_bytes_and_unparsable_text_are_errors() {
        let ser = serializer();
        let good = ser.serialize(&mq(Some("x"))).unwrap();
        for cut in 0..good.len() {
            assert!(ser.deserialize(&good[..cut]).is_err(), "cut {cut}");
        }
        let bad = ser.serialize(&mq(Some("bad"))).unwrap();
        assert_eq!(ser.deserialize(&bad).unwrap_err(), "cannot parse");
        assert!(format!("{ser:?}").contains("MonitorQuerySerializer"));
    }
}
