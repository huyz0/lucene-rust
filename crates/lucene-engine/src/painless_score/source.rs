//! A script given as Painless source: compiled here, by the Rust front end,
//! into the script the Java front end would have written.

use lucene_painless::front::source::{compile, doc_fields, reads_score, Refused};
use lucene_painless::ir::{self, Constant};

use super::{FieldKind, FieldSpec, Spec};
use crate::nodes::invalid;
use crate::status::Status;

/// Why source is not a script this engine runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceError {
    /// Painless rejects it: the message and where.
    Compile { offset: usize, message: String },
    /// A construct the front end does not serve yet.
    Unsupported(String),
    /// The runtime does not run it exactly as Painless would: it is
    /// [`Spec::check`]'s refusal.
    Refused(String),
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceError::Compile { offset, message } => write!(f, "{message} at offset {offset}"),
            SourceError::Unsupported(what) | SourceError::Refused(what) => f.write_str(what),
        }
    }
}

impl From<Refused> for SourceError {
    fn from(r: Refused) -> Self {
        match r {
            Refused::Compile { offset, message } => SourceError::Compile { offset, message },
            Refused::Unsupported(what) => SourceError::Unsupported(what),
        }
    }
}

/// The fields `source` reads by a constant name as a script of engine
/// context `context`: the ones a caller maps to [`FieldSpec`]s.
///
/// # Errors
/// [`SourceError`] when the source is not a script.
pub fn fields_of(source: &str, context: u8) -> Result<Vec<String>, SourceError> {
    Ok(doc_fields(&compile(source, context)?))
}

/// The script `source` is, for a context whose kind byte is `kind` (`0` and
/// `1` score, `2` filter, `3` and `4` number sort, `5` aggregation, `6`
/// `terms_set`), and whether it reads `_score`.
pub(super) fn script(source: &str, kind: u8) -> Result<(ir::Script, bool), SourceError> {
    let script = compile(source, kind)?;
    let score = reads_score(&script);
    Ok((script, score))
}

/// The script of a blob's source bytes, and whether it reads `_score`.
pub(super) fn decode(bytes: &[u8], kind: u8) -> Result<(ir::Script, bool), Status> {
    let source = std::str::from_utf8(bytes)
        .map_err(|_| invalid("query tree: painless source is not UTF-8"))?;
    script(source, kind).map_err(|e| invalid(format!("query tree: painless source: {e}")))
}

/// A spec of `source` as a script of `kind`, ready to run, or why it
/// cannot: not Painless, not served yet, or one [`Spec::check`] refuses.
///
/// # Errors
/// [`SourceError`] as above.
pub fn spec(
    kind: u8,
    source: &str,
    params: Constant,
    index: &str,
    shard: i32,
    fields: Vec<(String, String, FieldKind)>,
) -> Result<Spec, SourceError> {
    let spec = Spec::of_source(kind, source, params, index, shard, fields)?;
    spec.check().map_err(|r| SourceError::Refused(r.0))?;
    Ok(spec)
}

impl Spec {
    /// The spec of `source`, not yet checked.
    fn of_source(
        kind: u8,
        source: &str,
        params: Constant,
        index: &str,
        shard: i32,
        fields: Vec<(String, String, FieldKind)>,
    ) -> Result<Spec, SourceError> {
        let (script, needs_score) = script(source, kind)?;
        let params_blob = ir::encode_constant(&params);
        let (mut sort, mut agg, mut terms_set) = (false, false, false);
        match kind {
            3 | 4 => sort = true,
            5 => agg = true,
            6 => terms_set = true,
            _ => {}
        }
        Ok(Spec {
            script,
            ir_blob: source.as_bytes().to_vec(), // alloc-ok: the source, at most 65,535 bytes
            params,
            params_blob,
            needs_score: needs_score && matches!(kind, 0 | 1 | 3 | 4),
            filter: kind == 2,
            sort,
            agg,
            metric_map: kind == 11,
            terms_set,
            interval: false,
            similarity: false,
            similarity_weight: false,
            index: index.to_string(),
            shard,
            script_id: 0,
            script_id_at: 0,
            fields: fields
                .into_iter()
                .map(|(name, read, kind)| FieldSpec { name, read, kind })
                .collect(),
            js: None,
            expr: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_blob::Cursor;

    fn put(out: &mut Vec<u8>, b: &[u8]) {
        out.extend_from_slice(&(b.len() as i32).to_le_bytes());
        out.extend_from_slice(b);
    }

    /// The blob of a source script over one `long` field `price`.
    fn blob(kind: u8, source: &str) -> Vec<u8> {
        let mut out = vec![super::super::SOURCE_SCRIPT | kind];
        put(&mut out, source.as_bytes());
        put(&mut out, &ir::encode_constant(&Constant::Map(vec![])));
        put(&mut out, b"idx");
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&1i32.to_le_bytes());
        put(&mut out, b"price");
        put(&mut out, b"price");
        out.push(0);
        out
    }

    #[test]
    fn a_source_script_decodes_checks_and_says_whether_it_reads_the_score() {
        let s = super::super::decode_spec(&mut Cursor::new(&blob(0, "doc['price'].value * 2")))
            .expect("decodes");
        assert!(!s.needs_score && s.is_score());
        assert!(s.check().is_ok(), "{:?}", s.check());
        let s =
            super::super::decode_spec(&mut Cursor::new(&blob(0, "_score + doc['price'].value")))
                .expect("decodes");
        assert!(s.needs_score);
        let f = super::super::decode_spec(&mut Cursor::new(&blob(2, "doc['price'].value > 3")))
            .expect("decodes");
        assert!(f.filter);
        // Source Painless rejects, or one the front end does not serve.
        assert!(super::super::decode_spec(&mut Cursor::new(&blob(0, "int x = 'a';"))).is_err());
        assert!(super::super::decode_spec(&mut Cursor::new(&blob(0, "1 +"))).is_err());
    }

    #[test]
    fn the_fields_and_the_refusals_come_back_with_their_reasons() {
        assert_eq!(
            fields_of("doc['a'].value + doc['b'].value", 0).unwrap(),
            ["a", "b"]
        );
        assert!(matches!(
            fields_of("int x = 'a';", 0),
            Err(SourceError::Compile { .. })
        ));
        let f = || vec![("price".to_string(), "price".to_string(), FieldKind::Long)];
        let ok = spec(
            0,
            "doc['price'].value + 1",
            Constant::Map(vec![]),
            "i",
            0,
            f(),
        );
        assert!(ok.is_ok(), "{ok:?}");
        // A field the script reads that the index does not map has no reading.
        let refused = spec(
            0,
            "doc['nope'].value",
            Constant::Map(vec![]),
            "i",
            0,
            vec![("nope".into(), "nope".into(), FieldKind::Unsupported)],
        );
        assert!(
            matches!(refused, Err(SourceError::Refused(_))),
            "{refused:?}"
        );
    }
}
