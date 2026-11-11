//! The plugin's `max_sim` query over a `multi_vector` field, as node `44` of
//! the query tree ([`crate::jvm_queries`]).
//!
//! The sub-query's documents are scored by the `MaxSim` of the query's token
//! vectors against the field's stored multi-vector
//! ([`lucene_tensor::MaxSimValuesSource`], any of its encodings), through the
//! ported Lucene [`FunctionScoreQuery`]: a document without a value, or with a
//! negative `MaxSim`, scores 0. The plugin's Java `MaxSimQuery` scores the
//! same way.
//!
//! The node, after its kind byte:
//!
//! | field | layout |
//! |---|---|
//! | sub-query | a node |
//! | `field` | string |
//! | `dim` | `i32`, at least 1 |
//! | tokens | `i32` count, at least 1, then `count * dim` `f32` |

use std::sync::Arc;

use lucene_search::extended_query::ExtendedQuery;
use lucene_search::function::FunctionScoreQuery;
use lucene_search::query::Clause;
use lucene_tensor::{MaxSimQuery, MaxSimValuesSource};

use crate::error::FfiStatus;
use crate::jvm_nodes::invalid;
use crate::jvm_reader::Cursor;

pub(crate) const NODE_MAX_SIM: u8 = 44;

pub(crate) fn decode(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> std::result::Result<Clause, FfiStatus> {
    let sub = crate::jvm_reader::decode_node(c, depth + 1, nodes)?;
    let field = std::str::from_utf8(c.bytes()?)
        .map_err(|_| FfiStatus::InvalidUtf8)?
        .to_string();
    let dim = c.len()?;
    let count = c.len()?;
    if dim == 0 || count == 0 {
        return Err(invalid("query tree: max_sim without tokens"));
    }
    let floats = dim
        .checked_mul(count)
        .filter(|&n| n <= c.remaining() / 4)
        .ok_or_else(|| invalid("query tree: max_sim tokens past the blob"))?;
    let mut tokens = Vec::with_capacity(count); // alloc-ok: bounded by the blob
    let mut token = Vec::with_capacity(dim); // alloc-ok: bounded by the blob
    for _ in 0..floats {
        token.push(f32::from_bits(c.i32()? as u32));
        if token.len() == dim {
            tokens.push(std::mem::take(&mut token));
        }
    }
    let query = MaxSimQuery::new(&tokens).map_err(|e| invalid(format!("query tree: {e}")))?;
    let source = MaxSimValuesSource::new(&field, query);
    Ok(Clause::Extended(Box::new(ExtendedQuery::FunctionScore(
        FunctionScoreQuery::new(sub, Arc::new(source)),
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_nodes::tests::Blob;

    fn decoded(blob: &Blob) -> std::result::Result<Clause, FfiStatus> {
        let mut c = Cursor::new(&blob.0);
        let mut nodes = 0;
        crate::jvm_reader::decode_node(&mut c, 0, &mut nodes)
    }

    fn node(dim: i32, count: i32, floats: &[f32]) -> Blob {
        let mut b = Blob::default()
            .u8(NODE_MAX_SIM)
            .term("f", "x")
            .str("mv")
            .i32(dim)
            .i32(count);
        for &x in floats {
            b = b.f32(x);
        }
        b
    }

    #[test]
    fn scores_every_encoding_as_max_sim_and_missing_values_as_zero() {
        use crate::jvm_nodes::tests::{open, search};
        use crate::jvm_reader::ffi_close_jvm_reader;
        use lucene_index::document::{BinaryDocValuesField, Document, Store, StringField};
        use lucene_index::index_writer::IndexWriter;
        use lucene_store::FsDirectory;
        use lucene_tensor::{encode, TokenEncoding};
        use lucene_util::test_support::TempDir;

        const DIM: usize = 4;
        let encodings = [
            ("f32", TokenEncoding::Float32),
            ("f16", TokenEncoding::Float16),
            ("i8", TokenEncoding::Int8),
        ];
        let token = |seed: usize| -> Vec<f32> {
            (0..DIM)
                .map(|k| (((seed * 7 + k * 13) % 17) as f32 - 8.0) / 8.0)
                .collect()
        };
        let tmp = TempDir::new("max_sim_node");
        let dir = FsDirectory::open(tmp.path());
        let version = lucene_index::segment_info::LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        };
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", version).unwrap();
        let mut values = Vec::new();
        for i in 0..30usize {
            let mut d = Document::new();
            d.add(StringField::new("id", i.to_string(), Store::Yes));
            let tokens: Vec<f32> = (0..1 + i % 5).flat_map(|t| token(i * 5 + t)).collect();
            let mut per = Vec::new();
            if i % 7 != 3 {
                for (name, enc) in encodings {
                    let v = encode(&tokens, DIM, enc).unwrap();
                    d.add(BinaryDocValuesField::new(name, v.clone()));
                    per.push(Some(v));
                }
            } else {
                per = vec![None, None, None];
            }
            values.push(per);
            w.add_fields_documents(&[d]).unwrap();
            if i == 13 {
                w.commit().unwrap();
            }
        }
        w.commit().unwrap();
        let h = open(&tmp);
        let q = [token(3), token(11)];
        let query = MaxSimQuery::new(&q).unwrap();
        for (e, (name, _)) in encodings.iter().enumerate() {
            let mut want: Vec<(i32, f32)> = values
                .iter()
                .enumerate()
                .map(|(doc, v)| {
                    let s = v[e]
                        .as_ref()
                        .map_or(0.0, |v| query.score(v, &mut Vec::new()).unwrap().max(0.0));
                    (doc as i32, s)
                })
                .collect();
            want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            want.truncate(20);
            let mut blob = Blob::tree()
                .u8(NODE_MAX_SIM)
                .u8(5)
                .str(name)
                .i32(DIM as i32)
                .i32(2);
            for x in q.iter().flatten() {
                blob = blob.f32(*x);
            }
            let (hits, total) = search(h, &blob);
            assert_eq!(total, 30, "{name}");
            assert_eq!(hits, want, "{name}");
        }
        // the stock rescore over a window of every document, exists as the sub-query
        for (e, (name, _)) in encodings.iter().enumerate() {
            let mut blob = Blob::tree()
                .u8(NODE_MAX_SIM)
                .exists(name)
                .str(name)
                .i32(DIM as i32)
                .i32(2);
            for x in q.iter().flatten() {
                blob = blob.f32(*x);
            }
            let mut docs: Vec<i32> = (0..30).rev().collect();
            let mut scores: Vec<f32> = (0..30).map(|i| 1.0 + i as f32).collect();
            let n = crate::jvm_reader::rescore_hits(
                h,
                &blob.0,
                30,
                0.0,
                1.0,
                3,
                &mut docs,
                &mut scores,
            )
            .unwrap_or_else(|rc| panic!("{rc:?}: {}", crate::error::last_error()));
            assert_eq!(n, 30);
            for (d, s) in docs.iter().zip(&scores) {
                let want = values[*d as usize][e]
                    .as_ref()
                    .map_or(0.0, |v| query.score(v, &mut Vec::new()).unwrap().max(0.0));
                assert_eq!(*s, want, "{name} rescored doc {d}");
            }
        }
        // a dimension other than the field's is an error, not a score
        let blob = Blob::tree()
            .u8(NODE_MAX_SIM)
            .u8(5)
            .str("f32")
            .i32(2)
            .i32(1)
            .f32(1.0)
            .f32(0.0);
        let (_, msg) = crate::jvm_nodes::tests::fails(h, &blob);
        assert!(msg.contains("same dimension"), "{msg}");
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn decodes_and_refuses_bad_nodes() {
        assert!(decoded(&node(2, 2, &[1.0, 0.0, 0.0, 1.0])).is_ok());
        for bad in [
            node(0, 1, &[]),
            node(2, 0, &[]),
            node(2, 2, &[1.0, 0.0, 0.0]),
            node(i32::MAX, i32::MAX, &[1.0]),
            node(-1, 1, &[1.0]),
        ] {
            assert!(decoded(&bad).is_err());
        }
    }
}
