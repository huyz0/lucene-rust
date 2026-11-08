//! OpenSearch's `script_score` (`ScriptScoreQuery`) running the k-NN
//! plugin's `knn_score` script over a `knn_vector` space, as node `42` of the
//! query tree ([`crate::jvm_queries`]).
//!
//! The sub-query's documents are scored by the script's distance, with the
//! script's float arithmetic (the plugin's `KnnScoringScriptEngine` and
//! `KnnScoring`), and a document without a value scores 0. It runs as the
//! ported Lucene [`FunctionScoreQuery`] around [`KnnScoreSource`]: Lucene's
//! scorer computes `(float) (boost * value)` in `double`, which for a `float`
//! value and boost is the correctly rounded `float` product OpenSearch's
//! `ScriptScorer` computes.
//!
//! A document's vector comes from the field's vector files when the segment
//! has them (an `index.knn` index), else from its binary doc values: big-endian
//! floats for a float field, raw bytes for a byte or binary one, little-endian
//! half floats for a half_float one.
//!
//! The node, after its kind byte:
//!
//! | field | layout |
//! |---|---|
//! | sub-query | a node |
//! | `field` | string |
//! | `space` | `u8`: `0` l2, `1` cosinesimil, `2` innerproduct, `3` l1, `4` linf, `5` hamming |
//! | `data_type` | `u8`: `0` float, `1` byte or binary, `2` half_float |
//! | `half_cosine` | `u8`: `1` scores cosine as `(1 + cos) / 2` (indexes created on 2.19 or later) |
//! | target | `u8` `0` then `i32` count and that many `f32`, or `1` then bytes |

use std::sync::Arc;

use lucene_codecs::vectors::{
    dot_product, dot_product_bytes, f16_to_f32, scale_max_inner_product_score, square_distance,
    square_distance_bytes,
};
use lucene_search::extended_query::ExtendedQuery;
use lucene_search::function::FunctionScoreQuery;
use lucene_search::query::Clause;
use lucene_search::reader::doc_values::get_binary;
use lucene_search::reader::BinaryDocValues;
use lucene_search::values_source::{
    BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};
use lucene_search::{Error, Result};
use lucene_util::vector_util::{cosine_sums_i8, xor_bit_count};

use crate::error::FfiStatus;
use crate::jvm_nodes::invalid;
use crate::jvm_reader::Cursor;

pub(crate) const NODE_KNN_SCORE: u8 = 42;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Space {
    L2,
    Cosine,
    InnerProduct,
    L1,
    LInf,
    Hamming,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DataType {
    Float,
    Byte,
    HalfFloat,
}

#[derive(Debug, Clone)]
enum Target {
    Floats(Arc<[f32]>),
    Bytes(Arc<[u8]>),
}

/// `knn_score`'s value for each document of a segment.
#[derive(Debug, Clone)]
pub(crate) struct KnnScoreSource {
    field: String,
    space: Space,
    data_type: DataType,
    half_cosine: bool,
    target: Target,
    /// The float target's squared magnitude (`cosinesimilOptimized`'s).
    norm: f32,
}

pub(crate) fn decode(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> std::result::Result<Clause, FfiStatus> {
    let sub = crate::jvm_reader::decode_node(c, depth + 1, nodes)?;
    let field = std::str::from_utf8(c.bytes()?)
        .map_err(|_| FfiStatus::InvalidUtf8)?
        .to_string();
    let space = match c.u8()? {
        0 => Space::L2,
        1 => Space::Cosine,
        2 => Space::InnerProduct,
        3 => Space::L1,
        4 => Space::LInf,
        5 => Space::Hamming,
        other => {
            return Err(invalid(format!(
                "query tree: knn_score space {other} (expected 0..=5)"
            )))
        }
    };
    let data_type = match c.u8()? {
        0 => DataType::Float,
        1 => DataType::Byte,
        2 => DataType::HalfFloat,
        other => {
            return Err(invalid(format!(
                "query tree: knn_score data type {other} (expected 0..=2)"
            )))
        }
    };
    let half_cosine = match c.u8()? {
        0 => false,
        1 => true,
        other => {
            return Err(invalid(format!(
                "query tree: knn_score cosine flag {other} (expected 0..=1)"
            )))
        }
    };
    let target = match c.u8()? {
        0 => {
            let n = c.len()?;
            let mut v = Vec::with_capacity(n.min(c.remaining() / 4)); // alloc-ok: bounded by the blob
            for _ in 0..n {
                v.push(f32::from_bits(c.i32()? as u32));
            }
            Target::Floats(v.into())
        }
        1 => Target::Bytes(c.bytes()?.into()),
        other => {
            return Err(invalid(format!(
                "query tree: knn_score target kind {other} (expected 0..=1)"
            )))
        }
    };
    let floats = matches!(target, Target::Floats(_));
    if floats == (data_type == DataType::Byte) || (space == Space::Hamming && floats) {
        return Err(invalid(
            "query tree: knn_score target does not match the field's data type",
        ));
    }
    let norm = match &target {
        Target::Floats(q) => q.iter().fold(0.0f32, |n, x| n + x * x),
        Target::Bytes(_) => 0.0,
    };
    let source = KnnScoreSource {
        field,
        space,
        data_type,
        half_cosine,
        target,
        norm,
    };
    Ok(Clause::Extended(Box::new(ExtendedQuery::FunctionScore(
        FunctionScoreQuery::new(sub, Arc::new(source)),
    ))))
}

impl KnnScoreSource {
    /// The script's score of a float vector.
    fn score_floats(&self, q: &[f32], v: &[f32]) -> f32 {
        match self.space {
            Space::L2 => 1.0 / (1.0 + square_distance(q, v)),
            Space::L1 => {
                let d = q.iter().zip(v).fold(0.0f32, |d, (a, b)| d + (a - b).abs());
                1.0 / (1.0 + d)
            }
            Space::LInf => {
                let d = q
                    .iter()
                    .zip(v)
                    .fold(0.0f32, |d, (a, b)| (a - b).abs().max(d));
                1.0 / (1.0 + d)
            }
            Space::InnerProduct => scale_max_inner_product_score(dot_product(q, v)),
            Space::Cosine => {
                // `cosinesimilOptimized`: sequential float sums, one square root in double.
                let (mut dot, mut norm) = (0.0f32, 0.0f32);
                for (a, b) in q.iter().zip(v) {
                    dot += a * b;
                    norm += b * b;
                }
                let product = self.norm * norm;
                let cos = if product == 0.0 {
                    0.0
                } else {
                    (f64::from(dot) / f64::from(product).sqrt()) as f32
                };
                if self.half_cosine {
                    // `Math.max(x, 0)`: NaN stays NaN, for the script score check to refuse.
                    let half = (1.0 + cos) / 2.0;
                    if half > 0.0 || half.is_nan() {
                        half
                    } else {
                        0.0
                    }
                } else {
                    1.0 + cos
                }
            }
            Space::Hamming => 0.0,
        }
    }

    /// The script's score of a byte vector.
    fn score_bytes(&self, q: &[u8], v: &[u8]) -> f32 {
        let diffs = || {
            q.iter()
                .zip(v)
                .map(|(&a, &b)| (f32::from(a as i8) - f32::from(b as i8)).abs())
        };
        match self.space {
            Space::L2 => 1.0 / (1.0 + square_distance_bytes(q, v) as f32),
            Space::L1 => 1.0 / (1.0 + diffs().fold(0.0f32, |d, x| d + x)),
            Space::LInf => 1.0 / (1.0 + diffs().fold(0.0f32, |d, x| x.max(d))),
            Space::InnerProduct => scale_max_inner_product_score(dot_product_bytes(q, v) as f32),
            Space::Cosine => {
                // `VectorUtil.cosine(byte[], byte[])`: a zero document vector gives NaN.
                let (sum, n1, n2) = cosine_sums_i8(q, v);
                1.0 + (f64::from(sum) / (f64::from(n1) * f64::from(n2)).sqrt()) as f32
            }
            Space::Hamming => 1.0 / (1.0 + xor_bit_count(q, v) as f32),
        }
    }
}

/// A leaf's values: the documents with a vector, ascending, and how to score one.
struct Scores<'c> {
    next: Box<dyn FnMut(i32) -> Result<Option<f32>> + 'c>,
    value: f32,
}

impl DoubleValues for Scores<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        match (self.next)(doc)? {
            // `ScriptScoreQuery`'s check on every script value.
            Some(v) if v < 0.0 || v.is_nan() => Err(Error::IllegalArgument(format!(
                "script_score script returned an invalid score [{v}] for doc [{doc}]. Must be a non-negative score!"
            ))),
            Some(v) => {
                self.value = v;
                Ok(true)
            }
            None => Ok(false),
        }
    }
    fn double_value(&mut self) -> Result<f64> {
        Ok(f64::from(self.value))
    }
}

impl DoubleValuesSource for KnnScoreSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let reader = ctx.leaf_reader(leaf)?;
        let Some(info) = reader.field_infos().field_by_name(&self.field) else {
            return Ok(Box::new(Scores {
                next: Box::new(|_| Ok(None)),
                value: 0.0,
            }));
        };
        let number = info.number;
        let me = Arc::new(self.clone());
        if info.vector_dimension > 0 {
            let Some((flat, _)) = reader.knn_vectors_for(number)? else {
                return Ok(Box::new(Scores {
                    next: Box::new(|_| Ok(None)),
                    value: 0.0,
                }));
            };
            return match &self.target {
                Target::Floats(q) => {
                    let values = flat.float_vector_values(number)?;
                    let mut ords =
                        Ordinals::new(values.size(), reader.max_doc, |d| values.ord_to_docs(d))?;
                    let q = Arc::clone(q);
                    let mut scratch = vec![0f32; q.len()]; // alloc-ok: the target decoded from the blob, no longer than it
                    Ok(Box::new(Scores {
                        next: Box::new(move |doc| {
                            let Some(ord) = ords.ordinal(doc) else {
                                return Ok(None);
                            };
                            let v = values.vector_ref(ord, &mut scratch)?;
                            Ok(Some(me.score_floats(&q, v)))
                        }),
                        value: 0.0,
                    }))
                }
                Target::Bytes(q) => {
                    let values = flat.byte_vector_values(number)?;
                    let mut ords =
                        Ordinals::new(values.size(), reader.max_doc, |d| values.ord_to_docs(d))?;
                    let q = Arc::clone(q);
                    Ok(Box::new(Scores {
                        next: Box::new(move |doc| {
                            let Some(ord) = ords.ordinal(doc) else {
                                return Ok(None);
                            };
                            Ok(Some(me.score_bytes(&q, values.vector(ord)?)))
                        }),
                        value: 0.0,
                    }))
                }
            };
        }
        let mut dv: Box<dyn BinaryDocValues + 'c> = get_binary(reader, &self.field)?;
        let target = self.target.clone();
        let data_type = self.data_type;
        let mut scratch = Vec::new();
        Ok(Box::new(Scores {
            next: Box::new(move |doc| {
                if !dv.advance_exact(doc)? {
                    return Ok(None);
                }
                let raw = dv.binary_value();
                Ok(Some(match (&target, data_type) {
                    (Target::Bytes(q), _) => me.score_bytes(q, raw),
                    (Target::Floats(q), DataType::HalfFloat) => {
                        scratch.clear();
                        scratch.extend(
                            raw.chunks_exact(2)
                                .map(|h| f16_to_f32(u16::from_le_bytes([h[0], h[1]]))),
                        );
                        me.score_floats(q, &scratch)
                    }
                    (Target::Floats(q), _) => {
                        scratch.clear();
                        scratch.extend(
                            raw.chunks_exact(4)
                                .map(|f| f32::from_be_bytes([f[0], f[1], f[2], f[3]])),
                        );
                        me.score_floats(q, &scratch)
                    }
                }))
            }),
            value: 0.0,
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        format!("knn_score(field={}, space={:?})", self.field, self.space)
    }
    fn equality_key(&self) -> String {
        let target = match &self.target {
            Target::Floats(q) => q.iter().map(|x| format!("{:08x}", x.to_bits())).collect(),
            Target::Bytes(q) => q.iter().map(|b| format!("{b:02x}")).collect::<String>(),
        };
        format!(
            "knn_score(field={}, space={:?}, type={:?}, half_cosine={}, target={})",
            self.field, self.space, self.data_type, self.half_cosine, target
        )
    }
}

/// A segment's doc -> vector ordinal map. A field with a vector in every
/// document numbers its ordinals by doc id; a sparse field searches the
/// ordinal -> doc list, so documents may be asked in any order.
enum Ordinals {
    Dense(i32),
    Sparse(Vec<i32>),
}

impl Ordinals {
    fn new<E: Into<Error>>(
        size: i32,
        max_doc: i32,
        ord_to_docs: impl FnOnce(&mut Vec<i32>) -> std::result::Result<(), E>,
    ) -> Result<Ordinals> {
        if size == max_doc {
            return Ok(Ordinals::Dense(size));
        }
        let mut docs = Vec::new();
        ord_to_docs(&mut docs).map_err(Into::into)?;
        Ok(Ordinals::Sparse(docs))
    }

    fn ordinal(&mut self, doc: i32) -> Option<i32> {
        match self {
            Ordinals::Dense(size) => (doc >= 0 && doc < *size).then_some(doc),
            // An ordinal indexes an `i32`-sized list, so it fits.
            Ordinals::Sparse(docs) => docs.binary_search(&doc).ok().map(|o| o as i32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_nodes::tests::Blob;

    fn node(space: u8, data_type: u8, target: Blob) -> Blob {
        Blob::default()
            .u8(NODE_KNN_SCORE)
            .term("f", "x")
            .str("v")
            .u8(space)
            .u8(data_type)
            .u8(0)
            .raw(&target)
    }

    fn decoded(blob: &Blob) -> std::result::Result<Clause, FfiStatus> {
        let mut c = Cursor::new(&blob.0);
        let mut nodes = 0;
        crate::jvm_reader::decode_node(&mut c, 0, &mut nodes)
    }

    fn source(space: Space, half_cosine: bool, target: Target) -> KnnScoreSource {
        let norm = match &target {
            Target::Floats(q) => q.iter().fold(0.0f32, |n, x| n + x * x),
            Target::Bytes(_) => 0.0,
        };
        KnnScoreSource {
            field: "v".into(),
            space,
            data_type: DataType::Float,
            half_cosine,
            target,
            norm,
        }
    }

    #[test]
    fn decodes_and_refuses_bad_nodes() {
        let floats = Blob::default().u8(0).i32(2).f32(0.5).f32(1.0);
        let bytes = Blob::default().u8(1).bytes(&[1, 2]);
        assert!(decoded(&node(0, 0, floats.clone())).is_ok());
        assert!(decoded(&node(5, 1, bytes.clone())).is_ok());
        assert!(decoded(&node(6, 0, floats.clone())).is_err(), "space");
        assert!(decoded(&node(0, 3, floats.clone())).is_err(), "data type");
        assert!(
            decoded(&node(0, 1, floats.clone())).is_err(),
            "floats for a byte field"
        );
        assert!(
            decoded(&node(0, 0, bytes.clone())).is_err(),
            "bytes for a float field"
        );
        assert!(decoded(&node(5, 0, floats)).is_err(), "hamming over floats");
        assert!(
            decoded(&node(0, 1, Blob::default().u8(2))).is_err(),
            "target kind"
        );
    }

    #[test]
    fn float_spaces_score_as_the_script() {
        let q = vec![1.0f32, -2.0, 0.5];
        let v = [0.5f32, 1.0, -1.5];
        let s = |space, half| {
            source(space, half, Target::Floats(q.clone().into())).score_floats(&q, &v)
        };
        // Squared distance 0.25 + 9 + 4 = 13.25; l1 0.5 + 3 + 2 = 5.5; linf 3.
        assert_eq!(s(Space::L2, false), 1.0 / 14.25);
        assert_eq!(s(Space::L1, false), 1.0 / 6.5);
        assert_eq!(s(Space::LInf, false), 1.0 / 4.0);
        // Inner product 0.5 - 2 - 0.75 = -2.25 scales to 1 / 3.25.
        assert_eq!(s(Space::InnerProduct, false), 1.0 / 3.25);
        let cos = (-2.25f64 / (5.25f64 * 3.5).sqrt()) as f32;
        assert_eq!(s(Space::Cosine, false), 1.0 + cos);
        assert_eq!(s(Space::Cosine, true), ((1.0 + cos) / 2.0).max(0.0));
        // A positive inner product scales to ip + 1.
        assert_eq!(
            source(Space::InnerProduct, false, Target::Floats(q.clone().into()))
                .score_floats(&q, &q),
            5.25 + 1.0
        );
    }

    #[test]
    fn byte_spaces_score_as_the_script() {
        let q = vec![3u8, 0xFE, 1];
        let v = [1u8, 2, 0xFF];
        let s = |space| source(space, false, Target::Bytes(q.clone().into())).score_bytes(&q, &v);
        // Signed: q = (3, -2, 1), v = (1, 2, -1).
        assert_eq!(s(Space::L2), 1.0 / (1.0 + 4.0 + 16.0 + 4.0));
        assert_eq!(s(Space::L1), 1.0 / (1.0 + 8.0));
        assert_eq!(s(Space::LInf), 1.0 / 5.0);
        assert_eq!(
            s(Space::InnerProduct),
            scale_max_inner_product_score(3.0 - 4.0 - 1.0)
        );
        // Hamming over the raw bits: 3^1 = 2 (1 bit), 0xFE^2 = 0xFC (6), 1^0xFF = 0xFE (7).
        assert_eq!(s(Space::Hamming), 1.0 / 15.0);
    }

    #[test]
    fn equality_names_the_target_and_zero_byte_cosine_is_nan() {
        let a = source(Space::L2, false, Target::Floats(vec![1.0, 2.0].into()));
        let b = source(Space::L2, false, Target::Floats(vec![1.0, 3.0].into()));
        assert_ne!(a.equality_key(), b.equality_key());
        assert_eq!(a.equality_key(), a.clone().equality_key());
        let c = source(Space::Cosine, false, Target::Bytes(vec![1, 2].into()));
        assert!(c.score_bytes(&[1, 2], &[0, 0]).is_nan());
    }
}
