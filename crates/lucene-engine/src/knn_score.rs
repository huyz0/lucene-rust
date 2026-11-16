//! OpenSearch's `script_score` (`ScriptScoreQuery`) running the k-NN
//! plugin's `knn_score` script over a `knn_vector` space, as node `42` of the
//! query tree ([`crate::queries`]).
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
//! | `space` | `u8`: `0` l2, `1` cosinesimil, `2` innerproduct, `3` l1, `4` linf, `5` hamming, `6` hammingbit |
//! | `data_type` | `u8`: `0` float, `1` byte or binary, `2` half_float, `3` long (hammingbit only; the target is 8 big-endian bytes) |
//! | `half_cosine` | `u8`: `1` scores cosine as `(1 + cos) / 2` (indexes created on 2.19 or later) |
//! | target | `u8` `0` then `i32` count and that many `f32`, or `1` then bytes |
//! | formula | `u8`: `0` `knn_score`'s, `1` a Painless script's ([`crate::knn_painless`], which follows) |
//!
//! Under a Painless formula the space and cosine flag are unused, and a
//! document without a vector is an error rather than a score of 0.

use std::sync::Arc;

use lucene_codecs::vectors::{
    dot_product, dot_product_bytes, f16_to_f32, scale_max_inner_product_score, square_distance,
    square_distance_bytes,
};
use lucene_search::extended_query::ExtendedQuery;
use lucene_search::function::FunctionScoreQuery;
use lucene_search::query::Clause;
use lucene_search::reader::doc_values::{get_binary, get_sorted_numeric};
use lucene_search::reader::BinaryDocValues;
use lucene_search::values_source::{
    BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};
use lucene_search::{Error, Result};
use lucene_util::vector_util::{cosine_sums_i8, xor_bit_count};

use crate::nodes::invalid;
use crate::query_blob::Cursor;
use crate::status::Status;

pub const NODE_KNN_SCORE: u8 = 42;

/// `hammingbit`'s space number on the wire (`KnnScoringScriptEngine.HAMMING_BIT_CODE`).
const HAMMING_BIT_CODE: u8 = 6;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Space {
    L2,
    Cosine,
    InnerProduct,
    L1,
    LInf,
    Hamming,
    /// The `hammingbit` space over a `long` or a `binary` field: the bits in
    /// which the value and the query differ, as numbers (a `BigInteger`'s
    /// `xor`, right aligned).
    HammingBit,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DataType {
    Float,
    Byte,
    HalfFloat,
    /// A `long` field's doc values (with `hammingbit`, the target 8 bytes).
    Long,
}

#[derive(Debug, Clone)]
enum Target {
    Floats(Arc<[f32]>),
    Bytes(Arc<[u8]>),
}

/// `knn_score`'s value for each document of a segment.
#[derive(Debug, Clone)]
pub struct KnnScoreSource {
    field: String,
    space: Space,
    data_type: DataType,
    half_cosine: bool,
    target: Target,
    /// The float target's squared magnitude (`cosinesimilOptimized`'s).
    norm: f32,
    /// A Painless script's formula in place of `knn_score`'s.
    painless: Option<Arc<crate::knn_painless::Painless>>,
}

pub fn decode(
    c: &mut Cursor<'_>,
    depth: usize,
    nodes: &mut usize,
) -> std::result::Result<Clause, Status> {
    let sub = crate::query_blob::decode_node(c, depth.saturating_add(1), nodes)?;
    let field = std::str::from_utf8(c.bytes()?)
        .map_err(|_| Status::InvalidUtf8)?
        .to_string();
    let space = match c.u8()? {
        0 => Space::L2,
        1 => Space::Cosine,
        2 => Space::InnerProduct,
        3 => Space::L1,
        4 => Space::LInf,
        5 => Space::Hamming,
        HAMMING_BIT_CODE => Space::HammingBit,
        other => {
            return Err(invalid(format!(
                "query tree: knn_score space {other} (expected 0..=6)"
            )))
        }
    };
    let data_type = match c.u8()? {
        0 => DataType::Float,
        1 => DataType::Byte,
        2 => DataType::HalfFloat,
        3 => DataType::Long,
        other => {
            return Err(invalid(format!(
                "query tree: knn_score data type {other} (expected 0..=3)"
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
            let n = c.length()?;
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
    let painless = match c.u8()? {
        0 => None,
        1 => Some(Arc::new(crate::knn_painless::decode(c)?)),
        other => {
            return Err(invalid(format!(
                "query tree: knn_score formula {other} (expected 0..=1)"
            )))
        }
    };
    let floats = matches!(target, Target::Floats(_));
    // `hammingbit` reads a long or a binary field, any other space a vector.
    let hamming_bit = space == Space::HammingBit;
    let long = data_type == DataType::Long;
    if long && !(hamming_bit && matches!(&target, Target::Bytes(b) if b.len() == 8))
        || hamming_bit && (floats || painless.is_some() || !(long || data_type == DataType::Byte))
    {
        return Err(invalid(
            "query tree: knn_score hammingbit target does not match the field's data type",
        ));
    }
    if painless
        .as_ref()
        .is_some_and(|p| p.needs_float_target() && data_type == DataType::Byte)
    {
        return Err(invalid(
            "query tree: painless cosineSimilarity with a norm over a byte field",
        ));
    }
    if !long && (floats == (data_type == DataType::Byte) || (space == Space::Hamming && floats)) {
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
        painless,
    };
    Ok(Clause::Extended(Box::new(ExtendedQuery::FunctionScore(
        FunctionScoreQuery::new(sub, Arc::new(source)),
    ))))
}

impl KnnScoreSource {
    /// The script's value for a document without a vector: none for
    /// `knn_score` (the document scores 0), an error for Painless.
    fn missing(&self, doc: i32) -> Result<Option<f32>> {
        match &self.painless {
            None => Ok(None),
            Some(_) => Err(crate::knn_painless::missing(&self.field, doc)),
        }
    }

    /// The script's score of a float vector.
    fn floats(&self, q: &[f32], v: &[f32]) -> Result<Option<f32>> {
        match &self.painless {
            None => Ok(Some(self.score_floats(q, v))),
            Some(p) => p.score_floats(q, v).map(Some),
        }
    }

    /// The script's score of a byte vector.
    fn bytes(&self, q: &[u8], v: &[u8]) -> Result<Option<f32>> {
        match &self.painless {
            None => Ok(Some(self.score_bytes(q, v))),
            Some(p) => p.score_bytes(q, v).map(Some),
        }
    }

    /// `knn_score`'s score of a float vector.
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
            Space::Hamming | Space::HammingBit => 0.0,
        }
    }

    /// `knn_score`'s score of a byte vector.
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
            Space::HammingBit => 1.0 / (1.0 + hamming_bit_bytes(q, v) as f32),
        }
    }
}

/// The bits in which two big-endian unsigned numbers differ (`new
/// BigInteger(1, q).xor(new BigInteger(1, v)).bitCount()`): aligned at their
/// low ends, the longer one's extra high bytes counting whole.
fn hamming_bit_bytes(q: &[u8], v: &[u8]) -> u32 {
    let (long, short) = if q.len() >= v.len() { (q, v) } else { (v, q) };
    let split = long.len().saturating_sub(short.len());
    let (high, low) = long.split_at(split);
    let high: u32 = high.iter().map(|b| b.count_ones()).sum();
    let low: u32 = low
        .iter()
        .zip(short)
        .map(|(a, b)| (a ^ b).count_ones())
        .sum();
    high.saturating_add(low)
}

/// The first value of a `binary` field's doc value (`CustomBinaryDocValuesField`:
/// the count, then each value's length and bytes, sorted), as `doc[field]`'s
/// `getValue()` reads it; `None` for no value or one the encoding does not hold.
fn first_binary_value(raw: &[u8]) -> Option<&[u8]> {
    fn vint(raw: &[u8], at: &mut usize) -> Option<usize> {
        let mut value = 0usize;
        for shift in (0..35).step_by(7) {
            let b = *raw.get(*at)?;
            *at = at.checked_add(1)?;
            value |= usize::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Some(value);
            }
        }
        None
    }
    let mut at = 0;
    if vint(raw, &mut at)? == 0 {
        return None;
    }
    let len = vint(raw, &mut at)?;
    raw.get(at..at.checked_add(len)?)
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
        let me = Arc::new(self.clone());
        let none = |me: Arc<KnnScoreSource>| -> BoxDoubleValues<'c> {
            Box::new(Scores {
                next: Box::new(move |doc| me.missing(doc)),
                value: 0.0,
            })
        };
        let Some(info) = reader.field_infos().field_by_name(&self.field) else {
            return Ok(none(me));
        };
        let number = info.number;
        if self.data_type == DataType::Long {
            let Target::Bytes(q) = &self.target else {
                return Ok(none(me));
            };
            let q = i64::from_be_bytes(
                q[..8]
                    .try_into()
                    .map_err(|_| Error::IllegalState("hammingbit target is not 8 bytes".into()))?,
            );
            let mut values = get_sorted_numeric(reader, &self.field)?;
            return Ok(Box::new(Scores {
                next: Box::new(move |doc| {
                    if !values.advance_exact(doc)? || values.doc_value_count() == 0 {
                        return me.missing(doc);
                    }
                    // `ScriptDocValues.Longs.getValue()`: the smallest value.
                    let bits = (q ^ values.next_value()?).count_ones();
                    Ok(Some(1.0 / (1.0 + bits as f32)))
                }),
                value: 0.0,
            }));
        }
        if info.vector_dimension > 0 {
            let Some((flat, _)) = reader.knn_vectors_for(number)? else {
                return Ok(none(me));
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
                                return me.missing(doc);
                            };
                            let v = values.vector_ref(ord, &mut scratch)?;
                            me.floats(&q, v)
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
                                return me.missing(doc);
                            };
                            me.bytes(&q, values.vector(ord)?)
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
                    return me.missing(doc);
                }
                let raw = dv.binary_value();
                if me.space == Space::HammingBit {
                    // `doc[field].getValue()`: the first of the field's values.
                    let (Target::Bytes(q), Some(v)) = (&target, first_binary_value(raw)) else {
                        return me.missing(doc);
                    };
                    return me.bytes(q, v);
                }
                match (&target, data_type) {
                    (Target::Bytes(q), _) => me.bytes(q, raw),
                    (Target::Floats(q), DataType::HalfFloat) => {
                        scratch.clear();
                        scratch.extend(
                            raw.chunks_exact(2)
                                .map(|h| f16_to_f32(u16::from_le_bytes([h[0], h[1]]))),
                        );
                        me.floats(q, &scratch)
                    }
                    (Target::Floats(q), _) => {
                        scratch.clear();
                        scratch.extend(
                            raw.chunks_exact(4)
                                .map(|f| f32::from_be_bytes([f[0], f[1], f[2], f[3]])),
                        );
                        me.floats(q, &scratch)
                    }
                }
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
        let formula = self
            .painless
            .as_ref()
            .map_or_else(String::new, |p| p.describe());
        format!(
            "knn_score(field={}, space={:?}, type={:?}, half_cosine={}, target={}, formula={formula})",
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
mod tests;
