//! OpenSearch's `function_score` (`org.opensearch.common.lucene.search.function.FunctionScoreQuery`,
//! OpenSearch 3.8.0) as node `24` of the query tree ([`crate::jvm_nodes`]).
//!
//! OpenSearch's query is not Lucene's `FunctionScoreQuery`: it combines the
//! sub-query's score with its functions' in its own `double` arithmetic
//! (`FunctionFactorScorer.computeScore`, `CombineFunction.combine`). It runs
//! here as the ported Lucene [`FunctionScoreQuery`] around a values source
//! ([`OpenSearchFunctionScore`]) that computes exactly that arithmetic and
//! hands back the combined `float` widened to `double` -- which Lucene's
//! scorer (`(float) (boost * value)` with the boost `1`) narrows back to the
//! same bits. OpenSearch passes a weight's boost to its sub-query's weight
//! only: the source says so ([`DoubleValuesSource::boosts_wrapped_query`]),
//! and the scorer hands every boost from above -- a `BoostQuery`, a boosted
//! `bool`, `dis_max` or `nested` -- to the sub-query instead.
//!
//! **Not bit-exact by construction**: `field_value_factor`'s `log`/`log1p`/
//! `ln`/`ln1p`/`ln2p` and the decays' `exp` are libm's here and HotSpot's
//! `Math` intrinsics in OpenSearch, which Java allows to differ from the
//! correctly rounded result by one ulp (`StrictMath` would not, but
//! OpenSearch calls `Math`). A one-ulp `double` difference rarely survives
//! the final `(float)` cast; every value the tests compared was equal, but a
//! last-bit difference stays possible.
//!
//! What OpenSearch throws on -- a missing field value without `missing`, a
//! negative field-value score, a negative or `NaN` final score -- is a search
//! error here, so the query re-runs on Lucene and fails there as it would.
//!
//! The node, after its kind byte:
//!
//! | field | layout |
//! |---|---|
//! | `combine` | `u8`, `CombineFunction`'s ordinal: `0` multiply, `1` replace, `2` sum, `3` avg, `4` min, `5` max |
//! | `score_mode` | `u8`, `FunctionScoreQuery.ScoreMode`'s ordinal: `0` first, `1` avg, `2` max, `3` sum, `4` min, `5` multiply |
//! | `max_boost` | `f32` |
//! | sub-query | a node |
//! | `count` | `i32` (at least 1), then per function `has_filter: u8`, the filter (a node) when set, and the function ([`decode_function`]) |

use std::sync::Arc;

use lucene_search::extended_query::ExtendedQuery;
use lucene_search::function::FunctionScoreQuery;
use lucene_search::query::Clause;
use lucene_search::reader::doc_values::{get_sorted_numeric, get_sorted_set};
use lucene_search::reader::{SortedNumericDocValues, SortedSetDocValues};
use lucene_search::values_source::{
    BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};
use lucene_search::{Error, Result};

use crate::error::FfiStatus;
use crate::jvm_nodes::invalid;
use crate::jvm_reader::{decode_node, hex, Cursor};
use crate::query::check_clause_count;

/// `CombineFunction`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Combine {
    Multiply,
    Replace,
    Sum,
    Avg,
    Min,
    Max,
}

impl Combine {
    /// `combine(queryScore, funcScore, maxBoost)`.
    fn combine(self, query: f64, func: f64, max_boost: f64) -> f32 {
        let f = java_min(func, max_boost);
        (match self {
            Combine::Multiply => query * f,
            Combine::Replace => f,
            Combine::Sum => query + f,
            Combine::Avg => (f + query) / 2.0,
            Combine::Min => java_min(query, f),
            Combine::Max => java_max(query, f),
        }) as f32
    }
}

/// `Math.min` over doubles: `NaN` if either is, and `-0.0` below `0.0`.
fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else {
        a.min(b)
    }
}

/// `Math.max` over doubles.
fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == b {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else {
        a.max(b)
    }
}

/// `FunctionScoreQuery.ScoreMode`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ScoreMode {
    First,
    Avg,
    Max,
    Sum,
    Min,
    Multiply,
}

/// How a numeric field's sorted-numeric longs read as doubles
/// (`IndexNumericFieldData.load(ctx).getDoubleValues()`).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Numeric {
    /// An integral type or a date: `(double) v`.
    Long,
    /// `double`: `NumericUtils.sortableLongToDouble`.
    Double,
    /// `float`: `NumericUtils.sortableIntToFloat`, widened.
    Float,
}

impl Numeric {
    fn decode(self, v: i64) -> f64 {
        match self {
            Numeric::Long => v as f64,
            Numeric::Double => f64::from_bits((v ^ ((v >> 63) & i64::MAX)) as u64),
            Numeric::Float => {
                let i = v as i32;
                f64::from(f32::from_bits((i ^ ((i >> 31) & i32::MAX)) as u32))
            }
        }
    }
}

/// `FieldValueFactorFunction.Modifier`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Modifier {
    None,
    Log,
    Log1p,
    Log2p,
    Ln,
    Ln1p,
    Ln2p,
    Square,
    Sqrt,
    Reciprocal,
}

impl Modifier {
    fn apply(self, n: f64) -> f64 {
        match self {
            Modifier::None => n,
            Modifier::Log => n.log10(),
            Modifier::Log1p => (n + 1.0).log10(),
            Modifier::Log2p => (n + 2.0).log10(),
            Modifier::Ln => n.ln(),
            Modifier::Ln1p => n.ln_1p(),
            Modifier::Ln2p => (n + 1.0).ln_1p(),
            // `Math.pow(n, 2)`: HotSpot computes it as `n * n`.
            Modifier::Square => n * n,
            Modifier::Sqrt => n.sqrt(),
            Modifier::Reciprocal => 1.0 / n,
        }
    }
}

/// The decay functions' `DecayFunction.evaluate(value, scale)`, `scale`
/// already `processScale`d (OpenSearch keeps it so).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Decay {
    Gauss,
    Exp,
    Linear,
}

impl Decay {
    fn evaluate(self, value: f64, scale: f64) -> f64 {
        match self {
            // `Math.exp(0.5 * Math.pow(value, 2.0) / scale)`.
            Decay::Gauss => (0.5 * (value * value) / scale).exp(),
            Decay::Exp => (scale * value).exp(),
            Decay::Linear => java_max(0.0, (scale - value) / scale),
        }
    }
}

/// `MultiValueMode`, over a document's sorted distances.
#[derive(Debug, Clone, Copy, PartialEq)]
enum MultiValue {
    Sum,
    Avg,
    Median,
    Min,
    Max,
}

impl MultiValue {
    fn pick(self, sorted: &[f64]) -> f64 {
        let n = sorted.len();
        match self {
            MultiValue::Sum => sorted.iter().fold(0.0, |t, v| t + v),
            MultiValue::Avg => sorted.iter().fold(0.0, |t, v| t + v) / n as f64,
            MultiValue::Median if n.is_multiple_of(2) => {
                (sorted[(n - 1) / 2] + sorted[(n - 1) / 2 + 1]) / 2.0
            }
            MultiValue::Median => sorted[(n - 1) / 2],
            MultiValue::Min => sorted[0],
            MultiValue::Max => sorted[n - 1],
        }
    }
}

/// Where `random_score` hashes from.
#[derive(Debug, Clone, PartialEq)]
enum RandomField {
    /// No field: the document's index-wide ID.
    DocId,
    /// A keyword's first (smallest) term.
    Keyword(String),
    /// An integral numeric, boolean or date field (`LeafLongFieldData`):
    /// its values as `Long.toString` prints them, the bytewise smallest
    /// string hashed (`FieldData.toString` sorts them as bytes).
    Long(String),
}

/// One `ScoreFunction`.
#[derive(Debug, Clone, PartialEq)]
enum Function {
    /// `WeightFactorFunction`'s `ScoreOne`: `1`.
    One,
    /// `WeightFactorFunction`: the inner function times `weight`.
    Weight { weight: f32, inner: Box<Function> },
    /// `FieldValueFactorFunction`.
    FieldValue {
        field: String,
        numeric: Numeric,
        factor: f32,
        modifier: Modifier,
        missing: Option<f64>,
    },
    /// `RandomScoreFunction`, its seed already salted.
    Random {
        salted_seed: i32,
        field: RandomField,
    },
    /// `DecayFunctionBuilder.NumericFieldDataScoreFunction`.
    Decay {
        decay: Decay,
        field: String,
        numeric: Numeric,
        origin: f64,
        scale: f64,
        offset: f64,
        mode: MultiValue,
    },
}

impl Function {
    /// `getWeight()`.
    fn weight(&self) -> f32 {
        match self {
            Function::Weight { weight, .. } => *weight,
            _ => 1.0,
        }
    }

    fn fields<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Function::Weight { inner, .. } => inner.fields(out),
            Function::FieldValue { field, .. } | Function::Decay { field, .. } => out.push(field),
            Function::Random {
                field: RandomField::Keyword(f) | RandomField::Long(f),
                ..
            } => out.push(f),
            Function::One | Function::Random { .. } => {}
        }
    }
}

const FUNCTION_ONE: u8 = 0;
const FUNCTION_WEIGHT: u8 = 1;
const FUNCTION_FIELD_VALUE: u8 = 2;
const FUNCTION_RANDOM: u8 = 3;
const FUNCTION_DECAY: u8 = 4;

/// The node (kind `24`, its byte read at `start`).
pub(crate) fn decode(
    c: &mut Cursor<'_>,
    start: usize,
    depth: usize,
    nodes: &mut usize,
) -> std::result::Result<Clause, FfiStatus> {
    let combine = match c.u8()? {
        0 => Combine::Multiply,
        1 => Combine::Replace,
        2 => Combine::Sum,
        3 => Combine::Avg,
        4 => Combine::Min,
        5 => Combine::Max,
        other => {
            return Err(invalid(format!(
                "query tree: unknown boost mode {other} (expected 0..=5)"
            )))
        }
    };
    let score_mode = match c.u8()? {
        0 => ScoreMode::First,
        1 => ScoreMode::Avg,
        2 => ScoreMode::Max,
        3 => ScoreMode::Sum,
        4 => ScoreMode::Min,
        5 => ScoreMode::Multiply,
        other => {
            return Err(invalid(format!(
                "query tree: unknown function score mode {other} (expected 0..=5)"
            )))
        }
    };
    let max_boost = f32::from_bits(c.i32()? as u32);
    let sub = decode_node(c, depth + 1, nodes)?;
    let n = c.len()?;
    if n == 0 {
        return Err(invalid("query tree: a function score without functions"));
    }
    check_clause_count(nodes.saturating_add(n))?;
    let mut functions = Vec::new();
    for _ in 0..n {
        *nodes = nodes.saturating_add(1);
        let filter = match c.u8()? {
            0 => None,
            _ => Some(decode_node(c, depth + 1, nodes)?),
        };
        functions.push(Filtered {
            source: filter
                .as_ref()
                .map(|f| lucene_search::values_source::from_clause(f.clone())),
            filter,
            function: decode_function(c, 0)?,
        });
    }
    let source = OpenSearchFunctionScore {
        combine,
        score_mode,
        max_boost,
        functions,
        key: hex(c.since(start)),
    };
    Ok(Clause::Extended(Box::new(ExtendedQuery::FunctionScore(
        FunctionScoreQuery::new(sub, Arc::new(source)),
    ))))
}

fn field(c: &mut Cursor<'_>) -> std::result::Result<String, FfiStatus> {
    Ok(std::str::from_utf8(c.bytes()?)
        .map_err(|_| FfiStatus::InvalidUtf8)?
        .to_string())
}

fn numeric(c: &mut Cursor<'_>) -> std::result::Result<Numeric, FfiStatus> {
    Ok(match c.u8()? {
        0 => Numeric::Long,
        1 => Numeric::Double,
        2 => Numeric::Float,
        other => {
            return Err(invalid(format!(
                "query tree: unknown numeric field type {other} (expected 0..=2)"
            )))
        }
    })
}

/// One function, a tag and its fields:
///
/// | tag | layout | OpenSearch |
/// |---|---|---|
/// | `0` | nothing | `ScoreOne` (`weight` alone) |
/// | `1` | `weight: f32`, the inner function (not another weight) | `WeightFactorFunction` |
/// | `2` | `field`, `numeric: u8` (`0` long, `1` double, `2` float), `factor: f32`, `modifier: u8` (`Modifier`'s ordinal), `has_missing: u8`, `missing: f64` | `FieldValueFactorFunction` |
/// | `3` | `salted_seed: i32`, `kind: u8` (`0` doc ID, `1` keyword + `field`, `2` long + `field`) | `RandomScoreFunction` |
/// | `4` | `decay: u8` (`0` gauss, `1` exp, `2` linear), `field`, `numeric: u8`, `origin`, `scale` (processed), `offset` (`f64`), `mode: u8` (`MultiValueMode`'s ordinal) | a numeric or date decay function |
fn decode_function(c: &mut Cursor<'_>, depth: usize) -> std::result::Result<Function, FfiStatus> {
    Ok(match c.u8()? {
        FUNCTION_ONE => Function::One,
        FUNCTION_WEIGHT if depth == 0 => {
            let weight = f32::from_bits(c.i32()? as u32);
            Function::Weight {
                weight,
                inner: Box::new(decode_function(c, depth + 1)?),
            }
        }
        FUNCTION_FIELD_VALUE => {
            let field = field(c)?;
            let numeric = numeric(c)?;
            let factor = f32::from_bits(c.i32()? as u32);
            let modifier = match c.u8()? {
                0 => Modifier::None,
                1 => Modifier::Log,
                2 => Modifier::Log1p,
                3 => Modifier::Log2p,
                4 => Modifier::Ln,
                5 => Modifier::Ln1p,
                6 => Modifier::Ln2p,
                7 => Modifier::Square,
                8 => Modifier::Sqrt,
                9 => Modifier::Reciprocal,
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown field value modifier {other} (expected 0..=9)"
                    )))
                }
            };
            let has_missing = c.u8()? != 0;
            let missing = c.f64()?;
            Function::FieldValue {
                field,
                numeric,
                factor,
                modifier,
                missing: has_missing.then_some(missing),
            }
        }
        FUNCTION_RANDOM => {
            let salted_seed = c.i32()?;
            let field = match c.u8()? {
                0 => RandomField::DocId,
                1 => RandomField::Keyword(field(c)?),
                2 => RandomField::Long(field(c)?),
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown random score field kind {other} (expected 0..=2)"
                    )))
                }
            };
            Function::Random { salted_seed, field }
        }
        FUNCTION_DECAY => {
            let decay = match c.u8()? {
                0 => Decay::Gauss,
                1 => Decay::Exp,
                2 => Decay::Linear,
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown decay function {other} (expected 0..=2)"
                    )))
                }
            };
            let field = field(c)?;
            let numeric = numeric(c)?;
            let (origin, scale, offset) = (c.f64()?, c.f64()?, c.f64()?);
            let mode = match c.u8()? {
                0 => MultiValue::Sum,
                1 => MultiValue::Avg,
                2 => MultiValue::Median,
                3 => MultiValue::Min,
                4 => MultiValue::Max,
                other => {
                    return Err(invalid(format!(
                        "query tree: unknown multi-value mode {other} (expected 0..=4)"
                    )))
                }
            };
            Function::Decay {
                decay,
                field,
                numeric,
                origin,
                scale,
                offset,
                mode,
            }
        }
        other => {
            return Err(invalid(format!(
                "query tree: unknown score function {other} at depth {depth}"
            )))
        }
    })
}

/// The queries an extended clause runs beside its children: a function
/// score's filters.
pub(crate) fn queries(e: &ExtendedQuery) -> Vec<&Clause> {
    match e {
        ExtendedQuery::FunctionScore(q) => q.source.queries(),
        _ => Vec::new(),
    }
}

/// A function and the filter it applies under (`FilterScoreFunction`).
struct Filtered {
    filter: Option<Clause>,
    /// The filter's matches (`from_clause`): a document is in the filter where
    /// this has a value.
    source: Option<Arc<dyn DoubleValuesSource>>,
    function: Function,
}

/// OpenSearch's `FunctionFactorScorer.score()` as a values source over the
/// sub-query's scores.
pub(crate) struct OpenSearchFunctionScore {
    combine: Combine,
    score_mode: ScoreMode,
    max_boost: f32,
    functions: Vec<Filtered>,
    /// The node's bytes, hex: the source's identity.
    key: String,
}

impl DoubleValuesSource for OpenSearchFunctionScore {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let reader = ctx.leaf_reader(leaf)?;
        let mut functions = Vec::new();
        for f in &self.functions {
            let filter = match &f.source {
                Some(s) => Some(s.get_values(ctx, leaf, None)?),
                None => None,
            };
            functions.push((filter, LeafFunction::new(&f.function, reader)?));
        }
        Ok(Box::new(FunctionScoreValues {
            combine: self.combine,
            score_mode: self.score_mode,
            max_boost: self.max_boost,
            weights: self.functions.iter().map(|f| f.function.weight()).collect(),
            scores,
            functions,
            value: 0.0,
        }))
    }

    /// `subQueryScoreMode`: the sub-query scores unless the functions'
    /// score replaces it (no function here reads the score itself).
    fn needs_scores(&self) -> bool {
        self.combine != Combine::Replace
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        let mut fields = Vec::new();
        for f in &self.functions {
            f.function.fields(&mut fields);
        }
        fields
            .into_iter()
            .all(|f| lucene_search::values_source::doc_values_cacheable(ctx, leaf, f))
    }

    fn describe(&self) -> String {
        format!("opensearch_function_score({})", self.key)
    }

    /// `createWeight` passes its boost to `subQuery.createWeight`: under
    /// any boosted ancestor the sub-query's score is boosted, the combined
    /// score is not (and `replace` ignores it).
    fn boosts_wrapped_query(&self) -> bool {
        true
    }

    fn queries(&self) -> Vec<&Clause> {
        self.functions
            .iter()
            .filter_map(|f| f.filter.as_ref())
            .collect()
    }
}

/// One function's per-leaf state.
enum LeafFunction<'c> {
    One,
    Weight(f32, Box<LeafFunction<'c>>),
    FieldValue {
        values: Box<dyn SortedNumericDocValues + 'c>,
        field: String,
        numeric: Numeric,
        factor: f32,
        modifier: Modifier,
        missing: Option<f64>,
    },
    RandomDoc {
        doc_base: i32,
        salted_seed: i32,
    },
    RandomKeyword {
        values: Box<dyn SortedSetDocValues + 'c>,
        salted_seed: i32,
    },
    RandomLong {
        values: Box<dyn SortedNumericDocValues + 'c>,
        salted_seed: i32,
    },
    Decay {
        values: Box<dyn SortedNumericDocValues + 'c>,
        decay: Decay,
        numeric: Numeric,
        origin: f64,
        scale: f64,
        offset: f64,
        mode: MultiValue,
        distances: Vec<f64>,
    },
}

impl<'c> LeafFunction<'c> {
    fn new(
        f: &Function,
        reader: &'c lucene_search::directory_reader::SegmentReader,
    ) -> Result<Self> {
        Ok(match f {
            Function::One => LeafFunction::One,
            Function::Weight { weight, inner } => {
                LeafFunction::Weight(*weight, Box::new(LeafFunction::new(inner, reader)?))
            }
            Function::FieldValue {
                field,
                numeric,
                factor,
                modifier,
                missing,
            } => LeafFunction::FieldValue {
                values: get_sorted_numeric(reader, field)?,
                field: field.clone(),
                numeric: *numeric,
                factor: *factor,
                modifier: *modifier,
                missing: *missing,
            },
            Function::Random {
                salted_seed,
                field: RandomField::DocId,
            } => LeafFunction::RandomDoc {
                doc_base: reader.doc_base,
                salted_seed: *salted_seed,
            },
            Function::Random {
                salted_seed,
                field: RandomField::Keyword(field),
            } => LeafFunction::RandomKeyword {
                values: get_sorted_set(reader, field)?,
                salted_seed: *salted_seed,
            },
            Function::Random {
                salted_seed,
                field: RandomField::Long(field),
            } => LeafFunction::RandomLong {
                values: get_sorted_numeric(reader, field)?,
                salted_seed: *salted_seed,
            },
            Function::Decay {
                decay,
                field,
                numeric,
                origin,
                scale,
                offset,
                mode,
            } => LeafFunction::Decay {
                values: get_sorted_numeric(reader, field)?,
                decay: *decay,
                numeric: *numeric,
                origin: *origin,
                scale: *scale,
                offset: *offset,
                mode: *mode,
                distances: Vec::new(),
            },
        })
    }

    /// `LeafScoreFunction.score(docId, subQueryScore)`.
    fn score(&mut self, doc: i32) -> Result<f64> {
        Ok(match self {
            LeafFunction::One => 1.0,
            LeafFunction::Weight(w, inner) => inner.score(doc)? * f64::from(*w),
            LeafFunction::FieldValue {
                values,
                field,
                numeric,
                factor,
                modifier,
                missing,
            } => {
                let value = if values.advance_exact(doc)? {
                    numeric.decode(values.next_value()?)
                } else if let Some(m) = missing {
                    *m
                } else {
                    return Err(Error::IllegalArgument(format!(
                        "Missing value for field [{field}]"
                    )));
                };
                let result = modifier.apply(value * f64::from(*factor));
                if result < 0.0 {
                    return Err(Error::IllegalArgument(format!(
                        "field value function must not produce negative scores, but got: [{result}] for field value: [{value}]"
                    )));
                }
                result
            }
            LeafFunction::RandomDoc {
                doc_base,
                salted_seed,
            } => random(mix(doc_base.wrapping_add(doc), *salted_seed)),
            LeafFunction::RandomKeyword {
                values,
                salted_seed,
            } => random(if values.advance_exact(doc)? {
                let ord = values.next_ord()?;
                let term = values.lookup_ord(ord)?;
                lucene_util::string_helper::murmurhash3_x86_32(&term, *salted_seed)
            } else {
                *salted_seed
            }),
            LeafFunction::RandomLong {
                values,
                salted_seed,
            } => random(if values.advance_exact(doc)? {
                // `FieldData.toString`: every value as `Long.toString`, sorted
                // as bytes (`SortingBinaryDocValues`) -- the first is the
                // bytewise smallest string, not the smallest number.
                let mut first: Option<String> = None;
                for _ in 0..values.doc_value_count() {
                    let text = values.next_value()?.to_string();
                    if first
                        .as_ref()
                        .is_none_or(|f| text.as_bytes() < f.as_bytes())
                    {
                        first = Some(text);
                    }
                }
                let text = first.unwrap_or_default();
                lucene_util::string_helper::murmurhash3_x86_32(text.as_bytes(), *salted_seed)
            } else {
                *salted_seed
            }),
            LeafFunction::Decay {
                values,
                decay,
                numeric,
                origin,
                scale,
                offset,
                mode,
                distances,
            } => {
                let (decay, numeric, origin, scale, offset, mode) =
                    (*decay, *numeric, *origin, *scale, *offset, *mode);
                // `FieldData.replaceMissing(..., 0)`: no value is distance 0.
                let distance = if values.advance_exact(doc)? {
                    distances.clear();
                    for _ in 0..values.doc_value_count() {
                        let v = numeric.decode(values.next_value()?);
                        distances.push(java_max(0.0, (v - origin).abs() - offset));
                    }
                    distances.sort_by(f64::total_cmp);
                    if distances.is_empty() {
                        0.0
                    } else {
                        mode.pick(distances)
                    }
                } else {
                    0.0
                };
                decay.evaluate(distance, scale)
            }
        })
    }
}

/// HPPC's `BitMixer.mix(key, seed)`: `mix32(key ^ seed)`, MurmurHash3's
/// finalizer.
fn mix(key: i32, seed: i32) -> i32 {
    let mut k = (key ^ seed) as u32;
    k = (k ^ (k >> 16)).wrapping_mul(0x85eb_ca6b);
    k = (k ^ (k >> 13)).wrapping_mul(0xc2b2_ae35);
    (k ^ (k >> 16)) as i32
}

/// `(hash & 0x00FFFFFF) / (float) (1 << 24)`: a float in `[0, 1)`.
fn random(hash: i32) -> f64 {
    f64::from((hash & 0x00FF_FFFF) as f32 / (1u32 << 24) as f32)
}

struct FunctionScoreValues<'c> {
    combine: Combine,
    score_mode: ScoreMode,
    max_boost: f32,
    /// Each function's `getWeight()`.
    weights: Vec<f32>,
    scores: Option<BoxDoubleValues<'c>>,
    functions: Vec<(Option<BoxDoubleValues<'c>>, LeafFunction<'c>)>,
    value: f64,
}

impl FunctionScoreValues<'_> {
    /// `docSets[i].get(doc)`.
    fn in_filter(filter: &mut Option<BoxDoubleValues<'_>>, doc: i32) -> Result<bool> {
        match filter {
            Some(f) => f.advance_exact(doc),
            None => Ok(true),
        }
    }

    /// `FunctionFactorScorer.computeScore(doc, subQueryScore)`.
    fn compute(&mut self, doc: i32) -> Result<f64> {
        let mut factor = 1.0f64;
        match self.score_mode {
            ScoreMode::First => {
                for (filter, f) in &mut self.functions {
                    if Self::in_filter(filter, doc)? {
                        factor = f.score(doc)?;
                        break;
                    }
                }
            }
            ScoreMode::Max | ScoreMode::Min => {
                let max = self.score_mode == ScoreMode::Max;
                let mut best = if max {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                for (filter, f) in &mut self.functions {
                    if Self::in_filter(filter, doc)? {
                        let s = f.score(doc)?;
                        best = if max {
                            java_max(s, best)
                        } else {
                            java_min(s, best)
                        };
                    }
                }
                if best
                    != if max {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                {
                    factor = best;
                }
            }
            ScoreMode::Multiply => {
                for (filter, f) in &mut self.functions {
                    if Self::in_filter(filter, doc)? {
                        factor *= f.score(doc)?;
                    }
                }
            }
            ScoreMode::Avg | ScoreMode::Sum => {
                let (mut total, mut weights) = (0.0f64, 0.0f64);
                for ((filter, f), w) in self.functions.iter_mut().zip(&self.weights) {
                    if Self::in_filter(filter, doc)? {
                        total += f.score(doc)?;
                        weights += f64::from(*w);
                    }
                }
                if weights != 0.0 {
                    factor = total;
                    if self.score_mode == ScoreMode::Avg {
                        factor /= weights;
                    }
                }
            }
        }
        Ok(factor)
    }
}

impl DoubleValues for FunctionScoreValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        let sub = match (&mut self.scores, self.combine != Combine::Replace) {
            (Some(s), true) => {
                if s.advance_exact(doc)? {
                    s.double_value()? as f32
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        let factor = self.compute(doc)?;
        let score = self
            .combine
            .combine(f64::from(sub), factor, f64::from(self.max_boost));
        if score < 0.0 || score.is_nan() {
            return Err(Error::IllegalArgument(format!(
                "function score query returned an invalid score: {score} for doc: {doc}"
            )));
        }
        self.value = f64::from(score);
        Ok(true)
    }

    fn double_value(&mut self) -> Result<f64> {
        Ok(self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jvm_nodes::tests::{fails, index, open, search, Blob, BODIES};
    use crate::jvm_nodes::NODE_FUNCTION_SCORE;
    use crate::jvm_reader::ffi_close_jvm_reader;
    use std::collections::HashMap;

    /// Block `i`'s root document: after every earlier block and its own
    /// `i % 3` nested documents.
    fn root(i: usize) -> i32 {
        ((0..i).map(|k| k % 3 + 1).sum::<usize>() + i % 3) as i32
    }

    fn block_of(doc: i32) -> usize {
        (0..12).find(|&i| root(i) == doc).expect("a root")
    }

    /// The function score of `functions` (each an optional filter node and a
    /// function) over `body:alpha`.
    fn fs(combine: u8, mode: u8, max_boost: f32, functions: &[(Option<Blob>, Blob)]) -> Blob {
        let mut b = Blob::tree()
            .u8(NODE_FUNCTION_SCORE)
            .u8(combine)
            .u8(mode)
            .f32(max_boost)
            .term("body", "alpha")
            .i32(functions.len() as i32);
        for (filter, f) in functions {
            b = match filter {
                Some(node) => b.u8(1).raw(node),
                None => b.u8(0),
            }
            .raw(f);
        }
        b
    }

    fn weight(w: f32) -> Blob {
        Blob::default().u8(FUNCTION_WEIGHT).f32(w).u8(FUNCTION_ONE)
    }

    fn fvf(field: &str, numeric: u8, factor: f32, modifier: u8, missing: Option<f64>) -> Blob {
        Blob::default()
            .u8(FUNCTION_FIELD_VALUE)
            .str(field)
            .u8(numeric)
            .f32(factor)
            .u8(modifier)
            .u8(u8::from(missing.is_some()))
            .f64(missing.unwrap_or(0.0))
    }

    fn decay(
        kind: u8,
        field: &str,
        numeric: u8,
        origin: f64,
        scale: f64,
        offset: f64,
        mode: u8,
    ) -> Blob {
        Blob::default()
            .u8(FUNCTION_DECAY)
            .u8(kind)
            .str(field)
            .u8(numeric)
            .f64(origin)
            .f64(scale)
            .f64(offset)
            .u8(mode)
    }

    /// `body:alpha`'s scores by document.
    fn alpha(h: u64) -> HashMap<i32, f32> {
        search(h, &Blob::tree().term("body", "alpha"))
            .0
            .into_iter()
            .collect()
    }

    /// Every hit of `blob` scores `want(doc, sub_score)`, and they are
    /// `body:alpha`'s documents.
    fn scores_as(h: u64, blob: &Blob, want: impl Fn(i32, f32) -> f32) {
        let subs = alpha(h);
        let (hits, total) = search(h, blob);
        assert_eq!(total, subs.len() as i64);
        for (doc, score) in hits {
            let w = want(doc, subs[&doc]);
            assert_eq!(score.to_bits(), w.to_bits(), "doc {doc}: {score} vs {w}");
        }
    }

    #[test]
    fn every_boost_mode_combines_as_opensearch_does() {
        let tmp = index("jvm-fs-combine");
        let h = open(&tmp);
        for (combine, f) in [
            (0u8, (|q: f64, f: f64| q * f) as fn(f64, f64) -> f64),
            (1, |_, f| f),
            (2, |q, f| q + f),
            (3, |q, f| (f + q) / 2.0),
            (4, |q: f64, f: f64| q.min(f)),
            (5, |q: f64, f: f64| q.max(f)),
        ] {
            // weight 3 capped by max_boost 2.
            scores_as(h, &fs(combine, 0, 2.0, &[(None, weight(3.0))]), |_, s| {
                f(f64::from(s), 2.0) as f32
            });
        }
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn every_score_mode_folds_the_matching_functions() {
        let tmp = index("jvm-fs-modes");
        let h = open(&tmp);
        // Weight 2 on documents with "beta", weight 5 everywhere.
        let beta = Some(Blob::default().term("body", "beta"));
        let funcs = [(beta.clone(), weight(2.0)), (None, weight(5.0))];
        let has_beta = |doc: i32| BODIES[block_of(doc) % 6].contains("beta");
        for (mode, f) in [
            (
                0u8,
                (|b: bool| if b { 2.0 } else { 5.0 }) as fn(bool) -> f64,
            ),
            // (2 + 5) / (2 + 5), or 5 / 5.
            (1, |_| 1.0),
            (2, |_| 5.0),
            (3, |b| if b { 7.0 } else { 5.0 }),
            (4, |b| if b { 2.0 } else { 5.0 }),
            (5, |b| if b { 10.0 } else { 5.0 }),
        ] {
            scores_as(h, &fs(0, mode, f32::MAX, &funcs), |doc, s| {
                (f64::from(s) * f(has_beta(doc))) as f32
            });
        }
        // No function matches: the factor is 1 in every mode.
        let none = [(Some(Blob::default().term("body", "zzz")), weight(9.0))];
        for mode in 0..=5u8 {
            scores_as(h, &fs(0, mode, f32::MAX, &none), |_, s| s);
        }
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn field_value_factor_reads_each_numeric_type_through_its_modifier() {
        let tmp = index("jvm-fs-fvf");
        let h = open(&tmp);
        let pos = |doc: i32| block_of(doc) as f64 + 1.0;
        let modifiers: [fn(f64) -> f64; 10] = [
            |n| n,
            f64::log10,
            |n| (n + 1.0).log10(),
            |n| (n + 2.0).log10(),
            f64::ln,
            f64::ln_1p,
            |n| (n + 1.0).ln_1p(),
            |n| n * n,
            f64::sqrt,
            |n| 1.0 / n,
        ];
        for (m, f) in modifiers.iter().enumerate() {
            scores_as(
                h,
                &fs(1, 0, f32::MAX, &[(None, fvf("pos", 0, 1.5, m as u8, None))]),
                |doc, _| f(pos(doc) * 1.5) as f32,
            );
        }
        // A double and a float, squared (`sd` is negative on early blocks).
        let sd = |doc: i32| (block_of(doc) as f64 - 5.5) * 3.25;
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("sd", 1, 1.0, 7, None))]),
            |doc, _| (sd(doc) * sd(doc)) as f32,
        );
        let sf = |doc: i32| f64::from(block_of(doc) as f32 * 0.75 - 2.0);
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("sf", 2, 1.0, 7, None))]),
            |doc, _| (sf(doc) * sf(doc)) as f32,
        );
        // `si` is on even blocks only: its first value, or the missing value.
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("si", 0, 1.0, 0, Some(0.25)))]),
            |doc, _| {
                let i = block_of(doc);
                if i.is_multiple_of(2) {
                    i as f32
                } else {
                    0.25
                }
            },
        );
        // Without one, a missing value is OpenSearch's error; so is a negative score.
        let (rc, msg) = fails(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("si", 0, 1.0, 0, None))]),
        );
        assert_eq!(rc, FfiStatus::Search.code());
        assert!(msg.contains("Missing value for field [si]"), "{msg}");
        let (_, msg) = fails(
            h,
            &fs(1, 0, f32::MAX, &[(None, fvf("sd", 1, 1.0, 0, None))]),
        );
        assert!(msg.contains("must not produce negative scores"), "{msg}");
        let (_, msg) = fails(h, &fs(0, 0, f32::MAX, &[(None, weight(-1.0))]));
        assert!(msg.contains("invalid score"), "{msg}");
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn random_scores_hash_the_document_or_its_field() {
        let tmp = index("jvm-fs-random");
        let h = open(&tmp);
        let random_fn = |kind: u8, field: Option<&str>| {
            let b = Blob::default().u8(FUNCTION_RANDOM).i32(12345).u8(kind);
            match field {
                Some(f) => b.str(f),
                None => b,
            }
        };
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, random_fn(0, None))]),
            |doc, _| random(mix(doc, 12345)) as f32,
        );
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, random_fn(1, Some("kw")))]),
            |doc, _| {
                let term = format!("k{}", block_of(doc) % 4);
                random(lucene_util::string_helper::murmurhash3_x86_32(
                    term.as_bytes(),
                    12345,
                )) as f32
            },
        );
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, random_fn(2, Some("si")))]),
            |doc, _| {
                let i = block_of(doc);
                // `si` holds `i` and `3i + 1`; OpenSearch hashes the bytewise
                // smaller decimal string ("13" before "4" on block 4).
                let (a, b) = (i.to_string(), (3 * i + 1).to_string());
                let first = if a.as_bytes() <= b.as_bytes() { a } else { b };
                random(if i.is_multiple_of(2) {
                    lucene_util::string_helper::murmurhash3_x86_32(first.as_bytes(), 12345)
                } else {
                    12345
                }) as f32
            },
        );
        // A keyword with no value hashes the seed.
        scores_as(
            h,
            &fs(1, 0, f32::MAX, &[(None, random_fn(1, Some("nokw")))]),
            |_, _| random(12345) as f32,
        );
        // `BitMixer.mix` (HPPC 0.10), for a few keys and seeds.
        assert_eq!(mix(0, 0), 0);
        assert_eq!(mix(1, 0), 1_364_076_727);
        assert_eq!(mix(5, 42), mix(5 ^ 42, 0));
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn decay_functions_score_the_selected_distance() {
        let tmp = index("jvm-fs-decay");
        let h = open(&tmp);
        let evals: [fn(f64, f64) -> f64; 3] = [
            |v, s| (0.5 * (v * v) / s).exp(),
            |v, s| (s * v).exp(),
            |v, s| 0.0f64.max((s - v) / s),
        ];
        let (origin, offset) = (4.0, 0.5);
        for (kind, eval) in evals.iter().enumerate() {
            let scale = [-12.5, -0.1, 9.0][kind];
            for mode in 0..=4u8 {
                let blob = fs(
                    1,
                    0,
                    f32::MAX,
                    &[(
                        None,
                        decay(kind as u8, "si", 0, origin, scale, offset, mode),
                    )],
                );
                scores_as(h, &blob, |doc, _| {
                    let i = block_of(doc) as f64;
                    let d = if block_of(doc).is_multiple_of(2) {
                        let mut v = [
                            0.0f64.max((i - origin).abs() - offset),
                            0.0f64.max((3.0 * i + 1.0 - origin).abs() - offset),
                        ];
                        v.sort_by(f64::total_cmp);
                        match mode {
                            0 => v[0] + v[1],
                            1 => (v[0] + v[1]) / 2.0,
                            2 => (v[0] + v[1]) / 2.0,
                            3 => v[0],
                            _ => v[1],
                        }
                    } else {
                        0.0
                    };
                    eval(d, scale) as f32
                });
            }
        }
        // One value: the median is it.
        assert_eq!(MultiValue::Median.pick(&[1.0, 2.0, 7.0]), 2.0);
        assert_eq!(MultiValue::Avg.pick(&[1.0, 2.0]), 1.5);
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn java_min_and_max_keep_nan_and_signed_zeros() {
        assert!(java_min(f64::NAN, 1.0).is_nan());
        assert!(java_max(1.0, f64::NAN).is_nan());
        assert!(java_min(0.0, -0.0).is_sign_negative());
        assert!(java_min(-0.0, 0.0).is_sign_negative());
        assert!(java_max(-0.0, 0.0).is_sign_positive());
        assert!(java_max(0.0, -0.0).is_sign_positive());
        assert_eq!(java_min(1.0, 2.0), 1.0);
        assert_eq!(java_max(1.0, 2.0), 2.0);
        assert_eq!(Numeric::Long.decode(-3), -3.0);
    }

    #[test]
    fn the_source_describes_itself_and_its_filters() {
        let tmp = index("jvm-fs-source");
        let h = open(&tmp);
        let blob = fs(
            0,
            0,
            f32::MAX,
            &[(
                Some(Blob::default().term("body", "beta")),
                fvf("pos", 0, 1.0, 0, None),
            )],
        );
        let mut c = Cursor::new(&blob.0[1..]);
        let mut nodes = 0;
        let clause = decode_node(&mut c, 0, &mut nodes).unwrap();
        let Clause::Extended(e) = &clause else {
            panic!("{clause:?}")
        };
        assert_eq!(queries(e).len(), 1);
        assert!(
            queries(&ExtendedQuery::Span(lucene_search::spans::SpanNode::term(
                "f", "t"
            )))
            .is_empty()
        );
        let ExtendedQuery::FunctionScore(q) = e.as_ref() else {
            panic!()
        };
        assert!(q
            .source
            .describe()
            .starts_with("opensearch_function_score("));
        assert!(q.source.needs_scores());
        let dir = lucene_store::FsDirectory::open(tmp.path());
        let reader = lucene_search::directory_reader::DirectoryReader::open(&dir).unwrap();
        let ctx = ValuesContext::for_reader(&reader.segment_readers()[0]);
        assert!(q.source.is_cacheable(&ctx, 0));
        // Also as a filter, where the source is never asked.
        let filtered = Blob::tree()
            .u8(1)
            .i32(0)
            .i32(2)
            .u8(0)
            .term("body", "gamma")
            .u8(1)
            .raw(&Blob(blob.0[1..].to_vec()));
        assert!(search(h, &filtered).1 > 0);
        ffi_close_jvm_reader(h);
    }

    #[test]
    fn malformed_function_scores_are_refused() {
        let tmp = index("jvm-fs-bad");
        let h = open(&tmp);
        let bad = |b: Blob, want: &str| {
            let (rc, msg) = fails(h, &b);
            assert_eq!(rc, FfiStatus::InvalidArgument.code(), "{msg}");
            assert!(msg.contains(want), "{msg} lacks {want}");
        };
        let head = |combine: u8, mode: u8| {
            Blob::tree()
                .u8(NODE_FUNCTION_SCORE)
                .u8(combine)
                .u8(mode)
                .f32(1.0)
                .term("body", "alpha")
        };
        bad(head(9, 0).i32(1).u8(0).u8(0), "boost mode 9");
        bad(head(0, 9).i32(1).u8(0).u8(0), "function score mode 9");
        bad(head(0, 0).i32(0), "without functions");
        bad(
            fs(0, 0, 1.0, &[(None, Blob::default().u8(77))]),
            "unknown score function 77",
        );
        bad(
            fs(
                0,
                0,
                1.0,
                &[(
                    None,
                    Blob::default()
                        .u8(FUNCTION_WEIGHT)
                        .f32(1.0)
                        .raw(&weight(2.0)),
                )],
            ),
            "unknown score function 1 at depth 1",
        );
        bad(
            fs(0, 0, 1.0, &[(None, fvf("pos", 7, 1.0, 0, None))]),
            "numeric field type 7",
        );
        bad(
            fs(0, 0, 1.0, &[(None, fvf("pos", 0, 1.0, 42, None))]),
            "modifier 42",
        );
        bad(
            fs(
                0,
                0,
                1.0,
                &[(None, Blob::default().u8(FUNCTION_RANDOM).i32(1).u8(9))],
            ),
            "random score field kind 9",
        );
        bad(
            fs(0, 0, 1.0, &[(None, decay(5, "pos", 0, 0.0, 1.0, 0.0, 0))]),
            "decay function 5",
        );
        bad(
            fs(0, 0, 1.0, &[(None, decay(0, "pos", 0, 0.0, 1.0, 0.0, 8))]),
            "multi-value mode 8",
        );
        let utf = Blob::default().u8(FUNCTION_FIELD_VALUE).bytes(&[0xff]);
        assert_eq!(
            fails(h, &fs(0, 0, 1.0, &[(None, utf)])).0,
            FfiStatus::InvalidUtf8.code()
        );
        ffi_close_jvm_reader(h);
    }
}
