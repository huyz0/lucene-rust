//! OpenSearch's `function_score`
//! (`org.opensearch.common.lucene.search.function.FunctionScoreQuery`,
//! OpenSearch 3.8.0) as node `24` of the query tree ([`crate::nodes`]).
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
//! A node holding a Painless script function (tag `5`) never re-runs: those
//! errors carry the plugin's markers and fail the search as OpenSearch's
//! own exceptions, like the script's.
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

use crate::clauses::check_clause_count;
use crate::nodes::invalid;
use crate::painless_score::{opensearch_error, score_error, LeafScript, ScriptFunction};
use crate::query_blob::{decode_node, hex, Cursor};
use crate::status::Status;
use lucene_painless::number::double_to_string;

/// `CombineFunction`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Combine {
    Multiply,
    Replace,
    Sum,
    Avg,
    Min,
    Max,
}

impl Combine {
    /// `combine(queryScore, funcScore, maxBoost)`.
    pub fn combine(self, query: f64, func: f64, max_boost: f64) -> f32 {
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
pub fn java_min(a: f64, b: f64) -> f64 {
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
pub fn java_max(a: f64, b: f64) -> f64 {
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
pub enum ScoreMode {
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
pub enum Numeric {
    /// An integral type or a date: `(double) v`.
    Long,
    /// `double`: `NumericUtils.sortableLongToDouble`.
    Double,
    /// `float`: `NumericUtils.sortableIntToFloat`, widened.
    Float,
}

impl Numeric {
    pub fn decode(self, v: i64) -> f64 {
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
pub enum Modifier {
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
    pub fn apply(self, n: f64) -> f64 {
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
pub enum Decay {
    Gauss,
    Exp,
    Linear,
}

impl Decay {
    pub fn evaluate(self, value: f64, scale: f64) -> f64 {
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
pub enum MultiValue {
    Sum,
    Avg,
    Median,
    Min,
    Max,
}

impl MultiValue {
    pub fn pick(self, sorted: &[f64]) -> f64 {
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
pub enum RandomField {
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
pub enum Function {
    /// `WeightFactorFunction`'s `ScoreOne`: `1`.
    One,
    /// `WeightFactorFunction`: the inner function times `weight`.
    Weight { weight: f32, inner: Box<Function> },
    /// `FieldValueFactorFunction`.
    FieldValue {
        field: String,
        /// The field as the request names it (an alias's own name), in
        /// errors.
        name: String,
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
    /// `ScriptScoreFunction` over a Painless script the native runtime runs.
    Script(Arc<ScriptFunction>),
}

impl Function {
    /// `getWeight()`.
    pub fn weight(&self) -> f32 {
        match self {
            Function::Weight { weight, .. } => *weight,
            _ => 1.0,
        }
    }

    pub fn fields<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Function::Weight { inner, .. } => inner.fields(out),
            Function::FieldValue { field, .. } | Function::Decay { field, .. } => out.push(field),
            Function::Random {
                field: RandomField::Keyword(f) | RandomField::Long(f),
                ..
            } => out.push(f),
            Function::Script(s) => out.extend(s.fields()),
            Function::One | Function::Random { .. } => {}
        }
    }

    /// Where the scripts it is or wraps have their run numbers.
    pub fn script_ids(&self, out: &mut Vec<usize>) {
        match self {
            Function::Weight { inner, .. } => inner.script_ids(out),
            Function::Script(s) => out.push(s.script_id_at()),
            _ => {}
        }
    }

    /// Whether this is or wraps a script.
    pub fn scripted(&self) -> bool {
        match self {
            Function::Weight { inner, .. } => inner.scripted(),
            Function::Script(_) => true,
            _ => false,
        }
    }

    /// `needsScores()`: a script that reads `_score`.
    pub fn needs_score(&self) -> bool {
        match self {
            Function::Weight { inner, .. } => inner.needs_score(),
            Function::Script(s) => s.needs_score(),
            _ => false,
        }
    }
}

pub const FUNCTION_ONE: u8 = 0;
pub const FUNCTION_WEIGHT: u8 = 1;
pub const FUNCTION_FIELD_VALUE: u8 = 2;
pub const FUNCTION_RANDOM: u8 = 3;
pub const FUNCTION_DECAY: u8 = 4;
pub const FUNCTION_SCRIPT: u8 = 5;

/// The node (kind `24`, its byte read at `start`).
pub fn decode(
    c: &mut Cursor<'_>,
    start: usize,
    depth: usize,
    nodes: &mut usize,
) -> std::result::Result<Clause, Status> {
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
    let n = c.length()?;
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
    let scripted = functions.iter().any(|f| f.function.scripted());
    // The node's bytes, without its scripts' run numbers: two equal
    // functions compare equal wherever they sit in the request.
    let mut key = c.since(start).to_vec();
    let mut ids = Vec::new();
    for f in &functions {
        f.function.script_ids(&mut ids);
    }
    for at in ids {
        key[at - start..at - start + 4].fill(0);
    }
    let needs_sub =
        combine != Combine::Replace || functions.iter().any(|f| f.function.needs_score());
    let source = OpenSearchFunctionScore {
        combine,
        score_mode,
        max_boost,
        functions,
        scripted,
        needs_sub,
        key: hex(&key),
    };
    Ok(Clause::Extended(Box::new(ExtendedQuery::FunctionScore(
        FunctionScoreQuery::new(sub, Arc::new(source)),
    ))))
}

pub fn field(c: &mut Cursor<'_>) -> std::result::Result<String, Status> {
    Ok(std::str::from_utf8(c.bytes()?)
        .map_err(|_| Status::InvalidUtf8)?
        .to_string())
}

pub fn numeric(c: &mut Cursor<'_>) -> std::result::Result<Numeric, Status> {
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
/// | `2` | `field` (its doc values), `name` (the requested name, in errors), `numeric: u8` (`0` long, `1` double, `2` float), `factor: f32`, `modifier: u8` (`Modifier`'s ordinal), `has_missing: u8`, `missing: f64` | `FieldValueFactorFunction` |
/// | `3` | `salted_seed: i32`, `kind: u8` (`0` doc ID, `1` keyword + `field`, `2` long + `field`) | `RandomScoreFunction` |
/// | `4` | `decay: u8` (`0` gauss, `1` exp, `2` linear), `field`, `numeric: u8`, `origin`, `scale` (processed), `offset` (`f64`), `mode: u8` (`MultiValueMode`'s ordinal) | a numeric or date decay function |
/// | `5` | a Painless score script, checked ([`ScriptFunction::decode`]) | `ScriptScoreFunction` |
pub fn decode_function(c: &mut Cursor<'_>, depth: usize) -> std::result::Result<Function, Status> {
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
            let name = self::field(c)?;
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
                name,
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
        FUNCTION_SCRIPT => Function::Script(Arc::new(ScriptFunction::decode(c)?)),
        other => {
            return Err(invalid(format!(
                "query tree: unknown score function {other} at depth {depth}"
            )))
        }
    })
}

/// The queries an extended clause runs beside its children: a function
/// score's filters.
pub fn queries(e: &ExtendedQuery) -> Vec<&Clause> {
    match e {
        ExtendedQuery::FunctionScore(q) => q.source.queries(),
        _ => Vec::new(),
    }
}

/// A function and the filter it applies under (`FilterScoreFunction`).
pub struct Filtered {
    pub filter: Option<Clause>,
    /// The filter's matches (`from_clause`): a document is in the filter where
    /// this has a value.
    pub source: Option<Arc<dyn DoubleValuesSource>>,
    pub function: Function,
}

/// OpenSearch's `FunctionFactorScorer.score()` as a values source over the
/// sub-query's scores.
pub struct OpenSearchFunctionScore {
    pub combine: Combine,
    pub score_mode: ScoreMode,
    pub max_boost: f32,
    pub functions: Vec<Filtered>,
    /// Whether a function runs a script: the node's own errors then fail the
    /// search as OpenSearch's do instead of running it again in Java.
    pub scripted: bool,
    /// `subQueryScoreMode` needs scores: the combine reads the sub-query's
    /// score, or a script reads `_score`.
    pub needs_sub: bool,
    /// The node's bytes, hex: the source's identity.
    pub key: String,
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
            functions.push((filter, LeafFunction::new(&f.function, ctx, leaf, reader)?));
        }
        Ok(Box::new(FunctionScoreValues {
            scripted: self.scripted,
            needs_sub: self.needs_sub,
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
    /// score replaces it and no script reads `_score`.
    fn needs_scores(&self) -> bool {
        self.needs_sub
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
pub enum LeafFunction<'c> {
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
    Script(Box<LeafScript<'c>>),
}

impl<'c> LeafFunction<'c> {
    pub fn new(
        f: &Function,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        reader: &'c lucene_search::directory_reader::SegmentReader,
    ) -> Result<Self> {
        Ok(match f {
            Function::One => LeafFunction::One,
            Function::Weight { weight, inner } => LeafFunction::Weight(
                *weight,
                Box::new(LeafFunction::new(inner, ctx, leaf, reader)?),
            ),
            Function::Script(s) => LeafFunction::Script(Box::new(s.leaf(ctx, leaf)?)),
            Function::FieldValue {
                field,
                name,
                numeric,
                factor,
                modifier,
                missing,
            } => LeafFunction::FieldValue {
                values: get_sorted_numeric(reader, field)?,
                field: name.clone(),
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

    /// `LeafScoreFunction.score(docId, subQueryScore)`; `scripted` as the
    /// node's.
    pub fn score(&mut self, doc: i32, sub: f32, scripted: bool) -> Result<f64> {
        Ok(match self {
            LeafFunction::One => 1.0,
            LeafFunction::Weight(w, inner) => inner.score(doc, sub, scripted)? * f64::from(*w),
            LeafFunction::Script(s) => s.score(doc, sub)?,
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
                } else if scripted {
                    return Err(opensearch_error(format!(
                        "Missing value for field [{field}]"
                    )));
                } else {
                    return Err(Error::IllegalArgument(format!(
                        "Missing value for field [{field}]"
                    )));
                };
                let result = modifier.apply(value * f64::from(*factor));
                if result < 0.0 {
                    let (r, v) = (double_to_string(result), double_to_string(value));
                    let mut message = format!(
                        "field value function must not produce negative scores, but got: [{r}] for field value: [{v}]"
                    );
                    match modifier {
                        Modifier::Ln => message.push_str(
                            "; consider using ln1p or ln2p instead of ln to avoid negative scores",
                        ),
                        Modifier::Log => message.push_str(
                            "; consider using log1p or log2p instead of log to avoid negative scores",
                        ),
                        _ => {}
                    }
                    return Err(if scripted {
                        score_error(message)
                    } else {
                        Error::IllegalArgument(message)
                    });
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
pub fn mix(key: i32, seed: i32) -> i32 {
    let mut k = (key ^ seed) as u32;
    k = (k ^ (k >> 16)).wrapping_mul(0x85eb_ca6b);
    k = (k ^ (k >> 13)).wrapping_mul(0xc2b2_ae35);
    (k ^ (k >> 16)) as i32
}

/// `(hash & 0x00FFFFFF) / (float) (1 << 24)`: a float in `[0, 1)`.
pub fn random(hash: i32) -> f64 {
    f64::from((hash & 0x00FF_FFFF) as f32 / (1u32 << 24) as f32)
}

pub struct FunctionScoreValues<'c> {
    pub scripted: bool,
    pub needs_sub: bool,
    pub combine: Combine,
    pub score_mode: ScoreMode,
    pub max_boost: f32,
    /// Each function's `getWeight()`.
    pub weights: Vec<f32>,
    pub scores: Option<BoxDoubleValues<'c>>,
    pub functions: Vec<(Option<BoxDoubleValues<'c>>, LeafFunction<'c>)>,
    pub value: f64,
}

impl FunctionScoreValues<'_> {
    /// `docSets[i].get(doc)`.
    pub fn in_filter(filter: &mut Option<BoxDoubleValues<'_>>, doc: i32) -> Result<bool> {
        match filter {
            Some(f) => f.advance_exact(doc),
            None => Ok(true),
        }
    }

    /// `FunctionFactorScorer.computeScore(doc, subQueryScore)`.
    pub fn compute(&mut self, doc: i32, sub: f32) -> Result<f64> {
        let scripted = self.scripted;
        let mut factor = 1.0f64;
        match self.score_mode {
            ScoreMode::First => {
                for (filter, f) in &mut self.functions {
                    if Self::in_filter(filter, doc)? {
                        factor = f.score(doc, sub, scripted)?;
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
                        let s = f.score(doc, sub, scripted)?;
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
                        factor *= f.score(doc, sub, scripted)?;
                    }
                }
            }
            ScoreMode::Avg | ScoreMode::Sum => {
                let (mut total, mut weights) = (0.0f64, 0.0f64);
                for ((filter, f), w) in self.functions.iter_mut().zip(&self.weights) {
                    if Self::in_filter(filter, doc)? {
                        total += f.score(doc, sub, scripted)?;
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
        let sub = match (&mut self.scores, self.needs_sub) {
            (Some(s), true) => {
                if s.advance_exact(doc)? {
                    s.double_value()? as f32
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        let factor = self.compute(doc, sub)?;
        let score = self
            .combine
            .combine(f64::from(sub), factor, f64::from(self.max_boost));
        if score < 0.0 || score.is_nan() {
            let message = format!(
                "function score query returned an invalid score: {} for doc: {doc}",
                lucene_painless::number::float_to_string(score)
            );
            return Err(if self.scripted {
                opensearch_error(message)
            } else {
                Error::IllegalArgument(message)
            });
        }
        self.value = f64::from(score);
        Ok(true)
    }

    fn double_value(&mut self) -> Result<f64> {
        Ok(self.value)
    }
}
