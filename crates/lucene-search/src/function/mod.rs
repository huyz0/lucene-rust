//! `lucene-queries`' `function` package: per-document values computed by a
//! tree of [`ValueSource`]s (fields, constants, arithmetic, term statistics,
//! a query's scores, vector similarities), the [`FunctionValues`] a source
//! hands out per segment, and the queries built on them --
//! [`FunctionQuery`], [`FunctionRangeQuery`], [`FunctionMatchQuery`],
//! [`FunctionScoreQuery`] and the reader-wide sources of
//! [`index_reader_functions`].
//!
//! # Shape
//!
//! * [`ValueSource`] is a trait object, as Java's abstract class is: a tree
//!   of `Arc<dyn ValueSource>`. Its `getValues(context, readerContext)` is
//!   [`ValueSource::get_values`], with Java's `Map<Object,Object> context`
//!   as a [`FunctionContext`] and the `LeafReaderContext` as a
//!   [`ValueLeaf`].
//! * [`FunctionValues`] is a trait object too. Java's typed abstract bases
//!   (`FloatDocValues`, `IntDocValues`, ...) are the [`docvalues`] traits,
//!   each turned into a [`FunctionValues`] by a wrapper ([`docvalues::Float`],
//!   [`docvalues::Int`], ...), which supplies the base's derived getters.
//! * `MutableValue` and its seven subclasses are one enum, [`MutableValue`];
//!   a `ValueFiller` is [`FunctionValues::new_value`] (its `getValue()`'s
//!   empty value) with [`FunctionValues::fill_value`] (`fillValue(doc)`).
//! * `getRangeScorer` returns a [`RangeMatcher`] (the `matches(doc)` of the
//!   `ValueSourceScorer` each base builds), and [`ValueSourceScorer`] runs it.
//!
//! # Reader-wide state: `createWeight`
//!
//! Java's sources read reader-wide facts in two ways: through the searcher
//! the context carries (`DocFreqValueSource`, `MaxDocValueSource`,
//! `IDFValueSource`, `NormValueSource` and `TFValueSource` read
//! `context.get("searcher")` in `getValues`; `NumDocsValueSource`,
//! `JoinDocFreqValueSource` and `ScaleFloatFunction` reach the top-level
//! reader from the leaf), or from what their `createWeight` put in the
//! context (`TotalTermFreqValueSource`, `SumTotalTermFreqValueSource`,
//! `QueryValueSource`). A segment here cannot reach its siblings, so every
//! one of these facts is computed by [`ValueSource::create_weight`] over a
//! [`TopLevel`] (Java's searcher and top-level reader) and kept in the
//! [`FunctionContext`], keyed by the source (Java's `IdentityHashMap`). A
//! similarity is the one fact still read at `getValues` time, from the
//! leaf ([`ValueLeaf::similarity`]), as Java reads `searcher.getSimilarity()`
//! there. `getValues` on a source whose state `createWeight` never computed
//! is [`Error::IllegalState`] (Java computes some of it lazily from the
//! searcher instead).
//!
//! The queries compute their contexts once per search, with the other
//! reader-wide statistics ([`crate::multi_segment::global_boolean_stats`]),
//! as Java's weights do once per `createWeight`; a query searched without
//! them (one segment, no statistics pass) computes its context from the
//! segment alone, as if it were the whole index.
//!
//! # Deviations
//!
//! * `equals` compares [`ValueSource::description`]s; `hashCode` is not
//!   ported.
//! * Java's `toString(int doc)` is [`FunctionValues::to_string_doc`].
//! * Byte vectors are `Vec<u8>` holding Java's `byte` bit patterns, printed
//!   signed as `Arrays.toString(byte[])` prints them.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::exec::LeafContext;
use crate::explain::Explanation;
use crate::similarities::Similarity;
use crate::{Error, Result};

pub mod docvalues;
pub mod index_reader_functions;
mod queries;
pub mod valuesource;

#[cfg(test)]
mod tests;

pub(crate) use queries::ScorableView;
pub use queries::{
    as_double_values_source, as_long_values_source, from_double_values_source, sort_field,
    DoublePredicate, FunctionMatchQuery, FunctionQuery, FunctionRangeQuery, FunctionScoreQuery,
    ValueSourceScorer, DEFAULT_MATCH_COST,
};

/// A boxed per-segment [`FunctionValues`].
pub type BoxValues<'a> = Box<dyn FunctionValues + 'a>;

/// `docs were sent out-of-order` (the `IllegalArgumentException` a
/// doc-values-backed [`FunctionValues`] throws when asked for an earlier
/// document than the last).
pub(crate) fn out_of_order(last: i32, doc: i32) -> Error {
    Error::IllegalArgument(format!(
        "docs were sent out-of-order: lastDocID={last} vs docID={doc}"
    ))
}

/// Java's `UnsupportedOperationException` from a getter a
/// [`FunctionValues`] does not implement.
pub(crate) fn unsupported() -> Error {
    Error::Unsupported("java.lang.UnsupportedOperationException".into())
}

// ---------------------------------------------------------------------------
// Java's number formatting and parsing
// ---------------------------------------------------------------------------

/// `Float.toString(f)`: the shortest decimal that round-trips, with a decimal
/// point, in computerized scientific notation outside `[1e-3, 1e7)`.
pub fn java_float(v: f32) -> String {
    if v.is_subnormal() {
        if let Some(s) = java_subnormal_float(v) {
            return s;
        }
    }
    java_fp(f64::from(v), format!("{v:?}"), format!("{v:e}"))
}

/// `Float.toString` of a subnormal whose shortest decimal has one digit:
/// Java then chooses among the two-digit decimals that round to it, the
/// closest (`4.1E-44`, where the shortest is `4E-44`); `None` otherwise.
fn java_subnormal_float(v: f32) -> Option<String> {
    let sci = format!("{:e}", v.abs());
    let (mantissa, exp) = sci.split_once('e')?;
    if mantissa.contains('.') {
        return None;
    }
    let d: i32 = mantissa.parse().ok()?;
    let e: i32 = exp.parse().ok()?;
    let x = f64::from(v.abs());
    let mut best: Option<(i32, f64)> = None;
    for m in (d * 10 - 9).max(10)..=(d * 10 + 9).min(99) {
        let cand = format!("{m}e{}", e - 1);
        if cand.parse::<f32>().ok() != Some(v.abs()) {
            continue;
        }
        let diff = (m as f64 * 10f64.powi(e - 1) - x).abs();
        let better = match best {
            None => true,
            Some((bm, bd)) => diff < bd || (diff == bd && m % 2 == 0 && bm % 2 != 0),
        };
        if better {
            best = Some((m, diff));
        }
    }
    let (m, _) = best?;
    let sign = if v < 0.0 { "-" } else { "" };
    Some(format!("{sign}{}.{}E{e}", m / 10, m % 10))
}

/// `Double.toString(d)`, as [`java_float`].
pub fn java_double(v: f64) -> String {
    java_fp(v, format!("{v:?}"), format!("{v:e}"))
}

fn java_fp(v: f64, plain: String, sci: String) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let a = v.abs();
    if a == 0.0 || (1e-3..1e7).contains(&a) {
        // `{:?}` is the shortest round trip, always with a `.0` -- except
        // for a large value it may print without one in exponent form;
        // within this range it never does.
        return plain;
    }
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    if mantissa.contains('.') {
        format!("{mantissa}E{exp}")
    } else {
        format!("{mantissa}.0E{exp}")
    }
}

/// `NumberFormatException` (an `IllegalArgumentException`).
fn number_format(s: &str) -> Error {
    Error::IllegalArgument(format!("For input string: \"{s}\""))
}

/// The decimal (or `NaN`/`Infinity`) Java's `FloatingDecimal.readJavaFormatString`
/// accepts: surrounding whitespace trimmed, an optional sign, and an optional
/// `f`/`F`/`d`/`D` suffix. Hexadecimal floats are not accepted.
fn java_fp_body(s: &str) -> Result<(bool, String)> {
    let t = s.trim_matches(|c: char| c <= ' ');
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if body == "NaN" || body == "Infinity" {
        return Ok((neg, body.to_string()));
    }
    let body = body.strip_suffix(['f', 'F', 'd', 'D']).unwrap_or(body);
    let valid = !body.is_empty()
        && body.bytes().any(|b| b.is_ascii_digit())
        && body
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'));
    if !valid {
        return Err(number_format(s));
    }
    Ok((neg, body.to_string()))
}

/// `Float.parseFloat(s)`.
///
/// # Errors
/// [`Error::IllegalArgument`] (`NumberFormatException`).
pub fn parse_java_float(s: &str) -> Result<f32> {
    let (neg, body) = java_fp_body(s)?;
    let v = match body.as_str() {
        "NaN" => f32::NAN,
        "Infinity" => f32::INFINITY,
        b => b.parse::<f32>().map_err(|_| number_format(s))?,
    };
    Ok(if neg { -v } else { v })
}

/// `Double.parseDouble(s)`.
///
/// # Errors
/// [`Error::IllegalArgument`] (`NumberFormatException`).
pub fn parse_java_double(s: &str) -> Result<f64> {
    let (neg, body) = java_fp_body(s)?;
    let v = match body.as_str() {
        "NaN" => f64::NAN,
        "Infinity" => f64::INFINITY,
        b => b.parse::<f64>().map_err(|_| number_format(s))?,
    };
    Ok(if neg { -v } else { v })
}

/// `Integer.parseInt(s)`.
///
/// # Errors
/// [`Error::IllegalArgument`] (`NumberFormatException`).
pub fn parse_java_int(s: &str) -> Result<i32> {
    s.parse::<i32>().map_err(|_| number_format(s))
}

/// `Long.parseLong(s)`.
///
/// # Errors
/// [`Error::IllegalArgument`] (`NumberFormatException`).
pub fn parse_java_long(s: &str) -> Result<i64> {
    s.parse::<i64>().map_err(|_| number_format(s))
}

/// `Arrays.toString(float[])`.
pub fn java_float_array(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|&f| java_float(f)).collect();
    format!("[{}]", parts.join(", "))
}

/// `Arrays.toString(byte[])`: each byte as Java's signed `byte`.
pub fn java_byte_array(v: &[u8]) -> String {
    let parts: Vec<String> = v.iter().map(|&b| (b as i8).to_string()).collect();
    format!("[{}]", parts.join(", "))
}

// ---------------------------------------------------------------------------
// Values: objectVal, MutableValue
// ---------------------------------------------------------------------------

/// What `FunctionValues.objectVal(doc)` returns: a boxed number, string or
/// boolean, or `null`.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectVal {
    Null,
    Bool(bool),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Str(String),
}

impl fmt::Display for ObjectVal {
    /// `String.valueOf(object)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectVal::Null => f.write_str("null"),
            ObjectVal::Bool(b) => write!(f, "{b}"),
            ObjectVal::Int(i) => write!(f, "{i}"),
            ObjectVal::Long(l) => write!(f, "{l}"),
            ObjectVal::Float(v) => f.write_str(&java_float(*v)),
            ObjectVal::Double(v) => f.write_str(&java_double(*v)),
            ObjectVal::Str(s) => f.write_str(s),
        }
    }
}

/// `org.apache.lucene.util.mutable.MutableValue` and its subclasses
/// (`MutableValueBool`, `...Int`, `...Long`, `...Float`, `...Double`,
/// `...Str`, `...Date`): a value and whether it exists.
///
/// Equality is Java's `equals`: the same kind, the same `exists`, and the
/// same value by `==` -- so a `NaN` equals nothing (not even itself), as
/// Java's `value == b.value` says; `0.0` and `-0.0` are unequal here where
/// Java's `==` calls them equal, because Java's `hashCode`
/// (`floatToIntBits`) puts them in different buckets of every hash map
/// grouping uses, so they never meet.
#[derive(Debug, Clone)]
pub enum MutableValue {
    Bool {
        value: bool,
        exists: bool,
    },
    Int {
        value: i32,
        exists: bool,
    },
    Long {
        value: i64,
        exists: bool,
    },
    Float {
        value: f32,
        exists: bool,
    },
    Double {
        value: f64,
        exists: bool,
    },
    Str {
        value: Vec<u8>,
        exists: bool,
    },
    /// `MutableValueDate`: a `long` of milliseconds.
    Date {
        value: i64,
        exists: bool,
    },
}

impl MutableValue {
    /// A fresh `MutableValueFloat` (`exists` true, value `0`).
    pub fn float() -> Self {
        MutableValue::Float {
            value: 0.0,
            exists: true,
        }
    }

    /// `exists()`.
    pub fn exists(&self) -> bool {
        match self {
            MutableValue::Bool { exists, .. }
            | MutableValue::Int { exists, .. }
            | MutableValue::Long { exists, .. }
            | MutableValue::Float { exists, .. }
            | MutableValue::Double { exists, .. }
            | MutableValue::Str { exists, .. }
            | MutableValue::Date { exists, .. } => *exists,
        }
    }

    /// `toObject()`: the value, or `null` when it does not exist.
    pub fn to_object(&self) -> ObjectVal {
        if !self.exists() {
            return ObjectVal::Null;
        }
        match self {
            MutableValue::Bool { value, .. } => ObjectVal::Bool(*value),
            MutableValue::Int { value, .. } => ObjectVal::Int(*value),
            MutableValue::Long { value, .. } | MutableValue::Date { value, .. } => {
                ObjectVal::Long(*value)
            }
            MutableValue::Float { value, .. } => ObjectVal::Float(*value),
            MutableValue::Double { value, .. } => ObjectVal::Double(*value),
            MutableValue::Str { value, .. } => {
                ObjectVal::Str(String::from_utf8_lossy(value).into_owned())
            }
        }
    }

    /// `duplicate()`.
    pub fn duplicate(&self) -> Self {
        self.clone()
    }

    /// `compareSameType(other)` (`compareTo` across kinds is Java's
    /// `ClassCastException`, here `None`): a missing value sorts first.
    pub fn compare_same_type(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        fn by<T>(a: (T, bool), b: (T, bool), cmp: impl Fn(&T, &T) -> Ordering) -> Ordering {
            let c = cmp(&a.0, &b.0);
            if c != Ordering::Equal {
                return c;
            }
            if a.1 == b.1 {
                Ordering::Equal
            } else if a.1 {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        Some(match (self, other) {
            (
                MutableValue::Bool { value, exists },
                MutableValue::Bool {
                    value: v2,
                    exists: e2,
                },
            ) => {
                // `MutableValueBool.compareSameType`: `value == b.value ?
                // exists compare : value ? 1 : -1`.
                if value != v2 {
                    if *value {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                } else {
                    by(((), *exists), ((), *e2), |_, _| Ordering::Equal)
                }
            }
            (
                MutableValue::Int { value, exists },
                MutableValue::Int {
                    value: v2,
                    exists: e2,
                },
            ) => by((*value, *exists), (*v2, *e2), Ord::cmp),
            (
                MutableValue::Long { value, exists },
                MutableValue::Long {
                    value: v2,
                    exists: e2,
                },
            )
            | (
                MutableValue::Date { value, exists },
                MutableValue::Date {
                    value: v2,
                    exists: e2,
                },
            ) => by((*value, *exists), (*v2, *e2), Ord::cmp),
            (
                MutableValue::Float { value, exists },
                MutableValue::Float {
                    value: v2,
                    exists: e2,
                },
            ) => by((*value, *exists), (*v2, *e2), |a, b| {
                java_float_compare(*a, *b)
            }),
            (
                MutableValue::Double { value, exists },
                MutableValue::Double {
                    value: v2,
                    exists: e2,
                },
            ) => by((*value, *exists), (*v2, *e2), |a, b| {
                java_double_compare(*a, *b)
            }),
            (
                MutableValue::Str { value, exists },
                MutableValue::Str {
                    value: v2,
                    exists: e2,
                },
            ) => by((value, *exists), (v2, *e2), |a, b| a.cmp(b)),
            _ => return None,
        })
    }

    fn kind(&self) -> u8 {
        match self {
            MutableValue::Bool { .. } => 0,
            MutableValue::Int { .. } => 1,
            MutableValue::Long { .. } => 2,
            MutableValue::Float { .. } => 3,
            MutableValue::Double { .. } => 4,
            MutableValue::Str { .. } => 5,
            MutableValue::Date { .. } => 6,
        }
    }
}

/// `Float.compare`.
pub(crate) fn java_float_compare(a: f32, b: f32) -> std::cmp::Ordering {
    fn key(v: f32) -> i32 {
        let bits = if v.is_nan() {
            0x7fc0_0000_u32 as i32
        } else {
            v.to_bits() as i32
        };
        bits ^ (((bits >> 31) as u32) >> 1) as i32
    }
    key(a).cmp(&key(b))
}

/// `Double.compare`.
pub(crate) fn java_double_compare(a: f64, b: f64) -> std::cmp::Ordering {
    fn key(v: f64) -> i64 {
        let bits = if v.is_nan() {
            0x7ff8_0000_0000_0000_u64 as i64
        } else {
            v.to_bits() as i64
        };
        bits ^ (((bits >> 63) as u64) >> 1) as i64
    }
    key(a).cmp(&key(b))
}

impl PartialEq for MutableValue {
    fn eq(&self, other: &Self) -> bool {
        if self.kind() != other.kind() || self.exists() != other.exists() {
            return false;
        }
        match (self, other) {
            (MutableValue::Bool { value: a, .. }, MutableValue::Bool { value: b, .. }) => a == b,
            (MutableValue::Int { value: a, .. }, MutableValue::Int { value: b, .. }) => a == b,
            (MutableValue::Long { value: a, .. }, MutableValue::Long { value: b, .. })
            | (MutableValue::Date { value: a, .. }, MutableValue::Date { value: b, .. }) => a == b,
            (MutableValue::Float { value: a, .. }, MutableValue::Float { value: b, .. }) => {
                !a.is_nan() && a.to_bits() == b.to_bits()
            }
            (MutableValue::Double { value: a, .. }, MutableValue::Double { value: b, .. }) => {
                !a.is_nan() && a.to_bits() == b.to_bits()
            }
            (MutableValue::Str { value: a, .. }, MutableValue::Str { value: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl Eq for MutableValue {}

impl std::hash::Hash for MutableValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u8(self.kind());
        match self {
            MutableValue::Bool { value, .. } => value.hash(state),
            MutableValue::Int { value, .. } => value.hash(state),
            MutableValue::Long { value, .. } | MutableValue::Date { value, .. } => {
                value.hash(state)
            }
            MutableValue::Float { value, .. } => value.to_bits().hash(state),
            MutableValue::Double { value, .. } => value.to_bits().hash(state),
            MutableValue::Str { value, .. } => value.hash(state),
        }
    }
}

impl fmt::Display for MutableValue {
    /// `MutableValue.toString()`: the value, or `(null)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exists() {
            write!(f, "{}", self.to_object())
        } else {
            f.write_str("(null)")
        }
    }
}

// ---------------------------------------------------------------------------
// FunctionValues
// ---------------------------------------------------------------------------

/// `FunctionValues.getRangeScorer`'s `matches(doc)`: which value it reads
/// and the bounds it compares with, as each typed base builds it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RangeMatcher {
    /// `FunctionValues`' own: `floatVal` against float bounds.
    Float {
        lower: f32,
        upper: f32,
        include_lower: bool,
        include_upper: bool,
    },
    /// `DoubleDocValues`: `doubleVal` against double bounds.
    Double {
        lower: f64,
        upper: f64,
        include_lower: bool,
        include_upper: bool,
    },
    /// `IntDocValues` (and `EnumFieldSource`): `intVal` in `[lower, upper]`
    /// (the bounds already adjusted for exclusivity).
    Int { lower: i32, upper: i32 },
    /// `LongDocValues`: `longVal` in `[lower, upper]`.
    Long { lower: i64, upper: i64 },
    /// `DocTermsIndexDocValues`: the ordinal, as a `float`, in
    /// `[lower, upper]`.
    Ord { lower: i32, upper: i32 },
}

impl RangeMatcher {
    /// `FunctionValues.getRangeScorer`'s float bounds.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an unparseable bound.
    pub fn float(
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<Self> {
        Ok(RangeMatcher::Float {
            lower: lower.map_or(Ok(f32::NEG_INFINITY), parse_java_float)?,
            upper: upper.map_or(Ok(f32::INFINITY), parse_java_float)?,
            include_lower,
            include_upper,
        })
    }

    /// `DoubleDocValues.getRangeScorer`'s bounds.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an unparseable bound.
    pub fn double(
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<Self> {
        Ok(RangeMatcher::Double {
            lower: lower.map_or(Ok(f64::NEG_INFINITY), parse_java_double)?,
            upper: upper.map_or(Ok(f64::INFINITY), parse_java_double)?,
            include_lower,
            include_upper,
        })
    }

    /// `IntDocValues.getRangeScorer`'s bounds, from already-parsed
    /// endpoints (`None` unbounded).
    pub fn int(
        lower: Option<i32>,
        upper: Option<i32>,
        include_lower: bool,
        include_upper: bool,
    ) -> Self {
        let lower = match lower {
            None => i32::MIN,
            Some(l) if !include_lower && l < i32::MAX => l + 1,
            Some(l) => l,
        };
        let upper = match upper {
            None => i32::MAX,
            Some(u) if !include_upper && u > i32::MIN => u - 1,
            Some(u) => u,
        };
        RangeMatcher::Int { lower, upper }
    }

    /// `LongDocValues.getRangeScorer`'s bounds, from already-parsed
    /// endpoints.
    pub fn long(
        lower: Option<i64>,
        upper: Option<i64>,
        include_lower: bool,
        include_upper: bool,
    ) -> Self {
        let lower = match lower {
            None => i64::MIN,
            Some(l) if !include_lower && l < i64::MAX => l + 1,
            Some(l) => l,
        };
        let upper = match upper {
            None => i64::MAX,
            Some(u) if !include_upper && u > i64::MIN => u - 1,
            Some(u) => u,
        };
        RangeMatcher::Long { lower, upper }
    }

    /// `ValueSourceScorer.matches(doc)`: whether `doc` has a value in range.
    ///
    /// # Errors
    /// Whatever reading the value reports.
    pub fn matches(&self, values: &mut dyn FunctionValues, doc: i32) -> Result<bool> {
        if !values.exists(doc)? {
            return Ok(false);
        }
        Ok(match *self {
            RangeMatcher::Float {
                lower,
                upper,
                include_lower,
                include_upper,
            } => {
                let v = values.float_val(doc)?;
                (if include_lower { v >= lower } else { v > lower })
                    && (if include_upper { v <= upper } else { v < upper })
            }
            RangeMatcher::Double {
                lower,
                upper,
                include_lower,
                include_upper,
            } => {
                let v = values.double_val(doc)?;
                (if include_lower { v >= lower } else { v > lower })
                    && (if include_upper { v <= upper } else { v < upper })
            }
            RangeMatcher::Int { lower, upper } => {
                let v = values.int_val(doc)?;
                v >= lower && v <= upper
            }
            RangeMatcher::Long { lower, upper } => {
                let v = values.long_val(doc)?;
                v >= lower && v <= upper
            }
            RangeMatcher::Ord { lower, upper } => {
                // `float docVal = ordVal(doc); docVal >= ll && docVal <= uu`.
                let v = values.ord_val(doc)? as f32;
                v >= lower as f32 && v <= upper as f32
            }
        })
    }
}

/// `FunctionValues`: one segment's values of a [`ValueSource`]. Documents
/// are asked for in non-decreasing order (most values are doc-values
/// iterators underneath, and fail with [`Error::IllegalArgument`] when
/// sent backwards). Every getter a kind of values lacks is Java's
/// `UnsupportedOperationException`, here [`Error::Unsupported`].
pub trait FunctionValues {
    /// `byteVal(doc)`.
    fn byte_val(&mut self, _doc: i32) -> Result<i8> {
        Err(unsupported())
    }
    /// `shortVal(doc)`.
    fn short_val(&mut self, _doc: i32) -> Result<i16> {
        Err(unsupported())
    }
    /// `floatVal(doc)`.
    fn float_val(&mut self, _doc: i32) -> Result<f32> {
        Err(unsupported())
    }
    /// `intVal(doc)`.
    fn int_val(&mut self, _doc: i32) -> Result<i32> {
        Err(unsupported())
    }
    /// `longVal(doc)`.
    fn long_val(&mut self, _doc: i32) -> Result<i64> {
        Err(unsupported())
    }
    /// `doubleVal(doc)`.
    fn double_val(&mut self, _doc: i32) -> Result<f64> {
        Err(unsupported())
    }
    /// `strVal(doc)`; `None` is Java's `null`.
    fn str_val(&mut self, _doc: i32) -> Result<Option<String>> {
        Err(unsupported())
    }
    /// `boolVal(doc)`: `intVal(doc) != 0`.
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        Ok(self.int_val(doc)? != 0)
    }
    /// `floatVectorVal(doc)`.
    fn float_vector_val(&mut self, _doc: i32) -> Result<Option<Vec<f32>>> {
        Err(unsupported())
    }
    /// `byteVectorVal(doc)`.
    fn byte_vector_val(&mut self, _doc: i32) -> Result<Option<Vec<u8>>> {
        Err(unsupported())
    }
    /// `bytesVal(doc, target)`: the value's bytes into `target` (cleared),
    /// and whether there is one -- by default `strVal`'s UTF-8.
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        target.clear();
        match self.str_val(doc)? {
            Some(s) => {
                target.extend_from_slice(s.as_bytes());
                Ok(true)
            }
            None => Ok(false),
        }
    }
    /// `objectVal(doc)`: by default `floatVal` boxed.
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(ObjectVal::Float(self.float_val(doc)?))
    }
    /// `exists(doc)`: whether the document has a value; by default `true`.
    fn exists(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    /// `ordVal(doc)`.
    fn ord_val(&mut self, _doc: i32) -> Result<i32> {
        Err(unsupported())
    }
    /// `numOrd()`.
    fn num_ord(&self) -> Result<i32> {
        Err(unsupported())
    }
    /// `cost()`: an estimate of the cost of one value.
    fn cost(&self) -> f32 {
        100.0
    }
    /// `toString(doc)`.
    fn to_string_doc(&mut self, doc: i32) -> Result<String>;
    /// `getValueFiller().getValue()` before any fill: the mutable value this
    /// kind fills (a `MutableValueFloat` by default).
    fn new_value(&self) -> MutableValue {
        MutableValue::float()
    }
    /// `getValueFiller().fillValue(doc)` into `out` (a value from
    /// [`Self::new_value`]); by default `floatVal`, `exists` untouched.
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        let v = self.float_val(doc)?;
        if let MutableValue::Float { value, .. } = out {
            *value = v;
        }
        Ok(())
    }
    /// The multi-valued getters (`byteVal(doc, vals)` and the rest), for a
    /// [`valuesource::VectorValueSource`]: fills `vals`.
    fn byte_vals(&mut self, _doc: i32, _vals: &mut [i8]) -> Result<()> {
        Err(unsupported())
    }
    fn short_vals(&mut self, _doc: i32, _vals: &mut [i16]) -> Result<()> {
        Err(unsupported())
    }
    fn float_vals(&mut self, _doc: i32, _vals: &mut [f32]) -> Result<()> {
        Err(unsupported())
    }
    fn int_vals(&mut self, _doc: i32, _vals: &mut [i32]) -> Result<()> {
        Err(unsupported())
    }
    fn long_vals(&mut self, _doc: i32, _vals: &mut [i64]) -> Result<()> {
        Err(unsupported())
    }
    fn double_vals(&mut self, _doc: i32, _vals: &mut [f64]) -> Result<()> {
        Err(unsupported())
    }
    fn str_vals(&mut self, _doc: i32, _vals: &mut [Option<String>]) -> Result<()> {
        Err(unsupported())
    }
    /// `explain(doc)`: `floatVal` described by `toString(doc)`.
    fn explain(&mut self, doc: i32) -> Result<Explanation> {
        let v = self.float_val(doc)?;
        Ok(Explanation::match_(v, self.to_string_doc(doc)?))
    }
    /// `getRangeScorer(context, lowerVal, upperVal, includeLower,
    /// includeUpper)`: by default `floatVal` against float bounds.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an unparseable bound.
    fn range_matcher(
        &mut self,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<RangeMatcher> {
        RangeMatcher::float(lower, upper, include_lower, include_upper)
    }
}

// ---------------------------------------------------------------------------
// Leaves, the top-level view, the context
// ---------------------------------------------------------------------------

/// Java's `LeafReaderContext` as a value source reads it: one segment's
/// reader and postings, the reader-wide statistics its queries score with,
/// and the similarity the searcher was given.
#[derive(Clone)]
pub struct ValueLeaf<'a> {
    pub(crate) ctx: LeafContext<'a>,
    /// `context.get("scorer")`: the scores a wrapping
    /// [`as_double_values_source`] was handed, which
    /// [`from_double_values_source`] reads.
    pub(crate) scorer: Option<std::rc::Rc<std::cell::RefCell<ScorableView<'a>>>>,
}

impl<'a> ValueLeaf<'a> {
    pub(crate) fn new(ctx: LeafContext<'a>) -> Self {
        Self { ctx, scorer: None }
    }

    /// One segment of `searcher` (its norms, the searcher's similarity and
    /// `global`'s statistics for the queries a source scores).
    pub fn of_searcher(
        searcher: &'a crate::index_searcher::IndexSearcher<'a, 'a>,
        leaf: usize,
        global: Option<&'a crate::GlobalStats>,
    ) -> Result<Self> {
        let seg = searcher.segments().get(leaf).ok_or_else(|| {
            Error::IllegalArgument(format!("leaf {leaf} is not a segment of the searcher"))
        })?;
        Ok(Self::new(leaf_context(
            seg,
            searcher.norms(leaf),
            global,
            searcher.similarity().filter(|s| !s.is_default_bm25()),
        )))
    }

    /// One opened segment on its own: no norms, no reader-wide statistics,
    /// the default similarity (what a grouping selector sees).
    pub fn of_segment(seg: &crate::multi_segment::OpenSegment<'a>) -> Self {
        Self::new(leaf_context(seg, None, None, None))
    }

    /// The segment's reader.
    ///
    /// # Errors
    /// [`Error::MissingSegmentReader`] when the segment was opened without
    /// one.
    pub fn reader(&self) -> Result<&'a crate::directory_reader::SegmentReader> {
        self.ctx
            .reader
            .ok_or_else(|| Error::MissingSegmentReader("a value source".into()))
    }

    /// `reader().maxDoc()`.
    ///
    /// # Errors
    /// As [`Self::reader`].
    pub fn max_doc(&self) -> Result<i32> {
        match self.ctx.max_doc {
            Some(m) => Ok(m),
            None => Ok(self.reader()?.max_doc),
        }
    }

    /// The searcher's similarity, `None` for the default BM25.
    pub fn similarity(&self) -> Option<&'a dyn Similarity> {
        self.ctx.similarity
    }
}

/// An exec [`LeafContext`] for one opened segment.
pub(crate) fn leaf_context<'a>(
    seg: &crate::multi_segment::OpenSegment<'a>,
    norms: Option<&'a HashMap<String, crate::FieldNorms<'a>>>,
    global: Option<&'a crate::GlobalStats>,
    similarity: Option<&'a dyn Similarity>,
) -> LeafContext<'a> {
    LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: seg.live_docs,
        points: seg.points,
        norms,
        global,
        max_doc: seg.max_doc,
        cache: None,
        reader: seg.reader,
        similarity,
    }
}

/// Java's searcher and top-level reader, as `createWeight` reads them:
/// every leaf, and the similarity.
pub struct TopLevel<'a> {
    pub(crate) leaves: Vec<LeafContext<'a>>,
}

impl<'a> TopLevel<'a> {
    /// `searcher`'s segments, with `global` the statistics for the queries a
    /// source scores ([`queries_stats`]).
    pub fn of_searcher(
        searcher: &'a crate::index_searcher::IndexSearcher<'a, 'a>,
        global: Option<&'a crate::GlobalStats>,
    ) -> Self {
        let similarity = searcher.similarity().filter(|s| !s.is_default_bm25());
        Self {
            leaves: searcher
                .segments()
                .iter()
                .enumerate()
                .map(|(i, seg)| leaf_context(seg, searcher.norms(i), global, similarity))
                .collect(),
        }
    }

    /// The leaves, in order.
    pub fn leaves(&self) -> impl Iterator<Item = ValueLeaf<'a>> + '_ {
        self.leaves.iter().map(|c| ValueLeaf::new(*c))
    }

    /// `getIndexReader().maxDoc()`.
    ///
    /// # Errors
    /// [`Error::MissingSegmentReader`] for a leaf without a reader.
    pub fn max_doc(&self) -> Result<i32> {
        let mut sum = 0i32;
        for leaf in self.leaves() {
            sum = sum.saturating_add(leaf.max_doc()?);
        }
        Ok(sum)
    }

    /// `getIndexReader().numDocs()`.
    ///
    /// # Errors
    /// As [`Self::max_doc`].
    pub fn num_docs(&self) -> Result<i32> {
        let mut sum = 0i32;
        for leaf in self.leaves() {
            sum = sum.saturating_add(leaf.reader()?.num_docs());
        }
        Ok(sum)
    }

    /// `getIndexReader().docFreq(term)`: the leaves' summed.
    ///
    /// # Errors
    /// A corrupt terms dictionary.
    pub fn doc_freq(&self, field: &str, term: &[u8]) -> Result<i32> {
        let mut sum = 0i32;
        for c in &self.leaves {
            if let Some(t) = c.fields.field(field) {
                if let Some(s) = t.try_seek_exact(term)? {
                    sum = sum.saturating_add(s.doc_freq);
                }
            }
        }
        Ok(sum)
    }
}

/// Java's `Map<Object,Object> context` (`ValueSource.newContext`): what each
/// source's [`ValueSource::create_weight`] computed, keyed by the source
/// (an identity map, as Java's `IdentityHashMap`).
#[derive(Default)]
pub struct FunctionContext {
    entries: HashMap<usize, Arc<dyn Any + Send + Sync>>,
    /// The context a wrapped source ([`as_double_values_source`]) builds,
    /// which in Java holds the searcher but saw no `createWeight`: the
    /// sources that only `createWeight` sets up fail on it as Java's do.
    pub(crate) searcher_only: bool,
}

impl fmt::Debug for FunctionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FunctionContext({} entries)", self.entries.len())
    }
}

/// A source's identity: its address (it lives in an `Arc`, which never
/// moves it).
pub(crate) fn source_key<T: ?Sized>(source: &T) -> usize {
    (source as *const T).cast::<()>() as usize
}

impl FunctionContext {
    /// `ValueSource.newContext(searcher)`: empty.
    pub fn new() -> Self {
        Self::default()
    }

    /// `ValueSource.newContext(searcher)` followed by
    /// `source.createWeight(context, searcher)`.
    ///
    /// # Errors
    /// Whatever the sources' `createWeight` reports.
    pub fn create(source: &dyn ValueSource, top: &TopLevel<'_>) -> Result<Self> {
        let mut fcx = Self::new();
        source.create_weight(&mut fcx, top)?;
        Ok(fcx)
    }

    /// `context.put(source, value)`.
    pub(crate) fn put<T: Any + Send + Sync, S: ?Sized>(&mut self, source: &S, value: T) {
        self.entries.insert(source_key(source), Arc::new(value));
    }

    /// `context.get(source)`.
    pub(crate) fn get<T: Any + Send + Sync, S: ?Sized>(&self, source: &S) -> Option<&T> {
        self.entries
            .get(&source_key(source))
            .and_then(|v| v.downcast_ref::<T>())
    }
}

/// The state `createWeight` never computed (a source's `getValues` before
/// it): Java's `NullPointerException` on the missing context entry.
pub(crate) fn not_weighted(description: &str) -> Error {
    Error::IllegalState(format!(
        "{description}: createWeight has not computed this value source's reader-wide state"
    ))
}

// ---------------------------------------------------------------------------
// ValueSource
// ---------------------------------------------------------------------------

/// `ValueSource`: a source of per-document values.
pub trait ValueSource: Send + Sync {
    /// `getValues(context, readerContext)`.
    ///
    /// # Errors
    /// A field indexed with doc values of another kind, a source whose
    /// `createWeight` was never run, an unreadable segment.
    fn get_values<'a>(&self, fcx: &FunctionContext, leaf: &ValueLeaf<'a>) -> Result<BoxValues<'a>>;

    /// `description()` (and `toString()`).
    fn description(&self) -> String;

    /// `createWeight(context, searcher)`: computes reader-wide state into the
    /// context (by default only the sub-sources').
    ///
    /// # Errors
    /// Whatever computing the state reports.
    fn create_weight(&self, fcx: &mut FunctionContext, top: &TopLevel<'_>) -> Result<()> {
        for s in self.sources() {
            s.create_weight(fcx, top)?;
        }
        Ok(())
    }

    /// The sources this one combines, for walks over the tree.
    fn sources(&self) -> Vec<&dyn ValueSource> {
        Vec::new()
    }

    /// The queries the tree scores (a [`valuesource::QueryValueSource`]'s),
    /// whose reader-wide statistics a search gathers.
    fn queries(&self) -> Vec<&crate::query::Clause> {
        self.sources()
            .into_iter()
            .flat_map(|s| s.queries())
            .collect()
    }

    /// The sort a subclass's `getSortField(reverse)` gives itself (a field
    /// source's sort on its field); `None` for `ValueSource`'s own
    /// ([`sort_field`] builds it).
    fn native_sort_field(&self, _reverse: bool) -> Option<crate::top_field::SortField> {
        None
    }
}

impl fmt::Debug for dyn ValueSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.description())
    }
}

/// `equals`, by description (see the module doc).
pub fn same_source(a: &dyn ValueSource, b: &dyn ValueSource) -> bool {
    std::ptr::addr_eq(a as *const dyn ValueSource, b as *const dyn ValueSource)
        || a.description() == b.description()
}

/// The statistics the queries inside `source` score with across `searcher`
/// (`QueryValueSource.createWeight`'s `searcher.createWeight`), for
/// [`TopLevel::of_searcher`] and [`ValueLeaf::of_searcher`].
///
/// # Errors
/// A corrupt terms dictionary.
pub fn queries_stats(
    searcher: &crate::index_searcher::IndexSearcher<'_, '_>,
    source: &dyn ValueSource,
) -> Result<crate::GlobalStats> {
    let mut q = crate::query::BooleanQuery::new();
    q.should = source.queries().into_iter().cloned().collect();
    crate::multi_segment::global_boolean_stats(searcher.segments(), &q)
}

// ---------------------------------------------------------------------------
// The statistics pass: every function query's createWeight, once per search
// ---------------------------------------------------------------------------

/// The function queries' reader-wide state for one search: each value
/// source's `createWeight` context and each values source's rewrite, keyed
/// by the source (its `Arc`'s address, which every clone of the query
/// shares).
#[derive(Clone, Default)]
pub(crate) struct FunctionStats {
    contexts: HashMap<usize, Arc<FunctionContext>>,
    sources: HashMap<usize, Arc<dyn crate::values_source::DoubleValuesSource>>,
}

impl fmt::Debug for FunctionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FunctionStats({} contexts, {} values sources)",
            self.contexts.len(),
            self.sources.len()
        )
    }
}

impl PartialEq for FunctionStats {
    fn eq(&self, o: &Self) -> bool {
        self.contexts.len() == o.contexts.len()
            && self
                .contexts
                .iter()
                .all(|(k, v)| o.contexts.get(k).is_some_and(|w| Arc::ptr_eq(v, w)))
            && self.sources.len() == o.sources.len()
            && self
                .sources
                .iter()
                .all(|(k, v)| o.sources.get(k).is_some_and(|w| Arc::ptr_eq(v, w)))
    }
}

impl FunctionStats {
    pub(crate) fn is_empty(&self) -> bool {
        self.contexts.is_empty() && self.sources.is_empty()
    }

    /// The context `source`'s `createWeight` filled for this search.
    pub(crate) fn context(&self, source: &dyn ValueSource) -> Option<&Arc<FunctionContext>> {
        self.contexts.get(&source_key(source))
    }

    /// `source.rewrite(searcher)` for this search (the source itself when
    /// it does not rewrite); `None` when the source was not prepared.
    pub(crate) fn source(
        &self,
        source: &dyn crate::values_source::DoubleValuesSource,
    ) -> Option<&Arc<dyn crate::values_source::DoubleValuesSource>> {
        self.sources.get(&source_key(source))
    }
}

/// A function query's source, as the statistics pass meets it.
#[derive(Clone)]
pub(crate) enum FunctionSource<'q> {
    /// A `FunctionQuery`'s or `FunctionRangeQuery`'s `createWeight`.
    Value(&'q Arc<dyn ValueSource>),
    /// A `FunctionScoreQuery`'s or `FunctionMatchQuery`'s
    /// `source.rewrite(searcher)`.
    Double(&'q Arc<dyn crate::values_source::DoubleValuesSource>),
}

impl<'q> FunctionSource<'q> {
    /// The queries the source scores.
    pub(crate) fn queries(&self) -> Vec<&'q crate::query::Clause> {
        match self {
            FunctionSource::Value(s) => s.queries(),
            FunctionSource::Double(s) => s.queries(),
        }
    }
}

/// Every function query in `query`, in any position (a filter's needs its
/// state as much as a scoring clause's), including those inside the
/// queries a function scores.
pub(crate) fn collect_functions<'q>(
    query: &'q crate::query::BooleanQuery,
    out: &mut Vec<FunctionSource<'q>>,
) {
    for c in query
        .must
        .iter()
        .chain(&query.should)
        .chain(&query.filter)
        .chain(&query.must_not)
    {
        collect_clause(c, out);
    }
}

fn collect_clause<'q>(c: &'q crate::query::Clause, out: &mut Vec<FunctionSource<'q>>) {
    use crate::extended_query::ExtendedQuery;
    use crate::query::Clause;
    let start = out.len();
    match c {
        Clause::Boolean(b) => collect_functions(b, out),
        Clause::DisjunctionMax(d) => {
            for inner in &d.disjuncts {
                collect_clause(inner, out);
            }
        }
        Clause::Boost(b) => collect_clause(&b.inner, out),
        Clause::ConstantScore(cs) => collect_clause(&cs.inner, out),
        Clause::Extended(e) => {
            match e.as_ref() {
                ExtendedQuery::Function(q) => out.push(FunctionSource::Value(&q.source)),
                ExtendedQuery::FunctionRange(q) => out.push(FunctionSource::Value(&q.source)),
                ExtendedQuery::FunctionScore(q) => out.push(FunctionSource::Double(&q.source)),
                ExtendedQuery::FunctionMatch(q) => out.push(FunctionSource::Double(&q.source)),
                _ => {}
            }
            for child in e.children() {
                collect_clause(child, out);
            }
        }
        _ => {}
    }
    // The queries the new functions score may hold functions themselves.
    let added: Vec<FunctionSource<'q>> = out[start..].to_vec();
    for f in added {
        for q in f.queries() {
            collect_clause(q, out);
        }
    }
}

/// The norms of `fields` in every segment, each field's built with its
/// reader-wide average length (as `DirectoryReader::field_norms_by_field`
/// builds a searcher's).
fn segment_norms<'a>(
    segments: &[crate::multi_segment::OpenSegment<'a>],
    fields: &[String],
) -> Vec<HashMap<String, crate::FieldNorms<'a>>> {
    let avgs: Vec<(String, f32)> = fields
        .iter()
        .filter_map(|f| {
            let (mut stf, mut dc, mut seen) = (0i64, 0i64, false);
            for seg in segments {
                if let Some((s, d)) = seg.reader.and_then(|r| r.field_stats(f)) {
                    stf = stf.saturating_add(s);
                    dc = dc.saturating_add(i64::from(d));
                    seen = true;
                }
            }
            seen.then(|| (f.clone(), crate::field_norms::avg_field_length(stf, dc)))
        })
        .collect();
    segments
        .iter()
        .map(|seg| {
            let mut m = HashMap::new();
            if let Some(r) = seg.reader {
                for (f, avg) in &avgs {
                    if let Some(n) = r.field_norms_with_avg_field_length(f, *avg) {
                        m.insert(f.clone(), n);
                    }
                }
            }
            m
        })
        .collect()
}

/// The statistics pass's function step: every function query's
/// `createWeight` (a value source's context) or `rewrite(searcher)` (a
/// values source's), over every segment, with `global` the statistics the
/// queries they score read.
///
/// # Errors
/// Whatever a source's `createWeight` or `rewrite` reports.
pub(crate) fn prepare_functions(
    segments: &[crate::multi_segment::OpenSegment<'_>],
    functions: &[FunctionSource<'_>],
    global: &crate::GlobalStats,
    similarity: Option<&dyn Similarity>,
) -> Result<FunctionStats> {
    let mut out = FunctionStats::default();
    if functions.is_empty() {
        return Ok(out);
    }
    let mut terms = Vec::new();
    for f in functions {
        for q in f.queries() {
            collect_query_fields(q, &mut terms);
        }
    }
    terms.sort();
    terms.dedup();
    let norms = segment_norms(segments, &terms);
    let top = TopLevel {
        leaves: segments
            .iter()
            .zip(&norms)
            .map(|(seg, n)| leaf_context(seg, Some(n), Some(global), similarity))
            .collect(),
    };
    for f in functions {
        match f {
            FunctionSource::Value(s) => {
                let key = source_key(s.as_ref());
                if let std::collections::hash_map::Entry::Vacant(slot) = out.contexts.entry(key) {
                    slot.insert(Arc::new(FunctionContext::create(s.as_ref(), &top)?));
                }
            }
            FunctionSource::Double(s) => {
                let key = source_key(s.as_ref());
                if let std::collections::hash_map::Entry::Vacant(slot) = out.sources.entry(key) {
                    // A source that does not rewrite is recorded as itself:
                    // every function query of the search is then found.
                    let r = s.rewrite(&top)?.unwrap_or_else(|| Arc::clone(*s));
                    slot.insert(r);
                }
            }
        }
    }
    Ok(out)
}

/// The fields a query scores terms of (whose norms its scorer reads).
fn collect_query_fields(c: &crate::query::Clause, out: &mut Vec<String>) {
    use crate::query::Clause;
    match c {
        Clause::Term(t) => out.push(t.field.clone()),
        Clause::Phrase(p) => out.push(p.field.clone()),
        Clause::MultiPhrase(p) => out.push(p.field.clone()),
        Clause::Fuzzy(f) => out.push(f.field.clone()),
        Clause::Boolean(b) => {
            for inner in b.must.iter().chain(&b.should).chain(&b.must_not) {
                collect_query_fields(inner, out);
            }
        }
        Clause::DisjunctionMax(d) => {
            for inner in &d.disjuncts {
                collect_query_fields(inner, out);
            }
        }
        Clause::Boost(b) => collect_query_fields(&b.inner, out),
        Clause::Span(s) => {
            let mut leaves = Vec::new();
            crate::collect_span_leaves(s, &mut leaves);
            out.extend(leaves.into_iter().map(|(f, _)| f));
        }
        Clause::Extended(e) => {
            let mut terms = Vec::new();
            crate::exec::extended::collect_terms(e, &mut terms);
            out.extend(terms.into_iter().map(|(f, _)| f));
            for child in e.children() {
                collect_query_fields(child, out);
            }
        }
        _ => {}
    }
}

/// `source.isCacheable(ctx)` over a segment's reader (`false` without one).
pub(crate) fn source_cacheable(
    source: &dyn crate::values_source::DoubleValuesSource,
    reader: Option<&crate::directory_reader::SegmentReader>,
) -> bool {
    reader.is_some_and(|r| {
        source.is_cacheable(&crate::values_source::ValuesContext::for_reader(r), 0)
    })
}
