//! `org.apache.lucene.queries.function.docvalues`: the typed abstract bases
//! of [`FunctionValues`] -- `FloatDocValues`, `IntDocValues`,
//! `LongDocValues`, `DoubleDocValues`, `BoolDocValues`, `StrDocValues` --
//! and `DocTermsIndexDocValues`.
//!
//! Each base is a trait with Java's derived getters as its defaults (every
//! number from the one native getter with Java's narrowing casts, `strVal`
//! from `Float/Integer/Long/Double/Boolean.toString`, `objectVal` the boxed
//! value or `null`, `toString(doc)` as `description=strVal`, the base's
//! `ValueFiller` and range scorer); a wrapper ([`Float`], [`Int`], [`Long`],
//! [`Double`], [`Bool`], [`Str`]) makes an implementation a
//! [`FunctionValues`]. A Java subclass that overrides a getter overrides the
//! trait method. `DocTermsIndexDocValues`, whose two subclasses here differ
//! in nothing, is the concrete [`DocTermsIndexDocValues`].

use super::{
    java_double, java_float, out_of_order, parse_java_int, parse_java_long, FunctionValues,
    MutableValue, ObjectVal, RangeMatcher,
};
use crate::reader::SortedDocValues;
use crate::{Error, Result};

/// Java's `(byte)` of an `int`.
pub(crate) fn i2b(v: i32) -> i8 {
    v as i8
}

/// Java's `(short)` of an `int`.
pub(crate) fn i2s(v: i32) -> i16 {
    v as i16
}

/// Java's `(byte)` of a `float` or `double`: to `int` (saturating, `NaN` to
/// `0`), then truncated.
pub(crate) fn d2b(v: f64) -> i8 {
    (v as i32) as i8
}

/// Java's `(short)` of a `float` or `double`.
pub(crate) fn d2s(v: f64) -> i16 {
    (v as i32) as i16
}

macro_rules! forward_values {
    ($wrapper:ident, $base:ident) => {
        impl<T: $base> FunctionValues for $wrapper<T> {
            fn byte_val(&mut self, doc: i32) -> Result<i8> {
                $base::byte_val(&mut self.0, doc)
            }
            fn short_val(&mut self, doc: i32) -> Result<i16> {
                $base::short_val(&mut self.0, doc)
            }
            fn float_val(&mut self, doc: i32) -> Result<f32> {
                $base::float_val(&mut self.0, doc)
            }
            fn int_val(&mut self, doc: i32) -> Result<i32> {
                $base::int_val(&mut self.0, doc)
            }
            fn long_val(&mut self, doc: i32) -> Result<i64> {
                $base::long_val(&mut self.0, doc)
            }
            fn double_val(&mut self, doc: i32) -> Result<f64> {
                $base::double_val(&mut self.0, doc)
            }
            fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
                $base::str_val(&mut self.0, doc)
            }
            fn bool_val(&mut self, doc: i32) -> Result<bool> {
                $base::bool_val(&mut self.0, doc)
            }
            fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
                $base::bytes_val(&mut self.0, doc, target)
            }
            fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
                $base::object_val(&mut self.0, doc)
            }
            fn exists(&mut self, doc: i32) -> Result<bool> {
                $base::exists(&mut self.0, doc)
            }
            fn to_string_doc(&mut self, doc: i32) -> Result<String> {
                $base::to_string_doc(&mut self.0, doc)
            }
            fn new_value(&self) -> MutableValue {
                $base::new_value(&self.0)
            }
            fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
                $base::fill_value(&mut self.0, doc, out)
            }
            fn range_matcher(
                &mut self,
                lower: Option<&str>,
                upper: Option<&str>,
                include_lower: bool,
                include_upper: bool,
            ) -> Result<RangeMatcher> {
                $base::range_matcher(&mut self.0, lower, upper, include_lower, include_upper)
            }
            fn cost(&self) -> f32 {
                $base::cost(&self.0)
            }
            fn float_val_batch(&mut self, docs: &[i32], out: &mut [f32]) -> Result<()> {
                $base::float_val_batch(&mut self.0, docs, out)
            }
            fn double_val_batch(&mut self, docs: &[i32], out: &mut [f64]) -> Result<()> {
                $base::double_val_batch(&mut self.0, docs, out)
            }
            fn range_batch(
                &mut self,
                range: Option<&RangeMatcher>,
                docs: &[i32],
                matched: &mut Vec<i32>,
                mut values: Option<&mut Vec<f32>>,
            ) -> Result<()> {
                if let Some(r) = range {
                    let v = values.as_deref_mut();
                    if $base::range_batch_native(&mut self.0, r, docs, matched, v)? {
                        return Ok(());
                    }
                }
                super::range_batch_per_doc(self, range, docs, matched, values)
            }
        }
    };
}

/// The batch getters of a typed base ([`FunctionValues::float_val_batch`],
/// [`FunctionValues::double_val_batch`]): the per-document getters in a
/// loop, statically dispatched to the implementation, which a function of
/// other values overrides to read each of them a batch at a time.
macro_rules! batch_defaults {
    () => {
        /// [`FunctionValues::float_val_batch`].
        ///
        /// # Errors
        /// Whatever reading a value reports.
        fn float_val_batch(&mut self, docs: &[i32], out: &mut [f32]) -> Result<()> {
            for (&doc, o) in docs.iter().zip(out.iter_mut()) {
                *o = self.float_val(doc)?;
            }
            Ok(())
        }
        /// [`FunctionValues::double_val_batch`].
        ///
        /// # Errors
        /// Whatever reading a value reports.
        fn double_val_batch(&mut self, docs: &[i32], out: &mut [f64]) -> Result<()> {
            for (&doc, o) in docs.iter().zip(out.iter_mut()) {
                *o = self.double_val(doc)?;
            }
            Ok(())
        }
        /// [`FunctionValues::range_batch`] read natively, for values that
        /// can: `Ok(false)`, having read nothing, otherwise (the default),
        /// and the batch is then read one document at a time.
        ///
        /// # Errors
        /// Whatever reading a value reports.
        fn range_batch_native(
            &mut self,
            _range: &RangeMatcher,
            _docs: &[i32],
            _matched: &mut Vec<i32>,
            _values: Option<&mut Vec<f32>>,
        ) -> Result<bool> {
            Ok(false)
        }
    };
}

/// `bytesVal` from `strVal`: `FunctionValues`' own.
fn bytes_from_str(s: Option<String>, target: &mut Vec<u8>) -> bool {
    target.clear();
    match s {
        Some(s) => {
            target.extend_from_slice(s.as_bytes());
            true
        }
        None => false,
    }
}

// ---------------------------------------------------------------------------
// FloatDocValues
// ---------------------------------------------------------------------------

/// `FloatDocValues`: values whose native getter is `floatVal`.
pub trait FloatDocValues {
    batch_defaults!();
    /// `vs.description()`, which only the base's `toString(doc)` reads
    /// (empty for values that override it).
    fn description(&self) -> String {
        String::new()
    }
    fn float_val(&mut self, doc: i32) -> Result<f32>;
    fn exists(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn byte_val(&mut self, doc: i32) -> Result<i8> {
        Ok(d2b(f64::from(self.float_val(doc)?)))
    }
    fn short_val(&mut self, doc: i32) -> Result<i16> {
        Ok(d2s(f64::from(self.float_val(doc)?)))
    }
    fn int_val(&mut self, doc: i32) -> Result<i32> {
        Ok(self.float_val(doc)? as i32)
    }
    fn long_val(&mut self, doc: i32) -> Result<i64> {
        Ok(self.float_val(doc)? as i64)
    }
    fn double_val(&mut self, doc: i32) -> Result<f64> {
        Ok(f64::from(self.float_val(doc)?))
    }
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        Ok(self.float_val(doc)? != 0.0)
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        Ok(Some(java_float(self.float_val(doc)?)))
    }
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        let s = self.str_val(doc)?;
        Ok(bytes_from_str(s, target))
    }
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(if self.exists(doc)? {
            ObjectVal::Float(self.float_val(doc)?)
        } else {
            ObjectVal::Null
        })
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?;
        Ok(format!("{}={}", self.description(), opt_str(s)))
    }
    fn new_value(&self) -> MutableValue {
        MutableValue::float()
    }
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        let v = self.float_val(doc)?;
        let e = self.exists(doc)?;
        *out = MutableValue::Float {
            value: v,
            exists: e,
        };
        Ok(())
    }
    fn range_matcher(
        &mut self,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<RangeMatcher> {
        RangeMatcher::float(lower, upper, include_lower, include_upper)
    }
    fn cost(&self) -> f32 {
        100.0
    }
}

/// Java's `String.valueOf` of a possibly-`null` string.
pub(crate) fn opt_str(s: Option<String>) -> String {
    s.unwrap_or_else(|| "null".to_string())
}

/// A [`FloatDocValues`] as [`FunctionValues`].
pub struct Float<T>(pub T);
forward_values!(Float, FloatDocValues);

// ---------------------------------------------------------------------------
// IntDocValues
// ---------------------------------------------------------------------------

/// `IntDocValues`: values whose native getter is `intVal`.
pub trait IntDocValues {
    batch_defaults!();
    /// As [`FloatDocValues::description`].
    fn description(&self) -> String {
        String::new()
    }
    fn int_val(&mut self, doc: i32) -> Result<i32>;
    fn exists(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn byte_val(&mut self, doc: i32) -> Result<i8> {
        Ok(i2b(self.int_val(doc)?))
    }
    fn short_val(&mut self, doc: i32) -> Result<i16> {
        Ok(i2s(self.int_val(doc)?))
    }
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        Ok(self.int_val(doc)? as f32)
    }
    fn long_val(&mut self, doc: i32) -> Result<i64> {
        Ok(i64::from(self.int_val(doc)?))
    }
    fn double_val(&mut self, doc: i32) -> Result<f64> {
        Ok(f64::from(self.int_val(doc)?))
    }
    /// `FunctionValues.boolVal`: `intVal != 0`.
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        Ok(self.int_val(doc)? != 0)
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        Ok(Some(self.int_val(doc)?.to_string()))
    }
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        let s = self.str_val(doc)?;
        Ok(bytes_from_str(s, target))
    }
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(if self.exists(doc)? {
            ObjectVal::Int(self.int_val(doc)?)
        } else {
            ObjectVal::Null
        })
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?;
        Ok(format!("{}={}", self.description(), opt_str(s)))
    }
    fn new_value(&self) -> MutableValue {
        MutableValue::Int {
            value: 0,
            exists: true,
        }
    }
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        let v = self.int_val(doc)?;
        let e = self.exists(doc)?;
        *out = MutableValue::Int {
            value: v,
            exists: e,
        };
        Ok(())
    }
    /// `IntDocValues.getRangeScorer`: `Integer.parseInt` bounds.
    fn range_matcher(
        &mut self,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<RangeMatcher> {
        Ok(RangeMatcher::int(
            lower.map(parse_java_int).transpose()?,
            upper.map(parse_java_int).transpose()?,
            include_lower,
            include_upper,
        ))
    }
    fn cost(&self) -> f32 {
        100.0
    }
}

/// An [`IntDocValues`] as [`FunctionValues`].
pub struct Int<T>(pub T);
forward_values!(Int, IntDocValues);

// ---------------------------------------------------------------------------
// LongDocValues
// ---------------------------------------------------------------------------

/// `LongDocValues`: values whose native getter is `longVal`.
pub trait LongDocValues {
    batch_defaults!();
    /// As [`FloatDocValues::description`].
    fn description(&self) -> String {
        String::new()
    }
    fn long_val(&mut self, doc: i32) -> Result<i64>;
    fn exists(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn byte_val(&mut self, doc: i32) -> Result<i8> {
        Ok(self.long_val(doc)? as i8)
    }
    fn short_val(&mut self, doc: i32) -> Result<i16> {
        Ok(self.long_val(doc)? as i16)
    }
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        Ok(self.long_val(doc)? as f32)
    }
    fn int_val(&mut self, doc: i32) -> Result<i32> {
        Ok(self.long_val(doc)? as i32)
    }
    fn double_val(&mut self, doc: i32) -> Result<f64> {
        Ok(self.long_val(doc)? as f64)
    }
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        Ok(self.long_val(doc)? != 0)
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        Ok(Some(self.long_val(doc)?.to_string()))
    }
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        let s = self.str_val(doc)?;
        Ok(bytes_from_str(s, target))
    }
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(if self.exists(doc)? {
            ObjectVal::Long(self.long_val(doc)?)
        } else {
            ObjectVal::Null
        })
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?;
        Ok(format!("{}={}", self.description(), opt_str(s)))
    }
    /// `externalToLong(extVal)`: `Long.parseLong`.
    fn external_to_long(&self, ext: &str) -> Result<i64> {
        parse_java_long(ext)
    }
    fn new_value(&self) -> MutableValue {
        MutableValue::Long {
            value: 0,
            exists: true,
        }
    }
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        let v = self.long_val(doc)?;
        let e = self.exists(doc)?;
        *out = MutableValue::Long {
            value: v,
            exists: e,
        };
        Ok(())
    }
    /// `LongDocValues.getRangeScorer`: bounds through
    /// [`Self::external_to_long`].
    fn range_matcher(
        &mut self,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<RangeMatcher> {
        let lower = lower.map(|l| self.external_to_long(l)).transpose()?;
        let upper = upper.map(|u| self.external_to_long(u)).transpose()?;
        Ok(RangeMatcher::long(
            lower,
            upper,
            include_lower,
            include_upper,
        ))
    }
    fn cost(&self) -> f32 {
        100.0
    }
}

/// A [`LongDocValues`] as [`FunctionValues`].
pub struct Long<T>(pub T);
forward_values!(Long, LongDocValues);

// ---------------------------------------------------------------------------
// DoubleDocValues
// ---------------------------------------------------------------------------

/// `DoubleDocValues`: values whose native getter is `doubleVal`.
pub trait DoubleDocValues {
    batch_defaults!();
    /// As [`FloatDocValues::description`].
    fn description(&self) -> String {
        String::new()
    }
    fn double_val(&mut self, doc: i32) -> Result<f64>;
    fn exists(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn byte_val(&mut self, doc: i32) -> Result<i8> {
        Ok(d2b(self.double_val(doc)?))
    }
    fn short_val(&mut self, doc: i32) -> Result<i16> {
        Ok(d2s(self.double_val(doc)?))
    }
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        Ok(self.double_val(doc)? as f32)
    }
    fn int_val(&mut self, doc: i32) -> Result<i32> {
        Ok(self.double_val(doc)? as i32)
    }
    fn long_val(&mut self, doc: i32) -> Result<i64> {
        Ok(self.double_val(doc)? as i64)
    }
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        Ok(self.double_val(doc)? != 0.0)
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        Ok(Some(java_double(self.double_val(doc)?)))
    }
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        let s = self.str_val(doc)?;
        Ok(bytes_from_str(s, target))
    }
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(if self.exists(doc)? {
            ObjectVal::Double(self.double_val(doc)?)
        } else {
            ObjectVal::Null
        })
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?;
        Ok(format!("{}={}", self.description(), opt_str(s)))
    }
    fn new_value(&self) -> MutableValue {
        MutableValue::Double {
            value: 0.0,
            exists: true,
        }
    }
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        let v = self.double_val(doc)?;
        let e = self.exists(doc)?;
        *out = MutableValue::Double {
            value: v,
            exists: e,
        };
        Ok(())
    }
    /// `DoubleDocValues.getRangeScorer`: `Double.parseDouble` bounds.
    fn range_matcher(
        &mut self,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<RangeMatcher> {
        RangeMatcher::double(lower, upper, include_lower, include_upper)
    }
    fn cost(&self) -> f32 {
        100.0
    }
}

/// A [`DoubleDocValues`] as [`FunctionValues`].
pub struct Double<T>(pub T);
forward_values!(Double, DoubleDocValues);

// ---------------------------------------------------------------------------
// BoolDocValues
// ---------------------------------------------------------------------------

/// `BoolDocValues`: values whose native getter is `boolVal`.
pub trait BoolDocValues {
    batch_defaults!();
    /// As [`FloatDocValues::description`].
    fn description(&self) -> String {
        String::new()
    }
    fn bool_val(&mut self, doc: i32) -> Result<bool>;
    fn exists(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn byte_val(&mut self, doc: i32) -> Result<i8> {
        Ok(i8::from(self.bool_val(doc)?))
    }
    fn short_val(&mut self, doc: i32) -> Result<i16> {
        Ok(i16::from(self.bool_val(doc)?))
    }
    fn float_val(&mut self, doc: i32) -> Result<f32> {
        Ok(if self.bool_val(doc)? { 1.0 } else { 0.0 })
    }
    fn int_val(&mut self, doc: i32) -> Result<i32> {
        Ok(i32::from(self.bool_val(doc)?))
    }
    fn long_val(&mut self, doc: i32) -> Result<i64> {
        Ok(i64::from(self.bool_val(doc)?))
    }
    fn double_val(&mut self, doc: i32) -> Result<f64> {
        Ok(if self.bool_val(doc)? { 1.0 } else { 0.0 })
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        Ok(Some(self.bool_val(doc)?.to_string()))
    }
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        let s = self.str_val(doc)?;
        Ok(bytes_from_str(s, target))
    }
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(if self.exists(doc)? {
            ObjectVal::Bool(self.bool_val(doc)?)
        } else {
            ObjectVal::Null
        })
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?;
        Ok(format!("{}={}", self.description(), opt_str(s)))
    }
    fn new_value(&self) -> MutableValue {
        MutableValue::Bool {
            value: false,
            exists: true,
        }
    }
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        let v = self.bool_val(doc)?;
        let e = self.exists(doc)?;
        *out = MutableValue::Bool {
            value: v,
            exists: e,
        };
        Ok(())
    }
    /// `FunctionValues.getRangeScorer` (`BoolDocValues` keeps the float
    /// one).
    fn range_matcher(
        &mut self,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<RangeMatcher> {
        RangeMatcher::float(lower, upper, include_lower, include_upper)
    }
    fn cost(&self) -> f32 {
        100.0
    }
}

/// A [`BoolDocValues`] as [`FunctionValues`].
pub struct Bool<T>(pub T);
forward_values!(Bool, BoolDocValues);

// ---------------------------------------------------------------------------
// StrDocValues
// ---------------------------------------------------------------------------

/// `StrDocValues`: values whose native getter is `strVal`. Its numeric
/// getters are `FunctionValues`' (unsupported).
pub trait StrDocValues {
    batch_defaults!();
    /// As [`FloatDocValues::description`].
    fn description(&self) -> String {
        String::new()
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>>;
    fn exists(&mut self, _doc: i32) -> Result<bool> {
        Ok(true)
    }
    fn byte_val(&mut self, _doc: i32) -> Result<i8> {
        Err(super::unsupported())
    }
    fn short_val(&mut self, _doc: i32) -> Result<i16> {
        Err(super::unsupported())
    }
    fn float_val(&mut self, _doc: i32) -> Result<f32> {
        Err(super::unsupported())
    }
    fn int_val(&mut self, _doc: i32) -> Result<i32> {
        Err(super::unsupported())
    }
    fn long_val(&mut self, _doc: i32) -> Result<i64> {
        Err(super::unsupported())
    }
    fn double_val(&mut self, _doc: i32) -> Result<f64> {
        Err(super::unsupported())
    }
    /// `StrDocValues.boolVal`: `exists(doc)`.
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        self.exists(doc)
    }
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        let s = self.str_val(doc)?;
        Ok(bytes_from_str(s, target))
    }
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(if self.exists(doc)? {
            self.str_val(doc)?.map_or(ObjectVal::Null, ObjectVal::Str)
        } else {
            ObjectVal::Null
        })
    }
    /// `description='strVal'`.
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?;
        Ok(format!("{}='{}'", self.description(), opt_str(s)))
    }
    fn new_value(&self) -> MutableValue {
        MutableValue::Str {
            value: Vec::new(),
            exists: true,
        }
    }
    /// `mval.exists = bytesVal(doc, mval.value)`.
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        let mut value = Vec::new();
        let exists = self.bytes_val(doc, &mut value)?;
        *out = MutableValue::Str { value, exists };
        Ok(())
    }
    fn range_matcher(
        &mut self,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<RangeMatcher> {
        RangeMatcher::float(lower, upper, include_lower, include_upper)
    }
    fn cost(&self) -> f32 {
        100.0
    }
}

/// A [`StrDocValues`] as [`FunctionValues`].
pub struct Str<T>(pub T);
forward_values!(Str, StrDocValues);

// ---------------------------------------------------------------------------
// DocTermsIndexDocValues
// ---------------------------------------------------------------------------

/// `DocTermsIndexDocValues`: a `SORTED` field's term per document (through
/// a [`SortedDocValues`], which may be a `SORTED_SET` field's selected
/// value), with `objectVal` its `strVal` (as both of its subclasses here,
/// `BytesRefFieldSource`'s and `SortedSetFieldSource`'s, define it) and
/// `toTerm` the identity.
pub struct DocTermsIndexDocValues<'a> {
    terms_index: Box<dyn SortedDocValues + 'a>,
    description: String,
    last_doc: i32,
}

impl<'a> DocTermsIndexDocValues<'a> {
    pub fn new(description: String, terms_index: Box<dyn SortedDocValues + 'a>) -> Self {
        Self {
            terms_index,
            description,
            last_doc: 0,
        }
    }

    /// `DocTermsIndexDocValues.open(context, field)`: `DocValues.getSorted`,
    /// its failure wrapped as `DocTermsIndexException`.
    ///
    /// # Errors
    /// [`Error::IllegalState`] naming the field.
    pub fn open(
        description: String,
        reader: &'a crate::directory_reader::SegmentReader,
        field: &str,
    ) -> Result<Self> {
        let terms = crate::reader::doc_values::get_sorted(reader, field).map_err(|e| {
            Error::IllegalState(format!(
                "Can't initialize DocTermsIndex to generate (function) FunctionValues for \
                 field: {field} ({e})"
            ))
        })?;
        Ok(Self::new(description, terms))
    }

    /// `getOrdForDoc(doc)`: the document's ordinal, `-1` without one.
    // SENTINEL: `-1` = the document has no term (`getOrdForDoc`); `exists`
    // tests `>= 0`, `ordVal` hands it on as Java's does.
    fn ord_for_doc(&mut self, doc: i32) -> Result<i32> {
        if doc < self.last_doc {
            return Err(out_of_order(self.last_doc, doc));
        }
        self.last_doc = doc;
        let mut cur = self.terms_index.doc_id();
        if doc > cur {
            cur = self.terms_index.advance(doc)?;
        }
        Ok(if doc == cur {
            self.terms_index.ord_value()
        } else {
            -1
        })
    }

    fn term(&mut self, doc: i32) -> Result<Option<Vec<u8>>> {
        let ord = self.ord_for_doc(doc)?;
        if ord == -1 {
            return Ok(None);
        }
        Ok(Some(self.terms_index.lookup_ord(ord)?))
    }
}

impl FunctionValues for DocTermsIndexDocValues<'_> {
    fn exists(&mut self, doc: i32) -> Result<bool> {
        Ok(self.ord_for_doc(doc)? >= 0)
    }
    fn ord_val(&mut self, doc: i32) -> Result<i32> {
        self.ord_for_doc(doc)
    }
    fn num_ord(&self) -> Result<i32> {
        Ok(self.terms_index.value_count())
    }
    fn bytes_val(&mut self, doc: i32, target: &mut Vec<u8>) -> Result<bool> {
        target.clear();
        match self.term(doc)? {
            Some(t) => {
                target.extend_from_slice(&t);
                Ok(true)
            }
            None => Ok(false),
        }
    }
    fn str_val(&mut self, doc: i32) -> Result<Option<String>> {
        Ok(self
            .term(doc)?
            .map(|t| String::from_utf8_lossy(&t).into_owned()))
    }
    fn bool_val(&mut self, doc: i32) -> Result<bool> {
        self.exists(doc)
    }
    fn object_val(&mut self, doc: i32) -> Result<ObjectVal> {
        Ok(self.str_val(doc)?.map_or(ObjectVal::Null, ObjectVal::Str))
    }
    /// `DocTermsIndexDocValues.getRangeScorer`: the bounds looked up as
    /// ordinals (`lookupTerm`), an absent bound rounded inward.
    fn range_matcher(
        &mut self,
        lower: Option<&str>,
        upper: Option<&str>,
        include_lower: bool,
        include_upper: bool,
    ) -> Result<RangeMatcher> {
        let mut lo = i32::MIN;
        if let Some(l) = lower {
            lo = self.terms_index.lookup_term(l.as_bytes())?;
            if lo < 0 {
                lo = -lo - 1;
            } else if !include_lower {
                lo += 1;
            }
        }
        let mut hi = i32::MAX;
        if let Some(u) = upper {
            hi = self.terms_index.lookup_term(u.as_bytes())?;
            if hi < 0 {
                hi = -hi - 2;
            } else if !include_upper {
                hi -= 1;
            }
        }
        Ok(RangeMatcher::Ord {
            lower: lo,
            upper: hi,
        })
    }
    fn to_string_doc(&mut self, doc: i32) -> Result<String> {
        let s = self.str_val(doc)?;
        Ok(format!("{}={}", self.description, opt_str(s)))
    }
    fn new_value(&self) -> MutableValue {
        MutableValue::Str {
            value: Vec::new(),
            exists: true,
        }
    }
    fn fill_value(&mut self, doc: i32, out: &mut MutableValue) -> Result<()> {
        let value = self.term(doc)?;
        *out = MutableValue::Str {
            exists: value.is_some(),
            value: value.unwrap_or_default(),
        };
        Ok(())
    }
}
