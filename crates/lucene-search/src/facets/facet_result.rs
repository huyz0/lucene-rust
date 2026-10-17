//! `FacetResult` and `LabelAndValue`, with the `Number` their values are.
//!
//! Java types both values as `Number`: the counting facets put an `Integer`
//! there, the float-valued ones (`FloatTaxonomyFacets`) a `Float`.
//! [`FacetValue`] is that `Number`, with `Number.equals`' semantics: an `Int`
//! never equals a `Float`, and two floats are equal when their
//! `floatToIntBits` are (so `NaN` equals `NaN` and `-0.0` does not equal
//! `0.0`).

use std::fmt;

use lucene_util::geo::java_float_string;

/// The `Number` of a facet value: an `Integer` or a `Float`.
#[derive(Debug, Clone, Copy)]
pub enum FacetValue {
    Int(i32),
    Float(f32),
}

impl FacetValue {
    /// `Number.intValue()`: a float truncates toward zero, saturating, `NaN`
    /// to `0` (Java's `(int)` cast, which Rust's `as` is).
    pub fn int_value(self) -> i32 {
        match self {
            FacetValue::Int(v) => v,
            FacetValue::Float(v) => v as i32,
        }
    }

    /// `Number.floatValue()`.
    pub fn float_value(self) -> f32 {
        match self {
            FacetValue::Int(v) => v as f32,
            FacetValue::Float(v) => v,
        }
    }

    /// `Number.doubleValue()`.
    pub fn double_value(self) -> f64 {
        match self {
            FacetValue::Int(v) => f64::from(v),
            FacetValue::Float(v) => f64::from(v),
        }
    }

    /// `Number.hashCode()`: `Integer.hashCode` is the value,
    /// `Float.hashCode` its `floatToIntBits`.
    pub fn hash_code(self) -> i32 {
        match self {
            FacetValue::Int(v) => v,
            FacetValue::Float(v) => float_to_int_bits(v),
        }
    }
}

/// `Float.floatToIntBits`: every `NaN` collapses to the canonical one.
pub(crate) fn float_to_int_bits(v: f32) -> i32 {
    if v.is_nan() {
        0x7fc0_0000
    } else {
        v.to_bits() as i32
    }
}

impl PartialEq for FacetValue {
    /// `Integer.equals` / `Float.equals`.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FacetValue::Int(a), FacetValue::Int(b)) => a == b,
            (FacetValue::Float(a), FacetValue::Float(b)) => {
                float_to_int_bits(*a) == float_to_int_bits(*b)
            }
            _ => false,
        }
    }
}

impl Eq for FacetValue {}

impl fmt::Display for FacetValue {
    /// `Integer.toString` / `Float.toString`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FacetValue::Int(v) => write!(f, "{v}"),
            FacetValue::Float(v) => f.write_str(&java_float_string(*v)),
        }
    }
}

impl From<i32> for FacetValue {
    fn from(v: i32) -> Self {
        FacetValue::Int(v)
    }
}

impl From<f32> for FacetValue {
    fn from(v: f32) -> Self {
        FacetValue::Float(v)
    }
}

/// `LabelAndValue`: one child of a [`FacetResult`].
#[derive(Debug, Clone)]
pub struct LabelAndValue {
    /// `label`: the child's last path component.
    pub label: String,
    /// `value`: its aggregated value.
    pub value: FacetValue,
    /// `count`: how many documents contributed to it.
    pub count: i32,
}

impl LabelAndValue {
    /// `new LabelAndValue(label, value)`: the count is `value.intValue()`.
    pub fn new(label: impl Into<String>, value: FacetValue) -> Self {
        LabelAndValue {
            label: label.into(),
            value,
            count: value.int_value(),
        }
    }

    /// `new LabelAndValue(label, value, count)`.
    pub fn with_count(label: impl Into<String>, value: FacetValue, count: i32) -> Self {
        LabelAndValue {
            label: label.into(),
            value,
            count,
        }
    }

    /// `hashCode()`.
    pub fn hash_code(&self) -> i32 {
        java_string_hash(&self.label).wrapping_add(self.value.hash_code().wrapping_mul(1439))
    }
}

impl PartialEq for LabelAndValue {
    /// `equals`: label and value; the count is not compared.
    fn eq(&self, other: &Self) -> bool {
        self.label == other.label && self.value == other.value
    }
}

impl Eq for LabelAndValue {}

impl fmt::Display for LabelAndValue {
    /// `toString()`: `label (value)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.label, self.value)
    }
}

/// `FacetResult`: the children of one path of one dim.
#[derive(Debug, Clone)]
pub struct FacetResult {
    /// `dim`.
    pub dim: String,
    /// `path`: the path under `dim` whose children these are.
    pub path: Vec<String>,
    /// `value`: the path's aggregated value; `-1` (or `-1f`) for a
    /// multi-valued dim that does not require a dim count.
    pub value: FacetValue,
    /// `labelValues`.
    pub label_values: Vec<LabelAndValue>,
    /// `childCount`: how many children had a value, which can exceed
    /// `label_values.len()` when `topN` cut the list.
    pub child_count: i32,
}

impl FacetResult {
    /// `new FacetResult(dim, path, value, labelValues, childCount)`.
    pub fn new(
        dim: impl Into<String>,
        path: Vec<String>,
        value: FacetValue,
        label_values: Vec<LabelAndValue>,
        child_count: i32,
    ) -> Self {
        FacetResult {
            dim: dim.into(),
            path,
            value,
            label_values,
            child_count,
        }
    }

    /// `hashCode()`.
    pub fn hash_code(&self) -> i32 {
        let mut h = self
            .value
            .hash_code()
            .wrapping_add(self.child_count.wrapping_mul(31));
        for lv in &self.label_values {
            h = lv.hash_code().wrapping_add(h.wrapping_mul(31));
        }
        h
    }
}

impl PartialEq for FacetResult {
    /// `equals`: value, child count and children; `dim` and `path` are not
    /// compared.
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
            && self.child_count == other.child_count
            && self.label_values == other.label_values
    }
}

impl Eq for FacetResult {}

impl fmt::Display for FacetResult {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "dim={} path=[{}] value={} childCount={}",
            self.dim,
            self.path.join(", "),
            self.value,
            self.child_count
        )?;
        for lv in &self.label_values {
            writeln!(f, "  {lv}")?;
        }
        Ok(())
    }
}

/// `String.hashCode()`: over UTF-16 code units.
fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16()
        .fold(0i32, |h, c| h.wrapping_mul(31).wrapping_add(i32::from(c)))
}

#[cfg(test)]
mod tests;
