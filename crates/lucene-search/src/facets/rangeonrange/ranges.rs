//! `rangeonrange.Range`, `LongRange` and `DoubleRange`: labelled boxes of
//! one or more dimensions.

use std::fmt;

use lucene_util::geo::java_double_string;

use crate::facets::range::fail_no_match;
use crate::{Error, Result};

/// `rangeonrange.LongRange`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongRange {
    /// `label`.
    pub label: String,
    /// `min`, inclusive, per dimension.
    pub min: Vec<i64>,
    /// `max`, inclusive, per dimension.
    pub max: Vec<i64>,
}

impl LongRange {
    /// `new LongRange(label, min, minInclusive, max, maxInclusive)`: one
    /// dimension.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] (`range "label" matches nothing`) when the
    /// range is empty.
    pub fn new(
        label: &str,
        min: i64,
        min_inclusive: bool,
        max: i64,
        max_inclusive: bool,
    ) -> Result<Self> {
        let mut min = min;
        let mut max = max;
        if !min_inclusive {
            if min == i64::MAX {
                return fail_no_match(label);
            }
            min = min.wrapping_add(1);
        }
        if !max_inclusive {
            if max == i64::MIN {
                return fail_no_match(label);
            }
            max = max.wrapping_sub(1);
        }
        if min > max {
            return fail_no_match(label);
        }
        Ok(LongRange {
            label: label.to_string(),
            min: vec![min],
            max: vec![max],
        })
    }

    /// `new LongRange(label, min[], max[])`: inclusive ends per dimension.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] (`range "label" matches nothing`) for no
    /// dimensions, ends of different dimensions, or `min > max` in one.
    pub fn with_dims(label: &str, min: Vec<i64>, max: Vec<i64>) -> Result<Self> {
        if min.is_empty() || max.is_empty() || min.len() != max.len() {
            return fail_no_match(label);
        }
        if min.iter().zip(&max).any(|(a, b)| a > b) {
            return fail_no_match(label);
        }
        Ok(LongRange {
            label: label.to_string(),
            min,
            max,
        })
    }

    /// `dims`.
    pub fn dims(&self) -> usize {
        self.min.len()
    }

    /// `org.apache.lucene.document.LongRange.verifyAndEncode(min, max,
    /// bytes)`: the mins then the maxes, each as sortable bytes.
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        let n = self.min.len();
        let mut bytes = vec![0u8; 16 * n];
        for d in 0..n {
            if self.min[d] > self.max[d] {
                return Err(Error::IllegalArgument(format!(
                    "min value ({}) is greater than max value ({})",
                    self.min[d], self.max[d]
                )));
            }
            lucene_util::numeric_utils::long_to_sortable_bytes(self.min[d], &mut bytes, d * 8);
            lucene_util::numeric_utils::long_to_sortable_bytes(
                self.max[d],
                &mut bytes,
                (n + d) * 8,
            );
        }
        Ok(bytes)
    }
}

impl fmt::Display for LongRange {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LongRange(label: {}, min: {}, max: {})",
            self.label,
            java_array(self.min.iter().map(i64::to_string)),
            java_array(self.max.iter().map(i64::to_string))
        )
    }
}

/// `rangeonrange.DoubleRange`.
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleRange {
    /// `label`.
    pub label: String,
    /// `min`, inclusive, per dimension.
    pub min: Vec<f64>,
    /// `max`, inclusive, per dimension.
    pub max: Vec<f64>,
}

impl DoubleRange {
    /// `new DoubleRange(label, min, minInclusive, max, maxInclusive)`: one
    /// dimension, its exclusive ends moved in to the next double.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a `NaN` end and for an empty range.
    pub fn new(
        label: &str,
        min: f64,
        min_inclusive: bool,
        max: f64,
        max_inclusive: bool,
    ) -> Result<Self> {
        let r =
            crate::facets::range::DoubleRange::new(label, min, min_inclusive, max, max_inclusive)?;
        Ok(DoubleRange {
            label: r.label,
            min: vec![r.min],
            max: vec![r.max],
        })
    }

    /// `new DoubleRange(label, min[], max[])`: inclusive ends per
    /// dimension. A `NaN` end is accepted here and refused when the range
    /// is encoded, as in Java.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] (`range "label" matches nothing`) for no
    /// dimensions, ends of different dimensions, or `min > max` in one.
    pub fn with_dims(label: &str, min: Vec<f64>, max: Vec<f64>) -> Result<Self> {
        if min.is_empty() || max.is_empty() || min.len() != max.len() {
            return fail_no_match(label);
        }
        if min.iter().zip(&max).any(|(a, b)| a > b) {
            return fail_no_match(label);
        }
        Ok(DoubleRange {
            label: label.to_string(),
            min,
            max,
        })
    }

    /// `dims`.
    pub fn dims(&self) -> usize {
        self.min.len()
    }

    /// `org.apache.lucene.document.DoubleRange.verifyAndEncode(min, max,
    /// bytes)`.
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        let n = self.min.len();
        let mut bytes = vec![0u8; 16 * n];
        for d in 0..n {
            let (min, max) = (self.min[d], self.max[d]);
            if min.is_nan() {
                return Err(Error::IllegalArgument(
                    "invalid min value (NaN) in DoubleRange".into(),
                ));
            }
            if max.is_nan() {
                return Err(Error::IllegalArgument(
                    "invalid max value (NaN) in DoubleRange".into(),
                ));
            }
            if min > max {
                return Err(Error::IllegalArgument(format!(
                    "min value ({}) is greater than max value ({})",
                    java_double_string(min),
                    java_double_string(max)
                )));
            }
            lucene_util::numeric_utils::long_to_sortable_bytes(
                lucene_util::numeric_utils::double_to_sortable_long(min),
                &mut bytes,
                d * 8,
            );
            lucene_util::numeric_utils::long_to_sortable_bytes(
                lucene_util::numeric_utils::double_to_sortable_long(max),
                &mut bytes,
                (n + d) * 8,
            );
        }
        Ok(bytes)
    }
}

impl fmt::Display for DoubleRange {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoubleRange(label: {}, min: {}, max: {})",
            self.label,
            java_array(self.min.iter().map(|&v| java_double_string(v))),
            java_array(self.max.iter().map(|&v| java_double_string(v)))
        )
    }
}

/// `Arrays.toString`.
fn java_array(items: impl Iterator<Item = String>) -> String {
    format!("[{}]", items.collect::<Vec<_>>().join(", "))
}

#[cfg(test)]
mod tests;
