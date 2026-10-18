//! `FacetSetMatcher`, `ExactFacetSetMatcher`, `RangeFacetSetMatcher` and
//! `DimRange`.

use std::fmt;

use lucene_util::numeric_utils::{double_to_sortable_long, float_to_sortable_int};

use super::FacetSet;
use crate::{Error, Result};

/// `FacetSetMatcher`: a labelled test of a decoded facet set.
pub trait FacetSetMatcher: Send + Sync {
    /// `label`.
    fn label(&self) -> &str;
    /// `dims`.
    fn dims(&self) -> usize;
    /// `matches(dimValues)`.
    fn matches(&self, dim_values: &[i64]) -> bool;
}

/// `FacetSetMatcher`'s constructor checks: Java refuses `dims <= 0`, which
/// the callers' own checks already rule out.
fn check_dims(dims: usize) -> Result<()> {
    if dims == 0 {
        return Err(Error::IllegalArgument(
            "dims must be greater than 0: 0".into(),
        ));
    }
    Ok(())
}

/// `ExactFacetSetMatcher`: the facet set equal to one given set.
#[derive(Debug, Clone)]
pub struct ExactFacetSetMatcher {
    label: String,
    values: Vec<i64>,
}

impl ExactFacetSetMatcher {
    /// `new ExactFacetSetMatcher(label, facetSet)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a set of no dimensions.
    pub fn new(label: &str, facet_set: &dyn FacetSet) -> Result<Self> {
        check_dims(facet_set.dims())?;
        Ok(ExactFacetSetMatcher {
            label: label.to_string(),
            values: facet_set.get_comparable_values(),
        })
    }
}

impl FacetSetMatcher for ExactFacetSetMatcher {
    fn label(&self) -> &str {
        &self.label
    }
    fn dims(&self) -> usize {
        self.values.len()
    }
    fn matches(&self, dim_values: &[i64]) -> bool {
        dim_values == self.values.as_slice()
    }
}

/// `RangeFacetSetMatcher`: the facet sets inside a box, one [`DimRange`]
/// per dimension.
#[derive(Debug, Clone)]
pub struct RangeFacetSetMatcher {
    label: String,
    lower: Vec<i64>,
    upper: Vec<i64>,
}

impl RangeFacetSetMatcher {
    /// `new RangeFacetSetMatcher(label, dimRanges...)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for no ranges.
    pub fn new(label: &str, dim_ranges: &[DimRange]) -> Result<Self> {
        if dim_ranges.is_empty() {
            return Err(Error::IllegalArgument(
                "dimRanges cannot be null or empty".into(),
            ));
        }
        Ok(RangeFacetSetMatcher {
            label: label.to_string(),
            lower: dim_ranges.iter().map(|r| r.min).collect(),
            upper: dim_ranges.iter().map(|r| r.max).collect(),
        })
    }
}

impl FacetSetMatcher for RangeFacetSetMatcher {
    fn label(&self) -> &str {
        &self.label
    }
    fn dims(&self) -> usize {
        self.lower.len()
    }
    /// Java asserts the dimensions agree and indexes past the ranges
    /// otherwise; a set of more dimensions than the ranges does not match
    /// here.
    fn matches(&self, dim_values: &[i64]) -> bool {
        if dim_values.len() > self.lower.len() {
            return false;
        }
        dim_values
            .iter()
            .zip(self.lower.iter().zip(&self.upper))
            .all(|(v, (lo, hi))| v >= lo && v <= hi)
    }
}

/// `DimRange`: an inclusive range of one dimension's comparable values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimRange {
    /// `min`.
    pub min: i64,
    /// `max`.
    pub max: i64,
}

impl DimRange {
    /// `new DimRange(min, max)`.
    pub fn new(min: i64, max: i64) -> Self {
        DimRange { min, max }
    }

    /// `DimRange.fromLongs(min, minInclusive, max, maxInclusive)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an exclusive end that cannot move,
    /// and for an empty range.
    pub fn from_longs(
        min: i64,
        min_inclusive: bool,
        max: i64,
        max_inclusive: bool,
    ) -> Result<Self> {
        let mut min = min;
        let mut max = max;
        if !min_inclusive {
            if min == i64::MAX {
                return Err(Error::IllegalArgument(format!("Invalid min input: {min}")));
            }
            min = min.wrapping_add(1);
        }
        if !max_inclusive {
            if max == i64::MIN {
                return Err(Error::IllegalArgument(format!("Invalid max input: {max}")));
            }
            max = max.wrapping_sub(1);
        }
        if min > max {
            return Err(Error::IllegalArgument(format!(
                "Minimum cannot be greater than maximum, max={max}, min={min}"
            )));
        }
        Ok(DimRange { min, max })
    }

    /// `DimRange.fromDoubles(min, minInclusive, max, maxInclusive)`: as
    /// sortable longs.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a `NaN` end and for an empty range.
    pub fn from_doubles(
        min: f64,
        min_inclusive: bool,
        max: f64,
        max_inclusive: bool,
    ) -> Result<Self> {
        let mut min = min;
        let mut max = max;
        if min.is_nan() {
            return Err(Error::IllegalArgument("min cannot be NaN".into()));
        }
        if !min_inclusive {
            min = min.next_up();
        }
        if max.is_nan() {
            return Err(Error::IllegalArgument("max cannot be NaN".into()));
        }
        if !max_inclusive {
            max = max.next_down();
        }
        if min > max {
            return Err(Error::IllegalArgument(
                "Minimum cannot be greater than maximum".into(),
            ));
        }
        Ok(DimRange {
            min: double_to_sortable_long(min),
            max: double_to_sortable_long(max),
        })
    }

    /// `DimRange.fromFloats(min, minInclusive, max, maxInclusive)`: as
    /// sortable ints.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a `NaN` end and for an empty range.
    pub fn from_floats(
        min: f32,
        min_inclusive: bool,
        max: f32,
        max_inclusive: bool,
    ) -> Result<Self> {
        let mut min = min;
        let mut max = max;
        if min.is_nan() {
            return Err(Error::IllegalArgument("min cannot be NaN".into()));
        }
        if !min_inclusive {
            min = min.next_up();
        }
        if max.is_nan() {
            return Err(Error::IllegalArgument("max cannot be NaN".into()));
        }
        if !max_inclusive {
            max = max.next_down();
        }
        if min > max {
            return Err(Error::IllegalArgument(
                "Minimum cannot be greater than maximum".into(),
            ));
        }
        Ok(DimRange {
            min: i64::from(float_to_sortable_int(min)),
            max: i64::from(float_to_sortable_int(max)),
        })
    }
}

impl fmt::Display for DimRange {
    /// The record's `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DimRange[min={}, max={}]", self.min, self.max)
    }
}

#[cfg(test)]
mod tests;
