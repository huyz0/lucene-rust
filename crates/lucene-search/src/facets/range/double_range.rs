//! `DoubleRange`: an inclusive range of `double` values.

use std::fmt;
use std::sync::Arc;

use lucene_util::geo::java_double_string;
use lucene_util::numeric_utils::double_to_sortable_long;

use super::range_query::{RangeQuery, RangeValues};
use super::{fail_no_match, LongRange, Range};
use crate::facets::multi_double_values_source::MultiDoubleValuesSource;
use crate::query::BooleanQuery;
use crate::values_source::DoubleValuesSource;
use crate::{Error, Result};

/// `DoubleRange`: `[min, max]`, its exclusive ends moved in to the next
/// representable double.
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleRange {
    /// `label`.
    pub label: String,
    /// `min`, inclusive.
    pub min: f64,
    /// `max`, inclusive.
    pub max: f64,
}

impl DoubleRange {
    /// `new DoubleRange(label, min, minInclusive, max, maxInclusive)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a `NaN` end (`min cannot be NaN`,
    /// `max cannot be NaN`) and for an empty range.
    pub fn new(
        label: &str,
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
            // `Math.nextAfter(max, Double.NEGATIVE_INFINITY)`.
            max = max.next_down();
        }
        if min > max {
            return fail_no_match(label);
        }
        Ok(DoubleRange {
            label: label.to_string(),
            min,
            max,
        })
    }

    /// `accept(value)`.
    #[inline]
    pub fn accept(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }

    /// `toLongRange()`: the range in `NumericUtils.doubleToSortableLong`
    /// space.
    pub(crate) fn to_long_range(&self) -> LongRange {
        LongRange {
            label: self.label.clone(),
            min: double_to_sortable_long(self.min),
            max: double_to_sortable_long(self.max),
        }
    }

    /// `getQuery(fastMatchQuery, valueSource)`.
    pub fn get_query(
        &self,
        fast_match_query: Option<BooleanQuery>,
        value_source: Arc<dyn DoubleValuesSource>,
    ) -> RangeQuery {
        RangeQuery::new(
            self.to_string(),
            fast_match_query,
            RangeValues::Double {
                min: self.min,
                max: self.max,
                source: value_source,
            },
        )
    }

    /// `getQuery(fastMatchQuery, valuesSource)` with a
    /// [`MultiDoubleValuesSource`].
    pub fn get_query_multi(
        &self,
        fast_match_query: Option<BooleanQuery>,
        values_source: Arc<dyn MultiDoubleValuesSource>,
    ) -> RangeQuery {
        if let Some(single) = values_source.single_valued() {
            return self.get_query(fast_match_query, single);
        }
        RangeQuery::new(
            self.to_string(),
            fast_match_query,
            RangeValues::MultiDouble {
                min: self.min,
                max: self.max,
                source: values_source,
            },
        )
    }
}

impl Range for DoubleRange {
    fn label(&self) -> &str {
        &self.label
    }
}

impl fmt::Display for DoubleRange {
    /// `toString()`, the ends as Java prints a `double`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DoubleRange({}: {} to {})",
            self.label,
            java_double_string(self.min),
            java_double_string(self.max)
        )
    }
}

#[cfg(test)]
mod tests;
