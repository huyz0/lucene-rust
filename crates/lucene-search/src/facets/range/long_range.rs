//! `LongRange`: an inclusive range of `long` values.

use std::fmt;
use std::sync::Arc;

use super::range_query::{RangeQuery, RangeValues};
use super::{fail_no_match, Range};
use crate::facets::multi_long_values_source::MultiLongValuesSource;
use crate::query::BooleanQuery;
use crate::values_source::LongValuesSource;
use crate::Result;

/// `LongRange`: `[min, max]`, its exclusive ends moved in by one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongRange {
    /// `label`.
    pub label: String,
    /// `min`, inclusive.
    pub min: i64,
    /// `max`, inclusive.
    pub max: i64,
}

impl LongRange {
    /// `new LongRange(label, min, minInclusive, max, maxInclusive)`.
    ///
    /// # Errors
    /// [`crate::Error::IllegalArgument`] (`range "label" matches nothing`)
    /// when the range is empty.
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
            min,
            max,
        })
    }

    /// `accept(value)`.
    #[inline]
    pub fn accept(&self, value: i64) -> bool {
        value >= self.min && value <= self.max
    }

    /// `getQuery(fastMatchQuery, valueSource)`: the documents whose value is
    /// in the range (and that `fast_match_query` matches, when given).
    pub fn get_query(
        &self,
        fast_match_query: Option<BooleanQuery>,
        value_source: Arc<dyn LongValuesSource>,
    ) -> RangeQuery {
        RangeQuery::new(
            self.to_string(),
            fast_match_query,
            RangeValues::Long {
                min: self.min,
                max: self.max,
                source: value_source,
            },
        )
    }

    /// `getQuery(fastMatchQuery, valuesSource)` with a
    /// [`MultiLongValuesSource`]: a document matches when any of its values
    /// is in the range; a wrapped single-valued source is queried as one.
    pub fn get_query_multi(
        &self,
        fast_match_query: Option<BooleanQuery>,
        values_source: Arc<dyn MultiLongValuesSource>,
    ) -> RangeQuery {
        if let Some(single) = values_source.single_valued() {
            return self.get_query(fast_match_query, single);
        }
        RangeQuery::new(
            self.to_string(),
            fast_match_query,
            RangeValues::MultiLong {
                min: self.min,
                max: self.max,
                source: values_source,
            },
        )
    }
}

impl Range for LongRange {
    fn label(&self) -> &str {
        &self.label
    }
}

impl fmt::Display for LongRange {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LongRange({}: {} to {})", self.label, self.min, self.max)
    }
}

#[cfg(test)]
mod tests;
