//! The `org.apache.lucene.facet.range` package: counting documents into
//! `long` and `double` ranges, and choosing ranges from the data
//! ([`dynamic_range_util`]).
//!
//! [`LongRangeFacetCounts`] and [`DoubleRangeFacetCounts`] count a numeric
//! field, a [`crate::values_source::LongValuesSource`] /
//! [`crate::values_source::DoubleValuesSource`], or a multi-valued source
//! into ranges, through a [`LongRangeCounter`] (the double ranges as
//! `NumericUtils.doubleToSortableLong` ranges). [`LongRange::get_query`] /
//! [`DoubleRange::get_query`] are a range as a filter query.
//!
//! The older dim-less primitives in [`crate::facets`]
//! ([`crate::facets::NumericRange`], [`crate::facets::range_facet_counts`])
//! are what the FFI layer calls; they predate this package's port and count
//! the same way.
//!
//! Differs: `equals`/`hashCode` and `LongRangeNode.toString` are not ported;
//! `getQuery`'s query is a [`RangeQuery`], a clause plus Java's `toString`.

mod double_range;
mod double_range_facet_counts;
pub mod dynamic_range_util;
mod long_range;
mod long_range_counter;
mod long_range_facet_counts;
mod range_facet_counts;
mod range_query;

pub use double_range::DoubleRange;
pub use double_range_facet_counts::DoubleRangeFacetCounts;
pub use long_range::LongRange;
pub(crate) use long_range_counter::LongRangeCounter;
pub use long_range_facet_counts::LongRangeFacetCounts;
pub use range_query::RangeQuery;

use crate::{Error, Result};

/// `Range`: a labelled range.
pub trait Range {
    /// `label`.
    fn label(&self) -> &str;
}

/// `Range.failNoMatch()`.
pub(crate) fn fail_no_match<T>(label: &str) -> Result<T> {
    Err(Error::IllegalArgument(format!(
        "range \"{label}\" matches nothing"
    )))
}
