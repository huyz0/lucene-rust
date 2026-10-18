//! The `org.apache.lucene.facet.rangeonrange` package: documents carrying a
//! range field (`LongRangeDocValuesField` / `DoubleRangeDocValuesField`, a
//! `BINARY` doc value of the encoded box) counted into query ranges, each
//! document range related to each query range by a
//! [`lucene_index::document::RangeQueryType`].
//!
//! Differs: `equals`/`hashCode` are not ported. Java fails on an empty
//! range list, and on ranges of mixed dimensions, with an
//! `ArrayIndexOutOfBoundsException` (or encodes the shorter ones
//! partly); both are an [`crate::Error::IllegalArgument`] here.

mod range_on_range_facet_counts;
mod ranges;

pub use range_on_range_facet_counts::{DoubleRangeOnRangeFacetCounts, LongRangeOnRangeFacetCounts};
pub use ranges::{DoubleRange, LongRange};
