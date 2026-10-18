//! `DoubleRangeFacetCounts`: documents counted into [`DoubleRange`]s,
//! through `NumericUtils.doubleToSortableLong` ranges.

use std::fmt;

use lucene_util::numeric_utils::{double_to_sortable_long, sortable_double_bits};

use super::range_facet_counts::{RangeFacetCounts, SingleLongs};
use super::DoubleRange;
use crate::facets::facet_result::{FacetResult, FacetValue};
use crate::facets::facets_collector::FacetsCollector;
use crate::facets::facets_trait::Facets;
use crate::facets::multi_double_values_source::{sortable_longs, MultiDoubleValuesSource};
use crate::index_searcher::IndexSearcher;
use crate::query::BooleanQuery;
use crate::values_source::{BoxDoubleValues, DoubleValuesSource, ValuesContext};
use crate::Result;

/// `DoubleRangeFacetCounts`.
pub struct DoubleRangeFacetCounts {
    inner: RangeFacetCounts,
}

/// A `DoubleValues` as sortable longs.
struct SortableLongs<'c>(BoxDoubleValues<'c>);

impl SingleLongs for SortableLongs<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.0.advance_exact(doc)
    }
    fn value(&mut self) -> Result<i64> {
        Ok(double_to_sortable_long(self.0.double_value()?))
    }
}

/// `mapDocValue`: a field's `double` bits as a sortable long.
fn map_doc_value(v: i64) -> i64 {
    sortable_double_bits(v)
}

impl DoubleRangeFacetCounts {
    fn base(field: &str, ranges: &[DoubleRange], fast: Option<BooleanQuery>) -> RangeFacetCounts {
        RangeFacetCounts::new(
            field,
            ranges.iter().map(|r| r.label.clone()).collect(),
            ranges.iter().map(DoubleRange::to_long_range).collect(),
            fast,
        )
    }

    /// `new DoubleRangeFacetCounts(field, hits, ranges...)`: the field's
    /// `NUMERIC` / `SORTED_NUMERIC` doc values, each a `double`'s bits
    /// (`DoublePoint`'s doc values; a `float` field needs a values source).
    ///
    /// # Errors
    /// [`crate::Error::IllegalState`] for a field with doc values of another
    /// kind; whatever reading them reports.
    pub fn new(field: &str, hits: &FacetsCollector<'_>, ranges: &[DoubleRange]) -> Result<Self> {
        Self::with_fast_match(field, None, hits, None, ranges)
    }

    /// `new DoubleRangeFacetCounts(field, null, hits, fastMatchQuery,
    /// ranges...)`; `searcher` as in
    /// [`super::LongRangeFacetCounts::with_fast_match`].
    ///
    /// # Errors
    /// As [`Self::new`]; [`crate::Error::IllegalState`] for a query without
    /// a searcher.
    pub fn with_fast_match(
        field: &str,
        searcher: Option<&IndexSearcher<'_, '_>>,
        hits: &FacetsCollector<'_>,
        fast_match_query: Option<BooleanQuery>,
        ranges: &[DoubleRange],
    ) -> Result<Self> {
        let mut inner = Self::base(field, ranges, fast_match_query);
        inner.count_field(hits, searcher, map_doc_value)?;
        Ok(DoubleRangeFacetCounts { inner })
    }

    /// `new DoubleRangeFacetCounts(field, valueSource, hits,
    /// fastMatchQuery, ranges...)`.
    ///
    /// # Errors
    /// Whatever reading the values reports.
    pub fn with_values_source(
        field: &str,
        value_source: &dyn DoubleValuesSource,
        ctx: &ValuesContext<'_>,
        hits: &FacetsCollector<'_>,
        fast_match_query: Option<BooleanQuery>,
        ranges: &[DoubleRange],
    ) -> Result<Self> {
        let searcher = if fast_match_query.is_some() {
            Some(ctx.searcher()?)
        } else {
            None
        };
        let mut inner = Self::base(field, ranges, fast_match_query);
        inner.count_single(hits, searcher, |leaf| {
            Ok(
                Box::new(SortableLongs(value_source.get_values(ctx, leaf, None)?))
                    as Box<dyn SingleLongs>,
            )
        })?;
        Ok(DoubleRangeFacetCounts { inner })
    }

    /// `new DoubleRangeFacetCounts(field, valuesSource, hits,
    /// fastMatchQuery, ranges...)` with a [`MultiDoubleValuesSource`]: a
    /// wrapped single-valued source is counted as one.
    ///
    /// # Errors
    /// Whatever reading the values reports.
    pub fn with_multi_values_source(
        field: &str,
        values_source: &dyn MultiDoubleValuesSource,
        ctx: &ValuesContext<'_>,
        hits: &FacetsCollector<'_>,
        fast_match_query: Option<BooleanQuery>,
        ranges: &[DoubleRange],
    ) -> Result<Self> {
        if let Some(single) = values_source.single_valued() {
            return Self::with_values_source(
                field,
                single.as_ref(),
                ctx,
                hits,
                fast_match_query,
                ranges,
            );
        }
        let searcher = if fast_match_query.is_some() {
            Some(ctx.searcher()?)
        } else {
            None
        };
        let mut inner = Self::base(field, ranges, fast_match_query);
        inner.count_multi(hits, searcher, |leaf| {
            Ok(sortable_longs(values_source.get_values(ctx, leaf)?))
        })?;
        Ok(DoubleRangeFacetCounts { inner })
    }
}

impl Facets for DoubleRangeFacetCounts {
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
        self.inner.get_all_children(dim, path)
    }

    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        self.inner.get_top_children(top_n, dim, path)
    }

    fn get_specific_value(&self, dim: &str, path: &[&str]) -> Result<FacetValue> {
        self.inner.get_specific_value(dim, path)
    }

    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        self.inner.get_all_dims(top_n)
    }
}

impl fmt::Display for DoubleRangeFacetCounts {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl fmt::Debug for DoubleRangeFacetCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

#[cfg(test)]
mod tests;
