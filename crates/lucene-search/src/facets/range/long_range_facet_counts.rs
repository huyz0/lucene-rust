//! `LongRangeFacetCounts`: documents counted into [`LongRange`]s.

use std::fmt;

use super::range_facet_counts::{RangeFacetCounts, SingleLongs};
use super::LongRange;
use crate::facets::facet_result::{FacetResult, FacetValue};
use crate::facets::facets_collector::FacetsCollector;
use crate::facets::facets_trait::Facets;
use crate::facets::multi_long_values_source::MultiLongValuesSource;
use crate::index_searcher::IndexSearcher;
use crate::query::BooleanQuery;
use crate::values_source::{BoxLongValues, LongValuesSource, ValuesContext};
use crate::Result;

/// `LongRangeFacetCounts`.
pub struct LongRangeFacetCounts {
    inner: RangeFacetCounts,
}

/// A `LongValues` as [`SingleLongs`].
struct Longs<'c>(BoxLongValues<'c>);

impl SingleLongs for Longs<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        self.0.advance_exact(doc)
    }
    fn value(&mut self) -> Result<i64> {
        self.0.long_value()
    }
}

impl LongRangeFacetCounts {
    fn base(field: &str, ranges: &[LongRange], fast: Option<BooleanQuery>) -> RangeFacetCounts {
        RangeFacetCounts::new(
            field,
            ranges.iter().map(|r| r.label.clone()).collect(),
            ranges.to_vec(),
            fast,
        )
    }

    /// `new LongRangeFacetCounts(field, hits, ranges...)`: the field's
    /// `NUMERIC` / `SORTED_NUMERIC` doc values.
    ///
    /// # Errors
    /// [`crate::Error::IllegalState`] for a field with doc values of another
    /// kind; whatever reading them reports.
    pub fn new(field: &str, hits: &FacetsCollector<'_>, ranges: &[LongRange]) -> Result<Self> {
        Self::with_fast_match(field, None, hits, None, ranges)
    }

    /// `new LongRangeFacetCounts(field, null, hits, fastMatchQuery,
    /// ranges...)`: only the hits `fast_match_query` matches, run through
    /// `searcher` (the one the hits came from; needed only with a query).
    ///
    /// # Errors
    /// As [`Self::new`]; [`crate::Error::IllegalState`] for a query without
    /// a searcher.
    pub fn with_fast_match(
        field: &str,
        searcher: Option<&IndexSearcher<'_, '_>>,
        hits: &FacetsCollector<'_>,
        fast_match_query: Option<BooleanQuery>,
        ranges: &[LongRange],
    ) -> Result<Self> {
        let mut inner = Self::base(field, ranges, fast_match_query);
        inner.count_field(hits, searcher, |v| v)?;
        Ok(LongRangeFacetCounts { inner })
    }

    /// `new LongRangeFacetCounts(field, valueSource, hits, fastMatchQuery,
    /// ranges...)`: the values of `value_source`, read through `ctx` (a
    /// context over the searcher the hits came from, which also runs the
    /// query).
    ///
    /// # Errors
    /// Whatever reading the values reports.
    pub fn with_values_source(
        field: &str,
        value_source: &dyn LongValuesSource,
        ctx: &ValuesContext<'_>,
        hits: &FacetsCollector<'_>,
        fast_match_query: Option<BooleanQuery>,
        ranges: &[LongRange],
    ) -> Result<Self> {
        let searcher = if fast_match_query.is_some() {
            Some(ctx.searcher()?)
        } else {
            None
        };
        let mut inner = Self::base(field, ranges, fast_match_query);
        inner.count_single(hits, searcher, |leaf| {
            Ok(Box::new(Longs(value_source.get_values(ctx, leaf, None)?)) as Box<dyn SingleLongs>)
        })?;
        Ok(LongRangeFacetCounts { inner })
    }

    /// `new LongRangeFacetCounts(field, valuesSource, hits, fastMatchQuery,
    /// ranges...)` with a [`MultiLongValuesSource`]: a wrapped single-valued
    /// source is counted as one.
    ///
    /// # Errors
    /// Whatever reading the values reports.
    pub fn with_multi_values_source(
        field: &str,
        values_source: &dyn MultiLongValuesSource,
        ctx: &ValuesContext<'_>,
        hits: &FacetsCollector<'_>,
        fast_match_query: Option<BooleanQuery>,
        ranges: &[LongRange],
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
        inner.count_multi(hits, searcher, |leaf| values_source.get_values(ctx, leaf))?;
        Ok(LongRangeFacetCounts { inner })
    }
}

impl Facets for LongRangeFacetCounts {
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

impl fmt::Display for LongRangeFacetCounts {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl fmt::Debug for LongRangeFacetCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

#[cfg(test)]
mod tests;
