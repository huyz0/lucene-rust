//! `RangeFacetCounts`: the base of [`super::LongRangeFacetCounts`] and
//! [`super::DoubleRangeFacetCounts`] -- the count loops over a field's doc
//! values or a values source, and the results.

use std::fmt;

use super::{LongRange, LongRangeCounter};
use crate::facets::facet_counts_with_filter_query::FacetCountsWithFilterQuery;
use crate::facets::facet_result::{FacetResult, FacetValue, LabelAndValue};
use crate::facets::facet_utils::{
    for_each_hit, for_each_iter, sorted_numeric_is_single_valued, validate_dim_and_path,
};
use crate::facets::facets_collector::{FacetsCollector, MatchingDocs};
use crate::facets::facets_trait::{validate_top_n, Facets};
use crate::facets::multi_long_values_source::BoxMultiLongValues;
use crate::facets::taxonomy::facet_label::java_compare;
use crate::facets::top_ord_and_number_queue::PriorityQueue;
use crate::index_searcher::IndexSearcher;
use crate::query::BooleanQuery;
use crate::reader::doc_values::get_sorted_numeric;
use crate::reader::DocIdSetIterator;
use crate::{Error, Result};

/// One segment's single values, as `long`s (`LongValues`, or `DoubleValues`
/// through `NumericUtils.doubleToSortableLong`).
pub(crate) trait SingleLongs {
    /// `advanceExact(doc)`.
    fn advance_exact(&mut self, doc: i32) -> Result<bool>;
    /// The value, mapped to `long` space.
    fn value(&mut self) -> Result<i64>;
}

/// `RangeFacetCounts`.
pub(crate) struct RangeFacetCounts {
    base: FacetCountsWithFilterQuery,
    field: String,
    /// `ranges[i].label`.
    labels: Vec<String>,
    /// `getLongRanges()`.
    long_ranges: Vec<LongRange>,
    /// `counts`: `None` until something was counted.
    counts: Option<Vec<i32>>,
    tot_count: i32,
}

impl RangeFacetCounts {
    pub(crate) fn new(
        field: &str,
        labels: Vec<String>,
        long_ranges: Vec<LongRange>,
        fast_match_query: Option<BooleanQuery>,
    ) -> Self {
        RangeFacetCounts {
            base: FacetCountsWithFilterQuery::new(fast_match_query),
            field: field.to_string(),
            labels,
            long_ranges,
            counts: None,
            tot_count: 0,
        }
    }

    /// `createIterator(hits)`: the hits, less what the fast-match query (run
    /// through `searcher`) does not match.
    fn create_iterator<'i>(
        &self,
        searcher: Option<&IndexSearcher<'_, '_>>,
        hits: &'i MatchingDocs<'_>,
    ) -> Result<Option<Box<dyn DocIdSetIterator + 'i>>> {
        if self.base.fast_match_query().is_none() {
            return Ok(Some(Box::new(hits.bits.iterator())));
        }
        let Some(searcher) = searcher else {
            return Err(Error::IllegalState(
                "a fast-match query needs the searcher the hits came from".into(),
            ));
        };
        self.base.create_iterator(searcher, hits, Vec::new())
    }

    /// `setupCounter()`.
    fn setup_counter(&self) -> LongRangeCounter {
        LongRangeCounter::create(&self.long_ranges)
    }

    fn finish(&mut self, mut counter: LongRangeCounter, missing: i32) {
        let missing = missing.wrapping_add(counter.finish());
        self.tot_count = self.tot_count.wrapping_sub(missing);
        self.counts = Some(counter.into_counts());
    }

    /// `count(field, matchingDocs)`: the field's `NUMERIC` /
    /// `SORTED_NUMERIC` doc values, each mapped through `map`
    /// (`mapDocValue`).
    ///
    /// When no segment with hits is multi-valued, Java counts through the
    /// single-valued path, which adds each segment's `totalHits` to
    /// `totCount` -- the hits, not the fast-match query's subset of them --
    /// and so does this.
    pub(crate) fn count_field(
        &mut self,
        fc: &FacetsCollector<'_>,
        searcher: Option<&IndexSearcher<'_, '_>>,
        map: fn(i64) -> i64,
    ) -> Result<()> {
        let mut any = false;
        let mut found_multi_valued = false;
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            // `DocValues.getSortedNumeric` checks the field's type.
            get_sorted_numeric(hits.context.reader, &self.field)?;
            any = true;
            if !found_multi_valued
                && !sorted_numeric_is_single_valued(hits.context.reader, &self.field)
            {
                found_multi_valued = true;
            }
        }
        if !any {
            return Ok(());
        }
        let mut counter = self.setup_counter();
        let mut missing = 0i32;
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            if !found_multi_valued && self.base.fast_match_query().is_none() {
                // The hits alone: walked as a set, as `for_each_hit` does,
                // over `unwrapSingleton`'s `NUMERIC` values where the field
                // has them. A hit without a value is `missing`.
                self.tot_count = self.tot_count.wrapping_add(hits.total_hits);
                let mut seen = 0usize;
                let reader = hits.context.reader;
                if let Some(mut single) = reader.numeric_doc_values(&self.field)? {
                    for_each_hit(&hits.bits, single.as_mut(), |v| {
                        counter.add_single_valued(map(v.long_value()));
                        seen = seen.wrapping_add(1);
                        Ok(())
                    })?;
                } else {
                    let mut values = get_sorted_numeric(reader, &self.field)?;
                    for_each_hit(&hits.bits, values.as_mut(), |v| {
                        counter.add_single_valued(map(v.next_value()?));
                        seen = seen.wrapping_add(1);
                        Ok(())
                    })?;
                }
                let unseen = hits.bits.len().saturating_sub(seen);
                missing = missing.wrapping_add(i32::try_from(unseen).unwrap_or(i32::MAX));
                continue;
            }
            let Some(mut it) = self.create_iterator(searcher, hits)? else {
                continue;
            };
            let mut values = get_sorted_numeric(hits.context.reader, &self.field)?;
            if !found_multi_valued {
                self.tot_count = self.tot_count.wrapping_add(hits.total_hits);
                for_each_iter(it.as_mut(), |doc| {
                    if values.advance_exact(doc)? {
                        counter.add_single_valued(map(values.next_value()?));
                    } else {
                        missing = missing.wrapping_add(1);
                    }
                    Ok(())
                })?;
            } else {
                let tot = &mut self.tot_count;
                for_each_iter(it.as_mut(), |doc| {
                    if !values.advance_exact(doc)? {
                        return Ok(());
                    }
                    let limit = values.doc_value_count();
                    if limit == 1 {
                        counter.add_single_valued(map(values.next_value()?));
                        *tot = tot.wrapping_add(1);
                        return Ok(());
                    }
                    counter.start_multi_valued_doc();
                    let mut previous = 0i64;
                    for j in 0..limit {
                        let val = map(values.next_value()?);
                        if j == 0 || val != previous {
                            counter.add_multi_valued(val);
                            previous = val;
                        }
                    }
                    if counter.end_multi_valued_doc() {
                        *tot = tot.wrapping_add(1);
                    }
                    Ok(())
                })?;
            }
        }
        self.finish(counter, missing);
        Ok(())
    }

    /// `count(LongValuesSource | DoubleValuesSource, matchingDocs)`:
    /// `values(leaf)` is the segment's values.
    pub(crate) fn count_single<'c>(
        &mut self,
        fc: &FacetsCollector<'_>,
        searcher: Option<&IndexSearcher<'_, '_>>,
        mut values: impl FnMut(usize) -> Result<Box<dyn SingleLongs + 'c>>,
    ) -> Result<()> {
        let mut counter: Option<LongRangeCounter> = None;
        let mut missing = 0i32;
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            let Some(mut it) = self.create_iterator(searcher, hits)? else {
                continue;
            };
            let c = counter.get_or_insert_with(|| self.setup_counter());
            let mut fv = values(hits.context.ord)?;
            self.tot_count = self.tot_count.wrapping_add(hits.total_hits);
            for_each_iter(it.as_mut(), |doc| {
                if fv.advance_exact(doc)? {
                    c.add_single_valued(fv.value()?);
                } else {
                    missing = missing.wrapping_add(1);
                }
                Ok(())
            })?;
        }
        if let Some(counter) = counter {
            self.finish(counter, missing);
        }
        Ok(())
    }

    /// `count(MultiLongValuesSource | MultiDoubleValuesSource,
    /// matchingDocs)`: `values(leaf)` is the segment's values, as `long`s.
    pub(crate) fn count_multi<'c>(
        &mut self,
        fc: &FacetsCollector<'_>,
        searcher: Option<&IndexSearcher<'_, '_>>,
        mut values: impl FnMut(usize) -> Result<BoxMultiLongValues<'c>>,
    ) -> Result<()> {
        let mut counter: Option<LongRangeCounter> = None;
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            let Some(mut it) = self.create_iterator(searcher, hits)? else {
                continue;
            };
            let c = counter.get_or_insert_with(|| self.setup_counter());
            let mut mv = values(hits.context.ord)?;
            let tot = &mut self.tot_count;
            for_each_iter(it.as_mut(), |doc| {
                if !mv.advance_exact(doc)? {
                    return Ok(());
                }
                let limit = mv.get_value_count();
                if limit == 1 {
                    c.add_single_valued(mv.next_value()?);
                    *tot = tot.wrapping_add(1);
                    return Ok(());
                }
                c.start_multi_valued_doc();
                let mut previous = 0i64;
                let mut i = 0i64;
                while i < limit {
                    let val = mv.next_value()?;
                    if i == 0 || val != previous {
                        c.add_multi_valued(val);
                        previous = val;
                    }
                    i = i.saturating_add(1);
                }
                if c.end_multi_valued_doc() {
                    *tot = tot.wrapping_add(1);
                }
                Ok(())
            })?;
        }
        if let Some(counter) = counter {
            self.finish(counter, 0);
        }
        Ok(())
    }

    fn count_of(&self, i: usize) -> i32 {
        self.counts.as_ref().map_or(0, |c| c[i])
    }
}

/// `RangeFacetCounts.Entry`.
struct Entry<'a> {
    count: i32,
    label: &'a str,
}

impl Facets for RangeFacetCounts {
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
        validate_dim_and_path(&self.field, dim, path)?;
        let label_values: Vec<LabelAndValue> = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, l)| LabelAndValue::new(l.clone(), FacetValue::Int(self.count_of(i))))
            .collect();
        let n = i32::try_from(label_values.len()).unwrap_or(i32::MAX);
        Ok(Some(FacetResult::new(
            dim.to_string(),
            path.iter().map(|p| p.to_string()).collect(),
            FacetValue::Int(self.tot_count),
            label_values,
            n,
        )))
    }

    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        validate_top_n(top_n)?;
        validate_dim_and_path(&self.field, dim, path)?;
        let path_vec: Vec<String> = path.iter().map(|p| p.to_string()).collect();
        let Some(counts) = &self.counts else {
            return Ok(Some(FacetResult::new(
                dim.to_string(),
                path_vec,
                FacetValue::Int(self.tot_count),
                Vec::new(),
                0,
            )));
        };
        let max = usize::try_from(top_n).unwrap_or(0).min(counts.len());
        // Count descending, ties by label ascending.
        let mut pq = PriorityQueue::new(max, |a: &Entry<'_>, b: &Entry<'_>| {
            a.count < b.count || (a.count == b.count && java_compare(b.label, a.label).is_lt())
        });
        let mut child_count = 0i32;
        for (i, &count) in counts.iter().enumerate() {
            if count != 0 {
                child_count = child_count.wrapping_add(1);
                pq.insert_with_overflow(Entry {
                    count,
                    label: &self.labels[i],
                });
            }
        }
        let mut results = Vec::with_capacity(pq.size());
        while let Some(e) = pq.pop() {
            results.push(LabelAndValue::new(
                e.label.to_string(),
                FacetValue::Int(e.count),
            ));
        }
        results.reverse();
        Ok(Some(FacetResult::new(
            dim.to_string(),
            path_vec,
            FacetValue::Int(self.tot_count),
            results,
            child_count,
        )))
    }

    fn get_specific_value(&self, _dim: &str, _path: &[&str]) -> Result<FacetValue> {
        Err(Error::Unsupported(
            "RangeFacetCounts.getSpecificValue".into(),
        ))
    }

    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        validate_top_n(top_n)?;
        Ok(self
            .get_top_children(top_n, &self.field, &[])?
            .into_iter()
            .collect())
    }
}

impl fmt::Display for RangeFacetCounts {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RangeFacetCounts totCount={}:", self.tot_count)?;
        for (i, label) in self.labels.iter().enumerate() {
            writeln!(f, "  {label} -> count={}", self.count_of(i))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
