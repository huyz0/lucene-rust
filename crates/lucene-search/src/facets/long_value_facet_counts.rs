//! `LongValueFacetCounts`: a count per distinct `long` value of a field or
//! values source, values in `[0, 1024)` in an array and the rest in a hash
//! table.
//!
//! Differs: Java iterates its hash table (`getAllChildren`, `toString`) in an
//! order seeded per process; here the hashed values come in ascending order,
//! after the array's. `getTopChildren` and `getAllChildrenSortByValue` are
//! totally ordered and so the same. Java's `unwrapSingleton` fast path for a
//! single-valued field is the general path here (a document with one value
//! counts the same either way).

use std::fmt;

use super::facet_result::{FacetResult, FacetValue, LabelAndValue};
use super::facet_utils::{for_each_doc, for_each_hit, validate_dim_and_path};
use super::facets_collector::FacetsCollector;
use super::facets_trait::{validate_top_n, Facets};
use super::multi_long_values_source::{MultiLongValues, MultiLongValuesSource};
use super::top_ord_and_number_queue::PriorityQueue;
use crate::grouping::FxHashMap;
use crate::reader::doc_values::get_sorted_numeric;
use crate::reader::{IndexReader, SortedNumericDocValues, NO_MORE_DOCS};
use crate::values_source::{LongValuesSource, ValuesContext};
use crate::{Error, Result};

/// The values the array counts: `[0, COUNTS_LEN)`.
const COUNTS_LEN: usize = 1024;

/// `LongValueFacetCounts`.
pub struct LongValueFacetCounts {
    /// Counts of the values in `[0, 1024)`; empty until initialized.
    counts: Vec<i32>,
    /// Counts of the other values.
    hash_counts: FxHashMap<i64, i32>,
    /// Whether the counters were initialized (something was counted).
    initialized: bool,
    field: String,
    /// Documents with a value counted.
    tot_count: i32,
}

impl LongValueFacetCounts {
    fn empty(field: &str) -> Self {
        LongValueFacetCounts {
            counts: Vec::new(),
            hash_counts: FxHashMap::default(),
            initialized: false,
            field: field.to_string(),
            tot_count: 0,
        }
    }

    /// `new LongValueFacetCounts(field, hits)`: the field's `NUMERIC` or
    /// `SORTED_NUMERIC` doc values over the matching documents.
    ///
    /// # Errors
    /// [`Error::IllegalState`] for a field with doc values of another kind;
    /// whatever reading them reports.
    pub fn new(field: &str, hits: &FacetsCollector<'_>) -> Result<Self> {
        let mut c = Self::empty(field);
        c.count_field(hits)?;
        Ok(c)
    }

    /// `new LongValueFacetCounts(field, valueSource, hits)`: the values of
    /// `value_source`, read through `ctx` (the searcher `hits` came from).
    ///
    /// # Errors
    /// Whatever reading the values reports.
    pub fn with_values_source(
        field: &str,
        value_source: &dyn LongValuesSource,
        ctx: &ValuesContext<'_>,
        hits: &FacetsCollector<'_>,
    ) -> Result<Self> {
        let mut c = Self::empty(field);
        c.count_source(value_source, ctx, hits)?;
        Ok(c)
    }

    /// `new LongValueFacetCounts(field, valuesSource, hits)` with a
    /// [`MultiLongValuesSource`]: a wrapped single-valued source is counted
    /// as one.
    ///
    /// # Errors
    /// Whatever reading the values reports.
    pub fn with_multi_values_source(
        field: &str,
        values_source: &dyn MultiLongValuesSource,
        ctx: &ValuesContext<'_>,
        hits: &FacetsCollector<'_>,
    ) -> Result<Self> {
        let mut c = Self::empty(field);
        match values_source.single_valued() {
            Some(single) => c.count_source(single.as_ref(), ctx, hits)?,
            None => c.count_multi_source(values_source, ctx, hits)?,
        }
        Ok(c)
    }

    /// `new LongValueFacetCounts(field, reader)`: every live document's
    /// values of the field.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn count_all(field: &str, reader: &dyn IndexReader) -> Result<Self> {
        let mut c = Self::empty(field);
        c.initialize_counters();
        for leaf in reader.leaves() {
            let mut values = get_sorted_numeric(leaf.reader, field)?;
            match leaf.reader.live_docs() {
                None => {
                    let mut doc = values.next_doc()?;
                    while doc != NO_MORE_DOCS {
                        c.add_doc(values.as_mut())?;
                        doc = values.next_doc()?;
                    }
                }
                Some(live) => {
                    let mut doc = values.next_doc()?;
                    while doc != NO_MORE_DOCS {
                        if live.get_doc(doc) {
                            c.add_doc(values.as_mut())?;
                        }
                        doc = values.next_doc()?;
                    }
                }
            }
        }
        Ok(c)
    }

    /// `new LongValueFacetCounts(field, valueSource, reader)`: every
    /// document's value (deleted documents included, as Java's loop over
    /// `0..maxDoc` does -- `fixtures/data/value_facets`' `multi` index, which
    /// has deletions, records Lucene 10.5 counting them here while the field
    /// form skips them), read through `ctx`, a context over `reader`.
    ///
    /// # Errors
    /// Whatever reading the values reports.
    pub fn count_all_with_values_source(
        field: &str,
        value_source: &dyn LongValuesSource,
        ctx: &ValuesContext<'_>,
        reader: &dyn IndexReader,
    ) -> Result<Self> {
        let mut c = Self::empty(field);
        c.initialize_counters();
        c.count_all_source(value_source, ctx, reader)?;
        Ok(c)
    }

    /// `new LongValueFacetCounts(field, valuesSource, reader)` with a
    /// [`MultiLongValuesSource`].
    ///
    /// # Errors
    /// Whatever reading the values reports.
    pub fn count_all_with_multi_values_source(
        field: &str,
        values_source: &dyn MultiLongValuesSource,
        ctx: &ValuesContext<'_>,
        reader: &dyn IndexReader,
    ) -> Result<Self> {
        let mut c = Self::empty(field);
        c.initialize_counters();
        if let Some(single) = values_source.single_valued() {
            c.count_all_source(single.as_ref(), ctx, reader)?;
            return Ok(c);
        }
        for leaf in reader.leaves() {
            let mut values = values_source.get_values(ctx, leaf.ord)?;
            for doc in 0..leaf.reader.max_doc() {
                if values.advance_exact(doc)? {
                    c.add_multi(values.as_mut())?;
                }
            }
        }
        Ok(c)
    }

    fn count_all_source(
        &mut self,
        value_source: &dyn LongValuesSource,
        ctx: &ValuesContext<'_>,
        reader: &dyn IndexReader,
    ) -> Result<()> {
        for leaf in reader.leaves() {
            let mut fv = value_source.get_values(ctx, leaf.ord, None)?;
            for doc in 0..leaf.reader.max_doc() {
                if fv.advance_exact(doc)? {
                    self.increment(fv.long_value()?);
                    self.tot_count = self.tot_count.wrapping_add(1);
                }
            }
        }
        Ok(())
    }

    /// `initializeCounters()`.
    fn initialize_counters(&mut self) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        self.counts = vec![0; COUNTS_LEN];
    }

    /// `count(valueSource, matchingDocs)`.
    fn count_source(
        &mut self,
        value_source: &dyn LongValuesSource,
        ctx: &ValuesContext<'_>,
        fc: &FacetsCollector<'_>,
    ) -> Result<()> {
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            self.initialize_counters();
            let mut fv = value_source.get_values(ctx, hits.context.ord, None)?;
            for_each_doc(&hits.bits, |doc| {
                if fv.advance_exact(doc)? {
                    self.increment(fv.long_value()?);
                    self.tot_count = self.tot_count.wrapping_add(1);
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    /// `count(MultiLongValuesSource, matchingDocs)`.
    fn count_multi_source(
        &mut self,
        values_source: &dyn MultiLongValuesSource,
        ctx: &ValuesContext<'_>,
        fc: &FacetsCollector<'_>,
    ) -> Result<()> {
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            self.initialize_counters();
            let mut values = values_source.get_values(ctx, hits.context.ord)?;
            for_each_doc(&hits.bits, |doc| {
                if values.advance_exact(doc)? {
                    self.add_multi(values.as_mut())?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    /// `count(field, matchingDocs)`.
    fn count_field(&mut self, fc: &FacetsCollector<'_>) -> Result<()> {
        for hits in fc.matching_docs() {
            if hits.total_hits == 0 {
                continue;
            }
            self.initialize_counters();
            let mut values = get_sorted_numeric(hits.context.reader, &self.field)?;
            for_each_hit(&hits.bits, values.as_mut(), |dv| self.add_doc(dv))?;
        }
        Ok(())
    }

    /// One document's sorted numeric values, positioned on it.
    fn add_doc(&mut self, values: &mut (dyn SortedNumericDocValues + '_)) -> Result<()> {
        let limit = values.doc_value_count();
        if limit > 0 {
            self.tot_count = self.tot_count.wrapping_add(1);
        }
        // In bulk when the doc values hold the document's values decoded.
        if let Some(all) = values.take_values() {
            let mut previous = 0i64;
            for (i, &value) in all.iter().enumerate() {
                if i == 0 || value != previous {
                    self.increment(value);
                    previous = value;
                }
            }
            return Ok(());
        }
        let mut previous = 0i64;
        for i in 0..limit {
            let value = values.next_value()?;
            // Do not count a value repeated within the document.
            if i == 0 || value != previous {
                self.increment(value);
                previous = value;
            }
        }
        Ok(())
    }

    /// One document's multi-valued source values, positioned on it.
    fn add_multi(&mut self, values: &mut (dyn MultiLongValues + '_)) -> Result<()> {
        let limit = values.get_value_count();
        if limit > 0 {
            self.tot_count = self.tot_count.wrapping_add(1);
        }
        let mut previous = 0i64;
        let mut i = 0i64;
        while i < limit {
            let value = values.next_value()?;
            if i == 0 || value != previous {
                self.increment(value);
                previous = value;
            }
            i = i.saturating_add(1);
        }
        Ok(())
    }

    /// `increment(value)`.
    #[inline]
    fn increment(&mut self, value: i64) {
        match usize::try_from(value)
            .ok()
            .and_then(|v| self.counts.get_mut(v))
        {
            Some(c) => *c = c.wrapping_add(1),
            None => {
                let c = self.hash_counts.entry(value).or_insert(0);
                *c = c.wrapping_add(1);
            }
        }
    }

    /// The array's values with a count, ascending.
    fn dense(&self) -> impl Iterator<Item = (i64, i32)> + '_ {
        (0i64..)
            .zip(self.counts.iter().copied())
            .filter(|&(_, c)| c != 0)
    }

    /// The hash table's values with a count, ascending.
    fn hashed(&self) -> Vec<(i64, i32)> {
        let mut v: Vec<(i64, i32)> = self
            .hash_counts
            .iter()
            .filter(|(_, &c)| c != 0)
            .map(|(&k, &c)| (k, c))
            .collect();
        v.sort_unstable_by_key(|&(k, _)| k);
        v
    }

    fn not_counted(&self) -> FacetResult {
        FacetResult::new(
            self.field.clone(),
            Vec::new(),
            FacetValue::Int(self.tot_count),
            Vec::new(),
            0,
        )
    }

    fn result(&self, label_values: Vec<LabelAndValue>) -> FacetResult {
        let n = i32::try_from(label_values.len()).unwrap_or(i32::MAX);
        FacetResult::new(
            self.field.clone(),
            Vec::new(),
            FacetValue::Int(self.tot_count),
            label_values,
            n,
        )
    }

    /// `getAllChildrenSortByValue()`: every value with a count, ascending.
    pub fn get_all_children_sort_by_value(&self) -> FacetResult {
        if !self.initialized {
            return self.not_counted();
        }
        let hashed = self.hashed();
        let mut label_values = Vec::with_capacity(hashed.len());
        let mut counts_added = false;
        for &(v, c) in &hashed {
            if !counts_added && v >= COUNTS_LEN as i64 {
                counts_added = true;
                label_values.extend(self.dense_labels());
            }
            label_values.push(LabelAndValue::new(v.to_string(), FacetValue::Int(c)));
        }
        if !counts_added {
            label_values.extend(self.dense_labels());
        }
        self.result(label_values)
    }

    fn dense_labels(&self) -> impl Iterator<Item = LabelAndValue> + '_ {
        self.dense()
            .map(|(v, c)| LabelAndValue::new(v.to_string(), FacetValue::Int(c)))
    }
}

/// `LongValueFacetCounts.Entry`.
struct Entry {
    count: i32,
    value: i64,
}

impl Facets for LongValueFacetCounts {
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
        validate_dim_and_path(&self.field, dim, path)?;
        if !self.initialized {
            return Ok(Some(self.not_counted()));
        }
        let mut label_values: Vec<LabelAndValue> = self.dense_labels().collect();
        for (v, c) in self.hashed() {
            label_values.push(LabelAndValue::new(v.to_string(), FacetValue::Int(c)));
        }
        Ok(Some(self.result(label_values)))
    }

    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        validate_top_n(top_n)?;
        validate_dim_and_path(&self.field, dim, path)?;
        if !self.initialized {
            return Ok(Some(self.not_counted()));
        }
        let max = usize::try_from(top_n)
            .unwrap_or(0)
            .min(COUNTS_LEN.saturating_add(self.hash_counts.len()));
        // Count descending, ties by value ascending.
        let mut pq = PriorityQueue::new(max, |a: &Entry, b: &Entry| {
            a.count < b.count || (a.count == b.count && a.value > b.value)
        });
        let mut child_count = 0i32;
        for (value, count) in self.dense() {
            child_count = child_count.wrapping_add(1);
            pq.insert_with_overflow(Entry { count, value });
        }
        if !self.hash_counts.is_empty() {
            let n = i32::try_from(self.hash_counts.len()).unwrap_or(i32::MAX);
            child_count = child_count.wrapping_add(n);
            for (&value, &count) in &self.hash_counts {
                if count != 0 {
                    pq.insert_with_overflow(Entry { count, value });
                }
            }
        }
        let mut results = Vec::with_capacity(pq.size());
        while let Some(e) = pq.pop() {
            results.push(LabelAndValue::new(
                e.value.to_string(),
                FacetValue::Int(e.count),
            ));
        }
        results.reverse();
        Ok(Some(FacetResult::new(
            self.field.clone(),
            Vec::new(),
            FacetValue::Int(self.tot_count),
            results,
            child_count,
        )))
    }

    fn get_specific_value(&self, _dim: &str, _path: &[&str]) -> Result<FacetValue> {
        Err(Error::Unsupported(
            "LongValueFacetCounts.getSpecificValue".into(),
        ))
    }

    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        validate_top_n(top_n)?;
        let field = self.field.clone();
        Ok(self
            .get_top_children(top_n, &field, &[])?
            .into_iter()
            .collect())
    }
}

impl fmt::Display for LongValueFacetCounts {
    /// `toString()`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LongValueFacetCounts totCount={}", self.tot_count)?;
        if self.initialized {
            f.write_str(":\n")?;
            for (v, c) in self.dense() {
                writeln!(f, "  {v} -> count={c}")?;
            }
            for (v, c) in self.hashed() {
                writeln!(f, "  {v} -> count={c}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for LongValueFacetCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests;
