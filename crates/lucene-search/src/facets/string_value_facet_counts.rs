//! `StringValueFacetCounts` and `StringDocValuesReaderState`: a count per
//! distinct term of a `SORTED` or `SORTED_SET` field, without a facets
//! config or a dim layer.
//!
//! The state holds what Java's does: the reader and an [`OrdinalMap`] over
//! its leaves (none for a single leaf). The counter reads the terms back
//! through each leaf's own doc values, as Java's `MultiSortedSetDocValues`
//! does (`lookupOrd` from the first leaf holding the ordinal).
//!
//! Differs: Java's sparse counts are a hash table iterated (by
//! `getAllChildren`) in an order seeded per process; here they come in
//! ordinal order.

use std::cell::RefCell;

use super::facet_result::{FacetResult, FacetValue, LabelAndValue};
use super::facet_utils::{for_each_hit, validate_dim_and_path};
use super::facets_collector::{FacetsCollector, MatchingDocs};
use super::facets_trait::{validate_top_n, Facets};
use super::top_ord_and_number_queue::{OrdAndValue, TopOrdAndNumberQueue};
use crate::grouping::FxHashMap;
use crate::ordinal_map::OrdinalMap;
use crate::reader::doc_values::get_sorted_set;
use crate::reader::{
    DocValuesIterator, IndexReader, LeafReaderContext, SortedDocValues, SortedSetDocValues,
    NO_MORE_DOCS,
};
use crate::{Error, Result};

/// `StringDocValuesReaderState`: a reader and the ordinal map of one of its
/// fields, built once per reader.
pub struct StringDocValuesReaderState<'r> {
    reader: &'r dyn IndexReader,
    field: String,
    ordinal_map: Option<OrdinalMap>,
}

impl<'r> StringDocValuesReaderState<'r> {
    /// `new StringDocValuesReaderState(reader, field)`.
    ///
    /// # Errors
    /// [`Error::IllegalState`] when a leaf has the field with doc values of
    /// a kind other than `SORTED`/`SORTED_SET` (and the reader has more
    /// than one leaf); whatever reading the terms reports.
    pub fn new(reader: &'r dyn IndexReader, field: &str) -> Result<Self> {
        let leaves = reader.leaves();
        let ordinal_map = if leaves.len() <= 1 {
            None
        } else {
            let mut terms = Vec::with_capacity(leaves.len());
            for leaf in &leaves {
                let mut dv = get_sorted_set(leaf.reader, field)?;
                let n = dv.value_count();
                let mut t = Vec::new();
                let mut ord = 0i64;
                while ord < n {
                    t.push(dv.lookup_ord(ord)?);
                    ord = ord.saturating_add(1);
                }
                terms.push(t);
            }
            Some(OrdinalMap::build(&terms))
        };
        Ok(StringDocValuesReaderState {
            reader,
            field: field.to_string(),
            ordinal_map,
        })
    }

    /// The field.
    pub fn field(&self) -> &str {
        &self.field
    }
}

impl std::fmt::Debug for StringDocValuesReaderState<'_> {
    /// `toString()` (the reader as its leaf count).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StringDocValuesReaderState(field={} reader={} leaves)",
            self.field,
            self.reader.leaves().len()
        )
    }
}

/// The message of `validateState`'s `IllegalStateException`.
const STATE_MISMATCH: &str = "the SortedSetDocValuesReaderState provided to this class does not \
                              match the reader being searched; you must create a new \
                              SortedSetDocValuesReaderState every time you open a new IndexReader";

/// `StringValueFacetCounts`.
pub struct StringValueFacetCounts<'s> {
    state: &'s StringDocValuesReaderState<'s>,
    /// Each leaf's values, for `lookupOrd`/`lookupTerm`.
    lookups: RefCell<Vec<Box<dyn SortedSetDocValues + 's>>>,
    dense_counts: Option<Vec<i32>>,
    sparse_counts: Option<FxHashMap<i32, i32>>,
    initialized: bool,
    cardinality: i32,
    total_doc_count: i32,
}

impl<'s> StringValueFacetCounts<'s> {
    /// `new StringValueFacetCounts(state)`: every live document of the
    /// state's reader.
    ///
    /// # Errors
    /// As [`Self::with_hits`].
    pub fn new(state: &'s StringDocValuesReaderState<'s>) -> Result<Self> {
        let mut c = Self::open(state)?;
        c.dense_counts = Some(vec![0; usize::try_from(c.cardinality).unwrap_or(0)]);
        c.initialized = true;
        c.count_all()?;
        Ok(c)
    }

    /// `new StringValueFacetCounts(state, facetsCollector)`: the matching
    /// documents.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for more than `Integer.MAX_VALUE` terms;
    /// [`Error::IllegalState`] when `hits` was not collected from the
    /// state's reader, or for doc values of the wrong kind; whatever
    /// reading them reports.
    pub fn with_hits(
        state: &'s StringDocValuesReaderState<'s>,
        hits: &FacetsCollector<'_>,
    ) -> Result<Self> {
        let mut c = Self::open(state)?;
        if c.cardinality < 1024 {
            c.count(hits)?;
            return Ok(c);
        }
        let mut total_hits = 0i32;
        let mut total_docs = 0i32;
        for m in hits.matching_docs() {
            total_hits = total_hits.wrapping_add(m.total_hits);
            total_docs = total_docs.wrapping_add(m.context.reader.max_doc());
        }
        c.initialized = true;
        if total_hits == 0 {
            return Ok(c);
        }
        if total_hits < total_docs / 10 {
            c.sparse_counts = Some(FxHashMap::default());
        } else {
            c.dense_counts = Some(vec![0; usize::try_from(c.cardinality).unwrap_or(0)]);
        }
        c.count(hits)?;
        Ok(c)
    }

    /// The state's leaves' values and the cardinality (`getDocValues()`).
    fn open(state: &'s StringDocValuesReaderState<'s>) -> Result<Self> {
        let leaves = state.reader.leaves();
        let mut lookups = Vec::with_capacity(leaves.len());
        for leaf in &leaves {
            lookups.push(get_sorted_set(leaf.reader, &state.field)?);
        }
        let value_count = match (&state.ordinal_map, lookups.first()) {
            (Some(map), _) => map.value_count(),
            (None, Some(dv)) => dv.value_count(),
            (None, None) => 0,
        };
        let Ok(cardinality) = i32::try_from(value_count) else {
            return Err(Error::IllegalArgument(format!(
                "can only handle valueCount < Integer.MAX_VALUE; got {value_count}"
            )));
        };
        Ok(StringValueFacetCounts {
            state,
            lookups: RefCell::new(lookups),
            dense_counts: None,
            sparse_counts: None,
            initialized: false,
            cardinality,
            total_doc_count: 0,
        })
    }

    /// `validateState(context)`: the leaf is the state's reader's.
    fn validate_state(&self, context: &LeafReaderContext<'_>) -> Result<()> {
        let leaves = self.state.reader.leaves();
        let same = leaves
            .get(context.ord)
            .is_some_and(|l| std::ptr::addr_eq(l.reader, context.reader));
        if same {
            Ok(())
        } else {
            Err(Error::IllegalState(STATE_MISMATCH.into()))
        }
    }

    /// `count(facetsCollector)`.
    fn count(&mut self, fc: &FacetsCollector<'_>) -> Result<()> {
        let matching = fc.matching_docs();
        match matching {
            [] => Ok(()),
            [hits] => {
                if hits.total_hits == 0 {
                    return Ok(());
                }
                self.validate_state(&hits.context)?;
                self.count_one_segment(Some(hits), &hits.context)
            }
            [first, ..] => {
                self.validate_state(&first.context)?;
                for hits in matching {
                    if hits.total_hits == 0 {
                        continue;
                    }
                    self.count_one_segment(Some(hits), &hits.context)?;
                }
                Ok(())
            }
        }
    }

    /// `countAll()`.
    fn count_all(&mut self) -> Result<()> {
        for leaf in self.state.reader.leaves() {
            self.count_one_segment(None, &leaf)?;
        }
        Ok(())
    }

    /// `countOneSegment` (with `hits`) and `countOneSegmentNHLD` /
    /// `countOneSegment` over the live documents (without).
    fn count_one_segment(
        &mut self,
        hits: Option<&MatchingDocs<'_>>,
        context: &LeafReaderContext<'_>,
    ) -> Result<()> {
        if !self.initialized {
            self.dense_counts = Some(vec![0; usize::try_from(self.cardinality).unwrap_or(0)]);
            self.initialized = true;
        }
        let state = self.state;
        // `DocValues.unwrapSingleton`: a `SORTED` field is read as one.
        if let Some(mut single) = context.reader.sorted_doc_values(&state.field)? {
            let cardinality = single.value_count();
            return self.count_values(hits, context, single.as_mut(), cardinality);
        }
        let mut values = get_sorted_set(context.reader, &state.field)?;
        let cardinality = i32::try_from(values.value_count()).unwrap_or(i32::MAX);
        self.count_values(hits, context, values.as_mut(), cardinality)
    }

    /// `countOneSegment`'s three ways to count, over either doc-values form.
    fn count_values<V: Ords + ?Sized>(
        &mut self,
        hits: Option<&MatchingDocs<'_>>,
        context: &LeafReaderContext<'_>,
        values: &mut V,
        segment_cardinality: i32,
    ) -> Result<()> {
        let state = self.state;
        let segment_ords = match &state.ordinal_map {
            None => None,
            Some(map) => Some(
                map.segment_ords(context.ord)
                    .ok_or_else(|| bad_leaf(context.ord))?,
            ),
        };
        let mut total = self.total_doc_count;
        let Some(segment_ords) = segment_ords else {
            // No ordinal map: segment ordinals are global.
            let (dense, sparse) = (&mut self.dense_counts, &mut self.sparse_counts);
            for_each_doc_of(hits, context, values, |dv| {
                if dv.each_ord(|ord| increment(dense, sparse, ord, 1))? > 0 {
                    total = total.wrapping_add(1);
                }
                Ok(())
            })?;
            self.total_doc_count = total;
            return Ok(());
        };
        if hits.is_some_and(|h| h.total_hits < segment_cardinality / 10) {
            // Remap every ordinal as it comes.
            let (dense, sparse) = (&mut self.dense_counts, &mut self.sparse_counts);
            for_each_doc_of(hits, context, values, |dv| {
                let n =
                    dv.each_ord(|ord| increment(dense, sparse, global(segment_ords, ord)?, 1))?;
                if n > 0 {
                    total = total.wrapping_add(1);
                }
                Ok(())
            })?;
        } else {
            // Count in segment ordinals, then migrate to global ones.
            let mut seg_counts = vec![0i32; usize::try_from(segment_cardinality).unwrap_or(0)];
            for_each_doc_of(hits, context, values, |dv| {
                let n = dv.each_ord(|ord| {
                    let slot = usize::try_from(ord)
                        .ok()
                        .and_then(|o| seg_counts.get_mut(o))
                        .ok_or_else(|| bad_ord(ord))?;
                    *slot = slot.wrapping_add(1);
                    Ok(())
                })?;
                if n > 0 {
                    total = total.wrapping_add(1);
                }
                Ok(())
            })?;
            for (ord, &count) in (0i64..).zip(&seg_counts) {
                if count != 0 {
                    increment(
                        &mut self.dense_counts,
                        &mut self.sparse_counts,
                        global(segment_ords, ord)?,
                        count,
                    )?;
                }
            }
        }
        self.total_doc_count = total;
        Ok(())
    }

    /// `docValues.lookupOrd(ord).utf8ToString()`.
    fn label(&self, ord: i64) -> Result<String> {
        let term = self.lookup_ord(ord)?;
        Ok(String::from_utf8_lossy(&term).into_owned())
    }

    /// `docValues.lookupOrd(ord)`: from the first leaf holding it.
    fn lookup_ord(&self, ord: i64) -> Result<Vec<u8>> {
        let mut lookups = self.lookups.borrow_mut();
        let (seg, seg_ord) = match &self.state.ordinal_map {
            None => (Some(0), Some(ord)),
            Some(map) => (map.first_segment(ord), map.first_segment_ord(ord)),
        };
        let dv = seg.and_then(|s| lookups.get_mut(s));
        match (dv, seg_ord) {
            (Some(dv), Some(seg_ord)) => dv.lookup_ord(seg_ord),
            _ => Err(bad_ord(ord)),
        }
    }

    /// `docValues.lookupTerm(key)`: its ordinal, or a negative number.
    // SENTINEL: negative = absent, `-insertionPoint-1` (`SortedSetDocValues.lookupTerm`).
    fn lookup_term(&self, key: &[u8]) -> Result<i64> {
        if self.state.ordinal_map.is_none() {
            let mut lookups = self.lookups.borrow_mut();
            return match lookups.first_mut() {
                Some(dv) => dv.lookup_term(key),
                None => Ok(-1),
            };
        }
        // `SortedSetDocValues.lookupTerm`'s binary search over the global
        // ordinals.
        let (mut low, mut high) = (0i64, i64::from(self.cardinality).saturating_sub(1));
        while low <= high {
            let mid = low.saturating_add(high.saturating_sub(low) / 2);
            match self.lookup_ord(mid)?.as_slice().cmp(key) {
                std::cmp::Ordering::Less => low = mid.saturating_add(1),
                std::cmp::Ordering::Greater => high = mid.saturating_sub(1),
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Ok(low.saturating_add(1).saturating_neg())
    }

    fn result(&self, label_values: Vec<LabelAndValue>, child_count: i32) -> FacetResult {
        FacetResult::new(
            self.state.field.clone(),
            Vec::new(),
            FacetValue::Int(self.total_doc_count),
            label_values,
            child_count,
        )
    }

    /// The ordinals with a count, ascending.
    fn counted(&self) -> Vec<(i32, i32)> {
        if let Some(sparse) = &self.sparse_counts {
            let mut v: Vec<(i32, i32)> = sparse.iter().map(|(&o, &c)| (o, c)).collect();
            v.sort_unstable_by_key(|&(o, _)| o);
            v
        } else if let Some(dense) = &self.dense_counts {
            (0i32..)
                .zip(dense.iter().copied())
                .filter(|&(_, c)| c != 0)
                .collect()
        } else {
            Vec::new()
        }
    }
}

/// A segment's ordinals per document: `SORTED_SET` values, or `SORTED`
/// ones read as a singleton (`DocValues.unwrapSingleton`).
trait Ords: DocValuesIterator {
    /// Hands `f` each ordinal of the current document; returns how many.
    fn each_ord(&mut self, f: impl FnMut(i64) -> Result<()>) -> Result<i32>;
}

impl Ords for dyn SortedSetDocValues + '_ {
    #[inline]
    fn each_ord(&mut self, mut f: impl FnMut(i64) -> Result<()>) -> Result<i32> {
        let n = self.doc_value_count();
        for _ in 0..n {
            f(self.next_ord()?)?;
        }
        Ok(n)
    }
}

impl Ords for dyn SortedDocValues + '_ {
    #[inline]
    fn each_ord(&mut self, mut f: impl FnMut(i64) -> Result<()>) -> Result<i32> {
        f(i64::from(self.ord_value()))?;
        Ok(1)
    }
}

/// The documents of a segment to count, each handed to `f` with `values`
/// positioned on it: the hits with values, or (`hits` `None`) every live
/// document with values.
fn for_each_doc_of<V: Ords + ?Sized>(
    hits: Option<&MatchingDocs<'_>>,
    context: &LeafReaderContext<'_>,
    values: &mut V,
    mut f: impl FnMut(&mut V) -> Result<()>,
) -> Result<()> {
    if let Some(hits) = hits {
        return for_each_hit(&hits.bits, values, f);
    }
    let live = context.reader.live_docs();
    let mut doc = values.next_doc()?;
    while doc != NO_MORE_DOCS {
        if live.is_none_or(|l| l.get_doc(doc)) {
            f(values)?;
        }
        doc = values.next_doc()?;
    }
    Ok(())
}

/// `ordMap.get(ord)`.
fn global(segment_ords: &[i64], ord: i64) -> Result<i64> {
    usize::try_from(ord)
        .ok()
        .and_then(|o| segment_ords.get(o))
        .copied()
        .ok_or_else(|| bad_ord(ord))
}

fn bad_ord(ord: i64) -> Error {
    Error::IllegalState(format!("ordinal {ord} is out of bounds"))
}

fn bad_leaf(leaf: usize) -> Error {
    Error::IllegalState(format!("leaf {leaf} is not in the ordinal map"))
}

/// `increment(ordinal, amount)`.
fn increment(
    dense: &mut Option<Vec<i32>>,
    sparse: &mut Option<FxHashMap<i32, i32>>,
    ord: i64,
    amount: i32,
) -> Result<()> {
    let ordinal = i32::try_from(ord).map_err(|_| bad_ord(ord))?;
    if let Some(sparse) = sparse {
        let c = sparse.entry(ordinal).or_insert(0);
        *c = c.wrapping_add(amount);
    } else if let Some(dense) = dense {
        let c = usize::try_from(ordinal)
            .ok()
            .and_then(|o| dense.get_mut(o))
            .ok_or_else(|| bad_ord(ord))?;
        *c = c.wrapping_add(amount);
    }
    Ok(())
}

impl Facets for StringValueFacetCounts<'_> {
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
        validate_dim_and_path(&self.state.field, dim, path)?;
        let mut label_values = Vec::new();
        for (ord, count) in self.counted() {
            label_values.push(LabelAndValue::new(
                self.label(i64::from(ord))?,
                FacetValue::Int(count),
            ));
        }
        let n = i32::try_from(label_values.len()).unwrap_or(i32::MAX);
        Ok(Some(self.result(label_values, n)))
    }

    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        validate_top_n(top_n)?;
        validate_dim_and_path(&self.state.field, dim, path)?;
        let top_n = usize::try_from(top_n.min(self.cardinality)).unwrap_or(0);
        let mut q: Option<TopOrdAndNumberQueue> = None;
        let mut child_count = 0i32;
        let mut add = |ord: i32, count: i32| {
            child_count = child_count.wrapping_add(1);
            q.get_or_insert_with(|| TopOrdAndNumberQueue::int(top_n))
                .insert_with_overflow(OrdAndValue {
                    ord,
                    value: FacetValue::Int(count),
                });
        };
        if let Some(sparse) = &self.sparse_counts {
            for (&ord, &count) in sparse {
                add(ord, count);
            }
        } else if let Some(dense) = &self.dense_counts {
            for (ord, &count) in (0i32..).zip(dense) {
                if count != 0 {
                    add(ord, count);
                }
            }
        }
        let mut label_values = Vec::new();
        if let Some(mut q) = q {
            while let Some(e) = q.pop() {
                label_values.push(LabelAndValue::new(self.label(i64::from(e.ord))?, e.value));
            }
        }
        label_values.reverse();
        Ok(Some(self.result(label_values, child_count)))
    }

    fn get_specific_value(&self, dim: &str, path: &[&str]) -> Result<FacetValue> {
        if dim != self.state.field {
            return Err(Error::IllegalArgument(format!(
                "invalid dim \"{dim}\"; should be \"{}\"",
                self.state.field
            )));
        }
        let [label] = path else {
            return Err(Error::IllegalArgument("path must be length=1".into()));
        };
        // `(int) lookupTerm(...)`.
        let ord = self.lookup_term(label.as_bytes())? as i32;
        if ord < 0 {
            return Ok(FacetValue::Int(-1));
        }
        if let Some(sparse) = &self.sparse_counts {
            return Ok(FacetValue::Int(sparse.get(&ord).copied().unwrap_or(0)));
        }
        if let Some(dense) = &self.dense_counts {
            let c = usize::try_from(ord)
                .ok()
                .and_then(|o| dense.get(o))
                .copied()
                .ok_or_else(|| bad_ord(i64::from(ord)))?;
            return Ok(FacetValue::Int(c));
        }
        Ok(FacetValue::Int(0))
    }

    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        validate_top_n(top_n)?;
        let field = self.state.field.clone();
        Ok(self
            .get_top_children(top_n, &field, &[])?
            .into_iter()
            .collect())
    }
}

impl std::fmt::Debug for StringValueFacetCounts<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringValueFacetCounts")
            .field("field", &self.state.field)
            .field("cardinality", &self.cardinality)
            .field("total_doc_count", &self.total_doc_count)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
