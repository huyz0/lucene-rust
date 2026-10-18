//! `TaxonomyFacets`, `IntTaxonomyFacets` and `FloatTaxonomyFacets`: the
//! taxonomy facets' shared base -- per-ordinal counts (and int or float
//! values), the roll-up of hierarchical dims, and every `Facets` method.
//!
//! Java's three abstract classes are one struct here, told apart by its
//! [`Aggregation`]: counts only (`TaxonomyFacets` itself, which
//! `FastTaxonomyFacetCounts` extends), int values
//! (`IntTaxonomyFacets`) or float values (`FloatTaxonomyFacets`). What Java
//! dispatches through overridden methods -- the value of an ordinal, how two
//! values combine, the queue type, the missing value, the roll-up -- is a
//! `match` on it.
//!
//! Counting is dense (an array over the taxonomy) unless the taxonomy has at
//! least 1024 ordinals and the hits are under a tenth of the index, as in
//! Java. Java then keeps a hash table and walks it in its own (hash) order;
//! this port walks the sparse ordinals in ascending order. Only
//! `getAllChildren` lets that order show (its children come out in it, and a
//! float sum over them rounds in it); `getTopChildren`'s queue orders by
//! value then ordinal either way.

use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use crate::facets::facet_result::{float_to_int_bits, FacetResult, FacetValue, LabelAndValue};
use crate::facets::facets_collector::FacetsCollector;
use crate::facets::facets_trait::{validate_top_n, Facets};
use crate::facets::taxonomy::facet_label::java_compare;
use crate::facets::taxonomy::parallel_taxonomy_arrays::IntArrayReader;
use crate::facets::taxonomy::{
    AssociationAggregationFunction, FacetLabel, IntArray, ParallelTaxonomyArrays, TaxonomyReader,
    INVALID_ORDINAL, ROOT_ORDINAL,
};
use crate::facets::top_ord_and_number_queue::{OrdAndValue, PriorityQueue, TopOrdAndNumberQueue};
use crate::facets::{DimConfig, FacetsConfig};
use crate::grouping::FxHashMap;
use crate::{Error, Result};

/// What a [`TaxonomyFacets`] aggregates beside the counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    /// Counts only: `TaxonomyFacets` (`FastTaxonomyFacetCounts`).
    Count,
    /// An int per ordinal: `IntTaxonomyFacets`.
    Int(AssociationAggregationFunction),
    /// A float per ordinal: `FloatTaxonomyFacets`.
    Float(AssociationAggregationFunction),
}

/// A value per ordinal: an array over the taxonomy, or a hash table of the
/// ordinals set (a missing one is `T::default()`, hppc's `get`).
#[derive(Debug, Clone)]
pub(crate) enum OrdValues<T> {
    /// Not allocated (Java's `null` array).
    Unset,
    Dense(Vec<T>),
    Sparse(FxHashMap<i32, T>),
}

fn out_of_bounds(ord: i32, len: usize) -> Error {
    Error::IllegalArgument(format!("Index {ord} out of bounds for length {len}"))
}

/// `values[ord]` and `counts[ord]` of two dense arrays, an ordinal outside
/// either Java's `ArrayIndexOutOfBoundsException`.
#[inline]
pub(crate) fn dense_slots<'v, T>(
    values: &'v mut [T],
    counts: &'v mut [i32],
    ord: i32,
) -> Result<(&'v mut T, &'v mut i32)> {
    let len = values.len().min(counts.len());
    usize::try_from(ord)
        .ok()
        .and_then(|i| values.get_mut(i).zip(counts.get_mut(i)))
        .ok_or_else(|| out_of_bounds(ord, len))
}

/// `parents.get(ord)`, refusing an ordinal outside the array as Java's
/// array access does.
pub(crate) fn parent_of(parents: &dyn IntArray, ord: i32) -> Result<i32> {
    let len = parents.length();
    if (0..len).contains(&ord) {
        Ok(parents.get(ord))
    } else {
        Err(out_of_bounds(ord, usize::try_from(len).unwrap_or(0)))
    }
}

impl<T: Copy + Default> OrdValues<T> {
    fn new(sparse: bool, size: i32) -> Self {
        if sparse {
            OrdValues::Sparse(FxHashMap::default())
        } else {
            OrdValues::Dense(vec![T::default(); usize::try_from(size).unwrap_or(0)])
        }
    }

    /// `get(ordinal)`: a dense array's `ArrayIndexOutOfBoundsException` is
    /// an error.
    #[inline]
    pub(crate) fn get(&self, ord: i32) -> Result<T> {
        match self {
            OrdValues::Unset => Ok(T::default()),
            OrdValues::Dense(v) => usize::try_from(ord)
                .ok()
                .and_then(|i| v.get(i))
                .copied()
                .ok_or_else(|| out_of_bounds(ord, v.len())),
            OrdValues::Sparse(m) => Ok(m.get(&ord).copied().unwrap_or_default()),
        }
    }

    /// `set(ordinal, value)`.
    #[inline]
    pub(crate) fn set(&mut self, ord: i32, value: T) -> Result<()> {
        match self {
            OrdValues::Unset => Ok(()),
            OrdValues::Dense(v) => {
                let len = v.len();
                let slot = usize::try_from(ord)
                    .ok()
                    .and_then(|i| v.get_mut(i))
                    .ok_or_else(|| out_of_bounds(ord, len))?;
                *slot = value;
                Ok(())
            }
            OrdValues::Sparse(m) => {
                m.insert(ord, value);
                Ok(())
            }
        }
    }

    /// `set(ordinal, f(get(ordinal)))` with one lookup.
    #[inline]
    pub(crate) fn update(&mut self, ord: i32, f: impl FnOnce(T) -> T) -> Result<()> {
        match self {
            OrdValues::Unset => Ok(()),
            OrdValues::Dense(v) => {
                let len = v.len();
                let slot = usize::try_from(ord)
                    .ok()
                    .and_then(|i| v.get_mut(i))
                    .ok_or_else(|| out_of_bounds(ord, len))?;
                *slot = f(*slot);
                Ok(())
            }
            OrdValues::Sparse(m) => {
                let slot = m.entry(ord).or_default();
                *slot = f(*slot);
                Ok(())
            }
        }
    }
}

/// `TaxonomyFacets.TopChildrenForPath`: a path's value, how many children
/// had one, and the queue of the best.
pub(crate) struct TopChildrenForPath {
    path_value: FacetValue,
    child_count: i32,
    queue: TopOrdAndNumberQueue,
}

/// `TaxonomyFacets.AggregatedValue`, by [`Aggregation`].
enum AggregatedValue {
    Count(i32),
    Int(AssociationAggregationFunction, i32),
    Float(AssociationAggregationFunction, f32),
}

impl AggregatedValue {
    fn get(&self) -> FacetValue {
        match *self {
            AggregatedValue::Count(c) | AggregatedValue::Int(_, c) => FacetValue::Int(c),
            AggregatedValue::Float(_, v) => FacetValue::Float(v),
        }
    }
}

/// `Float.compare(a, b)`: `-0.0` before `0.0`, `NaN` after everything.
pub(crate) fn java_float_compare(a: f32, b: f32) -> Ordering {
    if a < b {
        return Ordering::Less;
    }
    if a > b {
        return Ordering::Greater;
    }
    float_to_int_bits(a).cmp(&float_to_int_bits(b))
}

/// `TaxonomyFacets` (with `IntTaxonomyFacets` / `FloatTaxonomyFacets`; see
/// the module docs).
pub struct TaxonomyFacets<'t> {
    index_field_name: String,
    taxo_reader: &'t dyn TaxonomyReader,
    config: &'t FacetsConfig,
    arrays: Arc<dyn ParallelTaxonomyArrays>,
    taxo_size: i32,
    /// `useHashTable(fc, taxoReader)`, decided up front.
    use_hash_table: bool,
    aggregation: Aggregation,
    pub(crate) counts: OrdValues<i32>,
    pub(crate) int_values: OrdValues<i32>,
    pub(crate) float_values: OrdValues<f32>,
    /// The sparse ordinals, ascending, once counting is done.
    pub(crate) sparse_ords: Vec<i32>,
    initialized: bool,
    children_loaded: Cell<bool>,
    siblings_loaded: Cell<bool>,
}

impl std::fmt::Debug for TaxonomyFacets<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaxonomyFacets")
            .field("index_field_name", &self.index_field_name)
            .field("aggregation", &self.aggregation)
            .field("initialized", &self.initialized)
            .field("use_hash_table", &self.use_hash_table)
            .finish()
    }
}

impl<'t> TaxonomyFacets<'t> {
    /// The constructor: `fc` is `None` when counting the whole index
    /// (`FastTaxonomyFacetCounts`' `countAll`).
    pub(crate) fn new(
        index_field_name: &str,
        taxo_reader: &'t dyn TaxonomyReader,
        config: &'t FacetsConfig,
        fc: Option<&FacetsCollector<'_>>,
        aggregation: Aggregation,
    ) -> Result<Self> {
        let arrays = taxo_reader.parallel_taxonomy_arrays()?;
        let taxo_size = taxo_reader.get_size()?;
        Ok(TaxonomyFacets {
            index_field_name: index_field_name.to_string(),
            taxo_reader,
            config,
            arrays,
            taxo_size,
            use_hash_table: use_hash_table(fc, taxo_size),
            aggregation,
            counts: OrdValues::Unset,
            int_values: OrdValues::Unset,
            float_values: OrdValues::Unset,
            sparse_ords: Vec::new(),
            initialized: false,
            children_loaded: Cell::new(false),
            siblings_loaded: Cell::new(false),
        })
    }

    /// `indexFieldName`.
    pub fn index_field_name(&self) -> &str {
        &self.index_field_name
    }

    /// `parents` (the arrays' `parents()`).
    pub(crate) fn parents(&self) -> &dyn IntArray {
        self.arrays.parents()
    }

    /// `getChildren()`.
    fn children(&self) -> &dyn IntArray {
        self.children_loaded.set(true);
        self.arrays.children()
    }

    /// `getSiblings()`.
    fn siblings(&self) -> &dyn IntArray {
        self.siblings_loaded.set(true);
        self.arrays.siblings()
    }

    /// `childrenLoaded()`: whether the children array was asked for.
    pub fn children_loaded(&self) -> bool {
        self.children_loaded.get()
    }

    /// `siblingsLoaded()`: whether the siblings array was asked for.
    pub fn siblings_loaded(&self) -> bool {
        self.siblings_loaded.get()
    }

    /// `initializeValueCounters()`.
    pub(crate) fn initialize_value_counters(&mut self) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        let sparse = self.use_hash_table;
        self.counts = OrdValues::new(sparse, self.taxo_size);
        match self.aggregation {
            Aggregation::Count => {}
            Aggregation::Int(_) => self.int_values = OrdValues::new(sparse, self.taxo_size),
            Aggregation::Float(_) => self.float_values = OrdValues::new(sparse, self.taxo_size),
        }
    }

    /// Records the sparse ordinals once counting and roll-up are done.
    pub(crate) fn finish_counting(&mut self) {
        if let OrdValues::Sparse(m) = &self.counts {
            let mut ords: Vec<i32> = m.keys().copied().collect();
            ords.sort_unstable();
            self.sparse_ords = ords;
        }
    }

    /// `getCount(ordinal)`.
    #[inline]
    pub(crate) fn get_count(&self, ord: i32) -> Result<i32> {
        self.counts.get(ord)
    }

    /// `setCount(ordinal, newValue)`.
    #[inline]
    pub(crate) fn set_count(&mut self, ord: i32, value: i32) -> Result<()> {
        self.counts.set(ord, value)
    }

    /// `getAggregationValue(ordinal)`.
    fn get_aggregation_value(&self, ord: i32) -> Result<FacetValue> {
        Ok(match self.aggregation {
            Aggregation::Count => FacetValue::Int(self.get_count(ord)?),
            Aggregation::Int(_) => FacetValue::Int(self.int_values.get(ord)?),
            Aggregation::Float(_) => FacetValue::Float(self.float_values.get(ord)?),
        })
    }

    /// `aggregate(existingVal, newVal)`.
    fn aggregate(&self, existing: FacetValue, new: FacetValue) -> FacetValue {
        match self.aggregation {
            Aggregation::Count => {
                FacetValue::Int(existing.int_value().wrapping_add(new.int_value()))
            }
            Aggregation::Int(f) => {
                FacetValue::Int(f.aggregate_int(existing.int_value(), new.int_value()))
            }
            Aggregation::Float(f) => {
                FacetValue::Float(f.aggregate_float(existing.float_value(), new.float_value()))
            }
        }
    }

    /// `missingAggregationValue()`: `-1`, or `-1f` for float values.
    fn missing_aggregation_value(&self) -> FacetValue {
        match self.aggregation {
            Aggregation::Float(_) => FacetValue::Float(-1.0),
            _ => FacetValue::Int(-1),
        }
    }

    /// `valueComparator`: int order, or `Float.compare` for float values.
    fn compare_values(&self, a: FacetValue, b: FacetValue) -> Ordering {
        compare_values(self.aggregation, a, b)
    }

    /// `verifyDim(dim)`.
    fn verify_dim(&self, dim: &str) -> Result<&'t DimConfig> {
        let dim_config = self.config.dim_config(dim);
        if self.config.is_dim_configured(dim)
            && dim_config.index_field_name != self.index_field_name
        {
            return Err(Error::IllegalArgument(format!(
                "dimension \"{dim}\" cannot be found in field \"{}\", since it was configured to \
                 be indexed into field \"{}\"",
                self.index_field_name, dim_config.index_field_name
            )));
        }
        Ok(dim_config)
    }

    /// `rollup()`: rolls up the single-valued hierarchical dims.
    pub(crate) fn rollup(&mut self) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }
        let config = self.config;
        for (dim, ft) in config.dim_configs() {
            if ft.hierarchical && !ft.multi_valued {
                let dim_root_ord = self.taxo_reader.get_ordinal(&FacetLabel::new([dim])?)?;
                // -1 when the dim was configured but never indexed.
                if dim_root_ord > 0 {
                    let child = self.children().get(dim_root_ord);
                    self.update_value_from_rollup(dim_root_ord, child)?;
                }
            }
        }
        Ok(())
    }

    /// `updateValueFromRollup(ordinal, childOrdinal)`: the counts, then (for
    /// `IntTaxonomyFacets`/`FloatTaxonomyFacets`) the values.
    ///
    /// Java's values roll-up calls this method again for every descendant,
    /// so a descendant's count is rolled up once more at every level above
    /// it -- which Java's counts then show, and so do these.
    fn update_value_from_rollup(&mut self, ord: i32, child: i32) -> Result<()> {
        let rolled = self.rollup_counts(child)?;
        let count = self.get_count(ord)?.wrapping_add(rolled);
        self.set_count(ord, count)?;
        match self.aggregation {
            Aggregation::Count => {}
            Aggregation::Int(f) => {
                let current = self.int_values.get(ord)?;
                let rolled = self.rollup_int(f, child)?;
                self.int_values.set(ord, f.aggregate_int(current, rolled))?;
            }
            Aggregation::Float(f) => {
                let current = self.float_values.get(ord)?;
                let rolled = self.rollup_float(f, child)?;
                self.float_values
                    .set(ord, f.aggregate_float(current, rolled))?;
            }
        }
        Ok(())
    }

    /// `TaxonomyFacets.rollup(ord)`: the counts of `ord` and its siblings,
    /// each first rolled up from its own children.
    fn rollup_counts(&mut self, mut ord: i32) -> Result<i32> {
        let mut aggregated = 0i32;
        while ord != INVALID_ORDINAL {
            let current = self.get_count(ord)?;
            let child = self.children().get(ord);
            let new_value = current.wrapping_add(self.rollup_counts(child)?);
            self.set_count(ord, new_value)?;
            aggregated = aggregated.wrapping_add(self.get_count(ord)?);
            ord = self.siblings().get(ord);
        }
        Ok(aggregated)
    }

    /// `IntTaxonomyFacets.rollup(ord)`.
    fn rollup_int(&mut self, f: AssociationAggregationFunction, mut ord: i32) -> Result<i32> {
        let mut aggregated = 0i32;
        while ord != INVALID_ORDINAL {
            let child = self.children().get(ord);
            self.update_value_from_rollup(ord, child)?;
            aggregated = f.aggregate_int(aggregated, self.int_values.get(ord)?);
            ord = self.siblings().get(ord);
        }
        Ok(aggregated)
    }

    /// `FloatTaxonomyFacets.rollup(ord)`.
    fn rollup_float(&mut self, f: AssociationAggregationFunction, mut ord: i32) -> Result<f32> {
        let mut aggregated = 0f32;
        while ord != INVALID_ORDINAL {
            let child = self.children().get(ord);
            self.update_value_from_rollup(ord, child)?;
            aggregated = f.aggregate_float(aggregated, self.float_values.get(ord)?);
            ord = self.siblings().get(ord);
        }
        Ok(aggregated)
    }

    /// `makeTopOrdAndNumberQueue(topN)`: at most `min(taxoSize, topN)`.
    fn make_queue(&self, top_n: i32) -> TopOrdAndNumberQueue {
        let size = usize::try_from(self.taxo_size.min(top_n)).unwrap_or(0);
        match self.aggregation {
            Aggregation::Float(_) => TopOrdAndNumberQueue::float(size),
            _ => TopOrdAndNumberQueue::int(size),
        }
    }

    /// `setIncomingValue`: the count, or the ordinal's value.
    fn queue_value(&self, ord: i32) -> Result<FacetValue> {
        Ok(match self.aggregation {
            Aggregation::Count => FacetValue::Int(self.get_count(ord)?),
            Aggregation::Int(_) => FacetValue::Int(self.int_values.get(ord)?),
            Aggregation::Float(_) => FacetValue::Float(self.float_values.get(ord)?),
        })
    }

    /// `newAggregatedValue()`.
    fn new_aggregated_value(&self) -> AggregatedValue {
        match self.aggregation {
            Aggregation::Count => AggregatedValue::Count(0),
            Aggregation::Int(f) => AggregatedValue::Int(f, 0),
            Aggregation::Float(f) => AggregatedValue::Float(f, 0.0),
        }
    }

    /// `AggregatedValue.aggregate(ord)`.
    fn aggregate_into(&self, agg: &mut AggregatedValue, ord: i32) -> Result<()> {
        match agg {
            AggregatedValue::Count(c) => *c = c.wrapping_add(self.get_count(ord)?),
            AggregatedValue::Int(f, v) => *v = f.aggregate_int(*v, self.int_values.get(ord)?),
            AggregatedValue::Float(f, v) => *v = f.aggregate_float(*v, self.float_values.get(ord)?),
        }
        Ok(())
    }

    /// Calls `visit(ord, count)` for each child of `path_ord` with a count
    /// above zero: from the sparse table (ascending ordinals), or down the
    /// children/siblings arrays.
    fn for_each_counted_child(
        &self,
        path_ord: i32,
        mut visit: impl FnMut(i32, i32) -> Result<()>,
    ) -> Result<()> {
        match &self.counts {
            OrdValues::Sparse(map) => {
                let parents = self.parents();
                for &ord in &self.sparse_ords {
                    let count = map.get(&ord).copied().unwrap_or(0);
                    // An ordinal off disk past the taxonomy: Java's
                    // `parents.get(ord)` throws.
                    if parent_of(parents, ord)? == path_ord && count > 0 {
                        visit(ord, count)?;
                    }
                }
            }
            _ => {
                let siblings = IntArrayReader::new(self.siblings());
                let mut ord = self.children().get(path_ord);
                while ord != INVALID_ORDINAL {
                    let count = self.get_count(ord)?;
                    if count > 0 {
                        visit(ord, count)?;
                    }
                    ord = siblings.get(ord);
                }
            }
        }
        Ok(())
    }

    /// `getTopChildrenForPath(dimConfig, pathOrd, topN)`.
    pub(crate) fn get_top_children_for_path(
        &self,
        dim_config: &DimConfig,
        path_ord: i32,
        top_n: i32,
    ) -> Result<TopChildrenForPath> {
        let mut queue = self.make_queue(top_n);
        let mut aggregated = self.new_aggregated_value();
        let mut child_count = 0i32;
        if let (Aggregation::Count, OrdValues::Dense(counts)) = (self.aggregation, &self.counts) {
            // The plain-count walk with the counts borrowed once: the same
            // children, sum and queue as the general loop below.
            let siblings = IntArrayReader::new(self.siblings());
            let mut sum = 0i32;
            let mut ord = self.children().get(path_ord);
            while ord != INVALID_ORDINAL {
                let count = usize::try_from(ord)
                    .ok()
                    .and_then(|i| counts.get(i))
                    .copied()
                    .ok_or_else(|| out_of_bounds(ord, counts.len()))?;
                if count > 0 {
                    sum = sum.wrapping_add(count);
                    child_count = child_count.wrapping_add(1);
                    // A count under a full queue's least loses to it
                    // whatever the ordinals: `insertWithOverflow`'s answer.
                    if queue.full_int_bottom().is_none_or(|bottom| count >= bottom) {
                        queue.insert_with_overflow(OrdAndValue {
                            ord,
                            value: FacetValue::Int(count),
                        });
                    }
                }
                ord = siblings.get(ord);
            }
            aggregated = AggregatedValue::Count(sum);
        } else {
            self.for_each_counted_child(path_ord, |ord, _| {
                self.aggregate_into(&mut aggregated, ord)?;
                child_count = child_count.wrapping_add(1);
                queue.insert_with_overflow(OrdAndValue {
                    ord,
                    value: self.queue_value(ord)?,
                });
                Ok(())
            })?;
        }
        let mut path_value = aggregated.get();
        if dim_config.multi_valued {
            path_value = if dim_config.require_dim_count {
                self.get_aggregation_value(path_ord)?
            } else {
                // Not accurate, in general.
                self.missing_aggregation_value()
            };
        }
        Ok(TopChildrenForPath {
            path_value,
            child_count,
            queue,
        })
    }

    /// `createFacetResult(topChildrenForPath, dim, path)`: `None` when no
    /// child had a value.
    fn create_facet_result(
        &self,
        top: TopChildrenForPath,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        if top.child_count == 0 {
            return Ok(None);
        }
        let mut queue = top.queue;
        let mut popped: Vec<OrdAndValue> = Vec::with_capacity(queue.size());
        while let Some(e) = queue.pop() {
            popped.push(e);
        }
        popped.reverse();
        let ordinals: Vec<i32> = popped.iter().map(|e| e.ord).collect();
        let bulk = self.taxo_reader.get_bulk_path(&ordinals)?;
        // The component right after the path; `+ 1` for the dim.
        let child_component = path.len().saturating_add(1);
        let mut label_values = Vec::with_capacity(popped.len());
        for (e, label) in popped.iter().zip(&bulk) {
            label_values.push(LabelAndValue::with_count(
                component(label, child_component)?,
                e.value,
                self.get_count(e.ord)?,
            ));
        }
        Ok(Some(FacetResult::new(
            dim,
            path.iter().map(|s| s.to_string()).collect(),
            top.path_value,
            label_values,
            top.child_count,
        )))
    }

    /// The first component of ordinal `ord`'s path (its dim).
    fn dim_of(&self, ord: i32) -> Result<String> {
        component(&self.taxo_reader.get_path(ord)?, 0)
    }

    fn top_dims(&self, top_n_dims: i32, top_n_children: i32) -> Result<Vec<FacetResult>> {
        if top_n_dims <= 0 || top_n_children <= 0 {
            return Err(Error::IllegalArgument("topN must be > 0".to_string()));
        }
        if !self.initialized {
            return Ok(Vec::new());
        }
        let aggregation = self.aggregation;
        let less_than =
            move |a: &DimValue, b: &DimValue| match compare_values(aggregation, a.value, b.value) {
                Ordering::Less => true,
                Ordering::Greater => false,
                Ordering::Equal => java_compare(&a.dim, &b.dim) == Ordering::Greater,
            };
        let max_size = usize::try_from(top_n_dims).unwrap_or(0);
        let mut pq = PriorityQueue::new(max_size, less_than);
        let mut intermediate: HashMap<String, TopChildrenForPath> = HashMap::new();

        let siblings = self.siblings();
        let mut ord = self.children().get(ROOT_ORDINAL);
        while ord != INVALID_ORDINAL {
            let dim = self.dim_of(ord)?;
            let dim_config = self.config.dim_config(&dim);
            if dim_config.index_field_name == self.index_field_name {
                let dim_ord = self
                    .taxo_reader
                    .get_ordinal(&FacetLabel::new([dim.as_str()])?)?;
                if dim_ord != INVALID_ORDINAL {
                    let dim_value = if dim_config.multi_valued {
                        if dim_config.require_dim_count {
                            self.get_aggregation_value(dim_ord)?
                        } else {
                            // An `Integer` -1 whatever the values' type.
                            FacetValue::Int(-1)
                        }
                    } else {
                        let top =
                            self.get_top_children_for_path(dim_config, dim_ord, top_n_children)?;
                        let value = top.path_value;
                        intermediate.insert(dim.clone(), top);
                        value
                    };
                    if self.compare_values(dim_value, FacetValue::Int(0)) != Ordering::Equal {
                        if pq.size() < max_size {
                            pq.add(DimValue {
                                dim,
                                dim_ord,
                                value: dim_value,
                            });
                        } else if let Some(top) = pq.top() {
                            let c = self.compare_values(dim_value, top.value);
                            if c == Ordering::Greater
                                || (c == Ordering::Equal
                                    && java_compare(&dim, &top.dim) == Ordering::Less)
                            {
                                // Java updates the dim and the value but keeps
                                // the replaced dim's ordinal.
                                if let Some(bottom) = pq.top_mut() {
                                    bottom.dim = dim;
                                    bottom.value = dim_value;
                                }
                                pq.update_top();
                            }
                        }
                    }
                }
            }
            ord = siblings.get(ord);
        }

        let mut results: Vec<Option<FacetResult>> = Vec::with_capacity(pq.size());
        while let Some(dim_value) = pq.pop() {
            let top = match intermediate.remove(&dim_value.dim) {
                Some(top) => top,
                None => {
                    let dim_config = self.config.dim_config(&dim_value.dim);
                    self.get_top_children_for_path(dim_config, dim_value.dim_ord, top_n_children)?
                }
            };
            results.push(self.create_facet_result(top, &dim_value.dim, &[])?);
        }
        // Popped worst first; Java fills its array from the back. A `null`
        // there (a dim with no child counted) is left out, where Java's list
        // keeps it. Java's queue also refuses a `topNDims` at or past
        // `ArrayUtil.MAX_ARRAY_LENGTH` (a JVM-dependent bound); this one
        // grows on demand, so any `topNDims` is taken.
        Ok(results.into_iter().rev().flatten().collect())
    }
}

/// `TaxonomyFacets.DimValue`.
struct DimValue {
    dim: String,
    dim_ord: i32,
    value: FacetValue,
}

/// `valueComparator.compare(a, b)` for `aggregation`.
fn compare_values(aggregation: Aggregation, a: FacetValue, b: FacetValue) -> Ordering {
    match aggregation {
        Aggregation::Float(_) => java_float_compare(a.float_value(), b.float_value()),
        _ => a.int_value().cmp(&b.int_value()),
    }
}

/// `label.components[i]`, Java's `ArrayIndexOutOfBoundsException` an error.
fn component(label: &FacetLabel, i: usize) -> Result<String> {
    label
        .components()
        .get(i)
        .cloned()
        .ok_or_else(|| out_of_bounds(i32::try_from(i).unwrap_or(i32::MAX), label.len()))
}

/// `useHashTable(fc, taxoReader)`: a hash table for a taxonomy of at least
/// 1024 ordinals when the hits are under a tenth of the index.
fn use_hash_table(fc: Option<&FacetsCollector<'_>>, taxo_size: i32) -> bool {
    if taxo_size < 1024 {
        return false;
    }
    let Some(fc) = fc else {
        return false;
    };
    let mut max_doc = 0i32;
    let mut sum_total_hits = 0i32;
    for docs in fc.matching_docs() {
        sum_total_hits = sum_total_hits.wrapping_add(docs.total_hits);
        max_doc = max_doc.wrapping_add(docs.context.reader.max_doc());
    }
    sum_total_hits < max_doc / 10
}

impl Facets for TaxonomyFacets<'_> {
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
        let dim_config = self.verify_dim(dim)?;
        let cp = FacetLabel::from_dim_path(dim, path)?;
        let dim_ord = self.taxo_reader.get_ordinal(&cp)?;
        if dim_ord == INVALID_ORDINAL || !self.initialized {
            return Ok(None);
        }
        let mut aggregated_value = FacetValue::Int(0);
        let mut aggregated_count = 0i32;
        let mut ordinals: Vec<i32> = Vec::new();
        let mut values: Vec<FacetValue> = Vec::new();
        self.for_each_counted_child(dim_ord, |ord, count| {
            let value = self.get_aggregation_value(ord)?;
            aggregated_count = aggregated_count.wrapping_add(count);
            aggregated_value = self.aggregate(aggregated_value, value);
            ordinals.push(ord);
            values.push(value);
            Ok(())
        })?;
        if aggregated_count == 0 {
            return Ok(None);
        }
        if dim_config.multi_valued {
            aggregated_value = if dim_config.require_dim_count {
                self.get_aggregation_value(dim_ord)?
            } else {
                self.missing_aggregation_value()
            };
        }
        let bulk = self.taxo_reader.get_bulk_path(&ordinals)?;
        let mut label_values = Vec::with_capacity(ordinals.len());
        for ((&ord, &value), label) in ordinals.iter().zip(&values).zip(&bulk) {
            label_values.push(LabelAndValue::with_count(
                component(label, cp.len())?,
                value,
                self.get_count(ord)?,
            ));
        }
        let child_count = i32::try_from(ordinals.len()).unwrap_or(i32::MAX);
        Ok(Some(FacetResult::new(
            dim,
            path.iter().map(|s| s.to_string()).collect(),
            aggregated_value,
            label_values,
            child_count,
        )))
    }

    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        validate_top_n(top_n)?;
        let dim_config = self.verify_dim(dim)?;
        let cp = FacetLabel::from_dim_path(dim, path)?;
        let dim_ord = self.taxo_reader.get_ordinal(&cp)?;
        if dim_ord == INVALID_ORDINAL || !self.initialized {
            return Ok(None);
        }
        let top = self.get_top_children_for_path(dim_config, dim_ord, top_n)?;
        self.create_facet_result(top, dim, path)
    }

    // SENTINEL: `-1` = "the path is not in the taxonomy", Java's own
    // `getSpecificValue` contract.
    fn get_specific_value(&self, dim: &str, path: &[&str]) -> Result<FacetValue> {
        let dim_config = self.verify_dim(dim)?;
        if path.is_empty()
            && !(dim_config.hierarchical && !dim_config.multi_valued)
            && !(dim_config.require_dim_count && dim_config.multi_valued)
        {
            return Err(Error::IllegalArgument(
                "cannot return dimension-level value alone; use getTopChildren instead".to_string(),
            ));
        }
        let ord = self
            .taxo_reader
            .get_ordinal(&FacetLabel::from_dim_path(dim, path)?)?;
        if ord < 0 {
            return Ok(FacetValue::Int(-1));
        }
        if self.initialized {
            self.get_aggregation_value(ord)
        } else {
            Ok(FacetValue::Int(0))
        }
    }

    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        validate_top_n(top_n)?;
        if !self.initialized {
            return Ok(Vec::new());
        }
        let siblings = self.siblings();
        let mut ord = self.children().get(ROOT_ORDINAL);
        let mut results = Vec::new();
        while ord != INVALID_ORDINAL {
            let dim = self.dim_of(ord)?;
            if self.config.dim_config(&dim).index_field_name == self.index_field_name {
                if let Some(r) = self.get_top_children(top_n, &dim, &[])? {
                    results.push(r);
                }
            }
            ord = siblings.get(ord);
        }
        // The highest value first, ties by dim. A `NaN` value ties with
        // everything, so the order is not total: an insertion sort, stable
        // like Java's `Collections.sort` and the same result whenever the
        // order is total, where `sort_by` may panic on one that is not.
        let before = |a: &FacetResult, b: &FacetResult| {
            let (x, y) = (a.value.double_value(), b.value.double_value());
            x > y
                || (y.partial_cmp(&x) != Some(Ordering::Greater)
                    && java_compare(&a.dim, &b.dim) == Ordering::Less)
        };
        for i in 1..results.len() {
            let mut j = i;
            while j > 0 && before(&results[j], &results[j.saturating_sub(1)]) {
                results.swap(j, j.saturating_sub(1));
                j = j.saturating_sub(1);
            }
        }
        Ok(results)
    }

    fn get_top_dims(&self, top_n_dims: i32, top_n_children: i32) -> Result<Vec<FacetResult>> {
        self.top_dims(top_n_dims, top_n_children)
    }
}

/// Implements [`Facets`] for a wrapper of a [`TaxonomyFacets`] field `inner`,
/// and derefs to it.
macro_rules! taxonomy_facets_wrapper {
    ($ty:ident) => {
        impl<'t> std::ops::Deref for $ty<'t> {
            type Target = $crate::facets::taxonomy::taxonomy_facets::TaxonomyFacets<'t>;
            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl $crate::facets::Facets for $ty<'_> {
            fn get_all_children(
                &self,
                dim: &str,
                path: &[&str],
            ) -> $crate::Result<Option<$crate::facets::FacetResult>> {
                self.inner.get_all_children(dim, path)
            }
            fn get_top_children(
                &self,
                top_n: i32,
                dim: &str,
                path: &[&str],
            ) -> $crate::Result<Option<$crate::facets::FacetResult>> {
                self.inner.get_top_children(top_n, dim, path)
            }
            fn get_specific_value(
                &self,
                dim: &str,
                path: &[&str],
            ) -> $crate::Result<$crate::facets::FacetValue> {
                self.inner.get_specific_value(dim, path)
            }
            fn get_all_dims(&self, top_n: i32) -> $crate::Result<Vec<$crate::facets::FacetResult>> {
                self.inner.get_all_dims(top_n)
            }
            fn get_top_dims(
                &self,
                top_n_dims: i32,
                top_n_children: i32,
            ) -> $crate::Result<Vec<$crate::facets::FacetResult>> {
                self.inner.get_top_dims(top_n_dims, top_n_children)
            }
        }
    };
}
pub(crate) use taxonomy_facets_wrapper;

#[cfg(test)]
mod tests;
