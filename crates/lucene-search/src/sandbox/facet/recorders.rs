//! `org.apache.lucene.sandbox.facet.recorders`: what is recorded per facet
//! ordinal -- a count ([`CountFacetRecorder`]), long aggregations
//! ([`LongAggregationsFacetRecorder`]), or several at once
//! ([`MultiFacetsRecorder`]).

use std::sync::{Arc, Mutex, MutexGuard};

use crate::directory_reader::SegmentReader;
use crate::values_source::{BoxLongValues, LongValuesSource, ValuesContext};
use crate::{Error, Result};

use super::{
    from_array, BoxOrdinalIterator, FacetCutter, FacetRecorder, FastMap, LeafFacetRecorder,
    LeafSlots, OrdinalIterator, NO_MORE_ORDS,
};

/// `Reducer`: how two values of one ordinal combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reducer {
    /// `Reducer.MAX`: `Math.max`.
    Max,
    /// `Reducer.SUM`: `Math.addExact` for integers, `+` for floats.
    Sum,
}

fn overflow(kind: &str) -> Error {
    Error::IllegalArgument(format!("{kind} overflow"))
}

impl Reducer {
    /// `reduce(int, int)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] ("integer overflow") where `SUM`'s
    /// `Math.addExact` throws `ArithmeticException`.
    pub fn reduce_i32(self, a: i32, b: i32) -> Result<i32> {
        match self {
            Reducer::Max => Ok(a.max(b)),
            Reducer::Sum => a.checked_add(b).ok_or_else(|| overflow("integer")),
        }
    }

    /// `reduce(long, long)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] ("long overflow") where `SUM`'s
    /// `Math.addExact` throws `ArithmeticException`.
    pub fn reduce_i64(self, a: i64, b: i64) -> Result<i64> {
        match self {
            Reducer::Max => Ok(a.max(b)),
            Reducer::Sum => a.checked_add(b).ok_or_else(|| overflow("long")),
        }
    }

    /// `reduce(float, float)`: `MAX` is `Math.max` (`NaN` wins, `0.0`
    /// beats `-0.0`).
    pub fn reduce_f32(self, a: f32, b: f32) -> f32 {
        match self {
            Reducer::Max => java_max_f32(a, b),
            Reducer::Sum => a + b,
        }
    }

    /// `reduce(double, double)`: as [`reduce_f32`](Self::reduce_f32).
    pub fn reduce_f64(self, a: f64, b: f64) -> f64 {
        match self {
            Reducer::Max => java_max_f64(a, b),
            Reducer::Sum => a + b,
        }
    }
}

/// `Math.max(float, float)`.
fn java_max_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `Math.max(double, double)`.
fn java_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a >= b {
        a
    } else {
        b
    }
}

fn lock<T>(m: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    m.lock()
        .map_err(|_| Error::IllegalState("facet recorder state poisoned".into()))
}

fn not_reduced() -> Error {
    Error::IllegalState("the recorder is read before reduce".into())
}

/// The cutter's children of `ord` (Java dereferences `null` here).
fn children<'c>(cutter: &'c dyn FacetCutter, ord: i32) -> Result<BoxOrdinalIterator<'c>> {
    cutter.children_ords(ord)?.ok_or_else(|| {
        Error::IllegalState(format!(
            "the cutter rolls ordinal {ord} up but has no children ordinals"
        ))
    })
}

/// `CountFacetRecorder`: how many documents each ordinal was recorded for.
#[derive(Default)]
pub struct CountFacetRecorder {
    values: Mutex<Option<FastMap<i32, i32>>>,
    per_leaf: Arc<LeafSlots<FastMap<i32, i32>>>,
}

impl CountFacetRecorder {
    /// `new CountFacetRecorder()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `getCount(ord)`: 0 for an ordinal never recorded.
    ///
    /// # Errors
    /// [`Error::IllegalState`] before [`FacetRecorder::reduce`] (Java's
    /// `NullPointerException`).
    pub fn get_count(&self, ord: i32) -> Result<i32> {
        let values = lock(&self.values)?;
        let values = values.as_ref().ok_or_else(not_reduced)?;
        Ok(values.get(&ord).copied().unwrap_or(0))
    }
}

/// `values.addTo(key, delta)`: the new value (wrapping, as Java's `int`).
fn add_to(values: &mut FastMap<i32, i32>, key: i32, delta: i32) -> i32 {
    let v = values.entry(key).or_insert(0);
    *v = v.wrapping_add(delta);
    *v
}

/// `CountFacetRecorder.rollup`: the descendants' counts summed into each
/// child with any, and the total returned.
fn rollup_counts(
    values: &mut FastMap<i32, i32>,
    ord: i32,
    cutter: &dyn FacetCutter,
) -> Result<i32> {
    let mut child_ords = children(cutter, ord)?;
    let mut accum = 0i32;
    loop {
        let child = child_ords.next_ord()?;
        if child == NO_MORE_ORDS {
            return Ok(accum);
        }
        let rolled_up = rollup_counts(values, child, cutter)?;
        // Zeros are not rolled up, not to add ordinals without counts.
        let v = if rolled_up > 0 {
            add_to(values, child, rolled_up)
        } else {
            values.get(&child).copied().unwrap_or(0)
        };
        accum = accum.wrapping_add(v);
    }
}

impl FacetRecorder for CountFacetRecorder {
    fn get_leaf_recorder<'a>(
        &self,
        _reader: &'a SegmentReader,
    ) -> Result<Box<dyn LeafFacetRecorder + 'a>> {
        let slot = self.per_leaf.reserve()?;
        Ok(Box::new(CountLeafFacetRecorder {
            values: FastMap::default(),
            slots: self.per_leaf.clone(),
            slot,
        }))
    }

    fn recorded_ords(&self) -> Result<BoxOrdinalIterator<'static>> {
        let values = lock(&self.values)?;
        let values = values.as_ref().ok_or_else(not_reduced)?;
        Ok(Box::new(from_array(values.keys().copied().collect())))
    }

    fn is_empty(&self) -> Result<bool> {
        let values = lock(&self.values)?;
        Ok(values.as_ref().ok_or_else(not_reduced)?.is_empty())
    }

    fn reduce(&self, cutter: &dyn FacetCutter) -> Result<()> {
        let mut leaves = self.per_leaf.take_all()?.into_iter();
        // The first leaf's map, then the others added to it.
        let mut values = leaves.next().unwrap_or_default();
        for leaf in leaves {
            for (k, v) in leaf {
                add_to(&mut values, k, v);
            }
        }
        if let Some(mut dim_ords) = cutter.ordinals_to_rollup()? {
            loop {
                let dim_ord = dim_ords.next_ord()?;
                if dim_ord == NO_MORE_ORDS {
                    break;
                }
                let rolled_up = rollup_counts(&mut values, dim_ord, cutter)?;
                if rolled_up > 0 {
                    add_to(&mut values, dim_ord, rolled_up);
                }
            }
        }
        *lock(&self.values)? = Some(values);
        Ok(())
    }

    fn contains(&self, ord: i32) -> Result<bool> {
        let values = lock(&self.values)?;
        Ok(values.as_ref().ok_or_else(not_reduced)?.contains_key(&ord))
    }
}

/// `CountFacetRecorder.CountLeafFacetRecorder`.
struct CountLeafFacetRecorder {
    values: FastMap<i32, i32>,
    slots: Arc<LeafSlots<FastMap<i32, i32>>>,
    slot: usize,
}

impl LeafFacetRecorder for CountLeafFacetRecorder {
    fn record(&mut self, _doc: i32, ord: i32) -> Result<()> {
        add_to(&mut self.values, ord, 1);
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        let this = *self;
        this.slots.fill(this.slot, this.values)
    }
}

/// `LongAggregationsFacetRecorder`: per ordinal, each source's values of
/// the ordinal's documents reduced by its [`Reducer`] (from 0).
pub struct LongAggregationsFacetRecorder {
    sources: Vec<Arc<dyn LongValuesSource>>,
    reducers: Vec<Reducer>,
    values: Mutex<Option<FastMap<i32, Vec<i64>>>>,
    per_leaf: Arc<LeafSlots<FastMap<i32, Vec<i64>>>>,
}

impl LongAggregationsFacetRecorder {
    /// `new LongAggregationsFacetRecorder(longValuesSources, reducers)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for different lengths (Java's assertion).
    pub fn new(sources: Vec<Arc<dyn LongValuesSource>>, reducers: Vec<Reducer>) -> Result<Self> {
        if sources.len() != reducers.len() {
            return Err(Error::IllegalArgument(format!(
                "{} values sources but {} reducers",
                sources.len(),
                reducers.len()
            )));
        }
        Ok(LongAggregationsFacetRecorder {
            sources,
            reducers,
            values: Mutex::new(None),
            per_leaf: Arc::default(),
        })
    }

    /// `getRecordedValue(ord, valuesId)`: 0 for an ordinal never recorded.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] ("Invalid request for ordinal values")
    /// for a `values_id` outside the sources; [`Error::IllegalState`] before
    /// [`FacetRecorder::reduce`].
    pub fn get_recorded_value(&self, ord: i32, values_id: i32) -> Result<i64> {
        let id = usize::try_from(values_id)
            .ok()
            .filter(|&i| i < self.sources.len())
            .ok_or_else(|| Error::IllegalArgument("Invalid request for ordinal values".into()))?;
        let values = lock(&self.values)?;
        let values = values.as_ref().ok_or_else(not_reduced)?;
        Ok(values
            .get(&ord)
            .and_then(|v| v.get(id))
            .copied()
            .unwrap_or(0))
    }

    /// `rollup(accum, ord, facetCutter)`: whether `ord` (now) has values.
    fn rollup(
        &self,
        values: &mut FastMap<i32, Vec<i64>>,
        ord: i32,
        cutter: &dyn FacetCutter,
    ) -> Result<bool> {
        let mut has = values.contains_key(&ord);
        let mut child_ords = children(cutter, ord)?;
        loop {
            let child = child_ords.next_ord()?;
            if child == NO_MORE_ORDS {
                return Ok(has);
            }
            if !self.rollup(values, child, cutter)? {
                continue;
            }
            let current = values.get(&child).cloned().unwrap_or_default();
            let accum = values
                .entry(ord)
                .or_insert_with(|| vec![0; self.sources.len()]);
            has = true;
            for ((a, &c), r) in accum.iter_mut().zip(&current).zip(&self.reducers) {
                *a = r.reduce_i64(*a, c)?;
            }
        }
    }
}

impl FacetRecorder for LongAggregationsFacetRecorder {
    fn get_leaf_recorder<'a>(
        &self,
        reader: &'a SegmentReader,
    ) -> Result<Box<dyn LeafFacetRecorder + 'a>> {
        let ctx = ValuesContext::for_reader(reader);
        let values = self
            .sources
            .iter()
            .map(|s| s.get_values(&ctx, 0, None))
            .collect::<Result<Vec<_>>>()?;
        let slot = self.per_leaf.reserve()?;
        Ok(Box::new(LongAggregationsLeafFacetRecorder {
            values,
            reducers: self.reducers.clone(),
            per_ordinal: FastMap::default(),
            slots: self.per_leaf.clone(),
            slot,
        }))
    }

    fn recorded_ords(&self) -> Result<BoxOrdinalIterator<'static>> {
        let values = lock(&self.values)?;
        let values = values.as_ref().ok_or_else(not_reduced)?;
        Ok(Box::new(from_array(values.keys().copied().collect())))
    }

    fn is_empty(&self) -> Result<bool> {
        let values = lock(&self.values)?;
        Ok(values.as_ref().ok_or_else(not_reduced)?.is_empty())
    }

    fn reduce(&self, cutter: &dyn FacetCutter) -> Result<()> {
        let mut leaves = self.per_leaf.take_all()?.into_iter();
        let mut values = leaves.next().unwrap_or_default();
        for leaf in leaves {
            for (k, v) in leaf {
                match values.get_mut(&k) {
                    None => {
                        values.insert(k, v);
                    }
                    Some(vals) => {
                        for ((a, &b), r) in vals.iter_mut().zip(&v).zip(&self.reducers) {
                            *a = r.reduce_i64(*a, b)?;
                        }
                    }
                }
            }
        }
        if let Some(mut dim_ords) = cutter.ordinals_to_rollup()? {
            loop {
                let dim_ord = dim_ords.next_ord()?;
                if dim_ord == NO_MORE_ORDS {
                    break;
                }
                self.rollup(&mut values, dim_ord, cutter)?;
            }
        }
        *lock(&self.values)? = Some(values);
        Ok(())
    }

    fn contains(&self, ord: i32) -> Result<bool> {
        let values = lock(&self.values)?;
        Ok(values.as_ref().ok_or_else(not_reduced)?.contains_key(&ord))
    }
}

/// `LongAggregationsFacetRecorder.LongAggregationsLeafFacetRecorder`.
struct LongAggregationsLeafFacetRecorder<'a> {
    values: Vec<BoxLongValues<'a>>,
    reducers: Vec<Reducer>,
    per_ordinal: FastMap<i32, Vec<i64>>,
    slots: Arc<LeafSlots<FastMap<i32, Vec<i64>>>>,
    slot: usize,
}

impl LeafFacetRecorder for LongAggregationsLeafFacetRecorder<'_> {
    fn record(&mut self, doc: i32, ord: i32) -> Result<()> {
        let n = self.values.len();
        let for_ord = self.per_ordinal.entry(ord).or_insert_with(|| vec![0; n]);
        for ((values, r), slot) in self
            .values
            .iter_mut()
            .zip(&self.reducers)
            .zip(for_ord.iter_mut())
        {
            if values.advance_exact(doc)? {
                *slot = r.reduce_i64(*slot, values.long_value()?)?;
            }
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        let this = *self;
        this.slots.fill(this.slot, this.per_ordinal)
    }
}

/// `MultiFacetsRecorder`: every ordinal recorded by each delegate.
pub struct MultiFacetsRecorder {
    delegates: Vec<Arc<dyn FacetRecorder>>,
}

impl MultiFacetsRecorder {
    /// `new MultiFacetsRecorder(delegates...)`.
    pub fn new(delegates: Vec<Arc<dyn FacetRecorder>>) -> Self {
        MultiFacetsRecorder { delegates }
    }
}

fn unsupported(what: &str) -> Error {
    Error::Unsupported(format!(
        "Not supported, call {what} for sub-recorders instead"
    ))
}

impl FacetRecorder for MultiFacetsRecorder {
    fn get_leaf_recorder<'a>(
        &self,
        reader: &'a SegmentReader,
    ) -> Result<Box<dyn LeafFacetRecorder + 'a>> {
        let delegates = self
            .delegates
            .iter()
            .map(|d| d.get_leaf_recorder(reader))
            .collect::<Result<Vec<_>>>()?;
        Ok(Box::new(MultiFacetsLeafRecorder { delegates }))
    }

    fn recorded_ords(&self) -> Result<BoxOrdinalIterator<'static>> {
        Err(unsupported("recordedOrds"))
    }

    fn is_empty(&self) -> Result<bool> {
        Err(unsupported("isEmpty"))
    }

    fn reduce(&self, cutter: &dyn FacetCutter) -> Result<()> {
        for d in &self.delegates {
            d.reduce(cutter)?;
        }
        Ok(())
    }

    fn contains(&self, _ord: i32) -> Result<bool> {
        Err(unsupported("contains"))
    }
}

/// `MultiFacetsRecorder.MultiFacetsLeafRecorder`.
struct MultiFacetsLeafRecorder<'a> {
    delegates: Vec<Box<dyn LeafFacetRecorder + 'a>>,
}

impl LeafFacetRecorder for MultiFacetsLeafRecorder<'_> {
    fn record(&mut self, doc: i32, ord: i32) -> Result<()> {
        for d in &mut self.delegates {
            d.record(doc, ord)?;
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        for d in self.delegates {
            d.finish()?;
        }
        Ok(())
    }
}
