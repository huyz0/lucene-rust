//! `org.apache.lucene.sandbox.facet.iterators.ComparableSupplier` and
//! `utils.ComparableUtils`: per-ordinal sort keys, greatest first.
//!
//! Each comparable is a Rust [`Ord`] key whose order is the Java
//! `compareTo`'s; ties on the ranked value go to the smaller ordinal (or
//! value), which Java expresses by comparing the ordinals reversed. Java's
//! `reuseComparable` -- an allocation saving for its object comparables --
//! has no counterpart: the keys are plain values.

use std::cmp::Reverse;

use crate::Result;

use super::cutters::LongValueFacetCutter;
use super::recorders::{CountFacetRecorder, LongAggregationsFacetRecorder};

/// `ComparableSupplier`: an ordinal's comparable.
pub trait ComparableSupplier {
    /// The comparable; greater sorts first.
    type Comparable: Ord;

    /// `createComparable(ord)`.
    ///
    /// # Errors
    /// Whatever reading the ordinal's value reports.
    fn create_comparable(&self, ord: i32) -> Result<Self::Comparable>;
}

impl<S: ComparableSupplier + ?Sized> ComparableSupplier for &S {
    type Comparable = S::Comparable;

    fn create_comparable(&self, ord: i32) -> Result<Self::Comparable> {
        (**self).create_comparable(ord)
    }
}

/// `ComparableUtils.byOrdinal()`: the smaller ordinal first.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByOrdinal;

impl ComparableSupplier for ByOrdinal {
    type Comparable = Reverse<i32>;

    fn create_comparable(&self, ord: i32) -> Result<Reverse<i32>> {
        Ok(Reverse(ord))
    }
}

/// `ComparableUtils.byCount(countRecorder)`: the greater count first, then
/// the smaller ordinal.
pub struct ByCount<'r> {
    recorder: &'r CountFacetRecorder,
}

impl<'r> ByCount<'r> {
    /// `byCount(recorder)`.
    pub fn new(recorder: &'r CountFacetRecorder) -> Self {
        ByCount { recorder }
    }
}

impl ComparableSupplier for ByCount<'_> {
    type Comparable = (i32, Reverse<i32>);

    fn create_comparable(&self, ord: i32) -> Result<Self::Comparable> {
        Ok((self.recorder.get_count(ord)?, Reverse(ord)))
    }
}

/// `ComparableUtils.byAggregatedValue(countRecorder, aggregationsRecorder,
/// aggregationId)`: the greater aggregated value first, then the greater
/// count, then the smaller ordinal.
pub struct ByAggregatedValue<'r> {
    counts: &'r CountFacetRecorder,
    aggregations: &'r LongAggregationsFacetRecorder,
    aggregation_id: i32,
}

impl<'r> ByAggregatedValue<'r> {
    /// `byAggregatedValue(countRecorder, longAggregationsFacetRecorder,
    /// aggregationId)`.
    pub fn new(
        counts: &'r CountFacetRecorder,
        aggregations: &'r LongAggregationsFacetRecorder,
        aggregation_id: i32,
    ) -> Self {
        ByAggregatedValue {
            counts,
            aggregations,
            aggregation_id,
        }
    }
}

impl ComparableSupplier for ByAggregatedValue<'_> {
    type Comparable = (i64, i32, Reverse<i32>);

    fn create_comparable(&self, ord: i32) -> Result<Self::Comparable> {
        Ok((
            self.aggregations
                .get_recorded_value(ord, self.aggregation_id)?,
            self.counts.get_count(ord)?,
            Reverse(ord),
        ))
    }
}

/// `ComparableUtils.byLongValue(longValueFacetCutter)`: the smaller value
/// first.
pub struct ByLongValue<'c> {
    cutter: &'c LongValueFacetCutter,
}

impl<'c> ByLongValue<'c> {
    /// `byLongValue(longValueFacetCutter)`.
    pub fn new(cutter: &'c LongValueFacetCutter) -> Self {
        ByLongValue { cutter }
    }
}

impl ComparableSupplier for ByLongValue<'_> {
    type Comparable = Reverse<i64>;

    fn create_comparable(&self, ord: i32) -> Result<Reverse<i64>> {
        Ok(Reverse(self.cutter.get_value(ord)?))
    }
}

/// `ComparableUtils.byCount(countFacetRecorder, longValueFacetCutter)`: the
/// greater count first, then the smaller value.
pub struct ByCountAndLongValue<'r> {
    counts: &'r CountFacetRecorder,
    cutter: &'r LongValueFacetCutter,
}

impl<'r> ByCountAndLongValue<'r> {
    /// `byCount(countFacetRecorder, longValueFacetCutter)`.
    pub fn new(counts: &'r CountFacetRecorder, cutter: &'r LongValueFacetCutter) -> Self {
        ByCountAndLongValue { counts, cutter }
    }
}

impl ComparableSupplier for ByCountAndLongValue<'_> {
    type Comparable = (i32, Reverse<i64>);

    fn create_comparable(&self, ord: i32) -> Result<Self::Comparable> {
        Ok((
            self.counts.get_count(ord)?,
            Reverse(self.cutter.get_value(ord)?),
        ))
    }
}

/// `ComparableUtils.sort(ordinals, comparableSupplier)`: greatest first,
/// equal comparables kept in their order (Java's `InPlaceMergeSorter` is
/// stable).
///
/// # Errors
/// Whatever the supplier reports.
pub fn sort<S: ComparableSupplier + ?Sized>(ordinals: &mut [i32], supplier: &S) -> Result<()> {
    let mut keyed = ordinals
        .iter()
        .map(|&o| Ok((supplier.create_comparable(o)?, o)))
        .collect::<Result<Vec<_>>>()?;
    keyed.sort_by(|a, b| b.0.cmp(&a.0));
    for (slot, (_, o)) in ordinals.iter_mut().zip(keyed) {
        *slot = o;
    }
    Ok(())
}
