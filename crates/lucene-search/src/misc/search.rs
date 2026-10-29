//! `org.apache.lucene.misc.search`: doc-value statistics, a diversified
//! top-docs collector, a memory-accounting bitset collector and a debugging
//! query wrapper.
//!
//! # How the collector model maps
//!
//! A [`ScoringCollector`] here sees global document ids and no leaf, where
//! Java hands `getLeafCollector` each `LeafReaderContext`. The collectors
//! that read per-leaf data take the leaves up front ([`Leaves`]) and find
//! the leaf of each hit. A collector cannot fail from `collect`, so an
//! error a leaf's doc values or a memory tracker raises is held and comes out
//! of `finish` / `reduce`.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::{ScoreMode, ScoringCollector};
use crate::collectors::CollectorManager;
use crate::misc::memory::MemoryTracker;
use crate::query::Clause;
use crate::reader::{LeafReader, NumericDocValues, SortedDocValues};
use crate::reader::{SortedNumericDocValues, SortedSetDocValues};
use crate::{Error, Result};

/// The leaves of an index in doc-base order, with each one's doc base.
#[derive(Clone, Copy)]
pub struct Leaves<'a> {
    readers: &'a [&'a dyn LeafReader],
}

impl<'a> Leaves<'a> {
    /// The leaves, in the order whose cumulative `max_doc` is the doc base.
    pub fn new(readers: &'a [&'a dyn LeafReader]) -> Self {
        Self { readers }
    }

    /// `(leaf index, doc base)` of the leaf holding global `doc`.
    fn locate(&self, doc: i32) -> Option<(usize, i32)> {
        let mut base = 0i32;
        for (i, r) in self.readers.iter().enumerate() {
            let end = base.saturating_add(r.max_doc());
            if doc >= base && doc < end {
                return Some((i, base));
            }
            base = end;
        }
        None
    }
}

/// What every `DocValuesStats` holds: the field and the two counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatsBase {
    field: String,
    missing: i32,
    count: i32,
}

impl StatsBase {
    fn new(field: &str) -> Self {
        Self {
            field: field.to_string(),
            missing: 0,
            count: 0,
        }
    }
}

/// `DocValuesStats<T>`: statistics of one doc-values field over a set of
/// documents.
pub trait DocValuesStats<'a> {
    /// The shared counts.
    fn base(&self) -> &StatsBase;
    /// The shared counts, mutably.
    fn base_mut(&mut self) -> &mut StatsBase;
    /// `init(context)`: opens the field's doc values in `leaf`; `false` when
    /// the leaf has none, so that every document of it is missing.
    ///
    /// # Errors
    /// Whatever opening the doc values fails with.
    fn init(&mut self, leaf: &'a dyn LeafReader) -> Result<bool>;
    /// `hasValue(doc)`, for a leaf-local `doc`.
    ///
    /// # Errors
    /// Whatever reading the doc values fails with.
    fn has_value(&mut self, doc: i32) -> Result<bool>;
    /// `doAccumulate(count)`: folds in the value `has_value` positioned on.
    ///
    /// # Errors
    /// Whatever reading the doc values fails with.
    fn do_accumulate(&mut self, count: i32) -> Result<()>;
    /// `merge(other)`: combines the statistics of two sets of documents.
    fn merge(&mut self, other: &Self)
    where
        Self: Sized;

    /// `accumulate(doc)`.
    ///
    /// # Errors
    /// Whatever the doc values fail with.
    fn accumulate(&mut self, doc: i32) -> Result<()> {
        if self.has_value(doc)? {
            self.base_mut().count += 1;
            let count = self.base().count;
            self.do_accumulate(count)
        } else {
            self.base_mut().missing += 1;
            Ok(())
        }
    }

    /// `addMissing()`.
    fn add_missing(&mut self) {
        self.base_mut().missing += 1;
    }

    /// `field()`
    fn field(&self) -> &str {
        &self.base().field
    }

    /// `count()`: the documents that had a value.
    fn count(&self) -> i32 {
        self.base().count
    }

    /// `missing()`: the documents that had none.
    fn missing(&self) -> i32 {
        self.base().missing
    }
}

/// The value type a numeric statistic is kept in: `Long` or `Double`.
pub trait StatValue: Copy + PartialEq + std::fmt::Debug {
    /// `Long.MAX_VALUE` / `Double.MAX_VALUE`
    const INITIAL_MIN: Self;
    /// `Long.MIN_VALUE` / `Double.MIN_VALUE` (the smallest *positive* double)
    const INITIAL_MAX: Self;
    /// The value a doc-values `long` stands for.
    fn from_bits(bits: i64) -> Self;
    /// `a.compareTo(b)`
    fn compare(a: Self, b: Self) -> std::cmp::Ordering;
    /// `sum + v`
    fn add(a: Self, b: Self) -> Self;
    /// `doubleValue()`
    fn to_f64(self) -> f64;
    /// `0`
    fn zero() -> Self;
}

impl StatValue for i64 {
    const INITIAL_MIN: Self = i64::MAX;
    const INITIAL_MAX: Self = i64::MIN;
    fn from_bits(bits: i64) -> Self {
        bits
    }
    fn compare(a: Self, b: Self) -> std::cmp::Ordering {
        a.cmp(&b)
    }
    fn add(a: Self, b: Self) -> Self {
        a.wrapping_add(b)
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn zero() -> Self {
        0
    }
}

impl StatValue for f64 {
    const INITIAL_MIN: Self = f64::MAX;
    const INITIAL_MAX: Self = f64::from_bits(1);
    fn from_bits(bits: i64) -> Self {
        f64::from_bits(bits as u64)
    }
    fn compare(a: Self, b: Self) -> std::cmp::Ordering {
        a.total_cmp(&b)
    }
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn zero() -> Self {
        0.0
    }
}

fn gt<V: StatValue>(a: V, b: V) -> bool {
    V::compare(a, b).is_gt()
}

fn lt<V: StatValue>(a: V, b: V) -> bool {
    V::compare(a, b).is_lt()
}

/// `NumericDocValuesStats` over `Long` ([`LongDocValuesStats`]) or `Double`
/// ([`DoubleDocValuesStats`]): count, min, max, sum, mean, variance, stdev.
pub struct NumericDocValuesStats<'a, V: StatValue> {
    base: StatsBase,
    min: V,
    max: V,
    sum: V,
    mean: f64,
    variance: f64,
    ndv: Option<Box<dyn NumericDocValues + 'a>>,
}

/// `DocValuesStats.LongDocValuesStats`
pub type LongDocValuesStats<'a> = NumericDocValuesStats<'a, i64>;
/// `DocValuesStats.DoubleDocValuesStats`
pub type DoubleDocValuesStats<'a> = NumericDocValuesStats<'a, f64>;

impl<'a, V: StatValue> NumericDocValuesStats<'a, V> {
    /// `new LongDocValuesStats(field)` / `new DoubleDocValuesStats(field)`.
    pub fn new(field: &str) -> Self {
        Self {
            base: StatsBase::new(field),
            min: V::INITIAL_MIN,
            max: V::INITIAL_MAX,
            sum: V::zero(),
            mean: 0.0,
            variance: 0.0,
            ndv: None,
        }
    }

    /// `min()`
    pub fn min(&self) -> V {
        self.min
    }

    /// `max()`
    pub fn max(&self) -> V {
        self.max
    }

    /// `sum()`
    pub fn sum(&self) -> V {
        self.sum
    }

    /// `mean()`
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// `variance()`
    pub fn variance(&self) -> f64 {
        if self.base.count > 0 {
            self.variance / f64::from(self.base.count)
        } else {
            0.0
        }
    }

    /// `stdev()`
    pub fn stdev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl<'a, V: StatValue> DocValuesStats<'a> for NumericDocValuesStats<'a, V> {
    fn base(&self) -> &StatsBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut StatsBase {
        &mut self.base
    }

    fn init(&mut self, leaf: &'a dyn LeafReader) -> Result<bool> {
        self.ndv = leaf.numeric_doc_values(&self.base.field)?;
        Ok(self.ndv.is_some())
    }

    fn has_value(&mut self, doc: i32) -> Result<bool> {
        match self.ndv.as_mut() {
            Some(ndv) => ndv.advance_exact(doc),
            None => Ok(false),
        }
    }

    fn do_accumulate(&mut self, count: i32) -> Result<()> {
        let Some(ndv) = self.ndv.as_ref() else {
            return Ok(());
        };
        let val = V::from_bits(ndv.long_value());
        if gt(val, self.max) {
            self.max = val;
        }
        if lt(val, self.min) {
            self.min = val;
        }
        self.sum = V::add(self.sum, val);
        let old_mean = self.mean;
        let v = val.to_f64();
        self.mean += (v - self.mean) / f64::from(count);
        self.variance += (v - self.mean) * (v - old_mean);
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        self.base.missing += other.base.missing;
        if other.base.count == 0 {
            return;
        }
        if self.base.count == 0 {
            self.base.count = other.base.count;
            self.min = other.min;
            self.max = other.max;
            self.mean = other.mean;
            self.variance = other.variance;
            self.sum = V::add(self.sum, other.sum);
            return;
        }
        let total_count = self.base.count + other.base.count;
        let combined_mean = (self.sum.to_f64() + other.sum.to_f64()) / f64::from(total_count);
        let target_delta = self.mean - combined_mean;
        let source_delta = other.mean - combined_mean;
        self.variance = self.variance
            + other.variance
            + target_delta * target_delta * f64::from(self.base.count)
            + source_delta * source_delta * f64::from(other.base.count);
        self.mean = combined_mean;
        self.base.count = total_count;
        if lt(other.min, self.min) {
            self.min = other.min;
        }
        if gt(other.max, self.max) {
            self.max = other.max;
        }
        self.sum = V::add(self.sum, other.sum);
    }
}

/// `SortedNumericDocValuesStats` over `Long` ([`SortedLongDocValuesStats`])
/// or `Double` ([`SortedDoubleDocValuesStats`]): the same, over every value
/// of every document, plus `values_count`.
pub struct SortedNumericDocValuesStats<'a, V: StatValue> {
    base: StatsBase,
    min: V,
    max: V,
    sum: V,
    values_count: i64,
    mean: f64,
    variance: f64,
    sndv: Option<Box<dyn SortedNumericDocValues + 'a>>,
}

/// `DocValuesStats.SortedLongDocValuesStats`
pub type SortedLongDocValuesStats<'a> = SortedNumericDocValuesStats<'a, i64>;
/// `DocValuesStats.SortedDoubleDocValuesStats`
pub type SortedDoubleDocValuesStats<'a> = SortedNumericDocValuesStats<'a, f64>;

impl<'a, V: StatValue> SortedNumericDocValuesStats<'a, V> {
    /// `new SortedLongDocValuesStats(field)` / `new SortedDoubleDocValuesStats(field)`.
    pub fn new(field: &str) -> Self {
        Self {
            base: StatsBase::new(field),
            min: V::INITIAL_MIN,
            max: V::INITIAL_MAX,
            sum: V::zero(),
            values_count: 0,
            mean: 0.0,
            variance: 0.0,
            sndv: None,
        }
    }

    /// `min()`
    pub fn min(&self) -> V {
        self.min
    }

    /// `max()`
    pub fn max(&self) -> V {
        self.max
    }

    /// `sum()`
    pub fn sum(&self) -> V {
        self.sum
    }

    /// `mean()`
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// `valuesCount()`
    pub fn values_count(&self) -> i64 {
        self.values_count
    }

    /// `variance()`: over the documents' count, as Java does.
    pub fn variance(&self) -> f64 {
        if self.base.count > 0 {
            self.variance / f64::from(self.base.count)
        } else {
            0.0
        }
    }

    /// `stdev()`
    pub fn stdev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl<'a, V: StatValue> DocValuesStats<'a> for SortedNumericDocValuesStats<'a, V> {
    fn base(&self) -> &StatsBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut StatsBase {
        &mut self.base
    }

    fn init(&mut self, leaf: &'a dyn LeafReader) -> Result<bool> {
        self.sndv = leaf.sorted_numeric_doc_values(&self.base.field)?;
        Ok(self.sndv.is_some())
    }

    fn has_value(&mut self, doc: i32) -> Result<bool> {
        match self.sndv.as_mut() {
            Some(sndv) => sndv.advance_exact(doc),
            None => Ok(false),
        }
    }

    fn do_accumulate(&mut self, _count: i32) -> Result<()> {
        let Some(sndv) = self.sndv.as_mut() else {
            return Ok(());
        };
        for _ in 0..sndv.doc_value_count() {
            let val = V::from_bits(sndv.next_value()?);
            if gt(val, self.max) {
                self.max = val;
            }
            if lt(val, self.min) {
                self.min = val;
            }
            self.sum = V::add(self.sum, val);
            let old_mean = self.mean;
            self.values_count += 1;
            let v = val.to_f64();
            self.mean += (v - self.mean) / self.values_count as f64;
            self.variance += (v - self.mean) * (v - old_mean);
        }
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        self.base.missing += other.base.missing;
        if other.base.count == 0 {
            return;
        }
        if self.base.count == 0 {
            self.base.count = other.base.count;
            self.min = other.min;
            self.max = other.max;
            self.mean = other.mean;
            self.variance = other.variance;
            self.values_count = other.values_count;
            self.sum = V::add(self.sum, other.sum);
            return;
        }
        let total_values_count = self.values_count + other.values_count;
        let combined_mean = (self.sum.to_f64() + other.sum.to_f64()) / total_values_count as f64;
        let target_delta = self.mean - combined_mean;
        let source_delta = other.mean - combined_mean;
        self.variance = self.variance
            + other.variance
            + target_delta * target_delta * self.values_count as f64
            + source_delta * source_delta * other.values_count as f64;
        self.mean = combined_mean;
        self.values_count = total_values_count;
        self.base.count += other.base.count;
        if lt(other.min, self.min) {
            self.min = other.min;
        }
        if gt(other.max, self.max) {
            self.max = other.max;
        }
        self.sum = V::add(self.sum, other.sum);
    }
}

/// The base `merge`: counts added, `min`/`max` the smaller / larger of the
/// two when both are set.
fn merge_bytes_stats(
    base: &mut StatsBase,
    min: &mut Option<Vec<u8>>,
    max: &mut Option<Vec<u8>>,
    other: (&StatsBase, &Option<Vec<u8>>, &Option<Vec<u8>>),
) {
    base.count += other.0.count;
    base.missing += other.0.missing;
    if let Some(o) = other.1 {
        if min.as_ref().is_none_or(|m| o < m) {
            *min = Some(o.clone());
        }
    }
    if let Some(o) = other.2 {
        if max.as_ref().is_none_or(|m| o > m) {
            *max = Some(o.clone());
        }
    }
}

fn fold_bytes(val: Vec<u8>, min: &mut Option<Vec<u8>>, max: &mut Option<Vec<u8>>) {
    if max.as_ref().is_none_or(|m| val > *m) {
        *max = Some(val.clone());
    }
    if min.as_ref().is_none_or(|m| val < *m) {
        *min = Some(val);
    }
}

/// `DocValuesStats.SortedDocValuesStats`: the smallest and largest term.
pub struct SortedDocValuesStats<'a> {
    base: StatsBase,
    min: Option<Vec<u8>>,
    max: Option<Vec<u8>>,
    sdv: Option<Box<dyn SortedDocValues + 'a>>,
}

impl<'a> SortedDocValuesStats<'a> {
    /// `new SortedDocValuesStats(field)`.
    pub fn new(field: &str) -> Self {
        Self {
            base: StatsBase::new(field),
            min: None,
            max: None,
            sdv: None,
        }
    }

    /// `min()`
    pub fn min(&self) -> Option<&[u8]> {
        self.min.as_deref()
    }

    /// `max()`
    pub fn max(&self) -> Option<&[u8]> {
        self.max.as_deref()
    }
}

impl<'a> DocValuesStats<'a> for SortedDocValuesStats<'a> {
    fn base(&self) -> &StatsBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut StatsBase {
        &mut self.base
    }

    fn init(&mut self, leaf: &'a dyn LeafReader) -> Result<bool> {
        self.sdv = leaf.sorted_doc_values(&self.base.field)?;
        Ok(self.sdv.is_some())
    }

    fn has_value(&mut self, doc: i32) -> Result<bool> {
        match self.sdv.as_mut() {
            Some(sdv) => sdv.advance_exact(doc),
            None => Ok(false),
        }
    }

    fn do_accumulate(&mut self, _count: i32) -> Result<()> {
        let Some(sdv) = self.sdv.as_mut() else {
            return Ok(());
        };
        let ord = sdv.ord_value();
        let val = sdv.lookup_ord(ord)?;
        fold_bytes(val, &mut self.min, &mut self.max);
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        merge_bytes_stats(
            &mut self.base,
            &mut self.min,
            &mut self.max,
            (&other.base, &other.min, &other.max),
        );
    }
}

/// `DocValuesStats.SortedSetDocValuesStats`: the smallest and largest term
/// over every value of every document.
pub struct SortedSetDocValuesStats<'a> {
    base: StatsBase,
    min: Option<Vec<u8>>,
    max: Option<Vec<u8>>,
    ssdv: Option<Box<dyn SortedSetDocValues + 'a>>,
}

impl<'a> SortedSetDocValuesStats<'a> {
    /// `new SortedSetDocValuesStats(field)`.
    pub fn new(field: &str) -> Self {
        Self {
            base: StatsBase::new(field),
            min: None,
            max: None,
            ssdv: None,
        }
    }

    /// `min()`
    pub fn min(&self) -> Option<&[u8]> {
        self.min.as_deref()
    }

    /// `max()`
    pub fn max(&self) -> Option<&[u8]> {
        self.max.as_deref()
    }
}

impl<'a> DocValuesStats<'a> for SortedSetDocValuesStats<'a> {
    fn base(&self) -> &StatsBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut StatsBase {
        &mut self.base
    }

    fn init(&mut self, leaf: &'a dyn LeafReader) -> Result<bool> {
        self.ssdv = leaf.sorted_set_doc_values(&self.base.field)?;
        Ok(self.ssdv.is_some())
    }

    fn has_value(&mut self, doc: i32) -> Result<bool> {
        match self.ssdv.as_mut() {
            Some(ssdv) => ssdv.advance_exact(doc),
            None => Ok(false),
        }
    }

    fn do_accumulate(&mut self, _count: i32) -> Result<()> {
        let Some(ssdv) = self.ssdv.as_mut() else {
            return Ok(());
        };
        for _ in 0..ssdv.doc_value_count() {
            let ord = ssdv.next_ord()?;
            let val = ssdv.lookup_ord(ord)?;
            fold_bytes(val, &mut self.min, &mut self.max);
        }
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        merge_bytes_stats(
            &mut self.base,
            &mut self.min,
            &mut self.max,
            (&other.base, &other.min, &other.max),
        );
    }
}

/// `DocValuesStatsCollector`: feeds every hit to a [`DocValuesStats`]. A leaf
/// that has no doc values for the field counts all its hits as missing.
pub struct DocValuesStatsCollector<'a, S: DocValuesStats<'a>> {
    stats: S,
    leaves: Leaves<'a>,
    /// `(leaf index, doc base, shouldProcess)` of the leaf being collected.
    current: Option<(usize, i32, bool)>,
    error: Option<Error>,
}

impl<'a, S: DocValuesStats<'a>> DocValuesStatsCollector<'a, S> {
    /// `new DocValuesStatsCollector(stats)` over the index's `leaves`.
    pub fn new(stats: S, leaves: Leaves<'a>) -> Self {
        Self {
            stats,
            leaves,
            current: None,
            error: None,
        }
    }

    /// `getStats()`, once the search is done.
    ///
    /// # Errors
    /// The first error the doc values raised while collecting.
    pub fn finish(self) -> Result<S> {
        match self.error {
            Some(e) => Err(e),
            None => Ok(self.stats),
        }
    }
}

impl<'a, S: DocValuesStats<'a>> ScoringCollector for DocValuesStatsCollector<'a, S> {
    fn collect(&mut self, doc_id: i32, _score: f32) {
        if self.error.is_some() {
            return;
        }
        let in_current = self.current.is_some_and(|(i, base, _)| {
            self.leaves
                .readers
                .get(i)
                .is_some_and(|r| doc_id >= base && doc_id - base < r.max_doc())
        });
        if !in_current {
            let Some((i, base)) = self.leaves.locate(doc_id) else {
                self.stats.add_missing();
                return;
            };
            match self.stats.init(self.leaves.readers[i]) {
                Ok(process) => self.current = Some((i, base, process)),
                Err(e) => {
                    self.error = Some(e);
                    return;
                }
            }
        }
        let Some((_, base, process)) = self.current else {
            return;
        };
        if process {
            if let Err(e) = self.stats.accumulate(doc_id - base) {
                self.error = Some(e);
            }
        } else {
            self.stats.add_missing();
        }
    }

    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
}

/// `DocValuesStatsCollectorManager`: a collector per slice over a fresh
/// statistics object, and their statistics merged.
pub struct DocValuesStatsCollectorManager<F> {
    stats_supplier: F,
}

impl<F> DocValuesStatsCollectorManager<F> {
    /// `new DocValuesStatsCollectorManager(statsSupplier)`.
    pub fn new(stats_supplier: F) -> Self {
        Self { stats_supplier }
    }

    /// `newCollector()`.
    pub fn new_collector<'a, S>(&self, leaves: Leaves<'a>) -> DocValuesStatsCollector<'a, S>
    where
        F: Fn() -> S,
        S: DocValuesStats<'a>,
    {
        DocValuesStatsCollector::new((self.stats_supplier)(), leaves)
    }

    /// `reduce(collectors)`: the collectors' statistics merged into a fresh
    /// one.
    ///
    /// # Errors
    /// The first error a collector held.
    pub fn reduce<'a, S>(&self, collectors: Vec<DocValuesStatsCollector<'a, S>>) -> Result<S>
    where
        F: Fn() -> S,
        S: DocValuesStats<'a>,
    {
        let mut merged = (self.stats_supplier)();
        for collector in collectors {
            let stats = collector.finish()?;
            merged.merge(&stats);
        }
        Ok(merged)
    }
}

/// `DiversifiedTopDocsCollector.ScoreDocKey`: a hit with the key it was
/// diversified on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreDocKey {
    /// The global document id.
    pub doc: i32,
    /// The hit's score.
    pub score: f32,
    /// The key (`0` for a document without one).
    pub key: i64,
}

/// `ScoreDocKeyQueue.lessThan` as a total order: ascending is worst first.
impl Eq for ScoreDocKey {}

impl PartialOrd for ScoreDocKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreDocKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `hitA.score == hitB.score ? hitA.doc > hitB.doc : hitA.score < hitB.score`
        match self.score.partial_cmp(&other.score) {
            Some(std::cmp::Ordering::Equal) | None => other.doc.cmp(&self.doc),
            Some(o) => o,
        }
    }
}

/// The per-document key source of a diversified collector: `getKeys`, taking a
/// global document id and returning its key, `0` when it has none.
pub type KeySource = Box<dyn FnMut(i32) -> i64 + Send>;

/// `insertWithOverflow` on a queue ordered by [`ScoreDocKey`], of at most
/// `capacity` entries: what is pushed out, which is `item` itself when it is
/// no better than the worst kept.
fn insert_with_overflow(
    queue: &mut BTreeSet<ScoreDocKey>,
    capacity: usize,
    item: ScoreDocKey,
) -> Option<ScoreDocKey> {
    if capacity == 0 {
        return Some(item);
    }
    if queue.len() < capacity {
        queue.insert(item);
        return None;
    }
    let worst = *queue.first()?;
    if item < worst {
        return Some(item);
    }
    queue.pop_first();
    queue.insert(item);
    Some(worst)
}

/// `DiversifiedTopDocsCollector`: the top documents by score with at most
/// `max_hits_per_key` of any one key.
pub struct DiversifiedTopDocsCollector {
    keys: KeySource,
    num_hits: usize,
    max_num_per_key: usize,
    total_hits: u64,
    global_queue: BTreeSet<ScoreDocKey>,
    per_key_queues: HashMap<i64, BTreeSet<ScoreDocKey>>,
}

impl DiversifiedTopDocsCollector {
    /// `new DiversifiedTopDocsCollector(numHits, maxHitsPerKey)` with `keys`
    /// for `getKeys`.
    pub fn new(num_hits: usize, max_hits_per_key: usize, keys: KeySource) -> Self {
        Self {
            keys,
            num_hits,
            max_num_per_key: max_hits_per_key,
            total_hits: 0,
            global_queue: BTreeSet::new(),
            per_key_queues: HashMap::new(),
        }
    }

    /// `insert(addition, docBase, keys)`: what the insertion pushed out.
    fn insert(&mut self, mut addition: ScoreDocKey) -> Option<ScoreDocKey> {
        if self.global_queue.len() >= self.num_hits
            && self.global_queue.first().is_none_or(|top| addition < *top)
        {
            return Some(addition);
        }
        addition.key = (self.keys)(addition.doc);
        let this_key_q = self.per_key_queues.entry(addition.key).or_default();
        let per_key_overflow = insert_with_overflow(this_key_q, self.max_num_per_key, addition);
        match per_key_overflow {
            Some(o) if o == addition => Some(addition),
            None => {
                let global_overflow =
                    insert_with_overflow(&mut self.global_queue, self.num_hits, addition);
                self.per_key_group_remove(global_overflow);
                global_overflow
            }
            Some(overflow) => {
                self.global_queue.remove(&overflow);
                self.global_queue.insert(addition);
                Some(overflow)
            }
        }
    }

    /// `perKeyGroupRemove(globalOverflow)`: the entry pushed out of the global
    /// queue leaves its key's queue too (it is that queue's lowest).
    fn per_key_group_remove(&mut self, global_overflow: Option<ScoreDocKey>) {
        let Some(overflow) = global_overflow else {
            return;
        };
        if let Some(q) = self.per_key_queues.get_mut(&overflow.key) {
            q.pop_first();
            if q.is_empty() {
                self.per_key_queues.remove(&overflow.key);
            }
        }
    }

    /// `topDocs()`: the kept hits, best first, and the hits collected.
    pub fn top_docs(&self) -> DiversifiedTopDocs {
        DiversifiedTopDocs {
            total_hits: self.total_hits,
            hits: self.global_queue.iter().rev().copied().collect(),
        }
    }
}

impl ScoringCollector for DiversifiedTopDocsCollector {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.total_hits += 1;
        self.insert(ScoreDocKey {
            doc: doc_id,
            score,
            key: 0,
        });
    }

    fn score_mode(&self) -> ScoreMode {
        ScoreMode::Complete
    }
}

/// What a diversified search returns: `TopDocs` of `ScoreDocKey`s.
#[derive(Debug, Clone, PartialEq)]
pub struct DiversifiedTopDocs {
    /// `totalHits.value()` (relation `EQUAL_TO`).
    pub total_hits: u64,
    /// `scoreDocs`, best first.
    pub hits: Vec<ScoreDocKey>,
}

/// `DiversifiedTopDocsCollectorManager`: collectors over `new_keys()` key
/// sources, their results reduced to the best hits with at most
/// `max_hits_per_key` per key.
pub struct DiversifiedTopDocsCollectorManager<F> {
    num_hits: usize,
    max_hits_per_key: usize,
    new_keys: F,
}

impl<F: Fn() -> KeySource + Sync> DiversifiedTopDocsCollectorManager<F> {
    /// `new DiversifiedTopDocsCollectorManager(numHits, maxHitsPerKey)`;
    /// `new_keys` is the subclass's `newCollector`, giving each collector its
    /// key source.
    pub fn new(num_hits: usize, max_hits_per_key: usize, new_keys: F) -> Self {
        Self {
            num_hits,
            max_hits_per_key,
            new_keys,
        }
    }
}

impl<F: Fn() -> KeySource + Sync> CollectorManager for DiversifiedTopDocsCollectorManager<F> {
    type Collector = DiversifiedTopDocsCollector;
    type Output = DiversifiedTopDocs;

    fn new_collector(&self) -> Result<Self::Collector> {
        Ok(DiversifiedTopDocsCollector::new(
            self.num_hits,
            self.max_hits_per_key,
            (self.new_keys)(),
        ))
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output> {
        let mut total_hits = 0u64;
        let mut all_docs = Vec::new();
        for collector in &collectors {
            let top = collector.top_docs();
            total_hits += top.total_hits;
            all_docs.extend(top.hits);
        }
        all_docs.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.doc.cmp(&b.doc)));
        let mut key_counts: HashMap<i64, usize> = HashMap::new();
        let mut hits = Vec::new();
        for sdk in all_docs {
            let count = key_counts.entry(sdk.key).or_insert(0);
            if *count < self.max_hits_per_key {
                hits.push(sdk);
                *count += 1;
                if hits.len() == self.num_hits {
                    break;
                }
            }
        }
        Ok(DiversifiedTopDocs { total_hits, hits })
    }
}

/// `FixedBitSet.ramBytesUsed()` as a 64-bit JVM with compressed references
/// reports it: the 24-byte object and the 16-byte header plus the words of
/// its `long[]`.
fn bitset_ram_bytes(bits: &FixedBitSet) -> i64 {
    let words = bits.len().div_ceil(64) as i64;
    24 + 16 + 8 * words
}

/// `MemoryAccountingBitsetCollector`: the matching documents as a bitset whose
/// growth is reported to a [`MemoryTracker`]. Documents are bits at their
/// global id; the set grows to the highest id seen, where Java grows to the
/// end of each leaf it visits.
pub struct MemoryAccountingBitsetCollector {
    tracker: Arc<dyn MemoryTracker + Send + Sync>,
    bit_set: FixedBitSet,
    /// `highestSetBit` (Lucene 10.6, branch_10_6, #16452): the highest
    /// document collected, `-1` before any.
    highest_set_bit: i64,
    error: Option<Error>,
}

impl MemoryAccountingBitsetCollector {
    /// `new MemoryAccountingBitsetCollector(tracker)`.
    ///
    /// # Errors
    /// What the tracker refuses.
    pub fn new(tracker: Arc<dyn MemoryTracker + Send + Sync>) -> Result<Self> {
        let bit_set = FixedBitSet::new(0);
        tracker.update_bytes(bitset_ram_bytes(&bit_set))?;
        Ok(Self {
            tracker,
            bit_set,
            highest_set_bit: -1,
            error: None,
        })
    }

    /// The bitset collected so far.
    pub fn bit_set(&self) -> &FixedBitSet {
        &self.bit_set
    }
}

impl ScoringCollector for MemoryAccountingBitsetCollector {
    fn collect(&mut self, doc_id: i32, _score: f32) {
        if self.error.is_some() {
            return;
        }
        let Ok(doc) = usize::try_from(doc_id) else {
            return;
        };
        if doc >= self.bit_set.len() {
            let before = bitset_ram_bytes(&self.bit_set);
            let old = std::mem::replace(&mut self.bit_set, FixedBitSet::new(0));
            self.bit_set = FixedBitSet::ensure_capacity(old, doc + 1);
            let after = bitset_ram_bytes(&self.bit_set);
            if let Err(e) = self.tracker.update_bytes(after - before) {
                self.error = Some(e);
                return;
            }
        }
        self.bit_set.set(doc);
        self.highest_set_bit = self.highest_set_bit.max(i64::from(doc_id));
    }

    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
}

/// `MemoryAccountingBitsetCollectorManager.Result`.
#[derive(Debug)]
pub struct MemoryAccountingBitsetResult {
    /// The union of the collectors' bitsets.
    pub bit_set: FixedBitSet,
    /// `tracker.getBytes()` after the union was allocated.
    pub total_bytes_used: i64,
}

/// `MemoryAccountingBitsetCollectorManager`: its result is sized to the
/// highest matched document plus one, at least one bit (Lucene 10.6,
/// branch_10_6, #16452; 10.5 sized it to the last visited leaf's end).
pub struct MemoryAccountingBitsetCollectorManager {
    tracker: Arc<dyn MemoryTracker + Send + Sync>,
}

impl MemoryAccountingBitsetCollectorManager {
    /// `new MemoryAccountingBitsetCollectorManager(tracker)`.
    pub fn new(tracker: Arc<dyn MemoryTracker + Send + Sync>) -> Self {
        Self { tracker }
    }
}

impl CollectorManager for MemoryAccountingBitsetCollectorManager {
    type Collector = MemoryAccountingBitsetCollector;
    type Output = MemoryAccountingBitsetResult;

    fn new_collector(&self) -> Result<Self::Collector> {
        MemoryAccountingBitsetCollector::new(Arc::clone(&self.tracker))
    }

    fn reduce(&self, mut collectors: Vec<Self::Collector>) -> Result<Self::Output> {
        // Sized to just cover the highest matched document across all
        // collectors, and at least one bit so that iterating an empty result
        // from 0 stays safe.
        let mut result_size = 0usize;
        for collector in &mut collectors {
            if let Some(e) = collector.error.take() {
                return Err(e);
            }
            if let Ok(last) = usize::try_from(collector.highest_set_bit) {
                result_size = result_size.max(last.saturating_add(1));
            }
        }
        let mut result = FixedBitSet::new(result_size.max(1));
        self.tracker.update_bytes(bitset_ram_bytes(&result))?;
        for collector in &collectors {
            if let Ok(last) = usize::try_from(collector.highest_set_bit) {
                let length = last.saturating_add(1);
                FixedBitSet::or_range(&collector.bit_set, 0, &mut result, 0, length);
            }
        }
        Ok(MemoryAccountingBitsetResult {
            bit_set: result,
            total_bytes_used: self.tracker.bytes(),
        })
    }
}

/// `HumanReadableQuery`: a query that prints a description before the query
/// it wraps, for debugging. It rewrites to the wrapped query and so is never
/// searched itself; Java's `createWeight` throws for the same reason.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanReadableQuery {
    inner: Clause,
    description: String,
}

impl HumanReadableQuery {
    /// `new HumanReadableQuery(in, description)`.
    pub fn new(inner: Clause, description: &str) -> Self {
        Self {
            inner,
            description: description.to_string(),
        }
    }

    /// `getWrappedQuery()`
    pub fn wrapped_query(&self) -> &Clause {
        &self.inner
    }

    /// `getDescription()`
    pub fn description(&self) -> &str {
        &self.description
    }

    /// `rewrite(searcher)`: the wrapped query.
    pub fn rewrite(&self) -> Clause {
        self.inner.clone()
    }

    /// `toString(field)`: the description, a colon and the wrapped query.
    pub fn to_string_field(&self, field: &str) -> String {
        format!(
            "{}:{}",
            self.description,
            crate::query_builder::to_query_string(Some(&self.inner), field)
        )
    }
}
