//! `org.apache.lucene.sandbox.facet`: faceting as a collector -- a
//! [`FacetCutter`] maps each matching document to facet ordinals, a
//! [`FacetRecorder`] records something per ordinal (a count, aggregated
//! values), and the ordinals are then read back through
//! [`OrdinalIterator`]s, sorted, cut to the top `n` and labelled.
//!
//! - [`FacetFieldCollector`] / [`FacetFieldCollectorManager`]: the
//!   collector, its leaf collector (`FacetFieldLeafCollector`) and manager.
//! - [`cutters`]: taxonomy ordinals, distinct long values; [`ranges`]: long
//!   and double ranges.
//! - [`recorders`]: counts, long aggregations, several recorders at once.
//! - [`iterators`] and [`comparable`]: filtering, top-`n` and sorting
//!   ordinals; [`labels`]: ordinal to label and back.
//! - [`builders`]: `FacetBuilder`s and the two orchestrators;
//!   [`post_collection`]: `PostCollectionFaceting`.
//! - [`histogram`]: `plain.histograms`' `HistogramCollectorManager`.
//!
//! Lifetimes rather than shared mutable objects: the Rust cutters and
//! recorders are shared through `Arc`, each recorder's per-segment state is
//! owned by its leaf recorder while a segment is collected (no lock on the
//! hot path) and handed back when the segment is finished, and a cutter's
//! leaf cutter reads the segment through the collector's
//! [`SegmentReader`]. The hash maps iterate in a fixed order where Java's
//! hppc maps iterate in a per-process one; every output Java sorts is sorted
//! here the same way.

pub mod builders;
pub mod comparable;
pub mod cutters;
pub mod histogram;
pub mod iterators;
pub mod labels;
pub mod post_collection;
pub mod ranges;
pub mod recorders;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use crate::collector::ScoreMode;
use crate::directory_reader::SegmentReader;
use crate::grouping::GroupingCollectorManager;
use crate::leaf_collector::SegmentCollector;
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

/// `OrdinalIterator.NO_MORE_ORDS`.
pub const NO_MORE_ORDS: i32 = -1;

/// `OrdinalIterator`: facet ordinals, one at a time, until
/// [`NO_MORE_ORDS`].
pub trait OrdinalIterator {
    /// `nextOrd()`.
    ///
    /// # Errors
    /// Whatever reading the ordinals' source reports.
    fn next_ord(&mut self) -> Result<i32>;

    /// `toArray()`: every remaining ordinal.
    ///
    /// # Errors
    /// As [`next_ord`](Self::next_ord).
    fn to_array(&mut self) -> Result<Vec<i32>> {
        let mut out = Vec::new();
        loop {
            let ord = self.next_ord()?;
            if ord == NO_MORE_ORDS {
                return Ok(out);
            }
            out.push(ord);
        }
    }
}

impl<T: OrdinalIterator + ?Sized> OrdinalIterator for Box<T> {
    fn next_ord(&mut self) -> Result<i32> {
        (**self).next_ord()
    }
}

/// `OrdinalIterator.fromArray(source)`: the array's ordinals, skipping
/// [`NO_MORE_ORDS`] entries.
#[derive(Debug, Clone)]
pub struct ArrayOrdinalIterator {
    source: Vec<i32>,
    cursor: usize,
}

impl ArrayOrdinalIterator {
    pub fn new(source: Vec<i32>) -> Self {
        ArrayOrdinalIterator { source, cursor: 0 }
    }
}

impl OrdinalIterator for ArrayOrdinalIterator {
    fn next_ord(&mut self) -> Result<i32> {
        while let Some(&ord) = self.source.get(self.cursor) {
            self.cursor = self.cursor.saturating_add(1);
            // NO_MORE_ORDS is returned only after the whole array is read.
            if ord != NO_MORE_ORDS {
                return Ok(ord);
            }
        }
        Ok(NO_MORE_ORDS)
    }
}

/// `OrdinalIterator.fromArray(source)`.
pub fn from_array(source: Vec<i32>) -> ArrayOrdinalIterator {
    ArrayOrdinalIterator::new(source)
}

/// `OrdinalIterator.EMPTY`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EmptyOrdinalIterator;

impl OrdinalIterator for EmptyOrdinalIterator {
    fn next_ord(&mut self) -> Result<i32> {
        Ok(NO_MORE_ORDS)
    }
}

/// A boxed ordinal iterator.
pub type BoxOrdinalIterator<'x> = Box<dyn OrdinalIterator + 'x>;

/// `LeafFacetCutter`: a segment's cutter -- [`advance_exact`](Self::advance_exact)
/// to a document, then its ordinals through
/// [`OrdinalIterator::next_ord`].
pub trait LeafFacetCutter: OrdinalIterator {
    /// `advanceExact(doc)`: whether `doc` has any ordinal.
    ///
    /// # Errors
    /// Whatever reading the segment reports.
    fn advance_exact(&mut self, doc: i32) -> Result<bool>;
}

/// `FacetCutter`: maps documents to facet ordinals.
pub trait FacetCutter {
    /// `createLeafCutter(context)`: the cutter of `reader`'s segment.
    ///
    /// # Errors
    /// Whatever opening the segment's values reports.
    fn create_leaf_cutter<'a>(
        &self,
        reader: &'a SegmentReader,
    ) -> Result<Box<dyn LeafFacetCutter + 'a>>;

    /// `getOrdinalsToRollup()`: the ordinals whose descendants' values a
    /// recorder rolls up into them; `None` (the default) when nothing rolls
    /// up.
    ///
    /// # Errors
    /// Whatever looking the ordinals up reports.
    fn ordinals_to_rollup(&self) -> Result<Option<BoxOrdinalIterator<'_>>> {
        Ok(None)
    }

    /// `getChildrenOrds(ord)`: `ord`'s children; `None` (the default) when
    /// the cutter has no hierarchy.
    ///
    /// # Errors
    /// Whatever reading the hierarchy reports.
    fn children_ords(&self, ord: i32) -> Result<Option<BoxOrdinalIterator<'_>>> {
        let _ = ord;
        Ok(None)
    }
}

/// `LeafFacetRecorder`: a segment's recorder.
pub trait LeafFacetRecorder {
    /// `record(docId, facetOrd)`.
    ///
    /// # Errors
    /// Whatever reading the segment's values reports, or an aggregation's
    /// overflow.
    fn record(&mut self, doc: i32, ord: i32) -> Result<()>;

    /// The segment is collected: hands what was recorded back to the
    /// [`FacetRecorder`] (Java's leaf recorder writes into a map its
    /// recorder already holds).
    ///
    /// # Errors
    /// A recorder whose shared state is poisoned.
    fn finish(self: Box<Self>) -> Result<()>;
}

/// `FacetRecorder`: records values per facet ordinal.
pub trait FacetRecorder {
    /// `getLeafRecorder(context)`.
    ///
    /// # Errors
    /// Whatever opening the segment's values reports.
    fn get_leaf_recorder<'a>(
        &self,
        reader: &'a SegmentReader,
    ) -> Result<Box<dyn LeafFacetRecorder + 'a>>;

    /// `recordedOrds()`: every ordinal with a value, after
    /// [`reduce`](Self::reduce).
    ///
    /// # Errors
    /// [`Error::Unsupported`] where Java throws
    /// `UnsupportedOperationException`; [`Error::IllegalState`] before
    /// `reduce`.
    fn recorded_ords(&self) -> Result<BoxOrdinalIterator<'static>>;

    /// `isEmpty()`.
    ///
    /// # Errors
    /// As [`recorded_ords`](Self::recorded_ords).
    fn is_empty(&self) -> Result<bool>;

    /// `reduce(facetCutter)`: merges the segments' records, then rolls up
    /// the cutter's [`ordinals_to_rollup`](FacetCutter::ordinals_to_rollup).
    ///
    /// # Errors
    /// Whatever the cutter reports, or an aggregation's overflow.
    fn reduce(&self, cutter: &dyn FacetCutter) -> Result<()>;

    /// `contains(ordinal)`.
    ///
    /// # Errors
    /// As [`recorded_ords`](Self::recorded_ords).
    fn contains(&self, ord: i32) -> Result<bool>;
}

/// `FacetFieldCollector` and its `FacetFieldLeafCollector`: per segment,
/// the leaf cutter and leaf recorder are created at the segment's first
/// hit, and every hit's ordinals are recorded.
pub struct FacetFieldCollector<'t, 'a> {
    cutter: Arc<dyn FacetCutter + 't>,
    recorder: Arc<dyn FacetRecorder + 't>,
    reader: Option<&'a SegmentReader>,
    leaf: Option<(
        Box<dyn LeafFacetCutter + 'a>,
        Box<dyn LeafFacetRecorder + 'a>,
    )>,
}

impl<'t, 'a> FacetFieldCollector<'t, 'a> {
    /// `new FacetFieldCollector(facetCutter, facetRecorder)`.
    pub fn new(cutter: Arc<dyn FacetCutter + 't>, recorder: Arc<dyn FacetRecorder + 't>) -> Self {
        FacetFieldCollector {
            cutter,
            recorder,
            reader: None,
            leaf: None,
        }
    }

    /// Moves to `reader`'s segment (`getLeafCollector(context)`), handing
    /// the previous segment's records back.
    pub(crate) fn enter(&mut self, reader: Option<&'a SegmentReader>) -> Result<()> {
        self.leave()?;
        self.reader = reader;
        Ok(())
    }

    /// The current segment is collected.
    pub(crate) fn leave(&mut self) -> Result<()> {
        self.reader = None;
        match self.leaf.take() {
            Some((_, recorder)) => recorder.finish(),
            None => Ok(()),
        }
    }

    /// `collect(doc)`.
    pub(crate) fn collect_doc(&mut self, doc: i32) -> Result<()> {
        if self.leaf.is_none() {
            let reader = self
                .reader
                .ok_or_else(|| Error::MissingSegmentReader("FacetFieldCollector".into()))?;
            let cutter = self.cutter.create_leaf_cutter(reader)?;
            let recorder = self.recorder.get_leaf_recorder(reader)?;
            self.leaf = Some((cutter, recorder));
        }
        let Some((cutter, recorder)) = self.leaf.as_mut() else {
            return Ok(());
        };
        if cutter.advance_exact(doc)? {
            loop {
                let ord = cutter.next_ord()?;
                if ord == NO_MORE_ORDS {
                    break;
                }
                recorder.record(doc, ord)?;
            }
        }
        Ok(())
    }
}

impl<'a> SegmentCollector<'a> for FacetFieldCollector<'_, 'a> {
    /// `scoreMode()`: `COMPLETE_NO_SCORES`.
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }

    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.enter(leaf.reader)
    }

    fn collect(&mut self, doc: i32, _score: f32) -> Result<()> {
        self.collect_doc(doc)
    }

    fn finish(&mut self) -> Result<()> {
        self.leave()
    }
}

/// `FacetFieldCollectorManager<V>`: collectors sharing one cutter and one
/// recorder; `reduce` reduces the recorder and returns it.
pub struct FacetFieldCollectorManager<'t, R: FacetRecorder + 't> {
    cutter: Arc<dyn FacetCutter + 't>,
    recorder: Arc<R>,
}

impl<'t, R: FacetRecorder + 't> FacetFieldCollectorManager<'t, R> {
    /// `new FacetFieldCollectorManager(facetCutter, facetRecorder)`.
    pub fn new(cutter: Arc<dyn FacetCutter + 't>, recorder: Arc<R>) -> Self {
        FacetFieldCollectorManager { cutter, recorder }
    }

    /// `newCollector()`.
    pub fn new_facet_collector<'a>(&self) -> FacetFieldCollector<'t, 'a> {
        let recorder: Arc<dyn FacetRecorder + 't> = self.recorder.clone();
        FacetFieldCollector::new(self.cutter.clone(), recorder)
    }

    /// `reduce(collectors)`: each collector's last segment handed back, then
    /// the recorder reduced.
    ///
    /// # Errors
    /// What [`FacetRecorder::reduce`] reports.
    pub fn reduce_collectors(
        &self,
        collectors: Vec<FacetFieldCollector<'t, '_>>,
    ) -> Result<Arc<R>> {
        for mut c in collectors {
            c.leave()?;
        }
        self.recorder.reduce(self.cutter.as_ref())?;
        Ok(self.recorder.clone())
    }

    /// The recorder.
    pub fn recorder(&self) -> &Arc<R> {
        &self.recorder
    }

    /// The cutter.
    pub fn cutter(&self) -> &Arc<dyn FacetCutter + 't> {
        &self.cutter
    }
}

impl<'t, 'a, R: FacetRecorder + 't> GroupingCollectorManager<'a>
    for FacetFieldCollectorManager<'t, R>
{
    type Collector = FacetFieldCollector<'t, 'a>;
    type Output = Arc<R>;

    fn new_collector(&self) -> Result<Self::Collector> {
        Ok(self.new_facet_collector())
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Arc<R>> {
        self.reduce_collectors(collectors)
    }
}

/// MurmurHash3's `fmix64` finalizer as a [`Hasher`] (in place of hppc's
/// maps' own mixing): the facet maps' keys are small dense integers, which
/// SipHash would only slow down.
#[derive(Default, Clone, Copy)]
pub(crate) struct MixHasher(u64);

impl Hasher for MixHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = mix64(self.0 ^ u64::from(b));
        }
    }

    fn write_i32(&mut self, v: i32) {
        self.0 = mix64(u64::from(v as u32));
    }

    fn write_i64(&mut self, v: i64) {
        self.0 = mix64(v as u64);
    }
}

/// MurmurHash3's 64-bit finalizer.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    z = (z ^ (z >> 33)).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    z ^ (z >> 33)
}

/// A map keyed by ordinal or value.
pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<MixHasher>>;

/// The shared per-segment record slots of a recorder: a slot is reserved
/// when a segment's leaf recorder is created (Java's
/// `perLeafValues.add(leafValues)`), and filled when it finishes.
pub(crate) struct LeafSlots<T>(std::sync::Mutex<Vec<Option<T>>>);

impl<T> Default for LeafSlots<T> {
    fn default() -> Self {
        LeafSlots(std::sync::Mutex::new(Vec::new()))
    }
}

impl<T> LeafSlots<T> {
    pub(crate) fn reserve(&self) -> Result<usize> {
        let mut slots = self.lock()?;
        slots.push(None);
        Ok(slots.len().saturating_sub(1))
    }

    pub(crate) fn fill(&self, slot: usize, value: T) -> Result<()> {
        if let Some(s) = self.lock()?.get_mut(slot) {
            *s = Some(value);
        }
        Ok(())
    }

    /// The filled slots, in reservation order.
    pub(crate) fn take_all(&self) -> Result<Vec<T>> {
        Ok(self.lock()?.iter_mut().filter_map(Option::take).collect())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Vec<Option<T>>>> {
        self.0
            .lock()
            .map_err(|_| Error::IllegalState("facet recorder state poisoned".into()))
    }
}
