//! Collectors that read each segment as they collect it: Java's
//! `Collector.getLeafCollector(context)` / `LeafCollector.collect(doc)` /
//! `LeafCollector.finish()`, for the collectors that open a leaf's doc values
//! when the search reaches it (the query-time join collectors of
//! [`crate::join`], the grouping collectors of [`crate::grouping`]).
//!
//! This crate's [`ScoringCollector`] sees reader-wide document ids only. A
//! [`SegmentCollector`] is driven through [`PerSegment`], a
//! [`ScoringCollector`] over the searcher's segments that turns each global
//! id back into its segment and segment-local id, calling
//! [`SegmentCollector::set_next_reader`] when the search enters a segment and
//! [`SegmentCollector::finish`] when it leaves it -- for **every** segment,
//! matching or not, in doc-base order, as `IndexSearcher.searchLeaf` calls
//! `getLeafCollector` and `finish` for every leaf whether or not its weight
//! has a scorer there.
//!
//! `collect` here returns a `Result` (Java's `IOException`); the first error
//! stops collection and is what [`search_segments`] returns.

use crate::collector::{ScoreMode, ScoringCollector};
use crate::index_searcher::IndexSearcher;
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::{Error, Result};

/// A collector that reads each segment it collects (`Collector` with its
/// `LeafCollector`s).
pub trait SegmentCollector<'a> {
    /// `Collector.scoreMode()`.
    fn score_mode(&self) -> ScoreMode;
    /// `getLeafCollector(context)`: the search enters segment `ord` of the
    /// searcher (`context.ord`), whose documents start at `leaf.doc_base`.
    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()>;
    /// `LeafCollector.collect(doc)` for a segment-local `doc`, with its
    /// score (`Scorable.score()`; meaningless under
    /// [`ScoreMode::CompleteNoScores`]).
    fn collect(&mut self, doc: i32, score: f32) -> Result<()>;
    /// `LeafCollector.finish()`: the search leaves the current segment.
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

/// The [`ScoringCollector`] that drives a [`SegmentCollector`] over a
/// searcher's segments (see the module doc). Call [`PerSegment::close`]
/// after the search: it visits the segments the search never reached and
/// reports the first error.
pub struct PerSegment<'c, 's, 'a, C: ?Sized> {
    segments: &'s [OpenSegment<'a>],
    /// Segment indices by doc base.
    order: Vec<usize>,
    /// Position in `order` of the segment being collected; `None` before
    /// the first.
    at: Option<usize>,
    /// One past the current segment's last global document.
    end: i32,
    inner: &'c mut C,
    error: Option<Error>,
}

impl<'c, 's, 'a, C: SegmentCollector<'a> + ?Sized> PerSegment<'c, 's, 'a, C> {
    /// Drives `inner` over `segments`.
    pub fn new(segments: &'s [OpenSegment<'a>], inner: &'c mut C) -> Self {
        let order: Vec<usize> = (0..segments.len()).collect();
        Self::over(segments, &order, inner)
    }

    /// Drives `inner` over the segments `order` names (a slice of the
    /// searcher's), in doc-base order.
    pub fn over(segments: &'s [OpenSegment<'a>], order: &[usize], inner: &'c mut C) -> Self {
        let mut order: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&i| i < segments.len())
            .collect();
        order.sort_by_key(|&i| segments[i].doc_base);
        Self {
            segments,
            order,
            at: None,
            end: i32::MIN,
            inner,
            error: None,
        }
    }

    /// Leaves the current segment (if any) and enters the next one.
    fn advance(&mut self) -> Result<bool> {
        if self.at.is_some() {
            self.inner.finish()?;
        }
        let next = self.at.map_or(0, |a| a.saturating_add(1));
        let Some(&i) = self.order.get(next) else {
            self.at = Some(next);
            self.end = i32::MAX;
            return Ok(false);
        };
        self.at = Some(next);
        let seg = &self.segments[i];
        let max_doc = seg
            .max_doc
            .or(seg.reader.map(|r| r.max_doc))
            .ok_or_else(|| Error::MissingSegmentReader("a leaf collector".into()))?;
        self.end = seg.doc_base.saturating_add(max_doc);
        self.inner.set_next_reader(i, seg)?;
        Ok(true)
    }

    fn collect_global(&mut self, doc: i32, score: f32) -> Result<()> {
        while self.at.is_none() || doc >= self.end {
            if !self.advance()? {
                return Err(Error::IllegalArgument(format!(
                    "document {doc} is in no segment of the searcher"
                )));
            }
        }
        let base = self
            .at
            .and_then(|a| self.order.get(a))
            .map_or(0, |&i| self.segments[i].doc_base);
        self.inner.collect(doc.saturating_sub(base), score)
    }

    /// Visits every segment the search did not reach (entering and
    /// finishing each, as Java does for a leaf without matches) and returns
    /// the first error any call reported.
    pub fn close(mut self) -> Result<()> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        while self.at.is_none_or(|a| a < self.order.len()) {
            self.advance()?;
        }
        Ok(())
    }
}

impl<'a, C: SegmentCollector<'a> + ?Sized> ScoringCollector for PerSegment<'_, '_, 'a, C> {
    fn collect(&mut self, doc_id: i32, score: f32) {
        if self.error.is_some() {
            return;
        }
        if let Err(e) = self.collect_global(doc_id, score) {
            self.error = Some(e);
        }
    }

    fn score_mode(&self) -> ScoreMode {
        self.inner.score_mode()
    }
}

/// `searcher.search(query, collector)` for a [`SegmentCollector`]: every
/// segment in doc-base order, each document of a segment in order.
///
/// # Errors
/// Whatever the search or the collector reports, the first one.
pub fn search_segments<'a, C: SegmentCollector<'a> + ?Sized>(
    searcher: &IndexSearcher<'_, 'a>,
    query: &BooleanQuery,
    collector: &mut C,
) -> Result<()> {
    let mut driver = PerSegment::new(searcher.segments(), collector);
    searcher.search_collector(query, &mut driver)?;
    driver.close()
}

/// [`search_segments`] over one slice of the searcher's segments (a
/// `CollectorManager`'s collector's share).
///
/// # Errors
/// As [`search_segments`].
pub fn search_slice<'a, C: SegmentCollector<'a> + ?Sized>(
    searcher: &IndexSearcher<'_, 'a>,
    query: &BooleanQuery,
    slice: &[usize],
    collector: &mut C,
) -> Result<()> {
    let mut driver = PerSegment::over(searcher.segments(), slice, collector);
    searcher.search_slice_collector(query, slice, &mut driver)?;
    driver.close()
}

/// Every slice of the searcher in turn, a fresh collector from `new` for
/// each: what a `CollectorManager` search hands `reduce` (Java may run the
/// slices concurrently; the collectors are the same either way).
///
/// # Errors
/// As [`search_segments`], and what `new` reports.
pub fn search_slices<'a, C: SegmentCollector<'a>>(
    searcher: &IndexSearcher<'_, 'a>,
    query: &BooleanQuery,
    mut new: impl FnMut() -> Result<C>,
) -> Result<Vec<C>> {
    let mut out = Vec::new();
    for slice in searcher.slices() {
        let mut c = new()?;
        search_slice(searcher, query, slice, &mut c)?;
        out.push(c);
    }
    if out.is_empty() {
        out.push(new()?);
    }
    Ok(out)
}
