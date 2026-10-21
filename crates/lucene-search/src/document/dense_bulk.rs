//! How `ConstantScoreScorerSupplier.bulkScorer` scores a constant-score
//! doc-values query, as far as which candidates are confirmed and collected
//! and when the search stops -- which decides `TopDocs.totalHits` once the
//! collector saturates -- and `SkipBlockRangeIterator`, the skip-index
//! approximation those queries iterate.
//!
//! Shared by the core doc-values range queries and lucene-sandbox's
//! multi-range ones, which Java scores through the same supplier.

use lucene_codecs::doc_values::{DocValuesSkipIndex, DocValuesSkipper, NO_MORE_DOCS};

use crate::collector::ScoringCollector;
use crate::document::collect_live;
use crate::multi_segment::OpenSegment;

/// `DenseConjunctionBulkScorer.WINDOW_SIZE`.
const WINDOW_SIZE: i32 = 4096;
/// `DenseConjunctionBulkScorer.DENSITY_THRESHOLD_INVERSE`.
const DENSITY_THRESHOLD_INVERSE: i32 = 32;

/// Whether the collector can no longer take a document scoring `score`:
/// the bulk scorers' `minCompetitiveScore > score`.
pub(crate) fn saturated(collector: &dyn ScoringCollector, score: f32) -> bool {
    crate::bulk_scorer::min_competitive_score(collector) > score
}

/// What the bulk scorer does with a candidate.
pub(crate) enum Step {
    Stop,
    Skip,
    Check,
}

/// The bulk scorer `ConstantScoreScorerSupplier.bulkScorer` picks, and its
/// candidate-by-candidate behaviour:
///
/// * `DenseConjunctionBulkScorer` (a segment of at least 2048 documents whose
///   approximation costs at least `maxDoc / 32`): windows of 4096 documents
///   from a candidate on -- shorter where the clause reports a run of
///   matches (`docIDRunEnd`) ending inside the window's second half, the
///   whole run when there are no deletions -- every candidate confirmed
///   before the live documents are masked, the collector's threshold
///   checked only between windows;
/// * `ConstantScoreBulkScorer` (no scores needed): every candidate
///   confirmed, then masked; nothing to stop for;
/// * `DefaultBulkScorer` (otherwise, and [`Bulk::default_scorer`]): only
///   live candidates confirmed, the search ending at the hit that saturates
///   the collector.
pub(crate) struct Bulk {
    windows: bool,
    live_first: bool,
    window_end: i32,
    max_doc: i32,
}

impl Bulk {
    pub(crate) fn new(needs_scores: bool, max_doc: i32, cost: i64) -> Self {
        let windows =
            max_doc >= WINDOW_SIZE / 2 && cost >= i64::from(max_doc / DENSITY_THRESHOLD_INVERSE);
        Bulk {
            windows,
            live_first: needs_scores && !windows,
            window_end: -1,
            max_doc,
        }
    }

    /// `DefaultBulkScorer`, what a plain `ScorerSupplier` (or a
    /// non-exhaustive `ConstantScoreQuery`) scores with: live candidates
    /// confirmed, the search ending at the hit that saturates the collector.
    pub(crate) fn default_scorer(max_doc: i32) -> Self {
        Bulk {
            windows: false,
            live_first: true,
            window_end: -1,
            max_doc,
        }
    }

    /// A candidate whose clause reports no run (`docIDRunEnd() == doc + 1`).
    pub(crate) fn candidate(
        &mut self,
        leaf: &OpenSegment<'_>,
        doc: i32,
        boost: f32,
        collector: &dyn ScoringCollector,
    ) -> Step {
        self.candidate_run(leaf, doc, doc.saturating_add(1), boost, collector)
    }

    /// A candidate whose clause matches every document of `doc..run_end`
    /// (`docIDRunEnd()`): `scoreWindow` leaves such a clause out of the
    /// window when the run reaches the window's second half, and the window
    /// then ends where the run does.
    pub(crate) fn candidate_run(
        &mut self,
        leaf: &OpenSegment<'_>,
        doc: i32,
        run_end: i32,
        boost: f32,
        collector: &dyn ScoringCollector,
    ) -> Step {
        if self.windows && doc >= self.window_end {
            if saturated(collector, boost) {
                return Step::Stop;
            }
            let threshold = doc.saturating_add(WINDOW_SIZE / 2).min(self.max_doc);
            let full = doc.saturating_add(WINDOW_SIZE);
            self.window_end = if run_end < threshold {
                full
            } else if leaf.live_docs.is_none() {
                // `collectRange(min, minDocIDRunEnd)`: the whole run.
                run_end.min(self.max_doc)
            } else {
                run_end.min(full)
            };
        }
        if self.live_first && leaf.live_docs.is_some_and(|b| !b.get_doc(doc)) {
            return Step::Skip;
        }
        Step::Check
    }

    /// Candidates `first..=last` that cannot match: only a window starting
    /// among them checks the collector. Whether to stop.
    pub(crate) fn pass(
        &mut self,
        first: i32,
        last: i32,
        boost: f32,
        collector: &dyn ScoringCollector,
    ) -> bool {
        if !self.windows {
            return false;
        }
        loop {
            let start = first.max(self.window_end);
            if start > last {
                return false;
            }
            if saturated(collector, boost) {
                return true;
            }
            self.window_end = start.saturating_add(WINDOW_SIZE);
        }
    }

    /// Collects a confirmed candidate if it is live; whether to stop.
    pub(crate) fn hit(
        &self,
        leaf: &OpenSegment<'_>,
        doc: i32,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> bool {
        collect_live(leaf, doc, boost, collector);
        !self.windows && saturated(collector, boost)
    }
}

/// `SkipBlockRangeIterator`: every document of the skip blocks whose value
/// range meets `[min, max]`.
pub(crate) struct SkipBlocks<'a> {
    skipper: DocValuesSkipper<'a>,
    min: i64,
    max: i64,
    doc: i32,
}

impl<'a> SkipBlocks<'a> {
    pub(crate) fn new(index: &'a DocValuesSkipIndex, min: i64, max: i64) -> Self {
        SkipBlocks {
            skipper: DocValuesSkipper::new(index),
            min,
            max,
            doc: -1,
        }
    }

    /// The current level-0 block: its last document and its value range.
    pub(crate) fn block(&self) -> (i32, i64, i64) {
        (
            self.skipper.max_doc_id(0),
            self.skipper.min_value(0),
            self.skipper.max_value(0),
        )
    }

    /// Moves to `doc`, a document of the current block.
    pub(crate) fn skip_to(&mut self, doc: i32) {
        self.doc = doc;
    }

    /// `nextDoc()`: `advance(doc + 1)`.
    pub(crate) fn next_doc(&mut self) -> i32 {
        let target = self.doc.saturating_add(1);
        if target <= self.skipper.max_doc_id(0) {
            if self.doc > -1 {
                self.doc = target;
                return target;
            }
        } else {
            self.skipper.advance(target);
        }
        self.skipper.advance_range(self.min, self.max);
        self.doc = target.max(self.skipper.min_doc_id(0));
        self.doc
    }

    /// Whether level `level`'s interval is `YES`: dense, every value in range.
    fn yes(&self, level: usize) -> bool {
        self.skipper.min_value(level) >= self.min
            && self.skipper.max_value(level) <= self.max
            && self
                .skipper
                .max_doc_id(level)
                .checked_sub(self.skipper.min_doc_id(level))
                == self.skipper.doc_count(level).checked_sub(1)
    }

    /// `docIDRunEnd()`: `doc + 1` unless the block is `YES`, then the end of
    /// the coarsest enclosing `YES` interval.
    pub(crate) fn doc_id_run_end(&self) -> i32 {
        // `classifyBlock`, which `advance` caches as `match`; the exhausted
        // iterator's `MAYBE` is `doc + 1` too.
        if self.doc == NO_MORE_DOCS || !self.yes(0) {
            return self.doc.saturating_add(1);
        }
        let mut max_doc = self.skipper.max_doc_id(0);
        let mut level = 1;
        while level < self.skipper.num_levels() && self.yes(level) {
            max_doc = self.skipper.max_doc_id(level);
            level = level.saturating_add(1);
        }
        max_doc.saturating_add(1)
    }
}
