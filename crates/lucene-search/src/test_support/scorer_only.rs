//! Operational `ScorerIndexSearcher` leaf dispatch, behind the asserting feature.
//!
//! The selection belongs to the searcher, so executor slices observe it too.
//! Unlike a thread-local guard, it also routes the default BM25 top-hits entry
//! point through the collector loop. Only the top-level bulk scorer is replaced;
//! the query's ordinary scorer tree and its two-phase iterators stay unchanged.

use crate::collector::ScoringCollector;
use crate::exec::{self, Bulk, LeafContext, Mode};
use crate::index_searcher::{IndexSearcher, SegmentNorms};
use crate::multi_segment::OpenSegment;
use crate::query::BooleanQuery;
use crate::reader::exitable::QueryTimeout;
use crate::similarities::Similarity;
use crate::{GlobalStats, Result};

/// Configures an owned searcher to use `DefaultBulkScorer` over the ordinary
/// top-level scorer for its collector, top-hit, count and manager entry points.
/// This does not configure the separately implemented field-sort/facet APIs.
/// Java: `ScorerIndexSearcher.searchLeaf`.
pub fn enable<'s, 'a>(mut searcher: IndexSearcher<'s, 'a>) -> IndexSearcher<'s, 'a> {
    searcher.scorer_only = true;
    searcher
}

/// Whether the searcher's leaf dispatch uses the scorer API.
pub fn enabled(searcher: &IndexSearcher<'_, '_>) -> bool {
    searcher.scorer_only
}

/// Java: `ScorerIndexSearcher.searchLeaf`: obtain the ordinary scorer through
/// `get(Long.MAX_VALUE)` and let the default bulk scorer drive it. Returns
/// whether the configured query timeout stopped this leaf.
pub(crate) fn score_leaf<C: ScoringCollector + ?Sized>(
    seg: &OpenSegment<'_>,
    query: &BooleanQuery,
    norms: SegmentNorms<'_, '_>,
    global: &GlobalStats,
    similarity: Option<&dyn Similarity>,
    collector: &mut C,
    timeout: Option<&dyn QueryTimeout>,
) -> Result<bool> {
    let ctx = LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: seg.live_docs,
        points: seg.points,
        norms,
        global: Some(global),
        max_doc: seg.max_doc,
        cache: seg.cache,
        reader: seg.reader,
        similarity,
    };
    let mode = Mode::of(collector);
    // The supplier wrapper does not forward setTopLevelScoringClause.
    let Some(scorer) = exec::build::build_boolean_root(&ctx, query, 1.0, mode, false)? else {
        return Ok(false);
    };
    let mut bulk = Bulk::scorer(scorer);
    if let Some(timeout) = timeout {
        exec::score_segment_time_limited(
            &mut bulk,
            mode,
            seg.live_docs,
            collector,
            seg.max_doc,
            || timeout.should_exit(),
        )
    } else {
        exec::score_segment(&mut bulk, mode, seg.live_docs, collector, seg.max_doc)?;
        Ok(false)
    }
}
