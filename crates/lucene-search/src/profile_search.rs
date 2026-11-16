//! A query run once more with a timer around every clause of its scorer tree, for the
//! `profile` option of a search ([`exec::profile`]).
//!
//! The run drives the same scorer tree the search does (the document-at-a-time one, which is
//! also what stock profiles: its `ProfileWeight` has no bulk scorer of its own), segment by
//! segment, in doc-base order, into the collector the search used: the shared top-docs
//! collector, which publishes the competitive score the scorers prune with, or a plain count
//! for a search that scores nothing.

use std::collections::HashMap;

use crate::collector::{ScoreMode, ScoringCollector};
use crate::exec;
use crate::field_norms::FieldNorms;
use crate::multi_segment::{
    global_boolean_stats, rewrite_points_ranges, search_leaves_shared_counting, OpenSegment,
};
use crate::query::{BooleanQuery, Clause};
use crate::{GlobalStats, Result, ScoreDoc};

pub use exec::profile::{clock, Guard, Kind, NodeReport, Session, Timer, TIMERS};

/// What the run collects.
#[derive(Debug, Clone, Copy)]
pub enum Collect {
    /// The best `top_n` by score, counting hits as `TopScoreDocCollectorManager` does under
    /// `total_hits_threshold` (the search's `track_total_hits`).
    TopScores {
        top_n: usize,
        total_hits_threshold: u64,
    },
    /// Every match, scoring nothing: a count, a sort by field, an aggregation.
    Matches,
}

/// What a profiled run found.
#[derive(Debug)]
pub struct Profiled {
    /// Each clause's timers over each segment it was built for.
    pub nodes: Vec<NodeReport>,
    /// The hits the run collected (what the search itself returns), best first.
    pub hits: Vec<ScoreDoc>,
    /// The documents the run counted.
    pub total: u64,
}

/// A counting collector that asks for no scores.
struct Count(u64);

impl ScoringCollector for Count {
    fn collect(&mut self, _doc_id: i32, _score: f32) {
        self.0 += 1;
    }

    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
}

/// The timers of `root`'s clauses over every segment, run as `collect` says.
///
/// `norms` is one entry per segment, as the search's own functions take it.
///
/// # Errors
/// What building or scoring a clause reports.
pub fn profile_query(
    segments: &[OpenSegment<'_>],
    root: &Clause,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    collect: Collect,
) -> Result<Profiled> {
    // The searcher's rewrite (point ranges), then the statistics of the whole reader.
    let (clause, stats_query) = match root {
        Clause::Boolean(b) => {
            let rewritten = rewrite_points_ranges(b, segments).unwrap_or_else(|| (**b).clone());
            (Clause::Boolean(Box::new(rewritten.clone())), rewritten)
        }
        other => {
            let wrapper = BooleanQuery {
                must: vec![other.clone()],
                ..Default::default()
            };
            let rewritten = rewrite_points_ranges(&wrapper, segments).unwrap_or(wrapper);
            let leaf = rewritten
                .must
                .first()
                .cloned()
                .unwrap_or_else(|| other.clone());
            (leaf, rewritten)
        }
    };
    let global = global_boolean_stats(segments, &stats_query)?;
    let mut count = Count(0);
    let session = Session::new();
    let _installed = Guard::install(&session);
    match collect {
        Collect::TopScores {
            top_n,
            total_hits_threshold,
        } if top_n > 0 => {
            let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
            let (hits, total) = search_leaves_shared_counting(
                &doc_bases,
                top_n,
                total_hits_threshold,
                |i, local| run_segment(&session, segments, i, norms, &global, &clause, local),
            )?;
            return Ok(Profiled {
                nodes: session.report(),
                hits,
                total: total.value,
            });
        }
        Collect::TopScores { .. } | Collect::Matches => {
            let mut order: Vec<usize> = (0..segments.len()).collect();
            order.sort_by_key(|&i| segments[i].doc_base);
            for i in order {
                run_segment(&session, segments, i, norms, &global, &clause, &mut count)?;
            }
        }
    }
    Ok(Profiled {
        nodes: session.report(),
        hits: Vec::new(),
        total: count.0,
    })
}

fn run_segment<C: ScoringCollector>(
    session: &Session,
    segments: &[OpenSegment<'_>],
    i: usize,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    global: &GlobalStats,
    clause: &Clause,
    collector: &mut C,
) -> Result<()> {
    let seg = &segments[i];
    session.set_segment(i);
    let ctx = exec::LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: seg.live_docs,
        points: seg.points,
        norms: norms.get(i).copied().flatten(),
        global: Some(global),
        max_doc: seg.max_doc,
        cache: seg.cache,
        reader: seg.reader,
        similarity: seg.similarity,
    };
    let mode = exec::Mode::of(collector);
    if let Some(mut bulk) = exec::profile::root(&ctx, clause, mode)? {
        exec::score_segment(&mut bulk, mode, seg.live_docs, collector, ctx.max_doc)?;
    }
    Ok(())
}
