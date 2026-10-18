//! `DrillSidewaysQuery`: the query a sequential drill-sideways search runs --
//! the base query and every dim's drill-down, scored together per segment by
//! a [`DrillSidewaysScorer`], which hands hits to the main collector and near
//! misses to the dims' sideways collectors.
//!
//! Java's query is searched through `IndexSearcher.search(query,
//! collectorManager)`, and its weight creates each segment's sideways
//! collectors (`managedDrillSidewaysCollectors`) as the search reaches the
//! segment; [`DrillSidewaysQuery::search`] is that search, the segments of
//! each slice in doc-base order, returning the main manager's result and the
//! sideways collectors of every segment.
//!
//! Differs: `toString`, `equals`/`hashCode` and `explain` are not ported (the
//! query is internal to [`crate::facets::DrillSideways`]). The searcher's
//! `QueryTimeout` is not applied (Java wraps the bulk scorer in a
//! `TimeLimitingBulkScorer`), and the slices are searched in turn.

use crate::collector::ScoreMode;
use crate::exec::{build::build, BoxScorer, LeafContext, Mode};
use crate::grouping::GroupingCollectorManager;
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::SegmentCollector;
use crate::query::{BooleanQuery, Clause};
use crate::{Error, Result};

use super::drill_sideways_scorer::{DocsAndCost, DrillSidewaysScorer, EmptyScorer};

/// `DrillSidewaysQuery`.
pub(crate) struct DrillSidewaysQuery {
    base_query: Clause,
    drill_down_queries: Vec<Clause>,
    score_sub_docs_at_once: bool,
}

/// The scorer mode a collector's `ScoreMode` asks for (`Mode::of`).
fn mode_of(score_mode: ScoreMode) -> Mode {
    match score_mode {
        ScoreMode::CompleteNoScores => Mode::NoScores,
        m if m.is_exhaustive() => Mode::Complete,
        _ => Mode::TopScores,
    }
}

impl DrillSidewaysQuery {
    /// `new DrillSidewaysQuery(baseQuery, managers, drillDownQueries,
    /// scoreSubDocsAtOnce)`.
    pub(crate) fn new(
        base_query: Clause,
        drill_down_queries: Vec<Clause>,
        score_sub_docs_at_once: bool,
    ) -> Self {
        DrillSidewaysQuery {
            base_query,
            drill_down_queries,
            score_sub_docs_at_once,
        }
    }

    /// `searcher.search(this, manager)`: per slice a main collector, per
    /// segment a sideways collector of each dim (`managedDrillSidewaysCollectors`,
    /// one list per segment, in search order).
    ///
    /// # Errors
    /// What the search, a collector or a reduction reports;
    /// [`Error::IllegalArgument`] when `sideways` does not have one manager
    /// per dim, or when there is no dim (Java's scorer would index past its
    /// empty dims array).
    #[allow(clippy::type_complexity)]
    pub(crate) fn search<'a, M, S>(
        &self,
        searcher: &IndexSearcher<'_, 'a>,
        manager: &M,
        sideways: &[S],
    ) -> Result<(M::Output, Vec<Vec<S::Collector>>)>
    where
        M: GroupingCollectorManager<'a>,
        S: GroupingCollectorManager<'a>,
    {
        if self.drill_down_queries.is_empty() {
            return Err(Error::IllegalArgument(
                "a drill-sideways query needs at least one dim".to_string(),
            ));
        }
        if sideways.len() != self.drill_down_queries.len() {
            return Err(Error::IllegalArgument(format!(
                "{} sideways collector managers for {} dims",
                sideways.len(),
                self.drill_down_queries.len()
            )));
        }
        // `rewrite` and `createWeight`: the base and the drill-downs
        // rewritten, the statistics reader-wide.
        let whole = BooleanQuery {
            must: vec![self.base_query.clone()],
            filter: self.drill_down_queries.clone(),
            ..Default::default()
        };
        let (rewritten, global) = searcher.prepare(&whole)?;
        let whole = rewritten.unwrap_or(whole);
        let mut parts = whole
            .must
            .into_iter()
            .chain(whole.filter)
            .map(Clause::rewrite);
        // The rewrite keeps each clause in its place: the base is the first.
        let base = parts.next().unwrap_or_else(|| self.base_query.clone());
        let dims: Vec<Clause> = parts.collect();

        let similarity = searcher.similarity().filter(|s| !s.is_default_bm25());
        let segments = searcher.segments();
        let default_slices = [Vec::new()];
        let slices = if searcher.slices().is_empty() {
            &default_slices[..]
        } else {
            searcher.slices()
        };
        let mut mains = Vec::with_capacity(slices.len());
        let mut managed: Vec<Vec<S::Collector>> = Vec::new();
        for slice in slices {
            let mut main = manager.new_collector()?;
            let mode = mode_of(main.score_mode());
            let mut order = slice.clone();
            order.sort_by_key(|&i| segments.get(i).map_or(i32::MAX, |s| s.doc_base));
            for i in order {
                let seg = segments.get(i).ok_or(Error::SliceOutOfRange {
                    segment: i,
                    segments: segments.len(),
                })?;
                let max_doc = seg
                    .max_doc
                    .or(seg.reader.map(|r| r.max_doc))
                    .ok_or_else(|| Error::MissingSegmentReader("DrillSidewaysQuery".into()))?;
                // `searchLeaf`: the leaf collector, then the scorer supplier.
                main.set_next_reader(i, seg)?;
                let ctx = LeafContext {
                    fields: seg.fields,
                    doc_in: seg.doc_in,
                    pos_in: seg.pos_in,
                    pay_in: seg.pay_in,
                    live_docs: seg.live_docs,
                    points: seg.points,
                    norms: searcher.norms(i),
                    global: Some(&global),
                    max_doc: Some(max_doc),
                    cache: seg.cache,
                    reader: seg.reader,
                    similarity,
                };
                let base_scorer = build(&ctx, &base, 1.0, mode, false)?;
                let mut collectors: Vec<S::Collector> = Vec::with_capacity(dims.len());
                let mut docs_and_costs: Vec<DocsAndCost<'_>> = Vec::with_capacity(dims.len());
                let mut null_count = 0usize;
                for (dim, (query, m)) in dims.iter().zip(sideways).enumerate() {
                    let scorer: BoxScorer<'_> =
                        match build(&ctx, query, 1.0, Mode::NoScores, false)? {
                            Some(s) => s,
                            None => {
                                null_count = null_count.saturating_add(1);
                                Box::new(EmptyScorer)
                            }
                        };
                    let mut c = m.new_collector()?;
                    c.set_next_reader(i, seg)?;
                    collectors.push(c);
                    docs_and_costs.push(DocsAndCost::new(scorer, dim));
                }
                match base_scorer {
                    Some(base_scorer) if null_count <= 1 => {
                        // Stable, as `Arrays.sort` of objects is.
                        docs_and_costs.sort_by_key(|d| d.cost);
                        let mut scorer = DrillSidewaysScorer::new(
                            max_doc,
                            base_scorer,
                            docs_and_costs,
                            self.score_sub_docs_at_once,
                            mode.needs_scores(),
                        );
                        scorer.score(&mut main, &mut collectors, seg.live_docs, 0, i32::MAX)?;
                    }
                    _ => {
                        for c in &mut collectors {
                            c.finish()?;
                        }
                    }
                }
                managed.push(collectors);
                main.finish()?;
            }
            mains.push(main);
        }
        Ok((manager.reduce(mains)?, managed))
    }
}

#[cfg(test)]
mod tests;
