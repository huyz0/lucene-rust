use super::*;

// Required plus optional clauses, every match: `ReqOptSumScorer`, a batch at a time.

/// `ReqOptSumScorer` over term legs outside `TOP_SCORES`, as a bulk scorer.
/// Lucene has none for this shape -- `BooleanScorerSupplier.booleanScorer()`
/// declines `MUST` + `SHOULD` and `DefaultBulkScorer` drives
/// `ReqOptSumScorer` a document at a time -- so this is the conjunction bulk
/// scorer's batch loop with the optional clauses applied the way
/// `ReqOptSumScorer.score` applies them:
///
/// - the required legs' scores summed in `double` and narrowed
///   (`ConjunctionScorer.score`; a `FILTER` leg contributes `0`),
/// - the optional legs' scores summed in `double` and narrowed
///   (`DisjunctionSumScorer.score`),
/// - the two added in `float` (`score += optScorer.score()`).
///
/// Outside `TOP_SCORES` `ReqOptSumScorer` iterates its required clause and
/// skips nothing, so neither does this. Under `TOP_SCORES` the dispatch runs
/// the scorer tree instead: the lower-bound total a collector reports is the
/// number of documents `ReqOptSumScorer`'s impacts approximation hands it,
/// and only that iteration reproduces the count.
pub(crate) struct ReqOptBulk {
    single: DocScores,
    acc: DocScoreAcc,
    opt_sums: Vec<f64>,
    opt_hit: Vec<bool>,
}

impl ReqOptBulk {
    pub(crate) fn new() -> Self {
        Self {
            single: DocScores::default(),
            acc: DocScoreAcc::default(),
            opt_sums: Vec::new(),
            opt_hit: Vec::new(),
        }
    }

    /// Collects the matches in `[min, max)`: documents every `req` leg
    /// matches (cheapest first; `req[0]` leads), each with every `opt` leg's
    /// score added where it matches too. Returns the next document to score.
    pub(crate) fn score<C: ScoringCollector + ?Sized>(
        &mut self,
        req: &mut [TermLeg<'_>],
        opt: &mut [TermLeg<'_>],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        debug_assert!(
            collector.pruning_threshold().is_none(),
            "ReqOptBulk is never built for TOP_SCORES"
        );
        if req[0].doc_id() < min {
            req[0].advance(min)?;
        }
        loop {
            req[0].next_docs_and_scores(max, live_docs, &mut self.single)?;
            if self.single.docs.is_empty() {
                return Ok(req[0].doc_id());
            }
            let acc = &mut self.acc;
            acc.copy_from(&self.single);
            for leg in req.iter_mut().skip(1) {
                apply_required_clause(acc, leg)?;
            }
            self.opt_sums.clear();
            self.opt_sums.resize(acc.docs.len(), 0.0);
            self.opt_hit.clear();
            self.opt_hit.resize(acc.docs.len(), false);
            for leg in opt.iter_mut() {
                let mut cur = leg.doc_id();
                for (k, &doc) in acc.docs.iter().enumerate() {
                    if cur < doc {
                        cur = leg.advance(doc)?;
                    }
                    if cur == doc {
                        self.opt_sums[k] += leg.score()? as f64;
                        self.opt_hit[k] = true;
                    }
                }
            }
            for (k, &doc) in acc.docs.iter().enumerate() {
                let mut score = acc.scores[k] as f32;
                if self.opt_hit[k] {
                    score += self.opt_sums[k] as f32;
                }
                collector.collect(doc, score);
            }
        }
    }
}
