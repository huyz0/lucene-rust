//! Block-at-a-time scoring of term clauses: ports of Lucene 10.5.0's
//! `TermScorer.nextDocsAndScores` + `BatchScoreBulkScorer` (one term),
//! `BlockMaxConjunctionBulkScorer` (a conjunction of terms) and
//! `MaxScoreBulkScorer` (a disjunction of terms), with the `ImpactsDISI` /
//! `MaxScoreCache` skipping underneath all three.
//!
//! # Why this module exists
//!
//! The per-document loops these replace walked every clause one document at a
//! time: advance every leg, read every frequency and norm, sum, collect, pull
//! the collector's threshold, repeat. Lucene stopped doing that in 10.x. Its
//! scorers hand back a whole postings block of `(doc, score)` pairs at once
//! (`nextDocsAndScores`) -- one `System.arraycopy` of the decoded block, one
//! norm gather, one vectorizable BM25 loop -- and the boolean bulk scorers then
//! work on those buffers:
//!
//! - a conjunction scores its cheapest clause first and **drops every document
//!   whose partial score plus the other clauses' block maxima cannot reach the
//!   threshold before advancing the other clauses to it**
//!   (`ScorerUtil.filterCompetitiveHits` then `applyRequiredClause`);
//! - a disjunction splits its clauses, per window, into *essential* ones (whose
//!   summed maxima alone can reach the threshold) and *non-essential* ones, and
//!   only ever iterates the essential ones; the others are consulted by
//!   `advance` for the survivors.
//!
//! Both need the threshold only once per batch, not once per document.
//!
//! # Scores are bit-for-bit Lucene's
//!
//! A clause's score is `BM25Scorer.score`'s `weight - weight / (1 + freq *
//! normInverse)` in `f32` (`similarity::do_score`). Clause scores are summed in
//! `f64` and narrowed once, as every Lucene bulk scorer does
//! (`DocAndScoreAccBuffer.scores` is a `double[]`). The per-document loops this
//! replaced summed in `f32`, which is only the same for two clauses. Summing a
//! handful of `f32`s in `f64` is exact (their exponents are far closer than the
//! 29 spare bits), so the order clauses are added in does not matter either.
//!
//! Thresholds use Lucene's convention, `minCompetitiveScore`: a document must
//! score **at least** it. The collector reports its bottom score, and
//! `TopScoreDocCollector` publishes `Math.nextUp(bottom)`, so
//! [`min_competitive_score`] is `next_up` of what the collector reports.

use lucene_codecs::postings::{Impact, LazyDocsCursor, NO_MORE_DOCS};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::ScoringCollector;
use crate::field_norms::FieldNormsCursor;
use crate::similarity;
use crate::{blocktree, Result};

/// `MaxScoreBulkScorer.INNER_WINDOW_SIZE`.
const INNER_WINDOW_SIZE: i32 = 1 << 12;
/// `BlockMaxConjunctionBulkScorer.MAX_WINDOW_SIZE`.
const MAX_WINDOW_SIZE: i32 = 65536;

/// `DocAndFloatFeatureBuffer`: one clause's batch of documents and scores.
#[derive(Debug, Default)]
pub(crate) struct DocScores {
    pub(crate) docs: Vec<i32>,
    pub(crate) scores: Vec<f32>,
}

/// `DocAndScoreAccBuffer`: documents and their partial scores, accumulated in
/// `f64` across clauses.
#[derive(Debug, Default)]
pub(crate) struct DocScoreAcc {
    pub(crate) docs: Vec<i32>,
    pub(crate) scores: Vec<f64>,
}

impl DocScoreAcc {
    fn copy_from(&mut self, src: &DocScores) {
        self.docs.clear();
        self.docs.extend_from_slice(&src.docs);
        self.scores.clear();
        self.scores.extend(src.scores.iter().map(|&s| s as f64));
    }

    fn truncate(&mut self, n: usize) {
        self.docs.truncate(n);
        self.scores.truncate(n);
    }
}

/// The collector's threshold in Lucene's `minCompetitiveScore` form: `0` while
/// nothing may be pruned, otherwise the smallest score that can still enter
/// the results (`Math.nextUp(bottom)`).
#[inline]
pub(crate) fn min_competitive_score<C: ScoringCollector + ?Sized>(collector: &C) -> f32 {
    match collector.pruning_threshold() {
        Some(t) => t.next_up(),
        None => 0.0,
    }
}

/// `MathUtil.sumRelativeErrorBound`.
fn sum_relative_error_bound(num_values: usize) -> f64 {
    if num_values <= 1 {
        return 0.0;
    }
    // u: the unit roundoff of a `double`, 2^-52.
    let u = f64::from_bits((1023u64 - 52) << 52);
    (num_values - 1) as f64 * u
}

/// `MathUtil.sumUpperBound`.
fn sum_upper_bound(sum: f64, num_values: usize) -> f64 {
    if num_values <= 2 {
        return sum;
    }
    (1.0 + 2.0 * sum_relative_error_bound(num_values)) * sum
}

/// `Math.ulp(float)`, widened.
fn ulp_f32(x: f32) -> f64 {
    let x = x.abs();
    (x.next_up() - x) as f64
}

/// `ScorerUtil.minRequiredScore`.
fn min_required_score(max_remaining: f64, min_competitive: f32, num_scorers: usize) -> f64 {
    let mut min_required = min_competitive as f64 - max_remaining;
    let subtraction = ulp_f32(min_competitive);
    while min_required > 0.0
        && (sum_upper_bound(min_required + max_remaining, num_scorers) as f32) >= min_competitive
    {
        min_required -= subtraction;
    }
    min_required
}

/// `ScorerUtil.filterCompetitiveHits`: drop every document whose partial score
/// plus `max_remaining` cannot reach `min_competitive`.
fn filter_competitive_hits(
    acc: &mut DocScoreAcc,
    max_remaining: f64,
    min_competitive: f32,
    num_scorers: usize,
) {
    let min_required = min_required_score(max_remaining, min_competitive, num_scorers);
    if min_required <= 0.0 {
        return;
    }
    // `VectorUtil.filterByScore`: a stable in-place compaction.
    let mut n = 0;
    for i in 0..acc.docs.len() {
        let s = acc.scores[i];
        if s >= min_required {
            acc.docs[n] = acc.docs[i];
            acc.scores[n] = s;
            n += 1;
        }
    }
    acc.truncate(n);
}

/// One term clause: a lazy postings cursor, its BM25 weight and norms, and the
/// `ImpactsDISI` / `MaxScoreCache` state that lets it skip blocks.
pub(crate) struct TermLeg<'a> {
    cursor: LazyDocsCursor<'a>,
    /// `boost * idf`; `0` for a non-scoring (`FILTER`) clause.
    weight: f32,
    /// `false` for a `FILTER` clause: every score is `0` and no frequency or
    /// norm is ever read (its cursor was opened docs-only).
    scoring: bool,
    /// `None` scores every document at the unnormed length, the same rule
    /// every other scoring path in this crate applies.
    norms: Option<FieldNormsCursor<'a, 'a>>,
    /// `avgFieldLength`, for bounding impacts the way the term path does.
    avg_field_length: f32,
    /// `DocIdSetIterator.cost()`: the term's document frequency.
    pub(crate) cost: i64,
    /// Upper bound on any document's score, whatever the impacts say --
    /// `MaxScoreCache.globalMaxScore`.
    global_max: f32,
    /// `MaxScoreCache`: the level-0 and level-1 bounds, keyed by the level's
    /// last doc id (unique per block/span, so a key match means the impacts
    /// are the same ones).
    l0_key: i32,
    l0_max: f32,
    l1_key: i32,
    l1_max: f32,
    /// `ImpactsDISI.minCompetitiveScore` / `upTo`.
    min_competitive: f32,
    up_to: i32,
    /// `DisiWrapper.doc`: this clause's doc as the disjunction last saw it.
    doc: i32,
    /// `DisiWrapper.maxWindowScore`.
    max_window_score: f32,
    freqs: Vec<i32>,
    norm_inv: Vec<f32>,
}

impl<'a> TermLeg<'a> {
    /// A scoring clause. `max_freq` is the highest frequency any one document
    /// can have for this term in this segment (`totalTermFreq - docFreq + 1`),
    /// which is what makes the global bound tight enough to stop a keyword
    /// field's scan as soon as the top-k fills.
    pub(crate) fn scoring(
        cursor: LazyDocsCursor<'a>,
        weight: f32,
        norms: Option<FieldNormsCursor<'a, 'a>>,
        avg_field_length: f32,
        cost: i64,
        max_freq: f32,
    ) -> Self {
        let max_norm_inverse = match &norms {
            Some(n) => n.max_norm_inverse(),
            None => similarity::UNNORMED_NORM_INVERSE,
        };
        let global_max = similarity::do_score(weight, max_freq, max_norm_inverse);
        Self::new(
            cursor,
            weight,
            true,
            norms,
            avg_field_length,
            cost,
            global_max,
        )
    }

    /// A `FILTER` clause: matches gate the conjunction and contribute `0`.
    pub(crate) fn filter(cursor: LazyDocsCursor<'a>, cost: i64) -> Self {
        Self::new(cursor, 0.0, false, None, 1.0, cost, 0.0)
    }

    fn new(
        cursor: LazyDocsCursor<'a>,
        weight: f32,
        scoring: bool,
        norms: Option<FieldNormsCursor<'a, 'a>>,
        avg_field_length: f32,
        cost: i64,
        global_max: f32,
    ) -> Self {
        Self {
            cursor,
            weight,
            scoring,
            norms,
            avg_field_length,
            cost,
            global_max,
            l0_key: i32::MIN,
            l0_max: 0.0,
            l1_key: i32::MIN,
            l1_max: 0.0,
            min_competitive: 0.0,
            up_to: NO_MORE_DOCS,
            doc: -1,
            max_window_score: 0.0,
            freqs: Vec::with_capacity(lucene_codecs::postings::BLOCK_SIZE as usize + 1),
            norm_inv: Vec::with_capacity(lucene_codecs::postings::BLOCK_SIZE as usize + 1),
        }
    }

    #[inline]
    pub(crate) fn doc_id(&self) -> i32 {
        self.cursor.doc_id()
    }

    #[inline]
    pub(crate) fn advance(&mut self, target: i32) -> Result<i32> {
        Ok(self
            .cursor
            .advance(target)
            .map_err(blocktree::Error::Postings)?)
    }

    #[inline]
    pub(crate) fn next_doc(&mut self) -> Result<i32> {
        Ok(self.cursor.next_doc().map_err(blocktree::Error::Postings)?)
    }

    #[inline]
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        Ok(self
            .cursor
            .advance_shallow(target)
            .map_err(blocktree::Error::Postings)?)
    }

    /// The largest score a block's impacts allow -- the same float expression
    /// the document is scored with, so the bound can never land one ULP under
    /// a real score.
    fn impacts_bound(&self, impacts: &[Impact]) -> f32 {
        match &self.norms {
            Some(_) => similarity::max_score_for_impacts_weighted(
                impacts,
                self.weight,
                self.avg_field_length,
            ),
            None => similarity::max_score_for_impacts_unnormed_weighted(impacts, self.weight),
        }
    }

    /// `MaxScoreCache.getMaxScoreForLevel(0)`; an empty level (the tail block,
    /// a field without frequencies) is bounded by the global maximum, where
    /// Lucene uses a dummy `freq = MAX_VALUE` impact that bounds it at
    /// `weight`.
    fn level0_max(&mut self) -> f32 {
        if !self.scoring {
            return 0.0;
        }
        let key = self.cursor.level0_last_doc_id();
        if key != self.l0_key {
            let impacts = self.cursor.level0_impacts();
            self.l0_max = if impacts.is_empty() {
                self.global_max
            } else {
                self.impacts_bound(impacts).min(self.global_max)
            };
            self.l0_key = key;
        }
        self.l0_max
    }

    /// Whether a level-1 span with impacts is available -- Lucene's
    /// `Impacts.numLevels() == 2`.
    fn has_level1(&self) -> bool {
        self.cursor.level1_last_doc_id() != NO_MORE_DOCS && !self.cursor.level1_impacts().is_empty()
    }

    fn level1_max(&mut self) -> f32 {
        if !self.scoring {
            return 0.0;
        }
        let key = self.cursor.level1_last_doc_id();
        if key != self.l1_key {
            self.l1_max = self
                .impacts_bound(self.cursor.level1_impacts())
                .min(self.global_max);
            self.l1_key = key;
        }
        self.l1_max
    }

    /// `Scorer.getMaxScore(upTo)` / `MaxScoreCache.getMaxScore`: a bound on
    /// every score of this clause up to and including `up_to`, from the first
    /// impacts level that covers it.
    pub(crate) fn max_score(&mut self, up_to: i32) -> f32 {
        if !self.scoring {
            return 0.0;
        }
        if up_to <= self.cursor.level0_last_doc_id() {
            return self.level0_max();
        }
        if self.has_level1() && up_to <= self.cursor.level1_last_doc_id() {
            return self.level1_max();
        }
        self.global_max
    }

    /// `MaxScoreCache.getSkipUpTo`: the last doc id of the highest impacts
    /// level whose bound is under `min_score`, or `-1` if even level 0's is not.
    //
    // SENTINEL: `-1` = "no level can be skipped", outside the domain of a doc
    // id. Its one caller, `advance_target`, tests `skip == -1`.
    fn skip_up_to(&mut self, min_score: f32) -> i32 {
        if self.level0_max() >= min_score {
            return -1;
        }
        if self.has_level1() {
            if self.level1_max() >= min_score {
                return self.cursor.level0_last_doc_id();
            }
            return self.cursor.level1_last_doc_id();
        }
        self.cursor.level0_last_doc_id()
    }

    /// `ImpactsDISI.setMinCompetitiveScore`.
    pub(crate) fn set_min_competitive_score(&mut self, min: f32) {
        if min > self.min_competitive {
            self.min_competitive = min;
            self.up_to = -1;
        }
    }

    /// `ImpactsDISI.advanceTarget`: the first doc at or after `target` whose
    /// block can hold a competitive score, deciding on impacts alone.
    fn advance_target(&mut self, mut target: i32) -> Result<i32> {
        if target <= self.up_to {
            return Ok(target);
        }
        // Nothing anywhere can compete: `MaxScoreCache.globalMaxScore` answers
        // without walking a single block header.
        if self.global_max < self.min_competitive {
            #[cfg(any(test, feature = "test-support"))]
            crate::test_only_maxscore_block_skip_counter::record_skip();
            return Ok(NO_MORE_DOCS);
        }
        self.up_to = self.advance_shallow(target)?;
        let mut max = self.level0_max();
        loop {
            if max >= self.min_competitive {
                return Ok(target);
            }
            if self.up_to == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            }
            #[cfg(any(test, feature = "test-support"))]
            crate::test_only_maxscore_block_skip_counter::record_skip();
            let skip = self.skip_up_to(self.min_competitive);
            target = if skip == -1 {
                self.up_to.saturating_add(1)
            } else if skip == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            } else {
                skip.saturating_add(1)
            };
            self.up_to = self.advance_shallow(target)?;
            max = self.level0_max();
        }
    }

    /// `ImpactsDISI.ensureCompetitive`.
    fn ensure_competitive(&mut self) -> Result<()> {
        let doc = self.cursor.doc_id();
        if doc == NO_MORE_DOCS {
            return Ok(());
        }
        let target = self.advance_target(doc)?;
        if target != doc {
            self.advance(target)?;
        }
        Ok(())
    }

    /// `Scorable.score()` for the document the cursor is on.
    #[inline]
    pub(crate) fn score(&mut self) -> Result<f32> {
        if !self.scoring {
            return Ok(0.0);
        }
        let doc = self.cursor.doc_id();
        let freq = self.cursor.freq().unwrap_or(1) as f32;
        let norm_inverse = match self.norms.as_mut() {
            Some(n) => n.norm_inverse(doc)?,
            None => similarity::UNNORMED_NORM_INVERSE,
        };
        Ok(similarity::do_score(self.weight, freq, norm_inverse))
    }

    /// `TermScorer.nextDocsAndScores`: the rest of the current postings block
    /// below `up_to`, scored, with deleted documents removed. Leaves the
    /// cursor on the first document not returned. An empty buffer means there
    /// are no more documents below `up_to`.
    pub(crate) fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut DocScores,
    ) -> Result<()> {
        loop {
            self.ensure_competitive()?;
            self.cursor
                .next_postings(up_to, &mut out.docs, &mut self.freqs)
                .map_err(blocktree::Error::Postings)?;
            if let Some(live) = live_docs {
                if !out.docs.is_empty() {
                    let mut n = 0;
                    for i in 0..out.docs.len() {
                        let d = out.docs[i];
                        if live.get_doc(d) {
                            out.docs[n] = d;
                            self.freqs[n] = self.freqs[i];
                            n += 1;
                        }
                    }
                    out.docs.truncate(n);
                    self.freqs.truncate(n);
                    // A whole batch of deleted documents is not the end of the
                    // postings: fetch the next one, as Java loops.
                    if n == 0 {
                        continue;
                    }
                }
            }
            break;
        }
        out.scores.clear();
        if !self.scoring {
            out.scores.resize(out.docs.len(), 0.0);
            return Ok(());
        }
        match self.norms.as_mut() {
            Some(n) => n.norm_inverse_batch(&out.docs, &mut self.norm_inv)?,
            None => {
                self.norm_inv.clear();
                self.norm_inv
                    .resize(out.docs.len(), similarity::UNNORMED_NORM_INVERSE);
            }
        }
        similarity::do_score_batch(self.weight, &self.freqs, &self.norm_inv, &mut out.scores);
        Ok(())
    }
}

/// `ScorerUtil.applyRequiredClause`: keep only the buffered documents `leg`
/// also matches, adding its score to theirs.
fn apply_required_clause(acc: &mut DocScoreAcc, leg: &mut TermLeg<'_>) -> Result<()> {
    let mut n = 0;
    let mut cur = leg.doc_id();
    for i in 0..acc.docs.len() {
        let target = acc.docs[i];
        if cur < target {
            cur = leg.advance(target)?;
        }
        if cur == target {
            acc.docs[n] = target;
            acc.scores[n] = acc.scores[i] + leg.score()? as f64;
            n += 1;
        }
    }
    acc.truncate(n);
    Ok(())
}

/// `ScorerUtil.applyOptionalClause`: add `leg`'s score to every buffered
/// document it matches, keeping the rest.
fn apply_optional_clause(acc: &mut DocScoreAcc, leg: &mut TermLeg<'_>) -> Result<()> {
    let mut cur = leg.doc_id();
    for i in 0..acc.docs.len() {
        let target = acc.docs[i];
        if cur < target {
            cur = leg.advance(target)?;
        }
        if cur == target {
            acc.scores[i] += leg.score()? as f64;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// One term: BatchScoreBulkScorer.
// ---------------------------------------------------------------------------

/// `BatchScoreBulkScorer.score` over the whole segment.
pub(crate) fn score_term<C: ScoringCollector>(
    leg: &mut TermLeg<'_>,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
) -> Result<()> {
    let mut min_competitive = min_competitive_score(collector);
    leg.set_min_competitive_score(min_competitive);
    if leg.doc_id() < 0 {
        leg.advance(0)?;
    }
    let mut buf = DocScores::default();
    loop {
        leg.next_docs_and_scores(NO_MORE_DOCS, live_docs, &mut buf)?;
        if buf.docs.is_empty() {
            return Ok(());
        }
        for (&doc, &score) in buf.docs.iter().zip(&buf.scores) {
            if score >= min_competitive {
                collector.collect(doc, score);
                min_competitive = min_competitive_score(collector);
            }
        }
        leg.set_min_competitive_score(min_competitive);
    }
}

// ---------------------------------------------------------------------------
// A conjunction: BlockMaxConjunctionBulkScorer.
// ---------------------------------------------------------------------------

/// `BlockMaxConjunctionBulkScorer.score` over the whole segment. `legs` must
/// hold at least two clauses; they are reordered by cost.
pub(crate) fn score_conjunction<C: ScoringCollector>(
    legs: &mut [TermLeg<'_>],
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
) -> Result<()> {
    if legs.len() == 1 {
        // One clause is not a conjunction: `BooleanScorerSupplier` hands a
        // single required clause straight to its own scorer, and a lone
        // `FILTER` clause scores 0 everywhere, which the term path already
        // stops the moment the queue fills.
        return score_term(&mut legs[0], live_docs, collector);
    }
    legs.sort_by_key(|l| l.cost);
    let n = legs.len();
    let mut sum_of_others = vec![f64::INFINITY; n];
    let mut single = DocScores::default();
    let mut acc = DocScoreAcc::default();
    let max = NO_MORE_DOCS;

    let mut window_min = legs[0].doc_id().max(0);
    if min_competitive_score(collector) == 0.0 {
        window_min = conjunction_doc_first(legs, live_docs, collector, 0, max)?;
    }
    while window_min < max {
        // The cheapest clause's block boundary is the window.
        let mut window_max = legs[0].advance_shallow(window_min)?.min(max - 1);
        window_max = window_max.min(window_min.saturating_add(MAX_WINDOW_SIZE));

        // `computeMaxScore`.
        for leg in legs.iter_mut() {
            leg.advance_shallow(window_min)?;
        }
        let mut max_window_score = 0.0f64;
        for (i, leg) in legs.iter_mut().enumerate() {
            let m = leg.max_score(window_max) as f64;
            sum_of_others[i] = m;
            max_window_score += m;
        }
        for i in (0..n - 1).rev() {
            sum_of_others[i] += sum_of_others[i + 1];
        }

        // `scoreWindowScoreFirst`.
        let window_end = window_max.saturating_add(1);
        let mut min_competitive = min_competitive_score(collector);
        if (max_window_score as f32) >= min_competitive {
            if legs[0].doc_id() < window_min {
                legs[0].advance(window_min)?;
            }
            if legs[0].doc_id() < window_end {
                loop {
                    legs[0].next_docs_and_scores(window_end, live_docs, &mut single)?;
                    if single.docs.is_empty() {
                        break;
                    }
                    acc.copy_from(&single);
                    for i in 1..n {
                        let remaining = sum_of_others[i];
                        // Two equal consecutive sums mean clause `i - 1` scores
                        // nothing, so filtering again would remove nothing.
                        if remaining != sum_of_others[i - 1] {
                            filter_competitive_hits(&mut acc, remaining, min_competitive, n);
                        }
                        apply_required_clause(&mut acc, &mut legs[i])?;
                    }
                    for (&doc, &score) in acc.docs.iter().zip(&acc.scores) {
                        collector.collect(doc, score as f32);
                    }
                    min_competitive = min_competitive_score(collector);
                }
                let mut max_other = -1;
                for leg in &legs[1..] {
                    max_other = max_other.max(leg.doc_id());
                }
                if legs[0].doc_id() < max_other {
                    legs[0].advance(max_other)?;
                }
            }
        } else {
            // The whole window goes without a single block body decoded.
            #[cfg(any(test, feature = "test-support"))]
            crate::test_only_maxscore_block_skip_counter::record_skip();
        }
        window_min = legs[0].doc_id().max(window_end);
    }
    Ok(())
}

/// `scoreDocFirstUntilDynamicPruning`: a plain leapfrog, scoring every match,
/// until the collector publishes a threshold. Returns the lead's next doc.
fn conjunction_doc_first<C: ScoringCollector>(
    legs: &mut [TermLeg<'_>],
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
    min: i32,
    max: i32,
) -> Result<i32> {
    let mut doc = legs[0].doc_id();
    if doc < min {
        doc = legs[0].advance(min)?;
    }
    'outer: while doc < max {
        if live_docs.is_none_or(|l| l.get_doc(doc)) {
            for i in 1..legs.len() {
                let mut other = legs[i].doc_id();
                if other < doc {
                    other = legs[i].advance(doc)?;
                }
                if other != doc {
                    doc = legs[0].advance(other)?;
                    continue 'outer;
                }
            }
            let mut score = 0.0f64;
            for leg in legs.iter_mut() {
                score += leg.score()? as f64;
            }
            collector.collect(doc, score as f32);
            if min_competitive_score(collector) > 0.0 {
                return legs[0].next_doc();
            }
        }
        doc = legs[0].next_doc()?;
    }
    Ok(doc)
}

// ---------------------------------------------------------------------------
// A disjunction: MaxScoreBulkScorer.
// ---------------------------------------------------------------------------

/// `MaxScoreBulkScorer`'s per-query state, minus the clauses themselves.
struct MaxScore {
    /// Indices into the legs, in `allScorers` order: non-essential first.
    order: Vec<usize>,
    scratch: Vec<usize>,
    first_essential: usize,
    first_required: usize,
    next_min_competitive: f32,
    max_score_sums: Vec<f64>,
    window_matches: FixedBitSet,
    window_scores: Vec<f64>,
    num_outer_windows: i64,
    num_candidates: i64,
    min_window_size: i32,
    single: DocScores,
    acc: DocScoreAcc,
    min_competitive: f32,
}

/// `MaxScoreBulkScorer.score` over the whole segment.
pub(crate) fn score_disjunction<C: ScoringCollector>(
    legs: &mut [TermLeg<'_>],
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
) -> Result<()> {
    let n = legs.len();
    if n == 0 {
        return Ok(());
    }
    let mut ms = MaxScore {
        order: (0..n).collect(),
        scratch: Vec::with_capacity(n),
        first_essential: 0,
        first_required: n,
        next_min_competitive: f32::INFINITY,
        max_score_sums: vec![0.0; n],
        window_matches: FixedBitSet::new(INNER_WINDOW_SIZE as usize),
        window_scores: vec![0.0; INNER_WINDOW_SIZE as usize],
        num_outer_windows: 0,
        num_candidates: 0,
        min_window_size: 1,
        single: DocScores::default(),
        acc: DocScoreAcc::default(),
        min_competitive: min_competitive_score(collector),
    };
    for leg in legs.iter_mut() {
        leg.doc = leg.doc_id();
    }
    let max = NO_MORE_DOCS;
    let mut outer_min = 0;
    'outer: while outer_min < max {
        let mut outer_max = ms.compute_outer_window_max(legs, outer_min)?.min(max);
        loop {
            ms.update_max_window_scores(legs, outer_min, outer_max)?;
            if !ms.partition_scorers(legs) {
                // No clause can compete anywhere in this window.
                #[cfg(any(test, feature = "test-support"))]
                crate::test_only_maxscore_block_skip_counter::record_skip();
                outer_min = outer_max;
                continue 'outer;
            }
            let new_max = ms.compute_outer_window_max(legs, outer_min)?;
            if new_max >= outer_max {
                break;
            }
            outer_max = new_max;
        }
        // A clause the threshold made non-essential is never iterated in this
        // window: its postings are only `advance`d to surviving candidates.
        #[cfg(any(test, feature = "test-support"))]
        if ms.first_essential > 0 {
            crate::test_only_maxscore_block_skip_counter::record_skip();
        }
        for &i in &ms.order[ms.first_essential..] {
            if legs[i].doc < outer_min {
                legs[i].doc = legs[i].advance(outer_min)?;
            }
        }
        while ms.top(legs).1 < outer_max {
            ms.score_inner_window(legs, live_docs, collector, outer_max)?;
            if ms.min_competitive >= ms.next_min_competitive {
                // The threshold rose enough for a better partition.
                break;
            }
        }
        outer_min = ms.top(legs).1.min(outer_max);
        ms.num_outer_windows += 1;
    }
    Ok(())
}

impl MaxScore {
    /// The essential clause with the smallest doc, and that doc; `NO_MORE_DOCS`
    /// when every essential clause is exhausted.
    fn top(&self, legs: &[TermLeg<'_>]) -> (usize, i32) {
        let mut best = (usize::MAX, NO_MORE_DOCS);
        for &i in &self.order[self.first_essential..] {
            if best.0 == usize::MAX || legs[i].doc < best.1 {
                best = (i, legs[i].doc);
            }
        }
        best
    }

    /// The second-smallest essential doc (`essentialQueue.top2()`), or `None`
    /// with a single essential clause.
    fn top2_doc(&self, legs: &[TermLeg<'_>], top: usize) -> Option<i32> {
        let mut best: Option<i32> = None;
        for &i in &self.order[self.first_essential..] {
            if i == top {
                continue;
            }
            let d = legs[i].doc;
            best = Some(best.map_or(d, |b| b.min(d)));
        }
        best
    }

    fn compute_outer_window_max(
        &mut self,
        legs: &mut [TermLeg<'_>],
        window_min: i32,
    ) -> Result<i32> {
        let n = legs.len();
        let first_window_lead = self.first_essential.min(n - 1);
        let mut window_max = NO_MORE_DOCS;
        for k in first_window_lead..n {
            let i = self.order[k];
            let target = legs[i].doc.max(window_min);
            let up_to = legs[i].advance_shallow(target)?;
            // `upTo + 1` in unsigned arithmetic: `NO_MORE_DOCS + 1` wraps to
            // a huge unsigned value, i.e. "no bound".
            window_max = window_max.min(up_to.saturating_add(1));
        }
        if n - first_window_lead > 1 {
            // Aim for at least 32 candidates per clause per outer window, so
            // that recomputing maxima does not dominate.
            let threshold = self.num_outer_windows * 32 * n as i64;
            if self.num_candidates < threshold {
                self.min_window_size = (self.min_window_size << 1).min(INNER_WINDOW_SIZE);
            } else {
                self.min_window_size = 1;
            }
            let min_window_max = window_min.saturating_add(self.min_window_size);
            window_max = window_max.max(min_window_max);
        }
        Ok(window_max)
    }

    fn update_max_window_scores(
        &mut self,
        legs: &mut [TermLeg<'_>],
        window_min: i32,
        window_max: i32,
    ) -> Result<()> {
        for leg in legs.iter_mut() {
            if leg.doc < window_max {
                if leg.doc < window_min {
                    leg.advance_shallow(window_min)?;
                }
                leg.max_window_score = leg.max_score(window_max - 1);
            } else {
                leg.max_window_score = 0.0;
            }
        }
        Ok(())
    }

    fn partition_scorers(&mut self, legs: &[TermLeg<'_>]) -> bool {
        let n = legs.len();
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.order);
        // Stable, like Java's `Arrays.sort` over objects.
        self.scratch.sort_by(|&a, &b| {
            let ka = legs[a].max_window_score as f64 / legs[a].cost.max(1) as f64;
            let kb = legs[b].max_window_score as f64 / legs[b].cost.max(1) as f64;
            ka.total_cmp(&kb)
        });
        let mut max_score_sum = 0.0f64;
        self.first_essential = 0;
        self.next_min_competitive = f32::INFINITY;
        for i in 0..n {
            let w = self.scratch[i];
            let new_sum = max_score_sum + legs[w].max_window_score as f64;
            let sum_f = sum_upper_bound(new_sum, self.first_essential + 1) as f32;
            if sum_f < self.min_competitive {
                max_score_sum = new_sum;
                self.order[self.first_essential] = w;
                self.max_score_sums[self.first_essential] = max_score_sum;
                self.first_essential += 1;
            } else {
                self.order[n - 1 - (i - self.first_essential)] = w;
                self.next_min_competitive = self.next_min_competitive.min(sum_f);
            }
        }
        self.first_required = n;
        if self.first_essential == n {
            return false;
        }
        if self.first_essential == n - 1 {
            // One essential clause: if it plus every non-essential clause but
            // the best one cannot compete, hits must also match that one.
            self.first_required = n - 1;
            let mut max_required = legs[self.order[self.first_essential]].max_window_score as f64;
            while self.first_required > 0 {
                let mut without_previous = max_required;
                if self.first_required > 1 {
                    without_previous += self.max_score_sums[self.first_required - 2];
                }
                if (without_previous as f32) >= self.min_competitive {
                    break;
                }
                self.first_required -= 1;
                max_required += legs[self.order[self.first_required]].max_window_score as f64;
            }
        }
        true
    }

    fn score_inner_window<C: ScoringCollector>(
        &mut self,
        legs: &mut [TermLeg<'_>],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        max: i32,
    ) -> Result<()> {
        let (top, top_doc) = self.top(legs);
        match self.top2_doc(legs, top) {
            None => self.score_single_essential(legs, live_docs, collector, top, max),
            Some(top2) if top2.saturating_sub(INNER_WINDOW_SIZE / 2) >= top_doc => {
                // The first half of the window only matches one clause: stream
                // it up to the next clause's doc.
                self.score_single_essential(legs, live_docs, collector, top, max.min(top2))
            }
            Some(_) => self.score_multiple_essential(legs, live_docs, collector, top_doc, max),
        }
    }

    fn score_single_essential<C: ScoringCollector>(
        &mut self,
        legs: &mut [TermLeg<'_>],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        top: usize,
        up_to: i32,
    ) -> Result<()> {
        loop {
            legs[top].next_docs_and_scores(up_to, live_docs, &mut self.single)?;
            if self.single.docs.is_empty() {
                break;
            }
            self.acc.copy_from(&self.single);
            self.score_non_essential(legs, collector)?;
        }
        legs[top].doc = legs[top].doc_id();
        Ok(())
    }

    fn score_multiple_essential<C: ScoringCollector>(
        &mut self,
        legs: &mut [TermLeg<'_>],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        inner_min: i32,
        max: i32,
    ) -> Result<()> {
        let inner_max = max.min(inner_min.saturating_add(INNER_WINDOW_SIZE));
        // `collectEssentialScoresIntoWindow`: drain every essential clause's
        // documents below `inner_max` into the window bitset.
        loop {
            let (top, top_doc) = self.top(legs);
            if top_doc >= inner_max {
                break;
            }
            loop {
                legs[top].next_docs_and_scores(inner_max, live_docs, &mut self.single)?;
                if self.single.docs.is_empty() {
                    break;
                }
                for (&doc, &score) in self.single.docs.iter().zip(&self.single.scores) {
                    let i = (doc - inner_min) as usize;
                    // FBS: `window_matches` is `FixedBitSet::new(INNER_WINDOW_SIZE)`,
                    // and `next_docs_and_scores(inner_max, ..)` only returns docs in
                    // `inner_min..inner_max`, a span of at most `INNER_WINDOW_SIZE`
                    // (`inner_max = min(max, inner_min + INNER_WINDOW_SIZE)`).
                    self.window_matches.set(i);
                    self.window_scores[i] += score as f64;
                }
            }
            legs[top].doc = legs[top].doc_id();
        }
        // `flushWindowToDocAndScoreAccBuffer`.
        self.acc.docs.clear();
        self.acc.scores.clear();
        let window_scores = &mut self.window_scores;
        let acc = &mut self.acc;
        self.window_matches.for_each_set_bit(|i| {
            acc.docs.push(inner_min + i as i32);
            acc.scores.push(window_scores[i]);
            window_scores[i] = 0.0;
        });
        self.window_matches.clear_all();
        self.score_non_essential(legs, collector)
    }

    fn score_non_essential<C: ScoringCollector>(
        &mut self,
        legs: &mut [TermLeg<'_>],
        collector: &mut C,
    ) -> Result<()> {
        self.num_candidates += self.acc.docs.len() as i64;
        let n = legs.len();
        for k in (0..self.first_essential).rev() {
            let i = self.order[k];
            filter_competitive_hits(
                &mut self.acc,
                self.max_score_sums[k],
                self.min_competitive,
                n,
            );
            if k >= self.first_required {
                apply_required_clause(&mut self.acc, &mut legs[i])?;
            } else {
                apply_optional_clause(&mut self.acc, &mut legs[i])?;
            }
            legs[i].doc = legs[i].doc_id();
        }
        for (&doc, &score) in self.acc.docs.iter().zip(&self.acc.scores) {
            collector.collect(doc, score as f32);
        }
        self.min_competitive = min_competitive_score(collector);
        Ok(())
    }
}
