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
pub(crate) fn sum_relative_error_bound(num_values: usize) -> f64 {
    if num_values <= 1 {
        return 0.0;
    }
    // u: the unit roundoff of a `double`, 2^-52.
    let u = f64::from_bits((1023u64 - 52) << 52);
    (num_values - 1) as f64 * u
}

/// `MathUtil.sumUpperBound`.
pub(crate) fn sum_upper_bound(sum: f64, num_values: usize) -> f64 {
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

/// What the conjunction bulk scorer needs of a clause: Lucene's `Scorer` as
/// `BlockMaxConjunctionBulkScorer` uses it. [`TermLeg`] implements it with
/// static dispatch; a boxed scorer tree implements it in `exec`, with
/// `Scorer.nextDocsAndScores`'s default batching.
pub(crate) trait BulkLeg {
    fn doc_id(&self) -> i32;
    fn advance(&mut self, target: i32) -> Result<i32>;
    fn next_doc(&mut self) -> Result<i32>;
    fn score(&mut self) -> Result<f32>;
    fn advance_shallow(&mut self, target: i32) -> Result<i32>;
    fn max_score(&mut self, up_to: i32) -> Result<f32>;
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut DocScores,
    ) -> Result<()>;
}

impl BulkLeg for TermLeg<'_> {
    #[inline]
    fn doc_id(&self) -> i32 {
        TermLeg::doc_id(self)
    }
    #[inline]
    fn advance(&mut self, target: i32) -> Result<i32> {
        TermLeg::advance(self, target)
    }
    #[inline]
    fn next_doc(&mut self) -> Result<i32> {
        TermLeg::next_doc(self)
    }
    #[inline]
    fn score(&mut self) -> Result<f32> {
        TermLeg::score(self)
    }
    #[inline]
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        TermLeg::advance_shallow(self, target)
    }
    #[inline]
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        Ok(TermLeg::max_score(self, up_to))
    }
    #[inline]
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut DocScores,
    ) -> Result<()> {
        TermLeg::next_docs_and_scores(self, up_to, live_docs, out)
    }
}

/// One term clause: a lazy postings cursor, its BM25 weight and norms, and the
/// `ImpactsDISI` / `MaxScoreCache` state that lets it skip blocks.
pub(crate) struct TermLeg<'a> {
    cursor: LazyDocsCursor<'a>,
    /// `boost * idf`; `0` for a non-scoring (`FILTER`) clause.
    weight: f32,
    /// `false` for a `FILTER` clause or a `ConstantScoreQuery` around a term:
    /// every score is [`Self::constant`] and no frequency or norm is ever read
    /// (its cursor was opened docs-only).
    scoring: bool,
    /// The score of every document when not `scoring`: `0` for a filter,
    /// the constant for a `ConstantScoreScorer`.
    constant: f32,
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

    /// `ConstantScoreScorer` over a term's documents: every one scores
    /// `score`, and once the threshold passes `score` nothing is left to
    /// visit (the global bound is `score`, so `advance_target` ends it).
    pub(crate) fn constant(cursor: LazyDocsCursor<'a>, cost: i64, score: f32) -> Self {
        let mut leg = Self::new(cursor, 0.0, false, None, 1.0, cost, score);
        leg.constant = score;
        leg
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
            constant: 0.0,
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
            return self.constant;
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
            return self.constant;
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
            return self.constant;
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

    /// `ImpactsDISI.advance`: the first document at or after `target` in a
    /// block that can still compete. Until a minimum competitive score is set
    /// this is a plain `advance`.
    pub(crate) fn impacts_advance(&mut self, target: i32) -> Result<i32> {
        let target = self.advance_target(target)?;
        self.advance(target)
    }

    /// `ImpactsDISI.nextDoc`.
    pub(crate) fn impacts_next_doc(&mut self) -> Result<i32> {
        let doc = self.doc_id();
        if doc < self.up_to {
            return self.next_doc();
        }
        self.impacts_advance(doc.saturating_add(1))
    }

    /// A `FILTER` leg: matches only, every score `0`.
    pub(crate) fn is_filter(&self) -> bool {
        !self.scoring && self.constant == 0.0
    }

    /// `PostingsEnum.docIDRunEnd()`.
    pub(crate) fn doc_id_run_end(&self) -> i32 {
        self.cursor.doc_id_run_end()
    }

    /// `TermScorer.advanceShallow`.
    pub(crate) fn shallow_advance(&mut self, target: i32) -> Result<i32> {
        self.advance_shallow(target)
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
            return Ok(self.constant);
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
            out.scores.resize(out.docs.len(), self.constant);
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
fn apply_required_clause<L: BulkLeg + ?Sized>(acc: &mut DocScoreAcc, leg: &mut L) -> Result<()> {
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
pub(crate) fn score_term<C: ScoringCollector + ?Sized>(
    leg: &mut TermLeg<'_>,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
) -> Result<()> {
    let mut buf = DocScores::default();
    score_term_window(leg, &mut buf, live_docs, collector, 0, NO_MORE_DOCS).map(|_| ())
}

/// `BatchScoreBulkScorer.score(collector, acceptDocs, min, max)`: the term's
/// documents in `[min, max)`. Returns the next document to score, at or past
/// `max`.
pub(crate) fn score_term_window<C: ScoringCollector + ?Sized>(
    leg: &mut TermLeg<'_>,
    buf: &mut DocScores,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
    min: i32,
    max: i32,
) -> Result<i32> {
    let mut min_competitive = min_competitive_score(collector);
    leg.set_min_competitive_score(min_competitive);
    if leg.doc_id() < min {
        leg.advance(min)?;
    }
    loop {
        leg.next_docs_and_scores(max, live_docs, buf)?;
        if buf.docs.is_empty() {
            return Ok(leg.doc_id());
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

/// `BlockMaxConjunctionBulkScorer`'s buffers, kept across windowed calls.
pub(crate) struct ConjunctionBulk {
    sum_of_others: Vec<f64>,
    single: DocScores,
    acc: DocScoreAcc,
}

impl ConjunctionBulk {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            sum_of_others: vec![f64::INFINITY; n],
            single: DocScores::default(),
            acc: DocScoreAcc::default(),
        }
    }

    /// `BlockMaxConjunctionBulkScorer.score(collector, acceptDocs, min, max)`
    /// over `legs`, which must be at least two, cheapest first. Returns the
    /// next document to score.
    pub(crate) fn score<L: BulkLeg, C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [L],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        let n = legs.len();
        let (sum_of_others, single, acc) =
            (&mut self.sum_of_others, &mut self.single, &mut self.acc);
        let mut window_min = legs[0].doc_id().max(min);
        if min_competitive_score(collector) == 0.0 {
            window_min = conjunction_doc_first(legs, live_docs, collector, min, max)?;
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
                let m = leg.max_score(window_max)? as f64;
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
                        legs[0].next_docs_and_scores(window_end, live_docs, single)?;
                        if single.docs.is_empty() {
                            break;
                        }
                        acc.copy_from(single);
                        for i in 1..n {
                            let remaining = sum_of_others[i];
                            // Two equal consecutive sums mean clause `i - 1` scores
                            // nothing, so filtering again would remove nothing.
                            if remaining != sum_of_others[i - 1] {
                                filter_competitive_hits(acc, remaining, min_competitive, n);
                            }
                            apply_required_clause(acc, &mut legs[i])?;
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
        Ok(window_min)
    }
}

/// `scoreDocFirstUntilDynamicPruning`: a plain leapfrog, scoring every match,
/// until the collector publishes a threshold. Returns the lead's next doc.
fn conjunction_doc_first<L: BulkLeg, C: ScoringCollector + ?Sized>(
    legs: &mut [L],
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
pub(crate) struct MaxScore {
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
pub(crate) fn score_disjunction<C: ScoringCollector + ?Sized>(
    legs: &mut [TermLeg<'_>],
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
) -> Result<()> {
    if legs.is_empty() {
        return Ok(());
    }
    MaxScore::new(legs)
        .score(legs, None, live_docs, collector, 0, NO_MORE_DOCS)
        .map(|_| ())
}

impl MaxScore {
    pub(crate) fn new(legs: &mut [TermLeg<'_>]) -> Self {
        let n = legs.len();
        for leg in legs.iter_mut() {
            leg.doc = leg.doc_id();
        }
        MaxScore {
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
            min_competitive: 0.0,
        }
    }

    /// `MaxScoreBulkScorer.score(collector, acceptDocs, min, max)`: the
    /// disjunction's documents in `[min, max)` -- those also matching
    /// `filter`, when there is one (`filteredOptionalBulkScorer`). Returns the
    /// next document to score.
    pub(crate) fn score<C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [TermLeg<'_>],
        mut filter: Option<&mut (dyn crate::exec::Scorer + '_)>,
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        let ms = self;
        ms.min_competitive = min_competitive_score(collector);
        let mut outer_min = min;
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
            // A clause the threshold made non-essential is never iterated in
            // this window: its postings are only `advance`d to surviving
            // candidates.
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
                match filter.as_deref_mut() {
                    Some(f) => {
                        ms.score_inner_window_with_filter(legs, f, live_docs, collector, outer_max)?
                    }
                    None => ms.score_inner_window(legs, live_docs, collector, outer_max)?,
                }
                if ms.min_competitive >= ms.next_min_competitive {
                    // The threshold rose enough for a better partition.
                    break;
                }
            }
            outer_min = ms.top(legs).1.min(outer_max);
            ms.num_outer_windows += 1;
        }
        Ok(Self::next_candidate(legs, max))
    }

    /// `nextCandidate(rangeEnd)`.
    fn next_candidate(legs: &[TermLeg<'_>], range_end: i32) -> i32 {
        let mut next = NO_MORE_DOCS;
        for leg in legs {
            if leg.doc < range_end {
                return range_end;
            }
            next = next.min(leg.doc);
        }
        next
    }

    /// `scoreInnerWindowWithFilter` by leapfrog (`fillScoreBufferViaLeapFrog`):
    /// the essential clauses' documents that the filter also matches, summed,
    /// then the non-essential clauses as usual.
    fn score_inner_window_with_filter<C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [TermLeg<'_>],
        filter: &mut (dyn crate::exec::Scorer + '_),
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        max: i32,
    ) -> Result<()> {
        let (mut top, mut top_doc) = self.top(legs);
        let mut filter_doc = filter.doc_id();
        while top_doc < filter_doc {
            legs[top].doc = legs[top].advance(filter_doc)?;
            (top, top_doc) = self.top(legs);
        }
        if top_doc >= max {
            return Ok(());
        }
        let inner_max = max.min(top_doc.saturating_add(INNER_WINDOW_SIZE));
        self.acc.docs.clear();
        self.acc.scores.clear();
        while top_doc < inner_max {
            if filter_doc < top_doc {
                filter_doc = filter.advance(top_doc)?;
            }
            if filter_doc != top_doc {
                while top_doc < filter_doc {
                    legs[top].doc = legs[top].advance(filter_doc)?;
                    (top, top_doc) = self.top(legs);
                }
            } else {
                let doc = top_doc;
                let matched = live_docs.is_none_or(|l| l.get_doc(doc))
                    && (!filter.two_phase() || filter.matches()?);
                let mut score = 0.0f64;
                while top_doc == doc {
                    if matched {
                        score += legs[top].score()? as f64;
                    }
                    legs[top].doc = legs[top].next_doc()?;
                    (top, top_doc) = self.top(legs);
                }
                if matched {
                    self.acc.docs.push(doc);
                    self.acc.scores.push(score);
                }
            }
        }
        self.score_non_essential(legs, collector)
    }

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

    /// `computeOuterWindowMax`.
    ///
    /// Deviation: with a filter, Lucene lets only the clauses at least as
    /// costly as the filter bound the window. Under a dense filter that is
    /// none of them, so the partition is computed over one huge window;
    /// bounding by every clause measured faster (`FILTER` + two `SHOULD`s on
    /// the benchmark corpus: 1.36x Lucene against 0.99x with the rule). The
    /// windows only decide how often the partition is recomputed, never
    /// which documents match or what they score.
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

    fn score_inner_window<C: ScoringCollector + ?Sized>(
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

    fn score_single_essential<C: ScoringCollector + ?Sized>(
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

    fn score_multiple_essential<C: ScoringCollector + ?Sized>(
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

    fn score_non_essential<C: ScoringCollector + ?Sized>(
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

/// Filter legs as one iterator: a leapfrog over their documents, the filter
/// a filtered `MaxScoreBulkScorer` checks candidates against.
pub(crate) struct FilterConjunction<'x, 'a> {
    pub(crate) legs: &'x mut [TermLeg<'a>],
}

impl FilterConjunction<'_, '_> {
    /// Leapfrog to the first document at or after `doc` every leg matches.
    ///
    /// Unlike `ConjunctionDISI.doNext`, a leg already *past* `doc` counts as
    /// a mismatch rather than a match: these legs come from the batch path,
    /// which advances the non-lead legs to candidate documents on its own, so
    /// one can sit ahead of the lead when this takes over.
    fn do_next(&mut self, mut doc: i32) -> Result<i32> {
        'head: loop {
            if doc == NO_MORE_DOCS {
                return Ok(doc);
            }
            for i in 1..self.legs.len() {
                let mut other = self.legs[i].doc_id();
                if other < doc {
                    other = self.legs[i].advance(doc)?;
                }
                if other != doc {
                    doc = self.legs[0].advance(other)?;
                    continue 'head;
                }
            }
            debug_assert!(
                self.legs.iter().all(|l| l.doc_id() == doc),
                "every filter leg must be on the document it reports"
            );
            return Ok(doc);
        }
    }

    /// Positions the conjunction on its first match at or after `target`.
    pub(crate) fn align(&mut self, target: i32) -> Result<i32> {
        let lead = self.legs[0].doc_id();
        let doc = if lead < target {
            self.legs[0].advance(target)?
        } else {
            lead
        };
        self.do_next(doc)
    }
}

impl crate::exec::Scorer for FilterConjunction<'_, '_> {
    fn doc_id(&self) -> i32 {
        self.legs[0].doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        let doc = self.legs[0].next_doc()?;
        self.do_next(doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.legs[0].advance(target)?;
        self.do_next(doc)
    }
    fn cost(&self) -> i64 {
        self.legs[0].cost
    }
    fn score(&mut self) -> Result<f32> {
        Ok(0.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
}

// ---------------------------------------------------------------------------
// Required plus optional clauses: `ReqOptSumScorer`, a batch at a time.
// ---------------------------------------------------------------------------

/// `ReqOptSumScorer` over term legs, as a bulk scorer. Lucene has none for
/// this shape -- `BooleanScorerSupplier.booleanScorer()` declines `MUST` +
/// `SHOULD` and `DefaultBulkScorer` drives `ReqOptSumScorer` a document at a
/// time -- so this is the conjunction bulk scorer's batch loop with the
/// optional clauses applied the way `ReqOptSumScorer.score` applies them:
///
/// - the required legs' scores summed in `double` and narrowed
///   (`ConjunctionScorer.score`; a `FILTER` leg contributes `0`),
/// - the optional legs' scores summed in `double` and narrowed
///   (`DisjunctionSumScorer.score`),
/// - the two added in `float` (`score += optScorer.score()`).
///
/// Each window is the lead's block. A window whose required and optional
/// block maxima cannot reach the threshold is skipped outright
/// (`ReqOptSumScorer`'s impacts approximation), and a batch document whose
/// required score plus the optional maxima cannot reach it is dropped before
/// the optional legs are advanced to it (`optIsRequired`, per document). Both
/// bounds are the score's own arithmetic over the maxima, so neither can drop
/// a document that would have entered the results.
pub(crate) struct ReqOptBulk {
    single: DocScores,
    acc: DocScoreAcc,
    /// `req_sums[i]`: the block maxima of required legs `i..`, summed.
    req_sums: Vec<f64>,
    opt_sums: Vec<f64>,
    opt_hit: Vec<bool>,
    /// The optional-led path's window: which documents an optional leg
    /// matched, and their summed optional scores.
    window_matches: FixedBitSet,
    window_scores: Vec<f64>,
    req_scores: Vec<f64>,
    /// Once a threshold exists over filter-only required legs: the filtered
    /// `MaxScoreBulkScorer` the rest of the segment runs on.
    filtered: Option<MaxScore>,
}

impl ReqOptBulk {
    pub(crate) fn new(num_required: usize) -> Self {
        Self {
            single: DocScores::default(),
            acc: DocScoreAcc::default(),
            req_sums: vec![0.0; num_required + 1],
            opt_sums: Vec::new(),
            opt_hit: Vec::new(),
            window_matches: FixedBitSet::new(INNER_WINDOW_SIZE as usize),
            window_scores: vec![0.0; INNER_WINDOW_SIZE as usize],
            req_scores: Vec::new(),
            filtered: None,
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
        let nr = req.len();
        let n = nr + opt.len();
        let filters_only = req.iter().all(TermLeg::is_filter);
        let mut window_min = req[0].doc_id().max(min);
        while window_min < max {
            let mut min_competitive = min_competitive_score(collector);
            // Every required leg is a filter, so a document matching none of
            // the optional legs scores 0. Once the threshold is above 0 it
            // cannot compete, and the rest is a filtered disjunction:
            // `MaxScoreBulkScorer` with the filters as its filter, which never
            // iterates the optional legs the threshold makes non-essential.
            if filters_only && min_competitive > 0.0 && opt.len() > 1 {
                let state = self.filtered.get_or_insert_with(|| MaxScore::new(opt));
                #[cfg(test)]
                test_only_req_opt_paths::record(3);
                let mut filter = FilterConjunction { legs: req };
                filter.align(window_min)?;
                return state.score(
                    opt,
                    Some(&mut filter),
                    live_docs,
                    collector,
                    window_min,
                    max,
                );
            }
            let mut window_max = req[0].advance_shallow(window_min)?.min(max - 1);
            window_max = window_max.min(window_min.saturating_add(MAX_WINDOW_SIZE));
            let window_end = window_max.saturating_add(1);

            for leg in req.iter_mut() {
                leg.advance_shallow(window_min)?;
            }
            self.req_sums[nr] = 0.0;
            for i in (0..nr).rev() {
                self.req_sums[i] = self.req_sums[i + 1] + req[i].max_score(window_max) as f64;
            }
            let mut opt_max = 0.0f64;
            for leg in opt.iter_mut() {
                if leg.doc_id() <= window_max {
                    if leg.doc_id() < window_min {
                        leg.advance_shallow(window_min)?;
                    }
                    opt_max += leg.max_score(window_max) as f64;
                }
            }
            let opt_max_f = opt_max as f32;
            let window_bound = (self.req_sums[0] as f32) + opt_max_f;
            if min_competitive > 0.0 && window_bound < min_competitive {
                // No document in the window can compete: nothing is decoded.
                #[cfg(any(test, feature = "test-support"))]
                crate::test_only_maxscore_block_skip_counter::record_skip();
                window_min = req[0].doc_id().max(window_end);
                continue;
            }

            // Once the required clauses alone cannot reach the threshold, a
            // hit must match an optional clause too (`optIsRequired`); when
            // the optional clauses are also the cheaper side, they lead.
            let opt_cost = opt.iter().fold(0i64, |a, l| a.saturating_add(l.cost));
            if min_competitive > 0.0
                && (self.req_sums[0] as f32) < min_competitive
                && opt_cost < req[0].cost
            {
                self.score_optional_led(req, opt, live_docs, collector, window_min, window_end)?;
                window_min = req[0].doc_id().max(window_end);
                continue;
            }

            #[cfg(test)]
            test_only_req_opt_paths::record(0);
            if req[0].doc_id() < window_min {
                req[0].advance(window_min)?;
            }
            loop {
                req[0].next_docs_and_scores(window_end, live_docs, &mut self.single)?;
                if self.single.docs.is_empty() {
                    break;
                }
                let acc = &mut self.acc;
                acc.copy_from(&self.single);
                // A partial required sum is not yet the narrowed score, so
                // these filters leave two ulps of slack below the threshold.
                let slack = min_competitive.next_down().next_down().max(0.0);
                for (i, leg) in req.iter_mut().enumerate().skip(1) {
                    filter_competitive_hits(acc, self.req_sums[i] + opt_max, slack, n);
                    apply_required_clause(acc, leg)?;
                }
                // The required score is final: drop what cannot compete even
                // with every optional clause at its maximum.
                if min_competitive > 0.0 {
                    let mut k = 0;
                    for i in 0..acc.docs.len() {
                        if (acc.scores[i] as f32) + opt_max_f >= min_competitive {
                            acc.docs[k] = acc.docs[i];
                            acc.scores[k] = acc.scores[i];
                            k += 1;
                        }
                    }
                    acc.truncate(k);
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
                min_competitive = min_competitive_score(collector);
            }
            window_min = req[0].doc_id().max(window_end);
        }
        Ok(window_min)
    }

    /// One window with the optional legs leading: their documents (a batch
    /// at a time, in inner windows of `INNER_WINDOW_SIZE`) are the candidates,
    /// each dropped unless its optional score plus the required maxima can
    /// compete, and the survivors advance the required legs -- which are
    /// never decoded or scored for a document no optional leg matches.
    #[allow(clippy::too_many_arguments)]
    fn score_optional_led<C: ScoringCollector + ?Sized>(
        &mut self,
        req: &mut [TermLeg<'_>],
        opt: &mut [TermLeg<'_>],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        window_min: i32,
        window_end: i32,
    ) -> Result<()> {
        let req_max = self.req_sums[0] as f32;
        if let [only] = opt {
            #[cfg(test)]
            test_only_req_opt_paths::record(1);
            // One optional leg: its batches are the candidates, already in
            // order, with no window to merge them through.
            if only.doc_id() < window_min {
                only.advance(window_min)?;
            }
            loop {
                only.next_docs_and_scores(window_end, live_docs, &mut self.single)?;
                if self.single.docs.is_empty() {
                    return Ok(());
                }
                let min_competitive = min_competitive_score(collector);
                let acc = &mut self.acc;
                acc.docs.clear();
                acc.scores.clear();
                for (&doc, &score) in self.single.docs.iter().zip(&self.single.scores) {
                    if req_max + score >= min_competitive {
                        acc.docs.push(doc);
                        acc.scores.push(score as f64);
                    }
                }
                self.apply_required_and_collect(req, collector)?;
            }
        }
        #[cfg(test)]
        test_only_req_opt_paths::record(2);
        let mut inner_min = window_min;
        while inner_min < window_end {
            let inner_end = window_end.min(inner_min.saturating_add(INNER_WINDOW_SIZE));
            let mut any = false;
            for leg in opt.iter_mut() {
                if leg.doc_id() < inner_min {
                    leg.advance(inner_min)?;
                }
                loop {
                    leg.next_docs_and_scores(inner_end, live_docs, &mut self.single)?;
                    if self.single.docs.is_empty() {
                        break;
                    }
                    any = true;
                    for (&doc, &score) in self.single.docs.iter().zip(&self.single.scores) {
                        let i = (doc - inner_min) as usize;
                        // FBS: `next_docs_and_scores(inner_end, ..)` returns
                        // documents in `inner_min..inner_end`, a span of at
                        // most `INNER_WINDOW_SIZE`, the bitset's size.
                        self.window_matches.set(i);
                        self.window_scores[i] += score as f64;
                    }
                }
            }
            if any {
                let min_competitive = min_competitive_score(collector);
                // Candidates, ascending, with their optional sums.
                let acc = &mut self.acc;
                acc.docs.clear();
                acc.scores.clear();
                let window_scores = &mut self.window_scores;
                self.window_matches.for_each_set_bit(|i| {
                    let opt_score = window_scores[i];
                    window_scores[i] = 0.0;
                    if req_max + (opt_score as f32) >= min_competitive {
                        acc.docs.push(inner_min + i as i32);
                        acc.scores.push(opt_score);
                    }
                });
                self.window_matches.clear_all();
                self.apply_required_and_collect(req, collector)?;
            }
            inner_min = inner_end;
        }
        Ok(())
    }

    /// The candidates in `acc` (optional sums in its scores) that every
    /// required leg matches, collected with `ReqOptSumScorer`'s arithmetic.
    fn apply_required_and_collect<C: ScoringCollector + ?Sized>(
        &mut self,
        req: &mut [TermLeg<'_>],
        collector: &mut C,
    ) -> Result<()> {
        let acc = &mut self.acc;
        // Every required leg must match; their scores sum apart from the
        // optional ones, as `ReqOptSumScorer` keeps them.
        self.req_scores.clear();
        self.req_scores.resize(acc.docs.len(), 0.0);
        let mut len = acc.docs.len();
        for leg in req.iter_mut() {
            let mut cur = leg.doc_id();
            let mut k = 0;
            for i in 0..len {
                let doc = acc.docs[i];
                if cur < doc {
                    cur = leg.advance(doc)?;
                }
                if cur == doc {
                    acc.docs[k] = doc;
                    acc.scores[k] = acc.scores[i];
                    self.req_scores[k] = self.req_scores[i] + leg.score()? as f64;
                    k += 1;
                }
            }
            len = k;
        }
        for k in 0..len {
            let score = (self.req_scores[k] as f32) + (acc.scores[k] as f32);
            collector.collect(acc.docs[k], score);
        }
        Ok(())
    }
}

/// Which of `ReqOptBulk`'s paths ran, per thread, for tests that must show
/// every path is reached: `[required-led batch, optional-led with one
/// optional leg, optional-led through the window bitset, filtered MaxScore]`.
#[cfg(test)]
pub(crate) mod test_only_req_opt_paths {
    use std::cell::Cell;

    thread_local! {
        static HITS: Cell<[u64; 4]> = const { Cell::new([0; 4]) };
    }

    pub(crate) fn record(path: usize) {
        HITS.with(|h| {
            let mut v = h.get();
            v[path] += 1;
            h.set(v);
        });
    }

    pub(crate) fn take() -> [u64; 4] {
        HITS.with(|h| h.replace([0; 4]))
    }
}
