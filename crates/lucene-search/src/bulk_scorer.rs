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

use lucene_codecs::postings::{Impact, Level0Impacts, NO_MORE_DOCS};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::ScoringCollector;
use crate::field_norms::FieldNormsCursor;
use crate::similarity;
use crate::Result;

mod postings;
pub(crate) use postings::{PhrasePostings, TermPostings};

/// `MaxScoreBulkScorer.INNER_WINDOW_SIZE`.
const INNER_WINDOW_SIZE: i32 = 1 << 12;
/// `BlockMaxConjunctionBulkScorer.MAX_WINDOW_SIZE`.
const MAX_WINDOW_SIZE: i32 = 65536;

/// A window's inclusive end: the lead's block boundary `shallow`, capped at
/// [`MAX_WINDOW_SIZE`] documents -- except in the lead's last block
/// (`advanceShallow` gives `NO_MORE_DOCS`), which is one window to `max`.
/// Capped there, the windows would march 64K ids at a time through the whole
/// `int` range once a skipped window left the lead behind, each paying for its
/// block maxima: 32K empty windows a segment. The window size decides only
/// how finely the maxima prune, never what matches.
fn last_window_or_capped(shallow: i32, window_min: i32, max: i32) -> i32 {
    // ARITH: max > window_min >= 0 in every caller's loop.
    let end = shallow.min(max - 1);
    if shallow == NO_MORE_DOCS {
        end
    } else {
        end.min(window_min.saturating_add(MAX_WINDOW_SIZE))
    }
}

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
    /// `scores` narrowed to the `float` Java collects, for
    /// [`Self::collect_all`].
    narrowed: Vec<f32>,
}

impl DocScoreAcc {
    /// Every document collected at its score as a `float`, in order: the
    /// collector takes as many as it can in one batch
    /// ([`ScoringCollector::collect_many`]), the rest one by one.
    fn collect_all<C: ScoringCollector + ?Sized>(&mut self, collector: &mut C) {
        self.narrowed.clear();
        self.narrowed.extend(self.scores.iter().map(|&s| s as f32));
        let from = collector.collect_many(&self.docs, &self.narrowed, 0);
        for (&doc, &score) in self.docs[from..].iter().zip(&self.narrowed[from..]) {
            collector.collect(doc, score);
        }
    }

    fn copy_from(&mut self, src: &DocScores) {
        self.docs.clear();
        self.docs.extend_from_slice(&src.docs);
        self.scores.clear();
        self.scores.extend(src.scores.iter().map(|&s| s as f64));
    }

    /// [`Self::copy_from`] then [`filter_competitive_hits`]'s compaction at
    /// `min`, in one pass: every pair is written, and the write position
    /// moves on only past a kept one, so a batch is read once rather than
    /// copied and then read again to be filtered.
    fn copy_from_filtered(&mut self, src: &DocScores, min: f64) {
        let len = src.docs.len().min(src.scores.len());
        self.docs.resize(len, 0);
        self.scores.resize(len, 0.0);
        let mut n = 0;
        for (&d, &s) in src.docs[..len].iter().zip(&src.scores[..len]) {
            let s = s as f64;
            // `n <= i < len`: in bounds, and written before it is read again.
            self.docs[n] = d;
            self.scores[n] = s;
            n += usize::from(s >= min);
        }
        self.truncate(n);
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
    // `VectorUtil.filterByScore` (`DefaultVectorUtilSupport`): a stable
    // in-place compaction without a branch per hit -- every hit is written,
    // and the write position moves on only past a kept one. Whether a score
    // clears the bar is close to a coin flip, so a branch mispredicts.
    let n = compact_by_score(&mut acc.docs, &mut acc.scores, min_required);
    acc.truncate(n);
}

/// Keeps the pairs whose score is at least `min`, in order, at the front;
/// returns how many.
fn compact_by_score(docs: &mut [i32], scores: &mut [f64], min: f64) -> usize {
    let len = docs.len().min(scores.len());
    let mut n = 0;
    for i in 0..len {
        let (d, s) = (docs[i], scores[i]);
        // `n <= i < len`: in bounds, and written before it is read again.
        docs[n] = d;
        scores[n] = s;
        n += usize::from(s >= min);
    }
    n
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
    /// [`Self::next_docs_and_scores`] in the batches Lucene's scorer of this
    /// clause fills, where a faster batch would differ.
    fn next_docs_and_scores_exact(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut DocScores,
    ) -> Result<()> {
        self.next_docs_and_scores(up_to, live_docs, out)
    }
    /// Whether the leg matches exactly `target` (past its current doc),
    /// when it can tell without moving past it: `Some(true)` leaves it on
    /// `target`, `Some(false)` where it was. `None`: use [`Self::advance`].
    #[inline]
    fn advance_exact_cheap(&mut self, _target: i32) -> Option<bool> {
        None
    }
}

/// What [`MaxScore`] keeps per clause besides the iterator itself
/// (`DisiWrapper`'s `doc`, `cost` and `maxWindowScore`): a term leg carries
/// them inline, any other scorer in [`crate::exec`]'s `ScorerLeg`.
pub(crate) trait MaxScoreLeg: BulkLeg {
    /// `DisiWrapper.doc`: the clause's doc as the disjunction last saw it.
    fn cur(&self) -> i32;
    fn set_cur(&mut self, doc: i32);
    /// `DisiWrapper.cost`.
    fn leg_cost(&self) -> i64;
    /// `DisiWrapper.maxWindowScore`.
    fn window_max(&self) -> f32;
    fn set_window_max(&mut self, score: f32);
}

impl MaxScoreLeg for TermLeg<'_> {
    #[inline]
    fn cur(&self) -> i32 {
        self.doc
    }
    #[inline]
    fn set_cur(&mut self, doc: i32) {
        self.doc = doc;
    }
    #[inline]
    fn leg_cost(&self) -> i64 {
        self.cost
    }
    #[inline]
    fn window_max(&self) -> f32 {
        self.max_window_score
    }
    #[inline]
    fn set_window_max(&mut self, score: f32) {
        self.max_window_score = score;
    }
}

impl BulkLeg for TermLeg<'_> {
    #[inline]
    fn doc_id(&self) -> i32 {
        TermLeg::doc_id(self)
    }
    #[inline]
    fn advance_exact_cheap(&mut self, target: i32) -> Option<bool> {
        self.cursor.advance_exact_in_bits(target)
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
    cursor: TermPostings<'a>,
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
    /// `DocIdSetIterator.cost()`: the term's document frequency.
    pub(crate) cost: i64,
    /// Upper bound on any document's score, whatever the impacts say: for
    /// a BM25 term, its score at the segment's highest possible frequency
    /// and shortest field -- tighter than `MaxScoreCache.globalMaxScore`, so
    /// a threshold above it ends the term without walking its blocks.
    global_max: f32,
    /// `MaxScoreCache.globalMaxScore`, `score(Float.MAX_VALUE, 1)`: what
    /// [`Self::max_score`] answers past the last impacts level. Kept apart
    /// from [`Self::global_max`] because a disjunction's windows and
    /// partitions are computed from it, and so, past the hit threshold, are
    /// the documents the collector counts.
    max_score_global: f32,
    /// `MaxScoreCache`: up to four levels' bounds, keyed by the level's
    /// last doc id (unique per block/span, so a key match means the impacts
    /// are the same ones).
    l0_key: i32,
    l0_max: f32,
    higher_max: [(i32, f32); 3],
    /// `ImpactsDISI.minCompetitiveScore` / `upTo`.
    min_competitive: f32,
    up_to: i32,
    /// `DisiWrapper.doc`: this clause's doc as the disjunction last saw it.
    doc: i32,
    /// `DisiWrapper.maxWindowScore`.
    max_window_score: f32,
    freqs: Vec<i32>,
    norm_inv: Vec<f32>,
    /// A term scored by a similarity other than the default BM25: `Some`
    /// only for such a leg, whose `scoring` is then `false` so every BM25
    /// branch above stays exactly as it was; the non-BM25 branches consult
    /// this before falling back to [`Self::constant`].
    sim: Option<Box<SimLeg<'a>>>,
    /// A `ConstantScoreScorer` (built by [`Self::constant`]) rather than a
    /// `TermScorer`: it scores its documents in batches of 64, not a
    /// postings block at a time.
    constant_scorer: bool,
}

/// A [`TermLeg`]'s scoring state under a general
/// [`crate::similarities::SimScorer`] -- `TermScorer`'s `SimScorer` and
/// norms, and `MaxScoreCache`'s bounds for it.
struct SimLeg<'a> {
    scorer: std::sync::Arc<dyn crate::similarities::SimScorer>,
    /// The field's norms; `None` scores every document with the norm `1`,
    /// Java's value for a field without norms.
    norms: Option<FieldNormsCursor<'a, 'a>>,
    /// Scratch for a batch's frequencies and norms.
    freqs: Vec<f32>,
    norms_buf: Vec<i64>,
}

impl SimLeg<'_> {
    /// `MaxScoreCache.computeMaxScore`: the largest score any impact of a
    /// level allows, `0` at least. Without norms every document scores with
    /// the norm `1`, so the impacts' norms are read as `1` too.
    fn bound(&self, impacts: &[Impact]) -> f32 {
        let with_norms = self.norms.is_some();
        impacts.iter().fold(0.0f32, |max, impact| {
            let norm = if with_norms { impact.norm } else { 1 };
            max.max(self.scorer.score(impact.freq as f32, norm))
        })
    }

    fn norm(&mut self, doc: i32) -> Result<i64> {
        Ok(match self.norms.as_mut() {
            Some(n) => n.norm_long(doc)?.unwrap_or(1),
            None => 1,
        })
    }
}

/// `MaxScoreCache.globalMaxScore`, `score(Float.MAX_VALUE, 1)`, as a bound
/// the pruning can compare against: a `NaN` (a similarity whose formula
/// overflows there) bounds nothing, so it becomes `+inf`, which only means
/// no pruning.
pub(crate) fn sim_global_max(scorer: &dyn crate::similarities::SimScorer, max_freq: f32) -> f32 {
    let max = scorer.score(max_freq, 1);
    if max.is_nan() {
        f32::INFINITY
    } else {
        max
    }
}

/// The largest score `impacts` allow for a term of `weight` -- the same float
/// expression the document is scored with, so the bound can never land one
/// ULP under a real score.
pub(crate) fn bound(
    norm_inverse_table: Option<&[f32; 256]>,
    weight: f32,
    impacts: &[Impact],
) -> f32 {
    match norm_inverse_table {
        Some(table) => similarity::max_score_for_impacts_table(impacts, weight, table),
        None => similarity::max_score_for_impacts_unnormed_weighted(impacts, weight),
    }
}

impl<'a> TermLeg<'a> {
    /// A scoring clause. `max_freq` is the highest frequency any one document
    /// can have for this term in this segment (`totalTermFreq - docFreq + 1`),
    /// which is what makes the global bound tight enough to stop a keyword
    /// field's scan as soon as the top-k fills.
    pub(crate) fn scoring(
        cursor: impl Into<TermPostings<'a>>,
        weight: f32,
        norms: Option<FieldNormsCursor<'a, 'a>>,
        cost: i64,
        max_freq: f32,
    ) -> Self {
        let cursor = cursor.into();
        let max_norm_inverse = match &norms {
            Some(n) => n.max_norm_inverse(),
            None => similarity::UNNORMED_NORM_INVERSE,
        };
        let global_max = similarity::do_score(weight, max_freq, max_norm_inverse);
        // `scorer.score(Float.MAX_VALUE, 1L)`: the norm `1` decoded through
        // the field's table, as `BM25Scorer` does whether or not the field
        // has norms.
        let norm_one = match &norms {
            Some(n) => n.norm_inverse_table()[1],
            None => similarity::UNNORMED_NORM_INVERSE,
        };
        let max_score_global = similarity::do_score(weight, f32::MAX, norm_one);
        let docs_only_impacts = cursor.binary_docs_only_impacts();
        let mut leg = Self::new(cursor, weight, true, norms, cost, global_max);
        leg.max_score_global = max_score_global;
        if docs_only_impacts {
            // `Lucene104PostingsReader`'s impacts for a field without
            // frequencies: one level up to `NO_MORE_DOCS` holding the impact
            // `(freq 1, norm 1)` -- every document's frequency is 1, and the
            // shortest non-empty field scores highest. The norm-free global
            // bound above takes the largest norm inverse there is (an empty
            // field's), which a keyword's one-token documents never reach, so
            // a full queue could never end the scan.
            let dummy = [Impact { freq: 1, norm: 1 }];
            leg.global_max = leg.global_max.min(leg.impacts_bound(&dummy));
            leg.max_score_global = leg.global_max;
        }
        leg
    }

    /// A scoring clause under a similarity other than the default BM25:
    /// `TermScorer` over `scorer`, with `MaxScoreCache`'s bounds for it --
    /// `score(Float.MAX_VALUE, 1)` globally, the largest score of a level's
    /// impacts per block. A field without frequencies has one impact,
    /// `(1, 1)`, for everything (see [`Self::scoring`]).
    pub(crate) fn scoring_sim(
        cursor: impl Into<TermPostings<'a>>,
        scorer: std::sync::Arc<dyn crate::similarities::SimScorer>,
        norms: Option<FieldNormsCursor<'a, 'a>>,
        cost: i64,
    ) -> Self {
        let cursor = cursor.into();
        let max_freq = if cursor.binary_docs_only_impacts() {
            1.0
        } else {
            f32::MAX
        };
        let global_max = sim_global_max(scorer.as_ref(), max_freq);
        let mut leg = Self::new(cursor, 0.0, false, None, cost, global_max);
        leg.sim = Some(Box::new(SimLeg {
            scorer,
            norms,
            freqs: Vec::new(),
            norms_buf: Vec::new(),
        }));
        leg
    }

    /// A `FILTER` clause: matches gate the conjunction and contribute `0`.
    pub(crate) fn filter(cursor: impl Into<TermPostings<'a>>, cost: i64) -> Self {
        Self::new(cursor.into(), 0.0, false, None, cost, 0.0)
    }

    /// `ConstantScoreScorer` over a term's documents: every one scores
    /// `score`, and once the threshold passes `score` nothing is left to
    /// visit (the global bound is `score`, so `advance_target` ends it).
    pub(crate) fn constant(cursor: impl Into<TermPostings<'a>>, cost: i64, score: f32) -> Self {
        let mut leg = Self::new(cursor.into(), 0.0, false, None, cost, score);
        leg.constant = score;
        leg.constant_scorer = true;
        leg
    }

    fn new(
        cursor: TermPostings<'a>,
        weight: f32,
        scoring: bool,
        norms: Option<FieldNormsCursor<'a, 'a>>,
        cost: i64,
        global_max: f32,
    ) -> Self {
        Self {
            cursor,
            weight,
            scoring,
            constant: 0.0,
            norms,
            cost,
            global_max,
            max_score_global: global_max,
            l0_key: i32::MIN,
            l0_max: 0.0,
            higher_max: [(i32::MIN, 0.0); 3],
            min_competitive: 0.0,
            up_to: NO_MORE_DOCS,
            doc: -1,
            max_window_score: 0.0,
            freqs: Vec::with_capacity(lucene_codecs::postings::BLOCK_SIZE as usize + 1),
            norm_inv: Vec::with_capacity(lucene_codecs::postings::BLOCK_SIZE as usize + 1),
            sim: None,
            constant_scorer: false,
        }
    }

    #[inline]
    pub(crate) fn doc_id(&self) -> i32 {
        self.cursor.doc_id()
    }

    #[inline]
    pub(crate) fn advance(&mut self, target: i32) -> Result<i32> {
        self.cursor.advance(target)
    }

    #[inline]
    pub(crate) fn next_doc(&mut self) -> Result<i32> {
        self.cursor.next_doc()
    }

    /// `Scorer.advanceShallow`. A leg without impacts -- a filter, whose
    /// `TermScorer` reads a `SlowImpactsEnum`, or a `ConstantScoreScorer` --
    /// has a single level that runs to `NO_MORE_DOCS`, so a window bounded
    /// by it is not cut at a postings block.
    #[inline]
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        if self.is_constant() {
            return Ok(NO_MORE_DOCS);
        }
        self.cursor.advance_shallow(target)
    }

    /// The largest score a block's impacts allow -- the same float expression
    /// the document is scored with, so the bound can never land one ULP under
    /// a real score.
    fn impacts_bound(&self, impacts: &[Impact]) -> f32 {
        bound(
            self.norms
                .as_ref()
                .map(FieldNormsCursor::norm_inverse_table),
            self.weight,
            impacts,
        )
    }

    /// `MaxScoreCache.getMaxScoreForLevel(0)`; an empty level (the tail block)
    /// is Lucene's dummy `(Integer.MAX_VALUE, 1)` impact, bounded at
    /// `MaxScoreCache.globalMaxScore`.
    fn level0_max(&mut self) -> f32 {
        if !self.scoring {
            return self.other_level_max(0);
        }
        let key = self.cursor.level0_last_doc_id();
        if key != self.l0_key {
            self.refresh_level0_max(key);
        }
        self.l0_max
    }

    /// [`Self::level0_max`] for a new block. Out of line on purpose: inlined,
    /// it grew the per-block scoring loop it sits in (`next_docs_and_scores`)
    /// and cost a four-term disjunction 7%.
    #[inline(never)]
    fn refresh_level0_max(&mut self, key: i32) {
        let table = self
            .norms
            .as_ref()
            .map(FieldNormsCursor::norm_inverse_table);
        // Still encoded (the usual case: the bound is the first thing to ask
        // for a new block's impacts): folded straight off the bytes.
        let folded = match (table, self.cursor.level0_impacts_view()) {
            (Some(table), Level0Impacts::Encoded(bytes)) => Some(
                similarity::max_score_for_encoded_impacts_table(bytes, self.weight, table),
            ),
            _ => None,
        };
        let max = folded.unwrap_or_else(|| {
            let impacts = self.cursor.level0_impacts();
            (!impacts.is_empty()
                && !matches!(
                    impacts,
                    [Impact {
                        freq: i32::MAX,
                        norm: 1
                    }]
                ))
            .then(|| bound(table, self.weight, impacts))
        });
        // No impacts (the tail block): Lucene's dummy `(Integer.MAX_VALUE,
        // 1)`, which bounds the block at `MaxScoreCache.globalMaxScore`.
        self.l0_max = max.map_or(self.max_score_global, |m| m.min(self.global_max));
        self.l0_key = key;
    }

    /// [`Self::level0_max`]/[`Self::higher_level_max`] for a leg that is not the
    /// BM25 fast form: a constant's score, or `MaxScoreCache` over a general
    /// similarity's impacts (an empty level bounded by the global maximum).
    #[inline(never)]
    fn other_level_max(&mut self, level: usize) -> f32 {
        let Some(sim) = self.sim.as_deref() else {
            return self.constant;
        };
        let (key, cached) = if level == 0 {
            (self.cursor.level0_last_doc_id(), (self.l0_key, self.l0_max))
        } else {
            (
                self.cursor.doc_id_up_to(level),
                self.higher_max[level.saturating_sub(1)],
            )
        };
        if key == cached.0 {
            return cached.1;
        }
        let impacts = if level == 0 {
            self.cursor.level0_impacts()
        } else {
            self.cursor.impacts(level)
        };
        let max = if impacts.is_empty() {
            self.global_max
        } else {
            sim.bound(impacts)
        };
        if level == 0 {
            (self.l0_key, self.l0_max) = (key, max);
        } else {
            self.higher_max[level.saturating_sub(1)] = (key, max);
        }
        max
    }

    fn higher_level_max(&mut self, level: usize) -> f32 {
        if !self.scoring {
            return self.other_level_max(level);
        }
        let key = self.cursor.doc_id_up_to(level);
        if key != self.higher_max[level.saturating_sub(1)].0 {
            let table = self
                .norms
                .as_ref()
                .map(FieldNormsCursor::norm_inverse_table);
            let impacts = self.cursor.impacts(level);
            let max = if impacts.is_empty()
                || matches!(
                    impacts,
                    [Impact {
                        freq: i32::MAX,
                        norm: 1
                    }]
                ) {
                self.max_score_global
            } else {
                bound(table, self.weight, impacts).min(self.global_max)
            };
            self.higher_max[level.saturating_sub(1)] = (key, max);
        }
        self.higher_max[level.saturating_sub(1)].1
    }

    /// `Scorer.getMaxScore(upTo)` / `MaxScoreCache.getMaxScore`: a bound on
    /// every score of this clause up to and including `up_to`, from the first
    /// impacts level that covers it.
    pub(crate) fn max_score(&mut self, up_to: i32) -> f32 {
        if !self.scoring && self.sim.is_none() {
            return self.constant;
        }
        if up_to <= self.cursor.level0_last_doc_id() {
            return self.level0_max();
        }
        for level in 1..self.cursor.num_levels() {
            if up_to <= self.cursor.doc_id_up_to(level) {
                return self.higher_level_max(level);
            }
        }
        self.max_score_global
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
        let mut skip = self.cursor.level0_last_doc_id();
        for level in 1..self.cursor.num_levels() {
            if self.higher_level_max(level) >= min_score {
                return skip;
            }
            skip = self.cursor.doc_id_up_to(level);
        }
        skip
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
    pub(crate) fn advance_target(&mut self, mut target: i32) -> Result<i32> {
        if target <= self.up_to {
            return Ok(target);
        }
        // Nothing anywhere can compete: `MaxScoreCache.globalMaxScore` bounds
        // every block Lucene's walk would visit (the tail's dummy impact
        // included), so it answers without walking a single block header.
        if self.max_score_global < self.min_competitive {
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

    /// `TermScorer.nextDoc` as the scorer tree runs it: through
    /// `ImpactsDISI` once a threshold is set, plainly before.
    pub(crate) fn tree_next_doc(&mut self) -> Result<i32> {
        if self.min_competitive > 0.0 {
            self.impacts_next_doc()
        } else {
            self.next_doc()
        }
    }

    /// `TermScorer.advance(target)` as the scorer tree runs it.
    pub(crate) fn tree_advance(&mut self, target: i32) -> Result<i32> {
        if self.min_competitive > 0.0 {
            self.impacts_advance(target)
        } else {
            self.advance(target)
        }
    }

    /// `ImpactsDISI.nextDoc`.
    pub(crate) fn impacts_next_doc(&mut self) -> Result<i32> {
        let doc = self.doc_id();
        if doc < self.up_to {
            return self.next_doc();
        }
        self.impacts_advance(doc.saturating_add(1))
    }

    /// A `ConstantScoreScorer`, built by [`Self::constant`].
    pub(crate) fn is_constant_scorer(&self) -> bool {
        self.constant_scorer
    }

    /// Every score is [`Self::constant`]: scoring reads nothing.
    pub(crate) fn is_constant(&self) -> bool {
        !self.scoring && self.sim.is_none()
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
        // No threshold yet: every block competes, so `advanceTarget` would
        // return `doc` after reading the next block's impacts for nothing.
        if doc == NO_MORE_DOCS || self.min_competitive <= 0.0 {
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
            return self.other_score();
        }
        let doc = self.cursor.doc_id();
        let freq = self.cursor.freq().unwrap_or(1) as f32;
        let norm_inverse = match self.norms.as_mut() {
            Some(n) => n.norm_inverse(doc)?,
            None => similarity::UNNORMED_NORM_INVERSE,
        };
        Ok(similarity::do_score(self.weight, freq, norm_inverse))
    }

    /// [`Self::score`] for a leg that is not the BM25 fast form:
    /// `simScorer.score(freq, norm)` under a general similarity, the
    /// constant otherwise.
    #[inline(never)]
    fn other_score(&mut self) -> Result<f32> {
        let doc = self.cursor.doc_id();
        let freq = self.cursor.freq().unwrap_or(1) as f32;
        let Some(sim) = self.sim.as_deref_mut() else {
            return Ok(self.constant);
        };
        let norm = sim.norm(doc)?;
        Ok(sim.scorer.score(freq, norm))
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
        if self.constant_scorer {
            return self.constant_docs_and_scores(up_to, usize::MAX, live_docs, out);
        }
        self.next_docs_and_scores_where(up_to, live_docs, |_| Ok(true), out)
    }

    /// [`Self::next_docs_and_scores`] over at most `max` of the postings'
    /// documents (live or not; `max >= 1`), the cursor left on the next.
    pub(crate) fn next_docs_and_scores_at_most(
        &mut self,
        up_to: i32,
        max: usize,
        live_docs: Option<&FixedBitSet>,
        out: &mut DocScores,
    ) -> Result<()> {
        if self.constant_scorer {
            return self.constant_docs_and_scores(up_to, max, live_docs, out);
        }
        self.next_docs_and_scores_limited::<false>(up_to, max, live_docs, |_| Ok(true), out)
    }

    /// `ConstantScoreScorer.nextDocsAndScores`: up to 64 live documents
    /// below `up_to`, stepping the iterator one document at a time, so an
    /// empty batch means `up_to` was reached. At most `max` documents are
    /// stepped over, unless none of them was live.
    fn constant_docs_and_scores(
        &mut self,
        up_to: i32,
        max: usize,
        live_docs: Option<&FixedBitSet>,
        out: &mut DocScores,
    ) -> Result<()> {
        const BATCH_SIZE: usize = 64;
        out.docs.clear();
        let mut doc = self.doc_id();
        let mut stepped = 0usize;
        while doc < up_to && out.docs.len() < BATCH_SIZE && (stepped < max || out.docs.is_empty()) {
            if live_docs.is_none_or(|l| l.get_doc(doc)) {
                out.docs.push(doc);
            }
            doc = self.tree_next_doc()?;
            stepped = stepped.saturating_add(1);
        }
        out.scores.clear();
        out.scores.resize(out.docs.len(), self.constant);
        Ok(())
    }

    /// [`Self::next_docs_and_scores`] keeping only the documents `accept`
    /// takes, dropped before they are scored: what a conjunction with
    /// non-scoring clauses scores (`exec::Bulk::FilteredTerm`).
    /// `accept` is asked about each document once, in ascending order. An
    /// empty buffer still means there are no more documents below `up_to`.
    pub(crate) fn next_docs_and_scores_where(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        accept: impl FnMut(i32) -> Result<bool>,
        out: &mut DocScores,
    ) -> Result<()> {
        self.next_docs_and_scores_limited::<true>(up_to, usize::MAX, live_docs, accept, out)
    }

    /// `FILTERED` is false when `accept` takes every document: then a batch
    /// with no deletions to drop is scored as read, without the compaction
    /// pass copying every document and frequency onto itself.
    fn next_docs_and_scores_limited<const FILTERED: bool>(
        &mut self,
        up_to: i32,
        max: usize,
        live_docs: Option<&FixedBitSet>,
        mut accept: impl FnMut(i32) -> Result<bool>,
        out: &mut DocScores,
    ) -> Result<()> {
        loop {
            self.ensure_competitive()?;
            self.cursor
                .next_postings_at_most(up_to, max, &mut out.docs, &mut self.freqs)?;
            if out.docs.is_empty() {
                break;
            }
            if !FILTERED && live_docs.is_none() {
                break;
            }
            // Branch-free compaction, as `filterCompetitiveHits`: every
            // document is written and the write position moves on only past
            // a kept one. A filter such as a cached range keeps a third of a
            // block's documents at random, so a branch on it mispredicted
            // over half of `+term #range`'s time. Deleted documents still
            // short-circuit, so `accept` sees each live document once.
            let mut n = 0;
            for i in 0..out.docs.len() {
                let d = out.docs[i];
                let keep = live_docs.is_none_or(|l| l.get_doc(d)) && (!FILTERED || accept(d)?);
                out.docs[n] = d;
                self.freqs[n] = self.freqs[i];
                n += usize::from(keep);
            }
            out.docs.truncate(n);
            self.freqs.truncate(n);
            // A whole batch of rejected documents is not the end of the
            // postings: fetch the next one, as Java loops.
            if n > 0 {
                break;
            }
        }
        self.score_batch(out)
    }

    /// Scores `out.docs` (with their frequencies in `self.freqs`) into
    /// `out.scores`: `TermScorer.nextDocsAndScores`'s second half.
    pub(crate) fn score_batch(&mut self, out: &mut DocScores) -> Result<()> {
        out.scores.clear();
        if !self.scoring {
            if let Some(sim) = self.sim.as_deref_mut() {
                // `TermScorer.nextDocsAndScores`: the norms in bulk, then
                // `BulkSimScorer.score` over the batch.
                match sim.norms.as_mut() {
                    Some(n) => n.norm_long_batch(&out.docs, &mut sim.norms_buf)?,
                    None => {
                        sim.norms_buf.clear();
                        sim.norms_buf.resize(out.docs.len(), 1);
                    }
                }
                sim.freqs.clear();
                sim.freqs.extend(self.freqs.iter().map(|&f| f as f32));
                sim.scorer
                    .score_bulk(&sim.freqs, &sim.norms_buf, &mut out.scores);
                return Ok(());
            }
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
///
/// A candidate the leg's current block answers by itself (a bit set) is
/// tested there, without moving past it: a miss leaves the leg behind the
/// candidate. Once the buffer is done the leg is advanced to its last
/// candidate if it is still behind it -- Lucene's position at that point too
/// (the first posting at or after the last candidate it had to advance for,
/// which is at or after the last candidate), so every caller sees the same
/// state.
fn apply_required_clause<L: BulkLeg + ?Sized>(acc: &mut DocScoreAcc, leg: &mut L) -> Result<()> {
    let Some(&last) = acc.docs.last() else {
        return Ok(());
    };
    let mut n = 0;
    let mut cur = leg.doc_id();
    for i in 0..acc.docs.len() {
        let target = acc.docs[i];
        if cur < target {
            match leg.advance_exact_cheap(target) {
                Some(true) => cur = target,
                Some(false) => continue,
                None => cur = leg.advance(target)?,
            }
        }
        if cur == target {
            acc.docs[n] = target;
            acc.scores[n] = acc.scores[i] + leg.score()? as f64;
            n += 1;
        }
    }
    acc.truncate(n);
    if leg.doc_id() < last {
        leg.advance(last)?;
    }
    Ok(())
}

/// `ScorerUtil.applyOptionalClause`: add `leg`'s score to every buffered
/// document it matches, keeping the rest.
fn apply_optional_clause<L: BulkLeg + ?Sized>(acc: &mut DocScoreAcc, leg: &mut L) -> Result<()> {
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
        // Until a threshold is published every document is collected; the
        // collector takes those in one batch, up to where one could be. The
        // per-document test `score >= 0` passes every score a term produces:
        // a non-negative boost times a positive idf and tf.
        debug_assert!(buf.scores.iter().all(|s| *s >= 0.0));
        let mut from = 0;
        if min_competitive == 0.0 {
            from = collector.collect_many(&buf.docs, &buf.scores, 0);
            if from > 0 {
                min_competitive = min_competitive_score(collector);
            }
        }
        for (&doc, &score) in buf.docs[from..].iter().zip(&buf.scores[from..]) {
            if score >= min_competitive {
                collector.collect(doc, score);
                min_competitive = min_competitive_score(collector);
            }
        }
        leg.set_min_competitive_score(min_competitive);
    }
}

/// A term with non-scoring clauses (`+term #filter...`): what Lucene scores
/// as `DefaultBulkScorer` over a `ConjunctionScorer` whose one scoring
/// clause takes the minimum competitive score (and skips blocks by it), a
/// batch at a time instead of a document at a time. Each batch of the term's
/// postings is narrowed to the documents `filter` matches -- by membership
/// when it can answer that, else leapfrogging it forward -- before they are
/// scored; the rest is [`score_term_window`]. The documents collected, their
/// scores and their order are the conjunction's; past the hit threshold,
/// where the total is a lower bound, fewer non-competitive ones are counted.
/// `filter` must not be two-phase.
pub(crate) fn score_filtered_term_window<C: ScoringCollector + ?Sized>(
    leg: &mut TermLeg<'_>,
    filter: &mut dyn crate::exec::Scorer,
    buf: &mut DocScores,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
    min: i32,
    max: i32,
) -> Result<i32> {
    debug_assert!(!filter.two_phase());
    let mut min_competitive = min_competitive_score(collector);
    leg.set_min_competitive_score(min_competitive);
    if leg.doc_id() < min {
        leg.advance(min)?;
    }
    // A filter held as a bit set (a cached or dense clause) is tested inline,
    // a bit per document, rather than through a virtual `contains` per
    // document -- which was a quarter of a term filtered by a cached range.
    // Taken once per window: nothing here sets the filter's threshold, so it
    // cannot be emptied meanwhile.
    let bits = filter.constant_bits();
    let bits = bits.as_ref().and_then(|(set, _)| match &**set {
        crate::exec::cache::CachedSet::Bits { bits, .. } => Some(bits),
        crate::exec::cache::CachedSet::Docs(_) => None,
    });
    loop {
        if let Some(bits) = bits {
            leg.next_docs_and_scores_where(max, live_docs, |d| Ok(bits.get_doc(d)), buf)?;
        } else {
            leg.next_docs_and_scores_where(
                max,
                live_docs,
                |d| {
                    Ok(match filter.contains(d) {
                        Some(m) => m,
                        None => {
                            let at = filter.doc_id();
                            (if at < d { filter.advance(d)? } else { at }) == d
                        }
                    })
                },
                buf,
            )?;
        }
        if buf.docs.is_empty() {
            return Ok(leg.doc_id());
        }
        let mut from = 0;
        if min_competitive == 0.0 {
            from = collector.collect_many(&buf.docs, &buf.scores, 0);
            if from > 0 {
                min_competitive = min_competitive_score(collector);
            }
        }
        for (&doc, &score) in buf.docs[from..].iter().zip(&buf.scores[from..]) {
            if score >= min_competitive {
                collector.collect(doc, score);
                min_competitive = min_competitive_score(collector);
            }
        }
        leg.set_min_competitive_score(min_competitive);
    }
}

/// A term with a two-phase filter over every document (`+term #frange`,
/// `+term #FunctionMatchQuery`): Lucene's `ConjunctionScorer` under
/// `DefaultBulkScorer`, the filter confirming a run of the term's documents
/// in one call ([`crate::exec::Scorer::matches_batch`]) instead of one
/// virtual `matches()` each.
///
/// The documents visited, collected, counted and skipped are the
/// document-at-a-time loop's. That loop moves the term as `ImpactsDISI`
/// does: a plain `nextDoc` inside the impacts block it last checked, a
/// check of the next block against the threshold past it, and -- once a
/// hit raises the threshold (`setMinCompetitiveScore` resets `upTo`) -- a
/// check of the block holding the next document. So a run never leaves
/// the term's current level-0 block and is read before any such check;
/// every match is collected (and counted) as the loop would, and a hit
/// that raises the threshold runs the check the next step would run, which
/// drops the rest of the run when its block can no longer compete. Scores
/// are computed for the confirmed documents only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_term_two_phase_window<C: ScoringCollector + ?Sized>(
    leg: &mut TermLeg<'_>,
    filter: &mut dyn crate::exec::Scorer,
    run: &mut TwoPhaseRun,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
    min: i32,
    max: i32,
) -> Result<i32> {
    debug_assert!(filter.batch_matches());
    // `publish` before the first document.
    let mut min_competitive = min_competitive_score(collector);
    leg.set_min_competitive_score(min_competitive);
    let mut doc = leg.doc_id();
    if doc < min {
        doc = if doc == min.saturating_sub(1) {
            leg.tree_next_doc()?
        } else {
            leg.tree_advance(min)?
        };
    }
    while doc < max {
        // The run: from `doc` through the end of its level-0 block (or the
        // batch's room, or `max`), stepping as `nextDoc` inside a block does.
        let block_end = leg.shallow_advance(doc)?;
        // The run's live documents and their frequencies; `last` the last
        // term document read into it (live or not).
        run.docs.clear();
        run.freqs.clear();
        // Whether the cursor stepped (inside the block, plainly) to a
        // document at or past `max`, which the run does not hold.
        let mut past = false;
        let mut last;
        let mut read = 0usize;
        loop {
            if live_docs.is_none_or(|l| l.get_doc(doc)) {
                run.docs.push(doc);
                run.freqs.push(leg.cursor.freq().unwrap_or(1));
            }
            last = doc;
            read += 1;
            if doc >= block_end || read >= RUN {
                break;
            }
            doc = leg.next_doc()?;
            if doc >= max {
                past = true;
                break;
            }
        }
        // The last term document the run read: a check after a hit at or
        // after it is the next step's own.
        let read_to = if past { doc } else { last };
        // Confirmed by the filter in one call, then scored as a batch
        // (norms read forward, once each).
        run.keep.clear();
        run.keep.resize(run.docs.len(), false);
        if !run.docs.is_empty() {
            filter.matches_batch(&run.docs, &mut run.keep)?;
        }
        run.out.docs.clear();
        leg.freqs.clear();
        for ((&d, &f), &k) in run.docs.iter().zip(&run.freqs).zip(&run.keep) {
            if k {
                run.out.docs.push(d);
                leg.freqs.push(f);
            }
        }
        leg.score_batch(&mut run.out)?;
        let mut skipped_to = None;
        for (&d, &score) in run.out.docs.iter().zip(&run.out.scores) {
            collector.collect(d, score);
            let m = min_competitive_score(collector);
            if m > min_competitive {
                min_competitive = m;
                leg.set_min_competitive_score(m);
                // The next step would check the block of `d + 1` -- this
                // run's block, which the cursor has not left -- against the
                // new threshold. Any term document after `d` in the run is
                // that next step's.
                if d < read_to {
                    let target = leg.advance_target(d.saturating_add(1))?;
                    if target != d.saturating_add(1) {
                        skipped_to = Some(target);
                        break;
                    }
                }
            }
        }
        doc = match skipped_to {
            // The rest of the run is in a block that can no longer compete:
            // `advance(target)`, past the cursor.
            Some(target) => leg.advance(target)?,
            None if past => doc,
            None => leg.tree_next_doc()?,
        };
    }
    Ok(doc)
}

/// The longest run [`score_term_two_phase_window`] reads before confirming
/// it (a level-0 block is 128 documents).
const RUN: usize = 128;

/// [`score_term_two_phase_window`]'s buffers.
#[derive(Debug, Default)]
pub(crate) struct TwoPhaseRun {
    docs: Vec<i32>,
    freqs: Vec<i32>,
    keep: Vec<bool>,
    out: DocScores,
}

/// `ReqOptSumScorer(req, opt)` under `DefaultBulkScorer`, where `req` is a
/// constant-scoring bit set (score `c`, a cached clause) and `opt` one
/// scoring term: every document of `req` scores `c`, plus the term's score
/// where the term matches too -- `float score = reqScore; score +=
/// optScore`.
///
/// Every match scored (`COMPLETE`): `ReqOptSumScorer`'s approximation is
/// then the required clause's iterator, so every document of `req` is
/// visited in order and the term advanced to it. Under `TOP_SCORES` the
/// scorer tree runs instead (`exec::Bulk::ReqBitsOpt` is not built there):
/// the collector's lower-bound total counts exactly the documents
/// `ReqOptSumScorer`'s impacts approximation hands it.
pub(crate) fn score_req_bits_opt_term<C: ScoringCollector + ?Sized>(
    set: &crate::exec::cache::CachedSet,
    c: f32,
    leg: &mut TermLeg<'_>,
    live_docs: Option<&FixedBitSet>,
    collector: &mut C,
    min: i32,
    max: i32,
) -> Result<i32> {
    debug_assert!(
        collector.pruning_threshold().is_none(),
        "ReqBitsOpt is exhaustive only"
    );
    // `Bulk::ReqBitsOpt` is only built from `constant_bits()`, which answers
    // for a bit set alone.
    debug_assert!(
        matches!(set, crate::exec::cache::CachedSet::Bits { .. }),
        "ReqBitsOpt over a sparse cached set"
    );
    let crate::exec::cache::CachedSet::Bits { bits, .. } = set else {
        return Ok(NO_MORE_DOCS);
    };
    let words = bits.words();
    let len = bits.len();
    let next_bit = |from: i32| -> i32 {
        let from = usize::try_from(from).unwrap_or(0);
        match lucene_util::fixed_bit_set::next_set_bit_in_words(words, from) {
            Some(i) if i < len => i as i32,
            _ => NO_MORE_DOCS,
        }
    };
    let mut doc = next_bit(min);
    while doc < max {
        if live_docs.is_none_or(|l| l.get_doc(doc)) {
            let mut score = c;
            if leg.doc_id() < doc {
                leg.advance(doc)?;
            }
            if leg.doc_id() == doc {
                score += leg.score()?;
            }
            collector.collect(doc, score);
            #[cfg(test)]
            test_only_req_bits_opt_docs::record();
        }
        doc = next_bit(doc.saturating_add(1));
    }
    Ok(doc)
}

// ---------------------------------------------------------------------------
// Optional terms with minimum_should_match > 1: a window at a time.
// ---------------------------------------------------------------------------

/// A disjunction of terms where at least `msm > 1` must match, scored.
/// Lucene runs `WANDScorer` (with `minShouldMatch`) under
/// `DefaultBulkScorer` here, a candidate at a time; this reads every term's
/// postings a decoded block at a time into a window of
/// [`INNER_WINDOW_SIZE`] documents, counting each document's matching
/// clauses and summing their scores in `f64`, then collects the documents
/// matching at least `msm` in order, each scoring its sum narrowed to `f32`
/// -- `WANDScorer.score`'s arithmetic (the order the `f64` sum takes does not
/// change a sum of a few BM25 `f32`s, as for [`DisMaxBulk`]). Once a
/// threshold exists, a window whose clause maxima cannot reach it, or where
/// fewer than `msm` clauses have a document, is skipped without decoding.
/// Until then every match is collected, so an exact total is WAND's; past it
/// (a lower bound) the count can differ.
pub(crate) struct MsmBulk {
    msm: u32,
    matches: FixedBitSet,
    counts: Vec<u32>,
    sums: Vec<f64>,
    buf: DocScores,
}

/// `ScorerUtil.costWithMinShouldMatch`: the summed costs of the
/// `costs.len() - msm + 1` cheapest clauses.
#[cfg_attr(not(any(test, feature = "asserting")), allow(dead_code))]
pub(crate) fn cost_with_min_should_match(costs: impl Iterator<Item = i64>, msm: usize) -> i64 {
    let mut costs: Vec<i64> = costs.collect();
    costs.sort_unstable();
    let keep = (costs.len() + 1).saturating_sub(msm);
    costs[..keep.min(costs.len())]
        .iter()
        .fold(0i64, |c, &x| c.saturating_add(x))
}

impl MsmBulk {
    /// The `minimum_should_match` it scores.
    #[cfg_attr(not(any(test, feature = "asserting")), allow(dead_code))]
    pub(crate) fn msm(&self) -> usize {
        self.msm as usize
    }

    pub(crate) fn new(msm: usize) -> Self {
        MsmBulk {
            msm: u32::try_from(msm).unwrap_or(u32::MAX),
            matches: FixedBitSet::new(INNER_WINDOW_SIZE as usize),
            counts: vec![0; INNER_WINDOW_SIZE as usize],
            sums: vec![0.0; INNER_WINDOW_SIZE as usize],
            buf: DocScores::default(),
        }
    }

    /// `BulkScorer.score(collector, acceptDocs, min, max)`; returns the next
    /// document to score.
    pub(crate) fn score<C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [TermLeg<'_>],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        let msm = self.msm as usize;
        let mut window_min = min;
        loop {
            let mut start = NO_MORE_DOCS;
            for leg in legs.iter_mut() {
                let mut doc = leg.doc_id();
                if doc < window_min {
                    doc = leg.advance(window_min)?;
                }
                start = start.min(doc);
            }
            if start >= max {
                return Ok(start);
            }
            let end = max.min(start.saturating_add(INNER_WINDOW_SIZE));
            let min_competitive = min_competitive_score(collector);
            // The clauses with a document in the window, and (with a
            // threshold) the sum of their maxima there.
            let mut present = 0usize;
            let mut bound = 0.0f64;
            for leg in legs.iter_mut() {
                let doc = leg.doc_id();
                if doc < end {
                    present += 1;
                    if min_competitive > 0.0 {
                        leg.advance_shallow(doc)?;
                        bound += f64::from(leg.max_score(end - 1));
                    }
                }
            }
            if present < msm || (min_competitive > 0.0 && (bound as f32) < min_competitive) {
                #[cfg(any(test, feature = "test-support"))]
                crate::test_only_maxscore_block_skip_counter::record_skip();
                window_min = end;
                continue;
            }
            for leg in legs.iter_mut() {
                if leg.doc_id() >= end {
                    continue;
                }
                loop {
                    leg.next_docs_and_scores(end, live_docs, &mut self.buf)?;
                    if self.buf.docs.is_empty() {
                        break;
                    }
                    for (&doc, &sub) in self.buf.docs.iter().zip(&self.buf.scores) {
                        // ARITH: `start <= doc < end <= start + INNER_WINDOW_SIZE`.
                        let i = (doc - start) as usize;
                        // FBS: `matches` holds `INNER_WINDOW_SIZE` bits, `i` below it.
                        self.matches.set(i);
                        self.counts[i] += 1;
                        self.sums[i] += f64::from(sub);
                    }
                }
            }
            let (counts, sums, want) = (&mut self.counts, &mut self.sums, self.msm);
            let mut threshold = min_competitive;
            self.matches.for_each_set_bit(|i| {
                if counts[i] >= want {
                    let score = sums[i] as f32;
                    if score >= threshold {
                        // ARITH: `i < INNER_WINDOW_SIZE`, `start + i < end`.
                        collector.collect(start + i as i32, score);
                        threshold = min_competitive_score(collector);
                    }
                }
                counts[i] = 0;
                sums[i] = 0.0;
            });
            self.matches.clear_all();
            window_min = end;
        }
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
            let shallow = legs[0].advance_shallow(window_min)?;
            let window_max = last_window_or_capped(shallow, window_min, max);

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
                        acc.collect_all(collector);
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
// A dismax of terms, a window at a time.
// ---------------------------------------------------------------------------

/// A `DisjunctionMaxQuery` over term legs, a window of
/// `INNER_WINDOW_SIZE` documents at a time.
///
/// With a tie-breaker of 0 under `TOP_SCORES` this is
/// `DisjunctionMaxBulkScorer`: each clause runs as its own
/// `BatchScoreBulkScorer` over the window -- skipping its own
/// non-competitive blocks and keeping only the documents whose score
/// reaches the threshold as the window began -- and the window's best
/// scores are collected in doc-id order. Any other tie-breaker (or score
/// mode) is `DefaultBulkScorer` over `DisjunctionMaxScorer` in Lucene, which
/// never skips: every match is collected, so is every match here, a window
/// at a time rather than a document at a time.
///
/// Scores are `DisjunctionMaxScorer.score`'s, bit for bit: per document, the
/// clauses fold in as `score` folds its top list (`>=` promotes to the max,
/// the loser joins the `f64` sum of the others), then `(float)(max + others *
/// tie)`. The fold visits clauses in query order rather than the heap's
/// order; the max is the same either way, and the others' `f64` sum of at
/// most a few `f32`s is exact unless their magnitudes differ by more than
/// 2^28, which BM25 scores do not.
pub(crate) struct DisMaxBulk {
    tie: f32,
    matches: FixedBitSet,
    max: Vec<f32>,
    others: Vec<f64>,
    buf: DocScores,
    /// `topLevelScorable.minCompetitiveScore`: `0` until the collector is
    /// first handed the scorable, after this leaf's first window.
    published: f32,
}

impl DisMaxBulk {
    pub(crate) fn new(tie: f32) -> Self {
        DisMaxBulk {
            tie,
            matches: FixedBitSet::new(INNER_WINDOW_SIZE as usize),
            max: vec![0.0; INNER_WINDOW_SIZE as usize],
            others: vec![0.0; INNER_WINDOW_SIZE as usize],
            buf: DocScores::default(),
            published: 0.0,
        }
    }

    /// `BulkScorer.score(collector, acceptDocs, min, max)`; returns the next
    /// document to score.
    pub(crate) fn score<C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [TermLeg<'_>],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        // Only `DisjunctionMaxBulkScorer` hands its clauses the threshold.
        let prune = self.tie == 0.0;
        loop {
            // `scorers.top().next`: where the earliest clause stands.
            let next = legs
                .iter()
                .map(TermLeg::doc_id)
                .min()
                .unwrap_or(NO_MORE_DOCS);
            if next >= max {
                return Ok(next);
            }
            let start = next.max(min);
            let window_max = max.min(start.saturating_add(INNER_WINDOW_SIZE));
            // `topLevelScorable.minCompetitiveScore` as the window begins:
            // nothing the clauses collect into the window changes it.
            let min_competitive = if prune { self.published } else { 0.0 };
            for leg in legs.iter_mut() {
                if leg.doc_id() >= window_max {
                    continue;
                }
                // The clause's `BatchScoreBulkScorer.score(innerCollector,
                // acceptDocs, windowMin, windowMax)`.
                if prune {
                    leg.set_min_competitive_score(min_competitive);
                }
                if leg.doc_id() < start {
                    leg.impacts_advance(start)?;
                }
                // A `ConstantScoreScorer` runs under `DefaultBulkScorer`,
                // which keeps every document its iterator returns.
                let keep_all = leg.is_constant_scorer();
                loop {
                    leg.next_docs_and_scores(window_max, live_docs, &mut self.buf)?;
                    if self.buf.docs.is_empty() {
                        break;
                    }
                    for (&doc, &sub) in self.buf.docs.iter().zip(&self.buf.scores) {
                        if !keep_all && sub < min_competitive {
                            continue;
                        }
                        let i = (doc - start) as usize;
                        // FBS: `matches` is `FixedBitSet::new(INNER_WINDOW_SIZE)`, and
                        // every doc here is in `start..window_max`, a span of at most
                        // `INNER_WINDOW_SIZE`.
                        self.matches.set(i);
                        let score_max = self.max[i];
                        if sub >= score_max {
                            self.others[i] += f64::from(score_max);
                            self.max[i] = sub;
                        } else {
                            self.others[i] += f64::from(sub);
                        }
                    }
                }
            }
            let tie = f64::from(self.tie);
            let (max_scores, others) = (&mut self.max, &mut self.others);
            self.matches.for_each_set_bit(|i| {
                let score = (f64::from(max_scores[i]) + others[i] * tie) as f32;
                max_scores[i] = 0.0;
                others[i] = 0.0;
                collector.collect(start + i as i32, score);
            });
            self.matches.clear_all();
            // `collector.setScorer(topLevelScorable)` before the window's
            // hits publishes the threshold so far, and each hit what it
            // raises it to.
            self.published = min_competitive_score(collector);
        }
    }
}

// ---------------------------------------------------------------------------
// A disjunction: MaxScoreBulkScorer.
// ---------------------------------------------------------------------------

/// `MaxScoreBulkScorer`'s per-query state, minus the clauses themselves.
pub(crate) struct MaxScore {
    /// Each leg's `cur()`, kept beside the order: the essential-clause scan
    /// (`essentialQueue`'s top) reads these contiguously.
    docs: Vec<i32>,
    /// `essentialQueue`: the essential legs as a binary min-heap on `docs`
    /// (ties by leg index), rebuilt when the partition changes. Only the top
    /// ever moves between rebuilds, so a move is one sift down -- where a
    /// scan of every essential clause per document was most of a 50-term
    /// fuzzy query.
    heap: Vec<usize>,
    /// Indices into the legs, in `allScorers` order: non-essential first.
    order: Vec<usize>,
    scratch: Vec<usize>,
    /// [`Self::partition_scorers`]' sort keys, paired with their leg.
    keyed: Vec<(f64, usize)>,
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
    /// `filterMatches`: the filter's documents in the current inner window,
    /// when [`Self::with_filter`] chose the bit-set path over leapfrogging.
    filter_matches: Option<FixedBitSet>,
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
        .score::<_, _, dyn crate::exec::Scorer>(legs, None, live_docs, collector, 0, NO_MORE_DOCS)
        .map(|_| ())
}

impl MaxScore {
    pub(crate) fn new<L: MaxScoreLeg>(legs: &mut [L]) -> Self {
        let n = legs.len();
        for leg in legs.iter_mut() {
            leg.set_cur(leg.doc_id());
        }
        MaxScore {
            docs: legs.iter().map(MaxScoreLeg::cur).collect(),
            heap: Vec::with_capacity(n),
            order: (0..n).collect(),
            scratch: Vec::with_capacity(n),
            keyed: Vec::with_capacity(n),
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
            filter_matches: None,
        }
    }

    /// The constructor's filter heuristic: a filter without two phases, on
    /// a segment of at least one inner window, is read a window at a time
    /// into a bit set when the sparsest clause is no sparser than it, or
    /// when more than four clauses together are no sparser -- the candidate
    /// stream is then dense enough that bulk gating beats an `advance` per
    /// candidate. Otherwise the filter leapfrogs with the essential clauses.
    pub(crate) fn with_filter<L: MaxScoreLeg>(
        mut self,
        legs: &[L],
        filter_cost: i64,
        filter_two_phase: bool,
        max_doc: Option<i32>,
    ) -> Self {
        let big_enough = max_doc.is_some_and(|m| m >= INNER_WINDOW_SIZE);
        if !filter_two_phase && big_enough && !legs.is_empty() {
            let min_cost = legs.iter().map(MaxScoreLeg::leg_cost).min().unwrap_or(0);
            let cost = legs
                .iter()
                .map(MaxScoreLeg::leg_cost)
                .fold(0i64, i64::saturating_add);
            if min_cost >= filter_cost || (legs.len() > 4 && cost >= filter_cost) {
                self.filter_matches = Some(FixedBitSet::new(INNER_WINDOW_SIZE as usize));
            }
        }
        self
    }

    /// `MaxScoreBulkScorer.score(collector, acceptDocs, min, max)`: the
    /// disjunction's documents in `[min, max)` -- those also matching
    /// `filter`, when there is one (`filteredOptionalBulkScorer`). Returns the
    /// next document to score.
    pub(crate) fn score<
        L: MaxScoreLeg,
        C: ScoringCollector + ?Sized,
        F: crate::exec::Scorer + ?Sized,
    >(
        &mut self,
        legs: &mut [L],
        mut filter: Option<&mut F>,
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        let ms = self;
        // The legs may have moved since the last call (a reused state).
        ms.docs.clear();
        ms.docs.extend(legs.iter().map(MaxScoreLeg::cur));
        ms.min_competitive = min_competitive_score(collector);
        let filter_cost = filter.as_deref().map(|f| f.cost());
        let mut outer_min = min;
        'outer: while outer_min < max {
            let mut outer_max = ms
                .compute_outer_window_max(legs, filter_cost, outer_min)?
                .min(max);
            loop {
                ms.update_max_window_scores(legs, outer_min, outer_max)?;
                if !ms.partition_scorers(legs) {
                    // No clause can compete anywhere in this window.
                    #[cfg(any(test, feature = "test-support"))]
                    crate::test_only_maxscore_block_skip_counter::record_skip();
                    outer_min = outer_max;
                    continue 'outer;
                }
                let new_max = ms.compute_outer_window_max(legs, filter_cost, outer_min)?;
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
                if legs[i].cur() < outer_min {
                    let d = legs[i].advance(outer_min)?;
                    legs[i].set_cur(d);
                    ms.docs[i] = d;
                }
            }
            ms.build_heap();
            while ms.top().1 < outer_max {
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
            outer_min = ms.top().1.min(outer_max);
            ms.num_outer_windows += 1;
        }
        Ok(Self::next_candidate(legs, max))
    }

    /// `nextCandidate(rangeEnd)`.
    fn next_candidate<L: MaxScoreLeg>(legs: &[L], range_end: i32) -> i32 {
        let mut next = NO_MORE_DOCS;
        for leg in legs {
            if leg.cur() < range_end {
                return range_end;
            }
            next = next.min(leg.cur());
        }
        next
    }

    /// `scoreInnerWindowWithFilter` by leapfrog (`fillScoreBufferViaLeapFrog`):
    /// the essential clauses' documents that the filter also matches, summed,
    /// then the non-essential clauses as usual.
    fn score_inner_window_with_filter<
        L: MaxScoreLeg,
        C: ScoringCollector + ?Sized,
        F: crate::exec::Scorer + ?Sized,
    >(
        &mut self,
        legs: &mut [L],
        filter: &mut F,
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        max: i32,
    ) -> Result<()> {
        let (mut top, mut top_doc) = self.top();
        let mut filter_doc = filter.doc_id();
        while top_doc < filter_doc {
            let d = legs[top].advance(filter_doc)?;
            legs[top].set_cur(d);
            self.docs[top] = legs[top].cur();
            self.sift_top();
            (top, top_doc) = self.top();
        }
        if top_doc >= max {
            return Ok(());
        }
        let inner_max = max.min(top_doc.saturating_add(INNER_WINDOW_SIZE));
        if self.filter_matches.is_some() {
            return self.fill_via_bit_set(legs, filter, live_docs, collector, top_doc, inner_max);
        }
        self.acc.docs.clear();
        self.acc.scores.clear();
        while top_doc < inner_max {
            if filter_doc < top_doc {
                filter_doc = filter.advance(top_doc)?;
            }
            if filter_doc != top_doc {
                while top_doc < filter_doc {
                    let d = legs[top].advance(filter_doc)?;
                    legs[top].set_cur(d);
                    self.docs[top] = legs[top].cur();
                    self.sift_top();
                    (top, top_doc) = self.top();
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
                    let d = legs[top].next_doc()?;
                    legs[top].set_cur(d);
                    self.docs[top] = legs[top].cur();
                    self.sift_top();
                    (top, top_doc) = self.top();
                }
                if matched {
                    self.acc.docs.push(doc);
                    self.acc.scores.push(score);
                }
            }
        }
        self.score_non_essential(legs, collector)
    }

    /// `fillScoreBufferViaBitSet`: the filter's documents in
    /// `[inner_min, inner_max)` into `filterMatches` (`intoBitSet`), deleted
    /// ones masked out, then every essential clause drained into the window
    /// as `collectEssentialScoresIntoWindow` drains it, keeping only the
    /// documents the filter set.
    fn fill_via_bit_set<
        L: MaxScoreLeg,
        C: ScoringCollector + ?Sized,
        F: crate::exec::Scorer + ?Sized,
    >(
        &mut self,
        legs: &mut [L],
        filter: &mut F,
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        inner_min: i32,
        inner_max: i32,
    ) -> Result<()> {
        let Some(mut bits) = self.filter_matches.take() else {
            return Ok(());
        };
        bits.clear_all();
        let mut doc = filter.doc_id();
        if doc < inner_max {
            if doc < inner_min {
                doc = filter.advance(inner_min)?;
            }
            // `DocIdSetIterator.intoBitSet`'s default loop.
            while doc < inner_max {
                // FBS: `inner_min <= doc < inner_max <= inner_min + INNER_WINDOW_SIZE`,
                // the bit set's length.
                bits.set((doc - inner_min) as usize);
                doc = filter.next_doc()?;
            }
        }
        if let Some(live) = live_docs {
            // `acceptDocs.applyMask(filterMatches, innerWindowMin)`.
            let mut i = bits.next_set_bit(0);
            while let Some(b) = i {
                if !live.get_doc(inner_min.wrapping_add(b as i32)) {
                    // FBS: `b` came from `next_set_bit` on `bits`.
                    bits.clear(b);
                }
                i = bits.next_set_bit(b + 1);
            }
        }
        loop {
            let (top, top_doc) = self.top();
            if top_doc >= inner_max {
                break;
            }
            loop {
                legs[top].next_docs_and_scores(inner_max, None, &mut self.single)?;
                if self.single.docs.is_empty() {
                    break;
                }
                for (&doc, &score) in self.single.docs.iter().zip(&self.single.scores) {
                    let i = (doc - inner_min) as usize;
                    // FBS: `next_docs_and_scores(inner_max, ..)` only returns docs in
                    // `inner_min..inner_max`, at most `INNER_WINDOW_SIZE` apart, and
                    // both bit sets are that long.
                    if !bits.get(i) {
                        continue;
                    }
                    // FBS: as above.
                    self.window_matches.set(i);
                    self.window_scores[i] += score as f64;
                }
            }
            legs[top].set_cur(legs[top].doc_id());
            self.docs[top] = legs[top].cur();
            self.sift_top();
        }
        self.filter_matches = Some(bits);
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

    /// The essential clause with the smallest doc, and that doc; `NO_MORE_DOCS`
    /// when every essential clause is exhausted.
    fn top(&self) -> (usize, i32) {
        self.heap
            .first()
            .map_or((usize::MAX, NO_MORE_DOCS), |&i| (i, self.docs[i]))
    }

    /// The second-smallest essential doc (`essentialQueue.top2()`), or `None`
    /// with a single essential clause: the smaller of the top's children.
    fn top2_doc(&self, _top: usize) -> Option<i32> {
        self.heap[1..self.heap.len().min(3)]
            .iter()
            .map(|&i| self.docs[i])
            .min()
    }

    /// Whether heap slot `a` orders before slot `b`.
    fn heap_less(&self, a: usize, b: usize) -> bool {
        let (x, y) = (self.heap[a], self.heap[b]);
        (self.docs[x], x) < (self.docs[y], y)
    }

    fn sift_down(&mut self, mut at: usize) {
        let n = self.heap.len();
        loop {
            // ARITH: `at < n <= legs.len()`, far below `usize::MAX / 2`.
            #[allow(clippy::arithmetic_side_effects)]
            let (l, r) = (2 * at + 1, 2 * at + 2);
            let mut least = at;
            if l < n && self.heap_less(l, least) {
                least = l;
            }
            if r < n && self.heap_less(r, least) {
                least = r;
            }
            if least == at {
                return;
            }
            self.heap.swap(at, least);
            at = least;
        }
    }

    /// The top's doc moved forward: restore the heap.
    fn sift_top(&mut self) {
        self.sift_down(0);
    }

    /// The essential legs, as the partition left them, heapified.
    fn build_heap(&mut self) {
        self.heap.clear();
        self.heap
            .extend_from_slice(&self.order[self.first_essential..]);
        for at in (0..self.heap.len() / 2).rev() {
            self.sift_down(at);
        }
    }

    /// `computeOuterWindowMax`: with a filter (of cost `filter_cost`), only
    /// the clauses at least as costly as the filter bound the window. The
    /// windows decide when the partition is recomputed, so past the hit
    /// threshold they decide which documents the collector counts.
    fn compute_outer_window_max<L: MaxScoreLeg>(
        &mut self,
        legs: &mut [L],
        filter_cost: Option<i64>,
        window_min: i32,
    ) -> Result<i32> {
        let n = legs.len();
        let first_window_lead = self.first_essential.min(n - 1);
        let mut window_max = NO_MORE_DOCS;
        for k in first_window_lead..n {
            let i = self.order[k];
            if filter_cost.is_some_and(|c| legs[i].leg_cost() < c) {
                continue;
            }
            let target = legs[i].cur().max(window_min);
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

    fn update_max_window_scores<L: MaxScoreLeg>(
        &mut self,
        legs: &mut [L],
        window_min: i32,
        window_max: i32,
    ) -> Result<()> {
        for leg in legs.iter_mut() {
            if leg.cur() < window_max {
                if leg.cur() < window_min {
                    leg.advance_shallow(window_min)?;
                }
                let m = leg.max_score(window_max - 1)?;
                leg.set_window_max(m);
            } else {
                leg.set_window_max(0.0);
            }
        }
        Ok(())
    }

    fn partition_scorers<L: MaxScoreLeg>(&mut self, legs: &[L]) -> bool {
        let n = legs.len();
        // Stable, like Java's `Arrays.sort` over objects. The keys are worked
        // out once per leg, not once per comparison: with a wide disjunction
        // (a fuzzy expansion's fifty terms) re-deriving them inside the
        // comparator was a tenth of the query.
        self.keyed.clear();
        self.keyed.extend(self.order.iter().map(|&w| {
            (
                legs[w].window_max() as f64 / legs[w].leg_cost().max(1) as f64,
                w,
            )
        }));
        self.keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.scratch.clear();
        self.scratch.extend(self.keyed.iter().map(|&(_, w)| w));
        let mut max_score_sum = 0.0f64;
        self.first_essential = 0;
        self.next_min_competitive = f32::INFINITY;
        for i in 0..n {
            let w = self.scratch[i];
            let new_sum = max_score_sum + legs[w].window_max() as f64;
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
            let mut max_required = legs[self.order[self.first_essential]].window_max() as f64;
            while self.first_required > 0 {
                let mut without_previous = max_required;
                if self.first_required > 1 {
                    without_previous += self.max_score_sums[self.first_required - 2];
                }
                if (without_previous as f32) >= self.min_competitive {
                    break;
                }
                self.first_required -= 1;
                max_required += legs[self.order[self.first_required]].window_max() as f64;
            }
        }
        true
    }

    fn score_inner_window<L: MaxScoreLeg, C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [L],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        max: i32,
    ) -> Result<()> {
        let (top, top_doc) = self.top();
        match self.top2_doc(top) {
            None => self.score_single_essential(legs, live_docs, collector, top, max),
            Some(top2) if top2.saturating_sub(INNER_WINDOW_SIZE / 2) >= top_doc => {
                // The first half of the window only matches one clause: stream
                // it up to the next clause's doc.
                self.score_single_essential(legs, live_docs, collector, top, max.min(top2))
            }
            Some(_) => self.score_multiple_essential(legs, live_docs, collector, top_doc, max),
        }
    }

    fn score_single_essential<L: MaxScoreLeg, C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [L],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        top: usize,
        up_to: i32,
    ) -> Result<()> {
        loop {
            // The batches decide when the threshold below is re-read, so they
            // are Lucene's: `nextDocsAndScores`' own, whatever the clause.
            legs[top].next_docs_and_scores_exact(up_to, live_docs, &mut self.single)?;
            if self.single.docs.is_empty() {
                break;
            }
            // The first non-essential clause's filter, folded into the copy.
            let candidates = self.single.docs.len();
            let prefiltered = self.first_essential > 0;
            if prefiltered {
                let min_required = min_required_score(
                    self.max_score_sums[self.first_essential - 1],
                    self.min_competitive,
                    legs.len(),
                );
                if min_required > 0.0 {
                    self.acc.copy_from_filtered(&self.single, min_required);
                } else {
                    self.acc.copy_from(&self.single);
                }
            } else {
                self.acc.copy_from(&self.single);
            }
            self.score_non_essential_from(legs, collector, candidates, prefiltered)?;
        }
        legs[top].set_cur(legs[top].doc_id());
        self.docs[top] = legs[top].cur();
        self.sift_top();
        Ok(())
    }

    fn score_multiple_essential<L: MaxScoreLeg, C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [L],
        live_docs: Option<&FixedBitSet>,
        collector: &mut C,
        inner_min: i32,
        max: i32,
    ) -> Result<()> {
        let inner_max = max.min(inner_min.saturating_add(INNER_WINDOW_SIZE));
        // `collectEssentialScoresIntoWindow`: drain every essential clause's
        // documents below `inner_max` into the window bitset.
        loop {
            let (top, top_doc) = self.top();
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
            legs[top].set_cur(legs[top].doc_id());
            self.docs[top] = legs[top].cur();
            self.sift_top();
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

    fn score_non_essential<L: MaxScoreLeg, C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [L],
        collector: &mut C,
    ) -> Result<()> {
        let candidates = self.acc.docs.len();
        self.score_non_essential_from(legs, collector, candidates, false)
    }

    /// `scoreNonEssentialClauses` over the buffered `candidates`, the first
    /// filter already applied when `prefiltered` (it is idempotent: the same
    /// threshold over the same hits keeps the same ones).
    fn score_non_essential_from<L: MaxScoreLeg, C: ScoringCollector + ?Sized>(
        &mut self,
        legs: &mut [L],
        collector: &mut C,
        candidates: usize,
        prefiltered: bool,
    ) -> Result<()> {
        self.num_candidates += candidates as i64;
        let n = legs.len();
        for k in (0..self.first_essential).rev() {
            let i = self.order[k];
            if !(prefiltered && k + 1 == self.first_essential) {
                filter_competitive_hits(
                    &mut self.acc,
                    self.max_score_sums[k],
                    self.min_competitive,
                    n,
                );
            }
            if k >= self.first_required {
                apply_required_clause(&mut self.acc, &mut legs[i])?;
            } else {
                apply_optional_clause(&mut self.acc, &mut legs[i])?;
            }
            legs[i].set_cur(legs[i].doc_id());
            self.docs[i] = legs[i].cur();
        }
        self.acc.collect_all(collector);
        self.min_competitive = min_competitive_score(collector);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Required plus optional clauses, every match: `ReqOptSumScorer`, a batch at
// a time.
// ---------------------------------------------------------------------------

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

/// Documents [`score_req_bits_opt_term`] collected, per thread.
#[cfg(test)]
pub(crate) mod test_only_req_bits_opt_docs {
    use std::cell::Cell;

    thread_local! {
        static HITS: Cell<u64> = const { Cell::new(0) };
    }

    pub(crate) fn record() {
        HITS.with(|h| h.set(h.get() + 1));
    }

    pub(crate) fn take() -> u64 {
        HITS.with(|h| h.replace(0))
    }
}

#[cfg(test)]
mod compact_tests {
    use super::compact_by_score;

    /// The documents scoring at least `min`, in order, each beside its own
    /// score -- a score equal to `min` kept -- and nothing from an empty or
    /// wholly non-competitive batch.
    #[test]
    fn compact_by_score_keeps_the_competitive_hits_in_order() {
        let mut docs = vec![1, 2, 3, 4, 5, 6];
        let mut scores = vec![0.5, 2.0, 1.0, 0.99, 3.0, 1.0];
        let n = compact_by_score(&mut docs, &mut scores, 1.0);
        assert_eq!(n, 4);
        assert_eq!(&docs[..n], &[2, 3, 5, 6]);
        assert_eq!(&scores[..n], &[2.0, 1.0, 3.0, 1.0]);
        let (mut d, mut s) = (vec![7], vec![0.1]);
        assert_eq!(compact_by_score(&mut d, &mut s, 1.0), 0);
        assert_eq!(compact_by_score(&mut [], &mut [], 0.0), 0);
    }

    /// The fused copy keeps exactly what copying then compacting keeps, in
    /// order, whatever the accumulator held before -- longer, shorter or
    /// empty.
    #[test]
    fn copy_from_filtered_is_copy_then_compact() {
        use crate::bulk_scorer::{DocScoreAcc, DocScores};
        let src = DocScores {
            docs: vec![3, 8, 9, 15, 40, 41],
            scores: vec![1.5, 0.25, 2.0, 1.0, 0.999, 7.0],
        };
        for min in [0.0, 1.0, 1.5, 8.0] {
            for prior in [0usize, 2, 10] {
                let mut fused = DocScoreAcc {
                    docs: vec![-1; prior],
                    scores: vec![-1.0; prior],
                    ..DocScoreAcc::default()
                };
                fused.copy_from_filtered(&src, min);
                let mut plain = DocScoreAcc::default();
                plain.copy_from(&src);
                let n = compact_by_score(&mut plain.docs, &mut plain.scores, min);
                plain.truncate(n);
                assert_eq!(fused.docs, plain.docs, "min {min}, prior {prior}");
                assert_eq!(fused.scores, plain.scores, "min {min}, prior {prior}");
            }
        }
    }
}
