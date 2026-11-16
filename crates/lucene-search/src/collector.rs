//! `Collector`/`LeafCollector`-equivalent, pared down to this slice's scope:
//! a plain callback trait invoked once per matching (live) doc ID, in
//! ascending order. Real Lucene's `Collector`/`LeafCollector` split exists to
//! let a collector rebind per-segment state (e.g. a `Scorer` reference) when
//! `IndexSearcher` moves to the next leaf; that split has no work to do yet
//! since this slice never federates more than one segment (see `lib.rs`'s
//! module doc), so a single flat trait stands in for both.
//!
//! `search_term_query` (`lib.rs`) is generic over `C: Collector` rather than
//! taking `&mut dyn Collector`, per the `rust-performance` skill's
//! "monomorphize per-doc loops, `dyn` only at Query/Weight level" rule — the
//! per-doc `collect()` call in the hot loop is a direct (inlinable) call, not
//! a vtable dispatch.
//!
//! ## `ScoringCollector` — a new trait, not a breaking
//! change to `Collector`
//!
//! `lib.rs`'s module doc, written when this file only had unscored matching,
//! already flagged that relevance scoring would need `Collector::collect` to
//! grow a `score: f32` parameter, and called that "a breaking signature
//! change, every existing `Collector` impl's signature changes". Having now
//! reached that point, this port takes the **non-breaking path instead**: a
//! separate [`ScoringCollector`] trait with its own `collect(doc_id, score)`
//! method, rather than editing [`Collector`] in place. Reasoning:
//!
//! - **Not every caller needs a score.** `CountCollector`/`VecCollector` (and
//!   `search_term_query`/`search_boolean_query`, which only ever determine
//!   matching, not ranking) have no use for a `score: f32` parameter — real
//!   Lucene's own `Collector`/`LeafCollector` doesn't force `TotalHitCountCollector`
//!   to touch a `Scorer` either (`LeafCollector.setScorer` is a no-op there).
//!   Forcing a score parameter onto every collector would make the two
//!   existing, already-shipped, already-tested unscored collectors either grow
//!   a dummy parameter or get deleted for no correctness reason.
//! - **A trait per shape, not one trait doing double duty.** `Collector` and
//!   `ScoringCollector` are different contracts (`fn(i32)` vs `fn(i32, f32)`);
//!   giving them different trait names (as opposed to one trait with both
//!   methods, one of them defaulted to a no-op) keeps each collector's impl
//!   exactly as small as the contract it actually fulfills, and keeps
//!   [`crate::search_term_query`]/[`crate::search_boolean_query`]'s existing generic bound
//!   (`C: Collector`) untouched — no existing caller's code breaks.
//! - **The cost is one more trait, not a hierarchy.** With exactly two shapes
//!   (unscored / scored) and no third on the horizon, this is the same
//!   "don't build the trait hierarchy until a second real shape needs it"
//!   call `lib.rs`'s module doc already made for `Weight`/`Scorer` — here the
//!   second shape *has* arrived, so it gets its own trait, but no further
//!   speculative generality (no shared supertrait, no `Collector: ScoringCollector`
//!   blanket impl) is introduced beyond that.

use std::sync::{Arc, OnceLock};

/// `org.apache.lucene.search.ScoreMode` -- what a collector needs from the
/// scorers underneath it, and therefore what work those scorers are allowed to
/// skip.
///
/// Real Lucene threads this from `Collector.scoreMode()` through
/// `Query.createWeight(searcher, scoreMode, boost)` into every `Weight`, and it
/// decides two independent things:
///
/// - **`needs_scores`** -- whether a score is ever asked for. A `Weight` built
///   for a mode with `needs_scores == false` may build a
///   `PostingsEnum` without frequencies at all (`PostingsEnum.DOCS`), and
///   `Lucene104PostingsReader` then `PForUtil.skip`s the entire frequency block
///   rather than decoding it. That is the concrete cost of not modelling this:
///   a filter-only clause in this port still decodes every frequency it will
///   never read.
/// - **`is_exhaustive`** -- whether the collector needs *every* match, or only
///   enough of them to fill a top-`n`. Dynamic pruning (`WANDScorer`,
///   `MaxScoreCache`-driven block skipping, `Scorable.setMinCompetitiveScore`)
///   is legal **only** in a non-exhaustive mode; in an exhaustive one a skipped
///   block would change `totalHits`, which the collector is promising to report
///   exactly.
///
/// This port models the enum and the two predicates faithfully, and uses them
/// where it has the machinery to act on them ([`TopDocsCollector`] reports its
/// own mode, and returns no competitive threshold in an exhaustive one, so
/// MAXSCORE block skipping shuts off).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreMode {
    /// Every match, with scores -- `IndexSearcher.count`-style exhaustive
    /// scoring. No pruning is legal.
    Complete,
    /// Every match, no scores -- a pure filter/count. No pruning is legal, and
    /// frequencies need never be decoded.
    CompleteNoScores,
    /// Top-`n` by score: pruning is legal, and scores are needed.
    TopScores,
    /// Top-`n` by something other than score (a field sort): pruning is legal
    /// and scores are not needed.
    TopDocs,
    /// Top-`n` by a non-score key that nevertheless reports scores.
    TopDocsWithScores,
}

impl ScoreMode {
    /// `ScoreMode.needsScores()`.
    pub fn needs_scores(self) -> bool {
        matches!(
            self,
            ScoreMode::Complete | ScoreMode::TopScores | ScoreMode::TopDocsWithScores
        )
    }

    /// `ScoreMode.isExhaustive()` -- `true` iff the consumer requires every
    /// match, which is exactly when dynamic pruning is **not** allowed.
    pub fn is_exhaustive(self) -> bool {
        matches!(self, ScoreMode::Complete | ScoreMode::CompleteNoScores)
    }
}

/// `TotalHits.Relation`: whether a reported hit count is exact, or a lower
/// bound because the search stopped early.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TotalHitsRelation {
    /// The count is the exact number of matches.
    EqualTo,
    /// The count is a lower bound -- pruning skipped documents that were never
    /// counted.
    GreaterThanOrEqualTo,
}

/// `org.apache.lucene.search.TotalHits`: a hit count plus whether it is exact.
///
/// Real `TopDocs` always carries one of these, and every caller that prints
/// "about N results" is reading its [`TotalHitsRelation`]. `top_docs().len()`
/// is only the *kept* hit count (capped at `top_n`) and says nothing about how
/// many documents matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TotalHits {
    pub value: u64,
    pub relation: TotalHitsRelation,
}

impl std::fmt::Display for TotalHits {
    /// `TotalHits.toString()`: `"12 hits"` / `"1000+ hits"`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{} hits",
            self.value,
            match self.relation {
                TotalHitsRelation::EqualTo => "",
                TotalHitsRelation::GreaterThanOrEqualTo => "+",
            }
        )
    }
}

/// Called once per matching, live doc ID, in ascending order — the entire
/// contract a collector needs for this slice (no scores, no per-segment
/// rebinding, no early-termination signal yet; see module doc).
pub trait Collector {
    fn collect(&mut self, doc_id: i32);
}

/// Called once per matching, live doc ID, in ascending-by-doc-ID order, with
/// that document's relevance score attached — the scored sibling of
/// [`Collector`] (see this module's doc comment for why it's a separate trait
/// rather than a breaking change to `Collector`).
pub trait ScoringCollector {
    fn collect(&mut self, doc_id: i32, score: f32);

    /// The score a document must beat to enter the results, once enough have
    /// been collected for that to be knowable. `None` means "everything is
    /// still competitive", which is the only safe answer for a collector that
    /// keeps every hit.
    ///
    /// Exposed on the trait so block-max pruning can be written once against
    /// any collector rather than only against [`TopDocsCollector`]. A collector
    /// that returns `None` simply disables pruning, which is always correct.
    fn min_competitive_score(&self) -> Option<f32> {
        None
    }

    /// `Collector.scoreMode()`: what this collector needs from the scorer, and
    /// therefore what the scorer may skip. Defaults to
    /// [`ScoreMode::Complete`] -- the conservative answer, which forbids
    /// pruning -- so a collector that has not thought about it cannot
    /// accidentally authorize an early exit.
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::Complete
    }

    /// The threshold a scorer may prune against: [`Self::min_competitive_score`],
    /// but only when this collector's [`ScoreMode`] permits pruning at all.
    ///
    /// This is the one place the two halves meet, and it exists so they cannot
    /// be consulted separately. In Java the decision is made once, at
    /// `Query.createWeight(searcher, scoreMode, boost)` time, and a `Weight`
    /// built for an exhaustive mode simply never receives a
    /// `setMinCompetitiveScore` call. This port has no `Weight` to make it at,
    /// so every scoring loop asks here instead — and asking
    /// `min_competitive_score()` directly is the bug this prevents: a collector promising an exact `totalHits` would
    /// still hand out a threshold and still get its blocks skipped.
    fn pruning_threshold(&self) -> Option<f32> {
        if self.score_mode().is_exhaustive() {
            None
        } else {
            self.min_competitive_score()
        }
    }

    /// How many more hits, all at one constant score and in ascending doc-id
    /// order, fill this collector to the point where no later document can
    /// enter it -- or `None` when that is not a fixed number (an exhaustive
    /// score mode, a `searchAfter` page, a threshold shared across leaves, a
    /// collector that keeps everything). A constant-score union uses it to
    /// stop reading postings once it has that many documents; see
    /// `cutoff_constant_score_union`. The default, `None`, is always correct.
    fn constant_score_hits_needed(&self) -> Option<u64> {
        None
    }

    /// Counts `n` more hits at `score` without collecting them one by one --
    /// allowed only when every one of them would lose to the kept hits
    /// anyway: the queue full, its worst hit scoring at least `score`, and
    /// the hits arriving after it in doc-id order so a tie loses too. Returns
    /// `false`, having counted nothing, whenever that cannot be proved; the
    /// caller then collects them itself. A constant-scored bit-set walk uses
    /// it to count the hits between a full queue and the `totalHits`
    /// threshold by popcount. The default, `false`, is always correct.
    fn count_losing_hits(&mut self, _n: u64, _score: f32) -> bool {
        false
    }

    /// Counts `n` more hits without collecting them one by one -- allowed
    /// only for a collector whose [`Self::collect`] does nothing but count
    /// (`TotalHitCountCollector`), where it is exactly `n` calls to it.
    /// Returns `false`, having counted nothing, for any other; a caller asks
    /// with `n == 0` before counting. A constant bit-set walk that reads no
    /// scores uses it to count a window by popcount. The default, `false`, is
    /// always correct.
    fn add_hits(&mut self, _n: u64) -> bool {
        false
    }

    /// [`Self::collect`] over a prefix of `docs` (each shifted by
    /// `doc_base`) and their `scores`, stopping before the first document
    /// that could change [`Self::pruning_threshold`]; returns how many were
    /// collected, which the caller follows with per-document calls. For a
    /// scoring loop that consults the threshold between documents, a batch
    /// of documents that cannot move it is collected the same either way.
    /// The default, `0`, is always correct.
    fn collect_many(&mut self, _docs: &[i32], _scores: &[f32], _doc_base: i32) -> usize {
        0
    }

    /// `LeafCollector.collectRange(min, max)`: every document in
    /// `[min, max)` matches, each scoring `score` (a constant-score bulk
    /// scorer's run of consecutive matches, `DenseConjunctionBulkScorer`'s
    /// `collectRange`). The default is Java's: [`Self::collect`] for each
    /// document in order, so a collector that does not override it collects
    /// exactly what the document-at-a-time loop would.
    #[inline]
    fn collect_range(&mut self, min: i32, max: i32, score: f32) {
        for doc in min..max {
            self.collect(doc, score);
        }
    }
}

/// `Collector.getLeafCollector(context)` for a collector shared by every
/// segment of a sequential search: forwards to `inner`, shifting each local doc
/// id by the segment's `doc_base`.
///
/// This is what lets a later segment start from the threshold the earlier ones
/// built. A fresh collector per segment starts every leaf at "everything is
/// competitive" and makes it fill its own top-k before a single block can be
/// skipped -- over a 15-segment index that cost most term and boolean queries
/// 2-4x against Lucene, which searches leaves in order through one
/// `TopScoreDocCollector`. Leaves must be visited in ascending `doc_base`
/// order: the ties-go-to-the-lower-doc-id rule the pruning threshold relies on
/// (a document scoring exactly the published threshold -- the worst kept hit --
/// cannot enter, so scorers skip anything whose bound is `<=` it) then holds
/// globally, since every doc of a later leaf has a higher global id.
pub struct LeafCollector<'c, C: ?Sized> {
    inner: &'c mut C,
    doc_base: i32,
}

impl<'c, C: ScoringCollector + ?Sized> LeafCollector<'c, C> {
    pub fn new(inner: &'c mut C, doc_base: i32) -> Self {
        Self { inner, doc_base }
    }
}

impl<C: ScoringCollector + ?Sized> ScoringCollector for LeafCollector<'_, C> {
    #[inline]
    fn collect(&mut self, doc_id: i32, score: f32) {
        // ARITH: `doc_base + doc_id` is a global doc id, below the reader's
        // `max_doc`, itself an `i32`.
        #[allow(clippy::arithmetic_side_effects)]
        self.inner.collect(self.doc_base + doc_id, score);
    }

    #[inline]
    fn min_competitive_score(&self) -> Option<f32> {
        self.inner.min_competitive_score()
    }

    #[inline]
    fn score_mode(&self) -> ScoreMode {
        self.inner.score_mode()
    }

    #[inline]
    fn pruning_threshold(&self) -> Option<f32> {
        self.inner.pruning_threshold()
    }

    #[inline]
    fn constant_score_hits_needed(&self) -> Option<u64> {
        self.inner.constant_score_hits_needed()
    }

    #[inline]
    fn count_losing_hits(&mut self, n: u64, score: f32) -> bool {
        self.inner.count_losing_hits(n, score)
    }

    #[inline]
    fn add_hits(&mut self, n: u64) -> bool {
        // A count is the same whatever the documents' base.
        self.inner.add_hits(n)
    }

    #[inline]
    fn collect_many(&mut self, docs: &[i32], scores: &[f32], doc_base: i32) -> usize {
        self.inner
            .collect_many(docs, scores, self.doc_base.saturating_add(doc_base))
    }

    #[inline]
    fn collect_range(&mut self, min: i32, max: i32, score: f32) {
        // ARITH: as in `collect`: both ends are global doc ids at most the
        // reader's `max_doc`, itself an `i32`.
        #[allow(clippy::arithmetic_side_effects)]
        self.inner
            .collect_range(self.doc_base + min, self.doc_base + max, score);
    }
}

/// OpenSearch's `MinimumScoreCollector` (`min_score`): only documents scoring
/// at least `min` reach `inner` -- counted, kept, aggregated. It needs scores:
/// `TOP_SCORES` when `inner` prunes by score (the threshold `inner` publishes
/// still holds, since a document below `min` never enters it), `COMPLETE`
/// otherwise.
pub struct MinScoreCollector<'c, C: ?Sized> {
    inner: &'c mut C,
    min: f32,
}

impl<'c, C: ScoringCollector + ?Sized> MinScoreCollector<'c, C> {
    pub fn new(inner: &'c mut C, min: f32) -> Self {
        Self { inner, min }
    }
}

impl<C: ScoringCollector + ?Sized> ScoringCollector for MinScoreCollector<'_, C> {
    #[inline]
    fn collect(&mut self, doc_id: i32, score: f32) {
        // `scorer.score() >= minimumScore`: a NaN score never passes.
        if score >= self.min {
            self.inner.collect(doc_id, score);
        }
    }

    #[inline]
    fn min_competitive_score(&self) -> Option<f32> {
        self.inner.min_competitive_score()
    }

    #[inline]
    fn score_mode(&self) -> ScoreMode {
        if self.inner.score_mode() == ScoreMode::TopScores {
            ScoreMode::TopScores
        } else {
            ScoreMode::Complete
        }
    }
}

/// Collects every matching doc ID into a `Vec<i32>`, ascending — the
/// `TopDocs`-shaped "give me the actual hits" collector.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VecCollector {
    pub docs: Vec<i32>,
}

impl Collector for VecCollector {
    fn collect(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }
}

/// `TotalHitCountCollector`-equivalent: counts matches without retaining doc
/// IDs, for callers that only need "how many docs match".
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CountCollector {
    pub count: i32,
}

impl Collector for CountCollector {
    fn collect(&mut self, _doc_id: i32) {
        self.count += 1;
    }
}

/// One scored hit: `ScoreDoc`-equivalent (`org.apache.lucene.search.ScoreDoc`),
/// minus the `shardIndex` field (meaningless — this port has no multi-shard
/// federation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreDoc {
    pub doc_id: i32,
    pub score: f32,
}

/// Real Lucene's ranking order: higher score first, and — verified against
/// `HitQueue.lessThan` (`org.apache.lucene.search.HitQueue`, not assumed —
/// `hitA.score == hitB.score ? hitA.doc > hitB.doc : hitA.score < hitB.score`,
/// i.e. on an exact score tie the *lower* doc ID is considered the better hit)
/// — **lower doc ID wins a score tie**. Returns `Ordering::Greater` when `a`
/// should rank ahead of `b`.
pub(crate) fn rank_order(a: &ScoreDoc, b: &ScoreDoc) -> std::cmp::Ordering {
    match a.score.total_cmp(&b.score) {
        std::cmp::Ordering::Equal => b.doc_id.cmp(&a.doc_id),
        other => other,
    }
}

/// A score as `DocScoreEncoder` stores it: `Float.floatToIntBits`
/// canonicalizes every NaN to `0x7fc00000`, the positive quiet NaN, so in
/// Java's heap a NaN of either sign ranks above `+inf` and reads back as that
/// one NaN (x86 produces a *negative* NaN at run time, arm64 a positive one;
/// Java hides the difference and so must this port). Every other value is
/// returned unchanged.
#[inline]
pub(crate) fn canonical_score(score: f32) -> f32 {
    if score.is_nan() {
        f32::NAN
    } else {
        score
    }
}

/// `DocScoreEncoder` (`org.apache.lucene.search.DocScoreEncoder`): a
/// `(doc, score)` pair packed into one `i64` whose **natural integer order is
/// the ranking order** -- higher score first, and on a score tie the lower doc
/// id first. That is what lets [`MaxScoreAccumulator`] be a single atomic
/// maximum rather than a lock around two fields.
///
/// ```text
/// encode(doc, score) = (floatToSortableInt(score) as i64) << 32 | (i32::MAX - doc)
/// ```
///
/// The score goes in the high half so it dominates the comparison, and the doc
/// id is stored *complemented* so that a smaller doc id makes a larger code.
mod doc_score_encoder {
    use lucene_util::numeric_utils::sortable_float_bits;

    pub(super) fn encode(doc_id: i32, score: f32) -> i64 {
        // `Float.floatToIntBits`, not `floatToRawIntBits`: every NaN is the
        // canonical positive one, so a NaN sorts above `+inf` whatever its
        // sign bit (see [`super::canonical_score`]).
        let sortable = sortable_float_bits(super::canonical_score(score).to_bits() as i32);
        // `(long) sortableInt << 32 | (Integer.MAX_VALUE - docId)`: the low
        // half is masked to 32 bits by the `|`, exactly as Java's widening of
        // an `int` operand does.
        ((sortable as i64) << 32) | (i64::from(i32::MAX.wrapping_sub(doc_id)) & 0xffff_ffff)
    }

    pub(super) fn to_score(code: i64) -> f32 {
        f32::from_bits(sortable_float_bits((code >> 32) as i32) as u32)
    }

    pub(super) fn doc_id(code: i64) -> i32 {
        i32::MAX.wrapping_sub(code as i32)
    }
}

/// `MaxScoreAccumulator` (`org.apache.lucene.search.MaxScoreAccumulator`): one
/// min-competitive score shared by every leaf a query is searched across
/// concurrently.
///
/// ## Why it exists
///
/// A per-leaf `TopScoreDocCollector` can only prune against **its own** queue,
/// so a leaf that has seen nothing competitive yet visits every document even
/// when another leaf has already filled its queue with far better hits. Sharing
/// one "worst hit worth keeping" across the leaves lets each of them start
/// pruning as soon as *any* of them can. Java creates one per
/// `TopScoreDocCollectorManager` whenever the search is concurrent, and passes
/// `null` when it is not.
///
/// ## Why a single atomic
///
/// Because [`doc_score_encoder`] packs the pair so that integer order *is*
/// ranking order, "the best competitive hit any leaf has proved" is just a
/// maximum, and a maximum over an `i64` is a compare-and-swap loop with no
/// lock and no allocation -- Java's `LongAccumulator(Math::max, Long.MIN_VALUE)`.
///
/// ## What a reader gets
///
/// [`Self::threshold_for`] returns the value **this port's** pruning rule
/// (`bound <= threshold` skips) needs, which is one ULP below the value Java's
/// rule (`bound < minCompetitiveScore` skips) is given. The two are the same
/// rule; see that method.
#[derive(Debug)]
pub struct MaxScoreAccumulator {
    /// `LongAccumulator(Math::max, Long.MIN_VALUE)`.
    acc: std::sync::atomic::AtomicI64,
}

impl Default for MaxScoreAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxScoreAccumulator {
    pub fn new() -> Self {
        Self {
            acc: std::sync::atomic::AtomicI64::new(i64::MIN),
        }
    }

    /// `accumulate(long code)`: fold `(doc_id, score)` into the running
    /// maximum. `doc_id` is **global** (leaf-local plus `docBase`), because the
    /// tie-break in [`Self::threshold_for`] compares it against another leaf's
    /// `docBase`.
    pub fn accumulate(&self, doc_id: i32, score: f32) {
        let code = doc_score_encoder::encode(doc_id, score);
        // `fetch_max` is `LongAccumulator`'s `Math::max` fold. `Relaxed` is
        // enough: the value is a *hint* -- every leaf's own queue is still the
        // authority on what it keeps, and a stale read can only cost pruning,
        // never correctness. Java's `LongAccumulator` gives no stronger
        // guarantee to its readers either (`get()` is explicitly documented as
        // not an atomic snapshot).
        self.acc
            .fetch_max(code, std::sync::atomic::Ordering::Relaxed);
    }

    /// The threshold a leaf whose documents start at `doc_base` may prune
    /// against, under this crate's `bound <= threshold` rule. `None` when
    /// nothing has been accumulated yet (Java's `getRaw() == Long.MIN_VALUE`).
    ///
    /// **The doc-id test, and the ULP.** Java's
    /// `updateGlobalMinCompetitiveScore` is
    ///
    /// ```java
    /// float score = DocScoreEncoder.toScore(maxMinScore);
    /// score = docBase >= DocScoreEncoder.docId(maxMinScore) ? Math.nextUp(score) : score;
    /// scorer.setMinCompetitiveScore(score);   // skips when bound < score
    /// ```
    ///
    /// -- if every document in this leaf sorts *after* the accumulated hit,
    /// then a document merely *tying* its score loses the doc-id tie-break and
    /// is not competitive, so the leaf may demand the next float up; otherwise
    /// a tie could still win and it may not.
    ///
    /// This port's rule skips when `bound <= threshold`, so the threshold it
    /// wants is the largest float Java's rule would still skip, i.e. the
    /// predecessor of Java's published value. `next_down(next_up(x)) == x`, so
    /// the two branches come out as `score` and `next_down(score)` -- the same
    /// pair, shifted by the same ULP the two rules differ by.
    pub fn threshold_for(&self, doc_base: i32) -> Option<f32> {
        let code = self.acc.load(std::sync::atomic::Ordering::Relaxed);
        if code == i64::MIN {
            return None;
        }
        let score = doc_score_encoder::to_score(code);
        Some(if doc_base >= doc_score_encoder::doc_id(code) {
            score
        } else {
            score.next_down()
        })
    }
}

/// `TopScoreDocCollector`-equivalent: keeps the top `n` `(doc_id, score)` hits
/// by score (ties broken by lower doc ID, matching real Lucene's `HitQueue` —
/// see [`rank_order`]), discarding everything else.
///
/// **Design**: up to [`SORTED_MAX`] hits start fully sorted (best-first) in a
/// plain `Vec`, a new hit shifted into place from the end, which beats a heap
/// at small sizes and when hits land near the end (equal or falling scores).
/// A larger `top_n`, or a full list above [`ALWAYS_SORTED`] once an insert
/// lands far from the end, keeps `hits` as a binary min-heap with the worst kept hit
/// at the root (Java's `HitQueue`), so an insert costs `O(log n)`;
/// [`TopDocsCollector::top_docs`] sorts a copy once on first read.
/// Both keep the same set: a hit enters only when it outranks the worst kept
/// hit under the total order [`rank_order`].
#[derive(Debug, Clone)]
pub struct TopDocsCollector {
    top_n: usize,
    /// Best-first while a sorted list, else a min-heap whose root is the
    /// worst kept hit (see `heap_ordered`).
    hits: Vec<ScoreDoc>,
    /// `hits` sorted best-first, built on the first [`Self::top_docs`] read of
    /// a heap-ordered collector and dropped by the next insert.
    sorted: OnceLock<Vec<ScoreDoc>>,
    /// Where the worst kept hit sits in a full `hits`: the root of a heap,
    /// the end of a sorted list.
    worst_at: usize,
    /// Every document handed to [`ScoringCollector::collect`], kept or not --
    /// real `TopScoreDocCollector`'s `totalHits`.
    total_hits: u64,
    /// `TopScoreDocCollector.totalHitsThreshold`: the collector reports no
    /// competitive threshold (and so authorizes no pruning) until it has seen
    /// more than this many hits, which is what keeps `total_hits` exact up to
    /// that point.
    total_hits_threshold: u64,
    /// Set once a threshold has actually been handed out, i.e. once a scorer
    /// was allowed to skip -- `TopScoreDocCollector.totalHitsRelation`.
    total_hits_relation: TotalHitsRelation,
    /// `TopScoreDocCollector.after`: the last hit of the previous page. Every
    /// document that would have ranked at or above it is skipped, so a caller
    /// gets page 2 without re-collecting page 1. `None` is Java's `after ==
    /// null`.
    ///
    /// **In this collector's own doc-ID space**, i.e. whatever `collect` is
    /// handed. Java stores a *global* `ScoreDoc` on the collector and subtracts
    /// `context.docBase` when it builds each leaf's collector; a caller
    /// federating segments here does the same subtraction -- see
    /// [`Self::with_after`].
    after: Option<ScoreDoc>,
    /// `TopScoreDocCollector.minScoreAcc`: one min-competitive score shared
    /// across every concurrently-searched leaf. `None` is Java's single-threaded
    /// case, where the collector's own queue is the only source.
    min_score_acc: Option<Arc<MaxScoreAccumulator>>,
    /// `LeafReaderContext.docBase`, needed only to translate this leaf's hits
    /// into the global doc-ID space [`Self::min_score_acc`] compares in.
    doc_base: i32,
    /// The highest threshold this collector has already published into
    /// [`Self::min_score_acc`] -- Java's per-leaf `minCompetitiveScore`, which
    /// gates `updateMinCompetitiveScore`'s work. Without it the accumulator's
    /// atomic read-modify-write would run once per collected document.
    published_min: f32,
}

impl TopDocsCollector {
    /// A collector that keeps at most `top_n` hits. `top_n == 0` is a defined
    /// "keep nothing" edge case (every `collect` call is a no-op), not a panic.
    ///
    /// The total-hits threshold is `0`, i.e. pruning is authorized as soon as
    /// the queue is full. Real Lucene's `IndexSearcher.search(query, n)` uses
    /// `TopScoreDocCollectorManager`'s default of `1000` instead, trading some
    /// pruning for an exact `totalHits` up to 1000; use
    /// [`Self::with_total_hits_threshold`] for that behavior. `0` is kept as
    /// this constructor's default because it is what every existing caller of
    /// this type already got, and because it is the right trade for the FFI
    /// callers this port serves, which ask for hits and not for counts.
    pub fn new(top_n: usize) -> Self {
        Self {
            top_n,
            hits: Vec::new(),
            sorted: OnceLock::new(),
            worst_at: worst_at(top_n),
            total_hits: 0,
            total_hits_threshold: 0,
            total_hits_relation: TotalHitsRelation::EqualTo,
            after: None,
            min_score_acc: None,
            doc_base: 0,
            published_min: f32::NEG_INFINITY,
        }
    }

    /// `TopScoreDocCollectorManager(numHits, totalHitsThreshold)`: keeps
    /// [`Self::total_hits`] exact (`EQUAL_TO`) for at least the first
    /// `total_hits_threshold` matches by refusing to publish a competitive
    /// score threshold before then. `u64::MAX` disables pruning entirely,
    /// which is exactly Java's `Integer.MAX_VALUE` sentinel that makes
    /// `scoreMode()` return `COMPLETE`.
    ///
    /// As in Java's constructor (`Math.max(totalHitsThreshold, numHits)`),
    /// the threshold is never below `top_n`: a query matching exactly `top_n`
    /// documents reports an exact (`EQUAL_TO`) count.
    pub fn with_total_hits_threshold(top_n: usize, total_hits_threshold: u64) -> Self {
        Self {
            top_n,
            hits: Vec::new(),
            sorted: OnceLock::new(),
            worst_at: worst_at(top_n),
            total_hits: 0,
            total_hits_threshold: total_hits_threshold.max(top_n as u64),
            total_hits_relation: TotalHitsRelation::EqualTo,
            after: None,
            min_score_acc: None,
            doc_base: 0,
            published_min: f32::NEG_INFINITY,
        }
    }

    /// `IndexSearcher.searchAfter(after, query, n)`: keep only hits that rank
    /// **strictly below** `after`, so the caller gets the next page without
    /// re-collecting the previous one.
    ///
    /// Java's test, verbatim (`TopScoreDocCollector.getLeafCollector`):
    ///
    /// ```java
    /// if (after != null && (score > afterScore || (score == afterScore && doc <= afterDoc)))
    ///   return;   // hit was collected on a previous page
    /// ```
    ///
    /// which is exactly "`after` outranks or equals this hit under `HitQueue`'s
    /// order" -- higher score first, lower doc id winning a tie. The document
    /// is still **counted**: `totalHits` is incremented before the test, so
    /// page 2 reports the same total page 1 did.
    ///
    /// **`after.doc_id` is in this collector's own doc-ID space.** Java holds a
    /// global `ScoreDoc` and builds each leaf's collector with
    /// `after.doc - context.docBase`; a multi-segment caller here does the same
    /// (see [`crate::multi_segment::merge_multi_segment_scored_after`], which
    /// does it for you). The subtraction is deliberately allowed to go
    /// negative or past `maxDoc`: for a leaf *before* the one holding `after`,
    /// every local doc id is `<= afterDoc` and only the score test can reject;
    /// for a leaf after it, no doc id is, which is the correct answer in both
    /// directions.
    pub fn with_after(mut self, after: ScoreDoc) -> Self {
        self.after = Some(after);
        self
    }

    /// `TopScoreDocCollectorManager`'s shared `MaxScoreAccumulator`: every
    /// concurrently-searched leaf publishes the score of its own worst kept hit
    /// into `acc`, and every leaf may then prune against the best of them
    /// instead of only against its own queue.
    ///
    /// `doc_base` is this leaf's `LeafReaderContext.docBase`, and it is not
    /// decoration: the accumulator stores `(global doc id, score)` because the
    /// tie-break is by doc id. A leaf whose documents all sort *after* the
    /// accumulated hit may require the next float up (a tie loses); a leaf that
    /// may contain smaller doc ids may not. See
    /// [`Self::min_competitive_score`].
    pub fn with_shared_max_score(mut self, acc: Arc<MaxScoreAccumulator>, doc_base: i32) -> Self {
        self.min_score_acc = Some(acc);
        self.doc_base = doc_base;
        self
    }

    /// `TopDocs.totalHits`: how many documents matched, and whether that count
    /// is exact. It is a lower bound exactly when this collector published a
    /// competitive threshold that a scorer could have pruned against.
    pub fn total_hits(&self) -> TotalHits {
        TotalHits {
            value: self.total_hits,
            relation: self.total_hits_relation,
        }
    }

    /// `TopScoreDocCollector.scoreMode()`: [`ScoreMode::Complete`] when the
    /// threshold is `u64::MAX` (pruning disabled, every hit counted),
    /// [`ScoreMode::TopScores`] otherwise -- the same `totalHitsThreshold ==
    /// Integer.MAX_VALUE ? COMPLETE : TOP_SCORES` test Java makes.
    #[inline]
    pub fn score_mode(&self) -> ScoreMode {
        if self.total_hits_threshold == u64::MAX {
            ScoreMode::Complete
        } else {
            ScoreMode::TopScores
        }
    }

    /// The kept hits, best-first (see [`rank_order`]) — `TopDocs.scoreDocs`-equivalent.
    /// This is capped at `top_n` and is **not** a hit count; for that, and for
    /// whether it is exact, see [`Self::total_hits`].
    pub fn top_docs(&self) -> &[ScoreDoc] {
        if !self.heap_ordered() {
            return &self.hits;
        }
        self.sorted.get_or_init(|| {
            let mut v = self.hits.clone();
            v.sort_unstable_by(|a, b| rank_order(b, a));
            v
        })
    }

    /// Whether `hits` is a heap rather than a sorted list.
    #[inline]
    fn heap_ordered(&self) -> bool {
        // A one-hit queue is the same either way.
        self.worst_at == 0 && self.top_n > 1
    }

    /// The worst kept hit. Only called on a full, non-empty queue.
    #[inline]
    fn worst(&self) -> ScoreDoc {
        self.hits[self.worst_at]
    }

    /// Adds `candidate`, evicting the worst kept hit when the queue is full.
    /// The caller has checked that it outranks that hit.
    fn insert_hit(&mut self, candidate: ScoreDoc) {
        if !self.heap_ordered() {
            let full = self.hits.len() == self.top_n;
            if full {
                self.hits.pop();
            }
            // Shift the hits it outranks one place down, from the end: a new
            // hit usually lands near the end (it beat the worst), and an
            // open-coded shift of a few elements beats `Vec::insert`'s call
            // into `memmove`.
            let h = &mut self.hits;
            h.push(candidate);
            let mut i = h.len() - 1;
            while i > 0 && rank_order(&h[i - 1], &candidate) != std::cmp::Ordering::Greater {
                h[i] = h[i - 1];
                i -= 1;
            }
            h[i] = candidate;
            // A hit landing far from the end (scores that rise as the
            // collection goes on) costs the list `O(n)`: on the first such
            // insert into a full queue it becomes a heap for good. The list
            // read in reverse, worst first, already is one.
            if full && self.top_n > ALWAYS_SORTED && h.len() - 1 - i > SHIFT_TO_HEAP {
                h.reverse();
                self.worst_at = 0;
            }
            return;
        }
        self.sorted.take();
        // `a` ranks below `b`: `a` belongs nearer the root.
        let worse = |a: &ScoreDoc, b: &ScoreDoc| rank_order(a, b) == std::cmp::Ordering::Less;
        let h = &mut self.hits;
        if h.len() < self.top_n {
            h.push(candidate);
            let mut i = h.len() - 1;
            while i > 0 {
                let parent = (i - 1) / 2;
                if worse(&h[i], &h[parent]) {
                    h.swap(i, parent);
                    i = parent;
                } else {
                    break;
                }
            }
            return;
        }
        // `updateTop`: replace the root and sift it down.
        let n = h.len();
        let mut i = 0;
        loop {
            let l = 2 * i + 1;
            if l >= n {
                break;
            }
            let r = l + 1;
            let child = if r < n && worse(&h[r], &h[l]) { r } else { l };
            if worse(&h[child], &candidate) {
                h[i] = h[child];
                i = child;
            } else {
                break;
            }
        }
        h[i] = candidate;
    }

    /// Real Lucene's `Scorable.setMinCompetitiveScore`-equivalent read side —
    /// the MAXSCORE/WAND mechanism's core value: once this collector is
    /// holding a full `top_n` hits, no candidate below the current worst kept
    /// hit's score (see [`rank_order`]) can possibly be kept, so that score is
    /// the threshold a block-level skip (e.g.
    /// `search_term_query_scored_maxscore`, via
    /// `crate::similarity::max_score_for_impacts`) compares a block's proven
    /// upper bound against. Returns `None` before the collector is full (every
    /// remaining candidate still has a chance, so there is no safe threshold
    /// yet) or when `top_n == 0`.
    #[inline]
    pub fn min_competitive_score(&self) -> Option<f32> {
        let local = self.local_min_competitive_score();
        // `updateGlobalMinCompetitiveScore`: what some other leaf has already
        // proved, translated into this leaf's terms. Java pushes it into the
        // scorer every `modInterval` documents to keep the atomic read off the
        // per-document path; this port's threshold is *pulled*, once per block
        // rather than once per document, so the interval has no work to do and
        // the read happens where the pruning decision is made.
        let global = self
            .min_score_acc
            .as_ref()
            .and_then(|acc| acc.threshold_for(self.doc_base));
        match (local, global) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }
    }

    /// The half of [`Self::min_competitive_score`] this collector's own queue
    /// knows: the worst kept hit's score, once the queue is full and the
    /// caller's exact-count budget is spent.
    #[inline]
    fn local_min_competitive_score(&self) -> Option<f32> {
        // `TopScoreDocCollector.updateMinCompetitiveScore`: a threshold is only
        // published once `totalHits > totalHitsThreshold`, so a caller that
        // asked for an exact count up to N gets one.
        if self.top_n == 0
            || self.hits.len() < self.top_n
            || self.total_hits <= self.total_hits_threshold
        {
            None
        } else {
            // `f32::max` returns the other operand for a NaN: Java pushes
            // nothing for a NaN worst hit (`Math.nextUp(NaN) >
            // minCompetitiveScore` is false), so the scorer keeps the bar it
            // was given before, `published_min` (`-inf` when there was none).
            // Otherwise the worst hit only ever rises, so this is the worst
            // hit's score, except after a queue full of NaN let a lower one in
            // -- and then too Java's scorer keeps its higher bar.
            Some(self.worst().score.max(self.published_min))
        }
    }

    /// Records that a threshold was handed out, so [`Self::total_hits`] stops
    /// claiming to be exact. Called from the `&mut self` collect path; the
    /// read-only [`Self::min_competitive_score`] cannot do it itself.
    fn note_threshold_published(&mut self) {
        match self.local_min_competitive_score() {
            Some(local) => {
                // `updateMinCompetitiveScore`'s `if (localMinScore >
                // minCompetitiveScore)` gate, which is also the only place the
                // relation flips (a bar at `-inf`, what a queue of NaN hits
                // yields, raises nothing and so leaves the count exact), then its
                // `minScoreAcc.accumulate(topCode)`. Two things are load-bearing
                // here. The gate: without it the atomic read-modify-write below
                // runs once per collected document, where Java runs it only when
                // this leaf's own bar actually rises -- `fetch_max` makes a
                // repeat write harmless, not free. And *what* is published: the
                // **worst kept hit** (Java's `topCode`, the heap top after the
                // insert), not the threshold derived from it, because the
                // accumulator stores the raw `(doc, score)` pair so a reader can
                // decide for itself whether a tie on that score is competitive
                // in its own doc-id range.
                if local > self.published_min {
                    self.total_hits_relation = TotalHitsRelation::GreaterThanOrEqualTo;
                    self.published_min = local;
                    if let Some(acc) = &self.min_score_acc {
                        let worst = self.worst();
                        acc.accumulate(self.doc_base.saturating_add(worst.doc_id), worst.score);
                    }
                } else {
                    self.note_shared_threshold();
                }
            }
            None => self.note_shared_threshold(),
        }
    }

    /// The relation flip for a bar this leaf did not raise itself.
    #[inline]
    fn note_shared_threshold(&mut self) {
        // A *shared* threshold can authorize a skip before this collector has
        // one of its own, and that makes the count a lower bound just the
        // same -- Java's `updateGlobalMinCompetitiveScore` sets the relation
        // too. Only reached while the relation is still `EqualTo`, so the
        // atomic load stops once it has flipped.
        if self.total_hits_relation == TotalHitsRelation::EqualTo
            && self
                .min_score_acc
                .as_ref()
                .is_some_and(|acc| acc.threshold_for(self.doc_base).is_some())
        {
            self.total_hits_relation = TotalHitsRelation::GreaterThanOrEqualTo;
        }
    }
}

/// How many hits [`TopDocsCollector`]'s `collect_many` tests against the
/// worst kept hit at once: long enough to vectorize, short enough that a hit
/// which does compete is reached without re-testing many before it.
const COLLECT_RUN: usize = 16;

/// The largest `top_n` [`TopDocsCollector`] starts as a sorted list; above it
/// the hits are a heap from the start.
const SORTED_MAX: usize = 64;

/// Up to this `top_n` the hits stay a sorted list whatever the inserts cost:
/// a shift of at most this many hits beats a heap's sift and its sort on read.
const ALWAYS_SORTED: usize = 16;

/// How far from the end an insert into a full sorted list may land before
/// [`TopDocsCollector`] turns the list into a heap.
const SHIFT_TO_HEAP: usize = 8;

/// [`TopDocsCollector::worst_at`] for a queue of `top_n`.
const fn worst_at(top_n: usize) -> usize {
    if top_n > SORTED_MAX {
        0
    } else {
        top_n.saturating_sub(1)
    }
}

/// Whether any of `scores` might not lose to `worst` -- `collect_many`'s run
/// test: true for a score above it, and whenever either side is a NaN, which
/// the per-document test does not reject (Java's `!(score <= topScore)`: a
/// NaN hit always enters, and a NaN worst hit -- a queue full of NaN --
/// lets every hit in). A whole run is compared as a fixed-size array, which
/// vectorizes; a shorter tail one by one.
#[inline]
fn run_competes(scores: &[f32], worst: f32) -> bool {
    // `!(s <= worst)`, not `s > worst`: the negation is what makes a NaN on
    // either side stop the run.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let competes = |any: bool, &s: &f32| any | !(s <= worst);
    match <&[f32; COLLECT_RUN]>::try_from(scores) {
        Ok(run) => run.iter().fold(false, competes),
        Err(_) => scores.iter().fold(false, competes),
    }
}

impl ScoringCollector for TopDocsCollector {
    #[inline]
    fn min_competitive_score(&self) -> Option<f32> {
        TopDocsCollector::min_competitive_score(self)
    }

    #[inline]
    fn pruning_threshold(&self) -> Option<f32> {
        // The trait's default, inline: every document-at-a-time loop asks
        // after each hit.
        if self.score_mode().is_exhaustive() {
            None
        } else {
            self.min_competitive_score()
        }
    }

    /// Full once `top_n` hits are kept *and* more than `total_hits_threshold`
    /// have been counted -- the two conditions
    /// [`Self::local_min_competitive_score`] publishes a threshold on. With
    /// every score equal and doc ids ascending, nothing after that point can
    /// displace a kept hit.
    fn constant_score_hits_needed(&self) -> Option<u64> {
        if self.top_n == 0
            || self.score_mode().is_exhaustive()
            || self.after.is_some()
            || self.min_score_acc.is_some()
        {
            return None;
        }
        let fill = (self.top_n as u64).max(self.total_hits_threshold.saturating_add(1));
        Some(fill.saturating_sub(self.total_hits))
    }

    /// [`ScoringCollector::collect`] per document, the count kept in a local
    /// while the documents lose to the worst kept hit. No threshold can be
    /// published until `total_hits` passes `total_hits_threshold` (and never
    /// by this leaf alone while a shared bar exists), so the batch stops at
    /// that count and declines a collector with a shared bar.
    fn collect_many(&mut self, docs: &[i32], scores: &[f32], doc_base: i32) -> usize {
        if self.min_score_acc.is_some() {
            return 0;
        }
        let room = self.total_hits_threshold.saturating_sub(self.total_hits);
        let k = docs
            .len()
            .min(scores.len())
            .min(usize::try_from(room).unwrap_or(usize::MAX));
        // The fast reject needs a full queue and no `searchAfter` page.
        let fast = self.after.is_none() && self.top_n != 0;
        let mut worst = (fast && self.hits.len() == self.top_n).then(|| self.worst());
        let (docs, scores) = (&docs[..k], &scores[..k]);
        let mut counted = 0u64;
        let mut i = 0;
        // Past a run that failed the test below, the rest of that run goes
        // one by one: testing it again from each next hit would re-read the
        // same scores up to `COLLECT_RUN` times.
        let mut one_by_one_until = 0;
        while i < k {
            if let Some(w) = worst.filter(|_| i >= one_by_one_until) {
                // A run of hits all past the worst kept hit's doc id (they
                // ascend, so the first one decides) loses exactly when none
                // scores above it and no NaN is involved (`run_competes`).
                // Checked a chunk at a time without a branch per hit.
                let end = i.saturating_add(COLLECT_RUN).min(k);
                if i64::from(doc_base) + i64::from(docs[i]) > i64::from(w.doc_id)
                    && !run_competes(&scores[i..end], w.score)
                {
                    #[cfg(any(test, feature = "test-support"))]
                    for _ in i..end {
                        crate::test_only_scored_docs_counter::record_scored();
                    }
                    // ARITH: `i < end <= k`.
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        counted += (end - i) as u64;
                    }
                    i = end;
                    continue;
                }
                one_by_one_until = end;
            }
            let (doc, score) = (docs[i], scores[i]);
            i += 1;
            // ARITH: a global doc id, below the reader's `i32` `max_doc`.
            #[allow(clippy::arithmetic_side_effects)]
            let doc_id = doc_base + doc;
            #[cfg(any(test, feature = "test-support"))]
            crate::test_only_scored_docs_counter::record_scored();
            if let Some(w) = worst {
                if score < w.score || (score == w.score && doc_id >= w.doc_id) {
                    counted += 1;
                    continue;
                }
            }
            self.total_hits += counted + 1;
            counted = 0;
            self.collect_counted(doc_id, score);
            if fast && self.hits.len() == self.top_n {
                worst = Some(self.worst());
            }
        }
        self.total_hits += counted;
        k
    }

    /// The per-document [`ScoringCollector::collect`] fast reject, `n` times
    /// at once: each would bump `total_hits` and lose to the worst kept hit.
    /// Only for a full queue with no `searchAfter` page and no shared bar,
    /// the case `constant_score_hits_needed` answers for; the caller hands
    /// its hits over in ascending doc-id order after every kept one.
    // Java's `score <= topScore` reject, negated below: a NaN on either side
    // is collected, not counted as losing.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn count_losing_hits(&mut self, n: u64, score: f32) -> bool {
        if self.after.is_some()
            || self.min_score_acc.is_some()
            || self.top_n == 0
            || self.hits.len() != self.top_n
            || !(score <= self.worst().score)
        {
            return false;
        }
        #[cfg(any(test, feature = "test-support"))]
        for _ in 0..n {
            crate::test_only_scored_docs_counter::record_scored();
        }
        self.total_hits = self.total_hits.saturating_add(n);
        if n != 0 && self.total_hits > self.total_hits_threshold {
            self.note_threshold_published();
        }
        true
    }

    #[inline]
    fn score_mode(&self) -> ScoreMode {
        TopDocsCollector::score_mode(self)
    }

    #[inline]
    fn collect(&mut self, doc_id: i32, score: f32) {
        // Counted before the fast reject, because the question this answers is
        // "how many documents did the scorer produce", not "how many did the
        // queue keep". See `crate::test_only_scored_docs_counter`.
        #[cfg(any(test, feature = "test-support"))]
        crate::test_only_scored_docs_counter::record_scored();
        // `TopScoreDocCollector`'s `int hitCountSoFar = ++totalHits;` -- counted
        // before the fast reject, because this is "how many documents matched",
        // not "how many the queue kept".
        self.total_hits += 1;
        // The common case, inline: a full queue the document loses to. Below
        // the count threshold and with no shared bar, publishing a threshold
        // would do nothing, so nothing else runs; nor once the relation has
        // flipped, since a losing hit moves neither bar (Java calls
        // `updateMinCompetitiveScore` from its reject branch only at
        // `hitCountSoFar == totalHitsThreshold + 1`).
        if self.after.is_none() && self.top_n != 0 && self.hits.len() == self.top_n {
            let worst = self.worst();
            if score < worst.score || (score == worst.score && doc_id >= worst.doc_id) {
                if self.total_hits_relation == TotalHitsRelation::EqualTo
                    && (self.total_hits > self.total_hits_threshold || self.min_score_acc.is_some())
                {
                    self.note_threshold_published();
                }
                return;
            }
        }
        self.collect_counted(doc_id, score);
    }
}

impl TopDocsCollector {
    /// [`ScoringCollector::collect`] past its inline fast path, the document
    /// already counted.
    #[inline(never)]
    fn collect_counted(&mut self, doc_id: i32, score: f32) {
        // `searchAfter`: a hit that ranks at or above the previous page's last
        // hit was already returned. Tested *before* the queue is consulted, as
        // Java does, and after the count, so page 2's `totalHits` still
        // includes page 1.
        if let Some(after) = self.after {
            if score > after.score || (score == after.score && doc_id <= after.doc_id) {
                self.note_threshold_published();
                return;
            }
        }
        if self.top_n == 0 {
            self.note_threshold_published();
            return;
        }
        // Fast reject, which is also `TopScoreDocCollector.collect`'s own
        // `if (score <= topScore)` reject. Once the queue is full, the
        // overwhelming majority of documents lose to the worst kept hit, and
        // deciding that takes one comparison; everything below only needs to
        // run for a hit that will actually be kept. The doc-id half of the
        // tie only matters to a caller handing documents over out of order
        // (Java assumes ascending order and rejects every tie).
        //
        // A NaN on either side is *not* rejected -- Java's test is
        // `!(score <= topScore)`, and NaN compares false -- so a NaN hit
        // replaces the worst kept hit, and while the worst kept hit is a NaN
        // (a queue full of them) every hit replaces it: with one slot the last
        // NaN document wins.
        if self.hits.len() == self.top_n {
            let worst = self.worst();
            if score < worst.score || (score == worst.score && doc_id >= worst.doc_id) {
                self.note_threshold_published();
                return;
            }
        } else if score == f32::NEG_INFINITY {
            // A queue that is not full has the `-inf` sentinel on top
            // (`DocScoreEncoder.LEAST_COMPETITIVE_CODE`), and `-inf <= -inf`:
            // Java never keeps a `-inf` hit.
            self.note_threshold_published();
            return;
        }
        let candidate = ScoreDoc {
            doc_id,
            score: canonical_score(score),
        };
        // `heap.updateTop(code)` when full: the worst hit leaves whatever the
        // new one's rank.
        self.insert_hit(candidate);
        self.note_threshold_published();
    }
}

/// Ascending/descending toggle for [`TopFieldCollector`] — real Lucene's
/// `SortField.setReverse` flag, generalized to any numeric sort key (this
/// port's `SortField.Type.LONG`/`INT` support; see `doc_value_query`'s
/// `sort_top_n_by_numeric_doc_value` for how a `DOUBLE` field would map onto
/// this same `i64` key if a caller bit-reinterprets it, which this port
/// doesn't do yet — see `docs/parity.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// One ranked-by-field hit: a doc ID plus its already-decoded numeric sort
/// value — the `FieldDoc`-equivalent minimal shape (no `shardIndex`, same
/// simplification [`ScoreDoc`] already makes, and no secondary sort fields —
/// see [`TopFieldCollector`]'s doc comment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldValueDoc {
    pub doc_id: i32,
    pub value: i64,
}

/// Real Lucene's `FieldValueHitQueue`-equivalent ranking order: ranks by
/// `value` in `direction` (ascending or descending), and — on an exact value
/// tie — **lower doc ID wins**, the same tie-break convention
/// [`TopDocsCollector`]'s [`rank_order`] already documents for a BM25 score
/// tie, kept consistent here for a sort-value tie. Returns
/// `Ordering::Greater` when `a` should rank ahead of `b`.
fn field_rank_order(
    a: &FieldValueDoc,
    b: &FieldValueDoc,
    direction: SortDirection,
) -> std::cmp::Ordering {
    let value_order = match direction {
        // Ascending: the *smaller* value ranks ahead, so `a` ranking ahead of
        // `b` (Greater) happens when `a.value < b.value`, i.e. when
        // `b.value.cmp(&a.value)` is `Greater`.
        SortDirection::Ascending => b.value.cmp(&a.value),
        // Descending: the *larger* value ranks ahead -- direct `a.cmp(b)`.
        SortDirection::Descending => a.value.cmp(&b.value),
    };
    match value_order {
        std::cmp::Ordering::Equal => b.doc_id.cmp(&a.doc_id),
        other => other,
    }
}

/// `TopFieldCollector`-equivalent (`org.apache.lucene.search.TopFieldCollector`,
/// scoped to a single numeric `SortField`): keeps the top `n` `(doc_id, value)`
/// hits ranked by a numeric doc-value field, ascending or descending per
/// [`SortDirection`], ties broken by ascending doc ID (see [`field_rank_order`]),
/// discarding everything else.
///
/// **Scope**: numeric doc-value fields only (`SortField.Type.LONG`/`INT`,
/// via the `i64` key `value` already carries — a `DOUBLE` field's sort key
/// would need a bit-reinterpret step this port doesn't add yet). No String/
/// `SortedDocValues`-based sort, no multiple sort fields/secondary keys
/// beyond the single documented doc-ID tie-break. Missing-value handling
/// (a candidate doc with no value for the sort field) is the caller's job —
/// this collector only ever sees `(doc_id, value)` pairs a caller already
/// decided to `offer`; see `doc_value_query::MissingValue` for the policy
/// its composition functions apply before calling [`TopFieldCollector::offer`].
/// See `docs/parity.md` for the precise, honest scope statement.
///
/// **Design**: not a [`Collector`]/[`ScoringCollector`] impl, because neither
/// trait's `collect` signature can carry a `Result` for a doc-value decode
/// error, and reading a doc's sort value is a fallible operation (the same
/// reason `doc_value_query::sort_by_numeric_doc_value` is a standalone
/// function rather than a `Collector` variant, see that function's doc
/// comment). Composition functions (e.g.
/// `doc_value_query::sort_top_n_by_numeric_doc_value`) decode each candidate
/// doc's value themselves (propagating any decode error via `Result`) and
/// call [`TopFieldCollector::offer`] with the already-decoded `i64`, which is
/// infallible. Internally this is the exact same bounded, always-sorted
/// `Vec` design [`TopDocsCollector`] already uses (see that struct's doc
/// comment for the tradeoff rationale) — same simple `O(n)`-per-insert cost.
#[derive(Debug, Clone)]
pub struct TopFieldCollector {
    top_n: usize,
    direction: SortDirection,
    hits: Vec<FieldValueDoc>,
}

impl TopFieldCollector {
    /// A collector that keeps at most `top_n` hits ranked by `direction`.
    /// `top_n == 0` is a defined "keep nothing" edge case (every `offer` call
    /// is a no-op), not a panic.
    pub fn new(top_n: usize, direction: SortDirection) -> Self {
        Self {
            top_n,
            direction,
            hits: Vec::new(),
        }
    }

    /// Offers one already-decoded `(doc_id, value)` pair. Only inserted if it
    /// ranks ahead of the current worst kept hit (or there's still room) --
    /// see [`field_rank_order`].
    pub fn offer(&mut self, doc_id: i32, value: i64) {
        if self.top_n == 0 {
            return;
        }
        let candidate = FieldValueDoc { doc_id, value };
        if self.hits.len() < self.top_n {
            let pos = self.hits.partition_point(|h| {
                field_rank_order(h, &candidate, self.direction) == std::cmp::Ordering::Greater
            });
            self.hits.insert(pos, candidate);
            return;
        }
        if let Some(worst) = self.hits.last() {
            if field_rank_order(&candidate, worst, self.direction) == std::cmp::Ordering::Greater {
                self.hits.pop();
                let pos = self.hits.partition_point(|h| {
                    field_rank_order(h, &candidate, self.direction) == std::cmp::Ordering::Greater
                });
                self.hits.insert(pos, candidate);
            }
        }
    }

    /// The kept hits, best-first per [`SortDirection`] (see [`field_rank_order`]).
    pub fn top_docs(&self) -> &[FieldValueDoc] {
        &self.hits
    }

    /// The worst kept value once the collector is full -- `LeafFieldComparator`'s
    /// `bottom` -- or `None` while every offer still gets in. A caller offering
    /// documents in ascending doc-id order can reject any value that does not
    /// strictly beat it without calling [`Self::offer`]: a tie loses on doc id.
    pub fn bottom_value(&self) -> Option<i64> {
        if self.top_n == 0 || self.hits.len() < self.top_n {
            return None;
        }
        self.hits.last().map(|h| h.value)
    }

    /// Whether `value` strictly beats the bottom (always, while not full).
    #[inline]
    pub fn competes(&self, bottom: Option<i64>, value: i64) -> bool {
        match (bottom, self.direction) {
            (None, _) => true,
            (Some(b), SortDirection::Ascending) => value < b,
            (Some(b), SortDirection::Descending) => value > b,
        }
    }
}

/// `CollapsingTopDocsCollector`-equivalent (real Lucene/Solr's field-collapse
/// mechanism, historically `org.apache.lucene.search.grouping.CollapsingTopDocsCollector`,
/// now living in Solr as `CollapsingQParserPlugin`'s collector): as docs are
/// scored, keep only the single highest-scoring doc for each distinct value
/// of a collapse-key doc-values field, then select the final top `n` from
/// those group-winners only.
///
/// **Algorithm**: single-pass, matching real `CollapsingTopDocsCollector`
/// (not a two-phase collect-then-rerank design) — for every candidate
/// `(doc_id, score, key)`, look up the current best doc for `key` in a
/// `HashMap` and replace it only if the candidate outranks it (per
/// [`rank_order`]'s score-desc/doc-id-asc convention, kept consistent with
/// [`TopDocsCollector`]). Group winners accumulate in the map (and, for the
/// null-group case, in a side `Vec` — see below) as collection proceeds;
/// [`Self::top_docs`] does the final top-`n` reduction over just those
/// winners on demand, not per-doc, since the winner set only shrinks in
/// membership count (never grows past one entry per key) as collection
/// proceeds.
///
/// **Scope**: NUMERIC doc-values collapse key only (an already-decoded
/// `i64`), not real Lucene's more common SORTED-field ordinal key. A SORTED
/// key needs an extra ordinal-to-value resolution step per doc (via the
/// field's terms dictionary, as `facets.rs`'s `resolve_labels` already does
/// for a different purpose) that this port doesn't add yet — NUMERIC-keyed
/// collapsing is a valid, narrower starting point (real Lucene's own
/// `CollapsingTopDocsCollector` special-cases `NUMERIC` vs `SORTED` keys
/// internally, so this isn't an invented split). See `docs/parity.md`.
///
/// **Missing collapse-key value**: a doc with no value for the collapse
/// field (`key: None`) is **not** discarded and **not** collapsed together
/// with other missing-value docs — each survives as its own singleton group,
/// i.e. this port's `null group` policy is "every null-key doc competes for
/// top-`n` on its own", the `CollapsingQParserPlugin` `nullPolicy=EXPAND`
/// behavior. Real Solr's *default* `nullPolicy` is `IGNORE` (drop null-key
/// docs entirely), but `EXPAND` is itself a documented, real Solr policy
/// value, not a fabricated behavior — this port picks it as the simpler
/// single case to implement first (no separate "drop" code path needed
/// alongside the "keep, uncollapsed" one) and documents the choice honestly
/// rather than silently assuming `IGNORE`. See `docs/parity.md`.
///
/// **Design**: like [`TopFieldCollector`], not a [`Collector`]/
/// [`ScoringCollector`] impl — the collapse key is a fallible doc-value
/// decode (same reasoning as that struct's doc comment), so a caller decodes
/// each candidate's key first and calls [`Self::offer`] with the
/// already-decoded `Option<i64>`.
#[derive(Debug, Clone)]
pub struct CollapsingCollector {
    top_n: usize,
    /// One entry per distinct collapse-key value seen so far, holding that
    /// key's current best-scoring doc.
    groups: std::collections::HashMap<i64, ScoreDoc>,
    /// Null-group docs (missing collapse-key value), each its own singleton
    /// group per this struct's documented `EXPAND`-equivalent policy.
    null_group: Vec<ScoreDoc>,
}

impl CollapsingCollector {
    /// A collector that will ultimately keep at most `top_n` hits, selected
    /// from group-winners. `top_n == 0` is a defined "keep nothing" edge
    /// case: [`Self::top_docs`] returns an empty slice regardless of how many
    /// groups were offered.
    pub fn new(top_n: usize) -> Self {
        Self {
            top_n,
            groups: std::collections::HashMap::new(),
            null_group: Vec::new(),
        }
    }

    /// Offers one already-scored, already-key-decoded candidate. `key` is
    /// the doc's decoded NUMERIC collapse-value, or `None` if the doc has no
    /// value for the collapse field (see this struct's doc comment for the
    /// null-group policy). Replaces the current group winner only if the
    /// candidate outranks it (or the group hasn't been seen yet); a `None`
    /// key always inserts a new singleton group.
    pub fn offer(&mut self, doc_id: i32, score: f32, key: Option<i64>) {
        let candidate = ScoreDoc { doc_id, score };
        match key {
            None => self.null_group.push(candidate),
            Some(k) => {
                self.groups
                    .entry(k)
                    .and_modify(|current| {
                        if rank_order(&candidate, current) == std::cmp::Ordering::Greater {
                            *current = candidate;
                        }
                    })
                    .or_insert(candidate);
            }
        }
    }

    /// The top `n` hits selected from group-winners only, best-first per
    /// [`rank_order`] — real Lucene's `CollapsingTopDocsCollector.topDocs`-
    /// equivalent final reduction. Every keyed group contributes at most one
    /// hit (its winner); every null-group doc contributes its own hit (see
    /// this struct's doc comment).
    ///
    /// Unlike [`TopDocsCollector::top_docs`]/[`TopFieldCollector::top_docs`]
    /// (which return a borrowed `&[T]` into an already-sorted `Vec`
    /// maintained incrementally on every `collect`/`offer` call), this
    /// returns an **owned, freshly sorted `Vec`** computed on demand: group
    /// winners live in a `HashMap` (see this struct's doc comment for why),
    /// so there is no single running best-first order to borrow until this
    /// reduction actually runs it. Calling this repeatedly re-sorts every
    /// time — cheap at the query sizes this collector targets, but a real
    /// difference from the other two collectors' O(1)/no-allocation
    /// `top_docs()`, not an oversight.
    pub fn top_docs(&self) -> Vec<ScoreDoc> {
        if self.top_n == 0 {
            return Vec::new();
        }
        let mut winners: Vec<ScoreDoc> = self
            .groups
            .values()
            .copied()
            .chain(self.null_group.iter().copied())
            .collect();
        winners.sort_by(|a, b| rank_order(a, b).reverse());
        winners.truncate(self.top_n);
        winners
    }
}

#[cfg(test)]
mod tests;
