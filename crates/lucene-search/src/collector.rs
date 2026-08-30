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
//! ## `ScoringCollector` (task #13's addition) — a new trait, not a breaking
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

use std::sync::Arc;

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
/// MAXSCORE block skipping shuts off). It does **not** yet have the other half
/// -- a `needs_scores == false` postings path -- because
/// `lucene_codecs::postings` has no `PostingsEnum`-flags plumbing at all; that
/// is a codec-side carry-over item recorded in `docs/sweep/m2/LEDGER.md`, and
/// this enum is the search-side half of it landing first.
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
/// "about N results" is reading its [`TotalHitsRelation`]. This port reported
/// only `top_docs().len()` before the b12 sweep, which is the *kept* hit count
/// (capped at `top_n`) and says nothing at all about how many documents
/// matched.
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
    /// `min_competitive_score()` directly, which several loops used to do, is
    /// the bug this prevents: a collector promising an exact `totalHits` would
    /// still hand out a threshold and still get its blocks skipped.
    fn pruning_threshold(&self) -> Option<f32> {
        if self.score_mode().is_exhaustive() {
            None
        } else {
            self.min_competitive_score()
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
    /// `NumericUtils.sortableFloatBits`: flips the sign bit for a positive
    /// float and every bit but the sign for a negative one, turning IEEE-754's
    /// sign-magnitude order into two's-complement order. Its own inverse.
    fn sortable_float_bits(bits: i32) -> i32 {
        bits ^ ((bits >> 31) & 0x7fff_ffff)
    }

    pub(super) fn encode(doc_id: i32, score: f32) -> i64 {
        let sortable = sortable_float_bits(score.to_bits() as i32);
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
/// **Design**: real `TopScoreDocCollector` is backed by a `HitQueue` (a binary
/// min-heap over the *worst* currently-kept hit, so a new hit only needs one
/// comparison against the heap's root to know whether it's worth keeping).
/// This port instead keeps `hits` fully sorted (best-first) after every
/// insert/eviction — a plain `Vec` with a binary-search insert position. This
/// is the same tradeoff this crate's `docid_set` module already made for
/// `Disjunction` ("simple first cut, revisit if scale demands it" — see that
/// module's doc comment): correct, `O(n)` per insert instead of `O(log n)`,
/// fine for the query sizes and `top_n` values this port's fixtures and tests
/// exercise today.
#[derive(Debug, Clone)]
pub struct TopDocsCollector {
    top_n: usize,
    hits: Vec<ScoreDoc>,
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
    pub fn with_total_hits_threshold(top_n: usize, total_hits_threshold: u64) -> Self {
        Self {
            top_n,
            hits: Vec::new(),
            total_hits: 0,
            total_hits_threshold,
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
        &self.hits
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
            self.hits.last().map(|h| h.score)
        }
    }

    /// Records that a threshold was handed out, so [`Self::total_hits`] stops
    /// claiming to be exact. Called from the `&mut self` collect path; the
    /// read-only [`Self::min_competitive_score`] cannot do it itself.
    fn note_threshold_published(&mut self) {
        match self.local_min_competitive_score() {
            Some(local) => {
                self.total_hits_relation = TotalHitsRelation::GreaterThanOrEqualTo;
                // `updateMinCompetitiveScore`'s `if (localMinScore >
                // minCompetitiveScore)` gate, then its
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
                    self.published_min = local;
                    if let Some(acc) = &self.min_score_acc {
                        if let Some(worst) = self.hits.last() {
                            acc.accumulate(self.doc_base.saturating_add(worst.doc_id), worst.score);
                        }
                    }
                }
            }
            None => {
                // A *shared* threshold can authorize a skip before this
                // collector has one of its own, and that makes the count a lower
                // bound just the same -- Java's
                // `updateGlobalMinCompetitiveScore` sets the relation too. Only
                // reached while the relation is still `EqualTo`, so the atomic
                // load stops once it has flipped.
                if self.min_score_acc.is_some()
                    && self.total_hits_relation == TotalHitsRelation::EqualTo
                    && TopDocsCollector::min_competitive_score(self).is_some()
                {
                    self.total_hits_relation = TotalHitsRelation::GreaterThanOrEqualTo;
                }
            }
        }
    }
}

impl ScoringCollector for TopDocsCollector {
    fn min_competitive_score(&self) -> Option<f32> {
        TopDocsCollector::min_competitive_score(self)
    }

    fn score_mode(&self) -> ScoreMode {
        TopDocsCollector::score_mode(self)
    }

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
        // Fast reject, which is also `TopScoreDocCollector.collect`'s own first
        // line (`if (score <= pqTop.score) return;`). Once the queue is full,
        // the overwhelming majority of documents lose to the worst kept hit,
        // and deciding that takes one comparison; everything below only needs
        // to run for a hit that will actually be kept. NaN falls through to the
        // general path below, which orders it via `total_cmp`.
        if self.hits.len() == self.top_n {
            let worst = self.hits[self.top_n - 1];
            if score < worst.score || (score == worst.score && doc_id >= worst.doc_id) {
                self.note_threshold_published();
                return;
            }
        }
        let candidate = ScoreDoc { doc_id, score };
        if self.hits.len() < self.top_n {
            let pos = self
                .hits
                .partition_point(|h| rank_order(h, &candidate) == std::cmp::Ordering::Greater);
            self.hits.insert(pos, candidate);
            self.note_threshold_published();
            return;
        }
        // Full: only replace the current worst (last) hit if the candidate outranks it.
        if let Some(worst) = self.hits.last() {
            if rank_order(&candidate, worst) == std::cmp::Ordering::Greater {
                self.hits.pop();
                let pos = self
                    .hits
                    .partition_point(|h| rank_order(h, &candidate) == std::cmp::Ordering::Greater);
                self.hits.insert(pos, candidate);
            }
        }
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
/// comment for the tradeoff rationale) — same `O(n)`-per-insert simple first
/// cut, revisit if scale demands it.
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
mod tests {
    use super::*;

    fn field_docs(v: &[(i32, i64)]) -> Vec<FieldValueDoc> {
        v.iter()
            .map(|&(doc_id, value)| FieldValueDoc { doc_id, value })
            .collect()
    }

    #[test]
    fn top_field_collector_empty_input_yields_no_hits() {
        let c = TopFieldCollector::new(3, SortDirection::Ascending);
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn top_field_collector_top_n_zero_keeps_nothing() {
        let mut c = TopFieldCollector::new(0, SortDirection::Ascending);
        c.offer(1, 5);
        c.offer(2, 9);
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn top_field_collector_ascending_orders_smallest_first() {
        let mut c = TopFieldCollector::new(5, SortDirection::Ascending);
        c.offer(1, 30);
        c.offer(2, 10);
        c.offer(3, 20);
        assert_eq!(
            c.top_docs().to_vec(),
            field_docs(&[(2, 10), (3, 20), (1, 30)])
        );
    }

    #[test]
    fn top_field_collector_descending_orders_largest_first() {
        let mut c = TopFieldCollector::new(5, SortDirection::Descending);
        c.offer(1, 30);
        c.offer(2, 10);
        c.offer(3, 20);
        assert_eq!(
            c.top_docs().to_vec(),
            field_docs(&[(1, 30), (3, 20), (2, 10)])
        );
    }

    #[test]
    fn top_field_collector_truncates_to_top_n_ascending() {
        let mut c = TopFieldCollector::new(2, SortDirection::Ascending);
        c.offer(1, 30);
        c.offer(2, 10);
        c.offer(3, 20);
        // Worst (doc 1, value 30) must be evicted, keeping the two smallest.
        assert_eq!(c.top_docs().to_vec(), field_docs(&[(2, 10), (3, 20)]));
    }

    #[test]
    fn top_field_collector_truncates_to_top_n_descending() {
        let mut c = TopFieldCollector::new(2, SortDirection::Descending);
        c.offer(1, 30);
        c.offer(2, 10);
        c.offer(3, 20);
        // Worst (doc 2, value 10) must be evicted, keeping the two largest.
        assert_eq!(c.top_docs().to_vec(), field_docs(&[(1, 30), (3, 20)]));
    }

    #[test]
    fn top_field_collector_tie_break_prefers_lower_doc_id() {
        let mut c = TopFieldCollector::new(2, SortDirection::Ascending);
        c.offer(5, 2);
        c.offer(2, 2);
        c.offer(9, 2);
        assert_eq!(c.top_docs().to_vec(), field_docs(&[(2, 2), (5, 2)]));
    }

    #[test]
    fn field_rank_order_ties_break_by_ascending_doc_id() {
        let a = FieldValueDoc {
            doc_id: 1,
            value: 5,
        };
        let b = FieldValueDoc {
            doc_id: 2,
            value: 5,
        };
        assert_eq!(
            field_rank_order(&a, &b, SortDirection::Ascending),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            field_rank_order(&a, &b, SortDirection::Descending),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn vec_collector_collects_in_call_order() {
        let mut c = VecCollector::default();
        c.collect(3);
        c.collect(7);
        assert_eq!(c.docs, vec![3, 7]);
    }

    #[test]
    fn count_collector_counts_calls_not_values() {
        let mut c = CountCollector::default();
        c.collect(0);
        c.collect(0);
        c.collect(5);
        assert_eq!(c.count, 3);
    }

    fn score_docs(v: &[(i32, f32)]) -> Vec<ScoreDoc> {
        v.iter()
            .map(|&(doc_id, score)| ScoreDoc { doc_id, score })
            .collect()
    }

    #[test]
    fn top_docs_collector_empty_input_yields_no_hits() {
        let c = TopDocsCollector::new(3);
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn top_docs_collector_top_n_zero_keeps_nothing() {
        let mut c = TopDocsCollector::new(0);
        c.collect(1, 5.0);
        c.collect(2, 9.0);
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn top_docs_collector_fewer_than_n_keeps_all_sorted_by_score_desc() {
        let mut c = TopDocsCollector::new(5);
        c.collect(1, 1.0);
        c.collect(2, 3.0);
        c.collect(3, 2.0);
        assert_eq!(
            c.top_docs().to_vec(),
            score_docs(&[(2, 3.0), (3, 2.0), (1, 1.0)])
        );
    }

    #[test]
    fn top_docs_collector_exactly_n_keeps_all_sorted() {
        let mut c = TopDocsCollector::new(3);
        c.collect(1, 1.0);
        c.collect(2, 3.0);
        c.collect(3, 2.0);
        assert_eq!(
            c.top_docs().to_vec(),
            score_docs(&[(2, 3.0), (3, 2.0), (1, 1.0)])
        );
    }

    #[test]
    fn top_docs_collector_more_than_n_evicts_the_worst() {
        let mut c = TopDocsCollector::new(2);
        c.collect(1, 1.0);
        c.collect(2, 3.0);
        c.collect(3, 2.0);
        // 1.0 (doc 1) is the worst score and gets evicted once a better candidate
        // (doc 3, score 2.0) arrives.
        assert_eq!(c.top_docs().to_vec(), score_docs(&[(2, 3.0), (3, 2.0)]));
    }

    #[test]
    fn top_docs_collector_candidate_worse_than_all_kept_hits_is_dropped() {
        let mut c = TopDocsCollector::new(2);
        c.collect(1, 5.0);
        c.collect(2, 4.0);
        c.collect(3, 1.0); // worse than both kept hits -- must not be kept.
        assert_eq!(c.top_docs().to_vec(), score_docs(&[(1, 5.0), (2, 4.0)]));
    }

    #[test]
    fn top_docs_collector_tie_break_prefers_lower_doc_id() {
        let mut c = TopDocsCollector::new(2);
        c.collect(5, 2.0);
        c.collect(2, 2.0);
        c.collect(9, 2.0);
        // All tied at score 2.0 -- lowest doc IDs (2, 5) must win over doc 9.
        assert_eq!(c.top_docs().to_vec(), score_docs(&[(2, 2.0), (5, 2.0)]));
    }

    #[test]
    fn top_docs_collector_tie_break_eviction_prefers_lower_doc_id() {
        let mut c = TopDocsCollector::new(1);
        c.collect(9, 3.0);
        c.collect(2, 3.0); // ties doc 9 on score; lower doc id must win.
        assert_eq!(c.top_docs().to_vec(), score_docs(&[(2, 3.0)]));
    }

    #[test]
    fn rank_order_orders_by_score_desc_then_doc_id_asc() {
        let a = ScoreDoc {
            doc_id: 1,
            score: 5.0,
        };
        let b = ScoreDoc {
            doc_id: 2,
            score: 5.0,
        };
        assert_eq!(rank_order(&a, &b), std::cmp::Ordering::Greater);
        assert_eq!(rank_order(&b, &a), std::cmp::Ordering::Less);
        let c = ScoreDoc {
            doc_id: 3,
            score: 6.0,
        };
        assert_eq!(rank_order(&c, &a), std::cmp::Ordering::Greater);
    }

    // --- CollapsingCollector ---

    #[test]
    fn collapsing_collector_empty_input_yields_no_hits() {
        let c = CollapsingCollector::new(3);
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn collapsing_collector_top_n_zero_keeps_nothing() {
        let mut c = CollapsingCollector::new(0);
        c.offer(1, 5.0, Some(1));
        c.offer(2, 9.0, Some(2));
        assert!(c.top_docs().is_empty());
    }

    #[test]
    fn collapsing_collector_keeps_only_highest_scoring_doc_per_key() {
        let mut c = CollapsingCollector::new(10);
        // Three docs share collapse-key 1; only the highest score (doc 2)
        // should survive.
        c.offer(1, 1.0, Some(1));
        c.offer(2, 5.0, Some(1));
        c.offer(3, 3.0, Some(1));
        let docs = c.top_docs();
        assert_eq!(
            docs,
            vec![ScoreDoc {
                doc_id: 2,
                score: 5.0
            }]
        );
    }

    #[test]
    fn collapsing_collector_distinct_keys_all_appear_subject_to_top_n() {
        let mut c = CollapsingCollector::new(10);
        c.offer(1, 1.0, Some(1));
        c.offer(2, 2.0, Some(2));
        c.offer(3, 3.0, Some(3));
        let docs = c.top_docs();
        assert_eq!(
            docs,
            vec![
                ScoreDoc {
                    doc_id: 3,
                    score: 3.0
                },
                ScoreDoc {
                    doc_id: 2,
                    score: 2.0
                },
                ScoreDoc {
                    doc_id: 1,
                    score: 1.0
                },
            ]
        );
    }

    #[test]
    fn collapsing_collector_truncates_group_winners_to_top_n() {
        let mut c = CollapsingCollector::new(2);
        c.offer(1, 1.0, Some(1));
        c.offer(2, 2.0, Some(2));
        c.offer(3, 3.0, Some(3));
        let docs = c.top_docs();
        assert_eq!(
            docs,
            vec![
                ScoreDoc {
                    doc_id: 3,
                    score: 3.0
                },
                ScoreDoc {
                    doc_id: 2,
                    score: 2.0
                },
            ]
        );
    }

    #[test]
    fn collapsing_collector_missing_key_docs_each_survive_as_own_group() {
        let mut c = CollapsingCollector::new(10);
        // Two docs with no collapse-key value -- both must survive
        // independently (this port's EXPAND-equivalent null-group policy),
        // not collapse together.
        c.offer(1, 1.0, None);
        c.offer(2, 2.0, None);
        c.offer(3, 3.0, Some(9));
        let mut docs = c.top_docs();
        docs.sort_by_key(|d| d.doc_id);
        assert_eq!(
            docs,
            vec![
                ScoreDoc {
                    doc_id: 1,
                    score: 1.0
                },
                ScoreDoc {
                    doc_id: 2,
                    score: 2.0
                },
                ScoreDoc {
                    doc_id: 3,
                    score: 3.0
                },
            ]
        );
    }

    #[test]
    fn collapsing_collector_replaces_group_winner_on_better_candidate() {
        let mut c = CollapsingCollector::new(10);
        c.offer(1, 5.0, Some(1));
        c.offer(2, 1.0, Some(1)); // worse -- must not replace doc 1.
        let docs = c.top_docs();
        assert_eq!(
            docs,
            vec![ScoreDoc {
                doc_id: 1,
                score: 5.0
            }]
        );
    }

    #[test]
    fn collapsing_collector_tie_break_prefers_lower_doc_id_within_group() {
        let mut c = CollapsingCollector::new(10);
        c.offer(9, 3.0, Some(1));
        c.offer(2, 3.0, Some(1)); // ties on score -- lower doc id wins.
        let docs = c.top_docs();
        assert_eq!(
            docs,
            vec![ScoreDoc {
                doc_id: 2,
                score: 3.0
            }]
        );
    }

    // ---- `ScoreMode` / `TotalHits` (`org.apache.lucene.search.ScoreMode`,
    // `TotalHits`, `TopScoreDocCollector.totalHitsThreshold`) ----

    #[test]
    fn score_mode_predicates_match_javas_enum_table() {
        // `ScoreMode`'s Java constructor arguments are `(isExhaustive,
        // needsScores)`: COMPLETE(true,true), COMPLETE_NO_SCORES(true,false),
        // TOP_SCORES(false,true), TOP_DOCS(false,false),
        // TOP_DOCS_WITH_SCORES(false,true). Asserted as a table so a future
        // edit to either predicate has to disagree with Java on purpose.
        let table = [
            (ScoreMode::Complete, true, true),
            (ScoreMode::CompleteNoScores, true, false),
            (ScoreMode::TopScores, false, true),
            (ScoreMode::TopDocs, false, false),
            (ScoreMode::TopDocsWithScores, false, true),
        ];
        for (mode, exhaustive, needs_scores) in table {
            assert_eq!(mode.is_exhaustive(), exhaustive, "{mode:?}");
            assert_eq!(mode.needs_scores(), needs_scores, "{mode:?}");
        }
    }

    #[test]
    fn total_hits_counts_every_collected_doc_not_just_the_kept_ones() {
        let mut c = TopDocsCollector::new(2);
        for doc in 0..10 {
            c.collect(doc, doc as f32);
        }
        assert_eq!(c.top_docs().len(), 2);
        assert_eq!(c.total_hits().value, 10);
    }

    #[test]
    fn total_hits_is_exact_until_a_competitive_threshold_is_published() {
        // Threshold 0 (this port's default): the queue fills at the second
        // document, a threshold is published, and the count stops being exact.
        let mut c = TopDocsCollector::new(2);
        c.collect(0, 1.0);
        assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
        assert_eq!(c.min_competitive_score(), None, "queue not full yet");
        c.collect(1, 2.0);
        assert_eq!(c.min_competitive_score(), Some(1.0));
        assert_eq!(
            c.total_hits().relation,
            TotalHitsRelation::GreaterThanOrEqualTo
        );
    }

    #[test]
    fn a_total_hits_threshold_keeps_the_count_exact_and_delays_pruning() {
        // `TopScoreDocCollectorManager(numHits, totalHitsThreshold)`: no
        // threshold is published while `totalHits <= totalHitsThreshold`, so
        // the first five counts are exact even though the queue filled at two.
        let mut c = TopDocsCollector::with_total_hits_threshold(2, 5);
        for doc in 0..5 {
            c.collect(doc, doc as f32);
            assert_eq!(c.min_competitive_score(), None, "doc {doc}");
            assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
        }
        c.collect(5, 5.0);
        assert!(c.min_competitive_score().is_some());
        assert_eq!(
            c.total_hits().relation,
            TotalHitsRelation::GreaterThanOrEqualTo
        );
        assert_eq!(c.total_hits().value, 6);
    }

    #[test]
    fn an_infinite_threshold_is_complete_mode_and_never_prunes() {
        // Java: `totalHitsThreshold == Integer.MAX_VALUE ? COMPLETE :
        // TOP_SCORES`. In COMPLETE mode no threshold may ever be published, so
        // `totalHits` stays exact no matter how many documents arrive.
        let mut c = TopDocsCollector::with_total_hits_threshold(2, u64::MAX);
        assert_eq!(c.score_mode(), ScoreMode::Complete);
        assert!(c.score_mode().is_exhaustive());
        for doc in 0..50 {
            c.collect(doc, doc as f32);
        }
        assert_eq!(c.min_competitive_score(), None);
        assert_eq!(c.total_hits().value, 50);
        assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
        assert_eq!(c.top_docs().len(), 2);
    }

    #[test]
    fn default_collector_reports_top_scores_mode() {
        let c = TopDocsCollector::new(3);
        assert_eq!(c.score_mode(), ScoreMode::TopScores);
        assert!(!c.score_mode().is_exhaustive(), "pruning must be legal");
        assert!(c.score_mode().needs_scores());
        // Reached through the trait too, since that is what the scorers use.
        assert_eq!(ScoringCollector::score_mode(&c), ScoreMode::TopScores);
    }

    #[test]
    fn a_collector_that_says_nothing_gets_the_conservative_default() {
        struct Bare(u32);
        impl ScoringCollector for Bare {
            fn collect(&mut self, _doc_id: i32, _score: f32) {
                self.0 += 1;
            }
        }
        let c = Bare(0);
        assert_eq!(c.score_mode(), ScoreMode::Complete);
        assert!(
            c.score_mode().is_exhaustive(),
            "no pruning without opting in"
        );
        assert_eq!(c.min_competitive_score(), None);
    }

    #[test]
    fn pruning_threshold_is_min_competitive_score_gated_on_the_score_mode() {
        // Non-exhaustive: the two agree.
        let mut top = TopDocsCollector::new(2);
        top.collect(0, 1.0);
        top.collect(1, 2.0);
        assert_eq!(top.min_competitive_score(), Some(1.0));
        assert_eq!(ScoringCollector::pruning_threshold(&top), Some(1.0));

        // Exhaustive: a threshold may not be handed out even if one exists.
        // (`TopDocsCollector` also refuses to compute one in this mode, so
        // force the interesting case with a collector that does.)
        struct AlwaysCompetitive;
        impl ScoringCollector for AlwaysCompetitive {
            fn collect(&mut self, _doc_id: i32, _score: f32) {}
            fn min_competitive_score(&self) -> Option<f32> {
                Some(42.0)
            }
            fn score_mode(&self) -> ScoreMode {
                ScoreMode::Complete
            }
        }
        let c = AlwaysCompetitive;
        assert_eq!(c.min_competitive_score(), Some(42.0));
        assert_eq!(
            c.pruning_threshold(),
            None,
            "an exhaustive collector must not authorize pruning, whatever its \
             min competitive score says"
        );

        struct TopScoresMode;
        impl ScoringCollector for TopScoresMode {
            fn collect(&mut self, _doc_id: i32, _score: f32) {}
            fn min_competitive_score(&self) -> Option<f32> {
                Some(7.0)
            }
            fn score_mode(&self) -> ScoreMode {
                ScoreMode::TopScores
            }
        }
        assert_eq!(TopScoresMode.pruning_threshold(), Some(7.0));
    }

    #[test]
    fn total_hits_displays_like_javas_to_string() {
        assert_eq!(
            TotalHits {
                value: 12,
                relation: TotalHitsRelation::EqualTo
            }
            .to_string(),
            "12 hits"
        );
        assert_eq!(
            TotalHits {
                value: 1000,
                relation: TotalHitsRelation::GreaterThanOrEqualTo
            }
            .to_string(),
            "1000+ hits"
        );
    }

    #[test]
    fn a_zero_top_n_collector_still_counts_every_hit_exactly() {
        let mut c = TopDocsCollector::new(0);
        for doc in 0..7 {
            c.collect(doc, 1.0);
        }
        assert!(c.top_docs().is_empty());
        assert_eq!(c.total_hits().value, 7);
        assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
    }

    // ---- `searchAfter` -------------------------------------------------

    /// A page walk over a queue that is deliberately full of score ties, so
    /// the boundary is decided by the doc-id half of the rule.
    #[test]
    fn after_pages_walk_the_ranking_without_repeating_or_losing_a_hit() {
        let hits: Vec<(i32, f32)> = (0..9).map(|d| (d, if d < 4 { 2.0 } else { 1.0 })).collect();
        let mut pages: Vec<Vec<ScoreDoc>> = Vec::new();
        let mut after: Option<ScoreDoc> = None;
        for _ in 0..3 {
            let mut c = TopDocsCollector::new(3);
            if let Some(a) = after {
                c = c.with_after(a);
            }
            for &(doc, score) in &hits {
                c.collect(doc, score);
            }
            let page = c.top_docs().to_vec();
            after = page.last().copied();
            pages.push(page);
        }
        let walked: Vec<i32> = pages.iter().flatten().map(|h| h.doc_id).collect();
        assert_eq!(walked, (0..9).collect::<Vec<_>>());
        // The score run boundary falls inside page 2, which is the case a
        // score-only comparison gets wrong.
        assert_eq!(
            pages[1].iter().map(|h| h.score).collect::<Vec<_>>(),
            vec![2.0, 1.0, 1.0]
        );
    }

    #[test]
    fn a_paged_hit_is_still_counted_in_total_hits() {
        // `TopScoreDocCollector` increments `totalHits` *before* the `after`
        // test, so page 2 reports the same total page 1 did.
        let mut c = TopDocsCollector::new(2).with_after(ScoreDoc {
            doc_id: 1,
            score: 5.0,
        });
        for doc in 0..6 {
            c.collect(doc, 5.0);
        }
        assert_eq!(c.total_hits().value, 6);
        assert_eq!(
            c.top_docs().iter().map(|h| h.doc_id).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn after_rejects_a_tie_on_a_lower_doc_id_and_keeps_a_higher_one() {
        // The exact boundary: `score == afterScore && doc <= afterDoc`.
        let mut c = TopDocsCollector::new(5).with_after(ScoreDoc {
            doc_id: 3,
            score: 1.0,
        });
        c.collect(3, 1.0); // the `after` hit itself
        c.collect(2, 1.0); // ranked above it (tie, lower doc id)
        c.collect(4, 1.0); // ranked below it
        c.collect(0, 2.0); // outranks it on score
        c.collect(9, 0.5); // below it on score
        assert_eq!(
            c.top_docs().iter().map(|h| h.doc_id).collect::<Vec<_>>(),
            vec![4, 9]
        );
    }

    // ---- `MaxScoreAccumulator` -----------------------------------------

    #[test]
    fn the_doc_score_code_orders_exactly_as_the_hit_queue_does() {
        // The claim the whole single-atomic design rests on: integer order over
        // the code is `rank_order` over the pair.
        let pairs = [
            (0i32, f32::NEG_INFINITY),
            (0, 0.0f32),
            (5, 0.0),
            (i32::MAX, 0.0),
            (0, 1.0),
            (7, 1.0),
            (3, 12.5),
            (0, f32::MAX),
        ];
        for &(da, sa) in &pairs {
            for &(db, sb) in &pairs {
                let code_order =
                    doc_score_encoder::encode(da, sa).cmp(&doc_score_encoder::encode(db, sb));
                let rank = rank_order(
                    &ScoreDoc {
                        doc_id: da,
                        score: sa,
                    },
                    &ScoreDoc {
                        doc_id: db,
                        score: sb,
                    },
                );
                assert_eq!(code_order, rank, "({da},{sa}) vs ({db},{sb})");
            }
        }
    }

    #[test]
    fn the_doc_score_code_round_trips_both_halves() {
        for &(doc, score) in &[
            (0i32, 0.0f32),
            (1, 1.0),
            (123_456, 0.25),
            (i32::MAX, f32::MAX),
            (0, f32::NEG_INFINITY),
        ] {
            let code = doc_score_encoder::encode(doc, score);
            assert_eq!(doc_score_encoder::doc_id(code), doc, "doc {doc}");
            assert_eq!(
                doc_score_encoder::to_score(code).to_bits(),
                score.to_bits(),
                "score {score}"
            );
        }
    }

    #[test]
    fn an_empty_accumulator_publishes_no_threshold() {
        let acc = MaxScoreAccumulator::new();
        assert_eq!(acc.threshold_for(0), None);
        assert_eq!(MaxScoreAccumulator::default().threshold_for(0), None);
    }

    #[test]
    fn the_accumulator_keeps_the_best_pair_and_tie_breaks_on_the_doc_base() {
        let acc = MaxScoreAccumulator::new();
        acc.accumulate(100, 1.0);
        acc.accumulate(50, 0.5); // worse score: ignored
                                 // Same score, higher doc id: also worse.
        acc.accumulate(200, 1.0);
        // A leaf starting at or after doc 100 cannot win a tie on 1.0, so it
        // may demand strictly more -- expressed under this crate's
        // `bound <= threshold` rule as the score itself.
        assert_eq!(acc.threshold_for(100), Some(1.0));
        assert_eq!(acc.threshold_for(1_000), Some(1.0));
        // A leaf that may hold smaller doc ids could still win a tie, so its
        // threshold is one ULP lower.
        assert_eq!(acc.threshold_for(0), Some(1.0f32.next_down()));
        assert_eq!(acc.threshold_for(99), Some(1.0f32.next_down()));
    }

    #[test]
    fn a_shared_accumulator_raises_a_leafs_threshold_above_its_own_queue() {
        let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
        // Another leaf has already proved 10.0 is competitive, at a doc id
        // below this leaf's base.
        acc.accumulate(0, 10.0);
        let mut c = TopDocsCollector::new(2).with_shared_max_score(std::sync::Arc::clone(&acc), 50);
        // Nothing collected yet: the local half has no answer, the shared half
        // does.
        assert_eq!(c.min_competitive_score(), Some(10.0));
        c.collect(0, 1.0);
        c.collect(1, 2.0);
        // The local half now says 1.0; the shared half still wins.
        assert_eq!(c.min_competitive_score(), Some(10.0));
        // Once this leaf beats it, its own queue takes over -- and it has
        // published its worst kept hit into the accumulator, at global doc ids.
        c.collect(2, 20.0);
        c.collect(3, 30.0);
        assert_eq!(c.min_competitive_score(), Some(20.0));
        assert_eq!(acc.threshold_for(0), Some(20.0f32.next_down()));
        assert_eq!(
            doc_score_encoder::doc_id(acc.acc.load(std::sync::atomic::Ordering::Relaxed)),
            52,
            "the accumulated doc id is global: doc 2 plus doc_base 50"
        );
    }

    #[test]
    fn the_accumulator_is_written_only_when_this_leafs_own_bar_rises() {
        // Java gates `minScoreAcc.accumulate` on `localMinScore >
        // minCompetitiveScore`; without that gate the atomic read-modify-write
        // is on the per-document path. What must not happen is the gate
        // *suppressing* a publish the accumulator needed.
        let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
        let mut c = TopDocsCollector::new(2).with_shared_max_score(std::sync::Arc::clone(&acc), 0);
        // Descending scores: the queue fills, then every later hit loses, so the
        // bar rises exactly once.
        for (doc, score) in [(0i32, 9.0f32), (1, 8.0), (2, 7.0), (3, 6.0), (4, 5.0)] {
            c.collect(doc, score);
        }
        assert_eq!(c.min_competitive_score(), Some(8.0));
        assert_eq!(acc.threshold_for(1), Some(8.0));

        // Ascending scores: the bar rises on nearly every hit, and the
        // accumulator must end on the last one, not the first.
        let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
        let mut c = TopDocsCollector::new(2).with_shared_max_score(std::sync::Arc::clone(&acc), 0);
        for (doc, score) in [(0i32, 1.0f32), (1, 2.0), (2, 3.0), (3, 4.0), (4, 5.0)] {
            c.collect(doc, score);
        }
        assert_eq!(c.min_competitive_score(), Some(4.0));
        assert_eq!(acc.threshold_for(4), Some(4.0));
    }

    #[test]
    fn a_shared_threshold_alone_makes_the_hit_count_a_lower_bound() {
        // The collector's own queue is not full, so it has published nothing --
        // but another leaf has, and a scorer is allowed to skip against it, so
        // `totalHits` can no longer claim to be exact. Java's
        // `updateGlobalMinCompetitiveScore` sets the relation for exactly this
        // reason.
        let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
        acc.accumulate(0, 100.0);
        let mut c = TopDocsCollector::new(5).with_shared_max_score(acc, 10);
        c.collect(0, 1.0);
        assert!(c.top_docs().len() < 5, "the queue is deliberately not full");
        assert_eq!(c.local_min_competitive_score(), None);
        assert_eq!(c.min_competitive_score(), Some(100.0));
        assert_eq!(
            c.total_hits().relation,
            TotalHitsRelation::GreaterThanOrEqualTo
        );
    }

    #[test]
    fn an_exhaustive_collector_publishes_nothing_into_the_accumulator() {
        // `total_hits_threshold == u64::MAX` is `ScoreMode::COMPLETE`, and a
        // collector promising an exact count must not let any leaf skip.
        let acc = std::sync::Arc::new(MaxScoreAccumulator::new());
        let mut c = TopDocsCollector::with_total_hits_threshold(1, u64::MAX)
            .with_shared_max_score(std::sync::Arc::clone(&acc), 0);
        for doc in 0..5 {
            c.collect(doc, 1.0);
        }
        assert_eq!(acc.threshold_for(0), None);
        assert_eq!(c.pruning_threshold(), None);
        assert_eq!(c.total_hits().relation, TotalHitsRelation::EqualTo);
    }
}
