//! The seam `lucene-test-framework`'s search checks (`QueryUtils`,
//! `CheckHits`, `AssertingIndexSearcher`) need into this crate's scorer
//! tree, which is otherwise private: a query's per-segment scorer and bulk
//! scorer (`Weight.scorer(leaf)`, `Weight.bulkScorer(leaf)`), and the
//! switch that wraps every scorer and bulk scorer a search builds in the
//! asserting wrappers (`AssertingWeight`, `AssertingScorer`,
//! `AssertingBulkScorer`, `AssertingLeafCollector`).
//!
//! Compiled only under the `asserting` feature, which nothing but
//! `lucene-test-framework` (itself only ever a dev-dependency) turns on: a
//! `-p lucene-ffi` build of the cdylib and the benchmark runner have none of
//! it (a whole-workspace build unifies the feature in).

use std::sync::{Arc, Mutex};

pub mod scorer_only;

use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::java_random::JavaRandom;

use crate::collector::{ScoreMode, ScoringCollector};
use crate::exec::asserting::{AssertingBulk, AssertingGuard};
use crate::exec::{self, BoxScorer, Bulk, LeafContext, Mode};
use crate::index_searcher::IndexSearcher;
use crate::query::{BooleanQuery, Clause};
use crate::{Error, Result};

/// `DocIdSetIterator.NO_MORE_DOCS`.
pub const NO_MORE_DOCS: i32 = exec::NO_MORE_DOCS;

/// `Query.toString()` with the default field empty, as explanations render it.
/// Exposed only through the existing test seam for differential query checks.
pub fn query_string(query: &Clause) -> String {
    crate::explain::describe_clause(query)
}

/// While held, every search on this thread runs under the asserting
/// wrappers (`new AssertingIndexSearcher(random, reader)`): each scorer the
/// tree is built from checks its iteration contract (`AssertingScorer`),
/// and each segment is scored in random windows (`AssertingBulkScorer`)
/// whose collectors check order and range (`AssertingLeafCollector`). A
/// broken contract panics. `seed` seeds the windows.
#[must_use = "the wrappers are removed when the guard is dropped"]
pub struct Asserting {
    _guard: AssertingGuard,
}

/// Installs the asserting wrappers until the returned guard is dropped.
pub fn asserting(seed: i64) -> Asserting {
    Asserting {
        _guard: AssertingGuard::new(seed),
    }
}

/// Whether the asserting wrappers are installed on this thread.
pub fn asserting_active() -> bool {
    exec::asserting::active()
}

fn mode_of(score_mode: ScoreMode) -> Mode {
    match score_mode {
        m if !m.needs_scores() => Mode::NoScores,
        ScoreMode::TopScores => Mode::TopScores,
        _ => Mode::Complete,
    }
}

/// One segment of a searcher, with a query prepared against the whole
/// reader (`searcher.rewrite(query)` and `createWeight`'s reader-wide
/// statistics): what [`with_leaf`] hands its closure.
pub struct Leaf<'q> {
    ctx: LeafContext<'q>,
    query: &'q BooleanQuery,
    doc_base: i32,
}

impl<'q> Leaf<'q> {
    /// `Weight.scorerSupplier(leaf).get(leadCost)` under `score_mode`, the
    /// query's top-level scorer (`setTopLevelScoringClause`); `None` when
    /// it matches nothing in this segment.
    ///
    /// # Errors
    /// Whatever building the scorer tree reports.
    pub fn scorer(&self, score_mode: ScoreMode) -> Result<Option<LeafScorer<'q>>> {
        let mode = mode_of(score_mode);
        // Borrow the prepared query, as Weight.scorer does, instead of cloning its tree.
        let s = exec::build::build_boolean(&self.ctx, self.query, 1.0, mode, true)?;
        // Preserve build()'s AssertingWeight wrapper at this borrowed root.
        let s = s.map(|inner| exec::asserting::wrap(inner, mode, true));
        Ok(s.map(|inner| LeafScorer { inner }))
    }

    /// `Weight.bulkScorer(leaf)` under `score_mode`; `None` when it matches
    /// nothing in this segment. Under [`asserting`] it is an
    /// `AssertingBulkScorer`.
    ///
    /// # Errors
    /// Whatever building the scorer tree reports.
    pub fn bulk_scorer(&self, score_mode: ScoreMode) -> Result<Option<LeafBulkScorer<'q>>> {
        let mode = mode_of(score_mode);
        let asserting = AssertingBulk::installed(self.ctx.max_doc);
        Ok(
            exec::bulk_boolean(&self.ctx, self.query, 1.0, mode)?.map(|bulk| LeafBulkScorer {
                bulk,
                mode,
                asserting,
            }),
        )
    }

    /// `context.reader().getLiveDocs()`.
    pub fn live_docs(&self) -> Option<&'q FixedBitSet> {
        self.ctx.live_docs
    }

    /// `context.reader().maxDoc()`, when known.
    pub fn max_doc(&self) -> Option<i32> {
        self.ctx.max_doc
    }

    /// `context.docBase`.
    pub fn doc_base(&self) -> i32 {
        self.doc_base
    }
}

/// Runs `f` over segment `leaf` of `searcher`, `query` prepared as a search
/// prepares it (rewritten against the reader, statistics reader-wide, the
/// searcher's similarity).
///
/// # Errors
/// [`Error::SliceOutOfRange`] for a segment the searcher lacks, whatever
/// preparing the query reports, and whatever `f` returns.
pub fn with_leaf<R>(
    searcher: &IndexSearcher<'_, '_>,
    query: &BooleanQuery,
    leaf: usize,
    f: impl FnOnce(&Leaf<'_>) -> Result<R>,
) -> Result<R> {
    let segments = searcher.segments();
    let seg = segments.get(leaf).ok_or(Error::SliceOutOfRange {
        segment: leaf,
        segments: segments.len(),
    })?;
    let (rewritten, global) = searcher.prepare(query)?;
    let query = rewritten.as_ref().unwrap_or(query);
    let norms = searcher.norms(leaf);
    let leaf = Leaf {
        ctx: LeafContext {
            fields: seg.fields,
            doc_in: seg.doc_in,
            pos_in: seg.pos_in,
            pay_in: seg.pay_in,
            live_docs: seg.live_docs,
            points: seg.points,
            norms,
            global: Some(&global),
            max_doc: seg.max_doc,
            cache: seg.cache,
            reader: seg.reader,
            similarity: searcher.similarity().filter(|s| !s.is_default_bm25()),
        },
        query,
        doc_base: seg.doc_base,
    };
    f(&leaf)
}

/// A segment's scorer: `Scorer`, its `iterator()` and its
/// `twoPhaseIterator()` in one, as this crate's scorer tree has them. For
/// a [`Self::two_phase`] scorer, [`Self::next_doc`] and [`Self::advance`]
/// move the approximation and [`Self::matches`] confirms; the exact
/// iterator is [`Self::exact_next`] and [`Self::exact_advance`].
pub struct LeafScorer<'a> {
    inner: BoxScorer<'a>,
}

impl LeafScorer<'_> {
    /// `docID()`.
    pub fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    /// The approximation's `nextDoc()`.
    ///
    /// # Errors
    /// Whatever reading the postings reports.
    pub fn next_doc(&mut self) -> Result<i32> {
        self.inner.next_doc()
    }
    /// The approximation's `advance(target)`.
    ///
    /// # Errors
    /// Whatever reading the postings reports.
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        self.inner.advance(target)
    }
    /// `iterator().nextDoc()`: the next match.
    ///
    /// # Errors
    /// Whatever reading the postings reports.
    pub fn exact_next(&mut self) -> Result<i32> {
        exec::exact_next(&mut *self.inner)
    }
    /// `iterator().advance(target)`: the first match at or after `target`.
    ///
    /// # Errors
    /// Whatever reading the postings reports.
    pub fn exact_advance(&mut self, target: i32) -> Result<i32> {
        exec::exact_advance(&mut *self.inner, target)
    }
    /// `iterator().cost()`.
    pub fn cost(&self) -> i64 {
        self.inner.cost()
    }
    /// Whether `twoPhaseIterator()` is not `null`.
    pub fn two_phase(&self) -> bool {
        self.inner.two_phase()
    }
    /// `twoPhaseIterator().matches()`.
    ///
    /// # Errors
    /// Whatever confirming the document reports.
    pub fn matches(&mut self) -> Result<bool> {
        self.inner.matches()
    }
    /// `twoPhaseIterator().matchCost()`.
    pub fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }
    /// `score()`.
    ///
    /// # Errors
    /// Whatever reading the frequencies or norms reports.
    pub fn score(&mut self) -> Result<f32> {
        self.inner.score()
    }
    /// `advanceShallow(target)`.
    ///
    /// # Errors
    /// Whatever reading the impacts reports.
    pub fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.inner.advance_shallow(target)
    }
    /// `getMaxScore(upTo)`.
    ///
    /// # Errors
    /// Whatever reading the impacts reports.
    pub fn max_score(&mut self, up_to: i32) -> Result<f32> {
        self.inner.max_score(up_to)
    }
    /// `setMinCompetitiveScore(min)`.
    ///
    /// # Errors
    /// Whatever the scorer reports.
    pub fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.inner.set_min_competitive_score(min)
    }
    /// `iterator().docIDRunEnd()`.
    pub fn doc_id_run_end(&self) -> i32 {
        self.inner.doc_id_run_end()
    }
}

/// A segment's bulk scorer (`BulkScorer`).
pub struct LeafBulkScorer<'a> {
    bulk: Bulk<'a>,
    mode: Mode,
    asserting: Option<AssertingBulk>,
}

impl LeafBulkScorer<'_> {
    /// Test-only introspection of Java's `Weight.DefaultBulkScorer` path.
    pub fn uses_default_scorer(&self) -> bool {
        matches!(&self.bulk, Bulk::Scorer(..))
    }

    /// `cost()`: the bulk scorer's own, which `AssertingBulkScorer`
    /// forwards.
    pub fn cost(&self) -> i64 {
        self.bulk.cost()
    }

    /// `score(collector, acceptDocs, min, max)`: collects the matches in
    /// `[min, max)` not deleted in `accept_docs`, returning the next
    /// document to score.
    ///
    /// # Errors
    /// Whatever scoring reports.
    pub fn score(
        &mut self,
        collector: &mut dyn ScoringCollector,
        accept_docs: Option<&FixedBitSet>,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        match &mut self.asserting {
            Some(a) => a.score(&mut self.bulk, self.mode, accept_docs, collector, min, max),
            None => self.bulk.score(self.mode, accept_docs, collector, min, max),
        }
    }
}

/// A query only tests build: lucene-test-framework's
/// `RandomApproximationQuery`, `BlockScoreQueryWrapper`,
/// `DisablingBulkScorerQuery`, or one of `TestQueryUtils`' broken queries.
/// Into a [`Clause`] with [`TestQuery::random_approximation`] and
/// [`TestQuery::broken`]; it reaches the scorer tree as
/// [`crate::extended_query::ExtendedQuery::Test`]. Its explanation and
/// matches are the wrapped query's. Transparent wrappers delegate cacheability;
/// random and broken scorers are never cached, so checks observe their scorers.
#[derive(Debug, Clone, PartialEq)]
pub struct TestQuery {
    query: Clause,
    kind: TestKind,
}

/// Which [`TestQuery`].
#[derive(Clone)]
pub(crate) enum TestKind {
    /// `RandomApproximationQuery`: the random each segment's scorer draws
    /// its own from (`new Random(random.nextLong())`).
    RandomApproximation(Arc<Mutex<JavaRandom>>),
    Broken(BrokenKind),
    BlockScore(i32),
    DisableBulk,
}

impl std::fmt::Debug for TestKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestKind::RandomApproximation(_) => f.write_str("RandomApproximation"),
            TestKind::Broken(k) => write!(f, "Broken({k:?})"),
            TestKind::BlockScore(length) => write!(f, "BlockScore({length})"),
            TestKind::DisableBulk => f.write_str("DisableBulk"),
        }
    }
}

/// `RandomApproximationQuery.equals` ignores the random; the broken queries
/// are equal when they break the same way.
impl PartialEq for TestKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TestKind::RandomApproximation(_), TestKind::RandomApproximation(_)) => true,
            (TestKind::Broken(a), TestKind::Broken(b)) => a == b,
            (TestKind::BlockScore(a), TestKind::BlockScore(b)) => a == b,
            (TestKind::DisableBulk, TestKind::DisableBulk) => true,
            _ => false,
        }
    }
}

/// How a `TestQueryUtils` broken scorer breaks its contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrokenKind {
    /// `BrokenAdvanceQuery`: `advance` always returns `NO_MORE_DOCS`.
    Advance,
    /// `SkipTooFarQuery`: `advance` steps one match past its target's.
    SkipTooFar,
    /// `UnstableScoreQuery`: every second `score()` of a document adds 1.
    UnstableScore,
    /// `ScoreMismatchQuery`: a document reached by `advance` scores 100 more.
    ScoreMismatch,
    /// `EarlyTerminationQuery`: `nextDoc` stops after its first call.
    EarlyTermination,
}

impl TestQuery {
    /// `new BlockScoreQueryWrapper(query, blockLength)`.
    /// Invalid lengths remain accepted at construction, as in Java.
    pub fn block_score(query: impl Into<Clause>, block_length: i32) -> Clause {
        Self::clause(Self {
            query: query.into(),
            kind: TestKind::BlockScore(block_length),
        })
    }

    /// `new DisablingBulkScorerQuery(query)`: use the ordinary scorer path.
    pub fn disabling_bulk_scorer(query: impl Into<Clause>) -> Clause {
        Self::clause(Self {
            query: query.into(),
            kind: TestKind::DisableBulk,
        })
    }

    /// Whether Java delegates visit, display, rewrite and cacheability.
    pub(crate) fn delegates_query_semantics(&self) -> bool {
        matches!(self.kind, TestKind::BlockScore(_) | TestKind::DisableBulk)
    }

    /// Java's transparent wrappers retain themselves around a rewritten child.
    pub(crate) fn rewrite(mut self) -> Self {
        if self.delegates_query_semantics() {
            self.query = self.query.rewrite();
        }
        self
    }

    /// `new RandomApproximationQuery(query, random)`.
    pub fn random_approximation(query: impl Into<Clause>, random: JavaRandom) -> Clause {
        Self::clause(TestQuery {
            query: query.into(),
            kind: TestKind::RandomApproximation(Arc::new(Mutex::new(random))),
        })
    }

    /// One of `TestQueryUtils`' broken queries around `query`.
    pub fn broken(query: impl Into<Clause>, kind: BrokenKind) -> Clause {
        Self::clause(TestQuery {
            query: query.into(),
            kind: TestKind::Broken(kind),
        })
    }

    fn clause(q: TestQuery) -> Clause {
        Clause::Extended(Box::new(crate::extended_query::ExtendedQuery::Test(q)))
    }

    /// The wrapped query.
    pub fn query(&self) -> &Clause {
        &self.query
    }

    /// Mutable child for leaf-local MatchAll explanation preparation.
    pub(crate) fn query_mut(&mut self) -> &mut Clause {
        &mut self.query
    }

    pub(crate) fn kind(&self) -> &TestKind {
        &self.kind
    }

    /// The query's class name.
    pub fn name(&self) -> &'static str {
        match self.kind {
            TestKind::RandomApproximation(_) => "RandomApproximationQuery",
            TestKind::Broken(_) => "BrokenQuery",
            TestKind::BlockScore(_) => "BlockScoreQueryWrapper",
            TestKind::DisableBulk => "DisablingBulkScorerQuery",
        }
    }
}

#[cfg(test)]
mod tests;
