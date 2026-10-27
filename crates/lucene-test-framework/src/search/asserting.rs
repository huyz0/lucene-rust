//! `AssertingIndexSearcher` and `AssertingCollector`.
//!
//! Java's searcher wraps every `Weight` it creates in an `AssertingWeight`,
//! whose scorers and bulk scorers check their contracts. This port's scorer
//! tree is private to `lucene-search`, so the wrappers live there
//! (`exec::asserting`, behind its `test-support` feature) and
//! [`AssertingIndexSearcher`] switches them on around each search with
//! [`lucene_search::test_support::asserting`]. The collector side --
//! documents arriving in global order, leaf after leaf -- is
//! [`AssertingCollector`], here.

use std::cell::RefCell;

use lucene_search::collector::{ScoreMode, ScoringCollector, TopDocsCollector};
use lucene_search::explain::Explanation;
use lucene_search::index_searcher::{IndexSearcher, TOTAL_HITS_THRESHOLD};
use lucene_search::query::BooleanQuery;
use lucene_search::test_support;
use lucene_search::top_docs::{ShardScoreDoc, TopDocs};
use lucene_search::Result;
use lucene_util::java_random::JavaRandom;

use super::query_utils;

/// `AssertingCollector`: every document is collected once, in ascending
/// global order across the whole search (not only within a segment), with a
/// score that is a number at or above zero.
pub struct AssertingCollector<'c, C: ?Sized> {
    inner: &'c mut C,
    /// The largest global document collected so far, `-1` before the first.
    max_doc: i32,
}

impl<'c, C: ScoringCollector + ?Sized> AssertingCollector<'c, C> {
    /// `AssertingCollector.wrap(in)`.
    pub fn wrap(inner: &'c mut C) -> Self {
        Self { inner, max_doc: -1 }
    }

    fn check(&mut self, doc: i32) {
        assert!(
            doc > self.max_doc,
            "collection is not in order: current doc={doc} while {} has already been collected",
            self.max_doc
        );
        self.max_doc = doc;
    }
}

impl<C: ScoringCollector + ?Sized> ScoringCollector for AssertingCollector<'_, C> {
    fn collect(&mut self, doc: i32, score: f32) {
        self.check(doc);
        assert!(score >= 0.0, "score={score} for doc {doc}");
        self.inner.collect(doc, score);
    }
    fn min_competitive_score(&self) -> Option<f32> {
        self.inner.min_competitive_score()
    }
    fn score_mode(&self) -> ScoreMode {
        self.inner.score_mode()
    }
    fn pruning_threshold(&self) -> Option<f32> {
        self.inner.pruning_threshold()
    }
    fn constant_score_hits_needed(&self) -> Option<u64> {
        self.inner.constant_score_hits_needed()
    }
    fn count_losing_hits(&mut self, n: u64, score: f32) -> bool {
        self.inner.count_losing_hits(n, score)
    }
    fn add_hits(&mut self, n: u64) -> bool {
        self.inner.add_hits(n)
    }
    fn collect_range(&mut self, min: i32, max: i32, score: f32) {
        assert!(max > min, "empty range [{min}, {max})");
        self.check(min);
        assert!(score >= 0.0, "score={score} for docs [{min}, {max})");
        self.inner.collect_range(min, max, score);
        self.max_doc = max - 1;
    }
    // `collect_many` keeps its default (collect nothing in bulk), so every
    // document comes through `collect` and its check.
}

/// `AssertingIndexSearcher`: a searcher whose every search runs its scorers
/// and bulk scorers under the asserting wrappers, and its collector under
/// [`AssertingCollector`].
pub struct AssertingIndexSearcher<'r, 's, 'a> {
    inner: &'r IndexSearcher<'s, 'a>,
    random: RefCell<JavaRandom>,
}

impl<'r, 's, 'a> AssertingIndexSearcher<'r, 's, 'a> {
    /// `new AssertingIndexSearcher(random, reader)`: draws its own random
    /// from `random`.
    pub fn new(random: &mut JavaRandom, inner: &'r IndexSearcher<'s, 'a>) -> Self {
        Self {
            inner,
            random: RefCell::new(JavaRandom::new(random.next_long())),
        }
    }

    /// The searcher wrapped.
    pub fn searcher(&self) -> &'r IndexSearcher<'s, 'a> {
        self.inner
    }

    fn guard(&self) -> test_support::Asserting {
        test_support::asserting(self.random.borrow_mut().next_long())
    }

    /// `search(query, collector)`.
    ///
    /// # Errors
    /// Whatever the search reports.
    pub fn search_collector<C: ScoringCollector + ?Sized>(
        &self,
        query: &BooleanQuery,
        collector: &mut C,
    ) -> Result<()> {
        self.rewrite(query)?;
        let _guard = self.guard();
        let mut asserting = AssertingCollector::wrap(collector);
        self.inner.search_collector(query, &mut asserting)
    }

    /// `rewrite(query)`: `QueryUtils.check` of the query and of its
    /// rewritten form, as every search here makes it.
    ///
    /// # Errors
    /// Whatever rewriting reports.
    pub fn rewrite(&self, query: &BooleanQuery) -> Result<BooleanQuery> {
        query_utils::check(query);
        let rewritten = self.inner.rewrite(query)?;
        query_utils::check(&rewritten);
        Ok(rewritten)
    }

    /// `search(query, n)`: through a `TopScoreDocCollector` (so through the
    /// scorer tree and [`AssertingCollector`]); the hits must be the ones the
    /// wrapped searcher's own `search(query, n)` returns.
    ///
    /// # Errors
    /// Whatever the search reports.
    pub fn search(&self, query: &BooleanQuery, n: usize) -> Result<TopDocs> {
        let mut c = TopDocsCollector::with_total_hits_threshold(n, TOTAL_HITS_THRESHOLD);
        self.search_collector(query, &mut c)?;
        Ok(TopDocs {
            total_hits: c.total_hits(),
            score_docs: c
                .top_docs()
                .iter()
                .map(|h| ShardScoreDoc::new(h.doc_id, h.score))
                .collect(),
        })
    }

    /// `count(query)`.
    ///
    /// # Errors
    /// Whatever the search reports.
    pub fn count(&self, query: &BooleanQuery) -> Result<u64> {
        self.rewrite(query)?;
        let _guard = self.guard();
        self.inner.count(query)
    }

    /// `explain(query, doc)`.
    ///
    /// # Errors
    /// Whatever the explanation reports.
    pub fn explain(&self, query: &BooleanQuery, doc: i32) -> Result<Explanation> {
        self.inner.explain(query, doc)
    }
}
