//! Lucene 10.5.0's `ScorerIndexSearcher`: use the ordinary scorer tree and
//! `DefaultBulkScorer` at every searched leaf, bypassing specialized bulk scorers.
//!
//! Java subclasses `IndexSearcher` and wraps its leaf weight. Rust owns a
//! configured `IndexSearcher`, whose feature-gated dispatch is explicit state
//! carried to executor slices. Query rewriting, reader-wide statistics,
//! similarities, live documents, doc bases and timeouts use the normal searcher.
//! The supported entry points are top hits, search-after, count, collector and
//! collector-manager searches. Field sorting/facets and leaf subranges are not
//! exposed: those APIs have separate dispatch in the current Rust searcher.

use std::sync::Arc;

use lucene_search::collector::{ScoreDoc, ScoringCollector};
use lucene_search::collectors::CollectorManager;
use lucene_search::explain::Explanation;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::BooleanQuery;
use lucene_search::reader::exitable::QueryTimeout;
use lucene_search::similarities::Similarity;
use lucene_search::test_support::scorer_only;
use lucene_search::top_docs::TopDocs;
use lucene_search::Result;

/// A searcher that always drives a `Scorer` through `DefaultBulkScorer`.
/// Java: `ScorerIndexSearcher`.
pub struct ScorerIndexSearcher<'s, 'a> {
    inner: IndexSearcher<'s, 'a>,
}

impl<'s, 'a> ScorerIndexSearcher<'s, 'a> {
    /// Java: `ScorerIndexSearcher(IndexReader)`.
    ///
    /// # Errors
    /// [`lucene_search::Error::IllegalArgument`] if norms and segment counts differ.
    pub fn new(segments: &'s [OpenSegment<'a>], norms: &'s [SegmentNorms<'s, 'a>]) -> Result<Self> {
        Ok(Self::from_searcher(IndexSearcher::new(segments, norms)?))
    }

    /// Keeps an existing searcher's similarity, slicing and timeout settings.
    pub fn from_searcher(searcher: IndexSearcher<'s, 'a>) -> Self {
        Self {
            inner: scorer_only::enable(searcher),
        }
    }

    /// Java: `IndexSearcher.setSimilarity`.
    pub fn set_similarity(&mut self, similarity: &'s dyn Similarity) {
        self.inner.set_similarity(similarity);
    }

    /// Java: `IndexSearcher.getSimilarity`; `None` is default BM25.
    pub fn similarity(&self) -> Option<&'s dyn Similarity> {
        self.inner.similarity()
    }

    /// Configures executor leaf slices; several slices run concurrently.
    /// Java: `ScorerIndexSearcher(IndexReader, Executor)`.
    ///
    /// # Errors
    /// [`lucene_search::Error::SliceOutOfRange`] for an invalid segment index.
    pub fn set_slices(&mut self, slices: Vec<Vec<usize>>) -> Result<()> {
        self.inner.set_slices(slices)
    }

    /// Java: `IndexSearcher.getSlices`.
    pub fn slices(&self) -> &[Vec<usize>] {
        self.inner.slices()
    }

    /// Java: `IndexSearcher.setTimeout`.
    pub fn set_timeout(&mut self, timeout: Option<Arc<dyn QueryTimeout>>) {
        self.inner.set_timeout(timeout);
    }

    /// Java: `IndexSearcher.getTimeout`.
    pub fn timeout(&self) -> Option<&Arc<dyn QueryTimeout>> {
        self.inner.timeout()
    }

    /// Java: `IndexSearcher.timedOut`.
    pub fn timed_out(&self) -> bool {
        self.inner.timed_out()
    }

    /// Java: `IndexSearcher.search(Query, int)`.
    ///
    /// # Errors
    /// Whatever query preparation or scorer execution reports.
    pub fn search(&self, query: &BooleanQuery, n: usize) -> Result<TopDocs> {
        self.inner.search(query, n)
    }

    /// Java: `IndexSearcher.searchAfter`.
    ///
    /// # Errors
    /// Whatever query preparation or scorer execution reports.
    pub fn search_after(&self, after: ScoreDoc, query: &BooleanQuery, n: usize) -> Result<TopDocs> {
        self.inner.search_after(after, query, n)
    }

    /// Java: `IndexSearcher.search(Query, Collector)`.
    ///
    /// # Errors
    /// Whatever query preparation or scorer execution reports.
    pub fn search_collector<C: ScoringCollector + ?Sized>(
        &self,
        query: &BooleanQuery,
        collector: &mut C,
    ) -> Result<()> {
        self.inner.search_collector(query, collector)
    }

    /// Java: `IndexSearcher.search(List<LeafReaderContext>, Weight, Collector)`.
    ///
    /// # Errors
    /// Invalid slice indices or query/scorer failures.
    pub fn search_slice_collector<C: ScoringCollector + ?Sized>(
        &self,
        query: &BooleanQuery,
        slice: &[usize],
        collector: &mut C,
    ) -> Result<()> {
        self.inner.search_slice_collector(query, slice, collector)
    }

    /// Java: `IndexSearcher.search(Query, CollectorManager)`. The explicit
    /// dispatch state is shared by every worker; no thread-local is installed.
    ///
    /// # Errors
    /// Collector creation/reduction or query/scorer failures.
    pub fn search_manager<M: CollectorManager>(
        &self,
        query: &BooleanQuery,
        manager: &M,
    ) -> Result<M::Output> {
        self.inner.search_manager(query, manager)
    }

    /// Java: `IndexSearcher.count`.
    ///
    /// # Errors
    /// Whatever query preparation or scorer execution reports.
    pub fn count(&self, query: &BooleanQuery) -> Result<u64> {
        self.inner.count(query)
    }

    /// Java: `IndexSearcher.explain`, independent of bulk scoring.
    ///
    /// # Errors
    /// Invalid document ids or query/explanation failures.
    pub fn explain(&self, query: &BooleanQuery, doc: i32) -> Result<Explanation> {
        self.inner.explain(query, doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_search::collector::TopDocsCollector;
    use lucene_search::query::{Clause, TermQuery};
    use lucene_search::reader::exitable::QueryTimeoutImpl;
    use lucene_search::similarities::BooleanSimilarity;

    struct EmptyManager;
    impl CollectorManager for EmptyManager {
        type Collector = TopDocsCollector;
        type Output = usize;
        fn new_collector(&self) -> Result<TopDocsCollector> {
            Ok(TopDocsCollector::new(1))
        }
        fn reduce(&self, collectors: Vec<TopDocsCollector>) -> Result<usize> {
            assert!(collectors
                .iter()
                .all(|collector| collector.top_docs().is_empty()));
            Ok(collectors.len())
        }
    }
    #[test]
    fn owned_configuration_and_empty_reader_entry_points() {
        let boolean = BooleanSimilarity;
        let mut searcher = ScorerIndexSearcher::new(&[], &[]).unwrap();
        assert!(searcher.similarity().is_none());
        searcher.set_similarity(&boolean);
        assert!(std::ptr::addr_eq(
            searcher.similarity().unwrap(),
            &boolean as &dyn Similarity
        ));
        let timeout: Arc<dyn QueryTimeout> = Arc::new(QueryTimeoutImpl::new(-1));
        searcher.set_timeout(Some(timeout.clone()));
        assert!(Arc::ptr_eq(searcher.timeout().unwrap(), &timeout));
        assert!(!searcher.timed_out());
        searcher.set_timeout(None);
        assert!(searcher.timeout().is_none());
        assert_eq!(searcher.slices(), &[Vec::<usize>::new()]);
        searcher.set_slices(Vec::new()).unwrap();
        assert!(searcher.slices().is_empty());
        assert!(searcher.set_slices(vec![vec![0]]).is_err());
        let query = BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "absent"))],
            ..BooleanQuery::default()
        };
        assert_eq!(searcher.search(&query, 3).unwrap().total_hits.value, 0);
        assert!(searcher
            .search_after(
                ScoreDoc {
                    doc_id: 0,
                    score: 1.0
                },
                &query,
                3
            )
            .unwrap()
            .score_docs
            .is_empty());
        assert_eq!(searcher.count(&query).unwrap(), 0);
        let mut collector = TopDocsCollector::new(1);
        searcher.search_collector(&query, &mut collector).unwrap();
        searcher
            .search_slice_collector(&query, &[], &mut collector)
            .unwrap();
        assert!(searcher
            .search_slice_collector(&query, &[0], &mut collector)
            .is_err());
        assert!(collector.top_docs().is_empty());
        assert_eq!(searcher.search_manager(&query, &EmptyManager).unwrap(), 1);
        assert!(searcher.explain(&query, 0).is_err());
        assert!(ScorerIndexSearcher::new(&[], &[None]).is_err());
    }
}
