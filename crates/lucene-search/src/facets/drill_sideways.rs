//! `DrillSideways`: a drill-down search that also counts, for each dim drilled
//! into, the facets of the documents that match every *other* dim -- the
//! "sideways" counts a faceted UI shows beside a chosen value, so the user
//! can switch to a sibling value or add one.
//!
//! The sequential search runs one [`DrillSidewaysQuery`] (the base query and
//! every dim scored together); the executor form (`new DrillSideways(...,
//! executor)`, [`DrillSideways::with_executor`]) runs the drill-down query
//! and, per dim, the drill-down query without that dim -- the same counts,
//! from separate searches.
//!
//! Differs:
//! * only taxonomy facets are counted: the `SortedSetDocValuesReaderState`
//!   constructors wait for the sorted-set facet counts over a
//!   `FacetsCollector` (T13.4d2);
//! * the executor's searches run one after the other on the calling thread
//!   (the results are the same);
//! * `search` with a `Sort` (`TopFieldCollectorManager`) is not ported:
//!   [`DrillSideways::search_filtered`] sorts by score;
//! * the `create*FacetsCollectorManager` and `buildFacetsResult` hooks are
//!   not overridable; [`DrillSideways::with_score_sub_docs_at_once`] is the
//!   `scoreSubDocsAtOnce` one;
//! * a collector manager is a [`GroupingCollectorManager`] (Java's
//!   `CollectorManager`); `search(query, null, managers)`'s refusal of a
//!   `null` drill-down manager has no Rust counterpart.

use lucene_codecs::completion::java_hash_map_order;

use crate::collector::{ScoreDoc, ScoreMode, ScoringCollector, TopDocsCollector};
use crate::facets::taxonomy::{FastTaxonomyFacetCounts, TaxonomyReader};
use crate::facets::{
    FacetResult, FacetValue, Facets, FacetsCollector, FacetsCollectorManager, FacetsConfig,
    MultiFacets,
};
use crate::grouping::{search_manager, GroupingCollectorManager};
use crate::index_searcher::IndexSearcher;
use crate::leaf_collector::SegmentCollector;
use crate::multi_segment::OpenSegment;
use crate::query::{Clause, MatchAllDocsQuery};
use crate::top_docs::{self, ShardScoreDoc, TieBreaker, TopDocs};
use crate::{Error, Result};

use super::drill_down_query::DrillDownQuery;
use super::drill_sideways_query::DrillSidewaysQuery;

/// `DrillSideways`.
pub struct DrillSideways<'r, 's, 'a, 't> {
    searcher: &'r IndexSearcher<'s, 'a>,
    config: &'t FacetsConfig,
    taxo_reader: &'t dyn TaxonomyReader,
    /// Whether an executor was given.
    executor: bool,
    score_sub_docs_at_once: bool,
}

/// `DrillSideways.DrillSidewaysResult` (and `ConcurrentDrillSidewaysResult`,
/// whose `collectorResult` is [`Self::hits`] here): the facets, the hits,
/// and the collectors they were counted from.
pub struct DrillSidewaysResult<'t, 'a, R> {
    /// `facets`: the drill-down counts, each drilled dim answered by its
    /// sideways counts.
    pub facets: DrillSidewaysFacets<'t>,
    /// `hits` (or `collectorResult`): the hit manager's result.
    pub hits: R,
    /// `drillDownFacetsCollector`: the documents matching the whole query.
    pub drill_down_facets_collector: FacetsCollector<'a>,
    /// `drillSidewaysFacetsCollector`: per drilled dim, the documents
    /// matching every other dim; `None` without a dim.
    pub drill_sideways_facets_collector: Option<Vec<FacetsCollector<'a>>>,
    /// `drillSidewaysDims`, in the query's order; `None` without a dim.
    pub drill_sideways_dims: Option<Vec<String>>,
}

/// `DrillSideways.Result<T, R>`: the drill-down manager's result and each
/// sideways manager's, in dim order (`None` when the query has no dim and
/// the search ran sequentially, as Java's `null`).
#[derive(Debug)]
pub struct DrillSidewaysSearchResult<T, R> {
    pub drill_down_result: T,
    pub drill_sideways_results: Option<Vec<R>>,
}

/// `buildFacetsResult`'s `Facets`: the drill-down counts alone when no dim
/// was drilled into, else a `MultiFacets` sending each drilled dim to its
/// sideways counts.
pub struct DrillSidewaysFacets<'t> {
    drill_down: FastTaxonomyFacetCounts<'t>,
    /// In Java's `HashMap` iteration order, which `getAllDims` walks.
    sideways: Vec<(String, FastTaxonomyFacetCounts<'t>)>,
}

impl DrillSidewaysFacets<'_> {
    fn with<T>(&self, f: impl FnOnce(&dyn Facets) -> Result<T>) -> Result<T> {
        if self.sideways.is_empty() {
            return f(&self.drill_down);
        }
        let dims: Vec<(String, &dyn Facets)> = self
            .sideways
            .iter()
            .map(|(d, f)| (d.clone(), f as &dyn Facets))
            .collect();
        f(&MultiFacets::new(dims, Some(&self.drill_down)))
    }
}

impl Facets for DrillSidewaysFacets<'_> {
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
        self.with(|f| f.get_all_children(dim, path))
    }

    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        self.with(|f| f.get_top_children(top_n, dim, path))
    }

    fn get_specific_value(&self, dim: &str, path: &[&str]) -> Result<FacetValue> {
        self.with(|f| f.get_specific_value(dim, path))
    }

    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        self.with(|f| f.get_all_dims(top_n))
    }
}

impl<'r, 's, 'a, 't> DrillSideways<'r, 's, 'a, 't> {
    /// `new DrillSideways(searcher, config, taxoReader)`.
    pub fn new(
        searcher: &'r IndexSearcher<'s, 'a>,
        config: &'t FacetsConfig,
        taxo_reader: &'t dyn TaxonomyReader,
    ) -> Self {
        DrillSideways {
            searcher,
            config,
            taxo_reader,
            executor: false,
            score_sub_docs_at_once: false,
        }
    }

    /// `new DrillSideways(searcher, config, taxoReader, null, executor)`:
    /// the drill-down query and each dim's sideways query searched
    /// separately (here one after the other).
    pub fn with_executor(mut self) -> Self {
        self.executor = true;
        self
    }

    /// Overriding `scoreSubDocsAtOnce()` to return `true`: every sub-scorer
    /// positioned on the document being collected (the query-first
    /// strategy, whatever the costs).
    pub fn with_score_sub_docs_at_once(mut self, at_once: bool) -> Self {
        self.score_sub_docs_at_once = at_once;
        self
    }

    /// `search(query, topN)`: the top `top_n` hits by score, and the facets.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when `top_n` is not positive, and whatever
    /// the search or the counting reports.
    pub fn search(
        &self,
        query: &DrillDownQuery<'_>,
        top_n: i32,
    ) -> Result<DrillSidewaysResult<'t, 'a, TopDocs>> {
        self.search_after(None, query, top_n)
    }

    /// `search(after, query, topN)`: the top `top_n` hits below `after` (a
    /// global document and its score).
    ///
    /// # Errors
    /// As [`Self::search`].
    pub fn search_after(
        &self,
        after: Option<ScoreDoc>,
        query: &DrillDownQuery<'_>,
        top_n: i32,
    ) -> Result<DrillSidewaysResult<'t, 'a, TopDocs>> {
        let mut limit = self.searcher.max_doc();
        if limit == 0 {
            // The collector does not allow `numHits = 0`.
            limit = 1;
        }
        let manager = TopScoreDocCollectorManager::new(top_n.min(limit), after)?;
        self.search_with(query, &manager)
    }

    /// `search(query, filter, after, topN, null, false)`: [`Self::search_after`]
    /// over the query with `filter` added to its base (as a `FILTER`).
    ///
    /// # Errors
    /// As [`Self::search`].
    pub fn search_filtered(
        &self,
        query: &DrillDownQuery<'_>,
        filter: Option<Clause>,
        after: Option<ScoreDoc>,
        top_n: i32,
    ) -> Result<DrillSidewaysResult<'t, 'a, TopDocs>> {
        match filter {
            Some(filter) => {
                let filtered = DrillDownQuery::with_filter(self.config, filter, query);
                self.search_after(after, &filtered, top_n)
            }
            None => self.search_after(after, query, top_n),
        }
    }

    /// `search(query, hitCollectorManager)`: the hit manager's result, and
    /// the facets, from one search.
    ///
    /// # Errors
    /// Whatever the search, a collector or the counting reports.
    pub fn search_with<M: GroupingCollectorManager<'a>>(
        &self,
        query: &DrillDownQuery<'_>,
        hit_manager: &M,
    ) -> Result<DrillSidewaysResult<'t, 'a, M::Output>> {
        let facets_manager = FacetsCollectorManager::new(false);
        // `MultiCollectorManager(drillDownFacetsCollectorManager,
        // hitCollectorManager)`.
        let main = PairManager {
            first: &facets_manager,
            second: hit_manager,
        };
        let sideways = vec![FacetsCollectorManager::new(false); query.dims().len()];
        let result = if self.executor {
            self.search_concurrently(query, &main, &sideways)?
        } else {
            self.search_sequentially(query, &main, &sideways)?
        };
        let (facets_collector, hits) = result.drill_down_result;
        let (collectors, dims) = if query.dims().is_empty() {
            (None, None)
        } else {
            (result.drill_sideways_results, Some(query.dims().to_vec()))
        };
        let facets =
            self.build_facets_result(&facets_collector, collectors.as_deref(), dims.as_deref())?;
        Ok(DrillSidewaysResult {
            facets,
            hits,
            drill_down_facets_collector: facets_collector,
            drill_sideways_facets_collector: collectors,
            drill_sideways_dims: dims,
        })
    }

    /// `search(query, drillDownCollectorManager,
    /// drillSidewaysCollectorManagers)`: any managers, one per dim for the
    /// sideways documents.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when the query has dims but `sideways` is
    /// `None`, or `sideways` does not have one manager per dim; whatever
    /// the search or a collector reports.
    pub fn search_managers<M, S>(
        &self,
        query: &DrillDownQuery<'_>,
        drill_down: &M,
        sideways: Option<&[S]>,
    ) -> Result<DrillSidewaysSearchResult<M::Output, S::Output>>
    where
        M: GroupingCollectorManager<'a>,
        S: GroupingCollectorManager<'a>,
    {
        let sideways = match sideways {
            None if !query.dims().is_empty() => {
                return Err(Error::IllegalArgument(
                    "The query requires not null drillSidewaysCollectorManagers".into(),
                ));
            }
            None => &[][..],
            Some(s) if s.len() != query.dims().len() => {
                return Err(Error::IllegalArgument(
                    "drillSidewaysCollectorManagers size must be equal to number of dimensions in \
                     the query."
                        .into(),
                ));
            }
            Some(s) => s,
        };
        if self.executor {
            self.search_concurrently(query, drill_down, sideways)
        } else {
            self.search_sequentially(query, drill_down, sideways)
        }
    }

    /// `buildFacetsResult`: `FastTaxonomyFacetCounts` over the drill-down
    /// documents, and over each dim's sideways documents.
    fn build_facets_result(
        &self,
        drill_downs: &FacetsCollector<'_>,
        drill_sideways: Option<&[FacetsCollector<'_>]>,
        dims: Option<&[String]>,
    ) -> Result<DrillSidewaysFacets<'t>> {
        let drill_down = FastTaxonomyFacetCounts::new(self.taxo_reader, self.config, drill_downs)?;
        let mut sideways = Vec::new();
        if let (Some(collectors), Some(dims)) = (drill_sideways, dims) {
            let keys: Vec<&str> = dims.iter().map(String::as_str).collect();
            for i in java_hash_map_order(&keys) {
                let (Some(dim), Some(fc)) = (dims.get(i), collectors.get(i)) else {
                    continue;
                };
                let counts = FastTaxonomyFacetCounts::new(self.taxo_reader, self.config, fc)?;
                sideways.push((dim.clone(), counts));
            }
        }
        Ok(DrillSidewaysFacets {
            drill_down,
            sideways,
        })
    }

    /// `searchSequentially`.
    fn search_sequentially<M, S>(
        &self,
        query: &DrillDownQuery<'_>,
        drill_down: &M,
        sideways: &[S],
    ) -> Result<DrillSidewaysSearchResult<M::Output, S::Output>>
    where
        M: GroupingCollectorManager<'a>,
        S: GroupingCollectorManager<'a>,
    {
        if query.dims().is_empty() {
            // No dim, so no sideways counts.
            return Ok(DrillSidewaysSearchResult {
                drill_down_result: search_manager(self.searcher, &query.rewrite(), drill_down)?,
                drill_sideways_results: None,
            });
        }
        // A pure browse searches every document.
        let base = query
            .base_query()
            .cloned()
            .unwrap_or_else(|| Clause::MatchAllDocs(MatchAllDocsQuery::new(0)));
        let dsq = DrillSidewaysQuery::new(
            base,
            query.drill_down_queries(),
            self.score_sub_docs_at_once,
        );
        let (result, per_segment) = dsq.search(self.searcher, drill_down, sideways)?;
        let mut per_dim: Vec<Vec<S::Collector>> = sideways
            .iter()
            .map(|_| Vec::with_capacity(per_segment.len()))
            .collect();
        for segment in per_segment {
            for (dim, collector) in per_dim.iter_mut().zip(segment) {
                dim.push(collector);
            }
        }
        let results = sideways
            .iter()
            .zip(per_dim)
            .map(|(m, collectors)| m.reduce(collectors))
            .collect::<Result<Vec<_>>>()?;
        Ok(DrillSidewaysSearchResult {
            drill_down_result: result,
            drill_sideways_results: Some(results),
        })
    }

    /// `searchConcurrently`: the drill-down query, then for each dim the
    /// query without it.
    fn search_concurrently<M, S>(
        &self,
        query: &DrillDownQuery<'_>,
        drill_down: &M,
        sideways: &[S],
    ) -> Result<DrillSidewaysSearchResult<M::Output, S::Output>>
    where
        M: GroupingCollectorManager<'a>,
        S: GroupingCollectorManager<'a>,
    {
        let result = search_manager(self.searcher, &query.rewrite(), drill_down)?;
        let filters = query.drill_down_queries();
        let mut results = Vec::with_capacity(sideways.len());
        for (dim, manager) in query.dims().iter().zip(sideways) {
            let q = self.drill_down_query_without(query, &filters, dim);
            results.push(search_manager(self.searcher, &q.rewrite(), manager)?);
        }
        Ok(DrillSidewaysSearchResult {
            drill_down_result: result,
            drill_sideways_results: Some(results),
        })
    }

    /// `getDrillDownQuery(query, queries, excludedDimension)`: the query's
    /// base and every dim's drill-down but `excluded`'s.
    fn drill_down_query_without(
        &self,
        query: &DrillDownQuery<'_>,
        queries: &[Clause],
        excluded: &str,
    ) -> DrillDownQuery<'t> {
        let mut ddl = match query.base_query() {
            Some(base) => DrillDownQuery::with_base(self.config, base.clone()),
            None => DrillDownQuery::new(self.config),
        };
        for (dim, q) in query.dims().iter().zip(queries) {
            if dim != excluded {
                ddl.add_query(dim, q.clone());
            }
        }
        ddl
    }
}

impl<'a> GroupingCollectorManager<'a> for FacetsCollectorManager {
    type Collector = FacetsCollector<'a>;
    type Output = FacetsCollector<'a>;

    fn new_collector(&self) -> Result<FacetsCollector<'a>> {
        Ok(FacetsCollectorManager::new_collector(self))
    }

    fn reduce(&self, collectors: Vec<FacetsCollector<'a>>) -> Result<FacetsCollector<'a>> {
        FacetsCollectorManager::reduce(self, collectors)
    }
}

/// `MultiCollectorManager` of two managers: each collector of a slice fed
/// every document, the results reduced apart.
struct PairManager<'m, A, B> {
    first: &'m A,
    second: &'m B,
}

/// `MultiCollector` of two collectors.
struct Pair<A, B> {
    first: A,
    second: B,
}

impl<'a, A: SegmentCollector<'a>, B: SegmentCollector<'a>> SegmentCollector<'a> for Pair<A, B> {
    /// `MultiCollector.scoreMode()`: the collectors' mode when they agree,
    /// else `COMPLETE`.
    fn score_mode(&self) -> ScoreMode {
        let (a, b) = (self.first.score_mode(), self.second.score_mode());
        if a == b {
            a
        } else {
            ScoreMode::Complete
        }
    }

    fn set_next_reader(&mut self, ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.first.set_next_reader(ord, leaf)?;
        self.second.set_next_reader(ord, leaf)
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        self.first.collect(doc, score)?;
        self.second.collect(doc, score)
    }

    fn collect_range(&mut self, min: i32, max: i32, score: f32) -> Result<()> {
        self.first.collect_range(min, max, score)?;
        self.second.collect_range(min, max, score)
    }

    fn finish(&mut self) -> Result<()> {
        self.first.finish()?;
        self.second.finish()
    }
}

impl<'a, A, B> GroupingCollectorManager<'a> for PairManager<'_, A, B>
where
    A: GroupingCollectorManager<'a>,
    B: GroupingCollectorManager<'a>,
{
    type Collector = Pair<A::Collector, B::Collector>;
    type Output = (A::Output, B::Output);

    fn new_collector(&self) -> Result<Self::Collector> {
        Ok(Pair {
            first: self.first.new_collector()?,
            second: self.second.new_collector()?,
        })
    }

    fn reduce(&self, collectors: Vec<Self::Collector>) -> Result<Self::Output> {
        let (firsts, seconds): (Vec<_>, Vec<_>) =
            collectors.into_iter().map(|p| (p.first, p.second)).unzip();
        Ok((self.first.reduce(firsts)?, self.second.reduce(seconds)?))
    }
}

/// `TopScoreDocCollectorManager(numHits, after, Integer.MAX_VALUE)`: every
/// hit counted, the top `numHits` by score kept.
struct TopScoreDocCollectorManager {
    num_hits: usize,
    after: Option<ScoreDoc>,
}

impl TopScoreDocCollectorManager {
    /// # Errors
    /// [`Error::IllegalArgument`] when `num_hits` is not positive.
    fn new(num_hits: i32, after: Option<ScoreDoc>) -> Result<Self> {
        match usize::try_from(num_hits) {
            Ok(n) if n > 0 => Ok(TopScoreDocCollectorManager { num_hits: n, after }),
            _ => Err(Error::IllegalArgument(
                "numHits must be > 0; please use TotalHitCountCollectorManager if you just need \
                 the total hit count"
                    .into(),
            )),
        }
    }
}

/// `TopScoreDocCollector`: a [`TopDocsCollector`] fed global document ids.
struct TopScoreDocCollector {
    top: TopDocsCollector,
    doc_base: i32,
}

impl<'a> SegmentCollector<'a> for TopScoreDocCollector {
    fn score_mode(&self) -> ScoreMode {
        self.top.score_mode()
    }

    fn set_next_reader(&mut self, _ord: usize, leaf: &OpenSegment<'a>) -> Result<()> {
        self.doc_base = leaf.doc_base;
        Ok(())
    }

    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        ScoringCollector::collect(&mut self.top, self.doc_base.saturating_add(doc), score);
        Ok(())
    }
}

impl<'a> GroupingCollectorManager<'a> for TopScoreDocCollectorManager {
    type Collector = TopScoreDocCollector;
    type Output = TopDocs;

    fn new_collector(&self) -> Result<TopScoreDocCollector> {
        let mut top = TopDocsCollector::with_total_hits_threshold(self.num_hits, u64::MAX);
        if let Some(after) = self.after {
            top = top.with_after(after);
        }
        Ok(TopScoreDocCollector { top, doc_base: 0 })
    }

    /// `TopDocs.merge(0, numHits, topDocs)`.
    fn reduce(&self, collectors: Vec<TopScoreDocCollector>) -> Result<TopDocs> {
        let shards: Vec<TopDocs> = collectors
            .iter()
            .map(|c| TopDocs {
                total_hits: c.top.total_hits(),
                score_docs: c
                    .top
                    .top_docs()
                    .iter()
                    .map(|h| ShardScoreDoc::new(h.doc_id, h.score))
                    .collect(),
            })
            .collect();
        top_docs::merge(0, self.num_hits, &shards, &TieBreaker::Default)
    }
}

#[cfg(test)]
mod tests;
