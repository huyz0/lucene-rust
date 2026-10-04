//! `IndexSearcher` as an object: the segments of one reader, their norms, a
//! similarity and a slicing, with `search`/`count`/`explain` and the
//! collector-driven entry points (`search(Query, Collector)`,
//! `search(Query, CollectorManager)`).
//!
//! Every method is a thin front over the free functions this crate already
//! had ([`crate::multi_segment`], [`crate::explain`]); what the object adds is
//! Lucene's shape -- the state `IndexSearcher` carries between calls, which
//! the rescorers ([`crate::rescorer`]), values sources
//! ([`crate::values_source`]) and reference managers
//! ([`crate::reference_manager`]) are written against.
//!
//! A collector is driven the way `IndexSearcher.search(List<LeafReaderContext>,
//! Weight, Collector)` drives one: the leaves in doc-base order, each document
//! shifted to its global id ([`LeafCollector`]), statistics reader-wide
//! (`IndexSearcher.collectionStatistics`/`termStatistics`).
//!
//! `setTimeout(queryTimeout)` wraps every leaf's bulk scorer in a
//! `TimeLimitingBulkScorer` ([`crate::exec::score_segment_time_limited`]),
//! which asks the timeout before each window of documents; a leaf it stops
//! keeps what it collected, the search moves on to the next leaf (which asks
//! again), and [`IndexSearcher::timed_out`] reports it from then on, as
//! Java's `partialResult` does (it is never reset).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::collector::{LeafCollector, ScoreMode, ScoringCollector};
use crate::collectors::CollectorManager;
use crate::explain::{explain_clause_with_stats, Explanation};
use crate::field_norms::FieldNorms;
use crate::multi_segment::{
    rewrite_points_ranges, search_boolean_query_multi_segment_maxscore_counting, OpenSegment,
};
use crate::query::{BooleanQuery, Clause};
use crate::reader::exitable::QueryTimeout;
use crate::similarities::Similarity;
use crate::top_docs::{ShardScoreDoc, TopDocs};
use crate::{Error, Result};

/// `IndexSearcher.TOTAL_HITS_THRESHOLD`: `search(query, n)` counts hits
/// exactly up to this many.
pub const TOTAL_HITS_THRESHOLD: u64 = 1000;

/// Per-segment norms, as the multi-segment functions take them.
pub type SegmentNorms<'s, 'a> = Option<&'s HashMap<String, FieldNorms<'a>>>;

/// `IndexSearcher` over already-opened segments.
pub struct IndexSearcher<'s, 'a> {
    segments: &'s [OpenSegment<'a>],
    norms: &'s [SegmentNorms<'s, 'a>],
    similarity: Option<&'s dyn Similarity>,
    slices: Vec<Vec<usize>>,
    /// `queryTimeout`.
    timeout: Option<Arc<dyn QueryTimeout>>,
    /// `partialResult`: set by the first search a timeout stopped.
    partial_result: AtomicBool,
}

impl<'s, 'a> IndexSearcher<'s, 'a> {
    /// `new IndexSearcher(reader)`: every segment in one slice (no executor),
    /// the default BM25 similarity. `norms` has one entry per segment.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when `norms` does not have one entry per
    /// segment.
    pub fn new(segments: &'s [OpenSegment<'a>], norms: &'s [SegmentNorms<'s, 'a>]) -> Result<Self> {
        if segments.len() != norms.len() {
            return Err(Error::IllegalArgument(format!(
                "{} segments but {} norms entries",
                segments.len(),
                norms.len()
            )));
        }
        Ok(Self {
            segments,
            norms,
            similarity: None,
            slices: vec![(0..segments.len()).collect()],
            timeout: None,
            partial_result: AtomicBool::new(false),
        })
    }

    /// `setTimeout(queryTimeout)`: every search from now on asks `timeout`
    /// before each window of documents a leaf's bulk scorer scores; `None`
    /// removes it.
    pub fn set_timeout(&mut self, timeout: Option<Arc<dyn QueryTimeout>>) {
        self.timeout = timeout;
    }

    /// `getTimeout()`.
    pub fn timeout(&self) -> Option<&Arc<dyn QueryTimeout>> {
        self.timeout.as_ref()
    }

    /// `timedOut()`: whether any search so far hit the timeout (its results
    /// were partial).
    pub fn timed_out(&self) -> bool {
        self.partial_result.load(Ordering::Relaxed)
    }

    /// `setSimilarity(similarity)`.
    pub fn set_similarity(&mut self, similarity: &'s dyn Similarity) {
        self.similarity = Some(similarity);
    }

    /// `getSimilarity()`: `None` is the default BM25.
    pub fn similarity(&self) -> Option<&'s dyn Similarity> {
        self.similarity
    }

    /// The slices a [`CollectorManager`] search runs (`getSlices()`): each a
    /// list of segment indices. Java builds them from its executor; here the
    /// caller does.
    ///
    /// # Errors
    /// [`Error::SliceOutOfRange`] for a segment index the searcher lacks.
    pub fn set_slices(&mut self, slices: Vec<Vec<usize>>) -> Result<()> {
        for &s in slices.iter().flatten() {
            if s >= self.segments.len() {
                return Err(Error::SliceOutOfRange {
                    segment: s,
                    segments: self.segments.len(),
                });
            }
        }
        self.slices = slices;
        Ok(())
    }

    /// `getSlices()`.
    pub fn slices(&self) -> &[Vec<usize>] {
        &self.slices
    }

    /// `getIndexReader().leaves()`.
    pub fn segments(&self) -> &'s [OpenSegment<'a>] {
        self.segments
    }

    /// The norms of segment `i`.
    pub fn norms(&self, i: usize) -> SegmentNorms<'s, 'a> {
        self.norms.get(i).copied().flatten()
    }

    /// `getIndexReader().maxDoc()`: the last segment's doc base plus its
    /// `maxDoc`, `0` when a segment's `maxDoc` is unknown.
    pub fn max_doc(&self) -> i32 {
        self.segments
            .iter()
            .map(|s| s.doc_base.saturating_add(s.max_doc.unwrap_or(0)))
            .max()
            .unwrap_or(0)
    }

    /// `ReaderUtil.subIndex(doc, leaves)`: the segment holding global `doc`.
    pub fn segment_of(&self, doc: i32) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, s) in self.segments.iter().enumerate() {
            if s.doc_base <= doc && best.is_none_or(|b| self.segments[b].doc_base <= s.doc_base) {
                best = Some(i);
            }
        }
        best.filter(|&i| {
            let s = &self.segments[i];
            s.max_doc.is_none_or(|m| doc - s.doc_base < m)
        })
    }

    /// `search(query, n)`: the top `n` by score, counted exactly up to
    /// [`TOTAL_HITS_THRESHOLD`] (`TopScoreDocCollectorManager(n, 1000)`).
    /// Under a similarity other than the default, or a timeout, this is one
    /// `TopScoreDocCollector` over the leaves in order (Java's single slice);
    /// a timeout leaves the relation what the collector counted, as Java
    /// does -- [`Self::timed_out`] is the flag.
    pub fn search(&self, query: &BooleanQuery, n: usize) -> Result<TopDocs> {
        let sim = self.similarity.filter(|s| !s.is_default_bm25());
        if sim.is_some() || self.timeout.is_some() {
            let mut c = crate::collector::TopDocsCollector::with_total_hits_threshold(
                n,
                TOTAL_HITS_THRESHOLD,
            );
            self.search_collector(query, &mut c)?;
            return Ok(TopDocs {
                total_hits: c.total_hits(),
                score_docs: c
                    .top_docs()
                    .iter()
                    .map(|h| ShardScoreDoc::new(h.doc_id, h.score))
                    .collect(),
            });
        }
        let rescored = self.rewrite_rescore(query)?;
        let query = rescored.as_ref().unwrap_or(query);
        let (hits, total_hits) = search_boolean_query_multi_segment_maxscore_counting(
            self.segments,
            query,
            self.norms,
            n,
            TOTAL_HITS_THRESHOLD,
        )?;
        Ok(TopDocs {
            total_hits,
            score_docs: hits
                .into_iter()
                .map(|h| ShardScoreDoc::new(h.doc_id, h.score))
                .collect(),
        })
    }

    /// `searchAfter(after, query, n)`: the top `n` below `after` (a global
    /// document id and its score; a hit ties it only with a larger id), counted
    /// as [`Self::search`] counts (`TopScoreDocCollectorManager(n, after,
    /// 1000)`).
    pub fn search_after(
        &self,
        after: crate::collector::ScoreDoc,
        query: &BooleanQuery,
        n: usize,
    ) -> Result<TopDocs> {
        let mut c =
            crate::collector::TopDocsCollector::with_total_hits_threshold(n, TOTAL_HITS_THRESHOLD)
                .with_after(after);
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

    /// `search(query, n, sort)` / `searchAfter(after, query, n, sort)`: the
    /// top `n` by `sort`, counted exactly up to [`TOTAL_HITS_THRESHOLD`]
    /// (`TopFieldCollectorManager(sort, n, after, 1000)`). `readers[i]` is
    /// the segment reader segment `i` was opened from (its doc values and
    /// field infos), which this object does not hold; see
    /// [`crate::top_field::search_sorted`]. A score key, and the function
    /// queries' preparation, use the searcher's similarity.
    pub fn search_sorted(
        &self,
        readers: &[crate::directory_reader::SegmentReader],
        query: &BooleanQuery,
        n: usize,
        sort: &[crate::top_field::SortField],
        after: Option<&crate::top_field::FieldDoc>,
    ) -> Result<crate::top_field::TopFieldDocs> {
        let norms: Vec<SegmentNorms<'s, 'a>> =
            (0..self.segments.len()).map(|i| self.norms(i)).collect();
        crate::top_field::search_sorted_leaves(
            self.segments,
            readers,
            query,
            &norms,
            sort,
            n,
            TOTAL_HITS_THRESHOLD,
            after,
            false,
            None,
            None,
            None,
            self.similarity.filter(|s| !s.is_default_bm25()),
        )
    }

    /// `search(query, collector)`: every segment, in doc-base order, into
    /// one collector, which sees global document ids.
    pub fn search_collector<C: ScoringCollector + ?Sized>(
        &self,
        query: &BooleanQuery,
        collector: &mut C,
    ) -> Result<()> {
        let order: Vec<usize> = (0..self.segments.len()).collect();
        self.search_leaves(query, &order, collector)
    }

    /// `search(leaves, weight, collector)` over the segments `slice` names
    /// (a `CollectorManager` slice): statistics reader-wide, the segments in
    /// doc-base order, documents shifted to global ids.
    pub fn search_slice_collector<C: ScoringCollector + ?Sized>(
        &self,
        query: &BooleanQuery,
        slice: &[usize],
        collector: &mut C,
    ) -> Result<()> {
        for &s in slice {
            if s >= self.segments.len() {
                return Err(Error::SliceOutOfRange {
                    segment: s,
                    segments: self.segments.len(),
                });
            }
        }
        self.search_leaves(query, slice, collector)
    }

    /// The query rewritten against the reader and its reader-wide statistics.
    /// `RescoreTopNQuery.rewrite(searcher)` for a query holding one (see
    /// [`crate::rescorer::rewrite_rescore_clauses`]; a vector-backed source
    /// needs the vectors, so such a query is rewritten by the caller with
    /// them first).
    fn rewrite_rescore(&self, query: &BooleanQuery) -> Result<Option<BooleanQuery>> {
        if !crate::rescorer::has_rescore_clauses(query) {
            return Ok(None);
        }
        let ctx = crate::values_source::ValuesContext::new(self);
        crate::rescorer::rewrite_rescore_clauses(query, &ctx)
    }

    fn prepare(&self, query: &BooleanQuery) -> Result<(Option<BooleanQuery>, crate::GlobalStats)> {
        let rescored = self.rewrite_rescore(query)?;
        let query = rescored.as_ref().unwrap_or(query);
        let rewritten = rewrite_points_ranges(query, self.segments).or(rescored.clone());
        let global = crate::multi_segment::global_boolean_stats_with_similarity(
            self.segments,
            rewritten.as_ref().unwrap_or(query),
            self.similarity.filter(|s| !s.is_default_bm25()),
        )?;
        Ok((rewritten, global))
    }

    /// `search(leaves, weight, collector)` over the segments `order` names.
    fn search_leaves<C: ScoringCollector + ?Sized>(
        &self,
        query: &BooleanQuery,
        order: &[usize],
        collector: &mut C,
    ) -> Result<()> {
        let (rewritten, global) = self.prepare(query)?;
        self.search_prepared(
            rewritten.as_ref().unwrap_or(query),
            &global,
            order,
            collector,
        )
    }

    fn search_prepared<C: ScoringCollector + ?Sized>(
        &self,
        query: &BooleanQuery,
        global: &crate::GlobalStats,
        order: &[usize],
        collector: &mut C,
    ) -> Result<()> {
        let sim = self.similarity.filter(|s| !s.is_default_bm25());
        let mut order = order.to_vec();
        order.sort_by_key(|&i| self.segments[i].doc_base);
        for i in order {
            let seg = &self.segments[i];
            let norms = self.norms(i);
            let mut leaf = LeafCollector::new(&mut *collector, seg.doc_base);
            if let Some(timeout) = &self.timeout {
                // `searchLeaf`: `new TimeLimitingBulkScorer(scorer,
                // queryTimeout)`; a `TimeExceededException` marks the
                // result partial and the next leaf is searched.
                if crate::search_boolean_query_scored_segment_time_limited(
                    seg,
                    query,
                    norms,
                    global,
                    sim,
                    &mut leaf,
                    || timeout.should_exit(),
                )? {
                    self.partial_result.store(true, Ordering::Relaxed);
                }
                continue;
            }
            match sim {
                Some(sim) => crate::search_boolean_query_scored_segment_with_similarity(
                    seg, query, norms, global, sim, &mut leaf,
                )?,
                None => crate::search_boolean_query_scored_segment(
                    seg,
                    query,
                    norms,
                    Some(global),
                    &mut leaf,
                )?,
            }
        }
        Ok(())
    }

    /// `search(query, collectorManager)`: a collector per slice, the slices
    /// concurrently when there are several, then `reduce` over the
    /// collectors in slice order.
    pub fn search_manager<M: CollectorManager>(
        &self,
        query: &BooleanQuery,
        manager: &M,
    ) -> Result<M::Output> {
        let (rewritten, global) = self.prepare(query)?;
        let query = rewritten.as_ref().unwrap_or(query);
        let parallel = self.slices.len() > 1;
        let run = |slice: &[usize]| -> Result<M::Collector> {
            let mut c = manager.new_collector()?;
            self.search_prepared(query, &global, slice, &mut c)?;
            Ok(c)
        };
        let collectors = if self.slices.is_empty() {
            vec![manager.new_collector()?]
        } else {
            crate::slices::run_slices_if(parallel, &self.slices, run)
                .into_iter()
                .collect::<Result<Vec<_>>>()?
        };
        manager.reduce(collectors)
    }

    /// `count(query)`: the number of live matches.
    pub fn count(&self, query: &BooleanQuery) -> Result<u64> {
        struct Count(u64);
        impl ScoringCollector for Count {
            fn collect(&mut self, _doc: i32, _score: f32) {
                self.0 += 1;
            }
            fn score_mode(&self) -> ScoreMode {
                ScoreMode::CompleteNoScores
            }
            fn add_hits(&mut self, n: u64) -> bool {
                self.0 += n;
                true
            }
        }
        let mut c = Count(0);
        self.search_collector(query, &mut c)?;
        Ok(c.0)
    }

    /// `explain(query, doc)` for a global document id: the query rewritten
    /// and its reader-wide statistics gathered as for a search
    /// (`createWeight` over the whole reader), then the document explained in
    /// its segment (`ReaderUtil.subIndex`, `weight.explain(leaf, doc -
    /// docBase)`) by [`explain_clause_with_stats`] -- so a term's `n` and `N`
    /// are the reader's, and the value is the score a search gives the
    /// document.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a document outside every segment, and
    /// whatever [`explain_clause_with_stats`] reports.
    pub fn explain(&self, query: &BooleanQuery, doc: i32) -> Result<Explanation> {
        let i = self
            .segment_of(doc)
            .ok_or_else(|| Error::IllegalArgument(format!("doc {doc} is in no segment")))?;
        let (rewritten, global) = self.prepare(query)?;
        let query = rewritten.as_ref().unwrap_or(query);
        let seg = &self.segments[i];
        // `IndexSearcher.explain` explains `rewrite(query)`: a one-clause
        // boolean is its clause.
        let mut clause = Clause::Boolean(Box::new(query.clone())).rewrite();
        if let Some(max_doc) = seg.max_doc {
            set_match_all_max_doc(&mut clause, max_doc);
        }
        crate::explain::with_leaf(
            seg.max_doc,
            seg.doc_base,
            seg.reader,
            self.similarity.filter(|s| !s.is_default_bm25()),
            || {
                explain_clause_with_stats(
                    seg.fields,
                    seg.doc_in,
                    seg.pos_in,
                    seg.pay_in,
                    seg.live_docs,
                    seg.points,
                    &clause,
                    doc - seg.doc_base,
                    self.norms(i),
                    Some(&global),
                )
            },
        )
    }

    /// `weight.scorer(leaf)` run to the end: every document of segment
    /// `leaf` that `query` matches, leaf-local and ascending, with its score
    /// (`ScoreMode.COMPLETE`, reader-wide statistics). With
    /// `include_deleted`, deleted documents are scored too, as a `Scorer`
    /// (which never reads live docs) reports them.
    pub fn leaf_scores(
        &self,
        query: &BooleanQuery,
        leaf: usize,
        include_deleted: bool,
    ) -> Result<Vec<(i32, f32)>> {
        struct All(Vec<(i32, f32)>);
        impl ScoringCollector for All {
            fn collect(&mut self, doc: i32, score: f32) {
                self.0.push((doc, score));
            }
        }
        let seg = self.segments.get(leaf).ok_or(Error::SliceOutOfRange {
            segment: leaf,
            segments: self.segments.len(),
        })?;
        let (rewritten, global) = self.prepare(query)?;
        let query = rewritten.as_ref().unwrap_or(query);
        let sim = self.similarity.filter(|s| !s.is_default_bm25());
        let one = OpenSegment {
            live_docs: if include_deleted { None } else { seg.live_docs },
            ..*seg
        };
        let mut all = All(Vec::new());
        match sim {
            Some(sim) => crate::search_boolean_query_scored_segment_with_similarity(
                &one,
                query,
                self.norms(leaf),
                &global,
                sim,
                &mut all,
            )?,
            None => crate::search_boolean_query_scored_segment(
                &one,
                query,
                self.norms(leaf),
                Some(&global),
                &mut all,
            )?,
        }
        all.0.sort_by_key(|&(d, _)| d);
        Ok(all.0)
    }

    /// The score `query` gives each of `docs` (global ids, any order) that it
    /// matches: what a `Weight`'s scorer, advanced to each, would report --
    /// the per-document scores `QueryRescorer` and a query-backed values
    /// source read. A document the query does not match is absent.
    pub fn scores_of(&self, query: &BooleanQuery, docs: &[i32]) -> Result<HashMap<i32, f32>> {
        struct Pick<'w> {
            wanted: &'w std::collections::HashSet<i32>,
            out: HashMap<i32, f32>,
        }
        impl ScoringCollector for Pick<'_> {
            fn collect(&mut self, doc: i32, score: f32) {
                if self.wanted.contains(&doc) {
                    self.out.insert(doc, score);
                }
            }
        }
        let wanted: std::collections::HashSet<i32> = docs.iter().copied().collect();
        let mut segs: Vec<usize> = docs.iter().filter_map(|&d| self.segment_of(d)).collect();
        segs.sort_unstable();
        segs.dedup();
        let mut pick = Pick {
            wanted: &wanted,
            out: HashMap::new(),
        };
        if !segs.is_empty() {
            self.search_leaves(query, &segs, &mut pick)?;
        }
        Ok(pick.out)
    }
}

/// A `MatchAllDocsQuery` built without a `maxDoc` (a query parsed once for
/// the whole reader) matches every document of the leaf it is explained in,
/// as its scorer does (`DocIdSetIterator.all(context.reader().maxDoc())`).
fn set_match_all_max_doc(clause: &mut Clause, max_doc: i32) {
    match clause {
        Clause::MatchAllDocs(m) => m.max_doc = max_doc,
        Clause::Boolean(b) => {
            for c in b
                .must
                .iter_mut()
                .chain(b.should.iter_mut())
                .chain(b.filter.iter_mut())
                .chain(b.must_not.iter_mut())
            {
                set_match_all_max_doc(c, max_doc);
            }
        }
        Clause::DisjunctionMax(d) => {
            for c in &mut d.disjuncts {
                set_match_all_max_doc(c, max_doc);
            }
        }
        Clause::ConstantScore(c) => set_match_all_max_doc(&mut c.inner, max_doc),
        Clause::Boost(b) => set_match_all_max_doc(&mut b.inner, max_doc),
        Clause::Extended(e) => {
            if let crate::extended_query::ExtendedQuery::FunctionScore(f) = e.as_mut() {
                set_match_all_max_doc(&mut f.in_query, max_doc);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::set_match_all_max_doc;
    use crate::query::{
        BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, MatchAllDocsQuery, TermQuery,
    };
    use crate::{BooleanQuery, Clause};

    #[test]
    fn a_match_all_takes_the_leafs_max_doc_wherever_it_is_nested() {
        let all = || Clause::MatchAllDocs(MatchAllDocsQuery::new(0));
        let mut b = BooleanQuery::new();
        b.must.push(all());
        b.should
            .push(Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
                vec![all(), Clause::Term(TermQuery::new("f", "t"))],
                0.0,
            ))));
        b.filter
            .push(Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                all(),
                1.0,
            ))));
        b.must_not
            .push(Clause::Boost(Box::new(BoostQuery::new(all(), 2.0))));
        let mut clause = Clause::Boolean(Box::new(b));
        set_match_all_max_doc(&mut clause, 7);
        let text = format!("{clause:?}");
        assert_eq!(text.matches("max_doc: 7").count(), 4, "{text}");
        assert!(!text.contains("max_doc: 0"), "{text}");
    }
}
