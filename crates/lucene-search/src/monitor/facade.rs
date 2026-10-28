//! `org.apache.lucene.monitor.Monitor` and its matchers: `QueryMatch`,
//! `ScoringMatch`, `ExplainingMatch`, `HighlightsMatch`, `MatchingQueries`
//! and `MultiMatchingQueries`. Queries are registered once; a batch of
//! documents is matched against them, the presearcher narrowing the
//! candidates and each candidate's query run against the batch.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lucene_analysis::Analyzer;

use crate::explain::Explanation;
use crate::index_searcher::IndexSearcher;
use crate::matches;
use crate::monitor::batch::{DocFields, DocumentBatch};
use crate::monitor::config::{MonitorConfiguration, MonitorUpdateListener, QueryTimeListener};
use crate::monitor::index::{
    batch_terms, AcceptAll, BatchTerms, CachedEntry, MonitorQuery, QueryCacheStats, QueryIndex,
};
use crate::monitor::presearcher::TermFilteredPresearcher;
use crate::query::Clause;
use crate::similarities::Similarity;
use crate::{Error, Result};

/// `HighlightsMatch.Hit`: a match's positions and offsets. Ordered, and so
/// equal, by start then end *position* only, as Java's `compareTo` has it.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub start_position: i32,
    pub start_offset: i32,
    pub end_position: i32,
    pub end_offset: i32,
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.start_position
            .cmp(&other.start_position)
            .then(self.end_position.cmp(&other.end_position))
    }
}

impl std::fmt::Display for Hit {
    /// `%d(%d)->%d(%d)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}({})->{}({})",
            self.start_position, self.start_offset, self.end_position, self.end_offset
        )
    }
}

/// `HighlightsMatch.getHits()`: each field's hits.
pub type Hits = BTreeMap<String, BTreeSet<Hit>>;

/// A matching query: `QueryMatch`, plus what the matcher that found it adds
/// -- a `ScoringMatch`'s score, an `ExplainingMatch`'s explanation, a
/// `HighlightsMatch`'s hits.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryMatch {
    pub query_id: String,
    pub score: Option<f32>,
    pub explanation: Option<Explanation>,
    pub hits: Option<Hits>,
}

impl QueryMatch {
    fn plain(query_id: &str) -> Self {
        QueryMatch {
            query_id: query_id.to_string(),
            score: None,
            explanation: None,
            hits: None,
        }
    }

    /// `CandidateMatcher.resolve(match1, match2)`: two matches of one query on
    /// one document -- two pieces of a decomposed query -- as one. Scores add,
    /// explanations become `sum of:`, hits merge.
    fn resolve(self, other: QueryMatch) -> QueryMatch {
        let score = match (self.score, other.score) {
            (Some(a), Some(b)) => Some(a + b),
            (a, b) => a.or(b),
        };
        let explanation = match (self.explanation, other.explanation) {
            (Some(a), Some(b)) => Some(
                Explanation::match_double(a.double_value() + b.double_value(), "sum of:")
                    .with_details(vec![a, b]),
            ),
            (a, b) => a.or(b),
        };
        let hits = match (self.hits, other.hits) {
            (Some(mut a), Some(b)) => {
                for (field, set) in b {
                    a.entry(field).or_default().extend(set);
                }
                Some(a)
            }
            (a, b) => a.or(b),
        };
        QueryMatch {
            query_id: self.query_id,
            score,
            explanation,
            hits,
        }
    }
}

/// The matcher a batch is matched with: `MatcherFactory`.
#[derive(Clone, Default)]
pub enum Matcher {
    /// `QueryMatch.SIMPLE_MATCHER`.
    #[default]
    Simple,
    /// `ScoringMatch.DEFAULT_MATCHER`, or `matchWithSimilarity(similarity)`.
    Scoring(Option<Arc<dyn Similarity>>),
    /// `ExplainingMatch.MATCHER`.
    Explaining,
    /// `HighlightsMatch.MATCHER`.
    Highlights,
}

/// `MultiMatchingQueries`: each document's matches, and the work done.
///
/// Equality leaves out the two timings, which no two runs share.
#[derive(Debug, Clone)]
pub struct MultiMatchingQueries {
    /// Per document, the matches by query id.
    pub matches: Vec<BTreeMap<String, QueryMatch>>,
    /// Queries that failed to run, by id, with the message.
    pub errors: BTreeMap<String, String>,
    /// The queries the presearcher selected.
    pub queries_run: usize,
    pub batch_size: usize,
    /// `getQueryBuildTime()`: nanoseconds spent building the presearcher's
    /// query from the batch's terms (0 for a partition's own run, as
    /// `PartitionMatcher`'s workers `finish(0, 0)`).
    pub query_build_time: u64,
    /// `getSearchTime()`: milliseconds from the matcher's creation to the
    /// end of the run.
    pub search_time: u64,
}

impl PartialEq for MultiMatchingQueries {
    fn eq(&self, other: &Self) -> bool {
        self.matches == other.matches
            && self.errors == other.errors
            && self.queries_run == other.queries_run
            && self.batch_size == other.batch_size
    }
}

impl MultiMatchingQueries {
    /// `matches(queryId, docId)`.
    pub fn matches(&self, query_id: &str, doc: usize) -> Option<&QueryMatch> {
        self.matches.get(doc)?.get(query_id)
    }

    /// `getMatchCount(docId)`.
    pub fn match_count(&self, doc: usize) -> usize {
        self.matches.get(doc).map_or(0, BTreeMap::len)
    }

    /// `singleton()`: a one-document batch's matches.
    fn singleton(mut self) -> MatchingQueries {
        let matches = self.matches.swap_remove(0).into_values().collect();
        MatchingQueries {
            matches,
            queries_run: self.queries_run,
            errors: self.errors,
            query_build_time: self.query_build_time,
            search_time: self.search_time,
        }
    }
}

/// `PresearcherMatches`: a match, with why the presearcher selected each query.
#[derive(Debug, Clone)]
pub struct PresearcherMatches {
    matching_terms: BTreeMap<String, String>,
    /// The batch's matches.
    pub matcher: MultiMatchingQueries,
}

/// `PresearcherMatch`.
#[derive(Debug, Clone, PartialEq)]
pub struct PresearcherMatch {
    /// The presearcher's matches on the query's indexed document(s): a space
    /// and the matching query for each matched token.
    pub presearcher_matches: String,
    /// The matcher's match, if the query matched the document.
    pub query_match: Option<QueryMatch>,
    pub query_id: String,
}

impl PresearcherMatches {
    /// `match(queryId, doc)`: `None` when the presearcher did not select the
    /// query.
    pub fn match_query(&self, query_id: &str, doc: usize) -> Option<PresearcherMatch> {
        let found = self.matching_terms.get(query_id)?;
        Some(PresearcherMatch {
            presearcher_matches: found.clone(),
            query_match: self.matcher.matches(query_id, doc).cloned(),
            query_id: query_id.to_string(),
        })
    }
}

/// `MatchingQueries`: one document's matches and the work done to find them.
///
/// Equality leaves out the two timings, which no two runs share.
#[derive(Debug, Clone)]
pub struct MatchingQueries {
    /// Sorted by query id.
    pub matches: Vec<QueryMatch>,
    /// The queries the presearcher selected.
    pub queries_run: usize,
    /// Queries that failed to run, by id, with the message.
    pub errors: BTreeMap<String, String>,
    /// `getQueryBuildTime()`: nanoseconds (see [`MultiMatchingQueries`]).
    pub query_build_time: u64,
    /// `getSearchTime()`: milliseconds (see [`MultiMatchingQueries`]).
    pub search_time: u64,
}

impl PartialEq for MatchingQueries {
    fn eq(&self, other: &Self) -> bool {
        self.matches == other.matches
            && self.queries_run == other.queries_run
            && self.errors == other.errors
    }
}

/// `System.nanoTime() - since` as a Java `long` would hold it.
fn nanos_since(since: std::time::Instant) -> u64 {
    u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// `CandidateMatcher.finish(buildTime, queryCount)`: the build time, and the
/// search time in milliseconds since the matcher was made.
fn finish(
    mut run: MultiMatchingQueries,
    build_nanos: u64,
    matcher_made: std::time::Instant,
) -> MultiMatchingQueries {
    run.query_build_time = build_nanos;
    run.search_time = nanos_since(matcher_made) / 1_000_000;
    run
}

impl MatchingQueries {
    /// `matches(queryId)`.
    pub fn matches(&self, query_id: &str) -> Option<&QueryMatch> {
        self.matches.iter().find(|m| m.query_id == query_id)
    }

    /// `getMatchCount()`.
    pub fn match_count(&self) -> usize {
        self.matches.len()
    }
}

/// `Monitor`.
pub struct Monitor {
    analyzer: Arc<Analyzer>,
    index: QueryIndex,
    filter_fields: Vec<String>,
}

impl Monitor {
    /// `new Monitor(analyzer)`: the default term-filtered presearcher.
    pub fn new(analyzer: Arc<Analyzer>) -> Self {
        Self::with_presearcher(analyzer, TermFilteredPresearcher::default(), Vec::new())
    }

    /// `new Monitor(analyzer, presearcher)`. `filter_fields` are the fields
    /// the presearcher was built to filter by.
    pub fn with_presearcher(
        analyzer: Arc<Analyzer>,
        presearcher: TermFilteredPresearcher,
        filter_fields: Vec<String>,
    ) -> Self {
        Monitor {
            analyzer,
            index: QueryIndex::new(presearcher),
            filter_fields,
        }
    }

    /// `register(queries)`: each replaces any query of the same id.
    pub fn register(&mut self, queries: Vec<MonitorQuery>) -> Result<()> {
        self.index.register(queries).map_err(Error::IllegalArgument)
    }

    /// `new Monitor(analyzer, presearcher, configuration)`: with an index path
    /// the monitor persists its queries there and reloads those already
    /// there.
    pub fn with_configuration(
        analyzer: Arc<Analyzer>,
        presearcher: TermFilteredPresearcher,
        filter_fields: Vec<String>,
        config: &MonitorConfiguration,
    ) -> Result<Self> {
        let index =
            QueryIndex::with_configuration(presearcher, config).map_err(Error::IllegalState)?;
        Ok(Monitor {
            analyzer,
            index,
            filter_fields,
        })
    }

    /// `purgeCache()`.
    pub fn purge_cache(&mut self) -> Result<()> {
        self.index.purge_cache().map_err(Error::IllegalState)
    }

    /// `getQueryCacheStats()`.
    pub fn query_cache_stats(&self) -> QueryCacheStats {
        self.index.cache_stats()
    }

    /// `addQueryIndexUpdateListener(listener)`.
    pub fn add_query_index_update_listener(&mut self, listener: Arc<dyn MonitorUpdateListener>) {
        self.index.add_listener(listener);
    }

    /// `deleteById(ids)`.
    pub fn delete_by_id(&mut self, ids: &[&str]) -> Result<()> {
        let ids: Vec<String> = ids.iter().map(|s| (*s).to_string()).collect();
        self.index.delete(&ids).map_err(Error::IllegalState)
    }

    /// `clear()`.
    pub fn clear(&mut self) -> Result<()> {
        self.index.clear().map_err(Error::IllegalState)
    }

    /// `getQuery(id)`.
    pub fn get_query(&self, id: &str) -> Option<&MonitorQuery> {
        self.index.get_query(id)
    }

    /// `getQueryIds()`, sorted.
    pub fn query_ids(&self) -> Vec<String> {
        self.index.query_ids()
    }

    /// `getQueryCount()`.
    pub fn query_count(&self) -> usize {
        self.index.query_ids().len()
    }

    /// `getDisjunctCount()`.
    pub fn disjunct_count(&self) -> usize {
        self.index.disjunct_count()
    }

    /// `match(doc, SIMPLE_MATCHER)`.
    pub fn match_doc(&self, doc: DocFields<'_>) -> Result<MatchingQueries> {
        Ok(self.match_docs(&[doc], &Matcher::Simple)?.singleton())
    }

    /// `match(doc, ScoringMatch.DEFAULT_MATCHER)`.
    pub fn match_doc_scored(&self, doc: DocFields<'_>) -> Result<MatchingQueries> {
        Ok(self
            .match_docs(&[doc], &Matcher::Scoring(None))?
            .singleton())
    }

    /// `match(doc, ScoringMatch.matchWithSimilarity(similarity))`.
    pub fn match_doc_with_similarity(
        &self,
        doc: DocFields<'_>,
        similarity: Arc<dyn Similarity>,
    ) -> Result<MatchingQueries> {
        Ok(self
            .match_docs(&[doc], &Matcher::Scoring(Some(similarity)))?
            .singleton())
    }

    /// `match(docs, QueryTimeListener.timingMatcher(matcher, listener))`: the
    /// same match, with each selected query's time told to `listener`.
    pub fn match_docs_timed(
        &self,
        docs: &[DocFields<'_>],
        matcher: &Matcher,
        listener: &mut dyn QueryTimeListener,
    ) -> Result<MultiMatchingQueries> {
        let prepared = self.prepare(docs, matcher)?;
        let candidates = self
            .index
            .select(&prepared.terms, &AcceptAll, prepared.filter.clone());
        let run = self.run_candidates(&prepared.batch, matcher, &candidates, Some(listener))?;
        Ok(finish(run, prepared.build_nanos, prepared.made))
    }

    /// The batch, its presearcher terms and its filter-field clause.
    fn prepare(&self, docs: &[DocFields<'_>], matcher: &Matcher) -> Result<Prepared> {
        let similarity = match matcher {
            Matcher::Scoring(s) => s.clone(),
            _ => None,
        };
        let batch = DocumentBatch::of(&self.analyzer, docs, similarity)?;
        let made = std::time::Instant::now();
        let built = std::time::Instant::now();
        let terms = batch_terms(batch.terms(), &self.filter_fields);
        let filter = self
            .index
            .batch_filter(
                &batch.doc_freqs(self.index.filter_fields()),
                batch.max_doc(),
            )
            .map_err(Error::IllegalArgument)?;
        let build_nanos = nanos_since(built);
        Ok(Prepared {
            batch,
            terms,
            filter,
            made,
            build_nanos,
        })
    }

    /// `debug(docs, matcher)`: the match, and for every query the presearcher
    /// selected, which of its indexed tokens did.
    pub fn debug_docs(
        &self,
        docs: &[DocFields<'_>],
        matcher: &Matcher,
    ) -> Result<PresearcherMatches> {
        let prepared = self.prepare(docs, matcher)?;
        let selected =
            self.index
                .select_debug(&prepared.terms, &AcceptAll, prepared.filter.clone());
        let mut matching_terms: BTreeMap<String, String> = BTreeMap::new();
        for (entry, why) in &selected {
            matching_terms
                .entry(entry.query_id.clone())
                .or_default()
                .push_str(why);
        }
        let candidates: Vec<&CachedEntry> = selected.iter().map(|(e, _)| *e).collect();
        let matcher = finish(
            self.run_candidates(&prepared.batch, matcher, &candidates, None)?,
            prepared.build_nanos,
            prepared.made,
        );
        Ok(PresearcherMatches {
            matching_terms,
            matcher,
        })
    }

    /// `match(docs, matcher)`: the presearcher selects the candidate entries
    /// for the batch, each candidate's query runs against it, and the matcher
    /// says what a match holds.
    pub fn match_docs(
        &self,
        docs: &[DocFields<'_>],
        matcher: &Matcher,
    ) -> Result<MultiMatchingQueries> {
        let prepared = self.prepare(docs, matcher)?;
        let candidates = self
            .index
            .select(&prepared.terms, &AcceptAll, prepared.filter.clone());
        let run = self.run_candidates(&prepared.batch, matcher, &candidates, None)?;
        Ok(finish(run, prepared.build_nanos, prepared.made))
    }

    /// Runs each candidate's query against the batch, under `matcher`.
    fn run_candidates(
        &self,
        batch: &DocumentBatch,
        matcher: &Matcher,
        candidates: &[&CachedEntry],
        timer: Option<&mut dyn QueryTimeListener>,
    ) -> Result<MultiMatchingQueries> {
        let max_doc = batch.max_doc();
        batch.with_searcher(|searcher| run_on(searcher, max_doc, matcher, candidates, timer))
    }
}

/// Runs each candidate's query with `searcher` over a batch of `max_doc`
/// documents, under `matcher`.
fn run_on(
    searcher: &IndexSearcher<'_, '_>,
    max_doc: usize,
    matcher: &Matcher,
    candidates: &[&CachedEntry],
    mut timer: Option<&mut dyn QueryTimeListener>,
) -> MultiMatchingQueries {
    let mut per_doc: Vec<BTreeMap<String, QueryMatch>> = vec![BTreeMap::new(); max_doc];
    let mut errors = BTreeMap::new();
    for entry in candidates {
        let Some(query) = entry.search_query() else {
            continue;
        };
        // a query's pieces on one document resolve into one match
        let mut add = |doc: usize, m: QueryMatch| {
            let slot = &mut per_doc[doc];
            match slot.remove(&m.query_id) {
                Some(old) => {
                    let id = m.query_id.clone();
                    slot.insert(id, old.resolve(m));
                }
                None => {
                    slot.insert(m.query_id.clone(), m);
                }
            }
        };
        let started = std::time::Instant::now();
        let outcome: Result<()> = (|| {
            match matcher {
                Matcher::Simple | Matcher::Scoring(_) => {
                    let top = searcher.search(&query.boolean, max_doc)?;
                    for sd in &top.score_docs {
                        let mut m = QueryMatch::plain(&entry.query_id);
                        if matches!(matcher, Matcher::Scoring(_)) {
                            m.score = Some(sd.score);
                        }
                        add(usize::try_from(sd.doc).unwrap_or(0), m);
                    }
                }
                Matcher::Explaining => {
                    for doc in 0..max_doc {
                        let ex =
                            searcher.explain(&query.boolean, i32::try_from(doc).unwrap_or(0))?;
                        if ex.matched {
                            let mut m = QueryMatch::plain(&entry.query_id);
                            m.explanation = Some(ex);
                            add(doc, m);
                        }
                    }
                }
                Matcher::Highlights => {
                    for doc in 0..max_doc {
                        let d = i32::try_from(doc).unwrap_or(0);
                        if let Some(found) = matches::matches(searcher, &query.clause, d)? {
                            let mut m = QueryMatch::plain(&entry.query_id);
                            m.hits = Some(build_hits(found.as_ref())?);
                            add(doc, m);
                        }
                    }
                }
            }
            Ok(())
        })();
        // `QueryTimeListener.timingMatcher`: each query's time, failed or not
        if let Some(t) = timer.as_deref_mut() {
            let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            t.log_query_time(&entry.query_id, nanos);
        }
        if let Err(e) = outcome {
            errors.insert(entry.query_id.clone(), e.to_string());
        }
    }
    MultiMatchingQueries {
        matches: per_doc,
        errors,
        queries_run: candidates.len(),
        batch_size: max_doc,
        query_build_time: 0,
        search_time: 0,
    }
}

/// `PartitionMatcher.partition`: `items` cut into `slices` runs of as equal a
/// size as a whole number allows, the last run taking the remainder.
pub fn partition<T>(items: &[T], slices: usize) -> Vec<&[T]> {
    let size = items.len() as f64 / slices as f64;
    let mut accum = 0f64;
    let mut start = 0usize;
    let mut out = Vec::with_capacity(slices);
    for i in 0..slices {
        let mut end = (accum + size).floor() as usize;
        if i + 1 == slices {
            end = items.len();
        }
        out.push(&items[start..end]);
        accum += size;
        start = accum.floor() as usize;
    }
    out
}

impl Monitor {
    /// `match(docs, PartitionMatcher.factory(executor, matcher, threads))`:
    /// the batch is built and the presearcher selects its candidates once;
    /// the candidates are cut into `threads` runs (`PartitionMatcher`'s
    /// tasks), run on the shared pool against the shared batch, and the
    /// per-document matches merge in run order. Equal to
    /// [`Monitor::match_docs`] for any thread count: a query's pieces resolve
    /// in candidate order whichever run they land in.
    ///
    /// A run is never shorter than [`MIN_CANDIDATES_PER_THREAD`] candidates,
    /// and there are no more runs than the machine has hardware threads
    /// (fewer threads are used otherwise), so a small candidate set is not
    /// slowed by handing work to another thread.
    pub fn match_docs_partitioned(
        &self,
        docs: &[DocFields<'_>],
        matcher: &Matcher,
        threads: usize,
    ) -> Result<MultiMatchingQueries> {
        let prepared = self.prepare(docs, matcher)?;
        let candidates = self
            .index
            .select(&prepared.terms, &AcceptAll, prepared.filter.clone());
        let hardware = std::thread::available_parallelism().map_or(1, usize::from);
        let threads = threads
            .min(hardware)
            .min(candidates.len().div_ceil(MIN_CANDIDATES_PER_THREAD))
            .max(1);
        if threads == 1 {
            let run = self.run_candidates(&prepared.batch, matcher, &candidates, None)?;
            return Ok(finish(run, prepared.build_nanos, prepared.made));
        }
        let max_doc = prepared.batch.max_doc();
        let parts = partition(&candidates, threads);
        // one searcher over the batch, shared by every run; the first run is
        // this thread's own and the rest go to the crate's shared pool (no
        // thread start-up per match), results in run order
        let runs: Vec<Vec<usize>> = (0..parts.len()).map(|i| vec![i]).collect();
        let results: Vec<MultiMatchingQueries> = prepared.batch.with_searcher(|searcher| {
            crate::slices::run_slices(&runs, |run| {
                run_on(searcher, max_doc, matcher, parts[run[0]], None)
            })
        })?;
        let mut merged: Vec<BTreeMap<String, QueryMatch>> = vec![BTreeMap::new(); docs.len()];
        let mut errors = BTreeMap::new();
        for part in results {
            for (doc, found) in part.matches.into_iter().enumerate() {
                for (id, m) in found {
                    // a query's pieces split across runs resolve in run order,
                    // the order the single pass resolves them in
                    match merged[doc].remove(&id) {
                        Some(old) => {
                            merged[doc].insert(id, old.resolve(m));
                        }
                        None => {
                            merged[doc].insert(id, m);
                        }
                    }
                }
            }
            errors.extend(part.errors);
        }
        let run = MultiMatchingQueries {
            matches: merged,
            errors,
            queries_run: candidates.len(),
            batch_size: docs.len(),
            query_build_time: 0,
            search_time: 0,
        };
        Ok(finish(run, prepared.build_nanos, prepared.made))
    }
}

/// The fewest candidates [`Monitor::match_docs_partitioned`] gives a thread.
/// A candidate's query costs a few microseconds against a one-document
/// batch; waking a pool thread costs tens, so a thread is given at
/// least a few hundred microseconds of work.
pub const MIN_CANDIDATES_PER_THREAD: usize = 64;

/// What a match needs before its candidates run: the batch, its presearcher
/// terms and filter clause, when the matcher was made and how long the
/// presearcher's query took to build.
struct Prepared {
    batch: DocumentBatch,
    terms: BatchTerms,
    filter: Option<Clause>,
    made: std::time::Instant,
    build_nanos: u64,
}

/// `HighlightsMatch.buildMatch`: every field's match positions and offsets,
/// descending into sub-matches where a match has them.
fn build_hits(found: &dyn matches::Matches) -> Result<Hits> {
    let mut hits = Hits::new();
    for field in found.fields() {
        let Some(mut mi) = found.get_matches(&field)? else {
            continue;
        };
        while mi.next()? {
            match mi.sub_matches()? {
                Some(mut sub) => {
                    while sub.next()? {
                        add_hit(&mut hits, &field, sub.as_ref());
                    }
                }
                None => add_hit(&mut hits, &field, mi.as_ref()),
            }
        }
    }
    Ok(hits)
}

fn add_hit(hits: &mut Hits, field: &str, it: &dyn matches::MatchesIterator) {
    hits.entry(field.to_string()).or_default().insert(Hit {
        start_position: it.start_position(),
        start_offset: it.start_offset(),
        end_position: it.end_position(),
        end_offset: it.end_offset(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{Clause, TermQuery};
    use lucene_analysis::StandardAnalyzer;

    fn monitor_with(queries: &[(&str, Clause)]) -> Monitor {
        let mut m = Monitor::new(Arc::new(Analyzer::new(StandardAnalyzer::new())));
        m.register(
            queries
                .iter()
                .map(|(id, q)| MonitorQuery::new(*id, q.clone(), None, BTreeMap::new()).unwrap())
                .collect(),
        )
        .unwrap();
        m
    }

    fn term(text: &str) -> Clause {
        Clause::Term(TermQuery::new("f", text.as_bytes().to_vec()))
    }

    #[test]
    fn hits_are_equal_and_ordered_by_position_alone() {
        let a = Hit {
            start_position: 1,
            start_offset: 4,
            end_position: 1,
            end_offset: 9,
        };
        let b = Hit {
            start_position: 1,
            start_offset: -1,
            end_position: 1,
            end_offset: -1,
        };
        let c = Hit {
            start_position: 2,
            start_offset: 0,
            end_position: 2,
            end_offset: 1,
        };
        assert_eq!(a, b);
        assert!(a < c);
        assert_eq!(a.to_string(), "1(4)->1(9)");
        let mut set = BTreeSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn two_matches_of_one_query_resolve_by_summing_and_merging() {
        let mut hits_a = Hits::new();
        hits_a.entry("f".into()).or_default().insert(Hit {
            start_position: 0,
            start_offset: 0,
            end_position: 0,
            end_offset: 1,
        });
        let mut hits_b = Hits::new();
        hits_b.entry("f".into()).or_default().insert(Hit {
            start_position: 3,
            start_offset: 5,
            end_position: 3,
            end_offset: 6,
        });
        hits_b.entry("g".into()).or_default().insert(Hit {
            start_position: 1,
            start_offset: 2,
            end_position: 1,
            end_offset: 3,
        });
        let a = QueryMatch {
            query_id: "q".into(),
            score: Some(1.5),
            explanation: Some(Explanation::match_(1.5, "a")),
            hits: Some(hits_a),
        };
        let b = QueryMatch {
            query_id: "q".into(),
            score: Some(0.25),
            explanation: Some(Explanation::match_(0.25, "b")),
            hits: Some(hits_b),
        };
        let r = a.resolve(b);
        assert_eq!(r.score, Some(1.75));
        let ex = r.explanation.unwrap();
        assert_eq!(ex.description, "sum of:");
        assert_eq!(ex.details.len(), 2);
        assert_eq!(r.hits.unwrap()["f"].len(), 2);
        // a match without detail keeps the other's
        let plain = QueryMatch::plain("q");
        let scored = QueryMatch {
            score: Some(2.0),
            ..QueryMatch::plain("q")
        };
        assert_eq!(plain.resolve(scored).score, Some(2.0));
    }

    #[test]
    fn an_empty_batch_is_refused_and_a_single_document_matches() {
        let m = monitor_with(&[("a", term("fox")), ("b", term("dog"))]);
        let err = m.match_docs(&[], &Matcher::Simple).err().unwrap();
        assert_eq!(
            err.to_string(),
            "A DocumentBatch must contain at least one document"
        );
        let one = m.match_doc(&[("f", "the quick fox")]).unwrap();
        assert_eq!(one.match_count(), 1);
        assert!(one.matches("a").is_some() && one.matches("b").is_none());
        assert_eq!(one.queries_run, 1);
        assert!(one.errors.is_empty());
    }

    #[test]
    fn a_batch_scores_with_its_own_statistics_and_reports_counts() {
        let m = monitor_with(&[("a", term("fox"))]);
        let docs: [DocFields<'_>; 2] = [&[("f", "fox")], &[("f", "dog")]];
        let r = m.match_docs(&docs, &Matcher::Scoring(None)).unwrap();
        assert_eq!(r.batch_size, 2);
        assert_eq!(r.match_count(0), 1);
        assert_eq!(r.match_count(1), 0);
        assert_eq!(r.match_count(9), 0);
        assert!(r.matches("a", 0).unwrap().score.unwrap() > 0.0);
        assert!(r.matches("a", 1).is_none());
        // the same batch under the explaining and highlighting matchers
        let e = m.match_docs(&docs, &Matcher::Explaining).unwrap();
        assert!(
            e.matches("a", 0)
                .unwrap()
                .explanation
                .as_ref()
                .unwrap()
                .matched
        );
        let h = m.match_docs(&docs, &Matcher::Highlights).unwrap();
        assert_eq!(
            h.matches("a", 0).unwrap().hits.as_ref().unwrap()["f"].len(),
            1
        );
    }

    #[test]
    fn registration_surface_and_similarity_pass_through() {
        let mut m = monitor_with(&[("a", term("fox")), ("b", term("dog"))]);
        assert_eq!((m.query_count(), m.disjunct_count()), (2, 2));
        assert_eq!(m.query_ids(), vec!["a".to_string(), "b".to_string()]);
        assert!(m.get_query("a").is_some());
        m.delete_by_id(&["a"]).unwrap();
        assert_eq!(m.query_count(), 1);
        m.clear().unwrap();
        assert_eq!(m.query_count(), 0);
        let m = monitor_with(&[("a", term("fox"))]);
        let sim: Arc<dyn Similarity> = Arc::new(crate::similarities::ClassicSimilarity::default());
        let classic = m
            .match_doc_with_similarity(&[("f", "fox fox")], sim)
            .unwrap();
        let bm25 = m.match_doc_scored(&[("f", "fox fox")]).unwrap();
        assert_ne!(classic.matches[0].score, bm25.matches[0].score);
    }
}

#[cfg(test)]
mod partition_tests {
    use super::*;

    #[test]
    fn partition_splits_as_java_does() {
        let items: Vec<u32> = (0..10).collect();
        let runs: Vec<Vec<u32>> = partition(&items, 3)
            .into_iter()
            .map(|r| r.to_vec())
            .collect();
        // size 10/3: runs of 3, 3, then the remainder
        assert_eq!(runs, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8, 9]]);
        let one: Vec<&[u32]> = partition(&items, 1);
        assert_eq!(one, vec![&items[..]]);
        let empty: Vec<u32> = Vec::new();
        assert_eq!(
            partition(&empty, 4).iter().map(|r| r.len()).sum::<usize>(),
            0
        );
    }

    #[test]
    fn partitioned_matching_equals_single_pass_for_any_thread_count() {
        use lucene_analysis::StandardAnalyzer;
        let mut m = Monitor::new(Arc::new(Analyzer::new(StandardAnalyzer::new())));
        // `fox AND quick` is two pieces on one query; the partitioned result must
        // equal the single pass's, whichever thread runs the query
        let mut both = crate::query::BooleanQuery::new();
        both.must
            .push(crate::query::Clause::Term(crate::query::TermQuery::new(
                "f",
                b"fox".to_vec(),
            )));
        both.must
            .push(crate::query::Clause::Term(crate::query::TermQuery::new(
                "f",
                b"quick".to_vec(),
            )));
        let mut queries: Vec<MonitorQuery> = ["fox", "dog", "quick"]
            .iter()
            .enumerate()
            .map(|(i, t)| {
                MonitorQuery::new(
                    format!("q{i}"),
                    crate::query::Clause::Term(crate::query::TermQuery::new(
                        "f",
                        t.as_bytes().to_vec(),
                    )),
                    None,
                    BTreeMap::new(),
                )
                .unwrap()
            })
            .collect();
        queries.push(
            MonitorQuery::new(
                "both",
                crate::query::Clause::Boolean(Box::new(both)),
                None,
                BTreeMap::new(),
            )
            .unwrap(),
        );
        m.register(queries).unwrap();
        let docs: [DocFields<'_>; 3] =
            [&[("f", "the quick fox")], &[("f", "dog")], &[("f", "zzz")]];
        for matcher in [Matcher::Simple, Matcher::Scoring(None)] {
            let whole = m.match_docs(&docs, &matcher).unwrap();
            for threads in [1, 2, 3, 8] {
                let parts = m.match_docs_partitioned(&docs, &matcher, threads).unwrap();
                assert_eq!(parts.matches, whole.matches, "threads {threads}");
            }
        }
    }
}

#[cfg(test)]
mod cache_and_timing_tests {
    use super::*;
    use crate::monitor::config::SlowLog;
    use crate::query::{Clause, TermQuery};
    use lucene_analysis::StandardAnalyzer;

    fn mq(id: &str, text: &str) -> MonitorQuery {
        MonitorQuery::new(
            id,
            Clause::Term(TermQuery::new("f", text.as_bytes().to_vec())),
            Some(text.to_string()),
            BTreeMap::new(),
        )
        .unwrap()
    }

    #[test]
    fn the_cache_stats_count_pieces_and_a_purge_stamps_the_time() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Purges(AtomicUsize);
        impl MonitorUpdateListener for Purges {
            fn on_purge(&self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let purges = Arc::new(Purges(AtomicUsize::new(0)));
        let mut m = Monitor::new(Arc::new(Analyzer::new(StandardAnalyzer::new())));
        m.add_query_index_update_listener(purges.clone());
        m.register(vec![mq("a", "fox"), mq("b", "dog")]).unwrap();
        let before = m.query_cache_stats();
        assert_eq!((before.queries, before.cached_queries), (2, 2));
        assert!(before.last_purged.is_none());
        m.purge_cache().unwrap();
        assert!(m.query_cache_stats().last_purged.is_some());
        assert_eq!(purges.0.load(Ordering::SeqCst), 1);
        assert_eq!(m.query_count(), 2);
    }

    #[test]
    fn a_purge_of_a_persisted_index_reloads_it() {
        let tmp = lucene_util::test_support::TempDir::new("monitor-purge");
        let path: &std::path::Path = &tmp;
        let ser =
            crate::monitor::serializer::MonitorQuerySerializer::from_parser(Arc::new(|t: &str| {
                Ok(Clause::Term(TermQuery::new("f", t.as_bytes().to_vec())))
            }));
        let config = MonitorConfiguration::default().set_directory(path, ser, false);
        let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
        let mut m = Monitor::with_configuration(
            Arc::clone(&analyzer),
            TermFilteredPresearcher::default(),
            Vec::new(),
            &config,
        )
        .unwrap();
        m.register(vec![mq("a", "fox"), mq("b", "dog")]).unwrap();
        // a second monitor deletes a query behind the first one's back
        let mut other = Monitor::with_configuration(
            analyzer,
            TermFilteredPresearcher::default(),
            Vec::new(),
            &config,
        )
        .unwrap();
        other.delete_by_id(&["b"]).unwrap();
        assert_eq!(m.query_count(), 2);
        m.purge_cache().unwrap();
        assert_eq!(m.query_ids(), vec!["a".to_string()]);
    }

    #[test]
    fn each_selected_query_is_timed() {
        let mut m = Monitor::new(Arc::new(Analyzer::new(StandardAnalyzer::new())));
        m.register(vec![mq("a", "fox"), mq("b", "dog"), mq("c", "cat")])
            .unwrap();
        let mut log = SlowLog::new();
        let docs: [DocFields<'_>; 1] = [&[("f", "fox dog")]];
        let result = m
            .match_docs_timed(&docs, &Matcher::Simple, &mut log)
            .unwrap();
        let timed: Vec<&str> = log.iter().map(|e| e.query_id.as_str()).collect();
        assert_eq!(result.queries_run, 2);
        assert_eq!(timed.len(), 2);
        assert!(timed.contains(&"a") && timed.contains(&"b"));
        assert!(log.to_string().contains("a ["));
    }

    #[test]
    fn a_run_reports_its_times_and_equality_ignores_them() {
        let mut m = Monitor::new(Arc::new(Analyzer::new(StandardAnalyzer::new())));
        m.register(vec![mq("a", "fox")]).unwrap();
        let docs: [DocFields<'_>; 1] = [&[("f", "fox")]];
        let mut first = m.match_docs(&docs, &Matcher::Simple).unwrap();
        // the build is measured on a monotonic clock; a run of a few
        // microseconds may read 0 milliseconds of search
        assert!(first.search_time < 60_000);
        let partitioned = m
            .match_docs_partitioned(&docs, &Matcher::Simple, 2)
            .unwrap();
        first.query_build_time = first.query_build_time.wrapping_add(1);
        first.search_time = first.search_time.wrapping_add(1);
        assert_eq!(first, partitioned);
        let mut one = m.match_doc(&[("f", "fox")]).unwrap();
        let other = one.clone();
        one.query_build_time = one.query_build_time.wrapping_add(1);
        one.search_time = one.search_time.wrapping_add(1);
        assert_eq!(one, other);
        one.queries_run += 1;
        assert_ne!(one, other);
        let debug = m.debug_docs(&docs, &Matcher::Simple).unwrap();
        assert_eq!(debug.matcher, partitioned);
        assert_eq!(nanos_since(std::time::Instant::now()) / 1_000_000_000, 0);
    }
}

/// Lucene's `TestFieldFilterPresearcherComponent`: a presearcher's filter
/// field narrows the candidates to the queries whose metadata carries the
/// batch's value.
#[cfg(test)]
mod filter_field_tests {
    use super::*;
    use crate::monitor::tree::TermWeightor;
    use crate::query::{Clause, MatchAllDocsQuery, TermQuery};
    use lucene_analysis::StandardAnalyzer;

    fn monitor() -> Monitor {
        let language = vec!["language".to_string()];
        let mut m = Monitor::with_presearcher(
            Arc::new(Analyzer::new(StandardAnalyzer::new())),
            TermFilteredPresearcher::new(TermWeightor::default(), language.clone()),
            language,
        );
        let mq = |id: &str, q: Clause, lang: Option<&str>| {
            let mut md = BTreeMap::new();
            if let Some(l) = lang {
                md.insert("language".to_string(), Some(l.to_string()));
            }
            MonitorQuery::new(id, q, None, md).unwrap()
        };
        let t = |s: &str| Clause::Term(TermQuery::new("f", s.as_bytes().to_vec()));
        m.register(vec![
            mq("1", t("test"), Some("en")),
            mq("2", t("test"), Some("de")),
            mq("3", t("wahl"), Some("de")),
            mq("4", t("wahl"), Some("es")),
            mq(
                "5",
                Clause::MatchAllDocs(MatchAllDocsQuery::new(0)),
                Some("es"),
            ),
            mq("6", t("test"), None),
        ])
        .unwrap();
        m
    }

    fn ids(found: &MatchingQueries) -> Vec<&str> {
        found.matches.iter().map(|m| m.query_id.as_str()).collect()
    }

    #[test]
    fn a_documents_filter_value_selects_only_the_queries_carrying_it() {
        let m = monitor();
        let en = m
            .match_doc(&[("f", "this is a test wahl"), ("language", "en")])
            .unwrap();
        assert_eq!(en.queries_run, 1);
        assert_eq!(ids(&en), vec!["1"]);
        let de = m
            .match_doc(&[("f", "das ist ein test wahl"), ("language", "de")])
            .unwrap();
        assert_eq!(de.queries_run, 2);
        assert_eq!(ids(&de), vec!["2", "3"]);
        // the any-token query of `es` runs only for an `es` document
        let es = m
            .match_doc(&[("f", "nothing here"), ("language", "es")])
            .unwrap();
        assert_eq!(es.queries_run, 1);
        assert_eq!(ids(&es), vec!["5"]);
        // two values on one document: either value selects
        let both = m
            .match_doc(&[("f", "test"), ("language", "en"), ("language", "de")])
            .unwrap();
        assert_eq!(both.queries_run, 2);
        assert_eq!(ids(&both), vec!["1", "2"]);
        // no filter value: every query the terms select, metadata or not
        let none = m.match_doc(&[("f", "test wahl")]).unwrap();
        assert_eq!(none.queries_run, 6);
        assert_eq!(ids(&none), vec!["1", "2", "3", "4", "5", "6"]);
    }

    #[test]
    fn a_batch_whose_documents_differ_in_filter_value_is_refused() {
        let m = monitor();
        let docs: [DocFields<'_>; 2] = [
            &[("f", "test"), ("language", "en")],
            &[("f", "test"), ("language", "de")],
        ];
        let err = m.match_docs(&docs, &Matcher::Simple).unwrap_err();
        assert!(
            err.to_string()
                .contains("Some documents in this batch do not have a term value of language:de"),
            "{err}"
        );
        let docs: [DocFields<'_>; 2] = [
            &[("f", "test"), ("language", "en")],
            &[("f", "test wahl"), ("language", "en")],
        ];
        let found = m.match_docs(&docs, &Matcher::Simple).unwrap();
        assert_eq!(found.queries_run, 1);
        assert_eq!(found.match_count(0), 1);
        assert_eq!(found.match_count(1), 1);
        let parted = m
            .match_docs_partitioned(&docs, &Matcher::Simple, 3)
            .unwrap();
        assert_eq!(parted, found);
        // debug selects the same candidates, its why naming the filter value
        let debug = m.debug_docs(&docs, &Matcher::Simple).unwrap();
        let hit = debug.match_query("1", 0).unwrap();
        assert!(hit.presearcher_matches.contains("language:en"), "{hit:?}");
        assert!(debug.match_query("2", 0).is_none());
    }

    #[test]
    fn partitioning_spreads_many_candidates_and_equals_one_thread() {
        let mut m = Monitor::new(Arc::new(Analyzer::new(StandardAnalyzer::new())));
        let words = ["alpha", "beta", "gamma", "delta"];
        m.register(
            (0..200)
                .map(|i| {
                    let mut b = crate::query::BooleanQuery::new();
                    b.should.push(Clause::Term(TermQuery::new(
                        "f",
                        words[i % 4].as_bytes().to_vec(),
                    )));
                    b.should.push(Clause::Term(TermQuery::new(
                        "f",
                        words[(i + 1) % 4].as_bytes().to_vec(),
                    )));
                    MonitorQuery::new(
                        format!("q{i}"),
                        Clause::Boolean(Box::new(b)),
                        None,
                        BTreeMap::new(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let docs: [DocFields<'_>; 1] = [&[("f", "alpha beta gamma delta")]];
        for matcher in [Matcher::Simple, Matcher::Scoring(None), Matcher::Explaining] {
            let whole = m.match_docs(&docs, &matcher).unwrap();
            assert_eq!(whole.queries_run, 400);
            for threads in [2, 4, 7, 64] {
                let parted = m.match_docs_partitioned(&docs, &matcher, threads).unwrap();
                assert_eq!(parted, whole, "threads {threads}");
            }
        }
    }
}
