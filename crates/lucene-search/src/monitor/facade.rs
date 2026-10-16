//! `org.apache.lucene.monitor.Monitor` and its matchers: `QueryMatch`,
//! `ScoringMatch`, `ExplainingMatch`, `HighlightsMatch`, `MatchingQueries`
//! and `MultiMatchingQueries`. Queries are registered once; a batch of
//! documents is matched against them, the presearcher narrowing the
//! candidates and each candidate's query run against the batch.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lucene_analysis::Analyzer;

use crate::explain::Explanation;
use crate::matches;
use crate::monitor::batch::{DocFields, DocumentBatch};
use crate::monitor::config::{MonitorConfiguration, MonitorUpdateListener, QueryTimeListener};
use crate::monitor::index::{
    as_search_query, batch_terms, AcceptAll, MonitorQuery, QueryCacheStats, QueryIndex,
};
use crate::monitor::presearcher::TermFilteredPresearcher;
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
#[derive(Debug, Clone, PartialEq)]
pub struct MultiMatchingQueries {
    /// Per document, the matches by query id.
    pub matches: Vec<BTreeMap<String, QueryMatch>>,
    /// Queries that failed to run, by id, with the message.
    pub errors: BTreeMap<String, String>,
    /// The queries the presearcher selected.
    pub queries_run: usize,
    pub batch_size: usize,
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
#[derive(Debug, Clone, PartialEq)]
pub struct MatchingQueries {
    /// Sorted by query id.
    pub matches: Vec<QueryMatch>,
    /// The queries the presearcher selected.
    pub queries_run: usize,
    /// Queries that failed to run, by id, with the message.
    pub errors: BTreeMap<String, String>,
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
        let similarity = match matcher {
            Matcher::Scoring(s) => s.clone(),
            _ => None,
        };
        let batch = DocumentBatch::of(&self.analyzer, docs, similarity)?;
        let terms = batch_terms(batch.terms(), &self.filter_fields);
        let candidates = self.index.select(&terms, &AcceptAll);
        self.run_candidates(&batch, matcher, &candidates, Some(listener))
    }

    /// `debug(docs, matcher)`: the match, and for every query the presearcher
    /// selected, which of its indexed tokens did.
    pub fn debug_docs(
        &self,
        docs: &[DocFields<'_>],
        matcher: &Matcher,
    ) -> Result<PresearcherMatches> {
        let similarity = match matcher {
            Matcher::Scoring(s) => s.clone(),
            _ => None,
        };
        let batch = DocumentBatch::of(&self.analyzer, docs, similarity)?;
        let terms = batch_terms(batch.terms(), &self.filter_fields);
        let selected = self.index.select_debug(&terms, &AcceptAll);
        let mut matching_terms: BTreeMap<String, String> = BTreeMap::new();
        for (entry, why) in &selected {
            matching_terms
                .entry(entry.query_id.clone())
                .or_default()
                .push_str(why);
        }
        let candidates: Vec<&crate::monitor::index::QueryCacheEntry> =
            selected.iter().map(|(e, _)| *e).collect();
        let matcher = self.run_candidates(&batch, matcher, &candidates, None)?;
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
        let similarity = match matcher {
            Matcher::Scoring(s) => s.clone(),
            _ => None,
        };
        let batch = DocumentBatch::of(&self.analyzer, docs, similarity)?;
        let terms = batch_terms(batch.terms(), &self.filter_fields);
        let candidates = self.index.select(&terms, &AcceptAll);
        self.run_candidates(&batch, matcher, &candidates, None)
    }

    /// Runs each candidate's query against the batch, under `matcher`.
    fn run_candidates(
        &self,
        batch: &DocumentBatch,
        matcher: &Matcher,
        candidates: &[&crate::monitor::index::QueryCacheEntry],
        mut timer: Option<&mut dyn QueryTimeListener>,
    ) -> Result<MultiMatchingQueries> {
        let max_doc = batch.max_doc();
        let mut per_doc: Vec<BTreeMap<String, QueryMatch>> = vec![BTreeMap::new(); max_doc];
        let mut errors = BTreeMap::new();
        batch.with_searcher(|searcher| {
            for entry in candidates {
                let Some(query) = as_search_query(&entry.match_query) else {
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
                                let ex = searcher
                                    .explain(&query.boolean, i32::try_from(doc).unwrap_or(0))?;
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
        })?;
        Ok(MultiMatchingQueries {
            matches: per_doc,
            errors,
            queries_run: candidates.len(),
            batch_size: max_doc,
        })
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
    /// `match(docs, PartitionMatcher.factory(executor, matcher, threads))`: the
    /// selected entries are split across `threads` workers, each running its
    /// share against the batch, and the per-document matches merge into one
    /// result. Equal to [`Monitor::match_docs`] for any thread count: a query
    /// sits in exactly one partition, and `resolve` combines its pieces the
    /// same way whichever partition they land in.
    pub fn match_docs_partitioned(
        &self,
        docs: &[DocFields<'_>],
        matcher: &Matcher,
        threads: usize,
    ) -> Result<MultiMatchingQueries> {
        let threads = threads.max(1);
        let whole = self.match_docs(docs, matcher)?;
        // The partitions run over the selected entries; the result is the
        // single-pass result, so the workers are the run's own parts. Each
        // part's matches are computed on its own thread and merged.
        let ids = self.index.query_ids();
        let parts: Vec<Vec<String>> = partition(&ids, threads)
            .into_iter()
            .map(|p| p.to_vec())
            .collect();
        let mut merged: Vec<BTreeMap<String, QueryMatch>> = vec![BTreeMap::new(); docs.len()];
        let mut errors = BTreeMap::new();
        std::thread::scope(|scope| -> Result<()> {
            let handles: Vec<_> = parts
                .iter()
                .map(|part| {
                    scope.spawn(move || -> Result<MultiMatchingQueries> {
                        // a worker's own match, over only its queries
                        let mut own = self.match_docs_for(docs, matcher, part)?;
                        own.queries_run = part.len();
                        Ok(own)
                    })
                })
                .collect();
            for h in handles {
                let part = h
                    .join()
                    .map_err(|_| Error::IllegalState("a matcher worker panicked".into()))??;
                for (doc, found) in part.matches.into_iter().enumerate() {
                    for (id, m) in found {
                        // Partitions hold whole queries (`PartitionMatcher` splits
                        // its tasks), so one query's pieces share a partition and
                        // this merge sees a query once; it is kept for a partition
                        // that splits a query's entries.
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
            Ok(())
        })?;
        Ok(MultiMatchingQueries {
            matches: merged,
            errors,
            queries_run: whole.queries_run,
            batch_size: docs.len(),
        })
    }

    /// [`Monitor::match_docs`] restricted to the queries named in `ids`.
    fn match_docs_for(
        &self,
        docs: &[DocFields<'_>],
        matcher: &Matcher,
        ids: &[String],
    ) -> Result<MultiMatchingQueries> {
        let similarity = match matcher {
            Matcher::Scoring(s) => s.clone(),
            _ => None,
        };
        let batch = DocumentBatch::of(&self.analyzer, docs, similarity)?;
        let terms = batch_terms(batch.terms(), &self.filter_fields);
        let candidates: Vec<_> = self
            .index
            .select(&terms, &AcceptAll)
            .into_iter()
            .filter(|e| ids.contains(&e.query_id))
            .collect();
        self.run_candidates(&batch, matcher, &candidates, None)
    }
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
}
