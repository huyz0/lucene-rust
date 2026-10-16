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
use crate::monitor::index::{as_search_query, batch_terms, AcceptAll, MonitorQuery, QueryIndex};
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

    /// `deleteById(ids)`.
    pub fn delete_by_id(&mut self, ids: &[&str]) {
        for id in ids {
            self.index.delete(id);
        }
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.index.clear();
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
        let max_doc = batch.max_doc();
        let mut per_doc: Vec<BTreeMap<String, QueryMatch>> = vec![BTreeMap::new(); max_doc];
        let mut errors = BTreeMap::new();
        batch.with_searcher(|searcher| {
            for entry in &candidates {
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
        m.delete_by_id(&["a"]);
        assert_eq!(m.query_count(), 1);
        m.clear();
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
