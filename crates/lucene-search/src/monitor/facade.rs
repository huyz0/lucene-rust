//! `org.apache.lucene.monitor.Monitor`, `QueryMatch`, `ScoringMatch` and
//! `MatchingQueries`: the percolator's public face. Queries are registered
//! once; a document is matched against them, the presearcher narrowing the
//! candidates and each candidate's query run against the document.

use std::collections::BTreeMap;
use std::sync::Arc;

use lucene_analysis::{Analyzer, TokenStream};

use crate::memory::MemoryIndex;
use crate::monitor::index::{match_batch, MonitorQuery, QueryIndex};
use crate::monitor::presearcher::TermFilteredPresearcher;
use crate::similarities::Similarity;
use crate::{Error, Result};

/// A matching query: `QueryMatch`, or `ScoringMatch` when `score` is set.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryMatch {
    pub query_id: String,
    pub score: Option<f32>,
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
    pub fn match_doc(&self, doc: &[(&str, &str)]) -> Result<MatchingQueries> {
        self.run(doc, false, None)
    }

    /// `match(doc, ScoringMatch.DEFAULT_MATCHER)`: each match carries the
    /// score of its query over the document. Differs: the score is that of a
    /// one-document batch, which is what `match(Document, ...)` is.
    pub fn match_doc_scored(&self, doc: &[(&str, &str)]) -> Result<MatchingQueries> {
        self.run(doc, true, None)
    }

    /// `ScoringMatch.matchWithSimilarity(similarity)`.
    pub fn match_doc_with_similarity(
        &self,
        doc: &[(&str, &str)],
        similarity: Arc<dyn Similarity>,
    ) -> Result<MatchingQueries> {
        self.run(doc, true, Some(similarity))
    }

    fn run(
        &self,
        doc: &[(&str, &str)],
        scored: bool,
        similarity: Option<Arc<dyn Similarity>>,
    ) -> Result<MatchingQueries> {
        let analysis = |e: lucene_analysis::AnalysisError| Error::IllegalArgument(e.to_string());
        let mut mi = MemoryIndex::new();
        if let Some(s) = similarity {
            mi.set_similarity(s)?;
        }
        let mut fields = Vec::new();
        for (field, value) in doc {
            mi.add_field_text(field, value, &self.analyzer)?;
            let mut ts = self.analyzer.token_stream(field, value).map_err(analysis)?;
            ts.reset().map_err(analysis)?;
            let mut terms = Vec::new();
            while ts.increment_token().map_err(analysis)? {
                terms.push(ts.attributes().term_bytes().to_vec());
            }
            ts.end().map_err(analysis)?;
            ts.close().map_err(analysis)?;
            fields.push(((*field).to_string(), terms));
        }
        mi.freeze()?;
        let mi = std::cell::RefCell::new(mi);
        let errors = std::cell::RefCell::new(BTreeMap::new());
        let scores = std::cell::RefCell::new(BTreeMap::<String, f32>::new());
        let result = match_batch(&self.index, &[fields], &self.filter_fields, &|entry, _| {
            match mi.borrow_mut().search(&entry.match_query) {
                Ok(score) if score > 0.0 => {
                    let mut scores = scores.borrow_mut();
                    // `ScoringMatch.resolve`: the pieces of one query add up
                    *scores.entry(entry.query_id.clone()).or_insert(0.0) += score;
                    true
                }
                Ok(_) => false,
                Err(e) => {
                    errors
                        .borrow_mut()
                        .insert(entry.query_id.clone(), e.to_string());
                    false
                }
            }
        });
        let scores = scores.into_inner();
        let mut matches: Vec<QueryMatch> = result.matches[0]
            .iter()
            .map(|id| QueryMatch {
                query_id: id.clone(),
                score: scored.then(|| scores.get(id).copied().unwrap_or(0.0)),
            })
            .collect();
        matches.sort_by(|a, b| a.query_id.cmp(&b.query_id));
        Ok(MatchingQueries {
            matches,
            queries_run: result.queries_run,
            errors: errors.into_inner(),
        })
    }
}
