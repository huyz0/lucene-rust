//! `org.apache.lucene.monitor.QueryIndex`, `WritableQueryIndex`,
//! `QueryCacheEntry`, `MonitorQuery`, `MatchingQueries` and `Monitor`'s
//! register/delete/match facade: the registered queries, and matching a
//! document batch against them.
//!
//! A registered query is decomposed (`QueryDecomposer`), and each piece is a
//! cache entry `id_<n>` with the presearcher's indexed document. A batch is
//! matched by building the presearcher's query over the batch's terms
//! ([`TermFilteredPresearcher::build_query`]), selecting the entries it
//! matches, and running each selected entry's query against the batch.
//!
//! Java keeps the entries in a Lucene index (`WritableQueryIndex`) that it
//! searches with the presearcher's query. This port keeps them in memory and
//! evaluates the presearcher's query over the entries' indexed documents: the
//! same candidate set, without a second on-disk index. The on-disk index and
//! its purge cycle are the persistence half (T13.5d).

use std::collections::BTreeMap;

use crate::monitor::analyzer::QueryDecomposer;
use crate::monitor::presearcher::{TermAcceptor, TermFilteredPresearcher};
use crate::query::{BooleanQuery, Clause};

/// `MonitorQuery`: an id, the query, its source text and its metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct MonitorQuery {
    pub id: String,
    pub query: Clause,
    pub query_string: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

impl MonitorQuery {
    /// `new MonitorQuery(id, query, queryString, metadata)`. A `null`
    /// metadata value is an error, as Java refuses it.
    pub fn new(
        id: impl Into<String>,
        query: Clause,
        query_string: Option<String>,
        metadata: BTreeMap<String, Option<String>>,
    ) -> Result<Self, String> {
        let mut out = BTreeMap::new();
        for (k, v) in metadata {
            match v {
                Some(v) => {
                    out.insert(k, v);
                }
                None => {
                    return Err(format!("Null value for key {k} in metadata map"));
                }
            }
        }
        Ok(MonitorQuery {
            id: id.into(),
            query,
            query_string,
            metadata: out,
        })
    }

    /// `MonitorQuery.toString()`: `id: source { k: v, ... }`, the source being
    /// the query string when given, else the query's own text.
    pub fn describe(&self) -> String {
        let mut sb = format!("{}: ", self.id);
        match &self.query_string {
            Some(q) => sb.push_str(q),
            None => sb.push_str(&crate::monitor::analyzer::leaf_string(&self.query)),
        }
        if !self.metadata.is_empty() {
            let pairs: Vec<String> = self
                .metadata
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect();
            sb.push_str(" { ");
            sb.push_str(&pairs.join(", "));
            sb.push_str(" }");
        }
        sb
    }
}

/// `QueryCacheEntry`: one decomposed piece of a registered query.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryCacheEntry {
    pub cache_id: String,
    pub query_id: String,
    pub match_query: Clause,
    pub metadata: BTreeMap<String, String>,
}

impl QueryCacheEntry {
    /// `QueryCacheEntry.decompose(mq, decomposer)`: one entry per decomposed
    /// sub-query, `id_0`, `id_1`, ... in the decomposer's order.
    pub fn decompose(mq: &MonitorQuery, decomposer: &QueryDecomposer) -> Vec<QueryCacheEntry> {
        decomposer
            .decompose(&mq.query)
            .into_iter()
            .enumerate()
            .map(|(upto, sub)| QueryCacheEntry {
                cache_id: format!("{}_{upto}", mq.id),
                query_id: mq.id.clone(),
                match_query: sub,
                metadata: mq.metadata.clone(),
            })
            .collect()
    }
}

/// One batch document: its fields, each with the terms it analyses to.
pub type BatchDoc = Vec<(String, Vec<Vec<u8>>)>;

/// The batch a presearcher's query is evaluated over: each field's distinct
/// terms.
pub type BatchTerms = BTreeMap<String, Vec<Vec<u8>>>;

/// `QueryIndex`: the registered queries, decomposed, and the presearcher that
/// narrows a batch to them.
pub struct QueryIndex {
    presearcher: TermFilteredPresearcher,
    decomposer: QueryDecomposer,
    /// The queries as registered, by id.
    queries: BTreeMap<String, MonitorQuery>,
    /// Each entry's id, its presearcher document, and the entry.
    entries: Vec<(IndexedEntry, QueryCacheEntry)>,
}

/// One entry's presearcher document: the fields and terms the presearcher's
/// batch query selects it by.
#[derive(Debug, Clone)]
struct IndexedEntry {
    fields: Vec<(String, Vec<Vec<u8>>)>,
}

impl QueryIndex {
    /// `new Monitor(analyzer, presearcher)`'s query index.
    pub fn new(presearcher: TermFilteredPresearcher) -> Self {
        QueryIndex {
            presearcher,
            decomposer: QueryDecomposer,
            queries: BTreeMap::new(),
            entries: Vec::new(),
        }
    }

    /// `Monitor.register`: each query replaces any of the same id, its entries
    /// are rebuilt, and the presearcher indexes them.
    pub fn register(&mut self, queries: Vec<MonitorQuery>) -> Result<(), String> {
        for mq in queries {
            self.delete(&mq.id);
            for entry in QueryCacheEntry::decompose(&mq, &self.decomposer) {
                let indexed = self
                    .presearcher
                    .index_query(&entry.match_query, &entry.metadata)?;
                self.entries.push((
                    IndexedEntry {
                        fields: indexed.fields,
                    },
                    entry,
                ));
            }
            self.queries.insert(mq.id.clone(), mq);
        }
        Ok(())
    }

    /// `Monitor.deleteById`: the query and its entries, if registered.
    pub fn delete(&mut self, query_id: &str) {
        self.queries.remove(query_id);
        self.entries.retain(|(_, e)| e.query_id != query_id);
    }

    /// `Monitor.clear`.
    pub fn clear(&mut self) {
        self.queries.clear();
        self.entries.clear();
    }

    /// `Monitor.getQuery(id)`.
    pub fn get_query(&self, query_id: &str) -> Option<&MonitorQuery> {
        self.queries.get(query_id)
    }

    /// `Monitor.getQueryIds()`.
    pub fn query_ids(&self) -> Vec<String> {
        self.queries.keys().cloned().collect()
    }

    /// `Monitor.getDisjunctCount()`: the entries, one per decomposed piece.
    pub fn disjunct_count(&self) -> usize {
        self.entries.len()
    }

    /// `QueryIndex.search(queryBuilder, matcher)`: every entry the presearcher's
    /// batch query selects, in registration order.
    ///
    /// The selection is evaluated over each entry's indexed fields: an entry is
    /// selected when the batch query matches it, which for the presearcher's
    /// `TermInSetQuery`/any-token shape is when one of its indexed terms is a
    /// batch term, or it carries the any-token marker.
    pub fn select(&self, batch: &BatchTerms, accept: &dyn TermAcceptor) -> Vec<&QueryCacheEntry> {
        let query = self.presearcher.build_query(batch, accept, None);
        self.entries
            .iter()
            .filter(|(indexed, _)| matches_presearcher(&query, indexed))
            .map(|(_, entry)| entry)
            .collect()
    }
}

/// Whether the presearcher's batch query selects an entry's indexed document.
/// The query is a `SHOULD`/`MUST` tree of `TermInSet` and `Term` clauses, so it
/// is evaluated directly: a `TermInSet` matches when the entry indexes one of
/// its terms in that field, a `Term` when it indexes that exact term, and a
/// boolean by its occurrences (`MUST` all, `SHOULD` any, at least one `SHOULD`
/// when there are no `MUST`s).
fn matches_presearcher(query: &Clause, indexed: &IndexedEntry) -> bool {
    match query {
        Clause::TermInSet(t) => indexed
            .fields
            .iter()
            .any(|(f, terms)| *f == t.field && terms.iter().any(|x| t.terms.contains(x))),
        Clause::Term(t) => indexed
            .fields
            .iter()
            .any(|(f, terms)| *f == t.field && terms.contains(&t.term)),
        Clause::Boolean(b) => {
            let must_ok = b.must.iter().all(|c| matches_presearcher(c, indexed));
            let filter_ok = b.filter.iter().all(|c| matches_presearcher(c, indexed));
            let not_ok = b.must_not.iter().all(|c| !matches_presearcher(c, indexed));
            let should_ok =
                b.should.is_empty() || b.should.iter().any(|c| matches_presearcher(c, indexed));
            must_ok && filter_ok && not_ok && should_ok
        }
        _ => false,
    }
}

/// `BooleanQuery` with no clauses, the empty batch's query.
pub fn empty_query() -> Clause {
    Clause::Boolean(Box::new(BooleanQuery::new()))
}

/// One document batch's terms, as the presearcher is asked about them: each
/// field's distinct analysed terms, sorted, from `(field, text)` values the
/// analyzer tokenizes. A field named in the filter set is left out, as
/// `TermFilteredPresearcher.buildQuery` leaves it out.
pub fn batch_terms(fields: &[(String, Vec<Vec<u8>>)], filter_fields: &[String]) -> BatchTerms {
    let mut out: BatchTerms = BTreeMap::new();
    for (field, terms) in fields {
        if filter_fields.contains(field) {
            continue;
        }
        let v = out.entry(field.clone()).or_default();
        for t in terms {
            if !v.contains(t) {
                v.push(t.clone());
            }
        }
    }
    for v in out.values_mut() {
        v.sort_unstable();
    }
    out
}

/// `TermAcceptor` that accepts every term: a batch matched in full.
pub struct AcceptAll;

impl TermAcceptor for AcceptAll {
    fn accept(&self, _field: &str, _term: &[u8]) -> bool {
        true
    }
}

/// `CandidateMatcher`'s per-document result: the matching query ids, and the
/// queries that failed to run, as Java's `MultiMatchingQueries` holds them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MatchResult {
    /// For each batch document, the ids of the queries that matched it.
    pub matches: Vec<Vec<String>>,
    /// The queries the presearcher selected.
    pub queries_run: usize,
}

/// `Monitor.match(docs, matcher)`: the presearcher selects the candidate
/// entries for the batch; each candidate's query runs against the batch's
/// documents (`run`), and a query matches a document when `run` says so.
/// `run` is the query evaluator, which the monitor's matchers supply.
pub fn match_batch(
    index: &QueryIndex,
    batch_docs: &[BatchDoc],
    filter_fields: &[String],
    run: &dyn Fn(&QueryCacheEntry, usize) -> bool,
) -> MatchResult {
    // the presearcher's view of the whole batch: its terms, unioned
    let mut union: Vec<(String, Vec<Vec<u8>>)> = Vec::new();
    for doc in batch_docs {
        for (f, t) in doc {
            match union.iter_mut().find(|(uf, _)| uf == f) {
                Some((_, ut)) => ut.extend(t.iter().cloned()),
                None => union.push((f.clone(), t.clone())),
            }
        }
    }
    let batch = batch_terms(&union, filter_fields);
    let candidates = index.select(&batch, &AcceptAll);
    let mut matches = vec![Vec::new(); batch_docs.len()];
    for entry in &candidates {
        for (doc, slot) in matches.iter_mut().enumerate() {
            if run(entry, doc) && !slot.contains(&entry.query_id) {
                slot.push(entry.query_id.clone());
            }
        }
    }
    MatchResult {
        matches,
        queries_run: candidates.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::TermQuery;

    fn term(field: &str, text: &str) -> Clause {
        Clause::Term(TermQuery::new(field, text.as_bytes().to_vec()))
    }

    fn mq(id: &str, query: Clause) -> MonitorQuery {
        MonitorQuery::new(id, query, None, BTreeMap::new()).unwrap()
    }

    #[test]
    fn a_null_metadata_value_is_refused_and_metadata_prints_sorted() {
        let mut md = BTreeMap::new();
        md.insert("b".to_string(), Some("2".to_string()));
        md.insert("a".to_string(), None);
        assert_eq!(
            MonitorQuery::new("q", term("f", "x"), None, md).unwrap_err(),
            "Null value for key a in metadata map"
        );
        let mut md = BTreeMap::new();
        md.insert("b".to_string(), Some("2".to_string()));
        md.insert("a".to_string(), Some("1".to_string()));
        let q = MonitorQuery::new("q", term("f", "x"), Some("f:x".into()), md).unwrap();
        assert_eq!(q.describe(), "q: f:x { a: 1, b: 2 }");
        assert_eq!(mq("q2", term("f", "x")).describe(), "q2: f:x");
    }

    #[test]
    fn registering_replaces_deletes_and_clears() {
        let mut index = QueryIndex::new(TermFilteredPresearcher::default());
        let mut or = BooleanQuery::new();
        or.should.push(term("f", "a"));
        or.should.push(term("f", "b"));
        index
            .register(vec![
                mq("one", Clause::Boolean(Box::new(or))),
                mq("two", term("f", "c")),
            ])
            .unwrap();
        // the disjunction decomposes into two entries `one_0`, `one_1`
        assert_eq!(index.disjunct_count(), 3);
        assert_eq!(
            index.query_ids(),
            vec!["one".to_string(), "two".to_string()]
        );
        index.register(vec![mq("one", term("f", "z"))]).unwrap();
        assert_eq!(index.disjunct_count(), 2);
        assert_eq!(index.get_query("one").unwrap().query, term("f", "z"));
        index.delete("two");
        index.delete("absent");
        assert_eq!(index.query_ids(), vec!["one".to_string()]);
        assert!(index.get_query("two").is_none());
        index.clear();
        assert_eq!(index.disjunct_count(), 0);
        assert!(index.query_ids().is_empty());
    }

    #[test]
    fn the_filter_fields_are_left_out_of_a_batch_and_terms_are_distinct_sorted() {
        let fields = vec![
            (
                "f".to_string(),
                vec![b"b".to_vec(), b"a".to_vec(), b"b".to_vec()],
            ),
            ("f".to_string(), vec![b"c".to_vec()]),
            ("tenant".to_string(), vec![b"t1".to_vec()]),
        ];
        let batch = batch_terms(&fields, &["tenant".to_string()]);
        assert_eq!(batch.len(), 1);
        assert_eq!(
            batch["f"],
            vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]
        );
    }

    #[test]
    fn a_batch_selects_by_term_and_any_token_and_runs_each_document() {
        let mut index = QueryIndex::new(TermFilteredPresearcher::default());
        index
            .register(vec![
                mq("fox", term("f", "fox")),
                mq("dog", term("f", "dog")),
                // a leaf no term narrows is selected for every batch
                mq(
                    "wild",
                    Clause::Prefix(crate::query::PrefixQuery::new("f", b"zz".to_vec())),
                ),
            ])
            .unwrap();
        let docs = vec![
            vec![("f".to_string(), vec![b"fox".to_vec()])],
            vec![("f".to_string(), vec![b"cat".to_vec()])],
        ];
        let result = match_batch(&index, &docs, &[], &|entry, doc| {
            // document 0 holds `fox`; document 1 matches nothing
            doc == 0 && entry.query_id == "fox"
        });
        // `dog` is not selected (no batch term), `wild` and `fox` are
        assert_eq!(result.queries_run, 2);
        assert_eq!(result.matches, vec![vec!["fox".to_string()], vec![]]);
        assert!(matches!(empty_query(), Clause::Boolean(_)));
    }
}
