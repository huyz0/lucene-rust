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
use crate::monitor::config::{MonitorConfiguration, MonitorUpdateListener};
use crate::monitor::persist::{EntryDoc, PersistentStore};
use crate::monitor::presearcher::{TermAcceptor, TermFilteredPresearcher};
use crate::monitor::serializer::MonitorQuerySerializer;
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
    /// The on-disk index and the serializer that fills it, when persisted.
    persist: Option<Persist>,
    listeners: Vec<std::sync::Arc<dyn MonitorUpdateListener>>,
    /// `getLastPurged()`: when the cache was last rebuilt from the index.
    last_purged: Option<std::time::SystemTime>,
}

/// `Monitor.QueryCacheStats`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryCacheStats {
    /// `numDocs()`: documents in the query index (the pieces of the queries).
    pub queries: usize,
    /// `cacheSize()`: pieces held in the query cache.
    pub cached_queries: usize,
    /// When the cache was last purged; `None` before the first purge (Java's -1).
    pub last_purged: Option<std::time::SystemTime>,
}

/// A persisted query index: where it lives and how queries are stored in it.
struct Persist {
    store: PersistentStore,
    serializer: MonitorQuerySerializer,
    read_only: bool,
}

/// One entry's presearcher document: the fields and terms the presearcher's
/// batch query selects it by.
#[derive(Debug, Clone)]
struct IndexedEntry {
    fields: Vec<(String, Vec<Vec<u8>>)>,
    metadata: Vec<(String, String)>,
}

impl QueryIndex {
    /// `new Monitor(analyzer, presearcher)`'s query index, in memory.
    pub fn new(presearcher: TermFilteredPresearcher) -> Self {
        QueryIndex {
            presearcher,
            decomposer: QueryDecomposer,
            queries: BTreeMap::new(),
            entries: Vec::new(),
            persist: None,
            listeners: Vec::new(),
            last_purged: None,
        }
    }

    /// `new Monitor(analyzer, presearcher, configuration)`'s query index. With
    /// an index path, the queries already in the directory are loaded
    /// (`populateQueryCache`); a non-empty directory needs a serializer.
    pub fn with_configuration(
        presearcher: TermFilteredPresearcher,
        config: &MonitorConfiguration,
    ) -> Result<Self, String> {
        let mut index = QueryIndex::new(presearcher);
        index.decomposer = config.query_decomposer.clone();
        let Some(path) = &config.index_path else {
            return Ok(index);
        };
        let store = PersistentStore::open(path).map_err(|e| e.to_string())?;
        let stored = store.load().map_err(|e| e.to_string())?;
        let Some(serializer) = config.serializer.clone() else {
            if store.num_docs().map_err(|e| e.to_string())? != 0 {
                return Err(
                    "Attempting to open a non-empty monitor query index with no MonitorQuerySerializer"
                        .to_string(),
                );
            }
            return Ok(index);
        };
        let mut errors = Vec::new();
        for (_, bytes) in &stored {
            match serializer.deserialize(bytes) {
                Ok(mq) => index.register_in_memory(vec![mq])?,
                Err(e) => errors.push(e),
            }
        }
        if !errors.is_empty() {
            return Err(format!(
                "Couldn't parse some queries from the index: {}",
                errors.join("; ")
            ));
        }
        index.persist = Some(Persist {
            store,
            serializer,
            read_only: config.read_only,
        });
        Ok(index)
    }

    /// `purgeCache()`: the cache rebuilt from the query index. A persisted
    /// index is read again, so a piece that left the index leaves the cache;
    /// an in-memory index is its own cache and only records the purge.
    pub fn purge_cache(&mut self) -> Result<(), String> {
        if let Some(p) = &self.persist {
            let stored = p.store.load().map_err(|e| e.to_string())?;
            let mut fresh = Vec::new();
            for (_, bytes) in &stored {
                fresh.push(p.serializer.deserialize(bytes)?);
            }
            self.queries.clear();
            self.entries.clear();
            self.register_in_memory(fresh)?;
        }
        self.last_purged = Some(std::time::SystemTime::now());
        for l in &self.listeners {
            l.on_purge();
        }
        Ok(())
    }

    /// `getQueryCacheStats()`.
    pub fn cache_stats(&self) -> QueryCacheStats {
        QueryCacheStats {
            queries: self.entries.len(),
            cached_queries: self.entries.len(),
            last_purged: self.last_purged,
        }
    }

    /// `addListener(listener)`.
    pub fn add_listener(&mut self, listener: std::sync::Arc<dyn MonitorUpdateListener>) {
        self.listeners.push(listener);
    }

    /// `Monitor.register`: each query replaces any of the same id, its entries
    /// are rebuilt and the presearcher indexes them; a persisted index writes
    /// them out and commits.
    pub fn register(&mut self, queries: Vec<MonitorQuery>) -> Result<(), String> {
        if let Some(p) = &self.persist {
            if p.read_only {
                return Err(
                    "java.lang.UnsupportedOperationException: Monitor is readOnly cannot commit"
                        .into(),
                );
            }
            if queries.iter().any(|q| q.query_string.is_none()) {
                return Err("Cannot add a MonitorQuery with a null string representation to a non-ephemeral Monitor".into());
            }
        }
        let updates = queries.clone();
        self.register_in_memory(queries)?;
        if let Some(p) = &self.persist {
            let mut docs = Vec::new();
            let mut ids = Vec::new();
            for mq in &updates {
                ids.push(mq.id.clone());
                let mq_bytes = p.serializer.serialize(mq)?;
                for (indexed, entry) in &self.entries {
                    if entry.query_id == mq.id {
                        docs.push(EntryDoc {
                            query_id: entry.query_id.clone(),
                            cache_id: entry.cache_id.clone(),
                            mq: mq_bytes.clone(),
                            fields: indexed.fields.clone(),
                            metadata: indexed.metadata.clone(),
                        });
                    }
                }
            }
            p.store.commit(&ids, &docs).map_err(|e| e.to_string())?;
        }
        for l in &self.listeners {
            l.after_update(&updates);
        }
        Ok(())
    }

    fn register_in_memory(&mut self, queries: Vec<MonitorQuery>) -> Result<(), String> {
        for mq in queries {
            self.remove(&mq.id);
            for entry in QueryCacheEntry::decompose(&mq, &self.decomposer) {
                let indexed = self
                    .presearcher
                    .index_query(&entry.match_query, &entry.metadata)?;
                self.entries.push((
                    IndexedEntry {
                        fields: indexed.fields,
                        metadata: indexed.metadata,
                    },
                    entry,
                ));
            }
            self.queries.insert(mq.id.clone(), mq);
        }
        Ok(())
    }

    fn remove(&mut self, query_id: &str) {
        self.queries.remove(query_id);
        self.entries.retain(|(_, e)| e.query_id != query_id);
    }

    /// `Monitor.deleteById`: the queries and their entries, if registered.
    pub fn delete(&mut self, query_ids: &[String]) -> Result<(), String> {
        if self.persist.as_ref().is_some_and(|p| p.read_only) {
            return Err("java.lang.UnsupportedOperationException: Monitor is readOnly cannot delete queries".into());
        }
        for id in query_ids {
            self.remove(id);
        }
        if let Some(p) = &self.persist {
            p.store.commit(query_ids, &[]).map_err(|e| e.to_string())?;
        }
        for l in &self.listeners {
            l.after_delete(query_ids);
        }
        Ok(())
    }

    /// `Monitor.clear`.
    pub fn clear(&mut self) -> Result<(), String> {
        if self.persist.as_ref().is_some_and(|p| p.read_only) {
            return Err(
                "java.lang.UnsupportedOperationException: Monitor is readOnly cannot clear".into(),
            );
        }
        self.queries.clear();
        self.entries.clear();
        if let Some(p) = &self.persist {
            p.store.clear().map_err(|e| e.to_string())?;
        }
        for l in &self.listeners {
            l.after_clear();
        }
        Ok(())
    }

    /// `Monitor.getQuery(id)`. Lucene reads the query back from its index
    /// documents, so a query whose decomposition is empty has none.
    pub fn get_query(&self, query_id: &str) -> Option<&MonitorQuery> {
        if self.entries.iter().any(|(_, e)| e.query_id == query_id) {
            self.queries.get(query_id)
        } else {
            None
        }
    }

    /// `Monitor.getQueryIds()`: the ids that have a document, sorted (Lucene
    /// scans the index).
    pub fn query_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .entries
            .iter()
            .map(|(_, e)| e.query_id.clone())
            .collect();
        ids.sort();
        ids.dedup();
        ids
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

    /// The entries `select` picks, each with why: the presearcher query's
    /// matches on its indexed document, as `Monitor.debug` writes them -- a
    /// space and the matching query for every matched token.
    pub fn select_debug(
        &self,
        batch: &BatchTerms,
        accept: &dyn TermAcceptor,
    ) -> Vec<(&QueryCacheEntry, String)> {
        let query = self.presearcher.build_query(batch, accept, None);
        self.entries
            .iter()
            .filter_map(|(indexed, entry)| {
                let fields = presearcher_matches(&query, indexed)?;
                let why: String = fields
                    .iter()
                    .flat_map(|(_, hits)| hits.iter())
                    .map(|(_, q)| format!(" {q}"))
                    .collect();
                Some((entry, why))
            })
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
        // a `BooleanQuery`: every `MUST` and `FILTER` clause, no `MUST_NOT`, and
        // -- with no required clause -- at least one `SHOULD`; with none at all
        // it matches nothing
        Clause::Boolean(b) => {
            let required = !b.must.is_empty() || !b.filter.is_empty();
            if !required && b.should.is_empty() {
                return false;
            }
            b.must.iter().all(|c| matches_presearcher(c, indexed))
                && b.filter.iter().all(|c| matches_presearcher(c, indexed))
                && b.must_not.iter().all(|c| !matches_presearcher(c, indexed))
                && (required || b.should.iter().any(|c| matches_presearcher(c, indexed)))
        }
        _ => false,
    }
}

/// One field's presearcher matches on an indexed document: each is the
/// position of the matched token and the query (`MatchesIterator.getQuery()`)
/// that matched it.
type FieldHits = (String, Vec<(usize, String)>);

/// `weight.matches(ctx, doc)` of the presearcher's batch query on an entry's
/// indexed document, as `BooleanWeight`, `MatchesUtils.fromSubMatches` and
/// `TermInSetQuery`'s per-term iterators give it: the fields in clause order,
/// each field's hits by position. `None` when the query does not match.
fn presearcher_matches(query: &Clause, indexed: &IndexedEntry) -> Option<Vec<FieldHits>> {
    // a term set (or one term) over a field: a hit per token of the field
    // whose term is in it
    let leaf = |field: &str, wanted: &dyn Fn(&[u8]) -> bool, shown: String| {
        let mut hits = Vec::new();
        let mut pos = 0;
        for (f, terms) in &indexed.fields {
            if f == field {
                for t in terms {
                    if wanted(t) {
                        hits.push((pos, shown.clone()));
                    }
                    pos += 1;
                }
            }
        }
        (!hits.is_empty()).then(|| vec![(field.to_string(), hits)])
    };
    match query {
        Clause::TermInSet(t) => {
            let names: Vec<String> = t
                .terms
                .iter()
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .collect();
            let shown = format!("{}:({})", t.field, names.join(" "));
            leaf(&t.field, &|x| t.terms.iter().any(|y| y == x), shown)
        }
        Clause::Term(t) => {
            let shown = format!("{}:{}", t.field, String::from_utf8_lossy(&t.term));
            leaf(&t.field, &|x| x == t.term.as_slice(), shown)
        }
        Clause::Boolean(b) => {
            let mut subs = Vec::new();
            for c in b.must.iter().chain(&b.filter) {
                subs.push(presearcher_matches(c, indexed)?);
            }
            if b.must_not
                .iter()
                .any(|c| presearcher_matches(c, indexed).is_some())
            {
                return None;
            }
            for c in &b.should {
                if let Some(m) = presearcher_matches(c, indexed) {
                    subs.push(m);
                }
            }
            if subs.is_empty() {
                return None;
            }
            if subs.len() == 1 {
                return subs.pop();
            }
            // `fromSubMatches`: the distinct fields in order, each field's
            // sub-iterators merged by position
            let mut fields: Vec<String> = Vec::new();
            for sub in &subs {
                for (f, _) in sub {
                    if !fields.contains(f) {
                        fields.push(f.clone());
                    }
                }
            }
            Some(
                fields
                    .into_iter()
                    .map(|f| {
                        let mut hits: Vec<(usize, String)> = subs
                            .iter()
                            .flatten()
                            .filter(|(g, _)| *g == f)
                            .flat_map(|(_, h)| h.iter().cloned())
                            .collect();
                        hits.sort_by_key(|(pos, _)| *pos);
                        (f, hits)
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// A query as a searcher runs it: rewritten, and as the boolean `search` takes.
pub struct SearchQuery {
    /// The rewritten clause (what a weight's `matches` is asked of).
    pub clause: Clause,
    /// The same, as the boolean the searcher takes.
    pub boolean: BooleanQuery,
}

/// `IndexSearcher.rewrite(query)`, then the boolean `search` takes; `None` for
/// a query that matches nothing.
pub fn as_search_query(query: &Clause) -> Option<SearchQuery> {
    let clause = query.clone().rewrite();
    let boolean = match &clause {
        Clause::MatchNoDocs(_) => return None,
        Clause::Boolean(b) => (**b).clone(),
        other => {
            let mut b = BooleanQuery::new();
            b.must.push(other.clone());
            b
        }
    };
    Some(SearchQuery { clause, boolean })
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
        index.delete(&["two".to_string()]).unwrap();
        index.delete(&["absent".to_string()]).unwrap();
        assert_eq!(index.query_ids(), vec!["one".to_string()]);
        assert!(index.get_query("two").is_none());
        index.clear().unwrap();
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
