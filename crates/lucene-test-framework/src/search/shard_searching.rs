//! `ShardSearchingTestBase`: simulated distributed search. Each node is
//! one index with its own writer; a [`ShardIndexSearcher`] matches the
//! documents of its own node's reader but scores them under statistics
//! summed over every node ("mock distributed scoring"), and its searches
//! fan out to every node at the versions it was created with, merging the
//! per-node hits with `TopDocs.merge`. A background loop
//! ([`ShardSearching::change_indices`], Java's `ChangeIndices` thread) adds,
//! updates and deletes `LineFileDocs` documents on random nodes, commits
//! and reopens them; each reopen broadcasts the node's new version and its
//! `body`/`title` collection statistics to every other node.
//!
//! Differs:
//!
//! - The statistics overrides are [`StatisticsSource`], which the
//!   searcher of a node's reader consults (Java's subclass overriding
//!   `termStatistics`/`collectionStatistics`); `rewrite` is
//!   [`ShardIndexSearcher::rewrite`], which every local search runs first,
//!   as Java's `IndexSearcher.search` calls the overridden `rewrite`.
//! - A node's `SearcherManager` is its current near-real-time
//!   [`DirectoryReader`], reopened with `openIfChanged` against its
//!   `Mutex<IndexWriter>`; the `SearcherLifetimeManager` records
//!   [`DirectoryReader`]s, and each search opens a searcher over the
//!   recorded reader. Readers are reference counted by [`Arc`], so
//!   `tryIncRef`'s retry loop and `release` have nothing to do.
//! - The directories are the caller's (Java's `newFSDirectory` per node),
//!   the writer is a plain `IndexWriter` with the caller's analyzer (Java's
//!   `MockAnalyzer` with a random maximum token length), and nothing is
//!   printed (`VERBOSE`).
//! - Java's `SearcherExpiredException` is [`ShardError::Expired`]; every
//!   other failure is [`ShardError::Failed`] with its message.
//! - Sorted searches run [`IndexSearcher::search_sorted`] and merge with
//!   [`top_docs::merge_field_docs`]: the hits carry no score (`NaN`), the
//!   sort's keys being what the merge compares.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use lucene_analysis::Analyzer;
use lucene_index::buffered_updates::Term as IndexTerm;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::collector::ScoreDoc;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms, StatisticsSource};
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::query_visitor::{extract_terms, Term};
use lucene_search::reference_manager::{PruneByAge, SearcherLifetimeManager};
use lucene_search::similarities::{CollectionStatistics, TermStatistics};
use lucene_search::top_docs::{self, ShardFieldDoc, ShardTopFieldDocs, TieBreaker, TopDocs};
use lucene_search::top_field::SortField;
use lucene_store::Directory;
use lucene_util::java_random::JavaRandom;

use crate::util::line_file_docs::LineFileDocs;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

/// The fields whose collection statistics every reopen broadcasts.
pub const FIELDS_TO_SHARE: [&str; 2] = ["body", "title"];

/// The fields a node's searchers load norms for.
const NORM_FIELDS: [&str; 2] = ["body", "titleTokenized"];

/// A failed shard operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardError {
    /// `SearcherExpiredException`: the node no longer holds the searcher of
    /// that version (its lease expired).
    Expired { node: usize, version: i64 },
    /// Anything else, as its message.
    Failed(String),
}

impl fmt::Display for ShardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expired { node, version } => write!(f, "nodeID={node} version={version}"),
            Self::Failed(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ShardError {}

/// Every failure but an expired lease, as its message.
fn failed(e: impl fmt::Display) -> ShardError {
    ShardError::Failed(e.to_string())
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// The result type of shard operations.
pub type Result<T> = std::result::Result<T, ShardError>;

/// Runs `f` over a plain searcher of `reader` (no statistics override),
/// with norms for the text fields.
///
/// # Errors
/// Whatever opening the reader's segments, or `f`, reports.
pub fn with_searcher<R>(
    reader: &DirectoryReader,
    f: impl FnOnce(&mut IndexSearcher<'_, '_>) -> Result<R>,
) -> Result<R> {
    let opened = reader.open_segments().map_err(failed)?;
    let segments = opened.as_open_segments();
    let fields: Vec<String> = NORM_FIELDS.iter().map(|f| (*f).to_string()).collect();
    let owned = reader.field_norms_by_field(&fields).map_err(failed)?;
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
    let mut searcher = IndexSearcher::new(&segments, &norms).map_err(failed)?;
    f(&mut searcher)
}

/// Runs `f` over one searcher of every reader in `readers`, in order (the
/// test's `new IndexSearcher(new MultiReader(subs))`), with the segment
/// readers its sorted searches take; `bases[i]` is `readers[i]`'s first
/// document.
///
/// # Errors
/// Whatever opening the readers' segments, or `f`, reports.
pub fn with_multi_searcher<R>(
    readers: &[Arc<DirectoryReader>],
    f: impl FnOnce(
        &IndexSearcher<'_, '_>,
        &[lucene_search::directory_reader::SegmentReader],
        &[i32],
    ) -> Result<R>,
) -> Result<R> {
    let fields: Vec<String> = NORM_FIELDS.iter().map(|f| (*f).to_string()).collect();
    let mut opened = Vec::with_capacity(readers.len());
    let mut owned = Vec::with_capacity(readers.len());
    for r in readers {
        opened.push(r.open_segments().map_err(failed)?);
        owned.push(r.field_norms_by_field(&fields).map_err(failed)?);
    }
    let mut segments = Vec::new();
    let mut leaves = Vec::new();
    let mut bases = Vec::with_capacity(readers.len());
    let mut base = 0i32;
    for (r, o) in readers.iter().zip(&opened) {
        bases.push(base);
        for mut s in o.as_open_segments() {
            s.doc_base += base;
            segments.push(s);
        }
        for s in r.segment_readers() {
            let mut s = s.clone();
            s.doc_base += base;
            leaves.push(s);
        }
        base += r.max_doc();
    }
    let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().flatten().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).map_err(failed)?;
    f(&searcher, &leaves, &bases)
}

/// The simulated cluster: every node, and the age past which a node prunes
/// its recorded searchers.
pub struct ShardSearching<'d> {
    pub nodes: Vec<NodeState<'d>>,
    pub max_searcher_age_seconds: f64,
}

/// `NodeState`: one simulated shard node.
pub struct NodeState<'d> {
    pub my_node_id: usize,
    pub writer: Mutex<IndexWriter<'d>>,
    /// `searchers`: every recorded reader, by version.
    pub searchers: SearcherLifetimeManager<DirectoryReader>,
    /// `mgr`: the current near-real-time reader.
    mgr: Mutex<Arc<DirectoryReader>>,
    /// `currentNodeVersions` and `currentShardSearcher`'s reader.
    current: Mutex<(Vec<i64>, Arc<DirectoryReader>)>,
    collection_stats_cache: Mutex<HashMap<(usize, i64, String), CollectionStatistics>>,
    term_stats_cache: Mutex<HashMap<(usize, i64, Term), TermStatistics>>,
}

impl<'d> NodeState<'d> {
    /// `new NodeState(random, nodeID, numNodes)`: a fresh index in `dir`
    /// (`OpenMode.CREATE`) and its first reader; the versions all `0` until
    /// [`ShardSearching::start`] broadcasts.
    ///
    /// # Errors
    /// Whatever opening the writer or its reader reports.
    pub fn new(
        dir: &'d dyn Directory,
        analyzer: Arc<Analyzer>,
        node_id: usize,
        num_nodes: usize,
    ) -> Result<Self> {
        let mut writer =
            IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).map_err(failed)?;
        writer.set_analyzer(Some(analyzer));
        let writer = Mutex::new(writer);
        let reader = Arc::new(DirectoryReader::open_from_writer(&writer).map_err(failed)?);
        Ok(Self {
            my_node_id: node_id,
            writer,
            searchers: SearcherLifetimeManager::new(),
            mgr: Mutex::new(Arc::clone(&reader)),
            current: Mutex::new((vec![0; num_nodes], reader)),
            collection_stats_cache: Mutex::new(HashMap::new()),
            term_stats_cache: Mutex::new(HashMap::new()),
        })
    }

    /// The current reader (`mgr.acquire()`).
    pub fn current_reader(&self) -> Arc<DirectoryReader> {
        Arc::clone(&lock(&self.mgr))
    }

    /// `updateNodeVersion(nodeID, version)`: a new current shard searcher,
    /// over the current reader, at the versions now known.
    fn update_node_version(&self, node_id: usize, version: i64) {
        let reader = self.current_reader();
        let mut current = lock(&self.current);
        current.0[node_id] = version;
        current.1 = reader;
    }
}

impl<'d> ShardSearching<'d> {
    /// `start(numNodes, runTimeSec, maxSearcherAgeSeconds)`: a node per
    /// directory, each one's first reader recorded and broadcast. The
    /// caller runs [`Self::change_indices`] (Java's `ChangeIndices`
    /// thread) alongside its searches.
    ///
    /// # Errors
    /// Whatever opening a node reports.
    pub fn start(
        dirs: &'d [&'d dyn Directory],
        analyzer: &Arc<Analyzer>,
        max_searcher_age_seconds: f64,
    ) -> Result<Self> {
        let nodes = dirs
            .iter()
            .enumerate()
            .map(|(id, dir)| NodeState::new(*dir, Arc::clone(analyzer), id, dirs.len()))
            .collect::<Result<Vec<_>>>()?;
        let this = Self {
            nodes,
            max_searcher_age_seconds,
        };
        let mut versions = Vec::with_capacity(this.nodes.len());
        for node in &this.nodes {
            versions.push(
                node.searchers
                    .record(&node.current_reader())
                    .map_err(failed)?,
            );
        }
        for (id, node) in this.nodes.iter().enumerate() {
            let reader = node.current_reader();
            debug_assert_eq!(node.searchers.record(&reader).ok(), Some(versions[id]));
            this.broadcast_node_reopen(id, versions[id], &reader)?;
        }
        Ok(this)
    }

    /// `broadcastNodeReopen`: the node's collection statistics for the
    /// shared fields into every other node's cache, then its new version to
    /// every node.
    fn broadcast_node_reopen(
        &self,
        node_id: usize,
        version: i64,
        reader: &DirectoryReader,
    ) -> Result<()> {
        for field in FIELDS_TO_SHARE {
            let stats = with_searcher(reader, |s| s.collection_statistics(field).map_err(failed))?;
            if let Some(stats) = stats {
                for node in &self.nodes {
                    // Don't put my own collection stats into the cache; we
                    // pull locally.
                    if node.my_node_id != node_id {
                        lock(&node.collection_stats_cache)
                            .insert((node_id, version, field.to_string()), stats);
                    }
                }
            }
        }
        for node in &self.nodes {
            node.update_node_version(node_id, version);
        }
        Ok(())
    }

    /// `acquire()`: the node's current (fresh) shard searcher.
    pub fn acquire(&self, node_id: usize) -> ShardIndexSearcher<'_, 'd> {
        let current = lock(&self.nodes[node_id].current);
        ShardIndexSearcher {
            base: self,
            node_versions: current.0.clone(),
            my_node_id: node_id,
            reader: Arc::clone(&current.1),
        }
    }

    /// `acquire(nodeVersions)`: a shard searcher over the node's reader of
    /// `node_versions[node_id]`.
    ///
    /// # Errors
    /// [`ShardError::Expired`] when the node no longer holds that version.
    pub fn acquire_versions(
        &self,
        node_id: usize,
        node_versions: &[i64],
    ) -> Result<ShardIndexSearcher<'_, 'd>> {
        let version = node_versions[node_id];
        let reader = self.nodes[node_id]
            .searchers
            .acquire(version)
            .map_err(failed)?
            .ok_or(ShardError::Expired {
                node: node_id,
                version,
            })?;
        Ok(ShardIndexSearcher {
            base: self,
            node_versions: node_versions.to_vec(),
            my_node_id: node_id,
            reader,
        })
    }

    /// `searchNode`: the node's hits at `node_versions` (Java's mock of
    /// sending the query over the wire).
    ///
    /// # Errors
    /// [`ShardError::Expired`] when the node, or a node whose statistics it
    /// needs, no longer holds the version asked for.
    pub fn search_node(
        &self,
        node_id: usize,
        node_versions: &[i64],
        q: &BooleanQuery,
        num_hits: usize,
        after: Option<ScoreDoc>,
    ) -> Result<TopDocs> {
        let s = self.acquire_versions(node_id, node_versions)?;
        match after {
            Some(after) => s.local_search_after(after, q, num_hits),
            None => s.local_search(q, num_hits),
        }
    }

    /// `searchNode` with a sort.
    ///
    /// # Errors
    /// As [`Self::search_node`].
    pub fn search_node_sorted(
        &self,
        node_id: usize,
        node_versions: &[i64],
        q: &BooleanQuery,
        num_hits: usize,
        sort: &[SortField],
    ) -> Result<ShardTopFieldDocs> {
        self.acquire_versions(node_id, node_versions)?
            .local_search_sorted(q, num_hits, sort)
    }

    /// `getNodeTermStats`: the statistics of each of `terms` the node's
    /// reader of `version` holds.
    ///
    /// # Errors
    /// [`ShardError::Expired`] when the node no longer holds that version.
    fn get_node_term_stats(
        &self,
        terms: &BTreeSet<Term>,
        node_id: usize,
        version: i64,
    ) -> Result<HashMap<Term, TermStatistics>> {
        let reader = self.nodes[node_id]
            .searchers
            .acquire(version)
            .map_err(failed)?
            .ok_or(ShardError::Expired {
                node: node_id,
                version,
            })?;
        with_searcher(&reader, |s| {
            let mut stats = HashMap::new();
            for t in terms {
                if let Some(ts) = s.term_statistics(&t.field, &t.bytes).map_err(failed)? {
                    stats.insert(t.clone(), ts);
                }
            }
            Ok(stats)
        })
    }

    /// `NodeState.reopen()`: refreshes the node's reader; a new one is
    /// recorded, the node's old searchers pruned by age, and the new
    /// version broadcast.
    ///
    /// # Errors
    /// Whatever reopening reports.
    pub fn reopen(&self, node_id: usize) -> Result<()> {
        let node = &self.nodes[node_id];
        let after = {
            let mut mgr = lock(&node.mgr);
            match mgr.open_if_changed_nrt(&node.writer).map_err(failed)? {
                Some(next) => {
                    let next = Arc::new(next);
                    *mgr = Arc::clone(&next);
                    next
                }
                None => return Ok(()),
            }
        };
        let version = node.searchers.record(&after).map_err(failed)?;
        node.searchers
            .prune(&PruneByAge::new(self.max_searcher_age_seconds).map_err(failed)?);
        self.broadcast_node_reopen(node_id, version, &after)
    }

    /// `ChangeIndices.run()`: until `end`, adds a document to a random
    /// node, or updates or deletes one by `docid`, committing and reopening
    /// a random node one time in 17 each.
    ///
    /// # Errors
    /// Whatever writing or reopening reports.
    pub fn change_indices(
        &self,
        random: &mut JavaRandom,
        docs: &mut LineFileDocs,
        end: Instant,
    ) -> Result<()> {
        let mut num_docs = 0i32;
        while Instant::now() < end {
            let what = random.next_int_bounded(3);
            let node = &self.nodes[random.next_int_bounded(self.nodes.len() as i32) as usize];
            if num_docs == 0 || what == 0 {
                let doc = docs.next_doc().map_err(failed)?.to_document();
                lock(&node.writer)
                    .add_fields_document(&doc)
                    .map_err(failed)?;
                num_docs += 1;
            } else if what == 1 {
                let id = random.next_int_bounded(num_docs).to_string();
                let doc = docs.next_doc().map_err(failed)?.to_document();
                lock(&node.writer)
                    .update_fields_documents(IndexTerm::new("docid", id.as_str()), &[doc])
                    .map_err(failed)?;
                num_docs += 1;
            } else {
                let id = random.next_int_bounded(num_docs).to_string();
                lock(&node.writer)
                    .delete_documents_by_term(&[IndexTerm::new("docid", id.as_str())])
                    .map_err(failed)?;
            }
            if random.next_int_bounded(17) == 12 {
                lock(&node.writer).commit().map_err(failed)?;
            }
            if random.next_int_bounded(17) == 12 {
                let n = random.next_int_bounded(self.nodes.len() as i32) as usize;
                self.reopen(n)?;
            }
        }
        Ok(())
    }
}

/// `ShardIndexSearcher`: matches the documents of one node's reader, scores
/// them under every node's statistics at `node_versions`.
pub struct ShardIndexSearcher<'b, 'd> {
    base: &'b ShardSearching<'d>,
    /// The version of every node's searcher this one searches.
    pub node_versions: Vec<i64>,
    pub my_node_id: usize,
    reader: Arc<DirectoryReader>,
}

impl<'d> ShardIndexSearcher<'_, 'd> {
    /// `getIndexReader()`.
    pub fn reader(&self) -> &Arc<DirectoryReader> {
        &self.reader
    }

    fn node(&self) -> &NodeState<'d> {
        &self.base.nodes[self.my_node_id]
    }

    /// Runs `f` over a searcher of this node's reader that scores under
    /// this searcher's statistics.
    fn with_local<R>(&self, f: impl FnOnce(&IndexSearcher<'_, '_>) -> Result<R>) -> Result<R> {
        let opened = self.reader.open_segments().map_err(failed)?;
        let segments = opened.as_open_segments();
        let fields: Vec<String> = NORM_FIELDS.iter().map(|f| (*f).to_string()).collect();
        let owned = self.reader.field_norms_by_field(&fields).map_err(failed)?;
        let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
        let mut searcher = IndexSearcher::new(&segments, &norms).map_err(failed)?;
        searcher.set_statistics(Some(self));
        f(&searcher)
    }

    /// `rewrite(original)`: fetches, in one request per other node, the
    /// statistics of every term of `q` that node's cache lacks at its
    /// version.
    ///
    /// # Errors
    /// [`ShardError::Expired`] when a node no longer holds its version.
    pub fn rewrite(&self, q: &BooleanQuery) -> Result<()> {
        let terms = extract_terms(&Clause::Boolean(Box::new(q.clone())));
        for (node_id, &version) in self.node_versions.iter().enumerate() {
            if node_id == self.my_node_id {
                continue;
            }
            let missing: BTreeSet<Term> = {
                let cache = lock(&self.node().term_stats_cache);
                terms
                    .iter()
                    .filter(|t| !cache.contains_key(&(node_id, version, (*t).clone())))
                    .cloned()
                    .collect()
            };
            if !missing.is_empty() {
                let stats = self.base.get_node_term_stats(&missing, node_id, version)?;
                let mut cache = lock(&self.node().term_stats_cache);
                for (term, s) in stats {
                    cache.insert((node_id, version, term), s);
                }
            }
        }
        Ok(())
    }

    /// `localSearch(query, numHits)`: `super.search` over this node.
    ///
    /// # Errors
    /// As [`Self::rewrite`], or what searching reports.
    pub fn local_search(&self, q: &BooleanQuery, num_hits: usize) -> Result<TopDocs> {
        self.rewrite(q)?;
        self.with_local(|s| s.search(q, num_hits).map_err(failed))
    }

    /// `localSearchAfter(after, query, numHits)`.
    ///
    /// # Errors
    /// As [`Self::local_search`].
    pub fn local_search_after(
        &self,
        after: ScoreDoc,
        q: &BooleanQuery,
        num_hits: usize,
    ) -> Result<TopDocs> {
        self.rewrite(q)?;
        self.with_local(|s| s.search_after(after, q, num_hits).map_err(failed))
    }

    /// `localSearch(query, numHits, sort)`, its hits marked with this
    /// node's shard index.
    ///
    /// # Errors
    /// As [`Self::local_search`].
    pub fn local_search_sorted(
        &self,
        q: &BooleanQuery,
        num_hits: usize,
        sort: &[SortField],
    ) -> Result<ShardTopFieldDocs> {
        self.rewrite(q)?;
        let td = self.with_local(|s| {
            s.search_sorted(self.reader.segment_readers(), q, num_hits, sort, None)
                .map_err(failed)
        })?;
        Ok(ShardTopFieldDocs {
            total_hits: td.total,
            hits: td
                .hits
                .into_iter()
                .map(|fields| ShardFieldDoc {
                    fields,
                    score: f32::NAN,
                    shard_index: self.my_node_id as i32,
                })
                .collect(),
        })
    }

    /// Every node's hits, each marked with its node as the shard index.
    fn per_node(&self, mut search: impl FnMut(usize) -> Result<TopDocs>) -> Result<Vec<TopDocs>> {
        (0..self.node_versions.len())
            .map(|node_id| {
                let mut hits = search(node_id)?;
                for h in &mut hits.score_docs {
                    h.shard_index = node_id as i32;
                }
                Ok(hits)
            })
            .collect()
    }

    /// `search(query, numHits)`: every node's top hits, merged.
    ///
    /// # Errors
    /// As [`ShardSearching::search_node`].
    pub fn search(&self, q: &BooleanQuery, num_hits: usize) -> Result<TopDocs> {
        let shard_hits = self.per_node(|node_id| {
            if node_id == self.my_node_id {
                self.local_search(q, num_hits)
            } else {
                self.base
                    .search_node(node_id, &self.node_versions, q, num_hits, None)
            }
        })?;
        top_docs::merge(0, num_hits, &shard_hits, &TieBreaker::Default).map_err(failed)
    }

    /// `searchAfter(after, query, numHits)`: hits are merged by score, then
    /// shard index, then document, so a node before `after`'s collects only
    /// worse scores, `after`'s own node those after `after`, and a later
    /// node every document with `after`'s score or worse.
    ///
    /// # Errors
    /// As [`ShardSearching::search_node`].
    pub fn search_after(
        &self,
        after: top_docs::ShardScoreDoc,
        q: &BooleanQuery,
        num_hits: usize,
    ) -> Result<TopDocs> {
        let shard_hits = self.per_node(|node_id| {
            let doc = match (node_id as i32).cmp(&after.shard_index) {
                std::cmp::Ordering::Less => {
                    // Setting the doc to maxDoc-1 tells the collector no more
                    // documents with that score should be collected.
                    let s = self.base.acquire_versions(node_id, &self.node_versions)?;
                    s.reader.max_doc() - 1
                }
                std::cmp::Ordering::Equal => after.doc,
                std::cmp::Ordering::Greater => -1,
            };
            let shard_after = ScoreDoc {
                doc_id: doc,
                score: after.score,
            };
            if node_id == self.my_node_id {
                self.local_search_after(shard_after, q, num_hits)
            } else {
                self.base
                    .search_node(node_id, &self.node_versions, q, num_hits, Some(shard_after))
            }
        })?;
        top_docs::merge(0, num_hits, &shard_hits, &TieBreaker::Default).map_err(failed)
    }

    /// `search(query, numHits, sort)`: every node's top hits by `sort`,
    /// merged.
    ///
    /// # Errors
    /// As [`ShardSearching::search_node`].
    pub fn search_sorted(
        &self,
        q: &BooleanQuery,
        num_hits: usize,
        sort: &[SortField],
    ) -> Result<ShardTopFieldDocs> {
        let shard_hits = (0..self.node_versions.len())
            .map(|node_id| {
                if node_id == self.my_node_id {
                    self.local_search_sorted(q, num_hits, sort)
                } else {
                    self.base
                        .search_node_sorted(node_id, &self.node_versions, q, num_hits, sort)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        top_docs::merge_field_docs(sort, 0, num_hits, &shard_hits, &TieBreaker::Default)
            .map_err(failed)
    }
}

impl StatisticsSource for ShardIndexSearcher<'_, '_> {
    /// `termStatistics`: this node's own plus each other node's cached
    /// statistics at its version (a node whose cache lacks the term does
    /// not hold it).
    fn term_statistics(
        &self,
        field: &str,
        term: &[u8],
        local: TermStatistics,
    ) -> lucene_search::Result<TermStatistics> {
        let key = Term {
            field: field.to_string(),
            bytes: term.to_vec(),
        };
        let cache = lock(&self.node().term_stats_cache);
        let mut sum = TermStatistics {
            doc_freq: 0,
            total_term_freq: 0,
        };
        for (node_id, &version) in self.node_versions.iter().enumerate() {
            let sub = if node_id == self.my_node_id {
                local
            } else {
                match cache.get(&(node_id, version, key.clone())) {
                    Some(s) => *s,
                    None => continue,
                }
            };
            sum.doc_freq += sub.doc_freq;
            sum.total_term_freq += sub.total_term_freq;
        }
        debug_assert!(sum.doc_freq > 0);
        Ok(sum)
    }

    /// `collectionStatistics`: this node's own plus each other node's
    /// broadcast statistics at its version; `None` when no node has the
    /// field.
    fn collection_statistics(
        &self,
        field: &str,
        local: Option<CollectionStatistics>,
    ) -> lucene_search::Result<Option<CollectionStatistics>> {
        let cache = lock(&self.node().collection_stats_cache);
        let mut sum = CollectionStatistics {
            max_doc: 0,
            doc_count: 0,
            sum_total_term_freq: 0,
            sum_doc_freq: 0,
        };
        for (node_id, &version) in self.node_versions.iter().enumerate() {
            let sub = if node_id == self.my_node_id {
                local
            } else {
                cache.get(&(node_id, version, field.to_string())).copied()
            };
            // Field not in the node at all.
            let Some(sub) = sub else { continue };
            sum.doc_count += sub.doc_count;
            sum.sum_total_term_freq += sub.sum_total_term_freq;
            sum.sum_doc_freq += sub.sum_doc_freq;
            debug_assert!(sub.max_doc >= 0);
            sum.max_doc += sub.max_doc;
        }
        Ok((sum.max_doc != 0).then_some(sum))
    }
}
