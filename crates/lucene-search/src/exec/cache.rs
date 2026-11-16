//! `LRUQueryCache` for the scorer tree: a clause built without scores (a
//! `FILTER` or `MUST_NOT` clause, or any clause of a count) that keeps being
//! used is computed once per segment into a doc-id set and iterated from
//! there afterwards.
//!
//! What Lucene does, and this follows: `IndexSearcher.createWeight` wraps
//! every weight created for `COMPLETE_NO_SCORES` in a `CachingWrapperWeight`;
//! `UsageTrackingQueryCachingPolicy` decides from how often a query was seen
//! (costly queries twice, compound ones four times, others five; term,
//! match-all, match-none and empty ones never); a segment under 10,000
//! documents is never cached; the cached set is the segment core's matches,
//! before deletions, so it survives a refresh that only changes live docs; a
//! dense set is a bitset and a sparse one a sorted doc list
//! (`LRUQueryCache.cacheImpl`'s 1% rule).
//!
//! Deviations: the cache belongs to the segment (it lives in
//! [`crate::directory_reader::SegmentReader`] and goes away with it, where
//! Lucene keys one node-wide cache on the core's cache helper), so its
//! bounds are per segment ([`MAX_ENTRIES`], [`MAX_BYTES`]) rather than
//! node-wide; the usage history is per segment too, which counts the same
//! uses since a query visits each segment once; `skipCacheFactor` (don't
//! cache a clause costing ten times its conjunction's lead) applies only to
//! an `IndexOrDocValuesQuery` clause, the one kind built here with its
//! conjunction's lead cost, and the "at least half the average leaf" size
//! rule is not ported, so a clause can be cached where Lucene would have run
//! it uncached. None of these changes a hit or a score.

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use lucene_util::fixed_bit_set::FixedBitSet;

use super::{exact_next, BoxScorer, Scorer, NO_MORE_DOCS};
use crate::query::Clause;
use crate::query_cache::{CachingCost, QueryCachingPolicy, UsageTrackingPolicy};
use crate::Result;

/// Cached queries per segment (OpenSearch's node-wide
/// `indices.queries.cache.count` is 10,000; Lucene's own default 1,000).
/// [`MAX_BYTES`] is the bound that matters: 64 entries let a workload of a
/// few hundred distinct filters evict each one before its next use, where
/// OpenSearch's cache would still hold it.
pub(crate) const MAX_ENTRIES: usize = 1_000;
/// Cached bytes per segment (Lucene's node-wide default is 32 MB).
pub(crate) const MAX_BYTES: usize = 16 << 20;
/// `LRUQueryCache`'s `MinSegmentSizePredicate` floor.
pub(crate) const MIN_SEGMENT_SIZE: i32 = 10_000;
/// `LRUQueryCache`'s default `skipCacheFactor`: a clause costing more than
/// this many times the lead cost of the conjunction it is in runs uncached.
pub(crate) const SKIP_CACHE_FACTOR: f32 = 10.0;

/// `cost / skipCacheFactor > leadCost`, in Java's `float` arithmetic.
pub(crate) fn skip_cache(cost: i64, lead_cost: i64) -> bool {
    cost as f32 / SKIP_CACHE_FACTOR > lead_cost as f32
}

/// A segment's matches for one query, before deletions.
pub(crate) enum CachedSet {
    Bits { bits: FixedBitSet, cardinality: i64 },
    Docs(Vec<i32>),
}

impl Drop for CachedSet {
    /// A set's bits go back to [`crate::bit_set_pool`] for the next query's
    /// set of the same segment, whether it was built for one query or
    /// evicted from the cache.
    fn drop(&mut self) {
        if let CachedSet::Bits { bits, .. } = self {
            crate::bit_set_pool::give(std::mem::replace(bits, FixedBitSet::new(0)));
        }
    }
}

impl CachedSet {
    /// How many documents it holds: its iterator's cost.
    #[cfg_attr(not(any(test, feature = "asserting")), allow(dead_code))]
    pub(crate) fn cardinality(&self) -> i64 {
        match self {
            CachedSet::Bits { cardinality, .. } => *cardinality,
            CachedSet::Docs(docs) => docs.len() as i64,
        }
    }

    /// How many of its documents `live_docs` keeps (all, without deletions).
    pub(crate) fn live_count(&self, live_docs: Option<&FixedBitSet>) -> u64 {
        let n = match (self, live_docs) {
            (CachedSet::Bits { cardinality, .. }, None) => {
                return u64::try_from(*cardinality).unwrap_or(0)
            }
            (CachedSet::Bits { bits, .. }, Some(live)) => {
                FixedBitSet::intersection_count(bits, live)
            }
            (CachedSet::Docs(docs), None) => docs.len(),
            (CachedSet::Docs(docs), Some(live)) => {
                docs.iter().filter(|&&d| live.get_doc(d)).count()
            }
        };
        n as u64
    }

    /// Whether the set holds no document (a cached empty answer).
    fn is_empty(&self) -> bool {
        match self {
            CachedSet::Bits { cardinality, .. } => *cardinality == 0,
            CachedSet::Docs(docs) => docs.is_empty(),
        }
    }

    fn ram_bytes(&self) -> usize {
        match self {
            CachedSet::Bits { bits, .. } => bits.words().len() * 8,
            CachedSet::Docs(docs) => docs.len() * 4,
        }
    }
}

struct Entry {
    set: Arc<CachedSet>,
    last_used: u64,
}

struct Inner {
    policy: UsageTrackingPolicy,
    entries: HashMap<Arc<str>, Entry>,
    bytes: usize,
    /// Whether this cache is in its node's list (once it first held a set).
    registered: bool,
    node: &'static Node,
}

impl Default for Inner {
    fn default() -> Self {
        Inner {
            policy: UsageTrackingPolicy::default(),
            entries: HashMap::new(),
            bytes: 0,
            registered: false,
            node: &NODE,
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.node.bytes.fetch_sub(self.bytes, Relaxed);
    }
}

impl Inner {
    /// Drops the least recently used entry, returning its bytes (0 when
    /// there is none).
    fn evict_oldest(&mut self) -> usize {
        let Some(oldest) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| k.clone())
        else {
            return 0;
        };
        let Some(e) = self.entries.remove(&oldest) else {
            return 0;
        };
        let freed = e.set.ram_bytes();
        self.bytes -= freed;
        self.node.bytes.fetch_sub(freed, Relaxed);
        self.node.evictions.fetch_add(1, Relaxed);
        freed
    }

    fn oldest(&self) -> Option<u64> {
        self.entries.values().map(|e| e.last_used).min()
    }
}

/// One segment's query cache. Shared by every search of the segment, hence
/// the lock; held only to look up or insert, never while a set is built.
/// Its sets also count against the node's budget ([`set_node_budget`]),
/// which evicts the least recently used sets of any segment.
#[derive(Default)]
pub struct SegmentQueryCache {
    inner: Arc<Mutex<Inner>>,
    columns: Mutex<Columns>,
}

use std::sync::atomic::Ordering::Relaxed;

/// Cached bytes across every segment of the process, by default (the host
/// sets its own: the plugin passes OpenSearch's `indices.queries.cache.size`).
pub const DEFAULT_NODE_BYTES: usize = 256 << 20;
/// The caches sharing one budget: every segment's, the process's one.
pub(crate) struct Node {
    /// Cached set bytes held across its segments.
    bytes: std::sync::atomic::AtomicUsize,
    /// The bound on `bytes`.
    budget: std::sync::atomic::AtomicUsize,
    /// Sets evicted, for stats.
    evictions: std::sync::atomic::AtomicU64,
    /// Every segment cache that has held a set.
    caches: Mutex<Vec<std::sync::Weak<Mutex<Inner>>>>,
}

impl Node {
    pub(crate) const fn new(budget: usize) -> Self {
        Node {
            bytes: std::sync::atomic::AtomicUsize::new(0),
            budget: std::sync::atomic::AtomicUsize::new(budget),
            evictions: std::sync::atomic::AtomicU64::new(0),
            caches: Mutex::new(Vec::new()),
        }
    }
}

static NODE: Node = Node::new(DEFAULT_NODE_BYTES);
/// One clock for every segment, so the least recently used set is found
/// across segments.
static CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Sets the bytes the query caches of every segment may hold together; past
/// them the least recently used sets go, whichever segment holds them.
pub fn set_node_budget(bytes: usize) {
    NODE.budget.store(bytes, Relaxed);
    evict_node(&NODE);
}

/// `(bytes held, budget, sets evicted)` of the query caches, process-wide.
pub fn node_stats() -> (usize, usize, u64) {
    (
        NODE.bytes.load(Relaxed),
        NODE.budget.load(Relaxed),
        NODE.evictions.load(Relaxed),
    )
}

/// Evicts across `node`'s segments, least recently used first, until it is
/// within its budget.
fn evict_node(node: &Node) {
    let budget = node.budget.load(Relaxed);
    if node.bytes.load(Relaxed) <= budget {
        return;
    }
    let live: Vec<Arc<Mutex<Inner>>> = {
        let mut all = lock(&node.caches);
        all.retain(|w| w.strong_count() > 0);
        all.iter().filter_map(std::sync::Weak::upgrade).collect()
    };
    // Each cache's oldest entry, oldest first; a cache gives up entries
    // until its oldest is newer than the next cache's.
    let mut order: Vec<(u64, &Arc<Mutex<Inner>>)> = live
        .iter()
        .filter_map(|c| lock(c).oldest().map(|o| (o, c)))
        .collect();
    order.sort_by_key(|(o, _)| *o);
    for (i, (_, c)) in order.iter().enumerate() {
        let next = order.get(i + 1).map_or(u64::MAX, |(o, _)| *o);
        let mut inner = lock(c);
        while node.bytes.load(Relaxed) > budget && inner.oldest().is_some_and(|o| o <= next) {
            if inner.evict_oldest() == 0 && inner.entries.is_empty() {
                break;
            }
        }
        if node.bytes.load(Relaxed) <= budget {
            return;
        }
    }
}

/// A sort column decoded once for the segment: every document's value, for
/// a column a per-document read would decode (a sparse, block-encoded or
/// multi-valued one). Lucene has no such cache -- its comparators read
/// doc values per document on every search -- and none of this changes a
/// value: it is the same read, done once.
pub(crate) enum SortColumn {
    /// Ordinals, `-1` for a document without one.
    Ords(Vec<i32>),
    /// A field's terms read back from its postings (the `_id` field sorts as OpenSearch's
    /// uninverted fielddata does): each document's term ordinal (`-1` for none), and the
    /// terms in order, `bytes[offsets[o]..offsets[o + 1]]` being term `o`.
    Terms {
        ords: Vec<i32>,
        offsets: Vec<u32>,
        bytes: Vec<u8>,
    },
    /// Values, and which documents have one.
    Longs { values: Vec<i64>, has: FixedBitSet },
    /// Every document's values (a multi-valued `SORTED_NUMERIC` column, the
    /// geo distance sorts' points): document `d`'s are
    /// `values[starts[d]..starts[d + 1]]`.
    Multi {
        starts: Vec<usize>,
        values: Vec<i64>,
    },
    /// A collapse field as group ids (a grouped KNN search's groups): each
    /// document's, `-1` for one without a value; a keyword field's are its
    /// ordinals, a numeric field's index `values`. `groups` counts the
    /// distinct ids, the missing one included.
    Groups {
        ids: Vec<i32>,
        values: Vec<i64>,
        groups: usize,
        /// Group id `g`'s documents are `members[starts[g + 1]..starts[g + 2]]`
        /// (the missing group, `-1`, first), in document order.
        starts: Vec<u32>,
        members: Vec<i32>,
        /// A numeric field's ids, by their values ascending.
        by_value: Vec<i32>,
    },
}

impl SortColumn {
    fn ram_bytes(&self) -> usize {
        match self {
            SortColumn::Ords(o) => o.len() * 4,
            SortColumn::Terms {
                ords,
                offsets,
                bytes,
            } => ords.len() * 4 + offsets.len() * 4 + bytes.len(),
            SortColumn::Longs { values, has } => values.len() * 8 + has.words().len() * 8,
            SortColumn::Multi { starts, values } => starts.len() * 8 + values.len() * 8,
            SortColumn::Groups {
                ids,
                values,
                starts,
                members,
                by_value,
                ..
            } => {
                ids.len() * 4
                    + values.len() * 8
                    + starts.len() * 4
                    + members.len() * 4
                    + by_value.len() * 4
            }
        }
    }
}

/// Sort columns: uses seen, and the columns built.
#[derive(Default)]
struct Columns {
    uses: HashMap<String, u32>,
    /// Terms columns that did not fit the budget ([`SegmentQueryCache::terms_column`]).
    too_big: std::collections::HashSet<String>,
    built: HashMap<String, Arc<SortColumn>>,
    bytes: usize,
}

/// A lock that a panicked holder left poisoned is still the data.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A sort column is decoded on its second use in a segment.
const COLUMN_MIN_USES: u32 = 2;
/// Decoded sort columns per segment.
pub(crate) const MAX_COLUMN_BYTES: usize = 32 << 20;
/// A terms column per segment ([`SegmentQueryCache::terms_column`]).
pub(crate) const MAX_TERMS_COLUMN_BYTES: usize = 256 << 20;
/// Decoded sort columns across every segment of the process: the host (a
/// JVM) cannot see this memory, so it is bounded here.
pub(crate) const MAX_TOTAL_COLUMN_BYTES: usize = 512 << 20;
/// [`MAX_TOTAL_COLUMN_BYTES`], or an eighth of the node's off-heap bound
/// when that is less ([`lucene_util::memory::process_limit`]): a small node
/// streams its sort values from the doc values instead of holding them.
fn total_column_budget() -> usize {
    match usize::try_from(lucene_util::memory::process_limit()) {
        Ok(limit) if limit > 0 => MAX_TOTAL_COLUMN_BYTES.min(limit / 8),
        _ => MAX_TOTAL_COLUMN_BYTES,
    }
}

/// Bytes of decoded sort columns held now, process-wide.
static TOTAL_COLUMN_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Drop for Columns {
    fn drop(&mut self) {
        TOTAL_COLUMN_BYTES.fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Decoded sort-column bytes held process-wide.
pub fn sort_column_bytes() -> usize {
    TOTAL_COLUMN_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

impl std::fmt::Debug for SegmentQueryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentQueryCache").finish_non_exhaustive()
    }
}

/// Formatted clause keys a thread keeps ([`clause_key`]).
const KEY_MEMO: usize = 8;

thread_local! {
    /// The last clauses a thread keyed, by address, with their keys: a
    /// search asks every segment's cache about the same clauses.
    static KEYS: std::cell::RefCell<Vec<(usize, Clause, Arc<str>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `clause`'s cache key, its `Debug` form (`Query.equals` and `hashCode`
/// for the cache's map). Formatted once per clause a search asks about,
/// not once per segment: a clause at the same address and equal to one
/// keyed before takes that key. Equal clauses format alike but for a float
/// `0.0` against `-0.0`, which only boosts, tie breakers and constant
/// scores carry: none changes what an unscored clause matches.
fn clause_key(clause: &Clause) -> Arc<str> {
    let addr = std::ptr::from_ref(clause).addr();
    let found = KEYS.with(|k| {
        k.borrow()
            .iter()
            .find(|(a, c, _)| *a == addr && c == clause)
            .map(|(_, _, key)| Arc::clone(key))
    });
    if let Some(key) = found {
        return key;
    }
    let key: Arc<str> = format!("{clause:?}").into();
    KEYS.with(|k| {
        let mut k = k.borrow_mut();
        if k.len() >= KEY_MEMO {
            k.remove(0);
        }
        k.push((addr, clause.clone(), Arc::clone(&key)));
    });
    key
}

/// What the policy is asked about: the clause's identity and shape.
struct PolicyKey<'k> {
    key: &'k str,
    costly: bool,
    composite: bool,
    never: bool,
}

impl Hash for PolicyKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl CachingCost for PolicyKey<'_> {
    fn never_cache(&self) -> bool {
        self.never
    }
    fn is_costly(&self) -> bool {
        self.costly
    }
    fn is_composite(&self) -> bool {
        self.composite
    }
}

/// `UsageTrackingQueryCachingPolicy.shouldNeverCache`, `isCostly` and its
/// compound-query rule, for the clause kinds the tree runs.
fn shape(clause: &Clause) -> (bool, bool, bool) {
    // (costly, composite, never)
    match clause {
        Clause::Term(_) | Clause::MatchAllDocs(_) | Clause::MatchNoDocs(_) => (false, false, true),
        Clause::Boolean(b) => {
            let empty = b.must.is_empty()
                && b.filter.is_empty()
                && b.should.is_empty()
                && b.must_not.is_empty();
            (false, true, empty)
        }
        Clause::DisjunctionMax(d) => (false, true, d.disjuncts.is_empty()),
        Clause::Wildcard(_)
        | Clause::Prefix(_)
        | Clause::Regexp(_)
        | Clause::Fuzzy(_)
        | Clause::TermInSet(_)
        | Clause::PointsRange(_) => (true, false, false),
        // `isCostly`: a `MultiTermQuery`, and `isPointQuery` (a class named
        // `Point*Query`).
        Clause::Extended(e) => match &**e {
            crate::extended_query::ExtendedQuery::MultiTerm(_)
            | crate::extended_query::ExtendedQuery::PointRange(_)
            | crate::extended_query::ExtendedQuery::PointInSet(_)
            | crate::extended_query::ExtendedQuery::PointInSetIncludingScore(_) => {
                (true, false, false)
            }
            _ => (false, false, false),
        },
        _ => (false, false, false),
    }
}

impl SegmentQueryCache {
    /// The cached scorer for `clause` on a segment of `max_doc` documents:
    /// from the cache, or built by `uncached` and stored when the policy
    /// says so. `None` means run `uncached`'s scorer as is.
    pub(crate) fn scorer<'a>(
        &self,
        clause: &Clause,
        max_doc: i32,
        uncached: impl FnOnce() -> Result<Option<BoxScorer<'a>>>,
    ) -> Result<Option<CacheResult>> {
        self.scorer_if_cacheable(clause, max_doc, true, || Ok(false), uncached)
    }

    /// [`Self::scorer`] for a clause whose weight says whether it may be
    /// cached against this segment (`SegmentCacheable.isCacheable`): the use
    /// is still counted, but an uncacheable clause is neither looked up nor
    /// stored (`CachingWrapperWeight.scorerSupplier`'s short-circuit).
    ///
    /// `skip` is asked only when the clause is not cached yet and the policy
    /// would cache it: `true` runs the clause uncached this time instead
    /// (`LRUQueryCache`'s `cost / skipCacheFactor > leadCost`).
    pub(crate) fn scorer_if_cacheable<'a>(
        &self,
        clause: &Clause,
        max_doc: i32,
        cacheable: bool,
        skip: impl FnOnce() -> Result<bool>,
        uncached: impl FnOnce() -> Result<Option<BoxScorer<'a>>>,
    ) -> Result<Option<CacheResult>> {
        let (costly, composite, never) = shape(clause);
        if never || max_doc < MIN_SEGMENT_SIZE {
            return Ok(None);
        }
        let key = clause_key(clause);
        let policy_key = PolicyKey {
            key: &key,
            costly,
            composite,
            never,
        };
        let cache = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let now = CLOCK.fetch_add(1, Relaxed);
            inner.policy.on_use(&policy_key);
            if !cacheable {
                return Ok(None);
            }
            if let Some(entry) = inner.entries.get_mut(&*key) {
                entry.last_used = now;
                if entry.set.is_empty() {
                    return Ok(Some(CacheResult::Empty));
                }
                return Ok(Some(CacheResult::Hit(Arc::clone(&entry.set))));
            }
            inner.policy.should_cache(&policy_key)
        };
        if !cache || skip()? {
            return Ok(None);
        }
        // Built without the lock: another search may build the same set at
        // the same time, and the second insert simply replaces the first.
        // An empty answer is cached too (`LRUQueryCache` keeps
        // `DocIdSet.EMPTY`): a filter that matches nothing in a segment is
        // not recomputed on every use.
        let Some(mut s) = uncached()? else {
            self.insert(key, Arc::new(CachedSet::Docs(Vec::new())));
            return Ok(Some(CacheResult::Empty));
        };
        let set = collect(&mut *s, max_doc)?;
        self.insert(key, Arc::clone(&set));
        Ok(Some(CacheResult::Hit(set)))
    }

    /// The sort column `key` of a segment of `max_doc` documents: cached,
    /// or built by `build` on its second use (and handed back, uncached,
    /// when it does not fit the budget). `None` means read the column per
    /// document.
    pub(crate) fn sort_column(
        &self,
        key: &str,
        max_doc: i32,
        build: &mut dyn FnMut() -> Result<SortColumn>,
    ) -> Result<Option<Arc<SortColumn>>> {
        if max_doc < MIN_SEGMENT_SIZE {
            return Ok(None);
        }
        {
            let mut c = lock(&self.columns);
            if let Some(col) = c.built.get(key) {
                return Ok(Some(Arc::clone(col)));
            }
            let uses = c.uses.entry(key.to_string()).or_insert(0);
            *uses = uses.saturating_add(1);
            if *uses != COLUMN_MIN_USES {
                return Ok(None);
            }
        }
        // Built without the lock; tried once (on the second use exactly).
        let col = Arc::new(build()?);
        let bytes = col.ram_bytes();
        let mut c = lock(&self.columns);
        // Built but over a cap: the search still uses it, uncached.
        if c.bytes.saturating_add(bytes) > MAX_COLUMN_BYTES {
            return Ok(Some(col));
        }
        // Reserve against the process-wide cap; the segment's share goes
        // back when its cache is dropped.
        let reserved = TOTAL_COLUMN_BYTES.fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |t| t.checked_add(bytes).filter(|&n| n <= total_column_budget()),
        );
        if reserved.is_err() {
            return Ok(Some(col));
        }
        c.bytes += bytes;
        c.built.insert(key.to_string(), Arc::clone(&col));
        Ok(Some(col))
    }

    /// A field's terms column ([`SortColumn::Terms`]), which a search cannot read per document
    /// (it takes the whole terms dictionary and its postings to build): built on first use and
    /// kept, as OpenSearch keeps the fielddata it loads. `None` when it does not fit the
    /// budget (and the same for every later call, without building it again).
    pub(crate) fn terms_column(
        &self,
        key: &str,
        build: &mut dyn FnMut() -> Result<SortColumn>,
    ) -> Result<Option<Arc<SortColumn>>> {
        {
            let c = lock(&self.columns);
            if let Some(col) = c.built.get(key) {
                return Ok(Some(Arc::clone(col)));
            }
            if c.too_big.contains(key) {
                return Ok(None);
            }
        }
        let col = Arc::new(build()?);
        let bytes = col.ram_bytes();
        let mut c = lock(&self.columns);
        if let Some(col) = c.built.get(key) {
            // Built by another search meanwhile.
            return Ok(Some(Arc::clone(col)));
        }
        let reserved = bytes <= MAX_TERMS_COLUMN_BYTES
            && TOTAL_COLUMN_BYTES
                .fetch_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |t| t.checked_add(bytes).filter(|&n| n <= total_column_budget()),
                )
                .is_ok();
        if !reserved {
            c.too_big.insert(key.to_string());
            return Ok(None);
        }
        c.bytes += bytes;
        c.built.insert(key.to_string(), Arc::clone(&col));
        Ok(Some(col))
    }

    fn insert(&self, key: Arc<str>, set: Arc<CachedSet>) {
        let bytes = set.ram_bytes();
        // A set the node's budget cannot hold beside a few others is not
        // kept at all (`LRUQueryCache`'s maxRamBytesUsed / 1000 rule, on the
        // node's budget).
        let node = lock(&self.inner).node;
        if bytes.saturating_mul(16) > node.budget.load(Relaxed) {
            return;
        }
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let now = CLOCK.fetch_add(1, Relaxed);
            inner.bytes += bytes;
            node.bytes.fetch_add(bytes, Relaxed);
            if let Some(old) = inner.entries.insert(
                key,
                Entry {
                    set,
                    last_used: now,
                },
            ) {
                let freed = old.set.ram_bytes();
                inner.bytes -= freed;
                node.bytes.fetch_sub(freed, Relaxed);
            }
            // `LRUQueryCache.evictIfNecessary`: least recently used first.
            while inner.entries.len() > MAX_ENTRIES || inner.bytes > MAX_BYTES {
                if inner.evict_oldest() == 0 && inner.entries.is_empty() {
                    break;
                }
            }
            if !inner.registered {
                inner.registered = true;
                lock(&node.caches).push(Arc::downgrade(&self.inner));
            }
        }
        evict_node(node);
    }

    /// A cache whose sets count against `node` rather than the process's
    /// (tests).
    #[cfg(test)]
    pub(crate) fn in_node(node: &'static Node) -> Self {
        let cache = SegmentQueryCache::default();
        lock(&cache.inner).node = node;
        cache
    }

    /// Cached queries and their bytes, for tests.
    #[cfg(test)]
    pub(crate) fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        (inner.entries.len(), inner.bytes)
    }
}

pub(crate) enum CacheResult {
    /// The cached set, for a [`CachedScorer`].
    Hit(Arc<CachedSet>),
    /// The clause matches nothing in this segment.
    Empty,
}

/// `LRUQueryCache.cacheImpl`: every match of `s` (exact, before deletions),
/// as a bitset when at least 1% of the segment matches, a doc list otherwise.
///
/// A scorer that already iterates a set of exactly this segment's bits (a
/// point range, a geo query) hands that set over as it is when it is dense,
/// and its documents read off the words when it is not. Otherwise the
/// matches go into a list while the scorer's cost says they are sparse, and
/// into a bit set otherwise, switching to the set once the list passes 1%.
fn collect(s: &mut dyn Scorer, max_doc: i32) -> Result<Arc<CachedSet>> {
    let len = usize::try_from(max_doc).unwrap_or(0);
    let dense = |n: i64| n.saturating_mul(100) >= i64::from(max_doc);
    if s.doc_id() == -1 {
        if let Some((set, _)) = s.constant_bits() {
            if let CachedSet::Bits { bits, cardinality } = &*set {
                if bits.len() == len {
                    if dense(*cardinality) {
                        return Ok(set);
                    }
                    let mut docs = Vec::with_capacity(usize::try_from(*cardinality).unwrap_or(0));
                    bits.for_each_set_bit(|i| docs.push(i as i32));
                    return Ok(Arc::new(CachedSet::Docs(docs)));
                }
            }
        }
    }
    let mut docs: Vec<i32> = Vec::new();
    let mut bits: Option<FixedBitSet> = None;
    let mut cardinality = 0i64;
    let into_bits = |docs: &[i32]| {
        let mut b = crate::bit_set_pool::take(len).unwrap_or_else(|| FixedBitSet::new(len));
        for &d in docs {
            // FBS: every listed id passed the `i < len` check below.
            b.set(d as usize);
        }
        b
    };
    if dense(s.cost()) {
        bits = Some(into_bits(&[]));
    }
    let mut doc = exact_next(s)?;
    while doc != NO_MORE_DOCS {
        if let Ok(i) = usize::try_from(doc) {
            if i < len {
                cardinality += 1;
                match &mut bits {
                    // FBS: `i < len`, the set's length, checked just above.
                    Some(b) => b.set(i),
                    None => {
                        docs.push(doc);
                        if dense(cardinality) {
                            bits = Some(into_bits(&docs));
                            docs = Vec::new();
                        }
                    }
                }
            }
        }
        doc = exact_next(s)?;
    }
    let Some(bits) = bits else {
        return Ok(Arc::new(CachedSet::Docs(docs)));
    };
    if dense(cardinality) {
        return Ok(Arc::new(CachedSet::Bits { bits, cardinality }));
    }
    let mut docs = Vec::with_capacity(usize::try_from(cardinality).unwrap_or(0));
    bits.for_each_set_bit(|i| docs.push(i as i32));
    crate::bit_set_pool::give(bits);
    Ok(Arc::new(CachedSet::Docs(docs)))
}

/// A cached set as a scorer: matches only, every score 0 -- or, built by
/// [`CachedScorer::constant`], every score one constant (`ConstantScoreScorer`
/// over the cached iterator, without a scorer in between).
pub(crate) struct CachedScorer {
    set: Arc<CachedSet>,
    doc: i32,
    /// `Docs`: the index of `doc`.
    at: usize,
    score: f32,
    /// `ConstantScoreScorer`'s `TOP_SCORES` mode: a threshold past the
    /// constant empties the iterator.
    top_scores: bool,
    emptied: bool,
    /// The cost the set was built with, when it is not its size: a
    /// `DocIdSetBuilder` bit set's estimate.
    cost: Option<i64>,
    /// `BitSetIterator`'s `runStart` and `runEnd`: the known run of set bits
    /// around the current document, `(-1, -1)` before any, so that
    /// [`Scorer::doc_id_run_end`] resumes its scan rather than rescanning
    /// the run on every call (Lucene 10.6, branch_10_6, #16849). A `Cell`
    /// because `docIDRunEnd` takes `&self` here.
    run: Cell<(i32, i32)>,
}

/// `BitSetIterator.MAX_RUN_END_SCAN` (Lucene 10.6, branch_10_6, #16849): the
/// most bits one `docIDRunEnd` scans past the current document, so that a
/// caller consuming only a prefix of a long run does not scan all of it.
pub(crate) const MAX_RUN_END_SCAN: i32 = 1 << 16;

impl CachedScorer {
    pub(crate) fn new(set: Arc<CachedSet>) -> Self {
        Self::constant(set, 0.0, false)
    }

    /// `ConstantScoreScorer(score, scoreMode, cachedIterator)`.
    pub(crate) fn constant(set: Arc<CachedSet>, score: f32, top_scores: bool) -> Self {
        Self {
            set,
            doc: -1,
            at: 0,
            score,
            top_scores,
            emptied: false,
            cost: None,
            run: Cell::new((-1, -1)),
        }
    }

    /// The set's cost as its builder estimated it (`BitDocIdSet(bits, cost)`).
    pub(crate) fn with_cost(mut self, cost: i64) -> Self {
        self.cost = Some(cost);
        self
    }
}

impl Scorer for CachedScorer {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        if let (CachedSet::Docs(docs), false) = (&*self.set, self.emptied) {
            // `at` indexes the current document once positioned: the next
            // one is the next entry.
            let next = if self.doc == -1 {
                0
            } else {
                self.at.saturating_add(1).min(docs.len())
            };
            self.at = next;
            self.doc = docs.get(next).copied().unwrap_or(NO_MORE_DOCS);
            return Ok(self.doc);
        }
        let target = self.doc.saturating_add(1);
        self.advance(target)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.emptied {
            self.doc = NO_MORE_DOCS;
            return Ok(NO_MORE_DOCS);
        }
        self.doc = match &*self.set {
            CachedSet::Bits { bits, .. } => usize::try_from(target)
                .ok()
                .and_then(|t| bits.next_set_bit(t))
                .map_or(NO_MORE_DOCS, |d| d as i32),
            CachedSet::Docs(docs) => {
                // Galloping from the current index (callers advance
                // forward): double the step until it passes `target`, then
                // binary-search the last step.
                let rest = docs.get(self.at..).unwrap_or(&[]);
                let mut lo = 0usize;
                let mut step = 1usize;
                while step < rest.len() && rest[step] < target {
                    lo = step;
                    step = step.saturating_mul(2);
                }
                let hi = step.saturating_add(1).min(rest.len());
                let skip = lo + rest[lo..hi].partition_point(|&d| d < target);
                self.at += skip;
                docs.get(self.at).copied().unwrap_or(NO_MORE_DOCS)
            }
        };
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        if let Some(cost) = self.cost {
            return cost;
        }
        match &*self.set {
            CachedSet::Bits { cardinality, .. } => *cardinality,
            CachedSet::Docs(docs) => docs.len() as i64,
        }
    }

    fn score(&mut self) -> Result<f32> {
        Ok(self.score)
    }

    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(self.score)
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if self.top_scores && min > self.score {
            // Emptied: no later document can compete.
            self.emptied = true;
        }
        Ok(())
    }

    fn constant_bits(&self) -> Option<(Arc<CachedSet>, f32)> {
        (!self.emptied && matches!(&*self.set, CachedSet::Bits { .. }))
            .then(|| (Arc::clone(&self.set), self.score))
    }

    /// A bit set answers with one bit test, moving onto `target` only when
    /// it is set.
    #[inline]
    fn advance_exact_cheap(&mut self, target: i32) -> Option<bool> {
        if self.emptied {
            return None;
        }
        match &*self.set {
            CachedSet::Bits { bits, .. } => {
                let hit = usize::try_from(target).is_ok_and(|d| d < bits.len() && bits.get(d));
                if hit {
                    self.doc = target;
                }
                Some(hit)
            }
            CachedSet::Docs(_) => None,
        }
    }

    fn contains(&self, doc: i32) -> Option<bool> {
        if self.emptied {
            return None;
        }
        match &*self.set {
            CachedSet::Bits { bits, .. } => {
                Some(usize::try_from(doc).is_ok_and(|d| d < bits.len() && bits.get(d)))
            }
            CachedSet::Docs(_) => None,
        }
    }

    /// `BitSetIterator.docIDRunEnd` (as of Lucene 10.6, branch_10_6,
    /// #16849): the run of set bits from here, resumed from the run end
    /// already known and scanned at most [`MAX_RUN_END_SCAN`] bits past the
    /// current document -- so a caller asking once per window pays for each
    /// bit once, not once per window, and may get a prefix of the run.
    fn doc_id_run_end(&self) -> i32 {
        if self.emptied {
            return self.doc.saturating_add(1);
        }
        match &*self.set {
            CachedSet::Bits { bits, .. } => {
                let doc = self.doc;
                let length = i32::try_from(bits.len()).unwrap_or(i32::MAX);
                let (mut run_start, mut run_end) = self.run.get();
                if doc < run_start || doc >= run_end {
                    run_start = doc;
                    run_end = doc.saturating_add(1);
                }
                let up_to = doc.saturating_add(length.saturating_sub(doc).min(MAX_RUN_END_SCAN));
                if run_end < up_to {
                    run_end = usize::try_from(run_end)
                        .ok()
                        .zip(usize::try_from(up_to).ok())
                        .and_then(|(from, to)| bits.next_clear_bit_in_range(from, to))
                        .map_or(up_to, |end| end as i32);
                }
                self.run.set((run_start, run_end));
                run_end
            }
            CachedSet::Docs(_) => self.doc.saturating_add(1),
        }
    }
}

#[cfg(test)]
mod tests;
