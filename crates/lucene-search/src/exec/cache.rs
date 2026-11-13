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
    /// Values, and which documents have one.
    Longs { values: Vec<i64>, has: FixedBitSet },
    /// Every document's values (a multi-valued `SORTED_NUMERIC` column, the
    /// geo distance sorts' points): document `d`'s are
    /// `values[starts[d]..starts[d + 1]]`.
    Multi {
        starts: Vec<usize>,
        values: Vec<i64>,
    },
}

impl SortColumn {
    fn ram_bytes(&self) -> usize {
        match self {
            SortColumn::Ords(o) => o.len() * 4,
            SortColumn::Longs { values, has } => values.len() * 8 + has.words().len() * 8,
            SortColumn::Multi { starts, values } => starts.len() * 8 + values.len() * 8,
        }
    }
}

/// Sort columns: uses seen, and the columns built.
#[derive(Default)]
struct Columns {
    uses: HashMap<String, u32>,
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
    /// or built by `build` on its second use when it fits the budget. `None`
    /// means read the column per document.
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
        if c.bytes.saturating_add(bytes) > MAX_COLUMN_BYTES {
            return Ok(None);
        }
        // Reserve against the process-wide cap; the segment's share goes
        // back when its cache is dropped.
        let reserved = TOTAL_COLUMN_BYTES.fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |t| t.checked_add(bytes).filter(|&n| n <= total_column_budget()),
        );
        if reserved.is_err() {
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
mod tests {

    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    /// **The node's budget holds across segments**: sets past it go least
    /// recently used first, from whichever segment holds them, and a set
    /// too large for the budget is not kept.
    #[test]
    fn the_node_budget_evicts_the_oldest_set_of_any_segment() {
        // 32 sets of 1 KiB fit (each a 32nd of the budget).
        let node: &'static Node = Box::leak(Box::new(Node::new(32 * 1024)));
        let (a, b) = (
            SegmentQueryCache::in_node(node),
            SegmentQueryCache::in_node(node),
        );
        let set = |n: usize| Arc::new(CachedSet::Docs(vec![0; n]));
        for i in 0..32 {
            let c = if i % 2 == 0 { &a } else { &b };
            c.insert(format!("s{i}").into(), set(256));
        }
        assert_eq!(node.bytes.load(Relaxed), 32 * 1024);
        assert_eq!(node.evictions.load(Relaxed), 0);
        // One more evicts the oldest, which is a's first, though b inserted.
        b.insert("new".into(), set(256));
        assert_eq!(node.bytes.load(Relaxed), 32 * 1024);
        assert!(
            !lock(&a.inner).entries.contains_key("s0"),
            "the oldest set went"
        );
        assert!(lock(&a.inner).entries.contains_key("s2"));
        assert!(lock(&b.inner).entries.contains_key("new"));
        assert_eq!(node.evictions.load(Relaxed), 1);
        // Past a sixteenth of the budget: not kept at all.
        b.insert("huge".into(), set(1024));
        assert!(!lock(&b.inner).entries.contains_key("huge"));
        // A segment's cache going away gives its bytes back.
        let held = lock(&a.inner).bytes;
        drop(a);
        assert_eq!(node.bytes.load(Relaxed), 32 * 1024 - held);
    }

    fn bits_scorer(bits: FixedBitSet) -> CachedScorer {
        let cardinality = bits.cardinality() as i64;
        CachedScorer::new(Arc::new(CachedSet::Bits { bits, cardinality }))
    }

    /// Lucene 10.6's `TestBitSetIterator.testDocIDRunEndScanIsBounded`
    /// (branch_10_6, #16849): one call scans at most `MAX_RUN_END_SCAN` bits
    /// past the current document.
    #[test]
    fn a_bit_set_run_end_scan_is_bounded() {
        let max = MAX_RUN_END_SCAN;
        let n = 2 * max + 5;
        let mut bits = FixedBitSet::new(n as usize);
        bits.set_range(0, n as usize);
        let mut s = bits_scorer(bits);
        assert_eq!(s.next_doc().unwrap(), 0);
        assert_eq!(s.doc_id_run_end(), max);
        assert_eq!(s.advance(10).unwrap(), 10);
        assert_eq!(s.doc_id_run_end(), 10 + max);
        assert_eq!(s.advance(2 * max).unwrap(), 2 * max);
        assert_eq!(s.doc_id_run_end(), n);
    }

    /// `TestBitSetIterator.testRandomDocIDRunEndLongRuns`: whatever is
    /// remembered between calls, each run end is the bounded scan from the
    /// current document.
    #[test]
    fn a_bit_set_run_end_resumes_to_the_bounded_scan() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        for _ in 0..10 {
            let n = 1 + next(4 * MAX_RUN_END_SCAN as u64) as usize;
            let mut bits = FixedBitSet::new(n);
            for _ in 0..1 + next(10) {
                let start = next(n as u64) as usize;
                let end = start + next((n - start) as u64 + 1) as usize;
                bits.set_range(start, end);
            }
            let expected = |doc: usize| {
                let len = doc + (n - doc).min(MAX_RUN_END_SCAN as usize);
                let mut end = doc + 1;
                while end < len && bits.get(end) {
                    end += 1;
                }
                end as i32
            };
            let mut s = bits_scorer(bits.clone());
            let mut d = s.next_doc().unwrap();
            while d != NO_MORE_DOCS {
                assert!(bits.get(d as usize));
                assert_eq!(s.doc_id_run_end(), expected(d as usize), "doc {d} of {n}");
                d = s.advance(d + 1 + next(5_000) as i32).unwrap();
            }
        }
    }

    #[test]
    fn a_dropped_bit_set_goes_back_to_the_pool_cleared() {
        // Test threads are shared, so this keys on a length of its own.
        const LEN: usize = 2_011;
        let mut bits = FixedBitSet::new(LEN);
        bits.set(3);
        bits.set(LEN - 1);
        let held = crate::bit_set_pool::held(LEN);
        drop(Arc::new(CachedSet::Bits {
            bits,
            cardinality: 2,
        }));
        assert_eq!(crate::bit_set_pool::held(LEN), held + 1);
        let back = crate::bit_set_pool::take(LEN).expect("pooled");
        assert_eq!(back.cardinality(), 0, "handed back cleared");
        // A doc list holds no bit set: nothing is pooled.
        drop(CachedSet::Docs(vec![1, 2]));
        assert_eq!(crate::bit_set_pool::held(LEN), held);
    }
    use crate::exec::leaf::DocList;
    use crate::query::{
        BooleanQuery, DisjunctionMaxQuery, MatchAllDocsQuery, PhraseQuery, PrefixQuery, TermQuery,
    };

    /// `ConstantScoreScorer` over a cached iterator: the constant as score
    /// and bound; under `TOP_SCORES` a threshold past it empties the
    /// iterator (no membership, no run, no bits, nothing left to advance
    /// to); a threshold at it, or a scorer not pruning, leaves it be.
    #[test]
    fn a_constant_cached_scorer_empties_like_constant_score_scorer() {
        let mut bits = FixedBitSet::new(100);
        for d in [3, 4, 50] {
            bits.set(d);
        }
        for set in [
            CachedSet::Bits {
                bits,
                cardinality: 3,
            },
            CachedSet::Docs(vec![3, 4, 50]),
        ] {
            let set = Arc::new(set);
            let mut s = CachedScorer::constant(Arc::clone(&set), 2.5, true);
            assert_eq!(s.next_doc().unwrap(), 3);
            assert_eq!(s.score().unwrap(), 2.5);
            assert_eq!(s.max_score(NO_MORE_DOCS).unwrap(), 2.5);
            s.set_min_competitive_score(2.5).unwrap();
            assert_eq!(s.next_doc().unwrap(), 4, "a tie still competes");
            s.set_min_competitive_score(3.0).unwrap();
            assert!(s.contains(50).is_none());
            assert_eq!(s.doc_id_run_end(), 5);
            assert!(s.constant_bits().is_none());
            assert_eq!(s.advance(10).unwrap(), NO_MORE_DOCS);
            let mut t = CachedScorer::constant(set, 1.0, false);
            t.set_min_competitive_score(9.0).unwrap();
            assert_eq!(t.advance(40).unwrap(), 50, "not pruning: never emptied");
        }
    }

    fn list(docs: Vec<i32>) -> Result<Option<BoxScorer<'static>>> {
        Ok(Some(Box::new(DocList::new(docs, Vec::new()))))
    }

    fn hit(r: Option<CacheResult>) -> BoxScorer<'static> {
        match r {
            Some(CacheResult::Hit(s)) => Box::new(CachedScorer::new(s)),
            _ => panic!("expected a cached scorer"),
        }
    }

    #[test]
    fn the_policy_shapes_follow_usage_tracking() {
        let term = Clause::Term(TermQuery::new("f", "t"));
        assert_eq!(shape(&term), (false, false, true));
        assert_eq!(
            shape(&Clause::MatchAllDocs(MatchAllDocsQuery::new(3))),
            (false, false, true)
        );
        assert_eq!(
            shape(&Clause::Boolean(Box::new(BooleanQuery::new()))),
            (false, true, true)
        );
        let mut b = BooleanQuery::new();
        b.must.push(term.clone());
        assert_eq!(shape(&Clause::Boolean(Box::new(b))), (false, true, false));
        let empty = DisjunctionMaxQuery::new(Vec::<Clause>::new(), 0.0);
        assert_eq!(
            shape(&Clause::DisjunctionMax(Box::new(empty))),
            (false, true, true)
        );
        assert_eq!(
            shape(&Clause::Prefix(PrefixQuery::new("f", "p"))),
            (true, false, false)
        );
        assert_eq!(
            shape(&Clause::Phrase(PhraseQuery::new("f", ["a", "b"]))),
            (false, false, false)
        );
    }

    #[test]
    fn a_clause_is_cached_after_the_policy_threshold_and_served_from_then_on() {
        let cache = SegmentQueryCache::default();
        assert!(format!("{cache:?}").contains("SegmentQueryCache"));
        let phrase = Clause::Phrase(PhraseQuery::new("f", ["a", "b"]));
        // Never for a small segment, nor for a term.
        assert!(cache
            .scorer(&phrase, 9_999, || list(vec![1]))
            .unwrap()
            .is_none());
        let term = Clause::Term(TermQuery::new("f", "t"));
        assert!(cache
            .scorer(&term, 20_000, || list(vec![1]))
            .unwrap()
            .is_none());
        // A phrase is an ordinary query: cached on its fifth use.
        for _ in 0..4 {
            let r = cache
                .scorer(&phrase, 20_000, || panic!("not cached yet"))
                .unwrap();
            assert!(r.is_none());
        }
        let mut s = hit(cache
            .scorer(&phrase, 20_000, || list(vec![3, 9, 17]))
            .unwrap());
        assert_eq!(cache.stats().0, 1);
        let mut again = hit(cache
            .scorer(&phrase, 20_000, || panic!("served from the cache"))
            .unwrap());
        // Sparse: a doc list.
        for scorer in [&mut s, &mut again] {
            assert_eq!(scorer.cost(), 3);
            assert_eq!(scorer.contains(3), None, "a doc list has no random access");
            assert_eq!(scorer.next_doc().unwrap(), 3);
            assert_eq!(scorer.doc_id_run_end(), 4);
            assert_eq!(scorer.advance(10).unwrap(), 17);
            assert_eq!(scorer.score().unwrap(), 0.0);
            assert_eq!(scorer.max_score(NO_MORE_DOCS).unwrap(), 0.0);
            assert_eq!(scorer.next_doc().unwrap(), NO_MORE_DOCS);
        }
    }

    #[test]
    fn an_uncacheable_clause_is_counted_but_never_cached() {
        // `CachingWrapperWeight`: `onUse` first, then the `isCacheable`
        // short-circuit, so the uses still count once it becomes cacheable
        // (a reopened segment without the doc-values update).
        let cache = SegmentQueryCache::default();
        let phrase = Clause::Phrase(PhraseQuery::new("f", ["a", "b"]));
        for _ in 0..6 {
            let r = cache
                .scorer_if_cacheable(
                    &phrase,
                    20_000,
                    false,
                    || Ok(false),
                    || panic!("never built"),
                )
                .unwrap();
            assert!(r.is_none());
        }
        assert_eq!(cache.stats().0, 0);
        let _ = hit(cache
            .scorer_if_cacheable(&phrase, 20_000, true, || Ok(false), || list(vec![5]))
            .unwrap());
        assert_eq!(cache.stats().0, 1);
        assert!(cache
            .scorer_if_cacheable(
                &phrase,
                20_000,
                false,
                || Ok(false),
                || panic!("not served either")
            )
            .unwrap()
            .is_none());
    }

    #[test]
    fn dense_sets_are_bitsets_with_runs_and_empty_ones_match_nothing() {
        let cache = SegmentQueryCache::default();
        let prefix = Clause::Prefix(PrefixQuery::new("f", "p"));
        // Costly: cached on the second use. 300 of 20,000 is over 1%.
        assert!(cache
            .scorer(&prefix, 20_000, || panic!("first use"))
            .unwrap()
            .is_none());
        let dense: Vec<i32> = (100..400).collect();
        let mut s = hit(cache.scorer(&prefix, 20_000, || list(dense)).unwrap());
        assert_eq!(s.cost(), 300);
        assert_eq!(s.next_doc().unwrap(), 100);
        assert_eq!(s.doc_id_run_end(), 400);
        assert_eq!(s.advance(399).unwrap(), 399);
        assert_eq!(s.advance(400).unwrap(), NO_MORE_DOCS);
        let other = Clause::Prefix(PrefixQuery::new("f", "q"));
        assert!(cache
            .scorer(&other, 20_000, || panic!("first use"))
            .unwrap()
            .is_none());
        let r = cache.scorer(&other, 20_000, || Ok(None)).unwrap();
        assert!(matches!(r, Some(CacheResult::Empty)));
        // The empty answer is cached, as `LRUQueryCache` keeps `DocIdSet.EMPTY`.
        let r = cache
            .scorer(&other, 20_000, || panic!("cached empty"))
            .unwrap();
        assert!(matches!(r, Some(CacheResult::Empty)));
        // So is a set that turned out empty once built.
        let none = Clause::Prefix(PrefixQuery::new("f", "r"));
        assert!(cache
            .scorer(&none, 20_000, || panic!("first use"))
            .unwrap()
            .is_none());
        let r = cache.scorer(&none, 20_000, || list(Vec::new())).unwrap();
        assert!(matches!(r, Some(CacheResult::Hit(_))));
        let r = cache.scorer(&none, 20_000, || panic!("cached")).unwrap();
        assert!(matches!(r, Some(CacheResult::Empty)));
    }

    /// Two searches building the same set at once: the second insert
    /// replaces the first, and the byte count stays that of one entry.
    #[test]
    fn a_racing_second_insert_replaces_the_first() {
        let cache = SegmentQueryCache::default();
        let set = || Arc::new(CachedSet::Docs(vec![1, 2, 3]));
        cache.insert("q".into(), set());
        cache.insert("q".into(), set());
        assert_eq!(cache.stats(), (1, 12));
    }

    #[test]
    fn the_least_recently_used_entries_are_evicted_past_the_bounds() {
        let cache = SegmentQueryCache::default();
        for i in 0..(MAX_ENTRIES + 6) {
            let q = Clause::Prefix(PrefixQuery::new("f", format!("p{i}")));
            assert!(cache
                .scorer(&q, 20_000, || panic!("first use"))
                .unwrap()
                .is_none());
            hit(cache.scorer(&q, 20_000, || list(vec![1, 2])).unwrap());
        }
        let (entries, bytes) = cache.stats();
        assert_eq!(entries, MAX_ENTRIES);
        assert_eq!(bytes, MAX_ENTRIES * 8);
        // The first ones went; the last is still served.
        let last = Clause::Prefix(PrefixQuery::new("f", format!("p{}", MAX_ENTRIES + 5)));
        hit(cache.scorer(&last, 20_000, || panic!("cached")).unwrap());
        // The first one went, and the policy's history (256 uses) has
        // forgotten it too: it is a first use again, then cached on the next.
        let first = Clause::Prefix(PrefixQuery::new("f", "p0"));
        assert!(cache
            .scorer(&first, 20_000, || list(vec![5]))
            .unwrap()
            .is_none());
        let r = cache.scorer(&first, 20_000, || list(vec![5])).unwrap();
        assert!(
            matches!(r, Some(CacheResult::Hit(_))),
            "rebuilt after eviction"
        );
        // A set over the byte bound evicts everything older, then itself fits
        // alone only if under the bound: 16 MB of bits is 134M documents.
        let huge = Clause::Prefix(PrefixQuery::new("f", "huge"));
        let max_doc = i32::try_from(MAX_BYTES * 8 + 64).unwrap();
        assert!(cache
            .scorer(&huge, max_doc, || panic!("first use"))
            .unwrap()
            .is_none());
        hit(cache
            .scorer(&huge, max_doc, || list((0..max_doc / 50).collect()))
            .unwrap());
        assert!(cache.stats().1 <= MAX_BYTES);
    }

    #[test]
    fn a_sort_column_is_decoded_on_its_second_use_within_the_budget() {
        let cache = SegmentQueryCache::default();
        let ords = |n: usize| move || Ok(SortColumn::Ords(vec![7; n]));
        let col = |key: &str, max_doc: i32, n: usize| cache.sort_column(key, max_doc, &mut ords(n));
        // A small segment never caches.
        assert!(col("k", 100, 100).unwrap().is_none());
        assert!(col("k", 100, 100).unwrap().is_none());
        // First use: nothing; second: built; afterwards: the same column,
        // whatever the builder would say.
        assert!(col("f", 20_000, 20_000).unwrap().is_none());
        let built = col("f", 20_000, 20_000).unwrap().unwrap();
        assert!(matches!(&*built, SortColumn::Ords(v) if v.len() == 20_000));
        let again = cache
            .sort_column("f", 20_000, &mut || -> Result<SortColumn> {
                unreachable!()
            })
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&built, &again));
        // A column past the budget is built once and dropped; not retried.
        let big = MAX_COLUMN_BYTES / 4 + 1;
        assert!(col("g", 20_000, big).unwrap().is_none());
        assert!(col("g", 20_000, big).unwrap().is_none());
        assert!(col("g", 20_000, big).unwrap().is_none());
        // A failed build is the search's error.
        assert!(col("h", 20_000, 1).unwrap().is_none());
        let err = cache.sort_column("h", 20_000, &mut || -> Result<SortColumn> {
            Err(crate::top_field::SortError::NoKeys.into())
        });
        assert!(err.is_err());
        // The process-wide count holds this cache's columns while it lives
        // (other tests' caches come and go concurrently, so no upper bound).
        assert!(sort_column_bytes() >= 20_000 * 4);
        drop(cache);
        let longs = SortColumn::Longs {
            values: vec![1, 2],
            has: FixedBitSet::new(2),
        };
        assert_eq!(longs.ram_bytes(), 16 + 8);
    }

    /// A clause keyed again takes its first key; a different clause at the
    /// same address gets its own; an equal clause elsewhere gets an equal
    /// key.
    #[test]
    fn a_clause_is_formatted_once_per_search() {
        let mut c = Clause::Prefix(PrefixQuery::new("f", "a"));
        let first = clause_key(&c);
        assert!(Arc::ptr_eq(&first, &clause_key(&c)));
        assert_eq!(&*first, format!("{c:?}"));
        c = Clause::Prefix(PrefixQuery::new("f", "b"));
        let other = clause_key(&c);
        assert_ne!(first, other);
        assert_eq!(&*other, format!("{c:?}"));
        let elsewhere = Box::new(Clause::Prefix(PrefixQuery::new("f", "a")));
        assert_eq!(clause_key(&elsewhere), first);
        // Past the memo's size, the oldest is formatted again, alike.
        let many: Vec<Clause> = (0..KEY_MEMO + 2)
            .map(|i| Clause::Prefix(PrefixQuery::new("f", format!("m{i}"))))
            .collect();
        for m in &many {
            assert_eq!(&*clause_key(m), format!("{m:?}"));
        }
        assert_eq!(&*clause_key(&many[0]), format!("{:?}", many[0]));
    }
}
