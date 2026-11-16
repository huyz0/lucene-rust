//! A size- and memory-bounded query-result cache: this port's
//! `org.apache.lucene.search.LRUQueryCache`. Given a `(segment, query)` pair,
//! caches the query's materialized matching-doc set so a repeat of the same
//! query against the same segment doesn't re-run its scorer/matcher.
//!
//! # What is ported
//!
//! - **The stored representation.** Not a raw [`FixedBitSet`] — a
//!   [`CachedDocIdSet`], which applies `LRUQueryCache.cacheImpl`'s exact rule:
//!   a set with ≥1% density stays a bitset (random access is what makes a
//!   cached filter useful as a conjunction lead), and anything sparser becomes
//!   a [`crate::docid_set::RoaringDocIdSet`]. This is the difference between charging
//!   `maxDoc / 8` bytes per cached entry regardless of match count and charging
//!   roughly the match count.
//! - **Both bounds.** `maxSize` (entry count) *and* `maxRamBytesUsed`, charged
//!   from each entry's own `ramBytesUsed` exactly as Java's
//!   `LRUQueryCachePartition.requiresEviction` does, evicting least-recently-used
//!   first until neither bound is exceeded.
//! - **The caching policy.** [`UsageTrackingPolicy`] is
//!   `UsageTrackingQueryCachingPolicy`: a fixed-size ring buffer of recently
//!   used query hashes, a per-query-shape minimum frequency before a query is
//!   worth caching (2 for costly queries, 5 for ordinary ones, 4 for composite
//!   ones), and a `shouldNeverCache` set that — notably — includes `TermQuery`,
//!   which Java refuses to cache because re-running it is already cheap.
//!   [`QueryCachingPolicy::on_use`] is the caller's to call, once per query
//!   execution, mirroring `CachingWrapperWeight`'s `used.compareAndSet` guard —
//!   see [`QueryCache::get_or_compute_with_policy`].
//! - **The leaf-size skip**, `LRUQueryCache`'s default
//!   `MinSegmentSizePredicate(10000)`: [`leaf_is_worth_caching`].
//!
//! # What is deliberately not ported
//!
//! - **Thread-safe sharing.** Java's cache is one shared object behind a
//!   `ReentrantReadWriteLock`, split into 16 partitions to cut contention.
//!   This one is `&mut`-exclusive: an owner holds it, and the borrow checker
//!   makes the data race impossible rather than a lock making it merely
//!   unlikely (the `rust-performance` skill's "ownership over locks"). A
//!   caller that wants one cache shared across rayon leaf threads wraps it
//!   itself; a per-thread cache needs nothing.
//! - **Segment identity.** Java keys on `IndexReader.CacheHelper`'s identity
//!   with a weak reference and a close listener, so closing a segment drops
//!   its entries automatically. This port has no such lifecycle object, so `S`
//!   is any caller-supplied key and [`QueryCache::invalidate_segment`] is
//!   called by hand.
//! - **`skipCacheFactor`** (Java skips caching a clause whose cost is 10x the
//!   lead cost of the conjunction it sits in): this port has no `cost()`
//!   estimate on its iterators to compare against.
//!
//! **Cache key**: a query type usable as a key needs `Eq + Hash + Clone` --
//! [`crate::query::TermQuery`] already derives all three. [`QueryCache`] is generic
//! over any `Q: Eq + Hash + Clone` and any segment identifier `S: Eq + Hash +
//! Clone`.
//!
//! **Wired entry point**: [`search_term_query_cached`] composes this cache with
//! [`crate::search_term_query`], opt-in — [`crate::search_term_query`] itself is
//! unchanged and still the uncached default. Note the tension with the ported
//! policy above: `UsageTrackingQueryCachingPolicy` would *never* cache a
//! `TermQuery`. That entry point exists because `TermQuery` is the only query
//! type in this port whose representation satisfies `Eq + Hash` (see its doc
//! comment), so it is what a cache can currently be demonstrated on; a caller
//! that wants Java's judgement applies [`UsageTrackingPolicy`] via
//! [`QueryCache::get_or_compute_with_policy`] and gets the uncached path.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::postings::DocInput;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::{Collector, VecCollector};
use crate::docid_set::CachedDocIdSet;
use crate::query::TermQuery;
use crate::Error;

/// A `(segment, query)` compound cache key. Two keys are equal iff both their
/// segment identifier and their query are equal -- the same query against a
/// different segment, or a different query against the same segment, are
/// always distinct entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey<S, Q> {
    segment: S,
    query: Q,
}

/// One cached entry: the materialized doc set plus an opaque monotonic "last
/// touched" stamp used to order eviction. Not a wall-clock timestamp -- a
/// strictly increasing counter bumped on every access (see
/// [`QueryCache::clock`]), which is what Java's access-ordered `LinkedHashMap`
/// encodes implicitly in its list order.
///
/// The set is behind an [`Arc`]: a cache *hit* must not copy the set it just
/// found. Handing back a clone of the whole thing would make a hit cost
/// `O(maxDoc)` bytes of memcpy -- on a 10M-document segment, 1.25 MB per hit,
/// which is more work than re-running many queries. Java
/// hands back a reference to the shared immutable `DocIdSet`; an `Arc` is that.
struct Entry {
    set: Arc<CachedDocIdSet>,
    last_used: u64,
    /// Charged against `max_ram_bytes`; cached rather than recomputed so
    /// eviction accounting can't drift from what was added.
    ram_bytes: usize,
}

/// A bounded, LRU-evicting cache from `(segment key, query)` to that query's
/// materialized match set in that segment -- see this module's doc comment for
/// what is and isn't ported from `LRUQueryCache`.
///
/// - `S`: a segment identifier. Any `Eq + Hash + Clone` type works -- a
///   segment name `String`, a generation number `u64`, whatever the caller
///   already has on hand to distinguish segments.
/// - `Q`: a query representation. Any `Eq + Hash + Clone` type works, e.g.
///   [`crate::query::TermQuery`].
pub struct QueryCache<S, Q> {
    max_entries: usize,
    max_ram_bytes: usize,
    entries: HashMap<CacheKey<S, Q>, Entry>,
    /// `last_used` stamp -> key, so finding the least-recently-used entry is
    /// `O(log n)` rather than a linear scan of every entry. The linear scan
    /// this replaces made a RAM-bounded eviction round (which can evict many
    /// entries in one insert) quadratic in cache size.
    ///
    /// Every live entry has exactly one stamp here and vice versa; the two are
    /// updated together on every hit, insert and removal.
    order: BTreeMap<u64, CacheKey<S, Q>>,
    /// Monotonic counter bumped on every access.
    clock: u64,
    ram_bytes_used: usize,
    hit_count: u64,
    miss_count: u64,
    cache_count: u64,
    eviction_count: u64,
}

impl<S, Q> QueryCache<S, Q>
where
    S: Eq + Hash + Clone,
    Q: Eq + Hash + Clone,
{
    /// Creates an empty cache holding at most `max_entries` entries, with no
    /// memory bound (`LRUQueryCache(maxSize, Long.MAX_VALUE)`).
    ///
    /// `max_entries == 0` is a valid, degenerate "cache nothing" bound:
    /// [`Self::get_or_compute`] always calls `compute` and never stores the
    /// result.
    pub fn new(max_entries: usize) -> Self {
        Self::with_ram_limit(max_entries, usize::MAX)
    }

    /// `LRUQueryCache(maxSize, maxRamBytesUsed)`: bounded by entry count *and*
    /// by the summed [`CachedDocIdSet::ram_bytes_used`] of everything cached.
    /// Whichever bound trips first evicts least-recently-used entries until
    /// neither is exceeded -- Java's `requiresEviction()`, which is
    /// `size > maxSize || ramBytesUsed > maxRamBytesUsed`.
    pub fn with_ram_limit(max_entries: usize, max_ram_bytes: usize) -> Self {
        Self {
            max_entries,
            max_ram_bytes,
            entries: HashMap::new(),
            order: BTreeMap::new(),
            clock: 0,
            ram_bytes_used: 0,
            hit_count: 0,
            miss_count: 0,
            cache_count: 0,
            eviction_count: 0,
        }
    }

    /// Number of entries currently cached (`LRUQueryCache.getCacheSize`).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Summed heap cost of every cached set (`LRUQueryCache.ramBytesUsed`).
    pub fn ram_bytes_used(&self) -> usize {
        self.ram_bytes_used
    }

    /// `LRUQueryCache.getHitCount`.
    pub fn hit_count(&self) -> u64 {
        self.hit_count
    }

    /// `LRUQueryCache.getMissCount`.
    pub fn miss_count(&self) -> u64 {
        self.miss_count
    }

    /// `LRUQueryCache.getCacheCount`: how many entries have ever been inserted.
    pub fn cache_count(&self) -> u64 {
        self.cache_count
    }

    /// `LRUQueryCache.getEvictionCount`.
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    /// Returns the cached set for `(segment, query)` if present (marking it
    /// most-recently-used), otherwise computes it via `compute`, stores it in
    /// whichever of the two representations `LRUQueryCache.cacheImpl` would
    /// pick (see [`CachedDocIdSet::from_bitset`]), and returns it.
    ///
    /// `compute` returns a `Result`, so a failed computation propagates
    /// straight out and **nothing is inserted** -- there is no window in which
    /// a placeholder "matches nothing" entry can shadow a later correct
    /// recompute of the same key.
    ///
    /// `compute` is only called on a miss.
    pub fn get_or_compute<E>(
        &mut self,
        segment: S,
        query: Q,
        compute: impl FnOnce() -> Result<FixedBitSet, E>,
    ) -> Result<Arc<CachedDocIdSet>, E> {
        let key = CacheKey { segment, query };
        self.clock += 1;
        let now = self.clock;

        if let Some(entry) = self.entries.get_mut(&key) {
            self.order.remove(&entry.last_used);
            entry.last_used = now;
            let set = Arc::clone(&entry.set);
            self.order.insert(now, key);
            self.hit_count += 1;
            return Ok(set);
        }

        self.miss_count += 1;
        let set = Arc::new(CachedDocIdSet::from_bitset(compute()?));
        self.insert(key, Arc::clone(&set), now);
        Ok(set)
    }

    /// [`Self::get_or_compute`] gated by a [`QueryCachingPolicy`], the way
    /// `LRUQueryCache.CachingWrapperWeight` consults its policy before ever
    /// touching the cache: on a miss the result is only *stored* if
    /// `policy.should_cache` says this query has been seen often enough to be
    /// worth it.
    ///
    /// An already-cached entry is still served on a hit regardless of what the
    /// policy now says -- same as Java, which checks the cache first and only
    /// asks the policy when it has to populate.
    ///
    /// **The caller records the use, once per query execution, not once per
    /// leaf.** `CachingWrapperWeight.scorerSupplier` guards its own call with
    /// `if (used.compareAndSet(false, true)) policy.onUse(getQuery())` -- one
    /// `onUse` per `Weight`, across every leaf that `Weight` is asked about.
    /// Calling it here instead would bump a query's tracked frequency once per
    /// segment, so an N-segment reader would cross Java's 5-observation
    /// threshold after 5/N executions rather than 5. So: call
    /// [`QueryCachingPolicy::on_use`] once before the per-leaf loop, then this
    /// per leaf.
    pub fn get_or_compute_with_policy<E, P>(
        &mut self,
        policy: &mut P,
        segment: S,
        query: Q,
        compute: impl FnOnce() -> Result<FixedBitSet, E>,
    ) -> Result<Arc<CachedDocIdSet>, E>
    where
        P: QueryCachingPolicy<Q>,
    {
        let key = CacheKey { segment, query };
        self.clock += 1;
        let now = self.clock;

        if let Some(entry) = self.entries.get_mut(&key) {
            self.order.remove(&entry.last_used);
            entry.last_used = now;
            let set = Arc::clone(&entry.set);
            self.order.insert(now, key);
            self.hit_count += 1;
            return Ok(set);
        }

        self.miss_count += 1;
        // Java asks the policy before it reads anything
        // (`CachingWrapperWeight.scorerSupplier`); a set that will not be
        // kept skips the roaring conversion it would only be cached in.
        if !policy.should_cache(&key.query) {
            let bits = compute()?;
            let cardinality = bits.cardinality();
            return Ok(Arc::new(CachedDocIdSet::Bits { bits, cardinality }));
        }
        let set = Arc::new(CachedDocIdSet::from_bitset(compute()?));
        self.insert(key, Arc::clone(&set), now);
        Ok(set)
    }

    /// Inserts `key -> set`, then evicts least-recently-used entries while
    /// either bound is exceeded. Insert-then-evict is Java's order
    /// (`put` followed by `evictIfNecessary`), and it matters for the RAM
    /// bound: a single entry larger than `max_ram_bytes` is inserted and then
    /// immediately evicted, exactly as Java's does, rather than silently
    /// bypassing accounting.
    fn insert(&mut self, key: CacheKey<S, Q>, set: Arc<CachedDocIdSet>, now: u64) {
        // Only ever reached on a miss, so `key` is not already present -- both
        // callers look it up first.
        debug_assert!(!self.entries.contains_key(&key));
        let ram_bytes = set.ram_bytes_used();
        self.ram_bytes_used += ram_bytes;
        self.cache_count += 1;
        self.entries.insert(
            key.clone(),
            Entry {
                set,
                last_used: now,
                ram_bytes,
            },
        );
        self.order.insert(now, key);
        while self.requires_eviction() {
            self.evict_lru();
        }
    }

    /// `LRUQueryCachePartition.requiresEviction`.
    fn requires_eviction(&self) -> bool {
        !self.entries.is_empty()
            && (self.entries.len() > self.max_entries || self.ram_bytes_used > self.max_ram_bytes)
    }

    /// Removes the entry with the smallest `last_used` stamp -- `O(log n)` via
    /// [`Self::order`], not a scan of every entry. A no-op on an empty cache.
    fn evict_lru(&mut self) {
        let Some((_, key)) = self.order.pop_first() else {
            return;
        };
        if let Some(entry) = self.entries.remove(&key) {
            self.ram_bytes_used -= entry.ram_bytes;
            self.eviction_count += 1;
        }
    }

    /// Removes every cached entry whose segment key equals `segment`,
    /// leaving every other segment's entries untouched. Returns the number
    /// of entries removed.
    ///
    /// This is the one piece of "cache goes stale" handling this module
    /// implements directly -- real automatic invalidation hooked to a
    /// segment's actual lifecycle (open/close/merge) is deferred, see this
    /// module's doc comment.
    pub fn invalidate_segment(&mut self, segment: &S) -> usize {
        let doomed: Vec<CacheKey<S, Q>> = self
            .entries
            .keys()
            .filter(|key| &key.segment == segment)
            .cloned()
            .collect();
        for key in &doomed {
            self.remove_key(key);
        }
        doomed.len()
    }

    /// Removes every cached entry, regardless of segment or query.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.ram_bytes_used = 0;
    }

    /// Removes the single entry for `(segment, query)`, if present. Returns
    /// whether an entry was actually removed.
    pub fn remove(&mut self, segment: &S, query: &Q) -> bool {
        self.remove_key(&CacheKey {
            segment: segment.clone(),
            query: query.clone(),
        })
    }

    fn remove_key(&mut self, key: &CacheKey<S, Q>) -> bool {
        match self.entries.remove(key) {
            Some(entry) => {
                self.order.remove(&entry.last_used);
                self.ram_bytes_used -= entry.ram_bytes;
                true
            }
            None => false,
        }
    }
}

/// `org.apache.lucene.search.QueryCachingPolicy`: the "is this query worth
/// caching at all" decision, kept separate from the cache itself exactly as
/// Java keeps it, so a caller can supply its own.
pub trait QueryCachingPolicy<Q> {
    /// Records one *execution* of `query`, whether or not it ends up cached.
    ///
    /// Once per query, not once per leaf -- see
    /// [`QueryCache::get_or_compute_with_policy`] for why the distinction
    /// changes what the frequency thresholds mean.
    fn on_use(&mut self, query: &Q);
    /// Whether `query` has earned a cache entry.
    fn should_cache(&mut self, query: &Q) -> bool;
}

/// What [`UsageTrackingPolicy`] needs to know about a query to apply Java's
/// per-shape thresholds -- the three `UsageTrackingQueryCachingPolicy` static
/// predicates (`shouldNeverCache`, `isCostly`, and the composite-query test
/// inside `minFrequencyToCache`), expressed as a trait so a query type
/// declares its own shape instead of the policy switching on a closed enum.
pub trait CachingCost {
    /// `UsageTrackingQueryCachingPolicy.shouldNeverCache`: a query so cheap to
    /// re-run that a cache entry is pure overhead. Java's list is `TermQuery`,
    /// `FieldExistsQuery`, `MatchAllDocsQuery`, `MatchNoDocsQuery`, and any
    /// empty `BooleanQuery`/`DisjunctionMaxQuery`.
    fn never_cache(&self) -> bool {
        false
    }
    /// `UsageTrackingQueryCachingPolicy.isCostly`: expensive to *build* a doc
    /// set for (multi-term, term-in-set and point queries), so worth caching
    /// after only 2 observations rather than 5.
    fn is_costly(&self) -> bool {
        false
    }
    /// A `BooleanQuery`/`DisjunctionMaxQuery`: cached one observation earlier
    /// than an ordinary query, so that "A OR B" makes it into the cache before
    /// A and B separately do.
    fn is_composite(&self) -> bool {
        false
    }
}

impl CachingCost for TermQuery {
    /// Java's first `shouldNeverCache` case, verbatim: "We do not bother
    /// caching term queries since they are already plenty fast."
    fn never_cache(&self) -> bool {
        true
    }
}

/// Java's `FrequencyTrackingRingBuffer`: a fixed-size window of the most
/// recently used query hashes plus the frequency of each hash inside that
/// window. Hashes, not queries, so a large query is never kept alive by the
/// policy -- Java's reasoning exactly ("this may cause rare false positives,
/// but at worst this just means we cache a query that was not in fact used
/// enough").
struct FrequencyRingBuffer {
    capacity: usize,
    window: VecDeque<u64>,
    frequencies: HashMap<u64, u32>,
}

impl FrequencyRingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            window: VecDeque::with_capacity(capacity.max(1)),
            frequencies: HashMap::new(),
        }
    }

    fn add(&mut self, hash: u64) {
        if self.window.len() == self.capacity {
            if let Some(evicted) = self.window.pop_front() {
                if let Some(count) = self.frequencies.get_mut(&evicted) {
                    *count -= 1;
                    if *count == 0 {
                        self.frequencies.remove(&evicted);
                    }
                }
            }
        }
        self.window.push_back(hash);
        *self.frequencies.entry(hash).or_insert(0) += 1;
    }

    fn frequency(&self, hash: u64) -> u32 {
        self.frequencies.get(&hash).copied().unwrap_or(0)
    }
}

/// `org.apache.lucene.search.UsageTrackingQueryCachingPolicy`: cache a query
/// only once it has been seen often enough in the recent past to believe it
/// will be seen again.
pub struct UsageTrackingPolicy {
    recently_used: FrequencyRingBuffer,
}

impl Default for UsageTrackingPolicy {
    /// Java's no-arg constructor: a 256-entry history.
    fn default() -> Self {
        Self::with_history_size(256)
    }
}

impl UsageTrackingPolicy {
    pub fn with_history_size(history_size: usize) -> Self {
        Self {
            recently_used: FrequencyRingBuffer::new(history_size),
        }
    }

    /// `UsageTrackingQueryCachingPolicy.minFrequencyToCache`: 2 for costly
    /// queries, 5 for ordinary ones, 4 for composite ones.
    fn min_frequency_to_cache(query: &impl CachingCost) -> u32 {
        if query.is_costly() {
            2
        } else if query.is_composite() {
            4
        } else {
            5
        }
    }

    fn hash_of<Q: Hash>(query: &Q) -> u64 {
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }

    /// How many times `query` appears in the tracked history window --
    /// `UsageTrackingQueryCachingPolicy.frequency`, exposed for tests the way
    /// Java exposes it package-private for its own.
    pub fn frequency<Q: Hash + CachingCost>(&self, query: &Q) -> u32 {
        self.recently_used.frequency(Self::hash_of(query))
    }
}

impl<Q: Hash + CachingCost> QueryCachingPolicy<Q> for UsageTrackingPolicy {
    fn on_use(&mut self, query: &Q) {
        if query.never_cache() {
            return;
        }
        self.recently_used.add(Self::hash_of(query));
    }

    fn should_cache(&mut self, query: &Q) -> bool {
        if query.never_cache() {
            return false;
        }
        self.recently_used.frequency(Self::hash_of(query)) >= Self::min_frequency_to_cache(query)
    }
}

/// Default minimum segment size below which `LRUQueryCache` does not cache at
/// all (`new MinSegmentSizePredicate(10000)`).
pub const DEFAULT_MIN_LEAF_SIZE: usize = 10_000;

/// `LRUQueryCache.MinSegmentSizePredicate`: caching a tiny leaf is not worth
/// the memory, because re-running the query there is cheap and the leaf is
/// likely to be merged away shortly. A leaf qualifies only if it holds at
/// least `min_size` documents *and* is at least half the average leaf size of
/// the reader it belongs to.
///
/// `num_leaves == 0` is treated as "not worth caching" rather than dividing by
/// zero (a reader with no leaves has nothing to cache anyway).
pub fn leaf_is_worth_caching(
    leaf_max_doc: usize,
    total_max_doc: usize,
    num_leaves: usize,
    min_size: usize,
) -> bool {
    if leaf_max_doc < min_size || num_leaves == 0 {
        return false;
    }
    let average_total_docs = total_max_doc / num_leaves;
    leaf_max_doc * 2 > average_total_docs
}

/// Cached wrapper around [`crate::search_term_query`] -- this module's only
/// query-execution entry point wired to [`QueryCache`].
///
/// Feeds `collector` exactly the documents `search_term_query(fields, doc_in,
/// live_docs, query, collector)` would, in the same ascending order, and
/// surfaces the same [`crate::Error`] on a decode failure -- except that a
/// repeat call with an `==` `query` against the same `segment` key reuses the
/// previously computed doc set from `cache` instead of re-running
/// `search_term_query`'s scorer/matcher. `live_docs` is applied on the way out
/// rather than being cached (see below), so an entry stays correct across
/// delete generations.
///
/// **Why only `TermQuery`, not also `BooleanQuery`**: [`QueryCache`]'s `Q` bound is `Eq + Hash + Clone`,
/// and `TermQuery` already derives all three (see that struct's doc comment).
/// `query::BooleanQuery` cannot cheaply gain the same derives: transitively,
/// via `Clause::DisjunctionMax(DisjunctionMaxQuery)`, it embeds an `f32`
/// `tie_breaker`, and `f32` has no total order/hash (`NaN`) -- `BooleanQuery`
/// and `Clause` deliberately derive only `PartialEq`, not `Eq`, for exactly
/// this reason (see `Clause`'s own derive-list comment in `query.rs`).
/// Bolting on `Hash`/`Eq` for `BooleanQuery` would mean inventing a
/// `NaN`-handling hash/equality convention with no precedent in this crate --
/// so `TermQuery`, which already satisfies `QueryCache`'s bound, is the one
/// wired up here. Caching `BooleanQuery` would need a hand-rolled key type
/// for its resolved `Clause` tree.
///
/// **Opt-in, not a replacement**: [`crate::search_term_query`] itself is
/// completely unchanged and still the uncached default -- this function is a
/// separate, additional entry point a caller reaches for explicitly when it
/// wants caching.
///
/// **Deletions are not baked into the cached set.** The entry is computed with
/// `live_docs == None` and `live_docs` is applied while *iterating* it, which is
/// what Java does: `LRUQueryCache.cacheIntoBitSet`/`cacheIntoRoaringDocIdSet`
/// both score with `acceptDocs == null`, and the cache keys on the segment's
/// *core* cache helper, precisely so an entry survives a new `.liv` generation.
/// Caching the post-deletion set instead would make the entry silently wrong
/// for the same `(segment, query)` key under any other `live_docs` -- and,
/// since a hit never looks at `live_docs` at all, wrong invisibly.
///
/// **Error handling**: `compute` propagates its own [`crate::Error`] out of
/// [`QueryCache::get_or_compute`] and nothing is inserted on failure, so a
/// decode error can never leave a poisoned "matches nothing" entry behind to
/// shadow a correct later recompute of the same `(segment, query)` key.
///
/// `num_docs` sizes the match set -- pass the segment's total doc count
/// (`maxDoc`-equivalent), the same value a caller already has on hand to
/// build `live_docs` itself. Whether the stored set ends up a bitset or a
/// [`crate::docid_set::RoaringDocIdSet`] is [`CachedDocIdSet::from_bitset`]'s decision, not the
/// caller's.
#[allow(clippy::too_many_arguments)]
pub fn search_term_query_cached<S>(
    cache: &mut QueryCache<S, TermQuery>,
    segment: S,
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    num_docs: usize,
    query: &TermQuery,
    collector: &mut impl Collector,
) -> crate::Result<()>
where
    S: Eq + Hash + Clone,
{
    let set = cache.get_or_compute(segment, query.clone(), || -> crate::Result<FixedBitSet> {
        let mut vec_collector = VecCollector::default();
        crate::search_term_query(fields, doc_in, None, query, &mut vec_collector)?;
        let mut bits = FixedBitSet::new(num_docs);
        // `num_docs` and the doc ids the postings walk produced arrive from
        // two different places -- the caller's claim about the segment, and
        // the segment's own `.doc` file. A doc id past the bitset is a ghost
        // bit or an index panic, so the bound is the bitset's own length and
        // a disagreement is reported rather than absorbed. Hoisted: one load,
        // not one per hit.
        let num_bits = bits.len();
        for doc_id in vec_collector.docs {
            let idx = usize::try_from(doc_id).ok().filter(|&i| i < num_bits);
            match idx {
                Some(idx) => bits.set(idx),
                None => return Err(Error::CachedDocOutOfRange { doc_id, num_docs }),
            }
        }
        Ok(bits)
    })?;

    // Iterating the set costs O(matches) for the Roaring shape and
    // O(maxDoc / 64) for the bitset shape -- never an O(maxDoc) bit-by-bit
    // scan, which would make a 3-hit query on a 10M-document segment walk 10M
    // bits per cache hit.
    for doc_id in set.iter() {
        // `live_docs` is a *separate* caller-supplied bitset from the cached
        // set being iterated; nothing here makes their lengths equal. A doc
        // the live-docs bitset does not cover is not live -- the same answer
        // `soft_deletes::is_live` and `vector_query`'s accept filter give.
        let live = live_docs.is_none_or(|bits| bits.get_doc(doc_id));
        if live {
            collector.collect(doc_id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
