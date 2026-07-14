//! A standalone, size-bounded query-result cache, analogous in spirit to
//! real Lucene's `LRUQueryCache` (`org.apache.lucene.search.LRUQueryCache`).
//! Given a `(segment, query)` pair, caches the matching-docs bitset
//! ([`FixedBitSet`]) so repeated identical queries against the same segment
//! don't need to re-run the query's scorer/matcher from scratch.
//!
//! This is **not** a byte-for-byte port of `LRUQueryCache.java` -- that class
//! tracks per-segment `IndexReader.CacheKey` identity via weak references,
//! bounds itself by both entry count *and* estimated RAM usage, and decides
//! per-query whether caching is even worthwhile (`shouldCache`, e.g. skipping
//! trivially cheap queries or ones already fully cached upstream). None of
//! that machinery is needed to prove the core value proposition: given a
//! segment identifier and a query, hand back a previously computed
//! [`FixedBitSet`] of matching doc IDs instead of recomputing it. This module
//! keeps only that core -- a bounded, least-recently-used-evicting map from
//! `(segment key, query)` to `FixedBitSet` -- as a from-scratch primitive.
//!
//! **Now wired to one real query-execution entry point, opt-in:**
//! [`search_term_query_cached`] composes this cache with
//! [`crate::search_term_query`] -- a caller that wants caching calls that
//! function instead, passing a `&mut QueryCache<S, TermQuery>` it owns
//! across calls; [`crate::search_term_query`] itself is completely
//! unchanged and still the uncached default every existing caller keeps
//! using. This is deliberately narrow: only `TermQuery` is wired (see
//! [`search_term_query_cached`]'s doc comment for exactly why
//! `BooleanQuery` isn't), there's still no automatic invalidation tied to a
//! real segment's open/close/merge lifecycle (a caller must call
//! [`QueryCache::invalidate_segment`] itself), and there's still no
//! `shouldCache`-style cache-worthiness heuristic. Integrating further --
//! more query types, automatic invalidation, `shouldCache` -- is future
//! work, tracked in `PLAN.md`/`docs/parity.md`.
//!
//! **Cache key**: a query type usable as a key needs `Eq + Hash + Clone` --
//! [`query::TermQuery`] already derives `PartialEq + Eq` (see `query.rs`);
//! this module adds a `Hash` derive to it (a pure additive change: `field:
//! String` and `term: Vec<u8>` are both already `Hash`, and no other code
//! depended on `TermQuery` *not* being `Hash`) rather than inventing a
//! parallel query representation just for caching. [`QueryCache`] itself is
//! generic over any `Q: Eq + Hash + Clone` and any segment identifier `S: Eq
//! + Hash + Clone` -- this port has no `IndexReader.CacheKey`-style segment
//! identity object yet (see [`crate::multi_segment::OpenSegment`]'s doc
//! comment for the current state of segment identity), so a simple
//! caller-supplied key (e.g. a segment name `String` or a generation number
//! `u64`) stands in, matching this task's explicit scope note that real
//! `IndexReader.CacheKey` mechanics aren't needed here.
//!
//! **Eviction policy**: bounded by entry *count* only (`max_entries`), evicting
//! the least-recently-used entry (by both `get_or_compute` hits and inserts)
//! when a new entry would exceed the bound. Real Lucene's RAM-based sizing
//! (`maxRamBytesUsed`, computed from each cached `FixedBitSet`'s actual byte
//! size) is deliberately not implemented -- see "Explicitly deferred" below.
//!
//! **Explicitly deferred** (see `docs/parity.md`):
//! - RAM-based cache sizing (real Lucene bounds by both count and estimated
//!   bytes; this module bounds by count alone).
//! - Automatic per-segment invalidation hooks tied to real segment
//!   open/close/merge lifecycle events -- [`QueryCache::invalidate_segment`]
//!   exists and is correct, but nothing in this port calls it yet, since
//!   nothing in this port owns a segment lifecycle to hook into.
//! - Cache-worthiness heuristics (real `LRUQueryCache.shouldCache` skips
//!   caching a query that's cheap to re-run, or a `MatchAllDocsQuery`, or one
//!   a `ConstantScoreQuery`/`FILTER`-clause caller already caches upstream).
//!   Every `get_or_compute` call here always inserts into the cache on a miss.
//! - Wiring `BooleanQuery` (or any other query type) into a cached entry
//!   point the way [`search_term_query_cached`] does for `TermQuery` -- see
//!   that function's doc comment for why `BooleanQuery` specifically can't
//!   cheaply do this yet.
//! - Automatic per-segment invalidation hooks tied to real segment
//!   open/close/merge lifecycle events remain a caller's own responsibility
//!   even through [`search_term_query_cached`] -- see that function's doc
//!   comment.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::postings::DocInput;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::{Collector, VecCollector};
use crate::query::TermQuery;

/// A `(segment, query)` compound cache key. Two keys are equal iff both their
/// segment identifier and their query are equal -- the same query against a
/// different segment, or a different query against the same segment, are
/// always distinct entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey<S, Q> {
    segment: S,
    query: Q,
}

/// One cached entry: the computed [`FixedBitSet`] plus an opaque monotonic
/// "last touched" timestamp used to find the least-recently-used entry on
/// eviction. Not a real wall-clock timestamp -- just a strictly increasing
/// counter bumped on every access (see [`QueryCache::clock`]).
struct Entry {
    bits: FixedBitSet,
    last_used: u64,
}

/// A bounded, LRU-evicting cache from `(segment key, query)` to a
/// [`FixedBitSet`] of that query's matching doc IDs in that segment -- see
/// this module's doc comment for the full design rationale and explicitly
/// deferred scope.
///
/// - `S`: a segment identifier. Any `Eq + Hash + Clone` type works -- a
///   segment name `String`, a generation number `u64`, whatever this port's
///   caller already has on hand to distinguish segments (see this module's
///   doc comment on why no dedicated segment-identity type is introduced
///   here).
/// - `Q`: a query representation. Any `Eq + Hash + Clone` type works, e.g.
///   [`crate::query::TermQuery`].
pub struct QueryCache<S, Q> {
    max_entries: usize,
    entries: HashMap<CacheKey<S, Q>, Entry>,
    /// Monotonic counter bumped on every access; each `Entry::last_used` is
    /// stamped with the counter's value at the time it was last touched
    /// (inserted or hit), so "least recently used" is just "entry with the
    /// smallest `last_used`" -- see [`Self::evict_lru`].
    clock: u64,
}

impl<S, Q> QueryCache<S, Q>
where
    S: Eq + Hash + Clone,
    Q: Eq + Hash + Clone,
{
    /// Creates an empty cache holding at most `max_entries` entries at once.
    ///
    /// `max_entries == 0` is a valid, degenerate "cache nothing" bound:
    /// [`Self::get_or_compute`] always calls `compute` and never actually
    /// stores the result (every insert would immediately need to evict
    /// itself), matching the intuitive meaning of a zero-sized cache.
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            entries: HashMap::new(),
            clock: 0,
        }
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the cached [`FixedBitSet`] for `(segment, query)` if present
    /// (marking it most-recently-used), computing and inserting it via
    /// `compute` on a miss.
    ///
    /// `compute` is only called on a miss -- a hit never touches it at all,
    /// which is the whole point (repeated identical queries against the same
    /// segment reuse the previously computed bitset instead of re-running
    /// the query's scorer/matcher; see this module's top doc comment).
    pub fn get_or_compute(
        &mut self,
        segment: S,
        query: Q,
        compute: impl FnOnce() -> FixedBitSet,
    ) -> FixedBitSet {
        let key = CacheKey { segment, query };
        self.clock += 1;
        let now = self.clock;

        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = now;
            return entry.bits.clone();
        }

        let bits = compute();
        self.insert(key, bits.clone(), now);
        bits
    }

    /// Inserts `key -> bits`, evicting the least-recently-used entry first if
    /// the cache is already at `max_entries` (or if `max_entries == 0`, in
    /// which case the newly inserted entry is itself immediately evicted --
    /// see [`Self::new`]'s doc comment).
    fn insert(&mut self, key: CacheKey<S, Q>, bits: FixedBitSet, now: u64) {
        if self.max_entries == 0 {
            return;
        }
        while self.entries.len() >= self.max_entries {
            self.evict_lru();
        }
        self.entries.insert(
            key,
            Entry {
                bits,
                last_used: now,
            },
        );
    }

    /// Removes the single entry with the smallest `last_used` stamp (ties
    /// broken arbitrarily by `HashMap` iteration order, which doesn't matter
    /// since a genuine tie means both entries are equally "least recently
    /// used"). A no-op on an empty cache.
    fn evict_lru(&mut self) {
        let Some(lru_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone())
        else {
            return;
        };
        self.entries.remove(&lru_key);
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
        let before = self.entries.len();
        self.entries.retain(|key, _| &key.segment != segment);
        before - self.entries.len()
    }

    /// Removes every cached entry, regardless of segment or query.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Removes the single entry for `(segment, query)`, if present. Returns
    /// whether an entry was actually removed.
    ///
    /// Used by [`search_term_query_cached`] to undo a poisoned insert: see
    /// that function's doc comment for why a failed `compute` still briefly
    /// inserts a placeholder entry, and why that entry must then be removed
    /// again rather than left in the cache to shadow a correct future
    /// recompute.
    pub fn remove(&mut self, segment: &S, query: &Q) -> bool {
        self.entries
            .remove(&CacheKey {
                segment: segment.clone(),
                query: query.clone(),
            })
            .is_some()
    }
}

/// Cached wrapper around [`crate::search_term_query`] -- this module's first,
/// and so far only, real query-execution entry point wired to [`QueryCache`]
/// (see this module's top doc comment: `QueryCache` itself was previously a
/// standalone primitive with nothing in this port actually calling it).
///
/// Behaves exactly like `search_term_query(fields, doc_in, live_docs, query,
/// collector)` -- same matched docs fed to `collector` in ascending order,
/// same [`crate::Error`] surfaced on a decode failure -- except that a repeat
/// call with an `==` `query` against the same `segment` key reuses the
/// previously computed doc set from `cache` instead of re-running
/// `search_term_query`'s scorer/matcher.
///
/// **Why only `TermQuery`, not also `BooleanQuery`** (the task's other
/// suggested candidate): [`QueryCache`]'s `Q` bound is `Eq + Hash + Clone`,
/// and `TermQuery` already derives all three (see that struct's doc comment).
/// `query::BooleanQuery` cannot cheaply gain the same derives: transitively,
/// via `Clause::DisjunctionMax(DisjunctionMaxQuery)`, it embeds an `f32`
/// `tie_breaker`, and `f32` has no total order/hash (`NaN`) -- `BooleanQuery`
/// and `Clause` deliberately derive only `PartialEq`, not `Eq`, for exactly
/// this reason (see `Clause`'s own derive-list comment in `query.rs`).
/// Bolting on `Hash`/`Eq` for `BooleanQuery` would mean inventing a
/// `NaN`-handling hash/equality convention with no existing precedent in this
/// crate to justify it, for a wrapper this task doesn't require to cover
/// every query type -- so `TermQuery`, which already satisfies `QueryCache`'s
/// bound with zero new derive risk, is the one wired up here. Wiring
/// `BooleanQuery` in later is still possible (e.g. by caching its resolved
/// `Clause` tree under a hand-rolled key type), just out of scope for this
/// task.
///
/// **Opt-in, not a replacement**: [`crate::search_term_query`] itself is
/// completely unchanged and still the uncached default -- this function is a
/// separate, additional entry point a caller reaches for explicitly when it
/// wants caching, exactly the "not wired into any live search path" gap this
/// module's top doc comment used to describe having now been closed *for this
/// one query type*, opt-in.
///
/// **Error handling**: [`QueryCache::get_or_compute`]'s `compute` closure
/// has to return a plain [`FixedBitSet`], not a `Result`, so a
/// `search_term_query` failure during a cache-miss computation is captured
/// out-of-band (via the `error` cell below), the closure hands back an
/// empty placeholder bitset (which `get_or_compute` inserts into `cache` as
/// normal, since it has no way to know that placeholder is meaningless),
/// and then -- before returning `Err` to the caller -- this function
/// immediately undoes that insert via [`QueryCache::remove`]. A subsequent
/// call with the same `(segment, query)` key therefore always sees a
/// genuine miss and retries the real computation, rather than getting stuck
/// on a poisoned "matches nothing" entry.
///
/// `num_docs` sizes the [`FixedBitSet`] used to represent this segment's
/// matched-doc set inside the cache -- pass the segment's total doc count
/// (`maxDoc`-equivalent), the same value a caller already has on hand to
/// build `live_docs` itself.
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
    let error: RefCell<Option<crate::Error>> = RefCell::new(None);
    let bits = cache.get_or_compute(segment.clone(), query.clone(), || {
        let mut vec_collector = VecCollector::default();
        match crate::search_term_query(fields, doc_in, live_docs, query, &mut vec_collector) {
            Ok(()) => {
                let mut bits = FixedBitSet::new(num_docs);
                for doc_id in vec_collector.docs {
                    bits.set(doc_id as usize);
                }
                bits
            }
            Err(err) => {
                *error.borrow_mut() = Some(err);
                FixedBitSet::new(num_docs)
            }
        }
    });

    if let Some(err) = error.borrow_mut().take() {
        // Undo the placeholder empty-bitset insert `compute` made above --
        // see this function's doc comment: a failed compute must not leave
        // a poisoned "matches nothing" entry behind to shadow a correct
        // future recompute of the same `(segment, query)` key.
        cache.remove(&segment, query);
        return Err(err);
    }

    for doc_id in 0..bits.len() {
        if bits.get(doc_id) {
            collector.collect(doc_id as i32);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A tiny bitset with exactly the bits in `docs` set, used throughout
    /// these tests as a stand-in for "the result of actually running a
    /// query's matcher" -- the specific bit pattern doesn't matter for cache
    /// correctness, only that distinct computations produce distinguishable
    /// results.
    fn bitset_with(docs: &[usize], num_bits: usize) -> FixedBitSet {
        let mut bits = FixedBitSet::new(num_bits);
        for &d in docs {
            bits.set(d);
        }
        bits
    }

    fn bit_indices(bits: &FixedBitSet) -> Vec<usize> {
        (0..bits.len()).filter(|&i| bits.get(i)).collect()
    }

    #[test]
    fn miss_computes_and_stores() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
        let calls = Cell::new(0);
        let bits = cache.get_or_compute("seg0", "q1", || {
            calls.set(calls.get() + 1);
            bitset_with(&[1, 3, 5], 8)
        });
        assert_eq!(bit_indices(&bits), vec![1, 3, 5]);
        assert_eq!(calls.get(), 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn hit_reuses_without_recomputing() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
        let calls = Cell::new(0);
        let compute = || {
            calls.set(calls.get() + 1);
            bitset_with(&[2, 4], 8)
        };

        let first = cache.get_or_compute("seg0", "q1", compute);
        let second = cache.get_or_compute("seg0", "q1", compute);

        assert_eq!(bit_indices(&first), bit_indices(&second));
        // `compute` (really the shared closure above) only actually ran on
        // the first, miss call -- the second call was a pure cache hit.
        assert_eq!(calls.get(), 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn distinct_queries_same_segment_are_distinct_entries() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
        let a = cache.get_or_compute("seg0", "q1", || bitset_with(&[1], 8));
        let b = cache.get_or_compute("seg0", "q2", || bitset_with(&[2], 8));

        assert_eq!(bit_indices(&a), vec![1]);
        assert_eq!(bit_indices(&b), vec![2]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn same_query_distinct_segments_are_distinct_entries() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
        let a = cache.get_or_compute("seg0", "q1", || bitset_with(&[1], 8));
        let b = cache.get_or_compute("seg1", "q1", || bitset_with(&[2], 8));

        assert_eq!(bit_indices(&a), vec![1]);
        assert_eq!(bit_indices(&b), vec![2]);
        assert_eq!(cache.len(), 2);

        // Confirm each segment's entry is independently still a cache hit
        // (not accidentally aliased to the other segment's entry).
        let calls = Cell::new(0);
        let hit = cache.get_or_compute("seg0", "q1", || {
            calls.set(calls.get() + 1);
            bitset_with(&[99], 8)
        });
        assert_eq!(bit_indices(&hit), vec![1]);
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn eviction_removes_least_recently_used_entry() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(2);
        cache.get_or_compute("seg0", "q1", || bitset_with(&[1], 8));
        cache.get_or_compute("seg0", "q2", || bitset_with(&[2], 8));
        assert_eq!(cache.len(), 2);

        // Touch q1 again so q2 becomes the least-recently-used entry.
        cache.get_or_compute("seg0", "q1", || bitset_with(&[1], 8));

        // Inserting a third distinct entry must evict q2 (LRU), not q1.
        cache.get_or_compute("seg0", "q3", || bitset_with(&[3], 8));
        assert_eq!(cache.len(), 2);

        let calls_q1 = Cell::new(0);
        let q1 = cache.get_or_compute("seg0", "q1", || {
            calls_q1.set(calls_q1.get() + 1);
            bitset_with(&[1], 8)
        });
        assert_eq!(bit_indices(&q1), vec![1]);
        assert_eq!(calls_q1.get(), 0, "q1 should still be cached, not evicted");

        let calls_q2 = Cell::new(0);
        let q2 = cache.get_or_compute("seg0", "q2", || {
            calls_q2.set(calls_q2.get() + 1);
            bitset_with(&[2], 8)
        });
        assert_eq!(bit_indices(&q2), vec![2]);
        assert_eq!(
            calls_q2.get(),
            1,
            "q2 should have been evicted and recomputed"
        );
    }

    #[test]
    fn invalidate_segment_removes_only_that_segments_entries() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(8);
        cache.get_or_compute("seg0", "q1", || bitset_with(&[1], 8));
        cache.get_or_compute("seg0", "q2", || bitset_with(&[2], 8));
        cache.get_or_compute("seg1", "q1", || bitset_with(&[3], 8));
        assert_eq!(cache.len(), 3);

        let removed = cache.invalidate_segment(&"seg0");
        assert_eq!(removed, 2);
        assert_eq!(cache.len(), 1);

        // seg1's entry survives untouched.
        let calls = Cell::new(0);
        let hit = cache.get_or_compute("seg1", "q1", || {
            calls.set(calls.get() + 1);
            bitset_with(&[99], 8)
        });
        assert_eq!(bit_indices(&hit), vec![3]);
        assert_eq!(calls.get(), 0);

        // seg0's entries were genuinely evicted, not just marked -- both
        // recompute on next access.
        let calls_q1 = Cell::new(0);
        cache.get_or_compute("seg0", "q1", || {
            calls_q1.set(calls_q1.get() + 1);
            bitset_with(&[1], 8)
        });
        assert_eq!(calls_q1.get(), 1);
    }

    #[test]
    fn zero_max_entries_never_actually_caches() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(0);
        let calls = Cell::new(0);
        let compute = || {
            calls.set(calls.get() + 1);
            bitset_with(&[1], 8)
        };
        cache.get_or_compute("seg0", "q1", compute);
        cache.get_or_compute("seg0", "q1", compute);
        assert_eq!(
            calls.get(),
            2,
            "every call recomputes with a zero-sized cache"
        );
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn clear_removes_every_entry_regardless_of_segment() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(8);
        cache.get_or_compute("seg0", "q1", || bitset_with(&[1], 8));
        cache.get_or_compute("seg1", "q1", || bitset_with(&[2], 8));
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn max_entries_of_one_keeps_only_the_most_recent_query() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(1);
        cache.get_or_compute("seg0", "q1", || bitset_with(&[1], 8));
        assert_eq!(cache.len(), 1);
        cache.get_or_compute("seg0", "q2", || bitset_with(&[2], 8));
        assert_eq!(
            cache.len(),
            1,
            "inserting a second query must evict the first"
        );

        let calls = Cell::new(0);
        let bits = cache.get_or_compute("seg0", "q1", || {
            calls.set(calls.get() + 1);
            bitset_with(&[1], 8)
        });
        assert_eq!(calls.get(), 1, "q1 was evicted, so it must recompute");
        assert_eq!(bit_indices(&bits), vec![1]);
    }

    #[test]
    fn evict_lru_on_an_empty_cache_is_a_documented_no_op() {
        let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
        assert!(cache.is_empty());
        cache.evict_lru();
        assert!(
            cache.is_empty(),
            "evicting from an empty cache must not panic"
        );
    }

    #[test]
    fn term_query_works_as_a_cache_key() {
        // Confirms `TermQuery`'s new `Hash` derive actually makes it usable
        // as this cache's `Q` type parameter, the concrete query type this
        // task calls out in its design rationale.
        use crate::query::TermQuery;

        let mut cache: QueryCache<u64, TermQuery> = QueryCache::new(4);
        let q1 = TermQuery::new("body", "cat");
        let q2 = TermQuery::new("body", "dog");

        let a = cache.get_or_compute(1, q1.clone(), || bitset_with(&[1], 8));
        let b = cache.get_or_compute(1, q2, || bitset_with(&[2], 8));
        assert_eq!(bit_indices(&a), vec![1]);
        assert_eq!(bit_indices(&b), vec![2]);

        let calls = Cell::new(0);
        let hit = cache.get_or_compute(1, q1, || {
            calls.set(calls.get() + 1);
            bitset_with(&[99], 8)
        });
        assert_eq!(bit_indices(&hit), vec![1]);
        assert_eq!(calls.get(), 0);
    }

    // -- `search_term_query_cached` tests -----------------------------------
    //
    // Same real checked-in fixture (`fixtures/data/blocktree_index/`)
    // `lib.rs`'s own `search_term_query` unit tests open -- see this crate's
    // `test-coverage` skill note (a real fixture beats a hand-built one
    // wherever one is already available). `lib.rs`'s `open_fixture` helper is
    // private to that module's own `#[cfg(test)]` block, so this module
    // duplicates the same small amount of fixture-opening logic rather than
    // exposing test-only plumbing across module boundaries.

    struct FixtureSegment {
        fields: BlockTreeFields,
        doc: Vec<u8>,
        id: [u8; 16],
        suffix: String,
        num_docs: usize,
    }

    impl FixtureSegment {
        fn doc_input(&self) -> DocInput<'_> {
            DocInput::open(&self.doc, &self.id, &self.suffix).expect("open .doc")
        }
    }

    fn open_fixture() -> FixtureSegment {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTree)");
        let get = |key: &str| -> String {
            manifest
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };
        let id_hex = get("id_hex");
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = get("segment_suffix");
        let max_doc: i32 = get("max_doc").parse().unwrap();

        let read_raw = |name: &str| -> Vec<u8> {
            std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
        };
        let fnm = read_raw(&get("fnm_file_name"));
        let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");
        let tim = read_raw(&get("tim_file_name"));
        let tip = read_raw(&get("tip_file_name"));
        let tmd = read_raw(&get("tmd_file_name"));
        let fields =
            lucene_codecs::blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
                .expect("open blocktree");
        let doc = read_raw(&get("doc_file_name"));
        FixtureSegment {
            fields,
            doc,
            id,
            suffix,
            num_docs: max_doc as usize,
        }
    }

    /// Expected doc list for `query` in `segment`, via the plain, uncached
    /// [`crate::search_term_query`] -- the ground truth `search_term_query_cached`'s
    /// output (on both a miss and a hit) must match exactly.
    fn uncached_docs(segment: &FixtureSegment, query: &TermQuery) -> Vec<i32> {
        let mut collector = VecCollector::default();
        crate::search_term_query(
            &segment.fields,
            Some(&segment.doc_input()),
            None,
            query,
            &mut collector,
        )
        .unwrap();
        collector.docs
    }

    /// Runs `query` through `search_term_query_cached` against `segment`
    /// under cache key `seg_key`, passing `.doc` input only when
    /// `with_doc_in` is `true`. Both `"cat"` and `"dog"` in the fixture have
    /// `docFreq == 2` (see `fixtures/data/blocktree_index/manifest.properties`),
    /// so a *genuine* execution of `search_term_query` without `.doc` input
    /// for either term always fails with `Error::BlockTree` (same as
    /// `lib.rs`'s own `multi_doc_term_without_doc_input_is_an_error` unit
    /// test) -- `with_doc_in: false` is therefore this test module's actual
    /// proof mechanism: an `Ok` result with `with_doc_in: false` is only
    /// possible if `search_term_query_cached` served a previously cached
    /// entry and never called `search_term_query` at all, while an `Err`
    /// result proves the opposite -- a real recompute was attempted (and,
    /// lacking `.doc` input, failed).
    fn call_cached(
        cache: &mut QueryCache<&'static str, TermQuery>,
        segment: &FixtureSegment,
        seg_key: &'static str,
        query: &TermQuery,
        with_doc_in: bool,
    ) -> crate::Result<Vec<i32>> {
        let mut collector = VecCollector::default();
        let doc_in = if with_doc_in {
            Some(segment.doc_input())
        } else {
            None
        };
        search_term_query_cached(
            cache,
            seg_key,
            &segment.fields,
            doc_in.as_ref(),
            None,
            segment.num_docs,
            query,
            &mut collector,
        )?;
        Ok(collector.docs)
    }

    #[test]
    fn cached_repeat_call_reuses_cached_result_without_recomputing() {
        // The decisive proof this task asks for: run the *same* query twice
        // against the *same* segment through `search_term_query_cached`, and
        // confirm the second call reused the cache rather than re-running
        // `search_term_query`. The mechanism: the first call supplies `.doc`
        // input and populates the cache with the correct result; the
        // *second*, identical call deliberately omits `.doc` input. `"cat"`
        // has `docFreq == 2` in this fixture, so a genuine re-execution of
        // `search_term_query` without `.doc` input would fail with
        // `Error::BlockTree` (see `call_cached`'s doc comment) -- the second
        // call succeeding, and returning the exact same correct doc IDs,
        // is only possible because it was served from the cache and never
        // called `search_term_query` again.
        let segment = open_fixture();
        let query = TermQuery::new("body", "cat");
        let expected = uncached_docs(&segment, &query);
        assert_eq!(expected, vec![0, 2], "sanity-check the fixture's own data");

        let mut cache: QueryCache<&'static str, TermQuery> = QueryCache::new(4);

        let first = call_cached(&mut cache, &segment, "seg-a", &query, true).unwrap();
        assert_eq!(
            first, expected,
            "first (miss) call must return the correct doc IDs"
        );
        assert_eq!(cache.len(), 1);

        let second = call_cached(&mut cache, &segment, "seg-a", &query, false).expect(
            "a genuine recompute without .doc input would error -- this must be a cache hit",
        );
        assert_eq!(
            second, expected,
            "cache hit must return the exact same correct doc IDs, without ever needing .doc input"
        );
        assert_eq!(
            cache.len(),
            1,
            "still exactly one entry -- no re-insertion happened"
        );
    }

    #[test]
    fn cached_different_query_same_segment_is_a_fresh_miss() {
        let segment = open_fixture();
        let cat = TermQuery::new("body", "cat");
        let dog = TermQuery::new("body", "dog");
        let expected_cat = uncached_docs(&segment, &cat);
        let expected_dog = uncached_docs(&segment, &dog);
        assert_ne!(
            expected_cat, expected_dog,
            "the two queries must have genuinely distinct results for this test to mean anything"
        );

        let mut cache: QueryCache<&'static str, TermQuery> = QueryCache::new(4);

        let a = call_cached(&mut cache, &segment, "seg-a", &cat, true).unwrap();
        assert_eq!(a, expected_cat);
        assert_eq!(cache.len(), 1);

        // "dog" has never been cached under "seg-a" -- calling without .doc
        // input proves this is a genuine miss (it must fail, since a real
        // recompute of a docFreq == 2 term without .doc input always fails).
        let err = call_cached(&mut cache, &segment, "seg-a", &dog, false).unwrap_err();
        assert!(matches!(err, crate::Error::BlockTree(_)));

        // With .doc input supplied, the same miss succeeds and gets cached.
        let b = call_cached(&mut cache, &segment, "seg-a", &dog, true).unwrap();
        assert_eq!(
            b, expected_dog,
            "a different query against the same segment must recompute the correct result"
        );
        assert_eq!(cache.len(), 2);

        // Now cached: a repeat call for "dog" without .doc input succeeds.
        let c = call_cached(&mut cache, &segment, "seg-a", &dog, false).unwrap();
        assert_eq!(c, expected_dog);
    }

    #[test]
    fn cached_same_query_different_segment_key_is_a_fresh_miss() {
        let segment = open_fixture();
        let query = TermQuery::new("body", "cat");
        let expected = uncached_docs(&segment, &query);

        let mut cache: QueryCache<&'static str, TermQuery> = QueryCache::new(4);

        let a = call_cached(&mut cache, &segment, "seg-a", &query, true).unwrap();
        assert_eq!(a, expected);
        assert_eq!(cache.len(), 1);

        // Same query, but a different segment key -- even though it happens
        // to point at the same underlying fixture data in this test, the
        // cache has no way to know that (and real distinct segments would
        // have distinct data), so this must be treated as a fresh miss: no
        // .doc input for "seg-b" must fail, proving it isn't accidentally
        // served from "seg-a"'s cached entry.
        let err = call_cached(&mut cache, &segment, "seg-b", &query, false).unwrap_err();
        assert!(matches!(err, crate::Error::BlockTree(_)));

        let b = call_cached(&mut cache, &segment, "seg-b", &query, true).unwrap();
        assert_eq!(b, expected);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cached_invalidate_segment_forces_a_fresh_recompute() {
        let segment = open_fixture();
        let query = TermQuery::new("body", "cat");
        let expected = uncached_docs(&segment, &query);

        let mut cache: QueryCache<&'static str, TermQuery> = QueryCache::new(4);

        let a = call_cached(&mut cache, &segment, "seg-a", &query, true).unwrap();
        assert_eq!(a, expected);

        // Repeat call before invalidation: still a hit, no .doc input needed.
        let b = call_cached(&mut cache, &segment, "seg-a", &query, false).unwrap();
        assert_eq!(b, expected);

        let removed = cache.invalidate_segment(&"seg-a");
        assert_eq!(removed, 1);

        // After invalidate_segment, the entry is genuinely gone: calling
        // without .doc input must now fail, proving a real recompute was
        // attempted rather than serving stale cached data.
        let err = call_cached(&mut cache, &segment, "seg-a", &query, false).unwrap_err();
        assert!(matches!(err, crate::Error::BlockTree(_)));

        let c = call_cached(&mut cache, &segment, "seg-a", &query, true).unwrap();
        assert_eq!(
            c, expected,
            "supplying .doc input again after invalidation must recompute the correct result"
        );
    }
}
