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
//! **Not wired into [`crate::directory_reader`]/[`crate::multi_segment`]'s
//! live search path yet.** `IndexSearcher`-equivalent query execution in this
//! port still always re-evaluates a query's scorer/matcher on every call, the
//! same way it did before this module existed. Integrating this cache into
//! that path (with real invalidation triggered by segment open/close, and a
//! cache-worthiness heuristic like real Lucene's `shouldCache`) is future
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
//! - Wiring into `IndexSearcher`-equivalent live query execution (see this
//!   module's top doc comment).

use std::collections::HashMap;
use std::hash::Hash;

use lucene_util::fixed_bit_set::FixedBitSet;

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
}
