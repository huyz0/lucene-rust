#![cfg(test)]
use super::*;
use std::cell::Cell;
use std::convert::Infallible;

/// A tiny bitset with exactly the bits in `docs` set, wrapped in the
/// `Result` [`QueryCache::get_or_compute`]'s `compute` closure returns --
/// used throughout these tests as a stand-in for "the result of actually
/// running a query's matcher". The specific bit pattern doesn't matter for
/// cache correctness, only that distinct computations produce
/// distinguishable results.
fn bitset_with(docs: &[usize], num_bits: usize) -> Result<FixedBitSet, Infallible> {
    Ok(raw_bitset(docs, num_bits))
}

fn raw_bitset(docs: &[usize], num_bits: usize) -> FixedBitSet {
    let mut bits = FixedBitSet::new(num_bits);
    for &d in docs {
        bits.set(d);
    }
    bits
}

/// The documents in a cached set, whichever representation it took.
fn docs_of(set: &CachedDocIdSet) -> Vec<usize> {
    set.iter().map(|d| d as usize).collect()
}

#[test]
fn miss_computes_and_stores() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
    let calls = Cell::new(0);
    let set = cache
        .get_or_compute("seg0", "q1", || {
            calls.set(calls.get() + 1);
            bitset_with(&[1, 3, 5], 8)
        })
        .unwrap();
    assert_eq!(docs_of(&set), vec![1, 3, 5]);
    assert_eq!(calls.get(), 1);
    assert_eq!(cache.len(), 1);
    assert_eq!((cache.hit_count(), cache.miss_count()), (0, 1));
}

#[test]
fn hit_reuses_without_recomputing() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
    let calls = Cell::new(0);
    let compute = || {
        calls.set(calls.get() + 1);
        bitset_with(&[2, 4], 8)
    };

    let first = cache.get_or_compute("seg0", "q1", compute).unwrap();
    let second = cache.get_or_compute("seg0", "q1", compute).unwrap();

    assert_eq!(docs_of(&first), docs_of(&second));
    // `compute` only actually ran on the first, miss call.
    assert_eq!(calls.get(), 1);
    assert_eq!(cache.len(), 1);
    assert_eq!((cache.hit_count(), cache.miss_count()), (1, 1));
}

/// A hit must hand back a *share* of the cached set, not a copy of it --
/// the whole reason a hit is cheaper than a recompute. `Arc::ptr_eq` is
/// the observable form of that: two hits on the same key point at one
/// allocation.
#[test]
fn a_hit_shares_the_cached_set_rather_than_copying_it() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
    let first = cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1, 2, 3], 4096))
        .unwrap();
    let second = cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1, 2, 3], 4096))
        .unwrap();
    assert!(
        Arc::ptr_eq(&first, &second),
        "a cache hit must not deep-copy the cached doc set"
    );
}

#[test]
fn distinct_queries_same_segment_are_distinct_entries() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
    let a = cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1], 8))
        .unwrap();
    let b = cache
        .get_or_compute("seg0", "q2", || bitset_with(&[2], 8))
        .unwrap();

    assert_eq!(docs_of(&a), vec![1]);
    assert_eq!(docs_of(&b), vec![2]);
    assert_eq!(cache.len(), 2);
}

#[test]
fn same_query_distinct_segments_are_distinct_entries() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
    let a = cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1], 8))
        .unwrap();
    let b = cache
        .get_or_compute("seg1", "q1", || bitset_with(&[2], 8))
        .unwrap();

    assert_eq!(docs_of(&a), vec![1]);
    assert_eq!(docs_of(&b), vec![2]);
    assert_eq!(cache.len(), 2);

    // Confirm each segment's entry is independently still a cache hit
    // (not accidentally aliased to the other segment's entry).
    let calls = Cell::new(0);
    let hit = cache
        .get_or_compute("seg0", "q1", || {
            calls.set(calls.get() + 1);
            bitset_with(&[99], 8)
        })
        .unwrap();
    assert_eq!(docs_of(&hit), vec![1]);
    assert_eq!(calls.get(), 0);
}

#[test]
fn eviction_removes_least_recently_used_entry() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(2);
    cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1], 8))
        .unwrap();
    cache
        .get_or_compute("seg0", "q2", || bitset_with(&[2], 8))
        .unwrap();
    assert_eq!(cache.len(), 2);

    // Touch q1 again so q2 becomes the least-recently-used entry.
    cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1], 8))
        .unwrap();

    // Inserting a third distinct entry must evict q2 (LRU), not q1.
    cache
        .get_or_compute("seg0", "q3", || bitset_with(&[3], 8))
        .unwrap();
    assert_eq!(cache.len(), 2);

    let calls_q1 = Cell::new(0);
    let q1 = cache
        .get_or_compute("seg0", "q1", || {
            calls_q1.set(calls_q1.get() + 1);
            bitset_with(&[1], 8)
        })
        .unwrap();
    assert_eq!(docs_of(&q1), vec![1]);
    assert_eq!(calls_q1.get(), 0, "q1 should still be cached, not evicted");

    let calls_q2 = Cell::new(0);
    let q2 = cache
        .get_or_compute("seg0", "q2", || {
            calls_q2.set(calls_q2.get() + 1);
            bitset_with(&[2], 8)
        })
        .unwrap();
    assert_eq!(docs_of(&q2), vec![2]);
    assert_eq!(
        calls_q2.get(),
        1,
        "q2 should have been evicted and recomputed"
    );
}

#[test]
fn no_hits_degrades_to_pure_fifo_eviction_order() {
    // Sanity check: with zero cache hits between inserts, "least recently
    // used" and "oldest inserted" coincide -- eviction order must be
    // exactly insertion order once the cache is at capacity and nothing
    // has ever been re-accessed.
    //
    // Note: probing whether a query is still cached is itself a
    // `get_or_compute` call, which would touch that entry's recency and
    // perturb the very order under test -- so this test only ever probes
    // *after* all insertions are done.
    let mut cache: QueryCache<&str, &str> = QueryCache::new(2);
    cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1], 8))
        .unwrap();
    cache
        .get_or_compute("seg0", "q2", || bitset_with(&[2], 8))
        .unwrap();
    assert_eq!(cache.len(), 2);

    cache
        .get_or_compute("seg0", "q3", || bitset_with(&[3], 8))
        .unwrap();
    assert_eq!(cache.len(), 2);
    cache
        .get_or_compute("seg0", "q4", || bitset_with(&[4], 8))
        .unwrap();
    assert_eq!(cache.len(), 2);

    let calls_q1 = Cell::new(0);
    cache
        .get_or_compute("seg0", "q1", || {
            calls_q1.set(calls_q1.get() + 1);
            bitset_with(&[1], 8)
        })
        .unwrap();
    assert_eq!(calls_q1.get(), 1, "q1 (oldest) should have been evicted");

    let calls_q2 = Cell::new(0);
    cache
        .get_or_compute("seg0", "q2", || {
            calls_q2.set(calls_q2.get() + 1);
            bitset_with(&[2], 8)
        })
        .unwrap();
    assert_eq!(
        calls_q2.get(),
        1,
        "q2 (second oldest) should have been evicted next, in insertion order"
    );
}

#[test]
fn invalidate_segment_removes_only_that_segments_entries() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(8);
    cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1], 8))
        .unwrap();
    cache
        .get_or_compute("seg0", "q2", || bitset_with(&[2], 8))
        .unwrap();
    cache
        .get_or_compute("seg1", "q1", || bitset_with(&[3], 8))
        .unwrap();
    assert_eq!(cache.len(), 3);

    let removed = cache.invalidate_segment(&"seg0");
    assert_eq!(removed, 2);
    assert_eq!(cache.len(), 1);

    // seg1's entry survives untouched.
    let calls = Cell::new(0);
    let hit = cache
        .get_or_compute("seg1", "q1", || {
            calls.set(calls.get() + 1);
            bitset_with(&[99], 8)
        })
        .unwrap();
    assert_eq!(docs_of(&hit), vec![3]);
    assert_eq!(calls.get(), 0);

    // seg0's entries were genuinely evicted, not just marked.
    let calls_q1 = Cell::new(0);
    cache
        .get_or_compute("seg0", "q1", || {
            calls_q1.set(calls_q1.get() + 1);
            bitset_with(&[1], 8)
        })
        .unwrap();
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
    cache.get_or_compute("seg0", "q1", compute).unwrap();
    cache.get_or_compute("seg0", "q1", compute).unwrap();
    assert_eq!(
        calls.get(),
        2,
        "every call recomputes with a zero-sized cache"
    );
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.ram_bytes_used(), 0);
}

#[test]
fn clear_removes_every_entry_regardless_of_segment() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(8);
    cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1], 8))
        .unwrap();
    cache
        .get_or_compute("seg1", "q1", || bitset_with(&[2], 8))
        .unwrap();
    assert_eq!(cache.len(), 2);
    cache.clear();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.ram_bytes_used(), 0);
}

#[test]
fn max_entries_of_one_keeps_only_the_most_recent_query() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(1);
    cache
        .get_or_compute("seg0", "q1", || bitset_with(&[1], 8))
        .unwrap();
    assert_eq!(cache.len(), 1);
    cache
        .get_or_compute("seg0", "q2", || bitset_with(&[2], 8))
        .unwrap();
    assert_eq!(
        cache.len(),
        1,
        "inserting a second query must evict the first"
    );

    let calls = Cell::new(0);
    let set = cache
        .get_or_compute("seg0", "q1", || {
            calls.set(calls.get() + 1);
            bitset_with(&[1], 8)
        })
        .unwrap();
    assert_eq!(calls.get(), 1, "q1 was evicted, so it must recompute");
    assert_eq!(docs_of(&set), vec![1]);
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
    use crate::query::TermQuery;

    let mut cache: QueryCache<u64, TermQuery> = QueryCache::new(4);
    let q1 = TermQuery::new("body", "cat");
    let q2 = TermQuery::new("body", "dog");

    let a = cache
        .get_or_compute(1, q1.clone(), || bitset_with(&[1], 8))
        .unwrap();
    let b = cache
        .get_or_compute(1, q2, || bitset_with(&[2], 8))
        .unwrap();
    assert_eq!(docs_of(&a), vec![1]);
    assert_eq!(docs_of(&b), vec![2]);

    let calls = Cell::new(0);
    let hit = cache
        .get_or_compute(1, q1, || {
            calls.set(calls.get() + 1);
            bitset_with(&[99], 8)
        })
        .unwrap();
    assert_eq!(docs_of(&hit), vec![1]);
    assert_eq!(calls.get(), 0);
}

/// A failed computation must leave the cache exactly as it was -- no
/// placeholder entry, no accounting drift -- so the next attempt is a
/// genuine miss that recomputes rather than a hit on an empty set.
#[test]
fn a_failed_computation_stores_nothing() {
    let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
    let err: Result<Arc<CachedDocIdSet>, &str> =
        cache.get_or_compute("seg0", "q1", || Err("decode failed"));
    assert_eq!(err.err(), Some("decode failed"));
    assert!(cache.is_empty());
    assert_eq!(cache.ram_bytes_used(), 0);

    let calls = Cell::new(0);
    cache
        .get_or_compute("seg0", "q1", || {
            calls.set(calls.get() + 1);
            bitset_with(&[1], 8)
        })
        .unwrap();
    assert_eq!(calls.get(), 1, "the failed key must still be a miss");
}

// -- RAM accounting (`maxRamBytesUsed`) ---------------------------------

/// `LRUQueryCachePartition.requiresEviction`'s second clause: the cache
/// evicts on *memory*, not just entry count. Without this, one query over
/// a huge segment can hold megabytes that `max_entries` never notices.
#[test]
fn ram_bound_evicts_even_when_the_entry_count_bound_is_far_from_reached() {
    // Dense sets over a 64k-doc segment: 8 KB of words each.
    let dense: Vec<usize> = (0..65_536).step_by(2).collect();
    let one_entry_bytes = CachedDocIdSet::from_bitset(raw_bitset(&dense, 65_536)).ram_bytes_used();
    // Room for two entries, not three.
    let mut cache: QueryCache<&str, u32> =
        QueryCache::with_ram_limit(1000, one_entry_bytes * 2 + 1);

    for q in 0..3u32 {
        cache
            .get_or_compute("seg0", q, || bitset_with(&dense, 65_536))
            .unwrap();
    }
    assert_eq!(cache.len(), 2, "the RAM bound, not max_entries, evicted");
    assert!(cache.ram_bytes_used() <= one_entry_bytes * 2 + 1);
    assert_eq!(cache.eviction_count(), 1);

    // The evicted one is the oldest.
    let calls = Cell::new(0);
    cache
        .get_or_compute("seg0", 0u32, || {
            calls.set(calls.get() + 1);
            bitset_with(&dense, 65_536)
        })
        .unwrap();
    assert_eq!(calls.get(), 1);
}

/// RAM accounting must stay exact across every path that removes an
/// entry: eviction, targeted removal, and per-segment invalidation. A
/// drift here would eventually wedge the cache into evicting forever (or
/// never).
#[test]
fn ram_accounting_returns_to_zero_through_every_removal_path() {
    let mut cache: QueryCache<&str, u32> = QueryCache::new(16);
    for q in 0..4u32 {
        cache
            .get_or_compute(if q % 2 == 0 { "seg0" } else { "seg1" }, q, || {
                bitset_with(&[1, 2, 3], 1024)
            })
            .unwrap();
    }
    assert!(cache.ram_bytes_used() > 0);
    assert!(cache.remove(&"seg0", &0));
    assert!(!cache.remove(&"seg0", &0), "removing twice is a no-op");
    assert_eq!(cache.invalidate_segment(&"seg1"), 2);
    assert_eq!(cache.len(), 1);
    assert!(cache.remove(&"seg0", &2));
    assert_eq!(cache.len(), 0);
    assert_eq!(
        cache.ram_bytes_used(),
        0,
        "every removal path must give its bytes back"
    );
}

/// A single entry larger than the whole RAM budget is inserted and then
/// immediately evicted -- Java's insert-then-`evictIfNecessary` order,
/// which keeps accounting exact rather than silently bypassing the bound.
#[test]
fn an_entry_bigger_than_the_whole_budget_does_not_stay() {
    let mut cache: QueryCache<&str, &str> = QueryCache::with_ram_limit(10, 8);
    cache
        .get_or_compute("seg0", "q1", || {
            bitset_with(&(0..4096).collect::<Vec<_>>(), 65_536)
        })
        .unwrap();
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.ram_bytes_used(), 0);
    assert_eq!(cache.cache_count(), 1);
    assert_eq!(cache.eviction_count(), 1);
}

/// The representation stored is `LRUQueryCache.cacheImpl`'s choice, and it
/// is what makes the memory bound meaningful: a sparse match set over a
/// large segment must not be charged `maxDoc / 8` bytes.
#[test]
fn a_sparse_match_set_is_cached_as_roaring_not_as_a_full_bitset() {
    let max_doc = 1_000_000;
    let docs: Vec<usize> = (0..50).map(|i| i * 19_997).collect();
    let mut cache: QueryCache<&str, &str> = QueryCache::new(4);
    let set = cache
        .get_or_compute("seg0", "q1", || bitset_with(&docs, max_doc))
        .unwrap();
    assert!(matches!(*set, CachedDocIdSet::Roaring(_)));
    assert_eq!(docs_of(&set), docs);
    assert!(
        cache.ram_bytes_used() * 20 < max_doc / 8,
        "sparse entry charged {} bytes, a bitset would be {}",
        cache.ram_bytes_used(),
        max_doc / 8
    );
}

// -- `UsageTrackingQueryCachingPolicy` ----------------------------------

/// A test query type that lets each of Java's three shape predicates be
/// exercised independently -- `TermQuery` can only ever demonstrate the
/// `shouldNeverCache` branch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ShapedQuery {
    name: &'static str,
    costly: bool,
    composite: bool,
}

impl CachingCost for ShapedQuery {
    fn is_costly(&self) -> bool {
        self.costly
    }
    fn is_composite(&self) -> bool {
        self.composite
    }
}

fn shaped(name: &'static str, costly: bool, composite: bool) -> ShapedQuery {
    ShapedQuery {
        name,
        costly,
        composite,
    }
}

/// Java's `minFrequencyToCache`: 2 observations for a costly query, 4 for
/// a composite one, 5 otherwise. The counts are the point of the class, so
/// pin each one at its exact boundary.
#[test]
fn policy_caches_only_after_javas_per_shape_observation_count() {
    for (costly, composite, min_frequency) in
        [(true, false, 2u32), (false, true, 4), (false, false, 5)]
    {
        let query = shaped("q", costly, composite);
        let mut policy = UsageTrackingPolicy::default();
        for seen in 1..min_frequency {
            policy.on_use(&query);
            assert!(
                !policy.should_cache(&query),
                "costly={costly} composite={composite}: cached after only {seen} uses"
            );
        }
        policy.on_use(&query);
        assert_eq!(policy.frequency(&query), min_frequency);
        assert!(
            policy.should_cache(&query),
            "costly={costly} composite={composite}: not cached after {min_frequency} uses"
        );
    }
}

/// `shouldNeverCache`: Java refuses to cache a `TermQuery` at all,
/// however often it is used, because re-running one is already cheap.
#[test]
fn policy_never_caches_a_term_query_however_often_it_is_used() {
    let query = TermQuery::new("body", "cat");
    let mut policy = UsageTrackingPolicy::default();
    for _ in 0..100 {
        policy.on_use(&query);
    }
    assert!(!policy.should_cache(&query));
    assert_eq!(
        policy.frequency(&query),
        0,
        "a never-cached query is not even tracked"
    );
}

/// The history is a *window*: a query used long ago, then displaced by
/// enough newer traffic, drops back out of the cacheable set. This is what
/// stops a one-off burst from pinning something in the cache forever.
#[test]
fn policy_history_window_forgets_displaced_queries() {
    let query = shaped("hot", false, false);
    let mut policy = UsageTrackingPolicy::with_history_size(8);
    for _ in 0..5 {
        policy.on_use(&query);
    }
    assert!(policy.should_cache(&query));
    for i in 0..8 {
        // Distinct names, so each is a distinct hash filling the window.
        policy.on_use(&shaped(
            ["a", "b", "c", "d", "e", "f", "g", "h"][i],
            false,
            false,
        ));
    }
    assert_eq!(policy.frequency(&query), 0);
    assert!(!policy.should_cache(&query));
}

/// The policy gate must actually stop the *storing*, not just report a
/// verdict: an under-used query is recomputed every time, and the moment
/// it crosses the threshold it starts being served from the cache.
#[test]
fn get_or_compute_with_policy_only_stores_once_the_policy_agrees() {
    let mut cache: QueryCache<&str, ShapedQuery> = QueryCache::new(8);
    let mut policy = UsageTrackingPolicy::default();
    let query = shaped("filter", false, false);
    let calls = Cell::new(0);

    // Java's threshold for this shape is 5 *executions*. `on_use` is the
    // caller's to call, once per execution across every leaf -- see
    // `get_or_compute_with_policy`.
    for _ in 0..4 {
        policy.on_use(&query);
        cache
            .get_or_compute_with_policy(&mut policy, "seg0", query.clone(), || {
                calls.set(calls.get() + 1);
                bitset_with(&[1, 2], 64)
            })
            .unwrap();
    }
    assert_eq!(calls.get(), 4, "nothing cached below the threshold");
    assert!(cache.is_empty());

    policy.on_use(&query);
    cache
        .get_or_compute_with_policy(&mut policy, "seg0", query.clone(), || {
            calls.set(calls.get() + 1);
            bitset_with(&[1, 2], 64)
        })
        .unwrap();
    assert_eq!(calls.get(), 5, "the fifth use still computes, but stores");
    assert_eq!(cache.len(), 1);

    policy.on_use(&query);
    let hit = cache
        .get_or_compute_with_policy(&mut policy, "seg0", query, || {
            calls.set(calls.get() + 1);
            bitset_with(&[9], 64)
        })
        .unwrap();
    assert_eq!(calls.get(), 5, "the sixth use is a hit");
    assert_eq!(docs_of(&hit), vec![1, 2]);
}

/// `on_use` is deliberately *not* called by
/// `get_or_compute_with_policy` -- Java guards its own call with
/// `used.compareAndSet(false, true)` so a query counts once per execution,
/// not once per leaf. Searching many leaves must therefore not pull a
/// query over the threshold by itself.
#[test]
fn searching_many_leaves_does_not_inflate_a_querys_tracked_frequency() {
    let mut cache: QueryCache<u32, ShapedQuery> = QueryCache::new(64);
    let mut policy = UsageTrackingPolicy::default();
    let query = shaped("filter", false, false);

    policy.on_use(&query); // one execution...
    for leaf in 0..20u32 {
        cache
            .get_or_compute_with_policy(&mut policy, leaf, query.clone(), || bitset_with(&[1], 64))
            .unwrap();
    }
    assert_eq!(
        policy.frequency(&query),
        1,
        "20 leaves is still one execution"
    );
    assert!(cache.is_empty(), "one execution is below the threshold");
}

/// A `TermQuery` routed through the policy is never stored, matching
/// Java -- which is precisely the divergence `search_term_query_cached`
/// (which caches unconditionally) makes deliberately.
#[test]
fn get_or_compute_with_policy_never_stores_a_term_query() {
    let mut cache: QueryCache<&str, TermQuery> = QueryCache::new(8);
    let mut policy = UsageTrackingPolicy::default();
    let query = TermQuery::new("body", "cat");
    for _ in 0..20 {
        policy.on_use(&query);
        cache
            .get_or_compute_with_policy(&mut policy, "seg0", query.clone(), || {
                bitset_with(&[1], 64)
            })
            .unwrap();
    }
    assert!(cache.is_empty());
    assert_eq!(cache.hit_count(), 0);
}

/// A query that overrides nothing exercises `CachingCost`'s defaults --
/// ordinary, non-costly, non-composite -- which is the shape most queries
/// have and which the deliberately-configurable `ShapedQuery` above never
/// reaches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlainQuery;

impl CachingCost for PlainQuery {}

#[test]
fn a_query_with_default_shape_uses_javas_ordinary_threshold_of_five() {
    let mut policy = UsageTrackingPolicy::with_history_size(0);
    // A zero-sized history is clamped to one slot, so the query is never
    // seen more than once and never reaches the threshold.
    for _ in 0..10 {
        policy.on_use(&PlainQuery);
    }
    assert_eq!(policy.frequency(&PlainQuery), 1);
    assert!(!policy.should_cache(&PlainQuery));

    let mut policy = UsageTrackingPolicy::default();
    for _ in 0..4 {
        policy.on_use(&PlainQuery);
    }
    assert!(!policy.should_cache(&PlainQuery));
    policy.on_use(&PlainQuery);
    assert!(policy.should_cache(&PlainQuery));
}

// -- `MinSegmentSizePredicate` ------------------------------------------

/// `LRUQueryCache`'s default leaf filter: at least 10000 documents, and at
/// least half the reader's average leaf size.
#[test]
fn leaf_size_predicate_matches_javas_two_conditions() {
    // Below the absolute minimum: never cached, however large the average.
    assert!(!leaf_is_worth_caching(
        9_999,
        9_999,
        1,
        DEFAULT_MIN_LEAF_SIZE
    ));
    // Big enough and it *is* the whole reader.
    assert!(leaf_is_worth_caching(
        10_000,
        10_000,
        1,
        DEFAULT_MIN_LEAF_SIZE
    ));
    // Big enough in absolute terms but a runt next to its siblings:
    // average is 1_000_000 / 4 = 250_000, and 10_000 * 2 <= 250_000.
    assert!(!leaf_is_worth_caching(
        10_000,
        1_000_000,
        4,
        DEFAULT_MIN_LEAF_SIZE
    ));
    // Exactly half the average is still rejected (`maxDoc * 2 > average`).
    assert!(!leaf_is_worth_caching(
        50_000,
        400_000,
        4,
        DEFAULT_MIN_LEAF_SIZE
    ));
    assert!(leaf_is_worth_caching(
        50_001,
        400_000,
        4,
        DEFAULT_MIN_LEAF_SIZE
    ));
    // A reader with no leaves has nothing to cache and must not divide by
    // zero.
    assert!(!leaf_is_worth_caching(10_000, 0, 0, DEFAULT_MIN_LEAF_SIZE));
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
    call_cached_with_live_docs(cache, segment, seg_key, query, with_doc_in, None)
}

fn call_cached_with_live_docs(
    cache: &mut QueryCache<&'static str, TermQuery>,
    segment: &FixtureSegment,
    seg_key: &'static str,
    query: &TermQuery,
    with_doc_in: bool,
    live_docs: Option<&FixedBitSet>,
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
        live_docs,
        segment.num_docs,
        query,
        &mut collector,
    )?;
    Ok(collector.docs)
}

/// `num_docs` and the doc ids the postings walk yields are two independent
/// inputs to this `pub fn`: the caller's claim about the segment, and the
/// segment's own `.doc` file. `FixedBitSet::new(num_docs)` then gets
/// indexed by a doc id nothing relates to it, on a public API. With
/// `num_docs = 1` and a term whose postings name doc 2, an unchecked write
/// would put a ghost bit into word 0 of a one-bit set (the doc it claimed
/// to cache is unrecoverable), and a `num_docs` of 0 would panic outright.
#[test]
fn a_num_docs_below_the_segments_own_doc_ids_is_reported_not_absorbed() {
    let segment = open_fixture();
    let query = TermQuery::new("body", "cat");
    assert_eq!(
        uncached_docs(&segment, &query),
        vec![0, 2],
        "the fixture's own postings name doc 2"
    );

    let mut cache: QueryCache<&'static str, TermQuery> = QueryCache::new(4);
    let mut collector = VecCollector::default();
    let doc_in = segment.doc_input();
    let err = search_term_query_cached(
        &mut cache,
        "seg-a",
        &segment.fields,
        Some(&doc_in),
        None,
        1, // the caller says the segment has one document; it has three
        &query,
        &mut collector,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            Error::CachedDocOutOfRange {
                doc_id: 2,
                num_docs: 1
            }
        ),
        "expected a typed out-of-range report, got: {err}"
    );
    assert_eq!(cache.len(), 0, "a failed compute must not be cached");
}

/// The other half: `live_docs` is a *separate* caller-supplied bitset from
/// the cached set being iterated, so a shorter one must not be indexed by a
/// doc id it does not cover. A doc outside the live-docs bitset is not
/// live -- the same answer `soft_deletes::is_live` and `vector_query`'s
/// accept filter give.
#[test]
fn a_live_docs_bitset_shorter_than_the_cached_set_hides_the_docs_it_does_not_cover() {
    let segment = open_fixture();
    let query = TermQuery::new("body", "cat");
    let mut cache: QueryCache<&'static str, TermQuery> = QueryCache::new(4);

    // One bit long: it covers doc 0 (live) and says nothing about doc 2.
    let mut live = FixedBitSet::new(1);
    live.set(0);
    let docs = call_cached_with_live_docs(&mut cache, &segment, "seg-a", &query, true, Some(&live))
        .expect("a short live-docs bitset must not panic");
    assert_eq!(docs, vec![0]);
}

#[test]
fn cached_repeat_call_reuses_cached_result_without_recomputing() {
    // The decisive proof: run the *same* query twice
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

    let second = call_cached(&mut cache, &segment, "seg-a", &query, false)
        .expect("a genuine recompute without .doc input would error -- this must be a cache hit");
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
/// A cached entry must be **deletion-agnostic**, the way Java's is: both
/// `cacheIntoBitSet` and `cacheIntoRoaringDocIdSet` score with
/// `acceptDocs == null`, and `LRUQueryCache` keys on the *core* cache
/// helper, so an entry survives a new `.liv` generation.
///
/// Baking `live_docs` into the entry instead would be wrong *invisibly*:
/// a hit never looks at `live_docs`, so the second call below would return
/// the first call's undeleted doc set with no way to notice.
#[test]
fn a_cached_entry_is_deletion_agnostic_and_live_docs_apply_on_the_way_out() {
    let segment = open_fixture();
    let query = TermQuery::new("body", "cat");
    let all = uncached_docs(&segment, &query);
    assert!(all.len() >= 2, "the fixture must have >1 hit to delete one");

    let mut cache: QueryCache<&'static str, TermQuery> = QueryCache::new(4);

    // Populate with no deletions.
    let first =
        call_cached_with_live_docs(&mut cache, &segment, "seg-a", &query, true, None).unwrap();
    assert_eq!(first, all);

    // Now the same segment key with the first hit deleted. `.doc` input is
    // withheld, so this can only be served from the cache -- and it must
    // still respect the *new* live docs.
    let mut live = FixedBitSet::new(segment.num_docs);
    for doc in 0..segment.num_docs {
        live.set(doc);
    }
    live.clear(all[0] as usize);
    let second =
        call_cached_with_live_docs(&mut cache, &segment, "seg-a", &query, false, Some(&live))
            .unwrap();
    assert_eq!(
        second,
        all[1..].to_vec(),
        "the cached set must be filtered by the live docs of this call"
    );
    assert_eq!(cache.len(), 1, "and still be one entry, not two");
}
