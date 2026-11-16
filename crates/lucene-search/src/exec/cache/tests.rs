#![cfg(test)]
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
    // A column past the budget is built once, handed back uncached, and
    // not retried.
    let big = MAX_COLUMN_BYTES / 4 + 1;
    assert!(col("g", 20_000, big).unwrap().is_none());
    assert!(col("g", 20_000, big).unwrap().is_some());
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
