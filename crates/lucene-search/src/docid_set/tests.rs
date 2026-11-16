#![cfg(test)]
use super::*;

// -----------------------------------------------------------------
// RoaringDocIdSet / CachedDocIdSet (`org.apache.lucene.util.
// RoaringDocIdSet`, `LRUQueryCache.cacheImpl`)
// -----------------------------------------------------------------

fn roaring_of(docs: &[i32], max_doc: usize) -> RoaringDocIdSet {
    let mut b = RoaringBuilder::new(max_doc);
    for &d in docs {
        b.add(d);
    }
    b.build()
}

fn block_of(set: &RoaringDocIdSet, i: usize) -> &Block {
    &set.blocks[i]
}

/// The whole point of the encoding: every block shape must round-trip the
/// exact doc set it was built from, and `contains` must agree with
/// iteration for every doc in range -- including the docs *between* the
/// matches, which is where an off-by-one in a block boundary would show.
#[test]
fn every_block_encoding_round_trips_and_contains_agrees_with_iteration() {
    let cases: Vec<(&str, Vec<i32>, usize)> = vec![
        ("empty", vec![], 200_000),
        ("single doc", vec![7], 200_000),
        ("sparse", vec![0, 5, 99, 65_535], 200_000),
        (
            "contiguous run (range encoding)",
            (100..350).collect(),
            200_000,
        ),
        (
            "second block only",
            vec![65_536, 65_537, 100_000, 131_071],
            200_000,
        ),
        (
            "straddles a block boundary",
            vec![65_534, 65_535, 65_536, 65_537],
            200_000,
        ),
        (
            "just over the array threshold (dense encoding)",
            (0..(1 << 12) + 1).map(|i| i * 3).collect(),
            200_000,
        ),
        (
            "whole first block set (all encoding)",
            (0..65_536).collect(),
            200_000,
        ),
        (
            "super dense: one hole (inverse encoding)",
            (0..65_536).filter(|d| *d != 4242).collect(),
            200_000,
        ),
    ];
    for (name, docs, max_doc) in cases {
        let set = roaring_of(&docs, max_doc);
        assert_eq!(set.cardinality(), docs.len(), "{name}: cardinality");
        assert_eq!(set.iter().collect::<Vec<_>>(), docs, "{name}: iteration");
        for doc in 0..max_doc as i32 {
            assert_eq!(
                set.contains(doc),
                docs.binary_search(&doc).is_ok(),
                "{name}: contains({doc})"
            );
        }
    }
}

/// The encoding actually chosen per block is Java's, not merely "something
/// that round-trips": `RoaringDocIdSet.Builder.flush`'s five cases, pinned.
/// `add`'s `max_doc` bound is what
/// `append_in_current_block`'s `// FBS:` proof stands on: a trailing
/// partial block's dense bitset is sized `min(BLOCK_SIZE, max_doc -
/// offset)`, so a doc id past `max_doc` indexes a bitset that was never
/// sized for it. With `max_doc = 100` and a doc in block 1, `num_bits`
/// would be `0`.
#[test]
#[should_panic(expected = "doc id 70000 is outside this builder's 0..100 range")]
fn a_doc_id_past_max_doc_is_a_producer_error_not_a_zero_bit_bitset() {
    let mut b = RoaringBuilder::new(100);
    b.add(0);
    b.add(70_000);
}

#[test]
#[should_panic(expected = "doc id -1 is outside this builder's 0..100 range")]
fn a_negative_doc_id_is_a_producer_error() {
    // Ascending against `last_doc_id == -1` is not enough: `-1` is not
    // ascending, but `-1 as u32` would have been `u32::MAX`.
    let mut b = RoaringBuilder::new(100);
    b.add(0);
    b.last_doc_id = -2;
    b.add(-1);
}

#[test]
fn builder_picks_javas_block_encoding_for_each_density() {
    assert_eq!(block_of(&roaring_of(&[], 70_000), 0), &Block::Empty);
    assert_eq!(
        block_of(&roaring_of(&(0..65_536).collect::<Vec<_>>(), 200_000), 0),
        &Block::All
    );
    assert_eq!(
        block_of(&roaring_of(&[10, 11, 12], 70_000), 0),
        &Block::Range { min: 10, max: 12 }
    );
    assert!(matches!(
        block_of(&roaring_of(&[1, 5, 900], 70_000), 0),
        Block::Sparse(_)
    ));
    // MAX_ARRAY_LENGTH docs still fit the short array; one more spills.
    let exactly_max: Vec<i32> = (0..MAX_ARRAY_LENGTH as i32).map(|i| i * 3).collect();
    assert!(matches!(
        block_of(&roaring_of(&exactly_max, 200_000), 0),
        Block::Sparse(_)
    ));
    let one_more: Vec<i32> = (0..MAX_ARRAY_LENGTH as i32 + 1).map(|i| i * 3).collect();
    assert!(matches!(
        block_of(&roaring_of(&one_more, 200_000), 0),
        Block::Dense(_)
    ));
    // A full block missing fewer than MAX_ARRAY_LENGTH docs inverts.
    let almost_all: Vec<i32> = (0..65_536).filter(|d| d % 100 != 0).collect();
    assert!(matches!(
        block_of(&roaring_of(&almost_all, 200_000), 0),
        Block::InverseSparse(_)
    ));
}

/// A short final block can never be `All`/`InverseSparse` (both are
/// defined over a whole 65536-doc block), and must still round-trip.
#[test]
fn trailing_partial_block_round_trips_without_the_full_block_encodings() {
    let max_doc = 65_536 + 10;
    let docs: Vec<i32> = (65_536..65_546).collect();
    let set = roaring_of(&docs, max_doc);
    assert_eq!(set.iter().collect::<Vec<_>>(), docs);
    assert_eq!(
        block_of(&set, 1),
        &Block::Range { min: 0, max: 9 },
        "10 contiguous docs in the trailing block is a range, not All"
    );
}

/// Doc IDs must be added in ascending order, same contract as Java's
/// builder (which throws `IllegalArgumentException`).
#[test]
#[should_panic(expected = "must be added in order")]
fn builder_rejects_out_of_order_docs() {
    let mut b = RoaringBuilder::new(1000);
    b.add(5);
    b.add(5);
}

/// The reason this encoding exists: a sparse match set must not cost
/// `maxDoc / 8` bytes. 100 matches in a 10M-doc segment is 1.25 MB as a
/// bitset; Roaring must be orders of magnitude smaller.
#[test]
fn sparse_set_costs_far_less_than_a_full_bitset() {
    let max_doc = 10_000_000;
    let docs: Vec<i32> = (0..100).map(|i| i * 97_777).collect();
    let mut bits = FixedBitSet::new(max_doc);
    for &d in &docs {
        bits.set(d as usize);
    }
    let bitset_bytes = bits.words().len() * 8;
    let roaring = roaring_of(&docs, max_doc);
    assert!(
        roaring.ram_bytes_used() * 100 < bitset_bytes,
        "roaring {} bytes vs bitset {bitset_bytes} bytes -- expected >100x smaller",
        roaring.ram_bytes_used()
    );
    assert_eq!(roaring.iter().collect::<Vec<_>>(), docs);
}

/// `LRUQueryCache.cacheImpl`'s exact rule: >= 1% density keeps the bitset,
/// below that switches to Roaring. Boundary included, since that is where
/// a `>` vs `>=` slip would hide.
#[test]
fn cached_doc_id_set_picks_javas_representation_at_the_one_percent_boundary() {
    let max_doc = 1000;
    // Exactly 1% (10 of 1000): Java keeps the bitset (`cost * 100 >= maxDoc`).
    let mut at = FixedBitSet::new(max_doc);
    for d in 0..10 {
        at.set(d * 7);
    }
    assert!(matches!(
        CachedDocIdSet::from_bitset(at),
        CachedDocIdSet::Bits { .. }
    ));
    // Just under 1% (9 of 1000): Roaring.
    let mut under = FixedBitSet::new(max_doc);
    for d in 0..9 {
        under.set(d * 7);
    }
    assert!(matches!(
        CachedDocIdSet::from_bitset(under),
        CachedDocIdSet::Roaring(_)
    ));
}

/// Both representations must be interchangeable at the API a consumer
/// actually uses: same docs, same cardinality, same `contains`.
#[test]
fn both_cached_representations_agree_on_contents() {
    let max_doc = 5000;
    for &density in &[1usize, 5, 50, 500, 4999] {
        let docs: Vec<i32> = (0..density).map(|i| (i * 997 % max_doc) as i32).collect();
        let mut sorted = docs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        let mut bits = FixedBitSet::new(max_doc);
        for &d in &sorted {
            bits.set(d as usize);
        }
        let cached = CachedDocIdSet::from_bitset(bits.clone());
        assert_eq!(cached.cardinality(), sorted.len());
        assert_eq!(cached.iter().collect::<Vec<_>>(), sorted);
        for doc in 0..max_doc as i32 {
            assert_eq!(cached.contains(doc), bits.get(doc as usize));
        }
    }
}

/// A bitset-shaped cached set iterates word-at-a-time; an empty one must
/// still terminate, and a negative doc is never contained.
#[test]
fn cached_bitset_edge_cases() {
    let cached = CachedDocIdSet::Bits {
        bits: FixedBitSet::new(0),
        cardinality: 0,
    };
    assert!(cached.is_empty());
    assert_eq!(cached.iter().count(), 0);
    assert!(!cached.contains(-1));
    assert!(!roaring_of(&[], 100).contains(-1));
    assert!(!roaring_of(&[1], 100).contains(1_000_000));
    // A document past `max_doc` but *inside* the last block: the block
    // exists, so the encoding's own bound is what has to catch it. A dense
    // trailing block is the case that would otherwise index past the end
    // of a short bitset.
    let short_dense: Vec<i32> = (0..(1 << 12) + 1).map(|i| i * 2).collect();
    let set = roaring_of(&short_dense, 9000);
    assert!(matches!(block_of(&set, 0), Block::Dense(_)));
    assert!(!set.contains(9000));
    assert!(!set.contains(60_000));
    let mut bits = FixedBitSet::new(100);
    bits.set(3);
    let cached = CachedDocIdSet::Bits {
        bits,
        cardinality: 1,
    };
    assert!(cached.contains(3));
    assert!(!cached.contains(100));
    assert!(!cached.contains(1_000_000));
    assert!(roaring_of(&[], 100).is_empty());
    assert!(CachedDocIdSet::from_bitset(FixedBitSet::new(64)).ram_bytes_used() > 0);
}

/// `ram_bytes_used` must charge every block encoding, not just the sparse
/// one -- the RAM bound `LRUQueryCache` enforces is only as good as the
/// accounting behind it. A dense block is 8 KB of words, an inverse-sparse
/// block is two bytes per *excluded* doc, and the free encodings (all /
/// range / empty) really are free.
#[test]
fn ram_accounting_covers_every_block_encoding() {
    let free = roaring_of(&(0..65_536).collect::<Vec<_>>(), 200_000).ram_bytes_used();
    let range = roaring_of(&(0..100).collect::<Vec<_>>(), 200_000).ram_bytes_used();
    let empty = roaring_of(&[], 200_000).ram_bytes_used();
    assert_eq!(free, empty, "an all-set block carries no payload");
    assert_eq!(range, empty, "a contiguous run carries no payload");

    let dense: Vec<i32> = (0..(1 << 13)).map(|i| i * 4).collect();
    let dense_bytes = roaring_of(&dense, 200_000).ram_bytes_used();
    assert!(
        dense_bytes >= empty + 8192,
        "a dense block must charge its 65536-bit word array, got {dense_bytes}"
    );

    let inverse: Vec<i32> = (0..65_536).filter(|d| d % 100 != 0).collect();
    let inverse_bytes = roaring_of(&inverse, 200_000).ram_bytes_used();
    assert!(
        inverse_bytes < dense_bytes,
        "inverting a super-dense block must be cheaper than storing it, \
             got {inverse_bytes} vs {dense_bytes}"
    );
    assert!(inverse_bytes > empty);
}

/// Block equality is used by the encoding-choice tests, so it has to
/// distinguish encodings rather than only compare within one. The dense arm
/// in particular compares bitset contents by hand, since `FixedBitSet` has
/// no `PartialEq`.
#[test]
fn block_equality_distinguishes_encodings_and_contents() {
    let all = roaring_of(&(0..65_536).collect::<Vec<_>>(), 200_000);
    let range = roaring_of(&(0..100).collect::<Vec<_>>(), 200_000);
    let sparse = roaring_of(&[1, 900], 200_000);
    let other_sparse = roaring_of(&[1, 901], 200_000);
    let dense_a = roaring_of(&(0..(1 << 13)).map(|i| i * 4).collect::<Vec<_>>(), 200_000);
    let dense_b = roaring_of(
        &(0..(1 << 13)).map(|i| i * 4 + 1).collect::<Vec<_>>(),
        200_000,
    );

    assert_ne!(block_of(&all, 0), block_of(&range, 0));
    assert_ne!(block_of(&sparse, 0), block_of(&other_sparse, 0));
    assert_ne!(block_of(&dense_a, 0), block_of(&dense_b, 0));
    assert_eq!(block_of(&dense_a, 0), block_of(&dense_a, 0));
    assert_ne!(block_of(&dense_a, 0), block_of(&sparse, 0));
    assert_eq!(
        block_of(&range, 0),
        &Block::Range { min: 0, max: 99 },
        "same encoding, same contents"
    );
    assert_ne!(block_of(&range, 0), &Block::Range { min: 0, max: 98 });
}

/// A `CachedDocIdSet` in the Roaring shape must answer the same shallow
/// questions the bitset shape does.
#[test]
fn cached_roaring_reports_emptiness_and_cost() {
    let cached = CachedDocIdSet::Roaring(roaring_of(&[], 1_000_000));
    assert!(cached.is_empty());
    assert_eq!(cached.cardinality(), 0);
    assert!(cached.ram_bytes_used() > 0);
    assert!(!cached.contains(0));
}

/// A zero-document segment has no blocks at all: building and iterating one
/// must terminate rather than index an empty block array.
#[test]
fn an_empty_doc_space_builds_an_empty_set() {
    let set = roaring_of(&[], 0);
    assert!(set.is_empty());
    assert_eq!(set.iter().count(), 0);
    assert!(!set.contains(0));
    assert_eq!(
        CachedDocIdSet::from_bitset(FixedBitSet::new(0)).cardinality(),
        0
    );
}

fn boxed(v: Vec<i32>) -> BoxDocIter<'static> {
    Box::new(v.into_iter())
}

fn collect_conjunction(inputs: Vec<Vec<i32>>) -> Vec<i32> {
    Conjunction::new(inputs.into_iter().map(boxed).collect()).collect()
}

fn collect_disjunction(inputs: Vec<Vec<i32>>) -> Vec<i32> {
    Disjunction::new(inputs.into_iter().map(boxed).collect()).collect()
}

fn collect_windowed(inputs: Vec<Vec<i32>>, min_should_match: usize) -> Vec<i32> {
    WindowedDisjunction::new(inputs, min_should_match).collect()
}

/// The reference answer, computed the obvious way rather than by calling
/// either implementation: the union, restricted to documents at least
/// `min_should_match` clauses agree on.
fn brute_force_disjunction(inputs: &[Vec<i32>], min_should_match: usize) -> Vec<i32> {
    let mut counts: std::collections::BTreeMap<i32, usize> = Default::default();
    for docs in inputs {
        for &doc in docs {
            *counts.entry(doc).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .filter(|&(_, n)| n >= min_should_match.max(1))
        .map(|(doc, _)| doc)
        .collect()
}

/// The windowed OR must be indistinguishable from the min-scan disjunction
/// it replaces -- including across window boundaries, which is where a
/// window-at-a-time implementation goes wrong.
///
/// The shapes cover: a document exactly on a window boundary (4095/4096),
/// a clause that spans many windows against one confined to a single
/// window, documents shared by several clauses (dedup), a clause that is
/// empty, one that starts far past the others (so whole windows are
/// skipped), and doc id 0.
#[test]
fn the_windowed_or_matches_the_min_scan_disjunction_on_every_shape() {
    let shapes: Vec<Vec<Vec<i32>>> = vec![
        vec![vec![], vec![]],
        vec![vec![0], vec![0]],
        vec![vec![0, 1, 2], vec![3, 4, 5]],
        vec![vec![4095, 4096], vec![4094, 4097]],
        // One clause spread over four windows, one confined to the third.
        vec![
            (0..16_384_i32).step_by(1_000).collect(),
            (8_192_i32..9_000).collect(),
        ],
        // A clause starting far past the others: whole windows are empty.
        vec![vec![1, 2, 3], vec![1_000_000, 1_000_001]],
        // Three clauses with heavy overlap.
        vec![
            (0..5_000_i32).collect(),
            (0..5_000_i32).step_by(2).collect(),
            (0..5_000_i32).step_by(3).collect(),
        ],
        vec![vec![], vec![7, 8, 9]],
        // The top of the doc-id space. `i32::MAX` is the last window's last
        // slot, and computing that window's exclusive end in `i32` overflows
        // -- saturating to `i32::MAX` would make `doc >= max` true for
        // the very document that selected the window, and **hang**: no clause
        // advanced, nothing was emitted, and the next iteration re-derived
        // the same window forever. `Disjunction` terminates on this input, so
        // the replacement must too.
        vec![vec![i32::MAX], vec![i32::MAX - 1]],
        vec![vec![i32::MAX - 1, i32::MAX], vec![i32::MAX]],
        // ... and the same window reached from below, so the boundary is
        // crossed rather than only landed on.
        vec![vec![i32::MAX - 4096, i32::MAX], vec![i32::MAX - 1]],
    ];
    // Under a deadline: the failure this guards against is an infinite
    // loop, and a plain assertion cannot observe one -- it would wedge the
    // whole test binary instead of failing it.
    run_before(std::time::Duration::from_secs(30), move || {
        for shape in shapes {
            for msm in 0..=3usize {
                let expected = brute_force_disjunction(&shape, msm);
                assert_eq!(
                    collect_windowed(shape.clone(), msm),
                    expected,
                    "windowed OR disagrees for min_should_match={msm} on {shape:?}"
                );
                if msm <= 1 {
                    // ... and with the implementation it replaces, so a
                    // future change to either is caught by this file alone.
                    assert_eq!(collect_disjunction(shape.clone()), expected);
                }
            }
        }
    });
}

/// Runs `body` on a worker thread and fails if it has not finished within
/// `limit`. Used where the defect under test is non-termination, which an
/// assertion cannot catch from inside the same thread.
fn run_before(limit: std::time::Duration, body: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        body();
        // A send failure means the receiver already gave up; the panic
        // below has been reported and this thread is just late.
        let _ = tx.send(());
    });
    match rx.recv_timeout(limit) {
        Ok(()) => worker.join().expect("worker panicked"),
        // Disconnected: the worker panicked, so surface that, not a timeout.
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            worker.join().expect("worker panicked")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!(
            "did not finish within {limit:?} -- almost certainly a \
                 non-terminating window loop"
        ),
    }
}

/// `min_should_match` greater than the clause count matches nothing, and
/// the per-window count must reset between windows -- a document in window
/// 2 must not inherit window 1's tally.
#[test]
fn window_counts_reset_between_windows() {
    // doc 10 is in both clauses (count 2); doc 4106 (window 1) is in one.
    let shape = vec![vec![10, 4106], vec![10]];
    assert_eq!(collect_windowed(shape.clone(), 2), vec![10]);
    assert_eq!(collect_windowed(shape.clone(), 3), Vec::<i32>::new());
    // And the other way round, so a stale count cannot hide in either
    // direction.
    let shape = vec![vec![10, 4106], vec![4106]];
    assert_eq!(collect_windowed(shape, 2), vec![4106]);
}

/// A single clause is not this scorer's shape (`BooleanScorer`'s own
/// constructor rejects fewer than two), so the caller must keep the plain
/// path -- pinned here because the choice lives in `lib.rs`.
#[test]
fn a_single_clause_is_not_the_windowed_shape() {
    assert!(!WindowedDisjunction::is_applicable(0));
    assert!(!WindowedDisjunction::is_applicable(1));
    assert!(WindowedDisjunction::is_applicable(2));
}

#[test]
fn conjunction_no_overlap_matches_nothing() {
    assert_eq!(
        collect_conjunction(vec![vec![1, 2], vec![3, 4]]),
        Vec::<i32>::new()
    );
}

#[test]
fn conjunction_full_overlap_matches_every_doc() {
    assert_eq!(
        collect_conjunction(vec![vec![1, 2, 3], vec![1, 2, 3]]),
        vec![1, 2, 3]
    );
}

#[test]
fn conjunction_partial_overlap_matches_only_shared_docs() {
    assert_eq!(
        collect_conjunction(vec![vec![0, 2, 5, 7], vec![0, 1, 5, 9]]),
        vec![0, 5]
    );
}

#[test]
fn conjunction_three_way_partial_overlap() {
    assert_eq!(
        collect_conjunction(vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5], vec![3, 4, 5, 6]]),
        vec![3, 4]
    );
}

#[test]
fn conjunction_one_iterator_empty_matches_nothing() {
    assert_eq!(
        collect_conjunction(vec![vec![1, 2, 3], vec![]]),
        Vec::<i32>::new()
    );
}

#[test]
fn conjunction_single_clause_passes_through() {
    assert_eq!(collect_conjunction(vec![vec![4, 5, 6]]), vec![4, 5, 6]);
}

#[test]
fn conjunction_no_clauses_matches_nothing() {
    assert_eq!(collect_conjunction(vec![]), Vec::<i32>::new());
}

#[test]
fn disjunction_no_overlap_merges_both_sorted() {
    assert_eq!(
        collect_disjunction(vec![vec![1, 3], vec![2, 4]]),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn disjunction_full_overlap_dedups() {
    assert_eq!(
        collect_disjunction(vec![vec![1, 2, 3], vec![1, 2, 3]]),
        vec![1, 2, 3]
    );
}

#[test]
fn disjunction_partial_overlap() {
    assert_eq!(
        collect_disjunction(vec![vec![0, 2, 4], vec![2, 3]]),
        vec![0, 2, 3, 4]
    );
}

#[test]
fn disjunction_one_iterator_empty() {
    assert_eq!(collect_disjunction(vec![vec![1, 2], vec![]]), vec![1, 2]);
}

#[test]
fn disjunction_no_clauses_matches_nothing() {
    assert_eq!(collect_disjunction(vec![]), Vec::<i32>::new());
}

#[test]
fn excluding_removes_shared_docs() {
    let base = boxed(vec![0, 1, 2, 3, 4]);
    let excluded = boxed(vec![1, 3]);
    let result: Vec<i32> = Excluding::new(base, excluded).collect();
    assert_eq!(result, vec![0, 2, 4]);
}

#[test]
fn excluding_with_no_exclusions_passes_base_through() {
    let base = boxed(vec![0, 2, 4]);
    let excluded = boxed(vec![]);
    let result: Vec<i32> = Excluding::new(base, excluded).collect();
    assert_eq!(result, vec![0, 2, 4]);
}

#[test]
fn excluding_everything_matches_nothing() {
    let base = boxed(vec![0, 1, 2]);
    let excluded = boxed(vec![0, 1, 2]);
    let result: Vec<i32> = Excluding::new(base, excluded).collect();
    assert_eq!(result, Vec::<i32>::new());
}

#[test]
fn excluding_docs_not_in_base_have_no_effect() {
    let base = boxed(vec![0, 2, 4]);
    let excluded = boxed(vec![1, 3, 5, 6]);
    let result: Vec<i32> = Excluding::new(base, excluded).collect();
    assert_eq!(result, vec![0, 2, 4]);
}

#[test]
fn excluding_empty_base_matches_nothing() {
    let base = boxed(vec![]);
    let excluded = boxed(vec![1, 2]);
    let result: Vec<i32> = Excluding::new(base, excluded).collect();
    assert_eq!(result, Vec::<i32>::new());
}
