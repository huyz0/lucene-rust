#![cfg(test)]

#[test]
fn count_set_bits_counts_each_window_exactly() {
    let mut bits = FixedBitSet::new(300);
    for i in [0usize, 1, 63, 64, 65, 127, 128, 200, 299] {
        bits.set(i);
    }
    for from in 0..300 {
        for to in from..=300 {
            let naive = (from..to).filter(|&i| bits.get(i)).count();
            assert_eq!(
                count_set_bits(bits.words(), from, to),
                naive,
                "{from}..{to}"
            );
        }
    }
}
#[test]
fn identity_accept_ords_is_live_and_filter_bit_for_bit() {
    let bits = |len: usize, every: usize| {
        let mut b = FixedBitSet::new(len);
        for i in (0..len).step_by(every) {
            b.set(i);
        }
        b
    };
    // Sources shorter, equal and longer than the field, word-aligned or not.
    for size in [0usize, 1, 63, 64, 65, 200] {
        for (live_len, filter_len) in [(size, size), (size / 2, size + 70), (size + 3, size / 3)] {
            let live = bits(live_len, 2);
            let filter = bits(filter_len, 3);
            for (l, f) in [
                (Some(&live), Some(&filter)),
                (Some(&live), None),
                (None, Some(&filter)),
            ] {
                let got = identity_accept_ords(l, f, size);
                assert_eq!(got.len(), size);
                for ord in 0..size {
                    let want = l.is_none_or(|b| b.get_doc(ord as i32))
                        && f.is_none_or(|b| b.get_doc(ord as i32));
                    assert_eq!(got.get(ord), want, "size {size} ord {ord}");
                }
            }
        }
    }
}

use super::*;

#[test]
fn the_filtered_walk_plan_follows_the_cost_model() {
    // 390k vectors, a queue of 100: about 1_287 unfiltered visits.
    let n = 390_000;
    // 70% passing: the walk is predicted cheap.
    assert_eq!(filtered_walk_plan(270_000, n, 100), FilteredWalk::Walk);
    // 1%: the scan is cheap next to any walk.
    assert_eq!(filtered_walk_plan(4_000, n, 100), FilteredWalk::Scan);
    // 5%: still cheap to scan.
    assert_eq!(filtered_walk_plan(20_000, n, 100), FilteredWalk::Scan);
    // 10%: a walk only pays if the filter passes the neighbourhood.
    match filtered_walk_plan(40_000, n, 100) {
        FilteredWalk::Probe {
            unfiltered_visits,
            share,
            scan_visits,
        } => {
            assert!((unfiltered_visits - (n as f64).ln() * 100.0).abs() < 1e-9);
            assert!((share - 40_000.0 / n as f64).abs() < 1e-12);
            assert_eq!(scan_visits, 10_000);
        }
        other => panic!("{other:?}"),
    }
    // A narrower queue makes walks cheaper: 10% walks at 50.
    assert_eq!(filtered_walk_plan(40_000, n, 50), FilteredWalk::Walk);
    // Degenerate sizes do not divide by zero.
    assert_eq!(filtered_walk_plan(1, 0, 10), FilteredWalk::Walk);
}

#[test]
fn exploration_slack_is_checked() {
    let q = KnnFloatVectorQuery::new("v", vec![1.0], 5).unwrap();
    assert!(q.clone().with_exploration_slack(0, 0.1).is_err());
    assert!(q.clone().with_exploration_slack(5, -0.1).is_err());
    assert!(q.clone().with_exploration_slack(5, f32::NAN).is_err());
    assert!(q.clone().with_exploration_slack(5, f32::INFINITY).is_err());
    let q = q.with_exploration_slack(5, 0.2).unwrap();
    assert_eq!(KnnQuery::exploration_slack(&q), Some((5, 0.2)));
    let b = KnnByteVectorQuery::new("v", vec![1], 5)
        .unwrap()
        .with_exploration_slack(3, 0.0)
        .unwrap();
    assert_eq!(KnnQuery::exploration_slack(&b), Some((3, 0.0)));
}

#[test]
fn a_partition_is_scored_exactly_by_its_documents_over_the_search() {
    let budget = ivf_accepted_target(10) * IVF_PARTITION_EXACT_RATIO;
    let want = ivf_accepted_target(10);
    // Within the budget over the search: exact in every leaf.
    assert_eq!(partition_exact_below(5_000, Some(budget), want, 10), 5_000);
    // Small in this leaf, too large over the search: probed.
    assert_eq!(
        partition_exact_below(5_000, Some(budget + 1), want, 10),
        want
    );
    // A leaf searched alone counts its own documents.
    assert_eq!(partition_exact_below(budget, None, want, 10), budget);
    assert_eq!(partition_exact_below(budget + 1, None, want, 10), want);
    // The search's count is never below the leaf's own.
    assert_eq!(partition_exact_below(budget + 1, Some(0), want, 10), want);
}

#[test]
fn live_passing_counts_the_live_documents_the_filter_passes() {
    let bits = |len: usize, every: usize| {
        let mut b = FixedBitSet::new(len);
        for i in (0..len).step_by(every) {
            b.set(i);
        }
        b
    };
    let (filter, live) = (bits(100, 2), bits(100, 3));
    assert_eq!(live_passing(Some(&filter), Some(&live), 100), 17);
    assert_eq!(live_passing(Some(&filter), None, 100), 50);
    assert_eq!(live_passing(None, Some(&live), 100), 34);
    assert_eq!(live_passing(None, None, 100), 100);
    assert_eq!(live_passing(None, None, -1), 0);
}

/// `HitQueue`'s order through `Float.compare`: `-0.0` below `0.0`,
/// every `NaN` equal and above everything; a lower score is "worse",
/// and between equal scores the higher document is.
#[test]
fn hit_queue_order_is_float_compare_then_document() {
    use std::cmp::Ordering::{Equal, Greater, Less};
    assert_eq!(java_float_compare(-0.0, 0.0), Less);
    assert_eq!(java_float_compare(f32::NAN, -f32::NAN), Equal);
    assert_eq!(java_float_compare(f32::NAN, f32::INFINITY), Greater);
    let w = |doc: i32, score: f32| WorstFirst((doc, 0, score));
    assert!(w(1, 0.5) == w(1, 0.5));
    assert!(w(1, 0.5) != w(2, 0.5));
    // The max-heap's top is the worst hit: the lower score...
    assert!(w(1, 0.25) > w(1, 0.5));
    // ...and, between equal scores, the higher document.
    assert!(w(9, 0.5) > w(1, 0.5));
    assert!(w(1, -0.0) > w(1, 0.0));
}

/// No KNN hit rewrites to `MatchNoDocsQuery`.
#[test]
fn no_knn_hits_is_match_no_docs() {
    assert!(matches!(
        knn_hits_to_clause(&[], &[]),
        crate::query::Clause::MatchNoDocs(_)
    ));
}

#[test]
fn per_leaf_top_k_is_javas_pro_rata_formula() {
    // One leaf: proportion 1, variance 0, so it is exactly `k`.
    assert_eq!(per_leaf_top_k(10, 1.0), 10);
    assert_eq!(per_leaf_top_k(1, 1.0), 1);
    // Four equal leaves, k = 10: 2.5 + 16*sqrt(10*0.25*0.75) = 24.4 -> 24.
    assert_eq!(per_leaf_top_k(10, 0.25), 24);
    // Two equal leaves, k = 10: 5 + 16*sqrt(2.5) = 30.29 -> 30.
    assert_eq!(per_leaf_top_k(10, 0.5), 30);
    // A vanishing leaf still gets a slot (Java's `Math.max(1, ..)`, and
    // its `assert perLeafTopK > 0`).
    assert_eq!(per_leaf_top_k(10, 0.0), 1);
    // A zero-document index divides by zero. Java's `Math.max`
    // propagates the NaN and yields 0 (and trips its own `assert
    // perLeafTopK > 0`); this returns 1 -- see the function's own comment
    // for why that divergence is observably nothing.
    assert_eq!(per_leaf_top_k(10, f32::NAN), 1);
    // And an absurd k cannot overflow the cast.
    assert_eq!(per_leaf_top_k(usize::MAX, 1.0), i32::MAX as usize);
}

#[test]
fn k_zero_is_rejected_like_javas_constructor() {
    let e = KnnFloatVectorQuery::new("f", vec![1.0], 0).unwrap_err();
    assert!(e.to_string().contains("k must be at least 1"), "{e}");
    let e = KnnByteVectorQuery::new("f", vec![1], 0).unwrap_err();
    assert!(e.to_string().contains("k must be at least 1"), "{e}");
    assert!(KnnFloatVectorQuery::new("f", vec![1.0], 1).is_ok());
}

#[test]
fn query_builders_set_exactly_what_they_name() {
    let q = KnnFloatVectorQuery::new("f", vec![1.0], 3)
        .unwrap()
        .with_ef_search(50)
        .with_visited_limit(7)
        .with_similarity(VectorSimilarityFunction::Cosine);
    assert_eq!((q.k, q.ef_search, q.visited_limit), (3, 50, 7));
    assert_eq!(q.similarity, Some(VectorSimilarityFunction::Cosine));
    assert_eq!(visit_limit(&q), 7);
    let plain = KnnByteVectorQuery::new("f", vec![1], 3).unwrap();
    // `visited_limit == 0` is Java's "unlimited".
    assert_eq!(visit_limit(&plain), u64::MAX);
    assert_eq!(plain.similarity, None);
}

#[test]
fn similarity_ordinals_are_the_pinned_file_format_order() {
    for ordinal in 0..4 {
        let s = similarity_from_ordinal(ordinal).unwrap();
        assert_eq!(similarity_ordinal(s), ordinal);
    }
    assert_eq!(similarity_from_ordinal(-1), None);
    assert_eq!(similarity_from_ordinal(4), None);
    assert_eq!(similarity_ordinal(VectorSimilarityFunction::Euclidean), 0);
    assert_eq!(similarity_ordinal(VectorSimilarityFunction::DotProduct), 1);
    assert_eq!(similarity_ordinal(VectorSimilarityFunction::Cosine), 2);
    assert_eq!(
        similarity_ordinal(VectorSimilarityFunction::MaximumInnerProduct),
        3
    );
}

#[test]
fn accept_bitset_drops_out_of_range_doc_ids() {
    let bits = accept_bitset([0, 3, 7, 99, -1], 8);
    assert_eq!(bits.len(), 8);
    assert!(bits.get(0) && bits.get(3) && bits.get(7));
    assert_eq!(bits.cardinality(), 3);
    assert_eq!(accept_bitset([1], -5).len(), 0);
}

#[test]
fn reentry_picks_exactly_the_leaves_that_are_not_tapped_out() {
    let sd = |doc, score| ScoreDoc { doc_id: doc, score };
    let per_leaf = vec![
        vec![sd(0, 0.9), sd(1, 0.8)], // worst 0.8 >= 0.75 -> re-enter
        vec![sd(2, 0.7), sd(3, 0.6)], // worst 0.6 <  0.75 -> tapped out
        vec![],                       // nothing at all    -> tapped out
    ];
    let merged = vec![sd(0, 0.9), sd(1, 0.8), sd(2, 0.75)];
    assert_eq!(reentry_leaves(&per_leaf, &merged), vec![0]);
    // With no merged hits at all there is nothing to compare against.
    assert!(reentry_leaves(&per_leaf, &[]).is_empty());
}

/// A scorer that counts the vector comparisons the walk performs, so a
/// test can assert on *work done* rather than on a recall number (recall
/// does not discriminate here -- mutating the
/// diversity rule took graph agreement to 1/4273 while recall rose).
struct Counting<S> {
    inner: S,
    comparisons: usize,
}

impl<S: VectorScorer> VectorScorer for Counting<S> {
    fn score(&mut self, node: i32) -> lucene_codecs::vectors::Result<f32> {
        self.comparisons += 1;
        self.inner.score(node)
    }

    fn max_ord(&self) -> i32 {
        self.inner.max_ord()
    }

    fn bulk_score(
        &mut self,
        nodes: &[i32],
        scores: &mut [f32],
    ) -> lucene_codecs::vectors::Result<f32> {
        self.comparisons += nodes.len();
        self.inner.bulk_score(nodes, scores)
    }
}

/// What seeding actually buys, over the real fixture graph and asserted
/// structurally rather than by a metric.
///
/// Two properties, and both fail on a "seeded" search that quietly
/// ignored its entry points:
///
/// 1. **Seeding a walk with its own answer is a fixpoint.** Feeding the
///    unseeded top-`k`'s ordinals back in as entry points returns exactly
///    the same hits -- which is the reason Java can substitute phase 2's
///    seeded walk for a fresh descent at all.
/// 2. **It skips `findBestEntryPoint`.** The seeded walk performs
///    strictly fewer vector comparisons, because the entire hill climb
///    over every level above 0 is not run.
///
/// A third assertion pins that the seeds are *used* and not merely
/// accepted: seeding from a single far-away ordinal reaches a different
/// (worse) answer.
#[test]
fn a_seeded_walk_restarts_from_its_entry_points_and_skips_the_descent() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/vectors_index/"
    );
    let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run scripts/gen-fixtures.sh first (GenVectors)");
    let kv: std::collections::HashMap<&str, &str> =
        text.lines().filter_map(|l| l.split_once('=')).collect();
    let mut id = [0u8; 16];
    let hex = kv["id_hex"];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = kv["segment_suffix"];
    let read = |name: &str| std::fs::read(format!("{dir}{name}")).expect("fixture file");
    let (vemf, vec_file) = (read(kv["vemf_file"]), read(kv["vec_file"]));
    let (vem, vex) = (read(kv["vem_file"]), read(kv["vex_file"]));
    let flat = FlatVectorsReader::open(&vemf, &vec_file, &id, suffix).unwrap();
    let hnsw = HnswVectorsReader::open(&vem, &vex, &id, suffix).unwrap();
    let field_number: i32 = kv["f0.number"].parse().unwrap();
    let values = flat.float_vector_values(field_number).unwrap();
    let graph = hnsw.graph(field_number).unwrap();
    assert!(graph.is_some(), "the dense fixture field carries a graph");
    let target: Vec<f32> = kv["q.f0.0.vec"]
        .split(',')
        .map(|s| f32::from_bits(s.parse::<i32>().unwrap() as u32))
        .collect();
    let k = 10;

    let run = |seeds: Option<&[i32]>, visit_limit: u64| {
        let mut scorer = Counting {
            inner: values.scorer(&target).unwrap(),
            comparisons: 0,
        };
        let (hits, _) = hnsw_vectors::search(
            &mut scorer,
            graph.as_ref(),
            k,
            visit_limit,
            SearchOptions {
                seed_ords: seeds,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        (hits, scorer.comparisons)
    };

    let (plain, plain_cost) = run(None, u64::MAX);
    assert_eq!(plain.len(), k);
    let mut seeds: Vec<i32> = plain.iter().map(|(ord, _)| *ord).collect();
    seeds.sort_unstable();

    let (seeded, seeded_cost) = run(Some(&seeds), u64::MAX);
    assert_eq!(seeded, plain, "seeding a walk with its own answer moved it");
    assert!(
        seeded_cost < plain_cost,
        "the seeded walk did {seeded_cost} comparisons against {plain_cost}: it did not \
             skip the entry-point descent"
    );

    // And the seeds are genuinely *the starting set*, not a hint the walk
    // may discard. Capped at exactly one visit per seed, the beam cannot
    // move at all, so the answer is the seed set itself -- and one
    // arbitrary seed under the same cap is not it. (Without the cap this
    // graph is well-connected enough that even a single far-away entry
    // point converges to the same top 10, which is why the assertion
    // needs the cap to discriminate.)
    let far = values.scorer(&target).unwrap().max_ord() - 1;
    let (pinned, _) = run(Some(&seeds), seeds.len() as u64);
    assert_eq!(pinned, plain, "a visit-capped seeded walk left its seeds");
    let (stranded, _) = run(Some(&[far]), 1);
    assert_eq!(stranded.len(), 1);
    assert_eq!(stranded[0].0, far);
}

/// An empty phase-1 hit list for a leaf is "not seeded", not "seeded
/// with nothing" -- Java's `ReentrantKnnCollectorManager` falls back to
/// the unseeded collector there, and the codec's seeded entry point
/// rejects an empty entry-point set outright (as Java's
/// `fromEntryPoints` does). Getting this wrong turns a leaf that
/// collected nothing in phase 1 into a hard error.
#[test]
fn an_empty_phase_one_hit_list_means_not_seeded() {
    let seeds = vec![vec![3, 7], Vec::new()];
    assert_eq!(seed_slice(Some(&seeds), 0), Some(&[3, 7][..]));
    assert_eq!(seed_slice(Some(&seeds), 1), None);
    // Phase 1 itself is never seeded.
    assert_eq!(seed_slice(None, 0), None);
}

/// Phase 2 restores the full `k` **on the collector only**. Java
/// recomputes `perLeafTopK` inside `getLeafResults` from `ctx.parent` on
/// every call, so the two cost thresholds stay pro-rata across both
/// passes; only the collector manager changes.
#[test]
fn reentry_restores_the_full_k_on_the_collector_and_nothing_else() {
    let phase1 = LeafPlan {
        k: 10,
        per_leaf_top_k: 24,
        collector_k: 24,
        visited_limit: 99,
        filtered: true,
        extras: LeafExtras::default(),
    };
    let phase2 = reentry_plan(&phase1, 0);
    assert_eq!(phase2.collector_k, 10);
    assert_eq!(phase2.per_leaf_top_k, 24, "the threshold stays pro-rata");
    assert_eq!(phase2.k, 10);
    assert_eq!(phase2.visited_limit, 99);
    assert!(phase2.filtered);
    // A caller-widened beam widens the collector and nothing else: the
    // cost thresholds stay Java's.
    let widened = reentry_plan(&phase1, 40);
    assert_eq!(widened.collector_k, 40);
    assert_eq!(widened.per_leaf_top_k, 24);
}
/// A scorer over a fixed table of similarities, one per ordinal.
struct Table(Vec<f32>);

impl VectorScorer for Table {
    fn score(&mut self, node: i32) -> lucene_codecs::vectors::Result<f32> {
        Ok(self.0[node as usize])
    }

    fn max_ord(&self) -> i32 {
        self.0.len() as i32
    }
}

fn bits(len: usize, set: &[usize]) -> FixedBitSet {
    let mut b = FixedBitSet::new(len);
    for &i in set {
        b.set(i);
    }
    b
}

fn doc_of(ord: i32) -> Result<i32> {
    Ok(ord * 2)
}

/// Children at even docs (ordinal `o` is doc `2o`), parents closing
/// blocks at docs 5, 11 and 19: blocks {0,2,4}, {6,8,10}, {12..18}, and
/// doc 20 after the last parent (`NO_MORE_DOCS` is its parent).
fn block_parents() -> FixedBitSet {
    bits(21, &[5, 11, 19])
}

#[test]
fn diversified_exact_search_keeps_each_parents_best_child() {
    let parents = block_parents();
    //             block 0          block 1          block 2                block after
    let mut t = Table(vec![0.1, 0.5, 0.5, 0.2, 0.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.6]);
    let all = bits(11, &(0..11).collect::<Vec<_>>());
    let hits = diversified_exact_search(&mut t, &all, 11, 10, &parents, &doc_of).unwrap();
    // One per block, best first; a tie inside a block keeps the first
    // child (`score > currentScore`), the orphan block counts too.
    assert_eq!(
        hits,
        vec![(8, 4, 0.9), (20, 10, 0.6), (2, 1, 0.5), (12, 6, 0.4)]
    );
    // `k` (and `cost`) bound the queue; a tie between blocks keeps the
    // lower document (`HitQueue`'s order), whatever came first.
    let mut t = Table(vec![0.7, 0.0, 0.0, 0.7, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0]);
    let hits = diversified_exact_search(&mut t, &all, 11, 2, &parents, &doc_of).unwrap();
    assert_eq!(hits, vec![(0, 0, 0.7), (6, 3, 0.7)]);
    let hits = diversified_exact_search(&mut t, &all, 1, 5, &parents, &doc_of).unwrap();
    assert_eq!(hits, vec![(0, 0, 0.7)]);
    // Only accepted ordinals count; nothing accepted is nothing.
    let some = bits(11, &[2, 9]);
    let hits = diversified_exact_search(&mut t, &some, 2, 5, &parents, &doc_of).unwrap();
    assert_eq!(hits, vec![(4, 2, 0.0), (18, 9, 0.0)]);
    assert!(
        diversified_exact_search(&mut t, &some, 0, 5, &parents, &doc_of)
            .unwrap()
            .is_empty()
    );
    // Negative similarities are dropped with the sentinels.
    let mut t = Table(vec![
        -0.5, -0.5, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    let hits = diversified_exact_search(&mut t, &all, 11, 2, &parents, &doc_of).unwrap();
    assert_eq!(hits, vec![(6, 3, 0.25), (12, 6, 0.0)]);
    // A failing ordinal lookup is the caller's error.
    let broken = |_: i32| -> Result<i32> { Err(Error::IllegalArgument("ord".into())) };
    assert!(diversified_exact_search(&mut t, &all, 11, 2, &parents, &broken).is_err());
}

#[test]
fn the_diversifying_collector_follows_javas_heap() {
    let parents = block_parents();
    let mut c = DiversifyingCollector::new(2, 100, &parents, &doc_of);
    assert_eq!(KnnCollect::k(&c), 2);
    assert_eq!(c.min_competitive_similarity(), f32::NEG_INFINITY);
    assert!(c.collect(0, 0.5)); // block 0
    assert!(!c.collect(1, 0.4)); // block 0, worse: kept 0
    assert!(c.collect(1, 0.6)); // block 0, better: replaces 0
    assert!(c.collect(4, 0.2)); // block 1
    assert_eq!(c.min_competitive_similarity(), 0.2);
    assert!(!c.collect(6, 0.1)); // block 2, below the floor
    assert!(!c.collect(7, 0.2)); // block 2, a tie with a higher child
    assert!(c.collect(10, 0.3)); // the orphan block evicts block 1
    assert!(c.collect(2, 0.7)); // block 0 improves again, in place
    assert!(c.collect(9, 0.3)); // block 2, a tie with a lower child: evicts doc 20
    c.inc_visited_count(3);
    assert_eq!((c.visited_count(), c.visit_limit()), (3, 100));
    assert!(!KnnCollect::early_terminated(&c));
    assert_eq!(c.top_docs().unwrap(), vec![(4, 2, 0.7), (18, 9, 0.3)]);

    // A worse score for a parent in the heap that must move down.
    let mut c = DiversifyingCollector::new(3, 1, &parents, &doc_of);
    for (ord, score) in [(0, 0.9), (3, 0.8), (6, 0.7)] {
        assert!(c.collect(ord, score));
    }
    assert!(c.collect(1, 0.95));
    assert!(c.collect(4, 0.85));
    c.inc_visited_count(1);
    assert!(KnnCollect::early_terminated(&c));
    assert_eq!(
        c.top_docs().unwrap(),
        vec![(2, 1, 0.95), (8, 4, 0.85), (12, 6, 0.7)]
    );

    // An ordinal that does not translate fails the search.
    let broken = |_: i32| -> Result<i32> { Err(Error::IllegalArgument("ord".into())) };
    let mut c = DiversifyingCollector::new(1, 10, &parents, &broken);
    assert!(!c.collect(0, 1.0));
    assert!(c.top_docs().is_err());
}

#[test]
fn diversified_leaf_results_take_javas_branches() {
    let parents = block_parents();
    let plan = |filtered: bool, per_leaf_top_k: usize| LeafPlan {
        k: 2,
        per_leaf_top_k,
        collector_k: 2,
        visited_limit: u64::MAX,
        filtered,
        extras: LeafExtras::default(),
    };
    let mut t = Table(vec![0.1, 0.5, 0.5, 0.2, 0.9, 0.3, 0.4, 0.4, 0.4, 0.4, 0.6]);
    // No parents in the segment: no results.
    let (hits, early) = diversified_leaf_results(
        &mut t,
        &LeafGraph::ScanAll,
        None,
        &doc_of,
        21,
        &plan(false, 2),
        None,
        None,
    )
    .unwrap();
    assert!(hits.hits.is_empty() && !early);
    // Unfiltered: the reader's search (a full scan here).
    let (hits, _) = diversified_leaf_results(
        &mut t,
        &LeafGraph::ScanAll,
        None,
        &doc_of,
        21,
        &plan(false, 2),
        None,
        Some(&parents),
    )
    .unwrap();
    let docs: Vec<(i32, f32)> = hits.hits.iter().map(|h| (h.doc_id, h.score)).collect();
    assert_eq!(docs, vec![(8, 0.9), (20, 0.6)]);
    assert_eq!(hits.ords, vec![4, 10]);
    // Filtered below perLeafTopK: the exact search.
    let some = bits(11, &[0, 1, 6]);
    let accept = AcceptOrds::Borrowed(&some);
    let (hits, _) = diversified_leaf_results(
        &mut t,
        &LeafGraph::Nothing,
        Some(&accept),
        &doc_of,
        21,
        &plan(true, 5),
        None,
        Some(&parents),
    )
    .unwrap();
    let docs: Vec<i32> = hits.hits.iter().map(|h| h.doc_id).collect();
    assert_eq!(docs, vec![2, 12]);
    // Filtered above it: the graph first; one that finds too little (a
    // flat format collects nothing) falls back to the exact search, one
    // that finds enough is kept.
    let (hits, _) = diversified_leaf_results(
        &mut t,
        &LeafGraph::Nothing,
        Some(&accept),
        &doc_of,
        21,
        &plan(true, 1),
        None,
        Some(&parents),
    )
    .unwrap();
    assert_eq!(hits.hits.len(), 2);
    let (hits, early) = diversified_leaf_results(
        &mut t,
        &LeafGraph::ScanAll,
        Some(&accept),
        &doc_of,
        21,
        &plan(true, 1),
        None,
        Some(&parents),
    )
    .unwrap();
    let docs: Vec<i32> = hits.hits.iter().map(|h| h.doc_id).collect();
    assert_eq!((docs, early), (vec![2, 12], false));
}

#[test]
fn the_rank_walk_accepts_what_the_lookups_accept() {
    use lucene_codecs::vectors::DocToOrdCursor;
    // Documents with a vector, in ordinal order, spanning several words.
    let docs: Vec<i32> = (0..300).filter(|d| d % 3 != 1).collect();
    let cursor = DocToOrdCursor::Explicit(&docs);
    let last = *docs.last().unwrap();
    let filter = bits(300, &(0..300).filter(|d| d % 5 != 0).collect::<Vec<_>>());
    let live = bits(300, &(0..300).filter(|d| d % 7 != 0).collect::<Vec<_>>());
    for live in [None, Some(&live)] {
        let mut got = FixedBitSet::new(docs.len());
        accept_ords_by_rank(&cursor, &filter, live, last, &mut got).unwrap();
        let want: Vec<usize> = docs
            .iter()
            .enumerate()
            .filter(|(_, &d)| filter.get(d as usize) && live.is_none_or(|l| l.get(d as usize)))
            .map(|(o, _)| o)
            .collect();
        let mut have = Vec::new();
        got.for_each_set_bit(|o| have.push(o));
        assert_eq!(have, want);
    }
    // A shorter filter or live set accepts nothing past its end, and an
    // ordinal past the accept set is dropped.
    let short = bits(64, &[2, 3, 63]);
    let mut got = FixedBitSet::new(2);
    accept_ords_by_rank(&cursor, &short, None, last, &mut got).unwrap();
    let mut have = Vec::new();
    got.for_each_set_bit(|o| have.push(o));
    assert_eq!(have, vec![1]);
    let mut got = FixedBitSet::new(docs.len());
    accept_ords_by_rank(&cursor, &filter, Some(&short), last, &mut got).unwrap();
    let mut have = Vec::new();
    got.for_each_set_bit(|o| have.push(o));
    // Docs 2, 3 and 63 are live; 63 is the 43rd document with a vector.
    assert_eq!(have, vec![1, 2, 42]);
}
