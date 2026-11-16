#![cfg(test)]
use super::*;
use lucene_codecs::field_infos::VectorSimilarityFunction;

#[test]
fn the_walk_planner_gives_up_after_a_probe_that_meets_few_accepted_nodes() {
    // 1000 unfiltered visits, a filter passing 10% of the graph, a scan
    // costing 2000 visits, a probe of 100.
    let mut p = FilteredWalkPlanner::new(KnnCollector::new(50, u64::MAX), 1000.0, 0.1, 2000, 100);
    // The probe meets 2 accepted nodes in 100: still counted at the
    // global 10%, 10_000 visits predicted, more than the scan.
    p.collect(1, 0.9);
    p.collect(2, 0.8);
    p.inc_visited_count(99);
    assert!(!p.early_terminated());
    p.inc_visited_count(1);
    assert!(p.gave_up() && p.early_terminated());
}

#[test]
fn the_walk_planner_keeps_walking_where_the_filter_passes_the_neighbourhood() {
    let mut p = FilteredWalkPlanner::new(KnnCollector::new(50, u64::MAX), 1000.0, 0.1, 2000, 100);
    for i in 0..60 {
        p.collect(i, 0.5);
    }
    p.inc_visited_count(100);
    // 60% passing: 1_667 predicted, cheaper than the scan.
    assert!(!p.gave_up());
    // Accepted nodes are no longer counted once decided.
    p.collect(99, 0.1);
    assert_eq!(p.accepted, 60);
    // Past twice the scan's cost it gives up all the same.
    p.inc_visited_count(3900);
    assert!(!p.gave_up());
    p.inc_visited_count(1);
    assert!(p.gave_up());
    let inner = p.into_inner();
    assert_eq!(inner.visited_count(), 4001);
}

#[test]
fn the_walk_planner_leaves_the_descent_out_of_the_probe() {
    let mut p = FilteredWalkPlanner::new(KnnCollector::new(100, u64::MAX), 1000.0, 0.1, 2000, 100);
    // 300 visits on the upper levels collect nothing.
    p.inc_visited_count(300);
    assert!(!p.gave_up());
    // Level 0: 60 of the first 100 pass.
    p.next_vectors_block();
    for i in 0..60 {
        p.collect(i, 0.5);
    }
    p.inc_visited_count(99);
    assert!(!p.decided);
    p.inc_visited_count(1);
    assert!(p.decided && !p.gave_up());
}

#[test]
fn the_walk_planner_decides_once_the_collector_is_full() {
    let mut p = FilteredWalkPlanner::new(KnnCollector::new(2, u64::MAX), 1000.0, 0.01, 50, 1000);
    p.collect(1, 0.9);
    p.collect(2, 0.8);
    // Full after 2 visits: share 1.0, 1000 predicted, more than 50.
    p.inc_visited_count(2);
    assert!(p.gave_up());
    assert_eq!(p.k(), 2);
    assert_eq!(p.visit_limit(), u64::MAX);
    assert!(p.min_competitive_similarity() > 0.0);
}

#[test]
fn adaptive_beam_bound_scales_the_kth_distance() {
    let mut c = AdaptiveBeamCollector::new(
        KnnCollector::new(2, u64::MAX),
        2,
        0.5,
        VectorSimilarityFunction::Cosine,
    )
    .unwrap();
    assert_eq!(c.min_competitive_similarity(), f32::NEG_INFINITY);
    assert!(!c.collect(1, 0.9));
    assert!(c.collect(2, 0.8));
    // 2nd best 0.8: distance 0.2, times 1.5 = 0.3.
    assert!((c.min_competitive_similarity() - 0.7).abs() < 1e-6);
    assert!(!c.collect(3, 0.1));
    assert!(c.collect(4, 0.95));
    assert!((c.min_competitive_similarity() - 0.85).abs() < 1e-6);
    assert_eq!(c.k(), 2);
    let inner = c.into_inner();
    assert_eq!(inner.top_docs().len(), 2);

    // Fewer hits than k: no bound until the delegate is full.
    let mut f = AdaptiveBeamCollector::new(
        KnnCollector::new(3, u64::MAX),
        1,
        0.0,
        VectorSimilarityFunction::DotProduct,
    )
    .unwrap();
    f.collect(1, 0.9);
    f.collect(2, 0.5);
    assert_eq!(f.min_competitive_similarity(), f32::NEG_INFINITY);
    f.collect(3, 0.4);
    assert!((f.min_competitive_similarity() - 0.9).abs() < 1e-6);

    let mut e = AdaptiveBeamCollector::new(
        KnnCollector::new(1, u64::MAX),
        1,
        1.0,
        VectorSimilarityFunction::Euclidean,
    )
    .unwrap();
    // score 0.5 = distance 1; doubled = 2, score 1 / (1 + 4).
    e.collect(1, 0.5);
    assert!((e.min_competitive_similarity() - 0.2).abs() < 1e-6);
    e.collect(2, 1.0);
    assert!((e.min_competitive_similarity() - 1.0).abs() < 1e-6);
    let mut z = AdaptiveBeamCollector::new(
        KnnCollector::new(1, u64::MAX),
        1,
        1.0,
        VectorSimilarityFunction::Euclidean,
    )
    .unwrap();
    z.collect(1, 0.0);
    assert_eq!(z.min_competitive_similarity(), f32::NEG_INFINITY);
    assert!(AdaptiveBeamCollector::new(
        KnnCollector::new(10, u64::MAX),
        1,
        1.0,
        VectorSimilarityFunction::MaximumInnerProduct,
    )
    .is_none());
}

#[test]
fn float_heap_keeps_the_largest() {
    let mut h = FloatHeap::new(3);
    for v in [5.0, 1.0, 4.0, 2.0, 3.0] {
        h.offer(v);
    }
    assert_eq!(h.size(), 3);
    assert_eq!(h.peek(), 3.0);
    assert!(!h.offer(0.5));
    assert_eq!(h.poll(), Some(3.0));
    assert_eq!(h.poll(), Some(4.0));
    assert_eq!(h.poll(), Some(5.0));
    assert_eq!(h.poll(), None);
    assert_eq!(h.peek(), 0.0);
    h.offer(1.0);
    assert_eq!(h.get_heap(), &[1.0]);
    h.clear();
    assert_eq!(h.size(), 0);
}

#[test]
fn blocking_heap_offers_sorted_runs_largest_first() {
    let g = BlockingFloatHeap::new(2);
    assert_eq!(g.offer(1.0), 1.0);
    assert_eq!(g.offer_sorted(&[0.5, 2.0, 3.0]), 2.0);
    assert_eq!(g.size(), 2);
    assert_eq!(g.offer(2.0), 2.0);
    assert_eq!(g.peek(), 2.0);
    assert_eq!(g.poll(), Some(2.0));
}

#[test]
fn similarity_collector_decays_its_bound() {
    let mut c = VectorSimilarityCollector::new(0.5, 0.5, 10);
    assert!(!c.collect(1, 0.7));
    assert!(c.collect(2, 0.1));
    assert!(c.min_competitive_similarity() < 0.0);
    c.inc_visited_count(10);
    assert!(c.early_terminated());
    assert_eq!(c.k(), 1);
    assert_eq!(c.visit_limit(), 10);
    assert_eq!(c.visited_count(), 10);
    assert_eq!(c.num_collected(), 1);
    let (hits, early) = c.into_hits();
    assert_eq!(hits, vec![(1, 0.7)]);
    assert!(early);
    let mut max_quality = VectorSimilarityCollector::new(0.5, 1.0, 10);
    assert!(!max_quality.collect(3, 0.1));
}

/// `TestVectorSimilarityCollector.testExplicitResultCollection`.
#[test]
fn explicit_collector_keeps_the_traversal_bound() {
    let limit = i32::MAX as u64;
    let mut c = ExplicitVectorSimilarityCollector::new(0.3, 0.5, limit).unwrap();
    let nodes = [1, 5, 10, 4, 8, 3, 2, 6, 7, 9];
    let scores = [0.1f32, 0.2, 0.3, 0.5, 0.2, 0.6, 0.9, 0.3, 0.7, 0.8];
    let mut mins = Vec::new();
    for (&n, &s) in nodes.iter().zip(&scores) {
        assert!(c.collect(n, s));
        mins.push(c.min_competitive_similarity());
    }
    assert_eq!(mins, [0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]);
    assert_eq!((c.k(), c.visit_limit(), c.num_collected()), (1, limit, 5));
    c.inc_visited_count(3);
    assert_eq!(c.visited_count(), 3);
    let (hits, early) = c.into_hits();
    assert_eq!(hits, vec![(4, 0.5), (3, 0.6), (2, 0.9), (7, 0.7), (9, 0.8)]);
    assert!(!early);
    assert!(ExplicitVectorSimilarityCollector::new(0.6, 0.5, 1).is_err());
    let mut nan = ExplicitVectorSimilarityCollector::new(0.0, 0.0, 1).unwrap();
    nan.collect(0, f32::NAN);
    assert!(nan.min_competitive_similarity().is_nan());
    let mut zero = ExplicitVectorSimilarityCollector::new(0.0, 0.0, 1).unwrap();
    zero.collect(0, -0.0);
    assert!(zero.min_competitive_similarity().is_sign_negative());
    let mut pos = ExplicitVectorSimilarityCollector::new(-0.0, 0.0, 1).unwrap();
    pos.collect(0, 0.0);
    assert!(pos.min_competitive_similarity().is_sign_negative());
    pos.inc_visited_count(1);
    assert!(pos.into_hits().1);
}

#[test]
fn saturation_ends_the_walk_after_patience_candidates() {
    let mut c = HnswQueueSaturationCollector::new(KnnCollector::new(2, u64::MAX), 0.9, 1);
    assert!(c.collect(1, 0.5));
    c.next_vectors_block();
    assert!(!c.early_terminated());
    c.next_vectors_block();
    c.next_vectors_block();
    assert!(c.early_terminated());
    assert!(!c.partial());
    assert_eq!(c.k(), 2);
    c.inc_visited_count(1);
    assert_eq!(c.visited_count(), 1);
    assert_eq!(c.visit_limit(), u64::MAX);
    assert_eq!(c.min_competitive_similarity(), f32::NEG_INFINITY);
    assert_eq!(c.into_inner().size(), 1);
}

#[test]
fn time_limit_terminates_once_past_the_deadline() {
    let past = Instant::now();
    let mut c = TimeLimitingKnnCollector::new(KnnCollector::new(1, u64::MAX), past);
    assert!(c.early_terminated());
    assert!(c.collect(1, 1.0));
    c.inc_visited_count(2);
    c.next_vectors_block();
    assert_eq!(
        (c.k(), c.visited_count(), c.visit_limit()),
        (1, 2, u64::MAX)
    );
    assert_eq!(c.min_competitive_similarity(), 1.0);
    assert_eq!(c.into_inner().size(), 1);
    let later = Instant::now() + std::time::Duration::from_secs(3600);
    assert!(
        !TimeLimitingKnnCollector::new(KnnCollector::new(1, u64::MAX), later).early_terminated()
    );
}

#[test]
fn multi_leaf_collector_shares_the_global_bar() {
    let global = BlockingFloatHeap::new(2);
    global.offer_sorted(&[0.8, 0.9]);
    let mut c = MultiLeafKnnCollector::new(2, &global, KnnCollector::new(2, u64::MAX)).unwrap();
    assert_eq!(c.min_competitive_similarity(), f32::NEG_INFINITY);
    c.collect(1, 0.1);
    c.collect(2, 0.2);
    // Two local results: the bar is at least the global queue's floor
    // capped by the non-competitive queue.
    assert!(c.min_competitive_similarity() >= 0.1);
    assert!(
        MultiLeafKnnCollector::with_params(2, 2.0, 1, &global, KnnCollector::new(2, 1)).is_err()
    );
    assert!(
        MultiLeafKnnCollector::with_params(2, 0.5, 0, &global, KnnCollector::new(2, 1)).is_err()
    );
    assert!(!c.early_terminated());
    c.inc_visited_count(1);
    assert_eq!(
        (c.k(), c.visited_count(), c.visit_limit()),
        (2, 1, u64::MAX)
    );
    assert_eq!(c.into_inner().size(), 2);
}
