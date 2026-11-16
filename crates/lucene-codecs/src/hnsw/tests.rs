#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::vectors::Error;

/// A one-dimensional scorer over an explicit vector list: every test below
/// can then reason about "closer" by reading numbers off the page.
#[derive(Debug, Clone)]
struct Line {
    values: Vec<f32>,
    query: f32,
}

impl Line {
    fn new(values: Vec<f32>) -> Self {
        Line { values, query: 0.0 }
    }
}

impl VectorScorer for Line {
    fn score(&mut self, node: i32) -> Result<f32> {
        let d = self.values[node as usize] - self.query;
        Ok(1.0 / (1.0 + d * d))
    }

    fn max_ord(&self) -> i32 {
        self.values.len() as i32
    }
}

impl UpdateableVectorScorer for Line {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        self.query = self.values[ord as usize];
        Ok(())
    }
}

// ---------------- SplittableRandom ----------------

// ---------------- heaps ----------------

/// The packing puts the score in the top 32 bits and the *complement* of
/// the node id in the bottom 32, so equal scores are broken in favour of
/// the smaller node id -- on both heap orders.
/// Both reset paths -- per-touched-word and full -- leave nothing set,
/// growth keeps the logical length, and `set` lists each word once.
#[test]
fn visited_set_resets_exactly_what_was_touched() {
    let mut v = VisitedSet::new(1000);
    for i in [0usize, 1, 63, 64, 500, 999] {
        v.set(i);
    }
    assert_eq!(v.touched, vec![0, 1, 7, 15]);
    v.ensure_capacity_and_clear(1000);
    assert!(v.touched.is_empty());
    assert!((0..1000).all(|i| !v.get(i)));
    // Touch more than a sixteenth of the words: the full clear.
    for i in (0..1000).step_by(64) {
        v.set(i);
    }
    v.ensure_capacity_and_clear(1000);
    assert!((0..1000).all(|i| !v.get(i)));
    // Growth by one node at a time reallocates only past the slack.
    v.ensure_capacity_and_clear(1001);
    assert_eq!(v.len(), 1001);
    let words = v.bits.words().len();
    v.set(1000);
    v.ensure_capacity_and_clear(1002);
    assert_eq!(v.bits.words().len(), words, "oversized: no reallocation");
    assert!(!v.get(1000));
    assert_eq!(v.len(), 1002);
}

#[test]
fn neighbor_queue_breaks_score_ties_toward_the_smaller_node() {
    let mut max = NeighborQueue::new(4, true);
    max.add(7, 0.5);
    max.add(3, 0.5);
    max.add(9, 0.9);
    assert_eq!(max.top_node(), 9);
    assert_eq!(max.top_score(), 0.9);
    assert_eq!(max.pop(), 9);
    assert_eq!(max.pop(), 3); // smaller id wins the 0.5 tie
    assert_eq!(max.pop(), 7);

    let mut min = NeighborQueue::new(4, false);
    min.add(7, 0.5);
    min.add(3, 0.5);
    min.add(9, 0.1);
    assert_eq!(min.top_node(), 9);
    // The worst kept is evicted first, and among equals the *larger* id.
    assert_eq!(min.pop(), 9);
    assert_eq!(min.pop(), 7);
    assert_eq!(min.pop(), 3);
}

#[test]
fn neighbor_queue_insert_with_overflow_is_bounded() {
    let mut q = NeighborQueue::new(2, false);
    assert!(q.insert_with_overflow(0, 0.1));
    assert!(q.insert_with_overflow(1, 0.5));
    assert!(!q.insert_with_overflow(2, 0.05));
    assert!(q.insert_with_overflow(3, 0.9));
    assert_eq!(q.size(), 2);
    let mut nodes = q.nodes();
    nodes.sort_unstable();
    assert_eq!(nodes, vec![1, 3]);
    q.clear();
    assert_eq!(q.size(), 0);
}

// ---------------- KnnCollector ----------------

#[test]
fn knn_collector_tracks_competitiveness_and_the_visit_limit() {
    let mut c = KnnCollector::new(2, 5);
    assert_eq!(c.k(), 2);
    assert_eq!(c.min_competitive_similarity(), f32::NEG_INFINITY);
    c.collect(0, 0.2);
    assert_eq!(c.min_competitive_similarity(), f32::NEG_INFINITY);
    c.collect(1, 0.8);
    assert_eq!(c.min_competitive_similarity(), 0.2);
    assert_eq!(c.minimum_score(), 0.2);
    assert!(!c.early_terminated());
    c.inc_visited_count(5);
    assert_eq!(c.visited_count(), 5);
    assert_eq!(c.visit_limit(), 5);
    assert!(c.early_terminated());
    assert_eq!(c.size(), 2);
    assert_eq!(c.top_docs(), vec![(1, 0.8), (0, 0.2)]);

    let mut unlimited = KnnCollector::unlimited(1);
    unlimited.inc_visited_count(1_000_000);
    assert!(!unlimited.early_terminated());
    unlimited.collect(4, 1.0);
    assert_eq!(unlimited.pop_node(), 4);
    unlimited.clear();
    assert_eq!(unlimited.size(), 0);
    assert_eq!(unlimited.visited_count(), 0);
}

#[test]
fn pop_until_nearest_k_nodes_trims_to_k() {
    let mut c = KnnCollector::unlimited(2);
    // `add`-style unbounded growth is what `searchLevel` does through
    // `collect`; here go past k deliberately.
    c.queue.add(0, 0.1);
    c.queue.add(1, 0.2);
    c.queue.add(2, 0.3);
    let mut nodes = c.pop_until_nearest_k_nodes();
    nodes.sort_unstable();
    assert_eq!(nodes, vec![1, 2]);
}

// ---------------- NeighborArray ----------------

#[test]
fn neighbor_array_add_in_order_keeps_descending_scores() {
    let mut a = NeighborArray::new(8, true);
    a.add_in_order(5, 0.9);
    a.add_in_order(6, 0.5);
    a.add_in_order(7, 0.5);
    assert_eq!(a.size(), 3);
    assert_eq!(a.nodes(), &[5, 6, 7]);
    assert_eq!(a.score(1), 0.5);
    assert_eq!(a.max_size(), 8);
    a.clear();
    assert_eq!(a.size(), 0);
}

#[test]
#[should_panic(expected = "No growth is allowed")]
fn neighbor_array_refuses_to_grow_past_max_size() {
    let mut a = NeighborArray::new(1, true);
    a.add_in_order(0, 1.0);
    a.add_in_order(1, 0.5);
}

#[test]
#[should_panic(expected = "No growth is allowed")]
fn neighbor_array_out_of_order_also_refuses_to_grow() {
    let mut a = NeighborArray::new(1, false);
    a.add_out_of_order(0, 1.0);
    a.add_out_of_order(1, 0.5);
}

/// `addAndEnsureDiversity` evicts the worst **non-diverse** neighbour,
/// which is not the same as the worst-scoring one.
///
/// Base node 0 on a line; neighbours 1 (at 1.0) and 3 (at 5.0). Inserting
/// node 2 (at 1.01) overflows the array. Node 3 is the one dropped -- not
/// because it scores worst, but because the newly inserted node 2 is
/// *closer to 3* (distance 3.99) than the base node is (distance 5), so a
/// search that reaches 2 can reach 3 from there and the direct link is
/// redundant. This is `isWorstNonDiverse`'s second branch: the candidate
/// was diverse when it was added, and only the unchecked newcomer can have
/// invalidated it.
#[test]
fn add_and_ensure_diversity_drops_a_link_reachable_through_a_newcomer() {
    let mut scorer = Line::new(vec![0.0, 1.0, 1.01, 5.0]);
    let mut a = NeighborArray::new(3, true);
    scorer.set_scoring_ordinal(0).unwrap();
    a.add_in_order(1, scorer.score(1).unwrap());
    a.add_in_order(3, scorer.score(3).unwrap());
    let score2 = scorer.score(2).unwrap();
    a.add_and_ensure_diversity(2, score2, 0, &mut scorer)
        .unwrap();
    assert_eq!(a.size(), 2);
    assert_eq!(a.nodes(), &[1, 2]);
}

/// The other branch of `isWorstNonDiverse`: the **new** node is the one
/// dropped, because it duplicates a neighbour that is already there.
/// Base 0, neighbours 1 (at 1.0) and 2 (at 10.0); the newcomer 3 sits at
/// 10.01, right on top of 2.
#[test]
fn add_and_ensure_diversity_drops_a_newcomer_that_duplicates_a_neighbour() {
    let mut scorer = Line::new(vec![0.0, 1.0, 10.0, 10.01]);
    let mut a = NeighborArray::new(3, true);
    scorer.set_scoring_ordinal(0).unwrap();
    a.add_in_order(1, scorer.score(1).unwrap());
    a.add_in_order(2, scorer.score(2).unwrap());
    let score3 = scorer.score(3).unwrap();
    a.add_and_ensure_diversity(3, score3, 0, &mut scorer)
        .unwrap();
    assert_eq!(a.nodes(), &[1, 2]);
}

#[test]
fn add_and_ensure_diversity_is_a_no_op_below_max_size() {
    let mut scorer = Line::new(vec![0.0, 1.0, 2.0]);
    let mut a = NeighborArray::new(4, true);
    scorer.set_scoring_ordinal(0).unwrap();
    a.add_and_ensure_diversity(1, 0.5, 0, &mut scorer).unwrap();
    assert_eq!(a.size(), 1);
}

#[test]
fn ascending_neighbor_array_inserts_at_the_right_most_equal_position() {
    let mut scorer = Line::new(vec![0.0, 1.0, 2.0, 3.0]);
    let mut a = NeighborArray::new(8, false);
    a.add_in_order(0, 0.1);
    a.add_in_order(1, 0.5);
    a.add_in_order(2, 0.5);
    a.add_out_of_order(3, 0.5);
    let unchecked = a.sort(&mut scorer).unwrap().unwrap();
    // Equal scores insert to the right of the existing run.
    assert_eq!(unchecked, vec![3]);
    assert_eq!(a.nodes(), &[0, 1, 2, 3]);
    // A second call has nothing left to do.
    assert!(a.sort(&mut scorer).unwrap().is_none());
}

#[test]
fn ascending_neighbor_array_inserts_a_new_low_score_at_the_front() {
    let mut scorer = Line::new(vec![0.0, 1.0, 2.0]);
    let mut a = NeighborArray::new(8, false);
    a.add_in_order(0, 0.4);
    a.add_in_order(1, 0.6);
    a.add_out_of_order(2, 0.1);
    let unchecked = a.sort(&mut scorer).unwrap().unwrap();
    assert_eq!(unchecked, vec![0]);
    assert_eq!(a.nodes(), &[2, 0, 1]);
}

// ---------------- OnHeapHnswGraph ----------------

#[test]
fn on_heap_graph_tracks_levels_entry_node_and_membership() {
    let mut g = OnHeapHnswGraph::new(4);
    assert_eq!(g.max_conn(), 4);
    assert_eq!(g.entry_node(), -1);
    assert_eq!(g.size(), 0);
    assert_eq!(HnswGraphView::max_node_id(&g), -1);
    for level in (0..=2).rev() {
        g.add_node(level, 0);
    }
    assert!(g.try_set_new_entry_node(0, 2));
    assert!(!g.try_set_new_entry_node(1, 3));
    assert_eq!(g.entry_node(), 0);
    assert_eq!(g.num_levels(), 3);
    assert_eq!(g.size(), 1);
    assert_eq!(HnswGraphView::max_node_id(&g), 0);
    g.add_node(0, 1);
    assert_eq!(g.size(), 2);
    assert!(g.node_exists_at_level(0, 1));
    assert!(!g.node_exists_at_level(1, 1));
    assert!(!g.node_exists_at_level(0, 7));
    assert_eq!(g.sorted_nodes_on_level(0).unwrap(), vec![0, 1]);
    assert_eq!(g.sorted_nodes_on_level(2).unwrap(), vec![0]);
    // A level-0 neighbour array is `2M + 1` wide, an upper one `M + 1`.
    assert_eq!(g.neighbors(0, 0).max_size(), 9);
    assert_eq!(g.neighbors(1, 0).max_size(), 5);
    g.neighbors_mut(0, 0).add_in_order(1, 0.5);
    let mut out = vec![99];
    g.neighbors_into(0, 0, &mut out).unwrap();
    assert_eq!(out, vec![1]);

    // Promotion only fires when the caller's view of the top level is
    // current.
    g.add_node(3, 1);
    assert!(!g.try_promote_new_entry_node(1, 3, 1));
    assert!(g.try_promote_new_entry_node(1, 3, 2));
    assert_eq!(g.entry_node(), 1);
    assert_eq!(g.num_levels(), 4);
}

#[test]
fn renumbered_moves_nodes_lists_levels_and_entry() {
    let mut g = OnHeapHnswGraph::new(4);
    g.add_node(1, 0);
    g.add_node(0, 0);
    assert!(g.try_set_new_entry_node(0, 1));
    g.add_node(0, 1);
    g.add_node(0, 2);
    g.neighbors_mut(0, 0).add_in_order(1, 0.9);
    g.neighbors_mut(0, 0).add_in_order(2, 0.5);
    g.neighbors_mut(0, 2).add_in_order(0, 0.5);
    // 0 -> 2, 1 -> 0, 2 -> 1.
    let r = g.clone().renumbered(&[2, 0, 1]).unwrap();
    assert_eq!(r.size(), 3);
    assert_eq!(r.entry_node(), 2);
    assert_eq!(r.num_levels(), 2);
    assert_eq!(r.sorted_nodes_on_level(1).unwrap(), vec![2]);
    assert_eq!(r.neighbors(0, 2).nodes(), &[0, 1]);
    assert_eq!(r.neighbors(0, 2).score(0), 0.9);
    assert_eq!(r.neighbors(0, 1).nodes(), &[2]);
    assert!(r.neighbors(0, 0).nodes().is_empty());
    assert_eq!(HnswGraphView::max_node_id(&r), 2);

    // Not a permutation: too short, a repeat, out of range.
    assert!(g.clone().renumbered(&[0, 1]).is_err());
    assert!(g.clone().renumbered(&[0, 0, 1]).is_err());
    assert!(g.clone().renumbered(&[0, 1, 3]).is_err());
    assert!(g.clone().renumbered(&[0, 1, -1]).is_err());
    // A hole: ordinal 1 never added.
    let mut holes = OnHeapHnswGraph::new(4);
    holes.add_node(0, 0);
    holes.add_node(0, 2);
    assert!(holes.renumbered(&[0, 1, 2]).is_err());
}

#[test]
#[should_panic(expected = "M (max connections) must be in 1..=")]
fn on_heap_graph_rejects_a_non_positive_m() {
    OnHeapHnswGraph::new(0);
}

/// `2 * m + 1` is an `i32` multiply: an absurd `M` would overflow it (a
/// panic in debug, a negative `nsize0` in release, and from there a
/// `NeighborArray` whose `max_size` is nonsense).
#[test]
#[should_panic(expected = "M (max connections) must be in 1..=")]
fn on_heap_graph_rejects_an_m_that_would_overflow_the_level_zero_budget() {
    OnHeapHnswGraph::new(i32::MAX);
}

/// `add_node` takes ordinals as `i32` and indexes with them: a negative
/// one would sign-extend into `resize_with(~2^64)` -- an allocation abort,
/// not a panic -- and a negative level would push neighbour arrays until
/// memory ran out.
#[test]
#[should_panic(expected = "node ordinal must be non-negative")]
fn add_node_rejects_a_negative_ordinal() {
    OnHeapHnswGraph::new(4).add_node(0, -1);
}

#[test]
#[should_panic(expected = "level must be non-negative")]
fn add_node_rejects_a_negative_level() {
    OnHeapHnswGraph::new(4).add_node(-1, 0);
}

// ---------------- thresholds ----------------

#[test]
fn expected_visited_nodes_and_the_graph_threshold() {
    assert_eq!(expected_visited_nodes(10, 50_000), 108);
    // k <= 0 disables the optimisation entirely.
    assert!(should_create_graph(0, 1));
    assert!(should_create_graph(-1, 1));
    // ln(1) == 0, so the expected count is 0 and no graph is worth it.
    assert!(!should_create_graph(100, 1));
    assert!(!should_create_graph(100, 500));
    assert!(should_create_graph(100, 5000));
}

// ---------------- builder + searcher ----------------

fn build_line_graph(n: usize, m: i32, beam: i32) -> (OnHeapHnswGraph, Line) {
    let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let scorer = Line::new(values.clone());
    let graph = HnswGraphBuilder::new(scorer, m, beam, DEFAULT_RAND_SEED)
        .unwrap()
        .build(n as i32)
        .unwrap();
    (graph, Line::new(values))
}

#[test]
fn builder_rejects_non_positive_parameters() {
    let scorer = Line::new(vec![0.0]);
    assert!(matches!(
        HnswGraphBuilder::new(scorer.clone(), 0, 10, 42),
        Err(Error::InvalidGraphParameter(_))
    ));
    assert!(matches!(
        HnswGraphBuilder::new(scorer.clone(), 4, 0, 42),
        Err(Error::InvalidGraphParameter(_))
    ));
    // The upper bounds too: a graph past them writes a `.vem` no reader
    // (this port's or Lucene's) will accept.
    assert!(matches!(
        HnswGraphBuilder::new(scorer.clone(), MAXIMUM_MAX_CONN + 1, 10, 42),
        Err(Error::InvalidGraphParameter(_))
    ));
    assert!(matches!(
        HnswGraphBuilder::new(scorer, 4, MAXIMUM_BEAM_WIDTH + 1, 42),
        Err(Error::InvalidGraphParameter(_))
    ));
}

#[test]
fn a_single_node_graph_is_its_own_entry_point() {
    let (graph, mut scorer) = build_line_graph(1, 4, 10);
    assert_eq!(graph.size(), 1);
    assert_eq!(graph.entry_node(), 0);
    scorer.query = 0.0;
    let mut collector = KnnCollector::new(1, u64::MAX);
    let mut searcher = HnswGraphSearcher::new(1, graph.size());
    searcher
        .search(&mut collector, &mut scorer, &graph, None)
        .unwrap();
    assert_eq!(collector.top_docs(), vec![(0, 1.0)]);
}

/// `SeededHnswGraphSearcher.fromEntryPoints`' two rejections, ported as
/// errors rather than as Java's `IllegalArgumentException`/`assert` pair.
/// The `assert` one matters: an ordinal outside the graph indexes the
/// `visited` bitset, so in a Java production build it is undefined
/// behaviour on a `Bits` and here it would be a panic -- neither is an
/// answer a caller can act on.
#[test]
fn seeded_entry_points_are_validated_before_they_reach_the_bitset() {
    let (graph, mut scorer) = build_line_graph(50, 4, 20);
    let mut collector = KnnCollector::new(3, u64::MAX);
    let mut searcher = HnswGraphSearcher::new(3, graph.size());
    for bad in [-1, graph.size(), i32::MAX] {
        let e = searcher
            .search_seeded(&mut collector, &mut scorer, &graph, None, &[0, bad])
            .unwrap_err();
        assert!(
            matches!(e, Error::InvalidGraphParameter(_))
                && e.to_string().contains("outside the graph"),
            "{bad}: {e}"
        );
    }
    let e = searcher
        .search_seeded(&mut collector, &mut scorer, &graph, None, &[])
        .unwrap_err();
    assert!(e.to_string().contains("must be > 0"), "{e}");
}

/// Seeding replaces the entry-point descent and nothing else. On a line
/// graph the two are directly comparable: seeding with the answer the
/// unseeded walk found returns that same answer, and seeding with one
/// far-away node under a visit cap of one returns exactly that node --
/// which is what says the seeds are the *starting set* rather than a
/// hint.
#[test]
fn a_seeded_search_starts_where_it_is_told() {
    let (graph, mut scorer) = build_line_graph(200, 8, 40);
    assert!(
        graph.num_levels() >= 2,
        "the descent must have work to skip"
    );
    scorer.query = 111.9;
    let mut plain = KnnCollector::new(3, u64::MAX);
    let mut searcher = HnswGraphSearcher::new(3, graph.size());
    searcher
        .search(&mut plain, &mut scorer, &graph, None)
        .unwrap();
    let plain = plain.top_docs();
    assert_eq!(
        plain.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        vec![112, 111, 113]
    );

    let mut seeds: Vec<i32> = plain.iter().map(|(n, _)| *n).collect();
    seeds.sort_unstable();
    let mut seeded = KnnCollector::new(3, u64::MAX);
    searcher
        .search_seeded(&mut seeded, &mut scorer, &graph, None, &seeds)
        .unwrap();
    assert_eq!(seeded.top_docs(), plain);

    let mut pinned = KnnCollector::new(3, 1);
    searcher
        .search_seeded(&mut pinned, &mut scorer, &graph, None, &[0])
        .unwrap();
    assert_eq!(
        pinned
            .top_docs()
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>(),
        vec![0]
    );
}

/// The accept set gates collection on the seeded path exactly as it does
/// on the unseeded one -- a seed that is not accepted is still scored and
/// still explored, but never collected. That is Java's `scoreEntryPoints`,
/// which both searchers share.
#[test]
fn a_seeded_search_still_honours_the_accept_set() {
    let (graph, mut scorer) = build_line_graph(200, 8, 40);
    scorer.query = 111.9;
    let mut accept = FixedBitSet::new(200);
    accept.set(111);
    accept.set(113);
    let mut collector = KnnCollector::new(3, u64::MAX);
    let mut searcher = HnswGraphSearcher::new(3, graph.size());
    searcher
        .search_seeded(
            &mut collector,
            &mut scorer,
            &graph,
            Some(&accept),
            &[111, 112, 113],
        )
        .unwrap();
    assert_eq!(
        collector
            .top_docs()
            .iter()
            .map(|(n, _)| *n)
            .collect::<Vec<_>>(),
        vec![111, 113]
    );
}

/// On a line, exact nearest neighbours are obvious, so a graph search that
/// returns anything else has a real defect -- this is the smallest end to
/// end check of build + descend + beam search there is.
#[test]
fn graph_search_finds_the_true_nearest_neighbours_on_a_line() {
    let (graph, mut scorer) = build_line_graph(200, 8, 40);
    assert!(graph.num_levels() >= 2, "expected a multi-level graph");
    for target in [0.5f32, 37.2, 111.9, 199.0] {
        scorer.query = target;
        let mut collector = KnnCollector::new(3, u64::MAX);
        let mut searcher = HnswGraphSearcher::new(3, graph.size());
        searcher
            .search(&mut collector, &mut scorer, &graph, None)
            .unwrap();
        let got: Vec<i32> = collector.top_docs().into_iter().map(|(n, _)| n).collect();
        let mut want: Vec<i32> = (0..200).collect();
        want.sort_by(|a, b| {
            (*a as f32 - target)
                .abs()
                .total_cmp(&(*b as f32 - target).abs())
        });
        assert_eq!(got, want[..3].to_vec(), "target {target}");
    }
}

#[test]
fn search_honours_accepted_ordinals_and_the_visit_limit() {
    let (graph, mut scorer) = build_line_graph(200, 8, 40);
    scorer.query = 100.0;
    let mut accept = FixedBitSet::new(200);
    for n in [10, 150, 199] {
        accept.set(n);
    }
    let mut collector = KnnCollector::new(2, u64::MAX);
    let mut searcher = HnswGraphSearcher::new(2, graph.size());
    searcher
        .search(&mut collector, &mut scorer, &graph, Some(&accept))
        .unwrap();
    for (node, _) in collector.top_docs() {
        assert!(accept.get(node as usize), "collected rejected node {node}");
    }

    // A visit limit of 1 stops the descent immediately.
    let mut limited = KnnCollector::new(2, 1);
    let mut searcher = HnswGraphSearcher::new(2, graph.size());
    searcher
        .search(&mut limited, &mut scorer, &graph, None)
        .unwrap();
    assert!(limited.early_terminated());
}

#[test]
fn find_best_entry_point_reports_no_entry_for_an_empty_graph() {
    let graph = OnHeapHnswGraph::new(4);
    let mut scorer = Line::new(vec![0.0]);
    let mut collector = KnnCollector::new(1, u64::MAX);
    let mut searcher = HnswGraphSearcher::new(1, 1);
    assert_eq!(
        searcher
            .find_best_entry_point(&mut scorer, &graph, &mut collector)
            .unwrap(),
        -1
    );
    // ... and `search` then collects nothing.
    searcher
        .search(&mut collector, &mut scorer, &graph, None)
        .unwrap();
    assert_eq!(collector.size(), 0);
}

#[test]
fn find_best_entry_point_stops_when_the_visit_limit_is_hit() {
    let (graph, mut scorer) = build_line_graph(200, 8, 40);
    assert!(graph.num_levels() > 1);
    scorer.query = 100.0;
    let mut collector = KnnCollector::new(2, 2);
    let mut searcher = HnswGraphSearcher::new(2, graph.size());
    assert_eq!(
        searcher
            .find_best_entry_point(&mut scorer, &graph, &mut collector)
            .unwrap(),
        -1
    );
}

/// `M == 1` is the degenerate `ml` branch (`1 / ln(1)` would be infinite,
/// so Java hard-codes 1).
#[test]
fn m_of_one_builds_a_usable_graph() {
    let (graph, mut scorer) = build_line_graph(40, 1, 8);
    assert_eq!(graph.max_conn(), 1);
    scorer.query = 20.0;
    let mut collector = KnnCollector::new(1, u64::MAX);
    let mut searcher = HnswGraphSearcher::new(1, graph.size());
    searcher
        .search(&mut collector, &mut scorer, &graph, None)
        .unwrap();
    assert_eq!(collector.size(), 1);
}

/// The diversity rule in isolation: a candidate closer to an existing
/// neighbour than to the new node is rejected.
#[test]
fn diversity_check_rejects_a_candidate_shadowed_by_a_neighbour() {
    let mut scorer = Line::new(vec![0.0, 1.0, 1.05]);
    let mut neighbors = NeighborArray::new(4, true);
    neighbors.add_in_order(1, 0.5);
    // Candidate 2 sits next to neighbour 1; scoring it against 1 gives a
    // higher similarity than its own score to the new node, so it is not
    // diverse.
    scorer.set_scoring_ordinal(2).unwrap();
    assert!(!HnswGraphBuilder::<Line>::diversity_check(0.5, &neighbors, &mut scorer).unwrap());
    // A far-away candidate is diverse.
    scorer.set_scoring_ordinal(0).unwrap();
    assert!(HnswGraphBuilder::<Line>::diversity_check(0.99, &neighbors, &mut scorer).unwrap());
    // An empty neighbour set accepts everything.
    assert!(HnswGraphBuilder::<Line>::diversity_check(
        0.0,
        &NeighborArray::new(4, true),
        &mut scorer
    )
    .unwrap());
}

#[test]
fn builder_exposes_its_graph_while_building() {
    let scorer = Line::new(vec![0.0, 1.0, 2.0]);
    let mut builder = HnswGraphBuilder::new(scorer, 4, 8, DEFAULT_RAND_SEED).unwrap();
    builder.add_graph_node(0).unwrap();
    builder.add_graph_node(1).unwrap();
    assert_eq!(builder.graph().size(), 2);
}

#[test]
fn next_up_matches_javas_math_next_up() {
    assert_eq!(next_up(f32::NEG_INFINITY), f32::MIN);
    assert_eq!(next_up(0.0), f32::from_bits(1));
    assert!(next_up(1.0) > 1.0);
}
// ---------------- merge-time construction ----------------

/// Every arc of a graph, level by level, in a form two graphs can be
/// compared by. Neighbour *order* is part of it: `writeGraph` sorts the
/// list before serializing, but `addAndEnsureDiversity` keeps it in score
/// order, so a difference in order is a difference in which arcs survived
/// pruning.
fn arcs<G: HnswGraphView>(graph: &G) -> Vec<(i32, i32, Vec<i32>)> {
    let mut out = Vec::new();
    let mut neighbors = Vec::new();
    for level in 0..graph.num_levels() {
        for node in graph.sorted_nodes_on_level(level).unwrap() {
            graph.neighbors_into(level, node, &mut neighbors).unwrap();
            out.push((level, node, neighbors.clone()));
        }
    }
    out
}

/// Structural invariants every graph handed to `write_hnsw_vectors` must
/// satisfy, whether it was built or merged.
fn assert_well_formed<G: HnswGraphView>(graph: &G, expected_nodes: i32) {
    let level0 = graph.sorted_nodes_on_level(0).unwrap();
    assert_eq!(
        level0.len() as i32,
        expected_nodes,
        "level 0 must contain every ordinal"
    );
    assert_eq!(level0, (0..expected_nodes).collect::<Vec<_>>());
    let mut neighbors = Vec::new();
    for level in 0..graph.num_levels() {
        let nodes = graph.sorted_nodes_on_level(level).unwrap();
        for node in &nodes {
            graph.neighbors_into(level, *node, &mut neighbors).unwrap();
            let budget = if level == 0 {
                graph.max_conn() * 2
            } else {
                graph.max_conn()
            };
            assert!(
                neighbors.len() as i32 <= budget,
                "level {level} node {node} has {} arcs, budget {budget}",
                neighbors.len()
            );
            let mut seen = std::collections::HashSet::new();
            for n in &neighbors {
                assert!(*n >= 0 && *n < expected_nodes, "arc to {n} out of range");
                assert_ne!(*n, *node, "self arc at level {level} node {node}");
                assert!(
                    seen.insert(*n),
                    "duplicate arc {node} -> {n} at level {level}"
                );
            }
        }
        if level > 0 {
            // Every node on a level must also be on the level below it --
            // the property `writeGraph`'s per-level node lists and
            // `findBestEntryPoint`'s descent both depend on.
            let below = graph.sorted_nodes_on_level(level - 1).unwrap();
            for node in &nodes {
                assert!(
                    below.contains(node),
                    "node {node} is on level {level} but not on level {}",
                    level - 1
                );
            }
        }
    }
}

#[test]
fn the_join_set_is_a_small_cover_of_the_graph() {
    let (graph, _) = build_line_graph(400, 16, 100);
    let join = compute_join_set(&graph).unwrap();
    assert!(!join.is_empty());
    assert!(
        (join.len() as i32) < graph.size(),
        "a join set covering the whole graph makes the merge no cheaper than a rebuild: \
             {} of {}",
        join.len(),
        graph.size()
    );
    assert!(join.iter().all(|n| *n >= 0 && *n < graph.size()));
    // Every node outside the join set must have at least one neighbour
    // inside it -- otherwise `updateGraph` would insert it with an empty
    // entry-point set and fall back to a full search, which is the cost
    // the join set exists to avoid.
    let mut neighbors = Vec::new();
    for node in 0..graph.size() {
        if join.contains(&node) {
            continue;
        }
        graph.neighbors_into(0, node, &mut neighbors).unwrap();
        assert!(
            neighbors.iter().any(|n| join.contains(n)) || neighbors.is_empty(),
            "node {node} is covered by nothing in the join set"
        );
    }
}

/// `coverage` clamps to the node's own degree, without which every leaf of
/// a degenerate (degree-1) graph joins the set and the join set becomes
/// the whole graph -- the case the Java comment calls out.
#[test]
fn join_set_coverage_is_clamped_to_the_degree() {
    assert_eq!(join_set_coverage(0), 0);
    assert_eq!(join_set_coverage(1), 1);
    assert_eq!(join_set_coverage(2), 2);
    assert_eq!(join_set_coverage(8), 2);
    assert_eq!(join_set_coverage(9), 3);
    assert_eq!(join_set_coverage(32), 8);
}

#[test]
fn the_gain_encoding_round_trips_and_orders_largest_first() {
    for (gain, node) in [(0i32, 0i32), (1, 5), (37, 99999), (i32::MAX / 2, 7)] {
        let e = encode_gain(gain, node);
        assert_eq!(decode_gain(e), gain);
        assert_eq!(decode_gain_node(e), node);
    }
    // The heap is a min-heap, so a bigger gain must encode smaller.
    assert!(encode_gain(10, 0) < encode_gain(1, 0));
}

/// With one source graph, no deletions and an identity ordinal map, the
/// merge is a pure structural copy: the merged graph must be **arc for
/// arc** the source graph. This is the assertion that actually
/// discriminates -- recall would not.
#[test]
fn merging_one_undeleted_graph_reproduces_it_arc_for_arc() {
    let n = 300;
    let (source, scorer) = build_line_graph(n, 16, 100);
    let ord_map: Vec<i32> = (0..n as i32).collect();
    let merged = merge_graphs(
        scorer,
        100,
        DEFAULT_RAND_SEED,
        &[&source],
        &[ord_map],
        n as i32,
        None,
    )
    .unwrap();
    assert_eq!(merged.num_levels(), source.num_levels());
    assert_eq!(merged.entry_node(), source.entry_node());
    assert_eq!(arcs(&merged), arcs(&source));
    assert_well_formed(&merged, n as i32);
}

/// Reversing the ordinal space is still a pure copy, just relabelled --
/// which is what catches a merge that copies the *structure* correctly but
/// forgets to remap, since with an identity map the two are the same
/// thing.
#[test]
fn a_permuted_ordinal_map_relabels_every_arc() {
    let n = 200i32;
    let (source, _) = build_line_graph(n as usize, 16, 100);
    // The scorer must score merged ordinals, so build one over the
    // permuted values.
    let permuted_values: Vec<f32> = (0..n).map(|i| (n - 1 - i) as f32).collect();
    let ord_map: Vec<i32> = (0..n).map(|i| n - 1 - i).collect();
    let merged = merge_graphs(
        Line::new(permuted_values),
        100,
        DEFAULT_RAND_SEED,
        &[&source],
        std::slice::from_ref(&ord_map),
        n,
        None,
    )
    .unwrap();
    assert_well_formed(&merged, n);
    let mut expected = Vec::new();
    let mut got = Vec::new();
    let mut buf = Vec::new();
    for level in 0..source.num_levels() {
        for node in source.sorted_nodes_on_level(level).unwrap() {
            source.neighbors_into(level, node, &mut buf).unwrap();
            expected.push((
                level,
                ord_map[node as usize],
                buf.iter().map(|n| ord_map[*n as usize]).collect::<Vec<_>>(),
            ));
        }
    }
    for level in 0..merged.num_levels() {
        for node in merged.sorted_nodes_on_level(level).unwrap() {
            merged.neighbors_into(level, node, &mut buf).unwrap();
            got.push((level, node, buf.clone()));
        }
    }
    expected.sort();
    got.sort();
    assert_eq!(got, expected);
}

/// Deleted ordinals must vanish from the merged graph, and no surviving
/// node may keep an arc to one -- a dangling arc is an ordinal out of
/// range for the merged `.vec`, which the writer would happily serialize
/// and the reader would reject (or worse, resolve to the wrong vector).
#[test]
fn merging_a_graph_with_deletions_leaves_no_dangling_arcs() {
    let n = 300i32;
    let (source, _) = build_line_graph(n as usize, 16, 100);
    // Drop every third vector.
    let mut ord_map = vec![-1i32; n as usize];
    let mut kept_values = Vec::new();
    let mut next = 0i32;
    for old in 0..n {
        if old % 3 != 0 {
            ord_map[old as usize] = next;
            kept_values.push(old as f32);
            next += 1;
        }
    }
    let total = next;
    let merged = merge_graphs(
        Line::new(kept_values),
        100,
        DEFAULT_RAND_SEED,
        &[&source],
        &[ord_map],
        total,
        None,
    )
    .unwrap();
    assert_well_formed(&merged, total);
    // Repair must actually reconnect: no surviving node may be isolated.
    let mut neighbors = Vec::new();
    for node in 0..total {
        merged.neighbors_into(0, node, &mut neighbors).unwrap();
        assert!(!neighbors.is_empty(), "node {node} lost every neighbour");
    }
}

/// Two graphs folded together: every ordinal of both must be present, and
/// the result must be searchable -- the second graph's nodes are inserted
/// with entry points derived from the first, which is the part that goes
/// wrong silently.
#[test]
fn merging_two_graphs_keeps_every_ordinal_and_stays_searchable() {
    // Two disjoint halves of one line, so the merged graph has to connect
    // regions neither source graph ever linked.
    let a_values: Vec<f32> = (0..200).map(|i| i as f32).collect();
    let b_values: Vec<f32> = (200..500).map(|i| i as f32).collect();
    let a = HnswGraphBuilder::new(Line::new(a_values.clone()), 16, 100, DEFAULT_RAND_SEED)
        .unwrap()
        .build(200)
        .unwrap();
    let b = HnswGraphBuilder::new(Line::new(b_values.clone()), 16, 100, DEFAULT_RAND_SEED)
        .unwrap()
        .build(300)
        .unwrap();
    let merged_values: Vec<f32> = (0..500).map(|i| i as f32).collect();
    // B is the larger graph, so it becomes the base; A is folded in.
    let b_map: Vec<i32> = (200..500).collect();
    let a_map: Vec<i32> = (0..200).collect();
    let merged = merge_graphs(
        Line::new(merged_values.clone()),
        100,
        DEFAULT_RAND_SEED,
        &[&b, &a],
        &[b_map, a_map],
        500,
        None,
    )
    .unwrap();
    assert_well_formed(&merged, 500);

    // Searchable across the seam: a query in A's range must find A's
    // vectors even though the search starts from B's entry node.
    let mut searcher = HnswGraphSearcher::new(10, merged.size());
    for target in [3.0f32, 199.0, 250.0, 499.0] {
        let mut collector = KnnCollector::new(10, u64::MAX);
        let mut scorer = Line {
            values: merged_values.clone(),
            query: target,
        };
        searcher
            .search(&mut collector, &mut scorer, &merged, None)
            .unwrap();
        let found: Vec<i32> = collector.top_docs().into_iter().map(|(n, _)| n).collect();
        assert!(
            found.contains(&(target as i32)),
            "searching for {target} over the merged graph returned {found:?}"
        );
    }
}

/// An ordinal no source graph covers is inserted from scratch -- that is
/// what `initialized_nodes`' clear bits are for. Without it the merged
/// graph is missing nodes and `write_hnsw_vectors` refuses it.
#[test]
fn ordinals_no_source_graph_covers_are_added_from_scratch() {
    let n = 200i32;
    let (source, _) = build_line_graph(n as usize, 16, 100);
    let total = n + 50;
    let values: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let ord_map: Vec<i32> = (0..n).collect();
    let mut initialized = FixedBitSet::new(total as usize);
    for i in 0..n {
        initialized.set(i as usize);
    }
    let merged = merge_graphs(
        Line::new(values),
        100,
        DEFAULT_RAND_SEED,
        &[&source],
        &[ord_map],
        total,
        Some(&initialized),
    )
    .unwrap();
    assert_well_formed(&merged, total);
}

#[test]
fn merge_graphs_rejects_mismatched_inputs() {
    let (source, scorer) = build_line_graph(50, 16, 100);
    let empty: [&OnHeapHnswGraph; 0] = [];
    assert!(matches!(
        merge_graphs(
            scorer.clone(),
            100,
            DEFAULT_RAND_SEED,
            &empty,
            &[],
            50,
            None
        ),
        Err(Error::InvalidGraphParameter(_))
    ));
    // One graph, no ordinal map.
    assert!(matches!(
        merge_graphs(
            scorer.clone(),
            100,
            DEFAULT_RAND_SEED,
            &[&source],
            &[],
            50,
            None
        ),
        Err(Error::InvalidGraphParameter(_))
    ));
    // An ordinal map too short for the source graph. This has to be
    // rejected up front, on the map's *length*: rejecting it only when a
    // lookup happens to fall off the end would depend on which ordinals the
    // level assignment put where, i.e. it would pass or panic by luck.
    assert!(matches!(
        merge_graphs(
            scorer.clone(),
            100,
            DEFAULT_RAND_SEED,
            &[&source],
            &[vec![0, 1, 2]],
            50,
            None
        ),
        Err(Error::InvalidGraphParameter(_))
    ));
    // ...and so is a map that is long enough but names an out-of-range
    // merged ordinal for a *neighbour* rather than for the node itself --
    // the lookup `copy_graph_structure` performs unchecked per entry.
    let short_but_long_enough: Vec<i32> = (0..50).collect();
    assert!(merge_graphs(
        scorer,
        100,
        DEFAULT_RAND_SEED,
        &[&source],
        &[short_but_long_enough],
        50,
        None
    )
    .is_ok());
}

/// A `.vex` can carry an upper-level node with more arcs than the level's
/// budget and still pass its checksum. Copying it must be an error, not a
/// `NeighborArray` panic three frames down.
#[test]
fn a_source_node_with_more_arcs_than_the_level_allows_is_rejected() {
    /// A graph that reports one over-full node on level 1.
    struct OverFull {
        m: i32,
    }
    impl HnswGraphView for OverFull {
        fn size(&self) -> i32 {
            64
        }
        fn num_levels(&self) -> i32 {
            2
        }
        fn entry_node(&self) -> i32 {
            0
        }
        fn max_conn(&self) -> i32 {
            self.m
        }
        fn neighbors_into(&self, level: i32, _node: i32, out: &mut Vec<i32>) -> Result<()> {
            out.clear();
            // `M + 1` arcs on an upper level, where the budget is `M`.
            let count = if level == 0 { 4 } else { self.m + 1 };
            out.extend(1..=count);
            Ok(())
        }
        fn sorted_nodes_on_level(&self, level: i32) -> Result<Vec<i32>> {
            Ok(if level == 0 {
                (0..64).collect()
            } else {
                vec![0]
            })
        }
    }
    let graph = OverFull { m: 8 };
    let err = merge_graphs(
        Line::new((0..64).map(|i| i as f32).collect()),
        100,
        DEFAULT_RAND_SEED,
        &[&graph],
        &[(0..64).collect()],
        64,
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::CorruptMeta(_)),
        "expected a corrupt-metadata error, got {err:?}"
    );
}

/// A graph whose neighbour lists name ordinals past its own `size()`.
/// Every `HnswGraphView` in this crate rejects those when it decodes them,
/// so this is the searcher's own bound: without it the ordinal indexes
/// `visited` and the query dies on `FixedBitSet`'s own out-of-range panic
/// instead of returning `Error::InvalidGraphParameter`.
struct WildArcs {
    neighbour: i32,
    count: usize,
}
impl HnswGraphView for WildArcs {
    fn size(&self) -> i32 {
        16
    }
    fn num_levels(&self) -> i32 {
        1
    }
    fn entry_node(&self) -> i32 {
        0
    }
    fn max_conn(&self) -> i32 {
        4
    }
    fn neighbors_into(&self, _level: i32, _node: i32, out: &mut Vec<i32>) -> Result<()> {
        out.clear();
        // Distinct ordinals: a repeated one is skipped as already visited
        // and never reaches the bulk buffer.
        out.extend((0..self.count as i32).map(|i| self.neighbour + i));
        Ok(())
    }
    fn sorted_nodes_on_level(&self, _level: i32) -> Result<Vec<i32>> {
        Ok((0..16).collect())
    }
}

#[test]
fn a_neighbour_ordinal_past_the_graph_never_reaches_the_visited_bitset() {
    let mut scorer = Line::new((0..16).map(|i| i as f32).collect());
    for neighbour in [16i32, 1_000_000, -1] {
        let graph = WildArcs {
            neighbour,
            count: 1,
        };
        let mut searcher = HnswGraphSearcher::new(4, graph.size());
        let mut collector = KnnCollector::unlimited(4);
        let err = searcher
            .search(&mut collector, &mut scorer, &graph, None)
            .unwrap_err();
        assert!(
            matches!(err, Error::CorruptMeta(_)),
            "neighbour {neighbour} should be rejected, got {err:?}"
        );
    }
}

/// The same loop writes `bulk_nodes[num_nodes]`, an array sized
/// `2 * maxConn`. Java asserts nothing about it at all: a longer neighbour
/// list is an `ArrayIndexOutOfBoundsException` there and was an index
/// panic here.
#[test]
fn more_neighbours_than_the_bulk_buffer_holds_is_an_error_not_a_panic() {
    let mut scorer = Line::new((0..16).map(|i| i as f32).collect());
    let graph = WildArcs {
        neighbour: 0,
        // Ten arcs, of which nine are unvisited when the entry node's
        // list is walked -- one more than the `2 * max_conn == 8` the bulk
        // buffer holds.
        count: 10,
    };
    let mut searcher = HnswGraphSearcher::new(4, graph.size());
    let mut collector = KnnCollector::unlimited(4);
    let err = searcher
        .search(&mut collector, &mut scorer, &graph, None)
        .unwrap_err();
    assert!(
        matches!(err, Error::CorruptMeta(_)),
        "expected a corrupt-metadata error, got {err:?}"
    );
}

/// `compute_join_set` sizes three arrays from the source graph's `size()`,
/// which a `.vem` supplies: two billion nodes is an 8.6 GB `vec![0i32; n]`
/// and an abort, reached before any ordinal map is consulted. One map
/// entry per source node -- what the first graph's copy already requires --
/// bounds it by memory the caller has already committed.
#[test]
fn a_source_graph_larger_than_its_ordinal_map_is_rejected() {
    let (a, _) = build_line_graph(64, 8, 32);
    let (b, _) = build_line_graph(32, 8, 32);
    let err = merge_graphs(
        Line::new((0..64).map(|i| i as f32).collect()),
        32,
        DEFAULT_RAND_SEED,
        &[&a, &b],
        &[(0..64).collect(), vec![0i32; 4]],
        64,
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidGraphParameter(_)),
        "expected an invalid-parameter error, got {err:?}"
    );
}

/// `merge_graphs`'s `initialized_nodes` is a caller's bitset bounded
/// against a *separate* caller argument -- the pair the arithmetic gate
/// names. A short one reads ghost bits past `num_bits` and decides that a
/// node needs no insertion.
#[test]
fn a_short_initialized_node_set_is_rejected() {
    let (source, _) = build_line_graph(64, 8, 32);
    let short = FixedBitSet::new(16);
    let err = merge_graphs(
        Line::new((0..64).map(|i| i as f32).collect()),
        32,
        DEFAULT_RAND_SEED,
        &[&source],
        &[(0..64).collect()],
        64,
        Some(&short),
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidGraphParameter(_)),
        "expected an invalid-parameter error, got {err:?}"
    );
}

/// `rebalanceGraph`'s `maxNodesAtLevel = size * (1/M)^level` never drops
/// to zero when `M == 1`, so Java's `for (int level = 1; ; level++)` runs
/// forever, promoting every node to level after level until it exhausts
/// memory. `M == 1` is a legal setting and a merge reads `M` from the base
/// graph's `.vem`, so this was reachable from a file -- and a hang is the
/// one failure `catch_unwind` cannot turn into a JVM exception.
///
/// The rebalance only runs when the merge dropped something, so the test
/// needs deletions as well as `M == 1`.
#[test]
fn merging_a_deleted_graph_at_m_one_terminates() {
    let n = 64i32;
    let (source, _) = build_line_graph(n as usize, 1, 32);
    assert_eq!(source.max_conn(), 1);
    let mut ord_map = vec![-1i32; n as usize];
    let mut kept = Vec::new();
    let mut next = 0i32;
    for old in 0..n {
        if old % 2 == 0 {
            ord_map[old as usize] = next;
            kept.push(old as f32);
            next += 1;
        }
    }
    let merged = merge_graphs(
        Line::new(kept),
        32,
        DEFAULT_RAND_SEED,
        &[&source],
        &[ord_map],
        next,
        None,
    )
    .unwrap();
    assert_eq!(merged.size(), next);
}

/// `with_graph` takes `M` from the graph it is handed, so it has to check
/// bounds `HnswGraphBuilder::new` checks before the graph exists -- an
/// `OnHeapHnswGraph` can legally be constructed past `MAXIMUM_MAX_CONN`,
/// and the `.vem` it would produce is rejected by this port's reader and by
/// Lucene's.
#[test]
fn with_graph_rejects_a_graph_or_beam_width_out_of_bounds() {
    let scorer = Line::new(vec![0.0, 1.0]);
    assert!(matches!(
        HnswGraphBuilder::with_graph(
            scorer.clone(),
            10,
            DEFAULT_RAND_SEED,
            OnHeapHnswGraph::new(MAXIMUM_MAX_CONN + 1)
        ),
        Err(Error::InvalidGraphParameter(_))
    ));
    assert!(matches!(
        HnswGraphBuilder::with_graph(
            scorer.clone(),
            0,
            DEFAULT_RAND_SEED,
            OnHeapHnswGraph::new(8)
        ),
        Err(Error::InvalidGraphParameter(_))
    ));
    assert!(matches!(
        HnswGraphBuilder::with_graph(
            scorer,
            MAXIMUM_BEAM_WIDTH + 1,
            DEFAULT_RAND_SEED,
            OnHeapHnswGraph::new(8)
        ),
        Err(Error::InvalidGraphParameter(_))
    ));
}

/// `InitializedHnswGraphBuilder.initGraph` on its own -- the public entry
/// point `merge_graphs` inlines. With no deletions it must be the pure
/// structural copy the merge depends on, arc for arc.
#[test]
fn init_graph_copies_a_graph_into_a_larger_ordinal_space() {
    let n = 300i32;
    let (source, scorer) = build_line_graph(n as usize, 16, 100);
    // A larger merged space, with the source occupying its front half.
    let total = n + 40;
    let copied = HnswGraphBuilder::init_graph(
        scorer,
        100,
        DEFAULT_RAND_SEED,
        &source,
        &(0..n).collect::<Vec<_>>(),
        total,
    )
    .unwrap();
    assert_eq!(copied.entry_node(), source.entry_node());
    assert_eq!(copied.num_levels(), source.num_levels());
    assert_eq!(arcs(&copied), arcs(&source));
    // The extra ordinals exist as slots but carry nothing yet -- that is
    // what `merge_graphs`' `initialized_nodes` scan is for.
    assert_eq!(copied.size(), n);
    assert_eq!(HnswGraphView::max_node_id(&copied), total - 1);
}

/// Euclidean scorer over random multi-dimensional points: unlike [`Line`],
/// diversity pruning leaves a node many neighbours, so a degree can be
/// compared against the level's budget.
#[derive(Debug, Clone)]
struct Points {
    values: Vec<Vec<f32>>,
    query: usize,
}

impl VectorScorer for Points {
    fn score(&mut self, node: i32) -> Result<f32> {
        let a = &self.values[self.query];
        let b = &self.values[node as usize];
        let d: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        Ok(1.0 / (1.0 + d))
    }

    fn max_ord(&self) -> i32 {
        self.values.len() as i32
    }
}

impl UpdateableVectorScorer for Points {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        self.query = ord as usize;
        Ok(())
    }
}

/// Modelled on Java's `TestHnswFloatVectorGraph
/// .testReconnectsNodeDecayedBelowFloorDuringMerge` (GITHUB#16552): a node
/// whose degree has already decayed below `CUMULATIVE_DEGREE_FLOOR_FACTOR`
/// of the level budget and loses one more neighbour is too small a loss
/// for the per-merge `DISCONNECTED_NODE_FACTOR` check, but the cumulative
/// check must still flag and reconnect it.
#[test]
fn a_node_decayed_below_the_degree_floor_is_reconnected_on_merge() {
    let (m, beam, size, dim) = (16, 100, 128usize, 16usize);
    let mut state = 42u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let values: Vec<Vec<f32>> = (0..size)
        .map(|_| (0..dim).map(|_| next()).collect())
        .collect();
    let scorer = Points { values, query: 0 };
    let mut graph = HnswGraphBuilder::new(scorer.clone(), m, beam, 42)
        .unwrap()
        .build(size as i32)
        .unwrap();
    let entry = graph.entry_node();
    let target = if entry == 0 { 1 } else { 0 };

    // Degree 10, below the floor (0.5 * 2M = 16); dropping one neighbour
    // leaves 9, which 9 >= 10 * 0.85 lets past the per-merge check.
    let degree_before = 10usize;
    let neighbors = graph.neighbors_mut(0, target);
    neighbors.clear();
    let mut added = 0;
    let mut node = 0;
    while added < degree_before {
        if node != target && node != entry {
            neighbors.add_out_of_order(node, f32::NAN);
            added += 1;
        }
        node += 1;
    }
    let deleted = neighbors.nodes()[0];
    let mut new_ord_map: Vec<i32> = (0..size as i32).collect();
    new_ord_map[deleted as usize] = -1;

    // The copy phase itself must flag the target: the end-to-end check
    // below can also pass through back-links other repaired nodes add.
    let mut copier = HnswGraphBuilder::with_graph(
        scorer.clone(),
        beam,
        42,
        OnHeapHnswGraph::with_size(m, size as i32),
    )
    .unwrap();
    let (disconnected, _, has_deletes) = copier.copy_graph_structure(&graph, &new_ord_map).unwrap();
    assert!(has_deletes);
    assert_eq!(copier.hnsw.neighbors(0, target).size(), degree_before - 1);
    assert!(
        disconnected[0].contains(&target),
        "the cumulative floor check should flag the target"
    );

    let merged =
        HnswGraphBuilder::init_graph(scorer, beam, 42, &graph, &new_ord_map, size as i32).unwrap();
    assert!(
        merged.neighbors(0, target).size() >= degree_before,
        "target below the cumulative floor should be reconnected during merge, has {}",
        merged.neighbors(0, target).size()
    );
}

/// Deletions that fall disproportionately on the upper levels leave the
/// hierarchy thinner than the HNSW model expects, and `rebalanceGraph`
/// promotes nodes back up. Constructed deliberately -- a uniform deletion
/// thins every level in proportion and never triggers it.
#[test]
fn deleting_the_upper_level_nodes_makes_the_merge_rebalance() {
    let n = 600i32;
    let (source, _) = build_line_graph(n as usize, 16, 100);
    assert!(source.num_levels() >= 2, "need an upper level to empty");

    // Drop every node that lives above level 0 (bar one, so the graph keeps
    // an entry node), and nothing else.
    let mut doomed: std::collections::HashSet<i32> = source
        .sorted_nodes_on_level(1)
        .unwrap()
        .into_iter()
        .collect();
    doomed.remove(&source.entry_node());
    let mut ord_map = vec![-1i32; n as usize];
    let mut kept_values = Vec::new();
    let mut next = 0i32;
    for old in 0..n {
        if !doomed.contains(&old) {
            ord_map[old as usize] = next;
            kept_values.push(old as f32);
            next += 1;
        }
    }
    let total = next;
    assert!(total > n / 2, "most nodes must survive: {total} of {n}");

    let merged = merge_graphs(
        Line::new(kept_values),
        100,
        DEFAULT_RAND_SEED,
        &[&source],
        &[ord_map],
        total,
        None,
    )
    .unwrap();
    assert_well_formed(&merged, total);
    // The rebalance must have refilled level 1 towards `size / M`, rather
    // than leaving it with the single survivor.
    let level1 = merged.sorted_nodes_on_level(1).unwrap();
    assert!(
        level1.len() > 1,
        "level 1 kept {} node(s); the rebalance did not run",
        level1.len()
    );
}

/// A graph view whose `getNodesOnLevel` names an ordinal outside its own
/// `size()` is malformed; copying it must be an error, not an index panic.
#[test]
fn a_source_node_outside_the_ordinal_map_is_rejected() {
    struct BadNodeIds;
    impl HnswGraphView for BadNodeIds {
        fn size(&self) -> i32 {
            4
        }
        fn num_levels(&self) -> i32 {
            1
        }
        fn entry_node(&self) -> i32 {
            0
        }
        fn max_conn(&self) -> i32 {
            8
        }
        fn neighbors_into(&self, _l: i32, _n: i32, out: &mut Vec<i32>) -> Result<()> {
            out.clear();
            Ok(())
        }
        fn sorted_nodes_on_level(&self, _l: i32) -> Result<Vec<i32>> {
            // `999` is past `size()`, so past any honest ordinal map.
            Ok(vec![0, 1, 2, 999])
        }
    }
    let err = merge_graphs(
        Line::new(vec![0.0, 1.0, 2.0, 3.0]),
        10,
        DEFAULT_RAND_SEED,
        &[&BadNodeIds],
        &[vec![0, 1, 2, 3]],
        4,
        None,
    )
    .unwrap_err();
    assert!(
        matches!(err, Error::InvalidGraphParameter(_)),
        "expected an invalid-parameter error, got {err:?}"
    );
}

/// `HnswGraphSearcher` sizes its visited bitset from `maxNodeId() + 1`,
/// not from `size()`. They differ exactly while a graph is being built out
/// of order, which is what merging does -- so a searcher that used
/// `size()` would index past the end of the bitset on the first merge.
#[test]
fn a_partly_built_graph_reports_max_node_id_from_its_declared_size() {
    let mut graph = OnHeapHnswGraph::with_size(16, 1000);
    graph.add_node(0, 900);
    assert_eq!(graph.size(), 1);
    assert_eq!(graph.max_node_id(), 999);
    // A graph that grows on demand reports the largest ordinal it has.
    let mut growing = OnHeapHnswGraph::new(16);
    growing.add_node(0, 4);
    assert_eq!(growing.max_node_id(), 4);
}
