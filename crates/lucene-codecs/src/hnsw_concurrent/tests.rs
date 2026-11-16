#![cfg(test)]
use super::*;
use crate::hnsw::{HnswGraphBuilder, VectorScorer};

/// A 1-D line: score is `1 / (1 + |a - b|)`.
#[derive(Clone)]
struct Line {
    points: Vec<f32>,
    ord: i32,
}

impl VectorScorer for Line {
    fn score(&mut self, node: i32) -> Result<f32> {
        let d = (self.points[self.ord as usize] - self.points[node as usize]).abs();
        Ok(1.0 / (1.0 + d))
    }
    fn max_ord(&self) -> i32 {
        self.points.len() as i32
    }
}

impl UpdateableVectorScorer for Line {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
        self.ord = ord;
        Ok(())
    }
}

// ARITH: test fixtures of a few thousand points.
#[allow(clippy::arithmetic_side_effects)]
fn line(n: usize) -> Line {
    Line {
        points: (0..n).map(|i| ((i * 7919) % 1000) as f32).collect(),
        ord: 0,
    }
}

fn edges(g: &OnHeapHnswGraph) -> Vec<Vec<Vec<i32>>> {
    g.graph
        .iter()
        .map(|levels| levels.iter().map(|a| a.nodes().to_vec()).collect())
        .collect()
}

// ARITH: test fixtures of a few thousand points.
#[allow(clippy::arithmetic_side_effects)]
#[test]
fn one_worker_is_the_sequential_builder() {
    let n = 300;
    let seq =
        HnswGraphBuilder::with_graph(line(n), 20, 42, OnHeapHnswGraph::with_size(6, n as i32))
            .unwrap()
            .build(n as i32)
            .unwrap();
    let conc = build_concurrent(
        vec![line(n)],
        6,
        20,
        42,
        OnHeapHnswGraph::with_size(6, n as i32),
        None,
        n as i32,
        7,
    )
    .unwrap();
    assert_eq!(edges(&seq), edges(&conc));
    assert_eq!(seq.entry_node(), conc.entry_node());
    assert_eq!(seq.num_levels(), conc.num_levels());
    assert_eq!(conc.size(), n as i32);
}

/// A growing graph, filled by one inserter past a chunk boundary, is
/// the sequential builder's graph over the same ordinals.
// ARITH: test fixtures of a few thousand points.
#[allow(clippy::arithmetic_side_effects)]
#[test]
fn a_growing_graph_is_the_sequential_builder() {
    let n = CHUNK + 500;
    let seq = HnswGraphBuilder::new(line(n), 6, 20, 42)
        .unwrap()
        .build(n as i32)
        .unwrap();
    let growing = ConcurrentHnswGraph::growable(6);
    let mut inserter = GraphInserter::new(&growing, line(n), 6, 20, 42).unwrap();
    let mut entries = Vec::new();
    for i in 0..n as i32 {
        inserter.insert(i).unwrap();
        assert_eq!(growing.published().len(), i + 1);
        entries.push(growing.entry());
    }
    drop(inserter);
    // Each prefix keeps the entry node it had when it was whole.
    for (i, &(entry, level)) in entries.iter().enumerate() {
        let prefix = growing.published_upto(i as i32 + 1);
        assert_eq!(
            (prefix.entry_node(), prefix.num_levels()),
            (entry, level + 1)
        );
        assert_eq!(prefix.len(), i as i32 + 1);
    }
    assert!(growing.published_upto(0).is_empty());
    assert_eq!(growing.published_upto(0).entry_node(), -1);
    let published = growing.published();
    assert_eq!(published.entry_node(), seq.entry_node());
    assert_eq!(published.num_levels(), seq.num_levels());
    let copied = growing.to_graph(n as i32).unwrap();
    assert_eq!(edges(&seq), edges(&copied));
    assert_eq!(copied.entry_node(), seq.entry_node());
    assert_eq!(copied.max_node_id(), seq.max_node_id());
    assert!(growing.to_graph(n as i32 + 1).is_err());
    let grown = growing.into_graph();
    assert_eq!(edges(&seq), edges(&grown));
    assert_eq!(grown.size(), n as i32);
}

/// Searches of the published graph while the inserter runs see only
/// whole nodes, and find each one's nearest neighbour.
// ARITH: test fixtures of a few thousand points.
#[allow(clippy::arithmetic_side_effects)]
#[test]
fn searches_read_a_graph_while_it_grows() {
    let n = 3000;
    let growing = ConcurrentHnswGraph::growable(8);
    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut inserter = GraphInserter::new(&growing, line(n), 8, 32, 42).unwrap();
            for i in 0..n as i32 {
                inserter.insert(i).unwrap();
            }
            done.store(true, Ordering::Release);
        });
        for _ in 0..2 {
            scope.spawn(|| {
                let mut searches = 0;
                while !done.load(Ordering::Acquire) || searches == 0 {
                    let g = growing.published();
                    if g.is_empty() {
                        continue;
                    }
                    let target = (searches * 31) % g.len() as usize;
                    let mut scorer = line(n);
                    scorer.set_scoring_ordinal(target as i32).unwrap();
                    let mut collector = KnnCollector::new(5, u64::MAX);
                    HnswGraphSearcher::new(16, g.len())
                        .search(&mut collector, &mut scorer, &g, None)
                        .unwrap();
                    let hits = collector.top_docs();
                    assert!(hits.iter().all(|&(d, _)| d < g.len()), "{hits:?}");
                    // The target itself (or a point at the same place).
                    assert_eq!(hits[0].1, 1.0, "target {target}: {hits:?}");
                    searches += 1;
                }
            });
        }
    });
}

#[test]
fn many_workers_build_a_connected_searchable_graph() {
    let n = 2000;
    let conc = build_concurrent(
        (0..4).map(|_| line(n)).collect(),
        8,
        32,
        42,
        OnHeapHnswGraph::with_size(8, n as i32),
        None,
        n as i32,
        64,
    )
    .unwrap();
    assert_eq!(conc.size(), n as i32);
    // Every node has neighbours on level 0, and the level-0 graph is
    // connected from the entry node.
    let mut seen = vec![false; n];
    let mut stack = vec![conc.entry_node()];
    while let Some(x) = stack.pop() {
        if std::mem::replace(&mut seen[x as usize], true) {
            continue;
        }
        stack.extend_from_slice(conc.neighbors(0, x).nodes());
    }
    assert!(seen.iter().filter(|s| **s).count() as f64 > 0.99 * n as f64);
    for levels in &conc.graph {
        assert!(!levels[0].nodes().is_empty());
    }
}

#[test]
fn initialized_nodes_are_skipped_and_errors_reported() {
    let n = 50;
    let mut init = FixedBitSet::new(n);
    init.set(3);
    let g = build_concurrent(
        vec![line(n), line(n)],
        4,
        10,
        1,
        OnHeapHnswGraph::with_size(4, n as i32),
        Some(&init),
        n as i32,
        DEFAULT_BATCH_SIZE,
    )
    .unwrap();
    assert!(g.graph[3].is_empty());
    assert_eq!(g.size(), n as i32 - 1);
    let bad = |scorers: Vec<Line>, m, batch, graph, init: Option<&FixedBitSet>, max| {
        build_concurrent(scorers, m, 10, 1, graph, init, max, batch).is_err()
    };
    assert!(bad(vec![], 4, 8, OnHeapHnswGraph::with_size(4, 5), None, 5));
    assert!(bad(
        vec![line(5)],
        0,
        8,
        OnHeapHnswGraph::with_size(4, 5),
        None,
        5
    ));
    assert!(bad(
        vec![line(5)],
        4,
        0,
        OnHeapHnswGraph::with_size(4, 5),
        None,
        5
    ));
    assert!(bad(
        vec![line(5)],
        4,
        8,
        OnHeapHnswGraph::with_size(4, 5),
        None,
        6
    ));
    assert!(bad(vec![line(5)], 4, 8, OnHeapHnswGraph::new(4), None, 0));
    let short = FixedBitSet::new(2);
    assert!(bad(
        vec![line(5)],
        4,
        8,
        OnHeapHnswGraph::with_size(4, 5),
        Some(&short),
        5
    ));
    let shared = ConcurrentHnswGraph::from_graph(OnHeapHnswGraph::with_size(4, 3)).unwrap();
    assert!(shared.node(9).is_err());
    let mut out = vec![1];
    shared.add_node(0, 1).unwrap();
    shared.neighbors_into(5, 1, &mut out).unwrap();
    assert!(out.is_empty());
    assert_eq!(shared.sorted_nodes_on_level(0).unwrap(), vec![1]);
    assert!(shared.try_set_new_entry_node(1, 0));
    assert!(!shared.try_set_new_entry_node(2, 0));
    assert!(!shared.try_promote_new_entry_node(2, 3, 1));
    assert_eq!(shared.max_node_id(), 2);
    assert_eq!(shared.max_conn(), 4);
}

/// A worker whose scorer panics fails the build with an error rather
/// than taking the merging thread down; the shared view reports the
/// graph's size as the graph it wraps.
#[test]
fn a_panicking_worker_is_an_error_not_a_crash() {
    #[derive(Clone)]
    struct Panics(Line);
    impl VectorScorer for Panics {
        fn score(&mut self, node: i32) -> Result<f32> {
            assert!(node < 50, "scorer failure at node {node}");
            self.0.score(node)
        }
        fn max_ord(&self) -> i32 {
            self.0.max_ord()
        }
    }
    impl UpdateableVectorScorer for Panics {
        fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
            self.0.set_scoring_ordinal(ord)
        }
    }
    let n = 200;
    assert_eq!(Panics(line(n)).max_ord(), n as i32, "one ordinal a point");
    let err = build_concurrent(
        vec![Panics(line(n)), Panics(line(n))],
        6,
        20,
        42,
        OnHeapHnswGraph::with_size(6, n as i32),
        None,
        n as i32,
        7,
    )
    .unwrap_err();
    assert!(err.to_string().contains("merge worker panicked"), "{err}");

    let built = build_concurrent(
        vec![line(30)],
        6,
        20,
        42,
        OnHeapHnswGraph::with_size(6, 30),
        None,
        30,
        7,
    )
    .unwrap();
    let shared = ConcurrentHnswGraph::from_graph(built).unwrap();
    assert_eq!(HnswGraphView::size(&shared), 30);
}
