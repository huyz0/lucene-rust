//! Port of `org.apache.lucene.util.hnsw.HnswConcurrentMergeBuilder` (and the
//! thread-safe half of `OnHeapHnswGraph`/`HnswGraphBuilder` it relies on): a
//! merge-time HNSW build spread over several worker threads.
//!
//! Java's design is kept: every worker is an `HnswGraphBuilder` with its own
//! scorer copy, its own `SplittableRandom` seeded with the same
//! `HnswGraphBuilder.randSeed`, and its own searcher; the workers claim
//! ordinals in batches of 2048 from a shared counter
//! (`workProgress.getAndAdd`) and insert them into one shared graph. The
//! graph's neighbour arrays are guarded by read/write locks (Java stripes 512
//! `ReentrantReadWriteLock`s by `(level, node)`; here each node has its own
//! lock), a search copies a neighbour list out under the read lock
//! (`MergeSearcher.graphSeek`), and linking a new node writes the neighbour's
//! array under the write lock. The entry node is a compare-and-set pair
//! (`AtomicReference<EntryNode>`), and an insertion whose level exceeds the
//! graph's re-runs the search for the levels another worker added in the
//! meantime before trying to promote itself again -- the `do { } while`
//! loop `HnswGraphBuilder.addGraphNodeInternal` carries for exactly this.
//!
//! Like Java's, the result depends on thread scheduling; with one worker it
//! is exactly `HnswGraphBuilder`'s graph.

use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};
use std::sync::{Mutex, RwLock};

use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::splittable_random::SplittableRandom;

use crate::hnsw::{
    pop_to_scratch, HnswGraphSearcher, HnswGraphView, KnnCollector, NeighborArray, OnHeapHnswGraph,
    UpdateableVectorScorer, MAXIMUM_BEAM_WIDTH, MAXIMUM_MAX_CONN,
};
use crate::vectors::{Error, Result};

/// `HnswConcurrentMergeBuilder.DEFAULT_BATCH_SIZE`: ordinals a worker inserts
/// per claim on the shared counter.
pub const DEFAULT_BATCH_SIZE: i32 = 2048;

/// An `OnHeapHnswGraph` of fixed ordinal space that several threads insert
/// into at once.
#[derive(Debug)]
pub struct ConcurrentHnswGraph {
    nsize: usize,
    nsize0: usize,
    /// Per node, its per-level neighbour arrays (empty: not added yet).
    nodes: Vec<RwLock<Vec<NeighborArray>>>,
    /// `(entryNode, level)`: Java's `AtomicReference<EntryNode>`.
    entry: Mutex<(i32, i32)>,
    size: AtomicI32,
}

impl ConcurrentHnswGraph {
    /// Takes over a merge graph (`OnHeapHnswGraph::with_size`), with whatever
    /// an initializer already copied into it.
    pub fn from_graph(graph: OnHeapHnswGraph) -> Result<Self> {
        if graph.fixed_size.is_none() {
            return Err(Error::InvalidGraphParameter(
                "concurrent insertion needs a graph of fixed size".to_string(),
            ));
        }
        Ok(ConcurrentHnswGraph {
            nsize: graph.nsize,
            nsize0: graph.nsize0,
            nodes: graph.graph.into_iter().map(RwLock::new).collect(),
            entry: Mutex::new((graph.entry_node, graph.entry_level)),
            size: AtomicI32::new(graph.size),
        })
    }

    /// The finished graph.
    pub fn into_graph(self) -> OnHeapHnswGraph {
        let graph: Vec<Vec<NeighborArray>> = self
            .nodes
            .into_iter()
            .map(|l| l.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        let max_node_id = graph
            .iter()
            .rposition(|l| !l.is_empty())
            .map_or(-1, |i| i as i32);
        let (entry_node, entry_level) = *self.entry.lock().unwrap_or_else(|e| e.into_inner());
        let fixed = graph.len() as i32;
        OnHeapHnswGraph {
            nsize: self.nsize,
            nsize0: self.nsize0,
            graph,
            entry_node,
            entry_level,
            size: self.size.load(Ordering::SeqCst),
            max_node_id,
            fixed_size: Some(fixed),
        }
    }

    fn node(&self, node: i32) -> Result<&RwLock<Vec<NeighborArray>>> {
        usize::try_from(node)
            .ok()
            .and_then(|n| self.nodes.get(n))
            .ok_or_else(|| {
                Error::InvalidGraphParameter(format!("node {node} is outside the graph"))
            })
    }

    /// `OnHeapHnswGraph.addNode` for every level `0..=level` at once: nodes
    /// are added top-down before any link, so the arrays all start empty.
    fn add_node(&self, level: i32, node: i32) -> Result<()> {
        let lock = self.node(node)?;
        let mut levels = lock.write().unwrap_or_else(|e| e.into_inner());
        if levels.is_empty() {
            self.size.fetch_add(1, Ordering::SeqCst);
        }
        levels.clear();
        for l in 0..=level.max(0) {
            let max = if l == 0 { self.nsize0 } else { self.nsize };
            levels.push(NeighborArray::new(max, true));
        }
        Ok(())
    }

    /// `trySetNewEntryNode`: only when the graph has none.
    fn try_set_new_entry_node(&self, node: i32, level: i32) -> bool {
        let mut e = self.entry.lock().unwrap_or_else(|e| e.into_inner());
        if e.0 == -1 {
            *e = (node, level);
            return true;
        }
        false
    }

    /// `tryPromoteNewEntryNode`: only if the level is still `expect_old_level`.
    fn try_promote_new_entry_node(&self, node: i32, level: i32, expect_old_level: i32) -> bool {
        let mut e = self.entry.lock().unwrap_or_else(|e| e.into_inner());
        if e.1 == expect_old_level {
            *e = (node, level);
            return true;
        }
        false
    }
}

impl HnswGraphView for ConcurrentHnswGraph {
    fn size(&self) -> i32 {
        self.size.load(Ordering::SeqCst)
    }

    // ARITH: the entry level is a random graph level, at most ~1100.
    #[allow(clippy::arithmetic_side_effects)]
    fn num_levels(&self) -> i32 {
        self.entry.lock().unwrap_or_else(|e| e.into_inner()).1 + 1
    }

    fn entry_node(&self) -> i32 {
        self.entry.lock().unwrap_or_else(|e| e.into_inner()).0
    }

    // ARITH: `nsize` is `M + 1` for an `M` of at most `MAXIMUM_MAX_CONN`.
    #[allow(clippy::arithmetic_side_effects)]
    fn max_conn(&self) -> i32 {
        self.nsize as i32 - 1
    }

    // ARITH: a vector's length, at most `i32::MAX` ordinals.
    #[allow(clippy::arithmetic_side_effects)]
    fn max_node_id(&self) -> i32 {
        self.nodes.len() as i32 - 1
    }

    /// `MergeSearcher.graphSeek`: copies the list out under the read lock.
    fn neighbors_into(&self, level: i32, node: i32, out: &mut Vec<i32>) -> Result<()> {
        out.clear();
        let levels = self.node(node)?.read().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = usize::try_from(level).ok().and_then(|l| levels.get(l)) {
            out.extend_from_slice(n.nodes());
        }
        Ok(())
    }

    fn sorted_nodes_on_level(&self, level: i32) -> Result<Vec<i32>> {
        let level = usize::try_from(level).unwrap_or(usize::MAX);
        Ok(self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, l)| l.read().unwrap_or_else(|e| e.into_inner()).len() > level)
            .map(|(n, _)| n as i32)
            .collect())
    }
}

/// One `ConcurrentMergeWorker`: an `HnswGraphBuilder` over the shared graph.
struct Worker<'g, S: UpdateableVectorScorer> {
    m: i32,
    ml: f64,
    random: SplittableRandom,
    scorer: S,
    searcher: HnswGraphSearcher,
    graph: &'g ConcurrentHnswGraph,
    entry_candidates: KnnCollector,
    beam_candidates: KnnCollector,
}

impl<S: UpdateableVectorScorer> Worker<'_, S> {
    /// `ConcurrentMergeWorker.run`: claim a batch, insert it, repeat.
    // ARITH: ordinals below `max_ord`, a positive `i32`, claimed through an
    // `i64` counter that cannot wrap.
    #[allow(clippy::arithmetic_side_effects)]
    fn run(
        &mut self,
        max_ord: i32,
        progress: &AtomicI64,
        batch_size: i32,
        initialized: Option<&FixedBitSet>,
    ) -> Result<()> {
        loop {
            let start = progress.fetch_add(i64::from(batch_size), Ordering::SeqCst);
            if start >= i64::from(max_ord) {
                return Ok(());
            }
            let end = (start + i64::from(batch_size)).min(i64::from(max_ord));
            for node in start as i32..end as i32 {
                // `ConcurrentMergeWorker.addGraphNode`: an initialized node is
                // already in the graph.
                // FBS: `node < max_ord`, and `build` checked
                // `initialized.len() >= max_ord`.
                if initialized.is_some_and(|b| b.get(node as usize)) {
                    continue;
                }
                self.add_graph_node(node)?;
            }
        }
    }

    /// `HnswGraphBuilder.getRandomGraphLevel`.
    fn random_graph_level(&mut self) -> i32 {
        let mut u;
        loop {
            u = self.random.next_double();
            if u != 0.0 {
                break;
            }
        }
        (-u.ln() * self.ml) as i32
    }

    /// `HnswGraphBuilder.addGraphNodeInternal` with its concurrent retry loop.
    // ARITH: levels are random graph levels (at most ~1100) and `m` is at most
    // `MAXIMUM_MAX_CONN`.
    #[allow(clippy::arithmetic_side_effects)]
    fn add_graph_node(&mut self, node: i32) -> Result<()> {
        self.scorer.set_scoring_ordinal(node)?;
        let node_level = self.random_graph_level();
        self.graph.add_node(node_level, node)?;
        if self.graph.try_set_new_entry_node(node, node_level) {
            return Ok(());
        }
        let mut lowest_unset_level = 0;
        loop {
            let cur_max_level = self.graph.num_levels() - 1;
            let mut eps = vec![self.graph.entry_node()];
            for level in ((node_level + 1)..=cur_max_level).rev() {
                self.entry_candidates.clear();
                self.searcher.search_level(
                    &mut self.entry_candidates,
                    &mut self.scorer,
                    level,
                    &eps,
                    self.graph,
                    None,
                )?;
                eps[0] = self.entry_candidates.pop_node();
            }
            let top = node_level.min(cur_max_level);
            let mut scratch_per_level: Vec<NeighborArray> = Vec::new();
            for level in (lowest_unset_level..=top).rev() {
                self.beam_candidates.clear();
                self.searcher.search_level(
                    &mut self.beam_candidates,
                    &mut self.scorer,
                    level,
                    &eps,
                    self.graph,
                    None,
                )?;
                eps = self.beam_candidates.pop_until_nearest_k_nodes();
                let mut scratch =
                    NeighborArray::new(self.beam_candidates.k().max(self.m as usize + 1), false);
                pop_to_scratch(&mut self.beam_candidates, &mut scratch);
                scratch_per_level.push(scratch);
            }
            // Searched top-down, linked bottom-up.
            scratch_per_level.reverse();
            for (i, scratch) in scratch_per_level.iter().enumerate() {
                self.add_diverse_neighbors(lowest_unset_level + i as i32, node, scratch)?;
            }
            lowest_unset_level += scratch_per_level.len() as i32;
            if lowest_unset_level == node_level + 1 {
                return Ok(());
            }
            if self
                .graph
                .try_promote_new_entry_node(node, node_level, cur_max_level)
            {
                return Ok(());
            }
            if self.graph.num_levels() == cur_max_level + 1 {
                return Err(Error::InvalidGraphParameter(format!(
                    "not able to promote node {node} at level {node_level} as entry node, \
                     but the max graph level {cur_max_level} has not changed"
                )));
            }
        }
    }

    /// `HnswGraphBuilder.addDiverseNeighbors` + `selectAndLinkDiverse` +
    /// `updateNeighbor` (never link repair on this path).
    // ARITH: `m` is at most `MAXIMUM_MAX_CONN`; `i` walks a candidate array
    // down to -1.
    #[allow(clippy::arithmetic_side_effects)]
    fn add_diverse_neighbors(
        &mut self,
        level: i32,
        node: i32,
        candidates: &NeighborArray,
    ) -> Result<()> {
        let max_conn_on_level = if level == 0 { self.m * 2 } else { self.m } as usize;
        let mut mask = vec![false; candidates.size()];
        {
            // Java writes the new node's own array unlocked: nothing links to
            // it yet on this level. Taking the lock costs nothing and keeps the
            // other levels' incoming links (which do take it) safe.
            let mut levels = self
                .graph
                .node(node)?
                .write()
                .unwrap_or_else(|e| e.into_inner());
            let neighbors = &mut levels[level as usize];
            let mut i = candidates.size() as i64 - 1;
            while neighbors.size() < max_conn_on_level && i >= 0 {
                let c_node = candidates.nodes()[i as usize];
                if c_node != node {
                    let c_score = candidates.score(i as usize);
                    self.scorer.set_scoring_ordinal(c_node)?;
                    let mut diverse = true;
                    for j in 0..neighbors.size() {
                        if self.scorer.score(neighbors.nodes()[j])? >= c_score {
                            diverse = false;
                            break;
                        }
                    }
                    if diverse {
                        mask[i as usize] = true;
                        neighbors.add_in_order(c_node, c_score);
                    }
                }
                i -= 1;
            }
        }
        for (i, selected) in mask.iter().enumerate() {
            if !*selected {
                continue;
            }
            let nbr = candidates.nodes()[i];
            let score = candidates.score(i);
            let mut levels = self
                .graph
                .node(nbr)?
                .write()
                .unwrap_or_else(|e| e.into_inner());
            let Some(arr) = levels.get_mut(level as usize) else {
                return Err(Error::InvalidGraphParameter(format!(
                    "neighbour {nbr} is not on level {level}"
                )));
            };
            arr.add_and_ensure_diversity(node, score, nbr, &mut self.scorer)?;
        }
        Ok(())
    }
}

/// Port of `HnswConcurrentMergeBuilder`: inserts every ordinal of
/// `0..max_ord` not in `initialized` into `graph`, one worker thread per
/// scorer in `scorers` (Java's `scorerSupplier.copy()` per worker).
///
/// `m` is the configured `M` (Java passes the merger's, which can differ from
/// an initializer graph's own `maxConn`). `batch_size` is
/// [`DEFAULT_BATCH_SIZE`] outside tests.
#[allow(clippy::too_many_arguments)]
pub fn build_concurrent<S: UpdateableVectorScorer + Send>(
    scorers: Vec<S>,
    m: i32,
    beam_width: i32,
    seed: u64,
    graph: OnHeapHnswGraph,
    initialized: Option<&FixedBitSet>,
    max_ord: i32,
    batch_size: i32,
) -> Result<OnHeapHnswGraph> {
    if scorers.is_empty() {
        return Err(Error::InvalidGraphParameter("no merge workers".to_string()));
    }
    if !(1..=MAXIMUM_MAX_CONN).contains(&m) || !(1..=MAXIMUM_BEAM_WIDTH).contains(&beam_width) {
        return Err(Error::InvalidGraphParameter(format!(
            "M {m} / beamWidth {beam_width} out of range"
        )));
    }
    if batch_size < 1 {
        return Err(Error::InvalidGraphParameter(format!(
            "batch size {batch_size}"
        )));
    }
    let slots = graph.graph.len();
    if usize::try_from(max_ord).map_or(true, |n| n > slots) {
        return Err(Error::InvalidGraphParameter(format!(
            "{max_ord} ordinals for a graph of {slots} slots"
        )));
    }
    if let Some(bits) = initialized {
        if (bits.len() as i64) < i64::from(max_ord) {
            return Err(Error::InvalidGraphParameter(format!(
                "the initialized-node set covers {} of {max_ord} ordinals",
                bits.len()
            )));
        }
    }
    let shared = ConcurrentHnswGraph::from_graph(graph)?;
    let progress = AtomicI64::new(0);
    let ml = if m == 1 { 1.0 } else { 1.0 / f64::from(m).ln() };
    let mut workers: Vec<Worker<'_, S>> = scorers
        .into_iter()
        .map(|scorer| Worker {
            m,
            ml,
            random: SplittableRandom::new(seed),
            scorer,
            searcher: HnswGraphSearcher::new(beam_width as usize, 1),
            graph: &shared,
            entry_candidates: KnnCollector::unlimited(1),
            beam_candidates: KnnCollector::unlimited(beam_width as usize),
        })
        .collect();
    let results: Vec<Result<()>> = std::thread::scope(|scope| {
        let handles: Vec<_> = workers
            .iter_mut()
            .map(|w| {
                let progress = &progress;
                scope.spawn(move || w.run(max_ord, progress, batch_size, initialized))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(Error::InvalidGraphParameter(
                        "a merge worker panicked".to_string(),
                    ))
                })
            })
            .collect()
    });
    drop(workers);
    for r in results {
        r?;
    }
    Ok(shared.into_graph())
}

#[cfg(test)]
mod tests {
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
}
