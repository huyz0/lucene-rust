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

use std::ops::Deref;
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
    nodes: Nodes,
    /// `(entryNode, level)`: Java's `AtomicReference<EntryNode>`.
    entry: Mutex<(i32, i32)>,
    size: AtomicI32,
    /// What [`GraphInserter`] has finished, read by [`Self::published`].
    published: Mutex<Published>,
}

/// The ordinals [`GraphInserter`] has finished, and the entry node each
/// prefix of them had.
#[derive(Debug, Default)]
struct Published {
    limit: i32,
    /// `(from, entry node, its level)`, `from` ascending: the entry of every
    /// prefix of at least `from` ordinals, until the next record.
    entries: Vec<(i32, i32, i32)>,
}

/// Nodes per chunk of [`Nodes`].
const CHUNK: usize = 4096;

/// The most nodes a growing graph holds (`2^24`): its chunk table is
/// allocated up front, 4096 slots.
const GROWABLE_NODES: usize = 1 << 24;

type Node = RwLock<Vec<NeighborArray>>;

/// The nodes' slots, in chunks of [`CHUNK`] allocated on first use and
/// never moved, so a slot stays put while the graph grows and readers hold
/// no lock on the table itself.
#[derive(Debug)]
struct Nodes {
    chunks: Box<[std::sync::OnceLock<Box<[Node]>>]>,
    /// Slots in use: one past the highest node added, or a fixed graph's size.
    len: std::sync::atomic::AtomicUsize,
}

impl Nodes {
    fn with_chunks(max: usize) -> Self {
        Nodes {
            chunks: (0..max.div_ceil(CHUNK))
                .map(|_| std::sync::OnceLock::new())
                .collect(),
            len: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn new_chunk() -> Box<[Node]> {
        (0..CHUNK).map(|_| RwLock::new(Vec::new())).collect()
    }

    // ARITH: chunk indexes and offsets of a table of at most
    // `GROWABLE_NODES` (or a fixed graph's `i32` size) slots.
    #[allow(clippy::arithmetic_side_effects)]
    fn from_levels(levels: Vec<Vec<NeighborArray>>) -> Result<Self> {
        let n = levels.len();
        let nodes = Nodes::with_chunks(n);
        let mut it = levels.into_iter();
        for c in nodes.chunks.iter() {
            let chunk: Box<[Node]> = (0..CHUNK)
                .map(|_| RwLock::new(it.next().unwrap_or_default()))
                .collect();
            let _ = c.set(chunk);
        }
        nodes.len.store(n, Ordering::SeqCst);
        Ok(nodes)
    }

    // ARITH: as `from_levels`.
    #[allow(clippy::arithmetic_side_effects)]
    fn into_levels(self) -> Vec<Vec<NeighborArray>> {
        let n = self.len.load(Ordering::SeqCst);
        let mut out: Vec<Vec<NeighborArray>> = self
            .chunks
            .into_vec()
            .into_iter()
            .filter_map(std::sync::OnceLock::into_inner)
            .flat_map(|c| c.into_vec())
            .map(|l| l.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        out.truncate(n);
        out
    }

    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    // ARITH: `n / CHUNK` and `n % CHUNK` of a slot index.
    #[allow(clippy::arithmetic_side_effects)]
    fn get(&self, n: usize) -> Option<&Node> {
        if n >= self.len() {
            return None;
        }
        self.chunks.get(n / CHUNK)?.get()?.get(n % CHUNK)
    }

    /// Allocates the chunk holding slot `n` (one thread grows a graph).
    // ARITH: as `get`; `n` is below `GROWABLE_NODES`.
    #[allow(clippy::arithmetic_side_effects)]
    fn ensure(&self, n: usize) -> Result<()> {
        let Some(slot) = self.chunks.get(n / CHUNK) else {
            return Err(Error::InvalidGraphParameter(format!(
                "node {n} is beyond the graph's {} slots",
                self.chunks.len() * CHUNK
            )));
        };
        if slot.get().is_none() {
            let _ = slot.set(Self::new_chunk());
        }
        // The chunk is set before the length shows its slot.
        self.len.fetch_max(n + 1, Ordering::AcqRel);
        Ok(())
    }
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
            nodes: Nodes::from_levels(graph.graph)?,
            entry: Mutex::new((graph.entry_node, graph.entry_level)),
            size: AtomicI32::new(graph.size),
            published: Mutex::new(Published::default()),
        })
    }

    /// The finished graph.
    pub fn into_graph(self) -> OnHeapHnswGraph {
        let graph: Vec<Vec<NeighborArray>> = self.nodes.into_levels();
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
        // One allocation per node, not a growth from four levels.
        levels.reserve_exact(usize::try_from(level.max(0)).unwrap_or(0).saturating_add(1));
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
        Ok((0..self.nodes.len())
            .filter(|&n| {
                self.nodes
                    .get(n)
                    .is_some_and(|l| l.read().unwrap_or_else(|e| e.into_inner()).len() > level)
            })
            .map(|n| n as i32)
            .collect())
    }
}

/// One `ConcurrentMergeWorker`: an `HnswGraphBuilder` over the shared graph.
struct Worker<G: Deref<Target = ConcurrentHnswGraph>, S: UpdateableVectorScorer> {
    m: i32,
    ml: f64,
    random: SplittableRandom,
    scorer: S,
    searcher: HnswGraphSearcher,
    graph: G,
    entry_candidates: KnnCollector,
    beam_candidates: KnnCollector,
}

impl<G: Deref<Target = ConcurrentHnswGraph>, S: UpdateableVectorScorer> Worker<G, S> {
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
                    &*self.graph,
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
                    &*self.graph,
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
    let mut workers: Vec<Worker<&ConcurrentHnswGraph, S>> = scorers
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

/// A graph growing while searches read it: one inserter adds ordinals in
/// order to a [`ConcurrentHnswGraph`] of fixed capacity, and any number of
/// threads search the graph meanwhile ([`HnswGraphView`] copies each
/// neighbour list out under its node's read lock). With one inserter the
/// graph is exactly [`crate::hnsw::HnswGraphBuilder`]'s over the same
/// ordinals, seed and parameters.
pub struct GraphInserter<G: Deref<Target = ConcurrentHnswGraph>, S: UpdateableVectorScorer> {
    worker: Worker<G, S>,
}

impl ConcurrentHnswGraph {
    /// An empty graph that grows as [`GraphInserter`] adds ordinals, up to
    /// `2^24` of them.
    pub fn growable(m: i32) -> Self {
        let empty = OnHeapHnswGraph::new(m);
        ConcurrentHnswGraph {
            nsize: empty.nsize,
            nsize0: empty.nsize0,
            nodes: Nodes::with_chunks(GROWABLE_NODES),
            entry: Mutex::new((-1, -1)),
            size: AtomicI32::new(0),
            published: Mutex::new(Published::default()),
        }
    }

    /// The graph as [`GraphInserter`] last finished it: its first `n`
    /// ordinals, whole, with the entry node they had. Searching it never
    /// reaches a node being inserted.
    pub fn published(&self) -> PublishedGraph<'_> {
        self.published_upto(i32::MAX)
    }

    /// [`Self::published`] cut to its first `max` ordinals, with the entry
    /// node those had: what a reader that saw only `max` vectors searches.
    pub fn published_upto(&self, max: i32) -> PublishedGraph<'_> {
        let p = self.published.lock().unwrap_or_else(|e| e.into_inner());
        let limit = p.limit.min(max.max(0));
        let at = p.entries.partition_point(|&(from, _, _)| from <= limit);
        let (entry, level) = at
            .checked_sub(1)
            .and_then(|i| p.entries.get(i))
            .map_or((-1, -1), |&(_, e, l)| (e, l));
        PublishedGraph {
            graph: self,
            limit,
            entry,
            level,
        }
    }

    /// Publishes the first `limit` ordinals as whole.
    fn publish(&self, limit: i32) {
        let (entry, level) = self.entry();
        let mut p = self.published.lock().unwrap_or_else(|e| e.into_inner());
        if p.entries
            .last()
            .is_none_or(|&(_, e, l)| (e, l) != (entry, level))
        {
            p.entries.push((limit, entry, level));
        }
        p.limit = limit;
    }

    /// A copy of the graph's first `n` ordinals as an on-heap graph, the
    /// graph an `HnswGraphBuilder` that added the same ordinals in order
    /// holds: what a flush writes while searches may still read this one.
    ///
    /// # Errors
    /// When fewer than `n` ordinals are published.
    pub fn to_graph(&self, n: i32) -> Result<OnHeapHnswGraph> {
        let published = self.published_upto(n);
        if published.len() != n {
            return Err(Error::InvalidGraphParameter(format!(
                "{n} ordinals of a graph that has published {}",
                published.len()
            )));
        }
        let count = usize::try_from(n).unwrap_or(0);
        let mut graph = Vec::with_capacity(count);
        for node in 0..count {
            let levels = self
                .nodes
                .get(node)
                .map(|l| l.read().unwrap_or_else(|e| e.into_inner()).clone())
                .unwrap_or_default();
            graph.push(levels);
        }
        Ok(OnHeapHnswGraph {
            nsize: self.nsize,
            nsize0: self.nsize0,
            graph,
            entry_node: published.entry,
            entry_level: published.level,
            size: n,
            // ARITH: `n` is at least 0.
            #[allow(clippy::arithmetic_side_effects)]
            max_node_id: n - 1,
            fixed_size: None,
        })
    }

    /// The entry node and its level, read together.
    pub fn entry(&self) -> (i32, i32) {
        *self.entry.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<G: Deref<Target = ConcurrentHnswGraph>, S: UpdateableVectorScorer> GraphInserter<G, S> {
    /// The inserter of `graph`, scoring with `scorer`.
    pub fn new(graph: G, scorer: S, m: i32, beam_width: i32, seed: u64) -> Result<Self> {
        if !(1..=MAXIMUM_MAX_CONN).contains(&m) || !(1..=MAXIMUM_BEAM_WIDTH).contains(&beam_width) {
            return Err(Error::InvalidGraphParameter(format!(
                "M {m} / beamWidth {beam_width} out of range"
            )));
        }
        let ml = if m == 1 { 1.0 } else { 1.0 / f64::from(m).ln() };
        Ok(GraphInserter {
            worker: Worker {
                m,
                ml,
                random: SplittableRandom::new(seed),
                scorer,
                searcher: HnswGraphSearcher::new(beam_width as usize, 1),
                graph,
                entry_candidates: KnnCollector::unlimited(1),
                beam_candidates: KnnCollector::unlimited(beam_width as usize),
            },
        })
    }

    /// Adds `node`, the next ordinal, then publishes the graph through it
    /// ([`ConcurrentHnswGraph::published`]).
    pub fn insert(&mut self, node: i32) -> Result<()> {
        let next = node.checked_add(1).ok_or_else(|| {
            Error::InvalidGraphParameter(format!("node {node} is the last ordinal"))
        })?;
        self.worker
            .graph
            .nodes
            .ensure(usize::try_from(node).unwrap_or(usize::MAX))?;
        self.worker.add_graph_node(node)?;
        self.worker.graph.publish(next);
        Ok(())
    }

    /// The graph it inserts into.
    pub fn graph(&self) -> &ConcurrentHnswGraph {
        &self.worker.graph
    }

    /// The scorer it inserts with.
    pub fn scorer_mut(&mut self) -> &mut S {
        &mut self.worker.scorer
    }
}

/// [`ConcurrentHnswGraph::published`]: the graph's first `limit` ordinals.
/// Links to later nodes are left out, so a search sees only nodes that
/// were whole when it started.
#[derive(Debug, Clone, Copy)]
pub struct PublishedGraph<'g> {
    graph: &'g ConcurrentHnswGraph,
    limit: i32,
    entry: i32,
    level: i32,
}

impl PublishedGraph<'_> {
    /// How many ordinals it holds.
    pub fn len(&self) -> i32 {
        self.limit
    }

    /// Whether it holds none.
    pub fn is_empty(&self) -> bool {
        self.limit == 0
    }
}

impl HnswGraphView for PublishedGraph<'_> {
    fn size(&self) -> i32 {
        self.limit
    }

    // ARITH: a random graph level, at most ~1100.
    #[allow(clippy::arithmetic_side_effects)]
    fn num_levels(&self) -> i32 {
        self.level + 1
    }

    fn entry_node(&self) -> i32 {
        self.entry
    }

    fn max_conn(&self) -> i32 {
        self.graph.max_conn()
    }

    // ARITH: `limit` is at least 0.
    #[allow(clippy::arithmetic_side_effects)]
    fn max_node_id(&self) -> i32 {
        self.limit - 1
    }

    fn neighbors_into(&self, level: i32, node: i32, out: &mut Vec<i32>) -> Result<()> {
        self.graph.neighbors_into(level, node, out)?;
        out.retain(|&n| n < self.limit);
        Ok(())
    }

    fn sorted_nodes_on_level(&self, level: i32) -> Result<Vec<i32>> {
        let mut nodes = self.graph.sorted_nodes_on_level(level)?;
        nodes.retain(|&n| n < self.limit);
        Ok(nodes)
    }
}

#[cfg(test)]
mod tests;
