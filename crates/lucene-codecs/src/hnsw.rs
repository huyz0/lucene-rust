//! HNSW: the hierarchical navigable small-world graph over a flat vector
//! store, and the beam search that walks it.
//!
//! Port of `org.apache.lucene.util.hnsw.{HnswGraph,OnHeapHnswGraph,
//! HnswGraphBuilder,HnswGraphSearcher,AbstractHnswGraphSearcher,NeighborQueue,
//! NeighborArray,RandomVectorScorer,UpdateableRandomVectorScorer,
//! IncrementalHnswGraphMerger's builders: MergingHnswGraphBuilder,
//! InitializedHnswGraphBuilder,UpdateGraphsUtils}`, plus the pieces of
//! `org.apache.lucene.search.{AbstractKnnCollector,TopKnnCollector}` they
//! depend on.
//!
//! `org.apache.lucene.util.{TernaryLongHeap,NumericUtils}` and
//! `java.util.SplittableRandom` are **not** here: none is HNSW-specific in
//! Java, and `util/hnsw` has no dependency on `codecs.lucene99` at all. They
//! live in [`lucene_util::ternary_long_heap`],
//! [`lucene_util::numeric_utils`] and [`lucene_util::splittable_random`], and
//! are imported downward.
//!
//! The *serialized* graph (`.vex`/`.vem`) lives in [`crate::hnsw_vectors`];
//! the vectors it indexes live in [`crate::vectors`].
//!
//! # What determines recall
//!
//! Four things, and all four are ports rather than reinventions, because a
//! subtly different one produces a graph that is fast and silently worse:
//!
//! 1. **`M` / `beamWidth`** (16 / 100 by default). `M` is the neighbour
//!    budget per node on levels above 0; level 0 gets `2 * M`.
//! 2. **Level assignment**: `(int)(-ln(U) * ml)` with `ml = 1 / ln(M)` and `U`
//!    drawn from `java.util.SplittableRandom(42)`.
//!    [`lucene_util::SplittableRandom`] here
//!    is a bit-exact port of that generator, so a Rust-built graph assigns the
//!    *same* levels to the same ordinals as Java's does.
//! 3. **The diversity rule** ([`HnswGraphBuilder::diversity_check`] and
//!    [`NeighborArray::find_worst_non_diverse`]): a candidate is kept only if
//!    it is closer to the new node than to every already-selected neighbour.
//!    Dropping this keeps the *nearest* neighbours instead of a spread, which
//!    leaves the graph locally over-connected and globally disconnected.
//! 4. **Entry-point descent** ([`HnswGraphSearcher::find_best_entry_point`]):
//!    greedy hill-climbing from the top level down to level 1, then one beam
//!    search on level 0.
//!
//! # Divergences from Java, all deliberate
//!
//! - **Single-threaded.** Java's `OnHeapHnswGraph` carries `AtomicReference`
//!   entry-node state and a `HnswLock` so several `HnswGraphBuilder`s can fill
//!   one fixed-size graph during a concurrent merge. This port builds on one
//!   thread, so the entry node is a plain field and `addGraphNode`'s
//!   `do { ... } while(true)` retry loop -- which exists only to cope with
//!   another thread having moved the entry node underneath it -- collapses to
//!   its single-iteration body. Every branch that a concurrent writer could
//!   have taken is unreachable here; see [`HnswGraphBuilder::add_graph_node`].
//! - **No `connectComponents`.** Java's `finish()` has it commented out
//!   upstream (apache/lucene#14214: "exceptionally expensive"), so the graph a
//!   current Lucene writes does not have it either.
//! - **No `FilteredHnswGraphSearcher`.** It is the strategy variant selected
//!   by `KnnSearchStrategy.Hnsw.useFilteredSearch`, and 10.5.0's
//!   `DEFAULT_FILTERED_SEARCH_THRESHOLD` is **`0`**, so
//!   `ratioPassingFilter * 100 < 0` is false for every ratio and no query
//!   reachable from `KnnFloatVectorQuery` ever selects it. (`main` raised the
//!   threshold to 60; that is a post-10.5.0 change -- see
//!   `docs/sweep/m2/c18-version-audit.md`.)
//! - **`SeededHnswGraphSearcher` is a method, not a class.**
//!   [`HnswGraphSearcher::search_seeded`] is Java's whole seeded searcher:
//!   the class exists there only to override `findBestEntryPoint` with a
//!   constant, and its `searchLevel` delegates verbatim to the plain
//!   searcher's. It is reached from `KnnSearchStrategy.Seeded`, which
//!   `AbstractKnnVectorQuery`'s optimistic re-entry pass builds -- see
//!   `lucene_search::vector_query`.
//! - **`FixedBitSet` always**, never `SparseFixedBitSet`. `createBitSet`'s
//!   choice is a memory trade-off, not a semantic one.

use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::numeric_utils::{float_to_sortable_int, sortable_int_to_float};
use lucene_util::splittable_random::SplittableRandom;
use lucene_util::ternary_long_heap::TernaryLongHeap;

use crate::vectors::{Error, Result};

/// `HnswGraphBuilder.DEFAULT_MAX_CONN`.
pub const DEFAULT_MAX_CONN: i32 = 16;
/// `HnswGraphBuilder.DEFAULT_BEAM_WIDTH`.
pub const DEFAULT_BEAM_WIDTH: i32 = 100;
/// `HnswGraphBuilder.DEFAULT_RAND_SEED`.
pub const DEFAULT_RAND_SEED: u64 = 42;
/// `Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN`.
pub const MAXIMUM_MAX_CONN: i32 = 512;
/// `Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH`.
pub const MAXIMUM_BEAM_WIDTH: i32 = 3200;
/// `Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD`.
pub const HNSW_GRAPH_THRESHOLD: i32 = 100;

/// `DocIdSetIterator.NO_MORE_DOCS`, the sentinel `HnswGraph.nextNeighbor`
/// returns.
pub const NO_MORE_DOCS: i32 = i32::MAX;

// ---------------------------------------------------------------------------
// SplittableRandom
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scorers
// ---------------------------------------------------------------------------

/// Port of `org.apache.lucene.util.hnsw.RandomVectorScorer`: scores an
/// abstract query against a vector ordinal.
///
/// `&mut self` rather than Java's `&self` because the concrete scorers in
/// [`crate::vectors`] decode into a reusable scratch buffer instead of
/// allocating a `float[]` per comparison. Java's own scorers are documented
/// "not thread-safe and should be used by a single thread", so nothing is
/// lost.
pub trait VectorScorer {
    fn score(&mut self, node: i32) -> Result<f32>;

    /// `RandomVectorScorer.maxOrd()`.
    fn max_ord(&self) -> i32;

    /// Port of `RandomVectorScorer.bulkScore`: fills `scores` and returns the
    /// maximum, or `-inf` for an empty batch. The default is Java's default --
    /// a plain loop; a scorer with a genuinely vectorised batch kernel may
    /// override it.
    /// Note `f32::max` is **not** `Math.max(float, float)` on a NaN: Rust
    /// returns the other operand, Java propagates the NaN. That difference is
    /// only reachable with a non-finite vector component, which
    /// [`crate::vectors::write_flat_vectors`] refuses to write (Java's
    /// `VectorUtil.checkFinite` refuses the same) -- recorded here because the
    /// consequence would be a silently *dropped* candidate rather than a loud
    /// failure.
    fn bulk_score(&mut self, nodes: &[i32], scores: &mut [f32]) -> Result<f32> {
        let mut max = f32::NEG_INFINITY;
        for (i, &node) in nodes.iter().enumerate() {
            scores[i] = self.score(node)?;
            max = max.max(scores[i]);
        }
        Ok(max)
    }
}

/// Port of `UpdateableRandomVectorScorer`: a scorer whose *query* is itself a
/// stored ordinal, re-targetable in place. This is what graph construction
/// uses -- every comparison during a build is vector-to-vector.
pub trait UpdateableVectorScorer: VectorScorer {
    fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()>;
}

// ---------------------------------------------------------------------------
// TernaryLongHeap / NeighborQueue
// ---------------------------------------------------------------------------

/// Port of `org.apache.lucene.util.hnsw.NeighborQueue`: `(node, score)` pairs
/// packed into a sortable `i64` and kept in a [`TernaryLongHeap`].
///
/// The packing is Java's, byte for byte: the sortable float bits go in the
/// top 32 and the **complemented** node id in the bottom 32, so that among
/// equal scores the *smaller* node id wins. A max-heap is the same min-heap
/// over `-1 - key`.
#[derive(Debug, Clone)]
pub struct NeighborQueue {
    heap: TernaryLongHeap,
    max_heap: bool,
}

impl NeighborQueue {
    /// `new NeighborQueue(initialSize, maxHeap)`.
    pub fn new(initial_size: usize, max_heap: bool) -> Self {
        NeighborQueue {
            heap: TernaryLongHeap::new(initial_size.max(1)),
            max_heap,
        }
    }

    fn apply(&self, v: i64) -> i64 {
        if self.max_heap {
            // Java's `-1 - v`, an `int`... `long` negation that wraps. The one
            // input that wraps is `v == i64::MIN`, which needs a NaN score
            // (the only float whose sortable bits are `i32::MIN`) together
            // with node `-1`; `write_flat_vectors` refuses to write a
            // non-finite component and `-1` is not an ordinal, so it is
            // unreachable -- but wrapping is what Java does with it, and a
            // panic here would be a different answer, not a safer one.
            (-1i64).wrapping_sub(v)
        } else {
            v
        }
    }

    fn encode(&self, node: i32, score: f32) -> i64 {
        self.apply(
            ((float_to_sortable_int(score) as i64) << 32) | (0xFFFF_FFFFi64 & !(node as i64)),
        )
    }

    fn decode_score(&self, heap_value: i64) -> f32 {
        sortable_int_to_float((self.apply(heap_value) >> 32) as i32)
    }

    fn decode_node_id(&self, heap_value: i64) -> i32 {
        !(self.apply(heap_value)) as i32
    }

    pub fn size(&self) -> usize {
        self.heap.size()
    }

    /// `NeighborQueue.add`: unbounded push.
    pub fn add(&mut self, node: i32, score: f32) {
        let v = self.encode(node, score);
        self.heap.push(v);
    }

    /// `NeighborQueue.insertWithOverflow`: bounded by the constructor's size.
    pub fn insert_with_overflow(&mut self, node: i32, score: f32) -> bool {
        let v = self.encode(node, score);
        self.heap.insert_with_overflow(v)
    }

    pub fn pop(&mut self) -> i32 {
        let v = self.heap.pop();
        self.decode_node_id(v)
    }

    pub fn top_node(&self) -> i32 {
        self.decode_node_id(self.heap.top())
    }

    pub fn top_score(&self) -> f32 {
        self.decode_score(self.heap.top())
    }

    /// `NeighborQueue.nodes()`: heap-array order, deliberately *not* sorted.
    pub fn nodes(&self) -> Vec<i32> {
        (1..=self.heap.size())
            .map(|i| self.decode_node_id(self.heap.get(i)))
            .collect()
    }

    pub fn clear(&mut self) {
        self.heap.clear();
    }
}

// ---------------------------------------------------------------------------
// KnnCollector
// ---------------------------------------------------------------------------

/// Port of `AbstractKnnCollector`/`TopKnnCollector` and of
/// `HnswGraphBuilder.GraphBuilderKnnCollector` -- one type, because they
/// differ only in whether `visitLimit` is finite.
#[derive(Debug, Clone)]
pub struct KnnCollector {
    queue: NeighborQueue,
    k: usize,
    visited_count: u64,
    visit_limit: u64,
}

impl KnnCollector {
    /// `new TopKnnCollector(k, visitLimit, null)`.
    pub fn new(k: usize, visit_limit: u64) -> Self {
        KnnCollector {
            queue: NeighborQueue::new(k.max(1), false),
            k,
            visited_count: 0,
            visit_limit,
        }
    }

    /// `new HnswGraphBuilder.GraphBuilderKnnCollector(k)`: no visit limit and
    /// therefore never early-terminated.
    pub fn unlimited(k: usize) -> Self {
        Self::new(k, u64::MAX)
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn size(&self) -> usize {
        self.queue.size()
    }

    pub fn early_terminated(&self) -> bool {
        self.visited_count >= self.visit_limit
    }

    pub fn inc_visited_count(&mut self, count: usize) {
        // Saturating, not checked: `visited_count` is only ever compared
        // against `visit_limit`, so a saturated count reads as "early
        // terminated", which is the safe direction. Reaching `u64::MAX` visits
        // is not physically possible; Java's `long +=` would wrap to a small
        // number and *disable* the visit limit instead.
        self.visited_count = self.visited_count.saturating_add(count as u64);
    }

    pub fn visited_count(&self) -> u64 {
        self.visited_count
    }

    pub fn visit_limit(&self) -> u64 {
        self.visit_limit
    }

    pub fn collect(&mut self, node: i32, similarity: f32) -> bool {
        self.queue.insert_with_overflow(node, similarity)
    }

    pub fn min_competitive_similarity(&self) -> f32 {
        if self.queue.size() >= self.k {
            self.queue.top_score()
        } else {
            f32::NEG_INFINITY
        }
    }

    /// `GraphBuilderKnnCollector.clear()`: also resets the visited count.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.visited_count = 0;
    }

    /// `GraphBuilderKnnCollector.popNode()`.
    pub fn pop_node(&mut self) -> i32 {
        self.queue.pop()
    }

    /// `GraphBuilderKnnCollector.minimumScore()`: the *worst* kept score.
    pub fn minimum_score(&self) -> f32 {
        self.queue.top_score()
    }

    /// `GraphBuilderKnnCollector.popUntilNearestKNodes()`.
    pub fn pop_until_nearest_k_nodes(&mut self) -> Vec<i32> {
        while self.queue.size() > self.k {
            self.queue.pop();
        }
        self.queue.nodes()
    }

    /// `TopKnnCollector.topDocs()`: `(node, score)` descending by score, node
    /// id ascending on ties.
    pub fn top_docs(mut self) -> Vec<(i32, f32)> {
        let mut out = Vec::with_capacity(self.queue.size());
        while self.queue.size() > 0 {
            let score = self.queue.top_score();
            out.push((self.queue.pop(), score));
        }
        out.reverse();
        out
    }
}

// ---------------------------------------------------------------------------
// NeighborArray
// ---------------------------------------------------------------------------

/// Port of `org.apache.lucene.util.hnsw.NeighborArray`: a node's neighbours
/// and their scores, kept sorted by score.
#[derive(Debug, Clone)]
pub struct NeighborArray {
    scores_desc_order: bool,
    max_size: usize,
    nodes: Vec<i32>,
    scores: Vec<f32>,
    /// How many of the leading entries are known to be in sorted order;
    /// `addOutOfOrder` appends past this mark and [`sort`](Self::sort)
    /// catches up.
    sorted_node_size: usize,
}

impl NeighborArray {
    pub fn new(max_size: usize, desc_order: bool) -> Self {
        NeighborArray {
            scores_desc_order: desc_order,
            max_size,
            nodes: Vec::with_capacity(max_size / 8),
            scores: Vec::with_capacity(max_size / 8),
            sorted_node_size: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }

    pub fn nodes(&self) -> &[i32] {
        &self.nodes
    }

    /// `NeighborArray.getScores(i)`.
    pub fn score(&self, i: usize) -> f32 {
        self.scores[i]
    }

    /// `NeighborArray.addInOrder`: the new node must be *worse* than every
    /// node already stored. Panics on a violated ordering or on overflow,
    /// where Java asserts / throws `IllegalStateException`: both are caller
    /// bugs inside the builder, not decodable input.
    pub fn add_in_order(&mut self, new_node: i32, new_score: f32) {
        debug_assert_eq!(
            self.nodes.len(),
            self.sorted_node_size,
            "cannot call add_in_order after add_out_of_order"
        );
        assert!(self.nodes.len() < self.max_size, "No growth is allowed");
        if let Some(&previous) = self.scores.last() {
            debug_assert!(
                (self.scores_desc_order && previous >= new_score)
                    || (!self.scores_desc_order && previous <= new_score),
                "Nodes are added in an incorrect order!"
            );
        }
        self.nodes.push(new_node);
        self.scores.push(new_score);
        // ARITH: the assert above establishes `nodes.len() < max_size` before
        // the push, and `sorted_node_size == nodes.len()` on entry, so this
        // stays at or below `max_size`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.sorted_node_size += 1;
        }
    }

    /// `NeighborArray.addOutOfOrder`.
    pub fn add_out_of_order(&mut self, new_node: i32, new_score: f32) {
        assert!(self.nodes.len() < self.max_size, "No growth is allowed");
        self.nodes.push(new_node);
        self.scores.push(new_score);
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.scores.clear();
        self.sorted_node_size = 0;
    }

    fn remove_index(&mut self, idx: usize) {
        self.nodes.remove(idx);
        self.scores.remove(idx);
        if idx < self.sorted_node_size {
            // ARITH: guarded by `idx < sorted_node_size`, so it is at least 1.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.sorted_node_size -= 1;
            }
        }
        self.sorted_node_size = self.sorted_node_size.min(self.nodes.len());
    }

    /// `NeighborArray.addAndEnsureDiversity`: append, then -- if that took the
    /// array to its maximum -- drop the *worst non-diverse* entry, so the
    /// array stays one below its cap and the node keeps a spread of
    /// neighbours rather than a cluster of near-duplicates.
    pub fn add_and_ensure_diversity<S: UpdateableVectorScorer>(
        &mut self,
        new_node: i32,
        new_score: f32,
        node_id: i32,
        scorer: &mut S,
    ) -> Result<()> {
        self.add_out_of_order(new_node, new_score);
        if self.nodes.len() < self.max_size {
            return Ok(());
        }
        scorer.set_scoring_ordinal(node_id)?;
        let worst = self.find_worst_non_diverse(scorer)?;
        self.remove_index(worst);
        Ok(())
    }

    /// `NeighborArray.sort`: insertion-sorts the unchecked tail into place and
    /// returns the sorted positions those entries landed in, ascending.
    fn sort<S: VectorScorer>(&mut self, scorer: &mut S) -> Result<Option<Vec<usize>>> {
        if self.nodes.len() == self.sorted_node_size {
            return Ok(None);
        }
        // ARITH: `sorted_node_size <= nodes.len()` is a type invariant --
        // every write to it either increments it alongside a push, decrements
        // it alongside a removal, or clamps with `.min(nodes.len())` -- and
        // the equality case returned above.
        #[allow(clippy::arithmetic_side_effects)]
        let mut unchecked = vec![0usize; self.nodes.len() - self.sorted_node_size];
        let mut count = 0usize;
        while self.sorted_node_size != self.nodes.len() {
            let inserted = self.insert_sorted_internal(scorer)?;
            unchecked[count] = inserted;
            for slot in unchecked.iter_mut().take(count) {
                // Everything already recorded that sat at or after the new
                // insertion point has shifted one to the right.
                // ARITH: both are positions in `nodes`, whose length is
                // capped at `max_size`, and `count` counts one iteration per
                // entry of `unchecked`.
                #[allow(clippy::arithmetic_side_effects)]
                if *slot >= inserted {
                    *slot += 1;
                }
            }
            // ARITH: one increment per iteration of a loop that runs exactly
            // `unchecked.len()` times (`unchecked[count]` on the line above would
            // index out of range otherwise), and that length is at most
            // `nodes.len() <= max_size`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                count += 1;
            }
        }
        unchecked.sort_unstable();
        Ok(Some(unchecked))
    }

    /// `NeighborArray.insertSortedInternal`.
    fn insert_sorted_internal<S: VectorScorer>(&mut self, scorer: &mut S) -> Result<usize> {
        debug_assert!(self.sorted_node_size < self.nodes.len());
        let tmp_node = self.nodes[self.sorted_node_size];
        let mut tmp_score = self.scores[self.sorted_node_size];
        if tmp_score.is_nan() {
            tmp_score = scorer.score(tmp_node)?;
        }
        let insertion_point = if self.scores_desc_order {
            self.desc_sort_right_most_insertion_point(tmp_score, self.sorted_node_size)
        } else {
            self.asc_sort_right_most_insertion_point(tmp_score, self.sorted_node_size)
        };
        self.nodes.remove(self.sorted_node_size);
        self.scores.remove(self.sorted_node_size);
        self.nodes.insert(insertion_point, tmp_node);
        self.scores.insert(insertion_point, tmp_score);
        // ARITH: the `debug_assert` above -- and the `while` in `sort`, which
        // is the only caller -- establish `sorted_node_size < nodes.len()`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.sorted_node_size += 1;
        }
        Ok(insertion_point)
    }

    /// `NeighborArray.ascSortFindRightMostInsertionPoint`.
    fn asc_sort_right_most_insertion_point(&self, new_score: f32, bound: usize) -> usize {
        // `Arrays.binarySearch` semantics: on a hit, walk right past every
        // equal score, then one more; on a miss, the insertion point.
        match self.scores[..bound].binary_search_by(|probe| probe.total_cmp(&new_score)) {
            // ARITH: `binary_search_by` returns `Ok(i)` only for a non-empty
            // slice, so `bound >= 1`; and `i < bound`, so `i + 1 <= bound`.
            #[allow(clippy::arithmetic_side_effects)]
            Ok(mut i) => {
                while i < bound - 1 && self.scores[i + 1] == self.scores[i] {
                    i += 1;
                }
                i + 1
            }
            Err(i) => i,
        }
    }

    /// `NeighborArray.descSortFindRightMostInsertionPoint`.
    // ARITH: `bound` is a position in `scores`, whose length is capped at
    // `max_size`; `start` and `end` stay inside `-1..=bound`, so the `i64`
    // arithmetic here is nowhere near overflow. (Java's `(start + end) / 2`
    // is `int` arithmetic that can overflow past `Integer.MAX_VALUE / 2`
    // entries; widening to `i64` removes that case rather than reproducing
    // it, and the two agree on every array a `NeighborArray` can hold.)
    #[allow(clippy::arithmetic_side_effects)]
    fn desc_sort_right_most_insertion_point(&self, new_score: f32, bound: usize) -> usize {
        let mut start = 0i64;
        let mut end = bound as i64 - 1;
        while start <= end {
            let mid = (start + end) / 2;
            if self.scores[mid as usize] < new_score {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        start as usize
    }

    /// `NeighborArray.findWorstNonDiverse`: walk from the most distant
    /// neighbour inward and return the first that is non-diverse, i.e. the
    /// first whose similarity to some *closer* neighbour is at least its
    /// similarity to the owning node.
    fn find_worst_non_diverse<S: UpdateableVectorScorer>(
        &mut self,
        scorer: &mut S,
    ) -> Result<usize> {
        let unchecked = self
            .sort(scorer)?
            .expect("addAndEnsureDiversity always leaves something unchecked");
        // ARITH: `sort` returned `Some`, so `nodes` is non-empty and
        // `unchecked` has one entry per unsorted node; both cursors walk
        // downward from a length and stop at 0 (`i`) or -1 (`cursor`).
        #[allow(clippy::arithmetic_side_effects)]
        let mut cursor = unchecked.len() as i64 - 1;
        // ARITH: `len()` is a `usize` that fits an `i64` on every supported
        // target, so `len - 1` is at worst -1 and cannot underflow.
        #[allow(clippy::arithmetic_side_effects)]
        let mut i = self.nodes.len() as i64 - 1;
        while i > 0 {
            if cursor < 0 {
                break;
            }
            scorer.set_scoring_ordinal(self.nodes[i as usize])?;
            if self.is_worst_non_diverse(i as usize, &unchecked, cursor as usize, scorer)? {
                return Ok(i as usize);
            }
            // ARITH: `cursor >= 0` here -- the loop breaks above when it goes
            // negative -- so the decrement bottoms out at -1.
            #[allow(clippy::arithmetic_side_effects)]
            if i as usize == unchecked[cursor as usize] {
                cursor -= 1;
            }
            // ARITH: the loop condition is `i > 0`, so the decrement bottoms out
            // at 0.
            #[allow(clippy::arithmetic_side_effects)]
            {
                i -= 1;
            }
        }
        // ARITH: `sort` returning `Some` means something was unsorted, which
        // means `nodes` is non-empty.
        #[allow(clippy::arithmetic_side_effects)]
        Ok(self.nodes.len() - 1)
    }

    /// `NeighborArray.isWorstNonDiverse`.
    fn is_worst_non_diverse<S: VectorScorer>(
        &self,
        candidate_index: usize,
        unchecked: &[usize],
        cursor: usize,
        scorer: &mut S,
    ) -> Result<bool> {
        let min_accepted_similarity = self.scores[candidate_index];
        if candidate_index == unchecked[cursor] {
            // The candidate itself has not been diversity-checked, so it must
            // be checked against every closer neighbour.
            for i in (0..candidate_index).rev() {
                if scorer.score(self.nodes[i])? >= min_accepted_similarity {
                    return Ok(true);
                }
            }
        } else {
            // Otherwise it was diverse when it was added; only the newly
            // inserted (unchecked) neighbours can have invalidated that.
            debug_assert!(candidate_index > unchecked[cursor]);
            for i in (0..=cursor).rev() {
                if scorer.score(self.nodes[unchecked[i]])? >= min_accepted_similarity {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

// ---------------------------------------------------------------------------
// HnswGraph
// ---------------------------------------------------------------------------

/// Port of the read side of `org.apache.lucene.util.hnsw.HnswGraph`.
///
/// Java exposes it as a stateful cursor (`seek(level, node)` then repeated
/// `nextNeighbor()`); this port hands the whole neighbour list over in one
/// call instead. Every caller in Lucene immediately drains the cursor into a
/// `bulkNodes` array anyway, so the cursor buys nothing here and costs a
/// `&mut self` that would force the builder to hand out an exclusive borrow of
/// a graph it is simultaneously reading.
pub trait HnswGraphView {
    /// Number of nodes on level 0.
    fn size(&self) -> i32;
    fn num_levels(&self) -> i32;
    /// The node the search starts from, expressed as a level-0 ordinal, or
    /// `-1` if the graph has none.
    fn entry_node(&self) -> i32;
    /// `M`. Level 0 nodes may have up to `2 * max_conn` neighbours.
    fn max_conn(&self) -> i32;
    /// `HnswGraph.maxNodeId()`: the largest ordinal any level may mention,
    /// inclusive. For a finished graph this is `size() - 1`, which is Java's
    /// default implementation and the only case the flush path ever sees.
    ///
    /// It is a **separate** quantity from `size()` while a graph is being
    /// built out of order -- which is exactly what merging does: a source
    /// segment's nodes are copied into their new ordinals before the rest of
    /// the merged ordinal space exists, so `size()` counts what has been added
    /// and `maxNodeId()` bounds where it can have been added. `HnswGraphSearcher`
    /// sizes its `visited` bitset off *this*, not off `size()`
    /// (`HnswGraphSearcher.getGraphSize`); using `size()` there is an
    /// out-of-bounds bitset access waiting for the first merge.
    fn max_node_id(&self) -> i32 {
        // ARITH: `size()` is a node count, so it is non-negative for every
        // implementation in this crate; the searcher clamps the result at 0
        // before using it as a length either way.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.size() - 1
        }
    }
    /// Replaces `out`'s contents with `node`'s neighbours on `level`.
    fn neighbors_into(&self, level: i32, node: i32, out: &mut Vec<i32>) -> Result<()>;
    /// `HnswGraph.getSortedNodes(level)`: the level's nodes, ascending.
    fn sorted_nodes_on_level(&self, level: i32) -> Result<Vec<i32>>;
}

/// Port of `org.apache.lucene.util.hnsw.OnHeapHnswGraph`: the graph as built,
/// before serialization.
///
/// `graph[node]` is the node's per-level neighbour arrays, level 0 first. An
/// absent node is an empty `Vec`, which is how "this ordinal has not been
/// added yet" is spelled (Java uses a `null` slot).
#[derive(Debug, Clone)]
pub struct OnHeapHnswGraph {
    /// `M + 1`: Java over-allocates by one so `addAndEnsureDiversity` can
    /// append before pruning.
    nsize: usize,
    /// `2 * M + 1`, the level-0 budget.
    nsize0: usize,
    pub(crate) graph: Vec<Vec<NeighborArray>>,
    entry_node: i32,
    entry_level: i32,
    size: i32,
    max_node_id: i32,
    /// `OnHeapHnswGraph`'s `noGrowth`: when the eventual node count is known
    /// up front (every merge knows it), Java refuses to grow the graph and
    /// reports `maxNodeId()` as `graph.length - 1` rather than the largest
    /// ordinal added so far. That matters because merging adds nodes **out of
    /// order**, so the largest-so-far is not a bound on what a search may
    /// visit.
    fixed_size: Option<i32>,
}

impl OnHeapHnswGraph {
    pub fn new(m: i32) -> Self {
        // The upper bound is deliberately *not* `MAXIMUM_MAX_CONN`: a graph
        // may legally be constructed past it (`with_graph` is what rejects
        // that, so the error names the format rule rather than an assertion).
        // It is the bound that keeps `2 * m + 1` inside `i32`.
        assert!(
            m > 0 && m <= (i32::MAX - 1) / 2,
            "M (max connections) must be in 1..={}",
            (i32::MAX - 1) / 2
        );
        // ARITH: bounded by the assertion directly above.
        #[allow(clippy::arithmetic_side_effects)]
        OnHeapHnswGraph {
            nsize: (m + 1) as usize,
            nsize0: (m * 2 + 1) as usize,
            graph: Vec::new(),
            // Java's initial `EntryNode(-1, 1)`; `entry_node == -1` is the
            // "no entry node yet" marker `trySetNewEntryNode` looks for.
            entry_node: -1,
            entry_level: 1,
            size: 0,
            max_node_id: -1,
            fixed_size: None,
        }
    }

    /// `new OnHeapHnswGraph(M, numNodes)`: the ordinal space is known up
    /// front, so every node slot exists from the start and no ordinal outside
    /// `0..num_nodes` may be added. Every merge uses this form.
    pub fn with_size(m: i32, num_nodes: i32) -> Self {
        assert!(m > 0, "M (max connections) must be positive");
        assert!(num_nodes >= 0, "num_nodes must be non-negative");
        let mut graph = Self::new(m);
        graph.graph = vec![Vec::new(); num_nodes as usize];
        graph.fixed_size = Some(num_nodes);
        graph
    }

    /// `OnHeapHnswGraph.addNode`. Nodes are always added from their top level
    /// downward, so this fills every level from 0 up to `level` at once (Java
    /// allocates the per-level array at the top level and leaves the lower
    /// slots null until their own `addNode` call fills them; since every
    /// intermediate array starts empty either way, the observable result is
    /// identical).
    pub fn add_node(&mut self, level: i32, node: i32) {
        // Both are ordinals, and `as usize` sign-extends: a negative `node`
        // would ask `resize_with` for ~2^64 slots (an **abort**, which
        // `catch_unwind` cannot intercept) and a negative `level` would push
        // neighbour arrays until memory ran out. Java has neither case because
        // `int[]`/array indexing throws on a negative index; here the failure
        // is a dead process, so it is asserted rather than left implicit.
        assert!(node >= 0, "node ordinal must be non-negative, got {node}");
        assert!(level >= 0, "level must be non-negative, got {level}");
        let node = node as usize;
        let level = level as usize;
        if node >= self.graph.len() {
            assert!(
                self.fixed_size.is_none(),
                "the graph does not expect to grow when an initial size is given"
            );
            // ARITH: `node` is a non-negative `i32` widened to `usize`.
            #[allow(clippy::arithmetic_side_effects)]
            self.graph.resize_with(node + 1, Vec::new);
        }
        if self.graph[node].is_empty() {
            // Saturating rather than a proof: one increment per node slot
            // that goes from empty to occupied, and the ordinals `0..=i32::MAX`
            // admit `i32::MAX + 1` of them -- one more than the counter can
            // hold. That graph cannot be built (it needs 2^31 `Vec`s), so the
            // saturation is unreachable; Java's `size++` would wrap negative
            // there, which is the worse of the two answers.
            self.size = self.size.saturating_add(1);
        }
        while self.graph[node].len() <= level {
            let l = self.graph[node].len();
            let max = if l == 0 { self.nsize0 } else { self.nsize };
            self.graph[node].push(NeighborArray::new(max, true));
        }
        let max = if level == 0 { self.nsize0 } else { self.nsize };
        self.graph[node][level] = NeighborArray::new(max, true);
        self.max_node_id = self.max_node_id.max(node as i32);
    }

    pub fn neighbors(&self, level: i32, node: i32) -> &NeighborArray {
        &self.graph[node as usize][level as usize]
    }

    pub fn neighbors_mut(&mut self, level: i32, node: i32) -> &mut NeighborArray {
        &mut self.graph[node as usize][level as usize]
    }

    pub fn node_exists_at_level(&self, level: i32, node: i32) -> bool {
        self.graph
            .get(node as usize)
            .is_some_and(|levels| levels.len() > level as usize)
    }

    /// `OnHeapHnswGraph.trySetNewEntryNode`.
    pub fn try_set_new_entry_node(&mut self, node: i32, level: i32) -> bool {
        // `num_levels()` is `entry_level + 1`; this and `try_promote` are the
        // only writers, so the bound lives here.
        assert!(
            (0..i32::MAX).contains(&level),
            "level must be in 0..i32::MAX, got {level}"
        );
        if self.entry_node == -1 {
            self.entry_node = node;
            self.entry_level = level;
            return true;
        }
        false
    }

    /// `OnHeapHnswGraph.tryPromoteNewEntryNode`. Single-threaded, so the
    /// `expectOldLevel` compare-and-set always succeeds; the parameter is kept
    /// so the assertion Java relies on is still checked.
    pub fn try_promote_new_entry_node(
        &mut self,
        node: i32,
        level: i32,
        expect_old_level: i32,
    ) -> bool {
        assert!(
            (0..i32::MAX).contains(&level),
            "level must be in 0..i32::MAX, got {level}"
        );
        debug_assert!(level > expect_old_level);
        if self.entry_level == expect_old_level {
            self.entry_node = node;
            self.entry_level = level;
            return true;
        }
        false
    }
}

impl HnswGraphView for OnHeapHnswGraph {
    fn size(&self) -> i32 {
        self.size
    }

    fn num_levels(&self) -> i32 {
        // ARITH: `entry_level` starts at 1 and is only ever assigned from
        // `try_set_new_entry_node`/`try_promote_new_entry_node`, both of which
        // assert `level < i32::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.entry_level + 1
        }
    }

    fn entry_node(&self) -> i32 {
        self.entry_node
    }

    fn max_conn(&self) -> i32 {
        // ARITH: `nsize` is `m + 1` for an `m` the constructor bounded at
        // `(i32::MAX - 1) / 2`, so it is in `2..=(i32::MAX + 1) / 2`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.nsize as i32 - 1
        }
    }

    fn max_node_id(&self) -> i32 {
        match self.fixed_size {
            // ARITH: `with_size` asserts `num_nodes >= 0`.
            #[allow(clippy::arithmetic_side_effects)]
            Some(n) => n - 1,
            None => self.max_node_id,
        }
    }

    fn neighbors_into(&self, level: i32, node: i32, out: &mut Vec<i32>) -> Result<()> {
        out.clear();
        out.extend_from_slice(self.neighbors(level, node).nodes());
        Ok(())
    }

    fn sorted_nodes_on_level(&self, level: i32) -> Result<Vec<i32>> {
        if level == 0 {
            return Ok((0..self.size).collect());
        }
        // Node ids are visited ascending, so the result is already sorted --
        // Java's `Arrays.sort` in `getSortedNodes` is a no-op on this input.
        Ok((0..self.graph.len())
            .filter(|&n| self.graph[n].len() > level as usize)
            .map(|n| n as i32)
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Searcher
// ---------------------------------------------------------------------------

/// `HnswGraphSearcher.expectedVisitedNodes`: `ln(graphSize) * k`, the
/// approximation Lucene uses to decide whether a graph search is even worth
/// it compared with an exhaustive scan.
pub fn expected_visited_nodes(k: i32, graph_size: i32) -> i32 {
    ((graph_size as f64).ln() * k as f64) as i32
}

/// Port of `org.apache.lucene.util.hnsw.HnswGraphSearcher` (and the shared
/// half of `AbstractHnswGraphSearcher`).
#[derive(Debug)]
pub struct HnswGraphSearcher {
    candidates: NeighborQueue,
    visited: FixedBitSet,
    bulk_nodes: Vec<i32>,
    bulk_scores: Vec<f32>,
    neighbor_scratch: Vec<i32>,
}

impl HnswGraphSearcher {
    pub fn new(k: usize, graph_size: i32) -> Self {
        HnswGraphSearcher {
            candidates: NeighborQueue::new(k.max(1), true),
            visited: FixedBitSet::new(graph_size.max(1) as usize),
            bulk_nodes: Vec::new(),
            bulk_scores: Vec::new(),
            neighbor_scratch: Vec::new(),
        }
    }

    /// `AbstractHnswGraphSearcher.search`: descend to the best entry point,
    /// then beam-search level 0.
    pub fn search<G: HnswGraphView, S: VectorScorer>(
        &mut self,
        results: &mut KnnCollector,
        scorer: &mut S,
        graph: &G,
        accept_ords: Option<&FixedBitSet>,
    ) -> Result<()> {
        let ep = self.find_best_entry_point(scorer, graph, results)?;
        if ep == -1 {
            return Ok(());
        }
        self.search_level(results, scorer, 0, &[ep], graph, accept_ords)
    }

    /// Port of `org.apache.lucene.util.hnsw.SeededHnswGraphSearcher`: the
    /// same beam search, started from entry points the caller already knows
    /// instead of from the graph's own entry node.
    ///
    /// Java makes this a subclass of `AbstractHnswGraphSearcher` whose
    /// `findBestEntryPoint` returns a constant and whose `searchLevel`
    /// delegates to the wrapped searcher, so the whole class *is* "skip the
    /// upper-level descent and use these ordinals". Here that is a second
    /// entry point on the one searcher, which also keeps the scratch state
    /// ([`Self::prepare_scratch_state`]'s bitset and bulk buffers) shared
    /// rather than duplicated behind a delegate.
    ///
    /// **What it changes and what it does not.** Only *where* level 0's beam
    /// starts. The collector, the accept set, the visit limit and the
    /// diversity of the walk are untouched, so a seeded search can return a
    /// different approximate answer but never a differently-*shaped* one.
    ///
    /// The cost is a **trade, not a saving in general**:
    /// `findBestEntryPoint`'s hill climb over every level above 0 is not run
    /// (nor its `collector.incVisitedCount(1)` for the entry node), but
    /// `searchLevel` then bulk-scores every seed instead of one entry point.
    /// The descent is `O(log n)` nodes and the seed set is whatever the
    /// caller passes, so seeding is a clear win for a narrow beam and can
    /// break even -- or lose -- once `|seeds|` approaches it. Measured on
    /// this port's fixtures: 26% fewer vector comparisons for `k = 10` over
    /// 4000 vectors with 10 seeds, but 4.1% for `k = 100` over 700 vectors
    /// with 93 (`docs/sweep/m2/c21-hnsw-seeded.md`).
    ///
    /// `seed_ords` is Java's `SeededHnswGraphSearcher.fromEntryPoints`
    /// argument, already resolved to level-0 ordinals (Java resolves them
    /// from doc ids through `SeededKnnVectorQuery.MappedDISI`). Its two
    /// rejections are ported as errors rather than as Java's
    /// `IllegalArgumentException`/`assert` pair, because an out-of-range
    /// ordinal indexes the `visited` bitset and Java's check for that is an
    /// `assert`, i.e. absent in production.
    pub fn search_seeded<G: HnswGraphView, S: VectorScorer>(
        &mut self,
        results: &mut KnnCollector,
        scorer: &mut S,
        graph: &G,
        accept_ords: Option<&FixedBitSet>,
        seed_ords: &[i32],
    ) -> Result<()> {
        // `fromEntryPoints`: "The number of entry points must be > 0".
        if seed_ords.is_empty() {
            return Err(Error::InvalidGraphParameter(
                "the number of seeded entry points must be > 0".to_string(),
            ));
        }
        let size = graph.size();
        for &ord in seed_ords {
            if ord < 0 || ord >= size {
                return Err(Error::InvalidGraphParameter(format!(
                    "seeded entry point {ord} is outside the graph's 0..{size} ordinals"
                )));
            }
        }
        self.search_level(results, scorer, 0, seed_ords, graph, accept_ords)
    }

    /// `HnswGraphSearcher.getGraphSize(graph)` -- `maxNodeId() + 1`, not
    /// `size()`; see [`HnswGraphView::max_node_id`]. Widened to `i64` because
    /// `max_node_id()` is a trait method: an implementation returning
    /// `i32::MAX` would overflow the `+ 1` Java performs in `int`.
    fn graph_capacity<G: HnswGraphView>(graph: &G) -> usize {
        // ARITH: an `i32` widened to `i64` cannot overflow a `+ 1`.
        #[allow(clippy::arithmetic_side_effects)]
        let capacity = i64::from(graph.max_node_id()) + 1;
        usize::try_from(capacity).unwrap_or(0)
    }

    /// `graph.maxConn() * 2`, the width of the bulk-scoring buffers. Same
    /// widening, same reason.
    fn bulk_width<G: HnswGraphView>(graph: &G) -> usize {
        // ARITH: an `i32` widened to `i64`, doubled, stays inside `i64`.
        #[allow(clippy::arithmetic_side_effects)]
        let width = i64::from(graph.max_conn()) * 2;
        usize::try_from(width).unwrap_or(0)
    }

    fn prepare_scratch_state(&mut self, capacity: usize, bulk_score_size: usize) {
        self.candidates.clear();
        if self.visited.len() < capacity {
            // `FixedBitSet.ensureCapacityAndClear`
            self.visited = FixedBitSet::new(capacity);
        } else {
            self.visited.clear_all();
        }
        if self.bulk_nodes.len() < bulk_score_size {
            self.bulk_nodes = vec![0; bulk_score_size];
            self.bulk_scores = vec![0.0; bulk_score_size];
        }
    }

    /// `HnswGraphSearcher.findBestEntryPoint`: greedy hill-climbing from the
    /// top level down to level 1. Returns `-1` for "no entry node, or the
    /// visit limit was hit" (Java's `UNK_EP`).
    pub fn find_best_entry_point<G: HnswGraphView, S: VectorScorer>(
        &mut self,
        scorer: &mut S,
        graph: &G,
        collector: &mut KnnCollector,
    ) -> Result<i32> {
        let mut current_ep = graph.entry_node();
        if current_ep == -1 || graph.num_levels() == 1 {
            return Ok(current_ep);
        }
        let size = Self::graph_capacity(graph);
        self.prepare_scratch_state(size, Self::bulk_width(graph));
        // Java asserts `friendOrd < size` inside the loop below -- an
        // assertion, i.e. absent in production, where the same ordinal then
        // indexes `visited`. `FixedBitSet::get`/`set` here index
        // `words[index >> 6]` behind a `debug_assert`, so an out-of-range
        // ordinal is a *ghost bit* in a release build (a silently wrong
        // "already visited" answer) or an index panic 64 bits later. The bound
        // is taken against the bitset's own length, per
        // `docs/arithmetic-gate.md`, and hoisted out of both loops.
        let visited_len = self.visited.len();
        if current_ep < 0 || current_ep as usize >= visited_len {
            return Err(Error::CorruptMeta(format!(
                "graph entry node {current_ep} is outside its own 0..{visited_len} ordinals"
            )));
        }
        let mut current_score = scorer.score(current_ep)?;
        collector.inc_visited_count(1);
        for level in (1..graph.num_levels()).rev() {
            let mut found_better = true;
            self.visited.set(current_ep as usize);
            while found_better {
                found_better = false;
                graph.neighbors_into(level, current_ep, &mut self.neighbor_scratch)?;
                // The neighbours just read off the `.vex` are file-derived
                // ordinals, and both of the uses below index a fixed-size
                // buffer with them: `visited` (a `FixedBitSet`, whose `get`
                // only `debug_assert`s its bound) and `bulk_nodes`. Checking
                // them here is what makes the two `#[allow]`s below sound --
                // `check_neighbors` is the enforcer their proofs name.
                check_neighbors(
                    &self.neighbor_scratch,
                    self.visited.len(),
                    self.bulk_nodes.len(),
                )?;
                let mut num_nodes = 0usize;
                for i in 0..self.neighbor_scratch.len() {
                    let friend = self.neighbor_scratch[i];
                    if self.visited.get(friend as usize) {
                        continue;
                    }
                    self.visited.set(friend as usize);
                    if collector.early_terminated() {
                        return Ok(-1);
                    }
                    self.bulk_nodes[num_nodes] = friend;
                    // ARITH: `check_neighbors` bounded the neighbour count by
                    // `bulk_nodes.len()`, and this counts a subset of them.
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        num_nodes += 1;
                    }
                }
                let max_score = if num_nodes > 0 {
                    scorer.bulk_score(
                        &self.bulk_nodes[..num_nodes],
                        &mut self.bulk_scores[..num_nodes],
                    )?
                } else {
                    f32::NEG_INFINITY
                };
                collector.inc_visited_count(num_nodes);
                if max_score > current_score {
                    for i in 0..num_nodes {
                        let score = self.bulk_scores[i];
                        if score > current_score {
                            current_score = score;
                            current_ep = self.bulk_nodes[i];
                            found_better = true;
                        }
                    }
                }
            }
        }
        Ok(if collector.early_terminated() {
            -1
        } else {
            current_ep
        })
    }

    /// `HnswGraphSearcher.searchLevel`: beam search from `eps` on one level.
    pub fn search_level<G: HnswGraphView, S: VectorScorer>(
        &mut self,
        results: &mut KnnCollector,
        scorer: &mut S,
        level: i32,
        eps: &[i32],
        graph: &G,
        accept_ords: Option<&FixedBitSet>,
    ) -> Result<()> {
        let size = Self::graph_capacity(graph);
        self.prepare_scratch_state(size, Self::bulk_width(graph));
        // Same bound, same reason as in `find_best_entry_point`: `eps` reaches
        // here from a caller (`search_seeded`) or from the level above, and
        // every `friend` below comes off a `.vex` through
        // `HnswGraphView::neighbors_into`.
        let visited_len = self.visited.len();
        check_neighbors(eps, visited_len, usize::MAX)?;
        // `accept_ords` is a *second* bitset, supplied by the caller and
        // indexed with the same ordinals. `crate::hnsw_vectors::search` sizes
        // it against this graph, but this method is public and Java's `Bits`
        // is unbounded-by-construction, so the pair is checked here too --
        // once per search, not per candidate.
        if let Some(bits) = accept_ords {
            if bits.len() < visited_len {
                return Err(Error::InvalidGraphParameter(format!(
                    "the accept-ordinal set covers {} ordinals, short of the {visited_len} this \
                     graph can name",
                    bits.len()
                )));
            }
        }
        if self.bulk_scores.len() < eps.len() {
            self.bulk_nodes.resize(eps.len(), 0);
            self.bulk_scores.resize(eps.len(), 0.0);
        }
        if results.early_terminated() {
            return Ok(());
        }

        // `AbstractHnswGraphSearcher.scoreEntryPoints`
        scorer.bulk_score(eps, &mut self.bulk_scores[..eps.len()])?;
        results.inc_visited_count(eps.len());
        for (i, &ep) in eps.iter().enumerate() {
            let score = self.bulk_scores[i];
            self.visited.set(ep as usize);
            self.candidates.add(ep, score);
            if accept_ords.is_none_or(|b| b.get(ep as usize)) {
                results.collect(ep, score);
            }
        }
        if results.early_terminated() {
            return Ok(());
        }

        let mut min_accepted_similarity = next_up(results.min_competitive_similarity());
        let mut should_explore_min_sim = true;
        while self.candidates.size() > 0 && !results.early_terminated() {
            let top_candidate_similarity = self.candidates.top_score();
            if top_candidate_similarity < min_accepted_similarity {
                if should_explore_min_sim
                    && next_up(top_candidate_similarity) == min_accepted_similarity
                {
                    should_explore_min_sim = false;
                } else {
                    break;
                }
            }
            let top_candidate_node = self.candidates.pop();
            graph.neighbors_into(level, top_candidate_node, &mut self.neighbor_scratch)?;
            // The neighbours just read off the `.vex` are file-derived
            // ordinals, and both of the uses below index a fixed-size
            // buffer with them: `visited` (a `FixedBitSet`, whose `get`
            // only `debug_assert`s its bound) and `bulk_nodes`. Checking
            // them here is what makes the two `#[allow]`s below sound --
            // `check_neighbors` is the enforcer their proofs name.
            check_neighbors(
                &self.neighbor_scratch,
                self.visited.len(),
                self.bulk_nodes.len(),
            )?;
            let mut num_nodes = 0usize;
            for i in 0..self.neighbor_scratch.len() {
                let friend = self.neighbor_scratch[i];
                if self.visited.get(friend as usize) {
                    continue;
                }
                self.visited.set(friend as usize);
                if results.early_terminated() {
                    break;
                }
                self.bulk_nodes[num_nodes] = friend;
                // ARITH: `check_neighbors` bounded the neighbour count by
                // `bulk_nodes.len()`, and this counts a subset of them.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    num_nodes += 1;
                }
            }
            // `Math.min(numNodes, visitLimit - visitedCount)`. The while
            // condition guarantees `visited_count < visit_limit` here, so
            // Java's signed subtraction cannot go negative either -- but
            // these are `u64`, where the same impossible case would be an
            // overflow panic rather than a negative count.
            let remaining = results
                .visit_limit()
                .saturating_sub(results.visited_count());
            num_nodes = num_nodes.min(remaining as usize);
            results.inc_visited_count(num_nodes);
            if num_nodes > 0 {
                let max = scorer.bulk_score(
                    &self.bulk_nodes[..num_nodes],
                    &mut self.bulk_scores[..num_nodes],
                )?;
                if max > results.min_competitive_similarity() {
                    for i in 0..num_nodes {
                        let node = self.bulk_nodes[i];
                        let score = self.bulk_scores[i];
                        if score >= min_accepted_similarity {
                            self.candidates.add(node, score);
                            if accept_ords.is_none_or(|b| b.get(node as usize))
                                && results.collect(node, score)
                            {
                                let old = min_accepted_similarity;
                                min_accepted_similarity =
                                    next_up(results.min_competitive_similarity());
                                if min_accepted_similarity > old {
                                    should_explore_min_sim = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// One pass over a neighbour list before it is walked: every ordinal must be
/// a legal index into the `visited` bitset, and there must not be more of them
/// than the bulk-scoring buffers hold.
///
/// Both are Java `assert`s (`friendOrd < size`, and the implicit
/// `bulkNodes[numNodes++]`), i.e. absent in production, where the first is a
/// ghost bit in a `FixedBitSet` and the second an
/// `ArrayIndexOutOfBoundsException`. A `.vex` that names an ordinal past the
/// graph is rejected by [`crate::hnsw_vectors::OffHeapHnswGraph`] before it
/// gets here, so this is the second line for any other `HnswGraphView` -- and
/// it is one pass over a list the caller is about to walk twice anyway.
fn check_neighbors(neighbors: &[i32], visited_len: usize, bulk_width: usize) -> Result<()> {
    if neighbors.len() > bulk_width {
        return Err(Error::CorruptMeta(format!(
            "a node has {} neighbours, more than the {bulk_width} its graph's M allows",
            neighbors.len()
        )));
    }
    for &friend in neighbors {
        if friend < 0 || friend as usize >= visited_len {
            return Err(Error::CorruptMeta(format!(
                "neighbour ordinal {friend} is outside the graph's 0..{visited_len} ordinals"
            )));
        }
    }
    Ok(())
}

/// `Math.nextUp(float)`. `-inf` maps to `-f32::MAX` in Java, which
/// `f32::next_up` also does; `NaN` stays `NaN`.
fn next_up(v: f32) -> f32 {
    v.next_up()
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Port of `org.apache.lucene.util.hnsw.HnswGraphBuilder`.
#[derive(Debug)]
pub struct HnswGraphBuilder<S: UpdateableVectorScorer> {
    m: i32,
    ml: f64,
    random: SplittableRandom,
    scorer: S,
    searcher: HnswGraphSearcher,
    hnsw: OnHeapHnswGraph,
    /// The seed this builder was constructed with. Kept because
    /// [`Self::rebalance_graph`] needs its own RNG (Java's is a separate,
    /// *unseeded* `SplittableRandom`), and drawing it from a module constant
    /// instead would make a merged graph a function of its inputs but not of
    /// the caller's seed.
    seed: u64,
    entry_candidates: KnnCollector,
    beam_candidates: KnnCollector,
    /// `beamCandidates0`: the collector used for level 0 **when the caller
    /// supplies its own entry points** (`MergingHnswGraphBuilder`). Java sizes
    /// it `min(beamWidth / 2, M * 3)` -- a narrower beam, on the grounds that
    /// prior knowledge of where the node belongs makes a wide one wasted work.
    beam_candidates0: KnnCollector,
    beam_width: usize,
}

impl<S: UpdateableVectorScorer> HnswGraphBuilder<S> {
    /// `HnswGraphBuilder.create(scorerSupplier, M, beamWidth, seed)`.
    pub fn new(scorer: S, m: i32, beam_width: i32, seed: u64) -> Result<Self> {
        // `HnswGraphBuilder`'s two checks plus `Lucene99HnswVectorsFormat`'s
        // constructor bounds. The upper bounds matter here and not only in the
        // format: without them a graph builds happily at `M = 1000` and the
        // `.vem` it produces is then rejected by this port's own reader (and by
        // Lucene's, whose `OffHeapHnswGraph` allocates `int[M * 2]`).
        if !(1..=MAXIMUM_MAX_CONN).contains(&m) {
            return Err(Error::InvalidGraphParameter(format!(
                "M (max connections) must be in 1..={MAXIMUM_MAX_CONN}, got {m}"
            )));
        }
        if !(1..=MAXIMUM_BEAM_WIDTH).contains(&beam_width) {
            return Err(Error::InvalidGraphParameter(format!(
                "beamWidth must be in 1..={MAXIMUM_BEAM_WIDTH}, got {beam_width}"
            )));
        }
        Self::with_graph(scorer, beam_width, seed, OnHeapHnswGraph::new(m))
    }

    /// `HnswGraphBuilder(scorerSupplier, beamWidth, seed, hnsw)`: build **onto
    /// an existing graph** rather than an empty one, which is what every
    /// merge-time builder does (`InitializedHnswGraphBuilder`,
    /// `MergingHnswGraphBuilder`). `M` comes from the graph, exactly as in
    /// Java (`this.M = hnsw.maxConn()`).
    pub fn with_graph(
        scorer: S,
        beam_width: i32,
        seed: u64,
        hnsw: OnHeapHnswGraph,
    ) -> Result<Self> {
        let m = hnsw.max_conn();
        if !(1..=MAXIMUM_MAX_CONN).contains(&m) {
            return Err(Error::InvalidGraphParameter(format!(
                "M (max connections) must be in 1..={MAXIMUM_MAX_CONN}, got {m}"
            )));
        }
        if !(1..=MAXIMUM_BEAM_WIDTH).contains(&beam_width) {
            return Err(Error::InvalidGraphParameter(format!(
                "beamWidth must be in 1..={MAXIMUM_BEAM_WIDTH}, got {beam_width}"
            )));
        }
        // Java writes `Math.min(beamWidth / 2, M * 3)`, which is **0** for
        // `beamWidth == 1` -- a zero-capacity collector whose `popNode` would
        // then be called. Unreachable in Lucene only because nothing configures
        // `beamWidth = 1`; clamped to 1 here rather than reproduced, since a
        // zero-capacity heap is a panic, not a behaviour.
        // ARITH: `m` is in `1..=MAXIMUM_MAX_CONN` (512) and `beam_width` in
        // `1..=MAXIMUM_BEAM_WIDTH` (3200), both checked immediately above.
        #[allow(clippy::arithmetic_side_effects)]
        let beam0 = ((beam_width / 2).min(m * 3)).max(1) as usize;
        Ok(HnswGraphBuilder {
            m,
            // `ml = M == 1 ? 1 : 1 / Math.log(1.0 * M)`
            ml: if m == 1 { 1.0 } else { 1.0 / (m as f64).ln() },
            random: SplittableRandom::new(seed),
            seed,
            scorer,
            searcher: HnswGraphSearcher::new(beam_width as usize, 1),
            hnsw,
            entry_candidates: KnnCollector::unlimited(1),
            beam_candidates: KnnCollector::unlimited(beam_width as usize),
            beam_candidates0: KnnCollector::unlimited(beam0),
            beam_width: beam_width as usize,
        })
    }

    /// `HnswGraphBuilder.build(maxOrd)`.
    pub fn build(mut self, max_ord: i32) -> Result<OnHeapHnswGraph> {
        for node in 0..max_ord {
            self.add_graph_node(node)?;
        }
        Ok(self.hnsw)
    }

    pub fn graph(&self) -> &OnHeapHnswGraph {
        &self.hnsw
    }

    /// `HnswGraphBuilder.getRandomGraphLevel`.
    fn random_graph_level(&mut self) -> i32 {
        let mut rand_double;
        loop {
            // Avoid 0: log(0) is undefined.
            rand_double = self.random.next_double();
            if rand_double != 0.0 {
                break;
            }
        }
        // `as i32` saturates in Rust where Java's `(int)` cast truncates
        // toward zero, but neither is reachable: `-ln(u)` is at most ~745 for
        // any non-zero `f64`, and `ml = 1 / ln(M)` is at most `1 / ln(2)`
        // (`M == 1` is special-cased to 1.0), so the product is under 1100.
        (-rand_double.ln() * self.ml) as i32
    }

    /// `HnswGraphBuilder.addGraphNode` + `addGraphNodeInternal`.
    ///
    /// Java wraps the body in `do { ... } while (true)`, re-running it when a
    /// concurrent builder moved the entry node between the read of
    /// `numLevels()` and the promotion attempt. This port builds on one
    /// thread: `tryPromoteNewEntryNode` therefore always succeeds on the
    /// first attempt (nothing else can have changed `entry_level`), so the
    /// loop is written out as its single iteration. The `IllegalStateException`
    /// Java throws when neither the promotion nor a level change happened is
    /// unreachable for the same reason.
    pub fn add_graph_node(&mut self, node: i32) -> Result<()> {
        self.add_graph_node_internal(node, None)
    }

    /// `HnswGraphBuilder.addGraphNode(node, eps0)`: the same insertion, but
    /// with the level-0 search seeded from `eps0` instead of from the entry
    /// node's descent -- `MergingHnswGraphBuilder`'s "we already know roughly
    /// where this node belongs" shortcut. A narrower beam
    /// ([`Self::beam_candidates0`]) is used with it, as in Java.
    pub fn add_graph_node_with_entry_points(&mut self, node: i32, eps0: &[i32]) -> Result<()> {
        self.add_graph_node_internal(node, Some(eps0))
    }

    fn add_graph_node_internal(&mut self, node: i32, eps0: Option<&[i32]>) -> Result<()> {
        self.scorer.set_scoring_ordinal(node)?;
        let node_level = self.random_graph_level();
        for level in (0..=node_level).rev() {
            self.hnsw.add_node(level, node);
        }
        if self.hnsw.try_set_new_entry_node(node, node_level) {
            return Ok(());
        }

        // ARITH: `num_levels()` is `entry_level + 1` and `entry_level` is at
        // least 0 once an entry node exists (which `try_set_new_entry_node`
        // returning false above establishes), so it is at least 1.
        #[allow(clippy::arithmetic_side_effects)]
        let cur_max_level = self.hnsw.num_levels() - 1;
        let mut eps = vec![self.hnsw.entry_node()];

        // Levels above the new node's: descend with topk = 1.
        // ARITH: `node_level` comes from `random_graph_level`, which cannot
        // exceed ~1100 (see there).
        #[allow(clippy::arithmetic_side_effects)]
        let above = (node_level + 1)..=cur_max_level;
        for level in above.rev() {
            self.entry_candidates.clear();
            self.searcher.search_level(
                &mut self.entry_candidates,
                &mut self.scorer,
                level,
                &eps,
                &self.hnsw,
                None,
            )?;
            eps[0] = self.entry_candidates.pop_node();
        }

        // Levels the new node is on: beam-search, then connect bottom-up.
        let top = node_level.min(cur_max_level);
        // ARITH: `top <= node_level`, which `random_graph_level` bounds at
        // ~1100.
        #[allow(clippy::arithmetic_side_effects)]
        let mut scratch_per_level: Vec<NeighborArray> =
            Vec::with_capacity((top + 1).max(0) as usize);
        for _ in 0..=top {
            scratch_per_level.push(NeighborArray::new(1, false));
        }
        for level in (0..=top).rev() {
            // Java swaps in `beamCandidates0` and the caller's entry points
            // for level 0 when `eps0` is non-empty, and keeps the wide beam
            // everywhere else.
            let use_eps0 = level == 0 && eps0.is_some_and(|e| !e.is_empty());
            let candidates = if use_eps0 {
                &mut self.beam_candidates0
            } else {
                &mut self.beam_candidates
            };
            candidates.clear();
            let search_eps: &[i32] = if use_eps0 { eps0.unwrap() } else { &eps };
            self.searcher.search_level(
                candidates,
                &mut self.scorer,
                level,
                search_eps,
                &self.hnsw,
                None,
            )?;
            eps = candidates.pop_until_nearest_k_nodes();
            // `new NeighborArray(Math.max(candidates.k(), M + 1), false)` --
            // `candidates.k()` differs between the two collectors, so this has
            // to read it off whichever one was used.
            // ARITH: `m` is in `1..=MAXIMUM_MAX_CONN` (512).
            #[allow(clippy::arithmetic_side_effects)]
            let mut scratch = NeighborArray::new(candidates.k().max(self.m as usize + 1), false);
            pop_to_scratch(candidates, &mut scratch);
            scratch_per_level[level as usize] = scratch;
        }
        for level in 0..=top {
            let scratch = std::mem::replace(
                &mut scratch_per_level[level as usize],
                NeighborArray::new(1, false),
            );
            self.add_diverse_neighbors(level, node, &scratch, false)?;
        }

        if node_level > cur_max_level {
            let promoted = self
                .hnsw
                .try_promote_new_entry_node(node, node_level, cur_max_level);
            debug_assert!(promoted, "single-threaded promotion cannot fail");
        }
        Ok(())
    }

    /// `HnswGraphBuilder.addDiverseNeighbors`.
    fn add_diverse_neighbors(
        &mut self,
        level: i32,
        node: i32,
        candidates: &NeighborArray,
        is_link_repair: bool,
    ) -> Result<()> {
        // ARITH: `m` is in `1..=MAXIMUM_MAX_CONN` (512).
        #[allow(clippy::arithmetic_side_effects)]
        let max_conn_on_level = if level == 0 { self.m * 2 } else { self.m };
        let mask = self.select_and_link_diverse(
            level,
            node,
            candidates,
            max_conn_on_level,
            is_link_repair,
        )?;
        // Uses `candidates` + `mask` rather than the neighbour array, because
        // adding an incoming link below can make this node discoverable and
        // therefore its neighbour array mutable underneath us.
        for (i, selected) in mask.iter().enumerate() {
            if !*selected {
                continue;
            }
            let nbr = candidates.nodes()[i];
            let score = candidates.score(i);
            // `updateNeighbor`: during link repair the same edge can already
            // exist (the node kept some of its old neighbours), and adding it
            // twice would leave a duplicate arc in the serialized graph. Java
            // pays the scan only in that case, and so does this.
            if is_link_repair && self.hnsw.neighbors(level, nbr).nodes().contains(&node) {
                continue;
            }
            self.hnsw
                .neighbors_mut(level, nbr)
                .add_and_ensure_diversity(node, score, nbr, &mut self.scorer)?;
        }
        Ok(())
    }

    /// `HnswGraphBuilder.selectAndLinkDiverse`: walk the candidates best-first
    /// and keep a candidate only if it is closer to `node` than to every
    /// already-kept neighbour.
    fn select_and_link_diverse(
        &mut self,
        level: i32,
        node: i32,
        candidates: &NeighborArray,
        max_conn_on_level: i32,
        is_link_repair: bool,
    ) -> Result<Vec<bool>> {
        let mut mask = vec![false; candidates.size()];
        let neighbors = &mut self.hnsw.graph[node as usize][level as usize];
        // ARITH: `i` walks a `NeighborArray`'s length down to -1, and that
        // length is capped at `max_size`.
        #[allow(clippy::arithmetic_side_effects)]
        let mut i = candidates.size() as i64 - 1;
        while neighbors.size() < max_conn_on_level as usize && i >= 0 {
            let c_node = candidates.nodes()[i as usize];
            if node == c_node {
                // ARITH: the loop condition is `i >= 0`, so the decrement bottoms out
                // at -1.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    i -= 1;
                }
                continue;
            }
            let c_score = candidates.score(i as usize);
            self.scorer.set_scoring_ordinal(c_node)?;
            if Self::diversity_check(c_score, neighbors, &mut self.scorer)? {
                mask[i as usize] = true;
                // During link repair the array already holds the node's
                // surviving neighbours, whose scores are unrelated to these
                // candidates' -- so the "each new entry is worse than the last"
                // precondition `addInOrder` asserts does not hold.
                if is_link_repair {
                    neighbors.add_out_of_order(c_node, c_score);
                } else {
                    neighbors.add_in_order(c_node, c_score);
                }
            }
            // ARITH: as above -- the loop condition is `i >= 0`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                i -= 1;
            }
        }
        Ok(mask)
    }

    /// `HnswGraphBuilder.diversityCheck`: the candidate (already the scorer's
    /// current ordinal) is diverse iff **no** already-selected neighbour is at
    /// least as similar to it as it is to the new node.
    ///
    /// Java chunks this through `bulkScore` to amortise its vectorised kernel;
    /// the answer is the same either way -- it short-circuits on the first
    /// chunk whose maximum reaches `score`, which is exactly "some neighbour
    /// scores >= score".
    fn diversity_check<T: VectorScorer>(
        score: f32,
        neighbors: &NeighborArray,
        scorer: &mut T,
    ) -> Result<bool> {
        for i in 0..neighbors.size() {
            if scorer.score(neighbors.nodes()[i])? >= score {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// `InitializedHnswGraphBuilder.DISCONNECTED_NODE_FACTOR`: a node that kept
/// less than this fraction of the neighbours it had in the source graph lost
/// too many in *this* merge and is repaired.
const DISCONNECTED_NODE_FACTOR: f64 = 0.85;
/// `InitializedHnswGraphBuilder.CUMULATIVE_DEGREE_FLOOR_FACTOR`: a node whose
/// out-degree has drifted below this fraction of the level's connection budget
/// is repaired even if it only lost one neighbour this merge -- the check that
/// catches slow decay across many merges rather than one big loss.
const CUMULATIVE_DEGREE_FLOOR_FACTOR: f64 = 0.5;

/// Port of `org.apache.lucene.util.hnsw.UpdateGraphsUtils.computeJoinSet`: the
/// set of level-0 nodes that between them "cover" the graph, where covering a
/// node means being one of its neighbours enough times
/// ([`join_set_coverage`]).
///
/// `MergingHnswGraphBuilder` inserts exactly this set into the target graph the
/// ordinary way (a full beam search each), and then inserts every *other* node
/// of the source graph with entry points derived from its already-inserted
/// neighbours -- which is where the merge saves its work. A join set that is
/// too small makes those derived entry points useless; one that is the whole
/// graph makes the merge no cheaper than a rebuild.
pub fn compute_join_set<G: HnswGraphView>(graph: &G) -> Result<std::collections::HashSet<i32>> {
    let size = graph.size();
    let mut heap = TernaryLongHeap::new(size.max(1) as usize);
    let mut join: std::collections::HashSet<i32> = std::collections::HashSet::new();
    let mut stale = vec![false; size.max(0) as usize];
    // Java's `short[] counts`. Widened to `i32`: a node's cover count is
    // bounded only by its in-degree, so Java's `short` can in principle wrap
    // negative on a large graph and make an already-covered node look
    // uncovered. Widening cannot change the answer for any input Java gets
    // right, and removes an input class where it does not.
    let mut counts = vec![0i32; size.max(0) as usize];
    let mut g_exit = 0i64;
    let mut neighbors: Vec<i32> = Vec::new();
    for v in 0..size {
        graph.neighbors_into(0, v, &mut neighbors)?;
        let degree = degree_of(&neighbors);
        let k = join_set_coverage(degree);
        // ARITH: `k <= degree <= i32::MAX / 4` (see `degree_of`), and `g_exit`
        // accumulates one `k` per node, so it is bounded by
        // `size * i32::MAX / 4` -- under 2^61.
        #[allow(clippy::arithmetic_side_effects)]
        {
            g_exit += i64::from(k);
            heap.push(encode_gain(k + degree, v));
        }
    }

    let mut g_tot = 0i64;
    let mut neighbors_of_neighbor: Vec<i32> = Vec::new();
    while g_tot < g_exit && heap.size() > 0 {
        let element = heap.pop();
        let gain = decode_gain(element);
        let v = decode_gain_node(element);
        graph.neighbors_into(0, v, &mut neighbors)?;
        let degree = degree_of(&neighbors);
        let k = join_set_coverage(degree);
        // `counts` and `stale` are both `size` entries long, and every
        // neighbour ordinal below indexes both. A `.vex` neighbour is already
        // bounded by `OffHeapHnswGraph::neighbors_into`; this is the bound for
        // any other `HnswGraphView`, which would otherwise index out of range.
        check_neighbors(&neighbors, counts.len(), usize::MAX)?;
        if stale[v as usize] {
            // The gain recorded when `v` was pushed is out of date because a
            // node picked since then already covers some of `v`'s neighbours;
            // recompute and re-push rather than trusting it.
            // ARITH: `k` and every `counts` entry are non-negative `i32`s, so
            // the difference cannot overflow; `new_gain` is then incremented
            // at most once per neighbour, and `degree_of` caps a neighbour
            // count at `i32::MAX / 4`.
            #[allow(clippy::arithmetic_side_effects)]
            let mut new_gain = 0.max(k - counts[v as usize]);
            for u in &neighbors {
                if counts[*u as usize] < k && !join.contains(u) {
                    // ARITH: `new_gain` starts at most at `k` and takes at most one
                    // increment per neighbour, so it is bounded by
                    // `k + degree <= i32::MAX / 2` (see `degree_of`).
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        new_gain += 1;
                    }
                }
            }
            if new_gain > 0 {
                heap.push(encode_gain(new_gain, v));
                stale[v as usize] = false;
            }
        } else {
            join.insert(v);
            // ARITH: one `gain` per join-set member, each at most
            // `i32::MAX / 2` (`encode_gain`'s argument is `k + degree`), over
            // at most `size` members -- bounded by 2^62.
            #[allow(clippy::arithmetic_side_effects)]
            {
                g_tot += i64::from(gain);
            }
            let mark_neighbours_stale = counts[v as usize] < k;
            for &u in &neighbors {
                if mark_neighbours_stale {
                    stale[u as usize] = true;
                }
                // ARITH: `k` is non-negative, so `k - 1` cannot underflow an
                // `i32`.
                #[allow(clippy::arithmetic_side_effects)]
                let k_minus_one = k - 1;
                if counts[u as usize] < k_minus_one {
                    graph.neighbors_into(0, u, &mut neighbors_of_neighbor)?;
                    check_neighbors(&neighbors_of_neighbor, stale.len(), usize::MAX)?;
                    for uu in &neighbors_of_neighbor {
                        stale[*uu as usize] = true;
                    }
                }
                // Saturating rather than a proof: a node's cover count rises
                // by at most one per join-set member, so it is bounded by the
                // graph's node count -- which is exactly `i32::MAX` at the
                // boundary. `counts` is only ever compared against `k` (at
                // most a node's degree), so a saturated entry answers every
                // comparison the same way an exact one would.
                counts[u as usize] = counts[u as usize].saturating_add(1);
            }
        }
    }
    Ok(join)
}

/// `UpdateGraphsUtils.coverage`: how many join-set members have to point at a
/// node before it counts as covered, clamped to the node's own degree (a node
/// can only ever be covered by its own neighbours, so asking for more than
/// `degree` would force every leaf of a degenerate graph into the join set).
fn join_set_coverage(degree: i32) -> i32 {
    // ARITH: `Math.ceilDiv(degree, 4)` for a `degree` in `9..=i32::MAX / 4`
    // (`degree_of` caps it), so `degree + 3` cannot overflow.
    #[allow(clippy::arithmetic_side_effects)]
    let k = if degree < 9 { 2 } else { (degree + 3) / 4 };
    k.min(degree)
}

/// A level-0 neighbour count as an `i32`, capped rather than truncated.
///
/// `len() as i32` is the shape that turns a 2^31-entry list into a *negative*
/// degree; nothing in this port can produce one (a graph's neighbour list is
/// at most `2 * M` entries, 1024 at `MAXIMUM_MAX_CONN`), and the cap keeps
/// `k + degree` and `degree + 3` inside `i32` for anything that could.
fn degree_of(neighbors: &[i32]) -> i32 {
    i32::try_from(neighbors.len())
        .unwrap_or(i32::MAX / 4)
        .min(i32::MAX / 4)
}

/// `UpdateGraphsUtils.encode`: the gain is negated into the high half so the
/// min-heap pops the **largest** gain first.
// ARITH: `gain` is at most `i32::MAX / 2` (`k + degree`, both capped at
// `i32::MAX / 4` by `degree_of`), so negating it and shifting it into the high
// half stays inside `i64`; the shift amount is a constant 32.
#[allow(clippy::arithmetic_side_effects)]
fn encode_gain(gain: i32, node: i32) -> i64 {
    ((-(gain as i64)) << 32) | (node as u32 as i64)
}

// ARITH: the inverse of `encode_gain`, whose high half is `-gain` for a `gain`
// in `0..=i32::MAX / 2`, so the negation cannot reach `i32::MIN`.
#[allow(clippy::arithmetic_side_effects)]
fn decode_gain(encoded: i64) -> i32 {
    -((encoded >> 32) as i32)
}

fn decode_gain_node(encoded: i64) -> i32 {
    (encoded & 0xFFFF_FFFF) as u32 as i32
}

impl<S: UpdateableVectorScorer> HnswGraphBuilder<S> {
    /// Port of `InitializedHnswGraphBuilder.initGraph`: build a graph whose
    /// ordinal space is `0..total_vectors` by **copying** `initializer`'s
    /// structure through `new_ord_map` (old ordinal -> new ordinal, `-1` for a
    /// vector this merge drops), then -- only if anything was dropped --
    /// repairing the nodes that lost too many neighbours and rebalancing the
    /// level distribution.
    ///
    /// With no deletions this is a pure structural copy: no search, no
    /// scoring, no RNG draw. That is the whole point of the incremental merge,
    /// and it is why `merge_graphs` is cheaper than rebuilding.
    pub fn init_graph<G: HnswGraphView>(
        scorer: S,
        beam_width: i32,
        seed: u64,
        initializer: &G,
        new_ord_map: &[i32],
        total_vectors: i32,
    ) -> Result<OnHeapHnswGraph> {
        let hnsw = OnHeapHnswGraph::with_size(initializer.max_conn(), total_vectors);
        let mut builder = Self::with_graph(scorer, beam_width, seed, hnsw)?;
        builder.initialize_from_graph(initializer, new_ord_map)?;
        Ok(builder.hnsw)
    }

    /// `InitializedHnswGraphBuilder.initializeFromGraph`'s three phases.
    fn initialize_from_graph<G: HnswGraphView>(
        &mut self,
        initializer: &G,
        new_ord_map: &[i32],
    ) -> Result<()> {
        let (disconnected_by_level, level_to_nodes, has_deletes) =
            self.copy_graph_structure(initializer, new_ord_map)?;
        if has_deletes {
            for level in (0..initializer.num_levels()).rev() {
                self.fix_disconnected_nodes(&disconnected_by_level[level as usize], level)?;
            }
            self.rebalance_graph(level_to_nodes)?;
        }
        Ok(())
    }

    /// `InitializedHnswGraphBuilder.copyGraphStructure`. Returns the
    /// per-level lists of nodes that need repairing, the per-level node lists
    /// (which the rebalance phase promotes from), and whether anything was
    /// dropped at all.
    #[allow(clippy::type_complexity)]
    fn copy_graph_structure<G: HnswGraphView>(
        &mut self,
        initializer: &G,
        new_ord_map: &[i32],
    ) -> Result<(Vec<Vec<i32>>, Vec<Vec<i32>>, bool)> {
        let num_levels = initializer.num_levels().max(0) as usize;
        let mut level_to_nodes: Vec<Vec<i32>> = vec![Vec::new(); num_levels];
        let mut disconnected: Vec<Vec<i32>> = vec![Vec::new(); num_levels];
        let mut has_deletes = false;
        let mut old_neighbors: Vec<i32> = Vec::new();

        // Top level first, so the first node seen at the top becomes the entry
        // node with the right level (`trySetNewEntryNode` only fires once).
        // Checked once, rather than per ordinal: every node id and every
        // neighbour id the source graph can produce is `< size`, so one bound
        // covers both. Without it a short map indexes out of bounds on a
        // neighbour -- the node lookup alone is not enough, since a node on the
        // top level can have neighbours with much larger ordinals.
        if new_ord_map.len() < initializer.size().max(0) as usize {
            return Err(Error::InvalidGraphParameter(format!(
                "ordinal map has {} entries for a {}-node source graph",
                new_ord_map.len(),
                initializer.size()
            )));
        }
        for level in (0..num_levels as i32).rev() {
            for old_ord in initializer.sorted_nodes_on_level(level)? {
                if old_ord < 0 || old_ord as usize >= new_ord_map.len() {
                    return Err(Error::InvalidGraphParameter(format!(
                        "ordinal map has no entry for source ordinal {old_ord}"
                    )));
                }
                let new_ord = new_ord_map[old_ord as usize];
                if new_ord == -1 {
                    has_deletes = true;
                    continue;
                }
                // `add_node` asserts on a negative ordinal (it would ask
                // `resize_with` for ~2^64 slots, an abort); an ordinal map is
                // a caller's array, so a wrong entry has to be an error here.
                if new_ord < 0 {
                    return Err(Error::InvalidGraphParameter(format!(
                        "ordinal map maps source ordinal {old_ord} to {new_ord}"
                    )));
                }
                self.hnsw.add_node(level, new_ord);
                level_to_nodes[level as usize].push(new_ord);
                self.hnsw.try_set_new_entry_node(new_ord, level);
                self.scorer.set_scoring_ordinal(new_ord)?;

                initializer.neighbors_into(level, old_ord, &mut old_neighbors)?;
                let old_neighbour_count = old_neighbors.len();
                let neighbors = self.hnsw.neighbors_mut(level, new_ord);
                // `NeighborArray` refuses to grow past its budget, and refuses
                // by panicking. A `.vex` can carry an upper-level node with
                // more than `M` arcs and still pass its checksum -- the reader
                // bounds arc counts at `2 * M` on every level, as Java's
                // `currentNeighborsBuffer` does -- so this has to be an error
                // here rather than a panic three frames down. Java throws
                // `IllegalStateException` for the same input.
                // ARITH: `max_size` is `nsize`/`nsize0`, i.e. `m + 1` or
                // `2 * m + 1` for an `m >= 1`, so it is at least 2.
                #[allow(clippy::arithmetic_side_effects)]
                let budget = neighbors.max_size() - 1;
                if old_neighbours_that_survive(&old_neighbors, new_ord_map) > budget {
                    return Err(Error::CorruptMeta(format!(
                        "source node {old_ord} has more neighbours on level {level} than the \
                         level's budget of {budget} allows"
                    )));
                }
                for old_neighbor in &old_neighbors {
                    // A dropped neighbour is simply not copied. Its score is
                    // never needed either -- `addOutOfOrder(_, NaN)` is Java's
                    // marker that these entries have not been scored, and
                    // `NeighborArray.sort` rescores them on demand.
                    // `new_ord_map` was length-checked against the source
                    // graph's `size()` above, and every ordinal a graph in
                    // this crate can name is `< size()`; `.get` rather than
                    // `[]` so that any other `HnswGraphView` gets an error
                    // instead of a panic.
                    let new_neighbor = *new_ord_map
                        .get(usize::try_from(*old_neighbor).unwrap_or(usize::MAX))
                        .ok_or_else(|| {
                            Error::InvalidGraphParameter(format!(
                                "ordinal map has no entry for source neighbour {old_neighbor}"
                            ))
                        })?;
                    if new_neighbor < -1 {
                        return Err(Error::InvalidGraphParameter(format!(
                            "ordinal map holds {new_neighbor}, which is neither an ordinal nor \
                             the -1 that marks a dropped vector"
                        )));
                    }
                    if new_neighbor != -1 {
                        neighbors.add_out_of_order(new_neighbor, f32::NAN);
                    }
                }
                // `M` on upper levels, `2 * M` at level 0.
                // ARITH: as above, `max_size >= 2`.
                #[allow(clippy::arithmetic_side_effects)]
                let max_conn_on_level = neighbors.max_size() - 1;
                let kept = neighbors.size();
                if (kept as f64) < old_neighbour_count as f64 * DISCONNECTED_NODE_FACTOR
                    || (kept < old_neighbour_count
                        && (kept as f64)
                            < max_conn_on_level as f64 * CUMULATIVE_DEGREE_FLOOR_FACTOR)
                {
                    disconnected[level as usize].push(new_ord);
                }
            }
        }
        Ok((disconnected, level_to_nodes, has_deletes))
    }

    /// `InitializedHnswGraphBuilder.fixDisconnectedNodes`: search out from
    /// whatever neighbours a node kept, and re-link it diversely to what that
    /// finds. A node that kept nothing has no entry point to search from, so it
    /// gets a full descent from the graph's entry node instead
    /// ([`Self::add_connections`]).
    fn fix_disconnected_nodes(&mut self, disconnected: &[i32], level: i32) -> Result<()> {
        if disconnected.is_empty() {
            return Ok(());
        }
        let beam_width = self.beam_width;
        let mut candidates = KnnCollector::unlimited(beam_width);
        let mut scratch = NeighborArray::new(beam_width, false);
        for node in disconnected {
            let node = *node;
            self.scorer.set_scoring_ordinal(node)?;
            let entry_points: Vec<i32> = self.hnsw.neighbors(level, node).nodes().to_vec();
            if !entry_points.is_empty() {
                self.searcher.search_level(
                    &mut candidates,
                    &mut self.scorer,
                    level,
                    &entry_points,
                    &self.hnsw,
                    None,
                )?;
                pop_to_scratch(&mut candidates, &mut scratch);
                self.add_diverse_neighbors(level, node, &scratch, true)?;
            } else {
                self.add_connections(node, level)?;
            }
            scratch.clear();
            candidates.clear();
        }
        Ok(())
    }

    /// `InitializedHnswGraphBuilder.addConnections`: descend from the entry
    /// node to `target_level`, beam-search there, and link diversely. Used for
    /// a node that has no neighbours left to search from, and for a node
    /// promoted to a new level by the rebalance.
    fn add_connections(&mut self, node: i32, target_level: i32) -> Result<()> {
        let beam_width = self.beam_width;
        let mut candidates = KnnCollector::unlimited(beam_width);
        let mut eps = [self.hnsw.entry_node()];
        if eps[0] == -1 {
            return Ok(());
        }
        // ARITH: `target_level` is a level of a graph being built, bounded by
        // `random_graph_level`'s ~1100 and by the source graph's level count.
        #[allow(clippy::arithmetic_side_effects)]
        let above = (target_level + 1)..self.hnsw.num_levels();
        for level in above.rev() {
            self.searcher.search_level(
                &mut candidates,
                &mut self.scorer,
                level,
                &eps,
                &self.hnsw,
                None,
            )?;
            eps[0] = candidates.pop_node();
            candidates.clear();
        }
        self.searcher.search_level(
            &mut candidates,
            &mut self.scorer,
            target_level,
            &eps,
            &self.hnsw,
            None,
        )?;
        let mut scratch = NeighborArray::new(beam_width, false);
        pop_to_scratch(&mut candidates, &mut scratch);
        self.add_diverse_neighbors(target_level, node, &scratch, true)
    }

    /// `InitializedHnswGraphBuilder.rebalanceGraph`: deletions thin the upper
    /// levels out of proportion, so nodes are promoted from the level below
    /// with probability `1/M` until each level holds the `size * M^-level`
    /// the HNSW model expects.
    ///
    /// **Java draws from `new SplittableRandom()` here -- unseeded** -- so a
    /// Lucene merge that hits this branch is not reproducible, even from
    /// identical inputs. This port seeds it, which makes the merged graph a
    /// function of its inputs (and so testable at all). Nothing about the
    /// format or the search depends on which nodes get promoted, only on the
    /// distribution.
    fn rebalance_graph(&mut self, mut level_to_nodes: Vec<Vec<i32>>) -> Result<()> {
        // `maxNodesAtLevel` is `size * (1/M)^level`, which for `M == 1` is
        // `size` at *every* level: Java's `for (int level = 1; ; level++)`
        // never breaks, promotes every node to level after level, and dies of
        // memory exhaustion. `M == 1` is a legal configuration
        // (`HnswGraphBuilder` and `Lucene99HnswVectorsFormat` both accept it)
        // and a merge takes `M` from the base graph's `.vem`, so this is
        // reachable from a file. There is nothing to rebalance in that case
        // anyway: the model wants every node on every level, which is not a
        // finite graph.
        if self.m <= 1 {
            return Ok(());
        }
        let mut random = SplittableRandom::new(self.seed);
        let size = self.hnsw.size();
        let inv_max_conn = 1.0 / self.m as f64;
        let mut level = 1usize;
        loop {
            let max_nodes_at_level = (size as f64 * inv_max_conn.powi(level as i32)) as i32;
            if max_nodes_at_level <= 0 {
                break;
            }
            let mut current_nodes_at_level = 0;
            // ARITH: `level` climbs by one per iteration and the loop stops
            // as soon as `size * (1/M)^level` rounds to 0, which for `M >= 2`
            // (guarded above) happens by level 31 -- `size` is an `i32`.
            #[allow(clippy::arithmetic_side_effects)]
            if level >= level_to_nodes.len() {
                level_to_nodes.resize(level + 1, Vec::new());
            } else {
                current_nodes_at_level = level_to_nodes[level].len() as i32;
            }
            if current_nodes_at_level >= max_nodes_at_level {
                // ARITH: as above -- at most 31 iterations for `M >= 2`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    level += 1;
                }
                continue;
            }
            // ARITH: the loop starts at `level == 1`.
            #[allow(clippy::arithmetic_side_effects)]
            let below: Vec<i32> = level_to_nodes[level - 1].clone();
            for node in below {
                if current_nodes_at_level >= max_nodes_at_level {
                    break;
                }
                if random.next_double() < inv_max_conn
                    && !self.hnsw.node_exists_at_level(level as i32, node)
                {
                    self.scorer.set_scoring_ordinal(node)?;
                    self.hnsw.add_node(level as i32, node);
                    if current_nodes_at_level == 0 {
                        // ARITH: `num_levels()` is at least 1 (see
                        // `OnHeapHnswGraph::num_levels`).
                        #[allow(clippy::arithmetic_side_effects)]
                        let expect_old = self.hnsw.num_levels() - 1;
                        if level as i32 > expect_old {
                            self.hnsw
                                .try_promote_new_entry_node(node, level as i32, expect_old);
                        }
                    } else {
                        self.add_connections(node, level as i32)?;
                    }
                    level_to_nodes[level].push(node);
                    // ARITH: bounded by `max_nodes_at_level`, itself at most
                    // `size`.
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        current_nodes_at_level += 1;
                    }
                }
            }
            // ARITH: as above -- at most 31 iterations for `M >= 2`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                level += 1;
            }
        }
        Ok(())
    }
}

/// Port of `org.apache.lucene.util.hnsw.MergingHnswGraphBuilder.fromGraphs` +
/// `build` -- the whole incremental merge, as one call.
///
/// `graphs[0]` is copied wholesale into the merged ordinal space
/// ([`HnswGraphBuilder::init_graph`]); every later graph is *folded in*: the
/// nodes that best cover it ([`compute_join_set`]) are inserted with an
/// ordinary beam search, and each remaining node is inserted with entry points
/// taken from its own neighbours in the small graph plus those neighbours'
/// neighbours in the big one -- which is a much cheaper search than starting at
/// the entry node. Finally, any ordinal no source graph covered
/// (`initialized_nodes`'s clear bits) is added from scratch.
///
/// `ord_maps[i][old_ord]` is the merged ordinal of graph `i`'s ordinal
/// `old_ord`, or `-1` if that vector is not in the merged segment.
/// `initialized_nodes` is `None` when every source segment contributed a graph,
/// meaning nothing is left over -- Java's `graphReaders.size() == numReaders`
/// test.
///
/// Java splits this across a builder class whose `build` reads fields captured
/// at construction; here it is one function, because the graphs it borrows and
/// the graph it produces would otherwise make the builder self-referential for
/// no gain. The RNG is shared across the two phases instead of being reset
/// between them, which is equivalent: the copy phase draws from it only in the
/// deletions branch, and that branch has its own RNG in both languages.
pub fn merge_graphs<G: HnswGraphView, S: UpdateableVectorScorer>(
    scorer: S,
    beam_width: i32,
    seed: u64,
    graphs: &[&G],
    ord_maps: &[Vec<i32>],
    total_vectors: i32,
    initialized_nodes: Option<&FixedBitSet>,
) -> Result<OnHeapHnswGraph> {
    if graphs.is_empty() || graphs.len() != ord_maps.len() {
        return Err(Error::InvalidGraphParameter(format!(
            "merge_graphs needs one ordinal map per graph, got {} graphs and {} maps",
            graphs.len(),
            ord_maps.len()
        )));
    }
    if graphs.iter().any(|g| g.size() == 0) {
        return Err(Error::InvalidGraphParameter(
            "merge_graphs: a source graph is empty".to_string(),
        ));
    }
    // Checked here, for *every* graph, rather than where each is used:
    // `update_graph` reaches `compute_join_set` first, and that sizes three
    // arrays from `gs.size()` -- a number a corrupt `.vem` supplies, so a
    // 2-billion-node claim is an 8.6 GB `vec![0i32; size]` and an **abort**
    // before any map is consulted. One ordinal map entry per source node is
    // what `copy_graph_structure` already requires of `graphs[0]`; requiring
    // it of all of them bounds the allocation by memory the caller has
    // already committed.
    for (i, (graph, map)) in graphs.iter().zip(ord_maps).enumerate() {
        if (map.len() as i64) < i64::from(graph.size()) {
            return Err(Error::InvalidGraphParameter(format!(
                "source {i}: ordinal map has {} entries for a {}-node graph",
                map.len(),
                graph.size()
            )));
        }
    }
    let hnsw = OnHeapHnswGraph::with_size(graphs[0].max_conn(), total_vectors);
    let mut builder = HnswGraphBuilder::with_graph(scorer, beam_width, seed, hnsw)?;
    builder.initialize_from_graph(graphs[0], &ord_maps[0])?;
    for i in 1..graphs.len() {
        builder.update_graph(graphs[i], &ord_maps[i])?;
    }
    if let Some(initialized) = initialized_nodes {
        // `initialized` is a caller's bitset and `total_vectors` a separate
        // caller argument: exactly the pair `docs/arithmetic-gate.md` names
        // ("never index a `FixedBitSet` with an index bounded against anything
        // other than that bitset's own `len()`"). A short one reads ghost bits
        // past `num_bits` in a release build -- silently deciding that a node
        // needs no insertion -- and panics 64 ordinals later.
        if (initialized.len() as i64) < i64::from(total_vectors) {
            return Err(Error::InvalidGraphParameter(format!(
                "the initialized-node set covers {} ordinals, short of the {total_vectors} \
                 vectors being merged",
                initialized.len()
            )));
        }
        for node in 0..total_vectors {
            if !initialized.get(node as usize) {
                builder.add_graph_node(node)?;
            }
        }
    }
    Ok(builder.into_graph())
}

impl<S: UpdateableVectorScorer> HnswGraphBuilder<S> {
    /// The graph this builder has been building, by value.
    pub fn into_graph(self) -> OnHeapHnswGraph {
        self.hnsw
    }

    /// `MergingHnswGraphBuilder.updateGraph`: fold one smaller graph into the
    /// one being built.
    fn update_graph<G: HnswGraphView>(&mut self, gs: &G, ord_map: &[i32]) -> Result<()> {
        let size = gs.size();
        let join = compute_join_set(gs)?;
        // Sorted for stability -- Java sorts the `IntHashSet`'s array for the
        // same reason (its iteration order is a hash order).
        let mut join_nodes: Vec<i32> = join.iter().copied().collect();
        join_nodes.sort_unstable();
        for node in &join_nodes {
            self.add_graph_node(map_ord(ord_map, *node)?)?;
        }

        let mut neighbors: Vec<i32> = Vec::new();
        let mut friends: Vec<i32> = Vec::new();
        let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
        let mut eps: Vec<i32> = Vec::new();
        for u in 0..size {
            if join.contains(&u) {
                continue;
            }
            seen.clear();
            eps.clear();
            gs.neighbors_into(0, u, &mut neighbors)?;
            for &v in &neighbors {
                // `v < u` means v is already in the big graph (we insert in
                // ascending order); `join.contains(v)` means it went in with
                // the join set. Either way its neighbourhood in the big graph
                // is a good guess for where `u` belongs.
                if v < u || join.contains(&v) {
                    let new_v = map_ord(ord_map, v)?;
                    if seen.insert(new_v) {
                        eps.push(new_v);
                    }
                    // `new_v` must already be in the target graph: it is
                    // either a join-set node (inserted first) or a node with a
                    // smaller ordinal (inserted earlier in this ascending
                    // loop). An ordinal map that maps two source ordinals onto
                    // one merged ordinal breaks that, and the index would
                    // otherwise panic rather than say so.
                    if !self.hnsw.node_exists_at_level(0, new_v) {
                        return Err(Error::InvalidGraphParameter(format!(
                            "ordinal map is not injective: merged ordinal {new_v} is referenced \
                             before it was inserted"
                        )));
                    }
                    self.hnsw.neighbors_into(0, new_v, &mut friends)?;
                    for friend in &friends {
                        if seen.insert(*friend) {
                            eps.push(*friend);
                        }
                    }
                }
            }
            // Java collects these into an `IntHashSet` and hands
            // `eps.toArray()` to the search, i.e. hppc's hash order. Insertion
            // order is used here instead: it is deterministic, which hppc's
            // is not across implementations, and the set is a *seed* for a beam
            // search, so its order only breaks score ties.
            self.add_graph_node_with_entry_points(map_ord(ord_map, u)?, &eps)?;
        }
        Ok(())
    }
}

/// How many of `old_neighbors` survive `new_ord_map` -- the count that has to
/// fit in the target level's `NeighborArray`.
fn old_neighbours_that_survive(old_neighbors: &[i32], new_ord_map: &[i32]) -> usize {
    old_neighbors
        .iter()
        .filter(|n| new_ord_map.get(**n as usize).is_some_and(|m| *m != -1))
        .count()
}

fn map_ord(ord_map: &[i32], old: i32) -> Result<i32> {
    match ord_map.get(old as usize) {
        Some(&new) if new >= 0 => Ok(new),
        _ => Err(Error::InvalidGraphParameter(format!(
            "ordinal map has no merged ordinal for source ordinal {old}"
        ))),
    }
}

/// `HnswGraphBuilder.popToScratch`: drains the beam (worst first) into an
/// ascending-score `NeighborArray`.
fn pop_to_scratch(candidates: &mut KnnCollector, scratch: &mut NeighborArray) {
    scratch.clear();
    let count = candidates.size();
    for _ in 0..count {
        let max_similarity = candidates.minimum_score();
        scratch.add_in_order(candidates.pop_node(), max_similarity);
    }
}

/// Convenience: is a graph worth building for `num_nodes` vectors at threshold
/// `k`? Port of `Lucene99HnswVectorsWriter.shouldCreateGraph`.
pub fn should_create_graph(k: i32, num_nodes: i32) -> bool {
    if k <= 0 {
        return true;
    }
    let expected = expected_visited_nodes(k, num_nodes);
    num_nodes > expected && expected > 0
}

#[cfg(test)]
mod tests {
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
    #[should_panic(expected = "M (max connections) must be in 1..=")]
    fn on_heap_graph_rejects_a_non_positive_m() {
        OnHeapHnswGraph::new(0);
    }

    /// `2 * m + 1` is an `i32` multiply: an absurd `M` used to overflow it
    /// (a panic in debug, a negative `nsize0` in release, and from there a
    /// `NeighborArray` whose `max_size` is nonsense).
    #[test]
    #[should_panic(expected = "M (max connections) must be in 1..=")]
    fn on_heap_graph_rejects_an_m_that_would_overflow_the_level_zero_budget() {
        OnHeapHnswGraph::new(i32::MAX);
    }

    /// `add_node` takes ordinals as `i32` and indexes with them: a negative
    /// one used to sign-extend into `resize_with(~2^64)` -- an allocation
    /// abort, not a panic -- and a negative level pushed neighbour arrays
    /// until memory ran out.
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
    /// discriminates -- recall would not (see `docs/sweep/m2/c5-vectors.md`'s
    /// Verification section).
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
    /// `visited`, which is `FixedBitSet::set(words[i >> 6])` behind a
    /// `debug_assert` -- a **ghost bit** in a release build (a silently wrong
    /// "already visited") and an index panic 64 ordinals further out.
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
}
