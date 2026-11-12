//! The search package's `KnnCollector`s beyond `TopKnnCollector`:
//! `VectorSimilarityCollector` (a similarity threshold instead of a top-k)
//! with its Lucene 10.4 form `ExplicitVectorSimilarityCollector`,
//! `HnswQueueSaturationCollector` (`PatienceKnnVectorQuery`'s early exit),
//! `TimeLimitingKnnCollectorManager`'s collector, and
//! `MultiLeafKnnCollector` with the `FloatHeap`/`BlockingFloatHeap` it shares
//! across leaves. Each implements [`lucene_codecs::hnsw::KnnCollect`], the
//! interface the graph walk and the exhaustive scan drive.

use std::sync::Mutex;
use std::time::Instant;

use lucene_codecs::hnsw::{KnnCollect, KnnCollector};

// ---------------------------------------------------------------------------
// VectorSimilarityCollector
// ---------------------------------------------------------------------------

/// `AbstractVectorSimilarityQuery.DECAY_MAX_QUALITY`.
pub const DECAY_MAX_QUALITY: f32 = 1.0;

/// `VectorSimilarityCollector`: every visited vector at or above
/// `result_similarity`; the walk's competitive bound starts just above
/// `-inf` and decays towards the similarities it passes over.
#[derive(Debug, Clone)]
pub struct VectorSimilarityCollector {
    result_similarity: f32,
    decay: f32,
    visited_count: u64,
    visit_limit: u64,
    hits: Vec<(i32, f32)>,
    min_competitive: f32,
}

impl VectorSimilarityCollector {
    pub fn new(result_similarity: f32, decay: f32, visit_limit: u64) -> Self {
        Self {
            result_similarity,
            decay,
            visited_count: 0,
            visit_limit,
            hits: Vec::new(),
            min_competitive: f32::NEG_INFINITY.next_up(),
        }
    }

    /// `topDocs()`: the collected `(node, similarity)`s in collection order,
    /// and whether the walk early-terminated.
    pub fn into_hits(self) -> (Vec<(i32, f32)>, bool) {
        let early = self.early_terminated();
        (self.hits, early)
    }

    /// `numCollected()`.
    pub fn num_collected(&self) -> usize {
        self.hits.len()
    }
}

impl KnnCollect for VectorSimilarityCollector {
    /// `super(1, ...)`: `k()` is 1, which is not the number collected.
    fn k(&self) -> usize {
        1
    }
    fn early_terminated(&self) -> bool {
        self.visited_count >= self.visit_limit
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.visited_count = self.visited_count.saturating_add(count as u64);
    }
    fn visited_count(&self) -> u64 {
        self.visited_count
    }
    fn visit_limit(&self) -> u64 {
        self.visit_limit
    }
    fn collect(&mut self, node: i32, similarity: f32) -> bool {
        if similarity >= self.result_similarity {
            self.hits.push((node, similarity));
        } else if self.decay < DECAY_MAX_QUALITY {
            self.min_competitive = (f64::from(similarity)
                + (f64::from(self.min_competitive) - f64::from(similarity)) * f64::from(self.decay))
                as f32;
            return true;
        }
        false
    }
    fn min_competitive_similarity(&self) -> f32 {
        self.min_competitive
    }
}

// ---------------------------------------------------------------------------
// ExplicitVectorSimilarityCollector
// ---------------------------------------------------------------------------

/// `ExplicitVectorSimilarityCollector` (Lucene 10.5.2, #16654; deprecated in
/// Java): Lucene 10.4's similarity collector, behind
/// `[Byte|Float]VectorSimilarityQuery.Explicit`. Every visited vector at or
/// above `result_similarity` is collected; the walk goes on while a better
/// node is available or the best candidate is not below
/// `traversal_similarity` (`minCompetitiveSimilarity` is the smaller of it
/// and the best similarity seen).
#[derive(Debug, Clone)]
pub struct ExplicitVectorSimilarityCollector {
    traversal_similarity: f32,
    result_similarity: f32,
    max_similarity: f32,
    visited_count: u64,
    visit_limit: u64,
    hits: Vec<(i32, f32)>,
}

impl ExplicitVectorSimilarityCollector {
    /// The constructor: `traversal_similarity > result_similarity` is Java's
    /// `IllegalArgumentException`.
    pub fn new(
        traversal_similarity: f32,
        result_similarity: f32,
        visit_limit: u64,
    ) -> crate::Result<Self> {
        if traversal_similarity > result_similarity {
            return Err(crate::Error::InvalidKnnQuery(
                "traversalSimilarity should be <= resultSimilarity".to_string(),
            ));
        }
        Ok(Self {
            traversal_similarity,
            result_similarity,
            max_similarity: f32::NEG_INFINITY,
            visited_count: 0,
            visit_limit,
            hits: Vec::new(),
        })
    }

    /// `topDocs()`: the collected `(node, similarity)`s in collection order,
    /// and whether the walk early-terminated.
    pub fn into_hits(self) -> (Vec<(i32, f32)>, bool) {
        let early = self.early_terminated();
        (self.hits, early)
    }

    /// `numCollected()`.
    pub fn num_collected(&self) -> usize {
        self.hits.len()
    }
}

impl KnnCollect for ExplicitVectorSimilarityCollector {
    /// `super(1, ...)`.
    fn k(&self) -> usize {
        1
    }
    fn early_terminated(&self) -> bool {
        self.visited_count >= self.visit_limit
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.visited_count = self.visited_count.saturating_add(count as u64);
    }
    fn visited_count(&self) -> u64 {
        self.visited_count
    }
    fn visit_limit(&self) -> u64 {
        self.visit_limit
    }
    /// `collect`: always `true`.
    fn collect(&mut self, node: i32, similarity: f32) -> bool {
        self.max_similarity = crate::values_source::java_float_max(self.max_similarity, similarity);
        if similarity >= self.result_similarity {
            self.hits.push((node, similarity));
        }
        true
    }
    /// `Math.min(traversalSimilarity, maxSimilarity)`: NaN wins, and `-0.0`
    /// beats `+0.0`.
    fn min_competitive_similarity(&self) -> f32 {
        let (a, b) = (self.traversal_similarity, self.max_similarity);
        if a.is_nan() || b.is_nan() {
            f32::NAN
        } else if a == 0.0 && b == 0.0 {
            if a.is_sign_negative() {
                a
            } else {
                b
            }
        } else if a <= b {
            a
        } else {
            b
        }
    }
}

// ---------------------------------------------------------------------------
// HnswQueueSaturationCollector
// ---------------------------------------------------------------------------

/// `HnswQueueSaturationCollector`: stops the walk once the delegate's queue
/// has stopped changing for `patience` consecutive candidates (its
/// saturation, `min(current, previous) / current`, at or above the
/// threshold).
#[derive(Debug, Clone)]
pub struct HnswQueueSaturationCollector<C> {
    delegate: C,
    saturation_threshold: f64,
    patience: usize,
    patience_finished: bool,
    count_saturated: usize,
    previous_queue_size: usize,
    current_queue_size: usize,
}

impl<C: KnnCollect> HnswQueueSaturationCollector<C> {
    pub fn new(delegate: C, saturation_threshold: f64, patience: usize) -> Self {
        Self {
            delegate,
            saturation_threshold,
            patience,
            patience_finished: false,
            count_saturated: 0,
            previous_queue_size: 0,
            current_queue_size: 0,
        }
    }

    /// `nextCandidate()`.
    pub fn next_candidate(&mut self) {
        let saturation = self.current_queue_size.min(self.previous_queue_size) as f64
            / self.current_queue_size as f64;
        self.previous_queue_size = self.current_queue_size;
        if saturation >= self.saturation_threshold {
            self.count_saturated += 1;
        } else {
            self.count_saturated = 0;
        }
        if self.count_saturated > self.patience {
            self.patience_finished = true;
        }
    }

    /// `topDocs()`'s relation: patience running out is a complete answer
    /// (`EQUAL_TO`); only the delegate's own early termination is partial.
    pub fn partial(&self) -> bool {
        self.delegate.early_terminated()
    }

    pub fn into_inner(self) -> C {
        self.delegate
    }
}

impl<C: KnnCollect> KnnCollect for HnswQueueSaturationCollector<C> {
    fn k(&self) -> usize {
        self.delegate.k()
    }
    fn early_terminated(&self) -> bool {
        self.delegate.early_terminated() || self.patience_finished
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.delegate.inc_visited_count(count)
    }
    fn visited_count(&self) -> u64 {
        self.delegate.visited_count()
    }
    fn visit_limit(&self) -> u64 {
        self.delegate.visit_limit()
    }
    fn collect(&mut self, node: i32, similarity: f32) -> bool {
        let collected = self.delegate.collect(node, similarity);
        if collected {
            self.current_queue_size += 1;
        }
        collected
    }
    fn min_competitive_similarity(&self) -> f32 {
        self.delegate.min_competitive_similarity()
    }
    /// `KnnSearchStrategy.Patience.nextVectorsBlock`.
    fn next_vectors_block(&mut self) {
        self.next_candidate();
        self.delegate.next_vectors_block();
    }
}

// ---------------------------------------------------------------------------
// TimeLimitingKnnCollectorManager
// ---------------------------------------------------------------------------

/// `TimeLimitingKnnCollectorManager.TimeLimitingKnnCollector`: the delegate,
/// early-terminated once the query's deadline passes (`QueryTimeout`).
#[derive(Debug, Clone)]
pub struct TimeLimitingKnnCollector<C> {
    delegate: C,
    deadline: Instant,
}

impl<C: KnnCollect> TimeLimitingKnnCollector<C> {
    pub fn new(delegate: C, deadline: Instant) -> Self {
        Self { delegate, deadline }
    }

    /// `queryTimeout.shouldExit()`.
    pub fn timed_out(&self) -> bool {
        Instant::now() >= self.deadline
    }

    pub fn into_inner(self) -> C {
        self.delegate
    }
}

impl<C: KnnCollect> KnnCollect for TimeLimitingKnnCollector<C> {
    fn k(&self) -> usize {
        self.delegate.k()
    }
    fn early_terminated(&self) -> bool {
        self.timed_out() || self.delegate.early_terminated()
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.delegate.inc_visited_count(count)
    }
    fn visited_count(&self) -> u64 {
        self.delegate.visited_count()
    }
    fn visit_limit(&self) -> u64 {
        self.delegate.visit_limit()
    }
    fn collect(&mut self, node: i32, similarity: f32) -> bool {
        self.delegate.collect(node, similarity)
    }
    fn min_competitive_similarity(&self) -> f32 {
        self.delegate.min_competitive_similarity()
    }
    fn next_vectors_block(&mut self) {
        self.delegate.next_vectors_block()
    }
}

// ---------------------------------------------------------------------------
// FilteredWalkPlanner
// ---------------------------------------------------------------------------

/// A filtered graph walk that gives up for an exact scan of the filter's
/// vectors once the walk is predicted to cost more than the scan.
///
/// An unfiltered walk visits about `unfiltered_visits` nodes; a filtered one
/// about that many divided by the share of the nodes it meets that the
/// filter passes. That share is measured over the first `probe` visits of
/// the walk on level 0 (or until it has collected `k`), and never counted below the
/// filter's share of the whole graph (`global_share`): a walk starting far
/// from the filter's documents still reaches them at the global rate. A
/// filter correlated with the query's neighbourhood -- a tenant queried from
/// inside its own region -- keeps walking; one the neighbourhood rarely
/// passes switches to the scan after the probe. A walk that goes on gives up
/// all the same past twice the scan's cost (`scan_visits`), where Lucene's
/// own limit is four times it.
#[derive(Debug, Clone)]
pub struct FilteredWalkPlanner<C> {
    delegate: C,
    unfiltered_visits: f64,
    global_share: f64,
    scan_visits: u64,
    probe: u64,
    accepted: u64,
    /// Visits before level 0's walk: the upper levels' descent, which
    /// collects nothing and so is left out of the probe.
    descent: u64,
    /// Whether level 0's walk has begun (a collect or an expanded
    /// candidate).
    level0: bool,
    decided: bool,
    gave_up: bool,
}

impl<C: KnnCollect> FilteredWalkPlanner<C> {
    pub fn new(
        delegate: C,
        unfiltered_visits: f64,
        global_share: f64,
        scan_visits: u64,
        probe: u64,
    ) -> Self {
        Self {
            delegate,
            unfiltered_visits,
            global_share,
            scan_visits,
            probe,
            accepted: 0,
            descent: 0,
            level0: false,
            decided: false,
            gave_up: false,
        }
    }

    /// Whether the walk stopped for the exact scan.
    pub fn gave_up(&self) -> bool {
        self.gave_up
    }

    pub fn into_inner(self) -> C {
        self.delegate
    }
}

impl<C: KnnCollect> KnnCollect for FilteredWalkPlanner<C> {
    fn k(&self) -> usize {
        self.delegate.k()
    }
    fn early_terminated(&self) -> bool {
        self.gave_up || self.delegate.early_terminated()
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.delegate.inc_visited_count(count);
        let visited = self.delegate.visited_count();
        if self.decided {
            if visited > self.scan_visits.saturating_mul(2) {
                self.gave_up = true;
            }
            return;
        }
        if !self.level0 {
            self.descent = visited;
            return;
        }
        let walked = visited.saturating_sub(self.descent);
        if walked >= self.probe || self.accepted >= self.delegate.k() as u64 {
            self.decided = true;
            // Until the collector is full every accepted node it meets is
            // collected, so `accepted` counts them.
            let share = (self.accepted as f64 / walked.max(1) as f64).max(self.global_share);
            let predicted = self.unfiltered_visits / share;
            if predicted - walked as f64 > self.scan_visits as f64 {
                self.gave_up = true;
            }
        }
    }
    fn visited_count(&self) -> u64 {
        self.delegate.visited_count()
    }
    fn visit_limit(&self) -> u64 {
        self.delegate.visit_limit()
    }
    fn collect(&mut self, node: i32, similarity: f32) -> bool {
        self.level0 = true;
        if !self.decided {
            self.accepted = self.accepted.saturating_add(1);
        }
        self.delegate.collect(node, similarity)
    }
    fn min_competitive_similarity(&self) -> f32 {
        self.delegate.min_competitive_similarity()
    }
    fn next_vectors_block(&mut self) {
        self.level0 = true;
        self.delegate.next_vectors_block()
    }
}

// ---------------------------------------------------------------------------
// AdaptiveBeamCollector
// ---------------------------------------------------------------------------

/// A distance-adaptive stop for a graph walk: the walk goes on while its
/// best unexplored candidate is within `1 + slack` times the distance of
/// the `hits`-th best node collected so far, instead of while it beats the
/// worst of the collector's `k` (Al-Jazzazi et al., "Distance Adaptive Beam
/// Search for Provably Accurate Graph-Based Nearest Neighbor Search", the
/// rule behind Vespa's `exploration-slack`). The walk then explores as far
/// as the query's neighbourhood needs rather than a fixed beam: at the same
/// number of visited nodes it finds more of the true `hits` nearest.
///
/// Distances come from the similarity's score: `1 - score` for cosine and
/// dot product (`(1 + cos) / 2`), `sqrt(1 / score - 1)` for Euclidean
/// (`1 / (1 + d^2)`). Maximum inner product has no such distance, and its
/// walk keeps the delegate's bound.
#[derive(Debug, Clone)]
pub struct AdaptiveBeamCollector<C> {
    delegate: C,
    /// The best `hits` scores collected, worst first once full.
    best: std::collections::BinaryHeap<std::cmp::Reverse<OrderedScore>>,
    hits: usize,
    slack: f32,
    euclidean: bool,
}

/// An `f32` ordered by `total_cmp`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedScore(f32);

impl Eq for OrderedScore {}

impl PartialOrd for OrderedScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl<C: KnnCollect> AdaptiveBeamCollector<C> {
    /// `None` for a similarity without a distance to scale.
    pub fn new(
        delegate: C,
        hits: usize,
        slack: f32,
        similarity: lucene_codecs::field_infos::VectorSimilarityFunction,
    ) -> Option<Self> {
        use lucene_codecs::field_infos::VectorSimilarityFunction as F;
        let euclidean = match similarity {
            F::Euclidean => true,
            F::Cosine | F::DotProduct => false,
            F::MaximumInnerProduct => return None,
        };
        Some(Self {
            delegate,
            best: std::collections::BinaryHeap::with_capacity(hits.saturating_add(1)),
            hits: hits.max(1),
            slack: slack.max(0.0),
            euclidean,
        })
    }

    pub fn into_inner(self) -> C {
        self.delegate
    }
}

impl<C: KnnCollect> KnnCollect for AdaptiveBeamCollector<C> {
    fn k(&self) -> usize {
        self.delegate.k()
    }
    fn early_terminated(&self) -> bool {
        self.delegate.early_terminated()
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.delegate.inc_visited_count(count)
    }
    fn visited_count(&self) -> u64 {
        self.delegate.visited_count()
    }
    fn visit_limit(&self) -> u64 {
        self.delegate.visit_limit()
    }
    fn collect(&mut self, node: i32, similarity: f32) -> bool {
        self.delegate.collect(node, similarity);
        let score = OrderedScore(similarity);
        if self.best.len() < self.hits {
            self.best.push(std::cmp::Reverse(score));
            return self.best.len() == self.hits;
        }
        match self.best.peek() {
            Some(std::cmp::Reverse(worst)) if score > *worst => {
                self.best.pop();
                self.best.push(std::cmp::Reverse(score));
                true
            }
            _ => false,
        }
    }
    fn min_competitive_similarity(&self) -> f32 {
        // No bound until the delegate is full too, so the walk still finds
        // `k` hits when `hits < k`.
        if self.delegate.min_competitive_similarity() == f32::NEG_INFINITY {
            return f32::NEG_INFINITY;
        }
        let Some(std::cmp::Reverse(OrderedScore(kth))) =
            self.best.peek().filter(|_| self.best.len() >= self.hits)
        else {
            return f32::NEG_INFINITY;
        };
        self.bound(*kth)
    }
    fn next_vectors_block(&mut self) {
        self.delegate.next_vectors_block()
    }
}

impl<C> AdaptiveBeamCollector<C> {
    /// The least score within `1 + slack` times the distance of `kth`.
    fn bound(&self, kth: f32) -> f32 {
        let scale = 1.0 + self.slack;
        if self.euclidean {
            if kth <= 0.0 {
                return f32::NEG_INFINITY;
            }
            let d = (1.0 / kth - 1.0).max(0.0).sqrt() * scale;
            1.0 / (1.0 + d * d)
        } else {
            1.0 - scale * (1.0 - kth)
        }
    }
}

// ---------------------------------------------------------------------------
// FloatHeap, BlockingFloatHeap, MultiLeafKnnCollector
// ---------------------------------------------------------------------------

/// `FloatHeap`: a bounded min-heap of floats, 1-based as Java's.
#[derive(Debug, Clone)]
pub struct FloatHeap {
    max_size: usize,
    heap: Vec<f32>,
}

impl FloatHeap {
    pub fn new(max_size: usize) -> Self {
        let mut heap = Vec::with_capacity(max_size + 1);
        heap.push(0.0);
        Self { max_size, heap }
    }

    /// `offer(value)`: whether the heap changed.
    pub fn offer(&mut self, value: f32) -> bool {
        if self.size() >= self.max_size {
            if value < self.heap[1] {
                return false;
            }
            self.update_top(value);
            return true;
        }
        self.push(value);
        true
    }

    /// `poll()`: the smallest, removed; `None` when empty (Java throws).
    pub fn poll(&mut self) -> Option<f32> {
        if self.size() == 0 {
            return None;
        }
        let result = self.heap[1];
        let last = self.heap.pop().unwrap_or(0.0);
        if self.size() > 0 {
            self.heap[1] = last;
            self.down_heap(1);
        }
        Some(result)
    }

    /// `peek()`: the smallest; `0` when empty, the array's unset slot.
    pub fn peek(&self) -> f32 {
        self.heap.get(1).copied().unwrap_or(0.0)
    }

    pub fn size(&self) -> usize {
        self.heap.len() - 1
    }

    /// `getHeap()`: the members in heap order.
    pub fn get_heap(&self) -> &[f32] {
        &self.heap[1..]
    }

    pub fn clear(&mut self) {
        self.heap.truncate(1);
    }

    fn push(&mut self, v: f32) {
        self.heap.push(v);
        let n = self.size();
        self.up_heap(n);
    }

    fn update_top(&mut self, v: f32) {
        self.heap[1] = v;
        self.down_heap(1);
    }

    fn down_heap(&mut self, mut i: usize) {
        let size = self.size();
        let value = self.heap[i];
        let mut j = i << 1;
        let mut k = j + 1;
        if k <= size && self.heap[k] < self.heap[j] {
            j = k;
        }
        while j <= size && self.heap[j] < value {
            self.heap[i] = self.heap[j];
            i = j;
            j = i << 1;
            k = j + 1;
            if k <= size && self.heap[k] < self.heap[j] {
                j = k;
            }
        }
        self.heap[i] = value;
    }

    fn up_heap(&mut self, orig: usize) {
        let mut i = orig;
        let value = self.heap[i];
        let mut j = i >> 1;
        while j > 0 && value < self.heap[j] {
            self.heap[i] = self.heap[j];
            i = j;
            j >>= 1;
        }
        self.heap[i] = value;
    }
}

/// `BlockingFloatHeap`: a [`FloatHeap`] shared by every leaf's collector,
/// guarded by a lock. Its `offer` keeps values at or above the smallest
/// (`>=`, where `FloatHeap.offer` keeps them above).
#[derive(Debug)]
pub struct BlockingFloatHeap {
    inner: Mutex<FloatHeap>,
}

impl BlockingFloatHeap {
    pub fn new(max_size: usize) -> Self {
        Self {
            inner: Mutex::new(FloatHeap::new(max_size)),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, FloatHeap> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `offer(value)`: the smallest kept afterwards.
    pub fn offer(&self, value: f32) -> f32 {
        let mut h = self.lock();
        if h.size() < h.max_size {
            h.push(value);
        } else if value >= h.heap[1] {
            h.update_top(value);
        }
        h.heap[1]
    }

    /// `offer(values, len)`: `values` ascending, offered largest first until
    /// one is not competitive; the smallest kept afterwards.
    pub fn offer_sorted(&self, values: &[f32]) -> f32 {
        let mut h = self.lock();
        for &v in values.iter().rev() {
            if h.size() < h.max_size {
                h.push(v);
            } else if v >= h.heap[1] {
                h.update_top(v);
            } else {
                break;
            }
        }
        h.peek()
    }

    pub fn peek(&self) -> f32 {
        self.lock().peek()
    }

    pub fn poll(&self) -> Option<f32> {
        self.lock().poll()
    }

    pub fn size(&self) -> usize {
        self.lock().size()
    }
}

/// `MultiLeafKnnCollector`: a leaf's [`KnnCollector`] that also shares its
/// similarities with every other leaf's through a [`BlockingFloatHeap`], so
/// a leaf whose results cannot make the global top `k` stops early.
#[derive(Debug)]
pub struct MultiLeafKnnCollector<'g> {
    global: &'g BlockingFloatHeap,
    non_competitive: FloatHeap,
    updates: FloatHeap,
    scratch: Vec<f32>,
    interval: u64,
    k_results_collected: bool,
    cached_global_min_sim: f32,
    sub: KnnCollector,
}

impl<'g> MultiLeafKnnCollector<'g> {
    /// `DEFAULT_GREEDINESS`.
    pub const DEFAULT_GREEDINESS: f32 = 0.9;
    /// `DEFAULT_INTERVAL`.
    pub const DEFAULT_INTERVAL: u64 = 0xff;

    pub fn new(k: usize, global: &'g BlockingFloatHeap, sub: KnnCollector) -> crate::Result<Self> {
        Self::with_params(
            k,
            Self::DEFAULT_GREEDINESS,
            Self::DEFAULT_INTERVAL,
            global,
            sub,
        )
    }

    pub fn with_params(
        k: usize,
        greediness: f32,
        interval: u64,
        global: &'g BlockingFloatHeap,
        sub: KnnCollector,
    ) -> crate::Result<Self> {
        if !(0.0..=1.0).contains(&greediness) {
            return Err(crate::Error::InvalidKnnQuery(
                "greediness must be in [0,1]".into(),
            ));
        }
        if interval == 0 {
            return Err(crate::Error::InvalidKnnQuery(
                "interval must be positive".into(),
            ));
        }
        // `Math.max(1, Math.round((1 - greediness) * k))`.
        let nc = (((1.0 - greediness) * k as f32).round() as i64).max(1) as usize;
        Ok(Self {
            global,
            non_competitive: FloatHeap::new(nc),
            updates: FloatHeap::new(k),
            scratch: vec![0.0; k],
            interval,
            k_results_collected: false,
            cached_global_min_sim: f32::NEG_INFINITY,
            sub,
        })
    }

    pub fn into_inner(self) -> KnnCollector {
        self.sub
    }
}

impl KnnCollect for MultiLeafKnnCollector<'_> {
    fn k(&self) -> usize {
        self.sub.k()
    }
    fn early_terminated(&self) -> bool {
        self.sub.early_terminated()
    }
    fn inc_visited_count(&mut self, count: usize) {
        self.sub.inc_visited_count(count)
    }
    fn visited_count(&self) -> u64 {
        self.sub.visited_count()
    }
    fn visit_limit(&self) -> u64 {
        self.sub.visit_limit()
    }
    fn collect(&mut self, node: i32, similarity: f32) -> bool {
        let local = self.sub.collect(node, similarity);
        let first_k = !self.k_results_collected && self.sub.size() == KnnCollect::k(&self.sub);
        if first_k {
            self.k_results_collected = true;
        }
        self.updates.offer(similarity);
        let mut global_updated = self.non_competitive.offer(similarity);
        if self.k_results_collected && (first_k || (self.sub.visited_count() & self.interval) == 0)
        {
            let len = self.updates.size();
            if len > 0 {
                for i in 0..len {
                    self.scratch[i] = self.updates.poll().unwrap_or(0.0);
                }
                self.cached_global_min_sim = self.global.offer_sorted(&self.scratch[..len]);
                global_updated = true;
            }
        }
        local || global_updated
    }
    fn min_competitive_similarity(&self) -> f32 {
        if !self.k_results_collected {
            return f32::NEG_INFINITY;
        }
        self.sub
            .min_competitive_similarity()
            .max(self.non_competitive.peek().min(self.cached_global_min_sim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::field_infos::VectorSimilarityFunction;

    #[test]
    fn the_walk_planner_gives_up_after_a_probe_that_meets_few_accepted_nodes() {
        // 1000 unfiltered visits, a filter passing 10% of the graph, a scan
        // costing 2000 visits, a probe of 100.
        let mut p =
            FilteredWalkPlanner::new(KnnCollector::new(50, u64::MAX), 1000.0, 0.1, 2000, 100);
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
        let mut p =
            FilteredWalkPlanner::new(KnnCollector::new(50, u64::MAX), 1000.0, 0.1, 2000, 100);
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
        let mut p =
            FilteredWalkPlanner::new(KnnCollector::new(100, u64::MAX), 1000.0, 0.1, 2000, 100);
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
        let mut p =
            FilteredWalkPlanner::new(KnnCollector::new(2, u64::MAX), 1000.0, 0.01, 50, 1000);
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
            !TimeLimitingKnnCollector::new(KnnCollector::new(1, u64::MAX), later)
                .early_terminated()
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
            MultiLeafKnnCollector::with_params(2, 2.0, 1, &global, KnnCollector::new(2, 1))
                .is_err()
        );
        assert!(
            MultiLeafKnnCollector::with_params(2, 0.5, 0, &global, KnnCollector::new(2, 1))
                .is_err()
        );
        assert!(!c.early_terminated());
        c.inc_visited_count(1);
        assert_eq!(
            (c.k(), c.visited_count(), c.visit_limit()),
            (2, 1, u64::MAX)
        );
        assert_eq!(c.into_inner().size(), 2);
    }
}
