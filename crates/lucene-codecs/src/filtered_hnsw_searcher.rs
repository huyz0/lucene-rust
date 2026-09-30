//! Port of `org.apache.lucene.util.hnsw.FilteredHnswGraphSearcher`: the
//! ACORN-style level-0 search used when a filter passes few enough vectors
//! that the plain beam search would waste its budget scoring rejected nodes.
//! Instead of scoring every unvisited neighbour, it scores only the accepted
//! ones and, when many were rejected, explores the rejected ones'
//! neighbours (two hops) to find more accepted nodes.
//!
//! It is chosen by `KnnSearchStrategy.Hnsw.useFilteredSearch`: when the share
//! of vectors passing the filter, in percent, is below the strategy's
//! `filteredSearchThreshold`. Lucene's default threshold is 0, so this is
//! opt-in; [`search_with_strategy`] is `HnswGraphSearcher.search(scorer,
//! collector, graph, acceptOrds, filteredDocCount)` with that decision made.

use lucene_util::fixed_bit_set::FixedBitSet;

use crate::hnsw::{HnswGraphSearcher, HnswGraphView, KnnCollector, NeighborQueue, VectorScorer};
use crate::hnsw_vectors::{self, SearchOptions};
use crate::vectors::{Error, Result};

/// `FilteredHnswGraphSearcher.EXPANDED_EXPLORATION_LAMBDA`.
const EXPANDED_EXPLORATION_LAMBDA: f32 = 0.10;

/// `KnnSearchStrategy.DEFAULT_FILTERED_SEARCH_THRESHOLD`.
pub const DEFAULT_FILTERED_SEARCH_THRESHOLD: i32 = 0;

/// `KnnSearchStrategy.Hnsw.useFilteredSearch(ratio)`.
pub fn use_filtered_search(filtered_search_threshold: i32, ratio_passing_filter: f32) -> bool {
    ratio_passing_filter * 100.0 < filtered_search_threshold as f32
}

/// `Math.round(double)` for the two parameters the constructor rounds.
fn round_half_up(x: f64) -> i64 {
    lucene_util::vector_util::java_round_f64(x)
}

/// `FilteredHnswGraphSearcher.IntArrayQueue`: a fixed-capacity FIFO.
#[derive(Debug)]
struct IntArrayQueue {
    nodes: Vec<i32>,
    upto: usize,
    size: usize,
}

impl IntArrayQueue {
    fn new(capacity: usize) -> Self {
        IntArrayQueue {
            nodes: vec![0; capacity],
            upto: 0,
            size: 0,
        }
    }
    fn capacity(&self) -> usize {
        self.nodes.len()
    }
    fn count(&self) -> usize {
        self.size.saturating_sub(self.upto)
    }
    fn is_full(&self) -> bool {
        self.size == self.nodes.len()
    }
    fn add(&mut self, node: i32) {
        // Java throws UnsupportedOperationException; every caller checks
        // `is_full` (or a bound below capacity) first.
        debug_assert!(!self.is_full());
        if let Some(slot) = self.nodes.get_mut(self.size) {
            *slot = node;
            self.size = self.size.saturating_add(1);
        }
    }
    /// `poll()`: `None` is Java's `NO_MORE_DOCS`.
    fn poll(&mut self) -> Option<i32> {
        if self.upto == self.size {
            return None;
        }
        let v = self.nodes[self.upto];
        self.upto = self.upto.saturating_add(1);
        Some(v)
    }
    fn clear(&mut self) {
        self.upto = 0;
        self.size = 0;
    }
}

/// `FilteredHnswGraphSearcher`.
#[derive(Debug)]
pub struct FilteredHnswGraphSearcher {
    candidates: NeighborQueue,
    visited: FixedBitSet,
    max_exploration_multiplier: usize,
    min_to_score: usize,
    /// The plain searcher, for the upper-level descent (Java's subclass
    /// inherits `findBestEntryPoint` from `HnswGraphSearcher`).
    base: HnswGraphSearcher,
    neighbors: Vec<i32>,
    friends_of_friend: Vec<i32>,
    bulk_scores: Vec<f32>,
}

impl FilteredHnswGraphSearcher {
    /// `FilteredHnswGraphSearcher.create(k, graph, filterSize, acceptOrds)`.
    pub fn create<G: HnswGraphView>(k: usize, graph: &G, filter_size: i32) -> Result<Self> {
        let graph_size = i64::from(graph.max_node_id()).saturating_add(1);
        if filter_size <= 0 || i64::from(filter_size) >= graph_size {
            return Err(Error::InvalidGraphParameter(
                "filterSize must be > 0 and < graph size".into(),
            ));
        }
        let max_conn = graph.max_conn();
        if max_conn <= 0 {
            return Err(Error::InvalidGraphParameter(
                "graph must have known max connections".into(),
            ));
        }
        let filter_ratio = filter_size as f32 / graph.size() as f32;
        let max_exploration_multiplier =
            round_half_up((1.0 / filter_ratio as f64).min(max_conn as f64 / 2.0)).max(0) as usize;
        let min_to_score = round_half_up(
            (1.0 / filter_ratio as f64 - 2.0 * max_conn as f64)
                .max(0.0)
                .min(max_conn as f64),
        )
        .max(0) as usize;
        let capacity = usize::try_from(graph_size).unwrap_or(0);
        Ok(FilteredHnswGraphSearcher {
            candidates: NeighborQueue::new(k.max(1), true),
            visited: FixedBitSet::new(capacity.max(1)),
            max_exploration_multiplier,
            min_to_score,
            base: HnswGraphSearcher::new(k, graph.size()),
            neighbors: Vec::new(),
            friends_of_friend: Vec::new(),
            bulk_scores: Vec::new(),
        })
    }

    /// `AbstractHnswGraphSearcher.search`.
    pub fn search<G: HnswGraphView, S: VectorScorer>(
        &mut self,
        results: &mut KnnCollector,
        scorer: &mut S,
        graph: &G,
        accept_ords: &FixedBitSet,
    ) -> Result<()> {
        let ep = self.base.find_best_entry_point(scorer, graph, results)?;
        if ep == -1 {
            return Ok(());
        }
        self.search_level(results, scorer, &[ep], graph, accept_ords)
    }

    fn check(&self, ord: i32) -> Result<usize> {
        usize::try_from(ord)
            .ok()
            .filter(|&o| o < self.visited.len())
            .ok_or_else(|| {
                Error::CorruptMeta(format!(
                    "neighbour ordinal {ord} is outside the graph's 0..{} ordinals",
                    self.visited.len()
                ))
            })
    }

    /// `visited.getAndSet(ord)`.
    // FBS: every caller passes an ordinal returned by `check`, which bounds
    // it against `self.visited.len()`.
    fn get_and_set(&mut self, ord: usize) -> bool {
        let was = self.visited.get(ord);
        self.visited.set(ord);
        was
    }

    /// `FilteredHnswGraphSearcher.searchLevel` (level 0 only).
    pub fn search_level<G: HnswGraphView, S: VectorScorer>(
        &mut self,
        results: &mut KnnCollector,
        scorer: &mut S,
        eps: &[i32],
        graph: &G,
        accept_ords: &FixedBitSet,
    ) -> Result<()> {
        if accept_ords.len() < self.visited.len() {
            return Err(Error::InvalidGraphParameter(format!(
                "the accept-ordinal set covers {} ordinals, short of the {} this graph can name",
                accept_ords.len(),
                self.visited.len()
            )));
        }
        // prepareScratchState
        self.candidates.clear();
        self.visited.clear_all();
        if self.bulk_scores.len() < eps.len() {
            self.bulk_scores.resize(eps.len(), 0.0);
        }
        if results.early_terminated() {
            return Ok(());
        }
        // scoreEntryPoints
        for &ep in eps {
            self.check(ep)?;
        }
        scorer.bulk_score(eps, &mut self.bulk_scores[..eps.len()])?;
        results.inc_visited_count(eps.len());
        for (i, &ep) in eps.iter().enumerate() {
            let score = self.bulk_scores[i];
            self.visited.set(ep as usize);
            self.candidates.add(ep, score);
            if accept_ords.get(ep as usize) {
                results.collect(ep, score);
            }
        }
        if results.early_terminated() {
            return Ok(());
        }
        let max_conn = usize::try_from(graph.max_conn()).unwrap_or(0);
        let queue_capacity = max_conn
            .saturating_mul(2)
            .saturating_mul(self.max_exploration_multiplier);
        let mut to_score = IntArrayQueue::new(queue_capacity);
        let mut to_explore = IntArrayQueue::new(queue_capacity);
        let mut min_accepted_similarity = results.min_competitive_similarity().next_up();
        while self.candidates.size() > 0 && !results.early_terminated() {
            let top_candidate_similarity = self.candidates.top_score();
            if min_accepted_similarity > top_candidate_similarity {
                break;
            }
            let top_candidate_node = self.candidates.pop();
            graph.neighbors_into(0, top_candidate_node, &mut self.neighbors)?;
            let neighbor_count = self.neighbors.len();
            to_score.clear();
            to_explore.clear();
            let mut idx = 0;
            // `while ((friendOrd = nextNeighbor()) != NO_MORE_DOCS && !toScore.isFull())`
            while idx < self.neighbors.len() && !to_score.is_full() {
                let friend = self.neighbors[idx];
                idx = idx.saturating_add(1);
                let f = self.check(friend)?;
                if self.get_and_set(f) {
                    continue;
                }
                if accept_ords.get(f) {
                    to_score.add(friend);
                } else {
                    to_explore.add(friend);
                }
            }
            let filtered_amount = to_explore.count() as f32 / neighbor_count as f32;
            let max_to_score_count = (neighbor_count as f32
                * (self.max_exploration_multiplier as f32).min(1.0 / (1.0 - filtered_amount)))
                as usize;
            let max_additional_to_explore_count = to_explore.capacity().saturating_sub(1);
            let mut total_explored = to_score.count().saturating_add(to_explore.count());
            if to_score.count() < max_to_score_count
                && filtered_amount > EXPANDED_EXPLORATION_LAMBDA
            {
                // The poll is evaluated first, so it consumes a node even when
                // one of the later conditions then ends the loop -- as in Java.
                while let Some(explore_friend) = to_explore.poll() {
                    if !(total_explored < max_additional_to_explore_count
                        && to_score.count() < max_to_score_count)
                    {
                        break;
                    }
                    graph.neighbors_into(0, explore_friend, &mut self.friends_of_friend)?;
                    let mut j = 0;
                    while j < self.friends_of_friend.len() && to_score.count() < max_to_score_count
                    {
                        let fof = self.friends_of_friend[j];
                        j = j.saturating_add(1);
                        let g = self.check(fof)?;
                        if self.get_and_set(g) {
                            continue;
                        }
                        total_explored = total_explored.saturating_add(1);
                        if accept_ords.get(g) {
                            to_score.add(fof);
                        } else if total_explored < max_additional_to_explore_count
                            && to_score.count() < self.min_to_score
                        {
                            to_explore.add(fof);
                        }
                    }
                }
            }
            let n = to_score.count();
            if self.bulk_scores.len() < n {
                self.bulk_scores.resize(n, 0.0);
            }
            let max_score = if n > 0 {
                scorer.bulk_score(&to_score.nodes[..to_score.size], &mut self.bulk_scores[..n])?
            } else {
                f32::NEG_INFINITY
            };
            results.inc_visited_count(n);
            if max_score > min_accepted_similarity {
                for i in 0..n {
                    let friend_similarity = self.bulk_scores[i];
                    if friend_similarity > min_accepted_similarity {
                        let ord = to_score.nodes[i];
                        self.candidates.add(ord, friend_similarity);
                        if results.collect(ord, friend_similarity) {
                            min_accepted_similarity =
                                results.min_competitive_similarity().next_up();
                        }
                    }
                }
            }
            to_score.upto = to_score.size;
        }
        Ok(())
    }
}

/// `HnswGraphSearcher.search(scorer, collector, graph, acceptOrds,
/// filteredDocCount)` inside `Lucene99HnswVectorsReader.search`, with the
/// `KnnSearchStrategy.Hnsw` threshold: [`crate::hnsw_vectors::search`] unless
/// the filter passes few enough vectors for the strategy to pick the
/// filtered searcher.
pub fn search_with_strategy<G: HnswGraphView, S: VectorScorer>(
    scorer: &mut S,
    graph: Option<&G>,
    k: usize,
    visit_limit: u64,
    options: SearchOptions<'_>,
    filtered_search_threshold: i32,
) -> Result<(Vec<(i32, f32)>, bool)> {
    if !(0..=100).contains(&filtered_search_threshold) {
        return Err(Error::InvalidGraphParameter(
            "filteredSearchThreshold must be >= 0 and <= 100".into(),
        ));
    }
    let (Some(g), Some(accept)) = (graph, options.accept_ords) else {
        return hnsw_vectors::search(scorer, graph, k, visit_limit, options);
    };
    let num_vectors = scorer.max_ord();
    let graph_size = g.size();
    let filtered = options
        .filtered_doc_count
        .unwrap_or(graph_size)
        .min(graph_size);
    let unfiltered_visit = crate::hnsw::expected_visited_nodes(k as i32, graph_size);
    let do_hnsw =
        (k as i64) < i64::from(num_vectors) && unfiltered_visit < filtered && graph_size > 0;
    let filtered_ok = options.seed_ords.is_none()
        && g.max_conn() > 0
        && filtered > 0
        && use_filtered_search(
            filtered_search_threshold,
            filtered as f32 / graph_size as f32,
        );
    if !do_hnsw || !filtered_ok || num_vectors == 0 || k == 0 {
        return hnsw_vectors::search(scorer, graph, k, visit_limit, options);
    }
    // The same accept-set bound hnsw_vectors::search applies.
    let needed = (i64::from(g.max_node_id()).saturating_add(1)).max(i64::from(num_vectors));
    if (accept.len() as i64) < needed {
        return Err(Error::InvalidGraphParameter(format!(
            "the accept-ordinal set covers {} ordinals, short of the {needed} this field can name",
            accept.len()
        )));
    }
    let mut collector = KnnCollector::new(k, visit_limit);
    let mut searcher = FilteredHnswGraphSearcher::create(k, g, filtered)?;
    searcher.search(&mut collector, scorer, g, accept)?;
    let early = collector.early_terminated();
    Ok((collector.top_docs(), early))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::hnsw::{self, HnswGraphBuilder, OnHeapHnswGraph, UpdateableVectorScorer};

    /// 2-d points on a line; score = 1 / (1 + squared distance).
    #[derive(Clone)]
    struct LineScorer {
        points: Vec<f32>,
        query: f32,
    }

    impl VectorScorer for LineScorer {
        fn score(&mut self, node: i32) -> Result<f32> {
            let d = self.points[node as usize] - self.query;
            Ok(1.0 / (1.0 + d * d))
        }
        fn max_ord(&self) -> i32 {
            self.points.len() as i32
        }
    }

    impl UpdateableVectorScorer for LineScorer {
        fn set_scoring_ordinal(&mut self, ord: i32) -> Result<()> {
            self.query = self.points[ord as usize];
            Ok(())
        }
    }

    fn graph(n: usize) -> (OnHeapHnswGraph, Vec<f32>) {
        let points: Vec<f32> = (0..n).map(|i| ((i * 7919) % 1000) as f32 / 10.0).collect();
        let g = HnswGraphBuilder::new(
            LineScorer {
                points: points.clone(),
                query: 0.0,
            },
            8,
            50,
            hnsw::DEFAULT_RAND_SEED,
        )
        .unwrap()
        .build(n as i32)
        .unwrap();
        (g, points)
    }

    #[test]
    fn filtered_search_finds_accepted_neighbours() {
        let n = 2000;
        let (g, points) = graph(n);
        // Accept every 20th node: 5% pass, below a threshold of 60.
        let mut accept = FixedBitSet::new(n);
        for i in (0..n).step_by(20) {
            accept.set(i);
        }
        let mut scorer = LineScorer {
            points: points.clone(),
            query: 50.0,
        };
        let opts = SearchOptions {
            accept_ords: Some(&accept),
            filtered_doc_count: Some(100),
            seed_ords: None,
        };
        let (hits, _) = search_with_strategy(&mut scorer, Some(&g), 5, u64::MAX, opts, 60).unwrap();
        // A filtered walk may collect fewer than k (Lucene then falls back to
        // an exact search in `AbstractKnnVectorQuery`).
        assert!(!hits.is_empty() && hits.len() <= 5, "{hits:?}");
        assert!(hits.iter().all(|&(o, _)| o % 20 == 0));
        // Exact answer among accepted nodes.
        let mut exact: Vec<(i32, f32)> = (0..n)
            .step_by(20)
            .map(|i| (i as i32, scorer.clone().score(i as i32).unwrap()))
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1));
        let best = exact[0].1;
        assert!(hits[0].1 >= best * 0.99, "{:?} vs {:?}", hits[0], exact[0]);
        // Threshold 0 (Lucene's default) takes the ordinary path.
        let (plain, _) = search_with_strategy(&mut scorer, Some(&g), 5, u64::MAX, opts, 0).unwrap();
        assert!(plain.iter().all(|&(o, _)| o % 20 == 0));
        // No filter: ordinary path.
        let (all, _) = search_with_strategy(
            &mut scorer,
            Some(&g),
            5,
            u64::MAX,
            SearchOptions::default(),
            60,
        )
        .unwrap();
        assert_eq!(all.len(), 5);
        assert!(search_with_strategy(&mut scorer, Some(&g), 5, u64::MAX, opts, 101).is_err());
        let short = FixedBitSet::new(10);
        let bad = SearchOptions {
            accept_ords: Some(&short),
            filtered_doc_count: Some(5),
            seed_ords: None,
        };
        assert!(search_with_strategy(&mut scorer, Some(&g), 5, u64::MAX, bad, 60).is_err());
    }

    #[test]
    fn create_validates_and_rounds_like_java() {
        let (g, _) = graph(300);
        assert!(FilteredHnswGraphSearcher::create(5, &g, 0).is_err());
        assert!(FilteredHnswGraphSearcher::create(5, &g, 300).is_err());
        let s = FilteredHnswGraphSearcher::create(5, &g, 30).unwrap();
        // ratio 0.1: min(10, 8 / 2) = 4; min(max(0, 10 - 16), 8) = 0.
        assert_eq!(s.max_exploration_multiplier, 4);
        assert_eq!(s.min_to_score, 0);
        let s = FilteredHnswGraphSearcher::create(5, &g, 10).unwrap();
        // ratio 1/30: min(30, 4) = 4; min(max(0, 30 - 16), 8) = 8.
        assert_eq!(s.min_to_score, 8);
        assert!(use_filtered_search(60, 0.5));
        assert!(!use_filtered_search(
            DEFAULT_FILTERED_SEARCH_THRESHOLD,
            0.01
        ));
        let mut q = IntArrayQueue::new(2);
        assert_eq!(q.poll(), None);
        q.add(1);
        q.add(2);
        assert!(q.is_full());
        assert_eq!(q.poll(), Some(1));
        assert_eq!(q.count(), 1);
        q.clear();
        assert_eq!(q.count(), 0);
    }
}
