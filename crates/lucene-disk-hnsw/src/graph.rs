//! Building the stock HNSW graph over a field's vectors, and the
//! breadth-first node order the records are written in.

use std::collections::VecDeque;

use lucene_codecs::hnsw::{self, HnswGraphView, OnHeapHnswGraph};
use lucene_codecs::hnsw_concurrent;

use crate::{Error, Result, Similarity};

/// How the graph is built: Lucene's `M`, `beamWidth` and seed, as for
/// `Lucene99HnswVectorsFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphParams {
    /// Neighbours per node on the upper levels; level 0 keeps up to `2 * m`.
    pub m: i32,
    /// Candidates kept while inserting a node.
    pub beam_width: i32,
    /// The level-assignment seed.
    pub seed: u64,
}

impl Default for GraphParams {
    fn default() -> Self {
        GraphParams {
            m: hnsw::DEFAULT_MAX_CONN,
            beam_width: hnsw::DEFAULT_BEAM_WIDTH,
            seed: hnsw::DEFAULT_RAND_SEED,
        }
    }
}

/// Scores stored vectors against a stored vector, as graph construction does.
#[derive(Debug, Clone)]
struct FlatScorer<'a> {
    vectors: &'a [f32],
    dimension: usize,
    similarity: Similarity,
    query: usize,
}

impl FlatScorer<'_> {
    fn row(&self, ord: usize) -> &[f32] {
        // ARITH: ordinals are below the vector count, so the row lies inside
        // `vectors`, whose length is that count times the dimension.
        #[allow(clippy::arithmetic_side_effects)]
        let start = ord * self.dimension;
        &self.vectors[start..][..self.dimension]
    }
}

impl hnsw::VectorScorer for FlatScorer<'_> {
    fn score(&mut self, node: i32) -> lucene_codecs::vectors::Result<f32> {
        let node = usize::try_from(node).map_err(|_| {
            lucene_codecs::vectors::Error::InvalidGraphParameter(format!("ordinal {node}"))
        })?;
        Ok(self.similarity.score(self.row(self.query), self.row(node)))
    }

    fn max_ord(&self) -> i32 {
        i32::try_from(self.vectors.len().checked_div(self.dimension).unwrap_or(0))
            .unwrap_or(i32::MAX)
    }
}

impl hnsw::UpdateableVectorScorer for FlatScorer<'_> {
    fn set_scoring_ordinal(&mut self, ord: i32) -> lucene_codecs::vectors::Result<()> {
        self.query = usize::try_from(ord).map_err(|_| {
            lucene_codecs::vectors::Error::InvalidGraphParameter(format!("ordinal {ord}"))
        })?;
        Ok(())
    }
}

/// Builds Lucene's HNSW graph over `vectors` (rows of `dimension` floats)
/// on full-precision scores, with one worker per rayon thread, as a merge
/// builds it (`HnswConcurrentMergeBuilder`).
///
/// # Errors
///
/// A zero dimension, vectors that are not whole rows, more than `i32::MAX`
/// vectors, or graph parameters Lucene refuses.
pub fn build_graph(
    vectors: &[f32],
    dimension: usize,
    similarity: Similarity,
    params: &GraphParams,
) -> Result<OnHeapHnswGraph> {
    if dimension == 0 || !vectors.len().is_multiple_of(dimension) {
        return Err(Error::IllegalArgument(format!(
            "{} floats are not whole vectors of dimension {dimension}",
            vectors.len()
        )));
    }
    let count = i32::try_from(vectors.len().checked_div(dimension).unwrap_or(0))
        .map_err(|_| Error::IllegalArgument("too many vectors".into()))?;
    if !(1..=hnsw::MAXIMUM_MAX_CONN).contains(&params.m) {
        return Err(Error::IllegalArgument(format!(
            "M must be in 1..={}, got {}",
            hnsw::MAXIMUM_MAX_CONN,
            params.m
        )));
    }
    let scorer = FlatScorer {
        vectors,
        dimension,
        similarity,
        query: 0,
    };
    let workers = rayon::current_num_threads().max(1);
    let graph = hnsw_concurrent::build_concurrent(
        vec![scorer; workers],
        params.m,
        params.beam_width,
        params.seed,
        OnHeapHnswGraph::with_size(params.m, count),
        None,
        count,
        hnsw_concurrent::DEFAULT_BATCH_SIZE,
    )?;
    Ok(graph)
}

/// The order records are written in: `order[new] = old`, breadth first over
/// level 0 from the entry node, then any node it does not reach in
/// ascending order. Neighbours get nearby numbers, so they tend to share
/// pages.
///
/// # Errors
///
/// The graph's own read errors, or a neighbour outside `0..size`.
pub(crate) fn breadth_first_order<G: HnswGraphView>(graph: &G, size: usize) -> Result<Vec<u32>> {
    let mut seen = vec![false; size];
    let mut order = Vec::with_capacity(size);
    let mut queue = VecDeque::new();
    let mut neighbors = Vec::new();
    let mut visit = |node: usize, order: &mut Vec<u32>, queue: &mut VecDeque<usize>| {
        if !seen[node] {
            seen[node] = true;
            // Node ids fit `u32`: the writer refuses larger fields.
            order.push(u32::try_from(node).unwrap_or(u32::MAX));
            queue.push_back(node);
        }
    };
    let entry = usize::try_from(graph.entry_node())
        .ok()
        .filter(|&e| e < size);
    let starts = entry.into_iter().chain(0..size);
    for start in starts {
        visit(start, &mut order, &mut queue);
        while let Some(node) = queue.pop_front() {
            // Node ids are below `size`, which fits `i32`.
            graph.neighbors_into(0, i32::try_from(node).unwrap_or(i32::MAX), &mut neighbors)?;
            for &nb in &neighbors {
                let nb = usize::try_from(nb)
                    .ok()
                    .filter(|&nb| nb < size)
                    .ok_or_else(|| Error::Graph(format!("neighbour {nb} of node {node}")))?;
                visit(nb, &mut order, &mut queue);
            }
        }
        if order.len() == size {
            break;
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Vec<f32> {
        (0..n).flat_map(|i| [i as f32, (i % 7) as f32]).collect()
    }

    #[test]
    fn builds_a_connected_graph() {
        let v = grid(500);
        let g = build_graph(&v, 2, Similarity::Euclidean, &GraphParams::default()).unwrap();
        assert_eq!(g.size(), 500);
        let order = breadth_first_order(&g, 500).unwrap();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..500).collect::<Vec<u32>>());
        assert_eq!(order[0], g.entry_node() as u32);
    }

    #[test]
    fn rejects_bad_input() {
        let p = GraphParams::default();
        assert!(build_graph(&[1.0; 3], 2, Similarity::Euclidean, &p).is_err());
        assert!(build_graph(&[1.0; 4], 0, Similarity::Euclidean, &p).is_err());
        let bad = GraphParams { m: 0, ..p };
        assert!(build_graph(&[1.0; 4], 2, Similarity::Euclidean, &bad).is_err());
        let bad = GraphParams { beam_width: 0, ..p };
        assert!(build_graph(&[1.0; 4], 2, Similarity::Euclidean, &bad).is_err());
    }

    #[test]
    fn scorer_rejects_negative_ordinals() {
        use hnsw::{UpdateableVectorScorer, VectorScorer};
        let v = grid(3);
        let mut s = FlatScorer {
            vectors: &v,
            dimension: 2,
            similarity: Similarity::Euclidean,
            query: 0,
        };
        assert_eq!(s.max_ord(), 3);
        assert!(s.score(-1).is_err());
        assert!(s.set_scoring_ordinal(-1).is_err());
        s.set_scoring_ordinal(2).unwrap();
        assert_eq!(s.score(2).unwrap(), 1.0);
    }

    struct Islands;
    impl HnswGraphView for Islands {
        fn size(&self) -> i32 {
            4
        }
        fn num_levels(&self) -> i32 {
            1
        }
        fn entry_node(&self) -> i32 {
            2
        }
        fn max_conn(&self) -> i32 {
            1
        }
        fn neighbors_into(
            &self,
            _: i32,
            node: i32,
            out: &mut Vec<i32>,
        ) -> lucene_codecs::vectors::Result<()> {
            out.clear();
            out.push(match node {
                0 => 1,
                1 => 0,
                2 => 3,
                _ => 2,
            });
            Ok(())
        }
        fn sorted_nodes_on_level(&self, _: i32) -> lucene_codecs::vectors::Result<Vec<i32>> {
            Ok((0..4).collect())
        }
    }

    #[test]
    fn unreached_nodes_follow_in_order() {
        let g = Islands;
        assert_eq!((g.size(), g.num_levels(), g.max_conn()), (4, 1, 1));
        assert_eq!(g.sorted_nodes_on_level(0).unwrap().len(), 4);
        assert_eq!(breadth_first_order(&g, 4).unwrap(), vec![2, 3, 0, 1]);
        assert!(breadth_first_order(&Islands, 3).is_err());
    }
}
