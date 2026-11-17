//! Cost of reusing a graph with deletions (copy + serial repair, as the merge
//! does today) against rebuilding the live vectors with parallel workers.
//!
//!   cargo run --release -p lucene-codecs --example hnsw_merge_repair_bench -- [n] [dim] [workers]
// ARITH: a benchmark over fixed-size synthetic data; every index is bounded by `n * dim`.
#![allow(clippy::arithmetic_side_effects, clippy::needless_range_loop)]

use std::sync::Arc;
use std::time::Instant;

use lucene_codecs::hnsw::{
    HnswGraphBuilder, HnswGraphView, OnHeapHnswGraph, UpdateableVectorScorer, VectorScorer,
};
use lucene_codecs::hnsw_concurrent::build_concurrent;
use lucene_util::fixed_bit_set::FixedBitSet;

#[derive(Clone)]
struct Flat {
    data: Arc<Vec<f32>>,
    dim: usize,
    query: usize,
}

impl VectorScorer for Flat {
    fn score(&mut self, node: i32) -> lucene_codecs::vectors::Result<f32> {
        let a = &self.data[self.query * self.dim..][..self.dim];
        let b = &self.data[node as usize * self.dim..][..self.dim];
        let d: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        Ok(1.0 / (1.0 + d))
    }
    fn max_ord(&self) -> i32 {
        (self.data.len() / self.dim) as i32
    }
}

impl UpdateableVectorScorer for Flat {
    fn set_scoring_ordinal(&mut self, ord: i32) -> lucene_codecs::vectors::Result<()> {
        self.query = ord as usize;
        Ok(())
    }
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let n = args.first().copied().unwrap_or(100_000);
    let dim = args.get(1).copied().unwrap_or(128);
    let workers = args.get(2).copied().unwrap_or(4);
    let (m, beam) = (16, 100);
    let mut state = 7u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let data: Vec<f32> = (0..n * dim).map(|_| next()).collect();
    let scorer = |d: Arc<Vec<f32>>| Flat {
        data: d,
        dim,
        query: 0,
    };
    let all = Arc::new(data);
    let t = Instant::now();
    let base = build_concurrent(
        (0..workers).map(|_| scorer(all.clone())).collect(),
        m,
        beam,
        42,
        OnHeapHnswGraph::with_size(m, n as i32),
        None,
        n as i32,
        2048,
    )
    .unwrap();
    println!(
        "base graph {n} x {dim}, {workers} workers: {:.1}s",
        t.elapsed().as_secs_f64()
    );

    println!("deletes  live    repair_serial_s  rebuild_parallel_s  rebuild/repair");
    for pct in [1usize, 5, 20, 35] {
        let mut map = vec![-1i32; n];
        let mut live = Vec::new();
        for i in 0..n {
            if (i.wrapping_mul(2_654_435_761) >> 7) % 100 >= pct {
                map[i] = live.len() as i32;
                live.push(i);
            }
        }
        let mut lv = Vec::with_capacity(live.len() * dim);
        for &i in &live {
            lv.extend_from_slice(&all[i * dim..][..dim]);
        }
        let lv = Arc::new(lv);
        let total = live.len() as i32;
        let t = Instant::now();
        let g =
            HnswGraphBuilder::init_graph(scorer(lv.clone()), beam, 42, &base, &map, total).unwrap();
        let repair = t.elapsed().as_secs_f64();
        let mut bits = FixedBitSet::new(total as usize);
        for i in 0..total as usize {
            bits.set(i);
        }
        let t = Instant::now();
        let g = build_concurrent(
            (0..workers).map(|_| scorer(lv.clone())).collect(),
            m,
            beam,
            42,
            g,
            Some(&bits),
            total,
            2048,
        )
        .unwrap();
        let tail = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let r = build_concurrent(
            (0..workers).map(|_| scorer(lv.clone())).collect(),
            m,
            beam,
            42,
            OnHeapHnswGraph::with_size(m, total),
            None,
            total,
            2048,
        )
        .unwrap();
        let rebuild = t.elapsed().as_secs_f64();
        assert_eq!(g.size(), r.size());
        println!(
            "{pct:>6}%  {total:>7}  {:>14.2}  {:>18.2}  {:>13.2}",
            repair + tail,
            rebuild,
            rebuild / (repair + tail)
        );
    }
}
