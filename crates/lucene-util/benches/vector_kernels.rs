//! The float similarity kernels and the IVF per-probe query setup, each
//! beside its portable specification: `simd::{dot_f32, square_distance_f32,
//! cosine_parts_f32, residual_stats, quantize_residual, nibble_luts}`. On
//! x86_64 the dispatched kernel is the AVX2 build baseline, on aarch64 NEON;
//! the `_scalar` rows are the same work through the specification.
//!
//! `gather` scores 32 random rows of a table far larger than the caches, as
//! an HNSW neighbour list is scored: `prefetch` asks for every row's cache
//! lines first (`simd::prefetch`, what `FloatVectorScorer::prefetch` does),
//! `no_prefetch` does not. Their gap is what the prefetch hint buys on the
//! target.
//!
//! Run with: `cargo bench -p lucene-util --bench vector_kernels`
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lucene_util::simd;

fn floats(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn bench_similarity(c: &mut Criterion) {
    let mut g = c.benchmark_group("similarity");
    for dim in [100usize, 384, 768] {
        let (a, b) = (floats(dim, 1), floats(dim, 2));
        g.throughput(Throughput::Elements(dim as u64));
        g.bench_function(format!("dot/{dim}"), |x| {
            x.iter(|| simd::dot_f32(black_box(&a), &b))
        });
        g.bench_function(format!("dot_scalar/{dim}"), |x| {
            x.iter(|| simd::dot_f32_scalar(black_box(&a), &b))
        });
        g.bench_function(format!("l2/{dim}"), |x| {
            x.iter(|| simd::square_distance_f32(black_box(&a), &b))
        });
        g.bench_function(format!("l2_scalar/{dim}"), |x| {
            x.iter(|| simd::square_distance_f32_scalar(black_box(&a), &b))
        });
        g.bench_function(format!("cosine/{dim}"), |x| {
            x.iter(|| simd::cosine_parts_f32(black_box(&a), &b))
        });
        g.bench_function(format!("cosine_scalar/{dim}"), |x| {
            x.iter(|| simd::cosine_parts_f32_scalar(black_box(&a), &b))
        });
    }
    g.finish();
}

fn bench_probe_setup(c: &mut Criterion) {
    let mut g = c.benchmark_group("probe_setup");
    for dim in [128usize, 384, 768] {
        let (q, centroid) = (floats(dim, 3), floats(dim, 4));
        let mut out = vec![0u8; dim];
        g.throughput(Throughput::Elements(dim as u64));
        g.bench_function(format!("residual_stats/{dim}"), |x| {
            x.iter(|| simd::residual_stats(black_box(&q), &centroid))
        });
        g.bench_function(format!("residual_stats_scalar/{dim}"), |x| {
            x.iter(|| simd::residual_stats_scalar(black_box(&q), &centroid))
        });
        let (iv, inv, step) = ([-1.5f32, 1.5], 63.0 / 3.0, 3.0 / 63.0);
        g.bench_function(format!("quantize_residual/{dim}"), |x| {
            x.iter(|| simd::quantize_residual(black_box(&q), &centroid, iv, inv, step, &mut out))
        });
        g.bench_function(format!("quantize_residual_scalar/{dim}"), |x| {
            x.iter(|| {
                simd::quantize_residual_scalar(black_box(&q), &centroid, iv, inv, step, &mut out)
            })
        });
        let values: Vec<u8> = (0..dim).map(|i| (i * 7 % 64) as u8).collect();
        let mut luts = vec![0u8; dim * 4];
        g.bench_function(format!("nibble_luts/{dim}"), |x| {
            x.iter(|| simd::nibble_luts(black_box(&values), &mut luts))
        });
        g.bench_function(format!("nibble_luts_scalar/{dim}"), |x| {
            x.iter(|| simd::nibble_luts_scalar(black_box(&values), &mut luts))
        });
    }
    g.finish();
}

fn bench_gather(c: &mut Criterion) {
    let mut g = c.benchmark_group("gather");
    const BATCH: usize = 32;
    for dim in [100usize, 768] {
        // 512 MiB of rows: every batch misses the caches.
        let rows = (512usize << 20) / 4 / dim;
        let table = floats(rows * dim, 5);
        let q = floats(dim, 6);
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut batch = move || {
            let mut out = [0usize; BATCH];
            for o in &mut out {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *o = (s % rows as u64) as usize;
            }
            out
        };
        g.throughput(Throughput::Elements(BATCH as u64));
        let score = |rows: &[usize], prefetch: bool| {
            if prefetch {
                for &r in rows {
                    let row = &table[r * dim..(r + 1) * dim];
                    let mut i = 0;
                    while i < dim {
                        simd::prefetch(row, i);
                        i += 16;
                    }
                    simd::prefetch(row, dim - 1);
                }
            }
            rows.iter()
                .map(|&r| simd::dot_f32(&q, &table[r * dim..(r + 1) * dim]))
                .sum::<f32>()
        };
        g.bench_function(format!("prefetch/{dim}"), |x| {
            x.iter(|| score(black_box(&batch()), true))
        });
        g.bench_function(format!("no_prefetch/{dim}"), |x| {
            x.iter(|| score(black_box(&batch()), false))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_similarity, bench_probe_setup, bench_gather);
criterion_main!(benches);
