//! Centroid routing micro-benchmarks: a query routed through each
//! `Routing` mode over synthetic clustered centroids of a few realistic
//! shapes, and the 8-bit scan kernel on its own.
//!
//! ```text
//! cargo bench -p lucene-ivf --bench routing
//! ```
#![allow(clippy::arithmetic_side_effects)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lucene_ivf::routing::{tree_parents, Router, Routing};
use lucene_ivf::Similarity;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

/// `n` rows of `dim` floats around `n / 50` random centres, like the
/// centroids of a clustered field.
fn clustered(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let centres: Vec<f32> = (0..(n / 50).max(1) * dim)
        .map(|_| rng.next() * 4.0)
        .collect();
    let k = centres.len() / dim;
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centres[(i % k) * dim..(i % k + 1) * dim];
        out.extend(c.iter().map(|&x| x + rng.next()));
    }
    out
}

fn routing(c: &mut Criterion) {
    let shapes = [
        (6400, 300, "6.4k x 300"),
        (27_000, 300, "27k x 300"),
        (24_000, 100, "24k x 100"),
    ];
    for (sim, sname) in [(Similarity::Euclidean, "l2"), (Similarity::Cosine, "cos")] {
        let mut g = c.benchmark_group(format!("route_{sname}"));
        for &(n, dim, label) in &shapes {
            let centroids = clustered(n, dim, 7);
            let queries = clustered(64, dim, 8);
            let parents = tree_parents(&centroids, dim, sim);
            for (mode, routing, stored) in [
                ("exact", Routing::Exact, None),
                ("int8", Routing::Quantized, None),
                ("tree", Routing::Auto, Some(&parents[..])),
            ] {
                let r = Router::new(&centroids, dim, sim, routing, stored);
                let mut i = 0;
                g.bench_with_input(BenchmarkId::new(mode, label), &r, |b, r| {
                    b.iter(|| {
                        i = (i + 1) % 64;
                        r.route(&centroids, &queries[i * dim..(i + 1) * dim], 64)
                    })
                });
            }
        }
        g.finish();
    }
}

fn kernel(c: &mut Criterion) {
    let mut g = c.benchmark_group("i8_scan");
    use lucene_util::simd::{dot_u7_i8, dot_u7_i8_x4};
    for (n, dim) in [(6400usize, 320usize), (27_000, 320), (24_000, 128)] {
        let codes: Vec<u8> = (0..n * dim).map(|i| (i * 31 % 251) as u8).collect();
        let q: Vec<u8> = (0..dim).map(|i| (i * 7 % 127) as u8).collect();
        g.bench_function(format!("i8 {n} x {dim}"), |b| {
            b.iter(|| {
                codes
                    .chunks_exact(dim)
                    .map(|c| lucene_util::vector_util::dot_product_i8(&q, c))
                    .fold(0i32, i32::wrapping_add)
            })
        });
        g.bench_function(format!("u7 {n} x {dim}"), |b| {
            b.iter(|| {
                codes
                    .chunks_exact(dim)
                    .map(|c| dot_u7_i8(&q, c))
                    .fold(0i32, i32::wrapping_add)
            })
        });
        g.bench_function(format!("u7x4 {n} x {dim}"), |b| {
            b.iter(|| {
                codes.chunks_exact(4 * dim).fold(0i32, |s, c| {
                    let (a, rest) = c.split_at(dim);
                    let (b2, rest) = rest.split_at(dim);
                    let (c2, d) = rest.split_at(dim);
                    dot_u7_i8_x4(&q, [a, b2, c2, d])
                        .iter()
                        .fold(s, |s, &x| s.wrapping_add(x))
                })
            })
        });
    }
    g.finish();
}

criterion_group!(benches, routing, kernel);
criterion_main!(benches);
