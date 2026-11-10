//! The full-text-search kernels with an AVX-512 path, each beside its
//! portable specification: hit compaction by score (`hit_compact`) and the
//! doc ids of a bit-set postings block (`simd::expand_bitset`).
//!
//! The dispatched kernel uses AVX-512 when the CPU has it. To compare it
//! against the AVX2 build baseline on the same machine, run once as is and
//! once with `LUCENE_RUST_SIMD=avx2`.
//!
//! Run with: `cargo bench -p lucene-util --bench fts_kernels`
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lucene_util::{hit_compact, simd};

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// A bulk scorer's batch: ascending docs, scores in `[0, 1)`, and a bar
/// that keeps `keep` of them.
fn batch(len: usize, seed: u64) -> (Vec<i32>, Vec<f32>) {
    let mut s = seed;
    let docs = (0..len as i32).map(|i| i * 3).collect();
    let scores = (0..len)
        .map(|_| (xorshift(&mut s) % 1000) as f32 / 1000.0)
        .collect();
    (docs, scores)
}

fn bench_hit_compact(c: &mut Criterion) {
    let mut g = c.benchmark_group("hit_compact");
    for (len, keep) in [(512usize, 0.1f64), (512, 0.5), (512, 0.9), (4096, 0.3)] {
        let (docs, scores) = batch(len, 0x1234_5678 + len as u64);
        let min = 1.0 - keep;
        g.throughput(Throughput::Elements(len as u64));
        let (mut d, mut s) = (Vec::new(), Vec::new());
        g.bench_function(format!("copy/{len}/keep{keep}"), |b| {
            b.iter(|| hit_compact::copy_by_score(black_box(&docs), &scores, min, &mut d, &mut s))
        });
        g.bench_function(format!("copy_scalar/{len}/keep{keep}"), |b| {
            b.iter(|| {
                hit_compact::copy_by_score_scalar(black_box(&docs), &scores, min, &mut d, &mut s)
            })
        });
        let wide: Vec<f64> = scores.iter().map(|&s| f64::from(s)).collect();
        g.bench_function(format!("compact/{len}/keep{keep}"), |b| {
            b.iter(|| {
                d.clone_from(&docs);
                s.clone_from(&wide);
                black_box(hit_compact::compact_by_score(&mut d, &mut s, min))
            })
        });
    }
    g.finish();
}

fn bench_expand_bitset(c: &mut Criterion) {
    let mut g = c.benchmark_group("expand_bitset");
    // 256 doc ids over `words` words: a bit-set postings block of density
    // 256 / (64 * words).
    for words in [4usize, 8, 16, 32] {
        let mut s = 0x9E37_79B9_u64 + words as u64;
        let mut bits = vec![0u64; words];
        let mut set = 0;
        while set < 256 {
            let b = (xorshift(&mut s) % (words as u64 * 64)) as usize;
            if bits[b / 64] & (1 << (b % 64)) == 0 {
                bits[b / 64] |= 1 << (b % 64);
                set += 1;
            }
        }
        let mut out = [0i32; 256];
        g.throughput(Throughput::Elements(256));
        g.bench_function(format!("words{words}"), |b| {
            b.iter(|| simd::expand_bitset(black_box(&bits), 7, &mut out))
        });
        g.bench_function(format!("words{words}_scalar"), |b| {
            b.iter(|| simd::expand_bitset_scalar(black_box(&bits), 7, &mut out))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_hit_compact, bench_expand_bitset);
criterion_main!(benches);
