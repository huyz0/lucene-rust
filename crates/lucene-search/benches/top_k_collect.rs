//! `TopDocsCollector` filling and updating its queue at several `top_n`
//! values, fed 1M hits with scores in random order (every hit competes while
//! the queue fills; after that about `k * ln(N/k)` do). Large `top_n` is the
//! OpenSearch `size=1000..10000` and rescoring-window case.
//!
//! Run with `cargo bench -p lucene-search --bench top_k_collect`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lucene_search::collector::{ScoringCollector, TopDocsCollector};

fn scores(n: usize) -> Vec<f32> {
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / 1024.0
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let s = scores(1_000_000);
    let mut g = c.benchmark_group("top_k_collect");
    g.sample_size(10);
    for &k in &[10usize, 100, 1000, 10_000] {
        g.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let mut c = TopDocsCollector::with_total_hits_threshold(k, 1000);
                for (doc, &score) in s.iter().enumerate() {
                    c.collect(doc as i32, score);
                }
                black_box(c.top_docs().len())
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
