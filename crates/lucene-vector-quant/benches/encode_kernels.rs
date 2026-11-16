//! Per-vector index-time and per-query cost of the quantizers: rotation,
//! OSQ and extended RaBitQ encoding, and query preparation.

// Fixture arithmetic: the gate is about values read off disk.
#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_util::quantization::ScalarEncoding;
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::rotation::fwht;
use lucene_vector_quant::{QuantizerFactory, Similarity};

fn data(n: usize, dim: usize) -> Vec<f32> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..n * dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn bench_fwht(c: &mut Criterion) {
    let mut g = c.benchmark_group("fwht");
    for n in [256usize, 1024, 2048] {
        let mut x = data(1, n);
        g.bench_function(format!("{n}"), |b| {
            b.iter(|| {
                fwht(black_box(&mut x));
                // keep the values bounded
                x[0] = 0.5;
            })
        });
    }
}

fn bench_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("encode");
    for dim in [128usize, 768] {
        let sample = data(64, dim);
        let centroid = data(1, dim);
        let v = data(1, dim);
        for (name, enc) in [
            ("osq1", ScalarEncoding::SingleBitQueryNibble),
            ("osq2", ScalarEncoding::DibitQueryNibble),
            ("osq4", ScalarEncoding::PackedNibble),
            ("osq7", ScalarEncoding::SevenBit),
        ] {
            let q = OsqFactory::new(enc)
                .train(dim, Similarity::Euclidean, &sample)
                .unwrap();
            let mut out = vec![0u8; q.code_len()];
            g.bench_function(format!("{name}/{dim}"), |b| {
                b.iter(|| q.encode(black_box(&v), black_box(&centroid), &mut out))
            });
        }
        for bits in [2u8, 4, 8] {
            let q = ExRabitqFactory::new(bits)
                .train(dim, Similarity::Euclidean, &sample)
                .unwrap();
            let prepared = q.prepare_centroid(&centroid).unwrap();
            let mut out = vec![0u8; q.code_len()];
            g.bench_function(format!("erq{bits}/{dim}"), |b| {
                b.iter(|| q.encode_prepared(black_box(&v), black_box(&prepared), &mut out))
            });
        }
    }
}

fn bench_query(c: &mut Criterion) {
    let mut g = c.benchmark_group("query_scorer");
    for dim in [128usize, 768] {
        let sample = data(64, dim);
        let centroid = data(1, dim);
        let v = data(1, dim);
        let q = OsqFactory::new(ScalarEncoding::SingleBitQueryNibble)
            .train(dim, Similarity::Euclidean, &sample)
            .unwrap();
        g.bench_function(format!("osq1/{dim}"), |b| {
            b.iter(|| {
                let mut s = q.query_scorer(black_box(&v));
                s.set_centroid(&centroid);
                black_box(&s);
            })
        });
        let q = ExRabitqFactory::new(4)
            .train(dim, Similarity::Euclidean, &sample)
            .unwrap();
        g.bench_function(format!("erq4/{dim}"), |b| {
            b.iter(|| {
                let mut s = q.query_scorer(black_box(&v));
                s.set_centroid(&centroid);
                black_box(&s);
            })
        });
    }
}

criterion_group!(benches, bench_fwht, bench_encode, bench_query);
criterion_main!(benches);
