//! `MaxSim` of one ColBERT-shaped query (23 tokens) against one document
//! (236 tokens, 96 dims: answerai-colbert-small on BEIR SciFact averages),
//! per stored encoding, against the Lucene port's `LateInteractionFloatValuesSource`
//! path (decode to `Vec<Vec<f32>>`, then `SUM_MAX_SIM` with dot product).

#![allow(clippy::arithmetic_side_effects)]

use criterion::{criterion_group, criterion_main, Criterion};
use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_search::values_source::{decode_multi_vector, MultiVectorSimilarity, ScoreFunction};
use lucene_tensor::{encode, MaxSimQuery, TokenEncoding};

fn unit_tokens(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut v: Vec<f32> = (0..n * dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    for t in v.chunks_exact_mut(dim) {
        let norm = t.iter().map(|x| x * x).sum::<f32>().sqrt();
        t.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

fn bench(c: &mut Criterion) {
    let (nq, nd, dim) = (23, 236, 96);
    let q = unit_tokens(nq, dim, 3);
    let d = unit_tokens(nd, dim, 5);
    let rows: Vec<Vec<f32>> = q.chunks(dim).map(<[f32]>::to_vec).collect();
    let query = MaxSimQuery::new(&rows).unwrap();
    let mut g = c.benchmark_group("max_sim_23x236x96");
    let lucene = encode(&d, dim, TokenEncoding::Float32).unwrap();
    g.bench_function("lucene_port_scalar", |b| {
        b.iter(|| {
            let doc = decode_multi_vector(&lucene).unwrap();
            ScoreFunction::SumMaxSim
                .compare(&rows, &doc, VectorSimilarityFunction::DotProduct)
                .unwrap()
        })
    });
    for enc in [
        TokenEncoding::Float32,
        TokenEncoding::Float16,
        TokenEncoding::Int8,
    ] {
        let v = encode(&d, dim, enc).unwrap();
        let mut scratch = Vec::new();
        g.bench_function(enc.name(), |b| {
            b.iter(|| query.score(&v, &mut scratch).unwrap())
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
