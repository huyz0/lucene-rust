//! Micro-benchmarks of the IVF search and build path below centroid
//! routing (see `routing.rs` for that): one cluster scored by each
//! quantizer, rerank by vector encoding, fixed and dynamic search depth,
//! several segments with and without a shared threshold, and building and
//! merging a field. Data are synthetic clusters of GloVe-like (100) and
//! fastText-like (300) dimensions.
//!
//! ```text
//! cargo bench -p lucene-ivf --bench ivf_path
//! ```
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lucene_ivf::{
    merge_field, FieldVectors, IvfFiles, IvfParams, IvfReader, IvfWriter, MergeClustering,
    MergeSource, Probes, Rerank, SearchParams, SharedThreshold, Similarity, VectorEncoding,
};
use lucene_util::quantization::ScalarEncoding;
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::rotation::RotatedFactory;
use lucene_vector_quant::scalar::ScalarFactory;
use lucene_vector_quant::{QuantizerFactory, QuantizerRegistry};

const ID: [u8; 16] = [9; 16];
const K: usize = 10;

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

/// `n` vectors of `dim` floats around `n / 400` random centres.
fn clustered(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let k = (n / 400).max(1);
    let centres: Vec<f32> = (0..k * dim).map(|_| rng.next() * 2.0).collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centres[(i % k) * dim..(i % k + 1) * dim];
        out.extend(c.iter().map(|&x| x + rng.next()));
    }
    out
}

fn factories() -> Vec<(&'static str, Arc<dyn QuantizerFactory>)> {
    let bbq: Arc<dyn QuantizerFactory> = Arc::new(OsqFactory::bbq());
    vec![
        ("bbq", Arc::clone(&bbq)),
        (
            "osq2",
            Arc::new(OsqFactory::new(ScalarEncoding::DibitQueryNibble)),
        ),
        (
            "osq4",
            Arc::new(OsqFactory::new(ScalarEncoding::PackedNibble)),
        ),
        ("sq7", Arc::new(ScalarFactory::int7())),
        ("rbbq", Arc::new(RotatedFactory::new(bbq))),
        ("erq3", Arc::new(ExRabitqFactory::new(3))),
        ("erq4", Arc::new(ExRabitqFactory::new(4))),
    ]
}

fn registry() -> QuantizerRegistry {
    let mut r = QuantizerRegistry::new();
    for (_, f) in factories() {
        r.register(f);
    }
    r
}

fn open(files: &IvfFiles) -> IvfReader {
    IvfReader::open(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &registry(),
    )
    .unwrap()
}

fn write(
    vectors: &[f32],
    dim: usize,
    sim: Similarity,
    factory: &dyn QuantizerFactory,
    params: &IvfParams,
) -> IvfFiles {
    let docs: Vec<i32> = (0..(vectors.len() / dim) as i32).collect();
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 0,
            similarity: sim,
            dimension: dim,
            docs: &docs,
            vectors,
        },
        factory,
        params,
    )
    .unwrap();
    w.finish()
}

/// One query against one 384-vector cluster: setting the centroid (the
/// query quantized against it), scoring the cluster's block of codes, and
/// the error margins of all of them.
fn scorer(c: &mut Criterion) {
    const CLUSTER: usize = 384;
    for (dim, sim, label) in [
        (100, Similarity::Cosine, "100d cos"),
        (300, Similarity::Euclidean, "300d l2"),
    ] {
        let mut g = c.benchmark_group(format!("scorer {label}"));
        g.throughput(Throughput::Elements(CLUSTER as u64));
        let data = clustered(CLUSTER * 4, dim, 3);
        let mut centroid = vec![0f32; dim];
        for v in data[..CLUSTER * dim].chunks_exact(dim) {
            centroid.iter_mut().zip(v).for_each(|(c, x)| *c += x);
        }
        centroid.iter_mut().for_each(|c| *c /= CLUSTER as f32);
        let query = &data[CLUSTER * 2 * dim..CLUSTER * 2 * dim + dim];
        for (name, factory) in factories() {
            let q = factory.train(dim, sim, &data).unwrap();
            let prepared = q.prepare_centroid(&centroid);
            let mut codes = vec![0u8; CLUSTER * q.code_len()];
            for (v, out) in data[..CLUSTER * dim]
                .chunks_exact(dim)
                .zip(codes.chunks_exact_mut(q.code_len()))
            {
                match &prepared {
                    Some(p) => q.encode_prepared(v, p, out),
                    None => q.encode(v, &centroid, out),
                }
            }
            let mut block = Vec::new();
            q.write_block(&codes, &mut block);
            let mut s = q.query_scorer(query);
            let set = |s: &mut dyn lucene_vector_quant::QueryScorer| match &prepared {
                Some(p) => s.set_prepared_centroid(p),
                None => s.set_centroid(&centroid),
            };
            g.bench_function(BenchmarkId::new("set_centroid", name), |b| {
                b.iter(|| set(s.as_mut()))
            });
            set(s.as_mut());
            let mut scores = vec![0f32; CLUSTER];
            g.bench_function(BenchmarkId::new("score_block", name), |b| {
                b.iter(|| s.score_block(&block, &mut scores))
            });
            s.score_block(&block, &mut scores);
            let mut bounds = vec![0f32; CLUSTER];
            if s.error_bounds(&block, &scores, &mut bounds) {
                g.bench_function(BenchmarkId::new("error_bounds", name), |b| {
                    b.iter(|| s.error_bounds(&block, &scores, &mut bounds))
                });
            }
        }
        g.finish();
    }
}

/// Whole searches of one 60k-vector field: rerank by each vector encoding
/// at a fixed depth, and fixed against calibrated, error-bounded depth.
fn search(c: &mut Criterion) {
    const N: usize = 60_000;
    for (dim, sim, label) in [
        (100, Similarity::Cosine, "100d cos"),
        (300, Similarity::Euclidean, "300d l2"),
    ] {
        let mut g = c.benchmark_group(format!("search {label}"));
        let data = clustered(N, dim, 11);
        let queries = clustered(64, dim, 12);
        let bbq = OsqFactory::bbq();
        let erq = ExRabitqFactory::new(4);
        let mut run = |name: String, files: &IvfFiles, params: SearchParams| {
            let r = open(files);
            let mut i = 0;
            g.bench_function(name, |b| {
                b.iter(|| {
                    let q = &queries[(i % 64) * dim..(i % 64 + 1) * dim];
                    i += 1;
                    r.search(0, q, &params, None).unwrap()
                })
            });
        };
        for enc in [
            VectorEncoding::F32,
            VectorEncoding::F16,
            VectorEncoding::Int8,
        ] {
            let p = IvfParams {
                rerank_encoding: enc,
                ..IvfParams::default()
            };
            let files = write(&data, dim, sim, &bbq, &p);
            run(
                format!("bbq rerank 200 {enc:?}"),
                &files,
                SearchParams::new(K, 16, 200),
            );
        }
        for (qname, factory) in [
            ("bbq", &bbq as &dyn QuantizerFactory),
            ("erq4", &erq as &dyn QuantizerFactory),
        ] {
            let files = write(&data, dim, sim, factory, &IvfParams::default());
            for (mode, rerank) in [
                ("fixed 0", Rerank::Fixed(0)),
                ("fixed 200", Rerank::Fixed(200)),
                ("bounded 200", Rerank::Bounded { cap: 200 }),
                ("auto", Rerank::Auto),
            ] {
                run(
                    format!("{qname} nprobe 16 {mode}"),
                    &files,
                    SearchParams {
                        k: K,
                        nprobe: Probes::Fixed(16),
                        rerank,
                    },
                );
            }
        }
        g.finish();
    }
}

/// Four 15k-vector segments searched one after another, each probing its
/// share of 16 clusters, with and without a threshold shared between them.
fn segments(c: &mut Criterion) {
    const N: usize = 15_000;
    let (dim, sim) = (100, Similarity::Cosine);
    let mut g = c.benchmark_group("segments 4x15k 100d cos");
    let readers: Vec<IvfReader> = (0..4)
        .map(|s| {
            let data = clustered(N, dim, 20 + s);
            open(&write(
                &data,
                dim,
                sim,
                &OsqFactory::bbq(),
                &IvfParams::default(),
            ))
        })
        .collect();
    let queries = clustered(64, dim, 30);
    for (name, share) in [("independent", false), ("shared threshold", true)] {
        let params = SearchParams {
            k: K,
            nprobe: Probes::Fixed(lucene_ivf::segment_nprobe(16, 4 * N, N)),
            rerank: Rerank::Bounded { cap: 100 },
        };
        let mut i = 0;
        g.bench_function(name, |b| {
            b.iter(|| {
                let q = &queries[(i % 64) * dim..(i % 64 + 1) * dim];
                i += 1;
                let shared = SharedThreshold::new();
                let mut hits = Vec::new();
                for r in &readers {
                    let out = r
                        .search_with(0, q, &params, None, share.then_some(&shared))
                        .unwrap();
                    hits.extend(out.hits);
                }
                hits
            })
        });
    }
    g.finish();
}

/// Building one 60k-vector field, and merging two 30k-vector segments by
/// reclustering and by reusing the larger one's centroids.
fn build(c: &mut Criterion) {
    const N: usize = 60_000;
    let mut g = c.benchmark_group("build");
    g.sample_size(10);
    for (dim, sim, label) in [
        (100, Similarity::Cosine, "100d cos"),
        (300, Similarity::Euclidean, "300d l2"),
    ] {
        let data = clustered(N, dim, 40);
        let bbq = OsqFactory::bbq();
        let params = IvfParams::default();
        g.bench_function(format!("write 60k {label}"), |b| {
            b.iter(|| write(&data, dim, sim, &bbq, &params))
        });
        let half = N / 2 * dim;
        let a = open(&write(&data[..half], dim, sim, &bbq, &params));
        let b2 = open(&write(&data[half..], dim, sim, &bbq, &params));
        let map_a = |d: i32| Some(d);
        let offset = (N / 2) as i32;
        let map_b = move |d: i32| Some(offset + d);
        let sources = [
            MergeSource {
                reader: &a,
                doc_map: &map_a,
            },
            MergeSource {
                reader: &b2,
                doc_map: &map_b,
            },
        ];
        for (name, policy) in [
            ("recluster", MergeClustering::Recluster),
            (
                "reuse",
                MergeClustering::ReuseLargest {
                    min_fraction: 0.5,
                    refine_rounds: 1,
                },
            ),
        ] {
            let p = IvfParams {
                merge_clustering: policy,
                ..params
            };
            g.bench_function(format!("merge 2x30k {label} {name}"), |b| {
                b.iter(|| {
                    let mut w = IvfWriter::new(&ID, "");
                    merge_field(&mut w, 0, &sources, &bbq, &p).unwrap();
                    w.finish()
                })
            });
        }
    }
    g.finish();
}

criterion_group!(benches, scorer, search, segments, build);
criterion_main!(benches);
