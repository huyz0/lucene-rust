//! Calibrated search depth, error-bounded rerank and thresholds shared
//! across segments.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_ivf::{
    segment_nprobe, Calibration, Error, FieldVectors, IvfFiles, IvfParams, IvfReader, IvfWriter,
    Probes, Rerank, ScoredDoc, SearchParams, SearchStats, SharedThreshold, Similarity,
    VectorEncoding, DEFAULT_PROBE_FRACTION, DEFAULT_RERANK_MULTIPLE,
};
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::scalar::ScalarFactory;
use lucene_vector_quant::{
    Quantizer, QuantizerFactory, QuantizerRegistry, QueryScorer, Result as QuantResult,
};

const ID: [u8; 16] = [7; 16];
const SIMS: [Similarity; 4] = [
    Similarity::Euclidean,
    Similarity::DotProduct,
    Similarity::Cosine,
    Similarity::MaximumInnerProduct,
];

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

/// `n` vectors around `centers` random centres, unit length for dot product.
fn dataset(n: usize, dim: usize, centers: usize, sim: Similarity, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let cs: Vec<Vec<f32>> = (0..centers)
        .map(|_| (0..dim).map(|_| rng.next() * 4.0).collect())
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &cs[i % centers];
        let mut v: Vec<f32> = c.iter().map(|x| x + rng.next() * 3.0).collect();
        if sim == Similarity::DotProduct {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
        }
        out.extend(v);
    }
    out
}

/// A quantizer that stores vectors exactly and estimates each score up to
/// `margin` below the exact one (by a hash of the code), so the exact
/// score is never above estimate plus `margin`; it reports `margin` as its
/// error bound when `report`.
#[derive(Debug, Clone, Copy)]
struct Noisy {
    margin: f32,
    report: bool,
}

const NOISY: &str = "noisy";

impl QuantizerFactory for Noisy {
    fn name(&self) -> &str {
        NOISY
    }
    fn train(&self, dim: usize, sim: Similarity, _: &[f32]) -> QuantResult<Box<dyn Quantizer>> {
        Ok(Box::new(NoisyQuantizer {
            dim,
            sim,
            noisy: *self,
        }))
    }
    fn read(&self, dim: usize, sim: Similarity, state: &[u8]) -> QuantResult<Box<dyn Quantizer>> {
        let margin = f32::from_le_bytes(state[..4].try_into().unwrap());
        Ok(Box::new(NoisyQuantizer {
            dim,
            sim,
            noisy: Noisy {
                margin,
                report: state[4] == 1,
            },
        }))
    }
}

#[derive(Debug)]
struct NoisyQuantizer {
    dim: usize,
    sim: Similarity,
    noisy: Noisy,
}

fn floats(code: &[u8]) -> Vec<f32> {
    code.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

impl Quantizer for NoisyQuantizer {
    fn name(&self) -> &str {
        NOISY
    }
    fn dimension(&self) -> usize {
        self.dim
    }
    fn similarity(&self) -> Similarity {
        self.sim
    }
    fn code_len(&self) -> usize {
        self.dim * 4
    }
    fn uses_centroid(&self) -> bool {
        false
    }
    fn encode(&self, vector: &[f32], _: &[f32], out: &mut [u8]) {
        for (o, x) in out.chunks_exact_mut(4).zip(vector) {
            o.copy_from_slice(&x.to_le_bytes());
        }
    }
    fn query_scorer(&self, query: &[f32]) -> Box<dyn QueryScorer + '_> {
        Box::new(NoisyScorer {
            q: self,
            query: query.to_vec(),
        })
    }
    fn state(&self) -> Vec<u8> {
        let mut s = self.noisy.margin.to_le_bytes().to_vec();
        s.push(u8::from(self.noisy.report));
        s
    }
}

struct NoisyScorer<'q> {
    q: &'q NoisyQuantizer,
    query: Vec<f32>,
}

impl QueryScorer for NoisyScorer<'_> {
    fn set_centroid(&mut self, _: &[f32]) {}
    fn score(&self, code: &[u8]) -> f32 {
        let exact = self.q.sim.score(&self.query, &floats(code));
        let margin = self.q.noisy.margin;
        if !margin.is_finite() {
            return exact;
        }
        let hash = code.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        exact - margin * 0.9 * ((hash >> 40) as f32 / (1u64 << 24) as f32)
    }
    fn error_bound(&self, _: &[u8], _: usize, _: f32) -> Option<f32> {
        self.q.noisy.report.then_some(self.q.noisy.margin)
    }
}

fn registry() -> QuantizerRegistry {
    let mut r = QuantizerRegistry::with_builtins();
    r.register(Arc::new(Noisy {
        margin: 0.0,
        report: false,
    }));
    r
}

fn write(
    sim: Similarity,
    dim: usize,
    docs: &[i32],
    vectors: &[f32],
    factory: &dyn QuantizerFactory,
    params: &IvfParams,
) -> IvfFiles {
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: sim,
            dimension: dim,
            docs,
            vectors,
        },
        factory,
        params,
    )
    .unwrap();
    w.finish()
}

fn params(target: usize) -> IvfParams {
    IvfParams {
        target_cluster_size: target,
        ..IvfParams::default()
    }
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

fn recall(found: &[ScoredDoc], truth: &[ScoredDoc]) -> usize {
    found
        .iter()
        .filter(|f| truth.iter().any(|t| t.doc == f.doc))
        .count()
}

#[test]
fn fixed_rerank_is_an_exact_rerank_of_the_best_estimates() {
    let (dim, n) = (16, 2000);
    for sim in SIMS {
        let data = dataset(n, dim, 12, sim, 3);
        let docs: Vec<i32> = (0..n as i32).collect();
        let r = open(&write(
            sim,
            dim,
            &docs,
            &data,
            &OsqFactory::bbq(),
            &params(100),
        ));
        for q in dataset(5, dim, 12, sim, 4).chunks_exact(dim) {
            for depth in [10, 37, 120] {
                // The candidates: the best `depth` estimates.
                let estimates = r
                    .search(1, q, &SearchParams::new(depth, 4, 0), None)
                    .unwrap();
                let mut qq = q.to_vec();
                if sim == Similarity::Cosine {
                    lucene_util::vector_util::l2normalize(&mut qq, false).unwrap();
                }
                let mut exact: Vec<ScoredDoc> = estimates
                    .iter()
                    .map(|e| ScoredDoc {
                        doc: e.doc,
                        score: sim.score(&qq, &r.vector(1, e.doc).unwrap().unwrap()),
                    })
                    .collect();
                exact.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
                exact.truncate(10);
                let p = SearchParams::new(10, 4, depth);
                let out = r.search_with(1, q, &p, None, None).unwrap();
                assert_eq!(out.hits, exact, "{sim:?} depth {depth}");
                assert_eq!(out.hits, r.search(1, q, &p, None).unwrap());
                assert_eq!(out.stats.candidates, depth.max(10));
                assert_eq!(out.stats.reranked, out.stats.candidates);
                assert_eq!(out.stats.clusters_scanned, 4);
                // A fresh shared threshold changes nothing.
                let fresh = SharedThreshold::new();
                assert_eq!(
                    r.search_with(1, q, &p, None, Some(&fresh)).unwrap().hits,
                    exact
                );
            }
        }
    }
}

#[test]
fn bounded_rerank_without_finite_bounds_is_fixed() {
    let (dim, n) = (12, 1500);
    let sim = Similarity::Euclidean;
    let data = dataset(n, dim, 8, sim, 5);
    let docs: Vec<i32> = (0..n as i32).collect();
    for noisy in [
        Noisy {
            margin: f32::INFINITY,
            report: true,
        },
        Noisy {
            margin: 0.05,
            report: false,
        },
    ] {
        let r = open(&write(sim, dim, &docs, &data, &noisy, &params(100)));
        for q in dataset(8, dim, 8, sim, 6).chunks_exact(dim) {
            for cap in [0, 5, 30, 200] {
                let fixed = r
                    .search_with(1, q, &SearchParams::new(10, 5, cap), None, None)
                    .unwrap();
                let bounded = r
                    .search_with(
                        1,
                        q,
                        &SearchParams {
                            rerank: Rerank::Bounded { cap },
                            ..SearchParams::new(10, 5, cap)
                        },
                        None,
                        None,
                    )
                    .unwrap();
                assert_eq!(bounded, fixed, "{noisy:?} cap {cap}");
            }
        }
    }
}

#[test]
fn early_stop_with_valid_bounds_keeps_the_hits() {
    let (dim, n) = (12, 3000);
    for sim in SIMS {
        let data = dataset(n, dim, 10, sim, 7);
        let docs: Vec<i32> = (0..n as i32).collect();
        let noisy = Noisy {
            margin: 0.01,
            report: true,
        };
        let r = open(&write(sim, dim, &docs, &data, &noisy, &params(150)));
        let (mut fixed_work, mut bounded_work) = (0, 0);
        for q in dataset(20, dim, 10, sim, 8).chunks_exact(dim) {
            let fixed = r
                .search_with(1, q, &SearchParams::new(10, 6, 300), None, None)
                .unwrap();
            let p = SearchParams {
                rerank: Rerank::Bounded { cap: 300 },
                ..SearchParams::new(10, 6, 300)
            };
            let bounded = r.search_with(1, q, &p, None, None).unwrap();
            assert_eq!(bounded.hits, fixed.hits, "{sim:?}");
            fixed_work += fixed.stats.reranked;
            bounded_work += bounded.stats.reranked;
        }
        assert!(
            bounded_work * 2 < fixed_work,
            "{sim:?}: {bounded_work} of {fixed_work} reranked"
        );
    }
}

#[test]
fn auto_reaches_deep_fixed_recall_with_less_rerank() {
    let (dim, n) = (32, 6000);
    let factories: Vec<Box<dyn QuantizerFactory>> = vec![
        Box::new(OsqFactory::bbq()),
        Box::new(ExRabitqFactory::new(4)),
        Box::new(ScalarFactory::int7()),
    ];
    for sim in [Similarity::Euclidean, Similarity::Cosine] {
        let data = dataset(n, dim, 30, sim, 9);
        let docs: Vec<i32> = (0..n as i32).collect();
        // Queries from the data's distribution: perturbed data vectors.
        let mut rng = Rng(10);
        let queries: Vec<f32> = data
            .chunks_exact(dim)
            .step_by(n / 40)
            .flat_map(|v| v.iter().map(|x| x + rng.next() * 2.0).collect::<Vec<_>>())
            .collect();
        for factory in &factories {
            let r = open(&write(
                sim,
                dim,
                &docs,
                &data,
                factory.as_ref(),
                &params(150),
            ));
            let f = r.field(1).unwrap();
            let c = f.calibration().expect("calibrated");
            assert!(c.probe_fraction > 0.0 && c.probe_fraction <= 1.0);
            assert!(c.rerank_multiple >= 1.0);
            let deep = SearchParams {
                k: 10,
                nprobe: Probes::Auto,
                rerank: Rerank::Fixed(300),
            };
            let (mut deep_hits, mut auto_hits) = (0, 0);
            let (mut deep_work, mut auto_work) = (0, 0);
            for q in queries.chunks_exact(dim) {
                let truth = r.exact_search(1, q, 10, None).unwrap();
                let d = r.search_with(1, q, &deep, None, None).unwrap();
                let a = r
                    .search_with(1, q, &SearchParams::auto(10), None, None)
                    .unwrap();
                assert_eq!(a.stats.clusters_scanned, f.auto_nprobe());
                assert!(a.stats.reranked <= f.auto_rerank(10));
                deep_hits += recall(&d.hits, &truth);
                auto_hits += recall(&a.hits, &truth);
                deep_work += d.stats.reranked;
                auto_work += a.stats.reranked;
            }
            let label = format!("{sim:?} {}: {c:?}", factory.name());
            assert!(
                auto_hits as f64 >= deep_hits as f64 * 0.97,
                "{label}: auto {auto_hits} vs deep {deep_hits} hits"
            );
            assert!(
                auto_work * 2 < deep_work,
                "{label}: auto reranked {auto_work}, deep {deep_work}"
            );
            assert!(
                auto_hits as f64 >= 0.9 * 400.0,
                "{label}: auto recall {auto_hits} of 400"
            );
        }
    }
}

#[test]
fn calibration_is_deterministic_and_optional() {
    let (dim, n) = (16, 2500);
    let sim = Similarity::DotProduct;
    let data = dataset(n, dim, 15, sim, 11);
    let docs: Vec<i32> = (0..n as i32).collect();
    let bbq = OsqFactory::bbq();
    let a = write(sim, dim, &docs, &data, &bbq, &params(100));
    let b = write(sim, dim, &docs, &data, &bbq, &params(100));
    assert_eq!(a, b);
    let ra = open(&a);
    let fa = ra.field(1).unwrap();
    let c = fa.calibration().unwrap();
    // nprobe is a whole number of clusters.
    let probes = c.probe_fraction * fa.clusters() as f32;
    assert!((probes - probes.round()).abs() < 1e-3, "{c:?}");
    assert_eq!(fa.auto_nprobe(), probes.round() as usize);
    assert_eq!(
        fa.auto_rerank(10),
        (c.rerank_multiple * 10.0).ceil() as usize
    );
    // Radii cover every vector of their cluster.
    for cl in 0..fa.clusters() {
        assert!(fa.radius(cl).unwrap() >= 0.0);
    }

    // A higher target probes and reranks no less.
    let strict = open(&write(
        sim,
        dim,
        &docs,
        &data,
        &bbq,
        &IvfParams {
            calibration_recall: Some(1.0),
            ..params(100)
        },
    ));
    let cs = strict.field(1).unwrap().calibration().unwrap();
    assert!(cs.probe_fraction >= c.probe_fraction, "{cs:?} vs {c:?}");
    let loose = open(&write(
        sim,
        dim,
        &docs,
        &data,
        &bbq,
        &IvfParams {
            calibration_recall: Some(0.3),
            ..params(100)
        },
    ));
    let cl = loose.field(1).unwrap().calibration().unwrap();
    assert!(cl.probe_fraction <= c.probe_fraction, "{cl:?} vs {c:?}");

    // Without calibration, Auto falls back to the defaults.
    let off = open(&write(
        sim,
        dim,
        &docs,
        &data,
        &bbq,
        &IvfParams {
            calibration_recall: None,
            ..params(100)
        },
    ));
    let f = off.field(1).unwrap();
    assert_eq!(f.calibration(), None);
    assert_eq!(
        f.auto_nprobe(),
        ((DEFAULT_PROBE_FRACTION * f.clusters() as f32).ceil() as usize).max(1)
    );
    assert_eq!(f.auto_rerank(7), (DEFAULT_RERANK_MULTIPLE * 7.0) as usize);
    assert_eq!(f.auto_rerank(0), 0);
    assert_eq!(f.auto_rerank(usize::MAX), usize::MAX);

    // Out of range targets are refused.
    for bad in [0.0, -1.0, 1.5, f32::NAN] {
        let mut w = IvfWriter::new(&ID, "");
        let err = w.add_field(
            &FieldVectors {
                number: 1,
                similarity: sim,
                dimension: dim,
                docs: &docs,
                vectors: &data,
            },
            &bbq,
            &IvfParams {
                calibration_recall: Some(bad),
                ..params(100)
            },
        );
        assert!(matches!(err, Err(Error::IllegalArgument(_))), "{bad}");
    }

    // Fields too small to sample are not calibrated, and spilled fields are.
    for (n, spill) in [(1usize, None), (2, None), (40, Some(1.0))] {
        let docs: Vec<i32> = (0..n as i32).collect();
        let r = open(&write(
            sim,
            dim,
            &docs,
            &data[..n * dim],
            &bbq,
            &IvfParams {
                spill,
                ..params(10)
            },
        ));
        let f = r.field(1).unwrap();
        assert_eq!(f.calibration().is_some(), n >= 2, "{n}");
        let hits = r
            .search(1, &data[..dim], &SearchParams::auto(3), None)
            .unwrap();
        assert_eq!(hits.len(), 3.min(n));
        assert_eq!(hits[0].doc, 0);
    }
}

#[test]
fn radii_bound_every_cluster_member() {
    for (sim, encoding) in [
        (Similarity::Euclidean, VectorEncoding::Int8),
        (Similarity::Cosine, VectorEncoding::Bf16),
        (Similarity::MaximumInnerProduct, VectorEncoding::F16),
        (Similarity::DotProduct, VectorEncoding::F32),
    ] {
        let (dim, n) = (8, 800);
        let data = dataset(n, dim, 6, sim, 12);
        let docs: Vec<i32> = (0..n as i32).collect();
        let r = open(&write(
            sim,
            dim,
            &docs,
            &data,
            &OsqFactory::bbq(),
            &IvfParams {
                rerank_encoding: encoding,
                spill: Some(1.0),
                ..params(60)
            },
        ));
        let f = r.field(1).unwrap();
        for c in 0..f.clusters() {
            let (members, _) = r.cluster(1, c).unwrap();
            let radius = f.radius(c).unwrap();
            for doc in members {
                let mut v = r.vector(1, doc).unwrap().unwrap();
                if sim == Similarity::Cosine {
                    lucene_util::vector_util::l2normalize(&mut v, false).unwrap();
                }
                let d: f32 = v
                    .iter()
                    .zip(f.centroid(c))
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    d <= radius * 1.0001 + 1e-6,
                    "{sim:?} cluster {c}: {d} > {radius}"
                );
            }
        }
    }
}

/// Searches `readers` one after another sharing a threshold, and each on
/// its own, and returns both merged top `k`s and the shared run's work.
fn shared_and_alone(
    readers: &[IvfReader],
    q: &[f32],
    p: &SearchParams,
) -> (Vec<ScoredDoc>, Vec<ScoredDoc>, SearchStats) {
    let shared = SharedThreshold::new();
    let (mut with, mut without) = (Vec::new(), Vec::new());
    let mut work = SearchStats::default();
    for r in readers {
        let out = r.search_with(1, q, p, None, Some(&shared)).unwrap();
        work.clusters_skipped += out.stats.clusters_skipped;
        work.clusters_scanned += out.stats.clusters_scanned;
        work.reranked += out.stats.reranked;
        with.extend(out.hits);
        without.extend(r.search(1, q, p, None).unwrap());
    }
    let top = |mut v: Vec<ScoredDoc>| {
        v.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
        v.truncate(p.k);
        v
    };
    (top(with), top(without), work)
}

#[test]
fn shared_thresholds_keep_the_merged_hits() {
    let dim = 10;
    for sim in SIMS {
        let all = dataset(4000, dim, 16, sim, 13);
        // Three segments of different sizes, documents numbered globally.
        let mut readers = Vec::new();
        let mut start = 0;
        for (i, size) in [2000usize, 1200, 800].into_iter().enumerate() {
            let docs: Vec<i32> = (start as i32..(start + size) as i32).collect();
            let factory: Box<dyn QuantizerFactory> = if i == 1 {
                Box::new(Noisy {
                    margin: 0.02,
                    report: true,
                })
            } else {
                Box::new(OsqFactory::bbq())
            };
            readers.push(open(&write(
                sim,
                dim,
                &docs,
                &all[start * dim..(start + size) * dim],
                factory.as_ref(),
                &params(80),
            )));
            start += size;
        }
        let mut skipped = 0;
        for q in dataset(12, dim, 16, sim, 14).chunks_exact(dim) {
            // Every cluster, every candidate: the exact top k either way.
            let full = SearchParams::new(10, usize::MAX, usize::MAX);
            let (with, without, work) = shared_and_alone(&readers, q, &full);
            assert_eq!(with, without, "{sim:?}");
            skipped += work.clusters_skipped;
            // Bounded rerank with sound bounds in the noisy segment.
            let bounded = SearchParams {
                rerank: Rerank::Bounded { cap: 4000 },
                ..full
            };
            let (with, _, _) = shared_and_alone(&readers[1..2], q, &bounded);
            let (alone, _, _) = shared_and_alone(&readers[1..2], q, &full);
            assert_eq!(with, alone, "{sim:?}");
        }
        assert!(skipped > 0, "{sim:?}: no cluster skipped");
    }
}

#[test]
fn version_two_fields_have_no_radii_or_calibration() {
    let (dim, n) = (8, 700);
    let sim = Similarity::Euclidean;
    let data = dataset(n, dim, 6, sim, 15);
    let docs: Vec<i32> = (0..n as i32).collect();
    let files = write(sim, dim, &docs, &data, &OsqFactory::bbq(), &params(100));
    let current = open(&files);
    let clusters = current.field(1).unwrap().clusters();
    let header = |codec| lucene_store::codec_util::index_header_length(codec, "");
    let rewrite = |file: &[u8], codec, version: i32, strip: usize| {
        let mut payload = file[header(codec)..file.len() - 16].to_vec();
        payload.truncate(payload.len() - strip);
        let mut out = Vec::new();
        lucene_store::codec_util::write_index_header(&mut out, codec, version, &ID, "");
        out.extend_from_slice(&payload);
        lucene_store::codec_util::write_footer(&mut out);
        out
    };
    let v2 = lucene_ivf::VERSION_VECTOR_ENCODING;
    // Radii, calibration, and the zero routing parent count.
    let meta = rewrite(
        &files.meta,
        lucene_ivf::META_CODEC,
        v2,
        clusters * 4 + 9 + 1,
    );
    let postings = rewrite(&files.postings, lucene_ivf::POSTINGS_CODEC, v2, 0);
    let vectors = rewrite(&files.vectors, lucene_ivf::VECTORS_CODEC, v2, 0);
    let old = IvfReader::open(
        &meta,
        Arc::new(postings),
        Arc::new(vectors),
        &ID,
        "",
        &registry(),
    )
    .unwrap();
    let f = old.field(1).unwrap();
    assert_eq!((f.calibration(), f.radius(0)), (None, None));
    let q = &data[..dim];
    for p in [
        SearchParams::new(10, 3, 40),
        SearchParams::auto(10),
        SearchParams::new(10, usize::MAX, usize::MAX),
    ] {
        let shared = SharedThreshold::new();
        shared.raise(0.5);
        let out = old.search_with(1, q, &p, None, Some(&shared)).unwrap();
        // No radii: no cluster can be ruled out.
        assert_eq!(out.stats.clusters_skipped, 0);
        if p.rerank == Rerank::Fixed(40) {
            assert_eq!(out.hits, current.search(1, q, &p, None).unwrap());
        }
    }
}

#[test]
fn corrupt_radii_and_calibration_are_rejected() {
    let (dim, n) = (6, 400);
    let sim = Similarity::Cosine;
    let data = dataset(n, dim, 5, sim, 16);
    let docs: Vec<i32> = (0..n as i32).collect();
    let files = write(sim, dim, &docs, &data, &OsqFactory::bbq(), &params(100));
    let clusters = open(&files).field(1).unwrap().clusters();
    let edit = |change: &dyn Fn(&mut Vec<u8>)| {
        let h = lucene_store::codec_util::index_header_length(lucene_ivf::META_CODEC, "");
        let mut payload = files.meta[h..files.meta.len() - 16].to_vec();
        change(&mut payload);
        let mut out = Vec::new();
        lucene_store::codec_util::write_index_header(
            &mut out,
            lucene_ivf::META_CODEC,
            lucene_ivf::VERSION_CURRENT,
            &ID,
            "",
        );
        out.extend_from_slice(&payload);
        lucene_store::codec_util::write_footer(&mut out);
        IvfReader::open(
            &out,
            Arc::new(files.postings.clone()),
            Arc::new(files.vectors.clone()),
            &ID,
            "",
            &registry(),
        )
    };
    edit(&|_| {}).unwrap();
    // The payload ends with the calibration (9 bytes) and a zero routing
    // parent count (1).
    let flag_at = |p: &Vec<u8>| p.len() - 10;
    let radius_at = |p: &Vec<u8>| p.len() - 10 - clusters * 4;
    let set_f32 = |p: &mut Vec<u8>, at: usize, x: f32| {
        p[at..at + 4].copy_from_slice(&x.to_le_bytes());
    };
    type Change<'a> = Box<dyn Fn(&mut Vec<u8>) + 'a>;
    let cases: Vec<(Change<'_>, &str)> = vec![
        (Box::new(|p| set_f32(p, radius_at(p), -1.0)), "radius"),
        (Box::new(|p| set_f32(p, radius_at(p), f32::NAN)), "radius"),
        (
            Box::new(|p| {
                let at = flag_at(p);
                p[at] = 2;
            }),
            "calibration flag",
        ),
        (Box::new(|p| set_f32(p, flag_at(p) + 1, 0.0)), "calibration"),
        (Box::new(|p| set_f32(p, flag_at(p) + 5, 0.5)), "calibration"),
        (
            Box::new(|p| {
                let at = radius_at(p);
                p.truncate(at + 2);
            }),
            "past the end",
        ),
    ];
    for (change, message) in cases {
        match edit(&*change) {
            Err(Error::Corrupted(m)) => assert!(m.contains(message), "{m}"),
            Err(e) => assert!(message == "past the end", "{e}"),
            Ok(_) => panic!("{message} accepted"),
        }
    }
    // A flag of 0 drops the calibration.
    let r = edit(&|p| {
        let at = flag_at(p);
        p[at] = 0;
        p.truncate(at + 1);
        p.push(0);
    })
    .unwrap();
    assert_eq!(r.field(1).unwrap().calibration(), None);
}

#[test]
fn segment_probe_budgets() {
    assert_eq!(segment_nprobe(64, 1000, 250), 16);
    assert_eq!(segment_nprobe(64, 1000, 1), 1);
    assert_eq!(segment_nprobe(10, 3, 1), 4);
    assert_eq!(segment_nprobe(64, 1000, 5000), 64);
    assert_eq!(segment_nprobe(0, 1000, 500), 1);
    assert_eq!(segment_nprobe(7, 0, 0), 7);
    assert_eq!(
        segment_nprobe(usize::MAX, usize::MAX, usize::MAX),
        usize::MAX
    );
}

#[test]
fn shared_thresholds_only_rise() {
    let t = SharedThreshold::default();
    assert_eq!(t.get(), f32::NEG_INFINITY);
    for (raise, expect) in [
        (-3.5, -3.5),
        (-7.0, -3.5),
        (f32::NAN, -3.5),
        (-0.0, -0.0),
        (0.25, 0.25),
        (0.125, 0.25),
        (2.0, 2.0),
        (f32::INFINITY, f32::INFINITY),
    ] {
        t.raise(raise);
        assert_eq!(t.get().to_bits(), f32::to_bits(expect), "{raise}");
    }
    let c = Calibration {
        probe_fraction: 0.5,
        rerank_multiple: 2.0,
    };
    assert_eq!(c, c);
}
