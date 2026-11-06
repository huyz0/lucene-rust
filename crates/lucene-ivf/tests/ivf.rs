//! Building, reading, searching and merging IVF indexes end to end.
#![allow(clippy::arithmetic_side_effects)]

use std::borrow::Cow;
use std::sync::Arc;

use lucene_ivf::kmeans::{Clustering, KMeansParams};
use lucene_ivf::{
    merge_field, Error, FieldVectors, IvfParams, IvfReader, IvfWriter, MergeClustering,
    MergeSource, RangeSource, ScoredDoc, SearchParams, Similarity, VectorEncoding,
};
use lucene_util::quantization::ScalarEncoding;
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
use lucene_vector_quant::rotation::RotatedFactory;
use lucene_vector_quant::scalar::ScalarFactory;
use lucene_vector_quant::{QuantizerFactory, QuantizerRegistry};

const ID: [u8; 16] = [9; 16];
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
        let mut v: Vec<f32> = c.iter().map(|x| x + rng.next()).collect();
        if sim == Similarity::DotProduct {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
        }
        out.extend(v);
    }
    out
}

fn open(files: &lucene_ivf::IvfFiles) -> IvfReader {
    IvfReader::open(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
    .unwrap()
}

fn build(
    number: i32,
    sim: Similarity,
    dim: usize,
    docs: &[i32],
    vectors: &[f32],
    factory: &dyn QuantizerFactory,
    target: usize,
) -> lucene_ivf::IvfFiles {
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number,
            similarity: sim,
            dimension: dim,
            docs,
            vectors,
        },
        factory,
        &IvfParams {
            target_cluster_size: target,
            ..IvfParams::default()
        },
    )
    .unwrap();
    w.finish()
}

fn recall(found: &[ScoredDoc], truth: &[ScoredDoc]) -> f64 {
    let hits = found
        .iter()
        .filter(|f| truth.iter().any(|t| t.doc == f.doc))
        .count();
    hits as f64 / truth.len() as f64
}

#[test]
fn every_quantizer_and_similarity_finds_the_neighbours() {
    let dim = 32;
    let n = 3000;
    let factories: Vec<Box<dyn QuantizerFactory>> = vec![
        Box::new(OsqFactory::bbq()),
        Box::new(OsqFactory::new(ScalarEncoding::DibitQueryNibble)),
        Box::new(OsqFactory::new(ScalarEncoding::PackedNibble)),
        Box::new(OsqFactory::new(ScalarEncoding::SevenBit)),
        Box::new(ScalarFactory::int7()),
        Box::new(ScalarFactory::int4()),
        Box::new(ExRabitqFactory::new(2)),
        Box::new(ExRabitqFactory::new(5)),
        Box::new(RotatedFactory::new(Arc::new(OsqFactory::bbq()))),
        Box::new(RotatedFactory::new(Arc::new(ScalarFactory::int7()))),
    ];
    for sim in SIMS {
        let data = dataset(n, dim, 20, sim, 11);
        let docs: Vec<i32> = (0..n as i32).map(|d| d * 2).collect();
        let queries = dataset(20, dim, 20, sim, 77);
        for factory in &factories {
            let files = build(5, sim, dim, &docs, &data, factory.as_ref(), 100);
            let r = open(&files);
            r.check_integrity().unwrap();
            let f = r.field(5).unwrap();
            assert_eq!((f.similarity(), f.dimension(), f.count()), (sim, dim, n));
            assert!(f.clusters() >= 30, "{} clusters", f.clusters());
            let mut total = 0.0;
            let mut total_est = 0.0;
            for q in queries.chunks_exact(dim) {
                let truth = r.exact_search(5, q, 10, None).unwrap();
                let hits = r.search(5, q, &SearchParams::new(10, 8, 50), None).unwrap();
                assert_eq!(hits.len(), 10);
                assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
                // Reranked scores are exact.
                for h in &hits {
                    let v = r.vector(5, h.doc).unwrap().unwrap();
                    let mut qq = q.to_vec();
                    if sim == Similarity::Cosine {
                        lucene_util::vector_util::l2normalize(&mut qq, false).unwrap();
                    }
                    assert_eq!(h.score, sim.score(&qq, &v));
                }
                total += recall(&hits, &truth);
                let est = r.search(5, q, &SearchParams::new(10, 8, 0), None).unwrap();
                total_est += recall(&est, &truth);
            }
            let avg = total / 20.0;
            let avg_est = total_est / 20.0;
            assert!(avg >= 0.9, "{sim:?} {factory:?}: recall {avg}");
            assert!(
                avg_est >= 0.4,
                "{sim:?} {factory:?}: estimated recall {avg_est}"
            );
        }
    }
}

#[test]
fn filters_apply_while_scanning() {
    let dim = 16;
    let data = dataset(2000, dim, 10, Similarity::Euclidean, 3);
    let docs: Vec<i32> = (0..2000).collect();
    let files = build(
        1,
        Similarity::Euclidean,
        dim,
        &docs,
        &data,
        &OsqFactory::bbq(),
        64,
    );
    let r = open(&files);
    let even = |d: i32| d % 2 == 0;
    let q = &data[..dim];
    let hits = r
        .search(1, q, &SearchParams::new(10, 100, 40), Some(&even))
        .unwrap();
    assert_eq!(hits.len(), 10);
    assert!(hits.iter().all(|h| h.doc % 2 == 0));
    let exact = r.exact_search(1, q, 10, Some(&even)).unwrap();
    assert!(exact.iter().all(|h| h.doc % 2 == 0));
    assert_eq!(exact[0].doc, 0);
    assert!(recall(&hits, &exact) >= 0.9);
    let none = |_: i32| false;
    assert!(r
        .search(1, q, &SearchParams::new(10, 4, 10), Some(&none))
        .unwrap()
        .is_empty());
}

#[test]
fn small_fields_are_one_cluster_and_search_exactly() {
    let dim = 8;
    let data = dataset(50, dim, 3, Similarity::Euclidean, 5);
    let docs: Vec<i32> = (0..50).collect();
    let files = build(
        2,
        Similarity::Euclidean,
        dim,
        &docs,
        &data,
        &ScalarFactory::int7(),
        384,
    );
    let r = open(&files);
    let f = r.field(2).unwrap();
    assert_eq!(f.clusters(), 1);
    assert_eq!(f.cluster_size(0), 50);
    assert_eq!(f.centroid(0).len(), dim);
    assert_eq!(f.quantizer().name(), "sq");
    assert_eq!(f.number(), 2);
    let (cdocs, codes) = r.cluster(2, 0).unwrap();
    assert_eq!(cdocs, docs);
    assert_eq!(codes.len(), f.quantizer().block_len(50));
    let q = &data[3 * dim..4 * dim];
    let hits = r.search(2, q, &SearchParams::new(5, 1, 50), None).unwrap();
    assert_eq!(hits, r.exact_search(2, q, 5, None).unwrap());
    assert_eq!(hits[0].doc, 3);
    // k = 0 and a k past the vector count.
    assert!(r
        .search(2, q, &SearchParams::new(0, 1, 5), None)
        .unwrap()
        .is_empty());
    assert_eq!(
        r.search(2, q, &SearchParams::new(80, 1, 0), None)
            .unwrap()
            .len(),
        50
    );
    assert_eq!(r.vector(2, 999).unwrap(), None);
    assert_eq!(r.slot_vector(2, 0).unwrap().len(), dim);
}

#[test]
fn several_fields_and_an_empty_one() {
    let dim = 4;
    let a = dataset(300, dim, 4, Similarity::Euclidean, 1);
    let b = dataset(200, 6, 4, Similarity::DotProduct, 2);
    let docs_a: Vec<i32> = (0..300).collect();
    let docs_b: Vec<i32> = (0..200).map(|d| d * 3).collect();
    let mut w = IvfWriter::new(&ID, "");
    let p = IvfParams {
        target_cluster_size: 40,
        ..IvfParams::default()
    };
    for (number, sim, d, docs, v) in [
        (1, Similarity::Euclidean, dim, &docs_a[..], &a[..]),
        (7, Similarity::DotProduct, 6, &docs_b[..], &b[..]),
        (9, Similarity::Cosine, 3, &[][..], &[][..]),
    ] {
        w.add_field(
            &FieldVectors {
                number,
                similarity: sim,
                dimension: d,
                docs,
                vectors: v,
            },
            &OsqFactory::bbq(),
            &p,
        )
        .unwrap();
    }
    let r = open(&w.finish());
    assert_eq!(r.fields().len(), 3);
    assert_eq!(r.field(7).unwrap().count(), 200);
    let empty = r.field(9).unwrap();
    assert_eq!((empty.count(), empty.clusters()), (0, 0));
    assert!(r
        .search(9, &[1.0, 0.0, 0.0], &SearchParams::new(3, 2, 3), None)
        .unwrap()
        .is_empty());
    assert!(r.doc_table(9).unwrap().is_empty());
    assert!(format!("{r:?}").contains("IvfReader"));
    // Bad calls.
    assert!(matches!(
        r.search(4, &[0.0; 4], &SearchParams::new(1, 1, 1), None),
        Err(Error::IllegalArgument(_))
    ));
    assert!(r
        .search(1, &[0.0; 3], &SearchParams::new(1, 1, 1), None)
        .is_err());
    assert!(r
        .search(1, &[f32::NAN; 4], &SearchParams::new(1, 1, 1), None)
        .is_err());
    assert!(r.cluster(1, 10_000).is_err());
    assert!(r.slot_vector(1, 10_000).is_err());
}

#[test]
fn merge_reclusters_live_vectors_under_new_numbers() {
    let dim = 12;
    let a = dataset(800, dim, 8, Similarity::Cosine, 21);
    let b = dataset(600, dim, 8, Similarity::Cosine, 22);
    let docs_a: Vec<i32> = (0..800).collect();
    let docs_b: Vec<i32> = (0..600).collect();
    let f = OsqFactory::new(ScalarEncoding::PackedNibble);
    let ra = open(&build(3, Similarity::Cosine, dim, &docs_a, &a, &f, 100));
    let rb = open(&build(3, Similarity::Cosine, dim, &docs_b, &b, &f, 100));
    // Drop every tenth document of `a`; `b` follows `a`'s 720 live ones.
    let map_a = |d: i32| (d % 10 != 0).then(|| d - d / 10 - 1);
    let map_b = |d: i32| Some(720 + d);
    let mut w = IvfWriter::new(&ID, "");
    let sources = [
        MergeSource {
            reader: &ra,
            doc_map: &map_a,
        },
        MergeSource {
            reader: &rb,
            doc_map: &map_b,
        },
    ];
    assert!(format!("{:?}", sources[0]).contains("MergeSource"));
    let p = IvfParams {
        target_cluster_size: 100,
        merge_clustering: MergeClustering::Recluster,
        ..IvfParams::default()
    };
    merge_field(&mut w, 3, &sources, &f, &p).unwrap();
    // A field no source has writes nothing.
    merge_field(&mut w, 4, &sources, &f, &p).unwrap();
    let merged = open(&w.finish());
    assert_eq!(merged.fields().len(), 1);
    let field = merged.field(3).unwrap();
    assert_eq!(field.count(), 1320);
    assert!(field.clusters() >= 14);
    // Document 721 of the merged segment is `b`'s document 1.
    let v = merged.vector(3, 721).unwrap().unwrap();
    assert_eq!(v, rb.vector(3, 1).unwrap().unwrap());
    let hits = merged
        .search(3, &v, &SearchParams::new(1, 4, 10), None)
        .unwrap();
    assert_eq!(hits[0].doc, 721);
    // Disagreeing and colliding sources.
    let rc = open(&build(3, Similarity::Euclidean, dim, &docs_b, &b, &f, 100));
    let mut w = IvfWriter::new(&ID, "");
    let same = |d: i32| Some(d);
    assert!(merge_field(
        &mut w,
        3,
        &[
            MergeSource {
                reader: &rb,
                doc_map: &same
            },
            MergeSource {
                reader: &rc,
                doc_map: &same
            },
        ],
        &f,
        &p
    )
    .is_err());
    assert!(merge_field(
        &mut w,
        3,
        &[
            MergeSource {
                reader: &rb,
                doc_map: &same
            },
            MergeSource {
                reader: &rb,
                doc_map: &same
            },
        ],
        &f,
        &p
    )
    .is_err());
}

/// A range source that copies, as a blob cache would.
struct Copying(Vec<u8>);
impl RangeSource for Copying {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn read(&self, offset: usize, len: usize) -> lucene_ivf::Result<Cow<'_, [u8]>> {
        Ok(Cow::Owned(self.0[offset..offset + len].to_vec()))
    }
}

#[test]
fn custom_range_sources_and_unaligned_vectors() {
    let dim = 5;
    let data = dataset(400, dim, 5, Similarity::MaximumInnerProduct, 8);
    let docs: Vec<i32> = (0..400).collect();
    let files = build(
        1,
        Similarity::MaximumInnerProduct,
        dim,
        &docs,
        &data,
        &ScalarFactory::int4(),
        50,
    );
    let src = Copying(files.vectors.clone());
    assert!(!src.is_empty());
    let r = IvfReader::open(
        &files.meta,
        Arc::new(Copying(files.postings.clone())),
        Arc::new(src),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
    .unwrap();
    r.check_integrity().unwrap();
    let q = &data[..dim];
    let a = r.search(1, q, &SearchParams::new(5, 3, 20), None).unwrap();
    assert_eq!(
        a,
        open(&files)
            .search(1, q, &SearchParams::new(5, 3, 20), None)
            .unwrap()
    );
    // Vectors read from an odd address decode the slow way, same values.
    let mut shifted = vec![0u8];
    shifted.extend_from_slice(&files.vectors);
    let shifted: Arc<[u8]> = shifted.into();
    struct Offset(Arc<[u8]>);
    impl RangeSource for Offset {
        fn len(&self) -> usize {
            self.0.len() - 1
        }
        fn read(&self, offset: usize, len: usize) -> lucene_ivf::Result<Cow<'_, [u8]>> {
            Ok(Cow::Borrowed(&self.0[offset + 1..offset + 1 + len]))
        }
    }
    let r2 = IvfReader::open(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(Offset(shifted)),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    )
    .unwrap();
    assert_eq!(
        r2.search(1, q, &SearchParams::new(5, 3, 20), None).unwrap(),
        a
    );
}

#[test]
fn corruption_is_detected() {
    let dim = 4;
    let data = dataset(200, dim, 4, Similarity::Euclidean, 4);
    let docs: Vec<i32> = (0..200).collect();
    let files = build(
        1,
        Similarity::Euclidean,
        dim,
        &docs,
        &data,
        &OsqFactory::bbq(),
        30,
    );
    let registry = QuantizerRegistry::with_builtins();
    let try_open = |meta: &[u8], p: Vec<u8>, v: Vec<u8>, id: &[u8; 16], suffix: &str| {
        IvfReader::open(meta, Arc::new(p), Arc::new(v), id, suffix, &registry)
    };
    let (m, p, v) = (&files.meta, files.postings.clone(), files.vectors.clone());
    assert!(try_open(m, p.clone(), v.clone(), &[1; 16], "").is_err());
    assert!(try_open(m, p.clone(), v.clone(), &ID, "x").is_err());
    assert!(try_open(m, p[..10].to_vec(), v.clone(), &ID, "").is_err());
    assert!(try_open(m, p.clone(), v[..p.len().min(30)].to_vec(), &ID, "").is_err());
    assert!(try_open(m, p.clone(), vec![0; 8], &ID, "").is_err());
    // Any flipped meta byte fails its checksum or a bound.
    for i in [40, m.len() / 2, m.len() - 20] {
        let mut bad = m.clone();
        bad[i] ^= 0x55;
        assert!(
            try_open(&bad, p.clone(), v.clone(), &ID, "").is_err(),
            "byte {i}"
        );
    }
    // A truncated postings file (footer kept) no longer holds the field.
    let mut short = p[..p.len() - 100].to_vec();
    short.extend_from_slice(&p[p.len() - 16..]);
    assert!(try_open(m, short, v.clone(), &ID, "").is_err());
    // A flipped postings byte passes open but fails the integrity check.
    let mut flipped = p.clone();
    flipped[100] ^= 1;
    let r = try_open(m, flipped, v.clone(), &ID, "").unwrap();
    assert!(r.check_integrity().is_err());
    // A meta file naming an unknown quantizer.
    let empty = QuantizerRegistry::new();
    assert!(matches!(
        IvfReader::open(m, Arc::new(p.clone()), Arc::new(v.clone()), &ID, "", &empty),
        Err(Error::Quantizer(_))
    ));
    // Out-of-range reads through the slice source.
    assert!(RangeSource::read(&p, p.len(), 1).is_err());
    assert!(RangeSource::read(&p, usize::MAX, 2).is_err());
}

#[test]
fn identical_vectors_and_tiny_clusters() {
    let dim = 3;
    let data = vec![0.5f32; 300 * dim];
    let docs: Vec<i32> = (0..300).collect();
    let files = build(
        1,
        Similarity::Euclidean,
        dim,
        &docs,
        &data,
        &OsqFactory::bbq(),
        32,
    );
    let r = open(&files);
    assert!(r.field(1).unwrap().clusters() >= 10);
    let hits = r
        .search(1, &[0.5; 3], &SearchParams::new(3, 2, 3), None)
        .unwrap();
    assert_eq!(
        hits.iter().map(|h| h.doc).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn clustering_params_are_honoured() {
    let dim = 6;
    let data = dataset(1000, dim, 10, Similarity::Euclidean, 13);
    let docs: Vec<i32> = (0..1000).collect();
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: Similarity::Euclidean,
            dimension: dim,
            docs: &docs,
            vectors: &data,
        },
        &OsqFactory::bbq(),
        &IvfParams {
            target_cluster_size: 50,
            kmeans: KMeansParams {
                iterations: 2,
                sample_per_centroid: 16,
                refine_rounds: 0,
                full_rounds: 0,
                seed: 1,
            },
            training_sample: 10,
            spill: None,
            rerank_encoding: VectorEncoding::F32,
            calibration_recall: None,
            merge_clustering: MergeClustering::Recluster,
            routing_tree_min: usize::MAX,
        },
    )
    .unwrap();
    let r = open(&w.finish());
    let f = r.field(1).unwrap();
    assert_eq!(f.entries(), 1000);
    assert!((0..f.clusters()).all(|c| f.cluster_size(c) <= 50));
}

#[test]
fn spilled_entries_are_searched_once() {
    let dim = 8;
    let data = dataset(2000, dim, 15, Similarity::Euclidean, 31);
    let docs: Vec<i32> = (0..2000).collect();
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: Similarity::Euclidean,
            dimension: dim,
            docs: &docs,
            vectors: &data,
        },
        &OsqFactory::bbq(),
        &IvfParams {
            target_cluster_size: 100,
            spill: Some(1.0),
            ..IvfParams::default()
        },
    )
    .unwrap();
    let r = open(&w.finish());
    let f = r.field(1).unwrap();
    assert!(f.entries() > f.count() && f.entries() <= 2 * f.count());
    let all = f.clusters();
    for rerank in [0, 100] {
        let hits = r
            .search(1, &data[..dim], &SearchParams::new(50, all, rerank), None)
            .unwrap();
        let distinct: std::collections::HashSet<i32> = hits.iter().map(|h| h.doc).collect();
        assert_eq!((hits.len(), distinct.len()), (50, 50));
    }
    let exact = r.exact_search(1, &data[..dim], 10, None).unwrap();
    let hits = r
        .search(1, &data[..dim], &SearchParams::new(10, all, 100), None)
        .unwrap();
    assert_eq!(hits, exact);
    let (cdocs, codes) = r.cluster(1, 0).unwrap();
    assert_eq!(cdocs.len(), f.cluster_size(0));
    assert_eq!(codes.len(), f.quantizer().block_len(cdocs.len()));
}

#[test]
fn bad_slots_and_doc_tables_are_rejected() {
    let dim = 4;
    let n = 300;
    let data = dataset(n, dim, 4, Similarity::Euclidean, 6);
    let docs: Vec<i32> = (0..n as i32).collect();
    let files = build(
        1,
        Similarity::Euclidean,
        dim,
        &docs,
        &data,
        &OsqFactory::bbq(),
        40,
    );
    let registry = QuantizerRegistry::with_builtins();
    let open_with = |p: Vec<u8>, v: Vec<u8>| {
        IvfReader::open(&files.meta, Arc::new(p), Arc::new(v), &ID, "", &registry)
    };
    // The doc table is the last `n * 8` bytes before the footer.
    let table = files.vectors.len() - 16 - n * 8;
    let mut unordered = files.vectors.clone();
    unordered.copy_within(table + 4..table + 8, table);
    assert!(matches!(
        open_with(files.postings.clone(), unordered),
        Err(Error::Corrupted(_))
    ));
    let mut twice = files.vectors.clone();
    let slots = table + n * 4;
    twice.copy_within(slots..slots + 4, slots + 4);
    assert!(matches!(
        open_with(files.postings.clone(), twice),
        Err(Error::Corrupted(_))
    ));
    // A posting naming a slot past the vectors fails the search that meets it
    // (k = n, so every entry is competitive).
    let mut postings = files.postings.clone();
    let first = lucene_store::codec_util::index_header_length(lucene_ivf::POSTINGS_CODEC, "");
    postings[first..first + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let r = open_with(postings, files.vectors.clone()).unwrap();
    let clusters = r.field(1).unwrap().clusters();
    assert!(matches!(
        r.search(1, &data[..dim], &SearchParams::new(n, clusters, 0), None),
        Err(Error::Corrupted(_))
    ));
    assert!(r.cluster(1, 0).is_err());
}

#[test]
fn huge_k_and_rerank_are_capped_and_vectors_found_by_doc() {
    let dim = 8;
    let n = 200;
    let data = dataset(n, dim, 4, Similarity::Euclidean, 9);
    let docs: Vec<i32> = (0..n as i32).map(|d| d * 3).collect();
    let files = build(
        0,
        Similarity::Euclidean,
        dim,
        &docs,
        &data,
        &OsqFactory::bbq(),
        50,
    );
    let registry = QuantizerRegistry::with_builtins();
    let r = IvfReader::open(
        &files.meta,
        Arc::new(files.postings),
        Arc::new(files.vectors),
        &ID,
        "",
        &registry,
    )
    .unwrap();
    let hits = r
        .search(
            0,
            &data[..dim],
            &SearchParams::new(usize::MAX, 4, usize::MAX),
            None,
        )
        .unwrap();
    assert!(!hits.is_empty() && hits.len() <= n);
    assert_eq!(
        r.exact_search(0, &data[..dim], usize::MAX, None)
            .unwrap()
            .len(),
        n
    );
    for (i, &doc) in docs.iter().enumerate() {
        assert_eq!(
            r.vector(0, doc).unwrap().unwrap(),
            &data[i * dim..(i + 1) * dim]
        );
        assert!(r.vector(0, doc + 1).unwrap().is_none());
    }
    assert!(r.vector(0, -1).unwrap().is_none());
}

#[test]
fn rotated_centroids_are_prepared_once_at_open() {
    let dim = 24;
    let n = 1500;
    let data = dataset(n, dim, 10, Similarity::Euclidean, 5);
    let docs: Vec<i32> = (0..n as i32).collect();
    let factories: [(Box<dyn QuantizerFactory>, Option<usize>); 3] = [
        (
            Box::new(RotatedFactory::new(Arc::new(OsqFactory::bbq()))),
            Some(64),
        ),
        (
            Box::new(RotatedFactory::new(Arc::new(ScalarFactory::int7()))),
            None,
        ),
        (Box::new(OsqFactory::bbq()), None),
    ];
    for (factory, prepared_len) in &factories {
        let files = build(
            2,
            Similarity::Euclidean,
            dim,
            &docs,
            &data,
            factory.as_ref(),
            64,
        );
        let r = open(&files);
        let f = r.field(2).unwrap();
        assert!(f.clusters() > 1);
        for c in 0..f.clusters() {
            let prepared = f.prepared_centroid(c);
            assert_eq!(prepared.map(<[f32]>::len), *prepared_len);
            assert_eq!(
                prepared.map(<[f32]>::to_vec),
                f.quantizer().prepare_centroid(f.centroid(c))
            );
        }
        let q = &data[..dim];
        let hits = r.search(2, q, &SearchParams::new(10, 4, 0), None).unwrap();
        let truth = r.exact_search(2, q, 10, None).unwrap();
        assert!(recall(&hits, &truth) >= 0.5, "{factory:?}");
    }
}

fn build_encoded(
    sim: Similarity,
    dim: usize,
    docs: &[i32],
    vectors: &[f32],
    encoding: VectorEncoding,
) -> lucene_ivf::IvfFiles {
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: sim,
            dimension: dim,
            docs,
            vectors,
        },
        &OsqFactory::bbq(),
        &IvfParams {
            target_cluster_size: 100,
            rerank_encoding: encoding,
            ..IvfParams::default()
        },
    )
    .unwrap();
    w.finish()
}

#[test]
fn rerank_encodings_round_trip_and_rank_like_f32() {
    let dim = 32;
    let n = 2000;
    let docs: Vec<i32> = (0..n as i32).collect();
    for sim in SIMS {
        let data = dataset(n, dim, 20, sim, 41);
        let queries = dataset(20, dim, 20, sim, 42);
        let exact_files = build_encoded(sim, dim, &docs, &data, VectorEncoding::F32);
        let exact = open(&exact_files);
        for (encoding, tol, min_overlap) in [
            (VectorEncoding::F16, 2f32.powi(-11), 1.0),
            (VectorEncoding::Bf16, 2f32.powi(-8), 0.97),
            (VectorEncoding::Int8, 1.0 / 255.0, 0.95),
        ] {
            let files = build_encoded(sim, dim, &docs, &data, encoding);
            let stored = n * encoding.vector_bytes(dim).unwrap();
            assert_eq!(
                exact_files.vectors.len() - files.vectors.len(),
                n * dim * 4 - stored
            );
            let r = open(&files);
            r.check_integrity().unwrap();
            let f = r.field(1).unwrap();
            assert_eq!(f.encoding(), encoding);
            // Decoded vectors are within the encoding's precision of the
            // f32 ones, relative to the vector's largest component (int8:
            // half a level of the vector's range).
            for &doc in &docs {
                let want = exact.vector(1, doc).unwrap().unwrap();
                let got = r.vector(1, doc).unwrap().unwrap();
                let max = want.iter().fold(0f32, |m, x| m.max(x.abs()));
                for (a, b) in want.iter().zip(&got) {
                    assert!((a - b).abs() <= max * tol, "{sim:?} {encoding:?} doc {doc}");
                }
            }
            let mut total = 0.0;
            for q in queries.chunks_exact(dim) {
                let p = SearchParams::new(10, 8, 50);
                let hits = r.search(1, q, &p, None).unwrap();
                let want = exact.search(1, q, &p, None).unwrap();
                assert_eq!(hits.len(), 10);
                assert_eq!(hits[0].doc, want[0].doc, "{sim:?} {encoding:?}");
                // Reranked scores are those of the decoded vectors.
                let mut qq = q.to_vec();
                if sim == Similarity::Cosine {
                    lucene_util::vector_util::l2normalize(&mut qq, false).unwrap();
                }
                for h in &hits {
                    let v = r.vector(1, h.doc).unwrap().unwrap();
                    assert_eq!(h.score, sim.score(&qq, &v));
                }
                total += recall(&hits, &want);
                let all = r.exact_search(1, q, 10, None).unwrap();
                assert_eq!(all[0].doc, hits[0].doc);
            }
            let avg = total / 20.0;
            assert!(
                avg >= min_overlap,
                "{sim:?} {encoding:?}: overlap with f32 {avg}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "out of range")]
fn prepared_centroid_checks_the_cluster() {
    let dim = 8;
    let data = dataset(200, dim, 4, Similarity::Euclidean, 6);
    let docs: Vec<i32> = (0..200).collect();
    let files = build(
        1,
        Similarity::Euclidean,
        dim,
        &docs,
        &data,
        &OsqFactory::bbq(),
        50,
    );
    let r = open(&files);
    let f = r.field(1).unwrap();
    let _ = f.prepared_centroid(f.clusters());
}

#[test]
fn merge_decodes_every_encoding() {
    let dim = 12;
    let sim = Similarity::Euclidean;
    let a = dataset(500, dim, 6, sim, 51);
    let b = dataset(400, dim, 6, sim, 52);
    let docs_a: Vec<i32> = (0..500).collect();
    let docs_b: Vec<i32> = (0..400).collect();
    let ra = open(&build_encoded(sim, dim, &docs_a, &a, VectorEncoding::F16));
    let rb = open(&build_encoded(sim, dim, &docs_b, &b, VectorEncoding::Int8));
    let map_a = |d: i32| Some(d);
    let map_b = |d: i32| Some(500 + d);
    let sources = [
        MergeSource {
            reader: &ra,
            doc_map: &map_a,
        },
        MergeSource {
            reader: &rb,
            doc_map: &map_b,
        },
    ];
    for encoding in VectorEncoding::ALL {
        let mut w = IvfWriter::new(&ID, "");
        let p = IvfParams {
            target_cluster_size: 100,
            rerank_encoding: encoding,
            ..IvfParams::default()
        };
        merge_field(&mut w, 1, &sources, &OsqFactory::bbq(), &p).unwrap();
        let merged = open(&w.finish());
        let field = merged.field(1).unwrap();
        assert_eq!((field.count(), field.encoding()), (900, encoding));
        for (doc, source, old) in [(3, &ra, 3), (499, &ra, 499), (500, &rb, 0), (777, &rb, 277)] {
            let want = source.vector(1, old).unwrap().unwrap();
            let got = merged.vector(1, doc).unwrap().unwrap();
            if encoding == VectorEncoding::F32 {
                // The decoded source vectors are carried over exactly.
                assert_eq!(got, want);
            } else {
                let max = want.iter().fold(0f32, |m, x| m.max(x.abs()));
                for (x, y) in want.iter().zip(&got) {
                    assert!((x - y).abs() <= max / 200.0, "{encoding:?} doc {doc}");
                }
            }
        }
        let q = merged.vector(1, 777).unwrap().unwrap();
        let hits = merged
            .search(1, &q, &SearchParams::new(1, 4, 20), None)
            .unwrap();
        assert_eq!(hits[0].doc, 777);
    }
}

/// `file` with its header rewritten at `version`, its payload (what
/// follows the header) passed through `edit`, under a fresh footer.
fn rewrite(file: &[u8], codec: &str, version: i32, edit: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let header = lucene_store::codec_util::index_header_length(codec, "");
    let mut payload = file[header..file.len() - 16].to_vec();
    edit(&mut payload);
    let mut out = Vec::new();
    lucene_store::codec_util::write_index_header(&mut out, codec, version, &ID, "");
    out.extend_from_slice(&payload);
    lucene_store::codec_util::write_footer(&mut out);
    out
}

/// The bytes `VERSION_CALIBRATION` and `VERSION_ROUTING_TREE` append to
/// the last field's meta entry: its radii, calibration and (for a field too
/// small for a routing tree) a zero parent count.
fn calibration_tail(files: &lucene_ivf::IvfFiles, field: i32) -> usize {
    let r = open(files);
    let f = r.field(field).unwrap();
    assert!(!f.has_routing_tree());
    f.clusters() * 4 + if f.calibration().is_some() { 9 } else { 1 } + 1
}

/// In a one-field meta payload: field count (1 byte), field number (4),
/// similarity (1), then the encoding byte.
const ENCODING_AT: usize = 6;

#[test]
fn version_one_files_read_as_f32() {
    let dim = 8;
    let sim = Similarity::Cosine;
    let data = dataset(600, dim, 5, sim, 61);
    let docs: Vec<i32> = (0..600).map(|d| d * 2).collect();
    let files = build_encoded(sim, dim, &docs, &data, VectorEncoding::F32);
    let v1 = lucene_ivf::VERSION_BLOCK_LAYOUT;
    let tail = calibration_tail(&files, 1);
    let meta = rewrite(&files.meta, lucene_ivf::META_CODEC, v1, |p| {
        p.truncate(p.len() - tail);
        p.remove(ENCODING_AT);
    });
    let postings = rewrite(&files.postings, lucene_ivf::POSTINGS_CODEC, v1, |_| {});
    let vectors = rewrite(&files.vectors, lucene_ivf::VECTORS_CODEC, v1, |_| {});
    let registry = QuantizerRegistry::with_builtins();
    let old = IvfReader::open(
        &meta,
        Arc::new(postings.clone()),
        Arc::new(vectors.clone()),
        &ID,
        "",
        &registry,
    )
    .unwrap();
    old.check_integrity().unwrap();
    let new = open(&files);
    assert_eq!(old.field(1).unwrap().encoding(), VectorEncoding::F32);
    let q = &data[5 * dim..6 * dim];
    let p = SearchParams::new(10, 3, 30);
    assert_eq!(
        old.search(1, q, &p, None).unwrap(),
        new.search(1, q, &p, None).unwrap()
    );
    assert_eq!(old.vector(1, 10).unwrap(), new.vector(1, 10).unwrap());
    // Files of different versions are not one index.
    for (m, pp, v) in [
        (&meta, &files.postings, &vectors),
        (&meta, &postings, &files.vectors),
        (&files.meta, &postings, &vectors),
    ] {
        assert!(matches!(
            IvfReader::open(
                m,
                Arc::new(pp.clone()),
                Arc::new(v.clone()),
                &ID,
                "",
                &registry
            ),
            Err(Error::Corrupted(_))
        ));
    }
    // Version 0 is refused.
    let v0 = |f: &[u8], codec| rewrite(f, codec, 0, |_| {});
    assert!(IvfReader::open(
        &v0(&meta, lucene_ivf::META_CODEC),
        Arc::new(v0(&postings, lucene_ivf::POSTINGS_CODEC)),
        Arc::new(v0(&vectors, lucene_ivf::VECTORS_CODEC)),
        &ID,
        "",
        &registry
    )
    .is_err());
}

#[test]
fn bad_encodings_are_rejected() {
    let dim = 6;
    let sim = Similarity::Euclidean;
    let data = dataset(300, dim, 4, sim, 71);
    let docs: Vec<i32> = (0..300).collect();
    let files = build_encoded(sim, dim, &docs, &data, VectorEncoding::F16);
    let registry = QuantizerRegistry::with_builtins();
    let open_meta = |meta: &[u8]| {
        IvfReader::open(
            meta,
            Arc::new(files.postings.clone()),
            Arc::new(files.vectors.clone()),
            &ID,
            "",
            &registry,
        )
    };
    let with_encoding = |ordinal: u8| {
        rewrite(
            &files.meta,
            lucene_ivf::META_CODEC,
            lucene_ivf::VERSION_CURRENT,
            |p| {
                assert_eq!(p[ENCODING_AT], VectorEncoding::F16.ordinal());
                p[ENCODING_AT] = ordinal;
            },
        )
    };
    open_meta(&with_encoding(VectorEncoding::F16.ordinal())).unwrap();
    // An unknown encoding.
    match open_meta(&with_encoding(9)) {
        Err(Error::Corrupted(m)) => assert!(m.contains("encoding"), "{m}"),
        other => panic!("{other:?}"),
    }
    // f32 vectors would run past the doc table of an f16 field.
    match open_meta(&with_encoding(VectorEncoding::F32.ordinal())) {
        Err(Error::Corrupted(m)) => assert!(m.contains("out of bounds"), "{m}"),
        other => panic!("{other:?}"),
    }

    // A value f16 cannot hold is refused, and the writer is left usable.
    let mut w = IvfWriter::new(&ID, "");
    let big = [1.0, 2.0, 70_000.0, 4.0];
    let field = FieldVectors {
        number: 3,
        similarity: sim,
        dimension: 2,
        docs: &[0, 1],
        vectors: &big,
    };
    let f16 = IvfParams {
        rerank_encoding: VectorEncoding::F16,
        ..IvfParams::default()
    };
    match w.add_field(&field, &OsqFactory::bbq(), &f16) {
        Err(Error::IllegalArgument(m)) => assert!(m.contains("f16"), "{m}"),
        other => panic!("{other:?}"),
    }
    for encoding in [VectorEncoding::Bf16, VectorEncoding::Int8] {
        let mut w = IvfWriter::new(&ID, "");
        let p = IvfParams {
            rerank_encoding: encoding,
            ..IvfParams::default()
        };
        w.add_field(&field, &OsqFactory::bbq(), &p).unwrap();
        let r = open(&w.finish());
        let v = r.vector(3, 1).unwrap().unwrap();
        assert!(
            (v[0] - 70_000.0).abs() <= 70_000.0 / 128.0,
            "{encoding:?} {v:?}"
        );
    }
    w.add_field(&field, &OsqFactory::bbq(), &IvfParams::default())
        .unwrap();
    let r = open(&w.finish());
    assert_eq!(r.vector(3, 1).unwrap().unwrap(), vec![70_000.0, 4.0]);
}

#[test]
fn int8_residuals_with_spilling_and_bad_primary_counts() {
    let dim = 8;
    let sim = Similarity::Euclidean;
    let data = dataset(1500, dim, 10, sim, 81);
    let docs: Vec<i32> = (0..1500).collect();
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: sim,
            dimension: dim,
            docs: &docs,
            vectors: &data,
        },
        &OsqFactory::bbq(),
        &IvfParams {
            target_cluster_size: 100,
            spill: Some(1.0),
            rerank_encoding: VectorEncoding::Int8,
            ..IvfParams::default()
        },
    )
    .unwrap();
    let r = open(&w.finish());
    let f = r.field(1).unwrap();
    assert!(f.entries() > f.count());
    // Every vector decodes near its original, whichever cluster holds it.
    for (i, v) in data.chunks_exact(dim).enumerate() {
        let got = r.vector(1, i as i32).unwrap().unwrap();
        for (a, b) in v.iter().zip(&got) {
            assert!((a - b).abs() < 0.01, "doc {i}: {a} vs {b}");
        }
    }
    let hits = r
        .search(1, &data[..dim], &SearchParams::new(5, 4, 50), None)
        .unwrap();
    assert_eq!(hits[0].doc, 0);

    // One cluster: the last vint before the radii and calibration is its
    // primary count.
    let small = dataset(90, dim, 3, sim, 82);
    let docs: Vec<i32> = (0..90).collect();
    let files = build_encoded(sim, dim, &docs, &small, VectorEncoding::Int8);
    let tail = calibration_tail(&files, 1);
    let registry = QuantizerRegistry::with_builtins();
    let with_primary = |count: u8| {
        rewrite(
            &files.meta,
            lucene_ivf::META_CODEC,
            lucene_ivf::VERSION_CURRENT,
            |p| {
                let at = p.len() - 1 - tail;
                assert_eq!(p[at], 90);
                p[at] = count;
            },
        )
    };
    let open_meta = |meta: &[u8]| {
        IvfReader::open(
            meta,
            Arc::new(files.postings.clone()),
            Arc::new(files.vectors.clone()),
            &ID,
            "",
            &registry,
        )
    };
    open_meta(&with_primary(90)).unwrap();
    for (count, message) in [(89, "primary entries"), (91, "vectors in a cluster")] {
        match open_meta(&with_primary(count)) {
            Err(Error::Corrupted(m)) => assert!(m.contains(message), "{m}"),
            other => panic!("{other:?}"),
        }
    }
}

fn segment(sim: Similarity, dim: usize, data: &[f32], params: &IvfParams) -> IvfReader {
    let docs: Vec<i32> = (0..(data.len() / dim) as i32).collect();
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: sim,
            dimension: dim,
            docs: &docs,
            vectors: data,
        },
        &OsqFactory::bbq(),
        params,
    )
    .unwrap();
    open(&w.finish())
}

fn merge_with(
    sources: &[MergeSource<'_>],
    params: &IvfParams,
) -> lucene_ivf::Result<lucene_ivf::IvfFiles> {
    let mut w = IvfWriter::new(&ID, "");
    merge_field(&mut w, 1, sources, &OsqFactory::bbq(), params)?;
    Ok(w.finish())
}

fn reuse(min_fraction: f32) -> MergeClustering {
    MergeClustering::ReuseLargest {
        min_fraction,
        refine_rounds: 1,
    }
}

/// Mean recall@10 at `nprobe` of each reader's field 1, over noisy copies
/// of every 13th vector of `data`, against the exact search of the first:
/// the readers hold the same documents and vectors.
fn merged_recall(readers: [&IvfReader; 2], data: &[f32], dim: usize, nprobe: usize) -> (f64, f64) {
    let mut rng = Rng(99);
    let mut total = [0.0; 2];
    let queries: Vec<Vec<f32>> = data
        .chunks_exact(dim)
        .step_by(13)
        .map(|v| v.iter().map(|x| x + rng.next() * 0.5).collect())
        .collect();
    for q in &queries {
        let truth = readers[0].exact_search(1, q, 10, None).unwrap();
        for (t, r) in total.iter_mut().zip(readers) {
            let hits = r
                .search(1, q, &SearchParams::new(10, nprobe, 50), None)
                .unwrap();
            *t += recall(&hits, &truth) / queries.len() as f64;
        }
    }
    (total[0], total[1])
}

fn sizes(r: &IvfReader) -> Vec<usize> {
    let f = r.field(1).unwrap();
    (0..f.clusters()).map(|c| f.cluster_size(c)).collect()
}

#[test]
fn merge_reuses_the_largest_source_only_when_it_dominates() {
    let dim = 8;
    let sim = Similarity::Euclidean;
    let data = dataset(2000, dim, 20, sim, 61);
    let p = IvfParams {
        target_cluster_size: 50,
        ..IvfParams::default()
    };
    let a = segment(sim, dim, &data[..1800 * dim], &p);
    let b = segment(sim, dim, &data[1800 * dim..], &p);
    let same = |d: i32| Some(d);
    let after = |d: i32| Some(1800 + d);
    let none_field = open(&IvfWriter::new(&ID, "").finish());
    let sources = [
        MergeSource {
            reader: &a,
            doc_map: &same,
        },
        MergeSource {
            reader: &b,
            doc_map: &after,
        },
        // A source without the field.
        MergeSource {
            reader: &none_field,
            doc_map: &same,
        },
    ];
    let with = |m: MergeClustering| {
        merge_with(
            &sources,
            &IvfParams {
                merge_clustering: m,
                ..p
            },
        )
        .unwrap()
    };
    let fresh = with(MergeClustering::Recluster);
    assert_eq!(p.merge_clustering, reuse(0.5));
    let reused = with(MergeClustering::default());
    assert_ne!(reused, fresh);
    // 90% of the vectors: reused at 0.9, not at 0.95; never with no vectors.
    assert_eq!(with(reuse(0.9)), reused);
    assert_eq!(with(reuse(0.95)), fresh);
    assert_eq!(with(reuse(f32::NAN)), fresh);
    let none = |_: i32| None;
    let empty = |m: MergeClustering| {
        merge_with(
            &[MergeSource {
                reader: &a,
                doc_map: &none,
            }],
            &IvfParams {
                merge_clustering: m,
                ..p
            },
        )
        .unwrap()
    };
    assert_eq!(empty(reuse(0.0)), empty(MergeClustering::Recluster));
    // A source of one cluster has nothing to reuse.
    let flat = segment(
        sim,
        dim,
        &data[..1800 * dim],
        &IvfParams {
            target_cluster_size: 5000,
            ..p
        },
    );
    let sources = [
        MergeSource {
            reader: &flat,
            doc_map: &same,
        },
        MergeSource {
            reader: &b,
            doc_map: &after,
        },
    ];
    let merged = |m: MergeClustering| {
        merge_with(
            &sources,
            &IvfParams {
                merge_clustering: m,
                ..p
            },
        )
        .unwrap()
    };
    assert_eq!(merged(reuse(0.5)), merged(MergeClustering::Recluster));
}

/// Recall@10 at 6 probes of a merge reusing `a`'s clusters and of one
/// re-clustering, after deleting every seventh document of `a`, over a
/// dataset of 4000 vectors. Checks the merged vectors, the cluster sizes and
/// that the reusing merge is deterministic.
fn reuse_and_recluster_recall(sim: Similarity, seed: u64) -> (f64, f64) {
    let (dim, n, split) = (16, 4000, 3600);
    let data = dataset(n, dim, 4, sim, seed);
    let p = IvfParams {
        target_cluster_size: 50,
        ..IvfParams::default()
    };
    let a = segment(sim, dim, &data[..split * dim], &p);
    let b = segment(sim, dim, &data[split * dim..], &p);
    let map_a = |d: i32| (d % 7 != 0).then(|| d - d / 7 - 1);
    let live_a = split as i32 - (split as i32 + 6) / 7;
    let map_b = move |d: i32| Some(live_a + d);
    let sources = [
        MergeSource {
            reader: &a,
            doc_map: &map_a,
        },
        MergeSource {
            reader: &b,
            doc_map: &map_b,
        },
    ];
    let live: Vec<f32> = data
        .chunks_exact(dim)
        .enumerate()
        .filter(|(i, _)| *i >= split || i % 7 != 0)
        .flat_map(|(_, v)| v.iter().copied())
        .collect();
    let fresh = open(
        &merge_with(
            &sources,
            &IvfParams {
                merge_clustering: MergeClustering::Recluster,
                ..p
            },
        )
        .unwrap(),
    );
    let files = merge_with(&sources, &p).unwrap();
    assert_eq!(files, merge_with(&sources, &p).unwrap(), "{sim:?}");
    let reused = open(&files);
    assert_eq!(reused.field(1).unwrap().count(), live.len() / dim);
    assert!(sizes(&reused).iter().all(|&s| s <= 100), "{sim:?}");
    for (doc, source, old) in [(0, &a, 1), (live_a - 1, &a, 3599), (live_a + 5, &b, 5)] {
        assert_eq!(
            reused.vector(1, doc).unwrap(),
            source.vector(1, old).unwrap(),
            "{sim:?}"
        );
    }
    merged_recall([&reused, &fresh], &live, dim, 6)
}

#[test]
fn reusing_merge_recalls_like_reclustering() {
    // Averaged over seeds: on this small dataset two clusterings differ by
    // a few points of recall either way, by up to 0.1 for
    // MaximumInnerProduct, whose probes favour large centroids.
    for sim in SIMS {
        let (mut got, mut want) = (0.0, 0.0);
        for seed in 71..77 {
            let (g, w) = reuse_and_recluster_recall(sim, seed);
            got += g / 6.0;
            want += w / 6.0;
        }
        assert!(got >= want - 0.02, "{sim:?}: {got} against {want}");
    }
}

#[test]
fn reusing_merge_splits_and_dissolves() {
    let dim = 8;
    let sim = Similarity::Euclidean;
    let data = dataset(2000, dim, 10, sim, 81);
    let p = IvfParams {
        target_cluster_size: 50,
        ..IvfParams::default()
    };
    let a = segment(sim, dim, &data, &p);
    // 900 new vectors around one of `a`'s.
    let mut rng = Rng(5);
    let crowd: Vec<f32> = (0..900)
        .flat_map(|_| data[..dim].to_vec())
        .map(|x| x + rng.next() * 0.01)
        .collect();
    let b = segment(sim, dim, &crowd, &p);
    let same = |d: i32| Some(d);
    let after = |d: i32| Some(2000 + d);
    let r = open(
        &merge_with(
            &[
                MergeSource {
                    reader: &a,
                    doc_map: &same,
                },
                MergeSource {
                    reader: &b,
                    doc_map: &after,
                },
            ],
            &p,
        )
        .unwrap(),
    );
    let s = sizes(&r);
    assert_eq!(s.iter().sum::<usize>(), 2900);
    assert!(s.iter().all(|&x| x <= 100), "{s:?}");
    let hits = r
        .search(1, &crowd[..dim], &SearchParams::new(1, 2, 10), None)
        .unwrap();
    assert_eq!(hits[0].doc, 2000);
    // Deleting all but three of one centre's vectors leaves no cluster
    // under a quarter of the target.
    let thin = |d: i32| (d % 10 != 0 || d < 30).then_some(d);
    let r = open(
        &merge_with(
            &[MergeSource {
                reader: &a,
                doc_map: &thin,
            }],
            &p,
        )
        .unwrap(),
    );
    let s = sizes(&r);
    assert_eq!(s.iter().sum::<usize>(), 1803);
    assert!(s.iter().all(|&x| (12..=100).contains(&x)), "{s:?}");
}

#[test]
fn reusing_merge_reads_spilled_sources() {
    let dim = 8;
    let sim = Similarity::Euclidean;
    let data = dataset(2000, dim, 15, sim, 91);
    for encoding in [VectorEncoding::F32, VectorEncoding::Int8] {
        let p = IvfParams {
            target_cluster_size: 50,
            spill: Some(1.0),
            rerank_encoding: encoding,
            ..IvfParams::default()
        };
        let a = segment(sim, dim, &data[..1700 * dim], &p);
        let b = segment(sim, dim, &data[1700 * dim..], &p);
        let same = |d: i32| Some(d);
        let after = |d: i32| Some(1700 + d);
        let sources = [
            MergeSource {
                reader: &a,
                doc_map: &same,
            },
            MergeSource {
                reader: &b,
                doc_map: &after,
            },
        ];
        let reused = open(&merge_with(&sources, &p).unwrap());
        let fresh = open(
            &merge_with(
                &sources,
                &IvfParams {
                    merge_clustering: MergeClustering::Recluster,
                    ..p
                },
            )
            .unwrap(),
        );
        let f = reused.field(1).unwrap();
        assert!(f.entries() > f.count() && f.count() == 2000);
        // int8 residuals are re-encoded against the merged centroid.
        let (got, want) = (
            reused.vector(1, 1900).unwrap().unwrap(),
            b.vector(1, 200).unwrap().unwrap(),
        );
        assert!(got.iter().zip(&want).all(|(x, y)| (x - y).abs() < 0.02));
        let (got, want) = merged_recall([&reused, &fresh], &data, dim, 4);
        assert!(got >= want - 0.02, "{encoding:?}: {got} against {want}");
    }
}

#[test]
fn reusing_merge_rejects_postings_that_lose_or_repeat_slots() {
    let dim = 4;
    let n = 400;
    let sim = Similarity::Euclidean;
    // Every vector spills but the 40 copies of one point, which sit on
    // their centroid.
    let mut data = dataset(n - 40, dim, 6, sim, 17);
    data.extend(std::iter::repeat_n(100.0, 40 * dim));
    let docs: Vec<i32> = (0..n as i32).collect();
    let mut w = IvfWriter::new(&ID, "");
    w.add_field(
        &FieldVectors {
            number: 1,
            similarity: sim,
            dimension: dim,
            docs: &docs,
            vectors: &data,
        },
        &OsqFactory::bbq(),
        &IvfParams {
            target_cluster_size: 40,
            spill: Some(1.0),
            ..IvfParams::default()
        },
    )
    .unwrap();
    let files = w.finish();
    let r = open(&files);
    let f = r.field(1).unwrap();
    let slot_of: std::collections::HashMap<i32, u32> =
        r.doc_table(1).unwrap().into_iter().collect();
    // Each cluster's slots, and where its block starts in `.ivp`.
    let mut at = lucene_store::codec_util::index_header_length(lucene_ivf::POSTINGS_CODEC, "");
    let mut blocks = Vec::new();
    let mut seen = std::collections::HashMap::<u32, usize>::new();
    for c in 0..f.clusters() {
        let slots: Vec<u32> = r
            .cluster(1, c)
            .unwrap()
            .0
            .iter()
            .map(|d| slot_of[d])
            .collect();
        for &s in &slots {
            *seen.entry(s).or_default() += 1;
        }
        blocks.push((at, slots));
        at += f.cluster_size(c) * 4 + f.quantizer().block_len(f.cluster_size(c));
    }
    let registry = QuantizerRegistry::with_builtins();
    let merge_postings = |postings: Vec<u8>| {
        let r = IvfReader::open(
            &files.meta,
            Arc::new(postings),
            Arc::new(files.vectors.clone()),
            &ID,
            "",
            &registry,
        )
        .unwrap();
        let same = |d: i32| Some(d);
        merge_with(
            &[MergeSource {
                reader: &r,
                doc_map: &same,
            }],
            &IvfParams {
                target_cluster_size: 40,
                ..IvfParams::default()
            },
        )
    };
    assert!(merge_postings(files.postings.clone()).is_ok());
    let edit = |pos: usize, slot: u32| {
        let mut p = files.postings.clone();
        p[pos..pos + 4].copy_from_slice(&slot.to_le_bytes());
        merge_postings(p)
    };
    let spilled = *seen.iter().find(|(_, &k)| k == 2).unwrap().0;
    let lonely: Vec<(usize, u32)> = blocks
        .iter()
        .flat_map(|(at, slots)| slots.iter().enumerate().map(move |(i, &s)| (at + 4 * i, s)))
        .filter(|(_, s)| seen[s] == 1)
        .collect();
    assert!(lonely.len() >= 2);
    for result in [
        edit(blocks[0].0, u32::MAX),
        // A third entry for a spilled slot.
        edit(lonely[0].0, spilled),
        // A slot's only entry naming another slot.
        edit(lonely[0].0, lonely[1].1),
    ] {
        assert!(matches!(result, Err(Error::Corrupted(_))), "{result:?}");
    }
}

#[test]
fn writer_takes_a_given_clustering() {
    let dim = 2;
    let data = [0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 11.0, 0.0];
    let docs = [0, 1, 2, 3];
    let field = FieldVectors {
        number: 1,
        similarity: Similarity::Euclidean,
        dimension: dim,
        docs: &docs,
        vectors: &data,
    };
    let f = OsqFactory::bbq();
    let p = IvfParams::default();
    let mut w = IvfWriter::new(&ID, "");
    // The empty middle cluster is dropped.
    let given = Clustering {
        centroids: vec![0.5, 0.0, 99.0, 0.0, 10.5, 0.0],
        assignments: vec![0, 0, 2, 2],
    };
    w.add_field_with_clustering(&field, &given, &f, &p).unwrap();
    let r = open(&w.finish());
    let field_r = r.field(1).unwrap();
    assert_eq!(field_r.clusters(), 2);
    assert_eq!(field_r.centroid(1), &[10.5, 0.0]);
    assert_eq!(r.cluster(1, 1).unwrap().0, vec![2, 3]);
    let bad = [
        Clustering {
            centroids: vec![0.5],
            ..given.clone()
        },
        Clustering {
            centroids: vec![f32::NAN, 0.0],
            assignments: vec![0; 4],
        },
        Clustering {
            assignments: vec![0; 3],
            ..given.clone()
        },
        Clustering {
            assignments: vec![0, 0, 3, 2],
            ..given.clone()
        },
    ];
    for c in bad {
        let mut w = IvfWriter::new(&ID, "");
        assert!(
            matches!(
                w.add_field_with_clustering(&field, &c, &f, &p),
                Err(Error::IllegalArgument(_))
            ),
            "{c:?}"
        );
    }
}
