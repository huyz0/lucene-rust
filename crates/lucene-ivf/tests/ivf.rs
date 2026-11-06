//! Building, reading, searching and merging IVF indexes end to end.
#![allow(clippy::arithmetic_side_effects)]

use std::borrow::Cow;
use std::sync::Arc;

use lucene_ivf::kmeans::KMeansParams;
use lucene_ivf::{
    merge_field, Error, FieldVectors, IvfParams, IvfReader, IvfWriter, MergeSource, RangeSource,
    ScoredDoc, SearchParams, Similarity,
};
use lucene_util::quantization::ScalarEncoding;
use lucene_vector_quant::osq::OsqFactory;
use lucene_vector_quant::rabitq::ExRabitqFactory;
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
