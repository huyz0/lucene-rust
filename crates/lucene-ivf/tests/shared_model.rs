//! Fields whose centroids and quantizer come from a model shared by many
//! segments rather than from the segment itself.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_ivf::kmeans::{self, KMeansParams, Metric};
use lucene_ivf::{
    merge_field_with_model, Error, FieldVectors, IvfFiles, IvfParams, IvfReader, IvfWriter,
    MergeSource, ScoredDoc, SearchParams, SharedModel, Similarity,
};
use lucene_vector_quant::flat::FlatFactory;
use lucene_vector_quant::pq::PqFactory;
use lucene_vector_quant::{QuantizerFactory, QuantizerRegistry};

const ID: [u8; 16] = [4; 16];
const CRC: u32 = 0xC0FF_EE11;
const DIM: usize = 16;

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

fn dataset(n: usize, centers: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let cs: Vec<Vec<f32>> = (0..centers)
        .map(|_| (0..DIM).map(|_| rng.next() * 4.0).collect())
        .collect();
    let mut out = Vec::with_capacity(n * DIM);
    for i in 0..n {
        let c = &cs[i % centers];
        out.extend(c.iter().map(|x| x + rng.next() * 0.5));
    }
    out
}

fn params() -> IvfParams {
    IvfParams {
        target_cluster_size: 60,
        ..IvfParams::default()
    }
}

fn nearest(v: &[f32], centroids: &[f32]) -> usize {
    centroids
        .chunks_exact(DIM)
        .enumerate()
        .map(|(i, c)| {
            (
                i,
                v.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
            )
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .0
}

/// A model trained on `data`: its centroids, and `factory` trained on the
/// residuals against them (or on the vectors, for a codeless factory).
fn model(id: &str, data: &[f32], factory: &dyn QuantizerFactory) -> Arc<SharedModel> {
    let clustering = kmeans::cluster(data, DIM, 60, &KMeansParams::default(), Metric::L2);
    let mut residuals = Vec::with_capacity(data.len());
    for v in data.chunks_exact(DIM) {
        let c = nearest(v, &clustering.centroids);
        let centroid = &clustering.centroids[c * DIM..(c + 1) * DIM];
        residuals.extend(v.iter().zip(centroid).map(|(a, b)| a - b));
    }
    let quantizer = factory
        .train(DIM, Similarity::Euclidean, &residuals)
        .unwrap();
    SharedModel::new(
        id,
        CRC,
        DIM,
        Similarity::Euclidean,
        clustering.centroids,
        Arc::from(quantizer),
    )
    .unwrap()
}

fn write(model: &Arc<SharedModel>, docs: &[i32], vectors: &[f32]) -> IvfFiles {
    let mut w = IvfWriter::new(&ID, "");
    w.add_field_with_model(
        &FieldVectors {
            number: 3,
            similarity: Similarity::Euclidean,
            dimension: DIM,
            docs,
            vectors,
        },
        model,
        &params(),
    )
    .unwrap();
    w.finish()
}

fn open_with(files: &IvfFiles, models: &[Arc<SharedModel>]) -> Result<IvfReader, Error> {
    IvfReader::open_with_models(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
        &|id| models.iter().find(|m| m.id() == id).cloned(),
    )
}

fn recall(found: &[ScoredDoc], truth: &[ScoredDoc]) -> f64 {
    let hits = found
        .iter()
        .filter(|f| truth.iter().any(|t| t.doc == f.doc))
        .count();
    hits as f64 / truth.len() as f64
}

#[test]
fn a_pq_model_field_names_its_model_and_finds_the_neighbours() {
    let data = dataset(3000, 20, 1);
    let m = model("m1", &data, &PqFactory::new(4));
    assert_eq!((m.id(), m.crc(), m.dimension()), ("m1", CRC, DIM));
    assert!(m.nlist() > 20);
    let docs: Vec<i32> = (0..3000).collect();
    let files = write(&m, &docs, &data);
    // The segment holds no centroids and no codebooks.
    assert!(
        files.meta.len() < m.nlist() * DIM * 4 / 2,
        "meta {}",
        files.meta.len()
    );
    let r = open_with(&files, &[Arc::clone(&m)]).unwrap();
    r.check_integrity().unwrap();
    let f = r.field(3).unwrap();
    assert_eq!((f.clusters(), f.count()), (m.nlist(), 3000));
    assert_eq!(f.model().map(|m| (m.id(), m.crc())), Some(("m1", CRC)));
    let queries = dataset(20, 20, 9);
    let mut total = 0.0;
    for q in queries.chunks_exact(DIM) {
        let truth = r.exact_search(3, q, 10, None).unwrap();
        let hits = r.search(3, q, &SearchParams::new(10, 8, 60), None).unwrap();
        assert_eq!(hits.len(), 10);
        total += recall(&hits, &truth);
    }
    assert!(total / 20.0 >= 0.9, "recall {}", total / 20.0);
}

#[test]
fn a_flat_model_field_is_exact_at_full_probe() {
    let data = dataset(1500, 15, 2);
    let m = model("flat", &data, &FlatFactory);
    let docs: Vec<i32> = (0..1500).collect();
    let r = open_with(&write(&m, &docs, &data), &[Arc::clone(&m)]).unwrap();
    let queries = dataset(10, 15, 5);
    for q in queries.chunks_exact(DIM) {
        let truth = r.exact_search(3, q, 10, None).unwrap();
        let hits = r
            .search(3, q, &SearchParams::new(10, m.nlist(), 0), None)
            .unwrap();
        assert_eq!(hits, truth);
        // Fewer probes still rank exactly what they scan.
        let some = r.search(3, q, &SearchParams::new(10, 3, 0), None).unwrap();
        assert!(some.windows(2).all(|w| w[0].score >= w[1].score));
        assert!(some
            .iter()
            .all(|h| (h.score - exact(&r, q, h.doc)).abs() < 1e-6));
    }
}

fn exact(r: &IvfReader, q: &[f32], doc: i32) -> f32 {
    Similarity::Euclidean.score(q, &r.vector(3, doc).unwrap().unwrap())
}

#[test]
fn a_filter_applies_to_a_flat_scan() {
    let data = dataset(600, 10, 3);
    let m = model("flat", &data, &FlatFactory);
    let docs: Vec<i32> = (0..600).collect();
    let r = open_with(&write(&m, &docs, &data), &[Arc::clone(&m)]).unwrap();
    let odd = |d: i32| d % 2 == 1;
    let hits = r
        .search(
            3,
            &data[..DIM],
            &SearchParams::new(5, m.nlist(), 0),
            Some(&odd),
        )
        .unwrap();
    assert_eq!(hits.len(), 5);
    assert!(hits.iter().all(|h| h.doc % 2 == 1));
    let truth = r.exact_search(3, &data[..DIM], 5, Some(&odd)).unwrap();
    assert_eq!(hits, truth);
}

#[test]
fn a_segment_smaller_than_the_model_keeps_empty_clusters() {
    let data = dataset(3000, 20, 1);
    let m = model("m1", &data, &PqFactory::new(4));
    let docs: Vec<i32> = (0..10).collect();
    let r = open_with(&write(&m, &docs, &data[..10 * DIM]), &[Arc::clone(&m)]).unwrap();
    let f = r.field(3).unwrap();
    assert_eq!((f.clusters(), f.count()), (m.nlist(), 10));
    assert!((0..m.nlist()).any(|c| f.cluster_size(c) == 0));
    let hits = r
        .search(3, &data[..DIM], &SearchParams::new(3, m.nlist(), 10), None)
        .unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].doc, 0);
}

#[test]
fn a_reader_refuses_a_missing_or_changed_model() {
    let data = dataset(600, 10, 4);
    let m = model("m1", &data, &PqFactory::new(4));
    let docs: Vec<i32> = (0..600).collect();
    let files = write(&m, &docs, &data);
    assert!(matches!(
        open_with(&files, &[]),
        Err(Error::IllegalArgument(_))
    ));
    let other = SharedModel::new(
        "m1",
        CRC + 1,
        DIM,
        Similarity::Euclidean,
        m.centroids().to_vec(),
        m.quantizer_arc(),
    )
    .unwrap();
    let err = open_with(&files, &[other]).unwrap_err();
    assert!(err.to_string().contains("m1"), "{err}");
    // A plain open has no models to give.
    let plain = IvfReader::open(
        &files.meta,
        Arc::new(files.postings.clone()),
        Arc::new(files.vectors.clone()),
        &ID,
        "",
        &QuantizerRegistry::with_builtins(),
    );
    assert!(matches!(plain, Err(Error::IllegalArgument(_))));
}

#[test]
fn writing_refuses_what_the_model_cannot_hold() {
    let data = dataset(600, 10, 4);
    let m = model("m1", &data, &PqFactory::new(4));
    let docs: Vec<i32> = (0..10).collect();
    let mut w = IvfWriter::new(&ID, "");
    let field = |dimension, similarity| FieldVectors {
        number: 3,
        similarity,
        dimension,
        docs: &docs,
        vectors: &data[..10 * DIM],
    };
    assert!(w
        .add_field_with_model(&field(8, Similarity::Euclidean), &m, &params())
        .is_err());
    assert!(w
        .add_field_with_model(&field(DIM, Similarity::DotProduct), &m, &params())
        .is_err());
    let spill = IvfParams {
        spill: Some(1.0),
        ..params()
    };
    assert!(w
        .add_field_with_model(&field(DIM, Similarity::Euclidean), &m, &spill)
        .is_err());
    // A refused field leaves the writer as it was.
    w.add_field_with_model(&field(DIM, Similarity::Euclidean), &m, &params())
        .unwrap();
}

#[test]
fn a_model_field_and_an_ordinary_field_share_a_segment() {
    let data = dataset(800, 10, 6);
    let m = model("m1", &data, &PqFactory::new(4));
    let docs: Vec<i32> = (0..800).collect();
    let mut w = IvfWriter::new(&ID, "");
    let field = |number| FieldVectors {
        number,
        similarity: Similarity::Euclidean,
        dimension: DIM,
        docs: &docs,
        vectors: &data,
    };
    w.add_field_with_model(&field(1), &m, &params()).unwrap();
    w.add_field(&field(2), &PqFactory::new(4), &params())
        .unwrap();
    let files = w.finish();
    let r = open_with(&files, &[Arc::clone(&m)]).unwrap();
    assert!(r.field(1).unwrap().model().is_some());
    assert!(r.field(2).unwrap().model().is_none());
    for number in [1, 2] {
        let hits = r
            .search(
                number,
                &data[..DIM],
                &SearchParams::new(3, m.nlist(), 800),
                None,
            )
            .unwrap();
        assert_eq!(hits[0].doc, 0);
    }
}

#[test]
fn a_merge_keeps_the_model_and_every_vectors_list() {
    let data = dataset(1200, 12, 7);
    let m = model("m1", &data, &PqFactory::new(4));
    let docs_a: Vec<i32> = (0..600).collect();
    let docs_b: Vec<i32> = (0..600).collect();
    let a = open_with(&write(&m, &docs_a, &data[..600 * DIM]), &[Arc::clone(&m)]).unwrap();
    let b = open_with(&write(&m, &docs_b, &data[600 * DIM..]), &[Arc::clone(&m)]).unwrap();
    let to_a = |d: i32| Some(d);
    let to_b = |d: i32| Some(d + 600);
    let mut w = IvfWriter::new(&ID, "");
    merge_field_with_model(
        &mut w,
        3,
        &[
            MergeSource {
                reader: &a,
                doc_map: &to_a,
            },
            MergeSource {
                reader: &b,
                doc_map: &to_b,
            },
        ],
        &m,
        &params(),
    )
    .unwrap();
    let merged = open_with(&w.finish(), &[Arc::clone(&m)]).unwrap();
    let f = merged.field(3).unwrap();
    assert_eq!((f.count(), f.clusters()), (1200, m.nlist()));
    assert_eq!(f.model().map(|m| m.crc()), Some(CRC));
    // No clustering ran: each list holds the vectors of both inputs' lists.
    for c in 0..m.nlist() {
        let both = a.field(3).unwrap().cluster_size(c) + b.field(3).unwrap().cluster_size(c);
        assert_eq!(f.cluster_size(c), both, "list {c}");
    }
    let hits = merged
        .search(
            3,
            &data[900 * DIM..901 * DIM],
            &SearchParams::new(3, m.nlist(), 30),
            None,
        )
        .unwrap();
    assert_eq!(hits[0].doc, 900);
}

/// `file` with its payload edited and its footer recomputed, under `version`.
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

#[test]
fn the_shared_flag_is_read_only_where_the_format_has_it() {
    let data = dataset(600, 10, 4);
    let m = model("m1", &data, &PqFactory::new(4));
    let docs: Vec<i32> = (0..600).collect();
    let files = write(&m, &docs, &data);
    // Field count (1 byte), field number (4), then the similarity byte.
    let at = 5;
    let with_meta = |version: i32, edit: &dyn Fn(&mut Vec<u8>)| {
        let meta = rewrite(&files.meta, lucene_ivf::META_CODEC, version, |p| edit(p));
        let postings = rewrite(&files.postings, lucene_ivf::POSTINGS_CODEC, version, |_| {});
        let vectors = rewrite(&files.vectors, lucene_ivf::VECTORS_CODEC, version, |_| {});
        IvfReader::open_with_models(
            &meta,
            Arc::new(postings),
            Arc::new(vectors),
            &ID,
            "",
            &QuantizerRegistry::with_builtins(),
            &|_| Some(Arc::clone(&m)),
        )
    };
    let current = lucene_ivf::VERSION_CURRENT;
    with_meta(current, &|p| assert_eq!(p[at] & 0x80, 0x80)).unwrap();
    // Before the flag existed the byte is not a similarity.
    let older = with_meta(lucene_ivf::VERSION_PARTITIONS, &|_| {});
    assert!(matches!(older, Err(Error::Corrupted(m)) if m.contains("similarity")));
    // Without the flag the entry is read as an ordinary field and does not
    // parse.
    assert!(with_meta(current, &|p| p[at] &= 0x7f).is_err());
}
