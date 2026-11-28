//! Fields clustered by a shared model (`IvfModelVectorsFormat`): every
//! flushed and merged segment names the model, keeps its lists and is
//! searched through it, and a reader without the model refuses the field.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use std::sync::Arc;

use lucene_codecs::per_field_knn_vectors::{
    KnnVectorsFormat, IVF_FLAT_QUANTIZER, IVF_PQ_QUANTIZER,
};
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_ivf::kmeans::{self, KMeansParams, Metric};
use lucene_ivf::{SharedModel, Similarity};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::query::{Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;
use lucene_vector_quant::flat::FlatFactory;
use lucene_vector_quant::pq::PqFactory;
use lucene_vector_quant::QuantizerFactory;

const DIM: usize = 16;
const PER_SEGMENT: usize = 1500;
const SEGMENTS: usize = 3;
const DOCS: usize = PER_SEGMENT * SEGMENTS;
const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn noise(seed: u64) -> impl FnMut() -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    move || {
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Points around 20 centres, so the clusters are real ones.
fn vector(id: usize) -> Vec<f32> {
    let mut centre = noise((id % 20) as u64 + 1_000_000);
    let mut jitter = noise(id as u64);
    (0..DIM).map(|_| centre() * 4.0 + jitter() * 0.5).collect()
}

/// `id` stored, the `parity` term (`even`/`odd`), the `rare` term
/// (`yes` for one document in 64) and the `centre` term (the centre the
/// vector is drawn around, `0`-`19`).
fn doc(id: usize) -> (ExplicitDocument, Vec<DocumentVector>) {
    let mut d = ExplicitDocument::default();
    d.stored.push(StoredField {
        field_number: 0,
        value: FieldValue::String(id.to_string()),
    });
    for (field_number, term) in [
        (
            1,
            if id.is_multiple_of(2) {
                &b"even"[..]
            } else {
                b"odd"
            },
        ),
        (
            2,
            if id.is_multiple_of(64) {
                &b"yes"[..]
            } else {
                b"no"
            },
        ),
        (3, (id % 20).to_string().as_bytes()),
    ] {
        let mut terms = InvertedTerms::default();
        terms.push::<Vec<u8>>(term, 1, &[], &[], &[]);
        d.fields.inverted.push(InvertedField {
            field_number,
            terms,
            norm: None,
        });
    }
    (d, vec![DocumentVector::float32("v", vector(id))])
}

fn writer(dir: &FsDirectory, format: KnnVectorsFormat) -> IndexWriter<'_> {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(FieldInfo::new("id", 0)).unwrap();
    for name in ["parity", "rare", "centre"] {
        w.register_field(
            FieldInfo::new(name, 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    }
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        DIM as i32,
        VectorEncoding::Float32,
        VectorSimilarityFunction::Euclidean,
    ))
    .unwrap();
    w.set_knn_vectors_format_for_field("v", format);
    w.set_ram_buffer_size_mb(256.0).unwrap();
    w.set_merge_policy(None);
    w
}

fn score(a: &[f32], b: &[f32]) -> f32 {
    VectorSimilarityFunction::Euclidean.score(a, b)
}

fn exact_top(target: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<usize> {
    let mut all: Vec<(usize, f32)> = (0..DOCS)
        .filter(|&id| keep(id))
        .map(|id| (id, score(target, &vector(id))))
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Recall@`k` of the reader's searches over `queries` targets against the
/// exhaustive top `k` of the documents `keep` passes; every hit's score is
/// checked against its exact score (`f32` rerank vectors).
fn recall(
    dir: &FsDirectory,
    k: usize,
    filter: Option<&Clause>,
    keep: impl Fn(usize) -> bool,
) -> f64 {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let queries = 30;
    let mut found = 0;
    for q in 0..queries {
        let target = vector(1_000_000 + q);
        let hits = knn_top_hits(
            &segments,
            "v",
            &KnnTarget::Float(target.clone()),
            k,
            k,
            filter,
        )
        .unwrap();
        assert_eq!(hits.len(), k);
        // Doc ids are ids: one writer thread, segments in order, no merge
        // reordering.
        for h in &hits {
            let id = h.doc_id as usize;
            assert!(keep(id), "doc {id} should not match");
            assert_eq!(h.score, score(&target, &vector(id)), "doc {id} score");
        }
        let want = exact_top(&target, k, &keep);
        found += hits
            .iter()
            .filter(|h| want.contains(&(h.doc_id as usize)))
            .count();
    }
    found as f64 / (queries * k) as f64
}

/// A model trained on every document's vector.
fn train(id: &str, factory: &dyn QuantizerFactory) -> Arc<SharedModel> {
    let data: Vec<f32> = (0..DOCS).flat_map(vector).collect();
    let clustering = kmeans::cluster(&data, DIM, 24, &KMeansParams::default(), Metric::L2);
    let mut residuals = Vec::with_capacity(data.len());
    for v in data.chunks_exact(DIM) {
        let (c, _) = clustering
            .centroids
            .chunks_exact(DIM)
            .enumerate()
            .map(|(i, c)| {
                (
                    i,
                    v.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
                )
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let centroid = &clustering.centroids[c * DIM..(c + 1) * DIM];
        residuals.extend(v.iter().zip(centroid).map(|(a, b)| a - b));
    }
    let quantizer = factory
        .train(DIM, Similarity::Euclidean, &residuals)
        .unwrap();
    SharedModel::new(
        id,
        0x1234_5678,
        DIM,
        Similarity::Euclidean,
        clustering.centroids,
        Arc::from(quantizer),
    )
    .unwrap()
}

fn model_writer<'a>(
    dir: &'a FsDirectory,
    model: &Arc<SharedModel>,
    quantizer: u8,
) -> IndexWriter<'a> {
    let mut w = writer(dir, KnnVectorsFormat::ivf_model(0, quantizer).unwrap());
    w.set_ivf_model_for_field("v", model);
    w
}

fn build_with_model(dir: &FsDirectory, model: &Arc<SharedModel>, quantizer: u8) {
    let w = ConcurrentIndexWriter::new(model_writer(dir, model, quantizer), 1).unwrap();
    for start in (0..DOCS).step_by(PER_SEGMENT) {
        let (docs, vectors): (Vec<_>, Vec<_>) = (start..start + PER_SEGMENT).map(doc).unzip();
        w.add_explicit_documents_with_vectors_in(None, docs, vectors)
            .unwrap();
        w.flush().unwrap();
    }
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
}

fn field_number(reader: &DirectoryReader) -> i32 {
    reader.segment_readers()[0]
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "v")
        .unwrap()
        .number
}

#[test]
fn flat_model_segments_share_lists_and_search_exactly() {
    let model = train("e2e-flat", &FlatFactory);
    lucene_index::ivf_vectors::register_model(model.clone());
    let tmp = TempDir::new("knn-ivf-model-flat");
    let dir = FsDirectory::open(&tmp);
    build_with_model(&dir, &model, IVF_FLAT_QUANTIZER);
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), SEGMENTS);
    let number = field_number(&reader);
    for seg in reader.segment_readers() {
        let index = seg.ivf_for(number).unwrap().expect("an IVF index");
        let field = index.field(number).unwrap();
        assert_eq!(field.model().unwrap().id(), "e2e-flat");
        assert_eq!(field.clusters(), model.nlist());
    }
    let r = recall(&dir, 10, None, |_| true);
    assert!(r >= 0.9, "recall {r}");
    let even = Clause::Term(TermQuery::new("parity", b"even".to_vec()));
    let r = recall(&dir, 10, Some(&even), |id| id.is_multiple_of(2));
    assert!(r >= 0.9, "filtered recall {r}");
}

#[test]
fn a_merge_keeps_the_model_and_its_lists() {
    let model = train("e2e-merge", &FlatFactory);
    lucene_index::ivf_vectors::register_model(model.clone());
    let tmp = TempDir::new("knn-ivf-model-merge");
    let dir = FsDirectory::open(&tmp);
    build_with_model(&dir, &model, IVF_FLAT_QUANTIZER);
    let mut w = model_writer(&dir, &model, IVF_FLAT_QUANTIZER);
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let number = field_number(&reader);
    let index = reader.segment_readers()[0]
        .ivf_for(number)
        .unwrap()
        .expect("a merged IVF index");
    let field = index.field(number).unwrap();
    assert_eq!(field.model().unwrap().id(), "e2e-merge");
    assert_eq!(field.clusters(), model.nlist());
    let r = recall(&dir, 10, None, |_| true);
    assert!(r >= 0.9, "recall after merge {r}");
}

#[test]
fn pq_model_segments_find_neighbours_and_rerank_exactly() {
    let model = train("e2e-pq", &PqFactory::new(4));
    lucene_index::ivf_vectors::register_model(model.clone());
    let tmp = TempDir::new("knn-ivf-model-pq");
    let dir = FsDirectory::open(&tmp);
    build_with_model(&dir, &model, IVF_PQ_QUANTIZER);
    let r = recall(&dir, 10, None, |_| true);
    assert!(r >= 0.8, "recall {r}");
}

#[test]
fn a_field_without_its_model_is_refused() {
    let model = train("e2e-unregistered", &FlatFactory);
    let tmp = TempDir::new("knn-ivf-model-missing");
    let dir = FsDirectory::open(&tmp);
    // The writer registers what it is given, so a reader finds it...
    build_with_model(&dir, &model, IVF_FLAT_QUANTIZER);
    let reader = DirectoryReader::open(&dir).unwrap();
    let number = field_number(&reader);
    assert!(reader.segment_readers()[0].ivf_for(number).is_ok());
    // ...and once it is gone, the segment is not searched under another.
    lucene_index::ivf_vectors::unregister_model("e2e-unregistered");
    let reader = DirectoryReader::open(&dir).unwrap();
    assert!(reader.segment_readers()[0].ivf_for(number).is_err());
}

#[test]
fn a_model_format_without_a_model_fails_the_flush() {
    let tmp = TempDir::new("knn-ivf-model-none");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(
        writer(
            &dir,
            KnnVectorsFormat::ivf_model(0, IVF_FLAT_QUANTIZER).unwrap(),
        ),
        1,
    )
    .unwrap();
    let (docs, vectors): (Vec<_>, Vec<_>) = (0..10).map(doc).unzip();
    w.add_explicit_documents_with_vectors_in(None, docs, vectors)
        .unwrap();
    assert!(w.flush().is_err());
}
