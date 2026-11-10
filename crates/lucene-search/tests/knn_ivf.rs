//! Float vectors in `LuceneRustIvfVectorsFormat`: flushed and merged
//! segments carry an IVF index beside their raw vectors, and KNN searches
//! probe it, with deletions, filters and a forced merge that reuses the
//! largest segment's clusters, checked against an exhaustive scan.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::query::{Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

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

/// `id` stored, the `parity` term (`even`/`odd`) and the `rare` term
/// (`yes` for one document in 64).
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
    for name in ["parity", "rare"] {
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

fn build(dir: &FsDirectory, format: KnnVectorsFormat) {
    let w = ConcurrentIndexWriter::new(writer(dir, format), 1).unwrap();
    for start in (0..DOCS).step_by(PER_SEGMENT) {
        let (docs, vectors): (Vec<_>, Vec<_>) = (start..start + PER_SEGMENT).map(doc).unzip();
        w.add_explicit_documents_with_vectors_in(None, docs, vectors)
            .unwrap();
        w.flush().unwrap();
    }
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
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

fn ivf() -> KnnVectorsFormat {
    KnnVectorsFormat::ivf(128, 0).unwrap()
}

#[test]
fn flushed_segments_are_searched_through_their_ivf_index() {
    let tmp = TempDir::new("knn-ivf-flush");
    let dir = FsDirectory::open(&tmp);
    build(&dir, ivf());
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), SEGMENTS);
    for seg in reader.segment_readers() {
        let field = seg
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "v")
            .unwrap();
        assert!(field.attributes.iter().any(|(k, v)| {
            k == "PerFieldKnnVectorsFormat.format" && v == "LuceneRustIvfVectorsFormat"
        }));
        let index = seg.ivf_for(field.number).unwrap().expect("an IVF index");
        assert!(index.field(field.number).unwrap().clusters() > 1);
    }
    let r = recall(&dir, 10, None, |_| true);
    assert!(r >= 0.9, "recall {r}");
}

#[test]
fn half_float_rerank_vectors_still_return_exact_scores() {
    let tmp = TempDir::new("knn-ivf-f16");
    let dir = FsDirectory::open(&tmp);
    build(&dir, KnnVectorsFormat::ivf(128, 1).unwrap());
    let r = recall(&dir, 10, None, |_| true);
    assert!(r >= 0.9, "recall {r}");
}

/// The quantizer name and bits each `IVF_QUANTIZERS` wire number writes
/// into the `.ivm` state.
#[test]
fn every_quantizer_writes_its_codes_and_keeps_exact_scores() {
    use lucene_codecs::per_field_knn_vectors::IVF_QUANTIZERS;
    for (number, name) in IVF_QUANTIZERS.iter().enumerate() {
        let tmp = TempDir::new(&format!("knn-ivf-{name}"));
        let dir = FsDirectory::open(&tmp);
        build(
            &dir,
            KnnVectorsFormat::ivf_quantized(128, 1, number as u8).unwrap(),
        );
        let reader = DirectoryReader::open(&dir).unwrap();
        for seg in reader.segment_readers() {
            let field = seg
                .field_infos()
                .fields
                .iter()
                .find(|f| f.name == "v")
                .unwrap();
            let index = seg.ivf_for(field.number).unwrap().expect("an IVF index");
            let want = if name.starts_with("erq") {
                "erq"
            } else {
                "osq"
            };
            assert_eq!(index.field(field.number).unwrap().quantizer().name(), want);
        }
        let r = recall(&dir, 10, None, |_| true);
        assert!(r >= 0.9, "{name} recall {r}");
    }
}

/// A merge re-quantizes the merged vectors with the writer's quantizer,
/// whatever the sources were written with.
#[test]
fn a_merge_requantizes_with_the_writers_quantizer() {
    let tmp = TempDir::new("knn-ivf-requant");
    let dir = FsDirectory::open(&tmp);
    build(&dir, ivf());
    let erq4 = KnnVectorsFormat::ivf_quantized(128, 1, 6).unwrap();
    let mut w = writer(&dir, erq4);
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];
    let number = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "v")
        .unwrap()
        .number;
    let index = seg.ivf_for(number).unwrap().expect("a merged IVF index");
    assert_eq!(index.field(number).unwrap().quantizer().name(), "erq");
}

#[test]
fn bad_ivf_parameters_are_refused() {
    assert!(KnnVectorsFormat::ivf(8, 0).is_err());
    assert!(KnnVectorsFormat::ivf(128, 4).is_err());
    assert!(KnnVectorsFormat::ivf_quantized(128, 0, 11).is_err());
    let tmp = TempDir::new("knn-ivf-bad");
    let dir = FsDirectory::open(&tmp);
    let w = ConcurrentIndexWriter::new(
        writer(
            &dir,
            KnnVectorsFormat::Ivf {
                target_cluster_size: 0,
                rerank_encoding: 0,
                quantizer: 0,
            },
        ),
        1,
    )
    .unwrap();
    let (docs, vectors): (Vec<_>, Vec<_>) = (0..10).map(doc).unzip();
    w.add_explicit_documents_with_vectors_in(None, docs, vectors)
        .unwrap();
    assert!(w.flush().is_err());
}

#[test]
fn filters_and_deletions_hold_through_the_ivf_index() {
    let tmp = TempDir::new("knn-ivf-filter");
    let dir = FsDirectory::open(&tmp);
    build(&dir, ivf());
    // Half the documents: probed through the index.
    let even = Clause::Term(TermQuery::new("parity", b"even".to_vec()));
    let r = recall(&dir, 10, Some(&even), |id| id.is_multiple_of(2));
    assert!(r >= 0.9, "filtered recall {r}");
    // One in 64: too few to probe for, scanned exactly.
    let rare = Clause::Term(TermQuery::new("rare", b"yes".to_vec()));
    assert_eq!(
        recall(&dir, 10, Some(&rare), |id| id.is_multiple_of(64)),
        1.0
    );
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.delete_documents_by_term(&[Term {
        field: "parity".into(),
        bytes: b"odd".to_vec(),
    }])
    .unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    let r = recall(&dir, 10, None, |id| id.is_multiple_of(2));
    assert!(r >= 0.9, "recall after deletes {r}");
}

#[test]
fn a_merge_writes_the_ivf_index_of_the_merged_vectors() {
    let tmp = TempDir::new("knn-ivf-merge");
    let dir = FsDirectory::open(&tmp);
    build(&dir, ivf());
    let mut w = writer(&dir, ivf());
    w.delete_documents_by_term(&[Term {
        field: "rare".into(),
        bytes: b"yes".to_vec(),
    }])
    .unwrap();
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    let seg = &reader.segment_readers()[0];
    let number = seg
        .field_infos()
        .fields
        .iter()
        .find(|f| f.name == "v")
        .unwrap()
        .number;
    let index = seg.ivf_for(number).unwrap().expect("a merged IVF index");
    let field = index.field(number).unwrap();
    assert_eq!(field.count(), DOCS - DOCS.div_ceil(64));
    // Ids no longer equal doc ids after the deletes: map them back.
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let stored = |doc: i32| -> usize {
        let stored = seg.stored_document(doc).unwrap().unwrap();
        match &stored.fields[0].value {
            FieldValue::String(s) => s.parse().unwrap(),
            other => panic!("id {other:?}"),
        }
    };
    let k = 10;
    let mut found = 0;
    for q in 0..30 {
        let target = vector(2_000_000 + q);
        let hits = knn_top_hits(
            &segments,
            "v",
            &KnnTarget::Float(target.clone()),
            k,
            k,
            None,
        )
        .unwrap();
        let want = exact_top(&target, k, |id| !id.is_multiple_of(64));
        for h in &hits {
            let id = stored(h.doc_id);
            assert!(!id.is_multiple_of(64), "deleted doc {id} returned");
            found += usize::from(want.contains(&id));
        }
    }
    let r = found as f64 / (30 * k) as f64;
    assert!(r >= 0.9, "recall after merge {r}");
}
