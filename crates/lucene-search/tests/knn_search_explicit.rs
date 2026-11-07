//! KNN searches over explicit documents with vectors, as the OpenSearch
//! engine writes them: several segments of HNSW (and scalar-quantized HNSW)
//! vectors through a `ConcurrentIndexWriter`, searched whole-reader
//! ([`knn_top_hits`]) and per segment ([`KnnLeafQuery`]), checked against an
//! exhaustive scan.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::buffered_updates::Term;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::search_all;
use lucene_search::knn_search::{knn_top_hits, KnnLeafKind, KnnLeafQuery, KnnTarget};
use lucene_search::query::{Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::quantization::ScalarEncoding;
use lucene_util::test_support::TempDir;

const DIM: usize = 8;
const DOCS: usize = 240;
const PER_SEGMENT: usize = 80;

fn vector(id: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| (((id * 31 + d * 17) % 97) as f32 / 97.0) - 0.5)
        .collect()
}

fn l2_score(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    1.0 / (1.0 + d)
}

/// `id` stored, the `parity` keyword term (`even`/`odd`) and the vector.
fn doc(id: usize) -> (ExplicitDocument, Vec<DocumentVector>) {
    let mut terms = InvertedTerms::default();
    let term: &[u8] = if id.is_multiple_of(2) {
        b"even"
    } else {
        b"odd"
    };
    terms.push::<Vec<u8>>(term, 1, &[], &[], &[]);
    let mut d = ExplicitDocument::default();
    d.stored.push(StoredField {
        field_number: 0,
        value: FieldValue::String(id.to_string()),
    });
    d.fields.inverted.push(InvertedField {
        field_number: 1,
        terms,
        norm: None,
    });
    (d, vec![DocumentVector::float32("v", vector(id))])
}

fn build(dir: &FsDirectory, format: KnnVectorsFormat) {
    let mut w = IndexWriter::open(
        dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(FieldInfo::new("id", 0)).unwrap();
    w.register_field(
        FieldInfo::new("parity", 0)
            .with_index_options(lucene_codecs::field_infos::IndexOptions::Docs)
            .with_omit_norms(true),
    )
    .unwrap();
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        DIM as i32,
        VectorEncoding::Float32,
        VectorSimilarityFunction::Euclidean,
    ))
    .unwrap();
    w.set_knn_vectors_format_for_field("v", format);
    w.set_ram_buffer_size_mb(64.0).unwrap();
    w.set_merge_policy(None);
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    for start in (0..DOCS).step_by(PER_SEGMENT) {
        let (docs, vectors): (Vec<_>, Vec<_>) = (start..start + PER_SEGMENT).map(doc).unzip();
        w.add_explicit_documents_with_vectors_in(None, docs, vectors)
            .unwrap();
        w.flush().unwrap();
    }
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
}

fn exact_top(target: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<(usize, f32)> {
    let mut all: Vec<(usize, f32)> = (0..DOCS)
        .filter(|&id| keep(id))
        .map(|id| (id, l2_score(target, &vector(id))))
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.truncate(k);
    all
}

fn check(format: KnnVectorsFormat, exact_scores: bool) {
    let tmp = TempDir::new("knn-search-explicit");
    let dir = FsDirectory::open(&tmp);
    build(&dir, format);
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), DOCS / PER_SEGMENT);
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let target = vector(7777);
    // Doc ids are ids: one writer thread, segments in order.
    let k = 10;
    let hits = knn_top_hits(
        &segments,
        "v",
        &KnnTarget::Float(target.clone()),
        k,
        100,
        None,
    )
    .unwrap();
    let want = exact_top(&target, k, |_| true);
    let got: Vec<usize> = hits.iter().map(|h| h.doc_id as usize).collect();
    let want_ids: Vec<usize> = want.iter().map(|w| w.0).collect();
    assert_eq!(got, want_ids, "global top {k}");
    if exact_scores {
        for (h, w) in hits.iter().zip(&want) {
            assert!((h.score - w.1).abs() < 1e-6, "{} vs {}", h.score, w.1);
        }
    }

    // Filtered to even documents.
    let even = Clause::Term(TermQuery::new("parity", b"even".to_vec()));
    let hits = knn_top_hits(
        &segments,
        "v",
        &KnnTarget::Float(target.clone()),
        k,
        100,
        Some(&even),
    )
    .unwrap();
    let got: Vec<usize> = hits.iter().map(|h| h.doc_id as usize).collect();
    let want: Vec<usize> = exact_top(&target, k, |id| id.is_multiple_of(2))
        .iter()
        .map(|w| w.0)
        .collect();
    assert_eq!(got, want, "filtered top {k}");

    // Per segment: k hits from every segment.
    let leaf = KnnLeafQuery::new(
        "v",
        KnnTarget::Float(target.clone()),
        KnnLeafKind::TopK {
            k: 5,
            ef_search: 100,
        },
    )
    .unwrap();
    let all = search_all(&segments, &leaf).unwrap();
    assert_eq!(all.len(), 5 * (DOCS / PER_SEGMENT));
    for seg in 0..DOCS / PER_SEGMENT {
        let lo = seg * PER_SEGMENT;
        let mut got: Vec<usize> = all
            .iter()
            .map(|h| h.doc_id as usize)
            .filter(|&d| d >= lo && d < lo + PER_SEGMENT)
            .collect();
        got.sort_unstable();
        let mut want: Vec<usize> = exact_top(&target, 5, |id| id >= lo && id < lo + PER_SEGMENT)
            .iter()
            .map(|w| w.0)
            .collect();
        want.sort_unstable();
        assert_eq!(got, want, "segment {seg}");
    }

    // Radial: everything scoring at least the 20th best's score.
    let threshold = exact_top(&target, 20, |_| true)[19].1;
    let radial = KnnLeafQuery::new(
        "v",
        KnnTarget::Float(target.clone()),
        KnnLeafKind::Radial {
            min_score: threshold,
        },
    )
    .unwrap();
    let all = search_all(&segments, &radial).unwrap();
    if exact_scores {
        assert_eq!(all.len(), 20);
    }
    assert!(all.iter().all(|h| h.score >= threshold));
}

#[test]
fn hnsw_float_vectors_find_the_exact_neighbours() {
    check(KnnVectorsFormat::hnsw(16, 100).unwrap(), true);
}

#[test]
fn quantized_hnsw_vectors_are_searched() {
    check(
        KnnVectorsFormat::hnsw_scalar_quantized(ScalarEncoding::SevenBit, 16, 100).unwrap(),
        true,
    );
}

#[test]
fn deleted_documents_are_never_returned() {
    let tmp = TempDir::new("knn-search-explicit-deletes");
    let dir = FsDirectory::open(&tmp);
    build(&dir, KnnVectorsFormat::default());
    {
        let mut w = IndexWriter::open(
            &dir,
            Vec::new(),
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        w.delete_documents_by_term(&[Term {
            field: "parity".into(),
            bytes: b"odd".to_vec(),
        }])
        .unwrap();
        w.commit().unwrap();
        w.close().unwrap();
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let target = vector(4242);
    let hits = knn_top_hits(
        &segments,
        "v",
        &KnnTarget::Float(target.clone()),
        10,
        100,
        None,
    )
    .unwrap();
    let got: Vec<usize> = hits.iter().map(|h| h.doc_id as usize).collect();
    let want: Vec<usize> = exact_top(&target, 10, |id| id.is_multiple_of(2))
        .iter()
        .map(|w| w.0)
        .collect();
    assert_eq!(got, want);
}
