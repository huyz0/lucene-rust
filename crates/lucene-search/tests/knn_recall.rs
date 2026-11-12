//! Recall-targeted KNN searches (`knn_recall`) over real HNSW and IVF
//! indexes: the effort scales what each format searches, the exact search
//! the checks compare with is exact, and a tuner moves its level toward
//! its target from sampled queries.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_recall::{
    check_sample, effort, knn_top_hits_at_effort, knn_top_hits_exact, knn_top_hits_tuned,
    RecallTarget, RecallTuner, BASE_LEVEL, LEVELS,
};
use lucene_search::knn_search::{knn_top_hits_with, KnnTarget};
use lucene_search::query::{Clause, TermQuery};
use lucene_search::ScoreDoc;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const DIM: usize = 32;
const PER_SEGMENT: usize = 1500;
const SEGMENTS: usize = 3;
const DOCS: usize = PER_SEGMENT * SEGMENTS;
const K: usize = 10;
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

/// Points around 20 centres, wide enough that a shallow search misses some.
fn vector(id: usize) -> Vec<f32> {
    let mut centre = noise((id % 20) as u64 + 1_000_000);
    let mut jitter = noise(id as u64);
    (0..DIM).map(|_| centre() * 2.0 + jitter() * 1.5).collect()
}

fn doc(id: usize) -> (ExplicitDocument, Vec<DocumentVector>) {
    let mut d = ExplicitDocument::default();
    d.stored.push(StoredField {
        field_number: 0,
        value: FieldValue::String(id.to_string()),
    });
    let mut terms = InvertedTerms::default();
    let parity: &[u8] = if id.is_multiple_of(2) {
        b"even"
    } else {
        b"odd"
    };
    terms.push::<Vec<u8>>(parity, 1, &[], &[], &[]);
    d.fields.inverted.push(InvertedField {
        field_number: 1,
        terms,
        norm: None,
    });
    (d, vec![DocumentVector::float32("v", vector(id))])
}

fn build(dir: &FsDirectory, format: KnnVectorsFormat) {
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", VERSION).unwrap();
    w.enable_explicit_documents().unwrap();
    w.register_field(FieldInfo::new("id", 0)).unwrap();
    w.register_field(
        FieldInfo::new("parity", 0)
            .with_index_options(IndexOptions::Docs)
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
    w.set_ram_buffer_size_mb(256.0).unwrap();
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

/// The exhaustive top `k` ids `keep` passes.
fn brute_force(target: &[f32], k: usize, keep: impl Fn(usize) -> bool) -> Vec<i32> {
    let mut all: Vec<(usize, f32)> = (0..DOCS)
        .filter(|&id| keep(id))
        .map(|id| {
            (
                id,
                VectorSimilarityFunction::Euclidean.score(target, &vector(id)),
            )
        })
        .collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    all.into_iter().take(k).map(|(id, _)| id as i32).collect()
}

fn ids(hits: &[ScoreDoc]) -> Vec<i32> {
    hits.iter().map(|h| h.doc_id).collect()
}

fn target(q: usize) -> KnnTarget {
    KnnTarget::Float(vector(1_000_000 + q))
}

fn even() -> Clause {
    Clause::Term(TermQuery::new("parity", b"even".to_vec()))
}

/// Each format's index under a temporary directory.
fn with_index(name: &str, format: KnnVectorsFormat, f: impl FnOnce(&DirectoryReader)) {
    let tmp = TempDir::new(name);
    let dir = FsDirectory::open(&tmp);
    build(&dir, format);
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), SEGMENTS);
    f(&reader);
}

fn formats() -> [(&'static str, KnnVectorsFormat); 2] {
    [
        ("knn-recall-hnsw", KnnVectorsFormat::hnsw(4, 8).unwrap()),
        ("knn-recall-ivf", KnnVectorsFormat::ivf(64, 0).unwrap()),
    ]
}

/// Recall@K of `search` over `queries` against the brute-force top K.
fn recall(queries: usize, search: impl Fn(usize) -> Vec<ScoreDoc>) -> f64 {
    let mut found = 0;
    for q in 0..queries {
        let want = brute_force(&vector(1_000_000 + q), K, |_| true);
        found += search(q)
            .iter()
            .filter(|h| want.contains(&h.doc_id))
            .count();
    }
    found as f64 / (queries * K) as f64
}

#[test]
fn the_exact_search_is_the_brute_force_top_k() {
    for (name, format) in formats() {
        with_index(name, format, |reader| {
            let opened = reader.open_segments().unwrap();
            let segs = opened.as_open_segments();
            for q in 0..10 {
                let t = target(q);
                let hits = knn_top_hits_exact(&segs, "v", &t, K, None, None).unwrap();
                let KnnTarget::Float(v) = &t else {
                    unreachable!()
                };
                assert_eq!(ids(&hits), brute_force(v, K, |_| true), "{name} q{q}");
                let even_hits = knn_top_hits_exact(&segs, "v", &t, K, Some(&even()), None).unwrap();
                assert_eq!(
                    ids(&even_hits),
                    brute_force(v, K, |id| id.is_multiple_of(2)),
                    "{name} q{q} filtered"
                );
            }
        });
    }
}

#[test]
fn effort_one_is_the_search_as_asked_and_more_effort_finds_more() {
    for (name, format) in formats() {
        let ivf = name.ends_with("ivf");
        with_index(name, format, |reader| {
            let opened = reader.open_segments().unwrap();
            let segs = opened.as_open_segments();
            let candidates = if ivf { K } else { 40 };
            let at = |q: usize, e: f32| {
                knn_top_hits_at_effort(&segs, "v", &target(q), K, candidates, None, None, 0, &[], e)
                    .unwrap()
            };
            for q in 0..5 {
                let plain =
                    knn_top_hits_with(&segs, "v", &target(q), K, candidates, None, 0).unwrap();
                assert_eq!(at(q, 1.0), plain, "{name} q{q}");
            }
            let low = recall(30, |q| at(q, effort(0)));
            let high = recall(30, |q| at(q, effort(LEVELS - 1)));
            assert!(high >= low, "{name}: {low} -> {high}");
            assert!(high >= 0.99, "{name}: deepest effort recall {high}");
            assert!(low < 0.99, "{name}: shallowest effort recall {low}");
        });
    }
}

#[test]
fn a_tuner_climbs_for_a_high_target_and_descends_for_a_low_one() {
    for (name, format) in formats() {
        let ivf = name.ends_with("ivf");
        with_index(name, format, |reader| {
            let opened = reader.open_segments().unwrap();
            let segs = opened.as_open_segments();
            let candidates = if ivf { K } else { 40 };
            let tune = |recall: f32, filter: Option<&Clause>| {
                let tuner = RecallTuner::new(
                    RecallTarget::new(recall, 1.0)
                        .unwrap()
                        .with_budget(1e9)
                        .unwrap(),
                );
                for q in 0..80 {
                    let hits = knn_top_hits_tuned(
                        &segs,
                        "v",
                        &target(q),
                        K,
                        candidates,
                        filter,
                        None,
                        0,
                        &[],
                        &tuner,
                    )
                    .unwrap();
                    assert_eq!(hits.len(), K);
                }
                tuner.stats()
            };
            let high = tune(1.0, None);
            assert_eq!(high.samples, 80, "{name}");
            assert_eq!(high.queries, 80, "{name}");
            assert!(high.level > BASE_LEVEL, "{name}: {high:?}");
            assert!(high.check_nanos > 0 && high.search_nanos > 0, "{name}");
            let low = tune(0.05, Some(&even()));
            assert!(low.level < BASE_LEVEL, "{name}: {low:?}");
            assert!(low.recall.unwrap() >= 0.05, "{name}: {low:?}");
        });
    }
}

#[test]
fn an_unsampled_search_is_not_checked() {
    let (name, format) = formats().into_iter().next().unwrap();
    with_index(name, format, |reader| {
        let opened = reader.open_segments().unwrap();
        let segs = opened.as_open_segments();
        let tuner = RecallTuner::new(RecallTarget::new(0.9, 0.5).unwrap());
        let first = tuner.begin();
        assert!(first.sampled);
        let second = tuner.begin();
        assert!(!second.sampled);
        let hits = knn_top_hits_at_effort(
            &segs,
            "v",
            &target(0),
            K,
            K,
            None,
            None,
            0,
            &[],
            second.effort,
        )
        .unwrap();
        let check = |search| {
            check_sample(
                &segs,
                "v",
                &target(0),
                K,
                K,
                None,
                None,
                0,
                &[],
                &tuner,
                search,
                &hits,
            )
            .unwrap()
        };
        check(&second);
        assert_eq!(tuner.stats().samples, 0);
        check(&first);
        assert_eq!(tuner.stats().samples, 1);
    });
}
