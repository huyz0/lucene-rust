//! Grouped KNN ([`knn_top_hits_grouped`]): the best documents of each of
//! the best `k` groups of a keyword or numeric doc-values field, over HNSW
//! and IVF fields of several segments, checked against an exhaustive scan,
//! with a filter, with deletions, and with groups whose documents all lie
//! close together (which crowd an ungrouped search's top `k`).

#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, HashMap};

use lucene_codecs::field_infos::{
    DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions, VectorEncoding,
    VectorSimilarityFunction,
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
use lucene_search::knn_search::{knn_top_hits_grouped, KnnGroupBy, KnnTarget};
use lucene_search::query::{Clause, TermQuery};
use lucene_search::top_field::CollapseKind;
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const DIM: usize = 16;
const PER_SEGMENT: usize = 2000;
const SEGMENTS: usize = 3;
const DOCS: usize = PER_SEGMENT * SEGMENTS;
const GROUPS: usize = 300;

fn noise(seed: u64) -> impl FnMut() -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    move || {
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Document `id`'s group: sizes from 1 to about 40, spread over the
/// segments; one document in 97 has none.
fn group(id: usize) -> Option<usize> {
    if id.is_multiple_of(97) {
        return None;
    }
    // Group g holds the ids whose hash lands in it; small groups first.
    let h = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40;
    let x = (h % 10_000) as f64 / 10_000.0;
    Some(((x * x) * GROUPS as f64) as usize)
}

/// A group's documents lie close together around the group's own point,
/// itself near one of 20 centres: siblings score alike.
fn vector(id: usize) -> Vec<f32> {
    let g = group(id).unwrap_or(GROUPS + id % 7);
    let mut centre = noise((g % 20) as u64 + 1_000_000);
    let mut own = noise(g as u64 + 2_000_000);
    let mut jitter = noise(id as u64);
    (0..DIM)
        .map(|_| centre() * 4.0 + own() * 1.0 + jitter() * 0.1)
        .collect()
}

/// One document in 211, about ten a segment.
fn rare(id: usize) -> bool {
    id.is_multiple_of(211)
}

fn doc(id: usize) -> (ExplicitDocument, Vec<DocumentVector>) {
    let mut d = ExplicitDocument::default();
    d.stored.push(StoredField {
        field_number: 0,
        value: FieldValue::String(id.to_string()),
    });
    let mut terms = InvertedTerms::default();
    let parity: &[u8] = if id.is_multiple_of(3) {
        b"third"
    } else {
        b"rest"
    };
    terms.push::<Vec<u8>>(parity, 1, &[], &[], &[]);
    d.fields.inverted.push(InvertedField {
        field_number: 1,
        terms,
        norm: None,
    });
    if rare(id) {
        let mut terms = InvertedTerms::default();
        terms.push::<Vec<u8>>(b"yes", 1, &[], &[], &[]);
        d.fields.inverted.push(InvertedField {
            field_number: 2,
            terms,
            norm: None,
        });
    }
    if let Some(g) = group(id) {
        d.fields.doc_values.push(StoredField {
            field_number: 3,
            value: FieldValue::Binary(format!("g{g}").into_bytes()),
        });
        d.fields.doc_values.push(StoredField {
            field_number: 4,
            value: FieldValue::Long(g as i64 * 1_000_003 - 7),
        });
    }
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
    for name in ["parity", "rare"] {
        w.register_field(
            FieldInfo::new(name, 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
        )
        .unwrap();
    }
    w.register_field(FieldInfo::new("cat", 0).with_doc_values(
        DocValuesType::SortedSet,
        DocValuesSkipIndexType::None,
        -1,
    ))
    .unwrap();
    w.register_field(FieldInfo::new("num", 0).with_doc_values(
        DocValuesType::SortedNumeric,
        DocValuesSkipIndexType::None,
        -1,
    ))
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

fn score(a: &[f32], b: &[f32]) -> f32 {
    VectorSimilarityFunction::Euclidean.score(a, b)
}

/// The exhaustive answer: ids of the best `per_group` documents of each of
/// the best `k` groups (no value is a group of its own), best first.
fn exact(
    target: &[f32],
    k: usize,
    per_group: usize,
    keep: impl Fn(usize) -> bool,
) -> Vec<(usize, f32)> {
    let mut by: HashMap<Option<usize>, Vec<(usize, f32)>> = HashMap::new();
    for id in (0..DOCS).filter(|&id| keep(id)) {
        by.entry(group(id))
            .or_default()
            .push((id, score(target, &vector(id))));
    }
    let order = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    let mut lists: Vec<Vec<(usize, f32)>> = by
        .into_values()
        .map(|mut l| {
            l.sort_by(order);
            l.truncate(per_group);
            l
        })
        .collect();
    lists.sort_by(|a, b| order(&a[0], &b[0]));
    lists.truncate(k);
    let mut out: Vec<(usize, f32)> = lists.into_iter().flatten().collect();
    out.sort_by(order);
    out
}

fn target(q: usize) -> Vec<f32> {
    let mut v = vector(q * 131 % DOCS);
    let mut n = noise(q as u64 + 9_000_000);
    for x in &mut v {
        *x += n() * 0.5;
    }
    v
}

/// Global doc id to stored id.
fn ids(reader: &DirectoryReader) -> Vec<usize> {
    let mut out = Vec::new();
    for seg in reader.segment_readers() {
        for d in 0..seg.max_doc {
            let stored = seg.stored_document(d).unwrap().unwrap();
            out.push(match &stored.fields[0].value {
                FieldValue::String(s) => s.parse().unwrap(),
                other => panic!("id {other:?}"),
            });
        }
    }
    out
}

/// Recall of the grouped searches' documents against the exhaustive ones,
/// every answer checked: at most `k` groups, at most `per_group` documents
/// each, exact scores, best first, only documents `keep` passes.
#[allow(clippy::too_many_arguments)]
fn recall(
    dir: &FsDirectory,
    kind: CollapseKind,
    k: usize,
    candidates: usize,
    per_group: usize,
    filter: Option<&Clause>,
    keep: impl Fn(usize) -> bool + Copy,
) -> f64 {
    let reader = DirectoryReader::open(dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let id_of = ids(&reader);
    let group_by = KnnGroupBy {
        field: match kind {
            CollapseKind::Keyword => "cat".into(),
            CollapseKind::Numeric => "num".into(),
        },
        kind,
        per_group,
    };
    let queries = 30;
    let (mut found, mut wanted) = (0, 0);
    for q in 0..queries {
        let t = target(q);
        let hits = knn_top_hits_grouped(
            &segments,
            "v",
            &KnnTarget::Float(t.clone()),
            k,
            candidates,
            filter,
            &group_by,
            0,
        )
        .unwrap();
        let mut per: BTreeMap<Option<usize>, usize> = BTreeMap::new();
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score, "hits out of order");
        }
        for h in &hits {
            let id = id_of[h.doc_id as usize];
            assert!(keep(id), "doc {id} does not pass the filter");
            let s = score(&t, &vector(id));
            assert!((s - h.score).abs() < 1e-5, "doc {id}: {} != {s}", h.score);
            *per.entry(group(id)).or_default() += 1;
        }
        assert!(per.len() <= k, "{} groups", per.len());
        assert!(per.values().all(|&n| n <= per_group));
        let want = exact(&t, k, per_group, keep);
        let got: Vec<usize> = hits.iter().map(|h| id_of[h.doc_id as usize]).collect();
        wanted += want.len();
        found += want.iter().filter(|(id, _)| got.contains(id)).count();
    }
    let r = found as f64 / wanted as f64;
    eprintln!(
        "{kind:?} k={k} per_group={per_group} filtered={}: recall {r:.3}",
        filter.is_some()
    );
    r
}

fn third(id: usize) -> bool {
    id.is_multiple_of(3)
}

fn check(format: KnnVectorsFormat, name: &str) {
    let tmp = TempDir::new(name);
    let dir = FsDirectory::open(&tmp);
    build(&dir, format);
    let third_clause = Clause::Term(TermQuery::new("parity", b"third".to_vec()));
    for kind in [CollapseKind::Keyword, CollapseKind::Numeric] {
        let r = recall(&dir, kind, 10, 100, 1, None, |_| true);
        assert!(r >= 0.95, "{name} {kind:?} best per group: recall {r}");
        let r = recall(&dir, kind, 10, 10, 3, None, |_| true);
        assert!(r >= 0.9, "{name} {kind:?} three per group: recall {r}");
        let r = recall(&dir, kind, 10, 100, 2, Some(&third_clause), third);
        assert!(r >= 0.9, "{name} {kind:?} filtered: recall {r}");
    }
    // A filter so selective every segment scores it exhaustively: exact.
    let rare_clause = Clause::Term(TermQuery::new("rare", b"yes".to_vec()));
    for kind in [CollapseKind::Keyword, CollapseKind::Numeric] {
        let r = recall(&dir, kind, 5, 5, 2, Some(&rare_clause), rare);
        assert_eq!(r, 1.0, "{name} {kind:?} rare");
    }
}

#[test]
fn hnsw_groups_by_keyword_and_number() {
    check(KnnVectorsFormat::hnsw(16, 100).unwrap(), "knn-grouped-hnsw");
}

#[test]
fn ivf_groups_by_keyword_and_number() {
    check(KnnVectorsFormat::ivf(64, 0).unwrap(), "knn-grouped-ivf");
}

#[test]
fn deleted_documents_are_skipped() {
    let tmp = TempDir::new("knn-grouped-deletes");
    let dir = FsDirectory::open(&tmp);
    build(&dir, KnnVectorsFormat::hnsw(16, 100).unwrap());
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
        bytes: b"third".to_vec(),
    }])
    .unwrap();
    w.commit().unwrap();
    w.close().unwrap();
    let r = recall(&dir, CollapseKind::Keyword, 10, 100, 2, None, |id| {
        !third(id)
    });
    assert!(r >= 0.9, "recall {r}");
}

#[test]
fn missing_values_wrong_types_and_unknown_fields() {
    let tmp = TempDir::new("knn-grouped-multi");
    let dir = FsDirectory::open(&tmp);
    build(&dir, KnnVectorsFormat::hnsw(16, 100).unwrap());
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    // `parity` has no doc values: every document is in the missing group.
    let by = |field: &str, kind| KnnGroupBy {
        field: field.into(),
        kind,
        per_group: 4,
    };
    let hits = knn_top_hits_grouped(
        &segments,
        "v",
        &KnnTarget::Float(target(1)),
        3,
        10,
        None,
        &by("parity", CollapseKind::Keyword),
        0,
    )
    .unwrap();
    assert_eq!(hits.len(), 4);
    // `num` read as a keyword: the wrong doc-values type.
    assert!(knn_top_hits_grouped(
        &segments,
        "v",
        &KnnTarget::Float(target(1)),
        3,
        10,
        None,
        &by("num", CollapseKind::Keyword),
        0,
    )
    .is_err());
    // An unknown vector field: nothing.
    assert!(knn_top_hits_grouped(
        &segments,
        "nope",
        &KnnTarget::Float(target(1)),
        3,
        10,
        None,
        &by("cat", CollapseKind::Keyword),
        0,
    )
    .unwrap()
    .is_empty());
}
