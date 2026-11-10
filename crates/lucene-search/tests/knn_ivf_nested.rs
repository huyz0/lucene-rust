//! Nested KNN ([`knn_top_hits_nested_with`]) over a field in
//! `LuceneRustIvfVectorsFormat`: the clusters are probed for the best child
//! of each parent, checked against an exhaustive scan, with a filter on the
//! children and with deletions.

#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lucene_codecs::field_infos::{
    FieldInfo, IndexOptions, VectorEncoding, VectorSimilarityFunction,
};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{
    DocumentVector, ExplicitDocument, IndexWriter, InvertedField, InvertedTerms,
};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::join::BitSetProducer;
use lucene_search::knn_search::{knn_top_hits_nested_with, KnnTarget};
use lucene_search::multi_segment::OpenSegment;
use lucene_search::query::{Clause, TermQuery};
use lucene_store::FsDirectory;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::test_support::TempDir;

const DIM: usize = 16;
const PARENTS_PER_SEGMENT: usize = 400;
const SEGMENTS: usize = 2;

fn noise(seed: u64) -> impl FnMut() -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    move || {
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Parent `p` has 1-8 children whose vectors lie close together around one
/// of 20 centres, as a document's chunks do.
fn children(p: usize) -> usize {
    1 + p * 7 % 8
}

fn child_vector(p: usize, c: usize) -> Vec<f32> {
    let mut centre = noise((p % 20) as u64 + 1_000_000);
    let mut parent = noise(p as u64 + 2_000_000);
    let mut jitter = noise((p * 16 + c) as u64);
    (0..DIM)
        .map(|_| centre() * 4.0 + parent() * 1.0 + jitter() * 0.2)
        .collect()
}

/// Every block: the children, then the parent. Returns, per doc, `Some((p,
/// c))` for a child and `None` for a parent.
fn layout() -> Vec<Option<(usize, usize)>> {
    (0..PARENTS_PER_SEGMENT * SEGMENTS)
        .flat_map(|p| {
            (0..children(p))
                .map(move |c| Some((p, c)))
                .chain(std::iter::once(None))
        })
        .collect()
}

#[derive(Debug)]
struct Parents;

impl BitSetProducer for Parents {
    fn bit_set(&self, leaf: &OpenSegment<'_>) -> lucene_search::Result<Option<Arc<FixedBitSet>>> {
        let reader = leaf.reader.unwrap();
        let max_doc = reader.max_doc as usize;
        let docs = layout();
        let base = leaf.doc_base as usize;
        let mut bits = FixedBitSet::new(max_doc);
        for d in 0..max_doc {
            if docs[base + d].is_none() {
                bits.set(d);
            }
        }
        Ok(Some(Arc::new(bits)))
    }

    fn key(&self) -> String {
        "parents".into()
    }
}

/// Children carry `parity` `even` or `odd` by child number.
fn build(dir: &FsDirectory) {
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
    w.set_knn_vectors_format_for_field("v", KnnVectorsFormat::ivf(64, 0).unwrap());
    w.set_ram_buffer_size_mb(256.0).unwrap();
    w.set_merge_policy(None);
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    let docs = layout();
    let mut start = 0;
    for s in 0..SEGMENTS {
        let end = docs
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_none())
            .nth((s + 1) * PARENTS_PER_SEGMENT - 1)
            .unwrap()
            .0
            + 1;
        let (batch, vectors): (Vec<_>, Vec<_>) = docs[start..end]
            .iter()
            .map(|d| match d {
                Some((p, c)) => {
                    let mut doc = ExplicitDocument::default();
                    let mut terms = InvertedTerms::default();
                    let term: &[u8] = if c % 2 == 0 { b"even" } else { b"odd" };
                    terms.push::<Vec<u8>>(term, 1, &[], &[], &[]);
                    doc.fields.inverted.push(InvertedField {
                        field_number: 0,
                        terms,
                        norm: None,
                    });
                    (
                        doc,
                        vec![DocumentVector::float32("v", child_vector(*p, *c))],
                    )
                }
                None => (ExplicitDocument::default(), Vec::new()),
            })
            .unzip();
        w.add_explicit_documents_with_vectors_in(None, batch, vectors)
            .unwrap();
        w.flush().unwrap();
        start = end;
    }
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
}

fn score(target: &[f32], v: &[f32]) -> f32 {
    VectorSimilarityFunction::Euclidean.score(target, v)
}

/// The exhaustive best `k` parents' best children: (doc, score).
fn exact(target: &[f32], k: usize, keep: impl Fn(usize, usize) -> bool) -> Vec<(usize, f32)> {
    let mut best: BTreeMap<usize, (usize, f32)> = BTreeMap::new();
    for (doc, d) in layout().iter().enumerate() {
        let Some((p, c)) = *d else { continue };
        if !keep(p, c) {
            continue;
        }
        let s = score(target, &child_vector(p, c));
        let e = best.entry(p).or_insert((doc, f32::NEG_INFINITY));
        if s > e.1 {
            *e = (doc, s);
        }
    }
    let mut v: Vec<(usize, f32)> = best.into_values().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(k);
    v
}

fn target(q: usize) -> Vec<f32> {
    let mut v = child_vector(q * 37 % (PARENTS_PER_SEGMENT * SEGMENTS), 0);
    let mut n = noise(q as u64 + 9_000_000);
    for x in &mut v {
        *x += n() * 0.5;
    }
    v
}

/// Recall of the nested search's parents against the exhaustive ones, with
/// every hit checked: one per parent, a child the filter passes, its exact
/// score.
fn recall(
    dir: &FsDirectory,
    nprobes: usize,
    filter: Option<&Clause>,
    keep: impl Fn(usize, usize) -> bool + Copy,
) -> f64 {
    let reader = DirectoryReader::open(dir).unwrap();
    assert_eq!(reader.segment_readers().len(), SEGMENTS);
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let docs = layout();
    let (k, queries) = (10, 30);
    let mut found = 0;
    for q in 0..queries {
        let t = target(q);
        let hits = knn_top_hits_nested_with(
            &segments,
            "v",
            &KnnTarget::Float(t.clone()),
            k,
            k,
            filter,
            &Parents,
            false,
            nprobes,
        )
        .unwrap();
        assert_eq!(hits.len(), k);
        let mut parents = BTreeSet::new();
        for h in &hits {
            let (p, c) = docs[h.doc_id as usize].expect("a child");
            assert!(keep(p, c), "doc {} should not match", h.doc_id);
            assert!(parents.insert(p), "parent {p} twice");
            assert_eq!(h.score, score(&t, &child_vector(p, c)));
        }
        let want: BTreeSet<usize> = exact(&t, k, keep).into_iter().map(|h| h.0).collect();
        found += hits
            .iter()
            .filter(|h| want.contains(&(h.doc_id as usize)))
            .count();
    }
    found as f64 / (queries * k) as f64
}

#[test]
fn nested_ivf_probes_for_the_best_child_of_each_parent() {
    let tmp = TempDir::new("knn-ivf-nested");
    let dir = FsDirectory::open(&tmp);
    build(&dir);
    // Every cluster probed: the exhaustive answer.
    assert_eq!(recall(&dir, 100_000, None, |_, _| true), 1.0);
    // The calibrated depth, and a single probe deepened until `k` parents
    // are in reach.
    let calibrated = recall(&dir, 0, None, |_, _| true);
    assert!(calibrated >= 0.9, "calibrated recall {calibrated}");
    let one = recall(&dir, 1, None, |_, _| true);
    assert!(one >= 0.5, "one-probe recall {one}");
    // A filter on the children: half of them.
    let even = Clause::Term(TermQuery::new("parity", b"even".to_vec()));
    let filtered = recall(&dir, 0, Some(&even), |_, c| c % 2 == 0);
    assert!(filtered >= 0.9, "filtered recall {filtered}");
    assert_eq!(recall(&dir, 100_000, Some(&even), |_, c| c % 2 == 0), 1.0);
}
