//! KNN over a nested field ([`knn_top_hits_nested`]): the best child of each
//! of the best parents, and with `expand` every child of those parents.

#![allow(clippy::arithmetic_side_effects)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use lucene_codecs::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{DocumentVector, ExplicitDocument, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::join::BitSetProducer;
use lucene_search::knn_search::{knn_top_hits_nested, KnnTarget};
use lucene_search::multi_segment::OpenSegment;
use lucene_store::FsDirectory;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::test_support::TempDir;

const DIM: usize = 8;
/// Each block: `BLOCK - 1` children, then the parent.
const BLOCK: usize = 4;
const PER_SEGMENT: usize = 200;
const SEGMENTS: usize = 3;
const DOCS: usize = PER_SEGMENT * SEGMENTS;

fn vector(id: usize) -> Vec<f32> {
    let mut x = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    (0..DIM)
        .map(|_| {
            x ^= x >> 29;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn is_parent(id: usize) -> bool {
    id % BLOCK == BLOCK - 1
}

/// Every block's second child has no vector.
fn has_vector(id: usize) -> bool {
    !is_parent(id) && id % BLOCK != 1
}

#[derive(Debug)]
struct EveryFourth;

impl BitSetProducer for EveryFourth {
    fn bit_set(&self, leaf: &OpenSegment<'_>) -> lucene_search::Result<Option<Arc<FixedBitSet>>> {
        let max_doc = leaf.reader.unwrap().max_doc as usize;
        let mut bits = FixedBitSet::new(max_doc);
        for d in (BLOCK - 1..max_doc).step_by(BLOCK) {
            bits.set(d);
        }
        Ok(Some(Arc::new(bits)))
    }

    fn key(&self) -> String {
        "every-fourth".into()
    }
}

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
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        DIM as i32,
        VectorEncoding::Float32,
        VectorSimilarityFunction::Euclidean,
    ))
    .unwrap();
    w.set_merge_policy(None);
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    for s in 0..SEGMENTS {
        let (docs, vectors): (Vec<_>, Vec<_>) = (s * PER_SEGMENT..(s + 1) * PER_SEGMENT)
            .map(|id| {
                let v = if has_vector(id) {
                    vec![DocumentVector::float32("v", vector(id))]
                } else {
                    Vec::new()
                };
                (ExplicitDocument::default(), v)
            })
            .unzip();
        w.add_explicit_documents_with_vectors_in(None, docs, vectors)
            .unwrap();
        w.flush().unwrap();
    }
    w.commit().unwrap();
    w.into_writer().unwrap().close().unwrap();
}

fn score(target: &[f32], id: usize) -> f32 {
    VectorSimilarityFunction::Euclidean.score(target, &vector(id))
}

#[test]
fn a_nested_search_keeps_one_child_per_parent_and_expands_to_all() {
    let tmp = TempDir::new("knn-nested");
    let dir = FsDirectory::open(&tmp);
    build(&dir);
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), SEGMENTS);
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let k = 10;
    let queries = 20;
    let mut recalled = 0;
    for q in 0..queries {
        let target = vector(100_000 + q);
        let t = KnnTarget::Float(target.clone());
        let hits =
            knn_top_hits_nested(&segments, "v", &t, k, 50, None, &EveryFourth, false).unwrap();
        assert_eq!(hits.len(), k);
        let parents: BTreeSet<usize> = hits.iter().map(|h| h.doc_id as usize / BLOCK).collect();
        assert_eq!(parents.len(), k, "one child per parent");
        for h in &hits {
            assert!(has_vector(h.doc_id as usize));
            assert_eq!(h.score, score(&target, h.doc_id as usize));
        }
        // The best child of every parent, exhaustively.
        let mut best: BTreeMap<usize, f32> = BTreeMap::new();
        for id in (0..DOCS).filter(|&id| has_vector(id)) {
            let s = score(&target, id);
            let e = best.entry(id / BLOCK).or_insert(f32::NEG_INFINITY);
            *e = e.max(s);
        }
        let mut exact: Vec<f32> = best.into_values().collect();
        exact.sort_by(|a, b| b.total_cmp(a));
        recalled += hits.iter().filter(|h| h.score >= exact[k - 1]).count();

        let expanded =
            knn_top_hits_nested(&segments, "v", &t, k, 50, None, &EveryFourth, true).unwrap();
        let mut want: Vec<(usize, f32)> = parents
            .iter()
            .flat_map(|p| p * BLOCK..(p + 1) * BLOCK)
            .filter(|&id| has_vector(id))
            .map(|id| (id, score(&target, id)))
            .collect();
        want.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let got: Vec<(usize, f32)> = expanded
            .iter()
            .map(|h| (h.doc_id as usize, h.score))
            .collect();
        assert_eq!(got, want);
    }
    assert!(
        recalled * 100 >= queries * k * 90,
        "recall {recalled}/{}",
        queries * k
    );
}
