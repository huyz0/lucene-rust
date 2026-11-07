//! Float vectors in the OpenSearch k-NN plugin's
//! `KNN1040HnswHalfFloatVectorsFormat`: stored as half floats, with no
//! similarity in `.vemf`, so flushed and merged segments and every search
//! score the half-float values by the field's similarity.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_codecs::vectors::{f16_to_f32, f32_to_f16};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{DocumentVector, ExplicitDocument, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

const DIM: usize = 12;
const PER_SEGMENT: usize = 150;
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

fn stored(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&c| f16_to_f32(f32_to_f16(c))).collect()
}

fn build(dir: &FsDirectory, similarity: VectorSimilarityFunction, merge: bool) {
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
    w.register_field(FieldInfo::new("v", 0).with_vectors(
        DIM as i32,
        VectorEncoding::Float32,
        similarity,
    ))
    .unwrap();
    w.set_knn_vectors_format_for_field("v", KnnVectorsFormat::hnsw_half_float(16, 100).unwrap());
    w.set_merge_policy(None);
    let w = ConcurrentIndexWriter::new(w, 1).unwrap();
    for s in 0..SEGMENTS {
        let (docs, vectors): (Vec<_>, Vec<_>) = (s * PER_SEGMENT..(s + 1) * PER_SEGMENT)
            .map(|id| {
                let mut d = ExplicitDocument::default();
                d.stored.push(StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.to_string()),
                });
                (d, vec![DocumentVector::float32("v", vector(id))])
            })
            .unzip();
        w.add_explicit_documents_with_vectors_in(None, docs, vectors)
            .unwrap();
        w.flush().unwrap();
    }
    w.commit().unwrap();
    let mut w = w.into_writer().unwrap();
    if merge {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    w.close().unwrap();
}

fn check(similarity: VectorSimilarityFunction, merge: bool) {
    let tmp = TempDir::new("knn-half-float");
    let dir = FsDirectory::open(&tmp);
    build(&dir, similarity, merge);
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(
        reader.segment_readers().len(),
        if merge { 1 } else { SEGMENTS }
    );
    for seg in reader.segment_readers() {
        let field = seg
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "v")
            .unwrap();
        assert!(field.attributes.iter().any(|(k, v)| {
            k == "PerFieldKnnVectorsFormat.format" && v == "KNN1040HnswHalfFloatVectorsFormat"
        }));
        let flat = seg.flat_vectors_reader().unwrap().unwrap();
        assert!(flat.stores_half_floats());
        assert_eq!(flat.field(field.number).unwrap().similarity, similarity);
    }
    // k-NN scores half-float cosine vectors (which it normalizes) by `(1 + dot) / 2`.
    let scoring = if similarity == VectorSimilarityFunction::Cosine {
        VectorSimilarityFunction::DotProduct
    } else {
        similarity
    };
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let k = 10;
    let mut recalled = 0;
    let queries = 20;
    for q in 0..queries {
        let target = vector(100_000 + q);
        let hits = knn_top_hits(
            &segments,
            "v",
            &KnnTarget::Float(target.clone()),
            k,
            100,
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), k);
        // Doc ids are ids: one writer, segments in order.
        for h in &hits {
            let want = scoring.score(&target, &stored(&vector(h.doc_id as usize)));
            assert_eq!(h.score, want, "doc {} score", h.doc_id);
        }
        let mut exact: Vec<f32> = (0..DOCS)
            .map(|id| scoring.score(&target, &stored(&vector(id))))
            .collect();
        exact.sort_by(|a, b| b.total_cmp(a));
        let kth = exact[k - 1];
        recalled += hits.iter().filter(|h| h.score >= kth).count();
    }
    assert!(
        recalled * 100 >= queries * k * 90,
        "recall {recalled}/{}",
        queries * k
    );
}

#[test]
fn flushed_segments_score_their_half_floats() {
    check(VectorSimilarityFunction::Euclidean, false);
}

#[test]
fn a_merged_segment_scores_its_half_floats() {
    check(VectorSimilarityFunction::Euclidean, true);
}

#[test]
fn the_similarity_comes_from_the_field() {
    check(VectorSimilarityFunction::Cosine, true);
    check(VectorSimilarityFunction::Cosine, false);
    check(VectorSimilarityFunction::MaximumInnerProduct, false);
}
