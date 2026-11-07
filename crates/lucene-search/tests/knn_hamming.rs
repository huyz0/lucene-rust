//! Packed bit vectors in the OpenSearch k-NN plugin's
//! `KNN990HnswBinaryVectorsFormat`: flushed and merged graphs and every
//! search score by Hamming distance, `1 / (1 + popcount(a ^ b))`, while the
//! field records `EUCLIDEAN` as k-NN's does.

#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::field_infos::{FieldInfo, VectorEncoding, VectorSimilarityFunction};
use lucene_codecs::per_field_knn_vectors::KnnVectorsFormat;
use lucene_codecs::stored_fields::{FieldValue, StoredField};
use lucene_index::concurrent_writer::ConcurrentIndexWriter;
use lucene_index::index_writer::{DocumentVector, ExplicitDocument, IndexWriter};
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

/// Bytes per vector: 64 bits.
const BYTES: usize = 8;
const PER_SEGMENT: usize = 150;
const SEGMENTS: usize = 3;
const DOCS: usize = PER_SEGMENT * SEGMENTS;

fn vector(id: usize) -> Vec<u8> {
    let mut x = (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x.to_le_bytes()[..BYTES].to_vec()
}

fn hamming(a: &[u8], b: &[u8]) -> f32 {
    let bits: u32 = a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum();
    1.0 / (1.0 + bits as f32)
}

fn build(dir: &FsDirectory, merge: bool) {
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
        BYTES as i32,
        VectorEncoding::Byte,
        VectorSimilarityFunction::Euclidean,
    ))
    .unwrap();
    w.set_knn_vectors_format_for_field("v", KnnVectorsFormat::hnsw_hamming(16, 100).unwrap());
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
                (d, vec![DocumentVector::byte("v", vector(id))])
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

fn check(merge: bool) {
    let tmp = TempDir::new("knn-hamming");
    let dir = FsDirectory::open(&tmp);
    build(&dir, merge);
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
        assert_eq!(
            field.vector_similarity_function,
            VectorSimilarityFunction::Euclidean
        );
        assert!(field.attributes.iter().any(|(k, v)| {
            k == "PerFieldKnnVectorsFormat.format" && v == "KNN990HnswBinaryVectorsFormat"
        }));
    }
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
            &KnnTarget::Byte(target.clone()),
            k,
            100,
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), k);
        // Doc ids are ids: one writer, segments in order.
        for h in &hits {
            let want = hamming(&target, &vector(h.doc_id as usize));
            assert_eq!(h.score, want, "doc {} score", h.doc_id);
        }
        let mut exact: Vec<f32> = (0..DOCS).map(|id| hamming(&target, &vector(id))).collect();
        exact.sort_by(|a, b| b.total_cmp(a));
        let kth = exact[k - 1];
        recalled += hits.iter().filter(|h| h.score >= kth).count();
    }
    // The graph is built by Hamming distance: a Euclidean-on-bytes graph
    // would rank these neighbours far worse.
    assert!(
        recalled * 100 >= queries * k * 90,
        "recall {recalled}/{}",
        queries * k
    );
}

#[test]
fn flushed_segments_score_by_hamming_distance() {
    check(false);
}

#[test]
fn a_merged_segment_scores_by_hamming_distance() {
    check(true);
}
