#![cfg(test)]
use super::*;
use lucene_codecs::points::{self, WritePointsField};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::FsDirectory;

fn long_bytes(v: i64) -> [u8; 8] {
    ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()
}

fn sci(segment_name: &str, del_gen: i64, del_count: i32) -> SegmentCommitInfo {
    SegmentCommitInfo {
        segment_name: segment_name.to_string(),
        segment_id: [7u8; ID_LENGTH],
        codec_name: "Lucene104".to_string(),
        del_gen,
        del_count,
        field_infos_gen: -1,
        doc_values_gen: -1,
        soft_del_count: 0,
        sci_id: None,
        field_infos_files: vec![],
        dv_update_files: vec![],
        ..Default::default()
    }
}

use lucene_util::test_support::TempDir;

/// A scratch directory that removes itself when the test ends -- unless
/// the test is panicking, in which case its bytes stay for inspection.
fn tempdir() -> TempDir {
    TempDir::new("points-delete")
}

/// Builds a tiny in-memory single-dimension `LongPoint`-shaped segment:
/// doc 0 -> 10, doc 1 -> 20, doc 2 -> 30, doc 3 -> 40, doc 4 -> 50.
fn build_single_dim_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, [u8; ID_LENGTH]) {
    let segment_id = [3u8; ID_LENGTH];
    let points: Vec<(i32, Vec<u8>)> = vec![
        (0, long_bytes(10).to_vec()),
        (1, long_bytes(20).to_vec()),
        (2, long_bytes(30).to_vec()),
        (3, long_bytes(40).to_vec()),
        (4, long_bytes(50).to_vec()),
    ];
    let field = WritePointsField {
        field_number: 1,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = points::write(&[field], 512, &segment_id, "").unwrap();
    (kdm, kdi, kdd, segment_id)
}

/// 2D fixture (e.g. `LatLonPoint`-shaped): doc 0 -> (0, 0), doc 1 -> (10,
/// 10), doc 2 -> (20, 20), doc 3 -> (10, 100) -- dimension 0 alone would
/// match doc 3 but dimension 1 must independently be in range too.
fn build_two_dim_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, [u8; ID_LENGTH]) {
    let segment_id = [5u8; ID_LENGTH];
    let pack = |a: i64, b: i64| -> Vec<u8> {
        let mut v = long_bytes(a).to_vec();
        v.extend_from_slice(&long_bytes(b));
        v
    };
    let points: Vec<(i32, Vec<u8>)> = vec![
        (0, pack(0, 0)),
        (1, pack(10, 10)),
        (2, pack(20, 20)),
        (3, pack(10, 100)),
    ];
    let field = WritePointsField {
        field_number: 1,
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = points::write(&[field], 512, &segment_id, "").unwrap();
    (kdm, kdi, kdd, segment_id)
}

// --- resolve_points_range_doc_ids ---

#[test]
fn exact_range_matches_only_docs_within_bounds() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(15);
    let max = long_bytes(35);
    let docs = resolve_points_range_doc_ids(&reader, None, 1, &min, &max).unwrap();
    assert_eq!(docs, vec![1, 2]); // values 20, 30
}

#[test]
fn boundary_values_are_inclusive_on_both_ends() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(10);
    let max = long_bytes(30);
    let docs = resolve_points_range_doc_ids(&reader, None, 1, &min, &max).unwrap();
    assert_eq!(docs, vec![0, 1, 2]); // 10, 20, 30 all included
}

#[test]
fn range_matching_zero_docs_is_empty() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(1000);
    let max = long_bytes(2000);
    let docs = resolve_points_range_doc_ids(&reader, None, 1, &min, &max).unwrap();
    assert!(docs.is_empty());
}

#[test]
fn range_matching_all_docs() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(i64::MIN);
    let max = long_bytes(i64::MAX);
    let docs = resolve_points_range_doc_ids(&reader, None, 1, &min, &max).unwrap();
    assert_eq!(docs, vec![0, 1, 2, 3, 4]);
}

#[test]
fn unknown_field_number_is_empty_not_an_error() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let min = long_bytes(i64::MIN);
    let max = long_bytes(i64::MAX);
    let docs = resolve_points_range_doc_ids(&reader, None, 99, &min, &max).unwrap();
    assert!(docs.is_empty());
}

#[test]
fn live_docs_filter_excludes_already_deleted_docs() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let mut live = FixedBitSet::new(5);
    for i in 0..5 {
        live.set(i);
    }
    live.clear(1); // doc 1 (value 20) already deleted
    let min = long_bytes(15);
    let max = long_bytes(35);
    let docs = resolve_points_range_doc_ids(&reader, Some(&live), 1, &min, &max).unwrap();
    assert_eq!(docs, vec![2]); // doc 1 filtered out, doc 2 remains
}

#[test]
fn multi_dimension_field_requires_every_dimension_in_range() {
    let (kdm, kdi, kdd, id) = build_two_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    // dim0 in [5, 15] would catch doc1 (10,10) and doc3 (10,100), but
    // dim1's range [5, 15] excludes doc3 (its dim1 value 100 is out).
    let mut min = long_bytes(5).to_vec();
    min.extend_from_slice(&long_bytes(5));
    let mut max = long_bytes(15).to_vec();
    max.extend_from_slice(&long_bytes(15));
    let docs = resolve_points_range_doc_ids(&reader, None, 1, &min, &max).unwrap();
    assert_eq!(docs, vec![1]);
}

/// A field whose *data* dimensions outnumber its *index* dimensions (a
/// real shape: `LatLonShape`/`*RangeField` pack extra non-indexed bytes
/// after the indexed ones). The query box covers only the index
/// dimensions, exactly as `PointRangeQuery` requires; the old
/// `decode_all_points` implementation looped over `field.num_dims` and
/// would have indexed past the end of a correctly-sized box.
#[test]
fn query_box_covers_index_dimensions_only_not_data_dimensions() {
    let segment_id = [9u8; ID_LENGTH];
    let pack = |a: i64, b: i64| -> Vec<u8> {
        let mut v = long_bytes(a).to_vec();
        v.extend_from_slice(&long_bytes(b));
        v
    };
    // dim 0 is indexed; dim 1 is a payload dimension the query never sees.
    let points: Vec<(i32, Vec<u8>)> =
        vec![(0, pack(10, 999)), (1, pack(20, -999)), (2, pack(30, 0))];
    let field = WritePointsField {
        field_number: 1,
        num_dims: 2,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = points::write(&[field], 512, &segment_id, "").unwrap();
    let reader = points::open(&kdm, &kdi, &kdd, &segment_id, "").unwrap();

    // One index dimension => an 8-byte box, not 16.
    let min = long_bytes(15);
    let max = long_bytes(35);
    let docs = resolve_points_range_doc_ids(&reader, None, 1, &min, &max).unwrap();
    assert_eq!(docs, vec![1, 2]);
}

// --- resolve_and_apply_points_range_delete: real Directory I/O ---

#[test]
fn resolves_and_applies_against_a_real_flushed_segment() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let info = sci("_0", -1, 0);

    let min = long_bytes(15);
    let max = long_bytes(45);
    let updated =
        resolve_and_apply_points_range_delete(&dir, &info, &reader, None, 5, 1, &min, &max)
            .unwrap();

    assert_eq!(updated.del_gen, 1);
    assert_eq!(updated.del_count, 3); // docs 1, 2, 3 (values 20, 30, 40)

    let bytes = std::fs::read(tmp.join("_0_1.liv")).unwrap();
    let parsed = lucene_codecs::live_docs::parse(&bytes, &info.segment_id, 1, 5, 3).unwrap();
    assert!(parsed.get(0));
    assert!(!parsed.get(1));
    assert!(!parsed.get(2));
    assert!(!parsed.get(3));
    assert!(parsed.get(4));
}

#[test]
fn a_range_matching_zero_docs_is_a_no_op_round_that_still_bumps_gen() {
    let (kdm, kdi, kdd, id) = build_single_dim_fixture();
    let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
    let tmp = tempdir();
    let dir = FsDirectory::open(&tmp);
    let info = sci("_0", -1, 0);

    let min = long_bytes(1000);
    let max = long_bytes(2000);
    let updated =
        resolve_and_apply_points_range_delete(&dir, &info, &reader, None, 5, 1, &min, &max)
            .unwrap();
    assert_eq!(updated.del_gen, 1);
    assert_eq!(updated.del_count, 0);
}
