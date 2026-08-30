//! Single-segment delete-by-point-range resolution: "which live doc IDs does
//! `deleteDocuments(new PointRangeQuery(...))`-shaped delete actually name, in
//! one already-opened segment" -- the BKD-point-range analog of
//! [`term_delete`](crate::term_delete)'s delete-by-term flow.
//! [`deletes.rs`](crate::deletes) is the apply half; this module is the
//! resolve half, scoped to one segment.
//!
//! # Why this lives in `lucene-index`, not `lucene-search`
//!
//! Same reasoning as `term_delete`'s module doc: the dependency graph is
//! strictly downward (`util ← store ← codecs ← index ← search ← core ← ffi`),
//! and `crates/lucene-search/Cargo.toml` depends on `lucene-index` (confirmed
//! by reading both `Cargo.toml`s), so `lucene-index` depending back on
//! `lucene-search` would invert that edge into a cycle.
//!
//! Unlike delete-by-term, there is no `PointRangeQuery` equivalent in
//! `lucene-search` to reuse (its only range-shaped queries are
//! `search_numeric_range`/`search_sorted_ord_range`/`search_multi_valued_range`
//! in `doc_value_query.rs`, which all walk **doc-values** data -- the "slow"
//! `SortedNumericDocValuesField.newSlowRangeQuery` fallback real Lucene uses
//! when no BKD index exists for a field, not the BKD tree at all).
//!
//! What this module *does* reuse is
//! [`lucene_codecs::points::PointsReader::range_query`], the port of
//! `PointValues.intersect` + `PointRangeQuery`'s `IntersectVisitor`: a real
//! BKD tree traversal that compares the query box against each cell's bounds
//! and prunes whole subtrees (`CELL_OUTSIDE_QUERY`), takes the doc-ids-only
//! fast path for fully-contained cells (`CELL_INSIDE_QUERY`, no packed value
//! decoded at all), and only decodes and per-point-checks the leaves that
//! actually straddle the boundary. This module previously called
//! `decode_all_points` and filtered in memory, which decoded every leaf of
//! every tree regardless of selectivity; see `docs/sweep/m2/b11-index-meta.md`
//! for the measured difference.
//!
//! # Scope
//!
//! Same single-segment scope as `term_delete`: resolution is scoped to one
//! already-opened segment's `PointsReader` + one field; there is no
//! multi-segment `IndexWriter`/`BufferedUpdates` orchestration in this port
//! (see `docs/parity.md`).

use lucene_codecs::points::PointsReader;
use lucene_util::fixed_bit_set::FixedBitSet;

use lucene_store::directory::Directory;

use crate::deletes;
use crate::segment_infos::SegmentCommitInfo;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Points(#[from] lucene_codecs::points::Error),
    #[error(transparent)]
    Deletes(#[from] deletes::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Resolves a `field_number`'s BKD points to every **live** doc ID whose
/// packed value falls within the inclusive `[min_packed, max_packed]` range,
/// ascending, de-duplicated (a doc with several matching values in a
/// multi-valued point field must only be deleted once).
///
/// `min_packed`/`max_packed` must each be exactly
/// `num_index_dims * bytes_per_dim` bytes for the field -- the **index**
/// dimensions, not the data dimensions, matching `PointRangeQuery`'s own
/// contract (`PointRangeQuery.checkValidPointValues` rejects a values instance
/// whose `getNumIndexDimensions()` differs from the query's `numDims`, and
/// only the index dimensions have cell bounds in the BKD packed index at
/// all). For every field this port writes today the two counts are equal;
/// for a field with trailing non-indexed data dimensions they are not, and
/// using the data-dimension count -- as this module did before -- would
/// compare bytes the query box does not describe.
///
/// An unknown `field_number` yields an empty `Vec`, not an error -- matches
/// `term_delete::resolve_term_doc_ids`'s "no matches, not a caller bug"
/// convention for a field/term absent from the segment.
///
/// `live_docs` is the segment's current `.liv` bitset (`None` means every doc
/// is live), same convention `term_delete`/`deletes::apply_deletes` use.
pub fn resolve_points_range_doc_ids(
    reader: &PointsReader<'_>,
    live_docs: Option<&FixedBitSet>,
    field_number: i32,
    min_packed: &[u8],
    max_packed: &[u8],
) -> Result<Vec<i32>> {
    if reader.field(field_number).is_none() {
        return Ok(Vec::new());
    }
    // `range_query` returns doc ids in leaf order, possibly repeated when a
    // doc has several matching points (exactly what Java's visitor sees
    // before it folds them into a `DocIdSetBuilder`); the sort+dedup below is
    // that fold.
    let mut doc_ids = reader.range_query(field_number, min_packed, max_packed)?;
    if let Some(bits) = live_docs {
        doc_ids.retain(|&doc_id| bits.get_doc(doc_id));
    }
    doc_ids.sort_unstable();
    doc_ids.dedup();
    Ok(doc_ids)
}

/// The full single-segment "resolve then apply" delete-by-point-range flow:
/// resolves `field_number`'s points within `[min_packed, max_packed]` to
/// their matching live doc IDs via [`resolve_points_range_doc_ids`], then
/// applies them via [`deletes::apply_deletes`] -- the BKD-range analog of
/// `term_delete::resolve_and_apply_term_delete`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_and_apply_points_range_delete(
    dir: &dyn Directory,
    sci: &SegmentCommitInfo,
    reader: &PointsReader<'_>,
    current_live_docs: Option<&FixedBitSet>,
    max_doc: usize,
    field_number: i32,
    min_packed: &[u8],
    max_packed: &[u8],
) -> Result<SegmentCommitInfo> {
    let doc_ids = resolve_points_range_doc_ids(
        reader,
        current_live_docs,
        field_number,
        min_packed,
        max_packed,
    )?;
    Ok(deletes::apply_deletes(
        dir,
        sci,
        current_live_docs,
        max_doc,
        doc_ids,
    )?)
}

#[cfg(test)]
mod tests {
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
}
