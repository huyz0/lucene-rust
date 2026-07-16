//! Search-side BKD points range query: "which live doc IDs does a
//! `PointRangeQuery`-shaped search actually match, in one already-opened
//! segment" -- the read-only, non-deleting sibling of
//! [`lucene_index::points_delete`]'s delete-by-point-range flow.
//!
//! ## Why this composes `lucene_index::points_delete`, not raw
//! `lucene_codecs::points`
//!
//! `lucene_index::points_delete::resolve_points_range_doc_ids` already is
//! exactly "every live doc ID whose packed BKD value falls in an inclusive
//! `[min_packed, max_packed]` range" -- that function's own doc comment
//! documents that it deliberately decodes every point via
//! [`lucene_codecs::points::PointsReader::decode_all_points`] and filters in
//! memory (`lucene_codecs::points` itself has no
//! intersection/tree-pruning logic to call into; see that module's doc
//! comment for why). This module doesn't reimplement any of that -- it
//! reuses `resolve_points_range_doc_ids` as-is (the dependency graph already
//! has `lucene-search -> lucene-index`, confirmed by
//! `crates/lucene-search/Cargo.toml`) and adapts its `Vec<i32>` result onto
//! this crate's [`Collector`] trait, the same "feed matches through a
//! collector" shape [`crate::doc_value_query::search_numeric_range`] and
//! [`crate::search_term_query`] use.
//!
//! The only new code here is that adaptation plus this crate's own
//! [`Error::Points`] wiring -- no BKD traversal, no packed-value comparison
//! logic, is duplicated or reimplemented.
//!
//! ## Scope
//!
//! **In scope:**
//! - [`search_points_range`]: single-dimension or multi-dimension range
//!   query (whatever `min_packed`/`max_packed`/`num_dims`/`bytes_per_dim`
//!   the field's [`lucene_codecs::points::PointsField`] declares --
//!   `resolve_points_range_doc_ids` itself is already dimension-agnostic,
//!   checking every dimension's slice independently per
//!   `PointRangeQuery.matches` semantics), filtered by an optional
//!   `live_docs` bitset, fed through any [`Collector`] (so it composes with
//!   [`crate::collector::VecCollector`] for "just the doc IDs" or any other
//!   `Collector` impl a caller wires up, e.g. as one clause of a larger
//!   boolean/conjunction search).
//!
//! **Deliberately out of scope** (tracked in `docs/parity.md`):
//! - **A scored variant.** Real Lucene's `PointRangeQuery` is a
//!   `ConstantScoreQuery`-shaped match-only query with no relevance score of
//!   its own -- there is no `ScoredCollector`-based sibling to add here (the
//!   distinction [`crate::doc_value_query`]'s module doc draws between
//!   `Collector` and `ScoredCollector` doesn't apply: `PointRangeQuery` never
//!   scores in real Lucene either).
//! - **Sublinear tree-pruning traversal.** Same honest gap
//!   `resolve_points_range_doc_ids` already documents: this decodes every
//!   point in the field via `decode_all_points` and filters in memory,
//!   correct but not a port of `BKDReader.intersect`'s pruning walk.
//! - **Multi-segment federation.** Single already-opened segment's
//!   `PointsReader` + one field, same scope every other query module in this
//!   crate takes (no `IndexSearcher`/`DirectoryReader` federation exists in
//!   this port yet).

use lucene_codecs::points::PointsReader;
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::{Collector, Error, Result};

/// Every live doc ID in `reader`'s `field_number` whose packed BKD value
/// falls within the inclusive `[min_packed, max_packed]` range, fed through
/// `collector` in ascending doc-ID order -- the search-side (non-deleting)
/// analog of
/// [`lucene_index::points_delete::resolve_points_range_doc_ids`], which this
/// function delegates to directly (see this module's doc comment for why no
/// BKD read/traversal logic is duplicated here).
///
/// `min_packed`/`max_packed` must each be exactly `num_dims * bytes_per_dim`
/// bytes for the field, same contract as
/// `resolve_points_range_doc_ids`/`PointsField::min_packed_value` (a caller
/// passing the wrong length gets a panic from the slice index, same as that
/// function).
///
/// An unknown `field_number` collects nothing and returns `Ok(())` -- matches
/// `resolve_points_range_doc_ids`'s "no matches, not a caller bug"
/// convention (and every other `search_*` function in this crate).
///
/// `live_docs` is the segment's current `.liv` bitset (`None` means every doc
/// is live), the same convention every other `search_*`/`resolve_*` function
/// in this workspace uses.
pub fn search_points_range<C: Collector>(
    reader: &PointsReader<'_>,
    live_docs: Option<&FixedBitSet>,
    field_number: i32,
    min_packed: &[u8],
    max_packed: &[u8],
    collector: &mut C,
) -> Result<()> {
    let doc_ids = lucene_index::points_delete::resolve_points_range_doc_ids(
        reader,
        live_docs,
        field_number,
        min_packed,
        max_packed,
    )
    .map_err(|err| match err {
        lucene_index::points_delete::Error::Points(e) => Error::Points(e),
        // `resolve_points_range_doc_ids` (unlike its `resolve_and_apply_*`
        // sibling) never calls `deletes::apply_deletes`, so this arm is
        // unreachable in practice -- kept exhaustive rather than adding an
        // `Error::Deletes` variant this module never otherwise produces.
        lucene_index::points_delete::Error::Deletes(e) => {
            unreachable!("resolve_points_range_doc_ids never applies deletes: {e}")
        }
    })?;
    for doc_id in doc_ids {
        collector.collect(doc_id);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::VecCollector;
    use lucene_codecs::points::{self, WritePointsField};
    use lucene_store::codec_util::ID_LENGTH;

    fn long_bytes(v: i64) -> [u8; 8] {
        ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()
    }

    /// Same single-dimension `LongPoint`-shaped fixture
    /// `lucene_index::points_delete`'s tests use: doc 0 -> 10, doc 1 -> 20,
    /// doc 2 -> 30, doc 3 -> 40, doc 4 -> 50.
    fn build_single_dim_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, [u8; ID_LENGTH]) {
        let segment_id = [9u8; ID_LENGTH];
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

    /// 2D `LatLonPoint`-shaped fixture: doc 0 -> (0, 0), doc 1 -> (10, 10),
    /// doc 2 -> (20, 20), doc 3 -> (10, 100) -- dimension 0 alone would match
    /// doc 3 but dimension 1 must independently be in range too, exercising
    /// this port's already-built multi-dimension BKD points support end to
    /// end through the search-side entry point (not just the delete-side
    /// one).
    fn build_two_dim_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>, [u8; ID_LENGTH]) {
        let segment_id = [11u8; ID_LENGTH];
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

    #[test]
    fn matches_only_docs_within_bounds() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let min = long_bytes(15);
        let max = long_bytes(35);
        let mut collector = VecCollector::default();
        search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![1, 2]); // values 20, 30
    }

    #[test]
    fn boundary_values_are_inclusive_on_both_ends() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let min = long_bytes(10);
        let max = long_bytes(30);
        let mut collector = VecCollector::default();
        search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![0, 1, 2]);
    }

    #[test]
    fn empty_range_matches_no_docs() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let min = long_bytes(1000);
        let max = long_bytes(2000);
        let mut collector = VecCollector::default();
        search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
        assert!(collector.docs.is_empty());
    }

    #[test]
    fn full_range_matches_every_doc() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let min = long_bytes(i64::MIN);
        let max = long_bytes(i64::MAX);
        let mut collector = VecCollector::default();
        search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn unknown_field_number_matches_nothing_not_an_error() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let min = long_bytes(i64::MIN);
        let max = long_bytes(i64::MAX);
        let mut collector = VecCollector::default();
        search_points_range(&reader, None, 99, &min, &max, &mut collector).unwrap();
        assert!(collector.docs.is_empty());
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
        let mut collector = VecCollector::default();
        search_points_range(&reader, Some(&live), 1, &min, &max, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![2]); // only doc 2 (value 30) is live
    }

    #[test]
    fn two_dimension_range_checks_every_dimension_independently() {
        let (kdm, kdi, kdd, id) = build_two_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let pack = |a: i64, b: i64| -> Vec<u8> {
            let mut v = long_bytes(a).to_vec();
            v.extend_from_slice(&long_bytes(b));
            v
        };
        let min = pack(0, 0);
        let max = pack(20, 20);
        let mut collector = VecCollector::default();
        search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
        // doc 3's dim-0 value (10) is in range but dim-1 (100) is not, so it
        // must be excluded even though a single-dimension check would match.
        assert_eq!(collector.docs, vec![0, 1, 2]);
    }

    #[test]
    fn corrupt_kdd_leaf_data_surfaces_as_points_error() {
        let (kdm, kdi, mut kdd, id) = build_single_dim_fixture();
        // `points::open` only validates the `.kdd` codec header + footer, not
        // every leaf block's content in between (that's only decoded lazily,
        // per field, by `decode_all_points`) -- so scrambling the bytes
        // strictly between the header and the trailing
        // `lucene_store::codec_util::FOOTER_LENGTH`-byte footer keeps `open`
        // itself succeeding, but forces `decode_all_points`'s leaf read to
        // fail partway through. This exercises the error path through
        // `search_points_range` itself (not just through `points::open`),
        // confirming the crate-level `Error::Points` wiring documented on
        // this module.
        let footer_start = kdd.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let header_end = 60; // past `check_index_header`'s magic+name+version+id+suffix prefix
        for b in kdd[header_end..footer_start].iter_mut() {
            *b = 0xFF;
        }
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let min = long_bytes(i64::MIN);
        let max = long_bytes(i64::MAX);
        let mut collector = VecCollector::default();
        let err = search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap_err();
        assert!(matches!(err, Error::Points(_)));
    }
}
