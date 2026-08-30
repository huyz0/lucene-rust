//! Search-side BKD points queries: "which live doc IDs does a
//! `PointRangeQuery`/`PointInSetQuery`-shaped search actually match, in one
//! already-opened segment" -- the read-only, non-deleting sibling of
//! [`lucene_index::points_delete`]'s delete-by-point-range flow.
//!
//! ## Why the range query composes `lucene_index::points_delete`
//!
//! `lucene_index::points_delete::resolve_points_range_doc_ids` already is
//! exactly "every live doc ID whose packed BKD value falls in an inclusive
//! `[min_packed, max_packed]` range", implemented on top of
//! [`lucene_codecs::points::PointsReader::range_query`] -- i.e. the real
//! `PointValues.intersect`/`BKDReader.BKDPointTree` pruning walk, with
//! `CELL_INSIDE_QUERY`/`CELL_OUTSIDE_QUERY`/`CELL_CROSSES_QUERY` dispatch,
//! plus the sort+dedup fold Java's `DocIdSetBuilder` performs. (Both this
//! module and that one predate that traversal and used to decode every point
//! via `decode_all_points` and filter in memory; the switch to the pruning
//! walk measured **577x** faster on a 200k-point tree -- see
//! `crates/lucene-codecs/benches/hot_paths.rs`'s
//! `points/range_query_selective_200k` vs
//! `points/decode_all_then_filter_selective_200k`.)
//! This module doesn't reimplement any of that -- it reuses
//! `resolve_points_range_doc_ids` as-is (the dependency graph already has
//! `lucene-search -> lucene-index`, confirmed by
//! `crates/lucene-search/Cargo.toml`) and adapts its `Vec<i32>` result onto
//! this crate's [`Collector`] trait, the same "feed matches through a
//! collector" shape [`crate::doc_value_query::search_numeric_range`] and
//! [`crate::search_term_query`] use.
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
//! - [`search_points_in_set`]: `PointInSetQuery`'s equivalent -- every live
//!   doc whose packed value is *exactly* one of a caller-supplied set. The
//!   one-dimensional case is a real port of Java's `MergePointVisitor`
//!   (a merge-sort of the sorted query set against the BKD tree's cells,
//!   including its `CELL_INSIDE_QUERY` shortcut for a cell whose min and max
//!   both equal the query point); see that function for the multi-dimension
//!   case.
//!
//! **Deliberately out of scope** (tracked in `docs/parity.md`):
//! - **A scored variant.** Real Lucene's `PointRangeQuery`/`PointInSetQuery`
//!   are `ConstantScoreQuery`-shaped match-only queries with no relevance
//!   score of their own -- there is no `ScoredCollector`-based sibling to add
//!   here (the distinction [`crate::doc_value_query`]'s module doc draws
//!   between `Collector` and `ScoredCollector` doesn't apply: these never
//!   score in real Lucene either).
//! - **`PointValues.estimateDocCount`/`estimatePointCount`** -- the cost
//!   estimate `ScorerSupplier.cost()` uses to pick a query plan. This port
//!   has no `ScorerSupplier`/`IndexOrDocValuesQuery` planner to feed.
//! - **Multi-segment federation.** Single already-opened segment's
//!   `PointsReader` + one field, same scope every other query module in this
//!   crate takes (no `IndexSearcher`/`DirectoryReader` federation exists in
//!   this port yet).

use lucene_codecs::field_infos::FieldInfos;
use lucene_codecs::points::{IntersectVisitor, PointsReader, Relation};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::{Collector, Error, Result};

/// Bundles an already-opened [`PointsReader`] with the segment's
/// [`FieldInfos`] -- the pairing [`crate::resolve_clause_docs`]/
/// [`crate::clause_scores`] need to resolve a [`crate::query::Clause::PointsRange`]'s
/// field *name* (that's all a query string / [`crate::query::PointsRangeQuery`]
/// ever has) down to the field *number* [`PointsReader::field`]/
/// [`search_points_range`] key on, the same "look the number up from
/// `field_infos`, then reopen the reader" two-step
/// `lucene_ffi::points_query::ffi_search_points_range` already performs at
/// the C-ABI boundary -- this is that same pairing, reused one layer down so
/// a pure-Rust caller (no FFI involved) gets the same capability.
pub struct PointsInput<'d> {
    pub reader: PointsReader<'d>,
    pub field_infos: &'d FieldInfos,
}

impl<'d> PointsInput<'d> {
    /// `field`'s number in `field_infos`, or `None` if this segment's field
    /// infos never heard of it -- mirrors
    /// `lucene_ffi::points_query::field_number_for`'s `find(|f| f.name ==
    /// field)` lookup, but returns `Option` instead of an FFI status since a
    /// caller here isn't crossing a C ABI: an unknown field name is treated
    /// the same "missing field means no matches" way every other clause
    /// resolver in this crate already treats a missing field (not an error).
    pub fn field_number(&self, field: &str) -> Option<i32> {
        self.field_infos
            .fields
            .iter()
            .find(|f| f.name == field)
            .map(|f| f.number)
    }
}

/// Packs an `i64` into real Lucene's `LongPoint`/`NumericUtils.longToSortableBytes`
/// big-endian, sign-bit-flipped 8-byte encoding -- the same encoding this
/// module's own tests (`long_bytes`) and `lucene_index::points_delete`'s
/// tests hand-roll locally, now available as a real (non-test-only) helper
/// for [`crate::resolve_clause_docs`]/[`crate::clause_scores`] to pack a
/// [`crate::query::PointsRangeQuery`]'s `i64` `min`/`max` bounds into the
/// `min_packed`/`max_packed` bytes [`search_points_range`] needs. Flipping the
/// sign bit before a big-endian byte view makes unsigned byte-wise comparison
/// (what the BKD tree's leaf/split-value comparisons use) agree with signed
/// `i64` ordering -- exactly why real Lucene does the same flip.
pub fn pack_i64(v: i64) -> [u8; 8] {
    ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()
}

/// Every live doc ID in `reader`'s `field_number` whose packed BKD value
/// falls within the inclusive `[min_packed, max_packed]` range, fed through
/// `collector` in ascending doc-ID order -- the search-side (non-deleting)
/// analog of
/// [`lucene_index::points_delete::resolve_points_range_doc_ids`], which this
/// function delegates to directly -- and which is itself
/// [`lucene_codecs::points::PointsReader::range_query`], i.e. the real
/// `PointValues.intersect` pruning walk (see this module's doc comment for
/// why no BKD read/traversal logic is duplicated here).
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

/// Port of `PointInSetQuery.MergePointVisitor` (the `numDims == 1` case):
/// a merge-sort of the already-sorted, deduplicated query points against the
/// BKD tree's cells, so a cell entirely below the next unconsumed query point
/// advances the cursor, a cell entirely above it prunes the whole subtree, and
/// a cell whose min *and* max both equal the query point short-circuits to
/// `CELL_INSIDE_QUERY` (the "> 512 docs share one value" case Java calls out).
struct MergePointVisitor<'q> {
    /// Sorted, deduplicated query points, each `bytes_per_dim` long.
    points: &'q [Vec<u8>],
    /// Index of the next unconsumed query point -- Java's `nextQueryPoint`.
    cursor: usize,
    docs: Vec<i32>,
}

impl MergePointVisitor<'_> {
    fn matches(&mut self, packed_value: &[u8]) -> bool {
        while let Some(query_point) = self.points.get(self.cursor) {
            match query_point.as_slice().cmp(packed_value) {
                std::cmp::Ordering::Equal => return true,
                // Query point is before the index point: advance the cursor.
                std::cmp::Ordering::Less => self.cursor += 1,
                // Query point is after the index point: this value can't match.
                std::cmp::Ordering::Greater => break,
            }
        }
        false
    }
}

impl IntersectVisitor for MergePointVisitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        while let Some(query_point) = self.points.get(self.cursor) {
            let cmp_min = query_point.as_slice().cmp(min_packed);
            if cmp_min == std::cmp::Ordering::Less {
                // Query point is before the start of this cell.
                self.cursor += 1;
                continue;
            }
            let cmp_max = query_point.as_slice().cmp(max_packed);
            if cmp_max == std::cmp::Ordering::Greater {
                // Query point is after the end of this cell.
                return Relation::CellOutsideQuery;
            }
            return if cmp_min == std::cmp::Ordering::Equal && cmp_max == std::cmp::Ordering::Equal {
                Relation::CellInsideQuery
            } else {
                Relation::CellCrossesQuery
            };
        }
        // Every query point has been consumed.
        Relation::CellOutsideQuery
    }

    fn visit(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if self.matches(packed_value) {
            self.docs.push(doc_id);
        }
    }
}

/// `PointInSetQuery`'s equivalent: every live doc ID in `reader`'s
/// `field_number` whose packed BKD value is **exactly equal** to one of
/// `points`, fed through `collector` in ascending doc-ID order.
///
/// Each entry of `points` must be exactly `num_index_dims * bytes_per_dim`
/// bytes for the field (real `PointInSetQuery`'s constructor enforces the
/// same `numDims * bytesPerDim` length and throws otherwise); an entry of the
/// wrong length simply never compares equal to an indexed value, so it
/// contributes no matches rather than panicking.
///
/// - **One index dimension**: a real port of Java's `MergePointVisitor` --
///   one `PointValues.intersect` traversal, merge-sorting the sorted query
///   set against the tree's cells so whole subtrees between two query points
///   are pruned without reading their `.kdd` bytes.
/// - **More than one index dimension**: Java falls back to
///   `SinglePointVisitor`, i.e. one traversal per query point against the
///   degenerate box `[point, point]`. This function does exactly that, via
///   [`lucene_codecs::points::PointsReader::range_query`] per point, then
///   folds the per-point results together (sort + dedup, Java's
///   `DocIdSetBuilder`).
///
/// An unknown `field_number`, or an empty `points`, collects nothing and
/// returns `Ok(())` -- the same "no matches, not a caller bug" convention
/// [`search_points_range`] documents.
pub fn search_points_in_set<C: Collector>(
    reader: &PointsReader<'_>,
    live_docs: Option<&FixedBitSet>,
    field_number: i32,
    points: &[Vec<u8>],
    collector: &mut C,
) -> Result<()> {
    let Some(field) = reader.field(field_number) else {
        return Ok(());
    };
    if points.is_empty() {
        return Ok(());
    }
    // `PointInSetQuery` sorts and deduplicates its packed points up front
    // (`sortedPackedPoints`, a `PrefixCodedTerms` built from a sorted set);
    // both the merge visitor's cursor and the multi-dimension fallback rely
    // on that ordering.
    let mut sorted: Vec<Vec<u8>> = points.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut doc_ids = if field.num_index_dims == 1 {
        let mut visitor = MergePointVisitor {
            points: &sorted,
            cursor: 0,
            docs: Vec::new(),
        };
        reader
            .intersect(field_number, &mut visitor)
            .map_err(Error::Points)?;
        visitor.docs
    } else {
        let mut docs = Vec::new();
        for point in &sorted {
            docs.extend(
                reader
                    .range_query(field_number, point, point)
                    .map_err(Error::Points)?,
            );
        }
        docs
    };

    if let Some(bits) = live_docs {
        doc_ids.retain(|&doc_id| bits.get(doc_id as usize));
    }
    doc_ids.sort_unstable();
    doc_ids.dedup();
    for doc_id in doc_ids {
        collector.collect(doc_id);
    }
    Ok(())
}

/// Port of `PointValues.estimateDocCount(visitor)`'s arithmetic: turns an
/// estimated *point* count into an estimated *document* count.
///
/// This is the number `IndexOrDocValuesQuery`'s planner consumes as its index
/// side's `ScorerSupplier.cost()` -- pass it to
/// [`crate::doc_value_query::plan_index_or_doc_values`] as `Some(cost)` --
/// and the number `PointRangeQuery`'s `DocIdSetIterator.cost()` reports.
///
/// - `estimated_point_count` is `PointValues.estimatePointCount(visitor)`: the
///   BKD walk that adds a whole subtree's `size()` for a cell entirely inside
///   the query, adds nothing for one entirely outside, and assumes half a leaf
///   matched when it can descend no further. **That walk is not ported** -- it
///   needs each BKD node's subtree size mid-traversal, which
///   `lucene_codecs::points` does not expose (see this batch's report for the
///   handoff). Until it is, a caller supplies the count some other way: the
///   exact match count when it has already run the query, or `None` to
///   `plan_index_or_doc_values`, whose documented default is Java's own answer
///   whenever this query leads.
/// - `size` is `PointValues.size()`, the field's total point count
///   (`PointsField::point_count`).
/// - `doc_count` is `PointValues.getDocCount()` (`PointsField::doc_count`).
///
/// The three branches are Java's, in order:
///
/// 1. an estimate at or above the whole field matches every document;
/// 2. a **single-valued** field (`size == docCount`) -- or an estimate of zero
///    -- has one point per document, so the point estimate *is* the document
///    estimate;
/// 3. otherwise a multi-valued field, where Java approximates the urn problem
///    `D * (1 - ((N - n) / N)^(N/D))` and floors the result at 1 so a
///    non-empty match never costs zero.
pub fn estimate_doc_count(estimated_point_count: i64, size: i64, doc_count: i32) -> i64 {
    let size_f = size as f64;
    if estimated_point_count >= size {
        return doc_count as i64;
    }
    if size == doc_count as i64 || estimated_point_count == 0 {
        return estimated_point_count;
    }
    // Java's `(long)` cast on a `double` truncates toward zero, which is what
    // `as i64` does for a finite value; `docCount` is positive here (a zero
    // `docCount` with a non-zero `size` cannot occur -- and would have taken
    // branch 2 anyway, since `estimated_point_count < size` and
    // `size == 0 == docCount` is impossible).
    let doc_estimate = (doc_count as f64
        * (1.0
            - ((size_f - estimated_point_count as f64) / size_f).powf(size_f / doc_count as f64)))
        as i64;
    if doc_estimate == 0 {
        1
    } else {
        doc_estimate
    }
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
        // per field, during the tree walk) -- so scrambling the bytes strictly
        // between the header and the trailing
        // `lucene_store::codec_util::FOOTER_LENGTH`-byte footer keeps `open`
        // itself succeeding, but forces the leaf-block read to fail partway
        // through. This exercises the error path through
        // `search_points_range` itself (not just through `points::open`),
        // confirming the crate-level `Error::Points` wiring documented on
        // this module.
        //
        // The range below has to be a *narrow* one. Since batch b7 replaced
        // the decode-every-point scan with the real `PointValues.intersect`
        // pruning walk, an all-encompassing `[i64::MIN, i64::MAX]` range hits
        // `CELL_INSIDE_QUERY` at the root, which -- exactly like Java's
        // `visitDocIDs` shortcut -- reads only the leaf's doc-id block and
        // never decodes a packed value, so it can happily return garbage doc
        // ids from garbage bytes without erroring (detecting *that* is
        // `checkIntegrity`'s/the footer checksum's job, not the tree walk's,
        // in Java too). A narrow range makes the root cell
        // `CELL_CROSSES_QUERY`, which is the branch that decodes the block.
        let footer_start = kdd.len() - lucene_store::codec_util::FOOTER_LENGTH;
        let header_end = 60; // past `check_index_header`'s magic+name+version+id+suffix prefix
        for b in kdd[header_end..footer_start].iter_mut() {
            *b = 0xFF;
        }
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let min = long_bytes(15);
        let max = long_bytes(35);
        let mut collector = VecCollector::default();
        let err = search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap_err();
        assert!(matches!(err, Error::Points(_)));
    }

    // --- `search_points_in_set` (`PointInSetQuery`) ---

    #[test]
    fn in_set_matches_exactly_the_requested_values() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let points = vec![
            long_bytes(20).to_vec(),
            long_bytes(50).to_vec(),
            long_bytes(45).to_vec(), // no doc has this value
        ];
        let mut collector = VecCollector::default();
        search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![1, 4]); // values 20 and 50
    }

    #[test]
    fn in_set_is_exact_equality_not_a_range() {
        // A set spanning the whole value space's endpoints must NOT pull in
        // the values between them -- the single sharpest difference between
        // `PointInSetQuery` and `PointRangeQuery`.
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let points = vec![long_bytes(10).to_vec(), long_bytes(50).to_vec()];
        let mut collector = VecCollector::default();
        search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![0, 4]);
    }

    #[test]
    fn in_set_unsorted_and_duplicated_input_is_normalized() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        // Deliberately descending, with a duplicate: `search_points_in_set`
        // must sort + dedup before the merge walk (Java's `sortedPackedPoints`).
        let points = vec![
            long_bytes(40).to_vec(),
            long_bytes(20).to_vec(),
            long_bytes(40).to_vec(),
        ];
        let mut collector = VecCollector::default();
        search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![1, 3]);
    }

    #[test]
    fn in_set_empty_and_unknown_field_collect_nothing() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let mut collector = VecCollector::default();
        search_points_in_set(&reader, None, 1, &[], &mut collector).unwrap();
        assert!(collector.docs.is_empty());
        search_points_in_set(
            &reader,
            None,
            99,
            &[long_bytes(10).to_vec()],
            &mut collector,
        )
        .unwrap();
        assert!(collector.docs.is_empty());
    }

    #[test]
    fn in_set_respects_live_docs() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let mut live = FixedBitSet::new(5);
        for i in 0..5 {
            live.set(i);
        }
        live.clear(1); // doc 1 (value 20) deleted
        let points = vec![long_bytes(20).to_vec(), long_bytes(30).to_vec()];
        let mut collector = VecCollector::default();
        search_points_in_set(&reader, Some(&live), 1, &points, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![2]);
    }

    #[test]
    fn in_set_multi_dimension_requires_every_dimension_to_match() {
        let (kdm, kdi, kdd, id) = build_two_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let pack = |a: i64, b: i64| -> Vec<u8> {
            let mut v = long_bytes(a).to_vec();
            v.extend_from_slice(&long_bytes(b));
            v
        };
        // (10, 10) is doc 1; (10, 20) matches no doc even though doc 1 shares
        // dim 0 and doc 2 shares dim 1.
        let points = vec![pack(10, 10), pack(10, 20)];
        let mut collector = VecCollector::default();
        search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![1]);
    }

    #[test]
    fn in_set_wrong_length_point_matches_nothing() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let mut collector = VecCollector::default();
        search_points_in_set(&reader, None, 1, &[vec![0u8; 3]], &mut collector).unwrap();
        assert!(collector.docs.is_empty());
    }

    #[test]
    fn in_set_every_indexed_value_matches_every_doc() {
        let (kdm, kdi, kdd, id) = build_single_dim_fixture();
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let points: Vec<Vec<u8>> = (1..=5).map(|i| long_bytes(i * 10).to_vec()).collect();
        let mut collector = VecCollector::default();
        search_points_in_set(&reader, None, 1, &points, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![0, 1, 2, 3, 4]);
    }

    // -----------------------------------------------------------------
    // `PointValues.estimateDocCount` (b14 §1.4 / c12 §5.4)
    // -----------------------------------------------------------------

    /// Java's three branches, each with a case only that branch produces.
    #[test]
    fn estimate_doc_count_follows_javas_three_branches() {
        // 1. An estimate at or above the field's whole point count matches
        //    every document, however the points are distributed.
        assert_eq!(estimate_doc_count(100, 100, 40), 40);
        assert_eq!(estimate_doc_count(1_000, 100, 40), 40);

        // 2. Single-valued (`size == docCount`): the point estimate *is* the
        //    document estimate.
        assert_eq!(estimate_doc_count(37, 100, 100), 37);
        // ...and a zero estimate is zero whatever the shape, so a
        // provably-empty range never costs anything.
        assert_eq!(estimate_doc_count(0, 100, 40), 0);

        // 3. Multi-valued: `D * (1 - ((N - n)/N)^(N/D))`, floored at 1.
        //    N = 100 points over D = 50 docs (2 points per doc), n = 50:
        //    50 * (1 - 0.5^2) = 37.5 -> truncates to 37.
        assert_eq!(estimate_doc_count(50, 100, 50), 37);
        //    N = 1000 over D = 100 (10 points/doc), n = 100:
        //    100 * (1 - 0.9^10) = 65.13... -> 65.
        assert_eq!(estimate_doc_count(100, 1000, 100), 65);
        //    A tiny estimate rounds to zero documents, which Java floors to 1
        //    rather than letting a non-empty match report no cost at all.
        assert_eq!(estimate_doc_count(1, 1_000_000, 1), 1);
    }

    /// The `IndexOrDocValuesQuery` question c12 §5.4 left open: with a real
    /// estimate in hand, does the planner make a *different* choice from the
    /// `None` default -- and when?
    ///
    /// The estimate is taken from the real `fixtures/data/points_index/`
    /// field, and for the exact ranges below `estimatePointCount` would be
    /// exact: every BKD cell a fully-covering range touches is either wholly
    /// inside or wholly outside it, which is the case Java's walk answers
    /// without approximating. The field is single-valued
    /// (`point_count == doc_count`), so the estimate passes straight through
    /// branch 2 of [`estimate_doc_count`].
    #[test]
    fn a_real_point_estimate_changes_the_index_or_doc_values_plan_only_when_another_clause_leads() {
        use crate::doc_value_query::{plan_index_or_doc_values, IndexOrDocValuesPlan};

        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/points_index/"
        );
        let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run scripts/gen-fixtures.sh first (GenPoints)");
        let kv: std::collections::HashMap<&str, &str> =
            text.lines().filter_map(|l| l.split_once('=')).collect();
        let mut id = [0u8; ID_LENGTH];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&kv["id_hex"][i * 2..i * 2 + 2], 16).unwrap();
        }
        let read = |key: &str| std::fs::read(format!("{dir}{}", kv[key])).unwrap();
        let (kdm, kdi, kdd) = (
            read("kdm_file_name"),
            read("kdi_file_name"),
            read("kdd_file_name"),
        );
        let reader = points::open(&kdm, &kdi, &kdd, &id, "").unwrap();
        let field_number: i32 = kv["field_number"].parse().unwrap();
        let field = reader.field(field_number).unwrap();
        let size = field.point_count;
        let doc_count = field.doc_count;
        assert_eq!(size, doc_count as i64, "the fixture field is single-valued");

        // The whole range: every point matches, so the estimate is the whole
        // field and `estimate_doc_count` short-circuits to `docCount`.
        let mut all = VecCollector::default();
        search_points_range(
            &reader,
            None,
            field_number,
            &long_bytes(i64::MIN),
            &long_bytes(i64::MAX),
            &mut all,
        )
        .unwrap();
        assert_eq!(all.docs.len(), doc_count as usize);
        let full_cost = estimate_doc_count(all.docs.len() as i64, size, doc_count);
        assert_eq!(full_cost, doc_count as i64);

        // A narrow range: the exact match count, which is what Java's walk
        // estimates.
        let mut narrow = VecCollector::default();
        search_points_range(
            &reader,
            None,
            field_number,
            &long_bytes(0),
            &long_bytes(100_000),
            &mut narrow,
        )
        .unwrap();
        let narrow_cost = estimate_doc_count(narrow.docs.len() as i64, size, doc_count);
        assert!(narrow_cost > 0 && narrow_cost < full_cost);

        // **The answer.** Java's rule is `cost >>> 3 <= leadCost`, so:
        //
        // - when this query leads (`lead_cost == cost`), the estimate cannot
        //   change the plan -- `cost/8 <= cost` for every non-negative cost --
        //   and the planner picks `Index`, which is exactly what `None`
        //   already picked. The estimate is *not* wasted work here; it is what
        //   proves the default was right rather than lucky.
        for cost in [full_cost, narrow_cost] {
            assert_eq!(
                plan_index_or_doc_values(Some(cost), cost),
                IndexOrDocValuesPlan::Index
            );
            assert_eq!(
                plan_index_or_doc_values(None, cost),
                IndexOrDocValuesPlan::Index
            );
        }

        // - the estimate changes the plan exactly when some *other* clause
        //   leads with fewer than `cost/8` documents. With the real numbers
        //   below that threshold is a concrete doc count, and on either side
        //   of it the two planners disagree.
        let threshold = full_cost / 8;
        assert!(threshold > 0);
        assert_eq!(
            plan_index_or_doc_values(Some(full_cost), threshold),
            IndexOrDocValuesPlan::Index,
            "at exactly cost/8 Java still runs the index"
        );
        assert_eq!(
            plan_index_or_doc_values(Some(full_cost), threshold - 1),
            IndexOrDocValuesPlan::DocValues,
            "below cost/8 Java verifies the lead clause's docs against doc values"
        );
        assert_eq!(
            plan_index_or_doc_values(None, threshold - 1),
            IndexOrDocValuesPlan::Index,
            "...and this is the one case the no-estimate default gets wrong"
        );

        // The narrower the query, the smaller its threshold, so a lead cost
        // that flips the full-range plan need not flip the narrow one -- the
        // estimate is doing real work, not scaling out of the comparison.
        assert!(narrow_cost / 8 < threshold);
        assert_eq!(
            plan_index_or_doc_values(Some(narrow_cost), threshold - 1),
            IndexOrDocValuesPlan::Index
        );
    }
}
