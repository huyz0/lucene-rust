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
//! actually straddle the boundary. Decoding every point and filtering in
//! memory would decode every leaf of every tree regardless of selectivity.
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
    // The BKD walk visits leaves in value order, not doc order: sort and
    // dedup the way `DocIdSetBuilder.build` does, with a radix sort.
    lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut doc_ids);
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
mod tests;
