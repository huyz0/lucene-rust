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
//! - **Multi-segment federation**, except for the top-`n` constant-score
//!   range ([`points_range_top_n_multi_segment`], over
//!   [`points_range_lowest_docs`] per segment). Everything else here takes a
//!   single already-opened segment's `PointsReader` + one field.

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

/// The `n` lowest live doc ids whose `field_number` value falls in
/// `[min_packed, max_packed]`, ascending, and how many documents match in
/// all -- what a top-`n` constant-score range query returns (every hit scores
/// the same, so the lowest doc ids win the ties).
///
/// [`search_points_range`] sorts every match before handing any out, as
/// Lucene's `DocIdSetBuilder` does; that sort is most of a selective range
/// query's cost. When the field holds one point per document (its
/// `pointCount == docCount`, so no doc id can repeat), the lowest `n` are
/// found with a linear-time selection instead and only those `n` are sorted.
/// Otherwise the matches are sorted and deduplicated as before.
pub fn points_range_lowest_docs(
    reader: &PointsReader<'_>,
    live_docs: Option<&FixedBitSet>,
    field_number: i32,
    min_packed: &[u8],
    max_packed: &[u8],
    n: usize,
) -> Result<(Vec<i32>, usize)> {
    let Some(field) = reader.field(field_number) else {
        return Ok((Vec::new(), 0));
    };
    let single_valued = field.point_count == i64::from(field.doc_count);
    let mut docs = reader.range_query(field_number, min_packed, max_packed)?;
    if let Some(bits) = live_docs {
        docs.retain(|&doc_id| bits.get_doc(doc_id));
    }
    if single_valued && docs.len() > n {
        let total = docs.len();
        if n == 0 {
            return Ok((Vec::new(), total));
        }
        // ARITH: `n > 0` was just checked, and `n < docs.len()`.
        #[allow(clippy::arithmetic_side_effects)]
        docs.select_nth_unstable(n - 1);
        docs.truncate(n);
        docs.sort_unstable();
        return Ok((docs, total));
    }
    lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut docs);
    let total = docs.len();
    docs.truncate(n);
    Ok((docs, total))
}

/// One segment's points for [`points_range_top_n_multi_segment`].
#[derive(Clone, Copy)]
pub struct PointsSegment<'a> {
    pub reader: &'a PointsReader<'a>,
    pub field_number: i32,
    pub live_docs: Option<&'a FixedBitSet>,
    pub doc_base: i32,
}

/// `IndexSearcher.search(PointRangeQuery, n)` over several segments: the `n`
/// lowest matching global doc ids (every hit scores the same constant, so doc
/// id breaks the ties), ascending, with the number of hits counted and whether
/// that count is exact.
///
/// Segments are visited in ascending `doc_base` order. Once `n` hits are held
/// and more than `total_hits_threshold` have been counted -- the state in
/// which `TopScoreDocCollector` publishes its threshold and reports its count
/// as a lower bound -- no later segment can change the hits (its documents
/// tie on score and lose on doc id), so the rest are not searched. The count
/// is then a lower bound (`exact == false`), as Lucene's is.
pub fn points_range_top_n_multi_segment(
    segments: &[PointsSegment<'_>],
    min_packed: &[u8],
    max_packed: &[u8],
    n: usize,
    total_hits_threshold: usize,
) -> Result<(Vec<i32>, usize, bool)> {
    let mut order: Vec<&PointsSegment<'_>> = segments.iter().collect();
    order.sort_by_key(|s| s.doc_base);
    let mut hits: Vec<i32> = Vec::new();
    let mut total = 0usize;
    for seg in &order {
        if hits.len() >= n && total > total_hits_threshold {
            return Ok((hits, total, false));
        }
        let (docs, matched) = points_range_lowest_docs(
            seg.reader,
            seg.live_docs,
            seg.field_number,
            min_packed,
            max_packed,
            n,
        )?;
        total = total.saturating_add(matched);
        let room = n.saturating_sub(hits.len());
        // ARITH: a global doc id, below the reader's `max_doc`.
        #[allow(clippy::arithmetic_side_effects)]
        hits.extend(docs.into_iter().take(room).map(|d| d + seg.doc_base));
    }
    // Past the threshold the count is a lower bound even when every segment
    // was searched, as `TopScoreDocCollector` reports it.
    Ok((hits, total, total <= total_hits_threshold))
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
        doc_ids.retain(|&doc_id| bits.get_doc(doc_id));
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
///   matched when it can descend no further. That walk **is** ported, as
///   `lucene_codecs::points::PointsReader::estimate_point_count` (and
///   `estimate_range_point_count` for `PointRangeQuery`'s own visitor), so a
///   caller with a reader can hand this function Java's own input rather than
///   an approximation of it. A caller that has already run the query may pass
///   the exact match count instead, and `plan_index_or_doc_values` still
///   accepts `None`, whose documented default is Java's own answer whenever
///   this query leads.
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
mod tests;
