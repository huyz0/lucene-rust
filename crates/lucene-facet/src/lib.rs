#![forbid(unsafe_code)]
//! Faceting over a SORTED_SET doc-values field — a port of real Lucene's
//! `SortedSetDocValuesFacetCounts` and the `lucene-facet` layer above it:
//! for every matching doc, increment a per-ordinal counter for each of that
//! doc's SortedSet ordinals, then resolve ordinals back to their string
//! labels via the field's terms dictionary (`lookupOrd`-equivalent, see
//! [`lucene_codecs::terms_dict::decode_all_terms`]).
//!
//! ## Scope decisions
//!
//! **Multi-segment faceting works, through an `OrdinalMap`.** A SORTED_SET
//! field's ordinals are per segment: each segment's terms dictionary numbers
//! its own terms 0..n, so ordinal 3 in segment 0 and ordinal 3 in segment 1
//! are in general different terms, and summing the per-segment count arrays
//! elementwise conflates them. [`lucene_search::ordinal_map::OrdinalMap`] solves it: build
//! one over every segment's term list, count each segment with
//! [`facet_counts`], and remap-and-sum with [`merge_segment_counts`], which is
//! `SortedSetDocValuesFacetCounts.countOneSegment`'s own
//! `counts[(int) ordMap.get(ord)] += count`. The result is indexed by global
//! ordinals, which is what [`FacetsState`] and [`SortedSetFacetCounts`] expect.
//! [`facet_counts`] itself is still the per-segment primitive it always was.
//!
//! **The dim layer exists.** [`FacetsConfig`] carries the per-dimension
//! configuration Lucene deliberately does not store in the index
//! (`hierarchical`, `multiValued`, `requireDimCount`), [`FacetsState`] is
//! `DefaultSortedSetDocValuesReaderState` (both the flat [`OrdRange`] and the
//! hierarchical [`DimTree`]), and [`SortedSetFacetCounts`] is
//! `SortedSetDocValuesFacetCounts`: `getTopChildren`, `getAllChildren`,
//! `getSpecificValue`, `getAllDims`, `getTopDims`, with Java's `-1` for a
//! multi-valued dim that does not require a dim count. The dim-less
//! [`top_children`]/[`all_children`] below remain as the raw one-field
//! primitives they were.
//!
//! **`FacetResult` semantics are now Java's.** [`top_children`]/
//! [`all_children`] (SORTED_SET) and [`top_range_children`] (ranges) return a
//! `FacetResult`-shaped struct with `label_values`/`value`/`child_count`, drop
//! zero-count children the way `AbstractSortedSetDocValueFacetCounts`/
//! `RangeFacetCounts` do, and report "a dim with no values in the matched set"
//! as `None` (Java's `null` `FacetResult`, which `getAllDims` filters out)
//! rather than a list of zeroes. [`top_n_facets`] is kept as the raw
//! ordering primitive it always was, explicitly *not* `getTopChildren`.
//!
//! **Double ranges are supported.** [`NumericRange::new_double`] reproduces
//! `DoubleRange`'s constructor plus `toLongRange`'s
//! `NumericUtils.doubleToSortableLong` transform (including the exclusive
//! bound's `Math.nextUp`/`nextDown` nudge happening in *double* space, before
//! the transform), and [`double_range_facet_counts`] applies
//! `mapDocValue`/[`sortable_double_bits`] to each stored doc value so the two
//! are compared in the same ordered space. [`NumericRange::new_long`] is
//! `LongRange`'s constructor, including its `failNoMatch` rejection of an
//! empty range.
//!
//! **Query-scoped counting is primary; "count everything" is the caller's
//! trivial special case.** Real Lucene's `FacetsCollector` always counts
//! over a matched-query doc set (there's no separate "count the whole
//! index" API distinct from running `MatchAllDocsQuery`). [`facet_counts`]
//! takes an explicit matching-doc-ID slice; a caller wanting "count every
//! doc in the segment" passes `0..max_doc` (every live doc ID) as that
//! slice — no separate code path is needed or added.
//!
//! **The taxonomy index exists too** ([`taxonomy`]): a
//! [`taxonomy::DirectoryTaxonomyWriter`] hands out category ordinals,
//! [`FacetsConfig::build`] indexes a document's [`FacetField`]s as those
//! ordinals (and its [`SortedSetDocValuesFacetField`]s as above), and a
//! [`taxonomy::DirectoryTaxonomyReader`] maps ordinals back to paths.
//!
//! **The counting layer is Java's too**. [`Facets`] is the
//! `Facets` contract (every method returns Java's errors as
//! `Err(IllegalArgument)`); a result's numbers are [`FacetValue`]s, Java's
//! `Number` (`Integer` for counts and int sums, `Float` for float sums).
//! [`FacetsCollector`]/[`FacetsCollectorManager`] gather the matching
//! documents per segment, [`taxonomy::FastTaxonomyFacetCounts`] and the
//! association facets count or aggregate them, and [`MultiFacets`] routes
//! dims to the facets that hold them.
//!
//! **Drill-down is Java's too**. [`DrillDownQuery`] narrows a
//! base query to chosen categories, [`FacetQuery`]/[`MultiFacetQuery`] are
//! a category's term as a query, and [`DrillSideways`] counts, for each dim
//! drilled into, the documents matching every other dim.
//!
//! **Value, range and facet-set counting are Java's too**.
//! [`LongValueFacetCounts`] and [`StringValueFacetCounts`] count each
//! distinct value of a field; [`range`] counts `long`/`double` values into
//! ranges ([`range::LongRangeFacetCounts`], [`range::DoubleRangeFacetCounts`],
//! [`range::dynamic_range_util`]), [`rangeonrange`] counts range fields into
//! ranges, and [`facetset`] counts tuples by matcher. The multi-valued
//! sources they read are [`multi_long_values_source`] and
//! [`multi_double_values_source`]. The older range primitives here
//! ([`NumericRange`], [`range_facet_counts`]) remain for the FFI layer.
//!
//! **Sorted-set counting over a reader, and sampling, are Java's too**. [`DefaultSortedSetDocValuesReaderState`] builds the
//! [`FacetsState`] of an index reader's SORTED_SET facet field (through an
//! [`lucene_search::ordinal_map::OrdinalMap`] when it has several segments);
//! [`SortedSetDocValuesFacetCounts`] and
//! [`ConcurrentSortedSetDocValuesFacetCounts`] count a collector's hits, or
//! every live document, into it; [`DrillSideways`] counts with either a
//! taxonomy or such a state; [`RandomSamplingFacetsCollector`] samples the
//! hits it collected and scales the counts back up. The taxonomy utilities
//! (merging, ordinal remapping, the searcher-and-taxonomy manager, stats and
//! the enriching writer) are under [`taxonomy`].

mod drill_down_query;
mod drill_sideways;
#[cfg(test)]
mod drill_sideways_query_tests;
mod facet_counts_with_filter_query;
mod facet_field;
mod facet_query;
mod facet_result;
mod facet_utils;
mod facets_collector;
mod facets_trait;
pub mod facetset;
mod long_value_facet_counts;
pub mod multi_double_values_source;
mod multi_facets;
pub mod multi_long_values_source;
mod random_sampling;
pub mod range;
pub mod rangeonrange;
mod sortedset;
mod string_value_facet_counts;
pub mod taxonomy;
#[cfg(test)]
pub(crate) mod test_index;
mod top_ord_and_number_queue;

pub use drill_down_query::DrillDownQuery;
pub use drill_sideways::{
    DrillSideways, DrillSidewaysFacets, DrillSidewaysResult, DrillSidewaysSearchResult,
};
pub use facet_counts_with_filter_query::FacetCountsWithFilterQuery;
pub use facet_field::{verify_label, FacetEntry, FacetField, SortedSetDocValuesFacetField};
pub use facet_query::{FacetQuery, MultiFacetQuery};
pub use facet_result::{FacetResult, FacetValue, LabelAndValue};
pub use facet_utils::{live_docs_disi, ConjunctionIterator, LiveDocsIterator};
pub use facets_collector::{
    DocIdSet, DocIdSetIter, FacetsCollector, FacetsCollectorManager, FacetsResult, MatchingDocs,
    SortedFacetsResult,
};
pub use facets_trait::{validate_top_n, Facets};
pub use long_value_facet_counts::LongValueFacetCounts;
pub use multi_double_values_source::{MultiDoubleValues, MultiDoubleValuesSource};
pub use multi_facets::MultiFacets;
pub use multi_long_values_source::{MultiLongValues, MultiLongValuesSource};
pub use random_sampling::{RandomSamplingFacetsCollector, RandomSamplingFacetsCollectorManager};
pub use sortedset::{
    ConcurrentSortedSetDocValuesFacetCounts, DefaultSortedSetDocValuesReaderState,
    SortedSetDocValuesFacetCounts,
};
pub use string_value_facet_counts::{StringDocValuesReaderState, StringValueFacetCounts};
pub use top_ord_and_number_queue::{OrdAndValue, TopOrdAndNumberQueue};

use lucene_codecs::doc_values::{self, NumericEntry, SortedNumericEntry};
use lucene_codecs::terms_dict::{self, TermsDictEntry};

use lucene_search::Result;

/// A single facet's ordinal, resolved label, and count — [`top_n_facets`]'s
/// element type, and also a convenient return shape for [`facet_counts`]
/// once resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacetCount {
    pub ord: i64,
    pub label: String,
    pub count: u64,
}

/// Counts, for every ordinal in a SORTED_SET field's terms dictionary, how
/// many of `matching_docs` have that ordinal among their values. Multi-valued
/// docs increment a counter for *every* one of their ordinals (real Lucene's
/// `SortedSetDocValuesFacetCounts.count` semantics), not just a "primary"
/// one. A doc not present in `matching_docs` contributes nothing. An empty
/// `matching_docs` produces a result with every ordinal present at count 0
/// (not an empty `Vec`) — this is the cleaner of the two options the task
/// brief allows: it keeps the result's ordinal set always equal to "every
/// term in the dictionary" regardless of how many docs matched, so callers
/// can always resolve labels for the full term set rather than special-
/// casing an empty match set.
///
/// `doc_values_data` is the segment's whole `.dvd` file, exactly as every
/// other function in [`lucene_search::doc_value_query`] takes it. Returns counts
/// indexed by ordinal (`result[ord] == count for ordinal ord`).
pub fn facet_counts(
    doc_values_data: &[u8],
    entry: &SortedNumericEntry,
    terms: &TermsDictEntry,
    matching_docs: &[i32],
) -> Result<Vec<u64>> {
    let mut counts = vec![0u64; terms.terms_dict_size as usize];
    // One forward cursor for the whole (ascending) candidate list and one
    // reused buffer: the free `sorted_numeric_values` re-walks the
    // docs-with-field blocks from the start and allocates, per document.
    let mut reader = doc_values::SortedNumericReader::new(doc_values_data, entry);
    let mut ords = Vec::new();
    for &doc_id in matching_docs {
        reader.values(doc_id, &mut ords)?;
        for &ord in &ords {
            if let Some(slot) = usize::try_from(ord).ok().and_then(|o| counts.get_mut(o)) {
                *slot += 1;
            }
        }
    }
    Ok(counts)
}

/// Resolves every ordinal's count (as returned by [`facet_counts`]) to its
/// string label via the field's terms dictionary, in ordinal order.
pub fn resolve_labels(
    doc_values_data: &[u8],
    terms: &TermsDictEntry,
    counts: &[u64],
) -> Result<Vec<FacetCount>> {
    // Streamed, not materialized: this walks the dictionary once in ordinal
    // order and turns each term straight into its `FacetCount`, so the
    // `Vec<Vec<u8>>` `decode_all_terms` would have built never exists beside
    // the result. Same "stop at the shorter of the two" behaviour a `zip`
    // over the whole dictionary had -- with one deliberate consequence:
    // stopping means corruption *past* `counts.len()` is no longer decoded,
    // so it is no longer reported here. That is the right trade (this
    // function's job is to label the counts it was given, and `check_index`
    // is what walks a dictionary to validate it), but it is a change in what
    // this call detects, not only in what it allocates.
    if terms.mask.is_some() {
        // The cursor reads the stored terms; the labels of a masked field are its masked ones.
        return Err(lucene_codecs::doc_values::Error::from(
            lucene_codecs::value_mask::masked_error("resolve_labels"),
        )
        .into());
    }
    let mut cursor = terms_dict::TermsCursor::open(doc_values_data, terms)
        .map_err(lucene_codecs::doc_values::Error::from)?;
    let mut labelled = Vec::with_capacity(counts.len());
    for (ord, &count) in counts.iter().enumerate() {
        let next = cursor
            .next_term()
            .map_err(lucene_codecs::doc_values::Error::from)?;
        let Some(label_bytes) = next else {
            break;
        };
        labelled.push(FacetCount {
            ord: ord as i64,
            label: String::from_utf8_lossy(label_bytes).into_owned(),
            count,
        });
    }
    Ok(labelled)
}

/// Real `AbstractSortedSetDocValueFacetCounts.getTopChildren(topN, dim)`:
/// keeps only children with a **non-zero** count, orders them by count
/// descending with ties broken by ascending ordinal
/// (`TopOrdAndIntQueue.OrdAndInt.lessThan` is `value <`, then `ord >`, i.e.
/// the *worst* entry is the lowest count and, on a tie, the highest ordinal —
/// so the best-first output is count DESC, ordinal ASC), truncates to `n`, and
/// reports `value`/`child_count` alongside.
///
/// Returns `None` when no child had a non-zero count — Java's
/// `createFacetResult` returns a `null` `FacetResult` for `childCount == 0`,
/// and `getAllDims` drops those, so "a dim with no values in the matched set"
/// is *absent*, not a row of zeros.
///
/// # Panics
///
/// If `n == 0`, with the same message real Lucene's `Facets.validateTopN`
/// throws its `IllegalArgumentException` with. Both are unchecked: a `topN` of
/// zero is a caller bug, not a runtime condition to branch on, and silently
/// returning an empty list would hide it.
pub fn top_children(facets: Vec<FacetCount>, n: usize) -> Option<FacetResult> {
    assert!(n > 0, "topN must be > 0 (got: {n})");
    let mut non_zero: Vec<FacetCount> = facets.into_iter().filter(|f| f.count > 0).collect();
    if non_zero.is_empty() {
        return None;
    }
    non_zero.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.ord.cmp(&b.ord)));
    Some(dimless_result(non_zero, Some(n)))
}

/// A dim-less [`FacetResult`] of `non_zero` (already in result order), its
/// labels cut to `keep`: `value` sums every count, `childCount` counts every
/// child, as Java's `int`s.
fn dimless_result(non_zero: Vec<FacetCount>, keep: Option<usize>) -> FacetResult {
    let as_int = |c: u64| i32::try_from(c).unwrap_or(i32::MAX);
    let child_count = i32::try_from(non_zero.len()).unwrap_or(i32::MAX);
    let value = non_zero
        .iter()
        .fold(0i32, |sum, f| sum.wrapping_add(as_int(f.count)));
    let label_values = non_zero
        .into_iter()
        .take(keep.unwrap_or(usize::MAX))
        .map(|f| LabelAndValue::new(f.label, FacetValue::Int(as_int(f.count))))
        .collect();
    FacetResult::new(
        String::new(),
        Vec::new(),
        FacetValue::Int(value),
        label_values,
        child_count,
    )
}

/// Real `AbstractSortedSetDocValueFacetCounts.getAllChildren(dim)`: every
/// child with a non-zero count, in **ordinal order** (Java iterates the dim's
/// `OrdRange` and appends in order, with no priority queue), plus the same
/// `value`/`child_count` aggregates [`top_children`] reports.
///
/// **Returns `None` when no child had a non-zero count**, which is *this
/// port's* convention for the dim-less primitives, not Java's:
/// `getAllChildren` has no `childCount == 0` guard and returns a
/// `FacetResult` with an empty `labelValues` array once
/// `prepareChildIteration` and `hasCounts` pass — only `getTopChildren`'s
/// `createFacetResult` nulls. [`SortedSetFacetCounts`]'s
/// [`Facets::get_all_children`], which
/// *does* have a dim and a state, follows Java and returns `Some` with an
/// empty child list.
pub fn all_children(facets: Vec<FacetCount>) -> Option<FacetResult> {
    let non_zero: Vec<FacetCount> = facets.into_iter().filter(|f| f.count > 0).collect();
    if non_zero.is_empty() {
        return None;
    }
    Some(dimless_result(non_zero, None))
}

/// Sorts `facets` descending by count, ties broken by ascending ordinal, then
/// truncates to at most `n`.
///
/// **This is not `Facets.getTopChildren`** — it keeps zero-count facets, which
/// Java never reports. It is kept as the raw ordering primitive (and for
/// callers that genuinely want "every ordinal, best first, padded with
/// zeroes"); use [`top_children`] for real `getTopChildren` semantics.
pub fn top_n_facets(mut facets: Vec<FacetCount>, n: usize) -> Vec<FacetCount> {
    facets.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.ord.cmp(&b.ord)));
    facets.truncate(n);
    facets
}

/// A single caller-defined numeric bucket for [`range_facet_counts`] — a
/// simplified port of real Lucene's `LongRange`/`DoubleRange`
/// (`lucene-facet` module): a `[min, max]` interval with each end
/// independently inclusive or exclusive, plus the label the bucket is
/// reported under. Values are `i64` here (the NUMERIC doc-values field's raw
/// stored representation); a `DoubleRange`-equivalent caller converts its
/// `f64` bounds to the field's `NumericUtils.doubleToSortableLong`-equivalent
/// `i64` encoding before constructing one of these, same as
/// [`lucene_search::doc_value_query`]'s numeric functions already assume for any
/// non-integer numeric field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericRange {
    pub label: String,
    pub min: i64,
    pub min_inclusive: bool,
    pub max: i64,
    pub max_inclusive: bool,
}

/// A caller-supplied range that cannot match anything — real Lucene's
/// `Range.failNoMatch()` (`IllegalArgumentException("range is empty: ...")`),
/// thrown from `LongRange`/`DoubleRange`'s constructors rather than silently
/// producing a bucket that always counts zero.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("range is empty: {label}")]
pub struct EmptyRange {
    pub label: String,
}

impl NumericRange {
    /// `LongRange(label, min, minInclusive, max, maxInclusive)`'s constructor,
    /// verbatim: an exclusive bound is normalized to the equivalent inclusive
    /// one by stepping it one *toward* the other end (`min++` / `max--`), and
    /// a bound already at the extreme it would step past — `min == i64::MAX`
    /// exclusive, `max == i64::MIN` exclusive — is `failNoMatch`, as is any
    /// range whose normalized `min > max`.
    ///
    /// The resulting range always has both ends inclusive, exactly like
    /// Java's; [`NumericRange`]'s `min_inclusive`/`max_inclusive` fields stay
    /// public so a caller that built one by hand keeps working, but this
    /// constructor is the one that reproduces Java's validation.
    pub fn new_long(
        label: impl Into<String>,
        min: i64,
        min_inclusive: bool,
        max: i64,
        max_inclusive: bool,
    ) -> std::result::Result<Self, EmptyRange> {
        let label = label.into();
        let fail = || EmptyRange {
            label: label.clone(),
        };
        let min = if min_inclusive {
            min
        } else {
            min.checked_add(1).ok_or_else(fail)?
        };
        let max = if max_inclusive {
            max
        } else {
            max.checked_sub(1).ok_or_else(fail)?
        };
        if min > max {
            return Err(fail());
        }
        Ok(NumericRange {
            label,
            min,
            min_inclusive: true,
            max,
            max_inclusive: true,
        })
    }

    /// `DoubleRange(label, min, minInclusive, max, maxInclusive)` followed by
    /// `DoubleRangeFacetCounts.getLongRanges`'s `toLongRange` conversion --
    /// i.e. the range a `DoubleDocValuesField` facet actually counts against.
    ///
    /// Java's order of operations matters and is reproduced exactly:
    /// 1. `NaN` on either end is rejected outright (`"min cannot be NaN"` /
    ///    `"max cannot be NaN"` — here, an [`EmptyRange`], since this port has
    ///    one error type for "this range can never be used").
    /// 2. An exclusive bound is nudged in **double** space
    ///    (`Math.nextUp(min)` / `Math.nextAfter(max, NEGATIVE_INFINITY)`),
    ///    *before* any integer transform — nudging after the transform would
    ///    be a different value for subnormals and around zero.
    /// 3. `min > max` after nudging is `failNoMatch`.
    /// 4. Both ends go through `NumericUtils.doubleToSortableLong`, which
    ///    flips the sign bit for a positive double and inverts every bit for a
    ///    negative one so that signed `i64` ordering matches `f64` ordering.
    ///
    /// The counted doc values must be mapped with [`sortable_double_bits`]
    /// (Java's `DoubleRangeFacetCounts.mapDocValue`) before being compared
    /// against a range built here — [`double_range_facet_counts`] does that
    /// for you.
    pub fn new_double(
        label: impl Into<String>,
        min: f64,
        min_inclusive: bool,
        max: f64,
        max_inclusive: bool,
    ) -> std::result::Result<Self, EmptyRange> {
        let label = label.into();
        let fail = || EmptyRange {
            label: label.clone(),
        };
        if min.is_nan() || max.is_nan() {
            return Err(fail());
        }
        let min = if min_inclusive { min } else { next_up(min) };
        let max = if max_inclusive { max } else { next_down(max) };
        if min > max {
            return Err(fail());
        }
        Ok(NumericRange {
            label,
            min: double_to_sortable_long(min),
            min_inclusive: true,
            max: double_to_sortable_long(max),
            max_inclusive: true,
        })
    }

    /// The range as inclusive `[min, max]` (an exclusive bound stepped one
    /// toward the other end, as [`Self::new_long`] normalises it), `None`
    /// when that leaves nothing to match.
    fn inclusive_bounds(&self) -> Option<(i64, i64)> {
        let min = if self.min_inclusive {
            self.min
        } else {
            self.min.checked_add(1)?
        };
        let max = if self.max_inclusive {
            self.max
        } else {
            self.max.checked_sub(1)?
        };
        (min <= max).then_some((min, max))
    }

    /// Whether `value` falls inside this range, honoring each bound's own
    /// inclusive/exclusive flag independently.
    #[cfg(test)]
    fn contains(&self, value: i64) -> bool {
        let above_min = if self.min_inclusive {
            value >= self.min
        } else {
            value > self.min
        };
        let below_max = if self.max_inclusive {
            value <= self.max
        } else {
            value < self.max
        };
        above_min && below_max
    }
}

/// `NumericUtils.doubleToSortableLong`: `sortableDoubleBits(Double
/// .doubleToLongBits(value))` — the raw IEEE-754 bit pattern with its sign
/// bit flipped for a positive value and every bit flipped for a negative one,
/// so that signed `i64` comparison reproduces `f64` comparison.
pub fn double_to_sortable_long(value: f64) -> i64 {
    sortable_double_bits(value.to_bits() as i64)
}

/// `NumericUtils.sortableDoubleBits(long bits)`: `bits ^ (bits >> 63) &
/// 0x7fffffffffffffffL`, i.e. its own inverse. Applied to a
/// `DoubleDocValuesField`'s stored `Double.doubleToRawLongBits` value it
/// yields the sortable form a range built by [`NumericRange::new_double`]
/// compares against — real `DoubleRangeFacetCounts.mapDocValue`.
pub use lucene_util::numeric_utils::sortable_double_bits;

/// `Math.nextUp(double)` for the finite/infinite cases this port needs (Rust
/// stabilized `f64::next_up` later than this project's pinned toolchain, so
/// it is spelled out here against the same bit-pattern rules Java uses).
fn next_up(v: f64) -> f64 {
    if v.is_nan() || v == f64::INFINITY {
        return v;
    }
    if v == 0.0 {
        // Both +0.0 and -0.0 step to the smallest positive subnormal.
        return f64::from_bits(1);
    }
    let bits = v.to_bits();
    if v > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// `Math.nextAfter(double, Double.NEGATIVE_INFINITY)`, i.e. `Math.nextDown`.
fn next_down(v: f64) -> f64 {
    if v.is_nan() || v == f64::NEG_INFINITY {
        return v;
    }
    if v == 0.0 {
        // Both +0.0 and -0.0 step to the largest negative subnormal.
        return f64::from_bits(1 | (1 << 63));
    }
    let bits = v.to_bits();
    if v > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Counts, for every range in `ranges`, how many of `matching_docs` have a
/// NUMERIC doc-value falling inside it — a simplified port of real Lucene's
/// `LongRangeFacetCounts`/`DoubleRangeFacetCounts.count`.
///
/// **Ranges are caller-defined and may overlap** (real Lucene doesn't
/// require ranges to partition the value space): a doc whose value matches
/// two or more ranges is counted in *each* one, independently — this
/// function makes one pass per range per doc's already-decoded value, with
/// no notion of "the" bucket a doc belongs to.
///
/// **A doc with no value for the field** (`numeric_value` returns `None` —
/// legitimate for a sparse NUMERIC field, see [`lucene_search::doc_value_query`]'s
/// module doc) contributes to **no** range, including an unbounded-looking
/// one like `i64::MIN..=i64::MAX` — the same missing-value rule
/// [`lucene_search::doc_value_query::search_numeric_range`] already documents,
/// applied here per-range instead of to a single range.
///
/// A doc not present in `matching_docs` contributes nothing. An empty
/// `matching_docs` produces every range present at count 0 (not an empty
/// `Vec`) — the same convention [`facet_counts`] documents for an empty
/// match set, kept consistent here.
///
/// Returns `(label, count)` pairs in the **same order** as `ranges` — that is
/// real Lucene's `RangeFacetCounts.getAllChildren`, which walks `ranges` in
/// caller order and reports every one, zero counts included. Feed the result
/// to [`top_range_children`] for `getTopChildren`'s count-descending,
/// zero-count-dropping ordering instead.
pub fn range_facet_counts(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
) -> Result<Vec<(String, u64)>> {
    Ok(range_facet_counts_with_total(doc_values_data, entry, ranges, matching_docs)?.counts)
}

/// `DoubleRangeFacetCounts.count`: the double-valued sibling of
/// [`range_facet_counts`]. `ranges` must have been built with
/// [`NumericRange::new_double`] (sortable-long space); each doc's raw stored
/// `Double.doubleToRawLongBits` doc value is mapped through
/// [`sortable_double_bits`] — Java's `mapDocValue` — before the comparison, so
/// the two live in the same ordered space.
///
/// Everything else (missing-value rule, overlapping ranges counting a doc in
/// each, caller-order output) is identical to [`range_facet_counts`].
pub fn double_range_facet_counts(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
) -> Result<Vec<(String, u64)>> {
    Ok(double_range_facet_counts_with_total(doc_values_data, entry, ranges, matching_docs)?.counts)
}

/// `RangeFacetCounts.getTopChildren(topN, dim)`: the ranges with a **non-zero**
/// count, ordered by count descending with ties broken by **ascending label**
/// (Java's priority queue orders by `count`, then `label` reversed, so the
/// best-first output is count DESC / label ASC — note this differs from the
/// SORTED_SET case's ordinal tie-break, because a range has no ordinal),
/// truncated to `n`.
///
/// `value` is `totCount`: the number of counted docs, which for
/// [`range_facet_counts`]/[`double_range_facet_counts`] is the number of
/// `matching_docs` that had a value at all — **not** the sum of the counts
/// (ranges may overlap, so summing would double-count). `child_count` is how
/// many ranges had a non-zero count.
///
/// Unlike the SORTED_SET case this returns a result even when every range is
/// empty (Java's `RangeFacetCounts.getTopChildren` returns a `FacetResult`
/// with an empty `labelValues` array rather than `null`).
///
/// # Panics
///
/// If `n == 0` — see [`top_children`].
pub fn top_range_children(
    counts: &[(String, u64)],
    total_count: u64,
    n: usize,
) -> RangeFacetResult {
    assert!(n > 0, "topN must be > 0 (got: {n})");
    let mut non_zero: Vec<(String, u64)> = counts.iter().filter(|(_, c)| *c > 0).cloned().collect();
    let child_count = non_zero.len();
    non_zero.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    non_zero.truncate(n);
    RangeFacetResult {
        label_values: non_zero,
        value: total_count,
        child_count,
    }
}

/// [`top_range_children`]'s return shape — real `FacetResult` for the range
/// case, whose children are labelled buckets rather than term ordinals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeFacetResult {
    /// `FacetResult.labelValues`, count descending / label ascending.
    pub label_values: Vec<(String, u64)>,
    /// `FacetResult.value` — `RangeFacetCounts.totCount`.
    pub value: u64,
    /// `FacetResult.childCount` — ranges with a non-zero count, which can
    /// exceed `label_values.len()` when `n` truncated the list.
    pub child_count: usize,
}

// ---------------------------------------------------------------------------
// `FacetsConfig` and the dim/path encoding
// ---------------------------------------------------------------------------

/// `FacetsConfig.DEFAULT_INDEX_FIELD_NAME` -- the doc-values field
/// `FacetsConfig.build` writes every dimension's ordinals into unless a dim
/// overrides it.
pub const DEFAULT_INDEX_FIELD_NAME: &str = "$facets";

/// `FacetsConfig.DELIM_CHAR` -- joins a dim and its path components into the
/// single SORTED_SET term that gets indexed.
pub const DELIM_CHAR: char = '\u{1F}';

/// `FacetsConfig`'s (private) `ESCAPE_CHAR`, escaping a literal [`DELIM_CHAR`]
/// or [`ESCAPE_CHAR`] inside a label.
pub const ESCAPE_CHAR: char = '\u{1E}';

/// `FacetsConfig.DimConfig` -- everything the search side needs to know about
/// a dimension that the *index* does not record.
///
/// Java's note applies verbatim and is the reason this type has to exist:
/// "this configuration is not saved into the index, but it's vital, and up to
/// the application to ensure, that at search time the provided `FacetsConfig`
/// matches what was used during indexing." Getting it wrong is not an error,
/// it is a wrong count -- a dim indexed as multi-valued but counted as
/// single-valued reports a `value` that double-counts documents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimConfig {
    /// `DimConfig.hierarchical`: paths under this dim have depth > 1 and every
    /// ancestor path was indexed too, so children are found by walking a
    /// [`DimTree`] rather than a contiguous [`OrdRange`].
    pub hierarchical: bool,
    /// `DimConfig.multiValued`: a document may carry more than one value for
    /// this dim, so summing child counts would double-count documents.
    pub multi_valued: bool,
    /// `DimConfig.requireDimCount`: the dim itself was indexed as its own
    /// ordinal, so an accurate document count for the whole dim is available
    /// without rolling children up.
    pub require_dim_count: bool,
    /// `DimConfig.drillDownTermsIndexing`: which drill-down `StringField`
    /// terms [`FacetsConfig::build_sorted_set_facet_fields`] indexes beside
    /// the doc-values values. Java's default is
    /// [`DrillDownTermsIndexing::All`], not "none".
    pub drill_down_terms_indexing: DrillDownTermsIndexing,
    /// `DimConfig.indexFieldName`.
    pub index_field_name: String,
}

/// `FacetsConfig.DrillDownTermsIndexing`: which of a facet path's prefixes get
/// indexed as searchable drill-down terms.
///
/// The examples are Java's own, for `FacetField("a", "foo/bar/baz")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DrillDownTermsIndexing {
    /// No drill-down terms at all, not even the full path.
    None,
    /// Only `"a/foo/bar/baz"`.
    FullPathOnly,
    /// `"a/foo"`, `"a/foo/bar"`, `"a/foo/bar/baz"` -- sub-paths and the full
    /// path, but not the bare dimension.
    AllPathsNoDim,
    /// `"a"` and `"a/foo/bar/baz"` -- the dimension and the full path, no
    /// intermediate sub-paths.
    DimensionAndFullPath,
    /// `"a"`, `"a/foo"`, `"a/foo/bar"`, `"a/foo/bar/baz"`. **Java's default.**
    #[default]
    All,
}

impl Default for DimConfig {
    /// `FacetsConfig.DEFAULT_DIM_CONFIG`: flat, single valued, no dim count,
    /// `DrillDownTermsIndexing.ALL`.
    fn default() -> Self {
        DimConfig {
            hierarchical: false,
            multi_valued: false,
            require_dim_count: false,
            drill_down_terms_indexing: DrillDownTermsIndexing::All,
            index_field_name: DEFAULT_INDEX_FIELD_NAME.to_string(),
        }
    }
}

/// `org.apache.lucene.facet.FacetsConfig` -- per-dimension configuration.
///
/// The read side is the setters and `getDimConfig` plus
/// [`path_to_string`]/[`string_to_path`]; the write side is
/// [`FacetsConfig::build`] (`FacetsConfig.build(taxoWriter, doc)`, taxonomy
/// and sorted-set facet fields) and
/// [`FacetsConfig::build_sorted_set_facet_fields`], the sorted-set half as
/// plain values.
///
/// `build` also records, per index field, the type of the
/// [`taxonomy::AssociationFacetField`]s indexed into it (`assocDimTypes`), to
/// refuse mixing them -- state that lives as long as the config, as in Java.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FacetsConfig {
    dims: std::collections::BTreeMap<String, DimConfig>,
    pub(crate) assoc_dim_types: AssocDimTypes,
}

/// `FacetsConfig.assocDimTypes`: index field name to association type
/// (`"int"`, `"float"` or `"bytes"`), filled by `build`.
#[derive(Debug, Default)]
pub(crate) struct AssocDimTypes(std::sync::Mutex<std::collections::BTreeMap<String, &'static str>>);

impl AssocDimTypes {
    fn snapshot(&self) -> std::collections::BTreeMap<String, &'static str> {
        self.0.lock().map(|m| m.clone()).unwrap_or_default()
    }

    /// The type recorded for `index_field_name`, recording `kind` when there
    /// is none yet.
    pub(crate) fn get_or_record(&self, index_field_name: &str, kind: &'static str) -> &'static str {
        match self.0.lock() {
            Ok(mut m) => {
                let recorded: &&'static str = m.entry(index_field_name.to_string()).or_insert(kind);
                recorded
            }
            Err(_) => kind,
        }
    }
}

impl Clone for AssocDimTypes {
    fn clone(&self) -> Self {
        AssocDimTypes(std::sync::Mutex::new(self.snapshot()))
    }
}

impl PartialEq for AssocDimTypes {
    fn eq(&self, other: &Self) -> bool {
        self.snapshot() == other.snapshot()
    }
}

impl Eq for AssocDimTypes {}

impl FacetsConfig {
    /// An empty config: every dim gets `DEFAULT_DIM_CONFIG`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `FacetsConfig.getDimConfig(dimName)`: the dim's configuration, or the
    /// default when it was never configured -- shared, as Java returns the
    /// one `DEFAULT_DIM_CONFIG` (a per-call copy was an allocation per
    /// `getTopChildren`).
    pub fn dim_config(&self, dim: &str) -> &DimConfig {
        static DEFAULT_DIM_CONFIG: std::sync::OnceLock<DimConfig> = std::sync::OnceLock::new();
        self.dims
            .get(dim)
            .unwrap_or_else(|| DEFAULT_DIM_CONFIG.get_or_init(DimConfig::default))
    }

    /// `FacetsConfig.isDimConfigured(dimName)`.
    pub fn is_dim_configured(&self, dim: &str) -> bool {
        self.dims.contains_key(dim)
    }

    /// `FacetsConfig.setHierarchical(dimName, v)`.
    pub fn set_hierarchical(&mut self, dim: impl Into<String>, v: bool) -> &mut Self {
        self.dims.entry(dim.into()).or_default().hierarchical = v;
        self
    }

    /// `FacetsConfig.setMultiValued(dimName, v)`.
    pub fn set_multi_valued(&mut self, dim: impl Into<String>, v: bool) -> &mut Self {
        self.dims.entry(dim.into()).or_default().multi_valued = v;
        self
    }

    /// `FacetsConfig.setRequireDimCount(dimName, v)`.
    pub fn set_require_dim_count(&mut self, dim: impl Into<String>, v: bool) -> &mut Self {
        self.dims.entry(dim.into()).or_default().require_dim_count = v;
        self
    }

    /// `FacetsConfig.setIndexFieldName(dimName, indexFieldName)`.
    pub fn set_index_field_name(
        &mut self,
        dim: impl Into<String>,
        index_field_name: impl Into<String>,
    ) -> &mut Self {
        self.dims.entry(dim.into()).or_default().index_field_name = index_field_name.into();
        self
    }

    /// `FacetsConfig.setDrillDownTermsIndexing(dimName, v)`.
    pub fn set_drill_down_terms_indexing(
        &mut self,
        dim: impl Into<String>,
        v: DrillDownTermsIndexing,
    ) -> &mut Self {
        self.dims
            .entry(dim.into())
            .or_default()
            .drill_down_terms_indexing = v;
        self
    }

    /// `FacetsConfig.getDimConfigs()`.
    pub fn dim_configs(&self) -> impl Iterator<Item = (&str, &DimConfig)> {
        self.dims.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// `FacetsConfig.build(Document)`'s `SortedSetDocValuesFacetField` half:
    /// turns one document's facet labels into the fields that actually get
    /// indexed.
    ///
    /// This is the **write-side** counterpart of everything else in this
    /// module, and the reason the read side works at all: [`facet_counts`]
    /// counts doc-values ordinals, [`FacetsState`] finds a dim's children by
    /// the `dim/child` prefix, and [`SortedSetFacetCounts::adjust_path_count`] reads the
    /// bare-dim ordinal --
    /// all three are only correct because `build` indexed those exact values.
    /// Without it a caller had to re-derive the encoding by hand and any
    /// mistake showed up as quietly wrong counts.
    ///
    /// `labels` is `(dim, path)` per facet field, in document order --
    /// `SortedSetDocValuesFacetField(dim, path...)`. The result is grouped by
    /// index field name, and within a group the values are in the order Java
    /// adds them.
    ///
    /// Per label, following `processSSDVFacetFields` exactly:
    ///
    /// - **hierarchical** dims index *every prefix* of the path
    ///   (`a`, `a/b`, `a/b/c`), because a hierarchical count needs an ordinal
    ///   for each level -- this is precisely what makes [`DimTree`] work;
    /// - **flat** dims must have exactly one path component beside the dim
    ///   (Java's `facetLabel.length != 2` check), and index the one full path;
    /// - a flat dim that is both `multi_valued` and `require_dim_count` also
    ///   indexes the **bare dimension**, which is the ordinal
    ///   [`SortedSetFacetCounts::adjust_path_count`] reads;
    /// - then `indexDrillDownTerms` adds the searchable `StringField` terms
    ///   its [`DrillDownTermsIndexing`] selects.
    ///
    /// A dim that is not `multi_valued` may appear only once per document
    /// (`checkSeen`).
    ///
    /// **Ordering note.** Java groups by index field name in a `HashMap` and
    /// iterates its entries, so the *relative order of two different index
    /// fields* is unspecified there; this returns them in first-appearance
    /// order. Nothing downstream can observe the difference -- doc-values
    /// values are sorted into a dictionary and drill-down terms are indexed
    /// terms -- and a deterministic order is worth more than reproducing a
    /// `HashMap`'s.
    pub fn build_sorted_set_facet_fields(
        &self,
        labels: &[(&str, &[&str])],
    ) -> std::result::Result<Vec<BuiltFacetField>, FacetBuildError> {
        let mut out: Vec<BuiltFacetField> = Vec::new();
        let mut seen_dims: Vec<&str> = Vec::new();

        for (dim, path) in labels {
            let config = self.dim_config(dim);
            if !config.multi_valued {
                // `checkSeen`.
                if seen_dims.contains(dim) {
                    return Err(FacetBuildError::NotMultiValued((*dim).to_string()));
                }
                seen_dims.push(dim);
            }

            // `FacetLabel(dim, path)`: the dim is component 0.
            let mut components: Vec<&str> = Vec::with_capacity(1 + path.len());
            components.push(dim);
            components.extend_from_slice(path);

            let field = match out
                .iter_mut()
                .position(|f| f.index_field_name == config.index_field_name)
            {
                Some(at) => &mut out[at],
                None => {
                    out.push(BuiltFacetField {
                        index_field_name: config.index_field_name.clone(),
                        sorted_set_values: Vec::new(),
                        drill_down_terms: Vec::new(),
                    });
                    out.last_mut().expect("just pushed")
                }
            };

            if config.hierarchical {
                // Every prefix, so every unique path has its own ordinal.
                for depth in 1..=components.len() {
                    field
                        .sorted_set_values
                        .push(path_components_to_string(&components[..depth])?);
                }
            } else {
                if components.len() != 2 {
                    return Err(FacetBuildError::NotHierarchical {
                        dim: (*dim).to_string(),
                        components: path.len(),
                    });
                }
                if config.multi_valued && config.require_dim_count {
                    field
                        .sorted_set_values
                        .push(path_components_to_string(&components[..1])?);
                }
                field
                    .sorted_set_values
                    .push(path_components_to_string(&components)?);
            }

            // `indexDrillDownTerms`.
            let indexing = config.drill_down_terms_indexing;
            if indexing != DrillDownTermsIndexing::None {
                field
                    .drill_down_terms
                    .push(path_components_to_string(&components)?);
                let prefixes: std::ops::Range<usize> = match indexing {
                    DrillDownTermsIndexing::None | DrillDownTermsIndexing::FullPathOnly => 0..0,
                    DrillDownTermsIndexing::DimensionAndFullPath => 1..2,
                    DrillDownTermsIndexing::AllPathsNoDim => 2..components.len(),
                    DrillDownTermsIndexing::All => 1..components.len(),
                };
                for depth in prefixes {
                    field
                        .drill_down_terms
                        .push(path_components_to_string(&components[..depth])?);
                }
            }
        }
        Ok(out)
    }
}

/// One index field's worth of [`FacetsConfig::build_sorted_set_facet_fields`]
/// output: what a caller must add to the document it hands `IndexWriter`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltFacetField {
    /// `DimConfig.indexFieldName` -- the field every value below belongs to.
    pub index_field_name: String,
    /// `SortedSetDocValuesField(indexFieldName, value)` values, in Java's
    /// order. These are what [`facet_counts`] counts.
    pub sorted_set_values: Vec<String>,
    /// `StringField(indexFieldName, term, Store.NO)` drill-down terms, in
    /// Java's order. These are what a drill-down `TermQuery` matches.
    pub drill_down_terms: Vec<String>,
}

/// Why [`FacetsConfig::build_sorted_set_facet_fields`] refused a document --
/// each variant is one of Java's own `IllegalArgumentException`s.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FacetBuildError {
    #[error("dimension {0:?} is not multiValued, but it appears more than once in this document")]
    NotMultiValued(String),
    #[error("dimension {dim:?} is not hierarchical yet has {components} components")]
    NotHierarchical { dim: String, components: usize },
    #[error(transparent)]
    EmptyPathComponent(#[from] EmptyPathComponent),
}

/// A path component that cannot be encoded -- real Lucene's
/// `IllegalArgumentException("each path component must have length > 0
/// (got: \"\")")` from `FacetsConfig.pathToString`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("each path component must have length > 0 (got: \"\")")]
pub struct EmptyPathComponent;

/// `FacetsConfig.pathToString(String[] path, int length)`: joins components
/// with [`DELIM_CHAR`], escaping any literal delimiter or escape character
/// inside a component with [`ESCAPE_CHAR`].
///
/// An empty `path` encodes to the empty string (Java's `length == 0` early
/// return); an empty *component* is rejected, because it would decode back to
/// a different path.
pub fn path_components_to_string(path: &[&str]) -> std::result::Result<String, EmptyPathComponent> {
    if path.is_empty() {
        return Ok(String::new());
    }
    let mut out = String::new();
    for (i, s) in path.iter().enumerate() {
        if s.is_empty() {
            return Err(EmptyPathComponent);
        }
        if i > 0 {
            out.push(DELIM_CHAR);
        }
        for ch in s.chars() {
            if ch == DELIM_CHAR || ch == ESCAPE_CHAR {
                out.push(ESCAPE_CHAR);
            }
            out.push(ch);
        }
    }
    Ok(out)
}

/// `FacetsConfig.pathToString(String dim, String... path)`: the dim prepended
/// to the path, then [`path_components_to_string`].
pub fn path_to_string(dim: &str, path: &[&str]) -> std::result::Result<String, EmptyPathComponent> {
    let mut full: Vec<&str> = Vec::with_capacity(1 + path.len());
    full.push(dim);
    full.extend_from_slice(path);
    path_components_to_string(&full)
}

/// `FacetsConfig.stringToPath(String s)`: the inverse of
/// [`path_components_to_string`].
///
/// Total, like Java's: a trailing escape character is dropped (Java asserts
/// `!lastEscape` and otherwise ignores it), and the empty string decodes to a
/// one-element path holding the empty string -- Java's `length == 0` early
/// return gives `String[0]`, which is the one case reproduced explicitly.
pub fn string_to_path(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    let mut parts = Vec::new();
    let mut buffer = String::new();
    let mut last_escape = false;
    for ch in s.chars() {
        if last_escape {
            buffer.push(ch);
            last_escape = false;
        } else if ch == ESCAPE_CHAR {
            last_escape = true;
        } else if ch == DELIM_CHAR {
            parts.push(std::mem::take(&mut buffer));
        } else {
            buffer.push(ch);
        }
    }
    parts.push(buffer);
    parts
}

/// What the dim layer reads of [`string_to_path`]`(s)`: its first component
/// and how many components it has, without building the others (borrowed
/// when the first holds no escape).
fn path_head(s: &str) -> (std::borrow::Cow<'_, str>, usize) {
    let bytes = s.as_bytes();
    // Both marks are ASCII: no byte of a multi-byte character equals one,
    // so the escaped character is skipped by skipping its first byte.
    let (delim, escape) = (DELIM_CHAR as u8, ESCAPE_CHAR as u8);
    let mut first_end = None;
    let mut first_escaped = false;
    let mut count = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == escape {
            if first_end.is_none() {
                first_escaped = true;
            }
            i += 2;
            continue;
        }
        if b == delim {
            if first_end.is_none() {
                first_end = Some(i);
            }
            count += 1;
        }
        i += 1;
    }
    let first = &s[..first_end.unwrap_or(s.len())];
    if !first_escaped {
        return (std::borrow::Cow::Borrowed(first), count);
    }
    let mut owned = string_to_path(first);
    (std::borrow::Cow::Owned(owned.swap_remove(0)), count)
}

// ---------------------------------------------------------------------------
// `SortedSetDocValuesReaderState`: the dim layer over one flat ordinal space
// ---------------------------------------------------------------------------

/// `SortedSetDocValuesReaderState.OrdRange` -- a flat dim's contiguous
/// `[start, end]` (both **inclusive**, as in Java) ordinal span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdRange {
    pub start: i64,
    pub end: i64,
}

/// `SortedSetDocValuesReaderState.DimTree` -- a hierarchical dim's sibling /
/// has-child links, built once from the sorted term list.
///
/// Java stores `hasChildren` as a `FixedBitSet` and `siblings` as an `int[]`;
/// this uses a `Vec<bool>` and a `Vec<i32>`, which is the same information at
/// 8x the bits for the flag. A `FixedBitSet` here would save bytes on a dim
/// with millions of paths and cost a shift-and-mask on every step of a walk
/// that is already following a random-access sibling pointer; not worth it
/// until a dim that big exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimTree {
    /// `DimTree.dimStartOrd` -- the dim's own ordinal, the root of the walk.
    pub dim_start_ord: i64,
    /// `siblings[i]` is the *dim-relative* ordinal of the next sibling of
    /// dim-relative ordinal `i`, or [`INVALID_ORDINAL`] when it has none.
    siblings: Vec<i32>,
    has_children: Vec<bool>,
}

/// `SortedSetDocValuesReaderState.INVALID_ORDINAL`.
pub const INVALID_ORDINAL: i32 = -1;

impl DimTree {
    /// `DimTree.iterator(pathOrd)` -- the immediate children of `path_ord`, in
    /// ordinal order. Empty when `path_ord` is outside the dim or is a leaf.
    pub fn children(&self, path_ord: i64) -> DimTreeChildren<'_> {
        DimTreeChildren {
            tree: self,
            current: path_ord - self.dim_start_ord,
            at_start: true,
        }
    }
}

/// [`DimTree::children`]'s iterator -- a port of the anonymous
/// `PrimitiveIterator.OfInt` in `DimTree.iterator(int)`.
#[derive(Debug)]
pub struct DimTreeChildren<'a> {
    tree: &'a DimTree,
    current: i64,
    at_start: bool,
}

impl Iterator for DimTreeChildren<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        if self.at_start {
            let idx = usize::try_from(self.current).ok()?;
            if !self.tree.has_children.get(idx).copied().unwrap_or(false) {
                return None;
            }
            self.at_start = false;
            self.current += 1;
            Some(self.current + self.tree.dim_start_ord)
        } else {
            let idx = usize::try_from(self.current).ok()?;
            let sibling = *self.tree.siblings.get(idx)?;
            if sibling == INVALID_ORDINAL {
                return None;
            }
            self.current = sibling as i64;
            Some(self.current + self.tree.dim_start_ord)
        }
    }
}

/// The term list handed to [`FacetsState::new`] is not a facet dictionary --
/// real Lucene's `DefaultSortedSetDocValuesReaderState` constructor throws
/// `IllegalArgumentException` for the same inputs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FacetsStateError {
    /// `"dimension not configured to handle hierarchical field; got: ..."` --
    /// a term with more than two components under a dim configured flat.
    #[error("dimension not configured to handle hierarchical field; got: {path:?} {term}")]
    NotHierarchical { path: Vec<String>, term: String },
    /// An **empty** term, which names no dimension. Java's constructor does
    /// `FacetsConfig.stringToPath(term)[0]` unguarded and throws
    /// `ArrayIndexOutOfBoundsException`; a facet dictionary written by
    /// `FacetsConfig.build` never contains one, but [`FacetsState::new`] takes
    /// a caller-supplied list, and a panic on caller data is not an
    /// improvement on Java's exception.
    #[error("ordinal {ord} has an empty label, which names no dimension")]
    EmptyLabel { ord: i64 },
}

/// `DefaultSortedSetDocValuesReaderState` -- the dim layer this port was
/// missing.
///
/// Built from one **global** ordinal space: either a single segment's
/// SORTED_SET terms, or the merged dictionary an [`lucene_search::ordinal_map::OrdinalMap`]
/// defines across segments. It parses every term into its `dim` plus path
/// (see [`string_to_path`]) and records, per dim, either the contiguous
/// [`OrdRange`] its ordinals occupy (flat dims) or the [`DimTree`] linking
/// each path to its siblings and children (hierarchical dims).
///
/// This is what [`FacetResult`]'s `dim`/`path` and [`SortedSetFacetCounts`]'s
/// `getAllDims`/`getSpecificValue` need and what
/// sit one layer above the counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacetsState {
    labels: Vec<String>,
    config: FacetsConfig,
    /// Java's `prefixToOrdRange`. A `BTreeMap`, not a `HashMap`, so
    /// [`Self::dims`] is deterministic -- see its doc.
    flat: std::collections::BTreeMap<String, OrdRange>,
    /// Java's `prefixToDimTree`.
    trees: std::collections::BTreeMap<String, DimTree>,
}

impl FacetsState {
    /// `DefaultSortedSetDocValuesReaderState(reader, field, config)`'s
    /// constructor, minus the doc-values plumbing: `labels` is the field's
    /// complete term list in ordinal order (what
    /// [`lucene_codecs::terms_dict::decode_all_terms`] returns, decoded to
    /// UTF-8, or the merged list across segments).
    pub fn new(
        labels: Vec<String>,
        config: FacetsConfig,
    ) -> std::result::Result<Self, FacetsStateError> {
        let mut state = FacetsState {
            labels,
            config,
            flat: std::collections::BTreeMap::new(),
            trees: std::collections::BTreeMap::new(),
        };
        // Java does `FacetsConfig.stringToPath(term)[0]` unguarded in three
        // places below and throws `ArrayIndexOutOfBoundsException` on an empty
        // term. Rejecting the whole list once, up front, turns that into a
        // typed error *and* makes every `[0]` in the two builders provably
        // in-bounds -- a facet dictionary written by `FacetsConfig.build`
        // never contains an empty term, but `new` takes a caller-supplied list.
        if let Some(ord) = state.labels.iter().position(String::is_empty) {
            return Err(FacetsStateError::EmptyLabel { ord: ord as i64 });
        }

        let value_count = state.labels.len() as i64;
        let mut ord = 0i64;
        while ord != value_count {
            let hierarchical = {
                let (dim, _) = path_head(&state.labels[ord as usize]);
                state.config.dim_config(&dim).hierarchical
            };
            ord = if hierarchical {
                state.build_one_hierarchical_dim(ord)? + 1
            } else {
                state.build_one_flat_dim(ord)? + 1
            };
        }
        Ok(state)
    }

    /// `createOneFlatFacetDimState` -- walk forward while the first component
    /// stays the same, rejecting any term deeper than `dim/child`.
    fn build_one_flat_dim(
        &mut self,
        dim_start_ord: i64,
    ) -> std::result::Result<i64, FacetsStateError> {
        let value_count = self.labels.len() as i64;
        let mut dim_end_ord = dim_start_ord;
        let not_hierarchical = |term: &String| FacetsStateError::NotHierarchical {
            path: string_to_path(term),
            term: term.clone(),
        };
        let first = &self.labels[dim_end_ord as usize];
        let (dim, len) = path_head(first);
        if len > 2 {
            return Err(not_hierarchical(first));
        }
        // The first component of every label: non-empty labels (checked by
        // `FacetsState::new`) have one.
        let dim = dim.into_owned();
        loop {
            if dim_end_ord + 1 == value_count {
                break;
            }
            let next = &self.labels[(dim_end_ord + 1) as usize];
            let (next_dim, next_len) = path_head(next);
            if next_dim != dim.as_str() {
                break;
            }
            if next_len != 2 {
                return Err(not_hierarchical(next));
            }
            dim_end_ord += 1;
        }
        self.flat.insert(
            dim,
            OrdRange {
                start: dim_start_ord,
                end: dim_end_ord,
            },
        );
        Ok(dim_end_ord)
    }

    /// `createOneHierarchicalFacetDimState` -- one forward pass building the
    /// sibling links, using a stack of paths whose sibling is still unknown.
    ///
    /// Ported structurally rather than idiomatically: the "pop every stacked
    /// path at least as deep as the current one, and if it is *exactly* as
    /// deep it was waiting for this ordinal as its sibling" step is the whole
    /// algorithm, and it only works because `FacetsConfig.build` indexes every
    /// ancestor path, so a depth can never jump by more than one.
    fn build_one_hierarchical_dim(
        &mut self,
        dim_start_ord: i64,
    ) -> std::result::Result<i64, FacetsStateError> {
        let value_count = self.labels.len() as i64;
        let mut has_children: Vec<bool> = Vec::new();
        let mut siblings: Vec<i32> = Vec::new();
        // (dim-relative ord, component count) for paths still awaiting a sibling.
        let mut sibling_stack: Vec<(usize, usize)> = Vec::new();

        let mut dim_end_ord = dim_start_ord;
        // Each label's first component and component count.
        let (dim, mut next_len) = path_head(&self.labels[dim_end_ord as usize]);
        let dim = dim.into_owned();

        loop {
            let len = next_len;
            let ord = (dim_end_ord - dim_start_ord) as usize;

            while let Some(&(stacked_ord, stacked_len)) = sibling_stack.last() {
                if stacked_len < len {
                    break;
                }
                sibling_stack.pop();
                if stacked_len == len {
                    siblings[stacked_ord] = ord as i32;
                }
            }

            if dim_end_ord + 1 == value_count {
                siblings.push(INVALID_ORDINAL);
                has_children.push(false);
                break;
            }

            let (next_dim, len_after) = path_head(&self.labels[(dim_end_ord + 1) as usize]);
            next_len = len_after;
            if next_dim != dim.as_str() {
                siblings.push(INVALID_ORDINAL);
                has_children.push(false);
                break;
            }

            match len.cmp(&next_len) {
                std::cmp::Ordering::Less => {
                    // The next ordinal is a direct child: every ancestor path
                    // is indexed, so a deeper next term is always exactly one
                    // level deeper.
                    has_children.push(true);
                    sibling_stack.push((ord, len));
                    siblings.push(INVALID_ORDINAL);
                }
                std::cmp::Ordering::Equal => {
                    siblings.push(ord as i32 + 1);
                    has_children.push(false);
                }
                std::cmp::Ordering::Greater => {
                    siblings.push(INVALID_ORDINAL);
                    has_children.push(false);
                }
            }

            dim_end_ord += 1;
        }

        self.trees.insert(
            dim,
            DimTree {
                dim_start_ord,
                siblings,
                has_children,
            },
        );
        Ok(dim_end_ord)
    }

    /// `SortedSetDocValuesReaderState.getDims()`.
    ///
    /// Java yields the hierarchical dims first and then the flat ones, each
    /// group in `HashMap` iteration order -- i.e. unspecified. This yields
    /// hierarchical first and then flat, each group in **ascending dim name**
    /// order, so a caller iterating dims gets a stable answer. `getAllDims`
    /// sorts its own output regardless, so the only observable difference is
    /// for a caller walking `dims()` directly.
    pub fn dims(&self) -> impl Iterator<Item = &str> {
        self.trees
            .keys()
            .chain(self.flat.keys())
            .map(String::as_str)
    }

    /// `SortedSetDocValuesReaderState.getOrdRange(dim)`. `None` when the dim
    /// was never indexed; Java additionally throws
    /// `UnsupportedOperationException` when the dim is configured
    /// hierarchical, which here is simply `None` (a hierarchical dim has a
    /// [`DimTree`], never an `OrdRange`).
    pub fn ord_range(&self, dim: &str) -> Option<&OrdRange> {
        self.flat.get(dim)
    }

    /// `SortedSetDocValuesReaderState.getDimTree(dim)`.
    pub fn dim_tree(&self, dim: &str) -> Option<&DimTree> {
        self.trees.get(dim)
    }

    /// `SortedSetDocValuesReaderState.getSize()` -- the number of ordinals.
    pub fn size(&self) -> usize {
        self.labels.len()
    }

    /// `SortedSetDocValuesReaderState.getFacetsConfig()`.
    pub fn config(&self) -> &FacetsConfig {
        &self.config
    }

    /// `SortedSetDocValues.lookupOrd(ord)`, on the state's own dictionary.
    pub fn label(&self, ord: i64) -> Option<&str> {
        usize::try_from(ord)
            .ok()
            .and_then(|i| self.labels.get(i))
            .map(String::as_str)
    }

    /// `SortedSetDocValues.lookupTerm(term)` -- the ordinal of an exact
    /// encoded path, or `None`. Binary search, since the dictionary is sorted.
    pub fn lookup_term(&self, term: &str) -> Option<i64> {
        self.labels
            .binary_search_by(|probe| probe.as_str().cmp(term))
            .ok()
            .map(|i| i as i64)
    }
}

// ---------------------------------------------------------------------------
// `SortedSetDocValuesFacetCounts`: the dim-aware result layer
// ---------------------------------------------------------------------------

/// `SortedSetDocValuesFacetCounts` / `AbstractSortedSetDocValueFacetCounts` --
/// counts for one global ordinal space, read through a [`FacetsState`].
///
/// The counting itself is still [`facet_counts`] (per segment) plus
/// [`merge_segment_counts`] (across segments); this type is the `Facets`
/// implementation on top of the resulting `counts` array.
#[derive(Debug, Clone)]
pub struct SortedSetFacetCounts<'a> {
    state: &'a FacetsState,
    /// Java's `int[] counts`, `None` until a segment was counted.
    counts: Option<Vec<i32>>,
}

/// `AbstractSortedSetDocValueFacetCounts.TopChildrenForPath` -- the
/// intermediate result before ordinals are resolved to labels.
struct TopChildrenForPath {
    path_count: i32,
    child_count: usize,
    top: Vec<(i64, i32)>,
}

impl<'a> SortedSetFacetCounts<'a> {
    /// `counts` is indexed by the state's ordinals. An **empty** `counts` is
    /// Java's `hasCounts() == false` (its `counts` array is allocated lazily
    /// and stays `null` when nothing matched): every accessor then reports
    /// "no results" rather than zeros.
    ///
    /// A count past `i32::MAX` saturates (Java's counts are `int`s).
    pub fn new(state: &'a FacetsState, counts: Vec<u64>) -> Self {
        let counts = if counts.is_empty() {
            None
        } else {
            Some(
                counts
                    .into_iter()
                    .map(|c| i32::try_from(c).unwrap_or(i32::MAX))
                    .collect(),
            )
        };
        SortedSetFacetCounts { state, counts }
    }

    /// Over Java's own `int[] counts` (`None`: nothing was counted).
    pub(crate) fn from_counts(state: &'a FacetsState, counts: Option<Vec<i32>>) -> Self {
        SortedSetFacetCounts { state, counts }
    }

    /// `AbstractSortedSetDocValueFacetCounts.hasCounts()`.
    fn has_counts(&self) -> bool {
        self.counts.is_some()
    }

    /// `AbstractSortedSetDocValueFacetCounts.getCount(ord)`: `0` outside the
    /// array, where Java would throw.
    fn count_i32(&self, ord: i64) -> i32 {
        usize::try_from(ord)
            .ok()
            .and_then(|i| self.counts.as_ref()?.get(i))
            .copied()
            .unwrap_or(0)
    }

    /// `prepareChildIteration(dim, dimConfig, path)`: the path's own ordinal
    /// plus its immediate children, or `None` when the path was never indexed.
    ///
    /// A non-empty `path` for a dim that is not configured hierarchical is
    /// Java's `IllegalArgumentException`, as is an empty path component.
    fn prepare_child_iteration(
        &self,
        dim: &str,
        dim_config: &DimConfig,
        path: &[&str],
    ) -> Result<Option<(i64, ChildOrds<'a>)>> {
        if dim_config.hierarchical {
            let Some(tree) = self.state.dim_tree(dim) else {
                return Ok(None);
            };
            let path_ord = if path.is_empty() {
                tree.dim_start_ord
            } else {
                let term = path_to_string(dim, path)
                    .map_err(|e| lucene_search::Error::IllegalArgument(e.to_string()))?;
                match self.state.lookup_term(&term) {
                    Some(ord) => ord,
                    // The path was never indexed.
                    None => return Ok(None),
                }
            };
            Ok(Some((path_ord, ChildOrds::Tree(tree.children(path_ord)))))
        } else {
            if !path.is_empty() {
                return Err(lucene_search::Error::IllegalArgument(
                    "Field is not configured as hierarchical, path should be 0 length".to_string(),
                ));
            }
            let Some(range) = self.state.ord_range(dim).copied() else {
                return Ok(None);
            };
            let mut start = range.start;
            if dim_config.multi_valued && dim_config.require_dim_count {
                // The dim's own ordinal was indexed and leads the range; skip
                // past it so the iteration starts on the first real child.
                start += 1;
            }
            Ok(Some((
                range.start,
                ChildOrds::Range {
                    next: start,
                    end: range.end,
                },
            )))
        }
    }

    /// `adjustPathCountIfNecessary` -- `-1` is Java's "no accurate count is
    /// obtainable for this dim", not a sentinel this port invented.
    //
    // SENTINEL: `-1` = "no accurate count obtainable", outside the domain of
    // a count.
    fn adjust_path_count(&self, dim_config: &DimConfig, path_ord: i64, computed: i32) -> i32 {
        if dim_config.hierarchical || (dim_config.multi_valued && dim_config.require_dim_count) {
            self.count_i32(path_ord)
        } else if dim_config.multi_valued {
            -1
        } else {
            computed
        }
    }

    /// `computeTopChildren`.
    ///
    /// Java keeps a `topN`-sized min-heap (`TopOrdAndIntQueue`, `lessThan` is
    /// `value <` then `ord >`), so the best-first order is count DESC / ord
    /// ASC. This collects the non-zero children and sorts on the same key,
    /// which is O(children log children) against Java's
    /// O(children log topN) -- the same trade [`top_children`] already makes,
    /// and the same answer.
    fn compute_top_children(
        &self,
        children: ChildOrds<'_>,
        top_n: usize,
        dim_config: &DimConfig,
        path_ord: i64,
    ) -> TopChildrenForPath {
        let mut path_count: i32 = 0;
        let mut top: Vec<(i64, i32)> = Vec::new();
        for ord in children {
            let count = self.count_i32(ord);
            if count > 0 {
                path_count = path_count.wrapping_add(count);
                top.push((ord, count));
            }
        }
        let child_count = top.len();
        let best_first = |a: &(i64, i32), b: &(i64, i32)| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0));
        if top_n == 0 {
            top.clear();
        } else if top.len() > top_n {
            top.select_nth_unstable_by(top_n.saturating_sub(1), best_first);
            top.truncate(top_n);
        }
        top.sort_unstable_by(best_first);
        TopChildrenForPath {
            path_count: self.adjust_path_count(dim_config, path_ord, path_count),
            child_count,
            top,
        }
    }

    /// `createFacetResult` -- resolves ordinals to their *last* path
    /// component, which is what `FacetResult.labelValues` carries.
    fn create_facet_result(
        &self,
        top: Option<TopChildrenForPath>,
        dim: &str,
        path: &[&str],
    ) -> Option<FacetResult> {
        let top = top?;
        if top.child_count == 0 {
            return None;
        }
        Some(FacetResult::new(
            dim,
            path.iter().map(|s| s.to_string()).collect(),
            FacetValue::Int(top.path_count),
            top.top
                .into_iter()
                .map(|(ord, count)| {
                    LabelAndValue::new(self.leaf_label(ord), FacetValue::Int(count))
                })
                .collect(),
            i32::try_from(top.child_count).unwrap_or(i32::MAX),
        ))
    }

    /// `FacetsConfig.stringToPath(term)[parts.length - 1]`.
    fn leaf_label(&self, ord: i64) -> String {
        self.state
            .label(ord)
            .map(|term| string_to_path(term).pop().unwrap_or_default())
            .unwrap_or_default()
    }

    /// `getTopChildrenForPath(topN, dim, path...)`.
    fn top_children_for_path(
        &self,
        top_n: usize,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<TopChildrenForPath>> {
        let dim_config = self.state.config().dim_config(dim);
        let Some((path_ord, children)) = self.prepare_child_iteration(dim, dim_config, path)?
        else {
            return Ok(None);
        };
        Ok(Some(self.compute_top_children(
            children, top_n, dim_config, path_ord,
        )))
    }
}

impl Facets for SortedSetFacetCounts<'_> {
    /// `getAllChildren(dim, path...)`: every non-zero child in **ordinal**
    /// order (no priority queue at all), plus the same `value`/`childCount`
    /// aggregates.
    fn get_all_children(&self, dim: &str, path: &[&str]) -> Result<Option<FacetResult>> {
        let dim_config = self.state.config().dim_config(dim);
        let Some((path_ord, children)) = self.prepare_child_iteration(dim, dim_config, path)?
        else {
            return Ok(None);
        };
        if !self.has_counts() {
            return Ok(None);
        }
        let mut path_count: i32 = 0;
        let mut label_values = Vec::new();
        for ord in children {
            let count = self.count_i32(ord);
            if count > 0 {
                path_count = path_count.wrapping_add(count);
                label_values.push(LabelAndValue::new(
                    self.leaf_label(ord),
                    FacetValue::Int(count),
                ));
            }
        }
        let child_count = i32::try_from(label_values.len()).unwrap_or(i32::MAX);
        Ok(Some(FacetResult::new(
            dim,
            path.iter().map(|s| s.to_string()).collect(),
            FacetValue::Int(self.adjust_path_count(dim_config, path_ord, path_count)),
            label_values,
            child_count,
        )))
    }

    /// `getTopChildren(topN, dim, path...)`.
    fn get_top_children(
        &self,
        top_n: i32,
        dim: &str,
        path: &[&str],
    ) -> Result<Option<FacetResult>> {
        validate_top_n(top_n)?;
        if !self.has_counts() {
            return Ok(None);
        }
        let top_n = usize::try_from(top_n).unwrap_or(usize::MAX);
        let top = self.top_children_for_path(top_n, dim, path)?;
        Ok(self.create_facet_result(top, dim, path))
    }

    /// `getSpecificValue(dim, path...)`: the count for one exact path, or
    /// `-1` when the path was never indexed. `0` when nothing was counted at
    /// all (Java's `hasCounts() == false` branch).
    //
    // SENTINEL: `-1` = "this path was never indexed", outside the domain of a
    // count. Java's own `Facets.getSpecificValue` contract.
    fn get_specific_value(&self, dim: &str, path: &[&str]) -> Result<FacetValue> {
        let dim_config = self.state.config().dim_config(dim);
        if !dim_config.hierarchical && path.len() != 1 {
            return Err(lucene_search::Error::IllegalArgument(format!(
                "{dim} is not configured as hierarchical, path must be length=1"
            )));
        }
        let term = path_to_string(dim, path)
            .map_err(|e| lucene_search::Error::IllegalArgument(e.to_string()))?;
        let Some(ord) = self.state.lookup_term(&term) else {
            return Ok(FacetValue::Int(-1));
        };
        Ok(FacetValue::Int(if self.has_counts() {
            self.count_i32(ord)
        } else {
            0
        }))
    }

    /// `getAllDims(topN)`: every dim that had a hit, each with its top `topN`
    /// children, sorted by `value` descending with ties broken by ascending
    /// dim name.
    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        validate_top_n(top_n)?;
        if !self.has_counts() {
            return Ok(Vec::new());
        }
        let top_n = usize::try_from(top_n).unwrap_or(usize::MAX);
        let mut results: Vec<FacetResult> = Vec::new();
        for dim in self.state.dims() {
            let top = self.top_children_for_path(top_n, dim, &[])?;
            if let Some(r) = self.create_facet_result(top, dim, &[]) {
                results.push(r);
            }
        }
        results.sort_by(|a, b| {
            b.value
                .int_value()
                .cmp(&a.value.int_value())
                .then_with(|| taxonomy::facet_label::java_compare(&a.dim, &b.dim))
        });
        Ok(results)
    }

    /// `getTopDims(topNDims, topNChildren)`: the dims ranked by their count
    /// -- read off the dim's own ordinal where it was indexed, `-1` for a
    /// multi-valued dim that does not require it, the children's sum for a
    /// single-valued flat dim -- and the best `top_n_dims` of them (count
    /// descending, then dim), each with its top children.
    ///
    /// Differs: a ranked dim with no counted child (a multi-valued dim's
    /// `-1`) is left out, where Java's list holds `null` for it.
    fn get_top_dims(&self, top_n_dims: i32, top_n_children: i32) -> Result<Vec<FacetResult>> {
        validate_top_n(top_n_dims)?;
        validate_top_n(top_n_children)?;
        if !self.has_counts() {
            return Ok(Vec::new());
        }
        let top_n = usize::try_from(top_n_children).unwrap_or(usize::MAX);
        let mut ranked: Vec<(&str, i32, Option<TopChildrenForPath>)> = Vec::new();
        for dim in self.state.dims() {
            let dim_config = self.state.config().dim_config(dim);
            let (dim_count, intermediate) = if dim_config.hierarchical {
                match self.state.dim_tree(dim) {
                    Some(tree) => (self.count_i32(tree.dim_start_ord), None),
                    None => continue,
                }
            } else {
                let Some(range) = self.state.ord_range(dim).copied() else {
                    continue;
                };
                if dim_config.multi_valued {
                    if dim_config.require_dim_count {
                        (self.count_i32(range.start), None)
                    } else {
                        (-1, None)
                    }
                } else {
                    let children = ChildOrds::Range {
                        next: range.start,
                        end: range.end,
                    };
                    let top = self.compute_top_children(children, top_n, dim_config, range.start);
                    (top.path_count, Some(top))
                }
            };
            if dim_count != 0 {
                ranked.push((dim, dim_count, intermediate));
            }
        }
        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| taxonomy::facet_label::java_compare(a.0, b.0))
        });
        ranked.truncate(usize::try_from(top_n_dims).unwrap_or(usize::MAX));
        let mut results = Vec::with_capacity(ranked.len());
        for (dim, _, intermediate) in ranked {
            let top = match intermediate {
                Some(top) => Some(top),
                None => self.top_children_for_path(top_n, dim, &[])?,
            };
            if let Some(r) = self.create_facet_result(top, dim, &[]) {
                results.push(r);
            }
        }
        Ok(results)
    }
}

/// [`SortedSetFacetCounts::prepare_child_iteration`]'s result: a flat dim's
/// contiguous ordinal span, or a hierarchical dim's sibling walk. An enum
/// rather than a `Box<dyn Iterator>` so the flat case -- the common one --
/// stays a plain counter with no allocation and no virtual call.
enum ChildOrds<'a> {
    /// Inclusive `[next, end]`.
    Range {
        next: i64,
        end: i64,
    },
    Tree(DimTreeChildren<'a>),
}

impl Iterator for ChildOrds<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        match self {
            ChildOrds::Range { next, end } => {
                if *next > *end {
                    None
                } else {
                    let v = *next;
                    *next += 1;
                    Some(v)
                }
            }
            ChildOrds::Tree(it) => it.next(),
        }
    }
}

/// `SortedSetDocValuesFacetCounts.countOneSegment`'s remap step:
/// `counts[(int) ordMap.get(ord)] += count`.
///
/// `segment_counts[i]` is segment `i`'s own [`facet_counts`] output, indexed
/// by that segment's local ordinals; the result is indexed by `map`'s global
/// ordinals. This is the operation whose absence made cross-segment faceting
/// unavailable: summing the per-segment arrays elementwise would add together
/// counts for unrelated terms that happen to share an ordinal number.
///
/// A segment with more local ordinals than the map knows about contributes
/// nothing for the excess -- it means the map was built from a different term
/// list than the counts were.
pub fn merge_segment_counts(
    map: &lucene_search::ordinal_map::OrdinalMap,
    segment_counts: &[Vec<u64>],
) -> Vec<u64> {
    let mut global = vec![0u64; map.value_count().max(0) as usize];
    for (segment, counts) in segment_counts.iter().enumerate() {
        let Some(ords) = map.segment_ords(segment) else {
            continue;
        };
        for (local, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            if let Some(&g) = ords.get(local) {
                if let Some(slot) = global.get_mut(g as usize) {
                    *slot += count;
                }
            }
        }
    }
    global
}

// ---------------------------------------------------------------------------
// Range counting: `totCount`, and the multi-valued branch
// ---------------------------------------------------------------------------

/// [`range_facet_counts_with_total`]'s result: the per-range counts *and*
/// `RangeFacetCounts.totCount`, which a caller cannot derive from the counts
/// alone once ranges overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeCounts {
    /// `(label, count)` per range, in caller order -- `getAllChildren`'s
    /// listing.
    pub counts: Vec<(String, u64)>,
    /// `RangeFacetCounts.totCount`: how many of `matching_docs` landed in **at
    /// least one** range. Java computes it as "every matching doc, minus the
    /// ones with no value, minus the ones whose value hit no range"; a doc in
    /// three overlapping ranges still counts once, which is why summing
    /// `counts` is not the same number.
    pub total_count: u64,
}

/// [`range_facet_counts`] plus `RangeFacetCounts.totCount` -- the number
/// [`top_range_children`] wants for `FacetResult.value` and which the older
/// signature made the caller invent.
pub fn range_facet_counts_with_total(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
) -> Result<RangeCounts> {
    count_single_valued(doc_values_data, entry, ranges, matching_docs, |v| v)
}

/// [`double_range_facet_counts`] plus `RangeFacetCounts.totCount`.
pub fn double_range_facet_counts_with_total(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
) -> Result<RangeCounts> {
    count_single_valued(
        doc_values_data,
        entry,
        ranges,
        matching_docs,
        sortable_double_bits,
    )
}

/// The ranges as the inclusive [`range::LongRange`]s a
/// [`range::LongRangeCounter`] counts into, with each counter slot's index
/// in `ranges`. A range that can match nothing once its exclusive bounds are
/// stepped inward (a hand-built [`NumericRange`]; the constructors refuse
/// those) is left out: it counts zero, as testing every value against it did.
fn counter_ranges(ranges: &[NumericRange]) -> (Vec<range::LongRange>, Vec<usize>) {
    let mut long_ranges = Vec::with_capacity(ranges.len());
    let mut slots = Vec::with_capacity(ranges.len());
    for (i, r) in ranges.iter().enumerate() {
        if let Some((min, max)) = r.inclusive_bounds() {
            long_ranges.push(range::LongRange {
                label: String::new(),
                min,
                max,
            });
            slots.push(i);
        }
    }
    (long_ranges, slots)
}

/// The counter's per-range counts back in `ranges`' order.
fn spread_counts(n: usize, slots: &[usize], counted: Vec<i32>) -> Vec<u64> {
    let mut counts = vec![0u64; n];
    for (&slot, c) in slots.iter().zip(counted) {
        if let Some(out) = counts.get_mut(slot) {
            // A count is of documents of one segment: never negative.
            *out = u64::try_from(c).unwrap_or(0);
        }
    }
    counts
}

/// `RangeFacetCounts.count`'s single-valued branch over a NUMERIC column:
/// one forward [`NumericReader`](doc_values::NumericReader) for the whole
/// candidate list, and each value placed with `LongRangeCounter`'s binary
/// search over the elementary intervals (a segment tree when ranges
/// overlap) rather than tested against every range.
fn count_single_valued(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
    map_doc_value: fn(i64) -> i64,
) -> Result<RangeCounts> {
    let (long_ranges, slots) = counter_ranges(ranges);
    let mut counter = range::LongRangeCounter::create(&long_ranges);
    let mut reader = doc_values::NumericReader::new(doc_values_data, entry);
    let mut with_value = 0u64;
    for &doc_id in matching_docs {
        if let Some(raw) = reader.value(doc_id)? {
            counter.add_single_valued(map_doc_value(raw));
            with_value += 1;
        }
    }
    // `totCount`: the documents with a value, minus those in no range.
    let missing = u64::try_from(counter.finish()).unwrap_or(0);
    let total_count = with_value.saturating_sub(missing);
    let counts = spread_counts(ranges.len(), &slots, counter.into_counts());
    Ok(RangeCounts {
        counts: label_counts(ranges, counts),
        total_count,
    })
}

/// `RangeFacetCounts.count`'s **multi-valued** branch: the SORTED_NUMERIC
/// sibling of [`range_facet_counts`].
///
/// Two rules make this more than "run the single-valued loop per value", and
/// both are load-bearing:
///
/// - **A document is counted at most once per range.** Java achieves it with
///   `startMultiValuedDoc`/`addMultiValued`/`endMultiValuedDoc`, which set a
///   bit per elementary interval and fold them in once at the end of the doc.
///   Counting per value instead would report a document with sizes
///   `{1, 2, 3}` three times in a `[1, 3]` bucket.
/// - **`totCount` counts the document once**, not once per matched range
///   (`endMultiValuedDoc()` returns a single boolean).
///
/// Java additionally skips a value equal to the immediately preceding one
/// (`if (j == 0 || val != previous)`); SORTED_NUMERIC values are stored
/// ascending, so that removes exact duplicates. It is subsumed here by the
/// per-range at-most-once rule and reproduced anyway, since it is what makes
/// the two implementations agree value-for-value.
///
/// A document with no value for the field contributes to nothing at all, not
/// even to `totCount` -- Java's `advanceExact` returning false.
pub fn multi_valued_range_facet_counts(
    doc_values_data: &[u8],
    entry: &SortedNumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
) -> Result<RangeCounts> {
    count_multi_valued(doc_values_data, entry, ranges, matching_docs, |v| v)
}

/// [`multi_valued_range_facet_counts`] for a multi-valued **double** field:
/// every stored value goes through [`sortable_double_bits`]
/// (`DoubleRangeFacetCounts.mapDocValue`) first, so it is compared in the same
/// ordered space a [`NumericRange::new_double`] range lives in.
pub fn multi_valued_double_range_facet_counts(
    doc_values_data: &[u8],
    entry: &SortedNumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
) -> Result<RangeCounts> {
    count_multi_valued(
        doc_values_data,
        entry,
        ranges,
        matching_docs,
        sortable_double_bits,
    )
}

fn count_multi_valued(
    doc_values_data: &[u8],
    entry: &SortedNumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
    map_doc_value: fn(i64) -> i64,
) -> Result<RangeCounts> {
    let (long_ranges, slots) = counter_ranges(ranges);
    let mut counter = range::LongRangeCounter::create(&long_ranges);
    let mut reader = doc_values::SortedNumericReader::new(doc_values_data, entry);
    let mut values = Vec::new();
    let mut total_count = 0u64;
    for &doc_id in matching_docs {
        reader.values(doc_id, &mut values)?;
        if values.is_empty() {
            continue;
        }
        for v in values.iter_mut() {
            *v = map_doc_value(*v);
        }
        // `addMultiValued` walks the intervals forward and needs ascending
        // values. Stored values are ascending; a mapping that reorders them
        // (`sortable_double_bits` over raw negative doubles) is sorted back,
        // which leaves the set of ranges each document hits unchanged.
        if !values.windows(2).all(|w| w[0] <= w[1]) {
            values.sort_unstable();
        }
        counter.start_multi_valued_doc();
        let mut previous = 0i64;
        for (j, &value) in values.iter().enumerate() {
            if j != 0 && value == previous {
                continue;
            }
            previous = value;
            counter.add_multi_valued(value);
        }
        if counter.end_multi_valued_doc() {
            total_count += 1;
        }
    }
    let counts = spread_counts(ranges.len(), &slots, counter.into_counts());
    Ok(RangeCounts {
        counts: label_counts(ranges, counts),
        total_count,
    })
}

fn label_counts(ranges: &[NumericRange], counts: Vec<u64>) -> Vec<(String, u64)> {
    ranges
        .iter()
        .zip(counts)
        .map(|(range, count)| (range.label.clone(), count))
        .collect()
}

#[cfg(test)]
mod tests;
