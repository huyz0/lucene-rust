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
//! elementwise conflates them. Until c12 this module had no answer to that and
//! its doc told callers to merge per-segment results by resolved *label*
//! instead. [`crate::ordinal_map::OrdinalMap`] removes the workaround: build
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

use lucene_codecs::doc_values::{self, NumericEntry, SortedNumericEntry};
use lucene_codecs::terms_dict::{self, TermsDictEntry};

use crate::Result;

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
/// other function in [`crate::doc_value_query`] takes it. Returns counts
/// indexed by ordinal (`result[ord] == count for ordinal ord`).
pub fn facet_counts(
    doc_values_data: &[u8],
    entry: &SortedNumericEntry,
    terms: &TermsDictEntry,
    matching_docs: &[i32],
) -> Result<Vec<u64>> {
    let mut counts = vec![0u64; terms.terms_dict_size as usize];
    for &doc_id in matching_docs {
        let ords = doc_values::sorted_numeric_values(doc_values_data, entry, doc_id)?;
        for ord in ords {
            if let Some(slot) = counts.get_mut(ord as usize) {
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
    let labels = terms_dict::decode_all_terms(doc_values_data, terms)
        .map_err(lucene_codecs::doc_values::Error::from)?;
    Ok(labels
        .into_iter()
        .zip(counts.iter().copied())
        .enumerate()
        .map(|(ord, (label_bytes, count))| FacetCount {
            ord: ord as i64,
            label: String::from_utf8_lossy(&label_bytes).into_owned(),
            count,
        })
        .collect())
}

/// Real `Facets.getTopChildren`'s return shape (`FacetResult`), for the
/// SORTED_SET string-facet case: the top-`n` children plus the two aggregate
/// numbers Java reports alongside them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FacetResult {
    /// `FacetResult.dim` — the dimension these children live under. Empty for
    /// the dim-less primitives [`top_children`]/[`all_children`], which count
    /// one flat field's ordinals with no [`FacetsState`] above them; set by
    /// every [`SortedSetFacetCounts`] method.
    pub dim: String,
    /// `FacetResult.path` — the path within [`Self::dim`] whose children these
    /// are. Always empty for a non-hierarchical dim.
    pub path: Vec<String>,
    /// `FacetResult.labelValues` — the top children, count descending, ties
    /// broken by ascending ordinal. Only children with a **non-zero** count
    /// ever appear (`AbstractSortedSetDocValueFacetCounts.computeTopChildren`
    /// skips `count == 0` entirely).
    ///
    /// `FacetCount::label` is the child's **last** path component
    /// (`stringToPath(term)[parts.length - 1]`) when the result came from a
    /// [`SortedSetFacetCounts`], and the raw term when it came from
    /// [`resolve_labels`] + [`top_children`].
    pub label_values: Vec<FacetCount>,
    /// `FacetResult.value` — the sum of **every** non-zero child's count, not
    /// just the ones that fit in the top `n` (Java accumulates `pathCount`
    /// over the full child iteration, before the priority queue truncates).
    ///
    /// **`-1` is a real Lucene value, not an error.** For a dim configured
    /// multi-valued *without* `requireDimCount`, summing children would
    /// double-count a document carrying two values of the same dim and no
    /// accurate count is obtainable, so `adjustPathCountIfNecessary` returns
    /// `-1`. For a hierarchical dim, or a multi-valued one with
    /// `requireDimCount`, the dim/path ordinal was itself indexed and its own
    /// count is reported instead of the sum. Which of the three applies is
    /// decided by [`DimConfig`] — see [`SortedSetFacetCounts`].
    ///
    /// The dim-less [`top_children`]/[`all_children`] have no config to
    /// consult and always report the plain sum.
    pub value: i64,
    /// `FacetResult.childCount` — how many children had a non-zero count,
    /// which can exceed `label_values.len()` when `n` truncated the list.
    pub child_count: usize,
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
    let child_count = non_zero.len();
    let value: i64 = non_zero.iter().map(|f| f.count as i64).sum();
    non_zero.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.ord.cmp(&b.ord)));
    non_zero.truncate(n);
    Some(FacetResult {
        dim: String::new(),
        path: Vec::new(),
        label_values: non_zero,
        value,
        child_count,
    })
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
/// `createFacetResult` nulls. [`SortedSetFacetCounts::all_children`], which
/// *does* have a dim and a state, follows Java and returns `Some` with an
/// empty child list.
pub fn all_children(facets: Vec<FacetCount>) -> Option<FacetResult> {
    let non_zero: Vec<FacetCount> = facets.into_iter().filter(|f| f.count > 0).collect();
    if non_zero.is_empty() {
        return None;
    }
    let child_count = non_zero.len();
    let value: i64 = non_zero.iter().map(|f| f.count as i64).sum();
    Some(FacetResult {
        dim: String::new(),
        path: Vec::new(),
        label_values: non_zero,
        value,
        child_count,
    })
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
/// [`crate::doc_value_query`]'s numeric functions already assume for any
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

    /// Whether `value` falls inside this range, honoring each bound's own
    /// inclusive/exclusive flag independently.
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
pub fn sortable_double_bits(bits: i64) -> i64 {
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

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
/// legitimate for a sparse NUMERIC field, see [`crate::doc_value_query`]'s
/// module doc) contributes to **no** range, including an unbounded-looking
/// one like `i64::MIN..=i64::MAX` — the same missing-value rule
/// [`crate::doc_value_query::search_numeric_range`] already documents,
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
/// [`FacetsConfig::build_sorted_set_facet_fields`], the
/// `SortedSetDocValuesFacetField` half of `FacetsConfig.build(Document)`.
///
/// **Not ported**: the taxonomy half (`processFacetFields`, which needs a
/// `TaxonomyWriter` to turn a path into an ordinal, and a taxonomy index this
/// port does not have) and `AssociationFacetField` (which needs an
/// association-reading facet source this port does not have either). Both are
/// scope, not oversight -- this port's faceting is SSDV-only end to end, as
/// this module's own doc comment says.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FacetsConfig {
    dims: std::collections::BTreeMap<String, DimConfig>,
}

impl FacetsConfig {
    /// An empty config: every dim gets `DEFAULT_DIM_CONFIG`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `FacetsConfig.getDimConfig(dimName)`: the dim's configuration, or the
    /// default when it was never configured.
    pub fn dim_config(&self, dim: &str) -> DimConfig {
        self.dims.get(dim).cloned().unwrap_or_default()
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
    /// the `dim/child` prefix, and [`dim_count`] reads the bare-dim ordinal --
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
    ///   indexes the **bare dimension**, which is the ordinal [`dim_count`]
    ///   reads;
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
/// SORTED_SET terms, or the merged dictionary an [`crate::ordinal_map::OrdinalMap`]
/// defines across segments. It parses every term into its `dim` plus path
/// (see [`string_to_path`]) and records, per dim, either the contiguous
/// [`OrdRange`] its ordinals occupy (flat dims) or the [`DimTree`] linking
/// each path to its siblings and children (hierarchical dims).
///
/// This is what [`FacetResult`]'s `dim`/`path` and [`SortedSetFacetCounts`]'s
/// `getAllDims`/`getSpecificValue` need and what
/// b14 recorded as "genuinely one layer up".
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
            let components = string_to_path(&state.labels[ord as usize]);
            let dim = components[0].clone();
            ord = if state.config.dim_config(&dim).hierarchical {
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
        let mut next_components = string_to_path(&self.labels[dim_end_ord as usize]);
        if next_components.len() > 2 {
            return Err(FacetsStateError::NotHierarchical {
                path: next_components,
                term: self.labels[dim_end_ord as usize].clone(),
            });
        }
        // In-bounds: `FacetsState::new` rejected every empty label up front.
        let dim = next_components[0].clone();
        loop {
            let components = next_components;
            if dim_end_ord + 1 == value_count {
                break;
            }
            next_components = string_to_path(&self.labels[(dim_end_ord + 1) as usize]);
            if next_components[0] != components[0] {
                break;
            }
            if next_components.len() != 2 {
                return Err(FacetsStateError::NotHierarchical {
                    path: next_components,
                    term: self.labels[(dim_end_ord + 1) as usize].clone(),
                });
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
        let mut next_components = string_to_path(&self.labels[dim_end_ord as usize]);
        // In-bounds: `FacetsState::new` rejected every empty label up front.
        let dim = next_components[0].clone();

        loop {
            let components = next_components;
            let ord = (dim_end_ord - dim_start_ord) as usize;

            while let Some(&(stacked_ord, stacked_len)) = sibling_stack.last() {
                if stacked_len < components.len() {
                    break;
                }
                sibling_stack.pop();
                if stacked_len == components.len() {
                    siblings[stacked_ord] = ord as i32;
                }
            }

            if dim_end_ord + 1 == value_count {
                siblings.push(INVALID_ORDINAL);
                has_children.push(false);
                break;
            }

            next_components = string_to_path(&self.labels[(dim_end_ord + 1) as usize]);
            if next_components[0] != components[0] {
                siblings.push(INVALID_ORDINAL);
                has_children.push(false);
                break;
            }

            match components.len().cmp(&next_components.len()) {
                std::cmp::Ordering::Less => {
                    // The next ordinal is a direct child: every ancestor path
                    // is indexed, so a deeper next term is always exactly one
                    // level deeper.
                    has_children.push(true);
                    sibling_stack.push((ord, components.len()));
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
    counts: Vec<u64>,
}

/// `AbstractSortedSetDocValueFacetCounts.TopChildrenForPath` -- the
/// intermediate result before ordinals are resolved to labels.
struct TopChildrenForPath {
    path_count: i64,
    child_count: usize,
    top: Vec<(i64, u64)>,
}

impl<'a> SortedSetFacetCounts<'a> {
    /// `counts` is indexed by the state's ordinals. An **empty** `counts` is
    /// Java's `hasCounts() == false` (its `counts` array is allocated lazily
    /// and stays `null` when nothing matched): every accessor then reports
    /// "no results" rather than zeros.
    pub fn new(state: &'a FacetsState, counts: Vec<u64>) -> Self {
        SortedSetFacetCounts { state, counts }
    }

    /// `AbstractSortedSetDocValueFacetCounts.hasCounts()`.
    fn has_counts(&self) -> bool {
        !self.counts.is_empty()
    }

    /// `AbstractSortedSetDocValueFacetCounts.getCount(ord)`.
    fn count(&self, ord: i64) -> u64 {
        usize::try_from(ord)
            .ok()
            .and_then(|i| self.counts.get(i))
            .copied()
            .unwrap_or(0)
    }

    /// `prepareChildIteration(dim, dimConfig, path)`: the path's own ordinal
    /// plus its immediate children, or `None` when the path was never indexed.
    ///
    /// # Panics
    ///
    /// With Java's `IllegalArgumentException` message when `path` is non-empty
    /// for a dim that is not configured hierarchical -- a caller bug, exactly
    /// as in Java.
    fn prepare_child_iteration(
        &self,
        dim: &str,
        dim_config: &DimConfig,
        path: &[&str],
    ) -> Option<(i64, ChildOrds<'a>)> {
        if dim_config.hierarchical {
            let tree = self.state.dim_tree(dim)?;
            let path_ord = if path.is_empty() {
                tree.dim_start_ord
            } else {
                self.state.lookup_term(&path_to_string(dim, path).ok()?)?
            };
            Some((path_ord, ChildOrds::Tree(tree.children(path_ord))))
        } else {
            assert!(
                path.is_empty(),
                "Field is not configured as hierarchical, path should be 0 length"
            );
            let range = *self.state.ord_range(dim)?;
            let mut start = range.start;
            if dim_config.multi_valued && dim_config.require_dim_count {
                // The dim's own ordinal was indexed and leads the range; skip
                // past it so the iteration starts on the first real child.
                start += 1;
            }
            Some((
                range.start,
                ChildOrds::Range {
                    next: start,
                    end: range.end,
                },
            ))
        }
    }

    /// `adjustPathCountIfNecessary` -- `-1` is Java's "no accurate count is
    /// obtainable for this dim", not a sentinel this port invented.
    fn adjust_path_count(&self, dim_config: &DimConfig, path_ord: i64, computed: i64) -> i64 {
        if dim_config.hierarchical || (dim_config.multi_valued && dim_config.require_dim_count) {
            self.count(path_ord) as i64
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
        let mut path_count: i64 = 0;
        let mut top: Vec<(i64, u64)> = Vec::new();
        for ord in children {
            let count = self.count(ord);
            if count > 0 {
                path_count += count as i64;
                top.push((ord, count));
            }
        }
        let child_count = top.len();
        top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        top.truncate(top_n);
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
        Some(FacetResult {
            dim: dim.to_string(),
            path: path.iter().map(|s| s.to_string()).collect(),
            label_values: top
                .top
                .into_iter()
                .map(|(ord, count)| FacetCount {
                    ord,
                    label: self.leaf_label(ord),
                    count,
                })
                .collect(),
            value: top.path_count,
            child_count: top.child_count,
        })
    }

    /// `FacetsConfig.stringToPath(term)[parts.length - 1]`.
    fn leaf_label(&self, ord: i64) -> String {
        self.state
            .label(ord)
            .map(|term| string_to_path(term).pop().unwrap_or_default())
            .unwrap_or_default()
    }

    /// `Facets.getTopChildren(topN, dim, path...)`.
    ///
    /// # Panics
    ///
    /// `Facets.validateTopN`'s message when `top_n == 0`; see
    /// [`top_children`].
    pub fn top_children(&self, top_n: usize, dim: &str, path: &[&str]) -> Option<FacetResult> {
        assert!(top_n > 0, "topN must be > 0 (got: {top_n})");
        if !self.has_counts() {
            return None;
        }
        let dim_config = self.state.config().dim_config(dim);
        let (path_ord, children) = self.prepare_child_iteration(dim, &dim_config, path)?;
        let top = self.compute_top_children(children, top_n, &dim_config, path_ord);
        self.create_facet_result(Some(top), dim, path)
    }

    /// `Facets.getAllChildren(dim, path...)`: every non-zero child in
    /// **ordinal** order (no priority queue at all), plus the same
    /// `value`/`childCount` aggregates.
    pub fn all_children(&self, dim: &str, path: &[&str]) -> Option<FacetResult> {
        let dim_config = self.state.config().dim_config(dim);
        let (path_ord, children) = self.prepare_child_iteration(dim, &dim_config, path)?;
        if !self.has_counts() {
            return None;
        }
        let mut path_count: i64 = 0;
        let mut label_values = Vec::new();
        for ord in children {
            let count = self.count(ord);
            if count > 0 {
                path_count += count as i64;
                label_values.push(FacetCount {
                    ord,
                    label: self.leaf_label(ord),
                    count,
                });
            }
        }
        let child_count = label_values.len();
        Some(FacetResult {
            dim: dim.to_string(),
            path: path.iter().map(|s| s.to_string()).collect(),
            label_values,
            value: self.adjust_path_count(&dim_config, path_ord, path_count),
            child_count,
        })
    }

    /// `Facets.getSpecificValue(dim, path...)`: the count for one exact path,
    /// or `-1` when the path was never indexed. `0` when nothing was counted
    /// at all (Java's `hasCounts() == false` branch).
    ///
    /// # Panics
    ///
    /// Java's `IllegalArgumentException` -- `"<dim> is not configured as
    /// hierarchical, path must be length=1"` -- for a non-hierarchical dim
    /// addressed with anything but a single path component.
    pub fn specific_value(&self, dim: &str, path: &[&str]) -> i64 {
        let dim_config = self.state.config().dim_config(dim);
        assert!(
            dim_config.hierarchical || path.len() == 1,
            "{dim} is not configured as hierarchical, path must be length=1"
        );
        let Ok(term) = path_to_string(dim, path) else {
            return -1;
        };
        let Some(ord) = self.state.lookup_term(&term) else {
            return -1;
        };
        if self.has_counts() {
            self.count(ord) as i64
        } else {
            0
        }
    }

    /// `AbstractSortedSetDocValueFacetCounts.getAllDims(topN)`: every dim that
    /// had a hit, each with its top `topN` children, sorted by `value`
    /// descending with ties broken by ascending dim name.
    ///
    /// # Panics
    ///
    /// `Facets.validateTopN`'s message when `top_n == 0`.
    pub fn all_dims(&self, top_n: usize) -> Vec<FacetResult> {
        assert!(top_n > 0, "topN must be > 0 (got: {top_n})");
        if !self.has_counts() {
            return Vec::new();
        }
        let mut results: Vec<FacetResult> = self
            .state
            .dims()
            .filter_map(|dim| self.top_children(top_n, dim, &[]))
            .collect();
        results.sort_by(|a, b| b.value.cmp(&a.value).then_with(|| a.dim.cmp(&b.dim)));
        results
    }

    /// `Facets.getTopDims(topNDims, topNChildren)`.
    ///
    /// This is the base-class contract verbatim -- "results should be the same
    /// as calling getAllDims and then only using the first topNDims" -- rather
    /// than `AbstractSortedSetDocValueFacetCounts`' override of it. That
    /// override exists to avoid computing children for dims that will not make
    /// the cut, by reading a dim's count directly where the encoding allows;
    /// here [`Self::all_dims`] is already a single pass over the dims with no
    /// per-dim I/O, so the optimization has nothing to buy and the two agree
    /// by construction.
    ///
    /// # Panics
    ///
    /// `Facets.validateTopN`'s message when either argument is 0.
    pub fn top_dims(&self, top_n_dims: usize, top_n_children: usize) -> Vec<FacetResult> {
        assert!(top_n_dims > 0, "topN must be > 0 (got: {top_n_dims})");
        let mut all = self.all_dims(top_n_children);
        all.truncate(top_n_dims);
        all
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
    map: &crate::ordinal_map::OrdinalMap,
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

fn count_single_valued(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    ranges: &[NumericRange],
    matching_docs: &[i32],
    map_doc_value: fn(i64) -> i64,
) -> Result<RangeCounts> {
    let mut counts = vec![0u64; ranges.len()];
    let mut total_count = 0u64;
    for &doc_id in matching_docs {
        if let Some(raw) = doc_values::numeric_value(doc_values_data, entry, doc_id)? {
            let value = map_doc_value(raw);
            let mut matched = false;
            for (range, count) in ranges.iter().zip(counts.iter_mut()) {
                if range.contains(value) {
                    *count += 1;
                    matched = true;
                }
            }
            if matched {
                total_count += 1;
            }
        }
    }
    Ok(RangeCounts {
        counts: label_counts(ranges, counts),
        total_count,
    })
}

/// `RangeFacetCounts.count`'s **multi-valued** branch: the SORTED_NUMERIC
/// sibling of [`range_facet_counts`], which b14 recorded as missing.
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
    let mut counts = vec![0u64; ranges.len()];
    let mut total_count = 0u64;
    // Reused across documents: one flag per range, "this doc already counted
    // here". Java's per-elementary-interval bit set, at range granularity.
    let mut hit = vec![false; ranges.len()];
    for &doc_id in matching_docs {
        let values = doc_values::sorted_numeric_values(doc_values_data, entry, doc_id)?;
        if values.is_empty() {
            continue;
        }
        hit.iter_mut().for_each(|h| *h = false);
        let mut previous = 0i64;
        for (j, raw) in values.into_iter().enumerate() {
            let value = map_doc_value(raw);
            if j != 0 && value == previous {
                continue;
            }
            previous = value;
            for (i, range) in ranges.iter().enumerate() {
                if !hit[i] && range.contains(value) {
                    hit[i] = true;
                    counts[i] += 1;
                }
            }
        }
        if hit.iter().any(|&h| h) {
            total_count += 1;
        }
    }
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
mod tests {
    use super::*;
    use lucene_codecs::doc_values::{DocValuesMeta, SortedSetKind};

    fn multi_dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/multi_valued_dv_index/"
        )
        .to_string()
    }

    struct Manifest {
        kv: Vec<(String, String)>,
    }

    impl Manifest {
        fn load(dir: &str) -> Self {
            let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
                .expect("run fixtures generator first");
            let kv = text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            Manifest { kv }
        }

        fn get(&self, key: &str) -> &str {
            self.kv
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
        }
    }

    fn id_from_hex(hex: &str) -> [u8; 16] {
        let mut id = [0u8; 16];
        for i in 0..16 {
            id[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        id
    }

    fn dv_suffix(manifest: &Manifest) -> String {
        let segment_name = manifest.get("segment_name");
        let name = manifest.get("dvm_file_name");
        name.strip_prefix(&format!("{segment_name}_"))
            .and_then(|s| s.strip_suffix(".dvm"))
            .unwrap()
            .to_string()
    }

    fn field_number(manifest: &Manifest, field: &str) -> i32 {
        manifest
            .get("field_numbers")
            .split(',')
            .find_map(|kv| {
                let (name, num) = kv.split_once(':').unwrap();
                (name == field).then(|| num.parse().unwrap())
            })
            .unwrap_or_else(|| panic!("field {field} missing from field_numbers"))
    }

    fn load_dv_meta(dir: &str) -> (Manifest, Vec<u8>, DocValuesMeta) {
        let manifest = Manifest::load(dir);
        let id = id_from_hex(manifest.get("id_hex"));
        let fnm = std::fs::read(format!("{dir}{}.raw", manifest.get("fnm_file_name"))).unwrap();
        let fis = lucene_codecs::field_infos::parse(&fnm, &id, "").unwrap();
        let meta_buf =
            std::fs::read(format!("{dir}{}.raw", manifest.get("dvm_file_name"))).unwrap();
        let data_buf =
            std::fs::read(format!("{dir}{}.raw", manifest.get("dvd_file_name"))).unwrap();
        let suffix = dv_suffix(&manifest);
        let (_, parsed) = doc_values::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
        (manifest, data_buf, parsed)
    }

    /// Ground truth taken directly from the manifest, which real Lucene wrote
    /// via a straightforward per-doc `SortedSetDocValues.nextOrd()` iteration
    /// (see `GenMultiValuedDocValues.java`'s `field.tags.ords`/`.terms`
    /// output) -- an honest differential check without depending on the
    /// `lucene-facet` module (not a project dependency), per this task's
    /// brief: "a straightforward manual per-doc iteration ... is a
    /// reasonable, real ground truth even without the facet module
    /// specifically."
    fn expected_counts_from_manifest(manifest: &Manifest, num_terms: usize) -> Vec<u64> {
        let mut counts = vec![0u64; num_terms];
        for doc_ords in manifest.get("field.tags.ords").split(';') {
            if doc_ords == "NONE" {
                continue;
            }
            for ord in doc_ords.split(',') {
                counts[ord.parse::<usize>().unwrap()] += 1;
            }
        }
        counts
    }

    fn tags_entry(
        meta: &DocValuesMeta,
        field_number: i32,
    ) -> (&SortedNumericEntry, &TermsDictEntry) {
        let entry = meta.sorted_set_entry(field_number).unwrap();
        match &entry.kind {
            SortedSetKind::Multi { ords, terms } => (ords, terms),
            SortedSetKind::Single(_) => panic!("expected a multi-valued SORTED_SET"),
        }
    }

    // --- `FacetsConfig` and the dim/path encoding ---

    #[test]
    fn dim_config_defaults_are_javas_default_dim_config() {
        let config = FacetsConfig::new();
        let d = config.dim_config("anything");
        assert!(!d.hierarchical && !d.multi_valued && !d.require_dim_count);
        assert_eq!(d.index_field_name, DEFAULT_INDEX_FIELD_NAME);
        assert_eq!(DEFAULT_INDEX_FIELD_NAME, "$facets");
        assert!(!config.is_dim_configured("anything"));
    }

    #[test]
    fn setters_are_independent_and_visible_through_dim_configs() {
        let mut config = FacetsConfig::new();
        config
            .set_hierarchical("a", true)
            .set_multi_valued("b", true)
            .set_require_dim_count("b", true)
            .set_index_field_name("c", "$other");
        assert!(config.dim_config("a").hierarchical);
        assert!(!config.dim_config("a").multi_valued);
        assert!(config.dim_config("b").multi_valued && config.dim_config("b").require_dim_count);
        assert_eq!(config.dim_config("c").index_field_name, "$other");
        assert!(config.is_dim_configured("a"));
        let names: Vec<&str> = config.dim_configs().map(|(n, _)| n).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn path_encoding_round_trips_and_escapes_the_delimiters() {
        assert_eq!(DELIM_CHAR, '\u{1F}');
        assert_eq!(ESCAPE_CHAR, '\u{1E}');
        assert_eq!(path_to_string("dim", &[]).unwrap(), "dim");
        assert_eq!(
            path_to_string("dim", &["a", "b"]).unwrap(),
            format!("dim{DELIM_CHAR}a{DELIM_CHAR}b")
        );
        assert_eq!(
            string_to_path(&path_to_string("dim", &["a", "b"]).unwrap()),
            vec!["dim", "a", "b"]
        );
        // A label containing the delimiter or the escape character survives.
        for label in [
            format!("x{DELIM_CHAR}y"),
            format!("x{ESCAPE_CHAR}y"),
            format!("{ESCAPE_CHAR}{DELIM_CHAR}"),
        ] {
            let encoded = path_to_string("dim", &[label.as_str()]).unwrap();
            assert_eq!(string_to_path(&encoded), vec!["dim".to_string(), label]);
        }
        // Java's `length == 0` early return, and its empty-component rejection.
        assert_eq!(path_components_to_string(&[]).unwrap(), "");
        assert!(string_to_path("").is_empty());
        assert_eq!(path_to_string("dim", &[""]), Err(EmptyPathComponent));
        assert_eq!(path_components_to_string(&[""]), Err(EmptyPathComponent));
        // Java asserts `!lastEscape` and otherwise ignores a trailing escape.
        assert_eq!(string_to_path(&format!("a{ESCAPE_CHAR}")), vec!["a"]);
    }

    // --- `SortedSetDocValuesReaderState` ---

    fn label(dim: &str, path: &[&str]) -> String {
        path_to_string(dim, path).unwrap()
    }

    fn flat_state() -> FacetsState {
        FacetsState::new(
            vec![
                label("Author", &["Bob"]),
                label("Author", &["Lisa"]),
                label("Tag", &["x"]),
            ],
            FacetsConfig::new(),
        )
        .unwrap()
    }

    #[test]
    fn a_flat_state_records_one_contiguous_ord_range_per_dim() {
        let state = flat_state();
        assert_eq!(state.size(), 3);
        assert_eq!(
            state.ord_range("Author"),
            Some(&OrdRange { start: 0, end: 1 })
        );
        assert_eq!(state.ord_range("Tag"), Some(&OrdRange { start: 2, end: 2 }));
        assert_eq!(state.ord_range("Nope"), None);
        assert_eq!(state.dim_tree("Author"), None);
        assert_eq!(state.dims().collect::<Vec<_>>(), vec!["Author", "Tag"]);
        assert_eq!(state.label(1), Some(label("Author", &["Lisa"]).as_str()));
        assert_eq!(state.label(99), None);
        assert_eq!(state.label(-1), None);
        assert_eq!(state.lookup_term(&label("Tag", &["x"])), Some(2));
        assert_eq!(state.lookup_term("nope"), None);
    }

    #[test]
    fn a_deeper_path_under_a_flat_dim_is_rejected_like_javas_constructor() {
        // `createOneFlatFacetDimState` throws
        // "dimension not configured to handle hierarchical field".
        let err =
            FacetsState::new(vec![label("Path", &["a", "b"])], FacetsConfig::new()).unwrap_err();
        assert!(matches!(err, FacetsStateError::NotHierarchical { .. }));
        assert!(err
            .to_string()
            .contains("not configured to handle hierarchical"));

        // The same rejection from the *second* term onward, which is a
        // different branch in Java.
        let err = FacetsState::new(
            vec![label("Path", &["a"]), label("Path", &["a", "b"])],
            FacetsConfig::new(),
        )
        .unwrap_err();
        assert!(matches!(err, FacetsStateError::NotHierarchical { .. }));
    }

    #[test]
    fn an_empty_label_is_a_typed_error_not_a_panic() {
        // Java's constructor does `stringToPath(term)[0]` unguarded and throws
        // `ArrayIndexOutOfBoundsException`; a facet dictionary written by
        // `FacetsConfig.build` never contains an empty term, but `new` takes a
        // caller-supplied list.
        let err = FacetsState::new(
            vec![label("Author", &["Bob"]), String::new()],
            FacetsConfig::new(),
        )
        .unwrap_err();
        assert_eq!(err, FacetsStateError::EmptyLabel { ord: 1 });
        assert!(err.to_string().contains("names no dimension"));
        // Including as the very first ordinal, and for a hierarchical dim.
        let mut config = FacetsConfig::new();
        config.set_hierarchical("Path", true);
        assert_eq!(
            FacetsState::new(vec![String::new(), label("Path", &["a"])], config).unwrap_err(),
            FacetsStateError::EmptyLabel { ord: 0 }
        );
        // An empty dictionary is fine -- it is just a state with no dims.
        let empty = FacetsState::new(Vec::new(), FacetsConfig::new()).unwrap();
        assert_eq!(empty.size(), 0);
        assert_eq!(empty.dims().count(), 0);
    }

    fn hier_state() -> FacetsState {
        let mut config = FacetsConfig::new();
        config.set_hierarchical("Path", true);
        FacetsState::new(
            vec![
                label("Path", &[]),
                label("Path", &["a"]),
                label("Path", &["a", "b"]),
                label("Path", &["a", "c"]),
                label("Path", &["d"]),
                label("Path", &["d", "e"]),
            ],
            config,
        )
        .unwrap()
    }

    #[test]
    fn a_dim_tree_links_children_and_siblings() {
        let state = hier_state();
        let tree = state.dim_tree("Path").expect("a hierarchical dim");
        assert_eq!(tree.dim_start_ord, 0);
        // The dim's own children are `a` (1) and `d` (4).
        assert_eq!(tree.children(0).collect::<Vec<_>>(), vec![1, 4]);
        // `a`'s children are `a/b` (2) and `a/c` (3).
        assert_eq!(tree.children(1).collect::<Vec<_>>(), vec![2, 3]);
        // `d`'s only child is `d/e` (5).
        assert_eq!(tree.children(4).collect::<Vec<_>>(), vec![5]);
        // A leaf, and an ordinal outside the dim, both iterate empty.
        assert!(tree.children(2).next().is_none());
        assert!(tree.children(99).next().is_none());
        assert!(tree.children(-5).next().is_none());
        assert_eq!(
            state.ord_range("Path"),
            None,
            "a hierarchical dim has no OrdRange"
        );
        assert_eq!(state.dims().collect::<Vec<_>>(), vec!["Path"]);
    }

    #[test]
    fn hierarchical_and_flat_dims_coexist_with_the_tree_dims_first() {
        let mut config = FacetsConfig::new();
        config.set_hierarchical("Path", true);
        let state = FacetsState::new(
            vec![
                label("Author", &["Bob"]),
                label("Path", &[]),
                label("Path", &["a"]),
                label("Zed", &["q"]),
            ],
            config,
        )
        .unwrap();
        assert_eq!(
            state.dims().collect::<Vec<_>>(),
            vec!["Path", "Author", "Zed"],
            "hierarchical dims first, then flat, each ascending"
        );
    }

    // --- `SortedSetDocValuesFacetCounts` over a hand-built state ---

    #[test]
    fn no_counts_at_all_is_javas_has_counts_false() {
        let state = flat_state();
        let facets = SortedSetFacetCounts::new(&state, Vec::new());
        assert!(facets.top_children(10, "Author", &[]).is_none());
        assert!(facets.all_children("Author", &[]).is_none());
        assert!(facets.all_dims(10).is_empty());
        assert!(facets.top_dims(10, 10).is_empty());
        assert_eq!(
            facets.specific_value("Author", &["Bob"]),
            0,
            "an indexed path with no counting done is 0, not -1"
        );
        assert_eq!(facets.specific_value("Author", &["Nobody"]), -1);
    }

    #[test]
    fn a_dim_whose_children_all_counted_zero_is_absent() {
        let state = flat_state();
        let facets = SortedSetFacetCounts::new(&state, vec![0, 0, 3]);
        assert!(facets.top_children(10, "Author", &[]).is_none());
        let tag = facets.top_children(10, "Tag", &[]).unwrap();
        assert_eq!(tag.dim, "Tag");
        assert_eq!(tag.value, 3);
        assert_eq!(tag.child_count, 1);
        // `getAllChildren` still returns a (child-less) result for the empty
        // dim -- it has no `childCount == 0 -> null` guard, unlike
        // `createFacetResult`.
        let author = facets.all_children("Author", &[]).unwrap();
        assert!(author.label_values.is_empty());
        assert_eq!(author.value, 0);
    }

    #[test]
    fn a_multi_valued_dim_with_require_dim_count_reads_the_dim_ordinal() {
        // `Publish Year` is indexed with its own ordinal leading the range.
        let mut config = FacetsConfig::new();
        config
            .set_multi_valued("Year", true)
            .set_require_dim_count("Year", true);
        let state = FacetsState::new(
            vec![
                label("Year", &[]),
                label("Year", &["1999"]),
                label("Year", &["2000"]),
            ],
            config,
        )
        .unwrap();
        // 5 documents carry the dim; their per-year counts sum to 7 because
        // documents carry more than one year.
        let facets = SortedSetFacetCounts::new(&state, vec![5, 4, 3]);
        let r = facets.top_children(10, "Year", &[]).unwrap();
        assert_eq!(
            r.value, 5,
            "the dim's own count, not the 7 its children sum to"
        );
        assert_eq!(
            r.child_count, 2,
            "the dim ordinal is not one of its children"
        );
        assert_eq!(
            r.label_values
                .iter()
                .map(|lv| lv.label.as_str())
                .collect::<Vec<_>>(),
            vec!["1999", "2000"]
        );
    }

    #[test]
    fn a_multi_valued_dim_without_require_dim_count_is_minus_one() {
        let mut config = FacetsConfig::new();
        config.set_multi_valued("Tag", true);
        let state =
            FacetsState::new(vec![label("Tag", &["x"]), label("Tag", &["y"])], config).unwrap();
        let facets = SortedSetFacetCounts::new(&state, vec![2, 1]);
        assert_eq!(facets.top_children(10, "Tag", &[]).unwrap().value, -1);
        assert_eq!(facets.all_children("Tag", &[]).unwrap().value, -1);
    }

    #[test]
    #[should_panic(expected = "topN must be > 0 (got: 0)")]
    fn top_children_rejects_a_zero_top_n() {
        let state = flat_state();
        SortedSetFacetCounts::new(&state, vec![1, 1, 1]).top_children(0, "Author", &[]);
    }

    #[test]
    #[should_panic(expected = "topN must be > 0 (got: 0)")]
    fn all_dims_rejects_a_zero_top_n() {
        let state = flat_state();
        SortedSetFacetCounts::new(&state, vec![1, 1, 1]).all_dims(0);
    }

    #[test]
    #[should_panic(expected = "topN must be > 0 (got: 0)")]
    fn top_dims_rejects_a_zero_top_n() {
        let state = flat_state();
        SortedSetFacetCounts::new(&state, vec![1, 1, 1]).top_dims(0, 5);
    }

    #[test]
    #[should_panic(expected = "path should be 0 length")]
    fn a_path_under_a_flat_dim_is_a_caller_bug() {
        let state = flat_state();
        SortedSetFacetCounts::new(&state, vec![1, 1, 1]).top_children(10, "Author", &["Bob"]);
    }

    #[test]
    #[should_panic(expected = "is not configured as hierarchical, path must be length=1")]
    fn specific_value_on_a_flat_dim_needs_exactly_one_component() {
        let state = flat_state();
        SortedSetFacetCounts::new(&state, vec![1, 1, 1]).specific_value("Author", &["a", "b"]);
    }

    #[test]
    fn specific_value_rejects_an_unencodable_path_as_not_found() {
        let mut config = FacetsConfig::new();
        config.set_hierarchical("Path", true);
        let state = FacetsState::new(vec![label("Path", &[])], config).unwrap();
        let facets = SortedSetFacetCounts::new(&state, vec![1]);
        // An empty component cannot be encoded, so it names no ordinal.
        assert_eq!(facets.specific_value("Path", &[""]), -1);
    }

    #[test]
    fn hierarchical_children_are_counted_and_labelled_by_their_leaf_component() {
        let state = hier_state();
        //         Path  a  a/b  a/c  d  d/e
        let counts = vec![6, 4, 3, 1, 2, 2];
        let facets = SortedSetFacetCounts::new(&state, counts);
        let top = facets.top_children(10, "Path", &[]).unwrap();
        assert_eq!(top.value, 6, "a hierarchical dim reports its own ordinal");
        assert_eq!(
            top.label_values
                .iter()
                .map(|lv| (lv.label.as_str(), lv.count))
                .collect::<Vec<_>>(),
            vec![("a", 4), ("d", 2)]
        );
        let under_a = facets.all_children("Path", &["a"]).unwrap();
        assert_eq!(under_a.path, vec!["a"]);
        assert_eq!(under_a.value, 4);
        assert_eq!(
            under_a
                .label_values
                .iter()
                .map(|lv| lv.label.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "c"]
        );
        assert_eq!(facets.specific_value("Path", &["a", "b"]), 3);
        assert_eq!(facets.specific_value("Path", &["a", "zz"]), -1);
        assert!(facets.top_children(10, "Path", &["zz"]).is_none());
    }

    #[test]
    fn all_dims_orders_by_value_descending_then_dim_ascending() {
        let mut config = FacetsConfig::new();
        config.set_multi_valued("Tag", true);
        let state = FacetsState::new(
            vec![
                label("Author", &["Bob"]),
                label("Tag", &["x"]),
                label("Zed", &["q"]),
            ],
            config,
        )
        .unwrap();
        // Author 5, Tag -1 (multi-valued, no dim count), Zed 5.
        let facets = SortedSetFacetCounts::new(&state, vec![5, 9, 5]);
        let all = facets.all_dims(10);
        let dims: Vec<(&str, i64)> = all.iter().map(|r| (r.dim.as_str(), r.value)).collect();
        assert_eq!(dims, vec![("Author", 5), ("Zed", 5), ("Tag", -1)]);
        // `getTopDims` is `getAllDims` truncated.
        let top = facets.top_dims(2, 10);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].dim, "Author");
    }

    #[test]
    fn top_children_truncates_but_child_count_does_not() {
        let state = FacetsState::new(
            vec![label("D", &["a"]), label("D", &["b"]), label("D", &["c"])],
            FacetsConfig::new(),
        )
        .unwrap();
        let facets = SortedSetFacetCounts::new(&state, vec![1, 3, 2]);
        let r = facets.top_children(2, "D", &[]).unwrap();
        assert_eq!(r.child_count, 3);
        assert_eq!(r.value, 6, "value sums every non-zero child, not the top 2");
        assert_eq!(
            r.label_values
                .iter()
                .map(|lv| lv.label.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "c"]
        );
        // `getAllChildren` is ordinal order, not count order.
        assert_eq!(
            facets
                .all_children("D", &[])
                .unwrap()
                .label_values
                .iter()
                .map(|lv| lv.label.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
    }

    // --- multi-valued range counting ---

    fn nums_entry(meta: &DocValuesMeta, field_number: i32) -> &SortedNumericEntry {
        meta.sorted_numeric_entry(field_number)
            .expect("a SORTED_NUMERIC entry")
    }

    #[test]
    fn a_multi_valued_doc_is_counted_once_per_range_not_once_per_value() {
        // `nums` is `[5,10]; NONE; [7]; [1,2,3]; NONE` -- doc 3 has three
        // values inside `[1, 3]`.
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = nums_entry(&meta, field_number(&manifest, "nums"));
        let ranges = vec![
            NumericRange::new_long("low", 1, true, 3, true).unwrap(),
            NumericRange::new_long("high", 5, true, 10, true).unwrap(),
        ];
        let out = multi_valued_range_facet_counts(&data, entry, &ranges, &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(
            out.counts,
            vec![("low".to_string(), 1), ("high".to_string(), 2)]
        );
        assert_eq!(
            out.total_count, 3,
            "docs 0, 2 and 3 each landed in at least one range; docs 1 and 4 have no values"
        );
        // Summing the counts would give 3 as well here, so make the
        // distinction sharp with overlapping ranges.
        let overlapping = vec![
            NumericRange::new_long("a", 1, true, 10, true).unwrap(),
            NumericRange::new_long("b", 1, true, 10, true).unwrap(),
        ];
        let out =
            multi_valued_range_facet_counts(&data, entry, &overlapping, &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(out.counts.iter().map(|(_, c)| *c).sum::<u64>(), 6);
        assert_eq!(out.total_count, 3);
    }

    #[test]
    fn a_doc_matching_no_range_is_not_in_tot_count() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = nums_entry(&meta, field_number(&manifest, "nums"));
        let ranges = vec![NumericRange::new_long("none", 100, true, 200, true).unwrap()];
        let out = multi_valued_range_facet_counts(&data, entry, &ranges, &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(out.counts, vec![("none".to_string(), 0)]);
        assert_eq!(out.total_count, 0);
        // An empty matching set, and an empty range list, are both clean.
        let out = multi_valued_range_facet_counts(&data, entry, &ranges, &[]).unwrap();
        assert_eq!(out.total_count, 0);
        let out = multi_valued_range_facet_counts(&data, entry, &[], &[0, 1, 2]).unwrap();
        assert!(out.counts.is_empty());
        assert_eq!(out.total_count, 0);
    }

    #[test]
    fn the_multi_valued_double_variant_maps_every_stored_value() {
        // `nums`' raw values are plain integers, so reading them as raw
        // `doubleToRawLongBits` patterns is meaningless as *numbers* -- but
        // the mapping is what is under test: `sortable_double_bits` is an
        // involution, so a range built over the mapped values must count the
        // same documents the unmapped one does over the unmapped values.
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = nums_entry(&meta, field_number(&manifest, "nums"));
        let mapped = vec![NumericRange {
            label: "mapped".into(),
            min: sortable_double_bits(1),
            min_inclusive: true,
            max: sortable_double_bits(3),
            max_inclusive: true,
        }];
        let raw = vec![NumericRange::new_long("raw", 1, true, 3, true).unwrap()];
        let a = multi_valued_double_range_facet_counts(&data, entry, &mapped, &[0, 1, 2, 3, 4])
            .unwrap();
        let b = multi_valued_range_facet_counts(&data, entry, &raw, &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(a.counts[0].1, b.counts[0].1);
        assert_eq!(a.total_count, b.total_count);
    }

    #[test]
    fn the_single_valued_with_total_variants_agree_with_the_older_signatures() {
        let (manifest, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();
        let ranges = vec![
            NumericRange::new_long("neg", i64::MIN, true, 0, false).unwrap(),
            NumericRange::new_long("pos", 0, true, i64::MAX, true).unwrap(),
        ];
        let plain = range_facet_counts(&data, entry, &ranges, &[0, 1, 2, 3, 4]).unwrap();
        let with_total =
            range_facet_counts_with_total(&data, entry, &ranges, &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(plain, with_total.counts);
        assert_eq!(with_total.total_count, 5, "every doc has a value in range");

        let dplain = double_range_facet_counts(&data, entry, &ranges, &[0, 1, 2, 3, 4]).unwrap();
        let dwith =
            double_range_facet_counts_with_total(&data, entry, &ranges, &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(dplain, dwith.counts);
    }

    #[test]
    fn a_doc_with_no_value_never_reaches_tot_count_in_the_single_valued_path() {
        let (manifest, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "sparse"))
            .unwrap();
        let ranges = vec![NumericRange::new_long("all", i64::MIN, true, i64::MAX, true).unwrap()];
        let out = range_facet_counts_with_total(&data, entry, &ranges, &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(
            out.total_count, 3,
            "`sparse` is 5, NONE, 15, NONE, 25 -- three docs have a value"
        );
    }

    // --- `merge_segment_counts` ---

    #[test]
    fn merge_segment_counts_remaps_rather_than_summing_elementwise() {
        use crate::ordinal_map::OrdinalMap;
        let segments = vec![
            vec![b"bar".to_vec(), b"foo".to_vec()],
            vec![b"cat".to_vec(), b"dog".to_vec()],
        ];
        let map = OrdinalMap::build(&segments);
        // Segment 0 counts bar=2 foo=1; segment 1 counts cat=5 dog=7.
        let merged = merge_segment_counts(&map, &[vec![2, 1], vec![5, 7]]);
        // Global order is bar, cat, dog, foo.
        assert_eq!(merged, vec![2, 5, 7, 1]);
        // Elementwise summing would have produced [7, 8] -- wrong length and
        // wrong meaning.
        assert_ne!(merged.len(), 2);
    }

    #[test]
    fn merge_segment_counts_tolerates_missing_and_oversized_inputs() {
        use crate::ordinal_map::OrdinalMap;
        let map = OrdinalMap::build(&[vec![b"a".to_vec()]]);
        // A count array longer than the segment's ordinal list: the excess is
        // ignored rather than panicking.
        assert_eq!(merge_segment_counts(&map, &[vec![3, 9, 9]]), vec![3]);
        // More count arrays than segments: the extras are ignored.
        assert_eq!(merge_segment_counts(&map, &[vec![3], vec![4]]), vec![3]);
        // No counts at all.
        assert_eq!(merge_segment_counts(&map, &[]), vec![0]);
    }

    // --- `getTopChildren`/`getAllChildren` (`FacetResult`) semantics ---

    fn fc(ord: i64, label: &str, count: u64) -> FacetCount {
        FacetCount {
            ord,
            label: label.into(),
            count,
        }
    }

    #[test]
    fn top_children_drops_zero_count_children_unlike_top_n_facets() {
        let facets = vec![fc(0, "a", 3), fc(1, "b", 0), fc(2, "c", 5)];
        // `top_n_facets` pads with the zero-count child; `top_children`, like
        // real `computeTopChildren`, never reports it at all.
        assert_eq!(top_n_facets(facets.clone(), 3).len(), 3);
        let result = top_children(facets, 3).unwrap();
        assert_eq!(
            result
                .label_values
                .iter()
                .map(|f| f.label.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "a"]
        );
        assert_eq!(result.child_count, 2);
        assert_eq!(result.value, 8);
    }

    #[test]
    fn top_children_orders_count_desc_then_ordinal_asc() {
        // `TopOrdAndIntQueue.OrdAndInt.lessThan`: lower count is worse, and on
        // a tie the *higher* ordinal is worse -- so best-first is count DESC,
        // ordinal ASC.
        let facets = vec![fc(2, "z", 4), fc(0, "x", 4), fc(1, "y", 9)];
        let result = top_children(facets, 3).unwrap();
        assert_eq!(
            result
                .label_values
                .iter()
                .map(|f| f.ord)
                .collect::<Vec<_>>(),
            vec![1, 0, 2]
        );
    }

    #[test]
    fn top_children_child_count_counts_every_non_zero_child_not_just_the_top_n() {
        let facets = vec![fc(0, "a", 1), fc(1, "b", 2), fc(2, "c", 3)];
        let result = top_children(facets, 1).unwrap();
        assert_eq!(result.label_values.len(), 1);
        assert_eq!(result.child_count, 3);
        assert_eq!(result.value, 6);
    }

    #[test]
    fn dim_with_no_matching_values_is_absent_not_a_row_of_zeroes() {
        // Java's `createFacetResult` returns `null` when `childCount == 0`,
        // and `getAllDims` drops it -- the dim simply isn't reported.
        let facets = vec![fc(0, "a", 0), fc(1, "b", 0)];
        assert!(top_children(facets.clone(), 5).is_none());
        assert!(all_children(facets).is_none());
    }

    #[test]
    #[should_panic(expected = "topN must be > 0 (got: 0)")]
    fn top_children_rejects_a_zero_top_n_like_validate_top_n() {
        top_children(vec![fc(0, "a", 1)], 0);
    }

    #[test]
    fn all_children_keeps_ordinal_order_and_drops_zero_counts() {
        let facets = vec![fc(0, "a", 3), fc(1, "b", 0), fc(2, "c", 5)];
        let result = all_children(facets).unwrap();
        assert_eq!(
            result
                .label_values
                .iter()
                .map(|f| (f.ord, f.count))
                .collect::<Vec<_>>(),
            vec![(0, 3), (2, 5)]
        );
        assert_eq!(result.value, 8);
        assert_eq!(result.child_count, 2);
    }

    #[test]
    fn top_children_over_real_fixture_counts_reports_only_used_tags() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let field_num = field_number(&manifest, "tags");
        let (ords, terms) = tags_entry(&meta, field_num);
        // Only doc 0 ([red, blue]) matches, so "green" must not be reported.
        let counts = facet_counts(&data, ords, terms, &[0]).unwrap();
        let resolved = resolve_labels(&data, terms, &counts).unwrap();
        let result = top_children(resolved, 10).unwrap();
        let labels: Vec<&str> = result
            .label_values
            .iter()
            .map(|f| f.label.as_str())
            .collect();
        assert_eq!(labels.len(), 2);
        assert!(labels.contains(&"red") && labels.contains(&"blue"));
        assert!(!labels.contains(&"green"));
        assert_eq!(result.value, 2);
        assert_eq!(result.child_count, 2);
    }

    // --- `LongRange`/`DoubleRange` constructors ---

    #[test]
    fn new_long_normalizes_exclusive_bounds_to_inclusive_ones() {
        // `LongRange`'s constructor: `min++` when exclusive, `max--` when
        // exclusive, and the resulting range is inclusive on both ends.
        let r = NumericRange::new_long("r", 7, false, 1000, false).unwrap();
        assert_eq!((r.min, r.max), (8, 999));
        assert!(r.min_inclusive && r.max_inclusive);
        assert!(!r.contains(7) && r.contains(8));
        assert!(r.contains(999) && !r.contains(1000));
    }

    #[test]
    fn new_long_rejects_empty_ranges_like_fail_no_match() {
        // min > max after normalization.
        assert!(NumericRange::new_long("r", 10, true, 5, true).is_err());
        // (42, 42] and [42, 42) are both empty.
        assert!(NumericRange::new_long("r", 42, false, 42, true).is_err());
        assert!(NumericRange::new_long("r", 42, true, 42, false).is_err());
        // An exclusive bound with nowhere to step: Java's `failNoMatch`.
        assert!(NumericRange::new_long("r", i64::MAX, false, i64::MAX, true).is_err());
        assert!(NumericRange::new_long("r", i64::MIN, true, i64::MIN, false).is_err());
        // [42, 42] is *not* empty.
        assert!(NumericRange::new_long("r", 42, true, 42, true).is_ok());
    }

    #[test]
    fn sortable_double_bits_is_its_own_inverse_and_order_preserving() {
        let values = [
            f64::NEG_INFINITY,
            -1e308,
            -1.5,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1.5,
            1e308,
            f64::INFINITY,
        ];
        let mut previous = i64::MIN;
        for (i, &v) in values.iter().enumerate() {
            let sortable = double_to_sortable_long(v);
            // Involution: applying the bit flip twice returns the raw bits.
            assert_eq!(sortable_double_bits(sortable), v.to_bits() as i64);
            if i > 0 && values[i - 1] != v {
                assert!(
                    sortable > previous,
                    "sortable order must follow f64 order at {v}"
                );
            }
            previous = sortable;
        }
    }

    #[test]
    fn new_double_nudges_exclusive_bounds_in_double_space_before_the_transform() {
        // `Math.nextUp(1.0)`/`Math.nextDown(2.0)`: the exclusive endpoints
        // themselves must fall outside, but every representable double
        // strictly between them must fall inside.
        let r = NumericRange::new_double("r", 1.0, false, 2.0, false).unwrap();
        assert!(!r.contains(double_to_sortable_long(1.0)));
        assert!(r.contains(double_to_sortable_long(next_up(1.0))));
        assert!(r.contains(double_to_sortable_long(next_down(2.0))));
        assert!(!r.contains(double_to_sortable_long(2.0)));
        assert!(r.contains(double_to_sortable_long(1.5)));

        // Nudging in double space, not sortable-long space, is what makes the
        // subnormal neighbourhood of zero come out right.
        let around_zero = NumericRange::new_double("z", 0.0, false, 1.0, true).unwrap();
        assert!(!around_zero.contains(double_to_sortable_long(0.0)));
        assert!(around_zero.contains(double_to_sortable_long(f64::from_bits(1))));
    }

    #[test]
    fn new_double_rejects_nan_and_empty_ranges() {
        assert!(NumericRange::new_double("r", f64::NAN, true, 1.0, true).is_err());
        assert!(NumericRange::new_double("r", 0.0, true, f64::NAN, true).is_err());
        assert!(NumericRange::new_double("r", 2.0, true, 1.0, true).is_err());
        assert!(NumericRange::new_double("r", 1.0, false, 1.0, true).is_err());
        assert!(NumericRange::new_double("r", 1.0, true, 1.0, true).is_ok());
    }

    #[test]
    fn double_range_facet_counts_maps_stored_bits_through_sortable_double_bits() {
        // Real-Lucene-written `varying` field: -100, 7, 42, 1000, -3 for docs
        // 0..4 (already differentially verified in `doc_value_query.rs`).
        // `sortable_double_bits` is the identity for a non-negative stored
        // value and flips the low 63 bits for a negative one, so a range
        // pinned to the *mapped* form of -100 counts doc 0 through
        // `double_range_facet_counts` (which applies `mapDocValue`) and
        // nothing at all through `range_facet_counts` (which doesn't). That
        // difference is exactly what `DoubleRangeFacetCounts.mapDocValue`
        // buys, asserted without hand-rolling a doc-values blob.
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        let mapped = sortable_double_bits(-100);
        assert_ne!(mapped, -100, "the mapping must actually move a negative");
        let ranges = vec![NumericRange {
            label: "exactly_mapped_minus_100".into(),
            min: mapped,
            min_inclusive: true,
            max: mapped,
            max_inclusive: true,
        }];
        let matching: Vec<i32> = (0..5).collect();

        let mapped_counts = double_range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(mapped_counts[0].1, 1, "doc 0's -100 maps into the range");

        let unmapped_counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(
            unmapped_counts[0].1, 0,
            "without mapDocValue the same range matches nothing"
        );
    }

    #[test]
    fn double_range_facet_counts_shares_the_missing_value_and_order_rules() {
        // `sparse` field: 5, NONE, 15, NONE, 25 -- all non-negative, so the
        // mapping is the identity and the bucket arithmetic is directly
        // checkable; a doc with no value still counts in no range, and the
        // output preserves caller-specified range order.
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "sparse")).unwrap();
        let ranges = vec![
            NumericRange {
                label: "high".into(),
                min: 15,
                min_inclusive: false,
                max: i64::MAX,
                max_inclusive: true,
            },
            NumericRange {
                label: "everything".into(),
                min: i64::MIN,
                min_inclusive: true,
                max: i64::MAX,
                max_inclusive: true,
            },
        ];
        let matching: Vec<i32> = (0..5).collect();
        let counts = double_range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(
            counts,
            vec![("high".to_string(), 1), ("everything".to_string(), 3)]
        );
    }

    #[test]
    fn double_range_facet_counts_propagates_decode_errors() {
        let entry = doc_values::NumericEntry {
            field_number: 0,
            docs_with_field_offset: -1,
            docs_with_field_length: 0,
            jump_table_entry_count: -1,
            dense_rank_power: 0xFF,
            num_values: 1,
            table: None,
            bits_per_value: 8,
            min_value: 0,
            gcd: 1,
            values_offset: 0,
            values_length: 1,
            block_shift: None,
            value_jump_table_offset: 0,
        };
        let ranges = vec![NumericRange::new_double("r", 0.0, true, 1.0, true).unwrap()];
        let err = double_range_facet_counts(&[], &entry, &ranges, &[0]).unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }

    // --- `RangeFacetCounts.getTopChildren` ---

    #[test]
    fn top_range_children_orders_count_desc_then_label_asc_and_drops_zeroes() {
        let counts = vec![
            ("b".to_string(), 3),
            ("a".to_string(), 3),
            ("c".to_string(), 7),
            ("d".to_string(), 0),
        ];
        let result = top_range_children(&counts, 10, 10);
        assert_eq!(
            result
                .label_values
                .iter()
                .map(|(l, c)| (l.as_str(), *c))
                .collect::<Vec<_>>(),
            vec![("c", 7), ("a", 3), ("b", 3)]
        );
        assert_eq!(result.child_count, 3);
        assert_eq!(result.value, 10);
    }

    #[test]
    fn top_range_children_truncates_but_keeps_the_full_child_count() {
        let counts = vec![("a".to_string(), 1), ("b".to_string(), 2)];
        let result = top_range_children(&counts, 3, 1);
        assert_eq!(result.label_values, vec![("b".to_string(), 2)]);
        assert_eq!(result.child_count, 2);
    }

    #[test]
    fn top_range_children_with_every_range_empty_is_still_a_result() {
        // Unlike the SORTED_SET case, `RangeFacetCounts.getTopChildren`
        // returns a `FacetResult` (with an empty child list), never `null`.
        let counts = vec![("a".to_string(), 0)];
        let result = top_range_children(&counts, 0, 5);
        assert!(result.label_values.is_empty());
        assert_eq!(result.child_count, 0);
        assert_eq!(result.value, 0);
    }

    #[test]
    #[should_panic(expected = "topN must be > 0 (got: 0)")]
    fn top_range_children_rejects_a_zero_top_n() {
        top_range_children(&[("a".to_string(), 1)], 1, 0);
    }

    #[test]
    fn facet_counts_match_real_lucene_ground_truth() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let field_num = field_number(&manifest, "tags");
        let (ords, terms) = tags_entry(&meta, field_num);

        // All 5 docs matching -- doc0=[red,blue], doc1=NONE, doc2=[green],
        // doc3=[blue], doc4=[red,green].
        let matching: Vec<i32> = (0..5).collect();
        let counts = facet_counts(&data, ords, terms, &matching).unwrap();

        let expected = expected_counts_from_manifest(&manifest, terms.terms_dict_size as usize);
        assert_eq!(counts, expected);

        // Cross-check against the manifest's resolved term strings too, not
        // just raw ordinal counts, so a label/ordinal mismatch would also
        // fail this test.
        let resolved = resolve_labels(&data, terms, &counts).unwrap();
        let expected_terms: Vec<&str> = manifest.get("field.tags.terms").split(',').collect();
        assert_eq!(resolved.len(), expected_terms.len());
        for (fc, expected_label) in resolved.iter().zip(expected_terms.iter()) {
            assert_eq!(&fc.label, expected_label);
        }
    }

    #[test]
    fn multi_valued_doc_increments_every_ordinal_not_just_first() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let field_num = field_number(&manifest, "tags");
        let (ords, terms) = tags_entry(&meta, field_num);

        // doc0 alone has ords [red, blue] (two ordinals) -- both must be
        // incremented, not just the first ("primary") one.
        let counts = facet_counts(&data, ords, terms, &[0]).unwrap();
        assert_eq!(counts.iter().sum::<u64>(), 2);

        let resolved = resolve_labels(&data, terms, &counts).unwrap();
        let red = resolved.iter().find(|f| f.label == "red").unwrap();
        let blue = resolved.iter().find(|f| f.label == "blue").unwrap();
        assert_eq!(red.count, 1);
        assert_eq!(blue.count, 1);
        assert!(resolved
            .iter()
            .all(|f| f.label == "red" || f.label == "blue" || f.count == 0));
    }

    #[test]
    fn doc_not_in_matching_set_contributes_nothing() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let field_num = field_number(&manifest, "tags");
        let (ords, terms) = tags_entry(&meta, field_num);

        // Every doc except doc4 ([red, green]) -- doc4's ordinals must not
        // show up.
        let matching: Vec<i32> = vec![0, 1, 2, 3];
        let counts = facet_counts(&data, ords, terms, &matching).unwrap();
        let expected = expected_counts_from_manifest(&manifest, terms.terms_dict_size as usize);
        // green appears in doc2 and doc4 in the full set; excluding doc4
        // should drop green's count by exactly one relative to full-set
        // expected.
        let resolved = resolve_labels(&data, terms, &counts).unwrap();
        let resolved_full = resolve_labels(&data, terms, &expected).unwrap();
        let green = resolved.iter().find(|f| f.label == "green").unwrap();
        let green_full = resolved_full.iter().find(|f| f.label == "green").unwrap();
        assert_eq!(green.count, green_full.count - 1);
    }

    #[test]
    fn empty_matching_set_yields_all_zero_counts() {
        let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
        let field_num = field_number(&manifest, "tags");
        let (ords, terms) = tags_entry(&meta, field_num);

        let counts = facet_counts(&data, ords, terms, &[]).unwrap();
        assert_eq!(counts.len(), terms.terms_dict_size as usize);
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn top_n_facets_sorts_descending_and_truncates() {
        let facets = vec![
            FacetCount {
                ord: 0,
                label: "a".into(),
                count: 3,
            },
            FacetCount {
                ord: 1,
                label: "b".into(),
                count: 7,
            },
            FacetCount {
                ord: 2,
                label: "c".into(),
                count: 5,
            },
        ];
        let top2 = top_n_facets(facets, 2);
        assert_eq!(
            top2.iter().map(|f| f.label.as_str()).collect::<Vec<_>>(),
            vec!["b", "c"]
        );
    }

    #[test]
    fn top_n_facets_ties_broken_by_ascending_ordinal() {
        let facets = vec![
            FacetCount {
                ord: 2,
                label: "z".into(),
                count: 4,
            },
            FacetCount {
                ord: 0,
                label: "x".into(),
                count: 4,
            },
            FacetCount {
                ord: 1,
                label: "y".into(),
                count: 4,
            },
        ];
        let top = top_n_facets(facets, 3);
        assert_eq!(top.iter().map(|f| f.ord).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn top_n_facets_n_larger_than_available_returns_all() {
        let facets = vec![FacetCount {
            ord: 0,
            label: "a".into(),
            count: 1,
        }];
        let top = top_n_facets(facets, 100);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn top_n_facets_empty_input_yields_empty_output() {
        let top = top_n_facets(Vec::new(), 5);
        assert!(top.is_empty());
    }

    #[test]
    fn facet_counts_propagates_decode_errors() {
        let mut numeric = doc_values::NumericEntry {
            field_number: 0,
            docs_with_field_offset: -1,
            docs_with_field_length: 0,
            jump_table_entry_count: -1,
            dense_rank_power: 0xFF,
            num_values: 1,
            table: None,
            bits_per_value: 8,
            min_value: 0,
            gcd: 1,
            values_offset: 0,
            values_length: 1,
            block_shift: None,
            value_jump_table_offset: 0,
        };
        numeric.bits_per_value = 8;
        let entry = SortedNumericEntry {
            field_number: 0,
            numeric,
            num_docs_with_field: 1,
            addresses: None,
        };
        let terms = TermsDictEntry {
            terms_dict_size: 1,
            max_term_length: 8,
            // Coherent with the single 8-byte term above; this test asserts
            // the *error* path, so the bound is never reached.
            max_block_length: 8,
            terms_data_offset: 0,
            terms_data_length: 0,
        };
        let err = facet_counts(&[], &entry, &terms, &[0]).unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }

    // --- range_facet_counts ---
    //
    // Reuses `doc_values_index`'s `varying` field (task #21/#31's own
    // fixture -- see `doc_value_query.rs`'s tests), whose 5 docs' real-Lucene-
    // recorded values are already differentially verified there: -100, 7, 42,
    // 1000, -3 for docs 0..4. Bucket assignment is hand-verified against
    // those recorded values rather than re-deriving decode correctness.

    fn dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/doc_values_index/"
        )
        .to_string()
    }

    fn dv_meta_and_data(dir: &str) -> (Manifest, Vec<u8>, DocValuesMeta) {
        let (manifest, data, meta) = load_dv_meta(dir);
        (manifest, data, meta)
    }

    fn field_num(manifest: &Manifest, field: &str) -> i32 {
        manifest
            .get("field_numbers")
            .split(',')
            .find_map(|kv| {
                let (name, num) = kv.split_once(':').unwrap();
                (name == field).then(|| num.parse().unwrap())
            })
            .unwrap_or_else(|| panic!("field {field} missing from field_numbers"))
    }

    #[test]
    fn range_facet_counts_partitions_non_overlapping_ranges() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // values: -100, 7, 42, 1000, -3 for docs 0..4.
        let ranges = vec![
            NumericRange {
                label: "negative".into(),
                min: i64::MIN,
                min_inclusive: true,
                max: 0,
                max_inclusive: false,
            },
            NumericRange {
                label: "small_positive".into(),
                min: 0,
                min_inclusive: true,
                max: 100,
                max_inclusive: true,
            },
            NumericRange {
                label: "large".into(),
                min: 100,
                min_inclusive: false,
                max: i64::MAX,
                max_inclusive: true,
            },
        ];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(
            counts,
            vec![
                ("negative".to_string(), 2),       // -100, -3
                ("small_positive".to_string(), 2), // 7, 42
                ("large".to_string(), 1),          // 1000
            ]
        );
    }

    #[test]
    fn range_facet_counts_overlapping_ranges_count_doc_in_both() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // doc2's value (42) falls in both overlapping ranges below.
        let ranges = vec![
            NumericRange {
                label: "0-50".into(),
                min: 0,
                min_inclusive: true,
                max: 50,
                max_inclusive: true,
            },
            NumericRange {
                label: "10-1000".into(),
                min: 10,
                min_inclusive: true,
                max: 1000,
                max_inclusive: true,
            },
        ];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        // "0-50": 7, 42 -> 2. "10-1000": 42, 1000 -> 2. doc2 (42) counted in both.
        assert_eq!(
            counts,
            vec![("0-50".to_string(), 2), ("10-1000".to_string(), 2)]
        );
    }

    #[test]
    fn range_facet_counts_boundary_inclusive_inclusive() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // [42, 42] inclusive-inclusive: only doc2 (value 42) matches.
        let ranges = vec![NumericRange {
            label: "exact".into(),
            min: 42,
            min_inclusive: true,
            max: 42,
            max_inclusive: true,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("exact".to_string(), 1)]);
    }

    #[test]
    fn range_facet_counts_boundary_inclusive_exclusive() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // [42, 42) inclusive-exclusive: the max bound equal to the value
        // itself must exclude it.
        let ranges = vec![NumericRange {
            label: "r".into(),
            min: 42,
            min_inclusive: true,
            max: 42,
            max_inclusive: false,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("r".to_string(), 0)]);
    }

    #[test]
    fn range_facet_counts_boundary_exclusive_inclusive() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // (42, 42] exclusive-inclusive: the min bound equal to the value
        // itself must exclude it.
        let ranges = vec![NumericRange {
            label: "r".into(),
            min: 42,
            min_inclusive: false,
            max: 42,
            max_inclusive: true,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("r".to_string(), 0)]);
    }

    #[test]
    fn range_facet_counts_boundary_exclusive_exclusive() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // (7, 1000) exclusive-exclusive: doc2 (42) still matches, but the
        // endpoints 7 and 1000 themselves (docs 1 and 3) must not.
        let ranges = vec![NumericRange {
            label: "r".into(),
            min: 7,
            min_inclusive: false,
            max: 1000,
            max_inclusive: false,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("r".to_string(), 1)]);
    }

    #[test]
    fn range_facet_counts_missing_value_never_counted_even_unbounded() {
        // `sparse` field: 5, NONE, 15, NONE, 25 -- docs 1 and 3 have no value.
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "sparse")).unwrap();
        // An unbounded-above range that would catch every value if a missing
        // doc wrongly counted.
        let ranges = vec![NumericRange {
            label: "everything".into(),
            min: i64::MIN,
            min_inclusive: true,
            max: i64::MAX,
            max_inclusive: true,
        }];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        // Only docs 0, 2, 4 (5, 15, 25) have a value.
        assert_eq!(counts, vec![("everything".to_string(), 3)]);
    }

    #[test]
    fn range_facet_counts_doc_not_in_matching_set_contributes_nothing() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // Excludes doc3 (value 1000); an unbounded range would otherwise
        // count it.
        let ranges = vec![NumericRange {
            label: "everything".into(),
            min: i64::MIN,
            min_inclusive: true,
            max: i64::MAX,
            max_inclusive: true,
        }];
        let matching: Vec<i32> = vec![0, 1, 2, 4];
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(counts, vec![("everything".to_string(), 4)]);
    }

    #[test]
    fn range_facet_counts_empty_matching_set_yields_all_zero_counts() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        let ranges = vec![
            NumericRange {
                label: "a".into(),
                min: 0,
                min_inclusive: true,
                max: 10,
                max_inclusive: true,
            },
            NumericRange {
                label: "b".into(),
                min: 10,
                min_inclusive: false,
                max: 100,
                max_inclusive: true,
            },
        ];
        let counts = range_facet_counts(&data, entry, &ranges, &[]).unwrap();
        assert_eq!(counts, vec![("a".to_string(), 0), ("b".to_string(), 0)]);
    }

    #[test]
    fn range_facet_counts_preserves_caller_specified_range_order() {
        let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
        let entry = meta.numeric_entry(field_num(&manifest, "varying")).unwrap();
        // Deliberately out of value order -- output must mirror input order,
        // not sort by count.
        let ranges = vec![
            NumericRange {
                label: "large".into(),
                min: 100,
                min_inclusive: false,
                max: i64::MAX,
                max_inclusive: true,
            },
            NumericRange {
                label: "negative".into(),
                min: i64::MIN,
                min_inclusive: true,
                max: 0,
                max_inclusive: false,
            },
        ];
        let matching: Vec<i32> = (0..5).collect();
        let counts = range_facet_counts(&data, entry, &ranges, &matching).unwrap();
        assert_eq!(
            counts.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(),
            vec!["large", "negative"]
        );
        assert_eq!(counts[0].1, 1); // large: 1000
        assert_eq!(counts[1].1, 2); // negative: -100, -3
    }

    #[test]
    fn range_facet_counts_propagates_decode_errors() {
        let mut entry = doc_values::NumericEntry {
            field_number: 0,
            docs_with_field_offset: -1,
            docs_with_field_length: 0,
            jump_table_entry_count: -1,
            dense_rank_power: 0xFF,
            num_values: 1,
            table: None,
            bits_per_value: 8,
            min_value: 0,
            gcd: 1,
            values_offset: 0,
            values_length: 1,
            block_shift: None,
            value_jump_table_offset: 0,
        };
        entry.bits_per_value = 8;
        let ranges = vec![NumericRange {
            label: "r".into(),
            min: 0,
            min_inclusive: true,
            max: 100,
            max_inclusive: true,
        }];
        let err = range_facet_counts(&[], &entry, &ranges, &[0]).unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }
}
