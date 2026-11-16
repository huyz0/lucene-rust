//! Doc-values-driven query support: a numeric range filter and a sorted-ordinal
//! range/equality filter (`NumericDocValuesField.newSlowRangeQuery`/
//! `SortedSetDocValuesField.newSlowRangeQuery`-equivalent, single-valued case
//! only — see [`crate::doc_value_query`]'s scope note below), the real
//! multi-valued `SortedNumericDocValuesField.newSlowRangeQuery`, plus a
//! post-processing "sort an already-matched doc set by a numeric doc value"
//! helper (the `SortField.Type.LONG`/`INT`-equivalent "real search sorting"
//! capability).
//!
//! ## Scope
//!
//! **In scope:**
//! - [`search_numeric_range`]: every live doc in `[0, max_doc)` whose
//!   [`lucene_codecs::doc_values::NumericEntry`] value falls in `[min, max]`
//!   (inclusive both ends) — a full doc-values sweep, not a skip-list-driven
//!   scorer (see the "why a full sweep" note on that function).
//! - [`search_sorted_ord_range`]: same shape, but compares ordinals of a
//!   [`lucene_codecs::doc_values::SortedEntry`] (single-valued SORTED field)
//!   against an inclusive `[min_ord, max_ord]` range. An equality predicate
//!   is just `min_ord == max_ord`; resolving a term's ordinal (via
//!   `terms_dict::seek_exact` or similar) to plug into this range is the
//!   caller's job — this function only ever compares already-known ordinals,
//!   mirroring `sorted_ord`'s own "just the ordinal" contract.
//! - [`sort_by_numeric_doc_value`]: given a candidate doc-ID list (e.g. the
//!   output of [`crate::search_term_query`]/[`crate::search_boolean_query`])
//!   and a `NumericEntry`, returns `(doc_id, value)` pairs ascending by value,
//!   ties broken by ascending doc ID (see that function's doc comment for the
//!   missing-value policy).
//! - [`sort_top_n_by_numeric_doc_value`]: the bounded, `TopFieldCollector`-
//!   driven sibling of [`sort_by_numeric_doc_value`] — same candidate-list
//!   contract, but ascending **or** descending (real search-time
//!   `SortField`/`TopFieldCollector` sorting, not just index-sort's
//!   ascending-only helper) and truncated to the top `n` hits without
//!   materializing a full sorted `Vec`. [`search_numeric_range_sorted_by_field`]
//!   is the concrete end-to-end wiring of this into an existing query
//!   ([`search_numeric_range`]).
//!
//! - [`ValueSelector`]/[`search_multi_valued_range`]/
//!   [`sort_by_multi_valued_doc_value`]: the multi-valued (SORTED_NUMERIC, and
//!   SORTED_SET's ordinal form) equivalents of the two functions above. Real
//!   Lucene reduces a multi-valued doc to one comparable value via a
//!   `SortedNumericSelector`/`SortedSetSelector` (MIN/MAX/MIDDLE_MIN/
//!   MIDDLE_MAX) before range-filtering or sorting; [`ValueSelector`] covers
//!   MIN/MAX (MIDDLE_MIN/MIDDLE_MAX are deferred, see below). Both functions
//!   take a [`lucene_codecs::doc_values::SortedNumericEntry`] — which is also
//!   exactly the ordinal-array shape a multi-valued
//!   [`lucene_codecs::doc_values::SortedSetKind::Multi`] field's `ords` uses
//!   (`sorted_numeric_values` already reads both the same way), so these two
//!   functions serve SORTED_SET range/sort too: pass the `Multi` variant's
//!   `ords` entry and the reduced value is a term ordinal instead of a raw
//!   number. A single-valued `SortedSetKind::Single` field still uses
//!   [`search_sorted_ord_range`] (no reduction needed, one ordinal per doc).
//!
//! - [`search_numeric_range_with_skip_index`]: same match semantics as
//!   [`search_numeric_range`], but driven by an already-decoded
//!   [`doc_values::DocValuesSkipIndex`] ([`doc_values::parse_skip_index`]'s
//!   output) instead of sweeping every doc — see that function's doc comment
//!   for exactly how a level-0 interval's `[min_value, max_value]` decides
//!   whether its whole doc range gets skipped without decoding a single
//!   value. Only usable for a field that actually has a skip index (real
//!   Lucene's `DocValuesSkipIndexType.RANGE`); [`search_numeric_range`]
//!   remains the correct (and only) choice for a field without one.
//!
//! **Deliberately out of scope** (tracked in `docs/parity.md`):
//! - **MIDDLE_MIN/MIDDLE_MAX selectors.** Real Lucene's `SortedSetSelector`
//!   also offers "the lower/upper of the two middle values" for even value
//!   counts — a niche median-ish reduction with no numeric-field equivalent.
//!   MIN/MAX cover the common range/sort use cases; MIDDLE_* would be a small
//!   follow-up (`values[(len - 1) / 2]` / `values[len / 2]` on a sorted
//!   `values`, which [`ValueSelector::reduce`] doesn't need to sort for since
//!   MIN/MAX only ever need one pass) if a caller needs it.
//! - **Multiple sort fields / secondary sort keys** beyond the single
//!   documented tie-break (ascending doc ID). Real `Sort` composes multiple
//!   `SortField`s; [`sort_by_numeric_doc_value`]/[`sort_top_n_by_numeric_doc_value`]
//!   are single-key sorts only (the latter does support ascending *and*
//!   descending, via [`SortDirection`] — only [`sort_by_numeric_doc_value`],
//!   the original index-sort-era helper, is ascending-only).
//! - **A DISI-driven range scorer for [`search_sorted_ord_range`] /
//!   [`search_multi_valued_range`].** [`search_numeric_range_with_skip_index`]
//!   covers the NUMERIC case; the SORTED/SORTED_NUMERIC/SORTED_SET range
//!   queries above still only offer the full `[0, max_doc)` sweep — a future
//!   follow-up if a skip index is ever recorded for those doc-values types.

use lucene_codecs::doc_values::{
    self, BinaryEntry, DocValuesMeta, NumericEntry, SortedEntry, SortedNumericEntry,
    SortedSetEntry, SortedSetKind,
};
use lucene_util::fixed_bit_set::FixedBitSet;

use crate::collector::{FieldValueDoc, SortDirection, TopFieldCollector};
use crate::{Collector, Result};

/// Feeds every **live** doc in `[0, max_doc)` whose numeric doc-value falls in
/// `[min, max]` (inclusive both ends) to `collector`, ascending by doc ID.
///
/// A doc with no value at all for this field (`numeric_value` returns `None`
/// — legitimate for a sparse NUMERIC field, see [`doc_values::NumericEntry`]'s
/// module doc) never matches any range, including an unbounded-looking one
/// like `i64::MIN..=i64::MAX` — this mirrors real Lucene's `DocValuesFieldExistsQuery`/
/// range-query semantics, where a doc missing the field is not a hit.
///
/// **Why a full `[0, max_doc)` sweep, not a candidate-set/skip-list scan**:
/// unlike [`crate::search_term_query`] (which starts from a term dictionary's
/// already-known matching-doc postings list), a numeric range has no
/// dictionary to seek into — every doc's value is independent, so without a
/// skip index there is no cheaper starting point than checking every doc.
/// This matches real Lucene's own fallback behavior (`PointRangeQuery`'s
/// "slow" `SortedNumericDocValuesField.newSlowRangeQuery` path, used
/// precisely when no BKD point index exists for the field, as is the case in
/// this port so far). **If the field does have a decoded skip index**
/// ([`doc_values::parse_skip_index`]), prefer
/// [`search_numeric_range_with_skip_index`] instead — it skips whole
/// doc-ID ranges the skip index proves can't match, without decoding a
/// single value in them.
pub fn search_numeric_range<C: Collector>(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    min: i64,
    max: i64,
    collector: &mut C,
) -> Result<()> {
    // One reader for the scan: the field's shape and values region are
    // resolved once, not per document (`doc_values::NumericReader`).
    // `for_each_value` decodes a dense column a chunk at a time; the value is
    // tested before liveness because it is already in hand and a deletion is
    // the rare case.
    let mut values = doc_values::NumericReader::new(doc_values_data, entry);
    values.for_each_value(0, max_doc, |doc_id, value| {
        if value >= min && value <= max && live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
            collector.collect(doc_id);
        }
    })?;
    Ok(())
}

/// Skip-index-driven sibling of [`search_numeric_range`]: same inclusive
/// `[min, max]` match semantics and the same missing-value rule (a doc with
/// no value never matches), but walks `skip_index`'s intervals in doc-ID
/// order instead of sweeping every doc in `[0, max_doc)`.
///
/// **How the skip works** (mirrors real Lucene's
/// `Lucene90DocValuesProducer.DocValuesSkipper`/`IndexedDISI` block-skipping,
/// used by `PointRangeQuery`'s doc-values fallback when a skip index is
/// present): each [`doc_values::SkipIndexInterval`] covers a contiguous
/// `[min_doc_id, max_doc_id]` doc range at its finest level
/// (`levels[0]` — see that type's doc comment on level ordering) and records
/// the `[min_value, max_value]` span of every value in that range. If that
/// span doesn't intersect the query's `[min, max]` at all, **no doc in the
/// interval can possibly match**, so the whole range is skipped without
/// decoding a single value — exactly the same "read the summary, skip the
/// block" trick a BKD point range query uses, just over doc-values blocks
/// instead of BKD leaf blocks. Otherwise every live doc in the interval's
/// range is decoded and checked individually (this port only ever consults
/// the finest level — real Lucene's own reader also falls back to per-doc
/// decoding once it reaches a competitive base interval; recursing through
/// coarser levels first is a possible future optimization, not a
/// correctness requirement, since level 0's summary alone is exact enough to
/// decide skip-or-scan for the whole interval).
///
/// Returns `Ok(decoded)`, the number of docs whose value was actually
/// decoded (i.e. docs inside intervals that couldn't be skipped) — callers
/// that don't care can ignore it; it exists so tests (and diagnostics) can
/// prove real skipping happened rather than just "didn't crash" (see this
/// module's tests).
///
/// **Panics**: never — every interval is assumed to carry at least one level
/// (`levels[0]` always exists), which [`doc_values::parse_skip_index`]
/// guarantees (it rejects a zero-level interval as
/// [`doc_values::Error::InvalidSkipIndexLevelCount`]).
#[allow(clippy::too_many_arguments)]
pub fn search_numeric_range_with_skip_index<C: Collector>(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    skip_index: &doc_values::DocValuesSkipIndex,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    min: i64,
    max: i64,
    collector: &mut C,
) -> Result<usize> {
    let mut decoded = 0usize;
    for interval in &skip_index.intervals {
        let finest = interval
            .levels
            .first()
            .expect("parse_skip_index never yields a zero-level interval");
        if finest.max_value < min || finest.min_value > max {
            // The whole interval's value range can't intersect [min, max] --
            // skip every doc in it without decoding anything.
            continue;
        }
        let start = finest.min_doc_id.max(0);
        let end = finest.max_doc_id.min(max_doc - 1);
        for doc_id in start..=end {
            if !live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
                continue;
            }
            if let Some(value) = doc_values::numeric_value(doc_values_data, entry, doc_id)? {
                decoded += 1;
                if value >= min && value <= max {
                    collector.collect(doc_id);
                }
            }
        }
    }
    Ok(decoded)
}

/// Same shape as [`search_numeric_range`], but for a single-valued SORTED
/// field's per-doc **ordinal** (`sorted_ord`) instead of a NUMERIC value —
/// `min_ord == max_ord` degenerates to an equality predicate ("every doc
/// whose SORTED value has this exact ordinal"). A doc with no ordinal at all
/// (`sorted_ord` returns `None`) never matches, same missing-value rule as
/// [`search_numeric_range`].
pub fn search_sorted_ord_range<C: Collector>(
    doc_values_data: &[u8],
    entry: &SortedEntry,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    min_ord: i64,
    max_ord: i64,
    collector: &mut C,
) -> Result<()> {
    // `sorted_ord` is `numeric_value` over the ords entry; one reader for the
    // whole scan instead.
    let mut ords = doc_values::NumericReader::new(doc_values_data, &entry.ords);
    ords.for_each_value(0, max_doc, |doc_id, ord| {
        if ord >= min_ord && ord <= max_ord && live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
            collector.collect(doc_id);
        }
    })?;
    Ok(())
}

/// How [`sort_by_numeric_doc_value`] treats a candidate doc with no value for
/// the sort field (real Lucene's `SortField.setMissingValue`-equivalent
/// choice, simplified to two policies rather than an arbitrary substitute
/// value per sort-key type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValue {
    /// Drop the doc from the result entirely — it never had a value to sort
    /// by. The simpler of the two policies and this function's recommended
    /// default for callers with no specific missing-value contract to match.
    Exclude,
    /// Substitute this value in the doc's place, so it still gets a sort
    /// position (real Lucene's most common case: `Long.MAX_VALUE`/`MIN_VALUE`
    /// to sort missing values last/first for an ascending sort).
    Default(i64),
}

/// Sorts `candidates` (e.g. [`crate::search_term_query`]/
/// [`crate::search_boolean_query`]'s output) by their numeric doc-value,
/// **ascending**, returning `(doc_id, value)` pairs. Ties (equal value) are
/// broken by **ascending doc ID** — the same "lower doc ID wins" convention
/// [`crate::collector::TopDocsCollector`] already documents for a BM25 score
/// tie, kept consistent here for a value tie.
///
/// **Missing-value handling** (a candidate doc with no value for this field —
/// legitimate for a sparse NUMERIC field): governed by `missing`, see
/// [`MissingValue`].
///
/// **Design note — a standalone function, not a `Collector` variant**: this
/// port's `Collector`/`ScoringCollector` traits exist for the "what happens
/// per matched doc, while the doc set is still being discovered" extension
/// point (`search_term_query`'s per-doc callback as it walks postings).
/// Sorting is a different shape: it needs the *entire* candidate set in hand
/// before it can decide the first output pair (the first result by value
/// might be the very last doc discovered), so a callback fired once per doc
/// during discovery has nothing useful to do until every doc has already
/// arrived. [`crate::collector::TopDocsCollector`]'s incremental top-`N` heap
/// is the shape that *does* fit the `Collector` model (it only ever needs the
/// current worst kept hit to decide, never the whole set) — but this
/// function does a full sort, not a bounded top-`N`, so that incremental
/// trick doesn't apply either. A plain function over an already-collected
/// `&[i32]` is the honest shape for "sort what a previous collector already
/// gathered"; introducing a trait for a single implementation with no second
/// caller would be the speculative generality `rust-performance` warns
/// against. A future bounded "`TopFieldCollector`-equivalent" (top-`N` by doc
/// value, incremental) would be the natural point to revisit this as a real
/// `Collector` variant.
pub fn sort_by_numeric_doc_value(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    candidates: &[i32],
    missing: MissingValue,
) -> Result<Vec<(i32, i64)>> {
    let mut pairs = Vec::with_capacity(candidates.len());
    for &doc_id in candidates {
        match doc_values::numeric_value(doc_values_data, entry, doc_id)? {
            Some(value) => pairs.push((doc_id, value)),
            None => {
                if let MissingValue::Default(default) = missing {
                    pairs.push((doc_id, default));
                }
            }
        }
    }
    pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    Ok(pairs)
}

/// `TopFieldCollector`-driven, **bounded** sibling of [`sort_by_numeric_doc_value`]:
/// decodes each `candidates` doc's numeric doc-value (applying `missing`'s
/// policy exactly as [`sort_by_numeric_doc_value`] does) and offers it to a
/// fresh [`TopFieldCollector`], returning only the top `top_n` hits ranked by
/// `direction` (ascending or descending — see [`SortDirection`]), ties broken
/// by ascending doc ID.
///
/// This is the general "run an already-matched doc set through
/// `TopFieldCollector`" composition point this module's doc comment
/// describes: `candidates` can come from any existing matched-doc source in
/// this crate ([`crate::search_term_query`], [`crate::search_boolean_query`],
/// [`search_numeric_range`], [`search_sorted_ord_range`], ...) — this
/// function only needs the resulting `&[i32]`, the same contract
/// [`sort_by_numeric_doc_value`] already has. Unlike that function, this one
/// never materializes a fully sorted `Vec` of every candidate: only up to
/// `top_n` hits are ever held at once (see [`TopFieldCollector`]'s doc
/// comment for why decode-then-`offer` is the shape, not a `Collector` impl).
///
/// **Scope**: numeric doc-value fields only (`SortField.Type.LONG`/`INT`),
/// single sort key, no descending-then-ascending secondary key. See
/// `docs/parity.md` for the precise statement.
pub fn sort_top_n_by_numeric_doc_value(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    candidates: &[i32],
    direction: SortDirection,
    missing: MissingValue,
    top_n: usize,
) -> Result<Vec<FieldValueDoc>> {
    let mut collector = TopFieldCollector::new(top_n, direction);
    let mut values = doc_values::NumericReader::new(doc_values_data, entry);
    for &doc_id in candidates {
        match values.value(doc_id)? {
            Some(value) => collector.offer(doc_id, value),
            None => {
                if let MissingValue::Default(default) = missing {
                    collector.offer(doc_id, default);
                }
            }
        }
    }
    Ok(collector.top_docs().to_vec())
}

/// End-to-end entry point wiring an EXISTING query execution path
/// ([`search_numeric_range`]) into [`sort_top_n_by_numeric_doc_value`]: runs
/// `search_numeric_range` (matching every live doc whose `range_entry` value
/// falls in `[min, max]`) into a [`crate::collector::VecCollector`] to gather
/// the matched-doc set, then sorts that set by `sort_entry`'s numeric
/// doc-value and returns the top `top_n` hits — `sort_entry` may be the same
/// field as `range_entry` (sort the matches by the field just queried) or a
/// different one (query by one field, sort results by another), both real
/// Lucene use cases (`IndexSearcher.search(query, n, sort)` with a
/// `PointRangeQuery`/`SortedNumericDocValuesField.newSlowRangeQuery` query and
/// a `SortField` on a different field).
///
/// **`range_data` and `sort_data` are separate buffers on purpose.** In Java
/// each field is served by its own `DocValuesProducer`
/// (`SegmentDocValuesProducer.dvProducersByField`), and after an
/// `IndexWriter.updateNumericDocValue` two fields of the same segment
/// genuinely live in *different* `.dvd` files -- the updated one in its
/// generation, the untouched one in the base column. A single
/// `doc_values_data` parameter shared by both entries silently assumed they
/// always coincide, which is true only until the first doc-values update.
/// Pass the same slice twice for the common case where they do.
#[allow(clippy::too_many_arguments)]
pub fn search_numeric_range_sorted_by_field(
    range_data: &[u8],
    range_entry: &NumericEntry,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    min: i64,
    max: i64,
    sort_data: &[u8],
    sort_entry: &NumericEntry,
    direction: SortDirection,
    missing: MissingValue,
    top_n: usize,
) -> Result<Vec<FieldValueDoc>> {
    let mut collector = TopFieldCollector::new(top_n, direction);
    collect_numeric_range_sorted_by_field(
        range_data,
        range_entry,
        live_docs,
        max_doc,
        min,
        max,
        sort_data,
        sort_entry,
        missing,
        0,
        &mut collector,
    )?;
    Ok(collector.top_docs().to_vec())
}

/// [`search_numeric_range_sorted_by_field`]'s scan, offering into a
/// caller-owned `collector` with every doc id shifted by `doc_base` -- so one
/// collector can be carried across the segments of a reader, visited in
/// ascending `doc_base` order, the way Lucene's `TopFieldCollector` is. A later
/// segment then offers against the bottom the earlier ones already set, and
/// ties still go to the lower global doc id.
#[allow(clippy::too_many_arguments)]
pub fn collect_numeric_range_sorted_by_field(
    range_data: &[u8],
    range_entry: &NumericEntry,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    min: i64,
    max: i64,
    sort_data: &[u8],
    sort_entry: &NumericEntry,
    missing: MissingValue,
    doc_base: i32,
    collector: &mut TopFieldCollector,
) -> Result<()> {
    // One pass: range-check and offer each document as it is read, rather
    // than collecting every match (all of them, for a wide range) and reading
    // its value a second time to sort. Same documents in the same ascending
    // order, so the same result; when the range and the sort are one column
    // the value is read once.
    let same_column = std::ptr::eq(range_data, sort_data) && std::ptr::eq(range_entry, sort_entry);
    let mut range_values = doc_values::NumericReader::new(range_data, range_entry);
    let mut sort_values = doc_values::NumericReader::new(sort_data, sort_entry);
    // The closure cannot return the sort column's error, so it parks the
    // first one here and ignores every later document.
    let mut sort_error = None;
    // `LeafFieldComparator.compareBottom`: once the collector is full, a
    // value that does not strictly beat its worst kept one is rejected here,
    // without a call into the queue -- documents arrive in ascending doc-id
    // order, so a tie loses. On a wide range this is the whole loop.
    let mut bottom = collector.bottom_value();
    range_values.for_each_value(0, max_doc, |doc_id, value| {
        if sort_error.is_some()
            || value < min
            || value > max
            || (same_column && !collector.competes(bottom, value))
            || !live_docs.is_none_or(|bits| bits.get_doc(doc_id))
        {
            return;
        }
        let sort_value = if same_column {
            Some(value)
        } else {
            match sort_values.value(doc_id) {
                Ok(v) => v,
                Err(e) => {
                    sort_error = Some(e);
                    return;
                }
            }
        };
        // ARITH: a global doc id, below the reader's `max_doc`.
        #[allow(clippy::arithmetic_side_effects)]
        let global = doc_base + doc_id;
        match sort_value {
            Some(v) => collector.offer(global, v),
            None => {
                if let MissingValue::Default(default) = missing {
                    collector.offer(global, default);
                }
            }
        }
        bottom = collector.bottom_value();
    })?;
    if let Some(e) = sort_error {
        return Err(e.into());
    }
    Ok(())
}

/// How a multi-valued doc's several values reduce to one comparable value
/// before range-filtering or sorting (real Lucene's
/// `SortedNumericSelector.Type`/`SortedSetSelector.Type`, scoped here to
/// MIN/MAX — see this module's doc comment for why MIDDLE_MIN/MIDDLE_MAX are
/// deferred). The same enum serves both SORTED_NUMERIC values and
/// SORTED_SET ordinals, since the reduction itself (smallest/largest of a
/// `Vec<i64>`) doesn't care what the numbers mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueSelector {
    /// The smallest of a doc's values (`SortedNumericSelector.Type.MIN` /
    /// `SortedSetSelector.Type.MIN`).
    Min,
    /// The largest of a doc's values (`SortedNumericSelector.Type.MAX` /
    /// `SortedSetSelector.Type.MAX`).
    Max,
}

impl ValueSelector {
    /// Reduces a doc's values to one, or `None` if the doc has none (mirrors
    /// [`doc_values::sorted_numeric_values`] returning an empty `Vec` for a
    /// doc with no values at all — a legitimately missing doc, not a tie to
    /// break).
    fn reduce(self, values: &[i64]) -> Option<i64> {
        match self {
            ValueSelector::Min => values.iter().copied().min(),
            ValueSelector::Max => values.iter().copied().max(),
        }
    }
}

/// Reduces each doc's values (from a [`SortedNumericEntry`] — a
/// SORTED_NUMERIC field, or a SORTED_SET field's ordinal array, see this
/// module's doc comment) via `selector`, then applies the same inclusive
/// `[min, max]` check [`search_numeric_range`] does. A doc with zero values
/// (`sorted_numeric_values` returns an empty `Vec`) never matches, same
/// missing-value rule as [`search_numeric_range`].
///
/// **This is not `SortedNumericDocValuesField.newSlowRangeQuery`** — that
/// query matches a doc if **any** of its values is in range, with no selector
/// involved (see [`search_sorted_numeric_range`]). `SortedNumericSelector` is
/// real Lucene's *sort-field* reduction (`SortedNumericSortField`), and this
/// function is the range-filter shaped after it: "filter on the same single
/// value this selector would sort by". The two genuinely differ — a doc with
/// values `[1, 50]` and a query range `[40, 60]` matches
/// [`search_sorted_numeric_range`] (50 is in range) but not this function
/// under [`ValueSelector::Min`] (the min, 1, is not).
#[allow(clippy::too_many_arguments)]
pub fn search_multi_valued_range<C: Collector>(
    doc_values_data: &[u8],
    entry: &SortedNumericEntry,
    selector: ValueSelector,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    min: i64,
    max: i64,
    collector: &mut C,
) -> Result<()> {
    for doc_id in 0..max_doc {
        if !live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
            continue;
        }
        let values = doc_values::sorted_numeric_values(doc_values_data, entry, doc_id)?;
        if let Some(reduced) = selector.reduce(&values) {
            if reduced >= min && reduced <= max {
                collector.collect(doc_id);
            }
        }
    }
    Ok(())
}

/// Real `SortedNumericDocValuesField.newSlowRangeQuery(field, lower, upper)` /
/// `SortedNumericDocValuesRangeQuery`: every live doc in `[0, max_doc)` with
/// **at least one** value in the inclusive `[min, max]` range, fed through
/// `collector` ascending by doc ID.
///
/// This is the multi-valued range query real Lucene actually ships. Java's
/// `TwoPhaseIterator.matches()` walks the doc's (already ascending) values,
/// skips every one below `lowerValue`, and answers with a single comparison
/// on the first one that isn't:
///
/// ```text
/// for (value : values) { if (value < lowerValue) continue; return value <= upperValue; }
/// return false;
/// ```
///
/// which is exactly "any value in `[min, max]`" given the ascending order
/// `sorted_numeric_values` also guarantees. This function reproduces that
/// early-exit shape rather than scanning every value, so a doc whose first
/// in-range candidate already exceeds `max` costs one comparison, not
/// `docValueCount` of them.
///
/// `min > max` matches nothing, the same way
/// `SortedNumericDocValuesField.newSlowRangeQuery` returns a
/// `MatchNoDocsQuery` for that case rather than erroring.
///
/// A doc with no values at all never matches, same missing-value rule as
/// [`search_numeric_range`].
#[allow(clippy::too_many_arguments)]
pub fn search_sorted_numeric_range<C: Collector>(
    doc_values_data: &[u8],
    entry: &SortedNumericEntry,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    min: i64,
    max: i64,
    collector: &mut C,
) -> Result<()> {
    if min > max {
        return Ok(());
    }
    for doc_id in 0..max_doc {
        if !live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
            continue;
        }
        let values = doc_values::sorted_numeric_values(doc_values_data, entry, doc_id)?;
        for value in values {
            if value < min {
                continue;
            }
            // Values are ascending, so the first one that isn't below `min`
            // decides the whole document.
            if value <= max {
                collector.collect(doc_id);
            }
            break;
        }
    }
    Ok(())
}

/// Multi-valued equivalent of [`sort_by_numeric_doc_value`]: reduces each
/// candidate doc's values via `selector`, then sorts ascending with the same
/// tie-break (ascending doc ID) and [`MissingValue`] policy.
pub fn sort_by_multi_valued_doc_value(
    doc_values_data: &[u8],
    entry: &SortedNumericEntry,
    selector: ValueSelector,
    candidates: &[i32],
    missing: MissingValue,
) -> Result<Vec<(i32, i64)>> {
    let mut pairs = Vec::with_capacity(candidates.len());
    for &doc_id in candidates {
        let values = doc_values::sorted_numeric_values(doc_values_data, entry, doc_id)?;
        match selector.reduce(&values) {
            Some(value) => pairs.push((doc_id, value)),
            None => {
                if let MissingValue::Default(default) = missing {
                    pairs.push((doc_id, default));
                }
            }
        }
    }
    pairs.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    Ok(pairs)
}

// ---------------------------------------------------------------------------
// `FieldExistsQuery`
// ---------------------------------------------------------------------------

/// One field's doc-values reader, whichever kind it has -- the Rust shape of
/// `FieldExistsQuery.getDocValuesDocIdSetIterator`'s `switch` on
/// `FieldInfo.getDocValuesType()`.
///
/// Resolved by [`doc_values_field`]; `None` there is Java's `NONE` case (and
/// its `fieldInfo == null` case), which returns a `null` iterator and so
/// matches nothing.
#[derive(Debug, Clone, Copy)]
pub enum DocValuesField<'a> {
    Numeric(&'a NumericEntry),
    Binary(&'a BinaryEntry),
    Sorted(&'a SortedEntry),
    SortedNumeric(&'a SortedNumericEntry),
    SortedSet(&'a SortedSetEntry),
}

/// `FieldExistsQuery.getDocValuesDocIdSetIterator(field, reader)`'s resolution
/// step: whichever doc-values kind this field was indexed with.
///
/// Lucene enforces (LUCENE-9334) that a field always uses the same data
/// structures across an index, so at most one of these can be present; this
/// checks them in Java's `switch` order and returns the first hit.
pub fn doc_values_field(meta: &DocValuesMeta, field_number: i32) -> Option<DocValuesField<'_>> {
    if let Some(e) = meta.numeric_entry(field_number) {
        return Some(DocValuesField::Numeric(e));
    }
    if let Some(e) = meta.binary_entry(field_number) {
        return Some(DocValuesField::Binary(e));
    }
    if let Some(e) = meta.sorted_entry(field_number) {
        return Some(DocValuesField::Sorted(e));
    }
    if let Some(e) = meta.sorted_numeric_entry(field_number) {
        return Some(DocValuesField::SortedNumeric(e));
    }
    if let Some(e) = meta.sorted_set_entry(field_number) {
        return Some(DocValuesField::SortedSet(e));
    }
    None
}

/// Whether `doc` has any value at all for `field` -- the per-document half of
/// `FieldExistsQuery`, i.e. "would this field's `DocIdSetIterator` land on
/// this doc".
///
/// A dense field answers `true` for every doc; a sparse one consults the
/// field's `IndexedDISI`, which is what each `doc_values::*_value` accessor
/// already does when it returns `None`/an empty list.
pub fn doc_has_value(doc_values_data: &[u8], field: DocValuesField<'_>, doc: i32) -> Result<bool> {
    Ok(match field {
        DocValuesField::Numeric(e) => doc_values::numeric_value(doc_values_data, e, doc)?.is_some(),
        DocValuesField::Binary(e) => doc_values::binary_value(doc_values_data, e, doc)?.is_some(),
        DocValuesField::Sorted(e) => doc_values::sorted_ord(doc_values_data, e, doc)?.is_some(),
        DocValuesField::SortedNumeric(e) => {
            !doc_values::sorted_numeric_values(doc_values_data, e, doc)?.is_empty()
        }
        DocValuesField::SortedSet(e) => match &e.kind {
            SortedSetKind::Single(sorted) => {
                doc_values::sorted_ord(doc_values_data, sorted, doc)?.is_some()
            }
            SortedSetKind::Multi { ords, .. } => {
                !doc_values::sorted_numeric_values(doc_values_data, ords, doc)?.is_empty()
            }
        },
    })
}

/// `FieldExistsQuery` over the **doc-values** source: every live doc in
/// `[0, max_doc)` that has a value for the field, collected in ascending doc
/// order at a constant score.
///
/// This is the whole of Java's `ConstantScoreWeight.scorerSupplier` for the
/// `fieldInfo.getDocValuesType() != NONE` branch:
/// `getDocValuesDocIdSetIterator` picks the field's iterator and
/// `ConstantScoreScorerSupplier.fromIterator` wraps it. A field with no
/// doc-values entry at all is Java's `null` iterator -- **no matches**, not an
/// error (`doc_values_field` returns `None`, and so does this function's
/// caller-visible behaviour: zero collected docs).
///
/// **This is one of three sources**, and not the one Java tries first:
/// [`field_exists_source`] is the `FieldInfo` switch that chooses between
/// [`search_field_exists_norms`], [`search_field_exists_vectors`] and this
/// function, in Java's own order. There is no fourth, points-based source --
/// `FieldExistsQuery`'s scorer never touches `PointValues`; points appear only
/// in its `rewrite`/`count` shortcuts as a *docCount* proxy, which is
/// [`field_exists_leaf_is_complete`]. (10.5.0's `FieldExistsQuery` has no
/// points iterator source.)
pub fn search_field_exists<C: Collector>(
    doc_values_data: &[u8],
    field: DocValuesField<'_>,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    collector: &mut C,
) -> Result<()> {
    for doc_id in 0..max_doc {
        if !live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
            continue;
        }
        if doc_has_value(doc_values_data, field, doc_id)? {
            collector.collect(doc_id);
        }
    }
    Ok(())
}

/// Which of `FieldExistsQuery`'s three sources a field's `FieldInfo` selects.
///
/// Java's `ConstantScoreWeight.scorerSupplier` is a three-way if/else on the
/// `FieldInfo` alone, in exactly this order, and the order matters: LUCENE-9334
/// guarantees a field uses the same data structures index-wide, but a field can
/// legitimately have *both* norms and doc values, and Java then iterates its
/// norms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldExistsSource {
    /// `fieldInfo.hasNorms()` -- `reader.getNormValues(field)`.
    Norms,
    /// `fieldInfo.getVectorDimension() != 0` --
    /// `getFloatVectorValues(field).iterator()` / `getByteVectorValues(...)`.
    Vectors,
    /// `fieldInfo.getDocValuesType() != NONE` -- the five-way switch
    /// [`doc_values_field`] resolves.
    DocValues,
}

/// `FieldExistsQuery`'s source selection, verbatim: norms, then vectors, then
/// doc values, decided by `FieldInfo` alone.
///
/// `Err(Error::FieldExistsUnsupported)` is Java's own `IllegalStateException`
/// ("FieldExistsQuery requires that the field indexes doc values, norms or
/// vectors, but field '...' exists and indexes neither of these data
/// structures") -- an error rather than an empty match set, because a field
/// that indexes none of the three is a caller mistake, not a document-less
/// field. A field name that is not in `FieldInfos` at all is Java's
/// `fieldInfo == null`: `null` scorer, no matches, no error -- that case never
/// reaches this function, which takes a `FieldInfo`.
pub fn field_exists_source(
    field: &lucene_codecs::field_infos::FieldInfo,
) -> Result<FieldExistsSource> {
    // `FieldInfo.hasNorms()`.
    if field.index_options != lucene_codecs::field_infos::IndexOptions::None && !field.omit_norms {
        return Ok(FieldExistsSource::Norms);
    }
    if field.vector_dimension != 0 {
        return Ok(FieldExistsSource::Vectors);
    }
    if field.doc_values_type != lucene_codecs::field_infos::DocValuesType::None {
        return Ok(FieldExistsSource::DocValues);
    }
    Err(crate::Error::FieldExistsUnsupported(field.name.clone()))
}

/// `FieldExistsQuery` over the **norms** source: every live doc that has a
/// norm for the field, ascending, at a constant score -- Java's
/// `iterator = context.reader().getNormValues(field)` branch, wrapped by
/// `ConstantScoreScorerSupplier.fromIterator`.
///
/// Takes an already-open [`crate::field_norms::FieldNorms`] rather than raw
/// `.nvd` bytes so the sparse case is one forward `IndexedDISI` walk for the
/// whole sweep (`FieldNormsCursor`), not a fresh block walk per document --
/// the same reason every scored query in this crate holds a cursor. Docs are
/// visited ascending, which is the order that cursor is cheap in.
pub fn search_field_exists_norms<C: Collector>(
    norms: &crate::field_norms::FieldNorms<'_>,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    collector: &mut C,
) -> Result<()> {
    let mut cursor = norms.cursor();
    for doc_id in 0..max_doc {
        if !live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
            continue;
        }
        if cursor.has_norm(doc_id)? {
            collector.collect(doc_id);
        }
    }
    Ok(())
}

/// `FieldExistsQuery` over the **vectors** source: every live doc that has a
/// vector for the field, ascending, at a constant score -- Java's
/// `getFloatVectorValues(field).iterator()` / `getByteVectorValues(field)
/// .iterator()` branch. `ord_to_doc` is the same mapping either encoding's
/// iterator walks, so one function serves both (the encoding decides which
/// `*VectorValues` the caller opened, not what "has a value" means).
///
/// **This iterates ordinals, not documents.** A `KnnVectorValues` iterator
/// visits exactly the documents that have a vector -- `size()` of them -- and
/// their doc ids come back ascending because ordinals are assigned in document
/// order (`OrdToDocDISIReaderConfiguration`'s ordinal -> doc map is a
/// `DirectMonotonicReader`). Sweeping `0..max_doc` and asking each document
/// instead would be `max_doc` `IndexedDISI` lookups to find the same set, and
/// is what a naive port of the doc-values branch would have done here.
///
/// `size` and `ord_to_doc` come from
/// [`lucene_codecs::vectors::FloatVectorValues`]/`ByteVectorValues`, which is
/// why this takes the two closures rather than one of those two concrete
/// types: the crate's `vector_values_common!` macro gives them the same
/// inherent methods but no shared trait to be generic over.
pub fn search_field_exists_vectors<C: Collector>(
    size: i32,
    ord_to_doc: impl Fn(i32) -> std::result::Result<i32, lucene_codecs::vectors::Error>,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    collector: &mut C,
) -> Result<()> {
    for ord in 0..size {
        let doc = ord_to_doc(ord)?;
        // `ord_to_doc` validates the *ordinal*; the doc id it decoded is a
        // value off disk and a `DirectMonotonicReader` will happily return
        // one past the segment. Reported rather than skipped: skipping would
        // turn a corrupt ordinal->doc map into a quietly short result, and
        // `live_docs.get(doc)` a line below indexes a bitset with it.
        if doc < 0 || doc >= max_doc {
            return Err(crate::Error::Vectors(
                lucene_codecs::vectors::Error::CorruptMeta(format!(
                    "ordToDoc maps ordinal {ord} to doc {doc}, outside 0..{max_doc}"
                )),
            ));
        }
        if !live_docs.is_none_or(|bits| bits.get_doc(doc)) {
            continue;
        }
        collector.collect(doc);
    }
    Ok(())
}

/// `FieldExistsQuery.rewrite`'s "every document in this leaf has the field, so
/// the whole query is `MatchAllDocsQuery`" test, for one leaf.
///
/// Java's `rewrite` asks a different question per source, and the *counts* it
/// compares are the ones it can get without decoding anything:
///
/// - **norms**: `reader.getDocCount(field) == reader.maxDoc()` (the terms
///   dictionary's per-field `docCount`);
/// - **vectors**: `getFloatVectorValues(field).size() == leaf.maxDoc()`;
/// - **doc values**: any one of the terms dictionary's `docCount`, the BKD
///   tree's `getDocCount()`, or the doc-values skipper's `docCount()` equal to
///   `maxDoc` (a field always uses the same structures index-wide, so any of
///   the three that exists is authoritative).
///
/// **Note which `reader` Java uses.** The norms branch reads
/// `reader.getDocCount(field)` and `reader.maxDoc()` off the *top-level*
/// `IndexReader`, while the vector and doc-values branches read `leaf`'s --
/// inside the same per-leaf loop. That asymmetry is Java's, not a
/// simplification here, so a caller reproducing `rewrite` exactly passes the
/// **reader-wide** `terms_doc_count`/`max_doc` for [`FieldExistsSource::Norms`]
/// and the **leaf's** for the other two. (For a single-segment reader the two
/// coincide, which is presumably why it has survived.)
///
/// This function takes whichever of those counts the caller could resolve
/// (`None` for one it has no reader for) and applies Java's rule; the caller
/// still has to make the *whole-reader* decision, since Java rewrites only
/// when **every** leaf says yes.
///
/// Note the deliberate asymmetry Java has here: the doc-values branch is an
/// OR over three optional counts, so a leaf where none of the three is
/// available is *not* rewritable, while the norms and vector branches have a
/// single count each.
pub fn field_exists_leaf_is_complete(
    source: FieldExistsSource,
    max_doc: i32,
    terms_doc_count: Option<i32>,
    vector_size: Option<i32>,
    points_doc_count: Option<i32>,
    skipper_doc_count: Option<i32>,
) -> bool {
    match source {
        FieldExistsSource::Norms => terms_doc_count == Some(max_doc),
        FieldExistsSource::Vectors => vector_size == Some(max_doc),
        FieldExistsSource::DocValues => {
            terms_doc_count == Some(max_doc)
                || points_doc_count == Some(max_doc)
                || skipper_doc_count == Some(max_doc)
        }
    }
}

// ---------------------------------------------------------------------------
// `IndexOrDocValuesQuery`
// ---------------------------------------------------------------------------

/// Which side of an [`IndexOrDocValuesQuery`-equivalent][plan_index_or_doc_values]
/// pair to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexOrDocValuesPlan {
    /// Run the index (points / terms) query: a costly scorer to build, but a
    /// good lead iterator.
    Index,
    /// Run the doc-values query: cheap to start, good at verifying individual
    /// documents another clause is already leading.
    DocValues,
}

/// `IndexOrDocValuesQuery`'s planner, verbatim:
///
/// ```text
/// final long threshold = cost() >>> 3;
/// if (threshold <= leadCost) { indexScorerSupplier.get(leadCost) }
/// else                       { dvScorerSupplier.get(leadCost) }
/// ```
///
/// `cost()` is the *index* side's `ScorerSupplier.cost()`, which for a point
/// range query is `PointValues.estimateDocCount(visitor)` -- a BKD tree walk.
/// The `>>> 3` is Java's own comment: "at equal costs, doc values tend to be
/// worse than points since they still need to perform one comparison per
/// document while points can do much better than that given how values are
/// organized. So we give an arbitrary 8x penalty to doc values."
///
/// **`index_cost: None` means "no estimate available" and plans
/// [`IndexOrDocValuesPlan::Index`].** That is not a guess: it is the answer
/// Java's own rule gives whenever this query is the *lead* iterator
/// (`leadCost == cost()` makes `cost >>> 3 <= leadCost` true for every
/// non-negative cost), and it is unconditionally what `bulkScorer()` does
/// ("bulk scorers need to consume the entire set of docs, so using an index
/// structure should perform better"). Choosing doc values without an estimate
/// would be the guess -- it is only ever right when some *other* clause is
/// leading with a much smaller doc set, which is information the caller has
/// and this function is told about through `lead_cost`.
pub fn plan_index_or_doc_values(index_cost: Option<i64>, lead_cost: i64) -> IndexOrDocValuesPlan {
    match index_cost {
        None => IndexOrDocValuesPlan::Index,
        // `>>> 3` on a non-negative long is `/ 8`; a negative cost is not a
        // value Java can produce (`assert cost >= 0`), and clamping it here
        // keeps the shift's meaning rather than sign-extending.
        Some(cost) => {
            let threshold = cost.max(0) >> 3;
            if threshold <= lead_cost {
                IndexOrDocValuesPlan::Index
            } else {
                IndexOrDocValuesPlan::DocValues
            }
        }
    }
}

#[cfg(test)]
mod tests;
