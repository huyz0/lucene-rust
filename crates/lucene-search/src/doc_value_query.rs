//! Doc-values-driven query support: a numeric range filter and a sorted-ordinal
//! range/equality filter (`SortedNumericDocValuesField.newSlowRangeQuery`/
//! `SortedSetDocValuesField.newSlowRangeQuery`-equivalent, single-valued case
//! only — see [`crate::doc_value_query`]'s scope note below), plus a
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
//! - **A skip-list/DISI-driven range scorer.** [`search_numeric_range`] is a
//!   full `[0, max_doc)` sweep (see that function's doc comment) — Lucene's
//!   real doc-values range query can use the field's optional skip index
//!   (`Lucene90DocValuesSkipIndex`) to skip whole blocks that can't match;
//!   `doc_values.rs` now decodes that skip index
//!   ([`doc_values::parse_skip_index`][lucene_codecs::doc_values::parse_skip_index]),
//!   but this module doesn't consult it yet, so there's nothing wired up to
//!   skip against — a full sweep is the only option this module implements
//!   today.

use lucene_codecs::doc_values::{self, NumericEntry, SortedEntry, SortedNumericEntry};
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
/// skip index (unsupported, see module doc) there is no cheaper starting
/// point than checking every doc. This matches real Lucene's own fallback
/// behavior (`PointRangeQuery`'s "slow" `SortedNumericDocValuesField
/// .newSlowRangeQuery` path, used precisely when no BKD point index exists
/// for the field, as is the case in this port so far).
pub fn search_numeric_range<C: Collector>(
    doc_values_data: &[u8],
    entry: &NumericEntry,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    min: i64,
    max: i64,
    collector: &mut C,
) -> Result<()> {
    for doc_id in 0..max_doc {
        if !live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
            continue;
        }
        if let Some(value) = doc_values::numeric_value(doc_values_data, entry, doc_id)? {
            if value >= min && value <= max {
                collector.collect(doc_id);
            }
        }
    }
    Ok(())
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
    for doc_id in 0..max_doc {
        if !live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
            continue;
        }
        if let Some(ord) = doc_values::sorted_ord(doc_values_data, entry, doc_id)? {
            if ord >= min_ord && ord <= max_ord {
                collector.collect(doc_id);
            }
        }
    }
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
    for &doc_id in candidates {
        match doc_values::numeric_value(doc_values_data, entry, doc_id)? {
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
#[allow(clippy::too_many_arguments)]
pub fn search_numeric_range_sorted_by_field(
    doc_values_data: &[u8],
    range_entry: &NumericEntry,
    live_docs: Option<&FixedBitSet>,
    max_doc: i32,
    min: i64,
    max: i64,
    sort_entry: &NumericEntry,
    direction: SortDirection,
    missing: MissingValue,
    top_n: usize,
) -> Result<Vec<FieldValueDoc>> {
    let mut matches = crate::collector::VecCollector::default();
    search_numeric_range(
        doc_values_data,
        range_entry,
        live_docs,
        max_doc,
        min,
        max,
        &mut matches,
    )?;
    sort_top_n_by_numeric_doc_value(
        doc_values_data,
        sort_entry,
        &matches.docs,
        direction,
        missing,
        top_n,
    )
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

/// Multi-valued equivalent of [`search_numeric_range`]: reduces each doc's
/// values (from a [`SortedNumericEntry`] — a SORTED_NUMERIC field, or a
/// SORTED_SET field's ordinal array, see this module's doc comment) via
/// `selector`, then applies the exact same inclusive `[min, max]` check. A
/// doc with zero values (`sorted_numeric_values` returns an empty `Vec`)
/// never matches, same missing-value rule as [`search_numeric_range`].
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
        if !live_docs.is_none_or(|bits| bits.get(doc_id as usize)) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::VecCollector;
    use lucene_codecs::doc_values::{self as ndv, DocValuesMeta};

    // Reuses the same real-Lucene fixtures `crates/lucene-codecs/tests/
    // doc_values_fixtures.rs`/`sorted_doc_values_fixtures.rs` already open --
    // see the `test-coverage` skill's "prefer a real fixture over a hand-built
    // one wherever available" guidance.
    fn dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/doc_values_index/"
        )
        .to_string()
    }

    fn sorted_dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/sorted_dv_index/"
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

    fn load_dv_meta(dir: &str) -> (Manifest, [u8; 16], Vec<u8>, DocValuesMeta) {
        let manifest = Manifest::load(dir);
        let id = id_from_hex(manifest.get("id_hex"));
        let fnm = std::fs::read(format!("{dir}{}.raw", manifest.get("fnm_file_name"))).unwrap();
        let fis = lucene_codecs::field_infos::parse(&fnm, &id, "").unwrap();
        let meta_buf =
            std::fs::read(format!("{dir}{}.raw", manifest.get("dvm_file_name"))).unwrap();
        let data_buf =
            std::fs::read(format!("{dir}{}.raw", manifest.get("dvd_file_name"))).unwrap();
        let suffix = dv_suffix(&manifest);
        let (_, parsed) = ndv::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
        (manifest, id, data_buf, parsed)
    }

    // --- search_numeric_range: real-Lucene fixture tests ---
    // `varying` field values across 5 docs: -100, 7, 42, 1000, -3.

    #[test]
    fn numeric_range_matches_values_in_bounds_real_fixture() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();
        let mut c = VecCollector::default();
        // values -100,7,42,1000,-3 for docs 0..4; [-3, 42] matches 7 (doc 1),
        // 42 (doc 2), and -3 (doc 4).
        search_numeric_range(&data, entry, None, 5, -3, 42, &mut c).unwrap();
        assert_eq!(c.docs, vec![1, 2, 4]);
    }

    #[test]
    fn numeric_range_boundary_values_are_inclusive() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();
        let mut c = VecCollector::default();
        // Exact min (-100, doc 0) and exact max (1000, doc 3) must both match.
        search_numeric_range(&data, entry, None, 5, -100, 1000, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn numeric_range_excludes_values_outside_bounds() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();
        let mut c = VecCollector::default();
        // Just short of doc 2's value (42) on both ends.
        search_numeric_range(&data, entry, None, 5, 43, 999, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn numeric_range_sparse_missing_doc_never_matches() {
        // `sparse` field: 5, NONE, 15, NONE, 25 -- docs 1 and 3 have no value.
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "sparse"))
            .unwrap();
        let mut c = VecCollector::default();
        // An enormous range that would catch every value if missing docs
        // wrongly matched -- only docs 0, 2, 4 (5, 15, 25) may appear.
        search_numeric_range(&data, entry, None, 5, i64::MIN, i64::MAX, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 2, 4]);
    }

    #[test]
    fn numeric_range_live_docs_filters_matches() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();
        let mut live_docs = FixedBitSet::new(5);
        for i in 0..5 {
            live_docs.set(i);
        }
        live_docs.clear(4); // doc 4 (-3) would otherwise match
        let mut c = VecCollector::default();
        search_numeric_range(&data, entry, Some(&live_docs), 5, -100, 1000, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1, 2, 3]);
    }

    // --- search_numeric_range: hand-built unit tests for constant/empty shapes ---

    fn constant_entry(field_number: i32, value: i64, num_values: i64) -> NumericEntry {
        NumericEntry {
            field_number,
            docs_with_field_offset: -1, // dense
            docs_with_field_length: 0,
            jump_table_entry_count: -1,
            dense_rank_power: 0xFF,
            num_values,
            table: None,
            bits_per_value: 0,
            min_value: value,
            gcd: 1,
            values_offset: 0,
            values_length: 0,
            block_shift: None,
            value_jump_table_offset: 0,
        }
    }

    #[test]
    fn numeric_range_constant_value_field() {
        let entry = constant_entry(0, 7, 3);
        let mut c = VecCollector::default();
        search_numeric_range(&[], &entry, None, 3, 5, 10, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1, 2]);

        let mut c2 = VecCollector::default();
        search_numeric_range(&[], &entry, None, 3, 8, 10, &mut c2).unwrap();
        assert!(c2.docs.is_empty());
    }

    #[test]
    fn numeric_range_empty_field_never_matches() {
        let mut entry = constant_entry(0, 7, 0);
        entry.docs_with_field_offset = -2; // empty
        let mut c = VecCollector::default();
        search_numeric_range(&[], &entry, None, 3, i64::MIN, i64::MAX, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn numeric_range_zero_max_doc_matches_nothing() {
        let entry = constant_entry(0, 7, 0);
        let mut c = VecCollector::default();
        search_numeric_range(&[], &entry, None, 0, 0, 100, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn numeric_range_propagates_decode_errors() {
        // bits_per_value != 0 with an empty `data` slice can't decode -- the
        // underlying `doc_values::Error` must surface through `Result`, not
        // panic or silently match.
        let mut entry = constant_entry(0, 0, 1);
        entry.bits_per_value = 8;
        entry.values_offset = 0;
        entry.values_length = 1;
        let mut c = VecCollector::default();
        let err = search_numeric_range(&[], &entry, None, 1, 0, 100, &mut c).unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }

    // --- search_sorted_ord_range: real-Lucene fixture tests ---
    // `sorted` field ords across 5 docs: 1,0,2,0,1 -- terms apple(0),
    // banana(1), cherry(2).

    #[test]
    fn sorted_ord_range_equality_matches_only_that_term() {
        let (manifest, _id, data, meta) = load_dv_meta(&sorted_dv_dir());
        let entry = meta
            .sorted_entry(field_number(&manifest, "sorted"))
            .unwrap();
        let mut c = VecCollector::default();
        // ord 0 == "apple" -- docs 1 and 3.
        search_sorted_ord_range(&data, entry, None, 5, 0, 0, &mut c).unwrap();
        assert_eq!(c.docs, vec![1, 3]);
    }

    #[test]
    fn sorted_ord_range_covers_multiple_ordinals() {
        let (manifest, _id, data, meta) = load_dv_meta(&sorted_dv_dir());
        let entry = meta
            .sorted_entry(field_number(&manifest, "sorted"))
            .unwrap();
        let mut c = VecCollector::default();
        // ords 0..=1 ("apple"/"banana") -- every doc except doc 2 (cherry, ord 2).
        search_sorted_ord_range(&data, entry, None, 5, 0, 1, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 1, 3, 4]);
    }

    #[test]
    fn sorted_ord_range_out_of_bounds_matches_nothing() {
        let (manifest, _id, data, meta) = load_dv_meta(&sorted_dv_dir());
        let entry = meta
            .sorted_entry(field_number(&manifest, "sorted"))
            .unwrap();
        let mut c = VecCollector::default();
        search_sorted_ord_range(&data, entry, None, 5, 5, 10, &mut c).unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn sorted_ord_range_live_docs_filters_matches() {
        let (manifest, _id, data, meta) = load_dv_meta(&sorted_dv_dir());
        let entry = meta
            .sorted_entry(field_number(&manifest, "sorted"))
            .unwrap();
        let mut live_docs = FixedBitSet::new(5);
        for i in 0..5 {
            live_docs.set(i);
        }
        live_docs.clear(1); // one of the two ord-0 docs
        let mut c = VecCollector::default();
        search_sorted_ord_range(&data, entry, Some(&live_docs), 5, 0, 0, &mut c).unwrap();
        assert_eq!(c.docs, vec![3]);
    }

    // --- sort_by_numeric_doc_value ---

    #[test]
    fn sort_by_value_ascending_order_real_fixture() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();
        // All 5 docs as candidates; values are -100,7,42,1000,-3 for docs 0..4.
        let candidates = [0, 1, 2, 3, 4];
        let sorted =
            sort_by_numeric_doc_value(&data, entry, &candidates, MissingValue::Exclude).unwrap();
        assert_eq!(sorted, vec![(0, -100), (4, -3), (1, 7), (2, 42), (3, 1000)]);
    }

    #[test]
    fn sort_by_value_ties_break_by_ascending_doc_id() {
        let entry = constant_entry(0, 42, 5);
        // Every doc has the same value -- must come out in ascending doc-ID order.
        let candidates = [4, 1, 3, 0, 2];
        let sorted =
            sort_by_numeric_doc_value(&[], &entry, &candidates, MissingValue::Exclude).unwrap();
        assert_eq!(sorted, vec![(0, 42), (1, 42), (2, 42), (3, 42), (4, 42)]);
    }

    #[test]
    fn sort_by_value_missing_doc_excluded_by_default_policy() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "sparse"))
            .unwrap();
        // sparse: 5, NONE, 15, NONE, 25 -- docs 1 and 3 have no value.
        let candidates = [0, 1, 2, 3, 4];
        let sorted =
            sort_by_numeric_doc_value(&data, entry, &candidates, MissingValue::Exclude).unwrap();
        assert_eq!(sorted, vec![(0, 5), (2, 15), (4, 25)]);
    }

    #[test]
    fn sort_by_value_missing_doc_gets_default_value() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "sparse"))
            .unwrap();
        let candidates = [0, 1, 2, 3, 4];
        let sorted =
            sort_by_numeric_doc_value(&data, entry, &candidates, MissingValue::Default(1_000_000))
                .unwrap();
        // Docs 1 and 3 substitute 1_000_000, sorting last (ascending).
        assert_eq!(
            sorted,
            vec![(0, 5), (2, 15), (4, 25), (1, 1_000_000), (3, 1_000_000)]
        );
    }

    #[test]
    fn sort_by_value_empty_candidates_yields_empty_result() {
        let entry = constant_entry(0, 1, 0);
        let sorted = sort_by_numeric_doc_value(&[], &entry, &[], MissingValue::Exclude).unwrap();
        assert!(sorted.is_empty());
    }

    // --- sort_top_n_by_numeric_doc_value / search_numeric_range_sorted_by_field
    //     (TopFieldCollector-driven search-time sort) ---

    fn field_docs(v: &[(i32, i64)]) -> Vec<crate::collector::FieldValueDoc> {
        v.iter()
            .map(|&(doc_id, value)| crate::collector::FieldValueDoc { doc_id, value })
            .collect()
    }

    #[test]
    fn sort_top_n_ascending_order_real_fixture() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();
        // values -100,7,42,1000,-3 for docs 0..4.
        let candidates = [0, 1, 2, 3, 4];
        let top = sort_top_n_by_numeric_doc_value(
            &data,
            entry,
            &candidates,
            SortDirection::Ascending,
            MissingValue::Exclude,
            10,
        )
        .unwrap();
        assert_eq!(
            top,
            field_docs(&[(0, -100), (4, -3), (1, 7), (2, 42), (3, 1000)])
        );
    }

    #[test]
    fn sort_top_n_descending_order_real_fixture() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();
        let candidates = [0, 1, 2, 3, 4];
        let top = sort_top_n_by_numeric_doc_value(
            &data,
            entry,
            &candidates,
            SortDirection::Descending,
            MissingValue::Exclude,
            10,
        )
        .unwrap();
        assert_eq!(
            top,
            field_docs(&[(3, 1000), (2, 42), (1, 7), (4, -3), (0, -100)])
        );
    }

    #[test]
    fn sort_top_n_truncates_ascending_and_descending() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();
        let candidates = [0, 1, 2, 3, 4];
        let top_asc = sort_top_n_by_numeric_doc_value(
            &data,
            entry,
            &candidates,
            SortDirection::Ascending,
            MissingValue::Exclude,
            2,
        )
        .unwrap();
        assert_eq!(top_asc, field_docs(&[(0, -100), (4, -3)]));

        let top_desc = sort_top_n_by_numeric_doc_value(
            &data,
            entry,
            &candidates,
            SortDirection::Descending,
            MissingValue::Exclude,
            2,
        )
        .unwrap();
        assert_eq!(top_desc, field_docs(&[(3, 1000), (2, 42)]));
    }

    #[test]
    fn sort_top_n_tie_break_prefers_lower_doc_id() {
        // Every doc has the same value (constant NUMERIC field) -- ascending
        // and descending must both fall back to the ascending-doc-ID
        // tie-break for equal values.
        let entry = constant_entry(0, 42, 5);
        let candidates = [4, 1, 3, 0, 2];
        let top = sort_top_n_by_numeric_doc_value(
            &[],
            &entry,
            &candidates,
            SortDirection::Ascending,
            MissingValue::Exclude,
            3,
        )
        .unwrap();
        assert_eq!(top, field_docs(&[(0, 42), (1, 42), (2, 42)]));

        let top_desc = sort_top_n_by_numeric_doc_value(
            &[],
            &entry,
            &candidates,
            SortDirection::Descending,
            MissingValue::Exclude,
            3,
        )
        .unwrap();
        assert_eq!(top_desc, field_docs(&[(0, 42), (1, 42), (2, 42)]));
    }

    #[test]
    fn sort_top_n_missing_doc_excluded_by_default_policy() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "sparse"))
            .unwrap();
        // sparse: 5, NONE, 15, NONE, 25 -- docs 1 and 3 have no value.
        let candidates = [0, 1, 2, 3, 4];
        let top = sort_top_n_by_numeric_doc_value(
            &data,
            entry,
            &candidates,
            SortDirection::Ascending,
            MissingValue::Exclude,
            10,
        )
        .unwrap();
        assert_eq!(top, field_docs(&[(0, 5), (2, 15), (4, 25)]));
    }

    #[test]
    fn sort_top_n_missing_doc_gets_default_value() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        let entry = meta
            .numeric_entry(field_number(&manifest, "sparse"))
            .unwrap();
        let candidates = [0, 1, 2, 3, 4];
        let top = sort_top_n_by_numeric_doc_value(
            &data,
            entry,
            &candidates,
            SortDirection::Ascending,
            MissingValue::Default(1_000_000),
            10,
        )
        .unwrap();
        assert_eq!(
            top,
            field_docs(&[(0, 5), (2, 15), (4, 25), (1, 1_000_000), (3, 1_000_000)])
        );
    }

    #[test]
    fn sort_top_n_propagates_decode_errors() {
        let mut entry = constant_entry(0, 0, 1);
        entry.bits_per_value = 8;
        entry.values_offset = 0;
        entry.values_length = 1;
        let err = sort_top_n_by_numeric_doc_value(
            &[],
            &entry,
            &[0],
            SortDirection::Ascending,
            MissingValue::Exclude,
            10,
        )
        .unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }

    #[test]
    fn search_numeric_range_sorted_by_field_end_to_end_real_fixture() {
        let (manifest, _id, data, meta) = load_dv_meta(&dv_dir());
        // Query: "gcd" in [1000, 1100] -- gcd values 1000,1025,1075,1200,1050
        // for docs 0..4 match docs 0, 1, 2, 4 (doc 3's 1200 is out of range).
        let range_entry = meta.numeric_entry(field_number(&manifest, "gcd")).unwrap();
        // Sort the matches by "varying" (-100,7,42,1000,-3 for docs 0..4):
        // doc0=-100, doc1=7, doc2=42, doc4=-3.
        let sort_entry = meta
            .numeric_entry(field_number(&manifest, "varying"))
            .unwrap();

        let asc = search_numeric_range_sorted_by_field(
            &data,
            range_entry,
            None,
            5,
            1000,
            1100,
            sort_entry,
            SortDirection::Ascending,
            MissingValue::Exclude,
            10,
        )
        .unwrap();
        assert_eq!(asc, field_docs(&[(0, -100), (4, -3), (1, 7), (2, 42)]));

        let desc = search_numeric_range_sorted_by_field(
            &data,
            range_entry,
            None,
            5,
            1000,
            1100,
            sort_entry,
            SortDirection::Descending,
            MissingValue::Exclude,
            10,
        )
        .unwrap();
        assert_eq!(desc, field_docs(&[(2, 42), (1, 7), (4, -3), (0, -100)]));

        // Top-2 truncation on the same query, descending.
        let desc_top2 = search_numeric_range_sorted_by_field(
            &data,
            range_entry,
            None,
            5,
            1000,
            1100,
            sort_entry,
            SortDirection::Descending,
            MissingValue::Exclude,
            2,
        )
        .unwrap();
        assert_eq!(desc_top2, field_docs(&[(2, 42), (1, 7)]));
    }

    #[test]
    fn sort_by_value_propagates_decode_errors() {
        let mut entry = constant_entry(0, 0, 1);
        entry.bits_per_value = 8;
        entry.values_offset = 0;
        entry.values_length = 1;
        let err = sort_by_numeric_doc_value(&[], &entry, &[0], MissingValue::Exclude).unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }

    // --- ValueSelector::reduce: isolated selector-reduction tests ---

    #[test]
    fn selector_min_picks_smallest_of_multiple_values() {
        assert_eq!(ValueSelector::Min.reduce(&[5, 10]), Some(5));
        assert_eq!(ValueSelector::Min.reduce(&[7]), Some(7));
        assert_eq!(ValueSelector::Min.reduce(&[1, 2, 3]), Some(1));
    }

    #[test]
    fn selector_max_picks_largest_of_multiple_values() {
        assert_eq!(ValueSelector::Max.reduce(&[5, 10]), Some(10));
        assert_eq!(ValueSelector::Max.reduce(&[7]), Some(7));
        assert_eq!(ValueSelector::Max.reduce(&[1, 2, 3]), Some(3));
    }

    #[test]
    fn selector_reduce_of_no_values_is_none() {
        assert_eq!(ValueSelector::Min.reduce(&[]), None);
        assert_eq!(ValueSelector::Max.reduce(&[]), None);
    }

    // --- search_multi_valued_range / sort_by_multi_valued_doc_value:
    //     real-Lucene fixture tests (multi_valued_dv_index) ---
    // `nums` (SORTED_NUMERIC) values per doc 0..4: [5,10], NONE, [7], [1,2,3],
    // NONE -- so MIN per doc is 5, none, 7, 1, none and MAX is 10, none, 7, 3,
    // none. `tags` (SORTED_SET) ords per doc: [0,2], NONE, [1], [0], [1,2].

    fn multi_dv_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/multi_valued_dv_index/"
        )
        .to_string()
    }

    #[test]
    fn multi_range_min_selector_matches_docs_whose_min_falls_in_range() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        let mut c = VecCollector::default();
        // mins: doc0=5, doc2=7, doc3=1 -- [1,5] matches doc0 and doc3, not
        // doc2 (min 7 is out of range) even though doc2's only value (7) is
        // outside too -- proving MIN, not MAX, drives the match here.
        search_multi_valued_range(&data, entry, ValueSelector::Min, None, 5, 1, 5, &mut c).unwrap();
        assert_eq!(c.docs, vec![0, 3]);
    }

    #[test]
    fn multi_range_max_selector_matches_docs_whose_max_falls_in_range() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        let mut c = VecCollector::default();
        // maxes: doc0=10, doc2=7, doc3=3 -- [7,10] matches doc0 and doc2, not
        // doc3 (max 3 is out of range).
        search_multi_valued_range(&data, entry, ValueSelector::Max, None, 5, 7, 10, &mut c)
            .unwrap();
        assert_eq!(c.docs, vec![0, 2]);
    }

    #[test]
    fn multi_range_selector_determines_match_not_both_values() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        // doc0 = [5, 10]: MIN (5) falls in [4,6] but MAX (10) does not.
        let mut c_min = VecCollector::default();
        search_multi_valued_range(&data, entry, ValueSelector::Min, None, 5, 4, 6, &mut c_min)
            .unwrap();
        assert_eq!(c_min.docs, vec![0]);

        let mut c_max = VecCollector::default();
        search_multi_valued_range(&data, entry, ValueSelector::Max, None, 5, 4, 6, &mut c_max)
            .unwrap();
        assert!(c_max.docs.is_empty());

        // doc3 = [1, 2, 3]: MAX (3) falls in [3,3] but MIN (1) does not.
        let mut c_max2 = VecCollector::default();
        search_multi_valued_range(&data, entry, ValueSelector::Max, None, 5, 3, 3, &mut c_max2)
            .unwrap();
        assert_eq!(c_max2.docs, vec![3]);

        let mut c_min2 = VecCollector::default();
        search_multi_valued_range(&data, entry, ValueSelector::Min, None, 5, 3, 3, &mut c_min2)
            .unwrap();
        assert!(c_min2.docs.is_empty());
    }

    #[test]
    fn multi_range_boundary_values_are_inclusive() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        let mut c = VecCollector::default();
        // Exact MIN boundary (doc3's min == 1) and MAX boundary (doc0's max
        // == 10) must both match.
        search_multi_valued_range(&data, entry, ValueSelector::Min, None, 5, 1, 1, &mut c).unwrap();
        assert_eq!(c.docs, vec![3]);

        let mut c2 = VecCollector::default();
        search_multi_valued_range(&data, entry, ValueSelector::Max, None, 5, 10, 10, &mut c2)
            .unwrap();
        assert_eq!(c2.docs, vec![0]);
    }

    #[test]
    fn multi_range_missing_docs_never_match() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        let mut c = VecCollector::default();
        // docs 1 and 4 have zero values -- an unbounded range must not catch
        // them under either selector.
        search_multi_valued_range(
            &data,
            entry,
            ValueSelector::Min,
            None,
            5,
            i64::MIN,
            i64::MAX,
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![0, 2, 3]);
    }

    #[test]
    fn multi_range_live_docs_filters_matches() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        let mut live_docs = FixedBitSet::new(5);
        for i in 0..5 {
            live_docs.set(i);
        }
        live_docs.clear(0); // doc0's min (5) would otherwise match
        let mut c = VecCollector::default();
        search_multi_valued_range(
            &data,
            entry,
            ValueSelector::Min,
            Some(&live_docs),
            5,
            1,
            5,
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![3]);
    }

    #[test]
    fn multi_sort_min_selector_ascending_order_real_fixture() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        // Candidates excluding the two missing docs (1, 4); mins: doc0=5,
        // doc2=7, doc3=1.
        let candidates = [0, 2, 3];
        let sorted = sort_by_multi_valued_doc_value(
            &data,
            entry,
            ValueSelector::Min,
            &candidates,
            MissingValue::Exclude,
        )
        .unwrap();
        assert_eq!(sorted, vec![(3, 1), (0, 5), (2, 7)]);
    }

    #[test]
    fn multi_sort_max_selector_ascending_order_real_fixture() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        // maxes: doc0=10, doc2=7, doc3=3.
        let candidates = [0, 2, 3];
        let sorted = sort_by_multi_valued_doc_value(
            &data,
            entry,
            ValueSelector::Max,
            &candidates,
            MissingValue::Exclude,
        )
        .unwrap();
        assert_eq!(sorted, vec![(3, 3), (2, 7), (0, 10)]);
    }

    #[test]
    fn multi_sort_missing_doc_excluded_by_default_policy() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        let candidates = [0, 1, 2, 3, 4];
        let sorted = sort_by_multi_valued_doc_value(
            &data,
            entry,
            ValueSelector::Min,
            &candidates,
            MissingValue::Exclude,
        )
        .unwrap();
        assert_eq!(sorted, vec![(3, 1), (0, 5), (2, 7)]);
    }

    #[test]
    fn multi_sort_missing_doc_gets_default_value() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_numeric_entry(field_number(&manifest, "nums"))
            .unwrap();
        let candidates = [0, 1, 2, 3, 4];
        let sorted = sort_by_multi_valued_doc_value(
            &data,
            entry,
            ValueSelector::Min,
            &candidates,
            MissingValue::Default(1_000_000),
        )
        .unwrap();
        // Docs 1 and 4 substitute 1_000_000, sorting last (ascending), tied
        // and broken by ascending doc ID.
        assert_eq!(
            sorted,
            vec![(3, 1), (0, 5), (2, 7), (1, 1_000_000), (4, 1_000_000)]
        );
    }

    #[test]
    fn multi_sort_ties_break_by_ascending_doc_id() {
        // Every doc's only value is 42 (addresses: None collapses to exactly
        // one value per doc, same as a single-valued constant NUMERIC field)
        // -- MIN and MAX agree, so this isolates the tie-break rule.
        let entry = SortedNumericEntry {
            field_number: 0,
            numeric: constant_entry(0, 42, 5),
            num_docs_with_field: 5,
            addresses: None,
        };
        let candidates = [4, 1, 3, 0, 2];
        let sorted = sort_by_multi_valued_doc_value(
            &[],
            &entry,
            ValueSelector::Max,
            &candidates,
            MissingValue::Exclude,
        )
        .unwrap();
        assert_eq!(sorted, vec![(0, 42), (1, 42), (2, 42), (3, 42), (4, 42)]);
    }

    #[test]
    fn multi_sort_empty_candidates_yields_empty_result() {
        let entry = SortedNumericEntry {
            field_number: 0,
            numeric: constant_entry(0, 1, 0),
            num_docs_with_field: 0,
            addresses: None,
        };
        let sorted = sort_by_multi_valued_doc_value(
            &[],
            &entry,
            ValueSelector::Min,
            &[],
            MissingValue::Exclude,
        )
        .unwrap();
        assert!(sorted.is_empty());
    }

    #[test]
    fn multi_range_propagates_decode_errors() {
        let mut numeric = constant_entry(0, 0, 1);
        numeric.bits_per_value = 8;
        numeric.values_offset = 0;
        numeric.values_length = 1;
        let entry = SortedNumericEntry {
            field_number: 0,
            numeric,
            num_docs_with_field: 1,
            addresses: None,
        };
        let mut c = VecCollector::default();
        let err =
            search_multi_valued_range(&[], &entry, ValueSelector::Min, None, 1, 0, 100, &mut c)
                .unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }

    #[test]
    fn multi_sort_propagates_decode_errors() {
        let mut numeric = constant_entry(0, 0, 1);
        numeric.bits_per_value = 8;
        numeric.values_offset = 0;
        numeric.values_length = 1;
        let entry = SortedNumericEntry {
            field_number: 0,
            numeric,
            num_docs_with_field: 1,
            addresses: None,
        };
        let err = sort_by_multi_valued_doc_value(
            &[],
            &entry,
            ValueSelector::Min,
            &[0],
            MissingValue::Exclude,
        )
        .unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }

    // --- search_multi_valued_range / sort_by_multi_valued_doc_value applied
    //     to a SORTED_SET field's ordinals (`tags`, real fixture) --
    //     confirms the same functions serve SORTED_SET's multi-valued case
    //     (see this module's doc comment) since ords are just i64s. ---

    #[test]
    fn multi_range_over_sorted_set_ordinals_real_fixture() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_set_entry(field_number(&manifest, "tags"))
            .unwrap();
        let ords_entry = match &entry.kind {
            doc_values::SortedSetKind::Multi { ords, .. } => ords,
            doc_values::SortedSetKind::Single(_) => panic!("expected a multi-valued SORTED_SET"),
        };
        // tags ords: doc0=[0,2], doc1=NONE, doc2=[1], doc3=[0], doc4=[1,2].
        // MIN ord 0 matches doc0 (min 0) and doc3 (min 0, its only ord).
        let mut c = VecCollector::default();
        search_multi_valued_range(&data, ords_entry, ValueSelector::Min, None, 5, 0, 0, &mut c)
            .unwrap();
        assert_eq!(c.docs, vec![0, 3]);

        // MAX ord 0 matches only doc3 (doc0's max ord is 2).
        let mut c2 = VecCollector::default();
        search_multi_valued_range(
            &data,
            ords_entry,
            ValueSelector::Max,
            None,
            5,
            0,
            0,
            &mut c2,
        )
        .unwrap();
        assert_eq!(c2.docs, vec![3]);
    }

    #[test]
    fn multi_sort_over_sorted_set_ordinals_real_fixture() {
        let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
        let entry = meta
            .sorted_set_entry(field_number(&manifest, "tags"))
            .unwrap();
        let ords_entry = match &entry.kind {
            doc_values::SortedSetKind::Multi { ords, .. } => ords,
            doc_values::SortedSetKind::Single(_) => panic!("expected a multi-valued SORTED_SET"),
        };
        // MIN ords: doc0=0, doc2=1, doc3=0, doc4=1 -- doc1 missing.
        let candidates = [0, 1, 2, 3, 4];
        let sorted = sort_by_multi_valued_doc_value(
            &data,
            ords_entry,
            ValueSelector::Min,
            &candidates,
            MissingValue::Exclude,
        )
        .unwrap();
        // Ascending by ord, ties (0 and 0; 1 and 1) broken by doc ID.
        assert_eq!(sorted, vec![(0, 0), (3, 0), (2, 1), (4, 1)]);
    }
}
