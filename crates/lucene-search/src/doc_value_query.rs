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
//!
//! **Deliberately out of scope** (tracked in `docs/parity.md`):
//! - **SORTED_NUMERIC/SORTED_SET (multi-valued) range/sort.** Real Lucene
//!   resolves a multi-valued field to a single comparable value via a
//!   `SortedNumericSelector`/`SortedSetSelector` (MIN/MAX/first/etc.) before
//!   sorting or range-filtering on it — that selector concept doesn't exist
//!   in this port yet. `sorted_numeric_values` (already read-side complete in
//!   `lucene-codecs`) is the building block a future slice would add a
//!   selector on top of; this slice only takes on the single-valued NUMERIC/
//!   SORTED case, which is the "real search sorting" core the task asked for.
//! - **Descending sort / multiple sort fields / secondary sort keys** beyond
//!   the single documented tie-break (ascending doc ID). Real `Sort`
//!   composes multiple `SortField`s; this is a single-key sort only.
//! - **A skip-list/DISI-driven range scorer.** [`search_numeric_range`] is a
//!   full `[0, max_doc)` sweep (see that function's doc comment) — Lucene's
//!   real doc-values range query can use the field's optional skip index
//!   (`Lucene90DocValuesSkipIndex`) to skip whole blocks that can't match;
//!   this port doesn't parse that skip index at all yet (see
//!   `doc_values.rs`'s `Error::UnsupportedSkipIndex`), so there's nothing to
//!   skip against — a full sweep is the only correct option available.

use lucene_codecs::doc_values::{self, NumericEntry, SortedEntry};
use lucene_util::fixed_bit_set::FixedBitSet;

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

    #[test]
    fn sort_by_value_propagates_decode_errors() {
        let mut entry = constant_entry(0, 0, 1);
        entry.bits_per_value = 8;
        entry.values_offset = 0;
        entry.values_length = 1;
        let err = sort_by_numeric_doc_value(&[], &entry, &[0], MissingValue::Exclude).unwrap_err();
        assert!(matches!(err, crate::Error::DocValues(_)));
    }
}
