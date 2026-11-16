#![cfg(test)]
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

// --- `FieldExistsQuery` ------------------------------------------------

/// The docs a manifest's `field.<x>.values`-style list says have a value:
/// every position whose entry is not `NONE`.
fn docs_with_values(list: &str) -> Vec<i32> {
    list.split(';')
        .enumerate()
        .filter(|(_, v)| *v != "NONE")
        .map(|(i, _)| i as i32)
        .collect()
}

#[test]
fn field_exists_matches_every_doc_with_a_value_for_all_five_dv_kinds() {
    // NUMERIC (sparse) and BINARY (sparse), from doc_values_index.
    let (m, _id, data, meta) = load_dv_meta(&dv_dir());
    for (field, key, expected) in [
        ("sparse", "field.sparse.values", vec![0i32, 2, 4]),
        ("bin_sparse", "field.bin_sparse.values_hex", vec![0, 2, 4]),
    ] {
        let f = doc_values_field(&meta, field_number(&m, field)).expect("a dv entry");
        let mut c = VecCollector::default();
        search_field_exists(&data, f, None, 5, &mut c).unwrap();
        assert_eq!(c.docs, expected, "{field}");
        assert_eq!(
            c.docs,
            m.get(key)
                .split(',')
                .enumerate()
                .filter(|(_, v)| *v != "NONE")
                .map(|(i, _)| i as i32)
                .collect::<Vec<_>>(),
            "{field}: manifest and port must agree on which docs have a value"
        );
    }
    // A dense field matches every doc.
    let f = doc_values_field(&meta, field_number(&m, "varying")).expect("a dv entry");
    let mut c = VecCollector::default();
    search_field_exists(&data, f, None, 5, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1, 2, 3, 4]);
}

#[test]
fn field_exists_covers_sorted_sorted_numeric_and_sorted_set() {
    // SORTED (dense).
    let (m, _id, data, meta) = load_dv_meta(&sorted_dv_dir());
    let f = doc_values_field(&meta, field_number(&m, "sorted")).expect("a dv entry");
    let mut c = VecCollector::default();
    search_field_exists(&data, f, None, 5, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 1, 2, 3, 4]);

    // SORTED_NUMERIC and SORTED_SET (both sparse) from the multi-valued
    // fixture: docs 1 and 4 have no `nums`, docs 1 has no `tags`.
    let (m, _id, data, meta) = load_dv_meta(&multi_dv_dir());
    let f = doc_values_field(&meta, field_number(&m, "nums")).expect("a dv entry");
    let mut c = VecCollector::default();
    search_field_exists(&data, f, None, 5, &mut c).unwrap();
    assert_eq!(c.docs, docs_with_values(m.get("field.nums.values")));

    let f = doc_values_field(&meta, field_number(&m, "tags")).expect("a dv entry");
    let mut c = VecCollector::default();
    search_field_exists(&data, f, None, 5, &mut c).unwrap();
    assert_eq!(c.docs, docs_with_values(m.get("field.tags.ords")));
}

#[test]
fn field_exists_respects_live_docs_and_an_unknown_field() {
    let (m, _id, data, meta) = load_dv_meta(&dv_dir());
    let f = doc_values_field(&meta, field_number(&m, "sparse")).expect("a dv entry");
    let mut live = FixedBitSet::new(5);
    for d in [0usize, 1, 2, 3, 4] {
        live.set(d);
    }
    live.clear(2);
    let mut c = VecCollector::default();
    search_field_exists(&data, f, Some(&live), 5, &mut c).unwrap();
    assert_eq!(c.docs, vec![0, 4], "a deleted doc never matches");

    // Java's `fieldInfo == null` / `DocValuesType.NONE`: a null iterator,
    // i.e. no matches -- not an error.
    assert!(doc_values_field(&meta, 9999).is_none());
}

#[test]
fn doc_has_value_agrees_with_the_iterator_document_by_document() {
    let (m, _id, data, meta) = load_dv_meta(&multi_dv_dir());
    let f = doc_values_field(&meta, field_number(&m, "nums")).expect("a dv entry");
    let mut c = VecCollector::default();
    search_field_exists(&data, f, None, 5, &mut c).unwrap();
    for doc in 0..5 {
        assert_eq!(
            doc_has_value(&data, f, doc).unwrap(),
            c.docs.contains(&doc),
            "doc {doc}"
        );
    }
}

// --- `IndexOrDocValuesQuery` -------------------------------------------

#[test]
fn the_planner_reproduces_javas_eight_times_penalty() {
    // `threshold = cost >>> 3; threshold <= leadCost ? index : doc values`.
    assert_eq!(
        plan_index_or_doc_values(Some(800), 100),
        IndexOrDocValuesPlan::Index,
        "800/8 == 100 <= 100: the index side wins on the boundary"
    );
    assert_eq!(
        plan_index_or_doc_values(Some(807), 100),
        IndexOrDocValuesPlan::Index,
        "807 >>> 3 == 100 (the shift truncates) -- still the index side"
    );
    assert_eq!(
        plan_index_or_doc_values(Some(808), 100),
        IndexOrDocValuesPlan::DocValues
    );
    // The lead-iterator case: `leadCost == cost` always chooses the index.
    for cost in [0i64, 1, 7, 8, 1_000_000, i64::MAX] {
        assert_eq!(
            plan_index_or_doc_values(Some(cost), cost),
            IndexOrDocValuesPlan::Index,
            "cost {cost} as its own lead cost"
        );
    }
}

#[test]
fn the_planner_defaults_to_the_index_side_without_an_estimate() {
    // No `PointValues.estimateDocCount` exists in this port, so a caller
    // with no estimate gets Java's own lead-iterator/bulk-scorer answer.
    assert_eq!(
        plan_index_or_doc_values(None, 0),
        IndexOrDocValuesPlan::Index
    );
    assert_eq!(
        plan_index_or_doc_values(None, i64::MAX),
        IndexOrDocValuesPlan::Index
    );
}

#[test]
fn the_planner_clamps_a_nonsensical_negative_cost() {
    // Java asserts `cost >= 0`; a negative one must not sign-extend into a
    // huge threshold and silently flip the plan.
    assert_eq!(
        plan_index_or_doc_values(Some(-1), 0),
        IndexOrDocValuesPlan::Index
    );
}

fn load_dv_meta(dir: &str) -> (Manifest, [u8; 16], Vec<u8>, DocValuesMeta) {
    let manifest = Manifest::load(dir);
    let id = id_from_hex(manifest.get("id_hex"));
    let fnm = std::fs::read(format!("{dir}{}.raw", manifest.get("fnm_file_name"))).unwrap();
    let fis = lucene_codecs::field_infos::parse(&fnm, &id, "").unwrap();
    let meta_buf = std::fs::read(format!("{dir}{}.raw", manifest.get("dvm_file_name"))).unwrap();
    let data_buf = std::fs::read(format!("{dir}{}.raw", manifest.get("dvd_file_name"))).unwrap();
    let suffix = dv_suffix(&manifest);
    let (_, parsed) = ndv::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
    (manifest, id, data_buf, parsed)
}

// --- search_numeric_range_with_skip_index: real-Lucene fixture tests ---
// `fixtures/data/doc_values_skip_index/` 36000
// docs, field "skip_numeric" with a strictly-increasing value
// `i*7 - 3`, giving a 2-level skip index (see GenDocValuesSkipIndex.java).

fn skip_dv_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/doc_values_skip_index/"
    )
    .to_string()
}

/// Loads the skip-index fixture's `.dvd` data, `NumericEntry`, and
/// decoded [`ndv::DocValuesSkipIndex`] together.
fn load_skip_index_fixture() -> (i32, Vec<u8>, NumericEntry, ndv::DocValuesSkipIndex) {
    let dir = skip_dv_dir();
    let (manifest, id, data, meta) = load_dv_meta(&dir);
    let field_number: i32 = manifest.get("field_number").parse().unwrap();
    let entry = meta.numeric_entry(field_number).unwrap().clone();
    let skipper_meta = meta.skipper_meta(field_number).copied().unwrap();
    let suffix = dv_suffix(&manifest);
    let dvs = std::fs::read(format!("{dir}{}.raw", manifest.get("dvs_file_name"))).unwrap();
    let skip_index = ndv::parse_skip_index(&dvs, &id, &suffix, &skipper_meta).unwrap();
    let max_doc: i32 = manifest.get("max_doc").parse().unwrap();
    (max_doc, data, entry, skip_index)
}

/// Asserts that the skip-index-driven scan and the full sweep agree on
/// `[min, max]`, and returns `(matches, decoded)` so callers can also
/// assert on how many values got decoded.
fn assert_skip_matches_full_sweep(min: i64, max: i64) -> (Vec<i32>, usize) {
    let (max_doc, data, entry, skip_index) = load_skip_index_fixture();

    let mut full = VecCollector::default();
    search_numeric_range(&data, &entry, None, max_doc, min, max, &mut full).unwrap();

    let mut skip = VecCollector::default();
    let decoded = search_numeric_range_with_skip_index(
        &data,
        &entry,
        &skip_index,
        None,
        max_doc,
        min,
        max,
        &mut skip,
    )
    .unwrap();

    assert_eq!(
        full.docs, skip.docs,
        "skip-index scan disagreed with full sweep for [{min}, {max}]"
    );
    (skip.docs, decoded)
}

#[test]
fn skip_index_matches_full_sweep_narrow_range() {
    // value = i*7 - 3, so value == 42 <=> i == (42+3)/7 == 45/7, not
    // integral -- pick an exact hit instead: i=100 -> value 697.
    let (docs, decoded) = assert_skip_matches_full_sweep(697, 697);
    assert_eq!(docs, vec![100]);
    // Only the base interval(s) whose value range covers 697 should ever
    // get decoded -- nowhere near all 36000 docs. Base interval size is
    // 4096 docs, so a single-value hit should decode at most one or two
    // intervals' worth (up to 2*4096), not a meaningful fraction of the
    // field.
    assert!(
        decoded <= 2 * 4096,
        "expected skip index to decode only a couple of base intervals, decoded={decoded}"
    );
}

#[test]
fn skip_index_matches_full_sweep_range_spanning_one_base_interval() {
    // Base interval size is 4096 docs; a range comfortably inside a
    // single 4096-doc interval (e.g. docs ~0..50, values -3..346).
    let (docs, decoded) = assert_skip_matches_full_sweep(-3, 346);
    assert_eq!(docs.len(), 50);
    assert!(
        decoded < 36000 / 8,
        "expected far fewer than 1/8 of all docs decoded, decoded={decoded}"
    );
}

#[test]
fn skip_index_matches_full_sweep_matches_everything() {
    let (docs, decoded) = assert_skip_matches_full_sweep(i64::MIN, i64::MAX);
    assert_eq!(docs.len(), 36000);
    // An unbounded range can't skip anything -- every doc still gets
    // decoded, same as the full sweep.
    assert_eq!(decoded, 36000);
}

#[test]
fn skip_index_matches_full_sweep_matches_nothing() {
    // Field's value range is [-3, 251990] (see manifest) -- anything
    // above the max can never match.
    let (docs, decoded) = assert_skip_matches_full_sweep(1_000_000, 2_000_000);
    assert!(docs.is_empty());
    // No interval's value range can intersect -- nothing gets decoded at
    // all, proving whole-interval skipping (not just fewer per-doc
    // hits).
    assert_eq!(
        decoded, 0,
        "expected zero decodes when no interval can match"
    );
}

#[test]
fn skip_index_matches_full_sweep_live_docs_filters() {
    let (max_doc, data, entry, skip_index) = load_skip_index_fixture();
    let mut live_docs = FixedBitSet::new(max_doc as usize);
    for i in 0..max_doc as usize {
        live_docs.set(i);
    }
    live_docs.clear(100); // doc 100 (value 697) would otherwise match

    // value = i*7 - 3: only i=100 (697) falls in [691, 703] -- i=99
    // (690) and i=101 (704) both fall just outside.
    let mut full = VecCollector::default();
    search_numeric_range(
        &data,
        &entry,
        Some(&live_docs),
        max_doc,
        691,
        703,
        &mut full,
    )
    .unwrap();
    let mut skip = VecCollector::default();
    search_numeric_range_with_skip_index(
        &data,
        &entry,
        &skip_index,
        Some(&live_docs),
        max_doc,
        691,
        703,
        &mut skip,
    )
    .unwrap();
    assert_eq!(full.docs, skip.docs);
    assert!(skip.docs.is_empty());
}

#[test]
fn skip_index_propagates_decode_errors() {
    // A skip index whose lone interval's value range does intersect
    // [min, max] forces the decode path to actually run against a
    // deliberately-broken NumericEntry, surfacing the underlying
    // `doc_values::Error` through `Result` rather than panicking.
    let mut entry = constant_entry(0, 0, 1);
    entry.bits_per_value = 8;
    entry.values_offset = 0;
    entry.values_length = 1;
    let skip_index = ndv::DocValuesSkipIndex {
        min_value: 0,
        max_value: 0,
        doc_count: 1,
        max_doc_id: 0,
        max_value_count: 1,
        intervals: vec![ndv::SkipIndexInterval {
            levels: vec![ndv::SkipIndexLevelInterval {
                min_doc_id: 0,
                max_doc_id: 0,
                min_value: 0,
                max_value: 0,
                doc_count: 1,
            }],
        }],
    };
    let mut c = VecCollector::default();
    let err =
        search_numeric_range_with_skip_index(&[], &entry, &skip_index, None, 1, 0, 100, &mut c)
            .unwrap_err();
    assert!(matches!(err, crate::Error::DocValues(_)));
}

#[test]
fn skip_index_empty_intervals_matches_nothing() {
    let skip_index = ndv::DocValuesSkipIndex {
        min_value: 0,
        max_value: 0,
        doc_count: 0,
        max_doc_id: -1,
        max_value_count: 0,
        intervals: vec![],
    };
    let entry = constant_entry(0, 7, 3);
    let mut c = VecCollector::default();
    let decoded =
        search_numeric_range_with_skip_index(&[], &entry, &skip_index, None, 3, 0, 100, &mut c)
            .unwrap();
    assert_eq!(decoded, 0);
    assert!(c.docs.is_empty());
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
        adaptive: None,
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
        &data,
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
        &data,
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
        &data,
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

// --- `search_sorted_numeric_range` (`SortedNumericDocValuesRangeQuery`) ---

#[test]
fn sorted_numeric_range_matches_a_doc_with_any_value_in_range() {
    let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
    let entry = meta
        .sorted_numeric_entry(field_number(&manifest, "nums"))
        .unwrap();
    // nums: doc0=[5,10], doc1=NONE, doc2=[7], doc3=[1,2,3], doc4=NONE.
    // [6, 8] catches doc0 (via 10? no -- via nothing below 6 except 5;
    // the first value >= 6 is 10, which is > 8) ... so only doc2 (7).
    let mut c = VecCollector::default();
    search_sorted_numeric_range(&data, entry, None, 5, 6, 8, &mut c).unwrap();
    assert_eq!(c.docs, vec![2]);

    // [9, 20] catches doc0 through its *second* value alone.
    let mut c = VecCollector::default();
    search_sorted_numeric_range(&data, entry, None, 5, 9, 20, &mut c).unwrap();
    assert_eq!(c.docs, vec![0]);

    // [2, 2] catches doc3 through a value that is neither its min nor its
    // max -- the case no `ValueSelector` can express.
    let mut c = VecCollector::default();
    search_sorted_numeric_range(&data, entry, None, 5, 2, 2, &mut c).unwrap();
    assert_eq!(c.docs, vec![3]);
}

#[test]
fn sorted_numeric_range_differs_from_both_selectors() {
    // The concrete divergence this function exists to fix: doc0 = [5, 10]
    // against [9, 20]. Java's `SortedNumericDocValuesRangeQuery` matches
    // it (10 is in range); neither MIN (5) nor MAX... MAX does match here,
    // so use [4, 6] as the MAX counter-example and [9, 20] as the MIN one.
    let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
    let entry = meta
        .sorted_numeric_entry(field_number(&manifest, "nums"))
        .unwrap();

    // [9, 20]: any-value matches doc0; MIN (5) does not.
    let mut any = VecCollector::default();
    search_sorted_numeric_range(&data, entry, None, 5, 9, 20, &mut any).unwrap();
    let mut min = VecCollector::default();
    search_multi_valued_range(&data, entry, ValueSelector::Min, None, 5, 9, 20, &mut min).unwrap();
    assert_eq!(any.docs, vec![0]);
    assert!(min.docs.is_empty());

    // [4, 6]: any-value matches doc0 (5); MAX (10) does not.
    let mut any = VecCollector::default();
    search_sorted_numeric_range(&data, entry, None, 5, 4, 6, &mut any).unwrap();
    let mut max = VecCollector::default();
    search_multi_valued_range(&data, entry, ValueSelector::Max, None, 5, 4, 6, &mut max).unwrap();
    assert_eq!(any.docs, vec![0]);
    assert!(max.docs.is_empty());
}

#[test]
fn sorted_numeric_range_never_matches_a_doc_with_no_values() {
    let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
    let entry = meta
        .sorted_numeric_entry(field_number(&manifest, "nums"))
        .unwrap();
    let mut c = VecCollector::default();
    search_sorted_numeric_range(&data, entry, None, 5, i64::MIN, i64::MAX, &mut c).unwrap();
    // docs 1 and 4 have no values at all.
    assert_eq!(c.docs, vec![0, 2, 3]);
}

#[test]
fn sorted_numeric_range_inverted_bounds_match_nothing() {
    // `SortedNumericDocValuesField.newSlowRangeQuery` returns a
    // `MatchNoDocsQuery` when `lowerValue > upperValue`.
    let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
    let entry = meta
        .sorted_numeric_entry(field_number(&manifest, "nums"))
        .unwrap();
    let mut c = VecCollector::default();
    search_sorted_numeric_range(&data, entry, None, 5, 10, 1, &mut c).unwrap();
    assert!(c.docs.is_empty());
}

#[test]
fn sorted_numeric_range_respects_live_docs() {
    let (manifest, _id, data, meta) = load_dv_meta(&multi_dv_dir());
    let entry = meta
        .sorted_numeric_entry(field_number(&manifest, "nums"))
        .unwrap();
    let mut live = FixedBitSet::new(5);
    for i in 0..5 {
        live.set(i);
    }
    live.clear(0);
    let mut c = VecCollector::default();
    search_sorted_numeric_range(&data, entry, Some(&live), 5, i64::MIN, i64::MAX, &mut c).unwrap();
    assert_eq!(c.docs, vec![2, 3]);
}

#[test]
fn sorted_numeric_range_propagates_decode_errors() {
    let numeric = doc_values::NumericEntry {
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
        adaptive: None,
    };
    let entry = SortedNumericEntry {
        field_number: 0,
        numeric,
        num_docs_with_field: 1,
        addresses: None,
    };
    let mut c = VecCollector::default();
    let err = search_sorted_numeric_range(&[], &entry, None, 1, 0, 10, &mut c).unwrap_err();
    assert!(matches!(err, crate::Error::DocValues(_)));
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
    search_multi_valued_range(&data, entry, ValueSelector::Max, None, 5, 7, 10, &mut c).unwrap();
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
    search_multi_valued_range(&data, entry, ValueSelector::Min, None, 5, 4, 6, &mut c_min).unwrap();
    assert_eq!(c_min.docs, vec![0]);

    let mut c_max = VecCollector::default();
    search_multi_valued_range(&data, entry, ValueSelector::Max, None, 5, 4, 6, &mut c_max).unwrap();
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
    search_multi_valued_range(&data, entry, ValueSelector::Max, None, 5, 10, 10, &mut c2).unwrap();
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
    let sorted =
        sort_by_multi_valued_doc_value(&[], &entry, ValueSelector::Min, &[], MissingValue::Exclude)
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
    let err = search_multi_valued_range(&[], &entry, ValueSelector::Min, None, 1, 0, 100, &mut c)
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

// -----------------------------------------------------------------
// `FieldExistsQuery`: the norms and vector sources
// -----------------------------------------------------------------

fn manifest_kv(dir: &str) -> std::collections::HashMap<String, String> {
    let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run scripts/gen-fixtures.sh first");
    text.lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn hex_id(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn field_info(
    name: &str,
    number: i32,
    index_options: lucene_codecs::field_infos::IndexOptions,
    omit_norms: bool,
    vector_dimension: i32,
    doc_values_type: lucene_codecs::field_infos::DocValuesType,
) -> lucene_codecs::field_infos::FieldInfo {
    lucene_codecs::field_infos::FieldInfo {
        name: name.to_string(),
        number,
        store_term_vectors: false,
        omit_norms,
        store_payloads: false,
        soft_deletes_field: false,
        parent_field: false,
        index_options,
        doc_values_type,
        doc_values_skip_index_type: lucene_codecs::field_infos::DocValuesSkipIndexType::None,
        doc_values_gen: -1,
        attributes: Vec::new(),
        point_dimension_count: 0,
        point_index_dimension_count: 0,
        point_num_bytes: 0,
        vector_dimension,
        vector_encoding: lucene_codecs::field_infos::VectorEncoding::Float32,
        vector_similarity_function: lucene_codecs::field_infos::VectorSimilarityFunction::Euclidean,
    }
}

/// Java's three-way if/else, including the order: a field with *both*
/// norms and doc values is answered from its norms, and the
/// "indexes neither" case is an error, not an empty result.
#[test]
fn field_exists_source_follows_javas_field_info_order() {
    use lucene_codecs::field_infos::{DocValuesType, IndexOptions};

    // Norms wins over doc values when a field has both.
    let both = field_info("b", 0, IndexOptions::Docs, false, 0, DocValuesType::Numeric);
    assert_eq!(
        field_exists_source(&both).unwrap(),
        FieldExistsSource::Norms
    );

    // omitNorms turns the same field into a doc-values source.
    let omitted = field_info("b", 0, IndexOptions::Docs, true, 0, DocValuesType::Numeric);
    assert_eq!(
        field_exists_source(&omitted).unwrap(),
        FieldExistsSource::DocValues
    );

    // Not indexed at all, but has vectors *and* doc values: vectors win.
    let vectors = field_info("v", 1, IndexOptions::None, true, 16, DocValuesType::Numeric);
    assert_eq!(
        field_exists_source(&vectors).unwrap(),
        FieldExistsSource::Vectors
    );

    // A vector field is not a norms field even when it is indexed with
    // norms enabled -- Java checks `hasNorms()` first, so this is Norms.
    let indexed_vectors = field_info("v", 1, IndexOptions::Docs, false, 16, DocValuesType::None);
    assert_eq!(
        field_exists_source(&indexed_vectors).unwrap(),
        FieldExistsSource::Norms
    );

    // Indexed with omitNorms, no vectors, no doc values: Java throws.
    let neither = field_info("n", 2, IndexOptions::Docs, true, 0, DocValuesType::None);
    let err = field_exists_source(&neither).unwrap_err();
    assert!(matches!(err, crate::Error::FieldExistsUnsupported(f) if f == "n"));
}

/// The norms source against real Lucene-written norms, for both a dense
/// field (every doc has a norm) and a sparse one (the fixture's manifest
/// records `NONE` for the documents that do not).
#[test]
fn field_exists_over_norms_matches_the_fixtures_own_norm_values() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/norms_index/"
    );
    let kv = manifest_kv(dir);
    let id = hex_id(&kv["id_hex"]);
    let max_doc: i32 = kv["max_doc"].parse().unwrap();
    let nvm = std::fs::read(format!("{dir}{}", kv["nvm_file_name"])).unwrap();
    let nvd = std::fs::read(format!("{dir}{}", kv["nvd_file_name"])).unwrap();
    let (_version, meta) = lucene_codecs::norms::parse_meta(&nvm, &id, "").unwrap();

    for field in ["body", "sparse_body"] {
        let number: i32 = kv[&format!("field.{field}.number")].parse().unwrap();
        let expected: Vec<bool> = kv[&format!("field.{field}.norm_values")]
            .split(',')
            .map(|v| v != "NONE")
            .collect();
        assert_eq!(expected.len(), max_doc as usize);

        let entry = meta.entry(number).expect("a norms entry for the field");
        let norms = crate::field_norms::FieldNorms::open(&nvd, *entry, max_doc, None).unwrap();
        let mut docs = crate::collector::VecCollector::default();
        search_field_exists_norms(&norms, None, max_doc, &mut docs).unwrap();

        let want: Vec<i32> = (0..max_doc).filter(|d| expected[*d as usize]).collect();
        assert_eq!(docs.docs, want, "field {field}");
    }

    // The sparse field must actually be sparse, or the dense case is the
    // only one under test.
    assert!(kv["field.sparse_body.norm_values"].contains("NONE"));
}

/// Deleted documents never match, exactly as
/// `ConstantScoreScorerSupplier.fromIterator` never emits one.
#[test]
fn field_exists_over_norms_skips_deleted_documents() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/norms_index/"
    );
    let kv = manifest_kv(dir);
    let id = hex_id(&kv["id_hex"]);
    let max_doc: i32 = kv["max_doc"].parse().unwrap();
    let nvm = std::fs::read(format!("{dir}{}", kv["nvm_file_name"])).unwrap();
    let nvd = std::fs::read(format!("{dir}{}", kv["nvd_file_name"])).unwrap();
    let (_version, meta) = lucene_codecs::norms::parse_meta(&nvm, &id, "").unwrap();
    let number: i32 = kv["field.body.number"].parse().unwrap();
    let entry = meta.entry(number).unwrap();
    let norms = crate::field_norms::FieldNorms::open(&nvd, *entry, max_doc, None).unwrap();

    let mut live = lucene_util::fixed_bit_set::FixedBitSet::new(max_doc as usize);
    for d in 0..max_doc as usize {
        live.set(d);
    }
    live.clear(2);
    let mut docs = crate::collector::VecCollector::default();
    search_field_exists_norms(&norms, Some(&live), max_doc, &mut docs).unwrap();
    assert_eq!(docs.docs, vec![0, 1, 3, 4]);
}

/// The vector source against real Lucene-written vectors: the dense field
/// (every document) and the sparse one, whose manifest spot-checks the
/// `ordinal -> doc` map this iterates.
#[test]
fn field_exists_over_vectors_matches_the_fixtures_ord_to_doc_map() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/vectors_index/"
    );
    let kv = manifest_kv(dir);
    let id = hex_id(&kv["id_hex"]);
    let max_doc: i32 = kv["max_doc"].parse().unwrap();
    let suffix = kv["segment_suffix"].clone();
    let vemf = std::fs::read(format!("{dir}{}", kv["vemf_file"])).unwrap();
    let vec_file = std::fs::read(format!("{dir}{}", kv["vec_file"])).unwrap();
    let flat =
        lucene_codecs::vectors::FlatVectorsReader::open(&vemf, &vec_file, &id, &suffix).unwrap();

    for prefix in ["f0", "f1"] {
        let number: i32 = kv[&format!("{prefix}.number")].parse().unwrap();
        let count: i32 = kv[&format!("{prefix}.count")].parse().unwrap();
        let values = flat.float_vector_values(number).unwrap();
        assert_eq!(values.size(), count);

        let mut docs = crate::collector::VecCollector::default();
        search_field_exists_vectors(
            values.size(),
            |ord| values.ord_to_doc(ord),
            None,
            max_doc,
            &mut docs,
        )
        .unwrap();

        // Exactly `size()` documents, ascending and distinct -- the
        // iterator contract `ConstantScoreScorerSupplier` relies on.
        assert_eq!(docs.docs.len(), count as usize, "field {prefix}");
        assert!(docs.docs.windows(2).all(|w| w[0] < w[1]), "field {prefix}");

        // ...and each one is where the fixture's own ord_to_doc spot
        // checks say it is.
        for spot in kv[&format!("{prefix}.ord_to_doc")].split(';') {
            let (ord, doc) = spot.split_once(':').unwrap();
            assert_eq!(
                docs.docs[ord.parse::<usize>().unwrap()],
                doc.parse::<i32>().unwrap(),
                "field {prefix} ordinal {ord}"
            );
        }
    }

    // The sparse field must really be sparse, or only the dense case is
    // under test.
    let sparse: i32 = kv["f1.count"].parse().unwrap();
    assert!(sparse < max_doc);
}

#[test]
fn field_exists_over_vectors_skips_deleted_documents() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/vectors_index/"
    );
    let kv = manifest_kv(dir);
    let id = hex_id(&kv["id_hex"]);
    let max_doc: i32 = kv["max_doc"].parse().unwrap();
    let vemf = std::fs::read(format!("{dir}{}", kv["vemf_file"])).unwrap();
    let vec_file = std::fs::read(format!("{dir}{}", kv["vec_file"])).unwrap();
    let flat = lucene_codecs::vectors::FlatVectorsReader::open(
        &vemf,
        &vec_file,
        &id,
        &kv["segment_suffix"],
    )
    .unwrap();
    let number: i32 = kv["f0.number"].parse().unwrap();
    let values = flat.float_vector_values(number).unwrap();

    let mut live = lucene_util::fixed_bit_set::FixedBitSet::new(max_doc as usize);
    for d in 0..max_doc as usize {
        live.set(d);
    }
    live.clear(0);
    live.clear(7);
    let mut docs = crate::collector::VecCollector::default();
    search_field_exists_vectors(
        values.size(),
        |ord| values.ord_to_doc(ord),
        Some(&live),
        max_doc,
        &mut docs,
    )
    .unwrap();
    assert_eq!(docs.docs.len(), max_doc as usize - 2);
    assert!(!docs.docs.contains(&0));
    assert!(!docs.docs.contains(&7));
}

/// An `ordinal -> doc` map that points outside the segment is corruption,
/// and must surface as an error rather than a quietly short result -- the
/// same rule every other decode in this crate follows. Unreachable
/// through a real fixture, so the map is supplied directly.
#[test]
fn field_exists_over_vectors_rejects_an_ord_to_doc_map_pointing_past_the_segment() {
    let mut docs = crate::collector::VecCollector::default();
    let err = search_field_exists_vectors(3, |_| Ok(9_999), None, 10, &mut docs).unwrap_err();
    assert!(matches!(err, crate::Error::Vectors(_)), "{err}");
    assert!(err.to_string().contains("outside 0..10"), "{err}");
    assert!(docs.docs.is_empty());

    // A negative doc is the same class of corruption.
    let err = search_field_exists_vectors(1, |_| Ok(-1), None, 10, &mut docs).unwrap_err();
    assert!(matches!(err, crate::Error::Vectors(_)));

    // ...and a decode error from the map itself propagates unchanged.
    let err = search_field_exists_vectors(
        1,
        |_| Err(lucene_codecs::vectors::Error::OrdOutOfRange(0, 0)),
        None,
        10,
        &mut docs,
    )
    .unwrap_err();
    assert!(matches!(err, crate::Error::Vectors(_)));
}

/// `rewrite`'s per-leaf shortcut, including the asymmetry: the doc-values
/// branch is an OR over three optional counts, the other two are one each.
#[test]
fn field_exists_leaf_completeness_follows_javas_rewrite() {
    use FieldExistsSource::*;
    // Norms: only the terms-dictionary docCount counts.
    assert!(field_exists_leaf_is_complete(
        Norms,
        10,
        Some(10),
        Some(10),
        Some(10),
        Some(10)
    ));
    assert!(!field_exists_leaf_is_complete(
        Norms,
        10,
        Some(9),
        Some(10),
        Some(10),
        Some(10)
    ));
    assert!(!field_exists_leaf_is_complete(
        Norms, 10, None, None, None, None
    ));

    // Vectors: only the vector count.
    assert!(field_exists_leaf_is_complete(
        Vectors,
        10,
        Some(1),
        Some(10),
        None,
        None
    ));
    assert!(!field_exists_leaf_is_complete(
        Vectors,
        10,
        Some(10),
        Some(9),
        Some(10),
        Some(10)
    ));

    // Doc values: any one of the three suffices, and none of them means
    // not rewritable.
    assert!(field_exists_leaf_is_complete(
        DocValues,
        10,
        None,
        None,
        Some(10),
        None
    ));
    assert!(field_exists_leaf_is_complete(
        DocValues,
        10,
        None,
        None,
        None,
        Some(10)
    ));
    assert!(field_exists_leaf_is_complete(
        DocValues,
        10,
        Some(10),
        None,
        None,
        None
    ));
    assert!(!field_exists_leaf_is_complete(
        DocValues,
        10,
        Some(9),
        Some(10),
        Some(9),
        Some(9)
    ));
    assert!(!field_exists_leaf_is_complete(
        DocValues, 10, None, None, None, None
    ));
}
