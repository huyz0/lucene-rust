#![cfg(test)]
use super::*;
use lucene_codecs::doc_values::{DocValuesMeta, SortedSetKind};

/// `path_head` answers what `string_to_path` does for the first
/// component and the count, escapes (of the marks, of a multi-byte
/// character, a trailing one) included.
#[test]
fn path_head_is_string_to_paths_head() {
    let (d, e) = (DELIM_CHAR, ESCAPE_CHAR);
    for s in [
        "a".to_string(),
        format!("dim{d}x"),
        format!("dim{d}x{d}y"),
        format!("d{e}{d}im{d}x"),
        format!("{e}é{d}{e}{d}z"),
        format!("{d}x"),
        format!("x{d}"),
        format!("x{e}"),
        format!("{e}"),
        format!("a{e}{e}{d}b{e}{d}c"),
    ] {
        let full = string_to_path(&s);
        let (head, n) = path_head(&s);
        assert_eq!((head.as_ref(), n), (full[0].as_str(), full.len()), "{s:?}");
    }
}

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
    let meta_buf = std::fs::read(format!("{dir}{}.raw", manifest.get("dvm_file_name"))).unwrap();
    let data_buf = std::fs::read(format!("{dir}{}.raw", manifest.get("dvd_file_name"))).unwrap();
    let suffix = dv_suffix(&manifest);
    let (_, parsed) = doc_values::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
    (manifest, data_buf, parsed)
}

/// Ground truth taken directly from the manifest, which real Lucene wrote
/// via a straightforward per-doc `SortedSetDocValues.nextOrd()` iteration
/// (see `GenMultiValuedDocValues.java`'s `field.tags.ords`/`.terms`
/// output) -- an honest differential check without depending on the
/// `lucene-facet` module (not a project dependency): a straightforward
/// manual per-doc iteration is a reasonable, real ground truth.
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

fn tags_entry(meta: &DocValuesMeta, field_number: i32) -> (&SortedNumericEntry, &TermsDictEntry) {
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
    let err = FacetsState::new(vec![label("Path", &["a", "b"])], FacetsConfig::new()).unwrap_err();
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

fn int(v: i32) -> FacetValue {
    FacetValue::Int(v)
}

fn ia_message<T: std::fmt::Debug>(r: Result<T>) -> String {
    match r {
        Err(lucene_search::Error::IllegalArgument(m)) => m,
        other => panic!("expected IllegalArgument, got {other:?}"),
    }
}

#[test]
fn no_counts_at_all_is_javas_has_counts_false() {
    let state = flat_state();
    let facets = SortedSetFacetCounts::new(&state, Vec::new());
    assert!(facets
        .get_top_children(10, "Author", &[])
        .unwrap()
        .is_none());
    assert!(facets.get_all_children("Author", &[]).unwrap().is_none());
    assert!(facets.get_all_dims(10).unwrap().is_empty());
    assert!(facets.get_top_dims(10, 10).unwrap().is_empty());
    assert_eq!(
        facets.get_specific_value("Author", &["Bob"]).unwrap(),
        int(0),
        "an indexed path with no counting done is 0, not -1"
    );
    assert_eq!(
        facets.get_specific_value("Author", &["Nobody"]).unwrap(),
        int(-1)
    );
}

#[test]
fn a_dim_whose_children_all_counted_zero_is_absent() {
    let state = flat_state();
    let facets = SortedSetFacetCounts::new(&state, vec![0, 0, 3]);
    assert!(facets
        .get_top_children(10, "Author", &[])
        .unwrap()
        .is_none());
    let tag = facets.get_top_children(10, "Tag", &[]).unwrap().unwrap();
    assert_eq!(tag.dim, "Tag");
    assert_eq!(tag.value, int(3));
    assert_eq!(tag.child_count, 1);
    // `getAllChildren` still returns a (child-less) result for the empty
    // dim -- it has no `childCount == 0 -> null` guard, unlike
    // `createFacetResult`.
    let author = facets.get_all_children("Author", &[]).unwrap().unwrap();
    assert!(author.label_values.is_empty());
    assert_eq!(author.value, int(0));
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
    let r = facets.get_top_children(10, "Year", &[]).unwrap().unwrap();
    assert_eq!(
        r.value,
        int(5),
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
    let state = FacetsState::new(vec![label("Tag", &["x"]), label("Tag", &["y"])], config).unwrap();
    let facets = SortedSetFacetCounts::new(&state, vec![2, 1]);
    assert_eq!(
        facets
            .get_top_children(10, "Tag", &[])
            .unwrap()
            .unwrap()
            .value,
        int(-1)
    );
    assert_eq!(
        facets.get_all_children("Tag", &[]).unwrap().unwrap().value,
        int(-1)
    );
}

#[test]
fn a_zero_top_n_is_rejected_everywhere() {
    let state = flat_state();
    let facets = SortedSetFacetCounts::new(&state, vec![1, 1, 1]);
    let want = "topN must be > 0 (got: 0)";
    assert_eq!(ia_message(facets.get_top_children(0, "Author", &[])), want);
    assert_eq!(ia_message(facets.get_all_dims(0)), want);
    assert_eq!(ia_message(facets.get_top_dims(0, 5)), want);
    assert_eq!(ia_message(facets.get_top_dims(5, 0)), want);
}

#[test]
fn a_path_under_a_flat_dim_is_an_error() {
    let state = flat_state();
    let facets = SortedSetFacetCounts::new(&state, vec![1, 1, 1]);
    assert!(ia_message(facets.get_top_children(10, "Author", &["Bob"]))
        .contains("path should be 0 length"));
}

#[test]
fn specific_value_on_a_flat_dim_needs_exactly_one_component() {
    let state = flat_state();
    let facets = SortedSetFacetCounts::new(&state, vec![1, 1, 1]);
    assert!(ia_message(facets.get_specific_value("Author", &["a", "b"]))
        .contains("is not configured as hierarchical, path must be length=1"));
}

#[test]
fn specific_value_rejects_an_unencodable_path() {
    let mut config = FacetsConfig::new();
    config.set_hierarchical("Path", true);
    let state = FacetsState::new(vec![label("Path", &[])], config).unwrap();
    let facets = SortedSetFacetCounts::new(&state, vec![1]);
    // An empty component cannot be encoded: Java's `pathToString` throws.
    assert!(facets.get_specific_value("Path", &[""]).is_err());
    assert!(facets.get_all_children("Path", &[""]).is_err());
}

#[test]
fn hierarchical_children_are_counted_and_labelled_by_their_leaf_component() {
    let state = hier_state();
    //         Path  a  a/b  a/c  d  d/e
    let counts = vec![6, 4, 3, 1, 2, 2];
    let facets = SortedSetFacetCounts::new(&state, counts);
    let top = facets.get_top_children(10, "Path", &[]).unwrap().unwrap();
    assert_eq!(
        top.value,
        int(6),
        "a hierarchical dim reports its own ordinal"
    );
    assert_eq!(
        top.label_values
            .iter()
            .map(|lv| (lv.label.as_str(), lv.value))
            .collect::<Vec<_>>(),
        vec![("a", int(4)), ("d", int(2))]
    );
    let under_a = facets.get_all_children("Path", &["a"]).unwrap().unwrap();
    assert_eq!(under_a.path, vec!["a"]);
    assert_eq!(under_a.value, int(4));
    assert_eq!(
        under_a
            .label_values
            .iter()
            .map(|lv| lv.label.as_str())
            .collect::<Vec<_>>(),
        vec!["b", "c"]
    );
    assert_eq!(
        facets.get_specific_value("Path", &["a", "b"]).unwrap(),
        int(3)
    );
    assert_eq!(
        facets.get_specific_value("Path", &["a", "zz"]).unwrap(),
        int(-1)
    );
    assert!(facets
        .get_top_children(10, "Path", &["zz"])
        .unwrap()
        .is_none());
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
    let all = facets.get_all_dims(10).unwrap();
    let dims: Vec<(&str, FacetValue)> = all.iter().map(|r| (r.dim.as_str(), r.value)).collect();
    assert_eq!(
        dims,
        vec![("Author", int(5)), ("Zed", int(5)), ("Tag", int(-1))]
    );
    // `getTopDims` is `getAllDims` truncated.
    let top = facets.get_top_dims(2, 10).unwrap();
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
    let r = facets.get_top_children(2, "D", &[]).unwrap().unwrap();
    assert_eq!(r.child_count, 3);
    assert_eq!(
        r.value,
        int(6),
        "value sums every non-zero child, not the top 2"
    );
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
            .get_all_children("D", &[])
            .unwrap()
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
    let a =
        multi_valued_double_range_facet_counts(&data, entry, &mapped, &[0, 1, 2, 3, 4]).unwrap();
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
    use lucene_search::ordinal_map::OrdinalMap;
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
    use lucene_search::ordinal_map::OrdinalMap;
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
    assert_eq!(result.value, FacetValue::Int(8));
    assert_eq!(result.dim, "");
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
            .map(|f| f.label.as_str())
            .collect::<Vec<_>>(),
        vec!["y", "x", "z"]
    );
}

#[test]
fn top_children_child_count_counts_every_non_zero_child_not_just_the_top_n() {
    let facets = vec![fc(0, "a", 1), fc(1, "b", 2), fc(2, "c", 3)];
    let result = top_children(facets, 1).unwrap();
    assert_eq!(result.label_values.len(), 1);
    assert_eq!(result.child_count, 3);
    assert_eq!(result.value, FacetValue::Int(6));
}

#[test]
fn a_count_past_i32_saturates() {
    let result = top_children(vec![fc(0, "a", u64::MAX)], 1).unwrap();
    assert_eq!(result.label_values[0].value, FacetValue::Int(i32::MAX));
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
            .map(|f| (f.label.as_str(), f.value))
            .collect::<Vec<_>>(),
        vec![("a", FacetValue::Int(3)), ("c", FacetValue::Int(5))]
    );
    assert_eq!(result.value, FacetValue::Int(8));
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
    assert_eq!(result.value, FacetValue::Int(2));
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
        disi: Default::default(),
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
    let ranges = vec![NumericRange::new_double("r", 0.0, true, 1.0, true).unwrap()];
    let err = double_range_facet_counts(&[], &entry, &ranges, &[0]).unwrap_err();
    assert!(matches!(err, lucene_search::Error::DocValues(_)));
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

/// `counts` longer than the dictionary stops at the dictionary, which is
/// what the `zip` this loop replaced did. Reachable from a caller that
/// pairs one field's counts with another field's terms; the alternative
/// -- labelling a count with a term that does not exist -- is the wrong
/// answer rather than a shorter one.
#[test]
fn more_counts_than_terms_stops_at_the_last_term() {
    let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
    let field_num = field_number(&manifest, "tags");
    let (_, terms) = tags_entry(&meta, field_num);
    let size = terms.terms_dict_size as usize;

    let counts = vec![7u64; size + 5];
    let resolved = resolve_labels(&data, terms, &counts).unwrap();
    assert_eq!(resolved.len(), size);
    assert_eq!(resolved.last().unwrap().ord, size as i64 - 1);
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
        disi: Default::default(),
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
    numeric.bits_per_value = 8;
    let entry = SortedNumericEntry {
        field_number: 0,
        numeric,
        num_docs_with_field: 1,
        addresses: None,
    };
    let terms = TermsDictEntry {
        mask: None,
        terms_dict_size: 1,
        max_term_length: 8,
        // Coherent with the single 8-byte term above; this test asserts
        // the *error* path, so the bound is never reached.
        max_block_length: 8,
        terms_data_offset: 0,
        terms_data_length: 0,
        index: None,
    };
    let err = facet_counts(&[], &entry, &terms, &[0]).unwrap_err();
    assert!(matches!(err, lucene_search::Error::DocValues(_)));
}

// --- range_facet_counts ---
//
// Reuses `doc_values_index`'s `varying` field (its own
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
        disi: Default::default(),
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
    entry.bits_per_value = 8;
    let ranges = vec![NumericRange {
        label: "r".into(),
        min: 0,
        min_inclusive: true,
        max: 100,
        max_inclusive: true,
    }];
    let err = range_facet_counts(&[], &entry, &ranges, &[0]).unwrap_err();
    assert!(matches!(err, lucene_search::Error::DocValues(_)));
}

/// The per-range-per-value scan the `LongRangeCounter` path replaced:
/// `(counts, totCount)` for single-valued (`values[doc]` one value) and
/// multi-valued documents alike.
fn brute_force(ranges: &[NumericRange], docs: &[Vec<i64>]) -> (Vec<u64>, u64) {
    let mut counts = vec![0u64; ranges.len()];
    let mut total = 0;
    for values in docs {
        let mut any = false;
        for (r, c) in ranges.iter().zip(counts.iter_mut()) {
            if values.iter().any(|&v| r.contains(v)) {
                *c += 1;
                any = true;
            }
        }
        total += u64::from(any);
    }
    (counts, total)
}

/// Random ranges, hand-built with every inclusive/exclusive mix (some
/// overlapping, some empty, some at the `i64` extremes).
fn random_ranges(seed: &mut u64, n: usize, lo: i64, hi: i64) -> Vec<NumericRange> {
    let mut next = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    };
    (0..n)
        .map(|i| {
            let span = (hi - lo) as u64;
            let a = lo + (next() % span) as i64;
            let b = lo + (next() % span) as i64;
            let (min, max) = match next() % 8 {
                0 => (i64::MIN, a.max(b)),
                1 => (a.min(b), i64::MAX),
                2 => (a, a),
                3 => (a.max(b), a.min(b)), // empty
                _ => (a.min(b), a.max(b)),
            };
            NumericRange {
                label: format!("r{i}"),
                min,
                min_inclusive: next() % 2 == 0,
                max,
                max_inclusive: next() % 2 == 0,
            }
        })
        .collect()
}

#[test]
fn the_counter_paths_count_what_testing_every_range_counts() {
    let (manifest, data, meta) = dv_meta_and_data(&dv_dir());
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for field in ["varying", "sparse"] {
        let entry = meta.numeric_entry(field_num(&manifest, field)).unwrap();
        let mut reader = doc_values::NumericReader::new(&data, entry);
        let docs: Vec<i32> = (0..5).collect();
        let raw: Vec<Option<i64>> = docs.iter().map(|&d| reader.value(d).unwrap()).collect();
        for n in [0usize, 1, 3, 8, 40] {
            let ranges = random_ranges(&mut seed, n, -200, 1100);
            let values: Vec<Vec<i64>> = raw.iter().flatten().map(|&v| vec![v]).collect();
            let (want, want_total) = brute_force(&ranges, &values);
            let got = range_facet_counts_with_total(&data, entry, &ranges, &docs).unwrap();
            assert_eq!(
                got.counts.iter().map(|(_, c)| *c).collect::<Vec<_>>(),
                want,
                "{field} {ranges:?}"
            );
            assert_eq!(got.total_count, want_total, "{field} {ranges:?}");
            // The double variant compares the mapped values.
            let mapped: Vec<Vec<i64>> = raw
                .iter()
                .flatten()
                .map(|&v| vec![sortable_double_bits(v)])
                .collect();
            let (want, want_total) = brute_force(&ranges, &mapped);
            let got = double_range_facet_counts_with_total(&data, entry, &ranges, &docs).unwrap();
            assert_eq!(got.counts.iter().map(|(_, c)| *c).collect::<Vec<_>>(), want);
            assert_eq!(got.total_count, want_total);
        }
    }
    let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
    let entry = nums_entry(&meta, field_number(&manifest, "nums"));
    let mut reader = doc_values::SortedNumericReader::new(&data, entry);
    let docs: Vec<i32> = (0..5).collect();
    let mut raw = Vec::new();
    for &d in &docs {
        let mut v = Vec::new();
        reader.values(d, &mut v).unwrap();
        raw.push(v);
    }
    for n in [0usize, 1, 3, 8, 40] {
        let ranges = random_ranges(&mut seed, n, -2, 14);
        let (want, want_total) = brute_force(&ranges, &raw);
        let got = multi_valued_range_facet_counts(&data, entry, &ranges, &docs).unwrap();
        assert_eq!(
            got.counts.iter().map(|(_, c)| *c).collect::<Vec<_>>(),
            want,
            "{ranges:?}"
        );
        assert_eq!(got.total_count, want_total, "{ranges:?}");
        // Mapped values that no longer come ascending are sorted back.
        let mapped: Vec<Vec<i64>> = raw
            .iter()
            .map(|v| v.iter().map(|&x| sortable_double_bits(x)).collect())
            .collect();
        let (want, want_total) = brute_force(&ranges, &mapped);
        let got = multi_valued_double_range_facet_counts(&data, entry, &ranges, &docs).unwrap();
        assert_eq!(got.counts.iter().map(|(_, c)| *c).collect::<Vec<_>>(), want);
        assert_eq!(got.total_count, want_total);
    }
}

#[test]
fn a_reordering_map_still_counts_each_document_once_per_range() {
    // `-1` and `1` as raw bits map through `sortable_double_bits` to
    // `i64::MIN + 1`-ish and `1`'s own image: out of the stored order.
    let (manifest, data, meta) = load_dv_meta(&multi_dv_dir());
    let entry = nums_entry(&meta, field_number(&manifest, "nums"));
    let all = vec![NumericRange::new_long("all", i64::MIN, true, i64::MAX, true).unwrap()];
    let got = multi_valued_double_range_facet_counts(&data, entry, &all, &[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(got.counts[0].1, 3);
    assert_eq!(got.total_count, 3);
}
