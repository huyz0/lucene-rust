//! Differential tests for faceting against **real Lucene 10.5.0's own
//! answers**, taken from the `lucene-facet` module rather than re-derived.
//!
//! `fixtures/src/GenFacets.java` builds a three-segment index through
//! `FacetsConfig.build`, runs `SortedSetDocValuesFacetCounts`,
//! `LongRangeFacetCounts` and `DoubleRangeFacetCounts` over a
//! `MatchAllDocsQuery`, and writes every `FacetResult` — plus the
//! `OrdinalMap`'s local→global ordinal table straight off
//! `MultiDocValues.getSortedSetValues(...).mapping` — into
//! `fixtures/data/facets_index/manifest.properties`.
//!
//! Three segments is the point: this port's per-segment SORTED_SET ordinals
//! genuinely disagree with the global ones (segment 0's ordinal 6 is
//! `Publish Year`, global ordinal 11), so a facet count summed without an
//! `OrdinalMap` is not merely imprecise, it adds together unrelated terms.
//! Self-round-tripping cannot catch that; only Lucene's own numbers can.

use lucene_codecs::doc_values::{self, DocValuesMeta, SortedNumericEntry, SortedSetKind};
use lucene_codecs::terms_dict::{self, TermsDictEntry};
use lucene_search::facets::{
    self, path_components_to_string, string_to_path, DrillDownTermsIndexing, FacetBuildError,
    FacetResult, FacetsConfig, FacetsState, NumericRange, SortedSetFacetCounts,
};
use lucene_search::ordinal_map::OrdinalMap;

const SEP: char = '\u{1}';

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/facets_index/"
    )
    .to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run scripts/gen-fixtures.sh first");
        Manifest {
            kv: text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }

    fn opt(&self, key: &str) -> Option<&str> {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn num(&self, key: &str) -> i64 {
        self.get(key).parse().expect("numeric manifest value")
    }

    /// A `SEP`-joined list, with `GenFacets.escape` undone.
    fn list(&self, key: &str) -> Vec<String> {
        let raw = self.get(key);
        if raw.is_empty() {
            return Vec::new();
        }
        raw.split(SEP).map(unescape).collect()
    }
}

/// Inverse of `GenFacets.escape`.
fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('n') => out.push('\n'),
            Some('u') => {
                let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                let cp = u32::from_str_radix(&hex, 16).expect("\\uXXXX escape");
                out.push(char::from_u32(cp).expect("valid code point"));
            }
            other => panic!("unknown escape \\{other:?}"),
        }
    }
    out
}

fn id_from_hex(hex: &str) -> [u8; 16] {
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

/// One segment's doc-values, opened exactly the way production does.
struct Segment {
    data: Vec<u8>,
    meta: DocValuesMeta,
    max_doc: i32,
    field_numbers: Vec<(String, i32)>,
}

impl Segment {
    fn field(&self, name: &str) -> i32 {
        self.field_numbers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, n)| *n)
            .unwrap_or_else(|| panic!("field {name} not in this segment"))
    }

    fn docs(&self) -> Vec<i32> {
        (0..self.max_doc).collect()
    }

    fn facet_ords(&self) -> (&SortedNumericEntry, &TermsDictEntry) {
        let entry = self.meta.sorted_set_entry(self.field("$facets")).unwrap();
        match &entry.kind {
            SortedSetKind::Multi { ords, terms } => (ords, terms),
            SortedSetKind::Single(_) => panic!("expected a multi-valued SORTED_SET"),
        }
    }

    /// This segment's `$facets` dictionary, decoded to UTF-8 in ordinal order.
    fn facet_terms(&self) -> Vec<String> {
        let (_, terms) = self.facet_ords();
        terms_dict::decode_all_terms(&self.data, terms)
            .expect("decode $facets terms")
            .into_iter()
            .map(|b| String::from_utf8(b).expect("facet labels are UTF-8"))
            .collect()
    }
}

fn open_segments(m: &Manifest) -> Vec<Segment> {
    let d = dir();
    (0..m.num("segment_count") as usize)
        .map(|i| {
            let id = id_from_hex(m.get(&format!("segment.{i}.id_hex")));
            let fnm =
                std::fs::read(format!("{d}{}.raw", m.get(&format!("segment.{i}.fnm")))).unwrap();
            let fis = lucene_codecs::field_infos::parse(&fnm, &id, "").unwrap();
            let dvm_name = m.get(&format!("segment.{i}.dvm"));
            let meta_buf = std::fs::read(format!("{d}{dvm_name}.raw")).unwrap();
            let data =
                std::fs::read(format!("{d}{}.raw", m.get(&format!("segment.{i}.dvd")))).unwrap();
            let suffix = dvm_name
                .strip_prefix(&format!("{}_", m.get(&format!("segment.{i}.name"))))
                .and_then(|s| s.strip_suffix(".dvm"))
                .unwrap()
                .to_string();
            let (_, meta) = doc_values::parse_meta(&meta_buf, &id, &suffix, &fis).unwrap();
            let field_numbers = m
                .get(&format!("segment.{i}.field_numbers"))
                .split(',')
                .map(|kv| {
                    let (n, num) = kv.split_once(':').unwrap();
                    (n.to_string(), num.parse().unwrap())
                })
                .collect();
            Segment {
                data,
                meta,
                max_doc: m.num(&format!("segment.{i}.max_doc")) as i32,
                field_numbers,
            }
        })
        .collect()
}

/// `FacetsConfig` matching `GenFacets.java`'s. Lucene does not store this in
/// the index — the application has to supply the same one at search time,
/// which is exactly why the type exists.
fn config() -> FacetsConfig {
    let mut c = FacetsConfig::new();
    c.set_multi_valued("Publish Year", true)
        .set_require_dim_count("Publish Year", true)
        .set_multi_valued("Tag", true)
        .set_hierarchical("Path", true);
    c
}

/// The whole read path in one place: per-segment ordinals, an `OrdinalMap`
/// over them, per-segment counts remapped and summed, and the dim layer built
/// over the merged dictionary.
fn global_counts(m: &Manifest) -> (FacetsState, Vec<u64>, OrdinalMap, Vec<Vec<String>>) {
    let segments = open_segments(m);
    let per_segment_terms: Vec<Vec<String>> = segments.iter().map(Segment::facet_terms).collect();
    let as_bytes: Vec<Vec<Vec<u8>>> = per_segment_terms
        .iter()
        .map(|ts| ts.iter().map(|t| t.as_bytes().to_vec()).collect())
        .collect();
    let map = OrdinalMap::build(&as_bytes);

    let per_segment_counts: Vec<Vec<u64>> = segments
        .iter()
        .map(|seg| {
            let (ords, terms) = seg.facet_ords();
            facets::facet_counts(&seg.data, ords, terms, &seg.docs()).expect("count one segment")
        })
        .collect();
    let counts = facets::merge_segment_counts(&map, &per_segment_counts);

    let global_terms: Vec<String> = (0..map.value_count())
        .map(|g| {
            let seg = map.first_segment(g).unwrap();
            per_segment_terms[seg][map.first_segment_ord(g).unwrap() as usize].clone()
        })
        .collect();
    let state = FacetsState::new(global_terms, config()).expect("build the dim state");
    (state, counts, map, per_segment_terms)
}

/// `FacetResult`, rendered the way `GenFacets.appendResult` renders Lucene's.
fn render(r: &FacetResult) -> (i64, usize, Vec<(String, u64)>) {
    (
        r.value,
        r.child_count,
        r.label_values
            .iter()
            .map(|lv| (lv.label.clone(), lv.count))
            .collect(),
    )
}

fn expected(m: &Manifest, prefix: &str) -> (i64, usize, Vec<(String, u64)>) {
    let children = m
        .list(&format!("{prefix}.children"))
        .into_iter()
        .map(|kv| {
            let (label, count) = kv.rsplit_once('=').expect("label=count");
            (label.to_string(), count.parse::<u64>().unwrap())
        })
        .collect();
    (
        m.num(&format!("{prefix}.value")),
        m.num(&format!("{prefix}.child_count")) as usize,
        children,
    )
}

// ---------------------------------------------------------------------------
// OrdinalMap
// ---------------------------------------------------------------------------

#[test]
fn ordinal_map_matches_real_lucenes_local_to_global_table() {
    let m = Manifest::load();
    let segment_terms: Vec<Vec<Vec<u8>>> = (0..m.num("segment_count") as usize)
        .map(|i| {
            m.list(&format!("ordmap.seg.{i}.terms"))
                .into_iter()
                .map(|t| t.into_bytes())
                .collect()
        })
        .collect();
    let map = OrdinalMap::build(&segment_terms);

    assert_eq!(
        map.value_count(),
        m.num("ordmap.global_count"),
        "global dictionary size must match OrdinalMap.getValueCount()"
    );
    for i in 0..segment_terms.len() {
        let want: Vec<i64> = m
            .get(&format!("ordmap.seg.{i}.to_global"))
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(
            map.segment_ords(i).unwrap(),
            want.as_slice(),
            "segment {i}'s local->global ordinals differ from Lucene's OrdinalMap"
        );
    }

    // The mapping is not the identity anywhere but the first few ordinals --
    // otherwise this test would pass with a stub that returned `local`.
    let any_shifted = (0..segment_terms.len()).any(|i| {
        map.segment_ords(i)
            .unwrap()
            .iter()
            .enumerate()
            .any(|(l, &g)| g != l as i64)
    });
    assert!(
        any_shifted,
        "the fixture must have segments whose ordinals actually shift"
    );
}

#[test]
fn the_global_dictionary_rebuilt_from_the_map_is_lucenes_own() {
    let m = Manifest::load();
    let (_, _, map, per_segment_terms) = global_counts(&m);
    let rebuilt: Vec<String> = (0..map.value_count())
        .map(|g| {
            per_segment_terms[map.first_segment(g).unwrap()]
                [map.first_segment_ord(g).unwrap() as usize]
                .clone()
        })
        .collect();
    assert_eq!(rebuilt, m.list("ordmap.global_terms"));
}

#[test]
fn the_decoded_per_segment_dictionaries_are_lucenes_own() {
    // Guards the rest of this file: if the doc-values/terms-dict read path
    // disagreed with Lucene about the ordinals, every count below would be
    // wrong for a reason that has nothing to do with faceting.
    let m = Manifest::load();
    for (i, terms) in open_segments(&m)
        .iter()
        .map(Segment::facet_terms)
        .enumerate()
    {
        assert_eq!(
            terms,
            m.list(&format!("ordmap.seg.{i}.terms")),
            "segment {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// SortedSetDocValuesFacetCounts
// ---------------------------------------------------------------------------

#[test]
fn get_all_dims_matches_real_lucene() {
    let m = Manifest::load();
    let (state, counts, _, _) = global_counts(&m);
    let facets = SortedSetFacetCounts::new(&state, counts);

    let got = facets.all_dims(10);
    assert_eq!(got.len(), m.num("alldims.count") as usize);
    for (i, r) in got.iter().enumerate() {
        assert_eq!(render(r), expected(&m, &format!("alldims.{i}")), "dim #{i}");
    }
    // The dims themselves, in Lucene's own value-DESC / dim-ASC order.
    assert_eq!(
        got.iter().map(|r| r.dim.as_str()).collect::<Vec<_>>(),
        vec!["Author", "Path", "Publish Year", "Tag"]
    );
    // `Tag` is multi-valued without `requireDimCount`: Lucene reports -1, not
    // the sum of its children (4 + 3 + 2 == 9, which would look plausible).
    let tag = got.iter().find(|r| r.dim == "Tag").unwrap();
    assert_eq!(tag.value, -1);
    assert_eq!(tag.child_count, 3);
}

#[test]
fn get_top_children_and_get_all_children_match_real_lucene() {
    let m = Manifest::load();
    let (state, counts, _, _) = global_counts(&m);
    let facets = SortedSetFacetCounts::new(&state, counts);

    for dim in ["Author", "Publish Year", "Tag"] {
        assert_eq!(
            render(&facets.top_children(10, dim, &[]).unwrap()),
            expected(&m, &format!("top.{dim}")),
            "getTopChildren(10, {dim})"
        );
        assert_eq!(
            render(&facets.top_children(2, dim, &[]).unwrap()),
            expected(&m, &format!("top2.{dim}")),
            "getTopChildren(2, {dim}) -- the truncation must not change childCount"
        );
        assert_eq!(
            render(&facets.all_children(dim, &[]).unwrap()),
            expected(&m, &format!("all.{dim}")),
            "getAllChildren({dim}) -- ordinal order, not count order"
        );
    }
}

#[test]
fn hierarchical_dims_walk_the_dim_tree_like_real_lucene() {
    let m = Manifest::load();
    let (state, counts, _, _) = global_counts(&m);
    let facets = SortedSetFacetCounts::new(&state, counts);

    assert_eq!(
        render(&facets.top_children(10, "Path", &[]).unwrap()),
        expected(&m, "top.Path"),
        "the dim's own children"
    );
    assert_eq!(
        render(&facets.top_children(10, "Path", &["a"]).unwrap()),
        expected(&m, "top.Path.a"),
        "one level down"
    );
    assert_eq!(
        render(&facets.all_children("Path", &["d"]).unwrap()),
        expected(&m, "all.Path.d")
    );
    // A path that was never indexed is `null` in Java, not an empty result.
    assert!(facets.top_children(10, "Path", &["nope"]).is_none());
    assert!(facets.all_children("Path", &["a", "b", "c"]).is_none());
}

#[test]
fn get_specific_value_matches_real_lucene_including_its_minus_one() {
    let m = Manifest::load();
    let (state, counts, _, _) = global_counts(&m);
    let facets = SortedSetFacetCounts::new(&state, counts);

    assert_eq!(
        facets.specific_value("Author", &["Bob"]),
        m.num("specific.Author.Bob")
    );
    assert_eq!(
        facets.specific_value("Author", &["Nobody"]),
        m.num("specific.Author.Nobody"),
        "a path that was never indexed is -1, not 0"
    );
    assert_eq!(
        facets.specific_value("Path", &["a"]),
        m.num("specific.Path.a")
    );
    assert_eq!(
        facets.specific_value("Path", &["a", "b"]),
        m.num("specific.Path.a.b")
    );
}

#[test]
fn a_dim_that_was_never_indexed_is_absent_everywhere() {
    let m = Manifest::load();
    let (state, counts, _, _) = global_counts(&m);
    let facets = SortedSetFacetCounts::new(&state, counts);
    assert!(facets.top_children(10, "NoSuchDim", &[]).is_none());
    assert!(facets.all_children("NoSuchDim", &[]).is_none());
    assert_eq!(facets.specific_value("NoSuchDim", &["x"]), -1);
}

#[test]
fn top_dims_is_get_all_dims_truncated() {
    let m = Manifest::load();
    let (state, counts, _, _) = global_counts(&m);
    let facets = SortedSetFacetCounts::new(&state, counts);
    let all = facets.all_dims(10);
    assert_eq!(facets.top_dims(2, 10), all[..2].to_vec());
    assert_eq!(facets.top_dims(99, 10), all);
}

#[test]
fn the_state_reports_lucenes_own_dims_and_size() {
    let m = Manifest::load();
    let (state, _, _, _) = global_counts(&m);
    assert_eq!(state.size(), m.num("ordmap.global_count") as usize);
    let mut dims: Vec<&str> = state.dims().collect();
    dims.sort_unstable();
    let mut want: Vec<String> = m.get("state.dims").split(',').map(str::to_string).collect();
    want.sort();
    assert_eq!(dims, want);
}

/// The defect an `OrdinalMap` exists to prevent, demonstrated rather than
/// asserted about: summing the per-segment count arrays elementwise -- the
/// only thing a caller *could* do before this batch -- gives visibly wrong
/// counts, because a given ordinal means different terms in different
/// segments.
#[test]
fn summing_raw_per_segment_counts_would_conflate_unrelated_terms() {
    let m = Manifest::load();
    let segments = open_segments(&m);
    let per_segment: Vec<Vec<u64>> = segments
        .iter()
        .map(|seg| {
            let (ords, terms) = seg.facet_ords();
            facets::facet_counts(&seg.data, ords, terms, &seg.docs()).unwrap()
        })
        .collect();

    let width = per_segment.iter().map(Vec::len).max().unwrap();
    let mut naive = vec![0u64; width];
    for counts in &per_segment {
        for (i, c) in counts.iter().enumerate() {
            naive[i] += c;
        }
    }

    let (state, correct, _, _) = global_counts(&m);
    assert_ne!(
        naive.len(),
        correct.len(),
        "the naive sum does not even have the right number of ordinals"
    );
    // And the correct counts are the ones Lucene reports.
    let facets = SortedSetFacetCounts::new(&state, correct);
    assert_eq!(
        render(&facets.top_children(10, "Author", &[]).unwrap()),
        expected(&m, "top.Author")
    );
}

// ---------------------------------------------------------------------------
// Range faceting
// ---------------------------------------------------------------------------

fn price_ranges() -> Vec<NumericRange> {
    vec![
        NumericRange::new_long("cheap", 0, true, 25, false).unwrap(),
        NumericRange::new_long("mid", 25, true, 70, false).unwrap(),
        NumericRange::new_long("dear", 70, true, i64::MAX, true).unwrap(),
        NumericRange::new_long("over40", 40, true, i64::MAX, true).unwrap(),
    ]
}

fn size_ranges() -> Vec<NumericRange> {
    vec![
        NumericRange::new_long("small", 1, true, 3, false).unwrap(),
        NumericRange::new_long("medium", 3, true, 9, false).unwrap(),
        NumericRange::new_long("large", 9, true, 100, true).unwrap(),
        NumericRange::new_long("upto3", 1, true, 3, true).unwrap(),
    ]
}

fn score_ranges() -> Vec<NumericRange> {
    vec![
        NumericRange::new_double("negative", f64::NEG_INFINITY, true, 0.0, false).unwrap(),
        NumericRange::new_double("zeroToOne", 0.0, true, 1.0, true).unwrap(),
        NumericRange::new_double("positive", 0.0, false, f64::INFINITY, true).unwrap(),
    ]
}

/// Sums per-segment `RangeCounts` the way a multi-segment range facet must:
/// counts add, and so does `totCount` (each document belongs to exactly one
/// segment, so there is no double-counting to guard against here).
fn sum_ranges(parts: Vec<facets::RangeCounts>) -> (Vec<(String, u64)>, u64) {
    let mut counts: Vec<(String, u64)> = parts[0].counts.clone();
    let mut total = parts[0].total_count;
    for part in &parts[1..] {
        for (slot, add) in counts.iter_mut().zip(part.counts.iter()) {
            assert_eq!(slot.0, add.0);
            slot.1 += add.1;
        }
        total += part.total_count;
    }
    (counts, total)
}

fn assert_ranges(m: &Manifest, prefix: &str, got: (Vec<(String, u64)>, u64)) {
    let (want_value, want_child_count, want_children) = expected(m, prefix);
    assert_eq!(got.1 as i64, want_value, "{prefix} totCount");
    let non_zero: Vec<(String, u64)> = got.0.iter().filter(|(_, c)| *c > 0).cloned().collect();
    assert_eq!(non_zero.len(), want_child_count, "{prefix} childCount");
    assert_eq!(non_zero, want_children, "{prefix} per-range counts");
}

#[test]
fn single_valued_long_range_counts_match_real_lucene() {
    let m = Manifest::load();
    let ranges = price_ranges();
    let parts = open_segments(&m)
        .iter()
        .map(|seg| {
            let entry = seg.meta.numeric_entry(seg.field("price")).unwrap();
            facets::range_facet_counts_with_total(&seg.data, entry, &ranges, &seg.docs()).unwrap()
        })
        .collect();
    assert_ranges(&m, "range.price", sum_ranges(parts));
}

#[test]
fn overlapping_ranges_do_not_inflate_tot_count() {
    // "over40" overlaps both "mid" and "dear", so the counts sum to more than
    // the 9 documents; Lucene's `totCount` is still 9.
    let m = Manifest::load();
    let ranges = price_ranges();
    let parts: Vec<facets::RangeCounts> = open_segments(&m)
        .iter()
        .map(|seg| {
            let entry = seg.meta.numeric_entry(seg.field("price")).unwrap();
            facets::range_facet_counts_with_total(&seg.data, entry, &ranges, &seg.docs()).unwrap()
        })
        .collect();
    let (counts, total) = sum_ranges(parts);
    let summed: u64 = counts.iter().map(|(_, c)| *c).sum();
    assert!(
        summed > total,
        "the fixture's ranges must overlap or this proves nothing"
    );
    assert_eq!(total as i64, m.num("range.price.value"));
}

#[test]
fn multi_valued_long_range_counts_match_real_lucene() {
    let m = Manifest::load();
    let ranges = size_ranges();
    let parts = open_segments(&m)
        .iter()
        .map(|seg| {
            let entry = seg.meta.sorted_numeric_entry(seg.field("sizes")).unwrap();
            facets::multi_valued_range_facet_counts(&seg.data, entry, &ranges, &seg.docs()).unwrap()
        })
        .collect();
    assert_ranges(&m, "range.sizes", sum_ranges(parts));
}

/// The specific rule the multi-valued branch exists for: document 8 has sizes
/// `{1, 2, 3}`, all three of which fall in `upto3` -- Lucene counts it once.
#[test]
fn a_document_with_several_values_in_one_range_is_counted_once() {
    let m = Manifest::load();
    let ranges = size_ranges();
    let parts: Vec<facets::RangeCounts> = open_segments(&m)
        .iter()
        .map(|seg| {
            let entry = seg.meta.sorted_numeric_entry(seg.field("sizes")).unwrap();
            facets::multi_valued_range_facet_counts(&seg.data, entry, &ranges, &seg.docs()).unwrap()
        })
        .collect();
    let (counts, _) = sum_ranges(parts);
    let upto3 = counts.iter().find(|(l, _)| l == "upto3").unwrap().1;
    let want = m
        .list("range.sizes.children")
        .into_iter()
        .find_map(|kv| {
            let (l, c) = kv.rsplit_once('=').unwrap();
            (l == "upto3").then(|| c.parse::<u64>().unwrap())
        })
        .unwrap();
    assert_eq!(upto3, want);
    assert_eq!(
        upto3, 5,
        "9 documents, 7 with sizes, 5 of them with a size <= 3"
    );
}

#[test]
fn double_range_counts_match_real_lucene() {
    let m = Manifest::load();
    let ranges = score_ranges();
    let parts = open_segments(&m)
        .iter()
        .map(|seg| {
            let entry = seg.meta.numeric_entry(seg.field("score")).unwrap();
            facets::double_range_facet_counts_with_total(&seg.data, entry, &ranges, &seg.docs())
                .unwrap()
        })
        .collect();
    assert_ranges(&m, "range.score", sum_ranges(parts));
}

#[test]
fn range_top_children_matches_real_lucenes_ordering() {
    let m = Manifest::load();
    let ranges = price_ranges();
    let parts: Vec<facets::RangeCounts> = open_segments(&m)
        .iter()
        .map(|seg| {
            let entry = seg.meta.numeric_entry(seg.field("price")).unwrap();
            facets::range_facet_counts_with_total(&seg.data, entry, &ranges, &seg.docs()).unwrap()
        })
        .collect();
    let (counts, total) = sum_ranges(parts);
    let top = facets::top_range_children(&counts, total, 10);
    let (want_value, want_child_count, want_children) = expected(&m, "rangetop.price");
    assert_eq!(top.value as i64, want_value);
    assert_eq!(top.child_count, want_child_count);
    assert_eq!(top.label_values, want_children);
}

#[test]
fn the_manifest_actually_carries_something() {
    // A guard against a silently truncated regeneration turning every
    // assertion above into a comparison of two empty lists.
    let m = Manifest::load();
    assert!(m.opt("alldims.count").is_some());
    assert_eq!(m.num("segment_count"), 3);
    assert_eq!(m.num("max_doc"), 9);
}

// ---------------------------------------------------------------------------
// `FacetsConfig.build` -- the write side (c12 §2.9)
// ---------------------------------------------------------------------------

const SUB: char = '\u{2}';

/// One manifest `build.N.*` case: the config, the input labels, and real
/// Lucene's own output.
struct BuildCase {
    name: String,
    config: FacetsConfig,
    labels: Vec<(String, Vec<String>)>,
    /// `(index field name, value)` in Lucene's own order.
    ssdv: Vec<(String, String)>,
    terms: Vec<(String, String)>,
}

fn pairs(m: &Manifest, key: &str) -> Vec<(String, String)> {
    m.get(key)
        .split(SEP)
        .filter(|e| !e.is_empty())
        .map(|e| {
            let (field, value) = e.split_once(SUB).expect("field SUB value");
            (unescape(field), unescape(value))
        })
        .collect()
}

fn build_cases(m: &Manifest) -> Vec<BuildCase> {
    let count: usize = m.num("build_count") as usize;
    (0..count)
        .map(|i| {
            let mut config = FacetsConfig::new();
            for dim in m.get(&format!("build.{i}.dims")).split(SEP) {
                let parts: Vec<&str> = dim.split(SUB).collect();
                assert_eq!(parts.len(), 6, "dim descriptor shape");
                let name = unescape(parts[0]);
                config.set_hierarchical(name.clone(), parts[1] == "true");
                config.set_multi_valued(name.clone(), parts[2] == "true");
                config.set_require_dim_count(name.clone(), parts[3] == "true");
                config.set_drill_down_terms_indexing(
                    name.clone(),
                    match parts[4] {
                        "NONE" => DrillDownTermsIndexing::None,
                        "FULL_PATH_ONLY" => DrillDownTermsIndexing::FullPathOnly,
                        "ALL_PATHS_NO_DIM" => DrillDownTermsIndexing::AllPathsNoDim,
                        "DIMENSION_AND_FULL_PATH" => DrillDownTermsIndexing::DimensionAndFullPath,
                        "ALL" => DrillDownTermsIndexing::All,
                        other => panic!("unknown DrillDownTermsIndexing {other}"),
                    },
                );
                config.set_index_field_name(name, unescape(parts[5]));
            }
            let labels = m
                .get(&format!("build.{i}.labels"))
                .split(SEP)
                .map(|entry| {
                    let mut parts = entry.split(SUB).map(unescape);
                    let dim = parts.next().expect("a dim");
                    (dim, parts.collect())
                })
                .collect();
            BuildCase {
                name: m.get(&format!("build.{i}.name")).to_string(),
                config,
                labels,
                ssdv: pairs(m, &format!("build.{i}.ssdv")),
                terms: pairs(m, &format!("build.{i}.terms")),
            }
        })
        .collect()
}

/// The load-bearing one: for every configuration real Lucene was run with,
/// this port's `build_sorted_set_facet_fields` emits exactly the doc-values
/// values and drill-down terms `FacetsConfig.build(Document)` emitted -- same
/// index field, same values, same order.
///
/// This is what makes the *read* side in this file meaningful rather than
/// circular: those tests decode an index Lucene wrote, and this proves the
/// port would have written the same one.
#[test]
fn facets_config_build_reproduces_real_lucenes_indexed_fields() {
    let m = Manifest::load();
    let cases = build_cases(&m);
    assert!(cases.len() >= 10);

    for case in &cases {
        let labels: Vec<(&str, Vec<&str>)> = case
            .labels
            .iter()
            .map(|(dim, path)| (dim.as_str(), path.iter().map(String::as_str).collect()))
            .collect();
        let as_slices: Vec<(&str, &[&str])> =
            labels.iter().map(|(d, p)| (*d, p.as_slice())).collect();
        let built = case
            .config
            .build_sorted_set_facet_fields(&as_slices)
            .unwrap_or_else(|e| panic!("case {}: {e}", case.name));

        // Flatten this port's per-field grouping back into Lucene's flat
        // document order, per field: the manifest's own order within one
        // index field is Java's insertion order, which is what must match.
        let mut got_ssdv: Vec<(String, String)> = Vec::new();
        let mut got_terms: Vec<(String, String)> = Vec::new();
        for field in &built {
            for v in &field.sorted_set_values {
                got_ssdv.push((field.index_field_name.clone(), v.clone()));
            }
            for t in &field.drill_down_terms {
                got_terms.push((field.index_field_name.clone(), t.clone()));
            }
        }

        // Java groups index fields in a `HashMap`, so compare per index field
        // (order within a field is what is specified; order across them is
        // not -- see `build_sorted_set_facet_fields`'s doc comment).
        for (label, got, want) in [
            ("ssdv", &got_ssdv, &case.ssdv),
            ("terms", &got_terms, &case.terms),
        ] {
            let mut fields: Vec<&str> = want.iter().map(|(f, _)| f.as_str()).collect();
            fields.sort_unstable();
            fields.dedup();
            for field in fields {
                let want_field: Vec<&str> = want
                    .iter()
                    .filter(|(f, _)| f == field)
                    .map(|(_, v)| v.as_str())
                    .collect();
                let got_field: Vec<&str> = got
                    .iter()
                    .filter(|(f, _)| f == field)
                    .map(|(_, v)| v.as_str())
                    .collect();
                assert_eq!(
                    got_field, want_field,
                    "case {} field {field} {label}",
                    case.name
                );
            }
            assert_eq!(got.len(), want.len(), "case {} {label} count", case.name);
        }
    }
}

/// The manifest must actually exercise the branches, or the test above could
/// pass on a `build` that only ever emitted the full path.
#[test]
fn the_build_fixture_covers_every_branch_the_port_implements() {
    let m = Manifest::load();
    let cases = build_cases(&m);
    let by_name = |n: &str| cases.iter().find(|c| c.name == n).expect(n);

    // A hierarchical dim indexes every prefix, so `a/b/c` is three values.
    assert_eq!(by_name("hierarchical_deep").ssdv.len(), 4);
    // A flat multi-valued dim with requireDimCount indexes the bare dim too,
    // so two labels become four values.
    assert_eq!(by_name("flat_multi_require_dim_count").ssdv.len(), 4);
    // ...and without it, just the two.
    assert_eq!(by_name("flat_multi_no_dim_count").ssdv.len(), 2);
    // The five drill-down modes really do differ.
    let terms = |n: &str| by_name(n).terms.len();
    assert_eq!(terms("drilldown_none"), 0);
    assert_eq!(terms("drilldown_full_path_only"), 1);
    assert_eq!(terms("drilldown_dimension_and_full_path"), 2);
    assert_eq!(terms("drilldown_all_paths_no_dim"), 3);
    assert_eq!(terms("hierarchical_deep"), 4);
    // Two index fields in one document.
    let custom = by_name("custom_index_field");
    let mut custom_fields: Vec<&str> = custom.ssdv.iter().map(|(f, _)| f.as_str()).collect();
    custom_fields.sort_unstable();
    custom_fields.dedup();
    assert_eq!(custom_fields, vec!["$author", "$facets"]);
    // A component containing a `/` is *not* a path separator: Lucene's own
    // delimiter is U+001F, which `FacetField.verifyLabel` forbids inside a
    // label, so `pathToString`'s escape branch is unreachable from `build`.
    // What must hold is that "a/b" stays one component.
    let escaped = by_name("escaped_component");
    assert!(escaped
        .ssdv
        .iter()
        .any(|(_, v)| string_to_path(v) == vec!["Path", "a/b"]));
    assert!(escaped
        .ssdv
        .iter()
        .any(|(_, v)| string_to_path(v) == vec!["Path", "a/b", "c"]));
}

/// The two `IllegalArgumentException`s Java raises, and the empty-component
/// one `pathToString` raises.
#[test]
fn facets_config_build_refuses_what_java_refuses() {
    let mut config = FacetsConfig::new();
    config.set_multi_valued("Tag", true);

    // A single-valued dim twice in one document.
    let err = config
        .build_sorted_set_facet_fields(&[("Author", &["a"]), ("Author", &["b"])])
        .unwrap_err();
    assert!(matches!(err, FacetBuildError::NotMultiValued(d) if d == "Author"));
    // ...but a multi-valued one is fine.
    assert!(config
        .build_sorted_set_facet_fields(&[("Tag", &["a"]), ("Tag", &["b"])])
        .is_ok());

    // A flat dim with more than one path component.
    let err = config
        .build_sorted_set_facet_fields(&[("Author", &["a", "b"])])
        .unwrap_err();
    assert!(matches!(
        err,
        FacetBuildError::NotHierarchical { ref dim, components: 2 } if dim == "Author"
    ));
    // ...which a hierarchical dim accepts.
    let mut hier = FacetsConfig::new();
    hier.set_hierarchical("Path", true);
    assert!(hier
        .build_sorted_set_facet_fields(&[("Path", &["a", "b"])])
        .is_ok());

    // An empty path component cannot be encoded.
    let err = config
        .build_sorted_set_facet_fields(&[("Author", &[""])])
        .unwrap_err();
    assert!(matches!(err, FacetBuildError::EmptyPathComponent(_)));

    // No labels at all is an empty result, not an error.
    assert!(config
        .build_sorted_set_facet_fields(&[])
        .unwrap()
        .is_empty());
}

/// End to end: build a document's facet fields, then count them with this
/// crate's own read side. The values `build` emits are exactly the ones
/// `facet_counts`/`FacetsState` expect, which is the property the two halves
/// have to share and the reason porting only one of them was a gap.
#[test]
fn what_build_emits_is_what_the_read_side_counts() {
    let mut config = FacetsConfig::new();
    config.set_hierarchical("Path", true);
    config.set_multi_valued("Tag", true);

    let built = config
        .build_sorted_set_facet_fields(&[("Path", &["a", "b"]), ("Tag", &["x"])])
        .unwrap();
    assert_eq!(built.len(), 1, "both dims use the default index field");
    let values = &built[0].sorted_set_values;

    // A hierarchical dim's own prefix chain is what `FacetsState` walks to
    // find children, so the dim, its child and its grandchild must all be
    // there and in dictionary order once sorted.
    assert!(values.contains(&"Path".to_string()));
    assert!(values
        .iter()
        .any(|v| string_to_path(v) == vec!["Path", "a"]));
    assert!(values
        .iter()
        .any(|v| string_to_path(v) == vec!["Path", "a", "b"]));
    // ...and every emitted value round-trips through the same decoder
    // `resolve_labels` uses.
    for v in values {
        let path = string_to_path(v);
        assert!(!path.is_empty());
        let rebuilt: Vec<&str> = path.iter().map(String::as_str).collect();
        assert_eq!(&path_components_to_string(&rebuilt).unwrap(), v);
    }
}
