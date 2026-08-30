//! Differential test against a real index-sorted index whose `Sort` this port
//! **could not open at all** before batch `c35`. Regenerate with
//! `scripts/gen-fixtures.sh --only GenSortedIndexWide`.
//!
//! `index_sort_fixtures.rs` covers the sort this port itself writes: two
//! `SortField(field, LONG, reverse)` tiers whose missing values are exactly
//! `Long.MIN_VALUE`/`Long.MAX_VALUE`. That was the whole of the old
//! `segment_info::IndexSortField`, and `parse` rejected anything else with
//! `Error::UnsupportedSortField` -- naming what it was, but refusing the
//! *file*, so an index a real `IndexWriter` wrote with an ordinary sort could
//! not be opened.
//!
//! Each of this fixture's three tiers was individually unrepresentable:
//!
//! 1. `rank`: `LONG` descending with an **arbitrary** missing value (`42`),
//!    which is neither the sort-first nor the sort-last sentinel -- and which
//!    sits *inside* the data's range, so the missing documents land in the
//!    middle of the order rather than at either end.
//! 2. `multi`: a `SortedNumericSortField` with the **`MAX` selector** and **no
//!    missing value at all** (Java then compares such a document as `0`).
//! 3. `name`: a `STRING` sort over `SortedDocValues`, compared by **term
//!    ordinal**.
//!
//! What is asserted, all against bytes and answers real Lucene produced: the
//! `.si` parses back to the exact `Sort` the writer was configured with,
//! Lucene's own `Sort.toString()` matches what this port renders, the
//! documents read back in Lucene's physical order, and -- the load-bearing one
//! -- this port's own comparator, applied to the columns Lucene wrote, agrees
//! that Lucene's order is sorted (through `check_index`'s
//! `sort.docs_in_index_sort_order`, the port of `CheckIndex.testSort`).
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture reader's own index arithmetic. See `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::segment_info::{
    IndexSortField, IndexSortKind, NumericSortKey, SortKeyComparator, SortedNumericSelector,
    StringMissingValue,
};
use lucene_index::{check_index, segment_info};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::FsDirectory;

fn dir_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/sorted_index_wide"
    ))
}

fn manifest() -> Vec<(String, String)> {
    let text = std::fs::read_to_string(dir_path().join("manifest.properties"))
        .expect("run `scripts/gen-fixtures.sh --only GenSortedIndexWide` first");
    text.lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn get<'a>(m: &'a [(String, String)], key: &str) -> &'a str {
    m.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("manifest key {key} missing"))
}

fn segment_id(m: &[(String, String)], segment: &str) -> [u8; ID_LENGTH] {
    let hex = get(m, &format!("segment.{segment}.id_hex"));
    let mut id = [0u8; ID_LENGTH];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    id
}

fn only_segment(m: &[(String, String)]) -> String {
    let names = get(m, "segment_names");
    assert!(
        !names.contains(','),
        "the fixture force-merges to one segment, got {names}"
    );
    names.to_string()
}

fn parsed_sort() -> Vec<IndexSortField> {
    let m = manifest();
    let segment = only_segment(&m);
    let id = segment_id(&m, &segment);
    let si_bytes = std::fs::read(dir_path().join(format!("{segment}.si"))).unwrap();
    segment_info::parse(&si_bytes, &id)
        .expect("this port must be able to open an index Lucene wrote with an ordinary sort")
        .index_sort
        .expect("the segment declares an index sort")
}

/// **The c35 fix.** Before it, this `parse` call returned
/// `Error::UnsupportedSortField` for tier 0 alone (`missing value 42 is
/// neither the sort-first nor the sort-last sentinel`) and the index was
/// unopenable.
#[test]
fn a_sort_the_old_model_rejected_now_parses_tier_for_tier() {
    let sort = parsed_sort();
    assert_eq!(
        sort,
        vec![
            IndexSortField {
                field: "rank".to_string(),
                reverse: true,
                kind: IndexSortKind::Numeric(NumericSortKey::Long(Some(42))),
            },
            IndexSortField {
                field: "multi".to_string(),
                reverse: false,
                kind: IndexSortKind::SortedNumeric {
                    key: NumericSortKey::Int(None),
                    selector: SortedNumericSelector::Max,
                },
            },
            IndexSortField {
                field: "name".to_string(),
                reverse: true,
                kind: IndexSortKind::String(StringMissingValue::First),
            },
        ]
    );
}

/// Cross-check against Lucene's own `Sort.toString()`, so the assertions
/// above cannot drift from what the generator configured -- and so this
/// port's `describe_index_sort` (the text in
/// `Error::IncongruentIndexSort`) is Java's, selector and type suffixes
/// included.
#[test]
fn our_rendering_matches_lucenes_own_sort_to_string() {
    let m = manifest();
    let printed = get(&m, "leaf_sort");
    assert_eq!(
        printed,
        "<long: \"rank\">! missingValue=42,\
         <sortednumeric: \"multi\"> selector=MAX type=INT,\
         <string: \"name\">! missingValue=SortField.STRING_FIRST"
    );
    assert_eq!(
        get(&m, &format!("segment.{}.sort", only_segment(&m))),
        printed
    );
    assert_eq!(
        segment_info::describe_index_sort(Some(&parsed_sort())),
        printed
    );
}

/// The documents read back in the physical order Lucene chose. The
/// discriminating part is where the `rank`-less documents land: the sentinel
/// is `42`, an ordinary value in the middle of the data, so under a
/// *descending* `rank` they sit between `43` and `13` -- neither first nor
/// last, which is the entire point.
#[test]
fn documents_come_back_in_lucenes_own_sort_order() {
    let m = manifest();
    let segment = only_segment(&m);
    let id = segment_id(&m, &segment);
    let max_doc: i32 = get(&m, "max_doc").parse().unwrap();

    let fdt = std::fs::read(dir_path().join(format!("{segment}.fdt"))).unwrap();
    let fdx = std::fs::read(dir_path().join(format!("{segment}.fdx"))).unwrap();
    let fdm = std::fs::read(dir_path().join(format!("{segment}.fdm"))).unwrap();
    let reader = lucene_codecs::stored_fields::open(&fdt, &fdx, &fdm, &id, "").unwrap();
    assert_eq!(reader.max_doc(), max_doc);

    let fnm = std::fs::read(dir_path().join(format!("{segment}.fnm"))).unwrap();
    let infos = lucene_codecs::field_infos::parse(&fnm, &id, "").unwrap();
    let id_field = infos.fields.iter().find(|f| f.name == "id").unwrap().number;

    let ids: Vec<String> = (0..max_doc)
        .map(|d| {
            let doc = reader.document(d).unwrap();
            doc.fields
                .iter()
                .find(|f| f.field_number == id_field)
                .and_then(|f| match &f.value {
                    lucene_codecs::stored_fields::FieldValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .expect("every document stores its id")
        })
        .collect();
    assert_eq!(ids, get(&m, "docs_in_order").split(',').collect::<Vec<_>>());

    // The missing-`rank` documents are in the *middle*: the two documents
    // before them have a `rank`, and so do the ones after.
    let ranks: Vec<&str> = get(&m, "rank_column").split(',').collect();
    let first_missing = ranks
        .iter()
        .position(|r| r.is_empty())
        .expect("some document has no rank");
    assert!(
        first_missing > 0,
        "a sentinel of 42 cannot put the missing documents first"
    );
    let last_missing = ranks.iter().rposition(|r| r.is_empty()).unwrap();
    assert!(last_missing < ranks.len() - 1, "nor last: {ranks:?}");
}

/// This port's comparator, driven by the parsed `.si` and fed the columns the
/// manifest records, must reproduce Lucene's physical order exactly -- tier
/// by tier, including the `MAX` selector's reduction of the multi-valued
/// column and the `0` a document with no `multi` compares as.
#[test]
fn our_comparator_reproduces_lucenes_order_from_lucenes_own_columns() {
    let m = manifest();
    let sort = parsed_sort();
    let max_doc: usize = get(&m, "max_doc").parse().unwrap();

    let ranks: Vec<Option<i64>> = column(get(&m, "rank_column"), max_doc)
        .into_iter()
        .map(|c| c.map(|s| s.parse().unwrap()))
        .collect();
    // `SortedNumericSelector.Type.MAX`: the largest of the document's values.
    let multi: Vec<Option<i64>> = column(get(&m, "multi_column"), max_doc)
        .into_iter()
        .map(|c| {
            c.map(|s| {
                s.split(' ')
                    .map(|v| v.parse::<i64>().unwrap())
                    .max()
                    .unwrap()
            })
        })
        .collect();
    let name_ords: Vec<Option<i64>> = column(get(&m, "name_ord_column"), max_doc)
        .into_iter()
        .map(|c| c.map(|s| s.parse().unwrap()))
        .collect();

    let columns = [&ranks, &multi, &name_ords];
    let cmps: Vec<SortKeyComparator> = sort
        .iter()
        .map(|sf| SortKeyComparator::new(sf).expect("every tier here has a single-i64 key"))
        .collect();

    for doc in 1..max_doc {
        let mut ordering = std::cmp::Ordering::Equal;
        for (cmp, keys) in cmps.iter().zip(columns) {
            ordering = cmp.compare(keys[doc - 1], keys[doc]);
            if ordering != std::cmp::Ordering::Equal {
                break;
            }
        }
        assert_ne!(
            ordering,
            std::cmp::Ordering::Greater,
            "docID={} sorts after docID={doc}",
            doc - 1
        );
    }
}

/// The whole-index statement: `check_index` -- this port's `CheckIndex`,
/// whose `testSort` re-derives the comparators from the `.si` and reads the
/// keys out of the *segment's own* doc-values files rather than the manifest
/// -- accepts a real Lucene index sorted three ways.
#[test]
fn our_own_check_index_accepts_a_real_lucene_wide_sorted_segment() {
    let dir = FsDirectory::open(dir_path());
    let results = check_index::check_directory(&dir).expect("check a real sorted index");
    let segment = results
        .iter()
        .find(|r| r.segment_name == only_segment(&manifest()))
        .expect("the fixture's segment must be checked");
    let sort_check = segment
        .checks
        .iter()
        .find(|c| c.name == "sort.docs_in_index_sort_order")
        .expect("the sorted segment must reach the index-sort check");
    assert!(
        sort_check.passed(),
        "our comparator disagrees with real Lucene's physical order: {}",
        sort_check.message
    );
    for result in &results {
        assert!(
            result.all_passed(),
            "check_index on a real Lucene sorted index ({}): {:?}",
            result.segment_name,
            result.failures()
        );
    }
}

/// `""` is an absent value in the manifest's column encoding.
fn column(line: &str, max_doc: usize) -> Vec<Option<&str>> {
    let cells: Vec<Option<&str>> = line
        .split(',')
        .map(|c| if c.is_empty() { None } else { Some(c) })
        .collect();
    assert_eq!(cells.len(), max_doc);
    cells
}
