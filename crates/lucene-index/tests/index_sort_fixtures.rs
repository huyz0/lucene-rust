//! Differential test against a real **index-sorted** index written by an
//! actual `IndexWriter` configured with `IndexWriterConfig.setIndexSort`.
//! Regenerate with `fixtures/src/GenSortedIndex.java`.
//!
//! The sort is two-tier and its first tier is **reversed**, which is the case
//! that discriminates: a missing value in Lucene is an ordinary *sentinel*
//! substituted into the column (`Long.MAX_VALUE` here for `rank`,
//! `Long.MIN_VALUE` for `tie`) and compared like any other value, so
//! `reverseMul` applies to it too and the missing documents land at the
//! **opposite** end from the one "missing last" suggests. The fixture's
//! `docs_in_order` shows exactly that: the six documents with no `rank` are
//! Lucene's *first* six, under a `rank` descending, missing-last sort.
//!
//! Three things are asserted, all against bytes and answers real Lucene
//! produced:
//!
//! 1. the `.si`'s `numSortFields` block parses back to the sort the writer was
//!    configured with (`b11` proved the encoding against a hand-built
//!    `SegmentInfo`; this proves it for a sort a real `IndexWriter` chose);
//! 2. this port's stored-fields reader returns the documents in the physical
//!    order Lucene put them in;
//! 3. **this port's own comparator reproduces that order** from the same
//!    doc-values columns a reader sees -- `segment_writer::sort_key_rank`, via
//!    `check_index`'s `sort.docs_in_index_sort_order` check, which is the
//!    function the sort-on-flush writer uses to *produce* an order. A
//!    comparator that disagrees with Lucene's is a writer that produces
//!    segments real Lucene's `CheckIndex.testSort` rejects.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::{check_index, segment_info, segment_infos};
use lucene_store::codec_util::ID_LENGTH;
use lucene_store::directory::FsDirectory;

fn dir_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/sorted_index"
    ))
}

fn manifest() -> Vec<(String, String)> {
    let text = std::fs::read_to_string(dir_path().join("manifest.properties"))
        .expect("run fixtures generator first (GenSortedIndex)");
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

/// The `.si` of a segment a real `IndexWriter` sorted parses back to the
/// configured `Sort`, tier for tier, including each tier's direction and
/// which sentinel its missing values take.
#[test]
fn a_real_index_writers_sorted_si_parses_back_to_its_sort() {
    let m = manifest();
    let segment = only_segment(&m);
    let id = segment_id(&m, &segment);
    let si_bytes = std::fs::read(dir_path().join(format!("{segment}.si"))).unwrap();
    let si = segment_info::parse(&si_bytes, &id).expect("parse a real sorted .si");

    let sort = si.index_sort.expect("the segment declares an index sort");
    assert_eq!(sort.len(), 2, "two tiers");
    assert_eq!(sort[0].field, "rank");
    assert!(sort[0].reverse, "rank is descending");
    // `missingValue=Long.MAX_VALUE` -- which the manifest's
    // `segment.<n>.sort` line spells out as 9223372036854775807.
    assert_eq!(sort[0].missing, segment_info::SortMissingValue::Last);
    assert_eq!(sort[1].field, "tie");
    assert!(!sort[1].reverse, "tie is ascending");
    assert_eq!(sort[1].missing, segment_info::SortMissingValue::First);

    // Cross-check against Lucene's own `Sort.toString()`, so the assertions
    // above cannot drift from what the generator configured.
    let printed = get(&m, &format!("segment.{segment}.sort"));
    assert_eq!(
        printed,
        "<long: \"rank\">! missingValue=9223372036854775807,\
         <long: \"tie\"> missingValue=-9223372036854775808"
    );
    // ...and the segment a reader would report the sort for is this one.
    assert_eq!(get(&m, "leaf_sort"), printed);
}

/// The documents come back in the order Lucene physically put them in --
/// including the six with no `rank`, which a *reversed* missing-last sort
/// places first.
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

    let expected: Vec<&str> = get(&m, "docs_in_order").split(',').collect();
    assert_eq!(ids, expected);

    // The first six are exactly the documents with no `rank`, which is the
    // half of the semantics a comparator that pins missing values to one end
    // regardless of `reverse` gets backwards.
    let ranks: Vec<&str> = get(&m, "rank_column").split(',').collect();
    assert_eq!(&ranks[..6], &["", "", "", "", "", ""]);
    assert!(!ranks[6].is_empty());
}

/// The load-bearing one: this port's own comparator, applied to the columns
/// Lucene wrote, must agree that Lucene's physical order is sorted. Run
/// through `check_index`'s `sort.docs_in_index_sort_order` check, which is
/// the port of `CheckIndex.testSort` and calls the exact function the
/// sort-on-flush writer uses to *produce* an order.
#[test]
fn our_own_sort_check_accepts_a_real_lucene_sorted_segment() {
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
        .expect("the sorted segment must reach the index-sort check, not skip it");
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

/// `segments_N` for the sorted index still reads back exactly, so nothing
/// about the sort perturbs the commit-level bookkeeping.
#[test]
fn the_commit_still_reads_back() {
    let m = manifest();
    let dir = FsDirectory::open(dir_path());
    let sis = segment_infos::read_latest(&dir).unwrap();
    assert_eq!(
        sis.segments.len(),
        get(&m, "num_segments").parse::<usize>().unwrap()
    );
    assert_eq!(sis.segments[0].segment_name, only_segment(&m));
}
