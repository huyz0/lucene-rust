//! `PerFieldDocValuesFormat` write routing against Java (`GenPerFieldDocValues`):
//! the `s_` fields routed to `Lucene90DocValuesFormat(16)` with
//! [`IndexWriter::set_doc_values_format_for_field`], the rest on the default
//! format. The same documents are written through the document API and every
//! segment must come out as Java's: the same instances (`Lucene90_0`,
//! `Lucene90_1`) with the same fields -- numbered in `IndexingChain`'s
//! field-hash order at flush, which puts the routed `s_key` first, and in
//! field-number order at a merge, which puts the default `d_num` first -- and
//! the same `.dvm`/`.dvd`/`.dvs` bytes, the segment id aside (the skip
//! indexes of the routed format close every 16 documents).

// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_index::check_index;
use lucene_index::document::{
    BinaryDocValuesField, Document, NumericDocValuesField, SortedDocValuesField,
    SortedSetDocValuesField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, TieredMergePolicy};
use lucene_index::segment_info::{self, LuceneVersion};
use lucene_index::segment_infos;
use lucene_store::{Directory, FsDirectory};
use lucene_util::test_support::TempDir;
use std::sync::Arc;

const PER_SEGMENT: usize = 300;
const WORDS: [&str; 10] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
];

/// `GenPerFieldDocValues.value`.
fn value(i: usize, k: i64) -> i64 {
    let x = (i as i64 + 1) * 2_654_435_761 + k * 40_503;
    (x ^ ((x as u64) >> 13) as i64) % 100_000
}

fn word(i: usize, k: i64) -> &'static str {
    WORDS[(value(i, k) % WORDS.len() as i64) as usize]
}

/// `GenPerFieldDocValues.doc`.
fn doc(i: usize) -> Document {
    let mut d = Document::new();
    d.add(NumericDocValuesField::indexed_field("d_num", value(i, 0)));
    if !i.is_multiple_of(5) {
        d.add(NumericDocValuesField::indexed_field(
            "s_num",
            value(i, 1) / 16,
        ));
    }
    d.add(SortedDocValuesField::indexed_field("s_key", word(i, 2)));
    d.add(SortedSetDocValuesField::new("d_set", word(i, 3)));
    d.add(SortedSetDocValuesField::new("d_set", word(i, 4)));
    d.add(BinaryDocValuesField::new(
        "s_bin",
        format!("b{}", value(i, 5)),
    ));
    d
}

fn write(dir: &FsDirectory, merge: bool) {
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version).unwrap();
    let small = Lucene90DocValuesFormat::new(16).unwrap();
    for field in ["s_key", "s_num", "s_bin"] {
        w.set_doc_values_format_for_field(field, small);
    }
    // Re-routing replaces, never duplicates.
    w.set_doc_values_format_for_field("s_bin", small);
    // The generator's writer: `TieredMergePolicy` (loose merged segments),
    // no merge-on-commit.
    w.set_max_full_flush_merge_wait_millis(0);
    if merge {
        let mut tmp = TieredMergePolicy::default();
        tmp.compound_file_settings_mut()
            .set_no_cfs_ratio(0.0)
            .unwrap();
        w.set_pluggable_merge_policy(Some(Arc::new(tmp)));
    }
    for seg in 0..2 {
        for i in seg * PER_SEGMENT..(seg + 1) * PER_SEGMENT {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    if merge {
        w.force_merge(1).unwrap();
    }
}

fn fixture(which: &str) -> String {
    format!(
        "{}/../../fixtures/data/per_field_doc_values/{which}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// `manifest.txt`'s shape, from a directory: per segment its doc-values
/// files, then each doc-values field's number and attributes.
fn describe(dir: &dyn Directory) -> Vec<String> {
    let sis = segment_infos::read_latest(dir).unwrap();
    let mut lines = Vec::new();
    for sci in &sis.segments {
        let si = segment_info::parse(
            &dir.open(&format!("{}.si", sci.segment_name)).unwrap(),
            &sci.segment_id,
        )
        .unwrap();
        let mut files: Vec<&String> = si.files.iter().filter(|f| f.contains("Lucene90")).collect();
        files.sort();
        let mut line = format!("segment {} files", sci.segment_name);
        for f in files {
            line.push(' ');
            line.push_str(f);
        }
        lines.push(line);
        let fnm = dir.open(&format!("{}.fnm", sci.segment_name)).unwrap();
        let infos = lucene_codecs::field_infos::parse(&fnm, &sci.segment_id, "").unwrap();
        for f in &infos.fields {
            let attr = |k: &str| {
                f.attributes
                    .iter()
                    .find(|(key, _)| key == k)
                    .map_or("null", |(_, v)| v.as_str())
            };
            lines.push(format!(
                "field {} {} {} {} {}",
                sci.segment_name,
                f.name,
                f.number,
                attr("PerFieldDocValuesFormat.format"),
                attr("PerFieldDocValuesFormat.suffix")
            ));
        }
    }
    lines
}

/// `ours` with its segment id replaced by `id` and its footer re-signed:
/// what the file would be had Java's segment written it.
fn with_segment_id(ours: &[u8], our_id: &[u8; 16], id: &[u8; 16]) -> Vec<u8> {
    let mut bytes = ours.to_vec();
    let at = bytes[..64]
        .windows(16)
        .position(|w| w == our_id)
        .expect("the index header carries the segment id");
    bytes[at..at + 16].copy_from_slice(id);
    let n = bytes.len();
    let crc = u64::from(crc32fast::hash(&bytes[..n - 8]));
    bytes[n - 8..].copy_from_slice(&crc.to_be_bytes());
    bytes
}

fn check(which: &str, merge: bool) {
    let tmp = TempDir::new("per-field-doc-values");
    let dir = FsDirectory::open(&tmp);
    write(&dir, merge);
    let java = FsDirectory::open(fixture(which));
    let expected: Vec<String> = std::fs::read_to_string(format!("{}/manifest.txt", fixture(which)))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(
        describe(&java),
        expected,
        "the reader agrees with Java's manifest"
    );
    assert_eq!(describe(&dir), expected, "{which}");

    let ours = segment_infos::read_latest(&dir).unwrap();
    let theirs = segment_infos::read_latest(&java).unwrap();
    let mut compared = 0;
    for (a, b) in ours.segments.iter().zip(&theirs.segments) {
        for ext in ["dvm", "dvd", "dvs"] {
            for suffix in 0..2 {
                let name = format!("{}_Lucene90_{suffix}.{ext}", a.segment_name);
                let mine = dir.open(&name).unwrap();
                let java_bytes = java.open(&name).unwrap();
                assert!(
                    with_segment_id(&mine, &a.segment_id, &b.segment_id) == java_bytes[..],
                    "{which}: {name} differs from Java's"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, ours.segments.len() * 6);
    let results = check_index::check_directory(&dir).unwrap();
    assert!(
        results.iter().all(|r| r.failures().is_empty()),
        "{results:?}"
    );
}

#[test]
fn flushed_segments_route_doc_values_as_java_does() {
    check("flushed", false);
}

#[test]
fn a_merge_regroups_doc_values_in_field_number_order_as_java_does() {
    check("merged", true);
}
