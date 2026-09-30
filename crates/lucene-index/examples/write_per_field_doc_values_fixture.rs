//! Writes two indexes whose doc values are split across two
//! `Lucene90DocValuesFormat` instances, for `VerifyPerFieldDocValues`: the
//! `s_` fields are routed to `Lucene90DocValuesFormat(16)` with
//! [`IndexWriter::set_doc_values_format_for_field`], the rest stay on the
//! default -- `GenPerFieldDocValues`' documents and routing.
//!
//! `<out>/flushed` holds two flushed segments, whose flush reaches the routed
//! `s_key` first (`Lucene90_0` is the routed instance); `<out>/merged` the
//! same force-merged into one segment, whose merge reaches the default
//! `d_num` first (`Lucene90_0` is the default instance).
//!
//! Usage: `write_per_field_doc_values_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_codecs::per_field_doc_values::Lucene90DocValuesFormat;
use lucene_index::document::{
    BinaryDocValuesField, Document, NumericDocValuesField, SortedDocValuesField,
    SortedSetDocValuesField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::{MergePolicy, TieredMergePolicy};
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

/// Documents per flushed segment; `VerifyPerFieldDocValues` recomputes them.
const PER_SEGMENT: usize = 300;
const WORDS: [&str; 10] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
];

fn value(i: usize, k: i64) -> i64 {
    let x = (i as i64 + 1) * 2_654_435_761 + k * 40_503;
    (x ^ ((x as u64) >> 13) as i64) % 100_000
}

fn word(i: usize, k: i64) -> &'static str {
    WORDS[(value(i, k) % WORDS.len() as i64) as usize]
}

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

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_per_field_doc_values_fixture <output-dir>");
    for (sub, merge) in [("flushed", false), ("merged", true)] {
        let path = std::path::Path::new(&out).join(sub);
        std::fs::create_dir_all(&path).unwrap();
        write(&FsDirectory::open(&path), merge);
    }
}
