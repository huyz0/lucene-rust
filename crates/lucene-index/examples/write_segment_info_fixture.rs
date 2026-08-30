//! Writes `segment_info::write`-produced `.si` files plus per-segment
//! manifests to the directory given as the first CLI argument.
//!
//! This is the reverse of this repo's usual differential-testing direction
//! (Java writes, Rust reads): here Rust writes, and
//! `fixtures/src/VerifySegmentInfo.java` reads the result back through real
//! Lucene's own `Lucene99SegmentInfoFormat.read`.
//!
//! Lives in `lucene-index` (not `lucene-codecs`, unlike the field-infos/
//! stored-fields write-path examples) because `SegmentInfo` itself lives in
//! `lucene-index::segment_info` -- the architecture skill's downward-only
//! dependency graph (`codecs ← index`) means `lucene-codecs` cannot depend on
//! `lucene-index` to reuse its types.
//!
//! Run: `cargo run -p lucene-index --example write_segment_info_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::segment_info::{
    self, IndexSortField, IndexSortKind, LuceneVersion, NumericSortKey, SegmentInfo,
    SortedNumericSelector, SortedSetSelector, StringMissingValue,
};
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_segment_info_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    // _0: compound file, with minVersion, with files/diagnostics/attributes.
    gen(
        &out_dir,
        "_0",
        SegmentInfo {
            id: *b"rustwrittensi000",
            version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: Some(LuceneVersion {
                major: 9,
                minor: 12,
                bugfix: 0,
            }),
            doc_count: 12345,
            is_compound_file: true,
            has_blocks: false,
            diagnostics: vec![
                ("source".to_string(), "flush".to_string()),
                ("lucene.version".to_string(), "10.0.0".to_string()),
                ("os".to_string(), "Linux".to_string()),
            ],
            files: vec![
                "_0.fdt".to_string(),
                "_0.fdx".to_string(),
                "_0_1.doc".to_string(),
            ],
            attributes: vec![(
                "Lucene90StoredFieldsFormat.mode".to_string(),
                "BEST_SPEED".to_string(),
            )],
            index_sort: None,
        },
    );

    // _1: not a compound file, no minVersion, no blocks/hasBlocks, empty
    // diagnostics/files/attributes -- exercises the "everything empty" path.
    gen(
        &out_dir,
        "_1",
        SegmentInfo {
            id: *b"rustwrittensi111",
            version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 7,
            is_compound_file: false,
            has_blocks: true,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: None,
        },
    );

    // _2: an index-sorted segment. This is the case real Lucene's
    // `SortFieldProvider` byte layout governs, and the only automated proof
    // that `segment_info::write`'s sort-field encoding is the real one and
    // not this port's own invention: `VerifySegmentInfo` reads it back with
    // `SortFieldProvider.forName(...).readSortField(...)` and compares field
    // name / reverse / missing value against the manifest.
    gen(
        &out_dir,
        "_2",
        SegmentInfo {
            id: *b"rustwrittensi222",
            version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: Some(LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            }),
            doc_count: 100,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: Some(vec![
                IndexSortField::long("timestamp", true, Some(i64::MAX)),
                IndexSortField::long("price", false, Some(i64::MIN)),
            ]),
        },
    );

    // _3: **every** shape `SortFieldProvider` can round-trip, in one `.si`.
    // `_2` above covers the sort this port's own sorted writers produce (two
    // `LONG` tiers with the `Long.MIN_VALUE`/`Long.MAX_VALUE` sentinels),
    // which was the whole of what `IndexSortField` could represent before
    // c35. This is the widened model's encoder under test: all four
    // providers, every `SortField.Type` that can be an index sort, both
    // selector enums, and every missing-value form -- an arbitrary numeric
    // sentinel, `STRING_FIRST`/`STRING_LAST`, and **no missing value at
    // all**. `VerifySegmentInfo` reads it back through
    // `SortFieldProvider.forName(...).readSortField(...)` and compares
    // Lucene's own `Sort.toString()` against ours.
    gen(
        &out_dir,
        "_3",
        SegmentInfo {
            id: *b"rustwrittensi333",
            version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 42,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: Some(vec![
                // An arbitrary INT sentinel: neither first nor last.
                IndexSortField {
                    field: "an_int".to_string(),
                    reverse: true,
                    kind: IndexSortKind::Numeric(NumericSortKey::Int(Some(-7))),
                },
                // No missing value at all -- Java compares such a document
                // as `0`, which is what made this unrepresentable before.
                IndexSortField {
                    field: "a_long".to_string(),
                    reverse: false,
                    kind: IndexSortKind::Numeric(NumericSortKey::Long(None)),
                },
                // FLOAT/DOUBLE go through `NumericUtils.floatToSortableInt`/
                // `doubleToSortableLong` on the way to disk.
                IndexSortField {
                    field: "a_float".to_string(),
                    reverse: false,
                    kind: IndexSortKind::Numeric(NumericSortKey::Float(Some(-1.5))),
                },
                IndexSortField {
                    field: "a_double".to_string(),
                    reverse: true,
                    kind: IndexSortKind::Numeric(NumericSortKey::Double(Some(2.25))),
                },
                // STRING's missing marker is `1 == FIRST, else LAST`, the
                // opposite way round from the sorted-set/binary marker.
                IndexSortField {
                    field: "a_string".to_string(),
                    reverse: true,
                    kind: IndexSortKind::String(StringMissingValue::Last),
                },
                IndexSortField {
                    field: "a_sorted_numeric".to_string(),
                    reverse: false,
                    kind: IndexSortKind::SortedNumeric {
                        key: NumericSortKey::Int(Some(9)),
                        selector: SortedNumericSelector::Max,
                    },
                },
                IndexSortField {
                    field: "a_sorted_set".to_string(),
                    reverse: true,
                    kind: IndexSortKind::SortedSet {
                        selector: SortedSetSelector::MiddleMin,
                        missing: StringMissingValue::First,
                    },
                },
                IndexSortField {
                    field: "a_binary".to_string(),
                    reverse: false,
                    kind: IndexSortKind::Binary(StringMissingValue::None),
                },
            ]),
        },
    );

    println!("wrote segment-info fixtures to {out_dir}");
}

fn gen(out_dir: &str, segment_name: &str, si: SegmentInfo) {
    let bytes = segment_info::write(&si, "");
    let file_name = format!("{segment_name}.si");
    let dir = FsDirectory::open(out_dir);
    let mut out = dir.create_output(&file_name).unwrap();
    out.write_bytes(&bytes);
    out.close().unwrap();
    dir.sync(&[file_name]).unwrap();

    let mut manifest =
        std::fs::File::create(format!("{out_dir}/{segment_name}.manifest.properties")).unwrap();
    writeln!(manifest, "segment_name={segment_name}").unwrap();
    writeln!(manifest, "id_hex={}", hex(&si.id)).unwrap();
    writeln!(manifest, "version_major={}", si.version.major).unwrap();
    writeln!(manifest, "version_minor={}", si.version.minor).unwrap();
    writeln!(manifest, "version_bugfix={}", si.version.bugfix).unwrap();
    writeln!(
        manifest,
        "has_min_version={}",
        if si.min_version.is_some() { 1 } else { 0 }
    )
    .unwrap();
    if let Some(mv) = si.min_version {
        writeln!(manifest, "min_version_major={}", mv.major).unwrap();
        writeln!(manifest, "min_version_minor={}", mv.minor).unwrap();
        writeln!(manifest, "min_version_bugfix={}", mv.bugfix).unwrap();
    }
    writeln!(manifest, "doc_count={}", si.doc_count).unwrap();
    writeln!(
        manifest,
        "is_compound_file={}",
        if si.is_compound_file { 1 } else { 0 }
    )
    .unwrap();
    writeln!(manifest, "has_blocks={}", if si.has_blocks { 1 } else { 0 }).unwrap();
    writeln!(manifest, "diagnostics={}", join_map(&si.diagnostics)).unwrap();
    writeln!(manifest, "attributes={}", join_map(&si.attributes)).unwrap();
    writeln!(manifest, "files={}", si.files.join(",")).unwrap();
    // `field:reverse:missing` triples, in priority order; empty for an
    // unsorted segment.
    // Lucene's own `Sort.toString()` rendering, produced by this port's
    // `describe_index_sort`. `VerifySegmentInfo` compares it against
    // `si.getIndexSort().toString()` on the `Sort` real Lucene reconstructed
    // from these very bytes, so the check covers every field of every
    // provider -- field name, direction, selector, type and missing value --
    // in one string, and it is Java that decides what that string is.
    writeln!(
        manifest,
        "index_sort={}",
        si.index_sort
            .as_ref()
            .map(|fields| segment_info::describe_index_sort(Some(fields)))
            .unwrap_or_default()
    )
    .unwrap();

    println!(
        "wrote {segment_name}.si ({} bytes)",
        std::fs::metadata(format!("{out_dir}/{segment_name}.si"))
            .unwrap()
            .len()
    );
}

fn join_map(m: &[(String, String)]) -> String {
    m.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
