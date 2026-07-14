//! Writes `doc_values::write_single_dense_*_field`-produced `.dvm`/`.dvd`/
//! `.dvs` triples (one per segment, one field per triple) plus a manifest to
//! the directory given as the first CLI argument.
//!
//! Reverse-direction differential test (Rust writes, Java reads), same
//! division of labor as `write_points_fixture.rs`: `fixtures/src/
//! VerifyDocValues.java` opens each result through real Lucene's
//! `Lucene90DocValuesFormat` with a hand-built `SegmentInfo`/`FieldInfos`,
//! so this slice doesn't also need a `.si`/`.fnm` writer.
//!
//! Covers all five `DocValuesType`s this port's write side supports, each
//! scoped to exactly one shape: dense (every doc has a value, or for the
//! multi-valued types, at least one) -- see each
//! `doc_values::write_single_dense_*_field` function's doc comment for the
//! full list of what's deliberately out of scope (sparse fields, per-field
//! doc-values skip indexes, multiple fields in one triple, etc).
//!
//! Run: `cargo run -p lucene-codecs --example write_doc_values_fixture -- <dir>`

use lucene_codecs::doc_values;
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittendvt01";
const FIELD_NUMBER: i32 = 0;

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_doc_values_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    // A mix of small/medium/negative values, dense over every doc -- enough
    // spread to force a real bits-per-value > 0 (not the all-equal/constant
    // path, which segment "_2" below covers) and `min <= 0` throughout (so
    // the min-shift-drop optimization, covered by segment "_1", never
    // triggers here).
    let values: Vec<i64> = vec![5, 250, 0, 1_000_000, -1_000_000, 42, 42, 999_999];

    // Regression case for the `gcd==1 && min>0 && unsignedBitsRequired(max)
    // == unsignedBitsRequired(max-min)` min-shift-drop optimization
    // (Lucene90DocValuesConsumer.java): every value here has `min == 1 > 0`,
    // and `bitsRequired(255) == bitsRequired(254) == 8`, so the shift must
    // be dropped -- a review pass before this writer's commit found this
    // exact branch had zero coverage against real Lucene (segment "_0"
    // above never has `min > 0`), only against this port's own reader.
    let shift_drop_values: Vec<i64> = vec![1, 255, 128, 64, 200, 1, 254, 100];

    // Regression case for the `min >= max` all-constant encoding
    // (`bitsPerValue == 0`, every doc decodes to `min_value` alone) -- also
    // previously unverified against real Lucene, only via a pure-Rust unit
    // test.
    let constant_values: Vec<i64> = vec![42; 6];

    let numeric_segments: [(&str, &[i64]); 3] = [
        ("_0", &values),
        ("_1", &shift_drop_values),
        ("_2", &constant_values),
    ];

    // BINARY: `_3` every value the same length (fixed-length/direct
    // addressing path), `_4` varying lengths including an empty value
    // (variable-length/DirectMonotonicReader address-block path).
    let binary_fixed: Vec<Vec<u8>> = vec![b"aaaa".to_vec(), b"bbbb".to_vec(), b"cccc".to_vec()];
    let binary_var: Vec<Vec<u8>> = vec![
        b"a".to_vec(),
        Vec::new(),
        b"medium".to_vec(),
        b"a much longer value than the others".to_vec(),
        b"z".to_vec(),
    ];

    // SORTED_NUMERIC: `_5` every doc has exactly one value (the address-array
    // collapse case: `numDocsWithField == numeric.numValues`), `_6` a mix of
    // 1-3 values per doc (forces the real DirectMonotonicReader address
    // range array).
    let sorted_numeric_single: Vec<Vec<i64>> = vec![vec![5], vec![-3], vec![100], vec![0]];
    let sorted_numeric_multi: Vec<Vec<i64>> =
        vec![vec![1, 2, 3], vec![9], vec![-1, -2], vec![0], vec![7, 7, 8]];

    // SORTED: `_7`, five docs with repeated values over a 3-term dictionary
    // (exercises both the terms-dict block encoding and repeated ordinals).
    let sorted_values: Vec<Vec<u8>> = vec![
        b"banana".to_vec(),
        b"apple".to_vec(),
        b"cherry".to_vec(),
        b"apple".to_vec(),
        b"banana".to_vec(),
    ];

    // SORTED_SET: `_8` every doc has exactly one distinct value (the
    // `multiValued = false` collapse case, same shape as SORTED), `_9` a mix
    // of 1-2 values per doc sharing a dictionary, including a doc whose raw
    // values repeat (deduped away per-doc).
    let sorted_set_single: Vec<Vec<Vec<u8>>> = vec![
        vec![b"red".to_vec()],
        vec![b"green".to_vec()],
        vec![b"blue".to_vec()],
    ];
    let sorted_set_multi: Vec<Vec<Vec<u8>>> = vec![
        vec![b"red".to_vec(), b"green".to_vec()],
        vec![b"blue".to_vec()],
        vec![b"green".to_vec(), b"green".to_vec()], // repeats dedup to one ord
        vec![b"red".to_vec(), b"blue".to_vec()],
    ];

    let dir = FsDirectory::open(&out_dir);
    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "segments=_0,_1,_2,_3,_4,_5,_6,_7,_8,_9").unwrap();

    for (name, seg_values) in numeric_segments {
        let seg_max_doc = seg_values.len() as i32;
        let (meta, data, skip_index) = doc_values::write_single_dense_numeric_field(
            FIELD_NUMBER,
            seg_values,
            seg_max_doc,
            &SEGMENT_ID,
            "",
        )
        .expect("single dense numeric field write");
        write_triple(&dir, name, &meta, &data, &skip_index);

        writeln!(manifest, "{name}.type=NUMERIC").unwrap();
        writeln!(manifest, "{name}.max_doc={seg_max_doc}").unwrap();
        writeln!(manifest, "{name}.field_number={FIELD_NUMBER}").unwrap();
        let rendered: Vec<String> = seg_values.iter().map(|v| v.to_string()).collect();
        writeln!(manifest, "{name}.values={}", rendered.join(";")).unwrap();
    }

    for (name, seg_values) in [("_3", &binary_fixed), ("_4", &binary_var)] {
        let seg_max_doc = seg_values.len() as i32;
        let (meta, data, skip_index) = doc_values::write_single_dense_binary_field(
            FIELD_NUMBER,
            seg_values,
            seg_max_doc,
            &SEGMENT_ID,
            "",
        )
        .expect("single dense binary field write");
        write_triple(&dir, name, &meta, &data, &skip_index);

        writeln!(manifest, "{name}.type=BINARY").unwrap();
        writeln!(manifest, "{name}.max_doc={seg_max_doc}").unwrap();
        writeln!(manifest, "{name}.field_number={FIELD_NUMBER}").unwrap();
        let rendered: Vec<String> = seg_values.iter().map(|v| hex(v)).collect();
        writeln!(manifest, "{name}.values={}", rendered.join(";")).unwrap();
    }

    for (name, seg_values) in [
        ("_5", &sorted_numeric_single),
        ("_6", &sorted_numeric_multi),
    ] {
        let seg_max_doc = seg_values.len() as i32;
        let (meta, data, skip_index) = doc_values::write_single_dense_sorted_numeric_field(
            FIELD_NUMBER,
            seg_values,
            &SEGMENT_ID,
            "",
        )
        .expect("single dense sorted numeric field write");
        write_triple(&dir, name, &meta, &data, &skip_index);

        writeln!(manifest, "{name}.type=SORTED_NUMERIC").unwrap();
        writeln!(manifest, "{name}.max_doc={seg_max_doc}").unwrap();
        writeln!(manifest, "{name}.field_number={FIELD_NUMBER}").unwrap();
        let rendered: Vec<String> = seg_values
            .iter()
            .map(|per_doc| {
                per_doc
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect();
        writeln!(manifest, "{name}.values={}", rendered.join(";")).unwrap();
    }

    {
        let seg_max_doc = sorted_values.len() as i32;
        let (meta, data, skip_index) = doc_values::write_single_dense_sorted_field(
            FIELD_NUMBER,
            &sorted_values,
            seg_max_doc,
            &SEGMENT_ID,
            "",
        )
        .expect("single dense sorted field write");
        write_triple(&dir, "_7", &meta, &data, &skip_index);

        writeln!(manifest, "_7.type=SORTED").unwrap();
        writeln!(manifest, "_7.max_doc={seg_max_doc}").unwrap();
        writeln!(manifest, "_7.field_number={FIELD_NUMBER}").unwrap();
        let rendered: Vec<String> = sorted_values.iter().map(|v| hex(v)).collect();
        writeln!(manifest, "_7.values={}", rendered.join(";")).unwrap();
    }

    for (name, seg_values) in [("_8", &sorted_set_single), ("_9", &sorted_set_multi)] {
        let seg_max_doc = seg_values.len() as i32;
        let (meta, data, skip_index) = doc_values::write_single_dense_sorted_set_field(
            FIELD_NUMBER,
            seg_values,
            seg_max_doc,
            &SEGMENT_ID,
            "",
        )
        .expect("single dense sorted set field write");
        write_triple(&dir, name, &meta, &data, &skip_index);

        writeln!(manifest, "{name}.type=SORTED_SET").unwrap();
        writeln!(manifest, "{name}.max_doc={seg_max_doc}").unwrap();
        writeln!(manifest, "{name}.field_number={FIELD_NUMBER}").unwrap();
        // The writer dedups each doc's values down to its distinct ordinal
        // set (sorted ascending) -- a doc whose raw values repeat (like `_9`'s
        // third doc, `[green, green]`) is stored, and thus read back, with
        // only one value. Mirror that here rather than the raw input.
        let rendered: Vec<String> = seg_values
            .iter()
            .map(|per_doc| {
                let mut distinct: Vec<&Vec<u8>> = per_doc.iter().collect();
                distinct.sort_unstable();
                distinct.dedup();
                distinct
                    .into_iter()
                    .map(|v| hex(v))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect();
        writeln!(manifest, "{name}.values={}", rendered.join(";")).unwrap();
    }

    println!("wrote doc values fixture to {out_dir}");
}

fn write_triple(dir: &FsDirectory, name: &str, meta: &[u8], data: &[u8], skip_index: &[u8]) {
    let mut files = Vec::new();
    for (suffix, bytes) in [("dvm", meta), ("dvd", data), ("dvs", skip_index)] {
        let file_name = format!("{name}.{suffix}");
        let mut out = dir.create_output(&file_name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
        files.push(file_name);
    }
    dir.sync(&files).unwrap();
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
