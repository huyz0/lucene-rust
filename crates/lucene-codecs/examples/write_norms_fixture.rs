//! Writes `norms::write_fields`-produced `.nvm`/`.nvd` pairs plus a manifest
//! to the directory given as the first CLI argument.
//!
//! Reverse-direction differential test (Rust writes, Java reads), same
//! division of labor as `write_doc_values_fixture.rs`: `fixtures/src/
//! VerifyNorms.java` opens the result through real Lucene's
//! `Lucene90NormsFormat` with a hand-built `SegmentInfo`/`FieldInfos`, so
//! this slice doesn't also need a `.si`/`.fnm` writer.
//!
//! One segment per shape `Lucene90NormsConsumer.addNormsField` can produce:
//! all five `numBytesPerValue` widths (constant, 1, 2, 4, 8 bytes per doc),
//! the sparse (`IndexedDISI`) docs-with-field structure, the "no document has
//! this field" marker, and several fields sharing one `.nvm`/`.nvd` pair.
//!
//! Run: `cargo run -p lucene-codecs --example write_norms_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::norms::{self, NormsField};
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittennrm01";

/// One field's expected per-doc norms, `None` for a doc with no value.
type Expected = Vec<Option<i64>>;

/// One fixture segment: its name, `maxDoc`, the fields written into its
/// `.nvm`/`.nvd` pair, and the per-field expected values the manifest records
/// for `VerifyNorms.java`.
struct Segment<'a> {
    name: &'a str,
    max_doc: i32,
    fields: Vec<NormsField<'a>>,
    expected: Vec<(i32, Expected)>,
}

fn dense(values: &[i64]) -> Expected {
    values.iter().map(|&v| Some(v)).collect()
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_norms_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    // Common case: varying small values (as real per-doc norms typically
    // are), all within a single signed byte -- forces bytesPerNorm == 1.
    let one_byte: Vec<i64> = vec![5, -100, 0, 127, -128, 42, 1, -1];
    // The `min >= max` all-equal encoding (`bytesPerNorm == 0`, every doc
    // decodes to the same constant, no per-doc array at all).
    let constant: Vec<i64> = vec![7; 6];
    // The three wider branches of `numBytesPerValue`, each just outside the
    // previous width's range so the narrower one cannot be chosen.
    let two_byte: Vec<i64> = vec![-129, 0, 300, i16::MAX as i64, i16::MIN as i64];
    let four_byte: Vec<i64> = vec![i16::MIN as i64 - 1, 0, 70_000, i32::MAX as i64];
    let eight_byte: Vec<i64> = vec![i32::MIN as i64 - 1, 0, 5_000_000_000, i64::MAX];
    // Sparse: only some docs have a norm (first and last doc missing, plus an
    // interior gap), so the writer emits an IndexedDISI structure and a
    // rank-indexed value array.
    let sparse_docs: [(i32, i64); 4] = [(1, 100), (4, -300), (6, 350), (7, 400)];
    let sparse_max_doc = 10;

    let dir = FsDirectory::open(&out_dir);
    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();

    let mut sparse_expected: Expected = vec![None; sparse_max_doc as usize];
    for &(doc, v) in &sparse_docs {
        sparse_expected[doc as usize] = Some(v);
    }
    // Several fields in one pair, the shape a real segment with more than one
    // normed field always takes: a dense wide-valued field, a sparse one, and
    // a constant one, interleaved into the same `.nvm`/`.nvd`.
    let multi_dense: Vec<i64> = (0..sparse_max_doc as i64).map(|d| d * 1000).collect();
    let multi_constant: Vec<i64> = vec![3; sparse_max_doc as usize];

    let segments = vec![
        Segment {
            name: "_0",
            max_doc: one_byte.len() as i32,
            fields: vec![NormsField::Dense(0, &one_byte)],
            expected: vec![(0, dense(&one_byte))],
        },
        Segment {
            name: "_1",
            max_doc: constant.len() as i32,
            fields: vec![NormsField::Dense(0, &constant)],
            expected: vec![(0, dense(&constant))],
        },
        Segment {
            name: "_2",
            max_doc: two_byte.len() as i32,
            fields: vec![NormsField::Dense(0, &two_byte)],
            expected: vec![(0, dense(&two_byte))],
        },
        Segment {
            name: "_3",
            max_doc: four_byte.len() as i32,
            fields: vec![NormsField::Dense(0, &four_byte)],
            expected: vec![(0, dense(&four_byte))],
        },
        Segment {
            name: "_4",
            max_doc: eight_byte.len() as i32,
            fields: vec![NormsField::Dense(0, &eight_byte)],
            expected: vec![(0, dense(&eight_byte))],
        },
        Segment {
            name: "_5",
            max_doc: sparse_max_doc,
            fields: vec![NormsField::Sparse(0, &sparse_docs)],
            expected: vec![(0, sparse_expected.clone())],
        },
        Segment {
            name: "_6",
            max_doc: sparse_max_doc,
            fields: vec![
                NormsField::Dense(0, &multi_dense),
                NormsField::Sparse(1, &sparse_docs),
                NormsField::Dense(2, &multi_constant),
            ],
            expected: vec![
                (0, dense(&multi_dense)),
                (1, sparse_expected),
                (2, dense(&multi_constant)),
            ],
        },
    ];

    let names: Vec<&str> = segments.iter().map(|s| s.name).collect();
    writeln!(manifest, "segments={}", names.join(",")).unwrap();

    for Segment {
        name,
        max_doc,
        fields,
        expected,
    } in &segments
    {
        let (meta, data) =
            norms::write_fields(fields, *max_doc, &SEGMENT_ID, "").expect("norms field write");

        let mut files = Vec::new();
        for (suffix, bytes) in [("nvm", &meta), ("nvd", &data)] {
            let file_name = format!("{name}.{suffix}");
            let mut out = dir.create_output(&file_name).unwrap();
            out.write_bytes(bytes);
            out.close().unwrap();
            files.push(file_name);
        }
        dir.sync(&files).unwrap();

        writeln!(manifest, "{name}.max_doc={max_doc}").unwrap();
        let numbers: Vec<String> = expected.iter().map(|(n, _)| n.to_string()).collect();
        writeln!(manifest, "{name}.field_numbers={}", numbers.join(",")).unwrap();
        for (number, values) in expected {
            // `-` marks a doc with no norm, so the sparse shape round-trips
            // through a positional list.
            let rendered: Vec<String> = values
                .iter()
                .map(|v| v.map_or("-".to_string(), |v| v.to_string()))
                .collect();
            writeln!(manifest, "{name}.{number}.values={}", rendered.join(";")).unwrap();
        }
    }

    println!("wrote norms fixture to {out_dir}");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
