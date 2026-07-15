//! Writes a `doc_values::write_single_sparse_numeric_field`-produced
//! `.dvm`/`.dvd`/`.dvs` triple plus a manifest to the directory given as the
//! first CLI argument.
//!
//! Reverse-direction differential test (Rust writes, Java reads), same
//! division of labor as `write_doc_values_fixture.rs`: `fixtures/src/
//! VerifySparseNumericDocValues.java` opens the result through real Lucene's
//! `Lucene90DocValuesFormat` with a hand-built `SegmentInfo`/`FieldInfos`, so
//! this slice doesn't also need a `.si`/`.fnm` writer.
//!
//! `write_single_sparse_numeric_field` was previously only checked against
//! this port's own reader (see `doc_values.rs`'s
//! `write_single_sparse_numeric_field_round_trips_through_own_reader` unit
//! test) -- never against a real, unmodified Lucene reader. This fixture
//! closes that gap.
//!
//! Run: `cargo run -p lucene-codecs --example write_sparse_numeric_doc_values_fixture -- <dir>`

use lucene_codecs::doc_values;
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittensnv01";
const FIELD_NUMBER: i32 = 0;

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_sparse_numeric_doc_values_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    // `_0`: a small segment (20 docs) with values missing on docs
    // *interspersed* throughout, not just trailing -- doc 0 has a value, doc
    // 1 doesn't, several consecutive docs in the middle are missing, and the
    // last doc has a value. This exercises IndexedDISI's sparse-bitset
    // decode on both sides of gaps, not just "docs run out at the end".
    let max_doc_0 = 20i32;
    let present_0: Vec<i32> = vec![0, 2, 3, 5, 6, 7, 8, 12, 15, 16, 19];
    let doc_values_0: Vec<(i32, i64)> = present_0
        .iter()
        .map(|&doc| (doc, (doc as i64) * 11 - 4))
        .collect();

    // `_1`: a larger segment (200,000 docs) with 1 of every 3 docs present,
    // forcing IndexedDISI to pick a shape per 65536-doc block from actual
    // density -- see `write_single_sparse_numeric_field_round_trips_through_
    // own_reader`'s doc comment in `doc_values.rs` for why 1/3 density lands
    // in the DENSE-bitset block shape (well above the 4095 SPARSE
    // threshold). This is the same shape that unit test already covers
    // against this port's own reader; here it's checked against real Lucene
    // too.
    let max_doc_1 = 200_000i32;
    let doc_values_1: Vec<(i32, i64)> = (0..max_doc_1)
        .step_by(3)
        .map(|doc| (doc, (doc as i64) * 7 - 3))
        .collect();

    let dir = FsDirectory::open(&out_dir);
    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "segments=_0,_1").unwrap();

    for (name, max_doc, seg_doc_values) in [
        ("_0", max_doc_0, &doc_values_0),
        ("_1", max_doc_1, &doc_values_1),
    ] {
        let (meta, data, skip_index) = doc_values::write_single_sparse_numeric_field(
            FIELD_NUMBER,
            seg_doc_values,
            max_doc,
            &SEGMENT_ID,
            "",
        )
        .expect("single sparse numeric field write");
        write_triple(&dir, name, &meta, &data, &skip_index);

        writeln!(manifest, "{name}.type=NUMERIC").unwrap();
        writeln!(manifest, "{name}.max_doc={max_doc}").unwrap();
        writeln!(manifest, "{name}.field_number={FIELD_NUMBER}").unwrap();
        if name == "_1" {
            // `_1` has ~66,667 present docs -- enumerating every `doc:value`
            // pair would bloat the checked-in manifest to ~900KB, an order
            // of magnitude past every sibling fixture (see
            // fixtures/README.md's "small, deterministic" convention).
            // Every present doc/value here follows a fixed arithmetic
            // pattern (`step_by(3)`, `doc * 7 - 3`), so the manifest encodes
            // that formula instead; the verifier reconstructs the expected
            // map from it.
            writeln!(manifest, "{name}.step=3").unwrap();
            writeln!(manifest, "{name}.value_mul=7").unwrap();
            writeln!(manifest, "{name}.value_sub=3").unwrap();
        } else {
            let rendered: Vec<String> = seg_doc_values
                .iter()
                .map(|(doc, v)| format!("{doc}:{v}"))
                .collect();
            writeln!(manifest, "{name}.values={}", rendered.join(";")).unwrap();
        }
    }

    println!("wrote sparse numeric doc values fixture to {out_dir}");
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
