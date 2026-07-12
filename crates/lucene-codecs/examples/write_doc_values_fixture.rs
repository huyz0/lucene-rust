//! Writes a `doc_values::write_single_dense_numeric_field`-produced
//! `.dvm`/`.dvd`/`.dvs` triple plus a manifest to the directory given as the
//! first CLI argument.
//!
//! Reverse-direction differential test (Rust writes, Java reads), same
//! division of labor as `write_points_fixture.rs`: `fixtures/src/
//! VerifyDocValues.java` opens the result through real Lucene's
//! `Lucene90DocValuesFormat` with a hand-built `SegmentInfo`/`FieldInfos`,
//! so this slice doesn't also need a `.si`/`.fnm` writer.
//!
//! Scoped to exactly one shape: a single NUMERIC field, dense (every doc has
//! a value), plain delta-compressed encoding -- see
//! `doc_values::write_single_dense_numeric_field`'s doc comment for the full
//! list of what's deliberately out of scope.
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

    let segments: [(&str, &[i64]); 3] = [
        ("_0", &values),
        ("_1", &shift_drop_values),
        ("_2", &constant_values),
    ];

    let dir = FsDirectory::open(&out_dir);
    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "segments=_0,_1,_2").unwrap();

    for (name, seg_values) in segments {
        let seg_max_doc = seg_values.len() as i32;
        let (meta, data, skip_index) = doc_values::write_single_dense_numeric_field(
            FIELD_NUMBER,
            seg_values,
            seg_max_doc,
            &SEGMENT_ID,
            "",
        )
        .expect("single dense numeric field write");

        let mut files = Vec::new();
        for (suffix, bytes) in [("dvm", &meta), ("dvd", &data), ("dvs", &skip_index)] {
            let file_name = format!("{name}.{suffix}");
            let mut out = dir.create_output(&file_name).unwrap();
            out.write_bytes(bytes);
            out.close().unwrap();
            files.push(file_name);
        }
        dir.sync(&files).unwrap();

        writeln!(manifest, "{name}.max_doc={seg_max_doc}").unwrap();
        writeln!(manifest, "{name}.field_number={FIELD_NUMBER}").unwrap();
        let rendered: Vec<String> = seg_values.iter().map(|v| v.to_string()).collect();
        writeln!(manifest, "{name}.values={}", rendered.join(";")).unwrap();
    }

    println!("wrote doc values fixture to {out_dir}");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
