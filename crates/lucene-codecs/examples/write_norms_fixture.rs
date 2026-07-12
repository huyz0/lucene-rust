//! Writes a `norms::write_single_dense_field`-produced `.nvm`/`.nvd` pair
//! plus a manifest to the directory given as the first CLI argument.
//!
//! Reverse-direction differential test (Rust writes, Java reads), same
//! division of labor as `write_doc_values_fixture.rs`: `fixtures/src/
//! VerifyNorms.java` opens the result through real Lucene's
//! `Lucene90NormsFormat` with a hand-built `SegmentInfo`/`FieldInfos`, so
//! this slice doesn't also need a `.si`/`.fnm` writer.
//!
//! Scoped to exactly one shape: a single norms field, dense (every doc has a
//! value), at most 1 byte per doc -- see
//! `norms::write_single_dense_field`'s doc comment for the full list of
//! what's deliberately out of scope.
//!
//! Run: `cargo run -p lucene-codecs --example write_norms_fixture -- <dir>`

use lucene_codecs::norms;
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittennrm01";
const FIELD_NUMBER: i32 = 0;

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_norms_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    // Common case: varying small values (as real per-doc norms typically
    // are), all within a single signed byte -- forces bytesPerNorm == 1.
    let values: Vec<i64> = vec![5, -100, 0, 127, -128, 42, 1, -1];

    // Regression case for the `min >= max` all-equal encoding
    // (`bytesPerNorm == 0`, every doc decodes to the same constant) -- the
    // doc-values write-side review found this exact branch shape
    // previously verified only against this port's own reader, not real
    // Lucene.
    let constant_values: Vec<i64> = vec![7; 6];

    let segments: [(&str, &[i64]); 2] = [("_0", &values), ("_1", &constant_values)];

    let dir = FsDirectory::open(&out_dir);
    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "segments=_0,_1").unwrap();

    for (name, seg_values) in segments {
        let seg_max_doc = seg_values.len() as i32;
        let (meta, data) =
            norms::write_single_dense_field(FIELD_NUMBER, seg_values, seg_max_doc, &SEGMENT_ID, "")
                .expect("single dense norms field write");

        let mut files = Vec::new();
        for (suffix, bytes) in [("nvm", &meta), ("nvd", &data)] {
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

    println!("wrote norms fixture to {out_dir}");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
