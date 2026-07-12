//! Writes a `write_best_speed`-produced `.fdt`/`.fdx`/`.fdm` triple plus a
//! manifest to the directory given as the first CLI argument.
//!
//! This is the reverse of this repo's usual differential-testing direction
//! (Java writes, Rust reads): here Rust writes, and
//! `fixtures/src/VerifyStoredFields.java` reads the result back through
//! real Lucene's own `Lucene90StoredFieldsFormat.fieldsReader`, constructing
//! a matching `SegmentInfo`/`FieldInfos` directly in Java code rather than
//! also requiring Rust to write `.si`/`.fnm` -- this keeps the first
//! write-path slice scoped to exactly the stored-fields format itself.
//!
//! Run: `cargo run -p lucene-codecs --example write_stored_fields_fixture -- <dir>`

use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittenseg01";

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_stored_fields_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    let docs = vec![
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("hello world".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(-42),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(1_234_567_890_123),
                },
                StoredField {
                    field_number: 3,
                    value: FieldValue::Float(1.5),
                },
                StoredField {
                    field_number: 4,
                    value: FieldValue::Double(2.25),
                },
                StoredField {
                    field_number: 5,
                    value: FieldValue::Binary(vec![1, 2, 3, 4, 5]),
                },
                StoredField {
                    field_number: 6,
                    // Genuine repetition, to prove `write_best_speed` now
                    // produces real LZ4 back-reference compression (via
                    // `lz4::compress`), not just a literal-wrapped block --
                    // real Lucene must decode the actually-compressed bytes
                    // here, not merely a valid-but-uncompressed unit.
                    value: FieldValue::String(
                        "the quick brown fox jumps over the lazy dog ".repeat(50),
                    ),
                },
            ],
        },
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("second document".to_string()),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(-9_999_999_999),
                },
            ],
        },
        Document { fields: vec![] },
    ];

    let (fdt, fdx, fdm) = stored_fields::write_best_speed(&docs, &SEGMENT_ID, "");

    // Route the encoded bytes through the real on-disk Directory/IndexOutput
    // primitive (rather than a hand-rolled `std::fs::write`), then fsync
    // them -- exercising the same write→sync contract a real IndexWriter
    // uses before referencing a segment's files from a commit.
    let dir = FsDirectory::open(&out_dir);
    for (name, bytes) in [("_0.fdt", &fdt), ("_0.fdx", &fdx), ("_0.fdm", &fdm)] {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }
    dir.sync(&[
        "_0.fdt".to_string(),
        "_0.fdx".to_string(),
        "_0.fdm".to_string(),
    ])
    .unwrap();

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "max_doc={}", docs.len()).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "num_fields=7").unwrap();
    for (doc_id, doc) in docs.iter().enumerate() {
        let rendered: Vec<String> = doc
            .fields
            .iter()
            .map(|f| {
                let (ty, val) = match &f.value {
                    FieldValue::String(s) => ("string".to_string(), s.clone()),
                    FieldValue::Binary(b) => ("binary".to_string(), hex(b)),
                    FieldValue::Int(v) => ("int".to_string(), v.to_string()),
                    FieldValue::Long(v) => ("long".to_string(), v.to_string()),
                    FieldValue::Float(v) => ("float".to_string(), v.to_string()),
                    FieldValue::Double(v) => ("double".to_string(), v.to_string()),
                };
                format!("{}:{ty}:{val}", f.field_number)
            })
            .collect();
        writeln!(manifest, "doc.{doc_id}.fields={}", rendered.join(";")).unwrap();
    }

    println!("wrote stored-fields fixture to {out_dir}");
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
