//! Writes two `.fdt`/`.fdx`/`.fdm` triples plus a manifest to the directory
//! given as the first CLI argument: segment `_0` via `write_best_speed`
//! (LZ4, `Mode.BEST_SPEED`) and segment `_1` via `write_best_compression`
//! (DEFLATE, `Mode.BEST_COMPRESSION`).
//!
//! This is the reverse of this repo's usual differential-testing direction
//! (Java writes, Rust reads): here Rust writes, and
//! `fixtures/src/VerifyStoredFields.java` reads each segment back through
//! real Lucene's own `Lucene90StoredFieldsFormat.fieldsReader`, constructing
//! a matching `SegmentInfo`/`FieldInfos` directly in Java code rather than
//! also requiring Rust to write `.si`/`.fnm` -- this keeps the write-path
//! slice scoped to exactly the stored-fields format itself.
//!
//! Run: `cargo run -p lucene-codecs --example write_stored_fields_fixture -- <dir>`

use lucene_codecs::stored_fields::{self, Document, FieldValue, StoredField};
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID_0: [u8; 16] = *b"rustwrittenseg01";
const SEGMENT_ID_1: [u8; 16] = *b"rustwrittenseg02";

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_stored_fields_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    let docs_best_speed = vec![
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

    let docs_best_compression = vec![
        Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String("hello compression".to_string()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Int(1_000_000),
                },
                StoredField {
                    field_number: 2,
                    value: FieldValue::Long(-1_234_567_890_123),
                },
                StoredField {
                    field_number: 3,
                    value: FieldValue::Float(-3.5),
                },
                StoredField {
                    field_number: 4,
                    value: FieldValue::Double(-2.25),
                },
                StoredField {
                    field_number: 5,
                    value: FieldValue::Binary(vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
                },
                StoredField {
                    field_number: 6,
                    // Large enough (~90KB) that write_best_compression's
                    // dictLength = len/60, blockLength = ceil((len-dict)/10)
                    // framing produces a real dictionary unit AND several
                    // sub-blocks -- not just a single trivial DEFLATE unit --
                    // so real Lucene's reader must walk the whole
                    // dictionary + multi-sub-block decode path.
                    value: FieldValue::String(
                        "the quick brown fox jumps over the lazy dog ".repeat(2000),
                    ),
                },
            ],
        },
        Document {
            fields: vec![StoredField {
                field_number: 0,
                value: FieldValue::String("second compressed document".to_string()),
            }],
        },
        Document { fields: vec![] },
    ];

    let (fdt0, fdx0, fdm0) = stored_fields::write_best_speed(&docs_best_speed, &SEGMENT_ID_0, "");
    let (fdt1, fdx1, fdm1) =
        stored_fields::write_best_compression(&docs_best_compression, &SEGMENT_ID_1, "");

    // Route the encoded bytes through the real on-disk Directory/IndexOutput
    // primitive (rather than a hand-rolled `std::fs::write`), then fsync
    // them -- exercising the same write→sync contract a real IndexWriter
    // uses before referencing a segment's files from a commit.
    let dir = FsDirectory::open(&out_dir);
    for (name, bytes) in [
        ("_0.fdt", &fdt0),
        ("_0.fdx", &fdx0),
        ("_0.fdm", &fdm0),
        ("_1.fdt", &fdt1),
        ("_1.fdx", &fdx1),
        ("_1.fdm", &fdm1),
    ] {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }
    dir.sync(&[
        "_0.fdt".to_string(),
        "_0.fdx".to_string(),
        "_0.fdm".to_string(),
        "_1.fdt".to_string(),
        "_1.fdx".to_string(),
        "_1.fdm".to_string(),
    ])
    .unwrap();

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "segments=_0,_1").unwrap();
    write_segment_manifest(
        &mut manifest,
        "_0",
        "BEST_SPEED",
        &SEGMENT_ID_0,
        &docs_best_speed,
    );
    write_segment_manifest(
        &mut manifest,
        "_1",
        "BEST_COMPRESSION",
        &SEGMENT_ID_1,
        &docs_best_compression,
    );

    println!("wrote stored-fields fixture to {out_dir}");
}

fn write_segment_manifest(
    manifest: &mut std::fs::File,
    seg: &str,
    mode: &str,
    id: &[u8; 16],
    docs: &[Document],
) {
    writeln!(manifest, "{seg}.mode={mode}").unwrap();
    writeln!(manifest, "{seg}.max_doc={}", docs.len()).unwrap();
    writeln!(manifest, "{seg}.id_hex={}", hex(id)).unwrap();
    writeln!(manifest, "{seg}.num_fields=7").unwrap();
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
        writeln!(manifest, "{seg}.doc.{doc_id}.fields={}", rendered.join(";")).unwrap();
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
