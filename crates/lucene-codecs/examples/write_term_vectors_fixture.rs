//! Writes a `write_best_speed`-produced `.tvd`/`.tvx`/`.tvm` triple plus a
//! manifest to the directory given as the first CLI argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyTermVectors.java` reads the result back through real
//! Lucene's own `Lucene90TermVectorsFormat`/`TermVectorsReader`, constructing
//! a matching `SegmentInfo`/`FieldInfos` directly in Java code (no `.si`/
//! `.fnm` needed from Rust), same pattern as `write_stored_fields_fixture.rs`.
//!
//! Scope note: `term_vectors::write_best_speed` only supports positions (no
//! offsets/payloads, no prefix sharing, single chunk) -- see its doc comment.
//!
//! Run: `cargo run -p lucene-codecs --example write_term_vectors_fixture -- <dir>`

use lucene_codecs::term_vectors::{self, TermVectorField, TermVectorTerm, TermVectorsDocument};
use lucene_store::{DataOutput, Directory, FsDirectory};
use std::io::Write;

const SEGMENT_ID: [u8; 16] = *b"rustwrittenseg02";

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: write_term_vectors_fixture <output-dir>");
    std::fs::create_dir_all(&out_dir).unwrap();

    let docs = vec![
        TermVectorsDocument {
            fields: vec![
                TermVectorField {
                    field_number: 0,
                    has_positions: true,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![
                        TermVectorTerm {
                            term: b"cat".to_vec(),
                            freq: 2,
                            positions: Some(vec![0, 3]),
                            start_offsets: None,
                            end_offsets: None,
                            payloads: None,
                        },
                        TermVectorTerm {
                            term: b"dog".to_vec(),
                            freq: 1,
                            positions: Some(vec![1]),
                            start_offsets: None,
                            end_offsets: None,
                            payloads: None,
                        },
                    ],
                },
                TermVectorField {
                    field_number: 1,
                    has_positions: false,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: b"hello".to_vec(),
                        freq: 1,
                        positions: None,
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                },
            ],
        },
        TermVectorsDocument { fields: vec![] },
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"gamma".to_vec(),
                    freq: 3,
                    positions: Some(vec![0, 1, 5]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        },
    ];

    // Regression case: every field across every doc in this chunk has
    // field_number == 0 (an entirely ordinary shape -- any single-field
    // index). This makes `max_field_num == 0`, which previously encoded
    // `bits_per_field_num` as 0 -- wire-format-valid for this port's own
    // (more permissive) reader, but real Lucene's reader unconditionally
    // indexes `packedBulkOps[bitsPerValue - 1]` and throws
    // `ArrayIndexOutOfBoundsException` on a 0-bit width. Written as a
    // second segment ("_1") so the primary multi-field-number fixture above
    // (which never hits this, since it always mixes field numbers 0 and 1)
    // is left untouched.
    let all_zero_docs = vec![
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"cat".to_vec(),
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        },
        TermVectorsDocument {
            fields: vec![TermVectorField {
                field_number: 0,
                has_positions: true,
                has_offsets: false,
                has_payloads: false,
                terms: vec![TermVectorTerm {
                    term: b"dog".to_vec(),
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: None,
                    end_offsets: None,
                    payloads: None,
                }],
            }],
        },
    ];

    let (tvd, tvx, tvm) = term_vectors::write_best_speed(&docs, &SEGMENT_ID, "");
    let (az_tvd, az_tvx, az_tvm) = term_vectors::write_best_speed(&all_zero_docs, &SEGMENT_ID, "");

    let dir = FsDirectory::open(&out_dir);
    for (name, bytes) in [
        ("_0.tvd", &tvd),
        ("_0.tvx", &tvx),
        ("_0.tvm", &tvm),
        ("_1.tvd", &az_tvd),
        ("_1.tvx", &az_tvx),
        ("_1.tvm", &az_tvm),
    ] {
        let mut out = dir.create_output(name).unwrap();
        out.write_bytes(bytes);
        out.close().unwrap();
    }
    dir.sync(&[
        "_0.tvd".to_string(),
        "_0.tvx".to_string(),
        "_0.tvm".to_string(),
        "_1.tvd".to_string(),
        "_1.tvx".to_string(),
        "_1.tvm".to_string(),
    ])
    .unwrap();

    let mut manifest = std::fs::File::create(format!("{out_dir}/manifest.properties")).unwrap();
    writeln!(manifest, "max_doc={}", docs.len()).unwrap();
    writeln!(manifest, "id_hex={}", hex(&SEGMENT_ID)).unwrap();
    writeln!(manifest, "num_fields=2").unwrap();
    for (doc_id, doc) in docs.iter().enumerate() {
        writeln!(manifest, "doc.{doc_id}.fields={}", render_doc(doc)).unwrap();
    }

    writeln!(manifest, "all_zero.max_doc={}", all_zero_docs.len()).unwrap();
    writeln!(manifest, "all_zero.num_fields=1").unwrap();
    for (doc_id, doc) in all_zero_docs.iter().enumerate() {
        writeln!(manifest, "all_zero.doc.{doc_id}.fields={}", render_doc(doc)).unwrap();
    }

    println!("wrote term-vectors fixture to {out_dir}");
}

fn render_doc(doc: &TermVectorsDocument) -> String {
    doc.fields
        .iter()
        .map(|f| {
            let terms: Vec<String> = f
                .terms
                .iter()
                .map(|t| {
                    let term_str = String::from_utf8(t.term.clone()).unwrap();
                    let positions = t
                        .positions
                        .as_ref()
                        .map(|p| {
                            p.iter()
                                .map(|v| v.to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        })
                        .unwrap_or_default();
                    format!("{term_str}:{}:{positions}", t.freq)
                })
                .collect();
            format!("{}[{}]", f.field_number, terms.join(","))
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
