//! Writes a `write_best_speed`-produced `.tvd`/`.tvx`/`.tvm` triple plus a
//! manifest to the directory given as the first CLI argument.
//!
//! Reverse-direction fixture (Rust writes, Java reads): the counterpart
//! `fixtures/src/VerifyTermVectors.java` reads the result back through real
//! Lucene's own `Lucene90TermVectorsFormat`/`TermVectorsReader`, constructing
//! a matching `SegmentInfo`/`FieldInfos` directly in Java code (no `.si`/
//! `.fnm` needed from Rust), same pattern as `write_stored_fields_fixture.rs`.
//!
//! Three segments are written: `_0` (the original multi-field-number
//! positions fixture), `_1` (every field number 0, a `bits_per_field_num`
//! regression case), and `_2` -- added by batch `c8-tv-chunking` -- a
//! **multi-chunk** segment with offsets and payloads and prefix-shared terms,
//! which is the only one that exercises the writer's real
//! `Lucene90CompressingTermVectorsWriter.flush` chunking (4 096 bytes / 128
//! documents), `flushOffsets`' derived `charsPerTerm`, `flushFlags`'
//! per-field-number encoding, and `startTerm`'s prefix compression. Batch c33
//! added its last document (400), whose terms are **non-ASCII with Java
//! `char` offsets** -- the shape `lucene_analysis` now emits -- so the
//! negative `length - prefixLength - suffixLength` values that follow from a
//! term occupying more bytes than it spans `char`s are read back by real
//! Lucene rather than only by this port's own reader.
//!
//! Run: `cargo run -p lucene-codecs --example write_term_vectors_fixture -- <dir>`
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

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

    // "_2": a multi-chunk segment. 400 documents at ~40 bytes of term text
    // each cross both of Java's flush triggers (4 096 bytes and 128
    // documents) many times over, so the `.tvd` holds a dozen-odd chunks --
    // and every one of them has to be found through `.tvx`'s
    // `DirectMonotonicReader` by real Lucene. Two fields per document with
    // constant flags exercise `flushFlags`' `nonChangingFlags` (selector 0)
    // encoding; the first field carries offsets and payloads, so
    // `flushOffsets`' `charsPerTerm` correction and `flushPayloadLengths`
    // are both live; consecutive terms share a long prefix, so
    // `startTerm`'s `bytesDifference` compression is too.
    let mut chunked_docs: Vec<TermVectorsDocument> = (0..400)
        .map(|n: i32| TermVectorsDocument {
            fields: vec![
                TermVectorField {
                    field_number: 0,
                    has_positions: true,
                    has_offsets: true,
                    has_payloads: true,
                    terms: (0..3)
                        .map(|t: i32| {
                            let position = t * 4 + n % 3;
                            let start = position * 7;
                            TermVectorTerm {
                                // A shared "chunked" prefix plus an ascending
                                // suffix: terms within a field must ascend
                                // (real Lucene's `CheckIndex` requires it)
                                // and the shared prefix is what
                                // `bytesDifference` compresses away.
                                term: format!("chunked{:02}{:04}", t, n).into_bytes(),
                                freq: 2,
                                positions: Some(vec![position, position + 1]),
                                start_offsets: Some(vec![start, start + 7]),
                                end_offsets: Some(vec![start + 6, start + 13]),
                                payloads: Some(vec![
                                    vec![(n % 251) as u8],
                                    if t % 2 == 0 { vec![] } else { vec![9, 9] },
                                ]),
                            }
                        })
                        .collect(),
                },
                TermVectorField {
                    field_number: 1,
                    has_positions: true,
                    has_offsets: false,
                    has_payloads: false,
                    terms: vec![TermVectorTerm {
                        term: format!("tail{n:04}").into_bytes(),
                        freq: 1,
                        positions: Some(vec![n % 11]),
                        start_offsets: None,
                        end_offsets: None,
                        payloads: None,
                    }],
                },
            ],
        })
        .collect();

    // Document 400, added by batch c33: **non-ASCII terms with Java `char`
    // offsets**, the shape `lucene_analysis` now produces for any non-ASCII
    // text. Every other document here is pure ASCII, where a term's UTF-8
    // byte length and its offset span are equal and
    // `flushOffsets`' `length - prefixLength - suffixLength` is therefore
    // always >= 0. With `OffsetAttribute`'s real unit that length goes
    // **negative** for every multi-byte term (`caf\u{e9}`: a 4-`char` span
    // over 5 bytes, so -1; `\u{4e16}`: 1 over 3, so -2), which is a
    // `block_packed` min-value-framing path no ASCII fixture can reach and
    // which real Lucene's reader has to undo exactly the same way.
    //
    // The offsets are the ones a real `StandardAnalyzer` reports for
    // "caf\u{e9} \u{4e16}\u{754c} dog" (see the `utf16_*` cases in
    // `fixtures/data/analysis/manifest.properties`). Terms ascend in byte
    // order, and the field's flags match every other document's so the chunk
    // still takes `flushFlags`' non-changing-flags encoding.
    chunked_docs.push(TermVectorsDocument {
        fields: vec![TermVectorField {
            field_number: 0,
            has_positions: true,
            has_offsets: true,
            has_payloads: true,
            terms: vec![
                TermVectorTerm {
                    term: "caf\u{e9}".as_bytes().to_vec(),
                    freq: 1,
                    positions: Some(vec![0]),
                    start_offsets: Some(vec![0]),
                    end_offsets: Some(vec![4]),
                    payloads: Some(vec![vec![1]]),
                },
                TermVectorTerm {
                    term: b"dog".to_vec(),
                    freq: 1,
                    positions: Some(vec![3]),
                    start_offsets: Some(vec![8]),
                    end_offsets: Some(vec![11]),
                    payloads: Some(vec![vec![]]),
                },
                TermVectorTerm {
                    term: "\u{4e16}".as_bytes().to_vec(),
                    freq: 1,
                    positions: Some(vec![1]),
                    start_offsets: Some(vec![5]),
                    end_offsets: Some(vec![6]),
                    payloads: Some(vec![vec![2, 2]]),
                },
                TermVectorTerm {
                    term: "\u{754c}".as_bytes().to_vec(),
                    freq: 1,
                    positions: Some(vec![2]),
                    start_offsets: Some(vec![6]),
                    end_offsets: Some(vec![7]),
                    payloads: Some(vec![vec![3]]),
                },
            ],
        }],
    });

    let (tvd, tvx, tvm) = term_vectors::write_best_speed(&docs, &SEGMENT_ID, "");
    let (az_tvd, az_tvx, az_tvm) = term_vectors::write_best_speed(&all_zero_docs, &SEGMENT_ID, "");
    let (ch_tvd, ch_tvx, ch_tvm) = term_vectors::write_best_speed(&chunked_docs, &SEGMENT_ID, "");

    let dir = FsDirectory::open(&out_dir);
    for (name, bytes) in [
        ("_0.tvd", &tvd),
        ("_0.tvx", &tvx),
        ("_0.tvm", &tvm),
        ("_1.tvd", &az_tvd),
        ("_1.tvx", &az_tvx),
        ("_1.tvm", &az_tvm),
        ("_2.tvd", &ch_tvd),
        ("_2.tvx", &ch_tvx),
        ("_2.tvm", &ch_tvm),
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
        "_2.tvd".to_string(),
        "_2.tvx".to_string(),
        "_2.tvm".to_string(),
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

    writeln!(manifest, "chunked.max_doc={}", chunked_docs.len()).unwrap();
    writeln!(manifest, "chunked.num_fields=2").unwrap();
    for (doc_id, doc) in chunked_docs.iter().enumerate() {
        writeln!(
            manifest,
            "chunked.doc.{doc_id}.fields={}",
            render_doc_full(doc)
        )
        .unwrap();
    }

    println!("wrote term-vectors fixture to {out_dir}");
}

/// Like [`render_doc`], but also renders each occurrence's start/end offsets
/// and payload bytes -- the streams only the `_2` segment carries.
fn render_doc_full(doc: &TermVectorsDocument) -> String {
    doc.fields
        .iter()
        .map(|f| {
            let terms: Vec<String> = f
                .terms
                .iter()
                .map(|t| {
                    let term_str = String::from_utf8(t.term.clone()).unwrap();
                    // A field without offsets renders `-1` per occurrence,
                    // and one without payloads an empty string per
                    // occurrence -- `PostingsEnum.startOffset()`/
                    // `getPayload()`'s own no-offsets/no-payload contract,
                    // which is what the Java verifier reads back.
                    let join = |v: &Option<Vec<i32>>| match v {
                        Some(p) => p
                            .iter()
                            .map(|x| x.to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                        None => vec!["-1"; t.freq as usize].join(","),
                    };
                    let payloads = match &t.payloads {
                        Some(ps) => ps.iter().map(|p| hex(p)).collect::<Vec<_>>().join(","),
                        None => vec![""; t.freq as usize].join(","),
                    };
                    format!(
                        "{term_str}:{}:{}:{}:{}:{}",
                        t.freq,
                        join(&t.positions),
                        join(&t.start_offsets),
                        join(&t.end_offsets),
                        payloads
                    )
                })
                .collect();
            format!("{}[{}]", f.field_number, terms.join(","))
        })
        .collect::<Vec<_>>()
        .join(";")
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
