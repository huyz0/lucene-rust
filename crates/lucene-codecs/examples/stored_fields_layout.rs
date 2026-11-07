//! Each chunk's place in the `.fdt` for each stored-fields mode, on a corpus
//! of one JSON `_source` per line: per chunk `start, end, doc base, docs`
//! and its documents' plain offsets, as TSV under `OUT_DIR`, for modelling
//! which file pages a document fetch reads.
//!
//! Run: `cargo run --release -p lucene-codecs --example stored_fields_layout -- FILE OUT_DIR`

// Benchmark arithmetic on its own counters, not on index bytes.
#![allow(clippy::arithmetic_side_effects)]

use std::fmt::Write as _;

use lucene_codecs::stored_fields::{
    self, Document, FieldValue, Mode, StoredField, StoredFieldsWriter, ZstdMode,
};

const ID: [u8; 16] = *b"zstd-bench-segid";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 2, "usage: stored_fields_layout FILE OUT_DIR");
    let text = std::fs::read_to_string(&args[0]).unwrap();
    let docs: Vec<Document> = text
        .lines()
        .enumerate()
        .map(|(i, line)| Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::Binary(i.to_string().into_bytes()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Binary(line.as_bytes().to_vec()),
                },
            ],
        })
        .collect();
    for (name, mode, train) in [
        ("lz4", Mode::BestSpeed, false),
        ("deflate", Mode::BestCompression, false),
        ("zstd", Mode::Zstd(ZstdMode::zstd(3)), false),
        ("paged", Mode::ZstdPaged(3), true),
    ] {
        let write = |dict: Option<&[u8]>| {
            let mut w = StoredFieldsWriter::new(mode, &ID, "");
            if let Some(d) = dict {
                w.set_zstd_dictionary(d);
            }
            for d in &docs {
                w.add_document(d);
            }
            w.finish()
        };
        let (mut fdt, mut fdx, mut fdm) = write(None);
        if train {
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &ID, "").unwrap();
            let dict = stored_fields::train_zstd_paged_dictionary(&[&reader])
                .unwrap()
                .unwrap();
            (fdt, fdx, fdm) = write(Some(&dict));
        }
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &ID, "").unwrap();
        let mut out = format!("# {name} fdt={}\n", fdt.len());
        let mut doc = 0;
        while doc < reader.max_doc() {
            let start = reader.chunk_for_doc(doc).unwrap().start_pointer;
            let chunk = reader.read_chunk(doc).unwrap();
            let next = chunk.doc_base() + chunk.num_docs();
            let end = if next < reader.max_doc() {
                reader.chunk_for_doc(next).unwrap().start_pointer
            } else {
                reader.max_pointer()
            };
            let mut offsets = Vec::new();
            let mut at = 0usize;
            for d in chunk.doc_base()..next {
                offsets.push(at.to_string());
                at += chunk.document(d).unwrap().1.len();
            }
            offsets.push(at.to_string());
            writeln!(
                out,
                "{start}\t{end}\t{}\t{}\t{}",
                chunk.doc_base(),
                chunk.num_docs(),
                offsets.join(",")
            )
            .unwrap();
            doc = next;
        }
        std::fs::write(format!("{}/{name}.tsv", args[1]), out).unwrap();
        println!("{name}: {} bytes", fdt.len());
    }
}
