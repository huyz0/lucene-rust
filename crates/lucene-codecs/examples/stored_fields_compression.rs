//! Stored-fields size and speed for each compression this port writes, on a
//! corpus of one JSON `_source` per line (`scripts/zstd/make-corpus.py`).
//! `scripts/zstd/StoredFieldsBench.java` measures stock Lucene and OpenSearch's
//! custom-codecs on the same documents and prints the same columns.
//!
//! Each document is OpenSearch's shape: `_id` (its line number) and `_source`
//! (the line), both binary. Per mode: the `.fdt` size; write throughput; the
//! mean time of one `document()` at 100k seeded random ids (the fetch phase);
//! and a full in-order scan through a [`ChunkCursor`] (a merge or reindex).
//!
//! Run: `cargo run --release -p lucene-codecs --example stored_fields_compression -- FILE...`

// Benchmark arithmetic on its own counters, not on index bytes.
#![allow(clippy::arithmetic_side_effects)]

use std::time::Instant;

use lucene_codecs::stored_fields::{
    self, ChunkCursor, Document, FieldValue, Mode, StoredField, StoredFieldsWriter, ZstdMode,
};

const ID: [u8; 16] = *b"zstd-bench-segid";
const RANDOM_READS: usize = 100_000;

fn modes() -> Vec<(String, Mode)> {
    let mut m = vec![
        ("lz4 (default)".to_string(), Mode::BestSpeed),
        ("deflate (best_compression)".to_string(), Mode::BestCompression),
    ];
    for level in [1, 3, 6] {
        m.push((format!("zstd level {level}"), Mode::Zstd(ZstdMode::zstd(level))));
    }
    for level in [1, 3, 6] {
        m.push((
            format!("zstd_no_dict level {level}"),
            Mode::Zstd(ZstdMode::zstd_no_dict(level)),
        ));
    }
    m
}

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    assert!(!files.is_empty(), "usage: stored_fields_compression FILE...");
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap();
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
        let raw: usize = text.lines().map(|l| l.len()).sum();
        println!(
            "{file}: {} docs, {:.1} MiB of _source",
            docs.len(),
            raw as f64 / (1 << 20) as f64
        );
        println!("| mode | .fdt MiB | ratio | write MB/s | random doc us | scan MB/s |");
        println!("|---|---|---|---|---|---|");
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let ids: Vec<i32> = (0..RANDOM_READS)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % docs.len() as u64) as i32
            })
            .collect();
        for (name, mode) in modes() {
            // Best of three for each timing.
            let mut write_s = f64::MAX;
            let mut files = None;
            for _ in 0..3 {
                let t = Instant::now();
                let mut w = StoredFieldsWriter::new(mode, &ID, "");
                for d in &docs {
                    w.add_document(d);
                }
                let out = w.finish();
                write_s = write_s.min(t.elapsed().as_secs_f64());
                files = Some(out);
            }
            let (fdt, fdx, fdm) = files.unwrap();
            let reader = stored_fields::open(&fdt, &fdx, &fdm, &ID, "").unwrap();
            let mut random_s = f64::MAX;
            let mut check = 0usize;
            for _ in 0..3 {
                let t = Instant::now();
                for &id in &ids {
                    let d = reader.document(id).unwrap();
                    check += d.fields.len();
                }
                random_s = random_s.min(t.elapsed().as_secs_f64());
            }
            let mut scan_s = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                let mut cursor = ChunkCursor::new();
                for id in 0..docs.len() as i32 {
                    let (_, bytes) = cursor.document(&reader, id).unwrap();
                    check += bytes.len();
                }
                scan_s = scan_s.min(t.elapsed().as_secs_f64());
            }
            assert!(check > 0);
            let mb = raw as f64 / 1e6;
            println!(
                "| {name} | {:.2} | {:.2} | {:.0} | {:.2} | {:.0} |",
                fdt.len() as f64 / (1 << 20) as f64,
                raw as f64 / fdt.len() as f64,
                mb / write_s,
                random_s * 1e6 / RANDOM_READS as f64,
                mb / scan_s,
            );
        }
        println!();
    }
}
