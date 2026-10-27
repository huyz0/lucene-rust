//! Writes indexes through `IndexWriter` with the test framework's six
//! `CompressingCodec`s installed (`IndexWriter::set_codec`), for
//! `VerifyCompressingCodecs`: each of [`CASES`] is one codec with its own
//! chunk size, documents per chunk, block shift and segment suffix choice,
//! small enough that chunks fill on both triggers and the occasional large
//! stored value makes a sliced chunk.
//!
//! Every document stores `id`, `body` (with term vectors: positions and
//! offsets) and the int `n`; every [`BIG_EVERY`]th also stores `big`. Two
//! flushed segments of [`PER_SEGMENT`] documents, then every [`deleted`]
//! document deleted, and, when `merge` is asked for, a force-merge to one.
//! `codec.txt` holds the codec's `toString()`.
//!
//! Usage: `write_compressing_codec_fixture <output-dir>`: every case into
//! `<output-dir>/<case>/flushed` and `<output-dir>/<case>/merged`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]
#![allow(dead_code)]

use std::sync::Arc;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, Field, FieldType, Store, StoredValue, StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::{DataOutput, Directory};
use lucene_test_framework::codecs::compressing_codec::{CodecKind, CompressingCodec};

/// One installed codec: `new <Kind>CompressingCodec(chunkSize,
/// maxDocsPerChunk, withSegmentSuffix, blockShift)`, or the no-argument
/// constructor.
pub struct Case {
    pub name: &'static str,
    pub kind: CodecKind,
    /// `None` for the subclass's defaults.
    pub settings: Option<(i32, i32, bool, i32)>,
}

pub const CASES: [Case; 8] = [
    Case {
        name: "fast-default",
        kind: CodecKind::Fast,
        settings: None,
    },
    Case {
        name: "fast-decompression",
        kind: CodecKind::FastDecompression,
        settings: Some((100, 7, true, 5)),
    },
    Case {
        name: "high-compression",
        kind: CodecKind::HighCompression,
        settings: Some((250, 64, false, 8)),
    },
    Case {
        name: "dummy",
        kind: CodecKind::Dummy,
        settings: Some((100, 10, true, 2)),
    },
    Case {
        name: "deflate-preset",
        kind: CodecKind::DeflateWithPreset,
        settings: Some((5000, 512, false, 10)),
    },
    Case {
        name: "deflate-preset-default",
        kind: CodecKind::DeflateWithPreset,
        settings: None,
    },
    Case {
        name: "lz4-preset",
        kind: CodecKind::Lz4WithPreset,
        settings: Some((90, 3, true, 22)),
    },
    Case {
        name: "fast-small",
        kind: CodecKind::Fast,
        settings: Some((64, 1000, false, 3)),
    },
];

/// Documents per flushed segment.
pub const PER_SEGMENT: usize = 400;
/// Flushed segments.
pub const SEGMENTS: usize = 2;
/// Every this many documents store a large `big` value.
pub const BIG_EVERY: usize = 50;

/// Whether document `i` is deleted after the flushes.
pub fn deleted(i: usize) -> bool {
    i % 17 == 3
}

pub fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

pub fn codec(case: &Case) -> CompressingCodec {
    match case.settings {
        None => case.kind.defaults(),
        Some((chunk, docs, suffix, shift)) => {
            case.kind.configured(chunk, docs, suffix, shift).unwrap()
        }
    }
}

/// `VerifyCompressingCodecs.body`.
pub fn body(i: usize) -> String {
    format!("w{} x{} common t{i}", i % 5, i % 7)
}

/// `VerifyCompressingCodecs.big`.
pub fn big(i: usize) -> Option<String> {
    i.is_multiple_of(BIG_EVERY)
        .then(|| format!("z{i} ").repeat(140))
}

/// `VerifyCompressingCodecs.doc`.
pub fn doc(i: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    let mut ty = FieldType::copy_of(&TextField::type_stored());
    ty.set_store_term_vectors(true).unwrap();
    ty.set_store_term_vector_positions(true).unwrap();
    ty.set_store_term_vector_offsets(true).unwrap();
    ty.freeze();
    d.add(Field::from_string("body", body(i), ty).unwrap());
    d.add(Field::stored("n", StoredValue::Int(i as i32)));
    if let Some(big) = big(i) {
        d.add(Field::stored("big", StoredValue::String(big)));
    }
    d
}

pub fn write(dir: &dyn Directory, case: &Case, merge: bool) {
    let codec = Arc::new(codec(case).installed().unwrap());
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version()).unwrap();
    w.set_codec(Some(codec.clone())).unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    for seg in 0..SEGMENTS {
        for i in seg * PER_SEGMENT..(seg + 1) * PER_SEGMENT {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    let terms: Vec<Term> = (0..SEGMENTS * PER_SEGMENT)
        .filter(|&i| deleted(i))
        .map(|i| Term::new("id", format!("d{i}")))
        .collect();
    w.delete_documents_by_term(&terms).unwrap();
    w.commit().unwrap();
    if merge {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
    let mut out = dir.create_output("codec.txt").unwrap();
    out.write_bytes(codec.to_string().as_bytes());
    out.close().unwrap();
}

pub fn main() {
    let usage = "usage: write_compressing_codec_fixture <output-dir>";
    let out = std::path::PathBuf::from(std::env::args().nth(1).expect(usage));
    for case in &CASES {
        for (sub, merge) in [("flushed", false), ("merged", true)] {
            let path = out.join(case.name).join(sub);
            std::fs::create_dir_all(&path).unwrap();
            write(&lucene_store::FsDirectory::open(&path), case, merge);
        }
    }
}
