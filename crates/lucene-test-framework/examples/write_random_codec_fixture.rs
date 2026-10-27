//! Writes indexes through `IndexWriter` with the test framework's
//! `RandomCodec` installed (`IndexWriter::set_codec`), for
//! `VerifyRandomCodec`: for each of [`SEEDS`], `new RandomCodec(new
//! Random(seed))` routes the postings of `id`, `body`, `kind` and `code` and the doc
//! values of `n`, `s` and `b` to its shuffled candidates, and writes the
//! three-dimensional point `p` with its leaf size and random splits.
//!
//! Two flushed segments of [`PER_SEGMENT`] documents, then every
//! [`DELETED`] document deleted, and, when `merge` is asked for, a
//! force-merge to one. Every segment records `Asserting`; `codec.txt`
//! holds the codec's `toString()` after the writes.
//!
//! Usage: `write_random_codec_fixture <output-dir> [<seed> [merge]]`; with
//! no seed, every one of [`SEEDS`] into `<output-dir>/<seed>/flushed` and
//! `<output-dir>/<seed>/merged`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]
#![allow(dead_code)]

use std::sync::Arc;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    BinaryDocValuesField, Document, IntPoint, NumericDocValuesField, SortedDocValuesField, Store,
    StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::{DataOutput, Directory};
use lucene_test_framework::index::random_codec::RandomCodec;
use lucene_util::java_random::JavaRandom;

/// The `new Random(seed)`s each index's codec is made from.
pub const SEEDS: [i64; 6] = [3, 5, 14, 20, 56, 58];

/// Documents per flushed segment.
pub const PER_SEGMENT: usize = 1500;
/// Flushed segments.
pub const SEGMENTS: usize = 2;

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

/// `new RandomCodec(new Random(seed))`, outside a nightly run.
pub fn codec(seed: i64) -> RandomCodec {
    RandomCodec::new(&mut JavaRandom::new(seed), false, 1)
}

/// `VerifyRandomCodec.point`.
pub fn point(i: usize) -> [i32; 3] {
    [(i % 97) as i32, ((i * 7) % 101) as i32 - 50, (i / 3) as i32]
}

/// `VerifyRandomCodec.doc`.
pub fn doc(i: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    d.add(TextField::new(
        "body",
        format!("w{} x{} common", i % 5, i % 7),
        Store::No,
    ));
    d.add(StringField::new("kind", format!("k{}", i % 11), Store::No));
    d.add(StringField::new("code", format!("c{}", i % 3), Store::No));
    d.add(NumericDocValuesField::new("n", i as i64));
    d.add(SortedDocValuesField::new("s", format!("s{}", i % 13)));
    d.add(BinaryDocValuesField::new("b", format!("b{i}")));
    d.add(IntPoint::new("p", &point(i)).unwrap());
    d
}

pub fn write(dir: &dyn Directory, seed: i64, merge: bool) {
    let codec = Arc::new(codec(seed).installed().unwrap());
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
    // `toString()`, for `VerifyRandomCodec` to compare with Lucene's.
    let mut out = dir.create_output("codec.txt").unwrap();
    out.write_bytes(codec.to_string().as_bytes());
    out.close().unwrap();
}

pub fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: write_random_codec_fixture <output-dir> [<seed> [merge]]";
    let out = std::path::PathBuf::from(args.next().expect(usage));
    let runs: Vec<(std::path::PathBuf, i64, bool)> = match args.next() {
        Some(seed) => vec![(
            out,
            seed.parse().expect(usage),
            args.next().as_deref() == Some("merge"),
        )],
        None => SEEDS
            .iter()
            .flat_map(|&seed| {
                [("flushed", false), ("merged", true)]
                    .map(|(sub, merge)| (out.join(seed.to_string()).join(sub), seed, merge))
            })
            .collect(),
    };
    for (path, seed, merge) in runs {
        std::fs::create_dir_all(&path).unwrap();
        write(&lucene_store::FsDirectory::open(&path), seed, merge);
    }
}
