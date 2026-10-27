//! Writes an index through `IndexWriter` with one of the test framework's
//! `Lucene104` filter codecs installed (`IndexWriter::set_codec`), for
//! `VerifyFilterCodecs`: `CheapBastard` (its direct
//! `Lucene104PostingsFormat(100, 200)`), `ConfigurableMCodec` (its direct
//! `Lucene99HnswVectorsFormat(128, 100)`) or `Asserting` (its `Asserting`
//! postings, doc-values and vectors formats over the defaults, and asserting
//! writers for everything else).
//!
//! Three flushed segments of [`PER_SEGMENT`] documents, `d5` and `d150`
//! deleted, then a force-merge to one when `merge` is asked for. Every
//! segment records the codec's name; a direct format's files have no codec
//! suffix (`_0.tim`, `_0.vec`), `Asserting`'s carry its name
//! (`_0_Asserting_0.tim`).
//!
//! Usage: `write_filter_codecs_fixture <output-dir> [<codec> [merge]]`; with
//! no codec, every one of [`CODECS`] into `<output-dir>/<codec>/flushed` and
//! `<output-dir>/<codec>/merged`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]
#![allow(dead_code)]

use std::sync::Arc;

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::field_infos::VectorSimilarityFunction;
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, KnnFloatVectorField, NumericDocValuesField, Store, StringField, TextField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::Directory;
use lucene_test_framework::codecs::asserting_installed::InstalledAssertingCodec;
use lucene_test_framework::codecs::cheap_bastard::CheapBastardCodec;
use lucene_test_framework::codecs::configurable_m::ConfigurableMCodec;

/// The installable filter codecs, by name.
pub const CODECS: [&str; 3] = ["CheapBastard", "ConfigurableMCodec", "Asserting"];

/// Documents per flushed segment.
pub const PER_SEGMENT: usize = 100;
/// Flushed segments.
pub const SEGMENTS: usize = 3;
/// The ids deleted after the flushes.
pub const DELETED: [usize; 2] = [5, 150];

pub fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

/// The installable codec named `name`.
pub fn codec(name: &str) -> Arc<dyn Lucene104Codec> {
    match name {
        "CheapBastard" => Arc::new(CheapBastardCodec::new()),
        "ConfigurableMCodec" => Arc::new(ConfigurableMCodec::new()),
        "Asserting" => Arc::new(InstalledAssertingCodec::new()),
        other => panic!("no installable filter codec {other:?}"),
    }
}

/// `VerifyFilterCodecs.vector`.
pub fn vector(i: usize) -> Vec<f32> {
    vec![(i % 10) as f32, (i % 3) as f32, 1.0, (i % 7) as f32 / 7.0]
}

/// `VerifyFilterCodecs.doc`.
pub fn doc(i: usize) -> Document {
    let mut d = Document::new();
    d.add(StringField::new("id", format!("d{i}"), Store::Yes));
    d.add(TextField::new(
        "body",
        format!("w{} x{} common", i % 5, i % 7),
        Store::No,
    ));
    d.add(KnnFloatVectorField::new("v", vector(i), VectorSimilarityFunction::Euclidean).unwrap());
    d.add(NumericDocValuesField::new("n", i as i64));
    d
}

/// Whether document `i` is still live after the writes.
pub fn live(i: usize) -> bool {
    !DELETED.contains(&i)
}

pub fn write(dir: &dyn Directory, codec_name: &str, merge: bool) {
    lucene_test_framework::codecs::register_codecs();
    let mut w = IndexWriter::open(dir, Vec::new(), "Lucene104", version()).unwrap();
    w.set_codec(Some(codec(codec_name))).unwrap();
    w.set_max_full_flush_merge_wait_millis(0);
    for seg in 0..SEGMENTS {
        for i in seg * PER_SEGMENT..(seg + 1) * PER_SEGMENT {
            w.add_fields_document(&doc(i)).unwrap();
        }
        w.commit().unwrap();
    }
    let terms: Vec<Term> = DELETED
        .iter()
        .map(|i| Term::new("id", format!("d{i}")))
        .collect();
    w.delete_documents_by_term(&terms).unwrap();
    w.commit().unwrap();
    if merge {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
}

pub fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: write_filter_codecs_fixture <output-dir> [<codec> [merge]]";
    let out = std::path::PathBuf::from(args.next().expect(usage));
    let runs: Vec<(std::path::PathBuf, String, bool)> = match args.next() {
        Some(codec) => vec![(out, codec, args.next().as_deref() == Some("merge"))],
        None => CODECS
            .iter()
            .flat_map(|codec| {
                [("flushed", false), ("merged", true)]
                    .map(|(sub, merge)| (out.join(codec).join(sub), codec.to_string(), merge))
            })
            .collect(),
    };
    for (path, codec, merge) in runs {
        std::fs::create_dir_all(&path).unwrap();
        write(&lucene_store::FsDirectory::open(&path), &codec, merge);
    }
}
