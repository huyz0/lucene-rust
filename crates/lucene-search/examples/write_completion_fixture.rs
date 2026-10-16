//! Writes completion indexes for `VerifyCompletion`: `SuggestField`s on a
//! field routed to `Completion104` beside a plain `TextField` on
//! `Lucene104`, as `<out>/flushed` (one segment) and `<out>/merged` (commits
//! after 70 and 100 documents, force-merged), plus `<out>/docs.tsv` (`surface
//! TAB weight`, one document per line), from which the verifier builds
//! Lucene's own index to compare every suggestion with.
//!
//! Usage: `write_completion_fixture <output-dir>`.
// Test-support code opts out of the arithmetic gate at the file boundary:
// see `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use std::path::Path;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_index::document::{Document, Store, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::suggest::suggest_field::SuggestField;
use lucene_store::FsDirectory;

const WORDS: [&str; 14] = [
    "cat",
    "car",
    "card",
    "care",
    "cart",
    "dog",
    "door",
    "dot",
    "tie",
    "many",
    "café",
    "日本",
    "new york",
    "new jersey",
];

/// Document `i`: a surface form and a weight, ties and shared prefixes on
/// purpose.
fn doc(i: usize) -> (&'static str, i32) {
    let x = (i as u64 + 1).wrapping_mul(2_654_435_761);
    (
        WORDS[(x % WORDS.len() as u64) as usize],
        ((x >> 7) % 20) as i32,
    )
}

fn write(dir: &Path, docs: &[(&str, i32)], commits: &[usize]) {
    std::fs::create_dir_all(dir).unwrap();
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    w.set_analyzer(Some(Arc::new(Analyzer::standard(None))));
    w.set_use_compound_file(false);
    w.set_postings_format_for_field("suggest", Lucene104PostingsFormat::completion104());
    for (i, (surface, weight)) in docs.iter().enumerate() {
        let mut d = Document::new();
        d.add_boxed(Box::new(
            SuggestField::new("suggest", *surface, *weight).unwrap(),
        ));
        d.add(TextField::new("body", *surface, Store::No));
        w.add_fields_document(&d).unwrap();
        if commits.contains(&(i + 1)) {
            w.commit().unwrap();
        }
    }
    w.commit().unwrap();
    if !commits.is_empty() {
        w.force_merge(1).unwrap();
        w.commit().unwrap();
    }
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_completion_fixture <dir>");
    let out = Path::new(&out);
    let docs: Vec<(&str, i32)> = (0..120).map(doc).collect();
    write(&out.join("flushed"), &docs, &[]);
    write(&out.join("merged"), &docs, &[70, 100]);
    let tsv: String = docs.iter().map(|(s, w)| format!("{s}\t{w}\n")).collect();
    std::fs::write(out.join("docs.tsv"), tsv).unwrap();
}
