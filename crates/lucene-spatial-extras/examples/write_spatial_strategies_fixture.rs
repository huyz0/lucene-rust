//! Writes `GenSpatialStrategies`' corpus through this port's spatial-extras
//! strategies and `IndexWriter` -- the same documents
//! (`fixtures/data/spatial_strategies/docs.tsv`), commits and deletes -- for
//! `VerifySpatialExtras` to open with real Lucene 10.5.0: `CheckIndex`, then
//! every question of `queries.tsv` answered over this index and over
//! Lucene's own, which must agree.
//!
//! Usage: `write_spatial_strategies_fixture <output-dir>`.
#![allow(clippy::arithmetic_side_effects)]

#[path = "../tests/common/spatial_corpus.rs"]
mod spatial_corpus;

use lucene_index::buffered_updates::Term;
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_spatial_strategies_fixture <output-dir>");
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/spatial_strategies");
    let read =
        |f: &str| std::fs::read_to_string(fixtures.join(f)).expect("GenSpatialStrategies fixtures");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).expect("create output dir");
    let corpus = spatial_corpus::corpus();
    let fs = FsDirectory::open(std::path::Path::new(&out));
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).expect("open writer");
    for (i, line) in read("docs.tsv").lines().enumerate() {
        w.add_fields_document(&corpus.document(line)).expect("add");
        // `GenSpatialStrategies.COMMITS`
        if matches!(i + 1, 120 | 240 | 360) {
            w.commit().expect("commit");
        }
    }
    for id in read("deletes.tsv").lines() {
        w.delete_documents_by_term(&[Term::new("id", id)])
            .expect("delete");
    }
    w.commit().expect("commit");
}
