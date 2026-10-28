//! `DirectoryReader::open` of one compound, multi-format segment: a text
//! field plus a completion field (`completion104`), so postings come in two
//! groups and every file is a `.cfs` member. Measures what opening pays for
//! the archive and for the per-field dictionary, which used to be copies of
//! the `.cfs` and of every group's `.tim`/`.tip`.
//!
//! Self-contained: the index is written into a scratch directory first.
//! Run with `cargo bench -p lucene-search --bench segment_open_compound`.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_analysis::Analyzer;
use lucene_codecs::per_field_postings::Lucene104PostingsFormat;
use lucene_index::document::{Document, Store, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::suggest::suggest_field::SuggestField;
use lucene_store::{FsDirectory, MmapDirectory};
use lucene_util::test_support::TempDir;

const DOCS: usize = 60_000;

fn write(dir: &std::path::Path) {
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    w.set_analyzer(Some(Arc::new(Analyzer::standard(None))));
    w.set_use_compound_file(true);
    w.set_postings_format_for_field("suggest", Lucene104PostingsFormat::completion104());
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut word = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        format!("w{:x}", s % 200_000)
    };
    for i in 0..DOCS {
        let body: Vec<String> = (0..12).map(|_| word()).collect();
        let mut doc = Document::new();
        doc.add(TextField::new("body", body.join(" "), Store::No));
        doc.add_boxed(Box::new(
            SuggestField::new("suggest", body[0].as_str(), i as i32).unwrap(),
        ));
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
    w.force_merge(1).unwrap();
    w.commit().unwrap();
}

fn bench(c: &mut Criterion) {
    let tmp = TempDir::new("segment_open_compound");
    write(tmp.path());
    let dir = MmapDirectory::open(tmp.path());
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.segment_readers().len(), 1);
    c.bench_function("segment_open_compound/two_postings_formats", |b| {
        b.iter(|| black_box(DirectoryReader::open(&dir).unwrap()))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
