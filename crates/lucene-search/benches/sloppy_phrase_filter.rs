//! A sloppy phrase as a filter (`IndexSearcher::count`, no scores): whether
//! a document matches needs its first match only, not the sloppy frequency
//! a score sums over every match.
//!
//! 100k documents of 60 words drawn from 30, so the phrase's terms co-occur
//! in nearly every document, often within the slop several times over.
//! Shapes: two terms (`"w1 w2"~3`), three terms (`"w1 w2 w3"~6`), and a
//! repeated term (`"w1 w2 w1"~6`).
//!
//! Run with `cargo bench -p lucene-search --bench sloppy_phrase_filter`.

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_analysis::Analyzer;
use lucene_index::document::{Document, Store, TextField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, PhraseQuery};
use lucene_store::ByteBuffersDirectory;

const DOCS: usize = 100_000;

fn build() -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut word = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        format!("w{}", s % 30)
    };
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_analyzer(Some(Arc::new(Analyzer::standard(None))));
        w.set_ram_buffer_size_mb(4096.0).unwrap();
        for _ in 0..DOCS {
            let body: Vec<String> = (0..60).map(|_| word()).collect();
            let mut doc = Document::new();
            doc.add(TextField::new("body", body.join(" "), Store::No));
            w.add_fields_document(&doc).unwrap();
        }
        w.commit().unwrap();
    }
    dir
}

fn bench(c: &mut Criterion) {
    let dir = build();
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let mut segs = opened.as_open_segments();
    for s in &mut segs {
        s.cache = None;
    }
    let norms: Vec<_> = segs.iter().map(|_| None).collect();
    let searcher = IndexSearcher::new(&segs, &norms).unwrap();
    let mut g = c.benchmark_group("sloppy_phrase_filter");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));
    g.sample_size(10);
    for (label, terms, slop) in [
        ("two", vec!["w1", "w2"], 3),
        ("three", vec!["w1", "w2", "w3"], 6),
        ("repeated", vec!["w1", "w2", "w1"], 6),
    ] {
        let q = BooleanQuery::new().with_filter([Clause::Phrase(
            PhraseQuery::new("body", terms.iter().map(|t| t.as_bytes().to_vec())).with_slop(slop),
        )]);
        g.bench_function(label, |b| b.iter(|| searcher.count(black_box(&q)).unwrap()));
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
