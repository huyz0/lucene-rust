//! What a `STRING_VAL` sort costs per collected document.
//!
//! `TermValComparator` reads each candidate's `BINARY` value and compares it
//! with the queue's bottom; until the search perf pass this port copied the
//! value into a fresh `Vec` for every comparison (and again into the slot on
//! a copy), where Java compares the `BytesRef` it was handed. This bench sorts
//! the `sort_val_index` fixture (two segments, 700 documents, a sparse `bv`
//! column of zero to three bytes) by that column, every document a candidate,
//! at a small and a large `topN` -- the small one spends its time in
//! `compareBottom`, the large one in `copy`.
//!
//! Run with `cargo bench -p lucene-search --bench string_val_sort`. Skips
//! itself when the fixture has not been generated.

#[path = "../tests/m7support/mod.rs"]
mod m7support;

use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::top_field::{search_sorted, SortField};
use lucene_store::FsDirectory;
use m7support::{fixture, Grammar};

fn bench(c: &mut Criterion) {
    let dir = fixture("sort_val_index");
    if !std::path::Path::new(&dir).exists() {
        eprintln!("string_val_sort: {dir} is absent -- run scripts/gen-fixtures.sh. Skipping.");
        return;
    }
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open reader");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let readers = reader.segment_readers();
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = vec![None; segments.len()];
    let grammar = Grammar {
        text: "body",
        range: "r",
    };
    let query = grammar.query("(all)");

    let mut group = c.benchmark_group("string_val_sort");
    for reverse in [false, true] {
        let sort = [SortField::string_val("bv", reverse)];
        for top_n in [10usize, 500] {
            let name = format!("top{top_n}{}", if reverse { "_rev" } else { "" });
            group.bench_function(name, |b| {
                b.iter(|| {
                    let hits = search_sorted(
                        &segments,
                        readers,
                        &query,
                        &norms,
                        &sort,
                        top_n,
                        u64::MAX,
                        None,
                    )
                    .unwrap();
                    black_box(hits.hits.len())
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
