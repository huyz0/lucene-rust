//! Doc-values skip indexes (`DocValuesSkipper`) on one 1M-document segment,
//! each field indexed twice: with a skip index (`*_skip`,
//! `NumericDocValuesField.indexedField`) and without (`*_plain`):
//!
//! - `ts` rises with the document number, as a timestamp indexed in arrival
//!   order does: the skipper's blocks have narrow, disjoint ranges;
//! - `rnd` is uniform in 0..1,000,000: every block spans almost the range.
//!
//! `range/*` counts a `NumericDocValuesRangeQuery` at 0.1%, 1% and 10% of
//! the range, alone and in a conjunction with a 9% term (`and_term`);
//! `sort/*` is the top 10 of every document sorted by the field, ascending
//! and descending (a skipper lets the competitive iterator skip blocks no
//! better than the current bottom).
//!
//! Run with `cargo bench -p lucene-search --bench skipper`.

#![allow(clippy::arithmetic_side_effects)]

mod support;

use std::collections::HashMap;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_index::document::{Document, NumericDocValuesField, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::NumericDocValuesRangeQuery;
use lucene_search::field_norms::FieldNorms;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::top_field::{search_sorted, SortField, SortType};
use lucene_store::ByteBuffersDirectory;

const DOCS: usize = 1_000_000;

fn build() -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let mut rng = support::Rng::new(0x5eed);
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_ram_buffer_size_mb(4096.0).unwrap();
        for i in 0..DOCS {
            let mut d = Document::new();
            let ts = i as i64 * 3 + rng.below(3) as i64;
            let rnd = rng.below(1_000_000) as i64;
            d.add(NumericDocValuesField::indexed_field("ts_skip", ts));
            d.add(NumericDocValuesField::new("ts_plain", ts));
            d.add(NumericDocValuesField::indexed_field("rnd_skip", rnd));
            d.add(NumericDocValuesField::new("rnd_plain", rnd));
            if i % 100 < 9 {
                d.add(StringField::new("lead", "mid", Store::No));
            }
            w.add_fields_document(&d).unwrap();
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
    assert_eq!(segs.len(), 1);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = segs.iter().map(|_| None).collect();
    let searcher = IndexSearcher::new(&segs, &norms).unwrap();

    let mut g = c.benchmark_group("skipper");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));
    g.sample_size(20);
    for (column, span) in [("ts", 3 * DOCS as i64), ("rnd", 1_000_000i64)] {
        for (label, width) in [
            ("0.1pct", span / 1000),
            ("1pct", span / 100),
            ("10pct", span / 10),
        ] {
            let lo = span / 3;
            let mut counts = Vec::new();
            for kind in ["skip", "plain"] {
                let field = format!("{column}_{kind}");
                let range =
                    || Clause::from(NumericDocValuesRangeQuery::new(&field, lo, lo + width));
                let alone = BooleanQuery::new().with_filter([range()]);
                let with_term = BooleanQuery::new()
                    .with_filter([Clause::from(TermQuery::new("lead", "mid")), range()]);
                counts.push((
                    searcher.count(&alone).unwrap(),
                    searcher.count(&with_term).unwrap(),
                ));
                g.bench_function(format!("range/{field}/{label}"), |b| {
                    b.iter(|| searcher.count(black_box(&alone)).unwrap())
                });
                g.bench_function(format!("range/{field}/{label}/and_term"), |b| {
                    b.iter(|| searcher.count(black_box(&with_term)).unwrap())
                });
            }
            assert_eq!(counts[0], counts[1], "{column} {label}");
        }
        for reverse in [false, true] {
            let mut tops = Vec::new();
            for kind in ["skip", "plain"] {
                let field = format!("{column}_{kind}");
                let sort = [SortField::numeric(&field, SortType::Long, reverse)];
                let all = BooleanQuery {
                    must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(DOCS as i32))],
                    ..Default::default()
                };
                let run = || {
                    search_sorted(
                        &segs,
                        reader.segment_readers(),
                        &all,
                        &norms,
                        &sort,
                        10,
                        1000,
                        None,
                    )
                    .unwrap()
                };
                tops.push(run().hits.iter().map(|h| h.doc).collect::<Vec<_>>());
                let dir = if reverse { "desc" } else { "asc" };
                g.bench_function(format!("sort/{field}/{dir}"), |b| {
                    b.iter(|| black_box(run().hits.len()))
                });
            }
            assert_eq!(tops[0], tops[1], "{column} sort");
        }
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
