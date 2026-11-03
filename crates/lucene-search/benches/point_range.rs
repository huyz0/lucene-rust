//! Point range queries over one 1M-document segment, with the query cache
//! off so every iteration runs the BKD walk:
//!
//! - `p` is a dense single-valued `LongPoint` (every document, 0..1,000,000);
//! - `pm` is a sparse multi-valued `LongPoint` (half the documents, 1-3
//!   values in 0..1,000,000);
//! - `t` is a dense `LongPoint` equal to the document's number, as a
//!   timestamp indexed in arrival order is: each leaf's doc ids are one run;
//! - `b` is `t` on every third document only: each leaf's doc ids are
//!   stored as a bit set.
//!
//! Each shape is counted (`IndexSearcher::count`, which iterates the scorer)
//! at a selectivity below `maxDoc / 128` (a sorted doc list), around 5% (a
//! bit set), and above half the segment (the dense field's inverse walk).
//! `points_range/*` is the `Clause::PointsRange` (i64) path, `point_range/*`
//! the `PointRangeQuery` (`LongPoint.newRangeQuery`) one.
//!
//! Run with `cargo bench -p lucene-search --bench point_range`.

#![allow(clippy::arithmetic_side_effects)]

mod support;

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_index::document::{Document, LongPoint, NumericDocValuesField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::long_point;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, PointsRangeQuery};
use lucene_store::ByteBuffersDirectory;

const DOCS: usize = 1_000_000;

fn build() -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let mut rng = support::Rng::new(0x9e37);
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_ram_buffer_size_mb(4096.0).unwrap();
        for i in 0..DOCS {
            let mut d = Document::new();
            let v = rng.below(1_000_000) as i64;
            d.add(LongPoint::new("p", &[v]).unwrap());
            d.add(NumericDocValuesField::new("p", v));
            d.add(LongPoint::new("t", &[i as i64]).unwrap());
            if i % 3 == 0 {
                d.add(LongPoint::new("b", &[i as i64]).unwrap());
            }
            if rng.below(2) == 0 {
                for _ in 0..1 + rng.below(3) {
                    d.add(LongPoint::new("pm", &[rng.below(1_000_000) as i64]).unwrap());
                }
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
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let mut segs = opened.as_open_segments();
    for s in &mut segs {
        s.cache = None;
    }
    let norms: Vec<_> = segs.iter().map(|_| None).collect();
    let searcher = IndexSearcher::new(&segs, &norms).unwrap();

    let mut g = c.benchmark_group("point_range");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));
    g.sample_size(20);
    for field in ["p", "pm", "t", "b"] {
        for (label, hi) in [("0.5pct", 5_000i64), ("5pct", 50_000), ("70pct", 700_000)] {
            let lo = 1_000i64;
            let q = BooleanQuery::new().with_filter([Clause::PointsRange(PointsRangeQuery::new(
                field,
                lo,
                lo + hi,
            ))]);
            let expected = searcher.count(&q).unwrap();
            g.bench_function(format!("points_range/{field}/{label}"), |b| {
                b.iter(|| {
                    let n = searcher.count(black_box(&q)).unwrap();
                    assert_eq!(n, expected);
                    n
                })
            });
            let q2 = BooleanQuery::new()
                .with_filter([Clause::from(long_point::new_range_query(field, lo, lo + hi).unwrap())]);
            assert_eq!(searcher.count(&q2).unwrap(), expected);
            g.bench_function(format!("point_range/{field}/{label}"), |b| {
                b.iter(|| searcher.count(black_box(&q2)).unwrap())
            });
        }
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
