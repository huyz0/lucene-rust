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
//! the `PointRangeQuery` (`LongPoint.newRangeQuery`) one. `walk/*` is the
//! BKD walk alone into a doc-id list, `sort/*` that list's sort and dedup.
//!
//! `index_sort_range/*` is `IndexSortSortedNumericDocValuesRangeQuery` over
//! a 500k-document segment sorted by its field, counted alone and in a
//! conjunction with a doc-values range.
//!
//! Run with `cargo bench -p lucene-search --bench point_range`.

#![allow(clippy::arithmetic_side_effects)]

mod support;

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_index::document::{Document, LongPoint, NumericDocValuesField, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::long_point;
use lucene_search::extended_query::{
    IndexSortSortedNumericDocValuesRangeQuery, NumericDocValuesRangeQuery,
};
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
            if i % 100 < 9 {
                d.add(StringField::new("lead", "mid", Store::No));
            }
            if i % 10_000 == 0 {
                d.add(StringField::new("lead", "few", Store::No));
            }
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
        for (label, hi) in [
            ("one", 0i64),
            ("0.5pct", 5_000),
            ("5pct", 50_000),
            ("70pct", 700_000),
        ] {
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
            let q2 = BooleanQuery::new().with_filter([Clause::from(
                long_point::new_range_query(field, lo, lo + hi).unwrap(),
            )]);
            assert_eq!(searcher.count(&q2).unwrap(), expected);
            g.bench_function(format!("point_range/{field}/{label}"), |b| {
                b.iter(|| searcher.count(black_box(&q2)).unwrap())
            });
            // A fresh query cache per iteration, and the query run twice:
            // `UsageTrackingQueryCachingPolicy` caches a point range on its
            // second use, so the second run builds the cached set.
            g.bench_function(format!("cache_fill/{field}/{label}"), |b| {
                b.iter(|| {
                    let cache = lucene_search::SegmentQueryCache::default();
                    let mut cached = opened.as_open_segments();
                    for s in &mut cached {
                        s.cache = Some(&cache);
                    }
                    let searcher = IndexSearcher::new(&cached, &norms).unwrap();
                    searcher.count(black_box(&q2)).unwrap() + searcher.count(&q2).unwrap()
                })
            });
            let points = segs[0].points.unwrap();
            let number = points.field_number(field).unwrap();
            let (pl, ph) = (
                lucene_search::points_query::pack_i64(lo),
                lucene_search::points_query::pack_i64(lo + hi),
            );
            g.bench_function(format!("walk/{field}/{label}"), |b| {
                b.iter(|| points.reader.range_query(number, &pl, &ph).unwrap().len())
            });
            let ids = points.reader.range_query(number, &pl, &ph).unwrap();
            g.bench_function(format!("sort/{field}/{label}"), |b| {
                b.iter(|| {
                    let mut d = ids.clone();
                    lucene_util::doc_id_sort::sort_dedup_doc_ids(&mut d);
                    d.len()
                })
            });
        }
    }
    g.finish();

    // An `IndexOrDocValuesQuery` filter (80% of `p`) behind a lead term,
    // with a fresh query cache, run five times: its fifth use fills the
    // cache. `mid` (9% of the documents) leads it to its doc-values side
    // uncached; `few` (100 documents) costs it more than `skipCacheFactor`
    // times the lead, so it is never cached.
    let mut g = c.benchmark_group("iodv_cache");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(3));
    g.sample_size(10);
    for lead in ["mid", "few"] {
        let mut q = BooleanQuery::new();
        q.must
            .push(Clause::Term(lucene_search::query::TermQuery::new(
                "lead",
                lead.as_bytes().to_vec(),
            )));
        q.filter.push(Clause::from(
            lucene_search::extended_query::IndexOrDocValuesQuery::new(
                Clause::from(long_point::new_range_query("p", 0, 799_999).unwrap()),
                Clause::from(NumericDocValuesRangeQuery::new("p", 0, 799_999)),
            ),
        ));
        g.bench_function(format!("five_runs/{lead}"), |b| {
            b.iter(|| {
                let cache = lucene_search::SegmentQueryCache::default();
                let mut cached = opened.as_open_segments();
                for s in &mut cached {
                    s.cache = Some(&cache);
                }
                let searcher = IndexSearcher::new(&cached, &norms).unwrap();
                (0..5)
                    .map(|_| searcher.count(black_box(&q)).unwrap())
                    .sum::<u64>()
            })
        });
    }
    g.finish();

    // `IndexSortSortedNumericDocValuesRangeQuery` over a segment sorted by
    // its field: the matches are one run of documents.
    let sorted = build_sorted();
    let reader = DirectoryReader::open(&sorted).unwrap();
    let opened = reader.open_segments().unwrap();
    let mut segs = opened.as_open_segments();
    for s in &mut segs {
        s.cache = None;
    }
    let norms: Vec<_> = segs.iter().map(|_| None).collect();
    let searcher = IndexSearcher::new(&segs, &norms).unwrap();
    let mut g = c.benchmark_group("index_sort_range");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));
    g.sample_size(20);
    for (label, hi) in [
        ("5pct", SORTED_DOCS as i64 / 20),
        ("50pct", SORTED_DOCS as i64 / 2),
    ] {
        let fallback = NumericDocValuesRangeQuery::new("v", 1_000, 1_000 + hi);
        let q = BooleanQuery::new().with_filter([Clause::from(
            IndexSortSortedNumericDocValuesRangeQuery::new("v", 1_000, 1_000 + hi, fallback),
        )]);
        // A conjunction iterates the range rather than counting it whole.
        let q_and = BooleanQuery::new().with_filter([
            Clause::from(IndexSortSortedNumericDocValuesRangeQuery::new(
                "v",
                1_000,
                1_000 + hi,
                NumericDocValuesRangeQuery::new("v", 1_000, 1_000 + hi),
            )),
            Clause::from(NumericDocValuesRangeQuery::new("w", 0, 499)),
        ]);
        g.bench_function(format!("count/{label}"), |b| {
            b.iter(|| searcher.count(black_box(&q)).unwrap())
        });
        g.bench_function(format!("and_count/{label}"), |b| {
            b.iter(|| searcher.count(black_box(&q_and)).unwrap())
        });
    }
    g.finish();
}

const SORTED_DOCS: usize = 500_000;

/// One segment of [`SORTED_DOCS`] documents sorted by `v` (the document's
/// number), with `w` a random value in 0..1,000.
fn build_sorted() -> ByteBuffersDirectory {
    use lucene_index::segment_info::{IndexSortField, IndexSortKind, NumericSortKey};
    let dir = ByteBuffersDirectory::new();
    let mut rng = support::Rng::new(0x51);
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_ram_buffer_size_mb(4096.0).unwrap();
        w.enable_explicit_documents().unwrap();
        w.set_index_sort(Some(&[IndexSortField {
            field: "v".to_string(),
            reverse: false,
            kind: IndexSortKind::Numeric(NumericSortKey::Long(None)),
        }]))
        .unwrap();
        for i in 0..SORTED_DOCS {
            let mut d = Document::new();
            d.add(NumericDocValuesField::new("v", i as i64));
            d.add(NumericDocValuesField::new("w", rng.below(1_000) as i64));
            w.add_fields_document(&d).unwrap();
        }
        w.commit().unwrap();
    }
    dir
}

criterion_group!(benches, bench);
criterion_main!(benches);
