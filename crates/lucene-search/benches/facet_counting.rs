//! Facet counting over a multi-segment index (the facets/grouping/join
//! perf pass):
//!
//! - `ffi/*`: the FFI-facing per-segment primitives in `facets.rs`
//!   ([`facets::facet_counts`] over a sparse multi-valued SORTED_SET,
//!   [`facets::range_facet_counts_with_total`] over a sparse and a dense
//!   NUMERIC with 4 / 32 / 256 ranges, and
//!   [`facets::multi_valued_range_facet_counts`] over a SORTED_NUMERIC),
//!   every segment of an 8-segment index, all documents (or every 10th)
//!   as candidates;
//! - `sorted_set/*`: `SortedSetDocValuesFacetCounts` and
//!   `ConcurrentSortedSetDocValuesFacetCounts` over a match-all collector's
//!   hits, on 1 and 8 segments;
//! - `search_sorted/*`: `FacetsCollectorManager::search_sorted` with a
//!   30%-selective term, with and without doc scores.
//!
//! Run with `cargo bench -p lucene-search --bench facet_counting`.

#![allow(clippy::arithmetic_side_effects)]

mod support;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::doc_values::SortedSetKind;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::facets::{
    self, ConcurrentSortedSetDocValuesFacetCounts, DefaultSortedSetDocValuesReaderState,
    FacetsCollectorManager, NumericRange, SortedSetDocValuesFacetCounts,
};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::top_field::{SortField, SortType};

const DOCS: usize = 400_000;

fn ranges(n: usize) -> Vec<NumericRange> {
    let width = 1_000_000 / n as i64;
    (0..n as i64)
        .map(|i| NumericRange::new_long(format!("r{i}"), i * width, true, (i + 1) * width, false))
        .collect::<Result<_, _>>()
        .unwrap()
}

fn bench_ffi(c: &mut Criterion) {
    let dir = support::build_index(DOCS, 8, 1000);
    let reader = DirectoryReader::open(&dir).unwrap();
    let segs = reader.segment_readers();
    assert_eq!(segs.len(), 8);
    let all: Vec<Vec<i32>> = segs.iter().map(|s| (0..s.max_doc).collect()).collect();
    let tenth: Vec<Vec<i32>> = segs
        .iter()
        .map(|s| (0..s.max_doc).step_by(10).collect())
        .collect();
    let field = |s: &lucene_search::directory_reader::SegmentReader, name: &str| {
        s.field_infos().field_by_name(name).unwrap().number
    };

    let mut g = c.benchmark_group("ffi");
    for (label, cands) in [("all", &all), ("tenth", &tenth)] {
        g.bench_function(format!("facet_counts_sparse_multi/{label}"), |b| {
            b.iter(|| {
                let mut total = 0u64;
                for (s, docs) in segs.iter().zip(cands.iter()) {
                    let n = field(s, "tags");
                    let (meta, data) = s.doc_values_for_field(n).unwrap();
                    let SortedSetKind::Multi { ords, terms } =
                        &meta.sorted_set_entry(n).unwrap().kind
                    else {
                        panic!("tags is multi-valued")
                    };
                    let counts = facets::facet_counts(data, ords, terms, docs).unwrap();
                    total += counts.iter().sum::<u64>();
                }
                black_box(total)
            })
        });
    }
    for field_name in ["num", "dense"] {
        for n_ranges in [4usize, 32, 256] {
            let rs = ranges(n_ranges);
            g.bench_function(format!("range_single_{field_name}/{n_ranges}"), |b| {
                b.iter(|| {
                    let mut total = 0u64;
                    for (s, docs) in segs.iter().zip(all.iter()) {
                        let n = field(s, field_name);
                        let (meta, data) = s.doc_values_for_field(n).unwrap();
                        let entry = meta.numeric_entry(n).unwrap();
                        let r =
                            facets::range_facet_counts_with_total(data, entry, &rs, docs).unwrap();
                        total += r.total_count;
                    }
                    black_box(total)
                })
            });
        }
    }
    for n_ranges in [4usize, 32] {
        let rs = ranges(n_ranges);
        g.bench_function(format!("range_multi_nums/{n_ranges}"), |b| {
            b.iter(|| {
                let mut total = 0u64;
                for (s, docs) in segs.iter().zip(all.iter()) {
                    let n = field(s, "nums");
                    let (meta, data) = s.doc_values_for_field(n).unwrap();
                    let entry = meta.sorted_numeric_entry(n).unwrap();
                    let r =
                        facets::multi_valued_range_facet_counts(data, entry, &rs, docs).unwrap();
                    total += r.total_count;
                }
                black_box(total)
            })
        });
    }
    g.finish();
}

fn match_all(max_doc: i32) -> BooleanQuery {
    BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc))],
        ..Default::default()
    }
}

fn bench_sorted_set(c: &mut Criterion) {
    let config = support::facets_config();
    let mut g = c.benchmark_group("sorted_set");
    for segments in [1usize, 8] {
        let dir = support::build_index(DOCS, segments, 1000);
        let reader = DirectoryReader::open(&dir).unwrap();
        let opened = reader.open_segments().unwrap();
        let segs = opened.as_open_segments();
        let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
        let norms: Vec<_> = owned.iter().map(Some).collect();
        let searcher = IndexSearcher::new(&segs, &norms).unwrap();
        let state = DefaultSortedSetDocValuesReaderState::new(&reader, &config).unwrap();
        let hits = FacetsCollectorManager::new(false)
            .collect(&searcher, &match_all(reader.max_doc()))
            .unwrap();
        g.bench_function(format!("sequential/{segments}seg"), |b| {
            b.iter(|| black_box(SortedSetDocValuesFacetCounts::with_hits(&state, &hits).unwrap()))
        });
        g.bench_function(format!("concurrent/{segments}seg"), |b| {
            b.iter(|| {
                black_box(
                    ConcurrentSortedSetDocValuesFacetCounts::with_hits(&state, &hits).unwrap(),
                )
            })
        });
        g.bench_function(format!("collect_match_all/{segments}seg"), |b| {
            b.iter(|| {
                black_box(
                    FacetsCollectorManager::new(false)
                        .collect(&searcher, &match_all(reader.max_doc()))
                        .unwrap(),
                )
            })
        });
    }
    g.finish();
}

fn bench_search_sorted(c: &mut Criterion) {
    let dir = support::build_index(DOCS, 8, 1000);
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<_> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segs, &norms).unwrap();
    let query = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("body", "t30"))],
        ..Default::default()
    };
    let sort = vec![SortField::numeric("dense", SortType::Long, false)];
    let mut g = c.benchmark_group("search_sorted");
    for scores in [false, true] {
        g.bench_function(format!("doc_scores={scores}"), |b| {
            b.iter(|| {
                black_box(
                    FacetsCollectorManager::search_sorted(
                        &searcher,
                        reader.segment_readers(),
                        &query,
                        10,
                        &sort,
                        scores,
                        &FacetsCollectorManager::new(false),
                    )
                    .unwrap(),
                )
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_ffi, bench_sorted_set, bench_search_sorted);
criterion_main!(benches);
