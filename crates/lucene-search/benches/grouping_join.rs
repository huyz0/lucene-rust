//! Grouping and query-time joins over a multi-segment index (the
//! facets/grouping/join perf pass):
//!
//! - `grouping/*`: `GroupingSearch` by a `TermGroupSelector` over a
//!   SORTED field of 24-byte terms (10k groups), top 10 groups, with the
//!   all-groups and all-group-heads collectors on and off;
//! - `join/*`: `create_join_query` (terms, `ScoreMode::None` and `Avg`) and
//!   the resulting query searched, `create_numeric_join_query` (`Avg`, the
//!   from side only), and `create_global_ordinals_join_query` searched, with
//!   a 30%-selective from query.
//!
//! Run with `cargo bench -p lucene-search --bench grouping_join`.

#![allow(clippy::arithmetic_side_effects)]

mod support;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::grouping::{GroupingSearch, TermGroupSelector};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::join::{
    create_global_ordinals_join_query, create_join_query, create_numeric_join_query, ordinal_map,
    NumericType, ScoreMode,
};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};

const DOCS: usize = 400_000;

fn bench(c: &mut Criterion) {
    let dir = support::build_index(DOCS, 8, 10_000);
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    let owned = reader
        .field_norms_by_field(&["body".to_string(), "join_to".to_string()])
        .unwrap();
    let norms: Vec<_> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segs, &norms).unwrap();
    let t30 = Clause::Term(TermQuery::new("body", "t30"));
    let q30 = BooleanQuery {
        must: vec![t30.clone()],
        ..Default::default()
    };

    let mut g = c.benchmark_group("grouping");
    for (label, all, heads) in [
        ("top10", false, false),
        ("top10_all_groups", true, false),
        ("top10_all_group_heads", false, true),
        ("top10_all_groups_heads", true, true),
    ] {
        g.bench_function(label, |b| {
            b.iter(|| {
                let gs = GroupingSearch::new(|| TermGroupSelector::new("group"))
                    .set_all_groups(all)
                    .set_all_group_heads(heads);
                black_box(gs.search(&searcher, &q30, 0, 10).unwrap())
            })
        });
    }
    g.finish();

    let mut g = c.benchmark_group("join");
    for (label, mode) in [("none", ScoreMode::None), ("avg", ScoreMode::Avg)] {
        g.bench_function(format!("terms_create/{label}"), |b| {
            b.iter(|| {
                black_box(create_join_query(
                    "join_key", false, "join_to", &t30, &searcher, mode,
                ))
                .unwrap()
            })
        });
        g.bench_function(format!("terms_create_and_search/{label}"), |b| {
            b.iter(|| {
                let q =
                    create_join_query("join_keys", true, "join_to", &t30, &searcher, mode).unwrap();
                let q = BooleanQuery {
                    must: vec![q],
                    ..Default::default()
                };
                black_box(searcher.search(&q, 10).unwrap())
            })
        });
    }
    g.bench_function("numeric_create/avg", |b| {
        b.iter(|| {
            black_box(
                create_numeric_join_query(
                    "join_num",
                    false,
                    "join_pt",
                    NumericType::Long,
                    &t30,
                    &searcher,
                    ScoreMode::Avg,
                )
                .unwrap(),
            )
        })
    });
    let map = ordinal_map(&searcher, "join_key").unwrap();
    let all = Clause::MatchAllDocs(MatchAllDocsQuery::new(reader.max_doc()));
    for (label, mode) in [("none", ScoreMode::None), ("max", ScoreMode::Max)] {
        g.bench_function(format!("global_ordinals_search/{label}"), |b| {
            b.iter(|| {
                let q = create_global_ordinals_join_query(
                    "join_key",
                    &t30,
                    &all,
                    &searcher,
                    mode,
                    Some(map.clone()),
                    0,
                    i32::MAX,
                )
                .unwrap();
                let q = BooleanQuery {
                    must: vec![q],
                    ..Default::default()
                };
                black_box(searcher.search(&q, 10).unwrap())
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
