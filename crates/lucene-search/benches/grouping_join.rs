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
//! - `grouping_sorted/*`: the same grouping with the groups sorted by a
//!   SORTED string field (`TermOrdValComparator`'s bottom compare on
//!   every document once the queue is full), top 10 and top 10,000 groups
//!   (the first pass's ordered set built once full);
//! - `grouping_block/*`: `GroupingSearch::by_blocks` over 200k documents
//!   in blocks of 5 (40k groups), sorted by a NUMERIC field, top 10 /
//!   1,000 / 10,000 groups (the group queue).
//!
//! Run with `cargo bench -p lucene-search --bench grouping_join`.

#![allow(clippy::arithmetic_side_effects)]

mod support;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_index::document::{Document, NumericDocValuesField, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::grouping::{GroupingSearch, Sort, TermGroupSelector};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::join::{
    create_global_ordinals_join_query, create_join_query, create_numeric_join_query, ordinal_map,
    NumericType, ScoreMode,
};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
use lucene_search::top_field::{SortField, SortType};
use lucene_store::ByteBuffersDirectory;

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

    let mut g = c.benchmark_group("grouping_sorted");
    let by_key = Sort::new(vec![SortField::string("join_key", false)]);
    for top_n in [10usize, 10_000] {
        g.bench_function(format!("string_sort/top{top_n}"), |b| {
            b.iter(|| {
                let gs = GroupingSearch::new(|| TermGroupSelector::new("group"))
                    .set_group_sort(by_key.clone());
                black_box(gs.search(&searcher, &q30, 0, top_n).unwrap())
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

/// 200k documents in blocks of 5, the last of
/// each block marked `end:x`, every document a random NUMERIC `price`; 4
/// segments.
fn build_block_index() -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let mut rng = support::Rng::new(0xb10c);
    {
        let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
        w.set_ram_buffer_size_mb(4096.0).unwrap();
        let blocks = 40_000;
        for b in 0..blocks {
            // One writer thread: the block's documents get adjacent ids, and
            // a flush only ever falls between blocks.
            for i in 0..5 {
                let d = {
                    let mut d = Document::new();
                    d.add(StringField::new("body", "all", Store::No));
                    d.add(NumericDocValuesField::new(
                        "price",
                        rng.below(1_000_000) as i64,
                    ));
                    if i == 4 {
                        d.add(StringField::new("end", "x", Store::No));
                    }
                    d
                };
                w.add_fields_document(&d).unwrap();
            }
            if (b + 1) % (blocks / 4) == 0 {
                w.flush().unwrap();
            }
        }
        w.commit().unwrap();
    }
    dir
}

fn bench_block(c: &mut Criterion) {
    let dir = build_block_index();
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<_> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segs, &norms).unwrap();
    let all = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("body", "all"))],
        ..Default::default()
    };
    let end = BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("end", "x"))],
        ..Default::default()
    };
    let by_price = Sort::new(vec![SortField::numeric("price", SortType::Long, true)]);
    let mut g = c.benchmark_group("grouping_block");
    for top_n in [10usize, 1000, 10_000] {
        g.bench_function(format!("price_desc/top{top_n}"), |b| {
            b.iter(|| {
                let gs = GroupingSearch::by_blocks(end.clone())
                    .set_group_sort(by_price.clone())
                    .set_sort_within_group(Sort::index_order());
                black_box(gs.search_blocks(&searcher, &all, 0, top_n).unwrap())
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench, bench_block);
criterion_main!(benches);
