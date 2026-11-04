//! Counting a conjunction of terms (`count(+a +b)`, filter-only `#a #b`) on
//! the benchmark corpus: the dense window-AND path against the leapfrog.
//! **Skips itself when `benchmarks/.corpus/merged` is absent.**
//!
//! Run with `cargo bench -p lucene-search --bench conjunction_count`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::MmapDirectory;

fn corpus_dir() -> Option<String> {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../benchmarks/.corpus/merged"
    );
    std::path::Path::new(dir)
        .join("segments_1")
        .exists()
        .then(|| dir.to_string())
}

fn term(t: &str) -> Clause {
    Clause::Term(TermQuery::new("body", t))
}

fn bench(c: &mut Criterion) {
    let Some(dir_path) = corpus_dir() else {
        eprintln!("conjunction_count: benchmarks/.corpus/merged is absent. Skipping.");
        return;
    };
    let dir = MmapDirectory::open(dir_path);
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    // No query cache: the count is computed every time.
    let s0 = &segments[0];
    let seg = &lucene_search::multi_segment::OpenSegment { cache: None, ..*s0 };
    let cases: Vec<(&str, BooleanQuery)> = vec![
        (
            "and_t0_t1",
            BooleanQuery::new().with_must([term("t0"), term("t1")]),
        ),
        (
            "and_t0_t1_t2",
            BooleanQuery::new().with_must([term("t0"), term("t1"), term("t2")]),
        ),
        (
            "filter_t0_t1",
            BooleanQuery::new().with_filter([term("t0"), term("t1")]),
        ),
        (
            "and_t0_tz",
            BooleanQuery::new().with_must([term("t0"), term("tz")]),
        ),
    ];
    let mut g = c.benchmark_group("conjunction_count");
    g.sample_size(20);
    for (name, q) in &cases {
        g.bench_function(*name, |b| {
            b.iter(|| black_box(lucene_search::count_boolean_query_segment(seg, q).unwrap()))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
