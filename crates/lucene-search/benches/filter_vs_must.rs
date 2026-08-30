//! Settles the c11 claim that an `Occur.FILTER` clause is **cheaper** than the
//! same clause as `Occur.MUST`.
//!
//! The claim is not self-evident. A filter clause is a leg of the same
//! conjunction, advanced identically, so it saves nothing on the *matching*
//! side; the whole saving is on the scoring side, and only if the executor
//! actually declines to do that work. In this port the saving is concretely:
//!
//! - `ConjunctionScorer.score()` iterates the scoring subset, so
//!   `try_conjunction_lazy` skips a filter leg's `freq()` -- which is a
//!   `PForUtil` frequency-block decode, not a field read;
//! - a filter leg gets no `FieldNormsCursor`, so its field's norms (and, for a
//!   sparse field, its `IndexedDISI` walk) are never touched;
//! - `BM25Scorer.score` is not called for it.
//!
//! Measured against the real benchmark corpus
//! (`benchmarks/.corpus/merged`, built by `scripts/bench-corpus.sh`) rather
//! than a synthetic list, because the saving is dominated by postings decode
//! and a synthetic `Vec<i32>` has none. **The bench skips itself when that
//! corpus is absent**, since it is gigabytes and deliberately not checked in.
//!
//! Shapes:
//! - `and_t0_t1` / `filter_t1` -- `+body:t0 +body:t1` against
//!   `+body:t0 #body:t1`: identical matched set, one fewer scored clause.
//! - `and_t0_tz` / `filter_tz` -- the same with a selective second clause,
//!   where the conjunction is small and the per-document saving is a smaller
//!   share of the total.
//! - `filter_only_t0_t1` -- `#body:t0 #body:t1`: no scoring clause at all.
//!
//! The last one is not comparable to `and_t0_t1` under a top-`n` collector, and
//! the bench says so by measuring both twice. A scoring conjunction *prunes*
//! (its block-max bound beats the queue's bottom score and whole blocks are
//! skipped); a filter-only conjunction has no score to bound and this port
//! deliberately does not let a zero bound authorize a skip, so it visits the
//! whole intersection. `*_exhaustive` re-runs both with a collector whose
//! `ScoreMode` forbids pruning on either side, which is the like-for-like
//! comparison of the *matching* work.
//!
//! Run with `cargo bench -p lucene-search --bench filter_vs_must`.

use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::norms;
use lucene_search::collector::TopDocsCollector;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_store::MmapDirectory;

const TOP_N: usize = 50;

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
        eprintln!(
            "filter_vs_must: benchmarks/.corpus/merged is absent -- \
             run scripts/bench-corpus.sh to measure this. Skipping."
        );
        return;
    };

    let dir = MmapDirectory::open(dir_path.clone());
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let seg = &segments[0];

    // Norms, read directly like the bench runner does -- without them BM25
    // loses length normalization, which is the very work a filter clause skips.
    let base = std::path::Path::new(&dir_path);
    let name = &reader.segment_readers()[0].segment_name;
    let commit_id = reader.segment_infos.segments[0].segment_id;
    let meta = std::fs::read(base.join(format!("{name}.nvm"))).expect("read .nvm");
    let data = std::fs::read(base.join(format!("{name}.nvd"))).expect("read .nvd");
    let data: &'static [u8] = Box::leak(data.into_boxed_slice());
    let (_, parsed) = norms::parse_meta(&meta, &commit_id, "").expect("parse .nvm");
    let mut norms_map: HashMap<String, FieldNorms<'static>> = HashMap::new();
    for fi in &reader.segment_readers()[0].field_infos().fields {
        if let (Some(entry), Some(f)) = (parsed.entry(fi.number), seg.fields.field(&fi.name)) {
            norms_map.insert(
                fi.name.clone(),
                FieldNorms::from_field_stats(data, *entry, f.sum_total_term_freq, f.doc_count),
            );
        }
    }

    let cases: Vec<(&str, BooleanQuery)> = vec![
        (
            "and_t0_t1",
            BooleanQuery::new().with_must([term("t0"), term("t1")]),
        ),
        (
            "filter_t1",
            BooleanQuery::new()
                .with_must([term("t0")])
                .with_filter([term("t1")]),
        ),
        (
            "and_t0_tz",
            BooleanQuery::new().with_must([term("t0"), term("tz")]),
        ),
        (
            "filter_tz",
            BooleanQuery::new()
                .with_must([term("t0")])
                .with_filter([term("tz")]),
        ),
        (
            "filter_only_t0_t1",
            BooleanQuery::new().with_filter([term("t0"), term("t1")]),
        ),
    ];

    // `total_hits_threshold == u64::MAX` is `ScoreMode::COMPLETE`: an exact
    // total-hit count, and therefore no pruning at all.
    let exhaustive: Vec<(&str, BooleanQuery)> = vec![
        (
            "and_t0_t1_exhaustive",
            BooleanQuery::new().with_must([term("t0"), term("t1")]),
        ),
        (
            "filter_only_t0_t1_exhaustive",
            BooleanQuery::new().with_filter([term("t0"), term("t1")]),
        ),
    ];

    let mut group = c.benchmark_group("filter_vs_must");
    for (name, query) in cases.iter().chain(exhaustive.iter()) {
        let prunes = !name.ends_with("_exhaustive");
        group.bench_function(*name, |b| {
            b.iter(|| {
                let mut top = if prunes {
                    TopDocsCollector::new(TOP_N)
                } else {
                    TopDocsCollector::with_total_hits_threshold(TOP_N, u64::MAX)
                };
                lucene_search::search_boolean_query_scored(
                    seg.fields,
                    seg.doc_in,
                    seg.pos_in,
                    seg.pay_in,
                    seg.live_docs,
                    None,
                    black_box(query),
                    Some(&norms_map),
                    &mut top,
                )
                .expect("search");
                black_box(top.top_docs().len())
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
