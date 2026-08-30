//! What `Weight.count(LeafReaderContext)` is actually worth.
//!
//! Java's `TotalHitCountCollector.getLeafCollector` asks `Weight.count(context)`
//! before it opens anything and throws `CollectionTerminatedException` when the
//! answer is not `-1`; `TermQuery.TermWeight.count` answers from
//! `termsEnum.docFreq()` whenever the leaf has no deletions. So
//! `IndexSearcher.count(new TermQuery(...))` on a deletion-free segment is a
//! terms-dictionary seek and nothing else -- no `.doc` file opened, no postings
//! block decoded, no per-document loop.
//!
//! Until c37 this port had no such shortcut: a caller wanting a count ran
//! `search_term_query` into a `CountCollector` and paid for the whole postings
//! walk. This bench is the two sides of that, on the real benchmark corpus
//! (`benchmarks/.corpus/merged`, ~5M documents, one segment) rather than on a
//! synthetic list, because what the shortcut skips is postings decode and a
//! synthetic list has none. **The bench skips itself when that corpus is
//! absent.**
//!
//! Shapes:
//! - `t0` -- a term in a large share of the corpus, where the walk is long.
//! - `tz` -- a selective term, where it is short. Included because the
//!   shortcut's advantage is not a constant: it is proportional to `docFreq`,
//!   and a bench that only measured the dense case would overstate it.
//!
//! Run with `cargo bench -p lucene-search --bench term_count`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::query::TermQuery;
use lucene_search::weight_count::count_term_query;
use lucene_search::CountCollector;
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

fn bench(c: &mut Criterion) {
    let Some(dir_path) = corpus_dir() else {
        eprintln!(
            "term_count: benchmarks/.corpus/merged is absent -- \
             run scripts/bench-corpus.sh to measure this. Skipping."
        );
        return;
    };

    let dir = MmapDirectory::open(dir_path);
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let seg = &segments[0];

    for term in ["t0", "tz"] {
        let query = TermQuery::new("body", term);

        // Sanity: the two must agree, or the comparison below is between two
        // different questions. Also prints the docFreq the shortcut returns, so
        // a reader can see how much walking the other side is doing.
        let shortcut = count_term_query(seg.fields, None, None, &query).expect("count");
        let mut counter = CountCollector::default();
        lucene_search::search_term_query(seg.fields, seg.doc_in, None, &query, &mut counter)
            .expect("collect");
        assert_eq!(
            shortcut,
            i64::from(counter.count),
            "the shortcut and the walk must agree on `body:{term}`"
        );
        eprintln!("term_count: body:{term} matches {shortcut} documents");

        let mut group = c.benchmark_group(format!("count_{term}"));
        group.bench_function("weight_count", |b| {
            b.iter(|| black_box(count_term_query(seg.fields, None, None, &query).unwrap()))
        });
        group.bench_function("collect_every_doc", |b| {
            b.iter(|| {
                let mut counter = CountCollector::default();
                lucene_search::search_term_query(
                    seg.fields,
                    seg.doc_in,
                    None,
                    &query,
                    &mut counter,
                )
                .unwrap();
                black_box(counter.count)
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
