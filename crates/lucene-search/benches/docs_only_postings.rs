//! Measures what `PostingsFlags::DocsOnly` actually buys the unscored paths
//! in `lucene-search` (c8's `PForUtil.skip` addition, wired here by c12).
//!
//! The A/B is in one build rather than across two: the same term's postings
//! are resolved twice, once with `Freqs` (what these paths asked for before)
//! and once with `DocsOnly` (what they ask for now). The difference is exactly
//! the frequency-block decode that the constant-score / `Occur::FILTER` /
//! `TermInSetQuery` / wildcard-family paths were paying for and discarding.
//!
//! c8 measured 1.07-1.32x and explained why it is not more: a `.doc` block is
//! dominated by the doc-delta bit-packing, not the frequency bit-packing. This
//! bench re-takes that number on this crate's own call shape rather than
//! quoting it, over three terms whose posting lists differ by two orders of
//! magnitude.
//!
//! Measured against `benchmarks/.corpus/merged` (`scripts/bench-corpus.sh`);
//! **the bench skips itself when that corpus is absent**.
//!
//! Run with `cargo bench -p lucene-search --bench docs_only_postings`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_codecs::postings::PostingsFlags;
use lucene_search::directory_reader::DirectoryReader;
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
            "docs_only_postings: benchmarks/.corpus/merged is absent -- \
             run scripts/bench-corpus.sh to measure this. Skipping."
        );
        return;
    };

    let dir = MmapDirectory::open(dir_path);
    let reader = DirectoryReader::open(&dir).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    let seg = &segments[0];
    let field_terms = seg.fields.field("body").expect("a body field");

    let mut group = c.benchmark_group("docs_only_postings");

    // Three terms, chosen for posting-list size rather than meaning: the
    // saving is per decoded block, so it should scale with docFreq and be
    // invisible on a short list.
    for term in ["t0", "t1", "t999"] {
        let df = field_terms
            .seek_exact(term.as_bytes())
            .map(|s| s.doc_freq)
            .unwrap_or(0);
        if df == 0 {
            continue;
        }
        group.bench_function(format!("{term}_df{df}/freqs"), |b| {
            b.iter(|| {
                let p = field_terms
                    .postings_with_flags(term.as_bytes(), seg.doc_in, PostingsFlags::Freqs)
                    .expect("postings")
                    .expect("term present");
                black_box(p.docs.len());
            })
        });
        group.bench_function(format!("{term}_df{df}/docs_only"), |b| {
            b.iter(|| {
                let p = field_terms
                    .postings_with_flags(term.as_bytes(), seg.doc_in, PostingsFlags::DocsOnly)
                    .expect("postings")
                    .expect("term present");
                black_box(p.docs.len());
            })
        });
    }

    // Deliberately no whole-query case here. A query-level number would be
    // one-sided -- there is no way to run the *old* clause shape in this build,
    // so it would measure "after" against nothing. The per-term A/B above is
    // the honest measurement of what changed; the query-level effect is a
    // fraction of it, since the postings decode is one part of a larger cost.

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
