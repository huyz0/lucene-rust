//! `MUST_NOT` bulk scoring (`ReqExclBulkScorer`) and the doc-values
//! rewrite's ordinal set, over one synthetic 500,000-document segment --
//! the cases Lucene 10.6's #16307, #16669 and #16297 change:
//!
//! - `dv_skip_excl/*`: `+body:all -ts:[0, 449]`, `ts` a skip-indexed
//!   NUMERIC field ascending with the document id, so the excluded range is
//!   whole `YES` skip blocks (#16307: a two-phase excluded clause skips its
//!   confirmed run at once);
//! - `terms_excl_dense/*`, `terms_excl_sparse/*`: `+body:all` minus a
//!   seven-term disjunction holding all but one document in 128 (dense) or
//!   one document in 128 (sparse) -- `MatchAllDocsQuery`-like required side,
//!   `MustNotBooleanQueryBenchmark`'s shapes (#16669);
//! - `dv_rewrite/*`: `+body:t1 #kw:k0001*` with `DOC_VALUES_REWRITE` over a
//!   SORTED field of 500,000 distinct keys -- a selective conjunction where
//!   building the matching-ordinal set is a large share of the work
//!   (#16297); `dv_rewrite_wildcard/*` the same with `kw:k00*5`, whose
//!   matching ordinals are not contiguous.
//!
//! Each case is run under a counting collector (`COMPLETE_NO_SCORES`,
//! through the scorer -- not `count`'s `Weight.count` shortcut) and a
//! top-10 search (`TOP_SCORES`).
//!
//! Run with `cargo bench -p lucene-search --bench req_excl`.

#![allow(clippy::arithmetic_side_effects)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_index::document::{
    Document, NumericDocValuesField, SortedDocValuesField, Store, StringField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::extended_query::{MultiTermQuery, MultiTermSource, RewriteMethod};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause, PrefixQuery, TermQuery, WildcardQuery};
use lucene_store::ByteBuffersDirectory;

const DOCS: usize = 500_000;
const INTERVAL: usize = 128;
const EXCL_TERMS: usize = 7;

fn build() -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
    w.set_ram_buffer_size_mb(4096.0).unwrap();
    for i in 0..DOCS {
        let mut d = Document::new();
        d.add(StringField::new("body", "all", Store::No));
        if i % 1000 == 7 {
            d.add(StringField::new("body", "t1", Store::No));
        }
        d.add(NumericDocValuesField::indexed_field(
            "ts",
            (i / 1000) as i64,
        ));
        if i % INTERVAL != 0 {
            d.add(StringField::new(
                "dense",
                format!("t{}", i % EXCL_TERMS),
                Store::No,
            ));
        } else {
            d.add(StringField::new(
                "sparse",
                format!("t{}", (i / INTERVAL) % EXCL_TERMS),
                Store::No,
            ));
        }
        d.add(SortedDocValuesField::new(
            "kw",
            format!("k{:07}", (i * 7919) % DOCS).into_bytes(),
        ));
        w.add_fields_document(&d).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    dir
}

/// `TotalHitCountCollector` without the `Weight.count` shortcut.
struct Count(u64);

impl ScoringCollector for Count {
    fn collect(&mut self, _doc: i32, _score: f32) {
        self.0 += 1;
    }
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
    fn add_hits(&mut self, n: u64) -> bool {
        self.0 += n;
        true
    }
}

fn count(searcher: &IndexSearcher<'_, '_>, q: &BooleanQuery) -> u64 {
    let mut c = Count(0);
    searcher.search_collector(q, &mut c).unwrap();
    c.0
}

fn excl_terms(field: &str) -> Clause {
    let mut q = BooleanQuery::new();
    for t in 0..EXCL_TERMS {
        q.should
            .push(Clause::Term(TermQuery::new(field, format!("t{t}"))));
    }
    Clause::Boolean(Box::new(q))
}

fn bench(c: &mut Criterion) {
    let dir = build();
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segs = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]).unwrap();
    let norms: Vec<_> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segs, &norms).unwrap();
    let all = Clause::Term(TermQuery::new("body", "all"));

    let mut cases: Vec<(&str, BooleanQuery)> = Vec::new();
    let mut q = BooleanQuery::new();
    q.must.push(all.clone());
    q.must_not.push(Clause::from(
        lucene_search::extended_query::NumericDocValuesRangeQuery::new("ts", 0, 449),
    ));
    cases.push(("dv_skip_excl", q));
    for (name, field) in [
        ("terms_excl_dense", "dense"),
        ("terms_excl_sparse", "sparse"),
    ] {
        let mut q = BooleanQuery::new();
        q.must.push(all.clone());
        q.must_not.push(excl_terms(field));
        cases.push((name, q));
    }
    let mut q = BooleanQuery::new();
    q.must.push(Clause::Term(TermQuery::new("body", "t1")));
    q.filter.push(Clause::from(MultiTermQuery {
        source: MultiTermSource::Prefix(PrefixQuery {
            field: "kw".to_string(),
            prefix: b"k0001".to_vec(),
        }),
        rewrite: RewriteMethod::DocValues,
    }));
    cases.push(("dv_rewrite", q));
    let mut q = BooleanQuery::new();
    q.must.push(Clause::Term(TermQuery::new("body", "t1")));
    q.filter.push(Clause::from(MultiTermQuery {
        source: MultiTermSource::Wildcard(WildcardQuery {
            field: "kw".to_string(),
            pattern: b"k00*5".to_vec(),
        }),
        rewrite: RewriteMethod::DocValues,
    }));
    cases.push(("dv_rewrite_wildcard", q));

    for (name, q) in &cases {
        let mut g = c.benchmark_group(*name);
        g.bench_function("count", |b| b.iter(|| black_box(count(&searcher, q))));
        g.bench_function("top10", |b| {
            b.iter(|| black_box(searcher.search(q, 10).unwrap()))
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
