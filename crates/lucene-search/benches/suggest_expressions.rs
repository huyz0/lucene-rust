//! The infix suggester's writer/reader cycle and the expression layer's
//! per-document evaluation.
//!
//! - `infix/build_50k`: `AnalyzingInfixSuggester::build` over 50k
//!   three-to-six-word suggestions (analysis, edge n-grams, the snapshot).
//! - `infix/refresh_clean`: `refresh` on an open writer with nothing added
//!   since the last one (Java's `SearcherManager.maybeRefresh` is a no-op).
//! - `infix/update_100_refresh`: 100 `update`s of existing texts, then one
//!   `refresh`.
//! - `infix/lookup_*`: top-10 lookups, a frequent first word plus a short
//!   prefix (`"ap pr"`) and a long prefix (`"apple con"`), unhighlighted.
//! - `expr/*`: an expression's values read for every document of a 1M-doc
//!   leaf, its variables bound to in-memory columns (one sparse) and the
//!   scores, as a sort's comparator reads them; `expr/*_batch` the same
//!   values 64 documents at a time, as a function query's scorer reads them.
//!
//! Run with `cargo bench -p lucene-search --bench suggest_expressions`.

use std::collections::BTreeSet;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_analysis::Analyzer;
use lucene_search::expressions::{compile, SimpleBindings};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::suggest::infix::AnalyzingInfixSuggester;
use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::Lookup;
use lucene_search::values_source::{
    self as vs, BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};

/// A deterministic xorshift.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const SYL: [&str; 16] = [
    "ap", "ple", "ex", "con", "pro", "re", "ta", "in", "ti", "on", "de", "st", "er", "a", "ly",
    "ma",
];

fn word(rng: &mut Rng) -> String {
    let parts = 1 + rng.below(4);
    (0..parts).map(|_| SYL[rng.below(SYL.len())]).collect()
}

/// `n` suggestions of 3-6 words, a fifth of them starting with "apple".
fn suggestions(n: usize) -> Vec<(String, i64)> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    (0..n)
        .map(|i| {
            let words = 3 + rng.below(4);
            let mut s = if i % 5 == 0 {
                "apple".to_string()
            } else {
                word(&mut rng)
            };
            for _ in 1..words {
                s.push(' ');
                s.push_str(&word(&mut rng));
            }
            // a unique tail, so every text is its own suggestion
            s.push_str(&format!(" s{i}"));
            (s, rng.below(1_000_000) as i64)
        })
        .collect()
}

struct Rows<'a> {
    rows: &'a [(String, i64)],
    next: usize,
}

impl InputIterator for Rows<'_> {
    fn next(&mut self) -> Option<Vec<u8>> {
        let row = self.rows.get(self.next)?;
        self.next += 1;
        Some(row.0.as_bytes().to_vec())
    }
    fn weight(&self) -> i64 {
        self.rows[self.next - 1].1
    }
    fn payload(&self) -> Option<&[u8]> {
        None
    }
    fn has_payloads(&self) -> bool {
        false
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        None
    }
    fn has_contexts(&self) -> bool {
        false
    }
}

fn built(rows: &[(String, i64)]) -> AnalyzingInfixSuggester {
    let analyzer = Arc::new(Analyzer::standard(None));
    // writer kept open, so add/update/refresh work after the build
    let mut s = AnalyzingInfixSuggester::with_options(
        analyzer.clone(),
        analyzer,
        4,
        false,
        true,
        false,
        false,
    )
    .unwrap();
    s.build(&mut Rows { rows, next: 0 }).unwrap();
    s
}

fn infix(c: &mut Criterion) {
    let rows = suggestions(50_000);
    let mut g = c.benchmark_group("infix");
    g.sample_size(10);
    g.bench_function("build_50k", |b| {
        b.iter(|| black_box(built(&rows).get_count().unwrap()))
    });
    let mut s = built(&rows);
    g.bench_function("refresh_clean", |b| {
        b.iter(|| s.refresh().unwrap());
    });
    let mut round = 0usize;
    g.bench_function("update_100_refresh", |b| {
        b.iter(|| {
            for k in 0..100 {
                let (text, weight) = &rows[(round * 100 + k) * 7 % rows.len()];
                s.update(text.as_bytes(), None, weight + 1, None).unwrap();
            }
            round += 1;
            s.refresh().unwrap();
        })
    });
    g.sample_size(30);
    for (name, key, all) in [
        ("lookup_ap_pr", "apple pr", true),
        ("lookup_apple_con", "apple con", true),
        ("lookup_prefix_appl", "appl", true),
        ("lookup_any_ap_pr", "apple pr", false),
    ] {
        g.bench_function(name, |b| {
            b.iter(|| black_box(s.lookup_with(key, None, 10, all, false).unwrap().len()))
        });
    }
    g.finish();
}

/// An in-memory `double` column; `every`-th document has no value.
struct Column {
    values: Arc<Vec<f64>>,
    every: usize,
    name: &'static str,
}

struct ColumnValues {
    values: Arc<Vec<f64>>,
    every: usize,
    doc: usize,
}

impl DoubleValues for ColumnValues {
    fn advance_exact(&mut self, doc: i32) -> lucene_search::Result<bool> {
        self.doc = doc as usize;
        Ok(self.every == 0 || !self.doc.is_multiple_of(self.every))
    }
    fn double_value(&mut self) -> lucene_search::Result<f64> {
        Ok(self.values[self.doc])
    }
    fn batch_capable(&self) -> bool {
        true
    }
}

impl DoubleValuesSource for Column {
    fn get_values<'c>(
        &self,
        _ctx: &ValuesContext<'c>,
        _leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> lucene_search::Result<BoxDoubleValues<'c>> {
        Ok(Box::new(ColumnValues {
            values: self.values.clone(),
            every: self.every,
            doc: 0,
        }))
    }
    fn needs_scores(&self) -> bool {
        false
    }
    fn is_cacheable(&self, _ctx: &ValuesContext<'_>, _leaf: usize) -> bool {
        true
    }
    fn describe(&self) -> String {
        self.name.to_string()
    }
}

const DOCS: usize = 1_000_000;
/// A function query's batch (a multiple of which `DOCS` is).
const BATCH: usize = 64;

fn column(seed: u64, every: usize, name: &'static str) -> Arc<dyn DoubleValuesSource> {
    let mut rng = Rng(seed);
    let values = (0..DOCS)
        .map(|_| 1.0 + rng.below(10_000) as f64 / 7.0)
        .collect();
    Arc::new(Column {
        values: Arc::new(values),
        every,
        name,
    })
}

fn expressions(c: &mut Criterion) {
    let mut bindings = SimpleBindings::new();
    bindings.add_source("popularity", column(1, 0, "popularity"));
    bindings.add_source("recency", column(2, 0, "recency"));
    bindings.add_source("price", column(3, 10, "price"));
    bindings.add_source("_score", vs::scores());
    let scores = column(4, 0, "scores");
    let searcher = IndexSearcher::new(&[], &[]).unwrap();
    let ctx = ValuesContext::new(&searcher);
    let mut g = c.benchmark_group("expr");
    g.sample_size(20);
    for (name, text) in [
        (
            "sort_3vars_1m",
            "ln(popularity) * 0.5 + recency / (1 + price)",
        ),
        (
            "score_4vars_1m",
            "sqrt(_score) + ln(popularity) * 0.5 + recency * 0.01 - price * 0.001",
        ),
    ] {
        let source = Arc::new(compile(text).unwrap())
            .double_values_source(&bindings)
            .unwrap();
        g.bench_function(name, |b| {
            b.iter(|| {
                let score_values = scores.get_values(&ctx, 0, None).unwrap();
                let mut v = source.get_values(&ctx, 0, Some(score_values)).unwrap();
                let mut sum = 0.0;
                for doc in 0..DOCS as i32 {
                    if v.advance_exact(doc).unwrap() {
                        sum += v.double_value().unwrap();
                    }
                }
                black_box(sum)
            })
        });
        // the same values a batch at a time, as a function query's
        // scorer reads them (`DoubleValues::fill_batch`)
        g.bench_function(format!("{name}_batch"), |b| {
            let docs: Vec<i32> = (0..BATCH as i32).collect();
            let batch_scores = vec![0f32; BATCH];
            let mut out = vec![0f64; BATCH];
            let mut has = vec![false; BATCH];
            b.iter(|| {
                let score_values = scores.get_values(&ctx, 0, None).unwrap();
                let mut v = source.get_values(&ctx, 0, Some(score_values)).unwrap();
                let mut sum = 0.0;
                let mut ids = docs.clone();
                for start in (0..DOCS).step_by(BATCH) {
                    for (k, d) in ids.iter_mut().enumerate() {
                        *d = (start + k) as i32;
                    }
                    v.fill_batch(&ids, &batch_scores, &mut out, &mut has)
                        .unwrap();
                    for (&o, &h) in out.iter().zip(&has) {
                        if h {
                            sum += o;
                        }
                    }
                }
                black_box(sum)
            })
        });
    }
    g.finish();
}

criterion_group!(benches, infix, expressions);
criterion_main!(benches);
