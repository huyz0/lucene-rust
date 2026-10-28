//! The percolator's hot paths and the spell checker's string distances.
//!
//! - `memory_index/add_query`: a `MemoryIndex` built from one 3000-token
//!   document over a prefix-heavy vocabulary (every token goes through the
//!   field's term hash), then searched with a term query.
//! - `monitor/match_*`: a one-document batch matched against 4000 registered
//!   queries (term queries and five-clause booleans), on one thread and
//!   partitioned over four.
//! - `monitor/batch_of_1doc`: building that batch alone, the part of a
//!   match no thread count can split.
//! - `monitor/register_10k`: 10k one-term queries registered in batches of
//!   1000 (each registration replaces any query of the same id).
//! - `spell/*`: each `StringDistance` over 2000 (query, candidate) pairs of
//!   4-12 ASCII characters.
//!
//! Run with `cargo bench -p lucene-search --bench monitor_memory_spell`.

use std::collections::BTreeMap;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_analysis::Analyzer;
use lucene_search::memory::MemoryIndex;
use lucene_search::monitor::batch::DocumentBatch;
use lucene_search::monitor::facade::{Matcher, Monitor};
use lucene_search::monitor::index::MonitorQuery;
use lucene_search::query::{BooleanQuery, Clause, TermQuery};
use lucene_search::spell::distance::{
    JaroWinklerDistance, LevenshteinDistance, LuceneLevenshteinDistance, NGramDistance,
    StringDistance,
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

/// English-like words from a few syllables: many share a first letter, a
/// prefix and a length.
fn vocabulary(n: usize, seed: u64) -> Vec<String> {
    const SYL: [&str; 16] = [
        "ap", "ple", "ex", "con", "pro", "re", "ta", "in", "ti", "on", "de", "st", "er", "a", "ly",
        "ma",
    ];
    let mut rng = Rng(seed);
    let mut out = std::collections::BTreeSet::new();
    while out.len() < n {
        let parts = 2 + rng.below(4);
        let w: String = (0..parts).map(|_| SYL[rng.below(SYL.len())]).collect();
        if (4..=12).contains(&w.len()) {
            out.insert(w);
        }
    }
    let mut v: Vec<String> = out.into_iter().collect();
    // shuffled, so a document's first-seen order is not sorted
    for i in (1..v.len()).rev() {
        let j = rng.below(i + 1);
        v.swap(i, j);
    }
    v
}

fn document(vocab: &[String], tokens: usize, seed: u64) -> String {
    let mut rng = Rng(seed);
    let mut s = String::new();
    for _ in 0..tokens {
        s.push_str(&vocab[rng.below(vocab.len())]);
        s.push(' ');
    }
    s
}

fn term(text: &str) -> Clause {
    Clause::Term(TermQuery::new("body", text.as_bytes().to_vec()))
}

fn memory_index(c: &mut Criterion) {
    let analyzer = Analyzer::standard(None);
    let vocab = vocabulary(1500, 7);
    let doc = document(&vocab, 3000, 11);
    let query = term(&vocab[3]);
    c.bench_function("memory_index/add_query_3000_tokens", |b| {
        b.iter(|| {
            let mut mi = MemoryIndex::with_options(true, true);
            mi.add_field_text("body", black_box(&doc), &analyzer)
                .unwrap();
            black_box(mi.search(&query).unwrap())
        })
    });
}

fn monitor_with(n_queries: usize, vocab: &[String]) -> Monitor {
    let mut m = Monitor::new(Arc::new(Analyzer::standard(None)));
    let mut rng = Rng(99);
    let mut qs = Vec::new();
    for i in 0..n_queries {
        let q = if i % 3 == 2 {
            let mut b = BooleanQuery::new();
            for k in 0..5 {
                let t = term(&vocab[rng.below(vocab.len())]);
                if k < 2 {
                    b.must.push(t);
                } else {
                    b.should.push(t);
                }
            }
            Clause::Boolean(Box::new(b))
        } else {
            term(&vocab[rng.below(vocab.len())])
        };
        qs.push(MonitorQuery::new(format!("q{i}"), q, None, BTreeMap::new()).unwrap());
    }
    m.register(qs).unwrap();
    m
}

fn monitor(c: &mut Criterion) {
    let vocab = vocabulary(6000, 3);
    let m = monitor_with(4000, &vocab);
    let doc = document(&vocab, 400, 5);
    let fields: Vec<(&str, &str)> = vec![("body", doc.as_str())];
    let docs = [fields.as_slice()];
    let single = m.match_docs(&docs, &Matcher::Simple).unwrap();
    let parted = m
        .match_docs_partitioned(&docs, &Matcher::Simple, 4)
        .unwrap();
    assert_eq!(single, parted);
    eprintln!(
        "monitor: {} candidates, {} matches",
        single.queries_run,
        single.match_count(0)
    );
    let mut g = c.benchmark_group("monitor");
    // the batch alone (analysis and the MemoryIndex), which both match paths
    // build once before any query runs
    let analyzer = Arc::new(Analyzer::standard(None));
    g.bench_function("batch_of_1doc", |b| {
        b.iter(|| {
            black_box(
                DocumentBatch::of(&analyzer, black_box(&docs), None)
                    .unwrap()
                    .max_doc(),
            )
        })
    });
    g.bench_function("match_1doc_4000q_single", |b| {
        b.iter(|| black_box(m.match_docs(black_box(&docs), &Matcher::Simple).unwrap()))
    });
    g.bench_function("match_1doc_4000q_partitioned4", |b| {
        b.iter(|| {
            black_box(
                m.match_docs_partitioned(black_box(&docs), &Matcher::Simple, 4)
                    .unwrap(),
            )
        })
    });
    // the highlighting matcher does more per candidate, so the run phase
    // dominates the batch build
    g.bench_function("highlight_1doc_4000q_single", |b| {
        b.iter(|| {
            black_box(
                m.match_docs(black_box(&docs), &Matcher::Highlights)
                    .unwrap(),
            )
        })
    });
    g.bench_function("highlight_1doc_4000q_partitioned4", |b| {
        b.iter(|| {
            black_box(
                m.match_docs_partitioned(black_box(&docs), &Matcher::Highlights, 4)
                    .unwrap(),
            )
        })
    });
    g.sample_size(10);
    g.bench_function("register_10k_in_batches_of_1000", |b| {
        b.iter(|| {
            let mut m = Monitor::new(Arc::new(Analyzer::standard(None)));
            for chunk in 0..10 {
                let qs = (0..1000)
                    .map(|i| {
                        let id = chunk * 1000 + i;
                        MonitorQuery::new(
                            format!("r{id}"),
                            term(&vocab[id % vocab.len()]),
                            None,
                            BTreeMap::new(),
                        )
                        .unwrap()
                    })
                    .collect();
                m.register(qs).unwrap();
            }
            black_box(m.disjunct_count())
        })
    });
    g.finish();
}

fn spell(c: &mut Criterion) {
    let vocab = vocabulary(4000, 21);
    let mut rng = Rng(5);
    let pairs: Vec<(String, String)> = (0..2000)
        .map(|_| {
            (
                vocab[rng.below(vocab.len())].clone(),
                vocab[rng.below(vocab.len())].clone(),
            )
        })
        .collect();
    let distances: Vec<(&str, Box<dyn StringDistance>)> = vec![
        ("levenshtein", Box::new(LevenshteinDistance::new())),
        ("jaro_winkler", Box::new(JaroWinklerDistance::new())),
        ("ngram2", Box::new(NGramDistance::new())),
        (
            "lucene_levenshtein",
            Box::new(LuceneLevenshteinDistance::new()),
        ),
    ];
    let mut g = c.benchmark_group("spell");
    for (name, d) in &distances {
        g.bench_function(*name, |b| {
            b.iter(|| {
                let mut s = 0f32;
                for (q, t) in &pairs {
                    s += d.get_distance(black_box(q), black_box(t));
                }
                black_box(s)
            })
        });
    }
    g.finish();
}

criterion_group!(benches, memory_index, monitor, spell);
criterion_main!(benches);
