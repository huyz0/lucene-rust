//! M13 T13.3c's pairs (`benchmarks/micro/java/AnalyzingSuggestMicro.java` is
//! the Java twin): `GenAnalyzingSuggest`'s 32 phrases
//! (`fixtures/data/suggest_analyzing/phrases.tsv`). `analyzing_lookup`/
//! `fuzzy_lookup`: top 5 for every one to three word prefix and each phrase
//! cut after four characters (so the fuzzy suggester has typos to forgive),
//! units: lookups; `analyzing_build`/`fuzzy_build`: a build over all phrases,
//! units: phrases.

use std::collections::{BTreeSet, HashSet};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::Analyzer;
use lucene_search::suggest::analyzing::{AnalyzingSuggester, FuzzySuggester};
use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::Lookup;

use super::measure;

/// The phrases with their weights, as an `InputIterator`.
pub(super) struct Entries<'a> {
    rows: &'a [(String, i64)],
    next: usize,
}

impl InputIterator for Entries<'_> {
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

pub(super) fn entries(rows: &[(String, i64)]) -> Entries<'_> {
    Entries { rows, next: 0 }
}

/// Word prefixes of each phrase, and each phrase cut after four characters,
/// first-seen order (`LinkedHashSet`).
pub(super) fn queries(rows: &[(String, i64)]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut add = |q: String| {
        if seen.insert(q.clone()) {
            out.push(q);
        }
    };
    for (t, _) in rows {
        let lower = t.to_lowercase();
        let mut acc = String::new();
        for (i, word) in lower.split(' ').take(3).enumerate() {
            if i > 0 {
                acc.push(' ');
            }
            acc.push_str(word);
            add(acc.clone());
        }
        add(lower.chars().take(4).collect());
    }
    out
}

pub(super) fn analyzer() -> Arc<Analyzer> {
    let stops: HashSet<String> = ["the", "of", "a"].iter().map(|s| s.to_string()).collect();
    Arc::new(Analyzer::standard(Some(&stops)))
}

pub(super) fn lookups(name: &str, lk: &dyn Lookup, queries: &[String], w: Duration, m: Duration) {
    measure(name, w, m, || {
        for q in queries {
            black_box(lk.lookup(q, None, false, 5).unwrap().len());
        }
        queries.len() as u64
    });
}

/// `GenAnalyzingSuggest`'s phrases with their weights.
pub(super) fn phrases() -> Vec<(String, i64)> {
    std::fs::read_to_string("fixtures/data/suggest_analyzing/phrases.tsv")
        .unwrap()
        .lines()
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            (c[0].to_string(), c[1].parse().unwrap())
        })
        .collect()
}

pub fn bench_analyzing(w: Duration, m: Duration) {
    let rows = phrases();
    let queries = queries(&rows);
    let mut a = AnalyzingSuggester::new(analyzer());
    a.build(&mut entries(&rows)).unwrap();
    let mut f = FuzzySuggester::new(analyzer());
    f.build(&mut entries(&rows)).unwrap();
    lookups("analyzing_lookup", &a, &queries, w, m);
    lookups("fuzzy_lookup", &f, &queries, w, m);
    measure("analyzing_build", w, m, || {
        let mut b = AnalyzingSuggester::new(analyzer());
        b.build(&mut entries(&rows)).unwrap();
        black_box(b.get_count().unwrap());
        rows.len() as u64
    });
    measure("fuzzy_build", w, m, || {
        let mut b = FuzzySuggester::new(analyzer());
        b.build(&mut entries(&rows)).unwrap();
        black_box(b.get_count().unwrap());
        rows.len() as u64
    });
}
