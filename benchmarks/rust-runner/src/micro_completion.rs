//! M13 T13.3d's pairs (`benchmarks/micro/java/CompletionMicro.java` is the
//! Java twin): `SuggestIndexSearcher::suggest` over `GenCompletionQuery`'s
//! index (three segments, one with deletions), top 10 with duplicates
//! skipped. `completion_prefix`: 12 prefixes on `suggest`;
//! `completion_fuzzy`: 6 misspellings, default fuzzy options;
//! `completion_regex`: 6 regular expressions; `completion_context`: 6
//! prefixes on `ctx` boosted by two contexts.

use std::collections::HashSet;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::Analyzer;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::suggest::completion_query::{
    CompletionAnalyzer, CompletionQuery, ContextQuery, FuzzyCompletionQuery, PrefixCompletionQuery,
    RegexCompletionQuery, SuggestIndexSearcher,
};

use super::measure;

const PREFIXES: [&str; 12] = [
    "b", "be", "the b", "beatles", "new", "new y", "n", "queen o", "music of", "caf", "日", "x",
];
const MISSPELT: [&str; 6] = ["nwe", "beatels", "bech", "quen", "cafe", "nigth"];
const REGEXES: [&str; 6] = ["b[a-z]*", "ne.", "(the|be)", "n.*t", "q.*e", "[^n]"];
const CONTEXT_PREFIXES: [&str; 6] = ["b", "the", "new y", "n", "q", "x"];

fn run(
    name: &str,
    s: &SuggestIndexSearcher<'_>,
    queries: &[CompletionQuery],
    w: Duration,
    m: Duration,
) {
    measure(name, w, m, || {
        for q in queries {
            black_box(s.suggest(q, 10, true).unwrap().len());
        }
        queries.len() as u64
    });
}

pub fn bench_completion(w: Duration, m: Duration) {
    let stops: HashSet<String> = ["the", "of"].iter().map(|s| s.to_string()).collect();
    let a = CompletionAnalyzer::new(Arc::new(Analyzer::standard(Some(&stops))));
    let reader = DirectoryReader::open(&lucene_store::FsDirectory::open(
        "fixtures/data/completion_query/index",
    ))
    .unwrap();
    let s = SuggestIndexSearcher::new(&reader);
    let prefix: Vec<CompletionQuery> = PREFIXES
        .iter()
        .map(|p| {
            CompletionQuery::Prefix(
                PrefixCompletionQuery::new(a.clone(), "suggest", p, None).unwrap(),
            )
        })
        .collect();
    let fuzzy: Vec<CompletionQuery> = MISSPELT
        .iter()
        .map(|p| {
            CompletionQuery::Fuzzy(
                FuzzyCompletionQuery::new(a.clone(), "suggest", p, None).unwrap(),
            )
        })
        .collect();
    let regex: Vec<CompletionQuery> = REGEXES
        .iter()
        .map(|r| CompletionQuery::Regex(RegexCompletionQuery::new("suggest", r, None).unwrap()))
        .collect();
    let context: Vec<CompletionQuery> = CONTEXT_PREFIXES
        .iter()
        .map(|p| {
            let inner = CompletionQuery::Prefix(
                PrefixCompletionQuery::new(a.clone(), "ctx", p, None).unwrap(),
            );
            let mut q = ContextQuery::new(inner).unwrap();
            q.add_context("music", 2.0, true).unwrap();
            q.add_context("mov", 3.0, false).unwrap();
            CompletionQuery::Context(q)
        })
        .collect();
    run("completion_prefix", &s, &prefix, w, m);
    run("completion_fuzzy", &s, &fuzzy, w, m);
    run("completion_regex", &s, &regex, w, m);
    run("completion_context", &s, &context, w, m);
}
