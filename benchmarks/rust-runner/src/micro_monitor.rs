//! M13 T13.5c's pairs (`benchmarks/micro/java/MonitorMicro.java` is the Java
//! twin): a `Monitor` over the 41 queries of
//! `fixtures/corpus/monitor-queries.txt`. `mon_register`: a fresh monitor,
//! every query registered; `mon_match`: four documents matched against the
//! registered set.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::monitor::facade::Monitor;
use lucene_search::monitor::index::MonitorQuery;
use lucene_search::queryparser::classic::QueryParser;

use super::measure;

const DOCS: &[&[(&str, &str)]] = &[
    &[("body", "the quick brown fox")],
    &[("body", "a b"), ("title", "quick")],
    &[("body", "fox fox fox"), ("title", "brown")],
    &[("body", "zzzz extraordinarily")],
];

pub fn bench_monitor(w: Duration, m: Duration) {
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let parser = QueryParser::new("body", Arc::clone(&analyzer));
    let texts: Vec<String> = std::fs::read_to_string("fixtures/corpus/monitor-queries.txt")
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    let queries: Vec<MonitorQuery> = texts
        .iter()
        .enumerate()
        .map(|(i, q)| {
            MonitorQuery::new(
                format!("q{i}"),
                parser.parse(q).unwrap().to_clause(1),
                Some(q.clone()),
                BTreeMap::new(),
            )
            .unwrap()
        })
        .collect();
    measure("mon_register", w, m, || {
        let mut monitor = Monitor::new(Arc::clone(&analyzer));
        monitor.register(queries.clone()).unwrap();
        black_box(monitor.query_count());
        queries.len() as u64
    });
    let mut monitor = Monitor::new(Arc::clone(&analyzer));
    monitor.register(queries.clone()).unwrap();
    measure("mon_match", w, m, || {
        for d in DOCS {
            black_box(monitor.match_doc(d).unwrap().match_count());
        }
        DOCS.len() as u64
    });
}
