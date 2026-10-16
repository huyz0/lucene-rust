//! Writes a persisted monitor query index for `VerifyMonitorIndex.java`: the
//! 41 queries of `fixtures/corpus/monitor-queries.txt` registered as `q0..q40`
//! in a monitor whose index lives at the given directory, in the layout
//! Lucene's `WritableQueryIndex` writes. Lucene's own `Monitor` must open it,
//! count the same queries and pieces, and match the same documents.
//!
//! Run from the repository root: `write_monitor_index_fixture <dir>`.

use std::collections::BTreeMap;
use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::monitor::config::MonitorConfiguration;
use lucene_search::monitor::facade::Monitor;
use lucene_search::monitor::index::MonitorQuery;
use lucene_search::monitor::presearcher::TermFilteredPresearcher;
use lucene_search::monitor::serializer::MonitorQuerySerializer;
use lucene_search::queryparser::classic::QueryParser;

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_monitor_index_fixture <dir>");
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let parser = Arc::new(QueryParser::new("body", Arc::clone(&analyzer)));
    let serializer = {
        let parser = Arc::clone(&parser);
        MonitorQuerySerializer::from_parser(Arc::new(move |t: &str| {
            parser
                .parse(t)
                .map(|p| p.to_clause(1))
                .map_err(|e| e.message)
        }))
    };
    let config = MonitorConfiguration::default().set_index_path(&out, serializer);
    let mut monitor = Monitor::with_configuration(
        Arc::clone(&analyzer),
        TermFilteredPresearcher::default(),
        Vec::new(),
        &config,
    )
    .expect("open the monitor");
    let queries: Vec<MonitorQuery> = std::fs::read_to_string("fixtures/corpus/monitor-queries.txt")
        .expect("run from the repository root")
        .lines()
        .enumerate()
        .map(|(i, q)| {
            MonitorQuery::new(
                format!("q{i}"),
                parser.parse(q).expect("a corpus query").to_clause(1),
                Some(q.to_string()),
                BTreeMap::new(),
            )
            .expect("a monitor query")
        })
        .collect();
    monitor.register(queries).expect("register");
}
