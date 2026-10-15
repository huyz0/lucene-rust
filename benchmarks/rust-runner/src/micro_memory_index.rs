//! M13 T13.5's pairs (`benchmarks/micro/java/MemoryIndexMicro.java` is the
//! Java twin): `MemoryIndex` over `fixtures/corpus/memory-index-docs.txt`
//! with offsets, the queries of `memory-index-queries.txt` parsed beforehand.
//! `mi_build`: every document indexed and frozen; `mi_build_search`: every
//! document indexed, then every query run against it (the monitor's shape);
//! `mi_search_frozen`: every query against documents frozen beforehand.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::memory::MemoryIndex;
use lucene_search::query::Clause;
use lucene_search::queryparser::classic::QueryParser;

use super::measure;

fn build(doc: &[(String, String)], a: &Analyzer) -> MemoryIndex {
    let mut mi = MemoryIndex::with_options(true, false);
    for (field, value) in doc {
        mi.add_field_text(field, value, a).unwrap();
    }
    mi.freeze().unwrap();
    mi
}

pub fn bench_memory_index(w: Duration, m: Duration) {
    let a = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let docs: Vec<Vec<(String, String)>> =
        std::fs::read_to_string("fixtures/corpus/memory-index-docs.txt")
            .unwrap()
            .lines()
            .map(|l| {
                l.split('\t')
                    .map(|p| {
                        let (f, v) = p.split_once('=').unwrap();
                        (f.to_string(), v.to_string())
                    })
                    .collect()
            })
            .collect();
    let parser = QueryParser::new("body", Arc::clone(&a));
    let queries: Vec<Clause> = std::fs::read_to_string("fixtures/corpus/memory-index-queries.txt")
        .unwrap()
        .lines()
        .map(|q| parser.parse(q).unwrap().to_clause(1))
        .collect();
    let units = |n: usize| u64::try_from(n).unwrap();
    measure("mi_build", w, m, || {
        for d in &docs {
            black_box(build(d, &a));
        }
        units(docs.len())
    });
    measure("mi_build_search", w, m, || {
        for d in &docs {
            let mut mi = build(d, &a);
            for q in &queries {
                black_box(mi.search(q).unwrap().to_bits());
            }
        }
        units(docs.len() * queries.len())
    });
    let mut frozen: Vec<MemoryIndex> = docs.iter().map(|d| build(d, &a)).collect();
    measure("mi_search_frozen", w, m, || {
        for mi in &mut frozen {
            for q in &queries {
                black_box(mi.search(q).unwrap().to_bits());
            }
        }
        units(frozen.len() * queries.len())
    });
}
