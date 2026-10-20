//! `org.apache.lucene.benchmark.quality`: lucene-benchmark's search-quality
//! measurement -- TREC topics and judgements run against an index, judged,
//! averaged and written as a TREC submission -- and its two command lines,
//! `QueryDriver` ([`trec::query_driver`]) and `QualityQueriesFinder`
//! ([`quality_utils::quality_queries_finder`]).
//!
//! The rest of lucene-benchmark (`byTask`'s algorithm files, tasks, content
//! sources and reports, `jmh`) is a Java performance harness: this project
//! measures itself with `scripts/bench-micro.sh` pairs and
//! `benchmarks/rust-runner` instead, so it is not ported (see
//! `docs/inventory/lucene-benchmark.tsv`).

pub mod quality;
pub mod quality_utils;
pub mod trec;
