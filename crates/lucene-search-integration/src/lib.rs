#![forbid(unsafe_code)]
//! Tests, benches and examples that drive `lucene-search` together with the
//! crates built on top of it (query parser, monitor, suggest, grouping,
//! expressions, facets). They live here, not in `lucene-search`, so that crate
//! carries no dev-dependency cycle and its own test build compiles once.
