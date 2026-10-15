# lucene-monitor -- the percolator (M13 T13.5)

[Index](../parity.md). `org.apache.lucene.monitor`: queries registered once, a document batch
matched against all of them through a presearcher over each query's term tree. Four sub-tasks
(T13.5a-d, [m13](../milestones/m13-search-application-modules.md)); this file covers T13.5a, the
query analysis the presearcher is built from.

| Java | Rust | Status |
|---|---|---|
| `monitor/QueryTree` | `lucene-search/src/monitor/tree.rs::QueryTree` | **ported** T13.5a: term, any-term marker, conjunction and disjunction nodes; `weight`, `collectTerms`, `toString(depth)` in Java's layout. The `Term` weight check is Java's `IllegalArgumentException`. Tests: `monitor_fixtures.rs` (`tree.tsv`). |
| `monitor/TermWeightor` | `lucene-search/src/monitor/tree.rs::TermWeightor` | **ported** T13.5a: `combine`, `fieldWeightor`, `termWeightor`, `termFreqWeightor`, `lengthWeightor`, `DEFAULT`; weights bit-equal (`weights.tsv`). Differs: `termAndFieldWeightor` is not ported (no caller in the core path). |
| `monitor/QueryAnalyzer` | `lucene-search/src/monitor/analyzer.rs::QueryAnalyzer` | **ported** T13.5a: `QueryBuilder`'s rules over `Clause` (`MUST`/`FILTER` conjunction children, `SHOULD` disjunction children unless a required clause makes them redundant, `MUST_NOT` only in a pure negative boolean, boosts as a `MUST` child, leaves through `consumeTerms` in visit order). Differs: the `CustomQueryHandler` mapper is not wired in (`unknownQueryMapper` is always null); equally weighted siblings come in the port's order, not Java's `HashMap`-by-`Class.hashCode()` order, which varies between JVM runs. Tests: `monitor_fixtures.rs` (`tree.tsv`). |
| `monitor/QueryDecomposer` | `lucene-search/src/monitor/analyzer.rs::QueryDecomposer` | **ported** T13.5a: disjunction-max, boost, boolean (mandatory, exclusions, minimum-should-match) decomposition with `LinkedHashSet` order. Tests: `monitor_fixtures.rs` (`decompose.tsv`). |
| `monitor/package-info` | `lucene-search/src/monitor/mod.rs` | not a class: the module's documentation and re-exports. |
