# accelerated-search

**A native search and indexing engine for OpenSearch, compatible with stock Lucene indices.**

[Documentation](https://huyz0.github.io/lucene-rust/) ·
[Get started](https://huyz0.github.io/lucene-rust/get-started.html) ·
[Performance](https://huyz0.github.io/lucene-rust/performance.html) ·
[Architecture](https://huyz0.github.io/lucene-rust/architecture.html)

accelerated-search is an OpenSearch 3.9.0 plugin that runs a shard's query phase, aggregations,
stored-field reads, indexing and merging in native code, loaded through Java's Foreign Function &
Memory API. Whatever it does not handle runs on OpenSearch as usual, and the index files are ordinary
Lucene segments, so removing the plugin leaves a normal index. The native code is a port of Apache
Lucene (pinned at 10.6.0) written in Rust.

- **Faster**: native search beats stock on most request shapes; log dashboards that bucket and
  summarise numbers run several times faster; indexing CPU drops by about 40% on log-shaped documents.
- **Leaner**: optional columnar doc values, zstd stored fields and leaner sequence numbers cut disk size;
  one page cache and explicit memory budgets bound the node's memory.
- **Drop-in**: serves the k-NN, neural-search, sparse-search and ml-commons APIs under their stock
  names, with opt-in IVF and disk HNSW vector indexes, and optional AES-256-GCM encrypted storage
  with a FIPS 140-3 build.
- **Reversible**: one node setting returns a node to OpenSearch's own engine, with no reindex.

## Quick start

Requirements: OpenSearch **3.9.0**, Linux x86_64 or aarch64 (glibc 2.34+), the JDK bundled with
OpenSearch (25).

```sh
scripts/lr build                                # native library and the plugin zip
bin/opensearch-plugin install file:///path/to/accelerated-search-0.1.0.zip
echo '--enable-native-access=ALL-UNNAMED' > config/jvm.options.d/accelerated-search.options
```

Restart the node, then `GET /_plugins/accelerated/stats` shows how many searches ran natively and,
per reason, how many fell back. The [documentation](https://huyz0.github.io/lucene-rust/) covers
the feature matrix, settings, compatibility with stock, operations, upgrades and rollback.

## Repository layout

| Path | Contents |
|---|---|
| `crates/` | Cargo workspace: one crate per Lucene module boundary, plus the engine and the C-ABI library |
| `opensearch-plugin/` | the Java plugin, its tests and end-to-end drivers |
| `fixtures/` | Java generators and the real-Lucene bytes the differential tests read |
| `benchmarks/` | benchmark harnesses and corpus builders |
| `scripts/` | gates, benchmarks, verification, and `scripts/lr`, the one entry point for build and test |
| `docs/` | reference documentation, the parity ledger and the Lucene class inventory |
| `site/` | the documentation site (mdBook), published to GitHub Pages |

## Build and test

```sh
scripts/fetch-data.py        # dictionaries and other generated data the repository does not store
scripts/lr doctor            # what this machine has and lacks
scripts/lr test quick        # static checks and the unit tests of the crates you changed
scripts/lr test full         # the whole gate and every OpenSearch node proof
cargo bench --workspace      # criterion microbenchmarks of the decode hot loops
```

`AGENTS.md` is the short orientation for contributors and coding agents; the documentation site's
Contributing page lists the rules the gates enforce.

## License

Apache-2.0. This is a derivative work of Apache Lucene; see `LICENSE` and `NOTICE`.
