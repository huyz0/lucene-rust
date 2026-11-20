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

## Performance

Measured on a 4-core cloud machine against stock OpenSearch 3.9.0, stock Apache Lucene 10.6 and the engine on its own
(no JVM, no OpenSearch). The plugin's numbers include OpenSearch's own request path, which both stock and the plugin pay:
about 0.4 ms per search and about 12 CPU-seconds per million documents indexed. The engine-only numbers show what is left
without it.

| | OpenSearch to plugin | Engine alone |
|---|---|---|
| Search, 195 request shapes (geometric mean) | 1.76x faster, 1.83x less CPU (1.27x with direct search off) | 2.8x over OpenSearch for the engine's share of a request; 2.7x over Lucene on 87 query shapes |
| Indexing, log documents (warm) | 1.4x docs/s, 0.64x the CPU | 2.2-2.9x Lucene's writer, same index size |
| k-NN, neural, hybrid, rerank | 1.15-3.2x | HNSW 1.4-1.9x Lucene |
| Highlight on long documents | 19x | |

Weak spots, measured: percentile aggregations about 1.1x, cheap bucket aggregations over REST around 1.0x, an hourly `date_histogram` with a filter 0.6x,
the plain highlighter and `has_child` queries below 1.0x, and end-to-end indexing capped near 1.3-1.5x by OpenSearch's bulk path. The
[performance page](https://huyz0.github.io/lucene-rust/performance.html) has the tables, what was re-measured and what was reused, and how to rerun each.

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
