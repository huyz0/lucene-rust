# opensearch-plugin

The lucene-rust plugin for **OpenSearch 3.8.0** (Lucene 10.5.0): it runs the
query phase of supported searches in Rust, through `liblucene_ffi.so` over JNI,
and leaves everything else — unsupported searches, fetch, indexing, refresh —
to OpenSearch and Lucene. Milestone:
[`docs/milestones/m2-opensearch-read-path.md`](../docs/milestones/m2-opensearch-read-path.md).
What runs native: [`docs/opensearch-native-queries.md`](../docs/opensearch-native-queries.md).

## Build

```
scripts/opensearch-dist.sh                 # once: lib/ of the pinned image -> target/opensearch-dist/
gradle -p opensearch-plugin bundlePlugin   # build/distributions/lucene-rust-0.1.0.zip
gradle -p opensearch-plugin check          # JVM-side differential self test
```

The plugin compiles against the jars of the distribution it installs into,
extracted from the `opensearchproject/opensearch:3.8.0` image — a plugin must
match its node exactly, and the build then needs no Maven repository.
`bundlePlugin` runs `cargo build --release -p lucene-ffi` and puts the library
under `native/<os>-<arch>/` in the zip; `-PextraNative=linux-aarch64=PATH`
adds one built elsewhere.

## Install

```
bin/opensearch-plugin install file:///path/to/lucene-rust-0.1.0.zip
```

The node loads the library and checks its ABI version at startup; a missing
or mismatched library stops the node with one line saying which file and
platform. OpenSearch accepts one `QueryPhaseSearcher` per node, so this plugin
cannot be installed alongside `neural-search`.

## Operate

| | |
|---|---|
| `index.lucene_rust.search.enabled` (index, dynamic, default `true`) | route this index's searches native when eligible |
| `index.lucene_rust.search.native_shapes` (index, dynamic, `fast`/`all`, default `fast`) | `all` also runs shapes that are correct natively but measured slower. Since read path R1 there are none (every encodable shape measures faster), so the two values agree |
| `GET /_plugins/lucene_rust/stats` | native queries, native errors, fallbacks by reason, open native readers |

A native failure is logged, counted as `native_errors`, and the query is re-run
on Lucene: it never fails a search Lucene can answer.

## Layout

| | |
|---|---|
| `RustSearchPlugin` | the plugin: loads the library, registers the searcher, settings and stats endpoint |
| `RustQueryPhaseSearcher` | eligibility, then native search or OpenSearch's own `QueryPhaseSearcherWrapper` |
| `QueryEncoder` | rewritten Lucene `Query` → the query blob `jvm_reader.rs` decodes, or a fallback reason; `isFast` is the measured routing |
| `NativeReaders` | one native reader per Java searcher reader: built from its `SegmentInfos`, `maxDoc`s and live docs, closed by its close listener |
| `NativeBridge`, `NativeLibrary` | the JNI surface and the loader/handshake |
| `src/test/…/NativeSelfTest` | the native path against Lucene's `IndexSearcher`: NRT readers with in-memory deletes, refreshes, merges, every Java-written fixture, JNI error paths |
| `src/test/…/NativeBench` | `gradle nativeBench`: JNI crossing cost and in-process latency |
| `src/yamlRestTest/…/LuceneRustYamlIT` | OpenSearch's REST YAML suites, for `verify-opensearch.sh --yaml` |
| `e2e/verify_opensearch.py` | the node-level harness `scripts/verify-opensearch.sh` runs |
| `docker/Dockerfile` | the test node: the pinned image, bundled plugins removed, this one installed |

The Rust side is `crates/lucene-ffi/src/jvm_reader.rs` (C ABI, unit-tested
without a JVM) and `crates/lucene-ffi/src/jni_bridge.rs` (marshalling only).
