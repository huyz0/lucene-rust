# opensearch-plugin

The lucene-rust plugin for **OpenSearch 3.8.0** (Lucene 10.5.0): it runs the
query phase of supported searches in Rust, through `liblucene_ffi.so` called
with Java's Foreign Function & Memory API (JDK 25, the JDK OpenSearch 3.8.0 bundles),
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

Give the node native access, in `config/jvm.options.d/lucene-rust.options`:

```
--enable-native-access=ALL-UNNAMED
```

Without it JDK 25 prints a warning the first time any code in the node calls a
restricted method (OpenSearch's own JNA usually gets there first); a later JDK
will refuse the call instead.

The node loads the library and checks its ABI version at startup; a missing
or mismatched library stops the node with one line saying which file and
platform. OpenSearch accepts one `QueryPhaseSearcher` per node, so this plugin
cannot be installed alongside `neural-search`.

## Operate

| | |
|---|---|
| `index.lucene_rust.search.enabled` (index, dynamic, default `true`) | route this index's searches native when eligible |
| `index.lucene_rust.search.native_shapes` (index, dynamic, `fast`/`all`, default `fast`) | `all` also runs shapes that are correct natively but measured slower. Since read path R1 there are none (every encodable shape measures faster), so the two values agree |
| `index.lucene_rust.fetch.enabled` (index, dynamic, default `true`) | read this index's stored fields (`_source`, `_id`, `stored_fields`) natively in the fetch phase and the get API (read path R6) |
| `lucene_rust.fetch.reader_wrapper` (node, default `true` unless the security plugin is installed) | install the index reader wrapper native stored fields need. An index takes one reader wrapper and the security plugin claims it for field- and document-level security, so with it installed the default is off and stored fields are read by Lucene; if another plugin set a wrapper first, this one logs and stands down |
| `GET /_plugins/lucene_rust/stats` | native queries, native errors, fallbacks by reason, open native readers, native stored-fields reads (`native_fetches`, `native_sequential_fetches`) and `StoredFields.document` time per path (`fetch_nanos`) |

A native failure is logged, counted as `native_errors`, and the query is re-run
on Lucene: it never fails a search Lucene can answer.

Native memory the JVM does not see, per segment of 10,000 documents or more:
the query cache (at most 16 MB) and, for sorted searches, decoded sort columns
(at most 32 MB per segment and 512 MB for the whole process; a column is
decoded on its second use in a segment, only when a per-document read would
decode it, and is freed with the segment).

## Layout

| | |
|---|---|
| `RustSearchPlugin` | the plugin: loads the library, registers the searcher, settings and stats endpoint |
| `RustQueryPhaseSearcher` | eligibility, then native search or OpenSearch's own `QueryPhaseSearcherWrapper` |
| `QueryEncoder` | rewritten Lucene `Query` → the query blob `jvm_reader.rs` decodes, or a fallback reason; `isFast` is the measured routing |
| `NativeReaders` | one native reader per Java searcher reader: built from its `SegmentInfos`, `maxDoc`s and live docs, closed by its close listener |
| `NativeBridge`, `NativeLibrary` | the FFM downcalls (per-thread native scratch memory for arguments and results) and the loader/handshake |
| `src/test/…/NativeSelfTest` | the native path against Lucene's `IndexSearcher`: NRT readers with in-memory deletes, refreshes, merges, every Java-written fixture, the bridge's error paths |
| `src/test/…/NativeBench` | `gradle nativeBench`: the downcall's crossing cost and in-process latency |
| `src/yamlRestTest/…/LuceneRustYamlIT` | OpenSearch's REST YAML suites, for `verify-opensearch.sh --yaml` |
| `e2e/verify_opensearch.py` | the node-level harness `scripts/verify-opensearch.sh` runs |
| `docker/Dockerfile` | the test node: the pinned image, bundled plugins removed, this one installed |

The Rust side is plain C ABI, unit-tested without a JVM:
`crates/lucene-ffi/src/jvm_reader.rs` and `engine_writer.rs`, which the
downcalls reach directly, and `crates/lucene-ffi/src/ffm_bridge.rs` for what
they lack (results handed back as Rust-allocated buffers, `docFreq`, the
last-error slot).
