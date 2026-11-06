# opensearch-plugin

The lucene-rust plugin for **OpenSearch 3.9.0** (Lucene 10.5.1). Through
`liblucene_ffi.so`, called with Java's Foreign Function & Memory API (JDK 25,
the JDK OpenSearch 3.9.0 bundles), it runs in Rust the query phase of supported
searches, their aggregations, stored-field reads in the fetch phase, and, with
the Rust engine, indexing and merging. Anything it cannot answer natively runs
on OpenSearch and Lucene as usual.
What runs native: [`docs/opensearch-native-queries.md`](../docs/opensearch-native-queries.md);
the engine: [`docs/opensearch-engine.md`](../docs/opensearch-engine.md).

## Build

```
scripts/opensearch-dist.sh                 # once: lib/ of the pinned image -> target/opensearch-dist/
gradle -p opensearch-plugin bundlePlugin   # build/distributions/lucene-rust-0.1.0.zip
gradle -p opensearch-plugin check          # JVM-side differential self test
```

The plugin compiles against the jars of the distribution it installs into,
extracted from the `opensearchproject/opensearch:3.9.0` image — a plugin must
match its node exactly, and the build then needs no Maven repository.
`bundlePlugin` runs `cargo build --profile dist -p lucene-ffi` (the release
build with LTO, `target/dist/`) and puts the library under `native/<os>-<arch>/`
in the zip; `-PextraNative=linux-aarch64=PATH` adds one built elsewhere.
`-PcargoProfile=release` builds without LTO, several times faster and about 20%
slower to run (what CI's test jobs use). `-PnativeLib=PATH` uses an already
built library instead, such as the profile-guided one
`scripts/pgo-build.sh` writes to `target/pgo/dist/` after training it on a
workload (search 1.27x faster on the queries it was trained on, untrained code
7-10% slower; see the script).

`gradle -p opensearch-plugin -Pfips bundlePlugin` builds the FIPS variant,
`lucene-rust-fips-0.1.0.zip`. It contains:

- the native library built against AWS-LC's FIPS 140-3 module
  (`scripts/build-fips.sh --ffi`, which needs CMake, Go and Perl);
- the Amazon Corretto Crypto Provider's FIPS build, which does the JVM's
  encryption for the `rust_encrypted` store.

See [`docs/pluggable-storage.md`](../docs/pluggable-storage.md#in-opensearch-the-rust_encrypted-store-type).

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
cannot be installed alongside `neural-search`. It serves neural-search's
learned sparse search itself: the `sparse_vector` field (`seismic` method,
`lucene` engine) on an `index.sparse` index, whose segments the Rust engine
writes in neural-search's format (neural-search opens them unchanged), and the
`neural_sparse` query by `query_tokens`, with `method_parameters` (`top_n`,
`k`, `heap_factor`, `filter`), also on `rank_features` fields. `query_text`
needs neural-search's model inference and is refused, as is the `native`
sparse engine.

## Operate

| | |
|---|---|
| `index.lucene_rust.search.enabled` (index, dynamic, default `true`) | route this index's searches native when eligible |
| `index.lucene_rust.search.native_shapes` (index, dynamic, `fast`/`all`, default `fast`) | `all` also runs shapes that are correct natively but measured slower. Every encodable shape currently measures faster, so the two values agree |
| `index.lucene_rust.fetch.enabled` (index, dynamic, default `true`) | read this index's stored fields (`_source`, `_id`, `stored_fields`) natively in the fetch phase and the get API |
| `lucene_rust.fetch.reader_wrapper` (node, default `true` unless the security plugin is installed) | install the index reader wrapper native stored fields need. An index takes one reader wrapper and the security plugin claims it for field- and document-level security, so with it installed the default is off and stored fields are read by Lucene; if another plugin set a wrapper first, this one logs and stands down |
| `index.lucene_rust.bp_reorder.enabled` (index, dynamic, default `false`) | Lucene's `BPReorderingMergePolicy` around the shard's merge policy: forced merges, and natural merges of at least `bp_reorder.min_merge_docs` (default 65536) documents, renumber documents so ones sharing terms sit together (smaller postings, faster queries). Output segments are ordinary Lucene segments. Tuning: `bp_reorder.fields` (default every indexed field), `min_doc_freq` (4096), `max_doc_freq` (1.0), `min_partition_size` (32), `max_iters` (20), `ram_budget` (256mb), `min_merge_ratio` (0.0). Ignored on sorted and sparse indexes, and a merge of segments holding nested documents keeps their order (Lucene needs a parent field to keep blocks together, and OpenSearch sets none) |
| `GET /_plugins/lucene_rust/stats` | native queries, native errors, fallbacks by reason, open native readers, native stored-fields reads (`native_fetches`, `native_sequential_fetches`) and `StoredFields.document` time per path (`fetch_nanos`) |

### Encryption at rest

`index.store.type: rust_encrypted` encrypts every shard file but commit
metadata (`segments_N`, `.si`), which OpenSearch reads on restart without the
store. Lucene and the
Rust engine both read and write it, and keys come from a KMS through
OpenSearch's `crypto-kms` plugin (BYOK):

```
PUT /tenant-a
{ "settings": {
    "index.store.type": "rust_encrypted",
    "index.store.rust_encrypted.key_provider_type": "aws-kms",
    "index.store.rust_encrypted.key_provider": "kms",
    "index.store.rust_encrypted.kms.key_arn": "arn:aws:kms:us-east-1:111122223333:key/..." } }
```

With no KMS, the provider type `lucene_rust_local` wraps data keys with a key
from the node keystore (`opensearch-keystore add lucene_rust.store.local_key`,
base64 of 32 random bytes). The Rust engine refuses other store types that
change bytes on disk, rather than bypass them. Settings, key rotation and
FIPS:
[`docs/pluggable-storage.md`](../docs/pluggable-storage.md#in-opensearch-the-rust_encrypted-store-type).

A native failure is logged, counted as `native_errors`, and the query is re-run
on Lucene: it never fails a search Lucene can answer.

Highlighters, suggesters and the query-string parsers stay OpenSearch's own:
`NamedRegistry` refuses a plugin that registers a built-in name again, so
they cannot be replaced. The query
phase they ride on runs natively, `query_string` and `simple_query_string`
included (but for the shapes that fall back by name: escaped wildcards,
non-default regexp flags, phrases with position gaps), and so does an index
with `completion` fields (the Completion90-104
postings formats are read natively; the suggesters read Lucene's reader).
Term vectors, which `fvh` and the unified highlighter may read, are Lucene's.
See [`docs/opensearch-native-queries.md`](../docs/opensearch-native-queries.md#suggesters-and-highlighters).

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
| `e2e/verify_upgrade.py` | `scripts/verify-opensearch-upgrade.sh`: an index OpenSearch 2.19 wrote, opened in place and restored from its snapshot by 3.9.0 with this plugin, answered as a stock 3.9.0 node answers it, natively, before and after a force merge |
| `docker/Dockerfile` | the test node: the pinned image, bundled plugins removed, this one installed |

The Rust side is plain C ABI, unit-tested without a JVM:
`crates/lucene-ffi/src/jvm_reader.rs` and `engine_writer.rs`, which the
downcalls reach directly, and `crates/lucene-ffi/src/ffm_bridge.rs` for what
they lack (results handed back as Rust-allocated buffers, `docFreq`, the
last-error slot).
