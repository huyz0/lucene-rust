# opensearch-plugin

The lucene-rust plugin for **OpenSearch 3.9.0** (Lucene 10.5.1). Through
`liblucene_ffi.so`, called with Java's Foreign Function & Memory API (JDK 25,
the JDK OpenSearch 3.9.0 bundles), it runs in Rust the query phase of supported
searches, their aggregations, stored-field reads in the fetch phase, and, with
the accelerated engine, indexing and merging. Anything it cannot answer natively runs
on OpenSearch and Lucene as usual.
What runs native: [`docs/accelerated-queries.md`](../docs/accelerated-queries.md);
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
  encryption for the `encrypted` store.

See [`docs/pluggable-storage.md`](../docs/pluggable-storage.md#in-opensearch-the-encrypted-store-type).

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
cannot be installed alongside `neural-search`. It serves neural-search
itself, over its own ml-commons and k-NN: the `neural` and `neural_knn_query`
queries on `knn_vector` fields; the `text_embedding`, `sparse_encoding`,
`text_image_embedding` and `text_chunking` ingest processors; the
`neural_query_enricher` and `neural_sparse_two_phase_processor` request
processors; the `rerank` response processor (`ml_opensearch` cross-encoder
and `by_field`) with its `ext.rerank` context; neural-search's settings and
`_plugins/_neural/warmup` and `clear_cache` for sparse indexes; the `semantic`
field, whose mapping gains the `knn_vector` or `rank_features` field its model
needs and whose documents are embedded at ingest without a pipeline (chunking,
`model_selection`, `dense_embedding_config`, `sparse_encoding_config`);
k-NN's `mmr` search extension and `knn_default_excludes` system processor
(on with `cluster.search.enabled_system_generated_factories`, as in stock); ml-commons' `bert-uncased` and
`mbert-uncased` analyzers and `hf_model_tokenizer` type; and learned sparse
search: the `sparse_vector` field (`seismic` method, `lucene` engine) on an
`index.sparse` index, whose segments either engine writes in neural-search's
format (neural-search opens them unchanged), and the `neural_sparse` query by
`query_tokens` or by `query_text` (a model, an analyzer, or `bert-uncased` by
default), with `method_parameters` (`top_n`, `k`, `heap_factor`, `filter`),
also on `rank_features` fields. `scripts/check-neural-plugin.py` compares them
with stock. The `native` sparse engine is refused. Model inference runs on
the plugin's own ml-commons, so with `opensearch-ml` installed the model-backed
queries and processors fail with a message saying so.

### `hex_id` fields

`"type": "hex_id"` is a `keyword` for trace, span and request ids. A value of
lowercase hex digits (even length) is indexed and doc-valued as its bytes,
half its characters; any other value (uppercase, odd length, dashes, other
text) is kept as its exact UTF-8. Results are a `keyword`'s on the same data:
`term`, `terms`, `exists` and `case_insensitive` `term` match the same
documents, and sort values, aggregation keys, `docvalue_fields`, `fields`,
scripts and derived `_source` give back the original strings. The
differences: non-canonical values sort after the canonical ones, so on mixed
data a document's values, a sort, a terms aggregation's `_key` order and
count ties, and composite paging order differently; `prefix`, `wildcard`,
`regexp`, `fuzzy` and `range` queries are refused, as on an `ip`; whatever
reads the index terms as text sees the bytes (a terms aggregation's regex
`include`/`exclude`, highlighting, the `term` suggester, `more_like_this`);
`cardinality` hashes the bytes, so its estimate can differ by a unit; and a
non-hex value may be 2 bytes shorter than Lucene's term limit. Parameters:
`index`, `doc_values`, `null_value`, `ignore_above`, `meta`. The segments are plain Lucene terms and `SORTED_SET` doc values. 5M
random 32-character trace ids on the accelerated engine, one segment: 101 MiB less
than `keyword` (shard 506 → 405 MiB: terms 144 → 110, doc values 163 → 96),
with `term`, `terms` and sorted lookups within 2% of `keyword`'s latency
(`e2e/bench_hex_id.py`; `e2e/hex_id_matrix.py` is the parity check).

### `compact_wildcard` fields

`"type": "compact_wildcard"` takes the queries of OpenSearch's `wildcard` field
(`term`, `terms`, `prefix`, `wildcard`, `regexp`, `match`, `exists`,
`query_string` wildcards, `case_insensitive` where stock takes it) and matches
the documents whose value stock's own test accepts. On the accelerated engine the
value is stored as `Substring90` doc values: LZ4 blocks of 32 values with a
trigram index over the blocks, kept through every flush and merge, and every
query runs natively but a `regexp` with `flags` other than `ALL`. Under
OpenSearch's engine the field is plain `BINARY` doc values, read and tested
by the same queries. Parameters: `null_value`,
`ignore_above`, `normalizer`, `meta`. The differences from `wildcard`: one
value per document (an array is refused); no sorting, aggregations or
scripts on it (`fields` and `_source` give the value back); `fuzzy` and
`range` refused, as on `wildcard`. Where stock's trigram index answers
differently from its own test, this field follows the test: a
`case_insensitive` pattern with non-ASCII letters (`*Ü*`, the Kelvin sign)
finds the documents stock misses, `terms` with `""` matches the empty
values, and `exists`, `*` and `.*` report a total that counts the empty
values, which stock's hits include but its total leaves out.

1M log lines (Loghub samples, digits varied), one shard, one segment, the
plugin's non-LTO release library on 4 cores: shard 256 MiB vs 328 for
`wildcard` (both with `_source`), bulk indexing 24.5 s vs 40.5 s (51.0 s for
`wildcard` on the accelerated engine). Median latency, `wildcard` → `compact_wildcard`:
`*error*` (57K hits) 606 → 89 ms, case-insensitive 752 → 92 ms, `*ab*` (142K)
1,737 → 97 ms, a regexp (54K) 698 → 103 ms, `*10.1*` 40 → 22 ms, `term` 2.6 →
2.0 ms; slower: `*session*root*` (1.4K hits from common grams) 27 → 57 ms and
`prefix` 2.9 → 4.3 ms, where every candidate block is decompressed
(`e2e/bench_compact_wildcard.py`; `e2e/compact_wildcard_matrix.py` is the parity
check against `wildcard` under both engines).

## Operate

| | |
|---|---|
| `index.search.accelerated` (index, dynamic, default `true`) | route this index's searches native when eligible |
| `index.search.accelerated_shapes` (index, dynamic, `fast`/`all`, default `fast`) | `all` also runs shapes that are correct natively but measured slower. Every encodable shape currently measures faster, so the two values agree |
| `index.fetch.accelerated` (index, dynamic, default `true`) | read this index's stored fields (`_source`, `_id`, `stored_fields`) natively in the fetch phase and the get API |
| `indices.fetch.accelerated_reader` (node, default `true` unless the security plugin is installed) | install the index reader wrapper native stored fields need. An index takes one reader wrapper and the security plugin claims it for field- and document-level security, so with it installed the default is off and stored fields are read by Lucene; if another plugin set a wrapper first, this one logs and stands down |
| `index.merge.reorder.enabled` (index, dynamic, default `false`) | Lucene's `BPReorderingMergePolicy` around the shard's merge policy: forced merges, and natural merges of at least `index.merge.reorder.min_merge_docs` (default 65536) documents, renumber documents so ones sharing terms sit together (smaller postings, faster queries). Output segments are ordinary Lucene segments. Tuning: `index.merge.reorder.fields` (default every indexed field), `min_doc_freq` (4096), `max_doc_freq` (1.0), `min_partition_size` (32), `max_iters` (20), `ram_budget` (256mb), `min_merge_ratio` (0.0). Ignored on sorted and sparse indexes, and a merge of segments holding nested documents keeps their order (Lucene needs a parent field to keep blocks together, and OpenSearch sets none) |
| `logs_preset@settings`, `logs_preset@mappings` (component templates) | the logs preset, opt-in per index template: see [`docs/logs-preset.md`](../docs/logs-preset.md). `indices.logs_preset.enabled` (node, default `true`) registers them |
| `GET /_plugins/lucene_rust/stats` | native queries, native errors, fallbacks by reason, open native readers, native stored-fields reads (`accelerated_fetches`, `accelerated_sequential_fetches`) and `StoredFields.document` time per path (`fetch_nanos`) |

### Encryption at rest

`index.store.type: encrypted` encrypts every shard file but commit
metadata (`segments_N`, `.si`), which OpenSearch reads on restart without the
store. Lucene and the
accelerated engine both read and write it, and keys come from a KMS through
OpenSearch's `crypto-kms` plugin (BYOK):

```
PUT /tenant-a
{ "settings": {
    "index.store.type": "encrypted",
    "index.store.encrypted.key_provider_type": "aws-kms",
    "index.store.encrypted.key_provider": "kms",
    "index.store.encrypted.kms.key_arn": "arn:aws:kms:us-east-1:111122223333:key/..." } }
```

With no KMS, the provider type `local_keystore` wraps data keys with a key
from the node keystore (`opensearch-keystore add indices.store.encrypted.local_key`,
base64 of 32 random bytes). The accelerated engine refuses other store types that
change bytes on disk, rather than bypass them. Settings, key rotation and
FIPS:
[`docs/pluggable-storage.md`](../docs/pluggable-storage.md#in-opensearch-the-encrypted-store-type).

A native failure is logged, counted as `accelerated_errors`, and the query is re-run
on Lucene: it never fails a search Lucene can answer.

The `unified` highlighter (named, or the default when a field names no
`type`) runs natively: `NamedRegistry` refuses a plugin that registers a
built-in name again, so the plugin registers it as `accelerated_unified` and points
each search's `unified` fields at it before the fetch phase. Its snippets are
stock's, on every offset source (postings, term vectors, re-analysis with the
field's analyzer built natively). `index.highlight.accelerated`
(dynamic, default `true`) turns it off per index. A field whose query or
analyzer cannot be sent falls back to OpenSearch's highlighter under a named
reason (`highlight_<Class>`, `highlight_function_score_phrase`, an analyzer
reason). `plain` and `fvh`, the suggesters and the query-string parsers stay
OpenSearch's own; the query phase they ride on runs natively, `query_string`
and `simple_query_string` included (but for the shapes that fall back by
name: escaped wildcards, non-default regexp flags, phrases with position
gaps), and so does an index with `completion` fields (the Completion90-104
postings formats are read natively; the suggesters read Lucene's reader).
Term vectors, which `fvh` reads, are Lucene's.
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
