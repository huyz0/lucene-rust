# Defaults of the plugin's own settings

A review of every setting, mapping option and codec option the plugin adds on
top of stock OpenSearch, and the default each should have. The order of
priorities:

1. **Portable to stock.** Data stays readable by stock OpenSearch and Lucene,
   and behaviour matches stock, unless deviating buys something real.
2. **Faster.**
3. **Less memory, IO and disk.**

Features that are opt-in by design (IVF, disk HNSW, realtime search, the
aggregation segment cache, partitioning) stay opt-in, and nothing is silently
remapped to one of them.

## Changed

| Setting | Was | Now | Evidence | Risk |
|---|---|---|---|---|
| `index.lucene_rust.stored_fields.merge_lz4_high_compression` | `false` | `true` | Same LZ4 block format, so stock reads it. Engine bench below: on 2M log events merged stored fields -28% (68.0 to 49.1 MB), index -16% (110 to 92 MB), for +2.7 CPU-s per 2M docs (+11% of the Rust writer's own CPU, about 1.4% of a node's indexing CPU at OpenSearch's ~100 CPU-s per 1M docs). On text (europarl) stored fields -9%, index -7%. Fetches: logs 5.9 to 4.1 us per random doc, scans 1.5x faster; text equal. | More merge CPU on text-heavy documents: the first merge of a flushed segment compresses at ~45 MB/s instead of ~130. Later merges copy the chunks as they are. `false` restores stock's bytes. |
| `index.lucene_rust.search.native_approximate` | `false` | `true` | A sorted `match_all` or range OpenSearch answers with its approximation: 17.6 to 3.3 ms through REST on 6.45M lines, hits identical to stock on all 18 shapes measured ([newest-n.md](newest-n.md)). | Where the last hit's sort value is tied across a BKD leaf, the documents kept among the tied ones can differ from OpenSearch's approximate pick. They are the ones OpenSearch returns with `track_total_hits: true`. `false` keeps OpenSearch's answer. The e2e suite checks the default against that exact answer. |

## Kept, and why

### Index settings

| Setting | Default | Why it stays |
|---|---|---|
| `index.lucene_rust.engine` | node default | Opt-in per index; see the node default below. |
| `index.lucene_rust.search.enabled`, `fetch.enabled`, `highlight.enabled` | `true` | Same answers as stock (e2e matrices), about 2x faster on heavy shapes. |
| `index.lucene_rust.search.native_shapes` | `fast` | `fast` and `all` behave the same today. |
| `index.lucene_rust.search.aggregation_segment_cache` | `false` | Opt-in feature. |
| `index.lucene_rust.realtime_search`, `realtime_max_staleness` | `false`, `20ms` | Opt-in feature. |
| `index.lucene_rust.adaptive_doc_values` | `false` | Own format (`RustAdaptive`): stock cannot read it. Scans 1.2-2x slower. |
| `index.lucene_rust.zstd_terms` | `false` | Own format: stock cannot read it. Wildcard walks up to 2x slower, merges 2x slower. |
| `index.lucene_rust.stored_fields.paged_zstd` | `false` | Own format: stock cannot read it. |
| `index.lucene_rust.store_mode` | `mmap` | Stock's IO. `page_cache` wins only under memory pressure and costs 5-30% on cold first touch. |
| `index.lucene_rust.bp_reorder.*` | `enabled: false` | Readable by stock, but forced merges take 3x as long and ascending sorts on time-ordered data regress; text queries gain 1.13x, size -6%. |
| `index.lucene_rust.seq_no.index_options`, `disable_sequence_numbers` | `points`, `false` | -3.3%/-5.3% size, but a stock node reading the index has no `_seq_no` points. |
| `index.lucene_rust.partition_field`, `knn.ivf_partition_clusters` | unset, `0` | Opt-in feature. |
| `index.lucene_rust.knn.target_recall` (+ sample rate, check budget) | `0` (off) | Opt-in feature; stock k-NN has no recall target. |
| `index.store.rust_encrypted.*` | `chunk_size` 64 KiB, `data_key_ttl` 1 h, `key_cache_ttl` 5 min | New feature. 64 KiB measured best for encrypted pages (128 KiB worse). |

### Node settings

| Setting | Default | Why it stays |
|---|---|---|
| `lucene_rust.engine.default` | `false` | Put to Huy: turning it on routes new indices to the Rust engine where it can. Recommended on; a separate change. |
| `lucene_rust.engine.node_enabled` | `true` | Rollback switch. |
| `lucene_rust.fetch.reader_wrapper` | `true` (off with the security plugin) | Needed for native stored-field reads. |
| `lucene_rust.store.page_cache.size` | 1 GB | A cap, filled only by `page_cache` and encrypted shards. |
| `lucene_rust.store.mmap.read_advice`, `warm_random` | per-file table, `auto` for `.fdt`; `true` | Measured in the stored-fields read-advice and storage IO work. |
| `lucene_rust.search.aggregation_segment_cache.size` | 256 MB | A cap, used only by indices that opt in. |
| `lucene_rust.logs_preset.templates.enabled` | `true` | Registers component templates only; an index uses them only through `composed_of`. |

### Mapping and codec options

| Option | Default | Why it stays |
|---|---|---|
| k-NN `lucene` engine: `m`, `ef_construction`, `ef_search`, encoder, mode, compression | stock k-NN's (16, 100, 100, ...) | Stock-compatible: same as k-NN. |
| k-NN `extended` engine `ivf`: `target_cluster_size`, `quantizer`, `rerank_encoding` | 384, `bbq`, `half_float` | New feature, measured in the IVF work. `auto` picks better per field but builds 1.2-1.6x slower. |
| k-NN `extended` engine `hnsw`: `quantizer` | `erq2` | Best recall per byte at 2 bits on GloVe and fastText. |
| `hex_id`, `rust_wildcard`, `pattern_text` | `ignore_above`, `null_value`, `normalizer` as `keyword`/`wildcard` | Same meaning as the stock types they stand in for. |

### Stock-named settings

The plugin's copies of stock k-NN (`index.knn.*`, `knn.*`), neural search and
ml-commons settings keep stock's defaults. The k-NN ones were compared with
k-NN's `KNNSettings.java` (3.3 and main): every default matches.

## Bench: high-compression LZ4 at merge

Two levels, both 4 vCPU, 15 GB:

1. Codec: `cargo run --release -p lucene-codecs --example stored_fields_compression`
   over `scripts/zstd/make-corpus.py`'s corpora (400k log events, europarl).
   Search depth 4 to 256 links gives the same size and speed, so the match
   finder's cost is not in the chain walk.

   | corpus | mode | .fdt MiB | write MB/s | random doc us | scan MB/s |
   |---|---|---:|---:|---:|---:|
   | logs (65 MiB) | lz4 | 12.34 | 385 | 5.89 | 1985 |
   | logs | lz4 hc | 9.08 | 77 | 4.08 | 2997 |
   | europarl (19 MiB) | lz4 | 14.49 | 132 | 7.11 | 1940 |
   | europarl | lz4 hc | 13.21 | 44 | 6.75 | 1987 |

2. Engine: `partition_cost index CORPUS DIR plain 3 1000` with
   `STORE_SOURCE=1` (and `HC=1`): the plugin's concurrent writer with a
   256 MB buffer, tiered merges, a refresh every second and three indexing
   threads, then `partition_cost forcemerge DIR plain 1`. CPU is the
   process's user + system time. Logs ran twice each (second shown; the
   first agreed within 0.5 s), europarl once.

   | corpus | HC | index CPU s | .fdt MB after indexing | force merge s | .fdt MB after | index MB after |
   |---|---|---:|---:|---:|---:|---:|
   | logs, 2M events (340 MB) | off | 25.2 | 68.0 | 2.7 | 68.0 | 110 |
   | logs | on | 27.7 | 52.4 | 3.4 | 49.1 | 92 |
   | europarl x4, 70k docs (82 MB) | off | 7.0 | 61.0 | 0.9 | 61.0 | 90 |
   | europarl x4 | on | 6.8 | 61.1 | 2.9 | 55.6 | 84 |
