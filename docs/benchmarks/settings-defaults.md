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
| `index.codec.stored_fields.recompress_on_merge` | `false` | `true` | Same LZ4 block format, so stock reads it. Engine bench below: on 2M log events merged stored fields -28% (68.0 to 49.1 MB), index -16% (110 to 92 MB), for +2.7 CPU-s per 2M docs (+11% of the accelerated writer's own CPU, about 1.4% of a node's indexing CPU at OpenSearch's ~100 CPU-s per 1M docs). On text (europarl) stored fields -9%, index -7%. Fetches: logs 5.9 to 4.1 us per random doc, scans 1.5x faster; text equal. | More merge CPU on text-heavy documents: the first merge of a flushed segment compresses at ~45 MB/s instead of ~130. Later merges copy the chunks as they are. `false` restores stock's bytes. |
| `index.search.approximation.exact` | `false` | `true` | A sorted `match_all` or range OpenSearch answers with its approximation: 17.6 to 3.3 ms through REST on 6.45M lines, hits identical to stock on all 18 shapes measured ([newest-n.md](newest-n.md)). | Where the last hit's sort value is tied across a BKD leaf, the documents kept among the tied ones can differ from OpenSearch's approximate pick. They are the ones OpenSearch returns with `track_total_hits: true`. `false` keeps OpenSearch's answer. The e2e suite checks the default against that exact answer. |

## Kept, and why

### Index settings

| Setting | Default | Why it stays |
|---|---|---|
| `index.engine.accelerated` | node default | Opt-in per index; see the node default below. |
| `index.search.accelerated`, `index.fetch.accelerated`, `index.highlight.accelerated` | `true` | Same answers as stock (e2e matrices), about 2x faster on heavy shapes. |
| `index.search.accelerated_shapes` | `fast` | `fast` and `all` behave the same today. |
| `index.search.aggregation_segment_cache.enabled` | `false` | Opt-in feature. |
| `index.search.realtime.enabled`, `max_staleness` | `false`, `20ms` | Opt-in feature. |
| `index.codec.adaptive_doc_values` | `indices.codec.adaptive_doc_values.default` (`auto`) | Own format (`Adaptive90`): stock cannot read it. `auto` writes it only for a shard whose first 64K documents have a column clustered in document order (aggregations 0.54-0.61x of stock end to end, store 27-38% smaller) and keeps stock doc values otherwise, where `true` makes filtered metrics up to 1.5x slower. The cluster setting `false` keeps new indices stock-readable. |
| `index.codec.terms_compression: zstd` | `false` | Own format: stock cannot read it. Wildcard walks up to 2x slower, merges 2x slower. |
| `index.codec.stored_fields.page_aligned` | `false` | Own format: stock cannot read it. |
| `index.store.read_mode` | `mmap` | Stock's IO. `page_cache` wins only under memory pressure and costs 5-30% on cold first touch. |
| `index.merge.reorder.*` | `enabled: false` | Readable by stock, but forced merges take 3x as long and ascending sorts on time-ordered data regress; text queries gain 1.13x, size -6%. |
| `index.seq_no.index_options`, `disable_sequence_numbers` | `points`, `false` | -3.3%/-5.3% size, but OpenSearch's own engine refuses such an index (it runs history ranges on `_seq_no` points), though the segments stay Lucene-readable. |
| `index.segment_partition.field`, `knn.ivf_partition_clusters` | unset, `0` | Opt-in feature. |
| `index.knn.target_recall` (+ sample rate, check budget) | `0` (off) | Opt-in feature; stock k-NN has no recall target. |
| `index.store.encrypted.*` | `chunk_size` 64 KiB, `data_key_ttl` 1 h, `key_cache_ttl` 5 min | New feature. 64 KiB measured best for encrypted pages (128 KiB worse). |

### Node settings

| Setting | Default | Why it stays |
|---|---|---|
| `indices.engine.accelerated.default` | `false` here | Turned on in its own change: every index without its own setting goes to the accelerated engine where it can, storing no setting, so stock still opens it. |
| `indices.engine.accelerated.node_enabled` | `true` | Rollback switch. |
| `indices.fetch.accelerated_reader` | `true` (off with the security plugin) | Needed for native stored-field reads. |
| `indices.store.page_cache.size` | 1 GB | A cap, filled only by `page_cache` and encrypted shards. |
| `indices.store.mmap.read_advice`, `warm_random` | per-file table, `auto` for `.fdt`; `true` | Measured in the stored-fields read-advice and storage IO work. |
| `indices.aggregation_segment_cache.size` | 256 MB | A cap, used only by indices that opt in. |
| `indices.logs_preset.enabled` | `true` | Registers component templates only; an index uses them only through `composed_of`. |

### Mapping and codec options

| Option | Default | Why it stays |
|---|---|---|
| k-NN `lucene` engine: `m`, `ef_construction`, `ef_search`, encoder, mode, compression | stock k-NN's (16, 100, 100, ...) | Stock-compatible: same as k-NN. |
| k-NN `extended` engine `ivf`: `target_cluster_size`, `quantizer`, `rerank_encoding` | 384, `bbq`, `half_float` | New feature, measured in the IVF work. `auto` picks better per field but builds 1.2-1.6x slower. |
| k-NN `extended` engine `hnsw`: `quantizer` | `erq2` | Best recall per byte at 2 bits on GloVe and fastText. |
| `hex_id`, `compact_wildcard`, `pattern_text` | `ignore_above`, `null_value`, `normalizer` as `keyword`/`wildcard` | Same meaning as the stock types they stand in for. |

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
