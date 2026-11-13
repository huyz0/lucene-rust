# Leaner sequence numbers on a logs index

What `index.seq_no.index_options: doc_values_only` and
`index.disable_sequence_numbers: true` (see
[opensearch-engine.md](../opensearch-engine.md#leaner-sequence-numbers)) save
on a logs index, and what they cost.

## Setup

- **Corpus**: 6,454,808 log lines, the first 250 MB of each of four Loghub sets
  (BGL, HDFS, OpenSSH, Thunderbird), interleaved with a fixed seed. Each document
  has `@timestamp` (date), `line` (long, its position), `source` and `host`
  (keyword) and `message` (text).
- **Node**: OpenSearch 3.9.0 in Docker, single node, 4 GB heap, 1 shard, 0
  replicas, `refresh_interval: 5s`. The host is 4 vCPUs (Intel Xeon @ 2.10 GHz)
  with 15 GB of RAM.
- **Run**: four bulk clients load the corpus in 5k-document batches. Then the
  harness refreshes, flushes, waits 35 s for the retention lease to renew, force
  merges to one segment, and times ten searches (3 warm-up and 20 timed rounds,
  request cache off). Sizes are the files in the shard's index directory after
  the merge.
- **Configs**:
  - `stock`: stock OpenSearch 3.9.0 with the bundled plugins removed.
  - `plugin-default`: the native engine with stock `_seq_no`.
  - `plugin-lean`: the native engine with `doc_values_only`.
  - `plugin-lean-disable`: `plugin-lean` plus `disable_sequence_numbers`.
- **Harness**: [`opensearch-plugin/e2e/lean_seq_no_bench/`](../../opensearch-plugin/e2e/lean_seq_no_bench/)
  (`gen.py` builds the corpus, `run.sh` runs all four configs, and the
  `res-*.json` files hold the results quoted here). One pass takes about 14
  minutes.

## Size after force merge (MiB)

| file | stock | plugin-default | plugin-lean | plugin-lean-disable |
|---|---:|---:|---:|---:|
| points (`.kdd` + `.kdi`) | 75.3 | 46.5 | 22.4 | 22.3 |
| doc values (`.dvd`) | 63.4 | 64.1 | 63.8 | 49.8 |
| stored fields (`.fdt`) | 443.2 | 380.1 | 380.2 | 380.3 |
| postings, terms, rest | 236.9 | 228.1 | 228.6 | 228.4 |
| **index total** | **818.8** | **718.8** | **695.0** | **680.8** |
| vs plugin-default | | | **-23.8 (-3.3%)** | **-38.0 (-5.3%)** |

- `doc_values_only` removes the `_seq_no` points, which were 24 MiB, more than
  half of all the points in this index. `@timestamp` and `line` keep theirs.
- `disable_sequence_numbers` also lets the merge write constant sentinels for
  `_seq_no` and `_primary_term` once their history is past retention. Both doc
  values columns then compress to almost nothing, saving another 14 MiB.
- Together the two settings save 38 MiB, 5.3% of the plugin index and 4.6% of
  stock.

## Indexing and merge

| | stock | plugin-default | plugin-lean | plugin-lean-disable |
|---|---:|---:|---:|---:|
| docs/s | 94,416 | 89,210 | 87,591 | 88,840 |
| CPU µs/doc | 31.37 | 37.12 | 37.80 | 37.44 |
| force merge (s) | 14.9 | 12.1 | 12.5 | 11.6 |
| merge CPU (s) | 14.9 | 17.1 | 17.1 | 15.5 |

An earlier full pass gave plugin-default 85,239 docs/s, plugin-lean 86,361 and
plugin-lean-disable 89,998. The order of the three plugin configs flipped
between passes, so indexing throughput is the same within run-to-run noise
(about ±3%). Merge time is also unchanged, and the sentinel config merges
slightly less data. The gap to stock is the plugin's existing indexing cost and
does not come from these settings.

## Search latency (median / p90 `took`, ms)

| query | stock | plugin-default | plugin-lean | plugin-lean-disable |
|---|---:|---:|---:|---:|
| `match message:error` top 10 | 3 / 3 | 2 / 4 | 2 / 2 | 2 / 3 |
| source + last 24 h filter | 3 / 6 | 2 / 4 | 2 / 3 | 2 / 3 |
| phrase "connection closed" | 2 / 5 | 3 / 5 | 3 / 5 | 3 / 3 |
| newest 100 by `@timestamp` | 5 / 7 | 7 / 11 | 6 / 9 | 7 / 9 |
| 7-day hourly date histogram | 4 / 6 | 4 / 6 | 4 / 5 | 3 / 4 |
| top 10 hosts for "failed" | 12 / 15 | 8.5 / 12 | 9 / 10 | 7 / 11 |
| `range` on `_seq_no` (100k hits) | 1 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| `term` on `_seq_no` | 1 / 1 | 1 / 1 | 2 / 3 | 1 / 1 |
| `terms` on `_seq_no` (4 values) | 1 / 1 | 1 / 2 | 1 / 2 | 1 / 64 |
| top 100 with `seq_no_primary_term` | 8 / 10 | 15.5 / 17 | 11.5 / 13 | 10 / 12 |

`took` has a 1 ms resolution, so differences of 1 ms are noise. Searches on
`_seq_no` run on the doc values and their skip index, and stay at about 1 ms.

## Results match stock

Each query's hits (for scored queries, sorted by score then `line`, with
scores to 4 decimals), totals and aggregations are hashed. Every query gives
the same hash in all four configs, with one expected exception. Under
`disable_sequence_numbers` every operation in this run was past retention when
the merge ran, so the merged segment holds the sentinel for every document:
the `_seq_no` searches match nothing, and `seq_no_primary_term` returns no
`_seq_no` for any of the 100 hits, against 100 in the other configs. That is
the documented behavior of the setting. Which document holds which `_seq_no`
depends on how the four bulk clients interleave, so `_seq_no` searches are
compared on their hit counts.

[`opensearch-plugin/e2e/seq_no_matrix.py`](../../opensearch-plugin/e2e/seq_no_matrix.py)
checks the same against stock on a small index. It covers `_seq_no` range,
term and terms searches, fetch and GET of `_seq_no`, nested documents, a
reopen, and `if_seq_no` writes.
