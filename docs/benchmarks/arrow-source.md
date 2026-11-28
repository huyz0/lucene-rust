# Arrow source: scan throughput against Parquet

What reading an index as an Arrow source costs against reading the same rows
from a Parquet file, in pyarrow, Polars, DuckDB and Spark. The feature is
described in [the index as an Arrow source](../arrow-source.md).

## Setup

- **Data**: 10,000,000 log-like rows from `cargo run --release -p lucene-sql
  --example arrow_gen`: `id` (stored key), `ts` (rising, 0-39 ms apart),
  `status` (9 values), `bytes` (0-100k), `latency`, and keywords `host` (1000
  values), `level` (5), `region` (20), in 8 segments. The Parquet file is
  written from the index's own scan (zstd, 1M-row groups): 136 MB. The index is
  216 MB with the stock doc-values format and 188 MB with the adaptive one.
- **Machine**: 4 vCPUs, 15 GB, warm page cache, one run of each case discarded
  then the median of 7. pyarrow 26.0.1, Polars 2.2.0, DuckDB 2.0.1 (4 threads),
  Spark 4.0.1 `local[4]`.
- **Program**: `benchmarks/arrow-source/bench.py` (the answers of the index and
  Parquet runs are checked equal for every case); Spark:
  `connectors/spark/e2e.py`.
- "index x4": four partitions read by four threads, as one table. A single
  stream already decodes on up to 8 threads, so it matters only for consumers
  that cannot take one stream apart.

## Scans, standard doc values (milliseconds, Parquet → index)

| Case | pyarrow | Polars | DuckDB |
|---|---|---|---|
| 2 long columns | 66 → 41 (x4: 39) | 55 → 36 | 97 → 61 |
| 2 keyword columns | 118 → 25 | 24 → 44 | 125 → 142 |
| 7 columns, no `id` | 314 → 118 (x4: 93) | 135 → 132 | 416 → 317 |
| `status = 500 and bytes > 90000` | 211 → 18 | 84 → 22 | 109 → 25 |
| `host = 'host-0042'` | 168 → 13 | 73 → 18 | 95 → 18 |
| `ts` in the last 1% (sorted) | 9.2 → 6.2 | 4.0 → 6.9 | 8.5 → 10.7 |
| `LIMIT 100` | 89 → 1.9 | 3.0 → 2.5 | 9.6 → 2.7 |

## Scans, adaptive doc values (zone statistics)

| Case | pyarrow | Polars | DuckDB |
|---|---|---|---|
| 2 long columns | 69 → 32 | 51 → 34 | 94 → 63 |
| 2 keyword columns | 119 → 25 | 26 → 43 | 140 → 144 |
| 7 columns, no `id` | 310 → 109 | 132 → 141 | 411 → 341 |
| `status = 500 and bytes > 90000` | 203 → 18 | 85 → 22 | 110 → 24 |
| `host = 'host-0042'` | 170 → 12 | 74 → 16 | 97 → 18 |
| `ts` in the last 1% (sorted) | 8.6 → 2.6 | 3.8 → 6.8 | 8.3 → 4.3 |
| `LIMIT 100` | 84 → 3.6 | 3.5 → 2.8 | 9.9 → 2.9 |

The zone statistics (per 1024 rows) let a range on `ts` skip 99% of the windows
without decoding them. Polars stays slower there (6.8 against 3.8 ms for its
own Parquet reader).

## Spark (`local[4]`, milliseconds, Parquet → index)

| Case | Parquet | Index |
|---|---|---|
| `count(*)` | 134 | 124 |
| `sum(bytes), sum(latency)` | 294 | 156 |
| `group by status` (count, sum) | 359 | 231 |
| `group by host` (keyword) | 362 | 356 |
| `status = 500 and bytes > 90000` (count, sum) | 700 | 92 |
| `host in (two values)` | 161 | 69 |
| `level is not null and ts >= cut` | 103 | 99 |
| `LIMIT 100` | 39 | 28 |

The `host` group-by is level because Spark's Arrow vectors cannot take a
dictionary, so keywords reach it as strings.

## Where the time goes

Single thread, per row: decoding the doc values is 0.4 to 0.8 ns
(`NumericReader::fill_window`); a long column scan is 1.0 ns and a keyword
column 1.9 to 2.3 ns, the rest being the Arrow buffers (a 512 KB buffer per
window is page-faulted in by the kernel) and, for a keyword, the lookup of each
ordinal in the scan's dictionary and the key check Arrow does on a dictionary
array. Four decoding threads take 2 long columns from 23 ms to 13 ms and 7
columns from 106 ms to 45 ms (inside the library; the Python timings above
include creating and dropping the result). Changes that moved these numbers:
reusing the window buffers, gathering keyword keys straight from the decoded
ordinals (two columns, one thread: 103 → 38 ms), one dictionary for the whole
scan so batches share it, 1M-row chunks in the Polars adapter (2 columns 101 →
42 ms), zone pruning of ranges, and shrinking the first window under a `LIMIT`
(8.8 ms to 1.9 ms).

## What loses, and what was tried

- Keyword columns in Polars and DuckDB. Measured on the 10M-row index, 4
  vCPUs, with the pieces timed apart:
  - *The scan is not the gap.* Two keyword columns read into pyarrow take
    24-28 ms (Parquet into pyarrow: 117 ms); DuckDB pulls the same two columns
    through its Arrow scan in 10 ms.
  - *Polars* gets the two columns as dictionary arrays and converts them to
    `Categorical` in `from_arrow`: 15 ms for 20M values (40 ms when each 64K-row
    batch is converted alone, which the adapter's 1M-row chunks avoid). Its
    Parquet reader decodes dictionary pages straight into `String` in 26 ms.
    Other ways to hand Polars the column were slower: plain `Utf8` 67-149 ms,
    `Utf8View` 129 ms (arrow-rs validates every view; skipping that needs
    `unsafe` and was not adopted), an `Enum` cast of the keys 53 ms.
  - *DuckDB* spends the time in its aggregate, not the scan: a group-by on
    both keywords costs 30 ns per row on one thread from the index's
    dictionary arrays against 14 ns from Parquet, and the same 30 ns from a
    pyarrow table already in memory, so it is how DuckDB handles Arrow
    dictionaries and not our stream. 8- and 16-bit keys, plain strings (485 ms
    against 374 ms) and `Utf8View` (408 ms) are no better. Streaming adds about
    10 ms over an in-memory table (41 against 30 ms for one keyword).
  - Skipping arrow-rs's per-batch key check in the dictionary array would take a
    keyword column from 2.3 to 1.5 ns per row inside the library (pyarrow 28 to
    24 ms) and changes nothing in Polars or DuckDB; it needs `unsafe` in a crate
    that forbids it, so it was left out.
  - Where this matters, a filter on the keyword (`host = ...`) or a projection
    that leaves it out still wins: the keyword is answered by term search or
    ordinals, not decoded.
- A column scan of a stored key (`id`) is not in the tables: it is one
  stored-fields read per row.

## Flight over loopback

A Python Flight server returning the index's scan (`RecordBatchStream` over the
C stream), read by a `pyarrow.flight` client on the same machine, one stream:

| Case | In process | Flight | Throughput |
|---|---|---|---|
| 2 long columns (160 MB) | 32 ms | 98 ms | 1.6 GB/s |
| 7 columns (663 MB) | 182 ms | 340 ms | 1.95 GB/s |
| `status = 500 and bytes > 90000` (3 MB) | 18 ms | 22 ms | |
