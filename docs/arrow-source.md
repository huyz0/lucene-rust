# The index as an Arrow source

An index can be read by pyarrow, Polars, DuckDB and Spark without going through
OpenSearch: the library hands them Arrow record batches over the Arrow C stream
interface, so a batch crosses from the index into the tool without a copy or a
serialisation step. Column pruning, comparisons and a row limit run inside the
index scan, over the doc values, so the tool only decodes what it asked for.

The table is declared once (an index does not say which of its fields are keys,
keywords or text) and is read as of the commit it was opened at.

| Column kind | Arrow type | How it is read |
|---|---|---|
| `long` | `Int64` | doc values, decoded straight into the Arrow buffer |
| `keyword` | `Dictionary(Int32, Utf8)`, or `Utf8` | sorted doc values: one dictionary for the scan and a key per row |
| `key` | `Utf8` | stored field (one stored-fields read per row; slow for a full scan) |
| `text`, `vector` | not readable | searched by `match_text` and `knn` in SQL |

## From Python

```python
import lucene_arrow as la

t = la.Table("/data/index", "logs", {"id": "key", "ts": "long", "status": "long",
                                     "host": "keyword", "level": "keyword"})

reader = t.scan(columns=["ts", "host"],
                filters=[("status", ">=", 500), ("host", "in", ["web-1", "web-2"])],
                limit=1000)
table = reader.read_all()                      # a pyarrow.Table

import duckdb, polars as pl
duckdb.sql("SELECT host, count(*) FROM reader GROUP BY host")   # the stream, as is
lf = t.polars()                                # a LazyFrame; filters and projections reach the scan
lf.filter((pl.col("status") == 500) & (pl.col("ts") > 1_700_000_000_000)).select("host").collect()
rows = t.sql("SELECT level, count(*) FROM logs GROUP BY level").read_all()   # read-only SQL
```

Filters are `=`, `!=`, `<`, `<=`, `>`, `>=`, `in` and `is_null` over `long` and
`keyword` columns, and `=`/`in` over a `key` column (a term lookup). A `NULL`
never satisfies a comparison, as in SQL. Whatever the scan does not take (an
`OR`, a function) the consumer applies to the batches it gets; the Polars
adapter does this itself.

`dictionary=False` returns keywords as plain strings. `threads` is how many
threads decode one stream (default: one per core, at most 8; batches still
arrive in document order). `partition`/`partitions` read a share of the
segments, for a consumer that reads several streams at once.

The library is `liblucene_ffi` built with `--features arrow-sql` (or `arrow`
without SQL); `lucene_arrow.py` (in `crates/lucene-ffi/python`) finds it through
`LUCENE_ARROW_LIB`. The entry points are `ffi_arrow_table_open`,
`ffi_arrow_scan` and `ffi_arrow_sql`; the scan is described by a JSON request,
documented in `crates/lucene-ffi/src/arrow_source.rs`.

## From Spark

`connectors/spark` is a DataSource V2 source (Spark 4.0, Java 17+) that reads
the same streams through Arrow Java's C data interface:

```python
df = (spark.read.format("lucene").option("path", "/data/index").option("table", "logs")
      .option("columns", "id:key,ts:long,status:long,host:keyword").load())
df.filter("status = 500 and host in ('web-1','web-2')").groupBy("host").count()
```

Required columns, `=`/`<`/... / `IN` / `IS [NOT] NULL` filters and `LIMIT` are
pushed into the scan; the other filters stay in Spark. Each Spark partition
reads a group of segments. The native library is `liblucene_ffi` built with
`--features spark`, found through `-Dlucene.native.library` or
`LUCENE_NATIVE_LIBRARY`.

## How fast

Ten million log rows, the same data as one Parquet file (zstd, 1M-row groups,
136 MB; the index is 216 MB), 4 cores, warm cache. Parquet → index:

| Question | pyarrow | Polars | DuckDB |
|---|---|---|---|
| 2 long columns | 66 → 41 ms | 55 → 36 ms | 97 → 61 ms |
| 2 keyword columns | 118 → 25 ms | 24 → 44 ms | 125 → 142 ms |
| 7 columns | 314 → 118 ms | 135 → 132 ms | 416 → 317 ms |
| `status = 500 and bytes > 90000` | 211 → 18 ms | 84 → 22 ms | 109 → 25 ms |
| `host = 'host-0042'` (1 in 1000) | 168 → 13 ms | 73 → 18 ms | 95 → 18 ms |
| `LIMIT 100` | 89 → 2 ms | 3.0 → 2.5 ms | 9.6 → 2.7 ms |

Filtered reads are 4 to 13 times faster than Parquet's, because the comparison
runs on doc values and a window no row survives is not decoded for the other
columns. Plain column scans are level or better except for keyword columns in
Polars and DuckDB, where their Parquet readers hand back dictionary pages
directly. An index written with adaptive doc values has zone statistics, and a
range on a sorted column skips whole windows. Method and the Spark numbers:
[Arrow source](benchmarks/arrow-source.md).

## Over the network: Flight SQL (design)

The library form is in-process. For a server with no JVM in front of it, the
same `ScanReader` and the SQL engine's stream are what a Flight SQL service
(`arrow-flight`, `tonic`) serves: `GetFlightInfo` plans a statement and returns
one endpoint per segment group, `DoGet` encodes that group's batches as Arrow
IPC. A Flight round trip of the same scan over loopback runs at 1.6 to 2.0 GB/s
for one stream, so the transport, not the scan, bounds a remote read. It is not
built: it adds an async gRPC stack and a port to secure, which the library form
does not need, and it should be built with the serverless engine's own
authentication and TLS rather than before them.
