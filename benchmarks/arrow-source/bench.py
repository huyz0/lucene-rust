"""Scan throughput of the index as an Arrow source against Parquet.

    python3 bench.py <index-dir> <work-dir> [reps]

Writes the table once to `<work-dir>/t.parquet` (row groups of 1M rows, zstd),
read back through the same engines, then times the same questions three ways
in pyarrow, Polars and DuckDB: from Parquet, and from the index through
`lucene_arrow` (single stream, and four partitions read side by side).
Medians of `reps` warm runs; the answer of each is checked against Parquet's.
"""
import json
import os
import statistics
import sys
import time

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../crates/lucene-ffi/python"))
import lucene_arrow as la  # noqa: E402

COLS = {"id": "key", "ts": "long", "status": "long", "bytes": "long", "latency": "long",
        "host": "keyword", "level": "keyword", "region": "keyword"}
THREADS = 4


def med(f, reps):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        r = f()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), r


def parallel_read(t, nparts, **kw):
    """The partitions of a scan read by threads, as one table."""
    from concurrent.futures import ThreadPoolExecutor
    n = t.partition_count(nparts)

    def one(i):
        return t.scan(partition=i, partitions=nparts, **kw).read_all()

    with ThreadPoolExecutor(n) as ex:
        return pa.concat_tables(list(ex.map(one, range(n))))


def main():
    index, work = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    os.makedirs(work, exist_ok=True)
    t = la.Table(index, "logs", COLS)
    print("index:", t.info, flush=True)
    pqf = os.path.join(work, "t.parquet")
    if not os.path.exists(pqf):
        t0 = time.time()
        w = None
        for b in t.scan(dictionary=False, columns=list(COLS)):
            if w is None:
                w = pq.ParquetWriter(pqf, b.schema, compression="zstd")
            w.write_batch(b, row_group_size=1 << 20)
        w.close()
        print(f"wrote parquet in {time.time() - t0:.0f}s", flush=True)
    idx_bytes = sum(os.path.getsize(os.path.join(index, f)) for f in os.listdir(index))
    print(f"parquet {os.path.getsize(pqf) / 1e6:.0f} MB; index {idx_bytes / 1e6:.0f} MB", flush=True)
    pl.thread_pool_size()
    duck = duckdb.connect()
    duck.execute(f"SET threads={THREADS}")
    pq_path = pqf.replace("'", "''")

    out = []

    def row(case, engine, source, secs, rows=None):
        out.append((case, engine, source, secs))
        print(f"{case:<34} {engine:<8} {source:<12} {secs * 1000:9.1f} ms", flush=True)

    cases = [
        ("2 long cols", ["bytes", "latency"], [], None),
        ("2 keyword cols", ["host", "level"], [], None),
        ("7 cols (no id)", ["ts", "status", "bytes", "latency", "host", "level", "region"], [], None),
        ("filter status=500 & bytes>90000", ["ts", "bytes", "host"],
         [("status", "=", 500), ("bytes", ">", 90000)], None),
        ("filter host=host-0042 (1/1000)", ["ts", "bytes"], [("host", "=", "host-0042")], None),
        ("filter ts in last 1% (sorted)", ["ts", "bytes"], [("ts", ">=", None)], None),
        ("limit 100", ["ts", "bytes", "host"], [], 100),
    ]
    ts_hi = pq.read_table(pqf, columns=["ts"])["ts"]
    cut = int(pc.quantile(ts_hi, 0.99)[0].as_py())
    del ts_hi

    for name, cols, filters, limit in cases:
        filters = [(c, o, cut if v is None else v) for c, o, v in filters]
        pq_filter = None
        for c, o, v in filters:
            e = {"=": ds.field(c) == v, ">": ds.field(c) > v, ">=": ds.field(c) >= v}[o]
            pq_filter = e if pq_filter is None else pq_filter & e
        sql_where = " AND ".join(f"{c} {'=' if o == '=' else o} " + (f"'{v}'" if isinstance(v, str) else str(v))
                                 for c, o, v in filters)
        where = f"WHERE {sql_where}" if sql_where else ""
        lim = f"LIMIT {limit}" if limit else ""
        sel = ", ".join(cols)

        # ---- pyarrow -----------------------------------------------------
        def pa_pq():
            if limit:
                return ds.dataset(pqf).scanner(columns=cols, filter=pq_filter).head(limit).num_rows
            return pq.read_table(pqf, columns=cols, filters=pq_filter).num_rows

        def pa_idx(parts=1):
            def f():
                if limit:
                    return t.scan(cols, filters, limit=limit).read_next_batch().num_rows
                if parts == 1:
                    return t.scan(cols, filters).read_all().num_rows
                return parallel_read(t, parts, columns=cols, filters=filters).num_rows
            return f

        s, n0 = med(pa_pq, reps); row(name, "pyarrow", "parquet", s)
        s, n1 = med(pa_idx(1), reps); row(name, "pyarrow", "index", s)
        if not limit:
            s, n2 = med(pa_idx(THREADS), reps); row(name, "pyarrow", f"index x{THREADS}", s)
            assert n0 == n1 == n2, (name, n0, n1, n2)

        # ---- Polars -----------------------------------------------------
        def expr():
            e = None
            for c, o, v in filters:
                x = {"=": pl.col(c) == v, ">": pl.col(c) > v, ">=": pl.col(c) >= v}[o]
                e = x if e is None else e & x
            return e

        def pl_q(lf):
            q = lf
            if filters:
                q = q.filter(expr())
            q = q.select(cols)
            if limit:
                q = q.head(limit)
            return q.collect().height

        s, n0 = med(lambda: pl_q(pl.scan_parquet(pqf)), reps); row(name, "polars", "parquet", s)
        lf = t.polars()
        s, n1 = med(lambda: pl_q(lf), reps); row(name, "polars", "index", s)
        assert n0 == n1, (name, n0, n1)

        # ---- DuckDB -----------------------------------------------------
        q_pq = f"SELECT {sel} FROM read_parquet('{pq_path}') {where} {lim}"
        s, n0 = med(lambda: duck.execute(q_pq).to_arrow_table().num_rows, reps); row(name, "duckdb", "parquet", s)

        def dk_idx():
            rd = t.scan(cols, filters, limit=limit)
            return duck.execute(f"SELECT {sel} FROM rd").to_arrow_table().num_rows

        s, n1 = med(dk_idx, reps); row(name, "duckdb", "index", s)
        assert n0 == n1, (name, n0, n1)

    json.dump(out, open(os.path.join(work, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
