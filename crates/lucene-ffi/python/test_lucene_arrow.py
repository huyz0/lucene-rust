"""Tests of lucene_arrow against an index written by the `arrow_gen` example.

    cargo build --release -p lucene-sql --example arrow_gen
    cargo build --release -p lucene-ffi --no-default-features --features crypto,arrow-sql
    LUCENE_ARROW_LIB=target/release/liblucene_ffi.so \
    LUCENE_ARROW_GEN=target/release/examples/arrow_gen python3 -m unittest -v test_lucene_arrow
"""
import os
import subprocess
import tempfile
import unittest

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

import lucene_arrow as la

COLS = {"id": "key", "ts": "long", "status": "long", "bytes": "long", "latency": "long",
        "host": "keyword", "level": "keyword", "region": "keyword"}
DOCS = 30_000


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        subprocess.run([os.environ["LUCENE_ARROW_GEN"], cls.tmp.name + "/index", str(DOCS), "3"],
                       check=True, capture_output=True)
        cls.t = la.Table(cls.tmp.name + "/index", "logs", COLS)
        cls.full = cls.t.scan(dictionary=False).read_all()

    @classmethod
    def tearDownClass(cls):
        cls.t.close()
        cls.tmp.cleanup()

    def test_info_and_full_scan(self):
        self.assertEqual(self.t.info["docs"], DOCS)
        self.assertEqual(self.full.num_rows, DOCS)
        self.assertEqual(self.full.column_names, list(COLS))
        ids = self.full["id"].to_pylist()
        self.assertEqual(ids, [f"doc-{i:09d}" for i in range(DOCS)])

    def test_dictionary_and_text_agree(self):
        d = self.t.scan(["host", "level"]).read_all()
        self.assertTrue(pa.types.is_dictionary(d.schema.field("host").type))
        self.assertEqual(d["host"].cast(pa.string()).to_pylist(), self.full["host"].to_pylist())

    def test_filters_match_pyarrow_compute(self):
        f = self.full
        cases = [
            ([("status", "=", 500)], pc.equal(f["status"], 500)),
            ([("bytes", ">", 90000), ("status", "!=", 200)],
             pc.and_(pc.greater(f["bytes"], 90000), pc.not_equal(f["status"], 200))),
            ([("host", "=", "host-0042")], pc.equal(f["host"], "host-0042")),
            ([("level", "in", ["warn", "error"])], pc.is_in(f["level"], pa.array(["warn", "error"]))),
            ([("ts", "<", int(f["ts"][1000].as_py()))], pc.less(f["ts"], f["ts"][1000])),
            ([("id", "=", "doc-000000123")], pc.equal(f["id"], "doc-000000123")),
        ]
        for filters, mask in cases:
            got = self.t.scan(["id", "bytes"], filters).read_all()
            want = f.filter(mask).select(["id", "bytes"])
            self.assertEqual(got.to_pylist(), want.to_pylist(), filters)

    def test_limit_and_empty_projection(self):
        self.assertEqual(self.t.scan(["ts"], limit=10).read_all().num_rows, 10)
        counted = self.t.scan([], [("status", "=", 500)]).read_all().num_rows
        self.assertEqual(counted, pc.sum(pc.equal(self.full["status"], 500)).as_py())

    def test_partitions_cover_the_table(self):
        n = self.t.partition_count(2)
        self.assertEqual(n, 2)
        rows = sum(self.t.scan(["ts"], partition=i, partitions=2).read_all().num_rows for i in range(n))
        self.assertEqual(rows, DOCS)

    def test_errors_carry_the_native_message(self):
        with self.assertRaisesRegex(RuntimeError, "no column"):
            self.t.scan(["nope"])
        with self.assertRaisesRegex(ValueError, "unknown operator"):
            self.t.scan(["ts"], [("ts", "~", 1)])
        with self.assertRaises(RuntimeError):
            la.Table(self.tmp.name + "/missing", "logs", COLS)

    def test_sql(self):
        got = self.t.sql("SELECT status, count(*) AS c FROM logs GROUP BY status").read_all()
        want = self.full.group_by("status").aggregate([("status", "count")])
        self.assertEqual(sorted((r["status"], r["c"]) for r in got.to_pylist()),
                         sorted((r["status"], r["status_count"]) for r in want.to_pylist()))

    def test_polars_pushdown_gives_polars_answers(self):
        lf = self.t.polars(dictionary=False)
        pdf = pl.from_arrow(self.full)
        queries = [
            lambda d: d.filter((pl.col("status") == 500) & (pl.col("bytes") > 90000)).select(["id", "bytes"]),
            lambda d: d.filter(pl.col("host") == "host-0042").select("ts"),
            lambda d: d.filter(pl.col("bytes") > 1000).filter(pl.col("latency") < 50).select(["id"]),
            lambda d: d.filter((pl.col("bytes") > 1000) | (pl.col("status") == 404)).select("id"),
            lambda d: d.select("ts", "host").head(7),
            lambda d: d.group_by("level").agg(pl.len()).sort("level"),
        ]
        for q in queries:
            self.assertTrue(q(lf).collect().equals(q(pdf.lazy()).collect()))

    def test_duckdb_reads_the_stream(self):
        rd = self.t.scan(["status", "bytes"], [("bytes", ">", 50000)])
        got = duckdb.sql("SELECT status, sum(bytes) FROM rd GROUP BY status ORDER BY status").fetchall()
        want = (pl.from_arrow(self.full).filter(pl.col("bytes") > 50000)
                .group_by("status").agg(pl.col("bytes").sum()).sort("status").rows())
        self.assertEqual(got, [tuple(r) for r in want])


if __name__ == "__main__":
    unittest.main()
