"""An index as an Arrow source for pyarrow, Polars and DuckDB.

    import lucene_arrow as la

    t = la.Table("/data/index", "logs", {"id": "key", "ts": "long", "host": "keyword"})
    reader = t.scan(columns=["ts", "host"], filters=[("ts", ">=", 1_700_000_000_000)])
    table = reader.read_all()                 # pyarrow.Table, decoded zero-copy
    df = t.polars(columns=...).collect()      # Polars LazyFrame with push-down
    duckdb.sql("SELECT host, count(*) FROM reader GROUP BY host")

Column pruning, comparisons (`=`, `!=`, `<`, `<=`, `>`, `>=`), `in`, `is null`
and a row limit are applied inside the scan, over the index's doc values;
what the scan does not take, the consumer applies to the batches it gets.
The batches cross the process boundary as the Arrow C stream interface, so
nothing is copied or serialised.
"""

from __future__ import annotations

import ctypes
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa

_LIB: Optional[ctypes.CDLL] = None
_STATUS = {
    1: "null pointer",
    3: "invalid handle",
    4: "io error",
    10: "invalid argument",
    11: "handle limit",
}


def _load(path: Optional[str] = None) -> ctypes.CDLL:
    global _LIB
    if _LIB is not None and path is None:
        return _LIB
    path = path or os.environ.get("LUCENE_ARROW_LIB") or "liblucene_ffi.so"
    lib = ctypes.CDLL(path)
    u8p = ctypes.POINTER(ctypes.c_ubyte)
    lib.ffi_arrow_table_open.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64)]
    lib.ffi_arrow_table_close.argtypes = [ctypes.c_uint64]
    lib.ffi_arrow_table_info.argtypes = [
        ctypes.c_uint64, ctypes.POINTER(u8p), ctypes.POINTER(ctypes.c_size_t)]
    lib.ffi_arrow_scan.argtypes = [
        ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_void_p]
    lib.ffi_arrow_partition_count.argtypes = [
        ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    lib.ffi_jvm_last_error.argtypes = [ctypes.POINTER(u8p), ctypes.POINTER(ctypes.c_size_t)]
    lib.ffi_jvm_free_bytes.argtypes = [u8p, ctypes.c_size_t]
    if hasattr(lib, "ffi_arrow_sql"):
        lib.ffi_arrow_sql.argtypes = [
            ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p]
    _LIB = lib
    return lib


def _take(lib, ptr, n) -> bytes:
    if not ptr:
        return b""
    data = ctypes.string_at(ptr, n)
    lib.ffi_jvm_free_bytes(ptr, n)
    return data


def _check(lib, status: int) -> None:
    if status == 0:
        return
    ptr, n = ctypes.POINTER(ctypes.c_ubyte)(), ctypes.c_size_t()
    lib.ffi_jvm_last_error(ctypes.byref(ptr), ctypes.byref(n))
    msg = _take(lib, ptr, n.value).decode("utf-8", "replace")
    raise RuntimeError(f"{msg or _STATUS.get(status, 'error')} (status {status})")


# An ArrowArrayStream: get_schema, get_next, get_last_error, release, private_data.
_STREAM_BYTES = 5 * ctypes.sizeof(ctypes.c_void_p)

_OPS = {"=": "=", "==": "=", "!=": "!=", "<>": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}


def _filters_json(filters) -> list:
    out = []
    for f in filters or []:
        col, op, val = f
        if op == "in":
            out.append({"column": col, "in": list(val)})
        elif op == "is_null":
            out.append({"column": col, "is_null": bool(val)})
        elif op in _OPS:
            out.append({"column": col, "op": _OPS[op], "value": val})
        else:
            raise ValueError(f"unknown operator {op!r}")
    return out


class Table:
    """An index read as the table `name` with the given columns.

    `columns` maps a column name to its kind: `key` (stored string), `long`,
    `keyword` (doc-values string), `text` or `vector` (`("vector", dim)`).
    """

    def __init__(self, path: str, name: str, columns: Dict[str, object], lib: Optional[str] = None):
        self._lib = _load(lib)
        cols = []
        for n, k in columns.items():
            if isinstance(k, tuple):
                cols.append({"name": n, "kind": k[0], "dim": k[1]})
            else:
                cols.append({"name": n, "kind": k})
        spec = json.dumps({"name": name, "columns": cols}).encode()
        p = os.fsencode(path)
        h = ctypes.c_uint64()
        _check(self._lib, self._lib.ffi_arrow_table_open(p, len(p), spec, len(spec), ctypes.byref(h)))
        self._h = h.value
        self.columns = dict(columns)
        info = ctypes.POINTER(ctypes.c_ubyte)()
        n = ctypes.c_size_t()
        _check(self._lib, self._lib.ffi_arrow_table_info(self._h, ctypes.byref(info), ctypes.byref(n)))
        self.info = json.loads(_take(self._lib, info, n.value))

    def close(self) -> None:
        if getattr(self, "_h", 0):
            self._lib.ffi_arrow_table_close(self._h)
            self._h = 0

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _import(self, fill) -> pa.RecordBatchReader:
        buf = ctypes.create_string_buffer(_STREAM_BYTES)
        _check(self._lib, fill(ctypes.addressof(buf)))
        return pa.RecordBatchReader._import_from_c(ctypes.addressof(buf))

    def scan(self, columns: Optional[Sequence[str]] = None, filters=None, limit: Optional[int] = None,
             batch_rows: Optional[int] = None, dictionary: bool = True,
             partition: int = 0, partitions: int = 1,
             threads: Optional[int] = None) -> pa.RecordBatchReader:
        """A stream of the rows the filters keep. `filters` is a list of
        `(column, op, value)`; `partition`/`partitions` read a share of the
        segments (see `partition_count`), the limit applies to each. `threads`
        decode side by side for this stream (default: one per core, at most 8,
        or one when reading a partition)."""
        req = {"filters": _filters_json(filters), "dictionary": dictionary}
        if columns is not None:
            req["columns"] = list(columns)
        if limit is not None:
            req["limit"] = int(limit)
        if batch_rows is not None:
            req["batch_rows"] = int(batch_rows)
        if threads is None and partitions > 1:
            threads = 1
        if threads is not None:
            req["threads"] = int(threads)
        body = json.dumps(req).encode()
        return self._import(lambda out: self._lib.ffi_arrow_scan(
            self._h, body, len(body), partition, partitions, out))

    def partition_count(self, partitions: int) -> int:
        n = ctypes.c_uint32()
        _check(self._lib, self._lib.ffi_arrow_partition_count(self._h, partitions, ctypes.byref(n)))
        return n.value

    def sql(self, statement: str) -> pa.RecordBatchReader:
        """The rows of a read-only SQL statement over the table."""
        if not hasattr(self._lib, "ffi_arrow_sql"):
            raise RuntimeError("this build of the library has no SQL (feature arrow-sql)")
        s = statement.encode()
        return self._import(lambda out: self._lib.ffi_arrow_sql(self._h, s, len(s), out))

    # ---- Polars -----------------------------------------------------------

    def polars(self, dictionary: bool = True):
        """A Polars LazyFrame that pushes the projection, a row limit and the
        comparisons of its filter into the scan."""
        import polars as pl
        from polars.io.plugins import register_io_source

        schema = self.scan(limit=0, dictionary=dictionary).schema
        pl_schema = pl.from_arrow(schema.empty_table()).schema

        def source(with_columns, predicate, n_rows, batch_size):
            pushed, rest = _polars_filters(predicate)
            cols = list(with_columns) if with_columns is not None else None
            if cols is not None and rest is not None:
                # The residual filter may need columns the query does not keep.
                need = set(rest.meta.root_names())
                extra = [c for c in need if c not in cols]
                scan_cols = cols + extra
            else:
                scan_cols = cols
            reader = self.scan(scan_cols, pushed, n_rows if rest is None else None,
                               batch_size, dictionary)
            done = 0
            for chunk in _chunks(reader, CHUNK_ROWS):
                df = pl.from_arrow(chunk)
                if rest is not None:
                    df = df.filter(rest)
                    if cols is not None:
                        df = df.select(cols)
                    if n_rows is not None:
                        df = df.head(n_rows - done)
                done += df.height
                yield df
                if n_rows is not None and done >= n_rows:
                    return

        return register_io_source(source, schema=pl_schema)


# Batches are joined into chunks of about this many rows before Polars sees
# them: a DataFrame per 64K-row batch costs more in Polars than the decoding.
CHUNK_ROWS = 1 << 20


def _chunks(reader, rows):
    acc, n = [], 0
    for b in reader:
        acc.append(b)
        n += b.num_rows
        if n >= rows:
            yield pa.Table.from_batches(acc)
            acc, n = [], 0
    if acc:
        yield pa.Table.from_batches(acc)


def _polars_filters(predicate):
    """Splits a Polars predicate into the `(column, op, value)` comparisons the
    scan takes and the expression left for Polars (or None)."""
    if predicate is None:
        return [], None
    try:
        tree = json.loads(predicate.meta.serialize(format="json"))
    except Exception:
        return [], predicate
    import polars as pl

    ops = {"Eq": "=", "NotEq": "!=", "Lt": "<", "LtEq": "<=", "Gt": ">", "GtEq": ">="}
    flip = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}

    def lit(node):
        v = node.get("Literal", node.get("Scalar")) if isinstance(node, dict) else None
        if v is None:
            return None
        if isinstance(v, dict):
            if len(v) == 1:
                inner = next(iter(v.values()))
                if isinstance(inner, dict) and len(inner) == 1:
                    inner = next(iter(inner.values()))
                if isinstance(inner, (int, str)) and not isinstance(inner, bool):
                    return inner
        return None

    def walk(node):
        """Comparisons ANDed together, or None when the node is anything else."""
        if not isinstance(node, dict) or "BinaryExpr" not in node:
            return None
        b = node["BinaryExpr"]
        op = b["op"]
        if op == "And":
            l, r = walk(b["left"]), walk(b["right"])
            return None if l is None or r is None else l + r
        if op in ops:
            left, right = b["left"], b["right"]
            if isinstance(left, dict) and "Column" in left and lit(right) is not None:
                return [(left["Column"], ops[op], lit(right))]
            if isinstance(right, dict) and "Column" in right and lit(left) is not None:
                return [(right["Column"], flip[ops[op]], lit(left))]
        return None

    got = walk(tree)
    if got is None:
        return [], predicate
    return got, None
