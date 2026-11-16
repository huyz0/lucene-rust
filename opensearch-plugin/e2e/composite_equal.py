"""Dump composite aggregation results (every page, via after_key) to a JSON file, for diffing
a plugin node against a stock node on the same corpus.

  composite_equal.py dump BASE OUT.json      writes {shape: [page, ...]}
  composite_equal.py diff A.json B.json      exits 1 on any difference

Index `bench` as made by bench_nodes.py (fields tag, category, num, price, ts, body)."""
import json
import sys

import benchhttp as bh

SRC = {
    "tag": {"terms": {"field": "tag"}},
    "cat": {"terms": {"field": "category"}},
    "num": {"terms": {"field": "num"}},
    "ts": {"date_histogram": {"field": "ts", "calendar_interval": "day"}},
    "tsm": {"date_histogram": {"field": "ts", "fixed_interval": "6h"}},
    "price": {"histogram": {"field": "price", "interval": 50}},
}
SHAPES = {
    "tag+cat": (["tag", "cat"], 100, None),
    "cat+tag": (["cat", "tag"], 100, None),
    "cat+tag s10": (["cat", "tag"], 10, None),
    "num+cat": (["num", "cat"], 100, None),
    "ts+cat": (["ts", "cat"], 100, None),
    "ts+tag s7": (["ts", "tag"], 7, None),
    "tsm+cat": (["tsm", "cat"], 50, None),
    "tag": (["tag"], 25, None),
    "num": (["num"], 1000, None),
    "tag+cat filtered": (["tag", "cat"], 100, {"match": {"body": "t8"}}),
    "num+cat filtered": (["num", "cat"], 100, {"range": {"price": {"lt": 300}}}),
    "num+tag range on num": (["num", "tag"], 100, {"range": {"num": {"gte": 1000, "lt": 500000}}}),
    "ts+cat range on ts": (["ts", "cat"], 100, {"range": {"ts": {"gte": 1_710_000_000_000}}}),
    "price+cat": (["price", "cat"], 100, None),
    "cat+num": (["cat", "num"], 100, None),
}
PAGES = 12


def dump(base):
    out = {}
    for name, (srcs, size, query) in SHAPES.items():
        pages, after = [], None
        for _ in range(PAGES):
            comp = {"size": size, "sources": [{s: SRC[s]} for s in srcs]}
            if after:
                comp["after"] = after
            body = {"size": 0, "aggs": {"a": {"composite": comp, "aggs": {"p": {"sum": {"field": "price"}}}}}}
            if query:
                body["query"] = query
            r = bh.req(base, "POST", "/bench/_search?request_cache=false", body)
            a = r["aggregations"]["a"]
            pages.append(a)
            after = a.get("after_key")
            if not after or not a["buckets"]:
                break
        out[name] = pages
    return out


def norm(x):
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v) for v in x]
    if isinstance(x, float):
        return round(x, 6)
    return x


if sys.argv[1] == "dump":
    json.dump(dump(sys.argv[2]), open(sys.argv[3], "w"))
else:
    a, b = (norm(json.load(open(p))) for p in sys.argv[2:4])
    bad = [k for k in a if a[k] != b.get(k)]
    for k in a:
        print(("DIFF " if k in bad else "same ") + k, f"({len(a[k])} pages)")
    sys.exit(1 if bad else 0)
