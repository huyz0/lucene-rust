#!/usr/bin/env python3
"""Edge cases against a stock node: empty indices, nulls and empty arrays, unicode (emoji, combining
marks, right-to-left, lone surrogates escaped, NUL), very long terms and values, a huge document,
deleted and updated documents across many segments, and several shards: every query / aggregation /
sort / fetch answer must be identical.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_edge_cases.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_api_stock as v  # noqa: E402

both = v.both
MAP = {"properties": {"t": {"type": "text", "fields": {"raw": {"type": "keyword", "ignore_above": 256}}}, "k": {"type": "keyword"}, "n": {"type": "long"},
                      "f": {"type": "double"}, "d": {"type": "date"}, "b": {"type": "boolean"}, "ip": {"type": "ip"}}}
QUERIES = [{"match_all": {}}, {"match": {"t": "hello"}}, {"term": {"k": "x"}}, {"range": {"n": {"gte": 0}}}, {"exists": {"field": "n"}},
           {"bool": {"must_not": [{"exists": {"field": "k"}}]}}, {"match_phrase": {"t": "a b"}}, {"prefix": {"k": "é"}}, {"wildcard": {"t.raw": "*😀*"}},
           {"query_string": {"query": "t:(hello OR wörld) AND NOT k:y"}}, {"fuzzy": {"t": {"value": "helo"}}}, {"regexp": {"k": ".*"}}]
AGGS = {"k": {"terms": {"field": "k", "size": 50}}, "n": {"stats": {"field": "n"}}, "f": {"avg": {"field": "f"}}, "c": {"cardinality": {"field": "k"}},
        "h": {"histogram": {"field": "n", "interval": 100, "min_doc_count": 1}}, "m": {"missing": {"field": "k"}},
        "tr": {"terms": {"field": "t.raw", "size": 50}}}


def make(index, shards, settings=None):
    for n in (v.plugin, v.stock):
        n.req("DELETE", f"/{index}")
        st, resp = n.req("PUT", f"/{index}", {"settings": {"number_of_shards": shards, "number_of_replicas": 0, **(settings or {})}, "mappings": MAP})
        assert st == 200, resp


def bulk(index, docs, with_ids=True):
    lines = []
    for i, d in enumerate(docs):
        lines += [json.dumps({"index": {"_id": f"d{i}"}}), json.dumps(d)]
    p = ("\n".join(lines) + "\n").encode()
    for n in (v.plugin, v.stock):
        st, resp = n.req("POST", f"/{index}/_bulk?refresh=true", p, raw=True)
        assert st == 200, resp


def check(index, label):
    for q in QUERIES:
        both("POST", f"/{index}/_search", {"query": q, "size": 20, "track_total_hits": True, "sort": [{"_seq_no": "asc"}], "aggs": AGGS},
             label=f"{label} {json.dumps(q)[:50]}", ign=v.IGN | {"_shard", "_node"})
        both("POST", f"/{index}/_count", {"query": q}, label=f"{label} count {json.dumps(q)[:40]}")
    for s in ("n", "k", "t.raw", "f"):   # not an all-missing key: equal keys across shards merge in arrival order on stock too
        for o in ("asc", "desc"):
            both("POST", f"/{index}/_search", {"size": 15, "sort": [{s: {"order": o, "missing": "_last"}}, {"_seq_no": "asc"}], "_source": False}, label=f"{label} sort {s} {o}")
    both("POST", f"/{index}/_search", {"size": 5, "query": {"match": {"t": "hello"}}, "highlight": {"fields": {"t": {}}}, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}]}, label=f"{label} highlight")


def main():
    # 1. empty indices (no docs at all, and docs all deleted)
    make("e_empty", 3)
    check("e_empty", "empty")
    make("e_alldel", 2)
    bulk("e_alldel", [{"t": "hello", "k": "x", "n": i} for i in range(50)])
    for i in range(50):
        for n in (v.plugin, v.stock):
            n.req("DELETE", f"/e_alldel/_doc/d{i}")
    for n in (v.plugin, v.stock):
        n.req("POST", "/e_alldel/_refresh")
    check("e_alldel", "alldeleted")
    # 2. nulls, empty arrays, odd shapes
    make("e_nulls", 1)
    bulk("e_nulls", [{}, {"t": None, "k": None, "n": None}, {"t": "", "k": "", "n": 0}, {"t": [], "k": [], "n": []}, {"t": ["hello", None, "a b"], "k": ["x", None, "x"], "n": [1, None, 1, 2]},
                     {"t": "hello", "k": "x", "n": -5, "f": 1e300, "d": "1970-01-01", "b": "true", "ip": "::ffff:1.2.3.4"}, {"f": "NaN-ish" if False else -0.0, "n": 2**63 - 1}])
    check("e_nulls", "nulls")
    # 3. unicode
    uni = ["hello wörld", "😀😀 hello", "é é", "שלום hello", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "ǅ ǆ Ǆ", "ß SS ss", "İstanbul ı", "\u0000nul", "a​b﻿c", "😀 a b", "日本語のテキスト hello", "x" * 400, "é" * 300]
    make("e_uni", 2)
    bulk("e_uni", [{"t": u, "k": u[:200], "n": i} for i, u in enumerate(uni * 4)])
    check("e_uni", "unicode")
    # 4. many segments with updates and deletes, several shards
    r = random.Random(11)
    make("e_seg", 4, {"index.merge.policy.segments_per_tier": 50, "index.merge.policy.max_merge_at_once": 50})
    for rnd in range(25):
        docs = [{"t": " ".join(r.choice(["hello", "a", "b", "wörld"]) for _ in range(4)), "k": r.choice("xyzé"), "n": r.randint(-500, 500), "f": r.random()} for _ in range(60)]
        lines = []
        for d in docs:
            lines += [json.dumps({"index": {"_id": f"s{r.randint(0, 600)}"}}), json.dumps(d)]
        p = ("\n".join(lines) + "\n").encode()
        dels = [f"s{r.randint(0, 600)}" for _ in range(10)]
        for n in (v.plugin, v.stock):
            n.req("POST", "/e_seg/_bulk?refresh=true", p, raw=True)
            for d in dels:
                n.req("DELETE", f"/e_seg/_doc/{d}")
    for n in (v.plugin, v.stock):
        n.req("POST", "/e_seg/_refresh")
    check("e_seg", "segments")
    for n in (v.plugin, v.stock):
        n.req("POST", "/e_seg/_forcemerge?max_num_segments=1")
    check("e_seg", "merged")
    # 5. a huge document and a huge term
    make("e_huge", 1)
    big = " ".join(r.choice(["alpha", "beta", "hello", "w%d" % r.randint(0, 5000)]) for _ in range(3_000_000))
    bulk("e_huge", [{"t": big, "k": "x", "n": 1}, {"t": "hello " + "z" * 40000, "k": "y", "n": 2}, {"t": "small hello", "k": "x", "n": 3}])
    for q in ({"match": {"t": "hello"}}, {"match": {"t": "alpha beta"}}, {"match_phrase": {"t": "alpha beta"}}, {"term": {"t": "w42"}}):
        both("POST", "/e_huge/_search", {"query": q, "size": 3, "_source": ["k", "n"], "sort": [{"_score": "desc"}, {"_seq_no": "asc"}], "highlight": {"fields": {"t": {"fragment_size": 50, "number_of_fragments": 1}}}},
             label=f"huge {json.dumps(q)}")
    both("POST", "/e_huge/_search", {"size": 0, "aggs": {"w": {"terms": {"field": "t", "size": 5}}}}, label="huge terms agg on text (fielddata off)")
    print(f"edge cases: {v.total} checks, mismatches: {v.bad}")
    return 1 if v.bad else 0


if __name__ == "__main__":
    sys.exit(main())
