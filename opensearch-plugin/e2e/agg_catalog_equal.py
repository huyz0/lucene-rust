"""Aggregation catalog check: the same requests on a plugin node and a stock node must give the same
aggregations, and the plugin's fallbacks for them must be zero.

  agg_catalog_equal.py dump BASE OUT.json     (creates index `catalog` on BASE, runs every request)
  agg_catalog_equal.py diff A.json B.json     (exit 1 on any difference)
  agg_catalog_equal.py fallbacks BASE         (prints the plugin's aggregation_* fallback counts)
"""
import json
import os
import random
import sys

import benchhttp as bh

N = 3000
MAPPING = {"properties": {
    "k": {"type": "keyword"}, "g": {"type": "keyword"}, "n": {"type": "long"}, "d": {"type": "double"},
    "f": {"type": "float"}, "i": {"type": "integer"}, "ts": {"type": "date"}, "t": {"type": "text"},
    "ip": {"type": "ip"}}}


def docs():
    r = random.Random(5)
    for i in range(N):
        d = {"t": "w%d" % r.randrange(20)}
        if r.random() < 0.7:
            d["k"] = "k%d" % r.randrange(30)
        if r.random() < 0.9:
            d["g"] = "g%d" % r.randrange(5)
        if r.random() < 0.6:
            d["n"] = r.randrange(-500, 5000)
        if r.random() < 0.6:
            d["d"] = round(r.uniform(-100, 1000), 3)
        if r.random() < 0.5:
            d["f"] = round(r.uniform(0, 50), 2)
        if r.random() < 0.5:
            d["i"] = r.randrange(0, 100)
            if r.random() < 0.3:
                d["i"] = [d["i"], r.randrange(0, 100)]
        if r.random() < 0.8:
            d["ts"] = 1_700_000_000_000 + r.randrange(200) * 3_600_000
        if r.random() < 0.4:
            d["ip"] = "10.0.%d.%d" % (r.randrange(4), r.randrange(256))
        yield d


def aggs():
    out = {}
    for fld in ["k", "n", "d", "ts", "t_none", "ip"]:
        out["missing " + fld] = {"a": {"missing": {"field": fld}}}
    out["missing + subs"] = {"a": {"missing": {"field": "k"}, "aggs": {"s": {"sum": {"field": "d"}}, "t": {"terms": {"field": "g"}}}}}
    out["missing under terms"] = {"a": {"terms": {"field": "g"}, "aggs": {"m": {"missing": {"field": "n"}}}}}
    for fld in ["n", "d", "f", "i", "ts"]:
        out["extended_stats " + fld] = {"a": {"extended_stats": {"field": fld}}}
    out["extended_stats sigma"] = {"a": {"extended_stats": {"field": "d", "sigma": 3}}}
    out["extended_stats under terms"] = {"a": {"terms": {"field": "g"}, "aggs": {"e": {"extended_stats": {"field": "d"}}}}}
    out["extended_stats under histogram"] = {"a": {"histogram": {"field": "n", "interval": 500}, "aggs": {"e": {"extended_stats": {"field": "f"}}}}}
    for fld in ["n", "d", "i", "f"]:
        out["median_absolute_deviation " + fld] = {"a": {"median_absolute_deviation": {"field": fld}}}
    out["mad compression"] = {"a": {"median_absolute_deviation": {"field": "d", "compression": 200}}}
    out["mad under terms"] = {"a": {"terms": {"field": "g"}, "aggs": {"m": {"median_absolute_deviation": {"field": "n"}}}}}
    return out


QUERIES = [None, {"term": {"t": "w3"}}, {"range": {"n": {"gte": 0, "lt": 2000}}}]


def dump(base):
    try:
        bh.req(base, "DELETE", "/catalog", None)
    except RuntimeError:
        pass
    settings = {"number_of_shards": 1, "number_of_replicas": 0}
    if os.environ.get("ACCEL"):
        settings["index.engine.accelerated"] = True
    bh.req(base, "PUT", "/catalog", {"settings": settings, "mappings": MAPPING})
    lines = []
    for i, d in enumerate(docs()):
        lines.append(json.dumps({"index": {"_id": str(i)}}))
        lines.append(json.dumps(d))
    bh.req(base, "POST", "/catalog/_bulk?refresh=true", "\n".join(lines) + "\n", ndjson=True)
    bh.req(base, "POST", "/catalog/_forcemerge?max_num_segments=1", None)
    res = {}
    for name, a in aggs().items():
        for qi, q in enumerate(QUERIES):
            body = {"size": 0, "aggs": a}
            if q:
                body["query"] = q
            try:
                r = bh.req(base, "POST", "/catalog/_search?request_cache=false", body)
                res["%s q%d" % (name, qi)] = r["aggregations"]
            except Exception as e:  # an error is part of the answer
                res["%s q%d" % (name, qi)] = "error: %s" % str(e)[:200]
    return res


def norm(x):
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v) for v in x]
    return x


if sys.argv[1] == "dump":
    json.dump(dump(sys.argv[2]), open(sys.argv[3], "w"), indent=1, sort_keys=True)
elif sys.argv[1] == "fallbacks":
    st = bh.req(sys.argv[2], "GET", "/_plugins/lucene_rust/stats", None)
    print({k: v for k, v in st.get("fallbacks", {}).items() if "aggregation" in k})
else:
    a, b = (norm(json.load(open(p))) for p in sys.argv[2:4])
    bad = [k for k in a if a[k] != b.get(k)]
    for k in sorted(a):
        print(("DIFF " if k in bad else "same ") + k)
    sys.exit(1 if bad else 0)
