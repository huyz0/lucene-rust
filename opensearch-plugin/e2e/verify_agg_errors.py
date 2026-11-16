"""Aggregation requests that fail or are rejected, run against a stock node and a plugin node on the same
data: status, error type and shape of the response must agree. Guards the plugin's deferred aggregator
construction (an error stock raises when it builds aggregators must still surface, as the same error).

  verify_agg_errors.py collect URL OUT.json     (a node with the `bench` index of bench_nodes.py loaded)
  verify_agg_errors.py diff STOCK.json PLUGIN.json
"""
import json
import sys

import benchhttp as bh

CASES = {
    "terms on text field": {"size": 0, "aggs": {"a": {"terms": {"field": "body"}}}},
    "terms on unmapped field": {"size": 0, "aggs": {"a": {"terms": {"field": "nope"}}}},
    "terms size 0": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 0}}}},
    "terms negative size": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": -1}}}},
    "terms bad include regex": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "include": "c(["}}}},
    "terms bad order path": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "order": {"zzz": "asc"}}}}},
    "terms order by missing sub": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "order": {"m": "asc"}}}}},
    "terms script error": {"size": 0, "aggs": {"a": {"terms": {"script": {"source": "doc['category'].value.foo()", "lang": "painless"}}}}},
    "terms on numeric with include string": {"size": 0, "aggs": {"a": {"terms": {"field": "num", "include": "x.*"}}}},
    "stats on keyword": {"size": 0, "aggs": {"a": {"stats": {"field": "category"}}}},
    "avg on text": {"size": 0, "aggs": {"a": {"avg": {"field": "body"}}}},
    "avg on unmapped": {"size": 0, "aggs": {"a": {"avg": {"field": "nope"}}}},
    "min with bad script": {"size": 0, "aggs": {"a": {"min": {"script": "1 +"}}}},
    "sum missing string": {"size": 0, "aggs": {"a": {"sum": {"field": "price", "missing": "abc"}}}},
    "duplicate agg names": {"size": 0, "aggs": {"a": {"min": {"field": "price"}}, "a ": {"max": {"field": "price"}}}},
    "agg without type": {"size": 0, "aggs": {"a": {}}},
    "agg two types": {"size": 0, "aggs": {"a": {"min": {"field": "price"}, "max": {"field": "price"}}}},
    "date_histogram bad interval": {"size": 0, "aggs": {"a": {"date_histogram": {"field": "ts", "calendar_interval": "x"}}}},
    "histogram on keyword": {"size": 0, "aggs": {"a": {"histogram": {"field": "category", "interval": 5}}}},
    "composite empty sources": {"size": 0, "aggs": {"a": {"composite": {"sources": []}}}},
    "pipeline missing path": {"size": 0, "aggs": {"a": {"terms": {"field": "category"}}, "b": {"max_bucket": {"buckets_path": "zz"}}}},
    "global with terms": {"size": 0, "aggs": {"g": {"global": {}, "aggs": {"a": {"terms": {"field": "nope", "size": -2}}}}}},
    "ok terms size 1": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 1}}}},
    "ok terms size 10 shard_size 10": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 10, "shard_size": 10}}}},
    "ok terms size 50": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 50}}}},
    "ok terms size 100": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 100}}}},
    "ok terms error stats": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 5, "show_term_doc_count_error": True}}}},
    "ok terms track false": {"size": 0, "track_total_hits": False, "aggs": {"a": {"terms": {"field": "category", "size": 5}}}},
    "ok terms track 5": {"size": 0, "track_total_hits": 5, "aggs": {"a": {"terms": {"field": "category", "size": 5}}}},
    "ok terms match_all explicit": {"size": 0, "query": {"match_all": {}}, "aggs": {"a": {"terms": {"field": "category", "size": 7}}}},
    "ok terms with query": {"size": 0, "query": {"match": {"body": "t8"}}, "aggs": {"a": {"terms": {"field": "category", "size": 7}}}},
    "ok terms tag 1000 terms": {"size": 0, "aggs": {"a": {"terms": {"field": "tag", "size": 20}}}},
    "ok terms min_doc_count 0": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "min_doc_count": 0}}}},
    "ok terms min_doc_count big": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "min_doc_count": 20000}}}},
    "ok terms include": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "include": "c1.*"}}}},
    "ok terms order key": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "order": {"_key": "desc"}}}}},
    "ok terms with hits": {"size": 3, "aggs": {"a": {"terms": {"field": "category", "size": 5}}}},
    "ok terms post_filter": {"size": 0, "post_filter": {"term": {"category": "c1"}}, "aggs": {"a": {"terms": {"field": "category", "size": 5}}}},
    "ok terms meta": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 3}, "meta": {"x": 1}}}},
    "ok terms": {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 5}}}},
    "ok stats": {"size": 0, "aggs": {"a": {"stats": {"field": "price"}}}},
    "ok terms+min": {"size": 0, "aggs": {"a": {"terms": {"field": "category"}}, "m": {"min": {"field": "price"}}}},
}


def run(base, body, max_buckets=None):
    c = bh.Client(base)
    try:
        return "ok", c.json("POST", "/bench/_search?request_cache=false", body)
    except RuntimeError as e:
        return "err", str(e)


def shape(kind, r):
    if kind == "ok":
        return ("ok", json.dumps(r.get("aggregations"), sort_keys=True), r["hits"].get("total"))
    s = r
    try:
        j = json.loads(s[s.index("{"):])
        e = j["error"]
        root = e.get("root_cause", [{}])[0]
        return ("err", j.get("status"), e.get("type"), root.get("type"))
    except Exception:
        return ("err", s[:200])


def collect(base):
    out = {name: shape(*run(base, body)) for name, body in CASES.items()}
    c = bh.Client(base)
    c.json("PUT", "/_cluster/settings", {"transient": {"search.max_buckets": 10}})
    try:
        out["too many buckets"] = shape(*run(base, {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 40}}}}))
    finally:
        c.json("PUT", "/_cluster/settings", {"transient": {"search.max_buckets": None}})
    return out


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        json.dump(collect(sys.argv[2]), open(sys.argv[3], "w"), indent=1)
    else:
        a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
        bad = 0
        for k in a:
            same = a[k] == b[k]
            bad += not same
            print(("PASS " if same else "FAIL ") + k, "" if same else f"\n   stock  {a[k]}\n   plugin {b[k]}")
        print("failures:", bad)
        sys.exit(1 if bad else 0)
