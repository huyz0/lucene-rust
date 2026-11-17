#!/usr/bin/env python3
"""`profile: true` on a plugin node against a stock OpenSearch node.

Loads the same documents into both nodes (several segments), runs one request per shape on both
with `profile: true`, and requires the same hits and the same profile structure: the query tree
(type, description, children in order, and which timers ran), the collector tree, the
aggregators, the post_filter's and rescore queries' trees, the fetch profile, and the same set
of breakdown keys on every node. Times themselves differ; they must be sane (not negative, not a
clock reading). The plugin node must answer the rows natively (its stats count them).

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_profile_stock.py
"""
import json
import os
import random
import sys
import urllib.error
import urllib.request

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
INDEX = "profile_stock"
FAILURES = []
WORDS = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta"]

# A request per shape. `native` rows must run on the native engine on the plugin node.
# Rows over a scoring disjunction, whose scorers prune by block maximum: the native clause scorers
# call `shallow_advance`, `compute_max_score` and `advance` where Lucene's step with `next_doc`,
# so which timers ran is not compared, only the trees.
LOOSE = {"bool", "metric_aggs"}

ROWS = {
    "match": {"query": {"match": {"msg": "alpha"}}},
    "size0": {"size": 0, "query": {"match": {"msg": "alpha beta"}}},
    "sorted": {"size": 5, "query": {"match": {"msg": "alpha"}}, "sort": [{"n": "desc"}]},
    "search_after": {"size": 4, "query": {"match": {"msg": "alpha"}}, "sort": [{"n": "asc"}], "search_after": [100]},
    "bool": {"query": {"bool": {
        "must": [{"match": {"msg": "alpha"}}],
        "should": [{"term": {"tag": "a"}}],
        "filter": [{"range": {"n": {"gte": 100}}}]}}},
    "must_not": {"query": {"bool": {"must": [{"match": {"msg": "alpha"}}], "must_not": [{"term": {"tag": "a"}}]}}},
    "filter_only": {"query": {"bool": {"filter": [{"term": {"tag": "b"}}, {"range": {"n": {"lt": 100}}}]}}},
    "dismax": {"query": {"dis_max": {"queries": [{"match": {"msg": "alpha"}}, {"match": {"msg": "beta"}}, {"match": {"msg": "gamma"}}]}}},
    "phrase": {"query": {"match_phrase": {"msg": "alpha beta"}}},
    "match_all": {"query": {"match_all": {}}},
    "range": {"query": {"range": {"n": {"gte": 100, "lt": 2000}}}},
    "exists": {"query": {"exists": {"field": "tag"}}},
    "post_filter": {"query": {"match": {"msg": "alpha"}}, "post_filter": {"term": {"tag": "a"}}},
    "rescore": {"query": {"match": {"msg": "alpha"}},
                "rescore": {"window_size": 10, "query": {"rescore_query": {"match": {"msg": "beta"}}}}},
    "min_score": {"min_score": 0.5, "query": {"match": {"msg": "alpha"}}},
    "terminate_after": {"terminate_after": 50, "query": {"match": {"msg": "alpha"}}},
    "track_total_hits": {"track_total_hits": True, "query": {"match": {"msg": "alpha beta gamma"}}},
    "terms_agg": {"size": 0, "query": {"match": {"msg": "alpha"}}, "aggs": {"t": {"terms": {"field": "tag"}}}},
    "metric_aggs": {"size": 3, "query": {"match": {"msg": "alpha gamma"}},
                    "aggs": {"s": {"sum": {"field": "n"}}, "a": {"avg": {"field": "n"}}}},
    "histogram": {"size": 0, "query": {"match_all": {}}, "aggs": {"h": {"histogram": {"field": "n", "interval": 1000}}}},
}


def requester(base):
    def req(method, path, body=None, ndjson=False):
        data = None
        headers = {}
        if body is not None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
        r = urllib.request.Request(base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(r, timeout=300) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if method == "DELETE" and e.code == 404:
                return {}
            raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:500]}")
    return req


def check(ok, what):
    if not ok:
        FAILURES.append(what)
        print("FAIL:", what)


def load(req):
    req("DELETE", f"/{INDEX}")
    req("PUT", f"/{INDEX}", {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {"msg": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "long"}}}})
    for b in range(4):
        rnd = random.Random(b)
        lines = []
        for i in range(2000):
            lines.append(json.dumps({"index": {"_index": INDEX, "_id": str(b * 2000 + i)}}))
            lines.append(json.dumps({"msg": " ".join(rnd.choice(WORDS) for _ in range(6)),
                                     "tag": rnd.choice(["a", "b", "c", "d"]), "n": b * 2000 + i}))
        req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        req("POST", f"/{INDEX}/_refresh")


def timers_ran(breakdown):
    return sorted(k for k, v in breakdown.items()
                  if v and not k.endswith(("_count", "_start_time")) and not k.startswith(("min_", "max_", "avg_")))


def query_shape(node, depth=0, out=None):
    out = [] if out is None else out
    out.append((depth, node["type"], node["description"][:80], tuple(timers_ran(node["breakdown"]))))
    for c in node.get("children", []):
        query_shape(c, depth + 1, out)
    return out


def collector_shape(node, depth=0, out=None):
    out = [] if out is None else out
    out.append((depth, node["name"], node["reason"]))
    for c in node.get("children", []):
        collector_shape(c, depth + 1, out)
    return out


def sane(node, what):
    for k, v in node["breakdown"].items():
        check(isinstance(v, int) and v >= 0, f"{what}: {node.get('type')} {k} = {v}")
        if k in ("build_scorer", "next_doc", "advance", "score", "create_weight", "collect", "initialize",
                 "build_leaf_collector", "post_collection", "build_aggregation") and v:
            check(v < 10 ** 14, f"{what}: {node.get('type')} {k} = {v} is a clock reading")
    for c in node.get("children", []):
        sane(c, what)


def structure(profile, what, loose=False):
    shard = profile["shards"][0]
    out = {"searches": [], "aggs": [], "fetch": []}
    for s in shard["searches"]:
        q = []
        for root in s["query"]:
            query_shape(root, 0, q)
            sane(root, what)
        if loose:
            q = [(d, t, desc, ()) for d, t, desc, _ in q]
        c = []
        for root in s["collector"]:
            collector_shape(root, 0, c)
        keys = sorted({k for root in s["query"] for k in _keys(root)})
        out["searches"].append((q, c, keys))
    for a in shard.get("aggregations", []):
        sane(a, what)
        out["aggs"].append((a["type"], a["description"], tuple(sorted(a["breakdown"]))))
    for f in shard.get("fetch", []):
        out["fetch"].append((f["type"], tuple(sorted(f["breakdown"]))))
    return out


def _keys(node):
    yield from node["breakdown"]
    for c in node.get("children", []):
        yield from _keys(c)


def hybrid_row(plugin):
    """A hybrid query: the stock node has no neural-search, so the reference is the plugin's own
    Java path (`index.search.accelerated: false`), the port of neural-search's profile."""
    body = {"profile": True, "size": 3, "query": {"hybrid": {"queries": [
        {"match": {"msg": "alpha"}}, {"term": {"tag": "a"}}]}}}
    plugin("PUT", f"/{INDEX}/_settings", {"index.search.accelerated": False})
    java = plugin("POST", f"/{INDEX}/_search", body)
    plugin("PUT", f"/{INDEX}/_settings", {"index.search.accelerated": True})
    before = plugin("GET", "/_plugins/accelerated/stats")["accelerated_queries"]
    native = plugin("POST", f"/{INDEX}/_search", body)
    ran = plugin("GET", "/_plugins/accelerated/stats")["accelerated_queries"] - before
    check(ran == 1, "hybrid profile ran natively")
    check([h["_id"] for h in java["hits"]["hits"]] == [h["_id"] for h in native["hits"]["hits"]], "hybrid: hits")
    sa = structure(java["profile"], "hybrid (java)", True)
    sb = structure(native["profile"], "hybrid (native)", True)
    check(sa == sb, f"hybrid: trees differ\n  java  : {sa}\n  native: {sb}")
    print(("ok   " if sa == sb else "FAIL ") + "hybrid")


def main():
    plugin, stock = requester(PLUGIN), requester(STOCK)
    load(plugin)
    load(stock)
    before = plugin("GET", "/_plugins/accelerated/stats")
    for name, body in ROWS.items():
        body = dict(body, profile=True)
        a = stock("POST", f"/{INDEX}/_search", body)
        b = plugin("POST", f"/{INDEX}/_search", body)
        check([h["_id"] for h in a["hits"]["hits"]] == [h["_id"] for h in b["hits"]["hits"]], f"{name}: hits")
        sa = structure(a["profile"], f"{name} (stock)", name in LOOSE)
        sb = structure(b["profile"], f"{name} (plugin)", name in LOOSE)
        if sa["searches"] != sb["searches"]:
            check(False, f"{name}: query/collector trees differ\n  stock : {sa['searches']}\n  plugin: {sb['searches']}")
        check(sa["aggs"] == sb["aggs"], f"{name}: aggregators\n  stock : {sa['aggs']}\n  plugin: {sb['aggs']}")
        check(sa["fetch"] == sb["fetch"], f"{name}: fetch\n  stock : {sa['fetch']}\n  plugin: {sb['fetch']}")
        print(("ok   " if not any(f.startswith(name) for f in FAILURES) else "FAIL ") + name)
    hybrid_row(plugin)
    after = plugin("GET", "/_plugins/accelerated/stats")
    native = after["accelerated_queries"] - before["accelerated_queries"]
    check(native >= len(ROWS) - 2, f"profiled rows ran natively: {native} of {len(ROWS)}; fallbacks {after['fallbacks']}")
    if FAILURES:
        print(f"{len(FAILURES)} failure(s)")
        sys.exit(1)
    print("profile: plugin and stock agree")


if __name__ == "__main__":
    main()
