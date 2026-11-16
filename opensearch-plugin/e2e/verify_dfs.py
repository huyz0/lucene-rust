#!/usr/bin/env python3
"""dfs_query_then_fetch against a stock node: three shards whose term statistics differ a lot (documents
routed by a skewed key), so the aggregated statistics change every score. Hits, scores and totals must
be identical, and the plugin must report no fallback.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 python3 opensearch-plugin/e2e/verify_dfs.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node  # noqa: E402

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
INDEX = "dfs_shapes"
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()


def queries():
    m = lambda q: {"match": {"title": q}}
    return [
        ("match", m("alpha")),
        ("match or", m("alpha beta gamma")),
        ("match and", {"match": {"title": {"query": "alpha beta", "operator": "and"}}}),
        ("bool should", {"bool": {"should": [m("alpha"), m("zeta"), {"term": {"tag": "x"}}]}}),
        ("bool must filter", {"bool": {"must": [m("beta delta")], "filter": [{"term": {"tag": "y"}}]}}),
        ("bool must_not", {"bool": {"must": [m("alpha")], "must_not": [m("beta")]}}),
        ("phrase", {"match_phrase": {"title": "alpha beta"}}),
        ("multi_match", {"multi_match": {"query": "alpha tau", "fields": ["title", "body"]}}),
        ("multi_match best", {"multi_match": {"query": "alpha tau", "fields": ["title^2", "body"], "type": "best_fields"}}),
        ("query_string", {"query_string": {"query": "title:alpha OR body:gamma", "default_field": "title"}}),
        ("prefix", {"prefix": {"title": "al"}}),
        ("fuzzy", {"fuzzy": {"title": {"value": "alpa"}}}),
        ("wildcard", {"wildcard": {"title": "al*a"}}),
        ("term", {"term": {"title": "alpha"}}),
        ("terms", {"terms": {"title": ["alpha", "beta"]}}),
        ("dis_max", {"dis_max": {"queries": [m("alpha"), m("beta")], "tie_breaker": 0.3}}),
        ("boosted", {"bool": {"should": [{"match": {"title": {"query": "alpha", "boost": 3}}}, m("beta")]}}),
        ("constant_score", {"constant_score": {"filter": m("alpha"), "boost": 2}}),
        ("function_score", {"function_score": {"query": m("alpha"), "field_value_factor": {"field": "n", "missing": 1}}}),
        ("sort by score", None),
    ]


def same(a, b):
    """Equal scores in order, totals, and the same ids within each group of equal scores but the last
    (which the page may cut, as it may the first): documents with equal scores come in either node's own segment order."""
    if not (isinstance(a[0], list) and isinstance(b[0], list)):
        return a == b
    if a[1] != b[1] or [(h[1], h[2][:1] if h[2] else None) for h in a[0]] != [(h[1], h[2][:1] if h[2] else None) for h in b[0]]:
        return False
    keys = [(h[1], tuple(h[2] or ())[:-1]) for h in a[0]]
    edges = {keys[0], keys[-1]} if keys else set()
    for k in set(keys) - edges:
        if {h[0] for h in a[0] if (h[1], tuple(h[2] or ())[:-1]) == k} != {h[0] for h in b[0] if (h[1], tuple(h[2] or ())[:-1]) == k}:
            return False
    return True


def main():
    r = random.Random(5)
    plugin, stock = Node(PLUGIN, 120), Node(STOCK, 120)
    docs = []
    for i in range(6000):
        shard = r.choice([0, 0, 0, 1, 2]) if i % 3 else r.choice([1, 2])
        pool = WORDS[: 4 + 5 * shard]
        docs.append({"title": " ".join(r.choice(pool) for _ in range(r.randint(1, 6))),
                     "body": " ".join(r.choice(WORDS) for _ in range(r.randint(2, 10))),
                     "tag": r.choice(["x", "y", "z"]), "n": i % 17, "_route": f"r{shard}"})
    for n in (plugin, stock):
        n.req("DELETE", f"/{INDEX}")
        st, resp = n.req("PUT", f"/{INDEX}", {"settings": {"number_of_shards": 3, "number_of_replicas": 0},
                                              "mappings": {"properties": {"title": {"type": "text"}, "body": {"type": "text"},
                                                                          "tag": {"type": "keyword"}, "n": {"type": "integer"}}}})
        assert st == 200, resp
        lines = []
        for i, d in enumerate(docs):
            route = d.pop("_route")
            lines += [json.dumps({"index": {"_id": f"d{i}", "routing": route}}), json.dumps(d)]
        st, resp = n.req("POST", f"/{INDEX}/_bulk?refresh=true", ("\n".join(lines) + "\n").encode(), raw=True)
        assert st == 200 and not resp.get("errors")
        n.req("POST", f"/{INDEX}/_forcemerge?max_num_segments=1")
        for d in docs:
            d["_route"] = None
    st, stats = plugin.req("GET", "/_plugins/lucene_rust/stats")
    before = dict(stats.get("fallbacks", {}))
    bad = total = 0
    for name, q in queries():
        for extra in ({}, {"size": 25, "track_total_hits": True}, {"size": 10, "from": 5, "explain": False},
                      {"size": 20, "sort": ["_score", {"n": "desc"}]}):
            body = {"query": q, "_source": False, "size": 20, **extra} if q else {"query": {"match": {"title": "alpha beta"}}, "sort": ["_score"], "size": 20, "_source": False}
            total += 1
            res = []
            for n in (plugin, stock):
                st, resp = n.req("POST", f"/{INDEX}/_search?search_type=dfs_query_then_fetch", body)
                if st != 200:
                    res.append((st, resp.get("error", {}).get("type")))
                else:
                    res.append(([(h["_id"], h["_score"], h.get("sort")) for h in resp["hits"]["hits"]], resp["hits"]["total"]))
            if same(res[0], res[1]) is False:
                bad += 1
                a, b = res
                k = next((j for j, (x, y) in enumerate(zip(a[0], b[0])) if x != y), None) if isinstance(a[0], list) and isinstance(b[0], list) else None
                print(f"{name} {json.dumps(extra)[:60]}: plugin {str(a)[:150] if k is None else a[0][k]} stock {str(b)[:150] if k is None else b[0][k]}")
    st, stats = plugin.req("GET", "/_plugins/lucene_rust/stats")
    after = stats.get("fallbacks", {})
    new = {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0) > 0}
    print(f"dfs searches checked: {total}, mismatches: {bad}, plugin fallbacks: {json.dumps(new)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
