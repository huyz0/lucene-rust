#!/usr/bin/env python3
"""Nested indexes on a plugin node against a stock OpenSearch node.

  * BP reorder: a nested index with index.lucene_rust.bp_reorder.* on is force-merged on the
    plugin node; its documents are renumbered (the root documents' `_doc` order differs from
    stock's insertion order), and every nested query finds and scores the same hits as stock.
  * Realtime: a nested index with index.lucene_rust.realtime_search and no refresh answers
    nested queries with the documents indexed before each search -- the same hits as a stock
    index refreshed after every bulk -- natively (the plugin's live searches count them).

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_nested_stock.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_join_stock import requester, check, same, shape, FAILURES  # noqa: E402

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
WORDS = "alpha beta gamma delta epsilon zeta eta theta".split()
MAPPING = {"properties": {
    "title": {"type": "text"},
    "tag": {"type": "keyword"},
    "comments": {"type": "nested", "properties": {
        "body": {"type": "text"},
        "stars": {"type": "integer"},
    }},
}}


def docs(seed, n, prefix):
    r = random.Random(seed)
    out = []
    for i in range(n):
        out.append((f"{prefix}{i}", {
            "title": " ".join(r.choice(WORDS) for _ in range(r.randint(1, 5))),
            "tag": f"t{i % 7}",
            "comments": [{"body": " ".join(r.choice(WORDS) for _ in range(r.randint(1, 6))),
                          "stars": r.randint(1, 5)} for _ in range(r.randint(0, 4))],
        }))
    return out


def bulk(req, index, items):
    lines = []
    for _id, src in items:
        lines.append(json.dumps({"index": {"_index": index, "_id": _id}}))
        lines.append(json.dumps(src))
    out = req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
    check(out.get("errors") is False, f"{index}: bulk indexed")


def nested(q, mode):
    return {"nested": {"path": "comments", "score_mode": mode, "query": q}}


def rows():
    body = {"match": {"comments.body": "alpha beta"}}
    q = [(f"nested {m}", {"query": nested(body, m)}) for m in ("avg", "max", "sum", "min", "none")]
    q.append(("nested bool", {"query": nested({"bool": {"must": [{"match": {"comments.body": "gamma"}}],
                                                        "filter": [{"range": {"comments.stars": {"gte": 3}}}]}}, "max")}))
    q.append(("nested in bool", {"query": {"bool": {
        "should": [nested(body, "avg"), {"match": {"title": "delta"}}],
        "must_not": [nested({"term": {"comments.body": "zeta"}}, "none")]}}}))
    q.append(("nested count", {"size": 0, "track_total_hits": True, "query": nested(body, "sum")}))
    q.append(("nested filter", {"query": {"bool": {"filter": [nested({"term": {"comments.stars": 5}}, "none")],
                                                    "must": [{"match": {"title": "eta"}}]}}}))
    for _, b in q:
        b.setdefault("size", 50)
    return q


def as_set(resp):
    """Every hit, as (score, id) ascending: documents renumbered break score ties differently."""
    return resp["hits"]["total"], sorted((round(h["_score"] or 0, 4), h["_id"]) for h in resp["hits"]["hits"])


def compare(label, index, plugin, stock, renumbered=False):
    for name, body in rows():
        url = f"/{index}/_search?request_cache=false"
        if renumbered:
            body = dict(body, size=10000, track_total_hits=True)
            a, b = as_set(plugin("POST", url, body)), as_set(stock("POST", url, body))
            diff = None if a == b else f"total {a[0]} vs {b[0]}, {len(a[1])} vs {len(b[1])} hits"
        else:
            diff = same(shape(plugin("POST", url, body)), shape(stock("POST", url, body)))
        check(diff is None, f"{label} [{name}]: plugin differs from stock: {diff}")
        print(f"  {label} [{name}]: {'ok' if diff is None else 'DIFF'}")


def order(req, index):
    out = req("POST", f"/{index}/_search?request_cache=false",
              {"size": 10000, "sort": ["_doc"], "query": {"match_all": {}}})
    return [h["_id"] for h in out["hits"]["hits"]]


def bp(plugin, stock):
    index = "nbp"
    items = docs(5, 3000, "d")
    for req, extra in ((plugin, {"index.lucene_rust.bp_reorder.enabled": True,
                                 "index.lucene_rust.bp_reorder.min_doc_freq": 2,
                                 "index.lucene_rust.bp_reorder.min_partition_size": 4,
                                 "index.lucene_rust.bp_reorder.min_merge_docs": 1}), (stock, {})):
        req("DELETE", f"/{index}")
        req("PUT", f"/{index}", {"settings": {"number_of_shards": 1, "number_of_replicas": 0,
                                              "refresh_interval": -1, **extra}, "mappings": MAPPING})
        for part in range(3):
            bulk(req, index, items[part * 1000:(part + 1) * 1000])
            req("POST", f"/{index}/_refresh")
        req("POST", f"/{index}/_forcemerge?max_num_segments=1")
        req("POST", f"/{index}/_refresh")
    on, off = order(plugin, index), order(stock, index)
    check(sorted(on) == sorted(off) and len(on) == 3000, "bp: every root document survives the merge")
    check(on != off, "bp: the forced merge renumbered the blocks")
    segs = plugin("GET", f"/{index}/_segments")["indices"][index]["shards"]["0"][0]["segments"]
    check(len(segs) == 1, f"bp: one segment ({len(segs)})")
    compare("bp", index, plugin, stock, renumbered=True)


def realtime(plugin, stock):
    index = "nrt"
    for req, extra in ((plugin, {"index.lucene_rust.realtime_search": True,
                                 "index.lucene_rust.realtime_max_staleness": "0ms"}), (stock, {})):
        req("DELETE", f"/{index}")
        req("PUT", f"/{index}", {"settings": {"number_of_shards": 1, "number_of_replicas": 0,
                                              "refresh_interval": -1, **extra}, "mappings": MAPPING})
    items = docs(9, 1200, "r")
    before = plugin("GET", "/_plugins/lucene_rust/stats")["live_queries"]
    for rnd in range(4):
        part = items[rnd * 300:(rnd + 1) * 300]
        # Updates of earlier blocks too: the old block deleted, the new one added whole.
        if rnd:
            part = part + [(i, dict(s, title=s["title"] + " omega")) for i, s in items[:rnd * 20]]
        bulk(plugin, index, part)
        bulk(stock, index, part)
        stock("POST", f"/{index}/_refresh")
        compare(f"realtime round {rnd}", index, plugin, stock)
        if rnd == 1:
            plugin("POST", f"/{index}/_refresh")
    after = plugin("GET", "/_plugins/lucene_rust/stats")["live_queries"]
    check(after - before >= 3 * len(rows()), f"realtime: nested searches ran live ({after - before})")


def main():
    plugin, stock = requester(PLUGIN), requester(STOCK)
    bp(plugin, stock)
    realtime(plugin, stock)
    print(f"verify_nested_stock: {len(FAILURES)} failures")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
