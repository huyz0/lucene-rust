#!/usr/bin/env python3
"""parent-join and nested queries on a plugin node against a stock OpenSearch node.

Loads the same documents into both nodes (`extended_matrix.create_join` /
`load_join`, several segments and some deletes), runs `extended_matrix.join_rows()`
plus inner-hits rows on both, and requires the same hits, scores (1e-5) and totals.
The plugin node must answer every join row natively (its stats count them).

With BENCH_QUESTIONS set it then loads that many questions into `pjb` on both
nodes, force-merges nothing, and reports each join row's median latency
(`took` and wall) over BENCH_ROUNDS searches per node.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_join_stock.py
"""
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extended_matrix  # noqa: E402

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
FAILURES = []


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


def shape(resp):
    hits = resp["hits"]
    out = {
        "total": hits.get("total"),
        "max_score": hits.get("max_score"),
        "hits": [(h["_id"], h.get("_score"), h.get("sort"), inner(h)) for h in hits["hits"]],
    }
    if "aggregations" in resp:
        out["aggs"] = resp["aggregations"]
    return out


def inner(hit):
    ih = hit.get("inner_hits")
    if not ih:
        return None
    return {k: [(x["_id"], x.get("_score")) for x in v["hits"]["hits"]] for k, v in sorted(ih.items())}


def close(a, b):
    if a is None or b is None:
        return a == b
    return abs(a - b) <= 1e-5 * max(1.0, abs(b))


def same(a, b):
    if a["total"] != b["total"]:
        return f"total {a['total']} vs {b['total']}"
    if a.get("aggs") != b.get("aggs"):
        return f"aggs {a.get('aggs')} vs {b.get('aggs')}"
    if not close(a["max_score"], b["max_score"]):
        return f"max_score {a['max_score']} vs {b['max_score']}"
    if len(a["hits"]) != len(b["hits"]):
        return f"{len(a['hits'])} hits vs {len(b['hits'])}"
    for x, y in zip(a["hits"], b["hits"]):
        if x[0] != y[0] or not close(x[1], y[1]) or x[2] != y[2] or x[3] != y[3]:
            return f"hit {x} vs {y}"
    return None


def rows():
    q = list(extended_matrix.join_rows())
    q.append(("has_child inner_hits", {"query": {"has_child": {
        "type": "answer", "score_mode": "max", "query": {"match": {"body": "alpha"}},
        "inner_hits": {"size": 2}}}}, "native"))
    q.append(("has_parent inner_hits", {"query": {"has_parent": {
        "parent_type": "question", "score": True, "query": {"match": {"body": "beta"}},
        "inner_hits": {}}}}, "native"))
    return q


def native_count(req):
    return req("GET", "/_plugins/lucene_rust/stats")["native_queries"]


def compare(index, plugin, stock):
    for name, body, expect in rows():
        url = f"/{index}/_search?request_cache=false"
        before = native_count(plugin)
        got = shape(plugin("POST", url, body))
        after = native_count(plugin)
        want = shape(stock("POST", url, body))
        diff = same(got, want)
        check(diff is None, f"{index} [{name}]: plugin differs from stock: {diff}")
        if expect == "native":
            check(after - before >= 1, f"{index} [{name}]: did not run natively")
        print(f"  {index} [{name}]: {'ok' if diff is None else 'DIFF'}, native {after - before}")


def bench(index, plugin, stock, rounds):
    print(f"bench {index}: median over {rounds} searches (ms): plugin took/wall vs stock took/wall")
    out = []
    for name, body, _ in extended_matrix.join_rows():
        res = {}
        for label, req in (("plugin", plugin), ("stock", stock)):
            url = f"/{index}/_search?request_cache=false"
            for _ in range(3):
                req("POST", url, body)
            took, wall = [], []
            for _ in range(rounds):
                t = time.perf_counter()
                r = req("POST", url, body)
                wall.append((time.perf_counter() - t) * 1000)
                took.append(r["took"])
            res[label] = (statistics.median(took), statistics.median(wall))
        p, s = res["plugin"], res["stock"]
        print(f"  [{name}]: plugin {p[0]:.1f}/{p[1]:.2f}  stock {s[0]:.1f}/{s[1]:.2f}  speedup {s[1] / p[1]:.2f}x")
        out.append({"row": name, "plugin_took": p[0], "plugin_wall": p[1], "stock_took": s[0], "stock_wall": s[1]})
    return out


def main():
    plugin, stock = requester(PLUGIN), requester(STOCK)
    for req in (plugin, stock):
        req("DELETE", "/pj")
        extended_matrix.create_join(req, "pj")
        extended_matrix.load_join(req, check, "pj", 400, 21)
    compare("pj", plugin, stock)
    questions = int(os.environ.get("BENCH_QUESTIONS", "0"))
    if questions:
        for req in (plugin, stock):
            req("DELETE", "/pjb")
            extended_matrix.create_join(req, "pjb")
            extended_matrix.load_join(req, check, "pjb", questions, 7)
        compare("pjb", plugin, stock)
        result = bench("pjb", plugin, stock, int(os.environ.get("BENCH_ROUNDS", "30")))
        if os.environ.get("BENCH_OUT"):
            with open(os.environ["BENCH_OUT"], "w") as f:
                json.dump(result, f, indent=1)
    print(f"verify_join_stock: {len(FAILURES)} failures")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
