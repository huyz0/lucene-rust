#!/usr/bin/env python3
"""Radial and nested k-NN on a plugin node against a stock OpenSearch node.

The same seeded documents go to both nodes (one shard, force-merged to one segment, graphs built with
a large ef so the approximate search is effectively exact on these sizes). For every space type:

  * radial: `max_distance` and `min_score` queries, plain, filtered, with a size cap, and at
    thresholds that keep none and all of the documents;
  * nested: a `knn` on a nested field (parents ranked by their best child), the same inside a
    `nested` query with each score_mode, with a filter on the nested documents, with a filter on the
    parents, with `expand_nested_docs`, with inner_hits, and radial on the nested field.

Each query must return the same hits with the same scores on both nodes (ids as sets where
scores tie, scores to 1e-5 relative).

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_knn_stock.py
"""
import json
import os
import random
import sys
import urllib.error
import urllib.request

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
DIM = 8
N = int(os.environ.get("KNN_DOCS", "300"))
QUERIES = int(os.environ.get("KNN_QUERIES", "6"))
SEED = int(os.environ.get("SEED", "11"))
SPACES = ("l2", "cosinesimil", "innerproduct")
FAILURES = []


def requester(base):
    def req(method, path, body=None, ndjson=False):
        data, headers = None, {}
        if body is not None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
        r = urllib.request.Request(base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(r, timeout=300) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
    return req


def vec(r):
    return [round(r.gauss(0, 1), 3) for _ in range(DIM)]


def field(space):
    return {"type": "knn_vector", "dimension": DIM, "method": {
        "name": "hnsw", "space_type": space, "engine": "lucene",
        "parameters": {"m": 16, "ef_construction": 512}}}


def settings():
    return {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0,
                      "refresh_interval": -1, "knn.algo_param.ef_search": 512}}


def flat_docs(r):
    return [(f"d{i}", {"v": vec(r), "cat": "abc"[i % 3], "n": i}) for i in range(N)]


def nested_docs(r):
    out = []
    for i in range(N // 3):
        out.append((f"p{i}", {"cat": "abc"[i % 3], "n": i, "items": [
            {"v": vec(r), "tag": "xyz"[j % 3]} for j in range(r.randint(0, 5))]}))
    return out


def load(req, index, mapping, docs):
    req("DELETE", f"/{index}")
    st, resp = req("PUT", f"/{index}", {"settings": settings(), "mappings": mapping})
    if st != 200:
        raise RuntimeError(f"create {index}: {st} {json.dumps(resp)[:300]}")
    lines = []
    for _id, src in docs:
        lines += [json.dumps({"index": {"_id": _id}}), json.dumps(src)]
    st, resp = req("POST", f"/{index}/_bulk?refresh=true", "\n".join(lines) + "\n", ndjson=True)
    if st != 200 or resp.get("errors"):
        raise RuntimeError(f"bulk {index}: {json.dumps(resp)[:300]}")
    req("POST", f"/{index}/_forcemerge?max_num_segments=1")
    req("POST", f"/{index}/_refresh")


def hits_of(resp):
    out = []
    for h in resp["hits"]["hits"]:
        inner = {}
        for name, ih in (h.get("inner_hits") or {}).items():
            inner[name] = sorted((x["_nested"]["offset"], round(x["_score"] or 0, 4))
                                 for x in ih["hits"]["hits"])
        out.append((h["_id"], h["_score"], inner))
    return out


def close(a, b):
    return abs(a - b) <= 1e-5 * max(1.0, abs(a), abs(b))


def compare(what, a, b):
    """a: plugin response (status, body), b: stock."""
    (sa, ra), (sb, rb) = a, b
    if sa != sb:
        FAILURES.append(what)
        print(f"FAIL {what}: status plugin {sa} stock {sb}\n  plugin: {json.dumps(ra)[:300]}\n  stock:  {json.dumps(rb)[:300]}")
        return
    if sa != 200:
        return
    ha, hb = hits_of(ra), hits_of(rb)
    ta, tb = ra["hits"]["total"], rb["hits"]["total"]
    problems = []
    if ta != tb:
        problems.append(f"total plugin {ta} stock {tb}")
    ia, ib = {h[0]: h for h in ha}, {h[0]: h for h in hb}
    if set(ia) != set(ib):
        problems.append(f"ids only plugin {sorted(set(ia) - set(ib))} only stock {sorted(set(ib) - set(ia))}")
    for i in set(ia) & set(ib):
        if not close(ia[i][1] or 0, ib[i][1] or 0):
            problems.append(f"{i}: score plugin {ia[i][1]} stock {ib[i][1]}")
        if ia[i][2] != ib[i][2]:
            problems.append(f"{i}: inner_hits plugin {ia[i][2]} stock {ib[i][2]}")
    if not problems and [h[0] for h in ha] != [h[0] for h in hb]:
        # an order that differs where the scores tie is not a difference
        sa_ = [h[1] for h in ha]
        sb_ = [h[1] for h in hb]
        if not all(close(x, y) for x, y in zip(sa_, sb_)):
            problems.append("order differs")
    if problems:
        FAILURES.append(what)
        print(f"FAIL {what}: " + "; ".join(problems[:4]))


def both(plugin, stock, what, index, body):
    compare(what, plugin("POST", f"/{index}/_search", body), stock("POST", f"/{index}/_search", body))


def threshold_values(space, hits_scores):
    return hits_scores


def radial(plugin, stock, space, r):
    index = f"knn_radial_{space}"
    mapping = {"properties": {"v": field(space), "cat": {"type": "keyword"}, "n": {"type": "integer"}}}
    docs = flat_docs(r)
    for req in (plugin, stock):
        load(req, index, mapping, docs)
    for qi in range(QUERIES):
        q = vec(r)
        # the score range this query sees, to choose thresholds that keep a few documents,
        # about half, none, and all
        st, probe = stock("POST", f"/{index}/_search", {"size": N, "query": {"knn": {"v": {"vector": q, "k": N}}}})
        scores = sorted((h["_score"] for h in probe["hits"]["hits"]), reverse=True)
        if not scores:
            continue
        cuts = {"few": scores[min(3, len(scores) - 1)], "half": scores[len(scores) // 2],
                "none": scores[0] * 2 + 1, "all": scores[-1] / 2}
        for name, ms in cuts.items():
            both(plugin, stock, f"{space} radial min_score {name} q{qi}", index,
                 {"size": N, "query": {"knn": {"v": {"vector": q, "min_score": ms}}}})
            both(plugin, stock, f"{space} radial min_score {name} filtered q{qi}", index,
                 {"size": N, "query": {"knn": {"v": {"vector": q, "min_score": ms,
                                                       "filter": {"term": {"cat": "a"}}}}}})
            both(plugin, stock, f"{space} radial min_score {name} size5 q{qi}", index,
                 {"size": 5, "query": {"knn": {"v": {"vector": q, "min_score": ms}}}})
        # max_distance is in the space's own distance: l2 squared distance, cosine distance
        # (1 - cos), inner product (negated)
        dists = {"l2": [1 / s - 1 for s in scores],
                 "cosinesimil": [2 - 2 * s for s in scores],
                 "innerproduct": [(1 - 1 / s) if s < 1 else (1 - s) for s in scores]}[space]
        dists = sorted(dists)
        for name, md in {"few": dists[min(3, len(dists) - 1)], "half": dists[len(dists) // 2],
                         "none": dists[0] - 1.0, "all": dists[-1] + 1.0}.items():
            both(plugin, stock, f"{space} radial max_distance {name} q{qi}", index,
                 {"size": N, "query": {"knn": {"v": {"vector": q, "max_distance": md}}}})
            both(plugin, stock, f"{space} radial max_distance {name} filtered q{qi}", index,
                 {"size": N, "query": {"knn": {"v": {"vector": q, "max_distance": md,
                                                       "filter": {"term": {"cat": "b"}}}}}})
        # k-NN with a filter and a bool around it, and a script-free compound
        both(plugin, stock, f"{space} radial inside bool q{qi}", index,
             {"size": N, "query": {"bool": {"must": [{"knn": {"v": {"vector": q, "min_score": cuts["half"]}}}],
                                            "filter": [{"range": {"n": {"gte": 100}}}]}}})


def nested(plugin, stock, space, r):
    index = f"knn_nested_{space}"
    mapping = {"properties": {
        "cat": {"type": "keyword"}, "n": {"type": "integer"},
        "items": {"type": "nested", "properties": {"v": field(space), "tag": {"type": "keyword"}}}}}
    docs = nested_docs(r)
    for req in (plugin, stock):
        load(req, index, mapping, docs)
    for qi in range(QUERIES):
        q = vec(r)
        knn = {"vector": q, "k": 10}
        both(plugin, stock, f"{space} nested knn top-level q{qi}", index,
             {"size": 10, "query": {"knn": {"items.v": knn}}})
        both(plugin, stock, f"{space} nested knn top-level k100 size100 q{qi}", index,
             {"size": 100, "query": {"knn": {"items.v": {"vector": q, "k": 100}}}})
        both(plugin, stock, f"{space} nested knn expand_nested_docs q{qi}", index,
             {"size": 10, "query": {"knn": {"items.v": {**knn, "expand_nested_docs": True}}},
              "_source": False})
        both(plugin, stock, f"{space} nested knn filter on nested q{qi}", index,
             {"size": 10, "query": {"knn": {"items.v": {**knn, "filter": {"nested": {
                 "path": "items", "query": {"term": {"items.tag": "x"}}}}}}}})
        both(plugin, stock, f"{space} nested knn filter on parent q{qi}", index,
             {"size": 10, "query": {"knn": {"items.v": {**knn, "filter": {"term": {"cat": "b"}}}}}})
        for mode in ("max", "avg", "sum", "min", "none"):
            both(plugin, stock, f"{space} nested query {mode} q{qi}", index,
                 {"size": 10, "query": {"nested": {"path": "items", "score_mode": mode,
                                                    "query": {"knn": {"items.v": knn}}}}})
        both(plugin, stock, f"{space} nested query inner_hits q{qi}", index,
             {"size": 10, "query": {"nested": {"path": "items", "inner_hits": {"size": 3},
                                                "query": {"knn": {"items.v": knn}}}}})
        both(plugin, stock, f"{space} nested query tag filter q{qi}", index,
             {"size": 10, "query": {"nested": {"path": "items", "query": {"knn": {"items.v": {
                 **knn, "filter": {"term": {"items.tag": "y"}}}}}}}})
        both(plugin, stock, f"{space} nested knn inside bool q{qi}", index,
             {"size": 10, "query": {"bool": {"must": [{"knn": {"items.v": knn}}],
                                             "filter": [{"range": {"n": {"gte": 20}}}]}}})
        st, probe = stock("POST", f"/{index}/_search", {"size": 100, "query": {"knn": {"items.v": {"vector": q, "k": 100}}}})
        scores = sorted((h["_score"] for h in probe["hits"]["hits"]), reverse=True)
        if not scores:
            continue
        for name, ms in {"few": scores[min(2, len(scores) - 1)], "half": scores[len(scores) // 2],
                         "none": scores[0] * 2 + 1}.items():
            both(plugin, stock, f"{space} nested radial min_score {name} q{qi}", index,
                 {"size": 100, "query": {"knn": {"items.v": {"vector": q, "min_score": ms}}}})
            both(plugin, stock, f"{space} nested radial in nested query {name} q{qi}", index,
                 {"size": 100, "query": {"nested": {"path": "items", "inner_hits": {"size": 5},
                                                     "query": {"knn": {"items.v": {"vector": q, "min_score": ms}}}}}})


def main():
    plugin, stock = requester(PLUGIN), requester(STOCK)
    r = random.Random(SEED)
    for space in SPACES:
        radial(plugin, stock, space, r)
        nested(plugin, stock, space, r)
        print(f"{space}: done, {len(FAILURES)} failures so far")
    print("knn stock differences:", len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
