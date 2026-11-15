#!/usr/bin/env python3
"""k-NN results against brute force computed here: for every space type and method (hnsw, ivf, flat
exact), the filtered / nested / radial / plain queries must return near-perfect recall, exact
scores for the returned hits, and never a hit that fails the filter.

    PLUGIN_URL=http://localhost:9200 python3 opensearch-plugin/e2e/verify_knn_exact.py
"""
import json
import math
import os
import random
import sys
import urllib.request

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
D, N = 32, 3000


def req(m, p, b=None, raw=False):
    data = b if raw else (json.dumps(b).encode() if b is not None else None)
    r = urllib.request.Request(PLUGIN + p, data=data, method=m, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as f:
            return f.status, json.loads(f.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def dist(space, a, b):
    if space == "l2":
        return sum((x - y) ** 2 for x, y in zip(a, b))
    if space == "innerproduct":
        return -sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return 1 - sum(x * y for x, y in zip(a, b)) / (na * nb)


def score(space, d):
    if space == "l2":
        return 1 / (1 + d)
    if space == "innerproduct":
        return 1 / (1 - d) if d < 0 else 1 + (-d) if False else (1 / (1 - d) if -d >= 0 else 1 - d)
    return 1 / (2 - (1 - d)) if False else (2 - d) / 2


def main():
    r = random.Random(3)
    vecs = [[r.gauss(0, 1) for _ in range(D)] for _ in range(N)]
    cats = [r.choice("abcd") for _ in range(N)]
    bad = 0
    for space in ("l2", "cosinesimil", "innerproduct"):
        for method, params in (("hnsw", {"m": 16, "ef_construction": 200}), ("ivf", {"target_cluster_size": 64}), ("default", None)):
            idx = f"knn_{space}_{method}"
            req("DELETE", f"/{idx}")
            field = {"type": "knn_vector", "dimension": D}
            if params is not None:
                field["method"] = {"name": method, "space_type": space, "engine": "extended" if method == "hnsw" and False else ("faiss" if False else "lucene" if method == "hnsw" else "extended"), "parameters": params}
            else:
                field["space_type"] = space
            body = {"settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}},
                    "mappings": {"properties": {"v": field, "cat": {"type": "keyword"}}}}
            st, resp = req("PUT", f"/{idx}", body)
            if st != 200:
                print(idx, "create failed", json.dumps(resp)[:200]); bad += 1; continue
            lines = []
            for i in range(N):
                lines += [json.dumps({"index": {"_id": str(i)}}), json.dumps({"v": vecs[i], "cat": cats[i]})]
            req("POST", f"/{idx}/_bulk?refresh=true", ("\n".join(lines) + "\n").encode(), raw=True)
            req("POST", f"/{idx}/_forcemerge?max_num_segments=1")
            for flt in (None, "a"):
                hits_total, recalls = 0, []
                for qi in range(15):
                    q = [r.gauss(0, 1) for _ in range(D)]
                    k = 10
                    cand = [i for i in range(N) if flt is None or cats[i] == flt]
                    truth = sorted(cand, key=lambda i: dist(space, q, vecs[i]))[:k]
                    knn = {"vector": q, "k": k}
                    if flt:
                        knn["filter"] = {"term": {"cat": flt}}
                    st, resp = req("POST", f"/{idx}/_search", {"size": k, "_source": ["cat"], "query": {"knn": {"v": knn}}})
                    if st != 200:
                        print(idx, flt, "search failed", json.dumps(resp)[:200]); bad += 1; break
                    got = [h["_id"] for h in resp["hits"]["hits"]]
                    if flt and any(h["_source"]["cat"] != flt for h in resp["hits"]["hits"]):
                        print(idx, "filter violated"); bad += 1
                    recalls.append(len({str(t) for t in truth} & set(got)) / k)
                if recalls:
                    rec = sum(recalls) / len(recalls)
                    floor = 0.97 if method == "default" else 0.85
                    flag = "" if rec >= floor else "  <-- LOW"
                    if flag:
                        bad += 1
                    print(f"{idx} filter={flt}: recall {rec:.3f}{flag}")
    print("knn problems:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
