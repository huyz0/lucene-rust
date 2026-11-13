#!/usr/bin/env python3
"""Realtime k-NN visibility on the plugin path: vectors written after the last refresh must be
found by a k-NN search with no refresh, for each method, unfiltered, filtered and nested.

    check_knn_realtime.py URL

Each case: 2000 docs are indexed and flushed (IVF gets clusters), refresh is turned off, then 100
more are written; a k=1 search for each new doc's own vector must return that doc, and a new doc's
_source must hold its vector (k-NN derived source puts it back from the buffer). Prints one line
per case and exits non-zero when any case misses.
"""
import json
import random
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1]
DIM = 32


def req(method, path, body=None, ndjson=False):
    data = None
    headers = {}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:600]}")


METHODS = {
    "lucene_hnsw": {"method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"}},
    "lucene_sq": {"method": {"name": "hnsw", "engine": "lucene", "space_type": "l2",
                             "parameters": {"encoder": {"name": "sq", "parameters": {"bits": 7}}}}},
    "extended_hnsw_rabitq": {"method": {"name": "hnsw", "engine": "extended", "space_type": "l2",
                                    "parameters": {"quantizer": "rabitq"}}},
    "extended_hnsw_erq4": {"method": {"name": "hnsw", "engine": "extended", "space_type": "l2",
                                  "parameters": {"quantizer": "erq4"}}},
    "ivf": {"method": {"name": "ivf", "engine": "extended", "space_type": "l2",
                       "parameters": {"target_cluster_size": 64}}},
    "ivf_auto": {"method": {"name": "ivf", "engine": "extended", "space_type": "l2",
                            "parameters": {"quantizer": "auto"}}},
    "default": {},
}


def vec(rng):
    return [round(rng.gauss(0, 1), 4) for _ in range(DIM)]


def run(name, method, nested, sort, rng):  # sort: realtime refuses an index sort
    index = f"rt_{name}{'_nested' if nested else ''}{'_sorted' if sort else ''}"
    try:
        req("DELETE", f"/{index}")
    except RuntimeError:
        pass
    field = {"type": "knn_vector", "dimension": DIM, **method}
    props = {"tenant": {"type": "keyword"}}
    if nested:
        props["parts"] = {"type": "nested", "properties": {"emb": field}}
    else:
        props["emb"] = field
    settings = {"number_of_shards": 1, "number_of_replicas": 0, "index.knn": True,
                "index.search.realtime.enabled": True, "index.search.realtime.max_staleness": "0ms"}
    if sort:
        settings.update({"index.sort.field": "tenant"})
    req("PUT", f"/{index}", {"settings": settings, "mappings": {"properties": props}})
    docs = {}

    def write(lo, hi):
        lines = []
        for i in range(lo, hi):
            v = vec(rng)
            docs[i] = v
            src = {"tenant": f"t{i % 5}"}
            if nested:
                src["parts"] = [{"emb": v}, {"emb": vec(rng)}]
            else:
                src["emb"] = v
            lines += [json.dumps({"index": {"_index": index, "_id": str(i)}}), json.dumps(src)]
        r = req("POST", "/_bulk", ("\n".join(lines) + "\n").encode(), ndjson=True)
        if r["errors"]:
            raise RuntimeError(json.dumps(r["items"][0])[:500])

    write(0, 2000)
    req("POST", f"/{index}/_flush")
    req("PUT", f"/{index}/_settings", {"index": {"refresh_interval": "-1"}})
    write(2000, 2100)
    missed = {"unfiltered": 0, "filtered": 0}
    for i in range(2000, 2100):
        for kind in missed:
            clause = {"vector": docs[i], "k": 1}
            if kind == "filtered":
                clause["filter"] = {"term": {"tenant": f"t{i % 5}"}}
            q = {"knn": {"emb": clause}}
            if nested:
                q = {"nested": {"path": "parts", "query": {"knn": {"parts.emb": clause}}}}
            hits = req("POST", f"/{index}/_search", {"size": 1, "_source": False, "query": q})["hits"]["hits"]
            if not hits or hits[0]["_id"] != str(i):
                missed[kind] += 1
    # Derived source: an unrefreshed document's _source holds its vectors, read from the buffer.
    sources = {}
    for i in range(2000, 2100, 7):
        hit = req("POST", f"/{index}/_search", {"size": 1, "query": {"ids": {"values": [str(i)]}}})["hits"]["hits"]
        sources[i] = hit[0]["_source"] if hit else None
    bad_source = 0
    for i, src in sources.items():
        got = None
        if src:
            got = src["parts"][0].get("emb") if nested else src.get("emb")
        if not isinstance(got, list) or any(abs(a - b) > 1e-6 for a, b in zip(got, docs[i])):
            bad_source += 1
    missed["source"] = bad_source
    ok = not any(missed.values())
    print(f"{'ok  ' if ok else 'MISS'} {index}: missed of 100 {missed}", flush=True)
    req("DELETE", f"/{index}")
    return ok


def main():
    rng = random.Random(7)
    only = sys.argv[2:] or list(METHODS)
    good = True
    for name in only:
        try:
            good &= run(name, METHODS[name], False, False, rng)
        except RuntimeError as e:
            good = False
            print(f"FAIL {name}: {e}", flush=True)
    for name, nested, sort in (("lucene_hnsw", True, False), ("ivf", True, False)):
        try:
            good &= run(name, METHODS[name], nested, sort, rng)
        except RuntimeError as e:
            good = False
            print(f"FAIL {name} nested={nested} sorted={sort}: {e}", flush=True)
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
