#!/usr/bin/env python3
"""neural_sparse speed: the lucene-rust plugin against stock neural-search.

Starts the two nodes scripts/check-sparse-plugin.py starts (or reuses them
when they are up), indexes the same synthetic SPLADE-like corpus into each in
turn, and runs the same `neural_sparse` queries one at a time. Reports
indexing throughput, query latency (client side and `took`), and each node's
recall@10 against the exact top 10 computed here with numpy.

  scripts/bench-sparse-plugin.py [--docs N] [--queries N] [--merged] [--keep]

--merged also force-merges each index to one segment and measures again.
Needs Docker, numpy and the plugin zip (see check-sparse-plugin.py).
"""
import argparse, importlib.util, json, os, random, statistics, sys, time

import numpy as np
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("check", os.path.join(ROOT, "scripts", "check-sparse-plugin.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

VOCAB, SEED, INDEX = 30000, 20261104, "bench"
PARAMS = {"approximate_threshold": 10000, "n_postings": 2000}
BATCH, CLIENTS = 1000, 4


def vector(rng, nnz):
    v = {}
    while len(v) < nnz:
        t = min(int(rng.paretovariate(0.6)) - 1, VOCAB - 1)
        v[t] = round(rng.uniform(0.01, 3.5), 4)
    return v


def corpus(n, queries):
    rng = random.Random(SEED)
    docs = [vector(rng, rng.randint(60, 160)) for _ in range(n)]
    qs = [vector(rng, rng.randint(10, 30)) for _ in range(queries)]
    return docs, qs


def exact_top10(docs, qs):
    """The exact dot-product top 10 per query, over FeatureField-rounded weights."""
    lens = [len(d) for d in docs]
    doc = np.repeat(np.arange(len(docs), dtype=np.int32), lens)
    tok = np.fromiter((t for d in docs for t in d), dtype=np.int32, count=len(doc))
    w = np.fromiter((x for d in docs for x in d.values()), dtype=np.float32, count=len(doc))
    w = (w.view(np.uint32) >> 15 << 15).view(np.float32)
    order = np.argsort(tok, kind="stable")
    doc, tok, w = doc[order], tok[order], w[order]
    starts = np.searchsorted(tok, np.arange(VOCAB + 1))
    out = []
    for q in qs:
        s = np.zeros(len(docs), dtype=np.float32)
        for t, qw in q.items():
            a, b = starts[t], starts[t + 1]
            np.add.at(s, doc[a:b], np.float32(qw) * w[a:b])
        top = np.argsort(-s, kind="stable")[:10]
        out.append({f"d{i}" for i in top if s[i] > 0})
    return out


def index(port, docs, rust):
    settings = {"index.sparse": True, "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    if rust:
        settings["index.engine.accelerated"] = True
    mapping = {"properties": {"emb": {"type": "sparse_vector", "method": {"name": "seismic", "parameters": PARAMS}}}}
    try:
        check.req(port, "DELETE", f"/{INDEX}")
    except RuntimeError:
        pass
    check.req(port, "PUT", f"/{INDEX}", {"settings": settings, "mappings": mapping})
    bodies = []
    for b in range(0, len(docs), BATCH):
        lines = []
        for i in range(b, min(b + BATCH, len(docs))):
            lines.append(json.dumps({"index": {"_index": INDEX, "_id": f"d{i}"}}))
            lines.append(json.dumps({"emb": {str(t): x for t, x in docs[i].items()}}))
        bodies.append("\n".join(lines) + "\n")

    def bulk(body):
        r = check.req(port, "POST", "/_bulk", body, ndjson=True, timeout=3600)
        assert not r["errors"], json.dumps(r)[:2000]

    start = time.perf_counter()
    with ThreadPoolExecutor(CLIENTS) as pool:
        list(pool.map(bulk, bodies))
    check.req(port, "POST", f"/{INDEX}/_refresh", timeout=3600)
    elapsed = time.perf_counter() - start
    check.req(port, "POST", f"/{INDEX}/_flush", timeout=3600)
    return len(docs) / elapsed


def segments(port):
    segs = check.req(port, "GET", f"/{INDEX}/_segments")["indices"][INDEX]["shards"]["0"][0]["segments"]
    return sum(1 for v in segs.values() if v["search"])


def run_queries(port, qs, mp, passes):
    lat, took, hits = [], [], []
    for p in range(passes + 1):
        for q in qs:
            body = {"size": 10, "_source": False,
                    "query": {"neural_sparse": {"emb": {"query_tokens": {str(t): w for t, w in q.items()},
                                                         "method_parameters": mp}}}}
            t0 = time.perf_counter()
            r = check.req(port, "POST", f"/{INDEX}/_search?request_cache=false", body)
            dt = (time.perf_counter() - t0) * 1000
            if p == 0:
                hits.append({h["_id"] for h in r["hits"]["hits"]})
                continue  # warm-up pass
            lat.append(dt)
            took.append(r["took"])
    return lat, took, hits


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=200000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--merged", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    nodes = {"plugin": check.OURS, "stock": check.STOCK}
    try:
        for _, port in nodes.values():
            check.req(port, "GET", "/_cluster/health")
    except Exception:
        sys.argv = [sys.argv[0], "--keep"]
        image = f"lucene-rust-opensearch:{check.VERSION}"
        check.start(check.OURS[0], check.OURS[1], image)
        check.start(check.STOCK[0], check.STOCK[1], f"opensearchproject/opensearch:{check.VERSION}")
        for _, port in nodes.values():
            check.wait(port)

    print(f"corpus: {a.docs} docs, {a.queries} queries, vocabulary {VOCAB}, params {PARAMS}", flush=True)
    docs, qs = corpus(a.docs, a.queries)
    exact = exact_top10(docs, qs)

    rate = {}
    for name, (_, port) in nodes.items():
        rate[name] = index(port, docs, name == "plugin")
        print(f"{name}: indexed {rate[name]:.0f} docs/s, {segments(port)} segments", flush=True)

    settings = [{"top_n": 10, "k": 10, "heap_factor": 1.0}, {"top_n": 20, "k": 50, "heap_factor": 0.8}]
    layouts = ["natural"] + (["merged"] if a.merged else [])
    for layout in layouts:
        if layout == "merged":
            for _, port in nodes.values():
                check.req(port, "POST", f"/{INDEX}/_forcemerge?max_num_segments=1", timeout=3600)
                check.req(port, "POST", f"/{INDEX}/_refresh", timeout=3600)
        for mp in settings:
            row = {}
            for name, (_, port) in nodes.items():
                lat, took, hits = run_queries(port, qs, mp, a.passes)
                recall = statistics.mean(len(h & e) / max(len(e), 1) for h, e in zip(hits, exact))
                row[name] = (statistics.median(lat), pct(lat, 0.99), statistics.mean(took), recall)
            print(f"{layout} {mp} ({segments(nodes['plugin'][1])} / {segments(nodes['stock'][1])} segments)")
            for name, (p50, p99, tk, rc) in row.items():
                print(f"  {name:6} p50 {p50:6.2f} ms  p99 {p99:6.2f} ms  took {tk:5.2f} ms  recall@10 {rc:.3f}")
            print(f"  stock/plugin p50 {row['stock'][0] / row['plugin'][0]:.2f}x", flush=True)
    print(f"indexing plugin/stock {rate['plugin'] / rate['stock']:.2f}x")
    if not a.keep:
        import subprocess
        subprocess.run(["docker", "rm", "-f", check.OURS[0], check.STOCK[0]], capture_output=True)


if __name__ == "__main__":
    main()
