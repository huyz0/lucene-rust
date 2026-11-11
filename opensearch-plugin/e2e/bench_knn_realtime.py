#!/usr/bin/env python3
"""Realtime k-NN against refresh: k-NN search latency and recall while vectors are being indexed.

    bench_knn_realtime.py URL CONTAINER VECTORS.npy LABEL OUT.json MAPPING_JSON SETTINGS_JSON

VECTORS.npy is an (n, d) float32 array (shuffled). The last QUERIES rows are the queries; the first
PRELOAD rows are bulk-loaded and flushed before the measured phase, which then indexes the next
STREAM rows at INDEX_RATE docs/s from one bulk client (bulks of BULK docs) while SEARCHERS threads
send k-NN searches at SEARCH_RATE each (alternating unfiltered and filtered to one of TENANTS
tenants), and a probe indexes a far-away vector every PROBE_EVERY s and searches for it until it is
found (acknowledged write -> visible).

Recall@10 of each search is against the exact top 10 over every document acknowledged before the
search was sent (plus any later one it returned), so it measures freshness and index quality
together. After the stream, SEARCH_ONLY searches run with no indexing (and no refresh first), then
a refresh and the same searches again. Then MAXRATE_DOCS more rows are indexed unpaced by CLIENTS
clients with no searches: the indexing throughput.

Each document: {"emb": vector, "tenant": "t<i % TENANTS>"}. Index `knn` has one shard, no replica.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

BASE = ""
K = 10


def req(method, path, body=None, ndjson=False, timeout=600):
    data = None
    headers = {}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:800]}")


def env(name, default):
    return type(default)(os.environ.get(name, default))


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    return round(s[min(len(s) - 1, int(p * len(s)))], 2)


def cpu_seconds():
    nodes = req("GET", "/_nodes/stats/process")["nodes"]
    return next(iter(nodes.values()))["process"]["cpu"]["total_in_millis"] / 1000


def memory(container):
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                         capture_output=True, text=True).stdout.strip()
    n = next(iter(req("GET", "/_nodes/stats/jvm")["nodes"].values()))
    return {"rss": out.split("/")[0].strip(), "heap_used_mb": round(n["jvm"]["mem"]["heap_used_in_bytes"] / 2**20)}


def bulk_body(vecs, start, tenants):
    lines = []
    for i, v in enumerate(vecs):
        d = start + i
        lines.append(json.dumps({"index": {"_index": "knn", "_id": str(d)}}))
        lines.append(json.dumps({"emb": [round(float(x), 5) for x in v], "tenant": f"t{d % tenants}"}))
    return ("\n".join(lines) + "\n").encode()


def bulk(body):
    res = req("POST", "/_bulk", body, ndjson=True)
    if res["errors"]:
        first = next(i for i in res["items"] if "error" in next(iter(i.values())))
        raise RuntimeError(f"bulk error: {json.dumps(first)[:1500]}")


def knn_body(q, tenant=None, k=K):
    clause = {"vector": [round(float(x), 5) for x in q], "k": k}
    if tenant is not None:
        clause["filter"] = {"term": {"tenant": tenant}}
    return {"size": k, "_source": False, "query": {"knn": {"emb": clause}}}


def main():
    global BASE
    BASE, container, vectors, label, out, mapping, settings = sys.argv[1:8]
    mapping, settings = json.loads(mapping), json.loads(settings)
    data = np.load(vectors).astype(np.float32)
    nq = env("QUERIES", 200)
    preload, stream, maxrate = env("PRELOAD", 100000), env("STREAM", 100000), env("MAXRATE_DOCS", 50000)
    tenants, bulk_docs = env("TENANTS", 50), env("BULK", 500)
    queries = data[-nq:]
    docs = data[:preload + stream + maxrate]
    dim = docs.shape[1]
    try:
        req("DELETE", "/knn")
    except RuntimeError:
        pass
    s = {"number_of_shards": 1, "number_of_replicas": 0, "index.knn": True, "refresh_interval": "1s"}
    s.update(settings)
    props = {"emb": {"type": "knn_vector", "dimension": dim, **mapping}, "tenant": {"type": "keyword"}}
    req("PUT", "/knn", {"settings": s, "mappings": {"properties": props}})

    t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(env("CLIENTS", 4)) as ex:
        list(ex.map(lambda a: bulk(bulk_body(docs[a:a + 2000], a, tenants)), range(0, preload, 2000)))
    req("POST", "/knn/_flush")
    req("POST", "/knn/_refresh")
    preload_s = time.perf_counter() - t
    print(f"{label}: preloaded {preload} in {preload_s:.1f}s", flush=True)

    acked = [preload]          # every doc below this is acknowledged
    sent = [preload]           # every doc below this has been sent
    done = [False]
    records = []               # (query index, tenant or None, acked at send, sent at response, ids, ms, took)
    probe_ms, probe_missed, errors = [], [0], []
    rate, index_rate = env("SEARCH_RATE", 20.0), env("INDEX_RATE", 1500.0)
    lock = threading.Lock()
    phases = {}

    def searcher(sid):
        i, last = sid, 0.0
        while not done[0]:
            now = time.perf_counter()
            if last:
                time.sleep(max(0.0, last + 1.0 / rate - now))
            last = time.perf_counter()
            qi = (i * 7919) % nq
            tenant = f"t{qi % tenants}" if i % 2 else None
            a = acked[0]
            t0 = time.perf_counter()
            try:
                r = req("POST", "/knn/_search?request_cache=false&phase_took=true", knn_body(queries[qi], tenant))
                ms = (time.perf_counter() - t0) * 1000
                for ph, v in (r.get("phase_took") or {}).items():
                    phases.setdefault(ph, []).append(float(v))
                ids = [int(h["_id"]) for h in r["hits"]["hits"] if not h["_id"].startswith("p")]
                with lock:
                    records.append((qi, tenant, a, sent[0], ids, ms, r.get("took", 0)))
            except (RuntimeError, OSError) as e:
                errors.append(str(e)[:300])
            i += env("SEARCHERS", 2)

    def probe():
        n = 0
        while not done[0]:
            n += 1
            v = np.zeros(dim, dtype=np.float32)
            v[0], v[1] = 1000.0, float(n)
            try:
                req("PUT", f"/knn/_doc/p{n}", {"emb": v.tolist(), "tenant": "probe"})
                t0 = time.perf_counter()
                while True:
                    r = req("POST", "/knn/_search?request_cache=false", knn_body(v, "probe", 1))
                    if any(h["_id"] == f"p{n}" for h in r["hits"]["hits"]):
                        probe_ms.append((time.perf_counter() - t0) * 1000)
                        break
                    if time.perf_counter() - t0 > 30:
                        probe_missed[0] += 1
                        break
                    time.sleep(0.002)
            except (RuntimeError, OSError) as e:
                errors.append(f"probe {n}: {str(e)[:300]}")
            time.sleep(env("PROBE_EVERY", 0.5))

    cpu0 = cpu_seconds()
    t0 = time.perf_counter()
    side = concurrent.futures.ThreadPoolExecutor(1 + env("SEARCHERS", 2))
    tasks = [side.submit(probe)] + [side.submit(searcher, k) for k in range(env("SEARCHERS", 2))]
    try:
        for a in range(preload, preload + stream, bulk_docs):
            due = t0 + (a - preload) / index_rate
            if due > time.perf_counter():
                time.sleep(due - time.perf_counter())
            b = min(a + bulk_docs, preload + stream)
            body = bulk_body(docs[a:b], a, tenants)
            sent[0] = b
            bulk(body)
            acked[0] = b
    finally:
        done[0] = True
        for f in tasks:
            f.result()
    stream_s = time.perf_counter() - t0
    stream_cpu = cpu_seconds() - cpu0
    mem = memory(container)
    try:
        shards = req("GET", "/knn/_segments")["indices"]["knn"]["shards"]["0"][0]["segments"]
        segs = {"count": len(shards), "docs": sorted((v["num_docs"] for v in shards.values()), reverse=True)[:12]}
    except (RuntimeError, OSError, KeyError):
        segs = None
    try:
        plugin_stats = req("GET", "/_plugins/lucene_rust/stats")
    except (RuntimeError, OSError):
        plugin_stats = None
    print(f"{label}: streamed {stream} in {stream_s:.1f}s, cpu {stream_cpu:.1f}s, {len(records)} searches", flush=True)

    total = preload + stream
    dist = ((queries ** 2).sum(1)[:, None] - 2 * queries @ docs[:total].T + (docs[:total] ** 2).sum(1)[None, :])
    tenant_of = np.arange(total) % tenants

    def recall(qi, tenant, a, s_, ids):
        cand = np.arange(a)
        late = [d for d in ids if a <= d < s_]
        if late:
            cand = np.concatenate([cand, np.array(late)])
        if tenant is not None:
            cand = cand[tenant_of[cand] == int(tenant[1:])]
        dq = dist[qi, cand]
        top = cand[np.argpartition(dq, K)[:K]] if len(cand) > K else cand
        return len(set(ids) & set(top.tolist())) / K

    def summarize(recs):
        out = {}
        for name, flt in (("unfiltered", lambda r: r[1] is None), ("filtered", lambda r: r[1] is not None)):
            rs = [r for r in recs if flt(r)]
            ms = [r[5] for r in rs]
            out[name] = {"n": len(rs), "p50_ms": pct(ms, 0.5), "p90_ms": pct(ms, 0.9), "p99_ms": pct(ms, 0.99),
                         "took_mean": round(sum(r[6] for r in rs) / max(1, len(rs)), 2),
                         "recall": round(float(np.mean([recall(*r[:5]) for r in rs])), 4) if rs else None}
        return out

    def quiet(n):
        recs = []
        for i in range(n):
            qi = i % nq
            tenant = f"t{qi % tenants}" if i % 2 else None
            t1 = time.perf_counter()
            r = req("POST", "/knn/_search?request_cache=false", knn_body(queries[qi], tenant))
            ms = (time.perf_counter() - t1) * 1000
            ids = [int(h["_id"]) for h in r["hits"]["hits"] if not h["_id"].startswith("p")]
            recs.append((qi, tenant, total, total, ids, ms, r.get("took", 0)))
        return summarize(recs)

    n_quiet = env("SEARCH_ONLY", 400)
    quiet_live = quiet(n_quiet)
    req("POST", "/knn/_refresh")
    quiet_refreshed = quiet(n_quiet)

    cpu1 = cpu_seconds()
    t1 = time.perf_counter()
    start = total
    with concurrent.futures.ThreadPoolExecutor(env("CLIENTS", 4)) as ex:
        list(ex.map(lambda a: bulk(bulk_body(docs[a:min(a + bulk_docs, start + maxrate)], a, tenants)),
                    range(start, start + maxrate, bulk_docs)))
    max_s = time.perf_counter() - t1
    max_cpu = cpu_seconds() - cpu1
    result = {
        "label": label, "mapping": mapping, "settings": settings, "preload": preload, "preload_seconds": round(preload_s, 1),
        "stream": {"docs": stream, "seconds": round(stream_s, 1), "docs_per_second": round(stream / stream_s),
                   "cpu_seconds": round(stream_cpu, 1), "searches": len(records)},
        "during_indexing": summarize(records),
        "visible_ms": {"p50": pct(probe_ms, 0.5), "p90": pct(probe_ms, 0.9), "p99": pct(probe_ms, 0.99),
                       "max": pct(probe_ms, 1.0), "n": len(probe_ms), "missed": probe_missed[0]},
        "quiet_unrefreshed": quiet_live, "quiet_refreshed": quiet_refreshed,
        "max_rate": {"docs": maxrate, "docs_per_second": round(maxrate / max_s), "cpu_seconds": round(max_cpu, 1)},
        "phase_took_mean": {ph: round(sum(v) / len(v), 2) for ph, v in phases.items() if v},
        "plugin_stats_after_stream": plugin_stats,
        "segments_after_stream": segs,
        "memory_after_stream": mem, "errors": errors[:10], "error_count": len(errors),
    }
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps({k: result[k] for k in ("stream", "during_indexing", "visible_ms", "quiet_unrefreshed",
                                             "max_rate", "memory_after_stream", "error_count")}), flush=True)


if __name__ == "__main__":
    main()
