"""Many tenants on one node: an index per tenant against one shared index,
with and without the plugin's segment partitions.

Usage:
  bench_tenants.py BASE CONTAINER MODE LABEL OUT [--tenants N] [--docs N]

MODE is one of
  per_tenant   an index per tenant (`t-<k>`), one shard each
  shared       one index `shared` with a `tenant` keyword; searches filter on it
  partitioned  `shared` with `index.lucene_rust.partition_field: tenant` (plugin node only)

Tenant sizes follow a Zipf law (s = 1), so a few tenants are large and most
are small; documents arrive interleaved across tenants. Every mode sees the
same documents (seeded). Measured:
  - index creation time, bulk load docs/s, refresh time, segment count, store size;
  - heap after a full GC (jcmd GC.run in the container), RSS, and the node's
    own segment/translog memory stats;
  - per-tenant search latency for a few request shapes, split by tenant size
    (large, mid, small), `request_cache=false`;
  - re-indexing a slice of documents spread over tenants (docs/s, then refresh);
  - single-document updates with `?refresh=true` (the update-to-visible cost).
The result (JSON) goes to OUT."""
import bisect
import concurrent.futures
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

VOCAB = 50_000
CLIENTS = 4
BULK = 5_000
TAGS = 200

BASE = ""


def b36(n):
    s = ""
    while True:
        n, d = divmod(n, 36)
        s = "0123456789abcdefghijklmnopqrstuvwxyz"[d] + s
        if n == 0:
            return s


def zipf_cum(n, s):
    acc, out = 0.0, []
    for r in range(1, n + 1):
        acc += 1.0 / r ** s
        out.append(acc)
    return out


def corpus(tenants, docs):
    """[(tenant, id, source)] in arrival order, and each tenant's size."""
    rnd = random.Random(7)
    words = ["t" + b36(i) for i in range(VOCAB)]
    wcum = zipf_cum(VOCAB, 1.07)
    tcum = zipf_cum(tenants, 1.0)
    out, sizes = [], [0] * tenants
    for i in range(docs):
        k = bisect.bisect_left(tcum, rnd.random() * tcum[-1])
        sizes[k] += 1
        out.append((k, "%d" % i, {
            "tenant": "u%d" % k,
            "body": " ".join(rnd.choices(words, cum_weights=wcum, k=rnd.randint(20, 80))),
            "tag": "g%d" % rnd.randrange(TAGS),
            "num": rnd.randrange(1_000_000),
            "ts": 1_700_000_000_000 + rnd.randrange(365 * 24 * 60) * 60_000,
        }))
    return out, sizes


def req(method, path, body=None, ndjson=False, timeout=1800):
    data, headers = None, {}
    if body is not None:
        data = body if isinstance(body, bytes) else (body.encode() if isinstance(body, str) else json.dumps(body).encode())
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:500]}")


MAPPING = {"properties": {
    "tenant": {"type": "keyword"}, "body": {"type": "text"}, "tag": {"type": "keyword"},
    "num": {"type": "long"}, "ts": {"type": "date"},
}}


def index_of(mode, k):
    return "t-%d" % k if mode == "per_tenant" else "shared"


def bulk_bodies(mode, docs):
    for start in range(0, len(docs), BULK):
        lines = []
        for k, i, src in docs[start:start + BULK]:
            lines.append(json.dumps({"index": {"_index": index_of(mode, k), "_id": i}}))
            lines.append(json.dumps(src, separators=(",", ":")))
        yield ("\n".join(lines) + "\n").encode()


def send_all(bodies):
    def send(body):
        res = req("POST", "/_bulk", body, ndjson=True)
        if res["errors"]:
            bad = next(it for it in res["items"] if "error" in next(iter(it.values())))
            raise RuntimeError(f"bulk had errors: {json.dumps(bad)[:400]}")
    with concurrent.futures.ThreadPoolExecutor(CLIENTS) as ex:
        list(ex.map(send, bodies))


PID = "$(/usr/share/opensearch/jdk/bin/jcmd -l | grep -m1 org.opensearch.bootstrap | cut -d' ' -f1)"


def full_gc(container):
    subprocess.run(["docker", "exec", container, "bash", "-c",
                    "/usr/share/opensearch/jdk/bin/jcmd " + PID + " GC.run"],
                   capture_output=True, check=False)
    time.sleep(2)


def rss_mib(container):
    out = subprocess.run(["docker", "exec", container, "bash", "-c",
                          "grep -E 'VmRSS|VmHWM' /proc/" + PID + "/status"],
                         capture_output=True, text=True).stdout
    vals = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in out.splitlines() if l}
    return vals.get("VmRSS"), vals.get("VmHWM")


def memory(container):
    full_gc(container)
    n = next(iter(req("GET", "/_nodes/stats/jvm,indices,breaker")["nodes"].values()))
    seg = n["indices"]["segments"]
    rss, hwm = rss_mib(container)
    return {
        "heap_after_gc_mib": round(n["jvm"]["mem"]["heap_used_in_bytes"] / 2**20),
        "rss_mib": rss, "hwm_mib": hwm,
        "segments": seg["count"],
        "segments_memory_mib": round(seg.get("memory_in_bytes", 0) / 2**20, 1),
        "index_writer_memory_mib": round(seg.get("index_writer_memory_in_bytes", 0) / 2**20, 1),
        "translog_ops": n["indices"]["translog"]["operations"],
        "shards": n["indices"].get("shard_stats", {}).get("total_count"),
    }


def shapes():
    return [
        ("term, frequent", {"term": {"body": "t0"}}, {}),
        ("match OR, 2 terms", {"match": {"body": "t3 t2s"}}, {}),
        ("match AND, 2 terms", {"match": {"body": {"query": "t1 t5", "operator": "and"}}}, {}),
        ("match, sort by num", {"match": {"body": "t9"}}, {"sort": [{"num": "desc"}]}),
        ("terms agg on tag", {"match_all": {}}, {"size": 0, "aggs": {"a": {"terms": {"field": "tag", "size": 10}}}}),
    ]


def search_body(mode, k, query, extra):
    if mode != "per_tenant":
        query = {"bool": {"must": query, "filter": {"term": {"tenant": "u%d" % k}}}}
    body = {"query": query, "track_total_hits": True}
    body.update(extra)
    return body


def classes(sizes):
    order = sorted(range(len(sizes)), key=lambda k: -sizes[k])
    n = len(order)
    pick = lambda xs: xs[:: max(1, len(xs) // 20)][:20]
    return {"large": pick(order[: max(1, n // 50)] if n >= 50 else order[:5]),
            "mid": pick(order[n // 10: n // 3]),
            "small": pick(order[2 * n // 3:])}


def time_search(mode, sizes, rounds):
    out = {}
    for cls, ks in classes(sizes).items():
        out[cls] = {"tenants": len(ks), "docs_per_tenant": round(statistics.mean(sizes[k] for k in ks))}
        for name, q, extra in shapes():
            for k in ks:  # warm-up pass
                req("POST", f"/{index_of(mode, k)}/_search?request_cache=false", search_body(mode, k, q, extra))
            wall, hits = [], 0
            for _ in range(rounds):
                for k in ks:
                    t = time.perf_counter()
                    r = req("POST", f"/{index_of(mode, k)}/_search?request_cache=false", search_body(mode, k, q, extra))
                    wall.append((time.perf_counter() - t) * 1000)
                    hits += r["hits"]["total"]["value"]
            wall.sort()
            out[cls][name] = {"median_ms": round(statistics.median(wall), 3),
                              "p90_ms": round(wall[int(len(wall) * 0.9) - 1], 3), "hits": hits // rounds}
            print(f"  {cls:5s} {name:22s} {out[cls][name]['median_ms']:7.2f} ms (p90 {out[cls][name]['p90_ms']:.2f})",
                  flush=True)
    return out


def run(base, container, mode, label, out, tenants, ndocs, rounds):
    global BASE
    BASE = base
    docs, sizes = corpus(tenants, ndocs)
    req("PUT", "/_cluster/settings", {"persistent": {"cluster.max_shards_per_node": max(1000, tenants + 10)}})
    idle = memory(container)
    settings = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    if mode == "partitioned":
        settings["lucene_rust.partition_field"] = "tenant"
    t = time.perf_counter()
    names = ["t-%d" % k for k in range(tenants)] if mode == "per_tenant" else ["shared"]

    def create(name):
        req("PUT", f"/{name}", {"settings": settings, "mappings": MAPPING})
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        list(ex.map(create, names))
    req("GET", "/_cluster/health?wait_for_status=green&timeout=600s")
    created = time.perf_counter() - t
    print(f"{label}: {len(names)} indices created in {created:.1f}s", flush=True)

    t = time.perf_counter()
    send_all(bulk_bodies(mode, docs))
    loaded = time.perf_counter() - t
    t = time.perf_counter()
    req("POST", "/_refresh")
    refreshed = time.perf_counter() - t
    count = req("GET", "/_count")["count"]
    store = sum(int(i["store.size"]) for i in req("GET", "/_cat/indices?format=json&bytes=b"))
    print(f"{label}: {count} docs in {loaded:.1f}s ({count / loaded:.0f} docs/s), refresh {refreshed:.2f}s, "
          f"{store / 2**20:.0f} MiB", flush=True)
    after_load = memory(container)
    print(f"{label}: memory {after_load}", flush=True)
    result = {"label": label, "mode": mode, "tenants": tenants, "docs": count, "create_seconds": round(created, 2),
              "load_seconds": round(loaded, 2), "docs_per_second": round(count / loaded),
              "refresh_seconds": round(refreshed, 3), "store_bytes": store,
              "memory_idle": idle, "memory_after_load": after_load}
    print(f"{label}: search")
    result["search"] = time_search(mode, sizes, rounds)

    # Re-index a slice spread over tenants like the load was, then refresh.
    rnd = random.Random(11)
    slice_ = rnd.sample(docs, min(50_000, len(docs) // 5))
    t = time.perf_counter()
    send_all(bulk_bodies(mode, slice_))
    reindexed = time.perf_counter() - t
    t = time.perf_counter()
    req("POST", "/_refresh")
    result["reindex"] = {"docs": len(slice_), "docs_per_second": round(len(slice_) / reindexed),
                         "refresh_seconds": round(time.perf_counter() - t, 3)}
    print(f"{label}: reindex {result['reindex']}", flush=True)

    # Single-document updates, each made visible at once.
    picks = [d for d in rnd.sample(docs, 200)]
    wall = []
    for k, i, src in picks:
        src = dict(src, num=src["num"] + 1)
        t = time.perf_counter()
        req("PUT", f"/{index_of(mode, k)}/_doc/{i}?refresh=true", src)
        wall.append((time.perf_counter() - t) * 1000)
    wall.sort()
    result["update_refresh_true"] = {"n": len(wall), "median_ms": round(statistics.median(wall), 2),
                                     "p90_ms": round(wall[int(len(wall) * 0.9) - 1], 2)}
    print(f"{label}: update+refresh {result['update_refresh_true']}", flush=True)
    result["memory_end"] = memory(container)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"{label}: wrote {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 5:
        sys.exit(__doc__)

    def opt(name, default):
        return int(a[a.index(name) + 1]) if name in a else default
    run(a[0], a[1], a[2], a[3], a[4], opt("--tenants", 500), opt("--docs", 500_000), opt("--rounds", 5))
