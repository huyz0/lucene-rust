"""One node against another on the same corpus: bulk indexing, store size,
memory, and REST latency per request shape.

Usage:
  bench_nodes.py gen DIR DOCS          writes the corpus as bulk bodies (DIR/bulk-*.ndjson)
  bench_nodes.py run BASE CONTAINER DIR LABEL OUT [--rounds N] [--java-too]
  bench_nodes.py visibility BASE CONTAINER DIR LABEL OUT SETTINGS_JSON

`run` creates index `bench` (one shard, no replica), sends the bulk bodies
with CLIENTS parallel clients, refreshes, then times each request shape:
WARMUP requests, then ROUNDS more, sequential, `request_cache=false`; the
median and 90th percentile of the round trip are kept. `--java-too` times the
shapes again with the plugin's native search off on the same index. The
result (JSON) goes to OUT.

The corpus is synthetic: a Zipf vocabulary of VOCAB terms (`t` plus the rank
in base 36, `t0` the most frequent), seeded, so every node reads the same
bytes. Generation is not part of any timing."""
import bisect
import concurrent.futures
import glob
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

VOCAB = 200_000
ZIPF_S = 1.07
PER_FILE = 5_000
CLIENTS = int(os.environ.get("CLIENTS", "4"))
WARMUP = int(os.environ.get("WARMUP", "30"))
CATEGORIES = 50
TAGS = 1_000


def b36(n):
    s = ""
    while True:
        n, d = divmod(n, 36)
        s = "0123456789abcdefghijklmnopqrstuvwxyz"[d] + s
        if n == 0:
            return s


def zipf_cum(n):
    acc, out = 0.0, []
    for r in range(1, n + 1):
        acc += 1.0 / r ** ZIPF_S
        out.append(acc)
    return out


def gen(out_dir, docs):
    os.makedirs(out_dir, exist_ok=True)
    rnd = random.Random(42)
    words = ["t" + b36(i) for i in range(VOCAB)]
    cum = zipf_cum(VOCAB)
    tag_cum = zipf_cum(TAGS)
    for f, start in enumerate(range(0, docs, PER_FILE)):
        lines = []
        for i in range(start, min(docs, start + PER_FILE)):
            body = rnd.choices(words, cum_weights=cum, k=rnd.randint(20, 120))
            title = rnd.choices(words, cum_weights=cum, k=rnd.randint(2, 8))
            lines.append('{"index":{"_id":"%d"}}' % i)
            lines.append(json.dumps({
                "body": " ".join(body),
                "title": " ".join(title),
                "tag": "g" + b36(bisect.bisect_left(tag_cum, rnd.random() * tag_cum[-1])),
                "category": "c%d" % rnd.randrange(CATEGORIES),
                "num": rnd.randrange(1_000_000),
                "price": round(rnd.uniform(0, 1000), 2),
                "ts": 1_700_000_000_000 + rnd.randrange(365 * 24 * 60) * 60_000,
            }, separators=(",", ":")))
        with open(os.path.join(out_dir, "bulk-%05d.ndjson" % f), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    print(f"gen: {docs} documents in {f + 1} files under {out_dir}")


BASE = ""


def req(method, path, body=None, ndjson=False, timeout=600):
    data = None
    headers = {}
    if body is not None:
        data = body if isinstance(body, bytes) else (body.encode() if isinstance(body, str) else json.dumps(body).encode())
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:500]}")


MAPPING = {
    "properties": {
        "body": {"type": "text"},
        "title": {"type": "text"},
        "tag": {"type": "keyword"},
        "category": {"type": "keyword"},
        "num": {"type": "long"},
        "price": {"type": "double"},
        "ts": {"type": "date"},
    }
}


def shapes():
    """(name, body): the request shapes timed, from cheap to heavy."""
    return [
        ("term, frequent (t0)", {"query": {"term": {"body": "t0"}}}),
        ("term, mid (t2s)", {"query": {"term": {"body": "t2s"}}}),
        ("term, rare (t1z4)", {"query": {"term": {"body": "t1z4"}}}),
        ("match OR, 2 frequent", {"query": {"match": {"body": "t1 t2"}}}),
        ("match OR, 4 mixed", {"query": {"match": {"body": "t0 t5 tz t2s"}}}),
        ("match AND, 2 frequent", {"query": {"match": {"body": {"query": "t1 t2", "operator": "and"}}}}),
        ("match AND, frequent + mid", {"query": {"match": {"body": {"query": "t0 tz", "operator": "and"}}}}),
        ("match_phrase, 2 frequent", {"query": {"match_phrase": {"body": "t0 t1"}}}),
        ("multi_match body+title", {"query": {"multi_match": {"query": "t3 t7", "fields": ["body", "title^2"]}}}),
        ("bool match + range filter", {"query": {"bool": {"must": {"match": {"body": "t4"}}, "filter": {"range": {"num": {"gte": 100000, "lt": 300000}}}}}}),
        ("range on long", {"query": {"range": {"num": {"gte": 400000, "lt": 410000}}}}),
        ("term on keyword", {"query": {"term": {"tag": "g5"}}}),
        ("prefix on title", {"query": {"prefix": {"title": "t1z"}}}),
        ("match_all, sort by num", {"query": {"match_all": {}}, "sort": [{"num": "desc"}]}),
        ("match, sort by price", {"query": {"match": {"body": "t9"}}, "sort": [{"price": "asc"}]}),
        ("count, track_total_hits", {"size": 0, "track_total_hits": True, "query": {"match": {"body": "t0 t1"}}}),
        ("terms agg on category", {"size": 0, "aggs": {"a": {"terms": {"field": "category"}}}}),
        ("terms agg on tag, filtered", {"size": 0, "query": {"match": {"body": "t8"}}, "aggs": {"a": {"terms": {"field": "tag", "size": 20}}}}),
        ("date_histogram by month", {"size": 0, "aggs": {"a": {"date_histogram": {"field": "ts", "calendar_interval": "month"}}}}),
        ("stats on price", {"size": 0, "aggs": {"a": {"stats": {"field": "price"}}}}),
    ]


def docker_mem(container):
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                         capture_output=True, text=True).stdout.strip()
    return out.split("/")[0].strip()


def jvm_heap():
    n = next(iter(req("GET", "/_nodes/stats/jvm")["nodes"].values()))
    return n["jvm"]["mem"]["heap_used_in_bytes"]


def node_snapshot(container):
    return {"rss": docker_mem(container), "heap_used_mb": round(jvm_heap() / 2**20)}


def load(files):
    t0 = time.perf_counter()

    def send(path):
        with open(path, "rb") as fh:
            res = req("POST", "/bench/_bulk", fh.read(), ndjson=True)
        if res["errors"]:
            raise RuntimeError(f"bulk {path} had errors")

    with concurrent.futures.ThreadPoolExecutor(CLIENTS) as ex:
        list(ex.map(send, files))
    indexed = time.perf_counter() - t0
    req("POST", "/bench/_refresh")
    return indexed, time.perf_counter() - t0


def time_shapes(rounds):
    out = {}
    for name, body in shapes():
        for _ in range(WARMUP):
            req("POST", "/bench/_search?request_cache=false", body)
        wall = []
        hits = None
        for _ in range(rounds):
            t = time.perf_counter()
            r = req("POST", "/bench/_search?request_cache=false", body)
            wall.append((time.perf_counter() - t) * 1000)
            hits = r["hits"]["total"]["value"]
        wall.sort()
        out[name] = {"median_ms": round(statistics.median(wall), 3),
                     "p90_ms": round(wall[int(len(wall) * 0.9) - 1], 3), "hits": hits}
        print(f"  {name:32s} {out[name]['median_ms']:8.2f} ms (p90 {out[name]['p90_ms']:.2f})", flush=True)
    return out


def run(base, container, corpus, label, out, rounds, java_too):
    global BASE
    BASE = base
    files = sorted(glob.glob(os.path.join(corpus, "bulk-*.ndjson")))
    try:
        req("DELETE", "/bench")
    except RuntimeError:
        pass
    req("PUT", "/bench", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1},
                          "mappings": MAPPING})
    idle = node_snapshot(container)
    indexed, refreshed = load(files)
    docs = req("GET", "/bench/_count")["count"]
    after_load = node_snapshot(container)
    idx = req("GET", "/_cat/indices/bench?format=json&bytes=b")[0]
    segs = len(req("GET", "/_cat/segments/bench?format=json"))
    print(f"{label}: {docs} docs indexed in {indexed:.1f}s ({docs / indexed:.0f} docs/s), "
          f"refreshed at {refreshed:.1f}s, {int(idx['store.size']) / 2**20:.0f} MiB, {segs} segments", flush=True)
    result = {
        "label": label,
        "docs": docs,
        "index_seconds": round(indexed, 2),
        "index_refresh_seconds": round(refreshed, 2),
        "docs_per_second": round(docs / indexed),
        "store_bytes": int(idx["store.size"]),
        "segments": segs,
        "memory_idle": idle,
        "memory_after_load": after_load,
    }
    print(f"{label}: search")
    result["search"] = time_shapes(rounds)
    result["memory_after_search"] = node_snapshot(container)
    if java_too:
        req("PUT", "/bench/_settings", {"index.lucene_rust.search.enabled": False})
        print(f"{label}: search, native search off")
        result["search_java"] = time_shapes(rounds)
        req("PUT", "/bench/_settings", {"index.lucene_rust.search.enabled": True})
    # Merged to one segment: the same shapes on the smallest index either engine makes.
    t = time.perf_counter()
    req("POST", "/bench/_forcemerge?max_num_segments=1", timeout=3600)
    # refresh_interval is -1, so the merged segment is not searchable until a refresh.
    req("POST", "/bench/_refresh", timeout=3600)
    result["force_merge_seconds"] = round(time.perf_counter() - t, 2)
    idx = req("GET", "/_cat/indices/bench?format=json&bytes=b")[0]
    result["merged_store_bytes"] = int(idx["store.size"])
    print(f"{label}: force-merged in {result['force_merge_seconds']}s, {int(idx['store.size']) / 2**20:.0f} MiB; search")
    result["search_merged"] = time_shapes(rounds)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"{label}: wrote {out}")


def cpu_seconds(container):
    """The node process's CPU time so far, in seconds, as the node reports it."""
    nodes = req("GET", "/_nodes/stats/process")["nodes"]
    return next(iter(nodes.values()))["process"]["cpu"]["total_in_millis"] / 1000


def pct(values, p):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(len(values) * p))], 1) if values else None


def visibility(base, container, corpus, label, out, settings):
    """Bulk-loads the corpus with CLIENTS clients while a probe indexes one document every
    PROBE_EVERY seconds and searches for it until it is found (the time from the write's
    acknowledgement to the first search that finds it), and a searcher runs a few request shapes
    in a loop. Index `bench` takes `settings` (JSON) beside one shard and no replica."""
    global BASE
    BASE = base
    files = sorted(glob.glob(os.path.join(corpus, "bulk-*.ndjson")))
    try:
        req("DELETE", "/bench")
    except RuntimeError:
        pass
    s = {"number_of_shards": 1, "number_of_replicas": 0}
    s.update(settings)
    req("PUT", "/bench", {"settings": s, "mappings": MAPPING})
    every = float(os.environ.get("PROBE_EVERY", "0.2"))
    done = [False]
    seen, missed, errors = [], [0], []
    searched = {}
    wanted = {"term, frequent (t0)", "match OR, 2 frequent", "terms agg on tag, filtered", "match, sort by price"}
    pairs = [(n, b) for n, b in shapes() if n in wanted]

    def probe_once(n):
        req("PUT", f"/bench/_doc/probe-{n}", {"body": f"probe{n}", "tag": "probe"})
        t0 = time.perf_counter()
        q = {"query": {"term": {"body": f"probe{n}"}}}
        while True:
            if req("POST", "/bench/_search?request_cache=false", q)["hits"]["total"]["value"] >= 1:
                seen.append((time.perf_counter() - t0) * 1000)
                return
            if time.perf_counter() - t0 > 30:
                missed[0] += 1
                return
            time.sleep(0.002)

    def probe():
        n = 0
        while not done[0]:
            n += 1
            try:
                probe_once(n)
            except (RuntimeError, OSError) as e:
                errors.append(f"probe {n}: {e}")
            time.sleep(every)

    def search():
        i = 0
        while not done[0]:
            name, body = pairs[i % len(pairs)]
            t = time.perf_counter()
            try:
                req("POST", "/bench/_search?request_cache=false", body)
                searched.setdefault(name, []).append((time.perf_counter() - t) * 1000)
            except (RuntimeError, OSError) as e:
                errors.append(f"{name}: {e}")
            i += 1

    cpu0 = cpu_seconds(container)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(2) as side:
        tasks = [side.submit(probe), side.submit(search)]
        try:
            def send(path):
                with open(path, "rb") as fh:
                    res = req("POST", "/bench/_bulk", fh.read(), ndjson=True)
                if res["errors"]:
                    raise RuntimeError(f"bulk {path} had errors")
            with concurrent.futures.ThreadPoolExecutor(CLIENTS) as ex:
                list(ex.map(send, files))
        finally:
            done[0] = True
            for t in tasks:
                t.result()
    indexed = time.perf_counter() - t0
    cpu = cpu_seconds(container) - cpu0
    docs = 0
    for f in files:
        with open(f, "rb") as fh:
            docs += sum(1 for _ in fh) // 2
    result = {
        "label": label, "settings": settings, "docs": docs, "index_seconds": round(indexed, 2),
        "docs_per_second": round(docs / indexed), "cpu_seconds": round(cpu, 1),
        "probes": len(seen), "probes_missed": missed[0], "request_errors": errors[:20], "request_error_count": len(errors),
        "visible_ms": {"p50": pct(seen, 0.5), "p90": pct(seen, 0.9), "p99": pct(seen, 0.99), "max": pct(seen, 1.0)},
        "search_ms_during_load": {n: {"p50": pct(v, 0.5), "p90": pct(v, 0.9), "n": len(v)} for n, v in searched.items()},
        "memory_after_load": node_snapshot(container),
    }
    try:
        result["plugin_stats"] = req("GET", "/_plugins/lucene_rust/stats")
    except Exception:  # optional: the stock node has no such endpoint
        pass
    print(f"{label}: {result['docs_per_second']} docs/s, {result['cpu_seconds']} CPU-s, visible p50/p99 "
          f"{result['visible_ms']['p50']}/{result['visible_ms']['p99']} ms ({len(seen)} probes, {missed[0]} missed, "
          f"{len(errors)} request errors)",
          flush=True)
    for n, v in result["search_ms_during_load"].items():
        print(f"  {n:32s} p50 {v['p50']} ms, p90 {v['p90']} ms ({v['n']} searches)", flush=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "run":
        a = sys.argv[2:]
        rounds = 50
        if "--rounds" in a:
            rounds = int(a[a.index("--rounds") + 1])
        run(a[0], a[1], a[2], a[3], a[4], rounds, "--java-too" in a)
    elif sys.argv[1] == "visibility":
        a = sys.argv[2:]
        visibility(a[0], a[1], a[2], a[3], a[4], json.loads(a[5]))
    else:
        sys.exit(__doc__)
