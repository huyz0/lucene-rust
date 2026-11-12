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
import threading
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
    geo = random.Random(7)
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
                # Its own stream, so the other fields stay those of corpora made before.
                "loc": {"lat": round(geo.uniform(-60, 60), 5), "lon": round(geo.uniform(-180, 180), 5)},
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
        "loc": {"type": "geo_point"},
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
        ("geo_distance filter", {"query": {"bool": {"filter": {"geo_distance": {"distance": "500km", "loc": {"lat": 10, "lon": 20}}}}}}),
        ("geo_bounding_box filter", {"query": {"bool": {"filter": {"geo_bounding_box": {"loc": {"top_left": {"lat": 15, "lon": 10}, "bottom_right": {"lat": 5, "lon": 25}}}}}}}),
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


def throughput(container, files, pairs, levels):
    """Closed-loop search throughput at each of `levels` concurrent clients."""
    seconds = float(os.environ.get("THROUGHPUT_SECONDS", "15"))
    index_rate = float(os.environ.get("THROUGHPUT_INDEX_RATE") or 0)
    rows = []
    for n in levels:
        stop = [False]
        lat, errors = [], []
        lock = threading.Lock()

        def client(k):
            i = k
            while not stop[0]:
                _, body = pairs[i % len(pairs)]
                t = time.perf_counter()
                try:
                    req("POST", "/bench/_search?request_cache=false", body)
                    with lock:
                        lat.append((time.perf_counter() - t) * 1000)
                except (RuntimeError, OSError) as e:
                    errors.append(str(e))
                i += 1

        def indexer(level):
            sent, t0 = 0, time.perf_counter()
            for f in files:
                if stop[0]:
                    return
                with open(f, "rb") as fh:
                    body = fh.read().replace(b'{"index":{"_id":"', b'{"index":{"_id":"x%d-' % level)
                due = t0 + sent / index_rate
                sent += body.count(b"\n") // 2
                wait = due - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                req("POST", "/bench/_bulk", body, ndjson=True)

        with concurrent.futures.ThreadPoolExecutor(n + 1) as ex:
            side = ex.submit(indexer, n) if index_rate > 0 else None
            if side:
                time.sleep(2)
            c0, t0 = cpu_seconds(container), time.perf_counter()
            tasks = [ex.submit(client, k) for k in range(n)]
            time.sleep(seconds)
            stop[0] = True
            for t in tasks:
                t.result()
            took, cpu = time.perf_counter() - t0, cpu_seconds(container) - c0
            if side:
                side.result()
        rows.append({
            "clients": n, "searches_per_second": round(len(lat) / took, 1),
            "p50": pct(lat, 0.5), "p99": pct(lat, 0.99),
            "node_cpu_ms_per_search": round(cpu * 1000 / max(1, len(lat)), 2),
            "indexing": index_rate, "errors": len(errors),
        })
    return rows


def pct(values, p):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(len(values) * p))], 1) if values else None


def index_work():
    """Merge, refresh and flush work the load caused, and the segments it left."""
    t = req("GET", "/bench/_stats/merge,refresh,flush,segments")["indices"]["bench"]["total"]
    m = t["merges"]
    return {
        "merges": m["total"], "merge_ms": m["total_time_in_millis"], "merge_docs": m["total_docs"],
        "merge_bytes": m["total_size_in_bytes"], "refreshes": t["refresh"]["total"],
        "refresh_ms": t["refresh"]["total_time_in_millis"], "flushes": t["flush"]["total"],
        "segments": t["segments"]["count"],
    }


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
    # Each search's own `took`: the node's time, without the client's and the HTTP round trip.
    server = {}
    phases = {}
    wanted = {"term, frequent (t0)", "match OR, 2 frequent", "terms agg on tag, filtered", "match, sort by price"}
    if os.environ.get("VIS_SHAPES"):
        wanted = set(os.environ["VIS_SHAPES"].split("|"))
    pairs = [(n, b) for n, b in shapes() if n in wanted]
    unknown = wanted - {n for n, _ in pairs}
    if unknown:
        raise SystemExit(f"VIS_SHAPES: no such shape {sorted(unknown)}")

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
            # In short steps: the load's end, not a long PROBE_EVERY, ends the run.
            until = time.perf_counter() + every
            while not done[0] and time.perf_counter() < until:
                time.sleep(min(0.05, every))

    # SEARCH_RATE=N: each searcher sends at most N searches a second, so
    # configs compare at the same search load.
    rate = float(os.environ.get("SEARCH_RATE") or 0)

    def search():
        i = 0
        last = 0.0
        while not done[0]:
            name, body = pairs[i % len(pairs)]
            t = time.perf_counter()
            if rate > 0 and i > 0:
                time.sleep(max(0.0, last + 1.0 / rate - t))
                t = time.perf_counter()
            last = t
            try:
                r = req("POST", "/bench/_search?request_cache=false&phase_took=true", body)
                searched.setdefault(name, []).append((time.perf_counter() - t) * 1000)
                server.setdefault(name, []).append(float(r.get("took", 0)))
                for ph, ms in (r.get("phase_took") or {}).items():
                    phases.setdefault(name, {}).setdefault(ph, []).append(float(ms))
            except (RuntimeError, OSError) as e:
                errors.append(f"{name}: {e}")
            i += 1

    cpu0 = cpu_seconds(container)
    t0 = time.perf_counter()
    searchers = int(os.environ.get("SEARCHERS", "1"))
    with concurrent.futures.ThreadPoolExecutor(1 + searchers) as side:
        tasks = [side.submit(probe)] + [side.submit(search) for _ in range(searchers)]
        try:
            # INDEX_RATE=N: bulks are sent no faster than N documents a
            # second in all, so each config indexes at the same rate and
            # leaves the same CPU to searches.
            index_rate = float(os.environ.get("INDEX_RATE") or 0)
            paced = [0, threading.Lock()]

            def send(path):
                with open(path, "rb") as fh:
                    body = fh.read()
                if index_rate > 0:
                    with paced[1]:
                        due = t0 + paced[0] / index_rate
                        paced[0] += body.count(b"\n") // 2
                    wait = due - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                res = req("POST", "/bench/_bulk", body, ndjson=True)
                if res["errors"]:
                    first = next((i for i in res["items"] if "error" in next(iter(i.values()))), None)
                    raise RuntimeError(f"bulk {path} had errors, first: {json.dumps(first)[:2000]}")
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
        "search_ms_during_load": {
            n: {
                "p50": pct(v, 0.5),
                "p90": pct(v, 0.9),
                "n": len(v),
                # took is whole milliseconds: its mean resolves a fraction of one.
                "took_mean": round(sum(server.get(n, [])) / max(1, len(server.get(n, []))), 2),
                "mean": round(sum(v) / max(1, len(v)), 2),
                "phase_took_mean": {ph: round(sum(x) / len(x), 2) for ph, x in phases.get(n, {}).items() if x},
            }
            for n, v in searched.items()
        },
        "memory_after_load": node_snapshot(container),
        "index_stats_after_load": index_work(),
    }
    # SEARCH_ONLY=N: then N searches of each shape, one at a time, with no
    # indexing: their CPU alone, a shape at a time. The buffered documents
    # stay as the load left them; nothing refreshes them first.
    rounds = int(os.environ.get("SEARCH_ONLY", "0"))
    if rounds:
        took = {n: [] for n, _ in pairs}
        cpu_by = {}
        # Merges the load started settle first.
        time.sleep(float(os.environ.get("SEARCH_ONLY_SETTLE", "15")))
        cpu1 = cpu_seconds(container)
        t1 = time.perf_counter()
        for name, body in pairs:
            c = cpu_seconds(container)
            for _ in range(rounds):
                t = time.perf_counter()
                req("POST", "/bench/_search?request_cache=false", body)
                took[name].append((time.perf_counter() - t) * 1000)
            cpu_by[name] = round((cpu_seconds(container) - c) * 1000 / rounds, 3)
        result["search_only"] = {
            "searches": rounds * len(pairs),
            "seconds": round(time.perf_counter() - t1, 2),
            "cpu_seconds": round(cpu_seconds(container) - cpu1, 2),
            "ms": {n: {"p50": pct(v, 0.5), "avg": round(sum(v) / len(v), 2), "cpu_ms": cpu_by[n]}
                   for n, v in took.items()},
        }
        print(f"  search only: {result['search_only']}", flush=True)
    # THROUGHPUT="1,8,32": then, at each concurrency, that many clients send
    # the shapes in a closed loop for THROUGHPUT_SECONDS; searches a second,
    # latency and node CPU per search are kept. With THROUGHPUT_INDEX_RATE=N,
    # the corpus is indexed again meanwhile (new ids) at N documents a second.
    levels = [int(n) for n in os.environ.get("THROUGHPUT", "").split(",") if n.strip()]
    if levels:
        result["throughput"] = throughput(container, files, pairs, levels)
        for row in result["throughput"]:
            print(f"  throughput: {row}", flush=True)
    try:
        result["plugin_stats"] = req("GET", "/_plugins/lucene_rust/stats")
    except Exception:  # optional: the stock node has no such endpoint
        pass
    print(f"{label}: {result['docs_per_second']} docs/s, {result['cpu_seconds']} CPU-s, visible p50/p99 "
          f"{result['visible_ms']['p50']}/{result['visible_ms']['p99']} ms ({len(seen)} probes, {missed[0]} missed, "
          f"{len(errors)} request errors)",
          flush=True)
    print(f"  index work: {result['index_stats_after_load']}", flush=True)
    for n, v in result["search_ms_during_load"].items():
        print(f"  {n:32s} p50 {v['p50']} ms, p90 {v['p90']} ms ({v['n']} searches), mean {v['mean']} ms, took mean {v['took_mean']} ms {v['phase_took_mean']}", flush=True)
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
