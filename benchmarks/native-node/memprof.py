"""Memory profile of one node (stock or plugin) started by native-node.sh, phase by phase.

  memprof.py BASE LABEL OUT CORPUS [PHASES]

PHASES (comma separated, default all): index, merge, search, aggs, fetch, soak.
A sampler thread reads the node process's anonymous and total resident memory (smaps_rollup) every
100 ms and the JVM heap and breakers (`_nodes/stats`) every second; each phase reports the peak
and end values of both. Heap numbers are used heap (not committed), so garbage counts until a
collection runs; `heap_after_gc` forces one (`_nodes/stats` after a `jcmd GC.run` is not
available without a JDK on the path, so the gap between peak and end is the thing to read).
The plugin node's off-heap counters (`_plugins/lucene_rust/stats`) are added when present.

SHARDS, DOCS_LIMIT, SOAK_CYCLES are read from the environment."""
import glob
import json
import os
import subprocess
import sys
import threading
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../../opensearch-plugin/e2e"))
import bench_nodes as bn  # noqa: E402


def node_pid():
    out = subprocess.run(["pgrep", "-f", "org.opensearch.bootstrap.OpenSearch"], capture_output=True, text=True).stdout.split()
    return int(out[0])


def rollup(pid):
    d = {}
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2 and p[0] in ("Rss:", "Anonymous:", "Pss_File:", "Pss_Anon:"):
                d[p[0][:-1]] = int(p[1]) // 1024
    return d


def heap_stats():
    n = next(iter(bn.req("GET", "/_nodes/stats/jvm,breaker")["nodes"].values()))
    br = {k: round(v["estimated_size_in_bytes"] / 2**20, 1) for k, v in n["breakers"].items() if v["estimated_size_in_bytes"]}
    return n["jvm"]["mem"]["heap_used_in_bytes"] // 2**20, br


def off_heap():
    try:
        s = bn.req("GET", "/_plugins/lucene_rust/stats")
    except Exception:
        return None
    m = s.get("off_heap_memory", {})
    return {k: (round(v / 2**20, 1) if k.endswith("_in_bytes") else v) for k, v in m.items() if not isinstance(v, dict)}


class Sampler(threading.Thread):
    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop_, self.lock = pid, False, threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.peak = {"rss": 0, "anon": 0, "heap": 0}
            self.last = {}

    def run(self):
        tick = 0
        while not self.stop_:
            try:
                r = rollup(self.pid)
                with self.lock:
                    self.peak["rss"] = max(self.peak["rss"], r["Rss"])
                    self.peak["anon"] = max(self.peak["anon"], r["Anonymous"])
                    self.last.update(rss=r["Rss"], anon=r["Anonymous"])
                if tick % 10 == 0:
                    h, _ = heap_stats()
                    with self.lock:
                        self.peak["heap"] = max(self.peak["heap"], h)
                        self.last["heap"] = h
            except Exception:
                pass
            tick += 1
            time.sleep(0.1)


def phase(sampler, name, fn, results):
    sampler.reset()
    t = time.perf_counter()
    fn()
    dt = time.perf_counter() - t
    time.sleep(1.2)
    h, br = heap_stats()
    r = rollup(sampler.pid)
    with sampler.lock:
        pk = dict(sampler.peak)
    res = {"seconds": round(dt, 1), "peak_rss_mb": pk["rss"], "peak_anon_mb": pk["anon"], "peak_heap_mb": pk["heap"],
           "end_rss_mb": r["Rss"], "end_anon_mb": r["Anonymous"], "end_heap_mb": h, "breakers_mb": br}
    oh = off_heap()
    if oh:
        res["off_heap"] = oh
    results[name] = res
    print(f"{name:28s} {dt:7.1f}s  peak rss {pk['rss']:5d} anon {pk['anon']:5d} heap {pk['heap']:5d} | end rss {r['Rss']:5d} anon {r['Anonymous']:5d} heap {h:5d}", flush=True)


def search_body(i):
    return bn.shapes()[i][1]


HEAVY = {
    "terms num size 10000 (high card)": {"size": 0, "aggs": {"a": {"terms": {"field": "num", "size": 10000}}}},
    "terms num size 100 sub avg": {"size": 0, "aggs": {"a": {"terms": {"field": "num", "size": 100}, "aggs": {"p": {"avg": {"field": "price"}}}}}},
    "terms tag size 1000": {"size": 0, "aggs": {"a": {"terms": {"field": "tag", "size": 1000}}}},
    "cardinality num": {"size": 0, "aggs": {"a": {"cardinality": {"field": "num"}}}},
    "percentiles price": {"size": 0, "aggs": {"a": {"percentiles": {"field": "price"}}}},
    "date_histogram hour": {"size": 0, "aggs": {"a": {"date_histogram": {"field": "ts", "fixed_interval": "1h"}}}},
    "date_histogram minute": {"size": 0, "aggs": {"a": {"date_histogram": {"field": "ts", "fixed_interval": "1m"}}}},
    "histogram num 1000": {"size": 0, "aggs": {"a": {"histogram": {"field": "num", "interval": 1000}}}},
    "composite num": {"size": 0, "aggs": {"a": {"composite": {"size": 10000, "sources": [{"n": {"terms": {"field": "num"}}}]}}}},
    "terms tag > terms category > avg": {"size": 0, "aggs": {"a": {"terms": {"field": "tag", "size": 200}, "aggs": {"b": {"terms": {"field": "category", "size": 50}, "aggs": {"p": {"avg": {"field": "price"}}}}}}}},
    "top_hits by tag": {"size": 0, "aggs": {"a": {"terms": {"field": "tag", "size": 100}, "aggs": {"t": {"top_hits": {"size": 3, "_source": False}}}}}},
    "sort num desc size 10000": {"size": 10000, "_source": False, "sort": [{"num": "desc"}], "query": {"match_all": {}}},
    "match_all size 1000 + source": {"size": 1000, "query": {"match_all": {}}},
    "match t0 size 5000 + source": {"size": 5000, "query": {"match": {"body": "t0"}}},
    "scroll-free from 9000 size 1000": {"from": 9000, "size": 1000, "query": {"match": {"body": "t0"}}, "sort": [{"num": "asc"}]},
}


def run_bodies(items, rounds):
    for _ in range(rounds):
        for _, body in items:
            try:
                bn.req("POST", "/bench/_search?request_cache=false", body)
            except RuntimeError as e:
                print("   refused:", str(e)[:90].replace("\n", " "), flush=True)


def main():
    base, label, out, corpus = sys.argv[1:5]
    phases = (sys.argv[5] if len(sys.argv) > 5 else "index,merge,search,aggs,fetch,soak").split(",")
    bn.BASE = base
    pid = node_pid()
    s = Sampler(pid)
    s.start()
    res = {"label": label, "pid": pid}
    phase(s, "idle", lambda: time.sleep(2), res)
    if "index" in phases:
        shards = int(os.environ.get("SHARDS", "1"))
        try:
            bn.req("DELETE", "/bench")
        except RuntimeError:
            pass
        st = {"number_of_shards": shards, "number_of_replicas": 0, "refresh_interval": os.environ.get("REFRESH", "5s")}
        if os.environ.get("ACCEL", "1") == "1" and "plugin" in label:
            st["index.engine.accelerated"] = True
        bn.req("PUT", "/bench", {"settings": st, "mappings": bn.MAPPING})
        files = sorted(glob.glob(os.path.join(corpus, "bulk-*.ndjson")))
        lim = int(os.environ.get("DOCS_LIMIT", "0"))
        if lim:
            files = files[:lim]
        phase(s, "index (bulk, live refresh)", lambda: bn.load(files), res)
        res["segments_after_index"] = len(bn.req("GET", "/_cat/segments/bench?format=json"))
    if "merge" in phases:
        def fm():
            bn.req("POST", "/bench/_forcemerge?max_num_segments=1", timeout=3600)
            bn.req("POST", "/bench/_refresh", timeout=3600)
        phase(s, "force merge to 1 segment", fm, res)
        idx = bn.req("GET", "/_cat/indices/bench?format=json&bytes=b")[0]
        res["merged_store_mb"] = int(idx["store.size"]) // 2**20
    n = int(os.environ.get("ROUNDS", "20"))
    if "search" in phases:
        light = [x for x in bn.shapes() if not any(k in x[0] for k in ("agg", "composite", "histogram", "stats"))]
        aggs = [x for x in bn.shapes() if any(k in x[0] for k in ("agg", "composite", "histogram", "stats"))]
        phase(s, "search: queries/sorts", lambda: run_bodies(light, n), res)
        phase(s, "search: standard aggs", lambda: run_bodies(aggs, n), res)
    if "aggs" in phases:
        for name, body in HEAVY.items():
            if name.startswith(("sort", "match", "scroll")):
                continue
            phase(s, "heavy: " + name, lambda b=body: run_bodies([("x", b)], 5), res)
    if "fetch" in phases:
        for name, body in HEAVY.items():
            if name.startswith(("sort", "match", "scroll")):
                phase(s, "fetch/sort: " + name, lambda b=body: run_bodies([("x", b)], 5), res)
    if "soak" in phases:
        allq = bn.shapes() + list(HEAVY.items())
        cycles = int(os.environ.get("SOAK_CYCLES", "10"))
        for c in range(cycles):
            phase(s, f"soak cycle {c + 1}", lambda: run_bodies(allq, 2), res)
    s.stop_ = True
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
