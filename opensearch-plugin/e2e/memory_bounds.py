"""Adversarial memory checks against one node: requests built to make the
node hold as much as possible, each sampled for the container's anonymous
memory (cgroup rss, no page cache) while it runs, and answered either with a
result or with OpenSearch's own refusal (429 circuit_breaking_exception, 400
too_many_buckets_exception) -- never by the node dying.

Usage:
  memory_bounds.py BASE CONTAINER LABEL OUT [load]

`load` indexes the corpus first (DOCS documents, default 2,000,000: a unique
`id` keyword, a distinct `ts` date, a 1,000-value `host`, a long `n`, a short
`msg` text, in SHARDS shards). Then each case runs ROUNDS times (default 3) and
the JSON at OUT holds, per case: status codes, error types, took, the peak
rss while it ran, and the plugin's off-heap stats after it.
"""
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DOCS = int(os.environ.get("DOCS", "2000000"))
SHARDS = int(os.environ.get("SHARDS", "1"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
INDEX = "membounds"


def req(base, method, path, body=None, ndjson=False, timeout=600):
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    headers = {"content-type": "application/x-ndjson" if ndjson else "application/json"}
    r = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {}
    except Exception as e:  # connection refused: the node died
        return -1, {"error": {"type": type(e).__name__, "reason": str(e)}}


def cgroup_rss(container):
    """Anonymous memory of the container (cgroup v1 total_rss, v2 anon)."""
    out = subprocess.run(
        ["docker", "exec", container, "sh", "-c",
         "cat /sys/fs/cgroup/memory/memory.stat 2>/dev/null || cat /sys/fs/cgroup/memory.stat"],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        k, _, v = line.partition(" ")
        if k in ("total_rss", "anon"):
            return int(v)
    return -1


class Sampler:
    def __init__(self, container):
        self.container, self.peak, self.stop = container, 0, False
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, cgroup_rss(self.container))
            time.sleep(0.2)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop = True
        self.t.join()


def load(base):
    req(base, "DELETE", "/" + INDEX)
    st, r = req(base, "PUT", "/" + INDEX, {
        "settings": {"number_of_shards": SHARDS, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {
            "id": {"type": "keyword"}, "ts": {"type": "date"}, "host": {"type": "keyword"},
            "n": {"type": "long"}, "msg": {"type": "text"}}}})
    assert st == 200, r
    rnd = random.Random(7)
    words = ["w%d" % i for i in range(5000)]
    batch, t0, sent = [], time.time(), 0
    base_ts = 1_700_000_000_000

    def flush(lines):
        st, r = req(base, "POST", "/_bulk", ("\n".join(lines) + "\n").encode(), ndjson=True)
        assert st == 200 and not r.get("errors"), str(r)[:500]

    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        futs = []
        for i in range(DOCS):
            batch.append('{"index":{"_index":"%s"}}' % INDEX)
            batch.append(json.dumps({
                "id": "%016x%016x" % (rnd.getrandbits(64), i), "ts": base_ts + i * 7,
                "host": "h%d" % rnd.randrange(1000), "n": rnd.randrange(1 << 40),
                "msg": " ".join(rnd.choice(words) for _ in range(8))}))
            if len(batch) >= 10000:
                futs.append(ex.submit(flush, batch))
                batch = []
                sent += 5000
                if len(futs) > 8:
                    futs.pop(0).result()
        if batch:
            futs.append(ex.submit(flush, batch))
        for f in futs:
            f.result()
    req(base, "POST", "/%s/_refresh" % INDEX)
    req(base, "POST", "/%s/_forcemerge?max_num_segments=4" % INDEX, timeout=1800)
    req(base, "POST", "/%s/_refresh" % INDEX)
    print("loaded %d docs in %.0f s" % (DOCS, time.time() - t0), flush=True)


def cases():
    q = {"size": 0, "track_total_hits": False}
    return {
        # Every document its own bucket, each bucket a metric: the shard holds DOCS buckets.
        "terms_unique_ids_sub_avg": dict(q, aggs={"t": {"terms": {"field": "id", "size": 60000, "shard_size": 60000},
                                                         "aggs": {"a": {"avg": {"field": "n"}}}}}),
        # Far more buckets than search.max_buckets: stock refuses with too_many_buckets_exception.
        "terms_unique_ids_over_max_buckets": dict(q, aggs={"t": {"terms": {"field": "id", "size": 200000}}}),
        # A bucket per millisecond over the whole corpus, with a sub-aggregation.
        "date_histogram_1ms": dict(q, aggs={"h": {"date_histogram": {"field": "ts", "fixed_interval": "7ms", "min_doc_count": 1},
                                                  "aggs": {"m": {"max": {"field": "n"}}}}}),
        # Two levels: 1,000 hosts x 1,000 nested terms each.
        "terms_host_x_terms_id": dict(q, aggs={"h": {"terms": {"field": "host", "size": 1000},
                                                     "aggs": {"i": {"terms": {"field": "id", "size": 50}}}}}),
        # High-cardinality count: HLL bounded by precision_threshold.
        "cardinality_ids": dict(q, aggs={"c": {"cardinality": {"field": "id", "precision_threshold": 40000}}}),
        # Deep pagination at the result window's end, sorted.
        "sort_deep_page": {"from": 9000, "size": 1000, "sort": [{"n": "asc"}], "_source": False},
        # Every hit counted, every document matched.
        "match_all_count": {"size": 10, "track_total_hits": True, "query": {"match_all": {}}},
    }


def run_case(base, container, name, body, rounds):
    out = {"status": [], "errors": [], "took_ms": []}
    with Sampler(container) as s:
        for _ in range(rounds):
            t = time.time()
            st, r = req(base, "POST", "/%s/_search?request_cache=false" % INDEX, body)
            out["status"].append(st)
            out["took_ms"].append(round((time.time() - t) * 1000))
            if st != 200:
                err = r.get("error", {})
                causes = err.get("root_cause") or [err]
                out["errors"].append(causes[0].get("type", "?") if causes else "?")
            elif r.get("_shards", {}).get("failed"):
                f = r["_shards"]["failures"][0].get("reason", {})
                out["errors"].append("shard:" + f.get("type", "?"))
    out["peak_rss_mb"] = round(s.peak / 2**20)
    return out


def concurrent_case(base, container, name, body, n):
    out = {"status": {}, "errors": {}}
    with Sampler(container) as s:
        with concurrent.futures.ThreadPoolExecutor(n) as ex:
            futs = [ex.submit(req, base, "POST", "/%s/_search?request_cache=false" % INDEX, body) for _ in range(n)]
            for f in futs:
                st, r = f.result()
                out["status"][str(st)] = out["status"].get(str(st), 0) + 1
                if st != 200:
                    err = r.get("error", {})
                    causes = err.get("root_cause") or [err]
                    t = causes[0].get("type", "?") if causes else "?"
                    out["errors"][t] = out["errors"].get(t, 0) + 1
    out["peak_rss_mb"] = round(s.peak / 2**20)
    return out


def plugin_stats(base):
    st, r = req(base, "GET", "/_plugins/lucene_rust/stats")
    return r.get("off_heap_memory") if st == 200 else None


def breaker_stats(base):
    st, r = req(base, "GET", "/_nodes/stats/breaker")
    if st != 200:
        return None
    node = next(iter(r["nodes"].values()))
    return {k: {"tripped": v["tripped"], "limit": v["limit_size_in_bytes"]} for k, v in node["breakers"].items()}


def main():
    base, container, label, out_path = sys.argv[1:5]
    if len(sys.argv) > 5 and sys.argv[5] == "load":
        load(base)
    result = {"label": label, "baseline_rss_mb": round(cgroup_rss(container) / 2**20), "cases": {}}
    heavy = {} if os.environ.get("ONLY_LATENCY") else cases()
    for name, body in heavy.items():
        r = run_case(base, container, name, body, ROUNDS)
        r["off_heap"] = plugin_stats(base)
        result["cases"][name] = r
        print(label, name, json.dumps({k: v for k, v in r.items() if k != "off_heap"}), flush=True)
        if r["status"] and r["status"][-1] == -1:
            print(label, "node died", flush=True)
            break
    # Ordinary requests, for the accounting's cost: p50 over LATENCY_ROUNDS.
    rounds = int(os.environ.get("LATENCY_ROUNDS", "60"))
    light = {
        "lat_terms_host": {"size": 0, "aggs": {"h": {"terms": {"field": "host", "size": 10}, "aggs": {"m": {"avg": {"field": "n"}}}}}},
        "lat_match": {"size": 10, "query": {"match": {"msg": "w1 w2"}}},
        "lat_range_sort": {"size": 100, "query": {"range": {"n": {"gte": 1 << 39}}}, "sort": [{"ts": "desc"}]},
    }
    for name, body in light.items():
        times = []
        for i in range(rounds):
            t = time.perf_counter()
            st, _ = req(base, "POST", "/%s/_search?request_cache=false" % INDEX, body)
            if i >= 5:
                times.append((time.perf_counter() - t) * 1000)
        times.sort()
        result["cases"][name] = {"status": [st], "errors": [], "p50_ms": round(times[len(times) // 2], 2),
                                 "p90_ms": round(times[int(len(times) * 0.9)], 2)}
        print(label, name, json.dumps(result["cases"][name]), flush=True)
    # Many heavy requests at once.
    body = cases()["terms_unique_ids_sub_avg"]
    if os.environ.get("ONLY_LATENCY"):
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
        return
    r = concurrent_case(base, container, "concurrent", body, CONCURRENCY)
    r["off_heap"] = plugin_stats(base)
    result["cases"]["concurrent_%d" % CONCURRENCY] = r
    print(label, "concurrent", json.dumps({k: v for k, v in r.items() if k != "off_heap"}), flush=True)
    result["breakers"] = breaker_stats(base)
    result["final_rss_mb"] = round(cgroup_rss(container) / 2**20)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    # What must hold on every build: the node answered every request, and the
    # refusals were OpenSearch's own.
    allowed = {"too_many_buckets_exception", "circuit_breaking_exception", "shard:circuit_breaking_exception",
               "shard:too_many_buckets_exception"}
    bad = []
    for name, c in result["cases"].items():
        statuses = c["status"] if isinstance(c["status"], list) else [int(k) for k in c["status"]]
        if -1 in statuses:
            bad.append("%s: the node did not answer" % name)
        errs = c["errors"] if isinstance(c["errors"], list) else list(c["errors"])
        bad += ["%s: %s" % (name, e) for e in errs if e not in allowed]
    for b in bad:
        print(label, "FAIL", b, flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
