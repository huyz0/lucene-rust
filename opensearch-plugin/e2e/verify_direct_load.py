#!/usr/bin/env python3
"""Slow searches must not hold the HTTP I/O threads. A plugin node (direct search on, the default) holds a 1M-document
index; with 8 clients running slow queries (wildcard, regexp, a script sort) other REST traffic -- cluster health,
the root endpoint, single-document indexing -- and cheap searches must stay quick, and a request the native path leaves
(a body feature it does not take) must still be answered by OpenSearch.

    python3 opensearch-plugin/e2e/verify_direct_load.py [SECONDS]        (plugin on :9250)"""
import json
import random
import sys
import threading
import time
import urllib.request
import concurrent.futures as cf

B = "http://localhost:9250"
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 15
LIMIT_MS = 100


def call(method, path, body=None, timeout=120):
    data = body if isinstance(body, bytes) else (None if body is None else json.dumps(body).encode())
    r = urllib.request.Request(B + path, data=data, method=method, headers={"content-type": "application/json"})
    s = time.time()
    with urllib.request.urlopen(r, timeout=timeout) as f:
        out = json.load(f)
    return (time.time() - s) * 1000, out


def build():
    try:
        call("DELETE", "/load")
    except Exception:
        pass
    call("PUT", "/load", {"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0, "index.engine.accelerated": True, "refresh_interval": "-1"},
                          "mappings": {"properties": {"tag": {"type": "keyword"}, "n": {"type": "long"}, "u": {"type": "keyword"}}}})

    def chunk(a):
        r = random.Random(a)
        lines = []
        for i in range(a * 50000, (a + 1) * 50000):
            lines += ['{"index":{}}', json.dumps({"tag": "t%d" % r.randint(0, 99), "n": r.randint(0, 10**6), "u": "u%d" % i})]
        call("POST", "/load/_bulk", ("\n".join(lines) + "\n").encode(), 600)
    with cf.ThreadPoolExecutor(3) as ex:
        list(ex.map(chunk, range(20)))
    call("POST", "/load/_refresh")
    call("POST", "/load/_forcemerge?max_num_segments=2", None, 600)


SLOW = [{"size": 10, "track_total_hits": True, "query": {"wildcard": {"u": "*99*"}}},
        {"size": 10, "track_total_hits": True, "query": {"regexp": {"u": ".*7.*3.*"}}},
        {"size": 10, "sort": {"_script": {"type": "number", "script": {"source": "doc['n'].value % 1000 + Math.sin(doc['n'].value)"}, "order": "asc"}},
         "query": {"match_all": {}}}]
CHEAP = {"size": 10, "query": {"term": {"tag": "t5"}}}


def main():
    build()
    for q in SLOW + [CHEAP]:
        call("POST", "/load/_search", q)
    lat = {"slow": [], "cheap": [], "health": [], "root": [], "index": []}
    stop = time.time() + DUR

    def loop(kind, fn, pause=0.0):
        while time.time() < stop:
            lat[kind].append(fn())
            time.sleep(pause)
    work = [("slow", lambda: call("POST", "/load/_search", random.choice(SLOW))[0], 0)] * 8
    work += [("cheap", lambda: call("POST", "/load/_search", CHEAP)[0], 0)] * 4
    work += [("health", lambda: call("GET", "/_cluster/health")[0], 0.05),
             ("root", lambda: call("GET", "/")[0], 0.05),
             ("index", lambda: call("PUT", "/other/_doc/%d" % random.randint(0, 999), {"a": 1})[0], 0.05)]
    threads = [threading.Thread(target=loop, args=w) for w in work]
    [t.start() for t in threads]
    [t.join() for t in threads]
    bad = []
    for kind, v in lat.items():
        v.sort()
        p50, p99 = v[len(v) // 2], v[max(int(len(v) * .99) - 1, 0)]
        print("%-7s n=%5d p50=%7.1f p99=%8.1f max=%8.1f ms" % (kind, len(v), p50, p99, v[-1]))
        if kind in ("health", "root", "index") and p50 > LIMIT_MS:
            bad.append("%s p50 %.0f ms while slow searches run" % (kind, p50))
    # A body feature the native path leaves must still be answered.
    status, r = None, call("POST", "/load/_search", {"size": 1, "query": {"match_all": {}}, "highlight": {"fields": {"u": {}}}})[1]
    if "hits" not in r:
        bad.append("a request left to OpenSearch got no answer")
    print("%d problems" % len(bad), *bad)
    sys.exit(1 if bad else 0)


main()
