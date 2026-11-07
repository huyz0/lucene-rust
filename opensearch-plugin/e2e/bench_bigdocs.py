"""Big documents: memory and throughput while documents of SIZE_MB each are
indexed one per bulk request, with CLIENTS parallel clients.

Usage:
  bench_bigdocs.py BASE CONTAINER LABEL OUT SETTINGS_JSON

The index has FIELDS fields besides `body`: a third keyword, a third long, a
third text. `body` is a text field of about SIZE_MB of Zipf words. Docker RSS
is sampled every 0.2 s (open files every 1 s) while the documents are sent (peak and last), then
after a refresh. A search for a rare term in the last document checks that
the documents were indexed."""
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

SIZE_MB = float(os.environ.get("SIZE_MB", "1"))
DOCS = int(os.environ.get("DOCS", "200"))
FIELDS = int(os.environ.get("FIELDS", "200"))
CLIENTS = int(os.environ.get("CLIENTS", "2"))


def req(base, method, path, body=None, ndjson=False):
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    headers = {"content-type": "application/x-ndjson" if ndjson else "application/json"}
    r = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(r, timeout=1200) as resp:
        return json.load(resp)


def mem_bytes(container):
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
                         capture_output=True, text=True).stdout.split("/")[0].strip()
    units = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "B": 1}
    for u, m in units.items():
        if out.endswith(u):
            return float(out[: -len(u)]) * m
    return 0.0


def open_files(container):
    top = subprocess.run(["docker", "top", container, "-o", "pid,args"], capture_output=True, text=True).stdout
    pids = [l.split()[0] for l in top.splitlines()[1:] if "org.opensearch.bootstrap" in l]
    try:
        return len(os.listdir("/proc/%s/fd" % pids[0])) if pids else 0
    except OSError:
        return 0


def breaker_bytes(base):
    try:
        n = next(iter(req(base, "GET", "/_nodes/stats/breaker")["nodes"].values()))
        return n["breakers"].get("lucene_rust_writer", {}).get("estimated_size_in_bytes", 0)
    except Exception:
        return 0


def mapping():
    props = {"body": {"type": "text"}}
    for i in range(1, int(SIZE_MB) // 40 + 3):
        props["body%d" % i] = {"type": "text"}
    for i in range(FIELDS):
        props["f%d" % i] = {"type": ("keyword", "long", "text")[i % 3]}
    return {"properties": props}


def doc(n, rnd, words):
    target = int(SIZE_MB * 2**20)
    parts, size = [], 0
    while size < target:
        w = words[min(int(rnd.paretovariate(1.0)) - 1, len(words) - 1)]
        parts.append(w)
        size += len(w) + 1
    parts.append("unique%d" % n)
    # OpenSearch refuses a JSON string over 50 MB: a bigger body is split.
    per = max(1, len(parts) * 40 // max(1, int(SIZE_MB) + 1)) if SIZE_MB > 40 else len(parts)
    d = {("body" if i == 0 else "body%d" % i): " ".join(parts[s:s + per])
         for i, s in enumerate(range(0, len(parts), per))}
    for i in range(FIELDS):
        kind = i % 3
        d["f%d" % i] = ("k%d" % rnd.randrange(100), rnd.randrange(10**6), "w%d x%d" % (rnd.randrange(50), n))[kind]
    return ('{"index":{"_id":"%d"}}\n' % n + json.dumps(d, separators=(",", ":")) + "\n").encode()


def main():
    base, container, label, out, settings = sys.argv[1:6]
    req(base, "PUT", "/big", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, **json.loads(settings)},
                              "mappings": mapping()})
    rnd = random.Random(1)
    words = ["t%d" % i for i in range(50_000)]
    bodies = [doc(n, rnd, words) for n in range(min(DOCS, 4))]
    idle = mem_bytes(container)
    samples, fds, breaker, stop = [], [open_files(container)], [0], threading.Event()

    def sample():
        i = 0
        while not stop.is_set():
            samples.append(mem_bytes(container))
            breaker.append(breaker_bytes(base))
            if i % 5 == 0:
                fds.append(open_files(container))
            i += 1
            stop.wait(0.2)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    t0 = time.perf_counter()

    def send(n):
        body = bodies[n % len(bodies)].replace(b"unique%d" % (n % len(bodies)), b"unique%d" % n, 1)
        res = req(base, "POST", "/big/_bulk", body.replace(b'"_id":"%d"' % (n % len(bodies)), b'"_id":"%d"' % n, 1), ndjson=True)
        if res["errors"]:
            raise RuntimeError(json.dumps(res)[:500])

    with concurrent.futures.ThreadPoolExecutor(CLIENTS) as ex:
        list(ex.map(send, range(DOCS)))
    indexed = time.perf_counter() - t0
    found = req(base, "POST", "/big/_search", {"size": 0, "query": {"term": {"body": "unique%d" % (DOCS - 1)}}})["hits"]["total"]["value"]
    stop.set()
    t.join()
    req(base, "POST", "/big/_refresh")
    after = mem_bytes(container)
    gib = lambda b: round(b / 2**30, 3)
    res = {"label": label, "size_mb": SIZE_MB, "docs": DOCS, "fields": FIELDS, "seconds": round(indexed, 2),
           "mb_per_sec": round(DOCS * SIZE_MB / indexed, 1), "rss_idle_gib": gib(idle),
           "rss_peak_gib": gib(max(samples or [0])), "rss_after_refresh_gib": gib(after),
           "visible_before_refresh": found,
           "open_files_idle": fds[0], "open_files_peak": max(fds),
           "breaker_peak_gib": gib(max(breaker))}
    if os.environ.get("BREAKER_SERIES"):
        print("breaker MB:", [round(b / 2**20) for b in breaker[::2]])
        print("rss GiB:", [round(b / 2**30, 2) for b in samples[::2]])
    print(f"{label}: {json.dumps(res)}")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
