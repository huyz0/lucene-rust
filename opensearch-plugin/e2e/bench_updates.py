"""Update-rate baseline: a counter field updated on a 1M-document index at a fixed rate
while merges run, a bulk update of a share of the corpus, and a trickle of updates.

Usage:
  bench_updates.py run BASE NODE LABEL OUT CORPUS_DIR [--docs N] [--rates 1000,5000,20000]
                       [--seconds S] [--mode update|index] [--dist hot|uniform]
                       [--bulk-shares 0.1,0.5] [--trickle-rate R] [--skip-load]

BASE is the node's URL, NODE its nodectl name (RSS). CORPUS_DIR is `bench_nodes.py gen`'s output.
Index `bench` is created with one shard, no replica, refresh_interval from REFRESH (default 1s).

Each stream phase: SENDERS clients send bulk requests of BATCH operations on a schedule that
meets the target rate; two probes run beside them (a term query, a point read by id), and the
node's CPU and RSS are read before and after. The counter is the mapping's `num` field.
`update` sends `{"update": ..., "doc": {"num": v}}` (OpenSearch fetches the source, merges,
reindexes); `index` sends the whole document again.

Result per phase: achieved updates per second, bulk latency, probe latency, CPU seconds and CPU
per update, RSS, and the index's segment count and deleted documents at the end."""
import bisect
import glob
import json
import os
import random
import sys
import threading
import time

import benchhttp as bh
import nodectl

SENDERS = int(os.environ.get("SENDERS", "4"))
BATCH = int(os.environ.get("BATCH", "200"))
REFRESH = os.environ.get("REFRESH", "1s")
HOT_SET = int(os.environ.get("HOT_SET", "10000"))
HOT_SHARE = float(os.environ.get("HOT_SHARE", "0.8"))

BASE = ""
NODE = ""
MAPPING = None


def req(method, path, body=None, ndjson=False, timeout=600):
    return bh.req(BASE, method, path, body, ndjson, timeout)


def cpu_seconds():
    nodes = req("GET", "/_nodes/stats/process")["nodes"]
    return next(iter(nodes.values()))["process"]["cpu"]["total_in_millis"] / 1000


def index_state():
    st = req("GET", "/bench/_stats/docs,segments,store")["_all"]["primaries"]
    return {"docs": st["docs"]["count"], "deleted": st["docs"]["deleted"],
            "segments": st["segments"]["count"], "store_bytes": st["store"]["size_in_bytes"]}


def pick(rnd, n, dist):
    if dist == "hot" and rnd.random() < HOT_SHARE:
        return rnd.randrange(min(HOT_SET, n))
    return rnd.randrange(n)


def bulk_body(ids, mode, rnd, full):
    out = []
    for i in ids:
        v = rnd.randrange(1_000_000_000)
        if mode == "update":
            out.append('{"update":{"_id":"%d"}}' % i)
            out.append('{"doc":{"num":%d}}' % v)
        else:
            doc = dict(full[i % len(full)])
            doc["num"] = v
            out.append('{"index":{"_id":"%d"}}' % i)
            out.append(json.dumps(doc, separators=(",", ":")))
    return ("\n".join(out) + "\n").encode()


class Probe(threading.Thread):
    """A term query and a point read, alternating, about PER_S a second."""

    def __init__(self, n, per_s):
        super().__init__(daemon=True)
        self.n, self.per_s = n, per_s
        self.stop = threading.Event()
        self.term, self.get, self.sort = [], [], []

    def run(self):
        rnd = random.Random(5)
        gap = 1.0 / self.per_s
        k = 0
        nxt = time.perf_counter()
        while not self.stop.is_set():
            nxt += gap
            t = time.perf_counter()
            kind = k % 3
            k += 1
            try:
                if kind == 0:
                    req("POST", "/bench/_search?request_cache=false",
                        {"query": {"term": {"body": "t2s"}}, "size": 10})
                    self.term.append((time.perf_counter() - t) * 1000)
                elif kind == 1:
                    req("GET", "/bench/_doc/%d" % pick(rnd, self.n, "hot"))
                    self.get.append((time.perf_counter() - t) * 1000)
                else:
                    req("POST", "/bench/_search?request_cache=false",
                        {"query": {"match_all": {}}, "sort": [{"num": "desc"}], "size": 10, "_source": False})
                    self.sort.append((time.perf_counter() - t) * 1000)
            except RuntimeError:
                pass
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)

    def result(self):
        return {name: bh.summarize(v) for name, v in
                (("term", self.term), ("point_read", self.get), ("sort_num", self.sort)) if v}


def stream(n, rate, seconds, mode, dist, full, label):
    """Hold `rate` updates a second for `seconds`; None for rate = 0 means as fast as possible
    is `bulk` below."""
    probe = Probe(n, 60)
    batch_ms = []
    lock = threading.Lock()
    done = [0]
    errs = [0]
    t_start = [0.0]
    deadline = [0.0]

    def sender(idx):
        rnd = random.Random(1000 + idx)
        interval = BATCH * SENDERS / rate
        nxt = t_start[0] + idx * interval / SENDERS
        while True:
            now = time.perf_counter()
            if nxt >= deadline[0]:
                return
            if nxt > now:
                time.sleep(nxt - now)
            ids = [pick(rnd, n, dist) for _ in range(BATCH)]
            body = bulk_body(ids, mode, rnd, full)
            t = time.perf_counter()
            res = req("POST", "/bench/_bulk", body, ndjson=True)
            ms = (time.perf_counter() - t) * 1000
            with lock:
                batch_ms.append(ms)
                done[0] += BATCH
                if res.get("errors"):
                    errs[0] += 1
            nxt += interval

    cpu0 = cpu_seconds()
    probe.start()
    t_start[0] = time.perf_counter()
    deadline[0] = t_start[0] + seconds
    ths = [threading.Thread(target=sender, args=(i,)) for i in range(SENDERS)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t_start[0]
    probe.stop.set()
    probe.join()
    cpu1 = cpu_seconds()
    out = {
        "target_rate": rate, "seconds": round(wall, 1), "updates": done[0],
        "achieved_rate": round(done[0] / wall), "bulk_errors": errs[0],
        "bulk_ms": bh.summarize(batch_ms) if batch_ms else None,
        "cpu_seconds": round(cpu1 - cpu0, 2),
        "cpu_us_per_update": round((cpu1 - cpu0) * 1e6 / max(1, done[0]), 1),
        "rss_mb": round(nodectl.rss(NODE) / 2**20) if NODE else None,
        "probes": probe.result(),
        **index_state(),
    }
    print(f"{label}: {json.dumps(out)}", flush=True)
    return out


def bulk(n, share, mode, full, label):
    """Update `share` of the corpus, distinct random ids, as fast as SENDERS clients can."""
    ids = random.Random(99).sample(range(n), int(n * share))
    chunks = [ids[i:i + BATCH * 5] for i in range(0, len(ids), BATCH * 5)]
    it = iter(chunks)
    lock = threading.Lock()
    probe = Probe(n, 20)

    def sender(idx):
        rnd = random.Random(2000 + idx)
        while True:
            with lock:
                c = next(it, None)
            if c is None:
                return
            res = req("POST", "/bench/_bulk", bulk_body(c, mode, rnd, full), ndjson=True)
            if res.get("errors"):
                raise RuntimeError("bulk errors")

    cpu0 = cpu_seconds()
    probe.start()
    t0 = time.perf_counter()
    ths = [threading.Thread(target=sender, args=(i,)) for i in range(SENDERS)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    probe.stop.set()
    probe.join()
    cpu1 = cpu_seconds()
    out = {"share": share, "updates": len(ids), "seconds": round(wall, 1),
           "updates_per_s": round(len(ids) / wall),
           "cpu_seconds": round(cpu1 - cpu0, 2),
           "cpu_us_per_update": round((cpu1 - cpu0) * 1e6 / len(ids), 1),
           "rss_mb": round(nodectl.rss(NODE) / 2**20) if NODE else None,
           "probes": probe.result(), **index_state()}
    print(f"{label}: {json.dumps(out)}", flush=True)
    return out


def trickle(n, rate, seconds, mode, full, label):
    """A few updates a second, one request each, beside the same probes, and the node's CPU over the
    same time with nothing sent for comparison."""
    probe = Probe(n, 20)
    cpu0 = cpu_seconds()
    probe.start()
    t0 = time.perf_counter()
    time.sleep(seconds)
    probe.stop.set()
    probe.join()
    idle_cpu = cpu_seconds() - cpu0
    idle_probes = probe.result()

    probe = Probe(n, 20)
    rnd = random.Random(77)
    lat = []
    sent = 0
    cpu0 = cpu_seconds()
    probe.start()
    t0 = time.perf_counter()
    nxt = t0
    while time.perf_counter() - t0 < seconds:
        nxt += 1.0 / rate
        i = rnd.randrange(n)
        t = time.perf_counter()
        req("POST", "/bench/_bulk", bulk_body([i], mode, rnd, full), ndjson=True)
        lat.append((time.perf_counter() - t) * 1000)
        sent += 1
        d = nxt - time.perf_counter()
        if d > 0:
            time.sleep(d)
    wall = time.perf_counter() - t0
    probe.stop.set()
    probe.join()
    cpu = cpu_seconds() - cpu0
    out = {"rate": rate, "updates": sent, "seconds": round(wall, 1),
           "update_ms": bh.summarize(lat), "cpu_seconds": round(cpu, 2),
           "idle_cpu_seconds": round(idle_cpu, 2),
           "cpu_ms_per_update_over_idle": round((cpu - idle_cpu) * 1000 / sent, 2),
           "probes": probe.result(), "idle_probes": idle_probes, **index_state()}
    print(f"{label}: {json.dumps(out)}", flush=True)
    return out


def load_corpus(files):
    t0 = time.perf_counter()
    lock = threading.Lock()
    it = iter(files)

    def worker():
        while True:
            with lock:
                f = next(it, None)
            if f is None:
                return
            with open(f, "rb") as fh:
                res = req("POST", "/bench/_bulk", fh.read(), ndjson=True)
            if res["errors"]:
                raise RuntimeError("load errors")

    ths = [threading.Thread(target=worker) for _ in range(SENDERS)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    req("POST", "/bench/_refresh")
    return time.perf_counter() - t0


def main(argv):
    global BASE, NODE, MAPPING
    import bench_nodes
    a = argv[1:]
    if not a or a[0] != "run":
        print(__doc__)
        return 2
    BASE, NODE, label, out, corpus = a[1], a[2], a[3], a[4], a[5]
    opts = {"--docs": "1000000", "--rates": "1000,5000,20000", "--seconds": "120", "--mode": "update",
            "--dist": "hot", "--bulk-shares": "0.1,0.5", "--trickle-rate": "20", "--settings": "{}",
            "--trickle-seconds": "60"}
    flags = {"--skip-load"}
    i = 6
    skip_load = False
    while i < len(a):
        if a[i] in flags:
            skip_load = True
            i += 1
        else:
            opts[a[i]] = a[i + 1]
            i += 2
    n = int(opts["--docs"])
    mode, dist = opts["--mode"], opts["--dist"]
    files = sorted(glob.glob(os.path.join(corpus, "bulk-*.ndjson")))
    full = []
    if mode == "index":
        with open(files[0]) as fh:
            lines = fh.read().split("\n")
        full = [json.loads(lines[j]) for j in range(1, 400, 2)]
    result = {"label": label, "mode": mode, "dist": dist, "docs": n, "hot_set": HOT_SET,
              "hot_share": HOT_SHARE, "batch": BATCH, "senders": SENDERS}
    if not skip_load:
        try:
            req("DELETE", "/bench")
        except RuntimeError:
            pass
        settings = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": REFRESH}
        settings.update(json.loads(opts["--settings"]))
        req("PUT", "/bench", {"settings": settings, "mappings": bench_nodes.MAPPING})
        result["load_seconds"] = round(load_corpus(files[:n // bench_nodes.PER_FILE]), 1)
        print(f"{label}: loaded {n} in {result['load_seconds']}s", flush=True)
    result["start"] = index_state()
    result["idle_rss_mb"] = round(nodectl.rss(NODE) / 2**20) if NODE else None
    time.sleep(15)
    result["streams"] = []
    for rate in [int(x) for x in opts["--rates"].split(",") if x]:
        result["streams"].append(stream(n, rate, int(opts["--seconds"]), mode, dist, full,
                                        f"{label} stream {rate}/s"))
        time.sleep(10)
    result["bulk"] = []
    for share in [float(x) for x in opts["--bulk-shares"].split(",") if x]:
        result["bulk"].append(bulk(n, share, mode, full, f"{label} bulk {share}"))
        time.sleep(10)
    if opts["--trickle-rate"] != "0":
        result["trickle"] = trickle(n, int(opts["--trickle-rate"]), int(opts["--trickle-seconds"]),
                                    mode, full, f"{label} trickle")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"{label}: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
