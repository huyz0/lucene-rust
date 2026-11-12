"""Where a many-tenant index spends its time and memory: one configuration of
bench_tenants.py's corpus, loaded and then taken apart.

Usage:
  analyze_tenants.py BASE CONTAINER LABEL OUT_DIR SETTINGS_JSON [--tenants N] [--docs N] [--perf PERF]

SETTINGS_JSON adds index settings to bench_tenants.py's (for example
'{"segment_partition.field": "tenant"}'
or '{"sort.field": "tenant"}'). Written to OUT_DIR/LABEL-*:
  - stats.json: node stats after load (refresh, flush, merge totals), segment
    count and sizes, heap after GC, and search latency for each query shape as
    a filter the plugin may prune (`term`) and one it cannot (a `bool` of one
    `should`), plus a query that matches nothing;
  - profile-*.json: the search profile API's breakdown for one tenant of each class;
  - histo.txt: the heap's class histogram after a full GC;
  - jfr-*.txt: hot Java methods (JFR) while searching and while updating with
    `refresh=true`; perf-*.txt: the same for native code when PERF is given.
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_tenants as bt  # noqa: E402

JDK = "/usr/share/opensearch/jdk/bin/"


def sh(container, cmd):
    return subprocess.run(["docker", "exec", container, "bash", "-c", cmd],
                          capture_output=True, text=True).stdout


def jpid(container):
    return sh(container, JDK + "jcmd -l | grep -m1 org.opensearch.bootstrap | cut -d' ' -f1").strip()


def host_pid(container):
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", container],
                         capture_output=True, text=True).stdout.strip()
    # The java process is the container's init or its child.
    kids = subprocess.run(["bash", "-c", f"ps -o pid=,comm= --ppid {out}; ps -o pid=,comm= -p {out}"],
                          capture_output=True, text=True).stdout.split("\n")
    for line in kids:
        if "java" in line:
            return line.split()[0]
    return out


class Profiled:
    """JFR (and perf, when given) around a block, reports to OUT_DIR."""

    def __init__(self, ctx, name):
        self.ctx, self.name = ctx, name

    def __enter__(self):
        c = self.ctx
        sh(c["container"], f"{JDK}jcmd {c['jpid']} JFR.start name=p settings=profile filename=/tmp/{self.name}.jfr")
        self.perf = None
        if c["perf"]:
            self.perf = subprocess.Popen([c["perf"], "record", "-F", "499", "-g", "-a",
                                          "-o", f"/tmp/{self.name}.perf"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc):
        c = self.ctx
        sh(c["container"], f"{JDK}jcmd {c['jpid']} JFR.stop name=p")
        for view in ("hot-methods", "native-methods"):
            out = sh(c["container"], f"{JDK}jfr view --width 200 --truncate beginning {view} /tmp/{self.name}.jfr")
            with open(os.path.join(c["out"], f"{c['label']}-jfr-{self.name}-{view}.txt"), "w") as fh:
                fh.write(out)
        raw = sh(c["container"], f"{JDK}jfr print --events jdk.ExecutionSample --stack-depth 40 /tmp/{self.name}.jfr")
        with open(os.path.join(c["out"], f"{c['label']}-jfr-{self.name}-inclusive.txt"), "w") as fh:
            fh.write(inclusive(raw))
        if self.perf:
            self.perf.send_signal(2)
            self.perf.wait()
            rep = subprocess.run([c["perf"], "report", "-i", f"/tmp/{self.name}.perf", "--no-children",
                                  "--sort", "dso,symbol", "--stdio", "-g", "none", "--percent-limit", "0.5"],
                                 capture_output=True, text=True).stdout
            with open(os.path.join(c["out"], f"{c['label']}-perf-{self.name}.txt"), "w") as fh:
                fh.write(rep)


def inclusive(raw):
    """Each Java method's share of samples with it anywhere on the stack."""
    import collections
    import re
    counts, total = collections.Counter(), 0
    for ev in raw.split("jdk.ExecutionSample")[1:]:
        frames = set(re.findall(r"^\s+([\w$.<>]+)\(", ev, re.M))
        if frames:
            total += 1
            counts.update(frames)
    lines = [f"{total} samples"] + [f"{n * 100 / total:6.1f}%  {m}" for m, n in counts.most_common(80)]
    return "\n".join(lines) + "\n"


def filtered(k, query, wrap):
    term = {"term": {"tenant": "u%d" % k}}
    filt = term if wrap == "term" else {"bool": {"should": [term]}}
    return {"bool": {"must": query, "filter": filt}}


SHAPES = [
    ("term, frequent", {"term": {"body": "t0"}}, {}),
    ("match OR, 2 terms", {"match": {"body": "t3 t2s"}}, {}),
    ("terms agg on tag", {"match_all": {}}, {"size": 0, "aggs": {"a": {"terms": {"field": "tag", "size": 10}}}}),
    ("matches nothing", {"term": {"body": "absent"}}, {}),
]


def time_shapes(sizes, rounds, wraps):
    out = {}
    for cls, ks in bt.classes(sizes).items():
        out[cls] = {}
        for name, q, extra in SHAPES:
            for wrap in wraps:
                def body(k):
                    b = {"query": filtered(k, q, wrap), "track_total_hits": True}
                    b.update(extra)
                    return b
                for k in ks:
                    bt.req("POST", "/shared/_search?request_cache=false", body(k))
                wall = []
                for _ in range(rounds):
                    for k in ks:
                        t = time.perf_counter()
                        bt.req("POST", "/shared/_search?request_cache=false", body(k))
                        wall.append((time.perf_counter() - t) * 1000)
                out[cls][f"{name} [{wrap}]"] = round(statistics.median(wall), 3)
                print(f"  {cls:5s} {name:18s} {wrap:6s} {out[cls][f'{name} [{wrap}]']:7.2f} ms", flush=True)
    return out


def node_stats():
    n = next(iter(bt.req("GET", "/_nodes/stats/indices")["nodes"].values()))["indices"]
    return {k: n[k] for k in ("indexing", "refresh", "flush", "merges", "segments")}


def run(base, container, label, out, settings_extra, tenants, ndocs, perf):
    bt.BASE = base
    os.makedirs(out, exist_ok=True)
    ctx = {"container": container, "jpid": jpid(container), "hpid": host_pid(container),
           "perf": perf, "out": out, "label": label}
    docs, sizes = bt.corpus(tenants, ndocs)
    settings = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    settings.update(settings_extra)
    bt.req("PUT", "/shared", {"settings": settings, "mappings": bt.MAPPING})
    bt.req("GET", "/_cluster/health?wait_for_status=green&timeout=600s")
    try:
        engine = json.dumps(bt.req("GET", "/_plugins/lucene_rust/stats").get("engines"))
    except RuntimeError:
        engine = "stock"
    with Profiled(ctx, "load"):
        t = time.perf_counter()
        bt.send_all(bt.bulk_bodies("shared", docs))
        load_s = time.perf_counter() - t
    t = time.perf_counter()
    bt.req("POST", "/_refresh")
    refresh_s = time.perf_counter() - t
    segs = bt.req("GET", "/_cat/segments/shared?format=json&bytes=b")
    docs_per_seg = sorted(int(s["docs.count"]) for s in segs)
    result = {"label": label, "settings": settings, "engines": engine,
              "load_seconds": round(load_s, 2), "docs_per_second": round(ndocs / load_s),
              "refresh_seconds": round(refresh_s, 3), "segments": len(segs),
              "segment_docs_quartiles": [docs_per_seg[int(len(docs_per_seg) * q)] for q in (0, .25, .5, .75)]
              + [docs_per_seg[-1]],
              "after_load": node_stats(), "memory": bt.memory(container)}
    print(f"{label}: {result['segments']} segments, load {load_s:.1f}s, refresh {refresh_s:.2f}s, "
          f"heap {result['memory']['heap_after_gc_mib']} MiB", flush=True)
    with open(os.path.join(out, f"{label}-histo.txt"), "w") as fh:
        fh.write(sh(container, f"{JDK}jcmd {ctx['jpid']} GC.class_histogram | head -45"))

    wraps = ["term", "bool"] if "segment_partition.field" in settings or "sort.field" in settings else ["term"]
    result["search"] = time_shapes(sizes, 5, wraps)
    for cls, ks in bt.classes(sizes).items():
        b = {"query": filtered(ks[0], SHAPES[0][1], "bool"), "profile": True}
        with open(os.path.join(out, f"{label}-profile-{cls}.json"), "w") as fh:
            json.dump(bt.req("POST", "/shared/_search?request_cache=false", b), fh, indent=1)
    big = bt.classes(sizes)["large"]
    small = bt.classes(sizes)["small"]
    with Profiled(ctx, "search"):
        t = time.perf_counter()
        while time.perf_counter() - t < 20:
            for k in big[:5] + small[:5]:
                bt.req("POST", "/shared/_search?request_cache=false",
                       {"query": filtered(k, SHAPES[0][1], wraps[-1]), "track_total_hits": True})

    import random
    rnd = random.Random(11)
    picks = rnd.sample(docs, 150)
    wall = []
    with Profiled(ctx, "update"):
        for k, i, src in picks:
            t = time.perf_counter()
            bt.req("PUT", f"/shared/_doc/{i}?refresh=true", dict(src, num=src["num"] + 1))
            wall.append((time.perf_counter() - t) * 1000)
    wall.sort()
    result["update_refresh_true_ms"] = {"p50": round(statistics.median(wall), 2),
                                        "p90": round(wall[int(len(wall) * .9) - 1], 2)}
    before = node_stats()["refresh"]
    t = time.perf_counter()
    for _ in range(20):
        bt.req("POST", "/shared/_refresh")
    result["noop_refresh_ms"] = round((time.perf_counter() - t) * 1000 / 20, 2)
    result["refresh_delta"] = {k: node_stats()["refresh"][k] - before[k] for k in before
                               if isinstance(before[k], int)}
    result["end"] = node_stats()
    result["segments_end"] = len(bt.req("GET", "/_cat/segments/shared?format=json"))
    print(f"{label}: update+refresh {result['update_refresh_true_ms']}, no-op refresh "
          f"{result['noop_refresh_ms']} ms, {result['segments_end']} segments", flush=True)
    with open(os.path.join(out, f"{label}-stats.json"), "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 5:
        sys.exit(__doc__)

    def opt(name, default, cast=int):
        return cast(a[a.index(name) + 1]) if name in a else default
    run(a[0], a[1], a[2], a[3], json.loads(a[4]), opt("--tenants", 500), opt("--docs", 500_000),
        opt("--perf", None, str))
