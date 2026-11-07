#!/usr/bin/env python3
"""The plugin's _plugins/_knn REST endpoints against stock k-NN's.

One plugin node and one stock node take the same indexes and the same knn
queries. Then both answer the k-NN stats, warmup and clear_cache calls, and
the answers are compared:

  stats         the same keys at every level, cluster and node, for the full
                listing, one stat, a group stat, ?stat= and a node filter;
                the query counters (knn_query_requests, with filter, min_score,
                max_distance) hold the same values
  warmup        the same _shards counts for knn indexes; the same error status,
  clear_cache   type and reason for a non-knn index and a missing one
  errors        an unknown stat and _all mixed with names give stock's 400

  scripts/check-knn-rest.py [--keep]

Needs Docker and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import glob, importlib.util, os, random, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("check_knn", os.path.join(ROOT, "scripts", "check-knn-plugin.py"))
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

OURS = ("knn-rest-ours", 19224)
STOCK = ("knn-rest-stock", 19225)
COUNTERS = ("knn_query_requests", "knn_query_with_filter_requests",
            "min_score_query_requests", "min_score_query_with_filter_requests",
            "max_distance_query_requests", "max_distance_query_with_filter_requests")


def setup(port):
    rng = random.Random(3)
    for index in ("vec-a", "vec-b"):
        ck.req(port, "PUT", f"/{index}", {
            "settings": {"index": {"knn": True, "number_of_shards": 2, "number_of_replicas": 0}},
            "mappings": {"properties": {"emb": {"type": "knn_vector", "dimension": 4,
                                                "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"}},
                                        "tag": {"type": "keyword"}}}})
        lines = []
        for i in range(50):
            lines.append('{"index":{"_id":"%d"}}' % i)
            lines.append('{"tag":"%s","emb":[%s]}' % ("even" if i % 2 == 0 else "odd",
                                                      ",".join(f"{rng.random():.4f}" for _ in range(4))))
        ck.req(port, "POST", f"/{index}/_bulk?refresh=true", "\n".join(lines) + "\n", ndjson=True)
    ck.req(port, "PUT", "/plain", {"settings": {"number_of_replicas": 0},
                                   "mappings": {"properties": {"tag": {"type": "keyword"}}}})
    target = [0.5, 0.5, 0.5, 0.5]
    flt = {"term": {"tag": "even"}}
    for body in ({"k": 5}, {"k": 5, "filter": flt}, {"min_score": 0.5}, {"min_score": 0.5, "filter": flt},
                 {"max_distance": 1.0}, {"max_distance": 1.0, "filter": flt}, {"k": 3}):
        ck.req(port, "POST", "/vec-a/_search", {"query": {"knn": {"emb": {"vector": target, **body}}}})


def shape(v):
    if isinstance(v, dict):
        return {k: shape(x) for k, x in v.items()}
    return type(v).__name__ if v is not None else "null"


def keys(v, path=""):
    out = set()
    if isinstance(v, dict):
        for k, x in v.items():
            out.add(f"{path}/{k}")
            out |= keys(x, f"{path}/{k}")
    return out


def node_stats(resp):
    nodes = resp.get("nodes", {})
    return next(iter(nodes.values())) if len(nodes) == 1 else None


def compare_stats(path, failures):
    ours, stock = ck.req(OURS[1], "GET", path), ck.req(STOCK[1], "GET", path)
    top_o = {k for k in ours if k != "nodes"}
    top_s = {k for k in stock if k != "nodes"}
    if top_o != top_s:
        failures.append(f"{path}: top-level keys {sorted(top_o ^ top_s)} differ")
    no, ns = node_stats(ours), node_stats(stock)
    if no is None or ns is None:
        failures.append(f"{path}: expected one node, got {ours.get('nodes')} / {stock.get('nodes')}")
        return
    ko, ks = keys(no), keys(ns)
    if ko != ks:
        failures.append(f"{path}: node keys only ours {sorted(ko - ks)[:10]}, only stock {sorted(ks - ko)[:10]}")
    for k in set(no) & set(ns):
        if shape(no[k]) != shape(ns[k]) and not (isinstance(no[k], (int, float)) and isinstance(ns[k], (int, float))):
            failures.append(f"{path}: {k} is {shape(no[k])} on ours, {shape(ns[k])} on stock")
    for c in COUNTERS:
        if c in ns and no.get(c) != ns[c]:
            failures.append(f"{path}: {c} = {no.get(c)}, stock {ns[c]}")
    for k in ("circuit_breaker_triggered", "model_index_status"):
        if k in stock and ours.get(k) != stock[k]:
            failures.append(f"{path}: {k} = {ours.get(k)!r}, stock {stock[k]!r}")


def error(port, method, path):
    r = ck.req(port, method, path, ok_errors=True)
    if "status" not in r or "error" not in r:
        return ("ok", r.get("_shards"))
    e = (r["error"] or {}).get("error", {})
    return (r["status"], e.get("type"), e.get("reason"))


def same_answer(method, path, failures):
    o, s = error(OURS[1], method, path), error(STOCK[1], method, path)
    if o != s:
        failures.append(f"{method} {path}: ours {o}, stock {s}")


def main():
    keep = "--keep" in sys.argv
    zips = glob.glob(os.path.join(ROOT, "opensearch-plugin", "build", "distributions", "lucene-rust-*.zip"))
    if not zips:
        sys.exit("no plugin zip: run gradle -p opensearch-plugin bundlePlugin")
    docker_dir = os.path.join(ROOT, "opensearch-plugin", "docker")
    for z in glob.glob(os.path.join(docker_dir, "*.zip")):
        os.remove(z)
    shutil.copy(zips[0], docker_dir)
    image = f"lucene-rust-opensearch:{ck.VERSION}"
    subprocess.run(["docker", "build", "-q", "-t", image, docker_dir], check=True, capture_output=True)
    ck.start(OURS[0], OURS[1], image)
    ck.start(STOCK[0], STOCK[1], f"opensearchproject/opensearch:{ck.VERSION}")
    failures = []
    try:
        ck.wait(OURS[1])
        ck.wait(STOCK[1])
        for port in (OURS[1], STOCK[1]):
            setup(port)
        stats = ["/_plugins/_knn/stats", "/_plugins/_knn/stats/knn_query_requests",
                 "/_plugins/_knn/stats/graph_stats", "/_plugins/_knn/stats/knn_query_requests,hit_count",
                 "/_plugins/_knn/stats?stat=knn_query_requests", "/_plugins/_knn/_local/stats",
                 "/_plugins/_knn/_all/stats/_all"]
        for path in stats:
            compare_stats(path, failures)
        calls = [("GET", "/_plugins/_knn/stats/bogus"), ("GET", "/_plugins/_knn/stats/_all,hit_count"),
                 ("GET", "/_plugins/_knn/warmup/vec-a"), ("GET", "/_plugins/_knn/warmup/vec-a,vec-b"),
                 ("GET", "/_plugins/_knn/warmup/vec-*"), ("GET", "/_plugins/_knn/warmup/plain"),
                 ("GET", "/_plugins/_knn/warmup/missing"), ("POST", "/_plugins/_knn/clear_cache/vec-a"),
                 ("POST", "/_plugins/_knn/clear_cache/plain"), ("POST", "/_plugins/_knn/clear_cache/missing")]
        for method, path in calls:
            same_answer(method, path, failures)
        print(f"{len(stats)} stats calls, {len(calls)} warmup/clear_cache/error calls compared")
    finally:
        if not keep:
            for name in (OURS[0], STOCK[0]):
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    if failures:
        print("\n".join(failures))
        sys.exit(1)
    print("ok")


if __name__ == "__main__":
    main()
