#!/usr/bin/env python3
"""The plugin's knn REST output against stock k-NN's.

One plugin node and one stock node take the same indexes and the same knn
queries. Then both answer the k-NN stats, warmup and clear_cache calls, the
mapping and explain APIs, and the answers are compared:

  stats         the same keys at every level, cluster and node, for the full
                listing, one stat, a group stat, ?stat= and a node filter;
                the query counters (knn_query_requests, with filter, min_score,
                max_distance) hold the same values
  warmup        the same _shards counts for knn indexes; the same error status,
  clear_cache   type and reason for a non-knn index and a missing one
  errors        an unknown stat and _all mixed with names give stock's 400
  mapping       GET _mapping persists each knn_vector mapping as stock does
                (method filled in with its engine, space type and parameters,
                data_type upper case); where no engine is named, stock's
                default is faiss and the plugin persists lucene, so a stock
                node reading the index sees the lucene field it is
  explain       a knn hit's explanation
  _source       a document's _source by get (from the translog, then from the
                index) and by search, numbers' types included, with derived
                source on as stock turns it on for a knn index (the plugin's
                index on the Rust engine, which masks the vectors); the masked
                index must store less than one with derived source off

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


MAPPINGS = {
    "plain": {"type": "knn_vector", "dimension": 4},
    "space": {"type": "knn_vector", "dimension": 4, "space_type": "innerproduct"},
    "method": {"type": "knn_vector", "dimension": 4, "method": {"name": "hnsw", "parameters": {"m": 8}}},
    "lucene": {"type": "knn_vector", "dimension": 4,
               "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2",
                          "parameters": {"m": 8, "ef_construction": 64}}},
    "bare": {"type": "knn_vector", "dimension": 4, "method": {"name": "hnsw", "engine": "lucene"}},
    "top": {"type": "knn_vector", "dimension": 4, "space_type": "innerproduct", "engine": "lucene",
            "method": {"name": "hnsw"}},
    "sq": {"type": "knn_vector", "dimension": 4,
           "method": {"name": "hnsw", "engine": "lucene", "parameters": {"encoder": {"name": "sq", "parameters": {"bits": 4}}}}},
    "mode": {"type": "knn_vector", "dimension": 4, "mode": "on_disk"},
    "mode_lucene": {"type": "knn_vector", "dimension": 4, "engine": "lucene", "mode": "on_disk"},
    "comp": {"type": "knn_vector", "dimension": 4, "compression_level": "4x"},
    "byte": {"type": "knn_vector", "dimension": 4, "data_type": "byte"},
    "float": {"type": "knn_vector", "dimension": 4, "data_type": "FLOAT"},
    "stored": {"type": "knn_vector", "dimension": 4, "doc_values": False, "store": True},
}


def stock_engine_to_lucene(spec, persisted):
    """The stock mapping as the plugin persists it: lucene where stock resolved faiss."""
    out = dict(persisted)
    if "method" in out:
        out["method"] = {**out["method"], "engine": "lucene"}
    elif "engine" not in spec:
        out["engine"] = "lucene"
    return out


def compare_mappings(failures):
    got = {}
    for port in (OURS[1], STOCK[1]):
        ck.req(port, "PUT", "/mappings", {"settings": {"index": {"knn": True, "number_of_replicas": 0}},
                                          "mappings": {"properties": MAPPINGS}})
        ck.req(port, "PUT", "/mappings/_doc/1?refresh=true", {k: [1, 2, 3, 4] for k in MAPPINGS})
        got[port] = ck.req(port, "GET", "/mappings/_mapping")["mappings"]["mappings"]["properties"]
        # The persisted mapping must parse again: an update repeating the request's mapping is no conflict.
        ck.req(port, "PUT", "/mappings/_mapping", {"properties": MAPPINGS})
    for name, spec in MAPPINGS.items():
        want = stock_engine_to_lucene(spec, got[STOCK[1]][name])
        if got[OURS[1]][name] != want:
            failures.append(f"mapping {name}: ours {got[OURS[1]][name]}, stock {got[STOCK[1]][name]}")
    return len(MAPPINGS)


def typed(v):
    """v with each number's JSON type kept: 1 and 1.0 differ."""
    if isinstance(v, dict):
        return {k: typed(x) for k, x in v.items()}
    if isinstance(v, list):
        return [typed(x) for x in v]
    return (type(v).__name__, v)


def sources(port, index):
    out = {"get": ck.req(port, "GET", f"/{index}/_doc/1")["_source"]}
    ck.req(port, "POST", f"/{index}/_refresh")
    out["get_refreshed"] = ck.req(port, "GET", f"/{index}/_doc/1")["_source"]
    out["search"] = ck.req(port, "POST", f"/{index}/_search", {"query": {"ids": {"values": ["1"]}}})["hits"]["hits"][0]["_source"]
    return out


def compare_sources(failures):
    doc = {"tag": "x", "secret": "s", "hidden": [5, 5, 5, 5], "nested_obj": {"inner": [9, 8, 7, 6]},
           **{k: [1, 2, 3, 4] for k in MAPPINGS}}
    props = {**MAPPINGS, "tag": {"type": "keyword"}, "secret": {"type": "keyword"},
             "hidden": {"type": "knn_vector", "dimension": 4},
             "nested_obj": {"properties": {"inner": {"type": "knn_vector", "dimension": 4}}}}
    n = 0
    # Filtered and whole _source; a copy_to field, which stock leaves whole.
    variants = (("sources", {"excludes": ["secret", "hidden"]}, {}),
                ("sources-whole", None, {"copied": {"type": "knn_vector", "dimension": 4, "copy_to": "copied_all"},
                                         "copied_all": {"type": "knn_vector", "dimension": 4}}))
    for index, source, extra in variants:
        got = {}
        mappings = {"properties": {**props, **extra}}
        if source:
            mappings["_source"] = source
        body = {**doc, **({"copied": [4, 3, 2, 1]} if extra else {})}
        for port in (OURS[1], STOCK[1]):
            settings = {"knn": True, "number_of_replicas": 0, **({"lucene_rust.engine": True} if port == OURS[1] else {})}
            ck.req(port, "PUT", f"/{index}", {"settings": {"index": settings}, "mappings": mappings})
            setting = ck.req(port, "GET", f"/{index}/_settings/index.knn.derived_source.enabled")
            got[port] = {"setting": setting[index]["settings"]["index"]["knn"]["derived_source"]["enabled"]}
            ck.req(port, "PUT", f"/{index}/_doc/1", body)
            got[port].update(sources(port, index))
        n += 3
        if typed(got[OURS[1]]) != typed(got[STOCK[1]]):
            failures.append(f"{index} _source: ours {got[OURS[1]]}, stock {got[STOCK[1]]}")
    # Masking stores less: the same documents with derived source off take more room.
    rng = random.Random(5)
    lines = []
    for i in range(400):
        lines.append('{"index":{"_id":"%d"}}' % i)
        lines.append('{"v":[%s]}' % ",".join(f"{rng.random():.6f}" for _ in range(64)))
    size = {}
    for index, derived in (("masked", True), ("whole", False)):
        ck.req(OURS[1], "PUT", f"/{index}", {"settings": {"index": {"knn": True, "number_of_replicas": 0,
                                                                    "lucene_rust.engine": True,
                                                                    "knn.derived_source.enabled": derived}},
                                             "mappings": {"properties": {"v": {"type": "knn_vector", "dimension": 64}}}})
        ck.req(OURS[1], "POST", f"/{index}/_bulk?refresh=true", "\n".join(lines) + "\n", ndjson=True)
        ck.req(OURS[1], "POST", f"/{index}/_forcemerge?max_num_segments=1")
        size[index] = ck.req(OURS[1], "GET", f"/{index}/_stats/store")["_all"]["primaries"]["store"]["size_in_bytes"]
        hit = ck.req(OURS[1], "GET", f"/{index}/_doc/7")["_source"]["v"]
        if len(hit) != 64:
            failures.append(f"{index}: _source vector {hit}")
    if size["masked"] >= size["whole"]:
        failures.append(f"masked index stores {size['masked']} bytes, not less than {size['whole']} unmasked")
    print(f"store: masked {size['masked']} bytes, whole {size['whole']}")
    return n


def compare_explain(failures):
    n = 0
    for body in ({"k": 1}, {"k": 3}, {"k": 3, "filter": {"term": {"tag": "even"}}}):
        query = {"explain": True, "size": 3, "query": {"knn": {"emb": {"vector": [0.5, 0.5, 0.5, 0.5], **body}}}}
        o, s = (ck.req(p, "POST", "/vec-a/_search", query)["hits"]["hits"] for p in (OURS[1], STOCK[1]))
        eo = [h.get("_explanation") for h in o]
        es = [h.get("_explanation") for h in s]
        if eo != es:
            failures.append(f"explain {body}: ours {eo}, stock {es}")
        n += 1
    return n


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
        calls = [("GET", "/_plugins/_knn/stats/bogus"), ("GET", "/_plugins/_knn/stats/hit_coutn"),
                 ("GET", "/_plugins/_knn/stats/_all,hit_count"),
                 ("GET", "/_plugins/_knn/warmup/vec-a"), ("GET", "/_plugins/_knn/warmup/vec-a,vec-b"),
                 ("GET", "/_plugins/_knn/warmup/vec-*"), ("GET", "/_plugins/_knn/warmup/plain"),
                 ("GET", "/_plugins/_knn/warmup/missing"), ("POST", "/_plugins/_knn/clear_cache/vec-a"),
                 ("POST", "/_plugins/_knn/clear_cache/plain"), ("POST", "/_plugins/_knn/clear_cache/missing")]
        for method, path in calls:
            same_answer(method, path, failures)
        mappings = compare_mappings(failures)
        explains = compare_explain(failures)
        reads = compare_sources(failures)
        print(f"{len(stats)} stats calls, {len(calls)} warmup/clear_cache/error calls, {mappings} mappings, "
              f"{explains} explains, {reads} _source reads compared")
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
