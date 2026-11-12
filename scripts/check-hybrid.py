#!/usr/bin/env python3
"""The plugin's hybrid search against stock neural-search's.

One plugin node and one stock node (which bundles neural-search) take the same
index: text, a keyword, a number and a knn_vector, two shards, a few segments
each. Both define the same search pipelines and answer the same hybrid
searches, and the answers are compared:

  pipelines   normalization-processor with min_max (plain, weights, lower
              bounds), l2 and z_score, combined by arithmetic, geometric and
              harmonic mean; score-ranker-processor (rrf, rank_constant)
  queries     match + knn, three lexical sub-queries, a filter, from and
              pagination_depth, size 0, a sort, collapse, aggregations,
              min_score, post_filter, explain with hybrid_score_explanation
  hits        the same ids and scores in the same order (equal scores may come
              in either order), the same total and max_score
  errors      the same status, type and reason: hybrid inside bool, too many
              sub-queries, a bad technique, weights that do not add up

  scripts/check-hybrid.py [--keep]

Needs Docker and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import glob, importlib.util, json, os, random, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("check_knn", os.path.join(ROOT, "scripts", "check-knn-plugin.py"))
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

OURS = ("hybrid-ours", 19226)
STOCK = ("hybrid-stock", 19227)
DIM = 8
DOCS = 600
WORDS = [f"w{i}" for i in range(120)]


def docs():
    rng = random.Random(11)
    out = []
    for i in range(DOCS):
        n = rng.randint(3, 14)
        out.append({
            "t": " ".join(rng.choices(WORDS, [1 / (j + 1) for j in range(len(WORDS))], k=n)),
            "tag": f"g{rng.randint(0, 9)}",
            "n": rng.randint(0, 1000),
            "v": [round(rng.uniform(-1, 1), 4) for _ in range(DIM)],
        })
    return out


def setup(port, rust, index="h"):
    settings = {"index": {"knn": True, "number_of_shards": 2, "number_of_replicas": 0, "refresh_interval": -1}}
    if rust:
        settings["index"]["engine.accelerated"] = True
    ck.req(port, "PUT", f"/{index}", {"settings": settings, "mappings": {"properties": {
        "t": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "integer"},
        "v": {"type": "knn_vector", "dimension": DIM, "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"}}}}})
    all_docs = docs()
    for start in range(0, DOCS, 200):
        lines = []
        for i in range(start, start + 200):
            lines.append(json.dumps({"index": {"_id": str(i), "routing": str(i % 7)}}))
            lines.append(json.dumps(all_docs[i]))
        ck.req(port, "POST", f"/{index}/_bulk", "\n".join(lines) + "\n", ndjson=True)
        ck.req(port, "POST", f"/{index}/_refresh")
    for name, body in PIPELINES.items():
        ck.req(port, "PUT", f"/_search/pipeline/{name}", body)


BIG = 15000


def setup_big(port):
    """Segments of 5000 documents: past the hybrid bulk scorer's 4096-document window, where
    stock stops counting documents no sub-query scores competitively."""
    rng = random.Random(11)
    ck.req(port, "PUT", "/hb", {"settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0,
                                                       "refresh_interval": -1}},
                                "mappings": {"properties": {"t": {"type": "text"}, "v": {"type": "knn_vector", "dimension": 4,
                                             "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"}}}}})
    for start in range(0, BIG, 5000):
        lines = []
        for i in range(start, start + 5000):
            lines.append(json.dumps({"index": {"_id": str(i)}}))
            lines.append(json.dumps({"t": " ".join(rng.choices(WORDS[:40], k=6)),
                                     "v": [round(rng.uniform(-1, 1), 4) for _ in range(4)]}))
        ck.req(port, "POST", "/hb/_bulk", "\n".join(lines) + "\n", ndjson=True)
        ck.req(port, "POST", "/hb/_refresh")


def big_searches():
    m = {"match": {"t": "w3 w17"}}
    k = {"knn": {"v": {"vector": [0.1, 0.2, -0.3, 0.4], "k": 10}}}
    return [(p, b) for p in ("minmax", "rrf") for b in (
        {"query": hybrid(m, k)},
        {"query": hybrid(m, k, pagination_depth=40), "from": 20, "size": 10},
        {"query": hybrid(m, {"match": {"t": "w5"}}, k), "size": 25},
        {"query": hybrid(m, k), "size": 0, "track_total_hits": True},
    )]


def norm(technique, combination="arithmetic_mean", **extra):
    n = {"technique": technique}
    c = {"technique": combination}
    if "weights" in extra:
        c["parameters"] = {"weights": extra.pop("weights")}
    if extra:
        n["parameters"] = extra
    return {"phase_results_processors": [{"normalization-processor": {"normalization": n, "combination": c}}]}


PIPELINES = {
    "minmax": norm("min_max"),
    "minmax_w": norm("min_max", weights=[0.3, 0.7]),
    "minmax_lb": norm("min_max", lower_bounds=[{"mode": "clip", "min_score": 0.1}, {"mode": "apply", "min_score": 0.2}]),
    "l2_geo": norm("l2", "geometric_mean"),
    "z_arith": norm("z_score"),
    "minmax_harm": norm("min_max", "harmonic_mean", weights=[0.6, 0.4]),
    "rrf": {"phase_results_processors": [{"score-ranker-processor": {"combination": {"technique": "rrf"}}}]},
    "rrf_k": {"phase_results_processors": [{"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": 10}}}]},
    "explain": {**norm("min_max"), "response_processors": [{"hybrid_score_explanation": {}}]},
}


def target(seed):
    rng = random.Random(seed)
    return [round(rng.uniform(-1, 1), 4) for _ in range(DIM)]


def hybrid(*queries, **extra):
    return {"hybrid": {"queries": list(queries), **extra}}


MATCH = {"match": {"t": "w3 w17 w40"}}
KNN = {"knn": {"v": {"vector": target(1), "k": 20}}}


def whole_searches():
    """Searches whose sub-queries keep every match (pagination_depth past the index), so equal
    scores at the edge of a sub-query's top hits cannot pick other documents."""
    deep = {"pagination_depth": DOCS}
    two = [
        {"query": hybrid(MATCH, KNN, **deep), "size": 20},
        {"query": hybrid({"match": {"t": "w5"}}, {"knn": {"v": {"vector": target(2), "k": 50}}}, **deep), "size": 30},
        {"query": hybrid(MATCH, KNN, filter={"range": {"n": {"gte": 200}}}, **deep), "size": 20},
        {"query": hybrid(MATCH, KNN, **deep), "from": 10, "size": 10},
    ]
    out = [(p, b) for p in ("minmax", "l2_geo", "z_arith", "rrf") for b in two]
    three = hybrid({"match": {"t": "w2"}}, {"term": {"tag": "g4"}}, {"range": {"n": {"gte": 100, "lte": 400}}}, **deep)
    return out + [(p, {"query": three, "size": 15}) for p in ("minmax", "rrf")]


def searches():
    """(pipeline, body) pairs, each two-sub-query search run under every two-query pipeline."""
    two = [
        {"query": hybrid(MATCH, KNN)},
        {"query": hybrid({"match": {"t": "w5"}}, {"knn": {"v": {"vector": target(2), "k": 50}}}), "size": 30},
        {"query": hybrid(MATCH, KNN, filter={"range": {"n": {"gte": 200}}})},
        {"query": hybrid(MATCH, KNN, pagination_depth=30), "from": 10, "size": 10},
        {"query": hybrid(MATCH, KNN), "size": 0, "track_total_hits": True},
        {"query": hybrid(MATCH, KNN), "aggs": {"tags": {"terms": {"field": "tag"}}, "avg_n": {"avg": {"field": "n"}}}},
        {"query": hybrid(MATCH, KNN), "post_filter": {"term": {"tag": "g1"}}},
        {"query": hybrid(MATCH, KNN), "min_score": 0.4},
    ]
    out = []
    for p in ("minmax", "minmax_w", "minmax_lb", "l2_geo", "z_arith", "minmax_harm", "rrf", "rrf_k"):
        out += [(p, b) for b in two]
    three = hybrid({"match": {"t": "w2"}}, {"term": {"tag": "g4"}}, {"range": {"n": {"gte": 100, "lte": 400}}})
    out += [(p, {"query": three, "size": 15}) for p in ("minmax", "l2_geo", "z_arith", "rrf")]
    out += [("minmax", {"query": hybrid(MATCH, KNN), "sort": [{"n": "desc"}], "size": 15}),
            ("minmax", {"query": hybrid(MATCH, KNN), "sort": [{"n": "asc"}, {"_id": "asc"}], "size": 15}),
            ("minmax", {"query": hybrid(MATCH, KNN), "collapse": {"field": "tag"}, "size": 10}),
            ("rrf", {"query": hybrid(MATCH, KNN), "collapse": {"field": "tag"}, "size": 10})]
    return out


def answer(port, pipeline, body, explain=False, index="h"):
    path = f"/{index}/_search?search_pipeline={pipeline}" + ("&explain=true" if explain else "")
    r = ck.req(port, "POST", path, body, ok_errors=True)
    if "error" in r and "status" in r:
        e = (r["error"] or {}).get("error", {})
        return {"error": (r["status"], e.get("type"), e.get("reason"))}
    hits = r["hits"]
    out = {"total": hits.get("total"), "max_score": hits.get("max_score"),
           "hits": [(h["_id"], h.get("_score"), h.get("sort"), (h.get("fields") or {}).get("tag")) for h in hits["hits"]]}
    if "aggregations" in r:
        out["aggregations"] = r["aggregations"]
    if explain:
        out["explanations"] = {h["_id"]: h.get("_explanation") for h in hits["hits"]}
    return out


def close(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def same_hits(o, s):
    """The same hits in the same order, where hits of one score may come in either order."""
    if len(o) != len(s):
        return False
    groups_o, groups_s = {}, {}
    for (oid, osc, osort, otag), (sid, ssc, ssort, stag) in zip(o, s):
        if not close(osc, ssc) or not close(osort, ssort):
            return False
        key = (round(osc, 5) if osc is not None else None, json.dumps(osort))
        groups_o.setdefault(key, set()).add(oid)
        groups_s.setdefault(key, set()).add(sid)
    return groups_o == groups_s


def compare(pipeline, body, failures, explain=False, ours="h", stock="h"):
    o, s = answer(OURS[1], pipeline, body, explain, ours), answer(STOCK[1], pipeline, body, explain, stock)
    label = f"{pipeline} {json.dumps(body)[:300]}"
    if "error" in o or "error" in s:
        if o.get("error") != s.get("error"):
            failures.append(f"{label}: ours {o.get('error', 'ok')}, stock {s.get('error', 'ok')}")
        return
    if o["total"] != s["total"] or not close(o["max_score"], s["max_score"]):
        failures.append(f"{label}: total/max ours {o['total']} {o['max_score']}, stock {s['total']} {s['max_score']}")
    if not same_hits(o["hits"], s["hits"]):
        failures.append(f"{label}: hits\n  ours  {o['hits'][:12]}\n  stock {s['hits'][:12]}")
    if not close(o.get("aggregations"), s.get("aggregations")):
        failures.append(f"{label}: aggregations ours {o.get('aggregations')}, stock {s.get('aggregations')}")
    if explain and not close(o["explanations"], s["explanations"]):
        k = next(iter(o["explanations"]), None)
        failures.append(f"{label}: explanation of {k}\n  ours  {json.dumps(o['explanations'].get(k))[:1500]}\n"
                        f"  stock {json.dumps(s['explanations'].get(k))[:1500]}")


ERRORS = [
    ("minmax", {"query": {"bool": {"should": [hybrid(MATCH, KNN)]}}}),
    ("minmax", {"query": hybrid(*[{"term": {"tag": f"g{i}"}} for i in range(6)])}),
    ("minmax", {"query": {"hybrid": {"queries": []}}}),
    ("minmax", {"query": {"hybrid": {"queries": [MATCH], "boost": 2}}}),
    ("minmax", {"query": hybrid(MATCH, KNN, pagination_depth=0)}),
    ("minmax", {"query": hybrid(MATCH, KNN), "from": 5, "size": 10}),
    ("minmax", {"query": hybrid(MATCH, KNN), "rescore": {"query": {"rescore_query": {"match": {"t": "w1"}}}}}),
]
BAD_PIPELINES = [
    norm("bogus"),
    norm("min_max", "bogus_mean"),
    norm("min_max", weights=[0.3, 0.3]),
    norm("min_max", weights=[1.5, -0.5]),
    norm("min_max", lower_bounds=[{"mode": "clip", "min_score": 2.0}, {"mode": "apply"}]),
    {"phase_results_processors": [{"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": 0}}}]},
    {"phase_results_processors": [{"score-ranker-processor": {"combination": {"technique": "min_max"}}}]},
]


def pipeline_error(port, body):
    r = ck.req(port, "PUT", "/_search/pipeline/bad", body, ok_errors=True)
    if "status" not in r:
        return "ok"
    e = (r["error"] or {}).get("error", {})
    return (r["status"], e.get("type"), e.get("reason"))


def stats_answer(port, path):
    """A stats call's status and body, its node ids, timestamps and memory left out."""
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=60) as r:
            status, body = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read().decode()
    try:
        body = json.loads(body)
    except ValueError:
        return status, body
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()
                    if k not in ("nodes", "minutes_since_last_event") and not k.startswith("memory")}
        return v
    return status, clean(body)


STATS_CALLS = [
    "/_plugins/_neural/stats",
    "/_plugins/_neural/stats?flat_stat_paths=true",
    "/_plugins/_neural/stats?include_metadata=true&flat_stat_paths=true",
    "/_plugins/_neural/stats/hybrid_query_requests,normalization_processor_executions",
    "/_plugins/_neural/stats?include_individual_nodes=false&include_info=false",
    "/_plugins/_neural/stats/bogus_stat",
    "/_plugins/_neural/stats?include_all_nodes=false",
]


def compare_stats(failures):
    for path in STATS_CALLS:
        o, s = stats_answer(OURS[1], path), stats_answer(STOCK[1], path)
        if o != s:
            failures.append(f"stats {path}:\n  ours  {json.dumps(o)[:1500]}\n  stock {json.dumps(s)[:1500]}")


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
    disabled = []
    try:
        ck.wait(OURS[1])
        ck.wait(STOCK[1])
        # Java-engine indexes on both nodes keep documents in the order they came, so equal
        # scores at the edge of a sub-query's top hits break the same way.
        for port in (OURS[1], STOCK[1]):
            disabled.append(stats_answer(port, "/_plugins/_neural/stats"))
            ck.req(port, "PUT", "/_cluster/settings", {"persistent": {"plugins.neural_search.stats_enabled": True}})
        setup(OURS[1], False)
        setup(STOCK[1], False)
        runs = searches()
        before = ck.req(OURS[1], "GET", "/_plugins/lucene_rust/stats")
        for pipeline, body in runs:
            compare(pipeline, body, failures)
        compare("explain", {"query": hybrid(MATCH, KNN), "size": 5}, failures, explain=True)
        for pipeline, body in ERRORS:
            compare(pipeline, body, failures)
        setup_big(OURS[1])
        setup_big(STOCK[1])
        big = big_searches()
        for pipeline, body in big:
            compare(pipeline, body, failures, ours="hb", stock="hb")
        # Before the plugin-only searches below, which the counters would see.
        if disabled[0] != disabled[1]:
            failures.append(f"stats while disabled: ours {disabled[0]}, stock {disabled[1]}")
        compare_stats(failures)
        # Rust-written segments: the plugin's Rust-engine index against stock's Java one.
        setup(OURS[1], True, "hr")
        whole = whole_searches()
        for pipeline, body in whole:
            compare(pipeline, body, failures, ours="hr")
        for body in BAD_PIPELINES:
            o, s = pipeline_error(OURS[1], body), pipeline_error(STOCK[1], body)
            if o != s:
                failures.append(f"pipeline {json.dumps(body)}: ours {o}, stock {s}")
        after = ck.req(OURS[1], "GET", "/_plugins/lucene_rust/stats")
        ran = after["accelerated_queries"] - before["accelerated_queries"]
        fallbacks = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
                     if v != before.get("fallbacks", {}).get(k, 0)}
        print(f"plugin: {ran} shard queries in Rust, fallbacks {fallbacks}")
        # What the native hybrid shard phase leaves to Java: the hybrid collectors' other modes.
        java = {"hybrid_aggregations", "hybrid_collapse", "hybrid_no_hits", "hybrid_post_filter", "hybrid_rescore",
                "hybrid_sort"}
        if ran == 0 or set(fallbacks) - java:
            failures.append(f"hybrid searches fell back to Java: {fallbacks}")
        print(f"{len(runs) + 1} hybrid searches, {len(whole)} on a Rust-engine index, {len(big)} over 5000-document segments, {len(ERRORS)} search errors, {len(BAD_PIPELINES)} pipeline errors, {len(STATS_CALLS)} stats calls compared")
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
