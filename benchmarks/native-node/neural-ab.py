#!/usr/bin/env python3
"""Vector, neural, sparse, semantic, hybrid, MMR and rerank search: stock against the plugin, no Docker.

Every round starts a fresh node of each side as a plain process (stock tarball with its bundled k-NN,
neural-search and ml-commons; the plugin variant of opensearch-plugin/e2e/nodectl.py), pinned to
NODE_CPUS, deploys the fixtures/ml models (tiny-bert text embedding, tiny-sparse encoder, tiny
cross-encoder, served over local HTTP), ingests DOCS documents through the pipelines, force-merges to
one segment and times every shape on one keep-alive socket (opensearch-plugin/e2e/benchhttp.py) from the
driver core. Sides alternate within a round; the report is the best median per shape.

  python3 benchmarks/native-node/neural-ab.py run  OUT.json [--docs 4000] [--queries 400] [--rounds 3]
  python3 benchmarks/native-node/neural-ab.py report OUT.json

The models are the repository's tiny test models (32-dim BERT, so inference is a small fraction of a
request and the engine's share shows); the ONNX runtime is the same one on both sides.
KEEP_JAVA_TOOL_OPTIONS=1 hands the shell's proxy/truststore -D flags to the nodes: stock ml-commons downloads its ONNX
runtime (DJL) at the first model deploy, and behind a TLS-intercepting proxy it needs them.
Env: NODE_CPUS (0-2), HEAP (1g), SIDES (stock,plugin), PLUGIN_LIB (a library to measure), SHAPES (a|b: substrings of
the shape names). Pin the driver yourself: taskset -c 3 python3 -I ...
"""
import argparse, hashlib, importlib.util, json, math, os, random, statistics, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "opensearch-plugin", "e2e"))
import benchhttp as bh  # noqa: E402
import nodectl  # noqa: E402


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ml = load("check_ml", "check-ml-plugin.py")
golden = ml.golden
PORT = int(os.environ.get("PORT", 9250))
CPUS = os.environ.get("NODE_CPUS", "0-2")
HEAP = os.environ.get("HEAP", "1g")
WORDS = ("search documents vectors sparse dense hybrid model query index shard segment token weight "
         "neural retrieval ranking score cluster embedding text field engine opensearch lucene "
         "vector latency cache memory thread merge commit shard replica").split()


def text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


class Api:
    def __init__(self, base):
        self.c = bh.Client(base)

    def ok(self, method, path, body=None):
        return self.c.json(method, path, body)

    def task(self, tid):
        for _ in range(600):
            t = self.ok("GET", f"/_plugins/_ml/tasks/{tid}")
            if t.get("state") in ("COMPLETED", "FAILED"):
                assert t["state"] == "COMPLETED", t
                return t
            time.sleep(0.5)
        raise RuntimeError("task timeout")


def deploy(n, name, zname, z, function=None, config=None):
    body = {"name": f"{name}-{ml.RUN}", "version": "1.0.0", "model_format": "ONNX",
            "model_content_hash_value": hashlib.sha256(z).hexdigest(), "model_content_size_in_bytes": len(z),
            "url": f"http://localhost:{ml.SERVE}/{zname}.zip"}
    if function:
        body["function_name"] = function
    if config:
        body["model_config"] = config
    mid = n.task(n.ok("POST", "/_plugins/_ml/models/_register", body)["task_id"])["model_id"]
    n.task(n.ok("POST", f"/_plugins/_ml/models/{mid}/_deploy")["task_id"])
    return mid


def bert_config(extra=None):
    c = {"model_type": "bert", "embedding_dimension": 32, "framework_type": "sentence_transformers",
         "pooling_mode": "mean", "normalize_result": True}
    if extra:
        c["additional_config"] = extra
    return c


def bulk(n, index, docs):
    lines = []
    for d in docs:
        lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
        lines.append(json.dumps({k: v for k, v in d.items() if k != "_id"}))
    r = n.c.json("POST", "/_bulk?refresh=true", "\n".join(lines) + "\n", ndjson=True)
    assert not r["errors"], json.dumps(r)[:1000]


def one_side(kind, zips, docs, queries, lib):
    name = f"nab-{kind}"
    nodectl.run(name, "plugin" if kind == "plugin" else "stock", PORT, transport=PORT + 100, heap=HEAP, cpus=CPUS,
                lib=lib if kind == "plugin" else None, java_opts=ml.proxy_opts(), wait=False)
    n = Api(f"http://localhost:{PORT}")
    for _ in range(150):
        try:
            if n.ok("GET", "/_cluster/health")["status"] in ("green", "yellow"):
                break
        except Exception:
            time.sleep(2)
    try:
        return bench(n, zips, docs, queries)
    finally:
        nodectl.stop(name)
        nodectl.rm(name)


def bench(n, zips, docs, queries):
    rng = random.Random(7)
    n.ok("PUT", "/_cluster/settings", ml.SETTINGS)
    dense = deploy(n, "tiny-bert", "tiny-bert", zips["tiny-bert"], None, bert_config())
    sem = deploy(n, "tiny-bert-sem", "tiny-bert", zips["tiny-bert"], None, bert_config({"space_type": "l2"}))
    sparse = deploy(n, "tiny-sparse", golden.SPARSE, zips[golden.SPARSE], "SPARSE_ENCODING")
    cross = deploy(n, "tiny-cross", golden.CROSS, zips[golden.CROSS], "TEXT_SIMILARITY",
                   {"model_type": "bert", "embedding_dimension": 1, "framework_type": "huggingface_transformers"})
    n.ok("PUT", "/_ingest/pipeline/bench-p", {"processors": [
        {"text_embedding": {"model_id": dense, "field_map": {"text": "emb"}}},
        {"sparse_encoding": {"model_id": sparse, "field_map": {"text": "feat"}}}]})
    n.ok("PUT", "/_search/pipeline/bench-rr", {"response_processors": [{"rerank": {
        "ml_opensearch": {"model_id": cross}, "context": {"document_fields": ["text"]}}}]})
    n.ok("PUT", "/_search/pipeline/bench-hy", {"phase_results_processors": [{"normalization-processor": {
        "normalization": {"technique": "min_max"}, "combination": {"technique": "arithmetic_mean"}}}]})
    n.ok("PUT", "/_search/pipeline/bench-rrf", {"phase_results_processors": [{"score-ranker-processor": {
        "combination": {"technique": "rrf", "rank_constant": 60}}}]})
    n.ok("PUT", "/bench", {"settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0,
                                        "default_pipeline": "bench-p", "refresh_interval": -1},
                           "mappings": {"properties": {"text": {"type": "text"}, "feat": {"type": "rank_features"},
                                                       "emb": {"type": "knn_vector", "dimension": 32}}}})
    all_docs = [{"_id": str(i), "text": text(rng, rng.randint(8, 40))} for i in range(docs)]
    out = {}
    t = time.perf_counter()
    for i in range(0, docs, 100):
        bulk(n, "bench", all_docs[i:i + 100])
    out["ingest embed+sparse (docs/s)"] = docs / (time.perf_counter() - t)
    n.ok("POST", "/bench/_refresh")
    n.ok("POST", "/bench/_forcemerge?max_num_segments=1")
    n.ok("PUT", "/bench-sem", {"settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0,
                                            "refresh_interval": -1},
                               "mappings": {"properties": {"text": {"type": "semantic", "model_id": sem}}}})
    t = time.perf_counter()
    for i in range(0, docs, 100):
        bulk(n, "bench-sem", all_docs[i:i + 100])
    out["ingest semantic field (docs/s)"] = docs / (time.perf_counter() - t)
    n.ok("POST", "/bench-sem/_refresh")
    n.ok("POST", "/bench-sem/_forcemerge?max_num_segments=1")
    qs = [text(rng, rng.randint(2, 6)) for _ in range(queries + 50)]
    vecs = [n.ok("POST", "/_plugins/_ml/_predict/text_embedding/" + dense,
                 {"text_docs": [q], "target_response": ["sentence_embedding"]})
            ["inference_results"][0]["output"][0]["data"] for q in qs]
    # a larger vector index (random 128-d clustered) comes from MODE=knn of ab.sh; here the 32-d embeddings

    def S(index, body, pipeline=None):
        path = f"/{index}/_search" + (f"?search_pipeline={pipeline}" if pipeline else "")
        return lambda i: n.ok("POST", path, body(i))

    neural = lambda i: {"query": {"neural": {"emb": {"query_text": qs[i], "model_id": dense, "k": 10}}}}
    knn = lambda i: {"query": {"knn": {"emb": {"vector": vecs[i], "k": 10}}}, "size": 10, "_source": False}
    match = lambda i: {"query": {"match": {"text": qs[i]}}, "size": 10}
    hyb = lambda i: {"query": {"hybrid": {"queries": [{"match": {"text": qs[i]}},
                                                      {"knn": {"emb": {"vector": vecs[i], "k": 50}}}]}}, "size": 10}
    hyn = lambda i: {"query": {"hybrid": {"queries": [{"match": {"text": qs[i]}},
                                                      {"neural": {"emb": {"query_text": qs[i], "model_id": dense, "k": 50}}}]}}, "size": 10}
    shapes = {
        "knn k=10 (precomputed vector)": S("bench", knn),
        "knn k=10 + term filter": S("bench", lambda i: {"query": {"knn": {"emb": {"vector": vecs[i], "k": 10, "filter": {
            "match": {"text": WORDS[i % len(WORDS)]}}}}}, "size": 10, "_source": False}),
        "knn k=100 size 100": S("bench", lambda i: {"query": {"knn": {"emb": {"vector": vecs[i], "k": 100}}}, "size": 100}),
        "neural (embed query + knn)": S("bench", neural),
        "neural_sparse by model": S("bench", lambda i: {"query": {"neural_sparse": {"feat": {"query_text": qs[i], "model_id": sparse}}}}),
        "neural_sparse by analyzer": S("bench", lambda i: {"query": {"neural_sparse": {"feat": {"query_text": qs[i]}}}}),
        "semantic field neural": S("bench-sem", lambda i: {"query": {"neural": {"text": {"query_text": qs[i], "k": 10}}}}),
        "mmr (50 candidates, diversity .5)": S("bench", lambda i: {**knn(i), "ext": {"mmr": {"candidates": 50, "diversity": 0.5}}}),
        "match (baseline)": S("bench", match),
        "hybrid match + knn (min_max)": S("bench", hyb, "bench-hy"),
        "hybrid match + neural (min_max)": S("bench", hyn, "bench-hy"),
        "hybrid match + knn (rrf)": S("bench", hyb, "bench-rrf"),
        "rerank top10 (cross-encoder)": S("bench", lambda i: {**match(i), "ext": {"rerank": {"query_context": {"query_text": qs[i]}}}}, "bench-rr"),
    }
    only = [x for x in os.environ.get("SHAPES", "").split("|") if x]
    for label, fn in shapes.items():
        if only and not any(x in label for x in only):
            continue
        k = [0]

        def call():
            k[0] += 1
            fn(k[0] % queries)
        # the mmr extension needs the system-generated search processors; they cost every search, so only here
        sysf = {"persistent": {"cluster.search.enabled_system_generated_factories": ["*"] if label.startswith("mmr") else None}}
        n.ok("PUT", "/_cluster/settings", sysf)
        ms = bh.measure(call, rounds=queries, warmup=40, warmup_s=3.0)
        n.ok("PUT", "/_cluster/settings", {"persistent": {"cluster.search.enabled_system_generated_factories": None}})
        out[label] = bh.summarize(ms)
    return out


def run(a):
    zips = {name: golden.model_zip(name) for name in ("tiny-bert", golden.SPARSE, golden.CROSS)}
    ml.serve(zips)
    sides = os.environ.get("SIDES", "stock,plugin").split(",")
    res = {s: [] for s in sides}
    for r in range(a.rounds):
        for s in sides:
            print(f"round {r + 1} {s}", flush=True)
            res[s].append(one_side(s, zips, a.docs, a.queries, os.environ.get("PLUGIN_LIB")))
        json.dump(res, open(a.out, "w"), indent=1)
    report(a.out)


def best(rounds, label):
    v = [r[label] for r in rounds if label in r]
    if not v:
        return None
    return min(x["median_ms"] for x in v) if isinstance(v[0], dict) else max(v)


def report(path):
    res = json.load(open(path))
    labels = list(res["stock"][0])
    print(f"{'shape':40} {'stock p50':>10} {'plugin p50':>11} {'speedup':>8}")
    ratios = []
    for l in labels:
        s, p = best(res["stock"], l), best(res["plugin"], l)
        if isinstance(res["stock"][0][l], dict):
            print(f"{l:40} {s:10.2f} {p:11.2f} {s / p:7.2f}x")
            ratios.append(s / p)
        else:
            print(f"{l:40} {s:10.1f} {p:11.1f} {p / s:7.2f}x (docs/s, higher is better)")
    print("geomean search speedup", round(math.exp(statistics.mean(math.log(r) for r in ratios)), 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("out")
    ap.add_argument("--docs", type=int, default=4000)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a.out)
