#!/usr/bin/env python3
"""neural-search's model-backed half through the lucene-rust plugin against stock.

Starts a node of the pinned OpenSearch image with the lucene-rust plugin (the
bundled plugins removed) and, unless --stock names a running one, a stock node
with k-NN, neural-search and ml-commons. Both deploy the same fixtures/ml
models, which this script serves over HTTP, run the same calls and the answers
are compared:

  analyze    the bert-uncased and mbert-uncased analyzers and tokenizers and a
             custom hf_model_tokenizer analyzer: tokens, offsets, payloads
  ingest     text_embedding, sparse_encoding and text_chunking pipelines,
             simulated and indexed: the documents they write
  neural     the neural query on a knn_vector field filled by text_embedding:
             k, min_score, max_distance, filters, inside bool and hybrid, the
             neural_query_enricher defaults, and its errors; on a Java-engine
             and (plugin only) a Rust-engine index, which must answer natively
  rerank     the rerank response processor (ml_opensearch with a cross-encoder,
             by_field) and its pipeline errors
  two_phase  neural_sparse_two_phase_processor: hits and pipeline errors
  warmup     _plugins/_neural/warmup and clear_cache on sparse and other indexes
  settings   every knn.* and neural_search setting's default, and bad values
  semantic   semantic fields (dense and sparse models, chunking, the field's
             options, model_selection, put mapping): the mappings the
             transformer writes, the documents the system processor embeds,
             neural and match queries on them, and the mapping errors
  sparse     neural_sparse by query_text with a model, by query_text with the
             default bert-uncased analyzer, by query_tokens, with the enricher,
             and its errors

  scripts/check-neural-plugin.py [--stock http://localhost:9200] [--keep]
                                 [--only analyze,ingest,neural,sparse,rerank,two_phase,warmup,settings,semantic]

Needs Docker and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import argparse, functools, glob, hashlib, importlib.util, json, os, re, shutil, subprocess, sys, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "3.9.0"
OURS = ("neural-check-rust", 19241)
STOCK = ("neural-check-stock", 19242)
SCORE_TOLERANCE = 1e-4


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ml = load("check_ml", "check-ml-plugin.py")
golden = ml.golden

TEXTS = [
    "hello world",
    "Neural search finds documents by meaning",
    "café über straße naïve",
    "Ünïcödé 😀 emoji and 中文字符 mixed with ASCII",
    "opensearch indexes documents into shards",
    "how are documents searched",
    "the model encodes text into vectors",
    "sparse retrieval weighs tokens",
    "dense retrieval compares embeddings",
    "hybrid search combines both",
    "unknownwordzzz qqq",
    "documents documents documents",
]
# Distinct texts, so that no two documents tie on score.
DOCS = [{"_id": str(i), "text": t + s, "tag": "even" if i % 2 == 0 else "odd", "n": i}
        for i, (s, t) in enumerate((s, t) for s in ("", " again", " once more") for t in TEXTS)]
LONG = " ".join(["opensearch indexes documents into shards"] * 150)


def same_answer(a, b):
    """Equal, floats within the score tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_answer(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_answer(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= SCORE_TOLERANCE * max(1.0, abs(b))
    return a == b


def answer(r):
    """A call's status with the error's type and reason, for comparing failures."""
    s, o = r
    if s != 200 and isinstance(o, dict) and "error" in o:
        e = o["error"]
        root = (e.get("root_cause") or [{}])[0]
        return s, e.get("type"), pkg(e.get("reason")), root.get("type"), pkg(root.get("reason"))
    return s, o


def pkg(reason):
    # The ported classes live under another package, which shows in Java exception messages.
    return reason.replace("org.lucenerust.opensearch.neural.", "org.opensearch.neuralsearch.") if isinstance(reason, str) else reason


def hits(r):
    s, o = r
    if s != 200:
        return answer(r)
    return [(h["_id"], h["_score"]) for h in o["hits"]["hits"]]


def compare(c, what, a, b):
    c.true(what, same_answer(a, b), f"ours {json.dumps(a, ensure_ascii=False)[:600]} stock {json.dumps(b, ensure_ascii=False)[:600]}")


def both(ours, stock, fn):
    return fn(ours), fn(stock)


def bulk(n, index, docs, pipeline=None):
    lines = []
    for d in docs:
        lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
        lines.append(json.dumps({k: v for k, v in d.items() if k != "_id"}))
    path = "/_bulk?refresh=true" + (f"&pipeline={pipeline}" if pipeline else "")
    s, r = bulk_call(n, path, "\n".join(lines) + "\n")
    assert s == 200 and not r["errors"], json.dumps(r)[:2000]


def bulk_call(n, path, body):
    req = urllib.request.Request(n.url + path, data=body.encode(), method="POST",
                                 headers={"Content-Type": "application/x-ndjson"})
    try:
        with urllib.request.urlopen(req, timeout=300) as f:
            return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def deploy(n, label, function, z, config=None):
    mid, _ = ml.onnx_model(n, label, function, z, config)
    return mid


def text_model(n, z):
    r = ml.register(n, "tiny-bert", z, "bert", "mean", True)
    mid = n.task(r["task_id"])["model_id"]
    n.task(n.ok("POST", f"/_plugins/_ml/models/{mid}/_deploy")["task_id"])
    return mid


# ---------------------------------------------------------------- analyze

def analyze(c, ours, stock):
    texts = TEXTS + ["", "   ", LONG, "Hello, WORLD! It's 3.14 o'clock — naïve café", "##s pieces and [CLS] [SEP] [MASK]"]
    for analyzer in ("bert-uncased", "mbert-uncased"):
        for i, t in enumerate(texts):
            for kind in ("analyzer", "tokenizer"):
                body = {kind: analyzer, "text": t}
                compare(c, f"analyze {kind} {analyzer} text {i}", *both(ours, stock, lambda n: answer(n.call("POST", "/_analyze", body))))
            body = {"analyzer": analyzer, "text": t, "explain": True, "attributes": ["payload", "type"]}
            compare(c, f"analyze explain {analyzer} text {i}", *both(ours, stock, lambda n: answer(n.call("POST", "/_analyze", body))))
    index = {"settings": {"analysis": {"analyzer": {"hf": {"type": "hf_model_tokenizer"}, "custom_hf": {"type": "custom", "tokenizer": "my_hf", "filter": ["lowercase"]}},
                                       "tokenizer": {"my_hf": {"type": "hf_model_tokenizer"}}}}}
    for n in (ours, stock):
        n.call("DELETE", "/hf-analyze")
    compare(c, "create index with hf analyzers", *both(ours, stock, lambda n: answer(n.call("PUT", "/hf-analyze", index))))
    for a in ("hf", "custom_hf"):
        body = {"analyzer": a, "text": "Neural Search finds Documents"}
        compare(c, f"index analyzer {a}", *both(ours, stock, lambda n: answer(n.call("POST", "/hf-analyze/_analyze", body))))


# ---------------------------------------------------------------- ingest

def ingest(c, ours, stock, text_ids, sparse_ids):
    docs = [{"_source": {"text": t, "nested": {"inner": t}}} for t in TEXTS[:5]]
    pipelines = [
        ("text_embedding", lambda mid: {"processors": [{"text_embedding": {"model_id": mid, "field_map": {"text": "emb", "nested": {"inner": "inner_emb"}}}}]}, text_ids),
        ("text_embedding batch", lambda mid: {"processors": [{"text_embedding": {"model_id": mid, "field_map": {"text": "emb"}, "batch_size": 2}}]}, text_ids),
        ("text_embedding skip_existing", lambda mid: {"processors": [{"text_embedding": {"model_id": mid, "field_map": {"text": "emb"}, "skip_existing": True}}]}, text_ids),
        ("sparse_encoding", lambda mid: {"processors": [{"sparse_encoding": {"model_id": mid, "field_map": {"text": "feat"}}}]}, sparse_ids),
        ("sparse_encoding prune", lambda mid: {"processors": [{"sparse_encoding": {"model_id": mid, "field_map": {"text": "feat"}, "prune_type": "max_ratio", "prune_ratio": 0.5}}]}, sparse_ids),
        ("chunking fixed", lambda mid: {"processors": [{"text_chunking": {"algorithm": {"fixed_token_length": {"token_limit": 3, "overlap_rate": 0.2, "tokenizer": "standard"}}, "field_map": {"text": "chunks"}}}]}, None),
        ("chunking delimiter", lambda mid: {"processors": [{"text_chunking": {"algorithm": {"delimiter": {"delimiter": " "}}, "field_map": {"text": "chunks"}}}]}, None),
        ("chunking then embedding", lambda mid: {"processors": [
            {"text_chunking": {"algorithm": {"fixed_token_length": {"token_limit": 4}}, "field_map": {"text": "chunks"}}},
            {"text_embedding": {"model_id": mid, "field_map": {"chunks": "chunk_emb"}}}]}, text_ids),
    ]
    bad = [
        ("text_embedding no model", {"processors": [{"text_embedding": {"field_map": {"text": "emb"}}}]}),
        ("text_embedding no field_map", {"processors": [{"text_embedding": {"model_id": "m"}}]}),
        ("sparse_encoding bad prune", {"processors": [{"sparse_encoding": {"model_id": "m", "field_map": {"text": "f"}, "prune_type": "max_ratio", "prune_ratio": 2}}]}),
        ("chunking two algorithms", {"processors": [{"text_chunking": {"algorithm": {"delimiter": {}, "fixed_token_length": {}}, "field_map": {"text": "c"}}}]}),
        ("chunking bad token_limit", {"processors": [{"text_chunking": {"algorithm": {"fixed_token_length": {"token_limit": 0}}, "field_map": {"text": "c"}}}]}),
        ("text_image_embedding no fields", {"processors": [{"text_image_embedding": {"model_id": "m", "embedding": "e", "field_map": {}}}]}),
    ]
    for label, make, ids in pipelines:
        a, b = (answer(n.call("POST", "/_ingest/pipeline/_simulate", {"pipeline": make(ids[i] if ids else None), "docs": docs}))
                for i, n in enumerate((ours, stock)))
        strip = lambda r: [d["doc"]["_source"] if "doc" in d else d for d in r[1]["docs"]] if r[0] == 200 else r
        compare(c, f"simulate {label}", strip(a), strip(b))
    for label, body in bad:
        a, b = (answer(n.call("PUT", "/_ingest/pipeline/bad", body)) for n in (ours, stock))
        compare(c, f"pipeline {label}", a, b)


# ---------------------------------------------------------------- neural

def neural_index(n, name, mid, rust):
    n.call("DELETE", f"/{name}")
    n.ok("PUT", f"/_ingest/pipeline/{name}-p", {"processors": [{"text_embedding": {"model_id": mid, "field_map": {"text": "emb"}}}]})
    settings = {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0, "default_pipeline": f"{name}-p"}
    if rust:
        settings["index.lucene_rust.engine"] = True
    n.ok("PUT", f"/{name}", {"settings": settings, "mappings": {"properties": {
        "text": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "integer"},
        "emb": {"type": "knn_vector", "dimension": 32, "space_type": "l2"}}}})
    bulk(n, name, DOCS)


def neural_queries(mid):
    q = lambda **kw: {"size": 10, "query": {"neural": {"emb": {"query_text": golden.QUERY, "model_id": mid, **kw}}}}
    return [
        ("k", q(k=5)),
        ("k all", {"size": 50, "query": {"neural": {"emb": {"query_text": "documents", "model_id": mid, "k": 50}}}}),
        ("boost name", q(k=3, boost=2.5, _name="nq")),
        ("filter", q(k=10, filter={"term": {"tag": "even"}})),
        ("min_score", q(min_score=0.5)),
        ("max_distance", q(max_distance=1.2)),
        ("method_parameters", q(k=5, method_parameters={"ef_search": 50})),
        ("in bool", {"query": {"bool": {"must": [{"neural": {"emb": {"query_text": "hybrid search", "model_id": mid, "k": 5}}}],
                                       "filter": [{"range": {"n": {"lt": 20}}}]}}}),
        ("expand_nested", q(k=4, expand_nested_docs=False)),
    ]


def neural_errors(mid):
    e = lambda body: {"query": {"neural": {"emb": body}}}
    return [
        ("no model", e({"query_text": "x", "k": 3})),
        ("missing model", e({"query_text": "x", "model_id": "missing-model", "k": 3})),
        ("k and min_score", e({"query_text": "x", "model_id": mid, "k": 3, "min_score": 0.1})),
        ("no text", e({"model_id": mid, "k": 3})),
        ("k zero", e({"query_text": "x", "model_id": mid, "k": 0})),
        ("unknown field", e({"query_text": "x", "model_id": mid, "k": 3, "bogus": 1})),
        ("two fields", {"query": {"neural": {"emb": {"query_text": "x", "model_id": mid, "k": 1}, "other": {}}}}),
        ("not a knn field", {"query": {"neural": {"text": {"query_text": "x", "model_id": mid, "k": 3}}}}),
        ("bad method parameter", e({"query_text": "x", "model_id": mid, "k": 3, "method_parameters": {"ef_search": 0}})),
        ("unknown method parameter", e({"query_text": "x", "model_id": mid, "k": 3, "method_parameters": {"bogus": 1}})),
        ("negative max_distance", e({"query_text": "x", "model_id": mid, "max_distance": -1})),
        ("min_score above one", e({"query_text": "x", "model_id": mid, "min_score": 1.5})),
        ("bad rescore", e({"query_text": "x", "model_id": mid, "k": 3, "rescore": {"oversample_factor": 1000}})),
    ]


def rust_stats(n):
    s, r = n.call("GET", "/_plugins/lucene_rust/stats")
    return r if s == 200 else {}


def neural(c, ours, stock, z):
    mids = [text_model(n, z) for n in (ours, stock)]
    neural_index(ours, "nq", mids[0], False)
    neural_index(ours, "nqr", mids[0], True)
    neural_index(stock, "nq", mids[1], False)
    for n, mid in zip((ours, stock), mids):
        n.ok("PUT", "/_search/pipeline/enrich", {"request_processors": [{"neural_query_enricher": {"default_model_id": mid}}]})
        n.ok("PUT", "/_search/pipeline/enrich-field", {"request_processors": [{"neural_query_enricher": {
            "default_model_id": "missing-model", "neural_field_default_id": {"emb": mid}}}]})
        n.ok("PUT", "/_search/pipeline/norm", {"phase_results_processors": [{"normalization-processor": {
            "normalization": {"technique": "min_max"}, "combination": {"technique": "arithmetic_mean"}}}]})
    before = rust_stats(ours)
    for (label, ob), (_, sb) in zip(neural_queries(mids[0]), neural_queries(mids[1])):
        a, b = hits(ours.call("POST", "/nq/_search", ob)), hits(stock.call("POST", "/nq/_search", sb))
        compare(c, f"neural {label}", a, b)
        compare(c, f"neural {label} rust engine", hits(ours.call("POST", "/nqr/_search", ob)), b)
    for pipeline in ("enrich", "enrich-field"):
        body = {"query": {"neural": {"emb": {"query_text": "dense retrieval", "k": 4}}}}
        for idx in ("nq", "nqr"):
            a = hits(ours.call("POST", f"/{idx}/_search?search_pipeline={pipeline}", body))
            b = hits(stock.call("POST", f"/nq/_search?search_pipeline={pipeline}", body))
            compare(c, f"neural enricher {pipeline} {idx}", a, b)
    for idx in ("nq", "nqr"):
        hy = lambda mid: {"query": {"hybrid": {"queries": [{"match": {"text": "documents"}},
                                                           {"neural": {"emb": {"query_text": "documents", "model_id": mid, "k": 5}}}]}}}
        a = hits(ours.call("POST", f"/{idx}/_search?search_pipeline=norm", hy(mids[0])))
        b = hits(stock.call("POST", "/nq/_search?search_pipeline=norm", hy(mids[1])))
        compare(c, f"hybrid with neural {idx}", a, b)
    after = rust_stats(ours)
    ran = after.get("native_queries", 0) - before.get("native_queries", 0)
    falls = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
             if v != before.get("fallbacks", {}).get(k, 0)}
    c.true("neural searches ran natively", ran > 0 and not falls, f"native {ran}, fallbacks {falls}")
    for (label, ob), (_, sb) in zip(neural_errors(mids[0]), neural_errors(mids[1])):
        a = hits(ours.call("POST", "/nq/_search", ob))
        b = hits(stock.call("POST", "/nq/_search", sb))
        # neural-search lists the field types from a Set.of, whose order differs between JVM runs.
        norm = lambda r: json.loads(json.dumps(r).replace(mids[0], "<id>").replace(mids[1], "<id>")
                                    .replace("[semantic,knn_vector]", "[knn_vector,semantic]"))
        compare(c, f"neural error {label}", norm(a), norm(b))
    # The documents the pipeline wrote, vectors compared approximately.
    src = lambda n: n.ok("POST", "/nq/_search", {"size": 50, "sort": [{"n": "asc"}]})["hits"]["hits"]
    compare(c, "neural indexed documents", [h["_source"] for h in src(ours)], [h["_source"] for h in src(stock)])


# ---------------------------------------------------------------- sparse

def sparse_index(n, name, mid):
    n.call("DELETE", f"/{name}")
    n.ok("PUT", f"/_ingest/pipeline/{name}-p", {"processors": [{"sparse_encoding": {"model_id": mid, "field_map": {"text": "feat"}}}]})
    n.ok("PUT", f"/{name}", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "default_pipeline": f"{name}-p"},
                             "mappings": {"properties": {"text": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "integer"},
                                                         "feat": {"type": "rank_features"}}}})
    bulk(n, name, DOCS)


# Seismic parameters under which no cluster is pruned, so both nodes score every candidate.
EXACT = {"approximate_threshold": 1, "n_postings": 100000, "cluster_ratio": 0.1, "summary_prune_ratio": 1.0}


def seismic_index(n, name, mid, rust):
    n.call("DELETE", f"/{name}")
    n.ok("PUT", f"/_ingest/pipeline/{name}-p", {"processors": [{"sparse_encoding": {"model_id": mid, "field_map": {"text": "emb"}}}]})
    settings = {"index.sparse": True, "number_of_shards": 1, "number_of_replicas": 0, "default_pipeline": f"{name}-p"}
    if rust:
        settings["index.lucene_rust.engine"] = True
    n.ok("PUT", f"/{name}", {"settings": settings, "mappings": {"properties": {
        "text": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "integer"},
        "emb": {"type": "sparse_vector", "method": {"name": "seismic", "parameters": EXACT}}}}})
    bulk(n, name, DOCS)
    n.ok("POST", f"/{name}/_flush")


def seismic_queries(mid):
    q = lambda **kw: {"size": 20, "query": {"neural_sparse": {"emb": kw}}}
    mp = {"top_n": 100, "heap_factor": 100.0, "k": 20}
    return [
        ("model", q(query_text=golden.QUERY, model_id=mid, method_parameters=mp)),
        ("default analyzer", q(query_text="Neural search finds documents", method_parameters=mp)),
        ("filter", q(query_text="documents", model_id=mid, method_parameters={**mp, "filter": {"term": {"tag": "even"}}})),
    ]


def sparse_queries(mid):
    q = lambda **kw: {"size": 20, "query": {"neural_sparse": {"feat": kw}}}
    return [
        ("model", q(query_text=golden.QUERY, model_id=mid)),
        ("model boost name", q(query_text="sparse retrieval", model_id=mid, boost=3.0, _name="s")),
        ("max_token_score", q(query_text="documents", model_id=mid, max_token_score=1.0)),
        ("default analyzer", q(query_text="Neural search finds documents")),
        ("bert-uncased analyzer", q(query_text="hybrid search combines", analyzer="bert-uncased")),
        ("mbert-uncased analyzer", q(query_text="hybrid search combines", analyzer="mbert-uncased")),
        ("standard analyzer", q(query_text="hybrid search combines", analyzer="standard")),
        ("query_tokens", q(query_tokens={"documents": 1.5, "search": 0.5})),
        ("in bool", {"query": {"bool": {"should": [{"neural_sparse": {"feat": {"query_text": "model vectors", "model_id": mid}}}],
                                        "filter": [{"term": {"tag": "odd"}}]}}}),
    ]


def sparse_errors(mid):
    q = lambda **kw: {"query": {"neural_sparse": {"feat": kw}}}
    return [
        ("model and analyzer", q(query_text="x", model_id=mid, analyzer="standard")),
        ("missing model", q(query_text="x", model_id="missing-model")),
        ("unknown analyzer", q(query_text="x", analyzer="no-such-analyzer")),
        ("nothing", q()),
        ("text and tokens", q(query_text="x", query_tokens={"a": 1.0})),
        ("negative token", q(query_tokens={"a": -1.0})),
        ("unknown field", q(query_text="x", bogus=1)),
        ("not rank_features", {"query": {"neural_sparse": {"text": {"query_tokens": {"a": 1.0}}}}}),
        ("bad max_token_score", q(query_text="x", model_id=mid, max_token_score="high")),
    ]


def sparse(c, ours, stock, z):
    mids = [deploy(n, golden.SPARSE, "SPARSE_ENCODING", z) for n in (ours, stock)]
    for n, mid in zip((ours, stock), mids):
        sparse_index(n, "sq", mid)
        n.ok("PUT", "/_search/pipeline/senrich", {"request_processors": [{"neural_query_enricher": {"default_model_id": mid}}]})
    before = rust_stats(ours)
    for (label, ob), (_, sb) in zip(sparse_queries(mids[0]), sparse_queries(mids[1])):
        compare(c, f"neural_sparse {label}", hits(ours.call("POST", "/sq/_search", ob)), hits(stock.call("POST", "/sq/_search", sb)))
    body = {"query": {"neural_sparse": {"feat": {"query_text": "how are documents searched"}}}}
    compare(c, "neural_sparse enricher", hits(ours.call("POST", "/sq/_search?search_pipeline=senrich", body)),
            hits(stock.call("POST", "/sq/_search?search_pipeline=senrich", body)))
    # Seismic clusters are written by the Rust engine; an OpenSearch-engine index on the plugin
    # node searches every segment exactly, so only the Rust-engine index is compared.
    seismic_index(ours, "sqsr", mids[0], True)
    seismic_index(stock, "sqs", mids[1], False)
    for (label, ob), (_, sb) in zip(seismic_queries(mids[0]), seismic_queries(mids[1])):
        b = hits(stock.call("POST", "/sqs/_search", sb))
        compare(c, f"neural_sparse seismic {label} rust engine", hits(ours.call("POST", "/sqsr/_search", ob)), b)
    after = rust_stats(ours)
    ran = after.get("native_queries", 0) - before.get("native_queries", 0)
    falls = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
             if v != before.get("fallbacks", {}).get(k, 0)}
    c.true("neural_sparse searches ran natively", ran > 0 and not falls, f"native {ran}, fallbacks {falls}")
    for (label, ob), (_, sb) in zip(sparse_errors(mids[0]), sparse_errors(mids[1])):
        norm = lambda r: json.loads(json.dumps(r).replace(mids[0], "<id>").replace(mids[1], "<id>"))
        compare(c, f"neural_sparse error {label}", norm(hits(ours.call("POST", "/sq/_search", ob))),
                norm(hits(stock.call("POST", "/sq/_search", sb))))
    src = lambda n: n.ok("POST", "/sq/_search", {"size": 50, "sort": [{"n": "asc"}]})["hits"]["hits"]
    compare(c, "sparse indexed documents", [h["_source"] for h in src(ours)], [h["_source"] for h in src(stock)])
    return mids


# ---------------------------------------------------------------- semantic

def semantic_dense_model(n, z):
    """tiny-bert with the additional_config.space_type a semantic field needs."""
    body = {"name": f"tiny-bert-sem-{ml.RUN}", "version": "1.0.0", "model_format": "ONNX",
            "model_content_hash_value": hashlib.sha256(z).hexdigest(), "model_content_size_in_bytes": len(z),
            "url": f"http://localhost:{ml.SERVE}/tiny-bert.zip",
            "model_config": {"model_type": "bert", "embedding_dimension": 32, "framework_type": "sentence_transformers",
                             "pooling_mode": "mean", "normalize_result": True, "additional_config": {"space_type": "l2"}}}
    mid = n.task(n.ok("POST", "/_plugins/_ml/models/_register", body)["task_id"])["model_id"]
    n.task(n.ok("POST", f"/_plugins/_ml/models/{mid}/_deploy")["task_id"])
    return mid


def semantic_mappings(dense, sparse, plain):
    return [
        ("dense", {"properties": {"text": {"type": "semantic", "model_id": dense}, "n": {"type": "integer"}}}),
        ("sparse", {"properties": {"text": {"type": "semantic", "model_id": sparse}, "n": {"type": "integer"}}}),
        ("sparse search model", {"properties": {"text": {"type": "semantic", "model_id": sparse, "semantic_field_search_analyzer": "bert-uncased"},
                                                "n": {"type": "integer"}}}),
        ("dense chunking", {"properties": {"text": {"type": "semantic", "model_id": dense, "chunking": True}, "n": {"type": "integer"}}}),
        ("dense chunking strategies", {"properties": {"text": {"type": "semantic", "model_id": dense, "chunking": [
            {"algorithm": "delimiter", "parameters": {"delimiter": " "}}, {"algorithm": "fixed_token_length", "parameters": {"token_limit": 3}}]},
            "n": {"type": "integer"}}}),
        ("dense options", {"properties": {"text": {"type": "semantic", "model_id": dense, "raw_field_type": "match_only_text",
                                                   "semantic_info_field_name": "info", "skip_existing_embedding": True,
                                                   "dense_embedding_config": {"method": {"name": "hnsw", "engine": "lucene", "parameters": {"m": 8}}}},
                                          "n": {"type": "integer"}}}),
        ("sparse prune", {"properties": {"text": {"type": "semantic", "model_id": sparse,
                                                  "sparse_encoding_config": {"prune_type": "max_ratio", "prune_ratio": 0.5}},
                                         "n": {"type": "integer"}}}),
        ("nested object", {"properties": {"obj": {"properties": {"text": {"type": "semantic", "model_id": sparse}}}, "n": {"type": "integer"}}}),
        ("keyword raw", {"properties": {"text": {"type": "semantic", "model_id": sparse, "raw_field_type": "keyword"}, "n": {"type": "integer"}}}),
        ("model_selection", {"properties": {"text": {"type": "semantic", "model_selection": {"language_option": "ENGLISH", "model_type": "SPARSE"}},
                                            "n": {"type": "integer"}}}),
    ], [
        ("no model", {"properties": {"text": {"type": "semantic"}}}),
        ("unknown model", {"properties": {"text": {"type": "semantic", "model_id": "no-such-model"}}}),
        ("wrong model kind", {"properties": {"text": {"type": "semantic", "model_id": plain}}}),
        ("bad raw type", {"properties": {"text": {"type": "semantic", "model_id": sparse, "raw_field_type": "integer"}}}),
        ("dense config on sparse", {"properties": {"text": {"type": "semantic", "model_id": sparse, "dense_embedding_config": {"method": {"name": "hnsw"}}}}}),
        ("sparse config on dense", {"properties": {"text": {"type": "semantic", "model_id": dense, "sparse_encoding_config": {"prune_type": "top_k", "prune_ratio": 3}}}}),
        ("bad chunking", {"properties": {"text": {"type": "semantic", "model_id": dense, "chunking": [{"algorithm": "nope"}]}}}),
        ("unknown param", {"properties": {"text": {"type": "semantic", "model_id": dense, "nope": 1}}}),
        ("model_selection unset", {"properties": {"text": {"type": "semantic", "model_selection": {"language_option": "MULTILINGUAL", "model_type": "DENSE"}}}}),
        ("model_selection bad", {"properties": {"text": {"type": "semantic", "model_selection": {"language_option": "KLINGON", "model_type": "DENSE"}}}}),
        ("model_selection and model_id", {"properties": {"text": {"type": "semantic", "model_id": dense,
                                                                  "model_selection": {"language_option": "ENGLISH", "model_type": "DENSE"}}}}),
    ]


def semantic_queries(label):
    q = [("neural", {"query": {"neural": {"text": {"query_text": "dense retrieval of documents", "k": 5}}}}),
         ("neural filter", {"query": {"neural": {"text": {"query_text": "documents", "k": 5, "filter": {"range": {"n": {"gte": 4}}}}}}}),
         ("match raw", {"query": {"match": {"text": "documents"}}}),
         ("bool", {"query": {"bool": {"must": [{"neural": {"text": {"query_text": "search", "k": 10}}}], "filter": [{"term": {"n": 3}}]}}})]
    if "sparse" in label or label in ("keyword raw", "model_selection"):
        q = [(l, json.loads(json.dumps(b).replace(', "k": 5', "").replace(', "k": 10', "").replace('"k": 5, ', ""))) for l, b in q]
        q.append(("neural_sparse", {"query": {"neural_sparse": {"text": {"query_text": "sparse retrieval of documents"}}}}))
    if label == "nested object":
        q = [(l, json.loads(json.dumps(b).replace('"text"', '"obj.text"'))) for l, b in q]
    return q


def semantic(c, ours, stock, z, zs):
    dense = [semantic_dense_model(n, z) for n in (ours, stock)]
    sparse = [deploy(n, golden.SPARSE, "SPARSE_ENCODING", zs) for n in (ours, stock)]
    plain = [text_model(n, z) for n in (ours, stock)]
    for n, mid in zip((ours, stock), sparse):
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.neural_search.model_selection.model_id.sparse.english": mid}})
    ids = dense + sparse + plain
    def norm(r):
        t = functools.reduce(lambda t, i: t.replace(i, "<id>"), ids, json.dumps(r))
        # Where no engine is named, stock's knn_vector default is faiss and the plugin's lucene.
        t = t.replace('"engine": "faiss"', '"engine": "lucene"')
        # neural-search lists allowed values from a Set.of, whose order differs between JVM runs.
        return json.loads(re.sub(r"\[([\w-]+(?:, [\w-]+)+)\]", lambda m: "[" + ", ".join(sorted(m.group(1).split(", "))) + "]", t))
    good = [semantic_mappings(d, sp, pl) for d, sp, pl in zip(dense, sparse, plain)]
    docs = [{"_id": d["_id"], "text": d["text"], "n": d["n"]} for d in DOCS[:12]] + [{"_id": "long", "text": LONG, "n": 99}]
    before = rust_stats(ours)
    for (label, mo), (_, ms) in zip(good[0][0], good[1][0]):
        idx = "sem-" + label.replace(" ", "-").replace("_", "-")
        outs = []
        for n, m in ((ours, mo), (stock, ms)):
            n.call("DELETE", f"/{idx}")
            outs.append(answer(n.call("PUT", f"/{idx}", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "index.knn": True}, "mappings": m})))
        compare(c, f"semantic create {label}", norm(outs[0]), norm(outs[1]))
        if outs[0][0] != 200 or outs[1][0] != 200:
            continue
        compare(c, f"semantic mapping {label}", *both(ours, stock, lambda n: norm(n.ok("GET", f"/{idx}/_mapping")[idx])))
        body = docs if label != "nested object" else [{"_id": d["_id"], "obj": {"text": d["text"]}, "n": d["n"]} for d in docs]
        for n in (ours, stock):
            bulk(n, idx, body)
        src = lambda n: norm([h["_source"] for h in n.ok("POST", f"/{idx}/_search", {"size": 50, "sort": [{"n": "asc"}]})["hits"]["hits"]])
        compare(c, f"semantic documents {label}", src(ours), src(stock))
        for qlabel, q in semantic_queries(label):
            compare(c, f"semantic {label} {qlabel}", *both(ours, stock, lambda n: norm(hits(n.call("POST", f"/{idx}/_search", q)))))
        # An update re-embeds only when the text changed (skip_existing_embedding) and keeps the model info.
        upd = {"doc": {"n": 50}} if label != "nested object" else {"doc": {"n": 50}}
        for n in (ours, stock):
            n.ok("POST", f"/{idx}/_update/1?refresh=true", upd)
        compare(c, f"semantic updated {label}", *both(ours, stock, lambda n: norm(n.ok("GET", f"/{idx}/_doc/1")["_source"])))
    for (label, mo), (_, ms) in zip(good[0][1], good[1][1]):
        outs = []
        for n, m in ((ours, mo), (stock, ms)):
            n.call("DELETE", "/sem-bad")
            outs.append(answer(n.call("PUT", "/sem-bad", {"settings": {"index.knn": True}, "mappings": m})))
        compare(c, f"semantic bad {label}", norm(outs[0]), norm(outs[1]))
    after = rust_stats(ours)
    falls = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
             if v != before.get("fallbacks", {}).get(k, 0)}
    c.true("semantic searches ran natively", after.get("native_queries", 0) > before.get("native_queries", 0) and not falls,
           f"fallbacks {falls}")
    # Adding a semantic field to an existing index through the put mapping API.
    outs = []
    for n, d in zip((ours, stock), dense):
        n.call("DELETE", "/sem-put")
        n.ok("PUT", "/sem-put", {"settings": {"index.knn": True}, "mappings": {"properties": {"n": {"type": "integer"}}}})
        outs.append(answer(n.call("PUT", "/sem-put/_mapping", {"properties": {"text": {"type": "semantic", "model_id": d}}})))
    compare(c, "semantic put mapping", norm(outs[0]), norm(outs[1]))
    compare(c, "semantic put mapping result", *both(ours, stock, lambda n: norm(n.ok("GET", "/sem-put/_mapping")["sem-put"])))
    for n in (ours, stock):
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.neural_search.model_selection.model_id.sparse.english": None}})


# ---------------------------------------------------------------- rerank

CROSS_CONFIG = {"model_type": "bert", "embedding_dimension": 1, "framework_type": "huggingface_transformers"}


def rerank(c, ours, stock, z):
    mids = [deploy(n, golden.CROSS, "TEXT_SIMILARITY", z, CROSS_CONFIG) for n in (ours, stock)]
    docs = [{**d, "meta": {"rank": (int(d["_id"]) * 7) % 11}} for d in DOCS]
    for n, mid in zip((ours, stock), mids):
        n.call("DELETE", "/rr")
        n.ok("PUT", "/rr", {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
                            "mappings": {"properties": {"text": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "integer"}}}})
        bulk(n, "rr", docs)
        n.ok("PUT", "/_search/pipeline/rr-ml", {"response_processors": [{"rerank": {
            "ml_opensearch": {"model_id": mid}, "context": {"document_fields": ["text"]}}}]})
        n.ok("PUT", "/_search/pipeline/rr-field", {"response_processors": [{"rerank": {
            "by_field": {"target_field": "meta.rank", "keep_previous_score": True}}}]})
        n.ok("PUT", "/_search/pipeline/rr-field-remove", {"response_processors": [{"rerank": {
            "by_field": {"target_field": "meta.rank", "remove_target_field": True}}}]})
    q = {"query": {"match": {"text": "documents search"}}, "size": 10}
    searches = [
        ("ml query_text", "rr-ml", {**q, "ext": {"rerank": {"query_context": {"query_text": golden.QUERY}}}}),
        ("ml query_text_path", "rr-ml", {**q, "ext": {"rerank": {"query_context": {"query_text_path": "query.match.text.query"}}}}),
        ("ml no context", "rr-ml", q),
        ("ml both contexts", "rr-ml", {**q, "ext": {"rerank": {"query_context": {"query_text": "a", "query_text_path": "x"}}}}),
        ("ml bad path", "rr-ml", {**q, "ext": {"rerank": {"query_context": {"query_text_path": "query.nope"}}}}),
        ("by_field", "rr-field", q),
        ("by_field remove", "rr-field-remove", q),
    ]
    for label, pipeline, body in searches:
        a, b = (n.call("POST", f"/rr/_search?search_pipeline={pipeline}", body) for n in (ours, stock))
        source = lambda r: [(h["_id"], h["_score"], h.get("_source")) for h in r[1]["hits"]["hits"]] if r[0] == 200 else answer(r)
        compare(c, f"rerank {label}", source(a), source(b))
    bad = [
        ("no type", {"response_processors": [{"rerank": {"context": {"document_fields": ["text"]}}}]}),
        ("two types", {"response_processors": [{"rerank": {"ml_opensearch": {"model_id": "m"}, "by_field": {"target_field": "x"},
                                                          "context": {"document_fields": ["text"]}}}]}),
        ("ml without context", {"response_processors": [{"rerank": {"ml_opensearch": {"model_id": "m"}}}]}),
        ("ml empty fields", {"response_processors": [{"rerank": {"ml_opensearch": {"model_id": "m"}, "context": {"document_fields": []}}}]}),
        ("ml too many fields", {"response_processors": [{"rerank": {"ml_opensearch": {"model_id": "m"},
                                                                   "context": {"document_fields": [f"f{i}" for i in range(51)]}}}]}),
        ("by_field without target", {"response_processors": [{"rerank": {"by_field": {}}}]}),
        ("by_field bad flag", {"response_processors": [{"rerank": {"by_field": {"target_field": "x", "keep_previous_score": "maybe"}}}]}),
    ]
    for label, body in bad:
        compare(c, f"rerank pipeline {label}", *both(ours, stock, lambda n: answer(n.call("PUT", "/_search/pipeline/rr-bad", body))))


# ---------------------------------------------------------------- two-phase

def two_phase(c, ours, stock, z):
    mids = [deploy(n, golden.SPARSE, "SPARSE_ENCODING", z) for n in (ours, stock)]
    for n, mid in zip((ours, stock), mids):
        sparse_index(n, "tp", mid)
        n.ok("PUT", "/_search/pipeline/tp", {"request_processors": [{"neural_sparse_two_phase_processor": {
            "tag": "t", "description": "d", "enabled": True,
            "two_phase_parameter": {"prune_ratio": 0.4, "expansion_rate": 5.0, "max_window_size": 10000}}}]})
        n.ok("PUT", "/_search/pipeline/tp-off", {"request_processors": [{"neural_sparse_two_phase_processor": {"enabled": False}}]})
    q = lambda mid: {"size": 15, "query": {"neural_sparse": {"feat": {"query_text": golden.QUERY, "model_id": mid}}}}
    bq = lambda mid: {"size": 15, "query": {"bool": {"should": [{"neural_sparse": {"feat": {"query_text": "documents", "model_id": mid}}},
                                                               {"match": {"text": "search"}}]}}}
    for label, pipeline, make in (("query", "tp", q), ("bool", "tp", bq), ("disabled", "tp-off", q)):
        a, b = (hits(n.call("POST", f"/tp/_search?search_pipeline={pipeline}", make(mid))) for n, mid in zip((ours, stock), mids))
        compare(c, f"two-phase {label}", a, b)
    bad = [("prune_ratio", {"two_phase_parameter": {"prune_ratio": 2.0}}),
           ("expansion_rate", {"two_phase_parameter": {"expansion_rate": 0.5}}),
           ("max_window_size", {"two_phase_parameter": {"max_window_size": 10}}),
           ("unknown", {"two_phase_parameter": {"bogus": 1}})]
    for label, body in bad:
        p = {"request_processors": [{"neural_sparse_two_phase_processor": body}]}
        compare(c, f"two-phase pipeline {label}", *both(ours, stock, lambda n: answer(n.call("PUT", "/_search/pipeline/tp-bad", p))))


# ---------------------------------------------------------------- warmup, clear_cache, settings

def warmup(c, ours, stock):
    for n in (ours, stock):
        for name, sparse_flag in (("wsparse", True), ("wplain", False)):
            n.call("DELETE", f"/{name}")
            n.ok("PUT", f"/{name}", {"settings": {"index.sparse": sparse_flag, "number_of_shards": 2, "number_of_replicas": 0},
                                     "mappings": {"properties": {"emb": {"type": "sparse_vector", "method": {"name": "seismic"}}}
                                                  if sparse_flag else {"t": {"type": "text"}}}})
    for op in ("warmup", "clear_cache"):
        for idx in ("wsparse", "wplain", "wsparse,wplain", "missing-index"):
            compare(c, f"{op} {idx}", *both(ours, stock, lambda n: answer(n.call("POST", f"/_plugins/_neural/{op}/{idx}"))))
        compare(c, f"{op} GET", *both(ours, stock, lambda n: n.call("GET", f"/_plugins/_neural/{op}/wsparse")[0]))


PREFIXES = ("knn.", "plugins.neural_search.", "index.knn.", "index.neural_search.", "seismic_thread_pool.")


def settings(c, ours, stock):
    def flat(n):
        r = n.ok("GET", "/_cluster/settings?include_defaults=true&flat_settings=true")
        out = {}
        for part in ("defaults", "persistent", "transient"):
            out.update({k: v for k, v in r.get(part, {}).items() if k.startswith(PREFIXES)})
        return out
    a, b = flat(ours), flat(stock)
    for k in sorted(set(a) | set(b)):
        compare(c, f"cluster setting {k}", a.get(k, "<missing>"), b.get(k, "<missing>"))
    def index_flat(n):
        n.call("DELETE", "/sset")
        n.ok("PUT", "/sset", {"settings": {"index.knn": True}})
        r = n.ok("GET", "/sset/_settings?include_defaults=true&flat_settings=true")["sset"]
        return {k: v for part in ("settings", "defaults") for k, v in r.get(part, {}).items() if k.startswith(PREFIXES)}
    a, b = index_flat(ours), index_flat(stock)
    for k in sorted(set(a) | set(b)):
        compare(c, f"index setting {k}", a.get(k, "<missing>"), b.get(k, "<missing>"))
    for key, value in (("plugins.neural_search.circuit_breaker.overhead", -1), ("knn.quantization.cache.size.limit", "50%"),
                       ("plugins.neural_search.sparse.algo_param.index_thread_qty", 0), ("knn.remote_index_build.poll.interval", "x"),
                       ("knn.memory.circuit_breaker.limit", "150%"), ("knn.memory.circuit_breaker.limit", "abc"),
                       ("knn.memory.circuit_breaker.limit.hot", "101%"), ("knn.algo_param.index_thread_qty", 0),
                       ("knn.algo_param.index_thread_qty", 33)):
        body = {"persistent": {key: value}}
        compare(c, f"bad setting {key}", *both(ours, stock, lambda n: answer(n.call("PUT", "/_cluster/settings", body))))
    for key, value in (("knn.memory.circuit_breaker.limit.hot", "40%"), ("knn.memory.circuit_breaker.limit", "60%"),
                       ("knn.algo_param.index_thread_qty", 32), ("knn.circuit_breaker.unset.percentage", 80)):
        for v in (value, None):
            body = {"persistent": {key: v}}
            compare(c, f"set {key}={v}", *both(ours, stock, lambda n: answer(n.call("PUT", "/_cluster/settings", body))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--only", help="comma-separated sections to run (default: all)")
    a = ap.parse_args()
    zips_dir = glob.glob(os.path.join(ROOT, "opensearch-plugin", "build", "distributions", "lucene-rust-*.zip"))
    if not zips_dir:
        sys.exit("no plugin zip: run gradle -p opensearch-plugin bundlePlugin")
    docker_dir = os.path.join(ROOT, "opensearch-plugin", "docker")
    for z in glob.glob(os.path.join(docker_dir, "*.zip")):
        os.remove(z)
    shutil.copy(zips_dir[0], docker_dir)
    image = f"lucene-rust-opensearch:{VERSION}"
    subprocess.run(["docker", "build", "-q", "-t", image, docker_dir], check=True, capture_output=True)
    ml.start(OURS[0], OURS[1], image)
    if a.stock:
        stock = ml.Node(a.stock)
    else:
        ml.start(STOCK[0], STOCK[1], f"opensearchproject/opensearch:{VERSION}")
        stock = ml.Node(f"http://localhost:{STOCK[1]}")
    ours = ml.Node(f"http://localhost:{OURS[1]}")
    zips = {name: golden.model_zip(name) for name in ("tiny-bert", golden.SPARSE, golden.CROSS)}
    ml.serve(zips)
    c = ml.Check()
    try:
        for n in (ours, stock):
            ml.wait(n)
            n.ok("PUT", "/_cluster/settings", ml.SETTINGS)
        sections = {
            "analyze": lambda: analyze(c, ours, stock),
            "neural": lambda: neural(c, ours, stock, zips["tiny-bert"]),
            "sparse": lambda: sparse(c, ours, stock, zips[golden.SPARSE]),
            "rerank": lambda: rerank(c, ours, stock, zips[golden.CROSS]),
            "two_phase": lambda: two_phase(c, ours, stock, zips[golden.SPARSE]),
            "warmup": lambda: warmup(c, ours, stock),
            "settings": lambda: settings(c, ours, stock),
            "semantic": lambda: semantic(c, ours, stock, zips["tiny-bert"], zips[golden.SPARSE]),
            "ingest": lambda: ingest(c, ours, stock, [text_model(n, zips["tiny-bert"]) for n in (ours, stock)],
                                     [deploy(n, golden.SPARSE, "SPARSE_ENCODING", zips[golden.SPARSE]) for n in (ours, stock)]),
        }
        for name in (a.only.split(",") if a.only else sections):
            sections[name]()
    finally:
        if not a.keep:
            subprocess.run(["docker", "rm", "-f", OURS[0]] + ([] if a.stock else [STOCK[0]]), capture_output=True)
    for f in c.failed:
        print("MISMATCH", f)
    print(f"check-neural-plugin: {c.rows - len(c.failed)}/{c.rows} rows match stock")
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
