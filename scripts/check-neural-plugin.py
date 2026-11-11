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
  mmr        k-NN's mmr search extension on knn, neural and match queries (the
             spaces and data types, explain, errors), and knn_default_excludes,
             with the system processors off and on
  semantic   semantic fields (dense and sparse models, chunking, the field's
             options, model_selection, put mapping): the mappings the
             transformer writes, the documents the system processor embeds,
             neural and match queries on them, and the mapping errors
  sparse     neural_sparse by query_text with a model, by query_text with the
             default bert-uncased analyzer, by query_tokens, with the enricher,
             and its errors

  scripts/check-neural-plugin.py [--stock http://localhost:9200] [--keep]
                                 [--only analyze,ingest,neural,sparse,rerank,two_phase,warmup,settings,semantic,mmr]

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
        # Updates: one leaving the text alone, one changing it, which re-embeds (skip_existing_embedding
        # reuses the stored embedding only for unchanged text).
        text_upd = {"obj": {"text": "a changed text"}} if label == "nested object" else {"text": "a changed text"}
        for step, upd in (("same text", {"n": 50}), ("new text", text_upd)):
            for n in (ours, stock):
                n.ok("POST", f"/{idx}/_update/1?refresh=true", {"doc": upd})
            compare(c, f"semantic updated {label} {step}", *both(ours, stock, lambda n: norm(n.ok("GET", f"/{idx}/_doc/1")["_source"])))
        if label == "dense options":
            info = lambda n: n.ok("GET", f"/{idx}/_doc/1")["_source"]["info"]["embedding"]
            first = info(ours)
            ours.ok("POST", f"/{idx}/_update/1?refresh=true", {"doc": {"n": 51}})
            kept = info(ours)
            ours.ok("POST", f"/{idx}/_update/1?refresh=true", {"doc": {"text": "yet another text"}})
            c.true("semantic skip_existing_embedding keeps, then re-embeds", kept == first and info(ours) != first,
                   "the embedding did not stay for the same text or did not change for a new one")
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


# ---------------------------------------------------------------- mmr

def mmr_docs():
    rng = __import__("random").Random(11)
    out = []
    for i in range(40):
        v = [round(rng.uniform(-1, 1), 3) for _ in range(4)]
        out.append({"_id": str(i), "text": TEXTS[i % len(TEXTS)] + f" {i}", "n": i, "emb": v, "cos": v,
                    "bytes": [int(x * 100) for x in v], "bits": [i % 256], "obj": {"inner": v},
                    "nest": [{"v": v}, {"v": [-x for x in v]}]})
    return out


def mmr(c, ours, stock, z):
    mids = [text_model(n, z) for n in (ours, stock)]
    mapping = {"properties": {"text": {"type": "text"}, "n": {"type": "integer"},
                              "emb": {"type": "knn_vector", "dimension": 4},
                              # lucene named: stock's default faiss engine returns cosine vectors normalized in _source.
                              "cos": {"type": "knn_vector", "dimension": 4, "space_type": "cosinesimil", "method": {"name": "hnsw", "engine": "lucene"}},
                              "bytes": {"type": "knn_vector", "dimension": 4, "data_type": "byte", "space_type": "innerproduct"},
                              "bits": {"type": "knn_vector", "dimension": 8, "data_type": "binary"},
                              "obj": {"properties": {"inner": {"type": "knn_vector", "dimension": 4}}},
                              "nest": {"type": "nested", "properties": {"v": {"type": "knn_vector", "dimension": 4}}},
                              "dense": {"type": "knn_vector", "dimension": 32}}}
    docs = mmr_docs()
    for n, mid in zip((ours, stock), mids):
        for idx in ("mmr", "mmr2"):
            n.call("DELETE", f"/{idx}")
            n.ok("PUT", f"/{idx}", {"settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0}, "mappings": mapping})
        n.ok("PUT", "/_ingest/pipeline/mmr-emb", {"processors": [{"text_embedding": {"model_id": mid, "field_map": {"text": "dense"}}}]})
        bulk(n, "mmr", docs, "mmr-emb")
        bulk(n, "mmr2", docs[:10], "mmr-emb")
        n.call("DELETE", "/mmr3")
        n.ok("PUT", "/mmr3", {"settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0},
                              "mappings": {"properties": {"bits": {"type": "knn_vector", "dimension": 16, "data_type": "binary"}}}})
        bulk(n, "mmr3", [{"_id": f"b{i}", "bits": [i, -i]} for i in range(10)])
    q = [0.5, -0.2, 0.1, 0.9]
    knn = lambda f="emb", k=10, vec=q: {"knn": {f: {"vector": vec, "k": k}}}
    ext = lambda **kw: {"mmr": {"diversity": 0.5, "candidates": 20, **kw}}
    no_vectors = lambda b: {"_source": {"excludes": ["emb", "cos", "bytes", "bits", "obj", "nest", "dense"]}, **b}
    disabled = [("mmr disabled", "/mmr/_search", {"query": knn(), "size": 5, "ext": ext()})]
    enabled = [
        ("knn", "/mmr/_search", {"query": knn(), "size": 5, "ext": ext()}),
        ("knn diversity 0", "/mmr/_search", {"query": knn(), "size": 5, "ext": ext(diversity=0)}),
        ("knn diversity 1", "/mmr/_search", {"query": knn(), "size": 5, "ext": ext(diversity=1)}),
        ("knn default candidates", "/mmr/_search", {"query": knn(), "size": 4, "ext": {"mmr": {}}}),
        # On the lucene-engine field: stock's default faiss explains its scores as returned from FAISS.
        ("knn explain", "/mmr/_search", {"query": knn("cos"), "size": 3, "explain": True, "ext": ext(explain=True)}),
        ("knn cosine", "/mmr/_search", {"query": knn("cos"), "size": 5, "ext": ext()}),
        ("knn byte innerproduct", "/mmr/_search", {"query": knn("bytes", vec=[50, -20, 10, 90]), "size": 5, "ext": ext()}),
        ("knn binary hamming", "/mmr/_search", {"query": knn("bits", vec=[7]), "size": 5, "ext": ext()}),
        ("knn object field", "/mmr/_search", {"query": knn("obj.inner"), "size": 5, "ext": ext()}),
        ("knn max_distance", "/mmr/_search", {"query": {"knn": {"emb": {"vector": q, "max_distance": 2.0}}}, "size": 5, "ext": ext()}),
        ("knn in bool", "/mmr/_search", {"query": {"bool": {"must": [knn()]}}, "size": 5, "ext": ext(vector_field_path="emb")}),
        ("match with path", "/mmr/_search", {"query": {"match": {"text": "documents search"}}, "size": 5, "ext": ext(vector_field_path="emb")}),
        ("match with path and types", "/mmr/_search", {"query": {"match": {"text": "documents"}}, "size": 5,
                                                       "ext": ext(vector_field_path="emb", vector_field_data_type="float", vector_field_space_type="innerproduct")}),
        ("two indices", "/mmr,mmr2/_search", {"query": knn(), "size": 5, "ext": ext()}),
        ("source excludes", "/mmr/_search", {"query": knn(), "size": 5, "_source": {"excludes": ["emb"]}, "ext": ext()}),
        ("source false", "/mmr/_search", {"query": knn(), "size": 5, "_source": False, "ext": ext()}),
        ("from", "/mmr/_search", {"query": knn(), "from": 2, "size": 3, "ext": ext()}),
        ("neural", "/mmr/_search", {"query": {"neural": {"dense": {"query_text": "documents search", "model_id": "<mid>", "k": 10}}}, "size": 5, "ext": ext()}),
        # errors
        ("bad diversity", "/mmr/_search", {"query": knn(), "ext": ext(diversity=2)}),
        ("bad candidates", "/mmr/_search", {"query": knn(), "ext": ext(candidates=0)}),
        ("size over candidates", "/mmr/_search", {"query": knn(), "size": 30, "ext": ext()}),
        ("unknown param", "/mmr/_search", {"query": knn(), "ext": ext(nope=1)}),
        ("bad space type", "/mmr/_search", {"query": knn(), "ext": ext(vector_field_space_type="nope")}),
        ("bad data type", "/mmr/_search", {"query": knn(), "ext": ext(vector_field_data_type="nope")}),
        ("l1 space type", "/mmr/_search", {"query": {"match": {"text": "documents"}}, "size": 5, "ext": ext(vector_field_path="emb", vector_field_space_type="l1")}),
        ("match without path", "/mmr/_search", {"query": {"match": {"text": "documents"}}, "ext": ext()}),
        ("path not a vector", "/mmr/_search", {"query": {"match": {"text": "documents"}}, "ext": ext(vector_field_path="text")}),
        ("path missing", "/mmr/_search", {"query": {"match": {"text": "documents"}}, "ext": ext(vector_field_path="nope")}),
        ("nested path", "/mmr/_search", {"query": {"match": {"text": "documents"}}, "ext": ext(vector_field_path="nest.v")}),
        ("size 0 no candidates", "/mmr/_search", {"query": knn(), "size": 0, "ext": {"mmr": {}}}),
        ("binary dims differ", "/mmr,mmr3/_search", {"query": knn("bits", vec=[7]), "size": 5, "ext": ext()}),
        ("stored_fields none", "/mmr/_search", {"query": knn(), "stored_fields": "_none_", "ext": ext()}),
        ("no query", "/mmr/_search", {"ext": ext(vector_field_path="emb")}),
    ]
    # knn_default_excludes: a search leaves knn_vector fields out of _source unless it asks for them.
    excludes = [
        ("default", {"query": {"term": {"n": 3}}}),
        ("source true", {"query": {"term": {"n": 3}}, "_source": True}),
        ("source includes vector", {"query": {"term": {"n": 3}}, "_source": ["emb", "text"]}),
        ("source includes wildcard", {"query": {"term": {"n": 3}}, "_source": {"includes": ["o*"]}}),
        ("source excludes other", {"query": {"term": {"n": 3}}, "_source": {"excludes": ["text"]}}),
        ("stored_fields", {"query": {"term": {"n": 3}}, "stored_fields": ["n"]}),
        ("inner_hits", {"query": {"nested": {"path": "nest", "query": {"match_all": {}}, "inner_hits": {"size": 1}}}, "size": 1, "sort": [{"n": "asc"}]}),
    ]

    def run(n, mid, path, body):
        b = json.loads(json.dumps(body).replace("<mid>", mid))
        s, r = n.call("POST", path, b)
        if s != 200:
            return answer((s, r))
        return [r["hits"].get("total"), r["hits"].get("max_score")] + [
            (h["_index"], h["_id"], h["_score"], h.get("_source"), h.get("_explanation"),
             {k: v["hits"]["hits"][0].get("_source") for k, v in h.get("inner_hits", {}).items()}) for h in r["hits"]["hits"]]

    def section(cases, prefix):
        for label, path, body in cases:
            a, b = (run(n, mid, path, body if "neural" in label or "_source" in body else no_vectors(body)) for n, mid in zip((ours, stock), mids))
            compare(c, f"{prefix} {label}", a, b)

    factories = "cluster.search.enabled_system_generated_factories"
    section(disabled, "mmr")
    for label, body in excludes:
        compare(c, f"default excludes off {label}", *both(ours, stock, lambda n: run(n, "", "/mmr/_search", body)))
    for n in (ours, stock):
        n.ok("PUT", "/_cluster/settings", {"persistent": {factories: ["*"]}})
    try:
        section(enabled, "mmr")
        for label, body in excludes:
            compare(c, f"default excludes {label}", *both(ours, stock, lambda n: run(n, "", "/mmr/_search", body)))
        compare(c, "default excludes msearch", *both(ours, stock, lambda n: (lambda r: [[h["_source"] for h in x["hits"]["hits"]] for x in r[1]["responses"]])(
            bulk_call(n, "/_msearch", '{"index":"mmr"}\n{"query":{"term":{"n":3}}}\n'))))
        compare(c, "default excludes get", *both(ours, stock, lambda n: n.ok("GET", "/mmr/_doc/3")["_source"]))
    finally:
        for n in (ours, stock):
            n.ok("PUT", "/_cluster/settings", {"persistent": {factories: None}})


# ---------------------------------------------------------------- ivf

IVF_FIELD = {"type": "knn_vector", "dimension": 32, "space_type": "l2",
             "method": {"name": "ivf", "engine": "lucene_rust", "parameters": {"target_cluster_size": 16, "rerank_encoding": "float"}}}
# Every cluster: the IVF search is then the exhaustive one, which stock's HNSW over 36 documents is too.
ALL_PROBES = {"nprobes": 1024}


def ivf(c, ours, stock, z):
    """The plugin's IVF fields (lucene_rust engine) through the stock k-NN and neural-search surface:
    the same documents, hits, scores and errors as stock's lucene HNSW field over the same documents."""
    mids = [text_model(n, z) for n in (ours, stock)]
    sem = [semantic_dense_model(n, z) for n in (ours, stock)]
    hnsw = {"type": "knn_vector", "dimension": 32, "space_type": "l2", "method": {"name": "hnsw", "engine": "lucene"}}
    cos = lambda f: {**f, "space_type": "cosinesimil"}
    ip = lambda f: {**f, "space_type": "innerproduct"}
    for n, mid, smid in zip((ours, stock), mids, sem):
        field = IVF_FIELD if n is ours else hnsw
        settings = {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0, "default_pipeline": "ivf-p"}
        if n is ours:
            settings["index.lucene_rust.engine"] = True
        n.ok("PUT", "/_ingest/pipeline/ivf-p", {"processors": [{"text_embedding": {"model_id": mid, "field_map": {"text": "emb"}}},
                                                               {"copy": {"source_field": "emb", "target_field": "cos"}},
                                                               {"copy": {"source_field": "emb", "target_field": "ip"}}]})
        n.call("DELETE", "/ivf")
        n.ok("PUT", "/ivf", {"settings": settings, "mappings": {"properties": {
            "text": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "integer"},
            "emb": field, "cos": cos(field), "ip": ip(field)}}})
        bulk(n, "ivf", DOCS)
        n.ok("PUT", "/_search/pipeline/ivf-hybrid", {"phase_results_processors": [{"normalization-processor": {
            "normalization": {"technique": "min_max"}, "combination": {"technique": "arithmetic_mean"}}}]})
        # A semantic field whose dense embeddings are an IVF field.
        n.call("DELETE", "/ivf-sem")
        dense = {"dense_embedding_config": {"method": IVF_FIELD["method"]}} if n is ours else {}
        n.ok("PUT", "/ivf-sem", {"settings": {k: v for k, v in settings.items() if k != "default_pipeline"},
                                 "mappings": {"properties": {"text": {"type": "semantic", "model_id": smid, **dense}, "n": {"type": "integer"}}}})
        bulk(n, "ivf-sem", [{"_id": d["_id"], "text": d["text"], "n": d["n"]} for d in DOCS])

    # Between two documents: a query equal to a document makes MMR's later picks ties.
    a, b = (ours.ok("GET", f"/ivf/_doc/{i}")["_source"]["emb"] for i in (5, 8))
    vec = [round((x + y) / 2, 4) for x, y in zip(a, b)]
    knn = lambda f="emb", **kw: {"knn": {f: {"vector": vec, **kw}}}
    neural = lambda f="emb", **kw: {"neural": {f: {"query_text": golden.QUERY, "model_id": "<mid>", **kw}}}  # kw may override query_text
    mmr = lambda **kw: {"mmr": {"diversity": 0.5, "candidates": 20, **kw}}
    cases = [
        ("knn", "/ivf", {"query": knn(k=10)}),
        ("knn cosine", "/ivf", {"query": knn("cos", k=10)}),
        ("knn innerproduct", "/ivf", {"query": knn("ip", k=10)}),
        ("knn filter", "/ivf", {"query": knn(k=10, filter={"term": {"tag": "even"}})}),
        ("knn min_score", "/ivf", {"size": 50, "query": knn(min_score=0.6)}),
        ("knn max_distance", "/ivf", {"size": 50, "query": knn(max_distance=0.8)}),
        ("knn cosine min_score", "/ivf", {"size": 50, "query": knn("cos", min_score=0.8)}),
        ("knn rescore", "/ivf", {"query": knn(k=10, rescore={"oversample_factor": 2.0})}),
        ("knn boost name", "/ivf", {"query": knn(k=5, boost=2.0, _name="q")}),
        ("knn in bool", "/ivf", {"query": {"bool": {"must": [knn(k=10)], "filter": [{"range": {"n": {"lt": 20}}}]}}}),
        ("knn explain", "/ivf", {"size": 3, "explain": True, "query": knn(k=3)}),
        ("knn source vector", "/ivf", {"size": 3, "_source": ["emb", "n"], "query": knn(k=3)}),
        ("knn fields", "/ivf", {"size": 3, "_source": False, "fields": ["emb"], "docvalue_fields": ["emb"], "query": knn(k=3)}),
        ("knn from", "/ivf", {"from": 3, "size": 4, "query": knn(k=10)}),
        ("knn two indices", "/ivf,ivf-sem", {"query": knn(k=5)}),
        ("knn_score script", "/ivf", {"size": 5, "query": {"script_score": {"query": {"match_all": {}}, "script": {
            "source": "knn_score", "lang": "knn", "params": {"field": "emb", "query_value": vec, "space_type": "l2"}}}}}),
        ("painless l2Squared", "/ivf", {"size": 5, "query": {"script_score": {"query": {"match_all": {}}, "script": {
            "source": "1 / (1 + l2Squared(params.q, doc['emb']))", "params": {"q": vec}}}}}),
        ("neural", "/ivf", {"query": neural(k=10)}),
        ("neural filter", "/ivf", {"query": neural(k=10, filter={"term": {"tag": "odd"}})}),
        ("neural min_score", "/ivf", {"size": 50, "query": neural(min_score=0.5)}),
        ("neural max_distance", "/ivf", {"size": 50, "query": neural(max_distance=1.2)}),
        ("neural cosine", "/ivf", {"query": neural("cos", k=10)}),
        ("neural in bool", "/ivf", {"query": {"bool": {"should": [neural(k=5), {"match": {"text": "documents"}}]}}}),
        ("hybrid", "/ivf/_search?search_pipeline=ivf-hybrid", {"query": {"hybrid": {"queries": [
            {"match": {"text": "documents search"}}, neural(k=10)]}}}),
        ("hybrid knn", "/ivf/_search?search_pipeline=ivf-hybrid", {"query": {"hybrid": {"queries": [
            {"match": {"text": "documents"}}, knn(k=10)]}}}),
        ("semantic neural", "/ivf-sem", {"query": {"neural": {"text": {"query_text": golden.QUERY, "k": 10}}}}),
        ("semantic neural filter", "/ivf-sem", {"query": {"neural": {"text": {"query_text": "documents", "k": 5, "filter": {"range": {"n": {"gte": 4}}}}}}}),
        ("mmr knn", "/ivf", {"size": 5, "query": knn(k=10), "ext": mmr()}),
        ("mmr neural", "/ivf", {"size": 5, "query": neural(k=10, query_text="dense retrieval of documents"), "ext": mmr()}),
        ("mmr cosine", "/ivf", {"size": 5, "query": knn("cos", k=10), "ext": mmr()}),
        ("mmr path", "/ivf", {"size": 5, "query": {"match": {"text": "documents"}}, "ext": mmr(vector_field_path="ip")}),
        ("mmr explain", "/ivf", {"size": 3, "explain": True, "query": knn(k=10), "ext": mmr(explain=True)}),
        ("default excludes", "/ivf", {"size": 3, "query": {"term": {"n": 3}}}),
        ("default excludes source true", "/ivf", {"size": 3, "query": {"term": {"n": 3}}, "_source": True}),
        # errors
        ("wrong dimension", "/ivf", {"query": {"knn": {"emb": {"vector": [1.0, 2.0], "k": 3}}}}),
        ("k zero", "/ivf", {"query": knn(k=0)}),
        ("k and min_score", "/ivf", {"query": knn(k=3, min_score=0.5)}),
        ("negative max_distance", "/ivf", {"query": knn(max_distance=-1.0)}),
        ("bad ef_search", "/ivf", {"query": knn(k=3, method_parameters={"ef_search": 0})}),
        ("k over 10000", "/ivf", {"query": knn(k=10001)}),
    ] + [(f"knn error {json.dumps(q)}", "/ivf", {"query": {"knn": {"emb": q}}}) for q in (
        {"vector": vec}, {"vector": [], "k": 3}, {"k": 3}, {"vector": vec, "min_score": 0}, {"vector": vec, "max_distance": -1},
        {"vector": vec, "k": 3, "method_parameters": {"nprobes": 0}}, {"vector": vec, "k": 3, "method_parameters": {"bogus": 1}},
        {"vector": vec, "k": 3, "method_parameters": {"ef_search": 0, "nprobes": 0}}, {"vector": vec, "k": 3, "method_parameters": 5},
        {"vector": vec, "k": 3, "method_parameters": {"ef_search": "x"}}, {"vector": vec, "k": 3, "method_parameters": {}},
        {"vector": vec, "k": 3, "rescore": {"oversample_factor": 1000}}, {"vector": vec, "k": 3, "rescore": {"oversample_factor": 0.5}},
        {"vector": vec, "k": 3, "rescore": {"bogus": 1}}, {"vector": vec, "k": 3, "bogus": 1}, {"vector": "x", "k": 3},
        {"vector": vec, "k": "x"}, {"vector": vec, "k": 3, "filter": 5}, {"vector": vec, "k": 3, "expand_nested_docs": "x"},
        {"vector": ["a"], "k": 3})] + [
        ("knn vector shorthand", "/ivf", {"query": {"knn": {"emb": vec}}}),
        ("knn two fields", "/ivf", {"query": {"knn": {"emb": {"vector": vec, "k": 3}, "cos": {"vector": vec, "k": 3}}}}),
    ]

    def probed(q):
        # Probe every cluster in each knn and neural query of the IVF index; stock's HNSW has no probes.
        if isinstance(q, dict):
            out = {k: probed(v) for k, v in q.items()}
            for kind in ("knn", "neural"):
                if kind in out and isinstance(out[kind], dict):
                    out[kind] = {f: ({**b, "method_parameters": {**b.get("method_parameters", {}), **ALL_PROBES}}
                                     if isinstance(b, dict) and "method_parameters" not in b and "k" in b and b["k"] not in (0, 10001) else b)
                                 for f, b in out[kind].items()}
            return out
        return [probed(x) for x in q] if isinstance(q, list) else q

    ids = mids + sem

    def run(n, mid, path, body, probe=True):
        b = json.loads(json.dumps(body).replace("<mid>", mid))
        if n is ours and probe:
            b = probed(b)
        p = path if "/_search" in path else path + "/_search"
        s, r = n.call("POST", p, b)
        if s != 200:
            return answer((s, r))
        rows = [r["hits"].get("total")]
        for h in r["hits"]["hits"]:
            rows.append((h["_index"], h["_id"], round(h["_score"], 4) if h["_score"] is not None else None,
                         h.get("_source"), h.get("fields"), h.get("_explanation")))
        return json.loads(functools.reduce(lambda t, i: t.replace(i, "<id>"), ids, json.dumps(rows)))

    factories = "cluster.search.enabled_system_generated_factories"
    for n in (ours, stock):
        n.ok("PUT", "/_cluster/settings", {"persistent": {factories: ["*"]}})
    try:
        for label, path, body in cases:
            compare(c, f"ivf {label}", *(run(n, mid, path, body, not label.startswith("knn error")) for n, mid in zip((ours, stock), mids)))
    finally:
        for n in (ours, stock):
            n.ok("PUT", "/_cluster/settings", {"persistent": {factories: None}})
    # Warmup and the k-NN stats on the IVF index.
    compare(c, "ivf warmup", *both(ours, stock, lambda n: answer(n.call("GET", "/_plugins/_knn/warmup/ivf"))))
    compare(c, "ivf clear_cache", *both(ours, stock, lambda n: answer(n.call("POST", "/_plugins/_knn/clear_cache/ivf"))))
    # The calibrated depth, with no nprobes: still most of the exhaustive hits.
    want = {h["_id"] for h in stock.ok("POST", "/ivf/_search", {"size": 10, "query": knn(k=10)})["hits"]["hits"]}
    got = {h["_id"] for h in ours.ok("POST", "/ivf/_search", {"size": 10, "query": knn(k=10)})["hits"]["hits"]}
    c.true("ivf calibrated depth recall", len(want & got) >= 8, f"{len(want & got)}/10")


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
            "mmr": lambda: mmr(c, ours, stock, zips["tiny-bert"]),
            "ivf": lambda: ivf(c, ours, stock, zips["tiny-bert"]),
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
