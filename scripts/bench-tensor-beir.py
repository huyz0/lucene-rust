#!/usr/bin/env python3
"""ColBERT retrieval through the lucene-rust plugin on a BEIR dataset: quality and latency.

Indexes the corpus (its text, and its ColBERT token vectors in a `multi_vector`
field with `pooled`) into one shard the Rust engine writes, then runs every
test query through each first pass and the `max_sim` rerank:

  bm25              `match` on the text alone
  bm25>maxsim       `match`, its top `window` reranked by `max_sim`
  pooled            `max_sim` with `candidates` (nearest pooled means), MaxSim-scored
  hybrid            `match` OR `max_sim` candidates, the top 2x`window` reranked by `max_sim`
  exact             `max_sim` over every document

and prints nDCG@10 and recall@100 against the qrels, with client latency p50 / p95.

  scripts/bench-tensor-beir.py <data dir> <dataset> [--windows=100,300] [--only=<pipeline prefix>] [--keep]

<data dir> holds `<dataset>.docs`, `<dataset>.queries` and `<dataset>.qrels`
(encode.py's mvbench files) and `<dataset>/corpus.jsonl`, `<dataset>/queries.jsonl`
(the BEIR download). Needs Docker, numpy, and the plugin zip.
"""
import json, math, os, struct, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

check = importlib.import_module("check-tensor-plugin")


def read_vectors(path):
    with open(path, "rb") as f:
        n, dim = struct.unpack("<II", f.read(8))
        out = []
        for _ in range(n):
            (t,) = struct.unpack("<I", f.read(4))
            out.append(np.frombuffer(f.read(4 * t * dim), dtype="<f4").reshape(t, dim))
    return out, dim


def ndcg10(ranked, rel):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(rel))))
    return dcg / ideal if ideal else 0.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data, ds = args[0], args[1]
    windows = [int(w) for w in next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--windows=")), "100,300").split(",")]
    keep = "--keep" in sys.argv
    docs, dim = read_vectors(f"{data}/{ds}.docs")
    queries, _ = read_vectors(f"{data}/{ds}.queries")
    corpus = [json.loads(l) for l in open(f"{data}/{ds}/corpus.jsonl")]
    texts = [(d.get("title", "") + " " + d["text"]).strip() for d in corpus]
    rels = {}
    for line in open(f"{data}/{ds}.qrels"):
        q, d = line.split()
        rels.setdefault(int(q), set()).add(f"d{d}")
    # encode.py kept the queries with qrels, in queries.jsonl's order.
    qids = set()
    import csv
    with open(f"{data}/{ds}/qrels/test.tsv") as f:
        rd = csv.reader(f, delimiter="\t")
        next(rd)
        for q, d, s in rd:
            if int(s) > 0:
                qids.add(q)
    qtexts = [q["text"] for q in (json.loads(l) for l in open(f"{data}/{ds}/queries.jsonl")) if q["_id"] in qids]
    assert len(qtexts) == len(queries), (len(qtexts), len(queries))

    image = check.build_image()
    check.start(image, heap="4g")
    try:
        must = check.must
        must("PUT", "/beir", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1,
                                            "index.engine.accelerated": True},
                               # The token vectors stay out of _source: each hit's fetch otherwise
                               # reads its whole stored _source, every token's numbers as JSON.
                               "mappings": {"_source": {"excludes": ["colbert"]},
                                            "properties": {"text": {"type": "text"},
                                                           "colbert": {"type": "multi_vector", "dimension": dim,
                                                                       "encoding": "float16", "pooled": True}}}})
        t0 = time.time()
        for s in range(0, len(docs), 100):
            lines = []
            for i in range(s, min(s + 100, len(docs))):
                lines.append(json.dumps({"index": {"_index": "beir", "_id": f"d{i}"}}))
                lines.append(json.dumps({"text": texts[i], "colbert": np.round(docs[i], 5).tolist()}))
            r = must("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
            if r["errors"]:
                sys.exit(f"bulk errors: {json.dumps(r)[:1500]}")
        must("POST", "/beir/_refresh")
        must("POST", "/beir/_forcemerge?max_num_segments=1")
        must("POST", "/beir/_refresh")
        print(f"# {ds}: {len(docs)} docs, {sum(len(d) for d in docs)} tokens, {len(queries)} queries, dim {dim}; indexed in {time.time() - t0:.0f} s")
        before = must("GET", "/_plugins/lucene_rust/stats")

        def maxsim(q, **extra):
            return {"max_sim": {"colbert": {"query_vectors": q, **extra}}}

        def rescore(q, window):
            return {"window_size": window, "query": {"rescore_query": maxsim(q), "query_weight": 0.0, "rescore_query_weight": 1.0}}

        pipelines = [("bm25", lambda t, q: {"query": {"match": {"text": t}}}),
                     ("exact", lambda t, q: {"query": maxsim(q)})]
        for w in windows:
            pipelines.append((f"bm25>maxsim w{w}", lambda t, q, w=w: {"query": {"match": {"text": t}}, "rescore": rescore(q, w)}))
            pipelines.append((f"pooled c{w}", lambda t, q, w=w: {"query": maxsim(q, candidates=w)}))
            pipelines.append((f"hybrid c{w} w{2 * w}", lambda t, q, w=w: {
                "query": {"bool": {"should": [{"match": {"text": t}}, maxsim(q, candidates=w)]}}, "rescore": rescore(q, 2 * w)}))
        print("pipeline,ndcg@10,recall@100,p50_ms,p95_ms,took_p50_ms")
        only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
        for name, body in pipelines:
            if only and not name.startswith(only):
                continue
            nd, rc, lat, took = [], [], [], []
            for rep in range(2):
                for qi, (t, q) in enumerate(zip(qtexts, queries)):
                    b = body(t, np.round(q, 5).tolist())
                    b["size"] = 100
                    b["_source"] = False
                    s = time.perf_counter()
                    r = must("POST", "/beir/_search", b)
                    e = (time.perf_counter() - s) * 1000
                    if rep == 0:
                        continue  # warm-up pass
                    lat.append(e)
                    took.append(r["took"])
                    ids = [h["_id"] for h in r["hits"]["hits"]]
                    rel = rels.get(qi, set())
                    nd.append(ndcg10(ids, rel))
                    rc.append(len(set(ids) & rel) / len(rel) if rel else 0.0)
            print(f"{name},{np.mean(nd):.4f},{np.mean(rc):.4f},{np.percentile(lat, 50):.2f},{np.percentile(lat, 95):.2f},{np.percentile(took, 50):.0f}")
        after = must("GET", "/_plugins/lucene_rust/stats")
        fb = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
              if v != before.get("fallbacks", {}).get(k, 0)}
        print(f"# shard queries in Rust {after['accelerated_queries'] - before['accelerated_queries']}, fallbacks {fb}")
    finally:
        if not keep:
            check.stop()


if __name__ == "__main__":
    main()
