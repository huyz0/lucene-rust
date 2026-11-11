#!/usr/bin/env python3
"""Neural search through the lucene-rust plugin against stock: ingest and query latency.

Uses the nodes scripts/check-neural-plugin.py --keep leaves running (the
plugin on 19241, stock on 19242), the fixtures/ml tiny-bert text embedding
and tiny-sparse encoding models. Per node, one at a time:

  ingest     DOCS documents through a text_embedding pipeline into a
             knn_vector field and through a sparse_encoding pipeline into a
             rank_features field, in bulks of 100: documents per second
  neural     QUERIES neural queries (k 10), one at a time: p50 / p99 ms
  sparse     neural_sparse by query_text with the model, and with the default
             bert-uncased analyzer: p50 / p99 ms
  rerank     a match query's top 10 reranked by the tiny cross-encoder through
             the rerank response processor, and the same query unreranked:
             p50 / p99 ms

  scripts/bench-neural-plugin.py [--docs 4000] [--queries 400]
"""
import argparse, importlib.util, json, os, random, statistics, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("check_neural", os.path.join(ROOT, "scripts", "check-neural-plugin.py"))
cn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cn)
ml = cn.ml
golden = cn.golden

WORDS = ("search documents vectors sparse dense hybrid model query index shard segment token weight "
         "neural retrieval ranking score cluster embedding text field engine opensearch lucene rust").split()


def text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def timed(fn, n):
    out = []
    for i in range(n):
        t = time.perf_counter()
        fn(i)
        out.append((time.perf_counter() - t) * 1000)
    out.sort()
    return statistics.median(out), out[int(len(out) * 0.99) - 1]


def model_id(n, prefix):
    r = n.ok("POST", "/_plugins/_ml/models/_search", {"size": 1, "query": {"bool": {"must": [
        {"term": {"model_state": "DEPLOYED"}}, {"wildcard": {"name.keyword": f"{prefix}*"}}]}}})
    return r["hits"]["hits"][0]["_id"]


def bench(n, docs, queries):
    rng = random.Random(7)
    dense, sparse = model_id(n, "tiny-bert-"), model_id(n, "tiny-sparse-")
    cross = model_id(n, golden.CROSS)
    n.call("DELETE", "/bench")
    n.ok("PUT", "/_ingest/pipeline/bench-p", {"processors": [
        {"text_embedding": {"model_id": dense, "field_map": {"text": "emb"}}},
        {"sparse_encoding": {"model_id": sparse, "field_map": {"text": "feat"}}}]})
    n.ok("PUT", "/_search/pipeline/bench-rr", {"response_processors": [{"rerank": {
        "ml_opensearch": {"model_id": cross}, "context": {"document_fields": ["text"]}}}]})
    n.ok("PUT", "/bench", {"settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0,
                                        "default_pipeline": "bench-p", "refresh_interval": -1},
                           "mappings": {"properties": {"text": {"type": "text"}, "feat": {"type": "rank_features"},
                                                       "emb": {"type": "knn_vector", "dimension": 32}}}})
    all_docs = [{"_id": str(i), "text": text(rng, rng.randint(8, 40))} for i in range(docs)]
    t = time.perf_counter()
    for i in range(0, docs, 100):
        cn.bulk(n, "bench", all_docs[i:i + 100])
    ingest = docs / (time.perf_counter() - t)
    n.ok("POST", "/bench/_refresh")
    n.ok("POST", "/bench/_forcemerge?max_num_segments=1")
    qs = [text(rng, rng.randint(2, 6)) for _ in range(queries)]

    def run(body, pipeline=None):
        s, r = n.call("POST", "/bench/_search" + (f"?search_pipeline={pipeline}" if pipeline else ""), body)
        assert s == 200, json.dumps(r)[:500]

    for q in qs[:20]:
        run({"query": {"neural": {"emb": {"query_text": q, "model_id": dense, "k": 10}}}})
    out = {"ingest docs/s": round(ingest, 1)}
    out["neural p50/p99 ms"] = timed(lambda i: run({"query": {"neural": {"emb": {"query_text": qs[i], "model_id": dense, "k": 10}}}}), queries)
    out["neural_sparse model p50/p99 ms"] = timed(lambda i: run({"query": {"neural_sparse": {"feat": {"query_text": qs[i], "model_id": sparse}}}}), queries)
    out["neural_sparse analyzer p50/p99 ms"] = timed(lambda i: run({"query": {"neural_sparse": {"feat": {"query_text": qs[i]}}}}), queries)
    match = lambda i: {"query": {"match": {"text": qs[i]}}, "size": 10}
    out["match p50/p99 ms"] = timed(lambda i: run(match(i)), queries)
    out["rerank p50/p99 ms"] = timed(lambda i: run({**match(i), "ext": {"rerank": {"query_context": {"query_text": qs[i]}}}}, "bench-rr"), queries)
    return {k: (tuple(round(x, 2) for x in v) if isinstance(v, tuple) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=4000)
    ap.add_argument("--queries", type=int, default=400)
    a = ap.parse_args()
    for label, port in (("plugin", cn.OURS[1]), ("stock", cn.STOCK[1])):
        print(label, json.dumps(bench(ml.Node(f"http://localhost:{port}"), a.docs, a.queries)))


if __name__ == "__main__":
    main()
