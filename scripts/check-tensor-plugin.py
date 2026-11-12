#!/usr/bin/env python3
"""multi_vector fields and the max_sim query through the lucene-rust plugin.

Starts one node of the pinned OpenSearch image with the plugin and checks,
on an index the Rust engine writes and on one OpenSearch's own engine writes
(two flushed segments each, some documents without a value, some deleted):

  query      `max_sim` on a field of each encoding (float, float16, int8)
             returns the documents with a value, scored by the MaxSim of the
             query's tokens against the stored ones (the stored tokens are
             rounded here as the encoding rounds them), in score order;
             with a `filter` too.
  rescore    a `match` query's window reranked by `max_sim` through the stock
             `rescore` (query_weight 0): the window's documents in MaxSim
             order, the rest after them.
  candidates `max_sim` with `candidates` on a `pooled` field: about the
             documents whose mean token is nearest the query's (HNSW: most of
             the exact top), each scored by its MaxSim, the filter honoured.
  formula    `script_score` over `max_sim` (its `_score`) with a doc field,
             in Painless.
  native     every one of those shard queries ran in Rust (the plugin's
             stats count no fallback).
  errors     a query of the wrong dimension, `max_sim` on another field
             type, and mappings or documents the field refuses, each a 400.
  source     `_source` keeps the vectors as sent.

  scripts/check-tensor-plugin.py [--keep]

Needs Docker, numpy, and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import glob, json, os, shutil, subprocess, sys, time, urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "3.9.0"
NAME, PORT = "tensor-check-rust", 19215
DIM = 32
DOCS = 400
ENCODINGS = ("float", "float16", "int8")
WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def req(method, path, body=None, ndjson=False, timeout=120, ok=(200, 201)):
    data = None
    if body is not None:
        data = body.encode() if ndjson else json.dumps(body).encode()
    r = urllib.request.Request(f"http://localhost:{PORT}{path}", data=data, method=method,
                               headers={"Content-Type": "application/x-ndjson" if ndjson else "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code in ok:
            return e.code, json.loads(body)
        return e.code, body


def must(method, path, body=None, ndjson=False):
    code, r = req(method, path, body, ndjson)
    if code not in (200, 201):
        raise RuntimeError(f"{method} {path}: {code} {str(r)[:2000]}")
    return r


def start(image, heap="1g"):
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", NAME, "-p", f"{PORT}:9200",
                    "-e", "discovery.type=single-node", "-e", "DISABLE_SECURITY_PLUGIN=true",
                    "-e", "DISABLE_INSTALL_DEMO_CONFIG=true", "-e", f"OPENSEARCH_JAVA_OPTS=-Xms{heap} -Xmx{heap}",
                    "-e", "cluster.routing.allocation.disk.threshold_enabled=false", image],
                   check=True, capture_output=True)
    for _ in range(150):
        try:
            if must("GET", "/_cluster/health")["status"] in ("green", "yellow"):
                return
        except Exception:
            pass
        time.sleep(2)
    sys.exit("node did not start")


def build_image():
    """The pinned OpenSearch image with the built plugin installed."""
    zips = glob.glob(os.path.join(ROOT, "opensearch-plugin", "build", "distributions", "lucene-rust-*.zip"))
    if not zips:
        sys.exit("no plugin zip: run gradle -p opensearch-plugin bundlePlugin")
    docker_dir = os.path.join(ROOT, "opensearch-plugin", "docker")
    for z in glob.glob(os.path.join(docker_dir, "*.zip")):
        os.remove(z)
    shutil.copy(zips[0], docker_dir)
    image = f"lucene-rust-opensearch:{VERSION}"
    subprocess.run(["docker", "build", "-q", "-t", image, docker_dir], check=True, capture_output=True)
    return image


def stop():
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)


def unit(rng, n):
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def stored(tokens, encoding):
    """The tokens as the encoding stores them, widened back to float32."""
    t = np.asarray(tokens, dtype=np.float32)
    if encoding == "float":
        return t
    if encoding == "float16":
        return t.astype(np.float16).astype(np.float32)
    out = np.empty_like(t)
    for i, row in enumerate(t):
        m = np.float32(np.abs(row).max())
        scale = np.float32(max(m / np.float32(127), np.float32(np.finfo(np.float32).smallest_subnormal))) if m > 0 else np.float32(1)
        v = (row / scale).astype(np.float32)
        q = np.clip(np.sign(v) * np.floor(np.abs(v.astype(np.float64)) + 0.5), -127, 127)
        out[i] = q.astype(np.float32) * scale
    return out


def max_sim(query, doc):
    return float(np.max(query.astype(np.float64) @ doc.astype(np.float64).T, axis=1).sum())


def pooled(tokens):
    m = np.asarray(tokens, dtype=np.float64).sum(axis=0)
    n = np.linalg.norm(m)
    return None if n == 0 else (m / n).astype(np.float32)


def close(a, b):
    return abs(a - b) <= 1e-4 * max(1.0, abs(b))


def ranked(got, want, size, what, failures):
    """`got` [(id, score)] is the top `size` of `want` {id: score} in score order, ties aside."""
    order = sorted(want.items(), key=lambda kv: -kv[1])
    if len(got) != min(len(order), size):
        failures.append(f"{what}: {len(got)} hits, expected {min(len(order), size)}")
        return
    for k, (i, s) in enumerate(got):
        if i not in want or not close(s, want[i]):
            failures.append(f"{what}: hit {i} scored {s}, expected {want.get(i)}")
            return
        if not close(s, order[k][1]):
            failures.append(f"{what}: rank {k} scored {s}, expected {order[k][1]}")
            return


def main():
    keep = "--keep" in sys.argv
    start(build_image())
    failures = []
    rng = np.random.default_rng(7)
    docs = []
    for i in range(DOCS):
        d = {"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd",
             "text": " ".join(WORDS[(i * 7 + j) % len(WORDS)] for j in range(1 + i % 4))}
        if i % 9 != 4:
            d["tokens"] = unit(rng, int(rng.integers(1, 60)))
        docs.append(d)
    deleted = {f"d{i}" for i in range(3, DOCS, 37)}
    try:
        before = must("GET", "/_plugins/lucene_rust/stats")
        for index, engine in (("rust", True), ("lucene", False)):
            settings = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
            if engine:
                settings["index.engine.accelerated"] = True
            props = {"tag": {"type": "keyword"}, "text": {"type": "text"}, "rank": {"type": "long"},
                     "o": {"properties": {"mv": {"type": "multi_vector", "dimension": DIM}}}}
            for e in ENCODINGS:
                props[f"mv_{e}"] = {"type": "multi_vector", "dimension": DIM, "encoding": e}
            props["mv_pooled"] = {"type": "multi_vector", "dimension": DIM, "encoding": "float", "pooled": True}
            must("PUT", f"/{index}", {"settings": settings, "mappings": {"properties": props}})
            half = len(docs) // 2
            for part in (docs[:half], docs[half:]):
                lines = []
                for d in part:
                    src = {"tag": d["tag"], "text": d["text"], "rank": int(d["_id"][1:]) % 10}
                    if "tokens" in d:
                        for e in ENCODINGS:
                            src[f"mv_{e}"] = d["tokens"].tolist()
                        src["mv_pooled"] = d["tokens"].tolist()
                    lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
                    lines.append(json.dumps(src))
                r = must("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
                if r["errors"]:
                    errs = [it["index"] for it in r["items"] if it["index"].get("error")]
                    failures.append(f"{index}: {len(errs)} bulk errors, first {json.dumps(errs[0])[:1500]}")
                    raise RuntimeError(failures[-1])
                must("POST", f"/{index}/_flush")
            for i in sorted(deleted):
                must("DELETE", f"/{index}/_doc/{i}")
            must("POST", f"/{index}/_refresh")

        live = [d for d in docs if d["_id"] not in deleted]
        queries = [unit(rng, int(rng.integers(1, 33))) for _ in range(8)]
        cases = 0
        for index in ("rust", "lucene"):
            for e in ENCODINGS:
                field = f"mv_{e}"
                for qi, q in enumerate(queries):
                    want = {d["_id"]: max(0.0, max_sim(q, stored(d["tokens"], e))) for d in live if "tokens" in d}
                    body = {"size": 50, "query": {"max_sim": {field: {"query_vectors": q.tolist()}}}}
                    r = must("POST", f"/{index}/_search", body)
                    if r["hits"]["total"]["value"] != len(want):
                        failures.append(f"{index} {field} q{qi}: total {r['hits']['total']['value']}, expected {len(want)}")
                    ranked([(h["_id"], h["_score"]) for h in r["hits"]["hits"]], want, 50, f"{index} {field} q{qi}", failures)
                    # filtered
                    odd = {i: s for i, s in want.items() if int(i[1:]) % 2 == 1}
                    body = {"size": 20, "query": {"max_sim": {field: {"query_vectors": q.tolist(), "filter": {"term": {"tag": "odd"}}}}}}
                    r = must("POST", f"/{index}/_search", body)
                    if r["hits"]["total"]["value"] != len(odd):
                        failures.append(f"{index} {field} q{qi} filtered: total {r['hits']['total']['value']}, expected {len(odd)}")
                    ranked([(h["_id"], h["_score"]) for h in r["hits"]["hits"]], odd, 20, f"{index} {field} q{qi} filtered", failures)
                    # rescore: a match query's window of 30, reranked by max_sim alone
                    word = WORDS[qi % len(WORDS)]
                    body = {"size": 40, "query": {"match": {"text": word}},
                            "rescore": {"window_size": 30, "query": {
                                "rescore_query": {"max_sim": {field: {"query_vectors": q.tolist()}}},
                                "query_weight": 0.0, "rescore_query_weight": 1.0}}}
                    # The window's last BM25 score is shared by more documents than fit, so which
                    # of those it holds is the shard's choice: check that each reranked hit
                    # matches, scores its MaxSim (or 0: outside the window, query_weight 0 zeroes
                    # it), and comes in score order, the window's documents first.
                    matched = {d["_id"] for d in live if word in d["text"].split()}
                    r = must("POST", f"/{index}/_search", body)
                    got = [(h["_id"], h["_score"]) for h in r["hits"]["hits"]][: min(30, len(matched))]
                    what = f"{index} {field} q{qi} rescore"
                    if len(got) != min(30, len(matched)):
                        failures.append(f"{what}: {len(got)} hits")
                    if sum(1 for _, sc in got if sc > 0) < min(10, len(got)):
                        failures.append(f"{what}: only {sum(1 for _, sc in got if sc > 0)} hits reranked")
                    for k, (i, sc) in enumerate(got):
                        # With every match in the window, a 0 is only right where MaxSim is not positive.
                        inside = len(matched) <= 30 or sc != 0.0
                        if i not in matched or (inside and not close(sc, max(want.get(i, 0.0), 0.0))) or (k and sc > got[k - 1][1]):
                            failures.append(f"{what}: hit {k} {i} scored {sc}, expected {want.get(i, 0.0)}")
                            break
                    cases += 3
        print(f"query/filter/rescore: {cases} searches checked")

        # candidates: the nearest by pooled mean, scored by MaxSim
        recalls = []
        for index in ("rust", "lucene"):
            for qi, q in enumerate(queries):
                pq = pooled(q)
                for flt in (None, "odd"):
                    pool = [d for d in live if "tokens" in d and (flt is None or d["tag"] == flt)]
                    near = sorted(pool, key=lambda d: -float(pooled(d["tokens"]).astype(np.float64) @ pq))[:20]
                    inner = {"query_vectors": q.tolist(), "candidates": 20}
                    if flt:
                        inner["filter"] = {"term": {"tag": flt}}
                    r = must("POST", f"/{index}/_search", {"size": 40, "query": {"max_sim": {"mv_pooled": inner}}})
                    hits = r["hits"]["hits"]
                    what = f"{index} candidates q{qi} {flt}"
                    want = {d["_id"]: max(0.0, max_sim(q, d["tokens"])) for d in pool}
                    if len(hits) != 20:
                        failures.append(f"{what}: {len(hits)} hits")
                    for k, h in enumerate(hits):
                        if h["_id"] not in want or not close(h["_score"], want[h["_id"]]) or (k and h["_score"] > hits[k - 1]["_score"]):
                            failures.append(f"{what}: hit {k} {h['_id']} scored {h['_score']}, expected {want.get(h['_id'])}")
                            break
                    recalls.append(len({h["_id"] for h in hits} & {d["_id"] for d in near}) / 20)
        # a ranking formula over max_sim: script_score's _score is the MaxSim, run by native Painless
        for index in ("rust", "lucene"):
            for qi, q in enumerate(queries[:4]):
                body = {"size": 30, "query": {"script_score": {"query": {"max_sim": {"mv_float": {"query_vectors": q.tolist()}}},
                        "script": {"source": "_score * params.w + doc['rank'].value", "params": {"w": 2.0}}}}}
                r = must("POST", f"/{index}/_search", body)
                want = {d["_id"]: 2.0 * max(0.0, max_sim(q, d["tokens"])) + int(d["_id"][1:]) % 10 for d in live if "tokens" in d}
                ranked([(h["_id"], h["_score"]) for h in r["hits"]["hits"]], want, 30, f"{index} script_score q{qi}", failures)
        print(f"candidates: {len(recalls)} searches, mean recall of the exact pooled top 20 {np.mean(recalls):.3f}")
        if np.mean(recalls) < 0.9:
            failures.append(f"candidates: mean recall {np.mean(recalls):.3f} below 0.9")

        # _source as sent
        d = next(d for d in live if "tokens" in d)
        src = must("GET", f"/rust/_doc/{d['_id']}")["_source"]
        if not np.array_equal(np.asarray(src["mv_int8"], dtype=np.float32), d["tokens"]):
            failures.append("_source does not keep the vectors as sent")

        after = must("GET", "/_plugins/lucene_rust/stats")
        ran = after["accelerated_queries"] - before["accelerated_queries"]
        fallbacks = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
                     if v != before.get("fallbacks", {}).get(k, 0)}
        print(f"plugin: {ran} shard queries in Rust, fallbacks {fallbacks}")
        if fallbacks:
            failures.append(f"queries fell back to Java: {fallbacks}")
        if ran < cases:
            failures.append(f"only {ran} of {cases} searches ran in Rust")

        # errors
        bad = [
            ("POST", "/rust/_search", {"query": {"max_sim": {"mv_float": {"query_vectors": [[1.0, 2.0]]}}}}, "dimension"),
            ("POST", "/rust/_search", {"query": {"max_sim": {"text": {"query_vectors": [[1.0] * DIM]}}}}, "multi_vector"),
            ("POST", "/rust/_search", {"query": {"max_sim": {"mv_float": {"query_vectors": []}}}}, "query_vectors"),
            ("POST", "/rust/_search", {"query": {"max_sim": {"mv_float": {"query_vectors": [[1.0] * DIM], "candidates": 5}}}}, "pooled"),
            ("POST", "/rust/_search", {"query": {"max_sim": {"mv_pooled": {"query_vectors": [[1.0] * DIM], "candidates": 0}}}}, "candidates"),
            ("POST", "/rust/_search", {"query": {"max_sim": {"mv_pooled": {"query_vectors": [[1.0] * DIM], "candidates": 2.5}}}}, "whole number"),
            ("POST", "/rust/_search", {"query": {"max_sim": {"mv_pooled": {"query_vectors": [[1.0] * DIM], "candidates": 1e10}}}}, "whole number"),
            ("POST", "/rust/_search", {"query": {"max_sim": {"mv_pooled": {"query_vectors": [[1.0] * DIM], "candidates": "5"}}}}, "whole number"),
            ("PUT", "/bad3", {"mappings": {"properties": {"v": {"type": "multi_vector", "dimension": 2048, "pooled": True}}}}, "1024"),
            ("PUT", "/bad4", {"mappings": {"properties": {"v": {"type": "multi_vector", "dimension": 4, "pooled": True,
                                                                "fields": {"_pooled": {"type": "keyword"}}}}}}, "multi-fields"),
            ("PUT", "/bad1", {"mappings": {"properties": {"v": {"type": "multi_vector"}}}}, "dimension"),
            ("PUT", "/bad2", {"mappings": {"properties": {"v": {"type": "multi_vector", "dimension": 4, "encoding": "int4"}}}}, "encoding"),
            ("PUT", "/rust/_doc/x1", {"mv_float": [[1.0, 2.0]]}, "dimension mismatch"),
            ("PUT", "/rust/_doc/x2", {"mv_float": []}, "at least one"),
            ("PUT", "/rust/_doc/x3", {"mv_float16": [[1e6] * DIM]}, "65504"),
            ("PUT", "/rust/_doc/x4", {"mv_float": [[1e39] * DIM]}, "finite"),
            ("PUT", "/rust/_doc/x5", {"o": [{"mv": [[1.0] * DIM]}, {"mv": [[1.0] * DIM]}]}, "one value per document"),
            ("PUT", "/rust/_mapping", {"properties": {"mv_float": {"type": "multi_vector", "dimension": DIM + 1, "encoding": "float"}}}, "dimension"),
            ("PUT", "/rust/_mapping", {"properties": {"mv_float": {"type": "multi_vector", "dimension": DIM, "encoding": "int8"}}}, "encoding"),
        ]
        for method, path, body, text in bad:
            code, r = req(method, path, body)
            if code != 400 or text not in str(r):
                failures.append(f"{method} {path}: expected a 400 naming {text!r}, got {code} {str(r)[:300]}")
        print(f"errors: {len(bad)} refusals checked")
        must("PUT", "/plain", {"mappings": {"properties": {"tag": {"type": "keyword"}}}})
        must("PUT", "/plain/_doc/p1?refresh=true", {"tag": "x"})
        r = must("POST", "/plain,rust/_search", {"query": {"max_sim": {"mv_float": {"query_vectors": [[1.0] * DIM]}}}})
        if r["_shards"]["failed"] or r["hits"]["total"]["value"] != sum(1 for d in live if "tokens" in d):
            failures.append(f"an index without the field: {json.dumps(r)[:500]}")
    finally:
        if failures:
            logs = subprocess.run(["docker", "logs", NAME], capture_output=True, text=True)
            print("\n".join(l for l in (logs.stdout + logs.stderr).splitlines() if "WARN" in l or "ERROR" in l)[-5000:])
        if not keep:
            stop()
    for f in failures[:40]:
        print("FAIL:", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
