#!/usr/bin/env python3
"""Writes the Seismic sparse fixture: indexes made by OpenSearch's neural-search
plugin (its `lucene` engine), plus the hits neural_sparse queries return on them.

Needs Docker. Runs opensearchproject/opensearch:VERSION, indexes seeded random
sparse vectors into two indexes (two flushed segments; the same docs force-merged
to one), runs a set of queries, then copies each shard's Lucene directory to
fixtures/data/seismic/<index>/ and the queries with their hits to
fixtures/data/seismic/expected.json.

  scripts/gen-seismic-fixture.py            # VERSION defaults to 3.9.0
"""
import json, os, random, shutil, subprocess, sys, time, urllib.request

VERSION = os.environ.get("VERSION", "3.9.0")
PORT = 19201
NAME = "seismic-fixture"
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures", "data", "seismic")
DOCS, VOCAB, SEED = 1000, 3000, 20261104
PARAMS = {"approximate_threshold": 100, "n_postings": 40, "cluster_ratio": 0.1,
          "summary_prune_ratio": 0.4, "quantization_ceiling_ingest": 3.0,
          "quantization_ceiling_search": 16.0}


def req(method, path, body=None, ndjson=False):
    data = None
    if body is not None:
        data = body.encode() if ndjson else json.dumps(body).encode()
    r = urllib.request.Request(f"http://localhost:{PORT}{path}", data=data, method=method,
                               headers={"Content-Type": "application/x-ndjson" if ndjson else "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read() or b"null")


def vector(rng, nnz):
    v = {}
    while len(v) < nnz:
        # Zipf-ish token ids so some posting lists are long.
        t = min(int(rng.paretovariate(0.8)) - 1, VOCAB - 1)
        v[str(t)] = round(rng.uniform(0.01, 3.5), 4)
    return v


def main():
    rng = random.Random(SEED)
    docs = [{"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd",
             "emb": vector(rng, rng.randint(10, 60))} for i in range(DOCS)]
    queries = [vector(rng, rng.randint(5, 40)) for _ in range(12)]

    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", NAME, "-p", f"{PORT}:9200",
                    "-e", "discovery.type=single-node", "-e", "DISABLE_SECURITY_PLUGIN=true",
                    "-e", "DISABLE_INSTALL_DEMO_CONFIG=true", "-e", "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g",
                    f"opensearchproject/opensearch:{VERSION}"], check=True)
    for _ in range(120):
        try:
            if req("GET", "/_cluster/health")["status"] in ("green", "yellow"):
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        sys.exit("node did not start")

    mapping = {"properties": {"tag": {"type": "keyword"},
               "emb": {"type": "sparse_vector", "method": {"name": "seismic", "parameters": PARAMS}}}}
    settings = {"index.sparse": True, "number_of_shards": 1, "number_of_replicas": 0,
                "refresh_interval": -1}
    expected = {"version": VERSION, "params": PARAMS, "indexes": {}, "queries": []}
    for index, merge in (("two_segments", False), ("merged", True)):
        req("PUT", f"/{index}", {"settings": settings, "mappings": mapping})
        for half in (docs[: DOCS // 2], docs[DOCS // 2:]):
            lines = []
            for d in half:
                lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
                lines.append(json.dumps({"tag": d["tag"], "emb": d["emb"]}))
            r = req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
            assert not r["errors"], r
            req("POST", f"/{index}/_flush")
        if merge:
            req("POST", f"/{index}/_forcemerge?max_num_segments=1")
        req("POST", f"/{index}/_refresh")
        segs = req("GET", f"/{index}/_segments")["indices"][index]["shards"]["0"][0]["segments"]
        expected["indexes"][index] = {"segments": {k: v["num_docs"] for k, v in segs.items()}}

    variants = [{"top_n": 10, "k": 10, "heap_factor": 1.0},
                {"top_n": 20, "k": 50, "heap_factor": 0.8},
                {"top_n": 5, "k": 20, "heap_factor": 1.5},
                {"top_n": 10, "k": 10, "heap_factor": 1.0, "filter": {"term": {"tag": "even"}}},
                {"top_n": 10, "k": 1000, "heap_factor": 1.0, "filter": {"term": {"tag": "odd"}}}]
    for qi, q in enumerate(queries):
        for vi, mp in enumerate(variants):
            entry = {"query_tokens": q, "method_parameters": mp, "hits": {}}
            for index in expected["indexes"]:
                body = {"size": 2000, "query": {"neural_sparse": {"emb": {
                    "query_tokens": q, "method_parameters": mp}}}}
                r = req("POST", f"/{index}/_search", body)
                entry["hits"][index] = [[h["_id"], h["_score"]] for h in r["hits"]["hits"]]
            expected["queries"].append(entry)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for index in expected["indexes"]:
        uuid = req("GET", f"/{index}/_settings")[index]["settings"]["index"]["uuid"]
        req("POST", f"/{index}/_flush")
        src = f"/usr/share/opensearch/data/nodes/0/indices/{uuid}/0/index"
        subprocess.run(["docker", "cp", f"{NAME}:{src}", os.path.join(OUT, index)], check=True)
        for f in os.listdir(os.path.join(OUT, index)):
            if f == "write.lock":
                os.remove(os.path.join(OUT, index, f))
    with open(os.path.join(OUT, "expected.json"), "w") as f:
        json.dump(expected, f, indent=1, sort_keys=True)
        f.write("\n")
    write_tsv(expected, os.path.join(OUT, "expected.tsv"))
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)


def write_tsv(expected, path):
    """expected.json flattened for the Rust test, which has no JSON parser."""
    with open(path, "w") as f:
        for q in expected["queries"]:
            mp = q["method_parameters"]
            filt = mp.get("filter", {}).get("term", {}).get("tag", "-")
            f.write(f"query\t{mp['top_n']}\t{mp['k']}\t{mp['heap_factor']!r}\t{filt}\n")
            f.write("tokens\t" + ",".join(f"{t}:{w!r}" for t, w in q["query_tokens"].items()) + "\n")
            for index, hits in sorted(q["hits"].items()):
                f.write(f"hits\t{index}\t" + ",".join(f"{i}:{s!r}" for i, s in hits) + "\n")


if __name__ == "__main__":
    main()
