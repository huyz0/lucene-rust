#!/usr/bin/env python3
"""neural_sparse through the lucene-rust plugin against stock neural-search.

Starts two nodes of the pinned OpenSearch image: one with the lucene-rust
plugin (built by scripts/verify-opensearch.sh's steps; neural-search removed,
the two cannot share a node) and a stock one, which bundles neural-search.
Both get the same `index.sparse` indexes, written in two flushed segments,
with the documents scripts/gen-seismic-fixture.py generates, and answer the
same queries. The Rust engine writes the plugin node's segments.

  exact      clusters never pruned (`n_postings` past every posting list,
             whole summaries, a huge heap factor): every document sharing a
             walked token is scored, so both nodes must return the same hits
             with the same scores, filtered or not.
  fallback   `approximate_threshold` past the segment size: every segment runs
             the sum of linear FeatureField queries; same hits and scores.
  features   neural_sparse on a `rank_features` field; same hits and scores.
  approx     the fixture's parameters. Clusters come from random centres,
             which the two writers draw differently, so the hits may differ:
             the check is that the plugin's top 10 overlap stock's by at least
             MIN_RECALL on average, and that every query ran in Rust.

  scripts/check-sparse-plugin.py [--keep]

Needs Docker and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import glob, importlib.util, json, os, shutil, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "3.9.0"
OURS, STOCK = ("sparse-check-rust", 19211), ("sparse-check-stock", 19212)
MIN_RECALL = 0.8

spec = importlib.util.spec_from_file_location("fixture", os.path.join(ROOT, "scripts", "gen-seismic-fixture.py"))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

EXACT = {"approximate_threshold": 100, "n_postings": 100000, "cluster_ratio": 0.1, "summary_prune_ratio": 1.0}
FALLBACK = {"approximate_threshold": 100000}


def req(port, method, path, body=None, ndjson=False):
    data = None
    if body is not None:
        data = body.encode() if ndjson else json.dumps(body).encode()
    r = urllib.request.Request(f"http://localhost:{port}{path}", data=data, method=method,
                               headers={"Content-Type": "application/x-ndjson" if ndjson else "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:2000]}") from None


def start(name, port, image, env=()):
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    args = ["docker", "run", "-d", "--name", name, "-p", f"{port}:9200",
            "-e", "discovery.type=single-node", "-e", "DISABLE_SECURITY_PLUGIN=true",
            "-e", "DISABLE_INSTALL_DEMO_CONFIG=true", "-e", "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g",
            "-e", "cluster.routing.allocation.disk.threshold_enabled=false"]
    for e in env:
        args += ["-e", e]
    subprocess.run(args + [image], check=True, capture_output=True)


def wait(port):
    for _ in range(150):
        try:
            if req(port, "GET", "/_cluster/health")["status"] in ("green", "yellow"):
                return
        except Exception:
            pass
        time.sleep(2)
    sys.exit(f"node on {port} did not start")


def create(port, index, params, docs, rust):
    settings = {"index.sparse": True, "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    if rust:
        settings["index.lucene_rust.engine"] = True
    mapping = {"properties": {"tag": {"type": "keyword"}, "feat": {"type": "rank_features"},
               "emb": {"type": "sparse_vector", "method": {"name": "seismic", "parameters": params}}}}
    req(port, "PUT", f"/{index}", {"settings": settings, "mappings": mapping})
    half = len(docs) // 2
    for part in (docs[:half], docs[half:]):
        lines = []
        for d in part:
            lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
            lines.append(json.dumps({"tag": d["tag"], "emb": d["emb"], "feat": d["emb"]}))
        r = req(port, "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        assert not r["errors"], json.dumps(r)[:2000]
        req(port, "POST", f"/{index}/_flush")
    req(port, "POST", f"/{index}/_refresh")
    segs = req(port, "GET", f"/{index}/_segments")["indices"][index]["shards"]["0"][0]["segments"]
    return sorted(v["num_docs"] for v in segs.values())


def search(port, index, field, tokens, mp):
    q = {"query_tokens": tokens}
    if mp is not None:
        q["method_parameters"] = mp
    r = req(port, "POST", f"/{index}/_search", {"size": 2000, "query": {"neural_sparse": {field: q}}})
    return [(h["_id"], h["_score"]) for h in r["hits"]["hits"]]


def main():
    keep = "--keep" in sys.argv
    zips = glob.glob(os.path.join(ROOT, "opensearch-plugin", "build", "distributions", "lucene-rust-*.zip"))
    if not zips:
        sys.exit("no plugin zip: run gradle -p opensearch-plugin bundlePlugin")
    docker_dir = os.path.join(ROOT, "opensearch-plugin", "docker")
    for z in glob.glob(os.path.join(docker_dir, "*.zip")):
        os.remove(z)
    shutil.copy(zips[0], docker_dir)
    image = f"lucene-rust-opensearch:{VERSION}"
    subprocess.run(["docker", "build", "-q", "-t", image, docker_dir], check=True, capture_output=True)
    start(OURS[0], OURS[1], image)
    start(STOCK[0], STOCK[1], f"opensearchproject/opensearch:{VERSION}")
    failures = []
    try:
        wait(OURS[1])
        wait(STOCK[1])
        rng = fixture.random.Random(fixture.SEED)
        docs = [{"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd",
                 "emb": fixture.vector(rng, rng.randint(10, 60))} for i in range(fixture.DOCS)]
        queries = [fixture.vector(rng, rng.randint(5, 40)) for _ in range(12)]
        for index, params in (("exact", EXACT), ("fallback", FALLBACK), ("approx", fixture.PARAMS)):
            ours = create(OURS[1], index, params, docs, True)
            stock = create(STOCK[1], index, params, docs, False)
            print(f"{index}: segments {ours} (plugin) {stock} (stock)")
            if ours != stock:
                failures.append(f"{index}: segment sizes differ")

        before = req(OURS[1], "GET", "/_plugins/lucene_rust/stats")
        exact_params = [{"top_n": 1000, "k": 10, "heap_factor": 1e30},
                        {"top_n": 1000, "k": 50, "heap_factor": 1e30, "filter": {"term": {"tag": "even"}}},
                        {"top_n": 1000, "k": 3, "heap_factor": 1e30, "filter": {"term": {"tag": "odd"}}}]
        cases = [("exact", "emb", mp) for mp in exact_params]
        cases += [("fallback", "emb", None), ("fallback", "emb", {"filter": {"term": {"tag": "odd"}}}),
                  ("exact", "feat", None)]
        checked = 0
        for index, field, mp in cases:
            for qi, tokens in enumerate(queries):
                got = search(OURS[1], index, field, tokens, mp)
                want = search(STOCK[1], index, field, tokens, mp)
                if got != want:
                    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                    failures.append(f"{index} {field} {mp} query {qi}: {len(got)} vs {len(want)} hits, "
                                    f"first difference at {first}: {got[first:first + 2]} vs {want[first:first + 2]}")
                checked += 1
        print(f"same hits and scores on {checked - len([f for f in failures if 'query' in f])} of {checked} queries")

        recalls = []
        for mp in ({"top_n": 10, "k": 10, "heap_factor": 1.0}, {"top_n": 20, "k": 50, "heap_factor": 0.8}):
            for tokens in queries:
                got = {i for i, _ in search(OURS[1], "approx", "emb", tokens, mp)[:10]}
                want = {i for i, _ in search(STOCK[1], "approx", "emb", tokens, mp)[:10]}
                recalls.append(len(got & want) / max(len(want), 1))
        recall = sum(recalls) / len(recalls)
        print(f"approx: plugin's top 10 overlap stock's by {recall:.3f} on average")
        if recall < MIN_RECALL:
            failures.append(f"approx: overlap {recall:.3f} < {MIN_RECALL}")

        after = req(OURS[1], "GET", "/_plugins/lucene_rust/stats")
        ran = after["native_queries"] - before["native_queries"]
        fallbacks = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
                     if v != before.get("fallbacks", {}).get(k, 0)}
        print(f"plugin: {ran} shard queries in Rust, fallbacks {fallbacks}")
        if fallbacks:
            failures.append(f"queries fell back to Java: {fallbacks}")
    finally:
        if failures:
            logs = subprocess.run(["docker", "logs", OURS[0]], capture_output=True, text=True)
            print("\n".join(l for l in (logs.stdout + logs.stderr).splitlines() if "WARN" in l or "ERROR" in l)[-5000:])
        if not keep:
            subprocess.run(["docker", "rm", "-f", OURS[0], STOCK[0]], capture_output=True)
    for f in failures:
        print("FAIL:", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
