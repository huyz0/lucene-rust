#!/usr/bin/env python3
"""knn_vector and knn through the lucene-rust plugin against the stock k-NN plugin.

Starts two nodes of the pinned OpenSearch image: one with the lucene-rust
plugin (the bundled plugins removed; the plugin serves knn_vector itself) and
a stock one, which bundles opensearch-knn. Both get the same `index.knn`
indexes -- one per mapping below, written in three flushed segments -- and
answer the same queries. The Rust engine writes the plugin node's segments.

  exact      full-precision lucene mappings (l2, cosinesimil,
             innerproduct; float and byte): the graphs are small
             enough that HNSW finds the true neighbours on both nodes, so every
             top-k, filtered, ef_search, boosted and radial query must return
             the same hits with the same scores (1e-4), and the same hits as an
             exhaustive scan computed here.
  quantized  sq / on_disk mappings: the stock plugin scores quantized vectors
             (and rescores some); the plugin searches full precision. The check
             is that its top 10 are the exhaustive scan's, and that they
             overlap stock's by at least MIN_RECALL on average.
  errors     requests the stock plugin refuses are refused here too; faiss
             mappings, which stock accepts, are refused here.

Every search must run in Rust (the plugin's stats show no fallback).

  scripts/check-knn-plugin.py [--keep]

Needs Docker and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import glob, json, math, os, random, shutil, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "3.9.0"
OURS, STOCK = ("knn-check-rust", 19221), ("knn-check-stock", 19222)
DIM = 16
DOCS = 600
SEGMENTS = 3
MIN_RECALL = 0.8
SEED = 7

EXACT = {
    "lucene_l2": {"engine": "lucene", "space_type": "l2"},
    "lucene_cos": {"engine": "lucene", "space_type": "cosinesimil"},
    "lucene_ip": {"engine": "lucene", "space_type": "innerproduct"},
    "lucene_byte": {"engine": "lucene", "space_type": "l2", "data_type": "byte"},
}
QUANTIZED = {
    "lucene_sq": {"engine": "lucene", "space_type": "l2", "encoder": {"name": "sq", "parameters": {"bits": 7}}},
    "lucene_sq_1bit": {"engine": "lucene", "space_type": "l2", "encoder": {"name": "sq", "parameters": {"bits": 1}}},
    "on_disk_32x": {"space_type": "l2", "mode": "on_disk"},
    "lucene_4x": {"space_type": "l2", "compression_level": "4x"},
}


def req(port, method, path, body=None, ndjson=False, timeout=120, ok_errors=False):
    data = None
    if body is not None:
        data = body.encode() if ndjson else json.dumps(body).encode()
    r = urllib.request.Request(f"http://localhost:{port}{path}", data=data, method=method,
                               headers={"Content-Type": "application/x-ndjson" if ndjson else "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        if ok_errors:
            return {"status": e.code, "error": json.loads(e.read() or b"null")}
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:2000]}") from None


def start(name, port, image):
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", name, "-p", f"{port}:9200",
                    "-e", "discovery.type=single-node", "-e", "DISABLE_SECURITY_PLUGIN=true",
                    "-e", "DISABLE_INSTALL_DEMO_CONFIG=true", "-e", "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g",
                    "-e", "cluster.routing.allocation.disk.threshold_enabled=false", image],
                   check=True, capture_output=True)


def wait(port):
    for _ in range(150):
        try:
            if req(port, "GET", "/_cluster/health")["status"] in ("green", "yellow"):
                return
        except Exception:
            pass
        time.sleep(2)
    sys.exit(f"node on {port} did not start")


def vector(rng, byte):
    if byte:
        return [rng.randint(-20, 20) for _ in range(DIM)]
    return [round(rng.uniform(-1, 1), 4) for _ in range(DIM)]


def field_mapping(spec):
    m = {"type": "knn_vector", "dimension": DIM}
    for k in ("data_type", "mode", "compression_level"):
        if k in spec:
            m[k] = spec[k]
    if "engine" in spec:
        method = {"name": "hnsw", "engine": spec["engine"], "space_type": spec["space_type"],
                  "parameters": {"m": 16, "ef_construction": 128}}
        if "encoder" in spec:
            method["parameters"]["encoder"] = spec["encoder"]
        m["method"] = method
    else:
        m["space_type"] = spec["space_type"]
    return m


def create(port, index, spec, docs, rust, extra=None):
    settings = {**(extra or {}), "index.knn": True, "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1,
                "index.merge.policy.max_merge_at_once": 100, "index.merge.policy.segments_per_tier": 100}
    if rust:
        settings["index.lucene_rust.engine"] = True
    mapping = {"properties": {"tag": {"type": "keyword"}, "n": {"type": "integer"}, "emb": field_mapping(spec)}}
    req(port, "PUT", f"/{index}", {"settings": settings, "mappings": mapping})
    per = len(docs) // SEGMENTS
    for s in range(SEGMENTS):
        lines = []
        for d in docs[s * per:(s + 1) * per]:
            lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
            lines.append(json.dumps({"tag": d["tag"], "n": d["n"], "emb": d["emb"]}))
        r = req(port, "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        assert not r["errors"], json.dumps(r)[:2000]
        req(port, "POST", f"/{index}/_flush")
    req(port, "POST", f"/{index}/_refresh")
    segs = req(port, "GET", f"/{index}/_segments")["indices"][index]["shards"]["0"][0]["segments"]
    return sorted(v["num_docs"] for v in segs.values())


def score(space, a, b):
    """The Lucene similarity scores the k-NN plugin's translations equal."""
    if space == "l2":
        return 1 / (1 + sum((x - y) ** 2 for x, y in zip(a, b)))
    dot = sum(x * y for x, y in zip(a, b))
    if space == "cosinesimil":
        return (1 + dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))) / 2
    return 1 / (1 - dot) if dot < 0 else dot + 1


def exact_top(spec, docs, target, k, keep=lambda d: True):
    scored = sorted(((score(spec["space_type"], target, d["emb"]), d["_id"]) for d in docs if keep(d)),
                    key=lambda t: (-t[0], t[1]))
    return scored[:k]


def search(port, index, body):
    r = req(port, "POST", f"/{index}/_search", body)
    return [(h["_id"], h["_score"]) for h in r["hits"]["hits"]]


def same(got, want, tol=1e-4):
    if [i for i, _ in got] != [i for i, _ in want]:
        return False
    return all(abs(a - b) <= tol * max(1.0, abs(b)) for (_, a), (_, b) in zip(got, want))


def knn(target, **params):
    return {"knn": {"emb": {"vector": target, **params}}}


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
        before = req(OURS[1], "GET", "/_plugins/lucene_rust/stats")
        checked = 0
        for group, specs in (("exact", EXACT), ("quantized", QUANTIZED)):
            for index, spec in specs.items():
                byte = spec.get("data_type") == "byte"
                rng = random.Random(f"{SEED}-{index}")
                docs = [{"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd", "n": i, "emb": vector(rng, byte)}
                        for i in range(DOCS)]
                ours = create(OURS[1], index, spec, docs, True)
                stock = create(STOCK[1], index, spec, docs, False)
                if ours != stock:
                    failures.append(f"{index}: segment sizes differ: {ours} vs {stock}")
                targets = [vector(rng, byte) for _ in range(8)]
                if group == "quantized":
                    overlaps = []
                    for t in targets:
                        body = {"size": 10, "query": knn(t, k=10)}
                        got = search(OURS[1], index, body)
                        want = exact_top(spec, docs, t, 10)
                        if [i for i, _ in got] != [i for _, i in want]:
                            failures.append(f"{index}: top 10 is not the exhaustive scan's: {got[:3]} vs {want[:3]}")
                        st = {i for i, _ in search(STOCK[1], index, body)}
                        overlaps.append(len(st & {i for i, _ in got}) / 10)
                        checked += 1
                    recall = sum(overlaps) / len(overlaps)
                    print(f"{index}: top 10 overlap stock's by {recall:.3f}")
                    if recall < MIN_RECALL:
                        failures.append(f"{index}: overlap with stock {recall:.3f} < {MIN_RECALL}")
                    continue
                for qi, t in enumerate(targets):
                    k = 10
                    # Halfway between the 25th and 26th best scores: no document sits on the
                    # boundary, where float rounding decides.
                    radial_k = exact_top(spec, docs, t, 26)
                    floor = (radial_k[-2][0] + radial_k[-1][0]) / 2
                    bodies = [
                        ("top k", {"size": 50, "query": knn(t, k=k)}, None),
                        ("filtered", {"size": 50, "query": knn(t, k=k, filter={"term": {"tag": "even"}})},
                         lambda d: d["tag"] == "even"),
                        ("range filter", {"size": 50, "query": knn(t, k=5, filter={"range": {"n": {"lt": 300}}})},
                         lambda d: d["n"] < 300),
                        ("ef_search", {"size": 50, "query": knn(t, k=k, method_parameters={"ef_search": 200})}, None),
                        ("boost", {"size": 20, "query": knn(t, k=k, boost=2.0)}, None),
                        ("in bool", {"size": 20, "query": {"bool": {"must": [knn(t, k=k)],
                                                                    "filter": [{"term": {"tag": "odd"}}]}}}, None),
                        ("min_score", {"size": 100, "query": knn(t, min_score=floor)}, None),
                    ]
                    if spec["space_type"] != "innerproduct":
                        dist = {"l2": lambda s: 1 / s - 1, "cosinesimil": lambda s: 2 - 2 * s}[spec["space_type"]](floor)
                        bodies.append(("max_distance", {"size": 100, "query": knn(t, max_distance=dist)}, None))
                    for label, body, keep_doc in bodies:
                        got = search(OURS[1], index, body)
                        want = search(STOCK[1], index, body)
                        checked += 1
                        if not same(got, want):
                            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a[0] != b[0]),
                                         min(len(got), len(want)))
                            failures.append(f"{index} {label} query {qi}: {len(got)} vs {len(want)} hits, first "
                                            f"difference at {first}: {got[first:first + 2]} vs {want[first:first + 2]}")
                        if label in ("top k", "filtered", "range filter"):
                            kk = 5 if label == "range filter" else k
                            scan = exact_top(spec, docs, t, kk, keep_doc or (lambda d: True))
                            if [i for i, _ in got] != [i for _, i in scan]:
                                failures.append(f"{index} {label} query {qi}: not the exhaustive scan's top {kk}")
        print(f"compared {checked} queries")

        errors = [
            ("k 0", "lucene_l2", {"query": knn([0.1] * DIM, k=0)}),
            ("k too big", "lucene_l2", {"query": knn([0.1] * DIM, k=10001)}),
            ("two of k/min_score", "lucene_l2", {"query": knn([0.1] * DIM, k=3, min_score=0.5)}),
            ("dimension", "lucene_l2", {"query": knn([0.1] * (DIM - 1), k=3)}),
            ("byte values", "lucene_byte", {"query": knn([0.5] * DIM, k=3)}),
            ("not knn", "lucene_l2", {"query": {"knn": {"tag": {"vector": [0.1] * DIM, "k": 3}}}}),
            ("min_score 0", "lucene_l2", {"query": knn([0.1] * DIM, min_score=0)}),
        ]
        for label, index, body in errors:
            a = req(OURS[1], "POST", f"/{index}/_search", body, ok_errors=True)
            b = req(STOCK[1], "POST", f"/{index}/_search", body, ok_errors=True)
            sa, sb = a.get("status", 200), b.get("status", 200)
            if sa != sb:
                failures.append(f"error {label}: status {sa} vs stock {sb}: {json.dumps(a)[:300]}")
        bad_mappings = [
            ("nmslib", {"type": "knn_vector", "dimension": 4, "method": {"name": "hnsw", "engine": "nmslib"}}),
            ("no dimension", {"type": "knn_vector"}),
            ("bad space", {"type": "knn_vector", "dimension": 4, "space_type": "nope"}),
        ]
        for label, m in bad_mappings:
            for port in (OURS[1], STOCK[1]):
                r = req(port, "PUT", "/bad-mapping", {"settings": {"index.knn": True},
                                                       "mappings": {"properties": {"v": m}}}, ok_errors=True)
                if r.get("status", 200) != 400:
                    failures.append(f"mapping {label} accepted on port {port}: {json.dumps(r)[:200]}")
                    req(port, "DELETE", "/bad-mapping", ok_errors=True)
        faiss_mappings = [
            ("faiss", {"type": "knn_vector", "dimension": 4, "method": {"name": "hnsw", "engine": "faiss"}}),
            ("top-level faiss", {"type": "knn_vector", "dimension": 4, "engine": "FAISS"}),
        ]
        for label, m in faiss_mappings:
            r = req(OURS[1], "PUT", "/bad-mapping", {"settings": {"index.knn": True},
                                                     "mappings": {"properties": {"v": m}}}, ok_errors=True)
            if r.get("status", 200) != 400 or "faiss" not in json.dumps(r).lower():
                failures.append(f"mapping {label} not refused: {json.dumps(r)[:200]}")
                req(OURS[1], "DELETE", "/bad-mapping", ok_errors=True)
        r = req(OURS[1], "PUT", "/no-engine", {"settings": {"index.knn": True},
                                                "mappings": {"properties": {"v": {"type": "knn_vector", "dimension": 4}}}})
        persisted = req(OURS[1], "GET", "/no-engine/_mapping")["no-engine"]["mappings"]["properties"]["v"]
        if persisted.get("engine") != "lucene":
            failures.append(f"a mapping without an engine persisted as {persisted}")
        r = req(OURS[1], "PUT", "/no-engine/_mapping", {"properties": {"v": {"type": "knn_vector", "dimension": 4}}},
                ok_errors=True)
        if "error" in r:
            failures.append(f"the same mapping again was refused: {json.dumps(r)[:300]}")
        req(OURS[1], "DELETE", "/no-engine")

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
