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
             exhaustive scan computed here; the plugin's own
             `method_parameters.exploration_slack` too (scan only).
  half_float float vectors stored as half floats (cosine ones normalized
             first, as stock does): checked as exact, against a scan over the
             half-float values.
  binary     packed bit vectors in the hamming space: the same scores as
             stock, each the hit's own Hamming score; distances tie often, so
             hits are compared by id only above the last score.
  quantized  sq / on_disk mappings: the stock plugin scores quantized vectors
             (and rescores some); the plugin searches full precision. The check
             is that its top 10 are the exhaustive scan's, and that they
             overlap stock's by at least MIN_RECALL on average.
  nested     a knn_vector under a nested field: the best child per parent,
             filters at the root and on the nested docs, and
             expand_nested_docs with inner hits, the same hits, scores and
             inner hits as stock; the same queries on a nested `ivf` field
             probing every cluster, stock's hits too.
  ivf        the plugin's own opt-in `extended` engine (`method: ivf`, no
             stock counterpart, so stock must refuse the mapping): every
             hit scores its exact score (f32 and half-float rerank vectors,
             1-bit, 2-bit OSQ and 4-bit Extended RaBitQ codes);
             probing every cluster, filtered or not, finds nearly all of the
             exhaustive scan's top 10 (its rerank is still approximate), a
             filter few documents pass returns exactly the scan's hits, all
             of it flushed, after deletes and after a force merge; the
             calibrated depth finds most of them; one probe finds fewer, so
             the index is what is searched.
  rust hnsw  the `extended` engine's `method: hnsw` (the graph walked on
             1-bit RaBitQ, 2-bit and 4-bit Extended RaBitQ codes; stock must
             refuse the mapping): every hit scores its exact score, the top 10
             is nearly the exhaustive scan's (filtered too), a filter few
             documents pass returns exactly the scan's hits, all of it
             flushed, after deletes and after a force merge.
  scripts    script_score with the `knn` language's `knn_score` in every
             space the field allows (plus `hammingbit` over long and binary
             fields), and the Painless functions (l2Squared, cosineSimilarity,
             innerProduct, l1Norm, lInfNorm, hamming, lateInteractionScore) in
             script_score and script_fields: the same hits, scores and values
             as stock, and the same refusals. Scripts run in Java for now, so
             they are left out of the no-fallback count.
  realtime   an `index.search.realtime.enabled` index (HNSW, l2, never
             refreshed): documents indexed after the last flush, an update
             and a delete since are seen by top-k, filtered and in-bool knn
             queries, which return the exhaustive scan's hits and scores
             over every live document, all of them in Rust.
  recall     `index.knn.target_recall` (opt-in) on an IVF and
             an HNSW index: at a 0.99 target the top 10 is nearly the scan's,
             the stats show the tuner's sampled checks, a query setting its
             own depth is not tuned, a target over 1 is refused, and 0
             turns tuning off.
  errors     requests the stock plugin refuses are refused here too; faiss
             mappings, which stock accepts, are refused here.

Every search must run in Rust (the plugin's stats show no fallback).

  scripts/check-knn-plugin.py [--keep]

Needs Docker and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import base64, glob, json, math, os, random, shutil, struct, subprocess, sys, time, urllib.request

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
    "lucene_binary": {"engine": "lucene", "space_type": "hamming", "data_type": "binary", "dimension": 64},
    "lucene_half_l2": {"engine": "lucene", "space_type": "l2", "data_type": "half_float"},
    "lucene_half_cos": {"engine": "lucene", "space_type": "cosinesimil", "data_type": "half_float"},
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


def vector(rng, byte, binary_dim=0):
    if binary_dim:
        return [rng.randint(-128, 127) for _ in range(binary_dim // 8)]
    if byte:
        return [rng.randint(-20, 20) for _ in range(DIM)]
    return [round(rng.uniform(-1, 1), 4) for _ in range(DIM)]


def vector_for(spec, rng):
    """A random value for `spec`'s field: floats, bytes, or a binary field's dimension / 8 bytes."""
    binary = spec.get("dimension", DIM) if spec.get("data_type") == "binary" else 0
    return vector(rng, spec.get("data_type") == "byte", binary)


def ties(spec):
    """Hamming scores tie often: hits at the last score may trade places."""
    return spec.get("space_type") == "hamming"


def field_mapping(spec):
    m = {"type": "knn_vector", "dimension": spec.get("dimension", DIM)}
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
        settings["index.engine.accelerated"] = True
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
    if space == "hamming":
        return 1 / (1 + sum(bin((x ^ y) & 0xFF).count("1") for x, y in zip(a, b)))
    dot = sum(x * y for x, y in zip(a, b))
    if space == "cosinesimil":
        return (1 + dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))) / 2
    return 1 / (1 - dot) if dot < 0 else dot + 1


def indexed(spec, v):
    """The vector a field indexes for `v`: a half_float field's half floats, normalized first for cosine."""
    if spec.get("data_type") != "half_float":
        return v
    if spec["space_type"] == "cosinesimil":
        v = normalized(v)
    return [struct.unpack("<e", struct.pack("<e", x))[0] for x in v]


def normalized(v):
    """VectorUtil.l2normalize, in float32."""
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    norm = f32(math.sqrt(f32(sum(f32(x * x) for x in v))))
    return [f32(x / norm) for x in v]


def half_cosine(spec):
    """k-NN scores half_float cosine fields' normalized vectors by its FP16 kernel, (1 + dot) / 2:
    in float32 here, and on CPUs with AVX-512-FP16 by FP16 FMAs on stock, whose scores then differ
    by up to a few 1e-4 and can swap near-equal hits."""
    return spec.get("data_type") == "half_float" and spec["space_type"] == "cosinesimil"


def exact_top(spec, docs, target, k, keep=lambda d: True):
    if half_cosine(spec):
        q = normalized(target)
        of = lambda d: (1 + sum(x * y for x, y in zip(q, indexed(spec, d["emb"])))) / 2
    else:
        of = lambda d: score(spec["space_type"], target, indexed(spec, d["emb"]))
    scored = sorted(((of(d), d["_id"]) for d in docs if keep(d)),
                    key=lambda t: (-t[0], t[1]))
    return scored[:k]


def search(port, index, body):
    r = req(port, "POST", f"/{index}/_search", body)
    return [(h["_id"], h["_score"]) for h in r["hits"]["hits"]]


def same(got, want, tol=1e-4, ties=False):
    """The same hits and scores; with `ties`, hits scoring the last score may differ."""
    if len(got) != len(want):
        return False
    if not all(abs(a - b) <= tol * max(1.0, abs(b)) for (_, a), (_, b) in zip(got, want)):
        return False
    if ties and got:
        last = want[-1][1]
        return [i for i, sc in got if sc > last] == [i for i, sc in want if sc > last]
    return [i for i, _ in got] == [i for i, _ in want]


def close(got, want, tol=1e-3):
    """Scores within `tol`; a hit missing from the other list scores within `tol` of its last."""
    if len(got) != len(want):
        return False
    theirs, last = dict(want), (want[-1][1] if want else 0)
    for (i, sc), (_, w) in zip(got, want):
        if abs(sc - w) > tol * max(1.0, abs(w)):
            return False
        if i in theirs and abs(sc - theirs[i]) > tol * max(1.0, abs(w)):
            return False
        if i not in theirs and abs(sc - last) > tol * max(1.0, abs(last)):
            return False
    return True


def knn(target, **params):
    return {"knn": {"emb": {"vector": target, **params}}}


def nested_check(failures):
    """Nested knn on both nodes: returns the number of queries compared."""
    index = "nested_l2"
    rng = random.Random(f"{SEED}-{index}")
    spec = {"engine": "lucene", "space_type": "l2"}
    docs = []
    for i in range(240):
        kids = []
        for j in range(rng.randint(1, 4)):
            kid = {"color": rng.choice(["red", "blue"])}
            if j != 1:
                kid["emb"] = vector_for(spec, rng)
            kids.append(kid)
        docs.append({"_id": f"p{i}", "tag": "even" if i % 2 == 0 else "odd", "n": i, "nest": kids})
    mapping = {"properties": {"tag": {"type": "keyword"}, "n": {"type": "integer"},
                              "nest": {"type": "nested", "properties": {"color": {"type": "keyword"},
                                                                       "emb": field_mapping(spec)}}}}
    ivf_index = "nested_ivf"
    ivf_nested = {"properties": {"tag": {"type": "keyword"}, "n": {"type": "integer"},
                                 "nest": {"type": "nested", "properties": {"color": {"type": "keyword"},
                                                                          "emb": ivf_mapping()}}}}
    for port, rust, name, m in ((OURS[1], True, index, mapping), (STOCK[1], False, index, mapping),
                                (OURS[1], True, ivf_index, ivf_nested)):
        settings = {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1,
                    "index.merge.policy.max_merge_at_once": 100, "index.merge.policy.segments_per_tier": 100}
        if rust:
            settings["index.engine.accelerated"] = True
        req(port, "PUT", f"/{name}", {"settings": settings, "mappings": m})
        per = len(docs) // SEGMENTS
        for s in range(SEGMENTS):
            lines = []
            for d in docs[s * per:(s + 1) * per]:
                lines.append(json.dumps({"index": {"_index": name, "_id": d["_id"]}}))
                lines.append(json.dumps({k: v for k, v in d.items() if k != "_id"}))
            r = req(port, "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
            assert not r["errors"], json.dumps(r)[:2000]
            req(port, "POST", f"/{name}/_flush")
        req(port, "POST", f"/{name}/_refresh")

    def nested(inner, **extra):
        return {"nested": {"path": "nest", "query": {"knn": {"nest.emb": inner}}, **extra}}

    def hits(port, body, name=index):
        r = req(port, "POST", f"/{name}/_search", body)
        out = []
        for h in r["hits"]["hits"]:
            inner = h.get("inner_hits", {}).get("nest", {}).get("hits", {}).get("hits", [])
            out.append((h["_id"], h["_score"], [(x["_nested"]["offset"], x["_score"]) for x in inner]))
        return out

    def agree(got, want):
        if not same([(i, sc) for i, sc, _ in got], [(i, sc) for i, sc, _ in want]):
            return False
        for (_, _, a), (_, _, b) in zip(got, want):
            if [o for o, _ in a] != [o for o, _ in b] or any(abs(x - y) > 1e-4 for (_, x), (_, y) in zip(a, b)):
                return False
        return True

    checked = 0
    for qi in range(6):
        t = vector_for(spec, rng)
        bodies = [
            ("nested top k", {"size": 20, "query": nested({"vector": t, "k": 10})}),
            ("nested ef_search", {"size": 20, "query": nested({"vector": t, "k": 5,
                                                               "method_parameters": {"ef_search": 100}})}),
            ("nested root filter", {"size": 20, "query": nested({"vector": t, "k": 10,
                                                                 "filter": {"term": {"tag": "even"}}})}),
            ("nested child filter", {"size": 20, "query": nested({"vector": t, "k": 10, "filter": {
                "nested": {"path": "nest", "query": {"term": {"nest.color": "red"}}}}})}),
            ("nested inner hits", {"size": 20, "query": nested({"vector": t, "k": 10},
                                                                inner_hits={"size": 4})}),
            ("expand", {"size": 20, "query": nested({"vector": t, "k": 10, "expand_nested_docs": True},
                                                     inner_hits={"size": 4}, score_mode="max")}),
            ("expand avg", {"size": 20, "query": nested({"vector": t, "k": 10, "expand_nested_docs": True},
                                                         inner_hits={"size": 4})}),
            ("expand filtered", {"size": 20, "query": nested({"vector": t, "k": 10, "expand_nested_docs": True,
                                                              "filter": {"nested": {"path": "nest", "query": {
                                                                  "term": {"nest.color": "blue"}}}}},
                                                             inner_hits={"size": 4}, score_mode="max")}),
        ]
        for label, body in bodies:
            got, want = hits(OURS[1], body), hits(STOCK[1], body)
            checked += 1
            if not agree(got, want):
                failures.append(f"{index} {label} query {qi}: {got[:2]} vs {want[:2]}")
            # The same nested queries on an IVF field probing every cluster:
            # the clusters are probed per parent, and find stock's hits.
            if "ef_search" in label:
                continue
            body = json.loads(json.dumps(body))
            body["query"]["nested"]["query"]["knn"]["nest.emb"]["method_parameters"] = {"nprobes": IVF_ALL}
            got = hits(OURS[1], body, ivf_index)
            checked += 1
            if not agree(got, want):
                failures.append(f"{ivf_index} {label} query {qi}: {got[:2]} vs stock's {want[:2]}")
        # The best child per parent, exhaustively: stock's graph is small enough to find it.
        best = {}
        for d in docs:
            for kid in d["nest"]:
                if "emb" in kid:
                    best[d["_id"]] = max(best.get(d["_id"], 0), score("l2", t, kid["emb"]))
        scan = sorted(best.items(), key=lambda x: (-x[1], x[0]))[:10]
        got = hits(OURS[1], bodies[0][1])
        if not same([(i, sc) for i, sc, _ in got], scan):
            failures.append(f"{index} nested top k query {qi}: not the exhaustive scan's: {got[:2]} vs {scan[:2]}")
    return checked


IVF = {"space_type": "l2"}
IVF_ALL = 100_000


def ivf_mapping(**params):
    return {"type": "knn_vector", "dimension": DIM,
            "method": {"name": "ivf", "engine": "extended", "space_type": "l2",
                       "parameters": {"target_cluster_size": 16, **params}}}


def ivf_check(failures):
    """The `extended` engine's IVF fields on the plugin node (see the module doc)."""
    checked = 0
    # ivf_tenants: sorted by tenant, each tenant's vectors clustered on their own.
    tenants = {"index.sort.field": "tenant", "index.knn.ivf.partition_min_vectors": 50}
    for index, params, extra in (("ivf_f32", {"rerank_encoding": "float"}, {}),
                                 ("ivf_f16", {"rerank_encoding": "half_float"}, {}),
                                 ("ivf_osq2", {"rerank_encoding": "float", "quantizer": "osq2"}, {}),
                                 ("ivf_erq4", {"quantizer": "erq4"}, {}),
                                 ("ivf_auto", {"quantizer": "auto"}, {}),
                                 ("ivf_tenants", {"rerank_encoding": "float"}, tenants)):
        rng = random.Random(f"{SEED}-{index}")
        docs = [{"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd", "n": i, "tenant": i % 3,
                 "emb": vector(rng, False)} for i in range(DOCS)]
        settings = {"index.knn": True, "index.engine.accelerated": True, "number_of_shards": 1,
                    "number_of_replicas": 0, "refresh_interval": -1,
                    "index.merge.policy.max_merge_at_once": 100, "index.merge.policy.segments_per_tier": 100, **extra}
        mapping = {"properties": {"tag": {"type": "keyword"}, "n": {"type": "integer"}, "tenant": {"type": "integer"},
                                  "emb": ivf_mapping(**params)}}
        r = req(STOCK[1], "PUT", f"/{index}", {"settings": {"index.knn": True}, "mappings": mapping}, ok_errors=True)
        if r.get("status", 200) != 400:
            failures.append(f"{index}: stock accepted the extended mapping: {json.dumps(r)[:200]}")
            req(STOCK[1], "DELETE", f"/{index}", ok_errors=True)
        req(OURS[1], "PUT", f"/{index}", {"settings": settings, "mappings": mapping})
        persisted = req(OURS[1], "GET", f"/{index}/_mapping")[index]["mappings"]["properties"]["emb"]
        if persisted.get("method", {}).get("engine") != "extended" or persisted["method"].get("name") != "ivf":
            failures.append(f"{index}: persisted as {persisted}")
        if "quantizer" in params and persisted["method"].get("parameters", {}).get("quantizer") != params["quantizer"]:
            failures.append(f"{index}: quantizer persisted as {persisted}")
        per = DOCS // SEGMENTS
        for s in range(SEGMENTS):
            lines = []
            for d in docs[s * per:(s + 1) * per]:
                lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
                lines.append(json.dumps({"tag": d["tag"], "n": d["n"], "tenant": d["tenant"], "emb": d["emb"]}))
            r = req(OURS[1], "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
            assert not r["errors"], json.dumps(r)[:2000]
            req(OURS[1], "POST", f"/{index}/_flush")
        req(OURS[1], "POST", f"/{index}/_refresh")
        targets = [vector(rng, False) for _ in range(8)]
        live = list(docs)

        def round_(stage):
            nonlocal checked
            found, one_probe = 0, 0
            wide = {"all clusters": 0, "filtered": 0}
            for qi, t in enumerate(targets):
                cases = [
                    ("all clusters", knn(t, k=10, method_parameters={"nprobes": IVF_ALL}), lambda d: True, 10),
                    ("filtered", knn(t, k=10, filter={"term": {"tag": "even"}}, method_parameters={"nprobes": IVF_ALL}),
                     lambda d: d["tag"] == "even", 10),
                    # 40 documents pass: scanned exactly, whatever the probe count.
                    ("rare filter", knn(t, k=10, filter={"range": {"n": {"gte": 100, "lt": 140}}},
                         method_parameters={"nprobes": 1}),
                     lambda d: 100 <= d["n"] < 140, 10),
                ]
                if extra:
                    cases.append(("tenant", knn(t, k=10, filter={"term": {"tenant": 1}}),
                                  lambda d: d["tenant"] == 1, 10))
                by_id = {d["_id"]: d["emb"] for d in live}
                for label, query, keep_doc, k in cases:
                    got = search(OURS[1], index, {"size": 50, "query": query})
                    want = [(i, sc) for sc, i in exact_top(IVF, live, t, k, keep_doc)]
                    checked += 1
                    wrong = [(i, sc) for i, sc in got
                             if i not in by_id or abs(sc - score("l2", t, by_id[i])) > 1e-4 * max(1.0, sc)]
                    if wrong:
                        failures.append(f"{index} {stage} {label} query {qi}: not exact scores or deleted docs {wrong[:3]}")
                    if label in ("rare filter", "tenant"):
                        if not same(got, want):
                            failures.append(f"{index} {stage} {label} query {qi}: {got[:3]} vs the scan's {want[:3]}")
                    else:
                        wide[label] += len({i for i, _ in got} & {i for i, _ in want})
                        if len(got) != k or not all(keep_doc({"tag": "even" if int(i[1:]) % 2 == 0 else "odd"}) for i, _ in got):
                            failures.append(f"{index} {stage} {label} query {qi}: {len(got)} hits or unfiltered ones")
                truth = {i for _, i in exact_top(IVF, live, t, 10)}
                found += len(truth & {i for i, _ in search(OURS[1], index, {"size": 10, "query": knn(t, k=10)})})
                one_probe += len(truth & {i for i, _ in search(
                    OURS[1], index, {"size": 10, "query": knn(t, k=10, method_parameters={"nprobes": 1})})})
                checked += 2
            recall, shallow = found / (10 * len(targets)), one_probe / (10 * len(targets))
            print(f"{index} {stage}: recall@10 {recall:.3f} calibrated, {shallow:.3f} with one probe")
            if recall < 0.9:
                failures.append(f"{index} {stage}: calibrated recall {recall:.3f} < 0.9")
            if shallow >= recall:
                failures.append(f"{index} {stage}: one probe ({shallow:.3f}) is no worse than the calibrated depth")
            for label, n in wide.items():
                if n / (10 * len(targets)) < 0.95:
                    failures.append(f"{index} {stage} {label}: recall@10 {n / (10 * len(targets)):.3f} < 0.95")

        round_("flushed")
        req(OURS[1], "POST", f"/{index}/_delete_by_query?refresh=true", {"query": {"range": {"n": {"lt": 100}}}})
        live = [d for d in docs if d["n"] >= 100]
        round_("after deletes")
        req(OURS[1], "POST", f"/{index}/_forcemerge?max_num_segments=1")
        req(OURS[1], "POST", f"/{index}/_refresh")
        segs = req(OURS[1], "GET", f"/{index}/_segments")["indices"][index]["shards"]["0"][0]["segments"]
        if len(segs) != 1:
            failures.append(f"{index}: {len(segs)} segments after the force merge")
        round_("merged")
    bad = [
        ("ivf byte", {**ivf_mapping(), "data_type": "byte"}),
        ("ivf half_float", {**ivf_mapping(), "data_type": "half_float"}),
        ("ivf small clusters", ivf_mapping(target_cluster_size=8)),
        ("ivf rerank", ivf_mapping(rerank_encoding="int4")),
        ("ivf quantizer", ivf_mapping(quantizer="pq")),
        ("ivf 1-bit erq", ivf_mapping(quantizer="erq1")),
        ("ivf 8-bit osq", ivf_mapping(quantizer="osq8")),
        ("ivf unknown parameter", ivf_mapping(nlist=4)),
        ("ivf on_disk", {**ivf_mapping(), "mode": "on_disk"}),
        ("extended hnsw osq", rust_hnsw_mapping(quantizer="osq2")),
        ("extended hnsw bbq", rust_hnsw_mapping(quantizer="bbq")),
        ("extended hnsw auto", rust_hnsw_mapping(quantizer="auto")),
        ("extended hnsw quantizer", rust_hnsw_mapping(quantizer="pq")),
        ("extended hnsw byte", {**rust_hnsw_mapping(), "data_type": "byte"}),
        ("extended hnsw on_disk", {**rust_hnsw_mapping(), "mode": "on_disk"}),
        ("extended hnsw encoder", rust_hnsw_mapping(encoder={"name": "sq", "parameters": {"bits": 4}})),
        ("extended hnsw ef_search", rust_hnsw_mapping(ef_search=100)),
        ("extended flat", {"type": "knn_vector", "dimension": DIM,
                              "method": {"name": "flat", "engine": "extended", "space_type": "l2"}}),
        ("lucene ivf", {"type": "knn_vector", "dimension": DIM,
                        "method": {"name": "ivf", "engine": "lucene", "space_type": "l2"}}),
    ]
    off = [("ivf off the accelerated engine", ivf_mapping(), False), ("extended hnsw off the accelerated engine", rust_hnsw_mapping(), False)]
    for label, m, rust in [(label, m, True) for label, m in bad] + off:
        settings = {"index.knn": True, **({"index.engine.accelerated": True} if rust else {})}
        r = req(OURS[1], "PUT", "/bad-mapping", {"settings": settings,
                                                 "mappings": {"properties": {"v": m}}}, ok_errors=True)
        if r.get("status", 200) != 400:
            failures.append(f"mapping {label} accepted: {json.dumps(r)[:200]}")
            req(OURS[1], "DELETE", "/bad-mapping", ok_errors=True)
    return checked


def recall_target_check(failures):
    """`index.knn.target_recall` on an IVF and an HNSW index (see the module doc)."""
    checked = 0
    for index, mapping in (("rt_ivf", ivf_mapping(rerank_encoding="float")),
                           ("rt_hnsw", {"type": "knn_vector", "dimension": DIM,
                                        "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"}})):
        rng = random.Random(f"{SEED}-{index}")
        docs = [{"_id": f"d{i}", "emb": vector(rng, False)} for i in range(DOCS)]
        settings = {"index.knn": True, "index.engine.accelerated": True, "number_of_shards": 1,
                    "number_of_replicas": 0, "refresh_interval": -1,
                    "index.knn.target_recall": 0.99,
                    "index.knn.target_recall_sample_rate": 1.0,
                    "index.knn.target_recall_check_budget": 1000.0}
        req(OURS[1], "PUT", f"/{index}", {"settings": settings, "mappings": {"properties": {"emb": mapping}}})
        lines = []
        for d in docs:
            lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
            lines.append(json.dumps({"emb": d["emb"]}))
        r = req(OURS[1], "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        assert not r["errors"], json.dumps(r)[:2000]
        req(OURS[1], "POST", f"/{index}/_refresh")
        uuid = req(OURS[1], "GET", f"/{index}/_settings")[index]["settings"]["index"]["uuid"]
        key = f"{uuid}/0/emb"

        def tuner():
            return req(OURS[1], "GET", "/_plugins/lucene_rust/stats").get("knn_recall", {}).get(key)

        targets = [vector(rng, False) for _ in range(40)]
        found = 0
        for t in targets:
            got = search(OURS[1], index, {"size": 10, "query": knn(t, k=10)})
            found += len({i for _, i in exact_top(IVF, docs, t, 10)} & {i for i, _ in got})
            checked += 1
        recall = found / (10 * len(targets))
        print(f"{index}: recall@10 {recall:.3f} at target 0.99")
        if recall < 0.95:
            failures.append(f"{index}: recall@10 {recall:.3f} at target 0.99")
        # The checks run behind the searches.
        stats = tuner()
        for _ in range(300):
            if stats and stats["samples"] > 0:
                break
            time.sleep(0.1)
            stats = tuner()
        if not stats or stats["samples"] == 0 or stats["queries"] < len(targets) or stats["target_recall"] != 0.99:
            failures.append(f"{index}: tuner stats {stats}")
            continue
        # A query asking for its own depth is not tuned.
        own = {"nprobes": 4} if index == "rt_ivf" else {"ef_search": 50}
        search(OURS[1], index, {"size": 10, "query": knn(targets[0], k=10, method_parameters=own)})
        if tuner()["queries"] != stats["queries"]:
            failures.append(f"{index}: a query with method_parameters {own} was tuned")
        r = req(OURS[1], "PUT", f"/{index}/_settings", {"index.knn.target_recall": 1.5}, ok_errors=True)
        if r.get("status", 200) != 400:
            failures.append(f"{index}: target_recall 1.5 accepted: {json.dumps(r)[:200]}")
        # Off again (the setting is dynamic): searched as asked.
        req(OURS[1], "PUT", f"/{index}/_settings", {"index.knn.target_recall": 0})
        before = tuner()["queries"]
        search(OURS[1], index, {"size": 10, "query": knn(targets[0], k=10)})
        if tuner()["queries"] != before:
            failures.append(f"{index}: target_recall 0 still tuned")
        checked += 3
    return checked


def rust_hnsw_mapping(**params):
    return {"type": "knn_vector", "dimension": DIM,
            "method": {"name": "hnsw", "engine": "extended", "space_type": "l2", "parameters": params}}


def rust_hnsw_check(failures):
    """The `extended` engine's quantized HNSW fields on the plugin node (see the module doc)."""
    checked = 0
    for index, params in (("rq_rabitq", {"quantizer": "rabitq"}), ("rq_erq2", {}),
                          ("rq_erq4", {"quantizer": "erq4", "m": 8, "ef_construction": 64})):
        rng = random.Random(f"{SEED}-{index}")
        docs = [{"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd", "n": i, "emb": vector(rng, False)}
                for i in range(DOCS)]
        settings = {"index.knn": True, "index.engine.accelerated": True, "number_of_shards": 1,
                    "number_of_replicas": 0, "refresh_interval": -1,
                    "index.merge.policy.max_merge_at_once": 100, "index.merge.policy.segments_per_tier": 100}
        mapping = {"properties": {"tag": {"type": "keyword"}, "n": {"type": "integer"},
                                  "emb": rust_hnsw_mapping(**params)}}
        r = req(STOCK[1], "PUT", f"/{index}", {"settings": {"index.knn": True}, "mappings": mapping}, ok_errors=True)
        if r.get("status", 200) != 400:
            failures.append(f"{index}: stock accepted the extended mapping: {json.dumps(r)[:200]}")
            req(STOCK[1], "DELETE", f"/{index}", ok_errors=True)
        req(OURS[1], "PUT", f"/{index}", {"settings": settings, "mappings": mapping})
        persisted = req(OURS[1], "GET", f"/{index}/_mapping")[index]["mappings"]["properties"]["emb"]
        if persisted.get("method", {}).get("engine") != "extended" or persisted["method"].get("name") != "hnsw":
            failures.append(f"{index}: persisted as {persisted}")
        if persisted["method"].get("parameters", {}).get("quantizer") != params.get("quantizer"):
            failures.append(f"{index}: quantizer persisted as {persisted}")
        per = DOCS // SEGMENTS
        for s in range(SEGMENTS):
            lines = []
            for d in docs[s * per:(s + 1) * per]:
                lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
                lines.append(json.dumps({"tag": d["tag"], "n": d["n"], "emb": d["emb"]}))
            r = req(OURS[1], "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
            assert not r["errors"], json.dumps(r)[:2000]
            req(OURS[1], "POST", f"/{index}/_flush")
        req(OURS[1], "POST", f"/{index}/_refresh")
        targets = [vector(rng, False) for _ in range(8)]
        live = list(docs)

        def round_(stage):
            nonlocal checked
            found = {"top k": 0, "filtered": 0}
            for qi, t in enumerate(targets):
                cases = [
                    ("top k", knn(t, k=10, method_parameters={"ef_search": 100}), lambda d: True),
                    ("filtered", knn(t, k=10, filter={"term": {"tag": "even"}}, method_parameters={"ef_search": 100}),
                     lambda d: d["tag"] == "even"),
                    # 40 documents pass: scanned exactly on the raw vectors.
                    ("rare filter", knn(t, k=10, filter={"range": {"n": {"gte": 100, "lt": 140}}}),
                     lambda d: 100 <= d["n"] < 140),
                ]
                by_id = {d["_id"]: d["emb"] for d in live}
                for label, query, keep_doc in cases:
                    got = search(OURS[1], index, {"size": 50, "query": query})
                    want = [(i, sc) for sc, i in exact_top(IVF, live, t, 10, keep_doc)]
                    checked += 1
                    wrong = [(i, sc) for i, sc in got
                             if i not in by_id or abs(sc - score("l2", t, by_id[i])) > 1e-4 * max(1.0, sc)]
                    if wrong:
                        failures.append(f"{index} {stage} {label} query {qi}: not exact scores or deleted docs {wrong[:3]}")
                    if len(got) != 10:
                        failures.append(f"{index} {stage} {label} query {qi}: {len(got)} hits")
                    if label == "rare filter":
                        if not same(got, want):
                            failures.append(f"{index} {stage} {label} query {qi}: {got[:3]} vs the scan's {want[:3]}")
                    else:
                        found[label] += len({i for i, _ in got} & {i for i, _ in want})
                        if not all(keep_doc({"tag": "even" if int(i[1:]) % 2 == 0 else "odd"}) for i, _ in got):
                            failures.append(f"{index} {stage} {label} query {qi}: unfiltered hits")
            for label, n in found.items():
                recall = n / (10 * len(targets))
                print(f"{index} {stage} {label}: recall@10 {recall:.3f}")
                if recall < 0.9:
                    failures.append(f"{index} {stage} {label}: recall@10 {recall:.3f} < 0.9")

        round_("flushed")
        req(OURS[1], "POST", f"/{index}/_delete_by_query?refresh=true", {"query": {"range": {"n": {"lt": 100}}}})
        live = [d for d in docs if d["n"] >= 100]
        round_("after deletes")
        req(OURS[1], "POST", f"/{index}/_forcemerge?max_num_segments=1")
        req(OURS[1], "POST", f"/{index}/_refresh")
        segs = req(OURS[1], "GET", f"/{index}/_segments")["indices"][index]["shards"]["0"][0]["segments"]
        if len(segs) != 1:
            failures.append(f"{index}: {len(segs)} segments after the force merge")
        round_("merged")
    return checked


SCRIPT_SPACES = {
    "lucene_l2": ["l2", "cosinesimil", "innerproduct", "l1", "linf"],
    "lucene_cos": ["cosinesimil", "l2"],
    "lucene_ip": ["innerproduct"],
    "lucene_byte": ["l2", "cosinesimil", "innerproduct", "l1", "linf"],
    "lucene_binary": ["hamming"],
    "lucene_half_l2": ["l2", "cosinesimil", "innerproduct"],
}
PAINLESS = [
    "1 / (1 + l2Squared(params.q, doc['emb']))",
    "1 + cosineSimilarity(params.q, doc['emb'])",
    "1 + cosineSimilarity(params.q, doc['emb'], params.norm)",
    "10000 + innerProduct(params.q, doc['emb'])",
    "1 / (1 + l1Norm(params.q, doc['emb']))",
    "1 / (1 + lInfNorm(params.q, doc['emb']))",
    "1.0 / (1.0 + l2Squared(params.q, doc['emb']))",
    "return 10000000 - l2Squared(params.q, doc[\"emb\"]);",
    "(2 * cosineSimilarity(params.q, doc['emb']) + 2f) / 4",
    "1 / (1 + l2Squared(params['q'], doc[params.f]))",
    "-innerProduct(params.q, doc['emb']) * -1.5e0 + 1e4",
]


def script_score(base, script, size=20):
    return {"size": size, "query": {"script_score": {"query": base, "script": script}}}


def knn_score(space, target, field="emb"):
    return {"lang": "knn", "source": "knn_score",
            "params": {"field": field, "query_value": target, "space_type": space}}


def realtime_check(failures):
    """knn over a realtime index's buffered documents on the plugin node: returns the queries compared."""
    index, spec = "realtime_l2", EXACT["lucene_l2"]
    rng = random.Random(f"{SEED}-{index}")
    docs = [{"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd", "n": i, "emb": vector(rng, False)}
            for i in range(DOCS)]
    settings = {"index.knn": True, "index.engine.accelerated": True, "index.search.realtime.enabled": True,
                "number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    mapping = {"properties": {"tag": {"type": "keyword"}, "n": {"type": "integer"}, "emb": field_mapping(spec)}}
    req(OURS[1], "PUT", f"/{index}", {"settings": settings, "mappings": mapping})

    def bulk(batch):
        lines = []
        for d in batch:
            lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
            lines.append(json.dumps({"tag": d["tag"], "n": d["n"], "emb": d["emb"]}))
        r = req(OURS[1], "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        assert not r["errors"], json.dumps(r)[:2000]

    half = DOCS // 2
    bulk(docs[:half])
    req(OURS[1], "POST", f"/{index}/_flush")
    # Buffered, never refreshed: a committed document updated, a buffered one deleted.
    bulk(docs[half:])
    docs[3] = {**docs[3], "emb": vector(rng, False)}
    bulk([docs[3]])
    req(OURS[1], "DELETE", f"/{index}/_doc/d{half + 5}")
    live = [d for d in docs if d["_id"] != f"d{half + 5}"]
    count = req(OURS[1], "POST", f"/{index}/_count", {"query": {"match_all": {}}})["count"]
    if count != len(live):
        failures.append(f"{index}: {count} documents counted, {len(live)} live")
    checked, buffered = 0, 0
    for qi in range(8):
        t = vector(rng, False)
        cases = [
            ("top k", knn(t, k=10), lambda d: True),
            ("filtered", knn(t, k=10, filter={"term": {"tag": "even"}}), lambda d: d["tag"] == "even"),
            ("in bool", {"bool": {"must": [knn(t, k=10)], "filter": [{"range": {"n": {"gte": half // 2}}}]}},
             None),
        ]
        for label, query, keep_doc in cases:
            got = search(OURS[1], index, {"size": 10, "query": query})
            if keep_doc is None:
                # The knn's own top 10, then the bool's filter.
                top = exact_top(spec, live, t, 10)
                want = [(i, sc) for sc, i in top if int(i[1:]) >= half // 2]
            else:
                want = [(i, sc) for sc, i in exact_top(spec, live, t, 10, keep_doc)]
            checked += 1
            buffered += sum(1 for i, _ in want if int(i[1:]) >= half or i == "d3")
            if not same(got, want):
                failures.append(f"{index} {label} query {qi}: {got[:3]} vs the scan's {want[:3]}")
    print(f"{index}: {checked} queries, {buffered} buffered hits expected")
    if buffered == 0:
        failures.append(f"{index}: no query's hits were buffered documents")
    return checked


def phases_check(failures):
    """knn over three shards, where each request runs a query and a fetch phase per shard: the
    plugin and stock return the same hits, scores, `_source`, inner hits, named queries, fields,
    script fields and highlights, whatever `_source` filter the request has; the plugin's fetch
    phase reuses the query phase's hits, and a filter that drops every vector reads `_source`
    without them. Then a realtime index's buffered documents the same way. Returns the queries
    compared."""
    index = "phases_l2"
    rng = random.Random(f"{SEED}-{index}")
    spec = {"engine": "lucene", "space_type": "l2"}
    words = ["red", "blue", "green", "fox", "dog"]
    docs = []
    for i in range(300):
        kids = [{"color": rng.choice(["red", "blue"]), "emb": vector_for(spec, rng)} for _ in range(rng.randint(1, 3))]
        docs.append({"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd", "n": i,
                     "title": " ".join(rng.choice(words) for _ in range(4)),
                     "obj": {"label": f"l{i % 7}", "emb": vector_for(spec, rng)},
                     "emb": vector_for(spec, rng), "nest": kids})
    mapping = {"properties": {"tag": {"type": "keyword"}, "n": {"type": "integer"}, "title": {"type": "text"},
                              "emb": field_mapping(spec),
                              "obj": {"properties": {"label": {"type": "keyword"}, "emb": field_mapping(spec)}},
                              "nest": {"type": "nested", "properties": {"color": {"type": "keyword"},
                                                                       "emb": field_mapping(spec)}}}}
    for port, rust in ((OURS[1], True), (STOCK[1], False)):
        settings = {"index.knn": True, "index.knn.derived_source.enabled": True, "number_of_shards": 3,
                    "number_of_replicas": 0, "refresh_interval": -1}
        if rust:
            settings["index.engine.accelerated"] = True
        req(port, "PUT", f"/{index}", {"settings": settings, "mappings": mapping})
        per = len(docs) // SEGMENTS
        for s in range(SEGMENTS):
            lines = []
            for d in docs[s * per:(s + 1) * per]:
                lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
                lines.append(json.dumps({k: v for k, v in d.items() if k != "_id"}))
            r = req(port, "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
            assert not r["errors"], json.dumps(r)[:2000]
            req(port, "POST", f"/{index}/_flush")
        req(port, "POST", f"/{index}/_refresh")

    def hits(port, body, params="", name=index):
        r = req(port, "POST", f"/{name}/_search{params}", body)
        pages = [r]
        while "_scroll_id" in r and r["hits"]["hits"] and len(pages) < 4:
            r = req(port, "POST", "/_search/scroll", {"scroll": "1m", "scroll_id": r["_scroll_id"]})
            pages.append(r)
        if "_scroll_id" in r:
            req(port, "DELETE", "/_search/scroll", {"scroll_id": r["_scroll_id"]}, ok_errors=True)
        out = []
        for page in pages:
            for h in page["hits"]["hits"]:
                inner = {k: [(x["_nested"]["offset"], x["_score"], x.get("_source"))
                             for x in v["hits"]["hits"]] for k, v in h.get("inner_hits", {}).items()}
                out.append((h["_id"], h["_score"], {"_source": h.get("_source"), "inner": inner,
                                                    "named": sorted(h.get("matched_queries", [])),
                                                    "fields": h.get("fields"), "highlight": h.get("highlight")}))
        return out

    def agree(got, want):
        return (same([(i, sc) for i, sc, _ in got], [(i, sc) for i, sc, _ in want])
                and json_close([x for _, _, x in got], [x for _, _, x in want], 1e-6))

    def unmasked(got):
        """Stock's inner hits under a root filter that drops `nest.emb` show its mask (`1`); the
        plugin's show the child's vector: the inner hits' sources without it, and whether every
        one of the plugin's is a vector."""
        vectors, out = True, []
        for i, sc, x in got:
            inner = {}
            for k, v in x["inner"].items():
                inner[k] = []
                for off, isc, src in v:
                    vectors &= isinstance((src or {}).get("emb"), list)
                    inner[k].append((off, isc, {a: b for a, b in (src or {}).items() if a != "emb"}))
            out.append((i, sc, {**x, "inner": inner}))
        return vectors, out

    def stats():
        st = req(OURS[1], "GET", "/_plugins/lucene_rust/stats")
        return st["knn_hits"]["reused"], st["accelerated_sources_without_vectors"]

    reused0, unspliced0 = stats()
    checked, fetches, dropping = 0, 0, 0
    for qi in range(4):
        t = vector_for(spec, rng)
        q = knn(t, k=10)
        nested_knn = {"nested": {"path": "nest", "query": {"knn": {"nest.emb": {"vector": t, "k": 10}}}}}
        no_vectors = {"excludes": ["emb", "obj.emb", "nest.emb"]}
        # (label, body, URL parameters, whether its _source filter drops every vector)
        cases = [
            ("whole _source", {"size": 10, "query": q}, "", False),
            ("no _source", {"size": 10, "query": q, "_source": False}, "", False),
            ("excludes vectors", {"size": 10, "query": q, "_source": no_vectors}, "", True),
            ("excludes wildcards", {"size": 10, "query": q, "_source": {"excludes": ["*emb"]}}, "", True),
            ("excludes parents", {"size": 10, "query": q, "_source": {"excludes": ["emb", "obj", "nest"]}}, "", True),
            ("includes", {"size": 10, "query": q, "_source": {"includes": ["tag", "obj.label", "nest.color"]}}, "", True),
            ("includes and excludes", {"size": 10, "query": q,
                                       "_source": {"includes": ["tag", "obj"], "excludes": ["obj.emb"]}},
             "", True),
            ("excludes one vector", {"size": 10, "query": q, "_source": {"excludes": ["emb", "nest.emb"]}}, "", False),
            ("k 100", {"size": 100, "query": knn(t, k=100), "_source": no_vectors}, "", True),
            ("from", {"from": 5, "size": 5, "query": q, "_source": no_vectors}, "", True),
            ("filtered", {"size": 10, "query": knn(t, k=10, filter={"term": {"tag": "even"}}), "_source": no_vectors},
             "", True),
            ("named", {"size": 10, "_source": no_vectors, "query": {"bool": {"should": [
                {"knn": {"emb": {"vector": t, "k": 10, "_name": "k"}}}, {"match": {"title": {"query": "fox", "_name": "t"}}}]}}},
             "", True),
            ("fields", {"size": 10, "query": q, "_source": no_vectors, "fields": ["tag", "obj.label"]}, "", False),
            ("script fields", {"size": 10, "query": q, "_source": no_vectors, "script_fields": {"dims": {"script": {
                "source": "def v = params['_source']['emb']; return v instanceof List ? v.size() : -1"}}}}, "", False),
            ("highlight", {"size": 10, "query": q, "_source": no_vectors, "highlight": {
                "fields": {"title": {}}, "highlight_query": {"match": {"title": "fox"}}}}, "", False),
            ("nested inner hits", {"size": 10, "_source": no_vectors, "query": {**nested_knn, "nested": {
                **nested_knn["nested"], "inner_hits": {"size": 3, "_source": {"excludes": ["emb"]}}}}}, "", False),
            ("nested inner hits whole", {"size": 10, "_source": no_vectors, "query": {**nested_knn, "nested": {
                **nested_knn["nested"], "inner_hits": {"size": 3}}}}, "", False),
            ("scroll", {"size": 4, "query": q, "_source": no_vectors}, "?scroll=1m", True),
        ]
        for label, body, params, drops in cases:
            got, want = hits(OURS[1], body, params), hits(STOCK[1], body, params)
            checked += 1
            fetches += 1
            dropping += drops
            if label == "nested inner hits whole":
                vectors, got = unmasked(got)
                _, want = unmasked(want)
                if not vectors:
                    failures.append(f"{index} {label} query {qi}: an inner hit's _source has no vector")
            if not agree(got, want):
                first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                failures.append(f"{index} {label} query {qi}: {len(got)} vs {len(want)} hits, first difference at "
                                f"{first}: {json.dumps(got[first:first + 1])[:600]} vs {json.dumps(want[first:first + 1])[:600]}")
    reused, unspliced = stats()
    print(f"{index}: {checked} queries, k-NN hits reused {reused - reused0} times, "
          f"{unspliced - unspliced0} _source read without vectors")
    # Each three-shard request rewrites its knn again in each shard's fetch phase (a shard with
    # no hit fetched has none): most of them take the query phase's hits.
    if reused - reused0 < fetches:
        failures.append(f"{index}: k-NN hits reused {reused - reused0} times over {fetches} requests")
    if dropping and unspliced - unspliced0 < dropping:
        failures.append(f"{index}: {unspliced - unspliced0} _source read without vectors over {dropping} requests")

    # A realtime index: half its documents flushed, half buffered.
    rt = "phases_realtime"
    settings = {"index.knn": True, "index.knn.derived_source.enabled": True, "index.engine.accelerated": True,
                "index.search.realtime.enabled": True, "number_of_shards": 3, "number_of_replicas": 0,
                "refresh_interval": -1}
    # No nested field: a realtime view does not search nested documents natively.
    flat = {"properties": {k: v for k, v in mapping["properties"].items() if k != "nest"}}
    req(OURS[1], "PUT", f"/{rt}", {"settings": settings, "mappings": flat})

    def bulk(batch):
        lines = []
        for d in batch:
            lines.append(json.dumps({"index": {"_index": rt, "_id": d["_id"]}}))
            lines.append(json.dumps({k: v for k, v in d.items() if k not in ("_id", "nest")}))
        r = req(OURS[1], "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        assert not r["errors"], json.dumps(r)[:2000]

    half = len(docs) // 2
    bulk(docs[:half])
    req(OURS[1], "POST", f"/{rt}/_flush")
    bulk(docs[half:])
    by_id = {d["_id"]: d for d in docs}
    buffered = 0
    for qi in range(4):
        t = vector_for(spec, rng)
        want = [(i, sc) for sc, i in exact_top(spec, docs, t, 10)]
        buffered += sum(1 for i, _ in want if int(i[1:]) >= half)
        for label, src in (("whole _source", None), ("excludes vectors", {"excludes": ["emb", "obj.emb", "nest.emb"]}),
                           ("includes", {"includes": ["tag", "obj.label"]})):
            body = {"size": 10, "query": knn(t, k=10)}
            if src is not None:
                body["_source"] = src
            got = hits(OURS[1], body, name=rt)
            checked += 1
            if not same([(i, sc) for i, sc, _ in got], want):
                failures.append(f"{rt} {label} query {qi}: {[(i, sc) for i, sc, _ in got][:3]} vs the scan's {want[:3]}")
                continue
            for i, _, x in got:
                d = {k: v for k, v in by_id[i].items() if k not in ("_id", "nest")}
                if label == "excludes vectors":
                    d = {k: v for k, v in d.items() if k != "emb"}
                    d["obj"] = {"label": d["obj"]["label"]}
                elif label == "includes":
                    d = {"tag": d["tag"], "obj": {"label": d["obj"]["label"]}}
                if not json_close(x["_source"], d):
                    failures.append(f"{rt} {label} query {qi}: {i}'s _source {json.dumps(x['_source'])[:300]}")
                    break
    if buffered == 0:
        failures.append(f"{rt}: no query's hits were buffered documents")
    return checked


def json_close(a, b, tol=1e-6):
    """The same JSON, numbers within `tol` (a vector read back as float32)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(json_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= tol * max(1.0, abs(b))
    return a == b


def fallback_counts():
    return req(OURS[1], "GET", "/_plugins/lucene_rust/stats").get("fallbacks", {})


def fell_back(before):
    """The fallback reasons counted since `before` (a `fallback_counts()`)."""
    return {k: v - before.get(k, 0) for k, v in fallback_counts().items() if v != before.get(k, 0)}


def script_check(failures):
    """Scoring scripts on both nodes over the exact indexes: returns the number of queries compared."""
    checked = 0
    bases = [{"match_all": {}}, {"term": {"tag": "even"}}]

    def compare(label, index, body, ties=False):
        nonlocal checked
        checked += 1
        a = req(OURS[1], "POST", f"/{index}/_search", body, ok_errors=True)
        b = req(STOCK[1], "POST", f"/{index}/_search", body, ok_errors=True)
        if "status" in a or "status" in b:
            if a.get("status", 200) != b.get("status", 200):
                failures.append(f"script {label} on {index}: status {a.get('status', 200)} vs stock "
                                f"{b.get('status', 200)}: {json.dumps(a)[:400]} / {json.dumps(b)[:400]}")
            return
        got = [(h["_id"], h["_score"] or 0.0) for h in a["hits"]["hits"]]
        want = [(h["_id"], h["_score"] or 0.0) for h in b["hits"]["hits"]]
        if not same(got, want, ties=ties):
            failures.append(f"script {label} on {index}: {got[:3]} vs stock {want[:3]}")
        fa = [(h["_id"], h.get("fields")) for h in a["hits"]["hits"]]
        fb = [(h["_id"], h.get("fields")) for h in b["hits"]["hits"]]
        if not ties and any(x[1] is not None for x in fb):
            vals = lambda f: [(i, [round(v, 3) for v in (d or {}).get("s", [])]) for i, d in f]
            if vals(fa) != vals(fb):
                failures.append(f"script_fields {label} on {index}: {fa[:2]} vs stock {fb[:2]}")

    for index, spaces in SCRIPT_SPACES.items():
        spec = EXACT[index]
        rng = random.Random(f"{SEED}-{index}-scripts")
        for qi in range(3):
            t = vector_for(spec, rng)
            before = fallback_counts()
            for space in spaces:
                for base in bases:
                    compare(f"knn_score {space} query {qi}", index, script_score(base, knn_score(space, t)),
                            ties=space == "hamming")
            fell = fell_back(before)
            if fell:
                failures.append(f"knn_score on {index} fell back to Java: {fell}")
            if spec.get("data_type") == "binary":
                scripts = ["1 / (1 + hamming(params.q, doc['emb']))"]
            else:
                scripts = PAINLESS
            norm = sum(x * x for x in t)
            for src in scripts:
                params = {"q": t, "norm": norm, "f": "emb"}
                before = fallback_counts()
                compare(f"painless {src} query {qi}", index,
                        script_score(bases[qi % 2], {"source": src, "params": params}), ties=spec.get("data_type") == "binary")
                # Every shape here runs natively, but hamming, and a norm over a byte field.
                native = "hamming" not in src and not ("params.norm" in src and spec.get("data_type") == "byte")
                fell = fell_back(before)
                if native and fell:
                    failures.append(f"painless {src} on {index} fell back to Java: {fell}")
                body = {"size": 5, "query": {"term": {"tag": "odd"}}, "sort": ["n"],
                        "script_fields": {"s": {"script": {"source": src, "params": params}}}}
                compare(f"script_fields {src} query {qi}", index, body)
    # Refusals: an unknown space, a space the field does not allow, a missing parameter, a wrong length.
    t = [0.1] * DIM
    for label, index, script in [
        ("bad space", "lucene_l2", knn_score("nope", t)),
        ("hamming on floats", "lucene_l2", knn_score("hamming", t)),
        ("hammingbit on knn_vector", "lucene_l2", knn_score("hammingbit", 3)),
        ("l2 on binary", "lucene_binary", knn_score("l2", [1] * 8)),
        ("missing field", "lucene_l2", {"lang": "knn", "source": "knn_score", "params": {"query_value": t, "space_type": "l2"}}),
        ("unknown source", "lucene_l2", {"lang": "knn", "source": "nope", "params": {}}),
        ("length", "lucene_l2", knn_score("l2", [0.1] * (DIM - 1))),
        ("byte values", "lucene_byte", knn_score("l2", [0.5] * DIM)),
        ("cosine zero", "lucene_l2", knn_score("cosinesimil", [0.0] * DIM)),
        ("painless dimension", "lucene_l2", {"source": "l2Squared(params.q, doc['emb'])", "params": {"q": [0.1] * 3}}),
        ("painless binary", "lucene_binary", {"source": "l2Squared(params.q, doc['emb'])", "params": {"q": [1] * 8}}),
    ]:
        compare(label, index, script_score({"match_all": {}}, script))
    # Shapes Painless reads differently from the plain parse: the search lookup's names shadow params, params.empty is
    # Map.isEmpty(), and a program past the native limits. Each runs in Java without a native error.
    deep = "1-(" * 16 + "l2Squared(params.q, doc['emb'])" + ")" * 16
    for label, script in [
        ("painless params._source", {"source": "l2Squared(params._source, doc['emb'])", "params": {"_source": t}}),
        ("painless doc[params.doc]", {"source": "1 / (1 + l2Squared(params.q, doc[params.doc]))", "params": {"q": t, "doc": "emb"}}),
        ("painless doc[params.empty]", {"source": "1 / (1 + l2Squared(params.q, doc[params.empty]))", "params": {"q": t, "empty": "emb"}}),
        ("painless depth 17", {"source": "1000000 + " + deep, "params": {"q": t}}),
        ("painless 300 operations", {"source": "l2Squared(params.q, doc['emb'])" + " + 1" * 150, "params": {"q": t}}),
    ]:
        before = fallback_counts()
        compare(label, "lucene_l2", script_score({"match_all": {}}, script))
        if fell_back(before).get("native_error"):
            failures.append(f"{label}: native error instead of a Java fallback")
    # hammingbit over long and binary fields.
    rng = random.Random(f"{SEED}-hammingbit")
    for port in (OURS[1], STOCK[1]):
        req(port, "PUT", "/hammingbit", {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
                                          "mappings": {"properties": {"h": {"type": "long"},
                                                                      "b": {"type": "binary", "doc_values": True},
                                                                      "mv": {"type": "object", "enabled": False}}}})
    lines = []
    for i in range(200):
        raw = bytes(rng.randrange(256) for _ in range(8))
        mv = [[round(rng.uniform(-1, 1), 3) for _ in range(4)] for _ in range(rng.randint(1, 3))]
        lines.append(json.dumps({"index": {"_index": "hammingbit", "_id": f"h{i}"}}))
        lines.append(json.dumps({"h": rng.randrange(-2 ** 62, 2 ** 62), "b": base64.b64encode(raw).decode(), "mv": mv}))
    for port in (OURS[1], STOCK[1]):
        req(port, "POST", "/_bulk?refresh=true", "\n".join(lines) + "\n", ndjson=True)
    for qi in range(3):
        compare(f"hammingbit long {qi}", "hammingbit",
                script_score({"match_all": {}}, knn_score("hammingbit", rng.randrange(-2 ** 62, 2 ** 62), "h")), ties=True)
        q = base64.b64encode(bytes(rng.randrange(256) for _ in range(8))).decode()
        compare(f"hammingbit binary {qi}", "hammingbit",
                script_score({"match_all": {}}, knn_score("hammingbit", q, "b")), ties=True)
        mq = [[round(rng.uniform(-1, 1), 3) for _ in range(4)] for _ in range(2)]
        for space in (None, "l2", "cosinesimil", "innerproduct", "l1", "hamming", "nope"):
            src = "lateInteractionScore(params.q, 'mv', params._source" + (", params.space)" if space else ")")
            body = script_score({"match_all": {}}, {"source": "100 + " + src, "params": {"q": mq, "space": space or ""}})
            compare(f"lateInteractionScore {space} {qi}", "hammingbit", body)
    for port in (OURS[1], STOCK[1]):
        req(port, "DELETE", "/hammingbit")
    # An index without index.knn: the fields are doc values only, the usual exact-search setup.
    types = {"f": "float", "b": "byte", "h": "half_float"}
    for port in (OURS[1], STOCK[1]):
        req(port, "PUT", "/no_knn", {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
                                      "mappings": {"properties": {f"emb_{k}": {"type": "knn_vector", "dimension": DIM,
                                                                               "data_type": t}
                                                                  for k, t in types.items()}}})
    rng = random.Random(f"{SEED}-no-knn")
    lines = []
    for i in range(150):
        lines.append(json.dumps({"index": {"_index": "no_knn", "_id": f"x{i}"}}))
        lines.append(json.dumps({"emb_f": vector(rng, False), "emb_b": vector(rng, True), "emb_h": vector(rng, False)}))
    for port in (OURS[1], STOCK[1]):
        req(port, "POST", "/_bulk?refresh=true", "\n".join(lines) + "\n", ndjson=True)
    for k in types:
        t = vector(rng, k == "b")
        before = fallback_counts()
        for space in ("l2", "cosinesimil", "innerproduct", "l1", "linf"):
            compare(f"no knn emb_{k} {space}", "no_knn", script_score({"match_all": {}}, knn_score(space, t, f"emb_{k}")))
            compare(f"no knn emb_{k} {space} boosted", "no_knn",
                    {"size": 20, "query": {"script_score": {"query": {"match_all": {}}, "boost": 2.5,
                                                            "script": knn_score(space, t, f"emb_{k}")}}})
            # Two targets in one bool: each clause keeps its own target.
            u = vector(rng, k == "b")
            compare(f"no knn emb_{k} {space} two targets", "no_knn",
                    {"size": 20, "query": {"bool": {"should": [
                        {"script_score": {"query": {"match_all": {}}, "script": knn_score(space, x, f"emb_{k}")}}
                        for x in (t, u)]}}})
        fell = fell_back(before)
        if fell:
            failures.append(f"knn_score on no_knn emb_{k} fell back to Java: {fell}")
        for src in PAINLESS:
            compare(f"no knn emb_{k} {src}", "no_knn",
                    script_score({"match_all": {}}, {"source": src.replace("'emb'", f"'emb_{k}'"),
                                                     "params": {"q": t, "norm": sum(x * x for x in t)}}))
    for port in (OURS[1], STOCK[1]):
        req(port, "DELETE", "/no_knn")
    return checked


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
                binary = ties(spec)
                rng = random.Random(f"{SEED}-{index}")
                docs = [{"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd", "n": i,
                         "emb": vector_for(spec, rng)} for i in range(DOCS)]
                ours = create(OURS[1], index, spec, docs, True)
                stock = create(STOCK[1], index, spec, docs, False)
                if ours != stock:
                    failures.append(f"{index}: segment sizes differ: {ours} vs {stock}")
                targets = [vector_for(spec, rng) for _ in range(8)]
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
                    ]
                    if not binary:
                        # Radial search is refused for binary vectors (checked under errors).
                        bodies.append(("min_score", {"size": 100, "query": knn(t, min_score=floor)}, None))
                    if spec["space_type"] in ("l2", "cosinesimil"):
                        dist = {"l2": lambda s: 1 / s - 1, "cosinesimil": lambda s: 2 - 2 * s}[spec["space_type"]](floor)
                        bodies.append(("max_distance", {"size": 100, "query": knn(t, max_distance=dist)}, None))
                    for label, body, keep_doc in bodies:
                        got = search(OURS[1], index, body)
                        want = search(STOCK[1], index, body)
                        checked += 1
                        if not (close(got, want) if half_cosine(spec) else same(got, want, ties=binary)):
                            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a[0] != b[0]),
                                         min(len(got), len(want)))
                            failures.append(f"{index} {label} query {qi}: {len(got)} vs {len(want)} hits, first "
                                            f"difference at {first}: {got[first:first + 2]} vs {want[first:first + 2]}")
                        if binary:
                            # Hamming graphs miss tied neighbours on both nodes; each score must be the hit's own.
                            by_id = {d["_id"]: d["emb"] for d in docs}
                            boost = 2.0 if label == "boost" else 1.0
                            if any(abs(sc - boost * score("hamming", t, by_id[i])) > 1e-6 for i, sc in got):
                                failures.append(f"{index} {label} query {qi}: scores are not the hits' Hamming scores")
                        elif label in ("top k", "filtered", "range filter"):
                            kk = 5 if label == "range filter" else k
                            scan = exact_top(spec, docs, t, kk, keep_doc or (lambda d: True))
                            if not same(got, [(i, sc) for sc, i in scan], ties=binary):
                                failures.append(f"{index} {label} query {qi}: not the exhaustive scan's top {kk}")
                    if not binary:
                        # The plugin's own opt-in adaptive beam stop (stock has no such method
                        # parameter): on these small graphs a slack of 0.5 still finds the scan's top k.
                        body = {"size": 50, "query": knn(t, k=k, method_parameters={"exploration_slack": 0.5})}
                        got = search(OURS[1], index, body)
                        scan = [(i, sc) for sc, i in exact_top(spec, docs, t, k)]
                        checked += 1
                        if not (close(got, scan) if half_cosine(spec) else same(got, scan)):
                            failures.append(f"{index} exploration_slack query {qi}: {got[:3]} vs the scan's {scan[:3]}")
        checked += nested_check(failures)
        checked += ivf_check(failures)
        checked += rust_hnsw_check(failures)
        checked += recall_target_check(failures)
        checked += realtime_check(failures)
        checked += phases_check(failures)
        print(f"compared {checked} queries")

        errors = [
            ("k 0", "lucene_l2", {"query": knn([0.1] * DIM, k=0)}),
            ("k too big", "lucene_l2", {"query": knn([0.1] * DIM, k=10001)}),
            ("two of k/min_score", "lucene_l2", {"query": knn([0.1] * DIM, k=3, min_score=0.5)}),
            ("dimension", "lucene_l2", {"query": knn([0.1] * (DIM - 1), k=3)}),
            ("byte values", "lucene_byte", {"query": knn([0.5] * DIM, k=3)}),
            ("not knn", "lucene_l2", {"query": {"knn": {"tag": {"vector": [0.1] * DIM, "k": 3}}}}),
            ("min_score 0", "lucene_l2", {"query": knn([0.1] * DIM, min_score=0)}),
            ("binary min_score", "lucene_binary", {"query": knn([1] * 8, min_score=0.1)}),
            ("binary max_distance", "lucene_binary", {"query": knn([1] * 8, max_distance=3)}),
            ("binary dimension", "lucene_binary", {"query": knn([1] * 7, k=3)}),
            ("binary values", "lucene_binary", {"query": knn([200] * 8, k=3)}),
            ("half_float dimension", "lucene_half_l2", {"query": knn([0.1] * (DIM + 1), k=3)}),
            ("expand not nested", "lucene_l2", {"query": knn([0.1] * DIM, k=3, expand_nested_docs=True)}),
            # Stock refuses the unknown parameter, the plugin a negative slack: both 400.
            ("negative exploration_slack", "lucene_l2",
             {"query": knn([0.1] * DIM, k=3, method_parameters={"exploration_slack": -1})}),
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
            ("binary dimension", {"type": "knn_vector", "dimension": 12, "data_type": "binary"}),
            ("binary l2", {"type": "knn_vector", "dimension": 16, "data_type": "binary", "space_type": "l2"}),
            ("hamming float", {"type": "knn_vector", "dimension": 16, "space_type": "hamming"}),
            ("half_float hamming", {"type": "knn_vector", "dimension": 16, "data_type": "half_float", "space_type": "hamming"}),
            ("half_float 4x", {"type": "knn_vector", "dimension": 16, "data_type": "half_float", "compression_level": "4x"}),
            ("half_float sq", {"type": "knn_vector", "dimension": 16, "data_type": "half_float",
                               "method": {"name": "hnsw", "engine": "lucene",
                                          "parameters": {"encoder": {"name": "sq", "parameters": {"bits": 1}}}}}),
            ("binary sq", {"type": "knn_vector", "dimension": 16, "data_type": "binary",
                           "method": {"name": "hnsw", "engine": "lucene",
                                      "parameters": {"encoder": {"name": "sq", "parameters": {"bits": 7}}}}}),
        ]
        for label, m in bad_mappings:
            for port in (OURS[1], STOCK[1]):
                r = req(port, "PUT", "/bad-mapping", {"settings": {"index.knn": True},
                                                       "mappings": {"properties": {"v": m}}}, ok_errors=True)
                if r.get("status", 200) != 400:
                    failures.append(f"mapping {label} accepted on port {port}: {json.dumps(r)[:200]}")
                    req(port, "DELETE", "/bad-mapping", ok_errors=True)
        for label, value in (("half_float range", [70000.0] * DIM), ("half_float nan", ["NaN"] * DIM)):
            for port in (OURS[1], STOCK[1]):
                r = req(port, "PUT", "/lucene_half_l2/_doc/bad", {"emb": value}, ok_errors=True)
                if r.get("status", 200) != 400:
                    failures.append(f"document {label} accepted on port {port}: {json.dumps(r)[:200]}")
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
        ran = after["accelerated_queries"] - before["accelerated_queries"]
        fallbacks = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
                     if v != before.get("fallbacks", {}).get(k, 0)}
        print(f"plugin: {ran} shard queries in Rust, fallbacks {fallbacks}")
        if fallbacks:
            failures.append(f"queries fell back to Java: {fallbacks}")
        print(f"compared {script_check(failures)} script queries")
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
