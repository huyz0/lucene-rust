"""The `modelops` section of check-ml-plugin.py: model update, register_meta and upload_chunk, profile and stats.

Both nodes run the same calls and every answer is compared: the update
API's merge and its refusals, the enable switch and rate limiter on a
deployed model, a model sent in chunks, the profile API in both views, and
the stats the calls above leave behind.
"""
import hashlib, json, time, urllib.error, urllib.request

MODEL = ["algorithm", "model_version", "model_format", "model_state", "model_content_hash_value",
         "model_content_size_in_bytes", "total_chunks", "model_config", "description", "name", "is_enabled",
         "rate_limiter", "is_hidden", "planning_worker_node_count", "current_worker_node_count"]


def pick(doc, fields):
    return {f: doc.get(f) for f in fields if f in doc}


def reason(r):
    return r["error"]["reason"] if isinstance(r, dict) and "error" in r else r


def upload(node, path, data):
    req = urllib.request.Request(node.url + path, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as f:
            return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


def node_stats(n):
    nodes = n.ok("GET", "/_plugins/_ml/stats")["nodes"]
    s = flat(next(iter(nodes.values())))
    return {k: v for k, v in s.items() if k not in ("ml_jvm_heap_usage", "ml_executing_task_count")}


def normalize(doc, ids):
    """A profile answer with the model and node ids replaced and the predictor's text dropped."""
    text = json.dumps(doc, sort_keys=True)
    for i, v in ids.items():
        text = text.replace(i, v)
    out = json.loads(text)

    def walk(x):
        if isinstance(x, dict):
            return {k: ("<predictor>" if k == "predictor" else
                        {"count": v["count"], "keys": sorted(v)} if k in ("model_inference_stats", "predict_request_stats") else walk(v))
                    for k, v in x.items() if k not in ("create_time", "last_update_time", "task_id")}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return walk(out)


def modelops(c, ours, stock, zips, register, run):
    for n in (ours, stock):
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.allow_registering_model_via_local_file": True}})
    z = zips["tiny-bert"]
    ids, node_ids = {}, {}
    for side, n in (("ours", ours), ("stock", stock)):
        r = register(n, "tiny-bert", z, "bert", "mean", True)
        ids[side] = n.task(r["task_id"])["model_id"]
        node_ids[side] = next(iter(n.ok("GET", "/_nodes")["nodes"]))
    before = {side: node_stats(n) for side, n in (("ours", ours), ("stock", stock))}

    def both(path, body=None, method="PUT"):
        out = {}
        for side, n in (("ours", ours), ("stock", stock)):
            s, r = n.call(method, path.replace("<id>", ids[side]), body)
            out[side] = (s, reason(r) if s != 200 else {k: r[k] for k in ("result", "_version") if k in r})
        return out

    def doc(n, side):
        return pick(n.ok("GET", f"/_plugins/_ml/models/{ids[side]}"), MODEL)

    updates = {
        "description": {"description": "d2"},
        "name": {"name": "renamed"},
        "disable": {"is_enabled": False},
        "rate limiter": {"rate_limiter": {"limit": "2", "unit": "MINUTES"}},
        "rate limiter limit only": {"rate_limiter": {"limit": "5"}},
        "empty rate limiter": {"rate_limiter": {}},
        "config merge": {"model_config": {"model_type": "bert", "embedding_dimension": 16}},
        "config without type": {"model_config": {"embedding_dimension": 3}},
        "bad unit": {"rate_limiter": {"limit": "2", "unit": "BOGUS"}},
        "bad limit": {"rate_limiter": {"limit": "abc", "unit": "MINUTES"}},
        "negative limit": {"rate_limiter": {"limit": "-1"}},
        "zero limit": {"rate_limiter": {"limit": "0", "unit": "MINUTES"}},
        "connector on local": {"connector_id": "zzz"},
        "unknown group": {"model_group_id": "zzz"},
        "empty": {},
        "unknown field": {"bogus": 1},
        "enable": {"is_enabled": True},
        "clear rate limiter": {"rate_limiter": {"limit": "100000", "unit": "SECONDS"}},
    }
    for label, body in updates.items():
        got = both("/_plugins/_ml/models/<id>", body)
        c.same(f"update {label}", got["ours"], got["stock"])
        c.same(f"update {label} doc", doc(ours, "ours"), doc(stock, "stock"))
    s_o, r_o = ours.call("PUT", "/_plugins/_ml/models/nosuch", {"description": "x"})
    s_s, r_s = stock.call("PUT", "/_plugins/_ml/models/nosuch", {"description": "x"})
    c.same("update unknown model", (s_o, reason(r_o)), (s_s, reason(r_s)))

    # Registrations that fail: the task, and what stays behind.
    cfg = {"model_type": "bert", "embedding_dimension": 32, "framework_type": "sentence_transformers"}
    failing = {
        "unreachable url": {"url": "http://127.0.0.1:1/x.zip", "model_config": cfg},
        "wrong hash": {"url": f"http://localhost:8766/tiny-bert.zip", "model_config": cfg,
                       "model_content_hash_value": "0" * 64},
        "unknown group": {"url": f"http://localhost:8766/tiny-bert.zip", "model_config": cfg, "model_group_id": "nope"},
    }
    for label, extra in failing.items():
        got = {}
        for side, n in (("ours", ours), ("stock", stock)):
            name = f"failing-{label.replace(' ', '-')}-{run}"
            r = n.ok("POST", "/_plugins/_ml/models/_register", {"name": name, "version": "1.0.0", "model_format": "ONNX", **extra})
            t = n.task(r["task_id"])
            left = n.ok("POST", "/_plugins/_ml/models/_search", {"query": {"term": {"name.keyword": name}}})["hits"]["hits"]
            got[side] = (pick(t, ["task_type", "function_name", "state", "error"]), sorted(h["_source"].get("model_state", "chunk") for h in left))
        c.same(f"failed registration: {label}", got["ours"], got["stock"])

    # Deployed: the enable switch and the rate limiter reach the node the model runs on.
    for side, n in (("ours", ours), ("stock", stock)):
        d = n.ok("POST", f"/_plugins/_ml/models/{ids[side]}/_deploy")
        n.task(d["task_id"])
    time.sleep(1)

    def predict(n, side):
        s, r = n.call("POST", f"/_plugins/_ml/_predict/text_embedding/{ids[side]}", {"text_docs": ["hello"]})
        return s, reason(r) if s != 200 else "ok"

    c.same("predict deployed", predict(ours, "ours"), predict(stock, "stock"))
    both("/_plugins/_ml/models/<id>", {"is_enabled": False})
    c.same("predict disabled", predict(ours, "ours"), predict(stock, "stock"))
    both("/_plugins/_ml/models/<id>", {"is_enabled": True})
    both("/_plugins/_ml/models/<id>", {"rate_limiter": {"limit": "2", "unit": "MINUTES"}})
    c.same("predict rate limited", [predict(ours, "ours") for _ in range(4)], [predict(stock, "stock") for _ in range(4)])
    both("/_plugins/_ml/models/<id>", {"rate_limiter": {"limit": "3", "unit": "MINUTES"}})
    both("/_plugins/_ml/models/<id>", {"rate_limiter": {"unit": "HOURS"}})
    c.same("predict after a partial rate limiter update", [predict(ours, "ours") for _ in range(5)], [predict(stock, "stock") for _ in range(5)])
    both("/_plugins/_ml/models/<id>", {"rate_limiter": {"limit": "0", "unit": "MINUTES"}})
    c.same("predict limit zero", predict(ours, "ours"), predict(stock, "stock"))
    both("/_plugins/_ml/models/<id>", {"rate_limiter": {"limit": "100000", "unit": "SECONDS"}})
    c.same("predict after raising the limit", predict(ours, "ours"), predict(stock, "stock"))

    # Profile.
    pid = {ids["ours"]: "<model>", ids["stock"]: "<model>", node_ids["ours"]: "<node>", node_ids["stock"]: "<node>"}
    for path in ("/profile", "/profile/models", f"/profile/models/{{id}}", "/profile/models?view=model",
                 "/profile/models?view=node", "/profile/tasks", "/profile/tasks/nosuch", "/profile/models/nosuch",
                 "/profile?view=model"):
        got = {}
        for side, n in (("ours", ours), ("stock", stock)):
            s, r = n.call("GET", "/_plugins/_ml" + path.replace("{id}", ids[side]))
            got[side] = (s, normalize(r, pid) if s == 200 else reason(r))
        c.same(f"profile {path}", got["ours"], got["stock"])

    # A model sent in chunks.
    size = len(z)
    parts = [z[: size // 3], z[size // 3: 2 * size // 3], z[2 * size // 3:]]
    meta = {"name": f"meta-{run}", "version": "1.0.0", "model_format": "ONNX", "model_content_size_in_bytes": size,
            "model_content_hash_value": hashlib.sha256(z).hexdigest(), "total_chunks": 3,
            "model_config": {"model_type": "bert", "embedding_dimension": 32, "framework_type": "sentence_transformers",
                             "pooling_mode": "MEAN", "normalize_result": True}}
    refusals = {
        "no name": {},
        "no format": {"name": "x"},
        "no hash": {"name": "x", "model_format": "ONNX"},
        "no config": {"name": "x", "model_format": "ONNX", "model_content_hash_value": "h"},
        "no model type": {"name": "x", "model_format": "ONNX", "model_content_hash_value": "h", "model_config": {"embedding_dimension": 4}},
        "text total chunks": {"name": "x", "model_format": "ONNX", "model_content_hash_value": "h", "total_chunks": "3",
                              "model_config": {"model_type": "bert", "embedding_dimension": 4}},
        "no total chunks": {"name": "x", "model_format": "ONNX", "model_content_hash_value": "h",
                            "model_config": {"model_type": "bert", "embedding_dimension": 4}},
    }
    for label, body in refusals.items():
        got = {side: pick_call(n, "POST", "/_plugins/_ml/models/meta", body) for side, n in (("ours", ours), ("stock", stock))}
        c.same(f"register_meta {label}", got["ours"], got["stock"])
    mids = {}
    for side, n in (("ours", ours), ("stock", stock)):
        s, r = n.call("POST", "/_plugins/_ml/models/meta", meta)
        c.same(f"register_meta {side} status", (s, r.get("status")), (200, "CREATED"))
        mids[side] = r["model_id"]
    got = {side: pick(n.ok("GET", f"/_plugins/_ml/models/{mids[side]}"), MODEL) for side, n in (("ours", ours), ("stock", stock))}
    c.same("register_meta model document", got["ours"], got["stock"])
    chunks = {"nosuch": ("nosuch", 0, parts[0]), "bad number": (None, "x", parts[0]), "too big": (None, 3, parts[0]),
              "empty": (None, 0, b""), "chunk 2": (None, 2, parts[2]), "chunk 0": (None, 0, parts[0])}
    for label, (mid, number, data) in chunks.items():
        got = {}
        for side, n in (("ours", ours), ("stock", stock)):
            s, r = upload(n, f"/_plugins/_ml/models/{mid or mids[side]}/upload_chunk/{number}", data)
            got[side] = (s, reason(r) if s != 200 else r)
        c.same(f"upload_chunk {label}", got["ours"], got["stock"])
    got = {side: n.ok("GET", f"/_plugins/_ml/models/{mids[side]}")["model_state"] for side, n in (("ours", ours), ("stock", stock))}
    c.same("model state with chunks missing", got["ours"], got["stock"])
    for side, n in (("ours", ours), ("stock", stock)):
        upload(n, f"/_plugins/_ml/models/{mids[side]}/upload_chunk/1", parts[1])
    got = {side: pick(n.ok("GET", f"/_plugins/_ml/models/{mids[side]}"), MODEL) for side, n in (("ours", ours), ("stock", stock))}
    c.same("model document with every chunk", got["ours"], got["stock"])
    got = {}
    for side, n in (("ours", ours), ("stock", stock)):
        hits = n.ok("POST", "/_plugins/_ml/models/_search", {"size": 10, "query": {"term": {"model_id": mids[side]}}})["hits"]["hits"]
        got[side] = sorted((h["_source"].get("chunk_number"), tuple(sorted(h["_source"]))) for h in hits)
    c.same("chunk documents", got["ours"], got["stock"])
    out = {}
    for side, n in (("ours", ours), ("stock", stock)):
        d = n.ok("POST", f"/_plugins/_ml/models/{mids[side]}/_deploy")
        n.task(d["task_id"])
        s, r = n.call("POST", f"/_plugins/_ml/_predict/text_embedding/{mids[side]}", {"text_docs": ["hello world"]})
        out[side] = (s, r["inference_results"][0]["output"][0]["data"] if s == 200 else reason(r))
    c.same("uploaded model predicts status", out["ours"][0], out["stock"][0])
    if out["ours"][0] == 200 and out["stock"][0] == 200:
        c.true("uploaded model vectors", all(abs(a - b) < 2e-5 for a, b in zip(out["ours"][1], out["stock"][1])), f"{out}")
    else:
        c.same("uploaded model predict answer", out["ours"][1], out["stock"][1])

    # Stats left by all of the above.
    after = {side: node_stats(n) for side, n in (("ours", ours), ("stock", stock))}
    diffs = {}
    for side in after:
        a, b = after[side], before[side]
        mid_ids = {ids[side]: "<model>", mids[side]: "<meta>"}
        diffs[side] = {}
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                key = k
                for i, v in mid_ids.items():
                    key = key.replace(i, v)
                diffs[side][key] = (b.get(k), a.get(k))
    # Stock's auto-redeployer can start a second deploy of a model while the first is loading, which counts
    # as one more deploy request; that timing is not part of the API, so the deploy counts are not compared
    # (nor is the total request count) but for the meta model, which is deployed with nothing else going on.
    for side in diffs:
        diffs[side] = {k: v for k, v in diffs[side].items()
                       if (".deploy." not in k or "<meta>" in k) and k != "ml_request_count"}
    c.same("stats after the calls", diffs["ours"], diffs["stock"])
    for path in ("/stats/ml_request_count", "/stats/ml_model_count", "/stats/ml_task_index_status,ml_connector_count",
                 "/stats/ml_nosuch", "/nosuchnode/stats", "/nosuchnode/stats/ml_request_count", "/stats?bogus=1"):
        got = {}
        for side, n in (("ours", ours), ("stock", stock)):
            s, r = n.call("GET", "/_plugins/_ml" + path)
            if s == 200 and isinstance(r, dict):
                r = {k: v for k, v in r.items() if k != "nodes"} | ({"nodes": len(r["nodes"])} if "nodes" in r else {})
            got[side] = (s, reason(r) if s != 200 else r)
        c.same(f"stats {path}", got["ours"], got["stock"])
    keys = {side: sorted(next(iter(n.ok("GET", "/_plugins/_ml/stats")["nodes"].values()))) for side, n in (("ours", ours), ("stock", stock))}
    c.same("stats node keys", keys["ours"], keys["stock"])
    top = {side: sorted(k for k in n.ok("GET", "/_plugins/_ml/stats") if k != "nodes") for side, n in (("ours", ours), ("stock", stock))}
    c.same("stats cluster keys", top["ours"], top["stock"])


def pick_call(n, method, path, body):
    s, r = n.call(method, path, body)
    return s, reason(r) if s != 200 else "ok"
