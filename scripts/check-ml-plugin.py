#!/usr/bin/env python3
"""ml-commons through the lucene-rust plugin against stock ml-commons.

Starts a node of the pinned OpenSearch image with the lucene-rust plugin (the
bundled plugins removed, ml-commons among them) and, unless --stock names a
running one, a stock node with ml-commons. Both run the same calls of the
model registry, deploy and predict APIs on the fixtures/ml models, which this
script serves over HTTP, and the answers are compared:

  groups     register, duplicate name, get, update, search, delete
  register   a model at a URL, into a named group or a new one; the task and
             the model document stock writes
  deploy     the task, the model's state, predict before and after
  predict    every fixture text under every model and pooling mode: the same
             vectors (relative 2e-5) or the same error status; target_response
  sparse     a SPARSE_TOKENIZE model (fixtures/ml tokenizer + an idf.json):
             the token weights by token and by token id
  encoding   a SPARSE_ENCODING ONNX model: token weights by token and by id,
             result filters, bad parameters
  similarity a TEXT_SIMILARITY cross-encoder: scores, bytes, bad inputs
  asymmetric query/passage prefixes, missing content types, result filters
  torchscript TorchScript models: sentence-transformers and Hugging Face text
             embedding (every pooling mode), sparse encoding, cross-encoder
  lifecycle  delete while deployed, undeploy, delete, get after delete, tasks

A stock node needs libtorch for DJL, which it downloads from publish.djl.ai
unless --libtorch names a local libtorch directory (DJL's
pytorch/<version>-cpu-linux-x86_64 layout, PyTorch 2.5.1). The plugin's node
finds or downloads its own the same way, unless --our-libtorch names one
(scripts/fetch-libtorch.sh fetches it), given as PYTORCH_LIBRARY_PATH.

  scripts/check-ml-plugin.py [--stock http://localhost:9200] [--libtorch DIR] [--our-libtorch DIR]
                             [--keep] [--only groups,sparse,encoding,similarity,asymmetric,torchscript,models,tasks]

Needs Docker and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import argparse, base64, glob, hashlib, http.server, importlib.util, io, json, os, shutil, subprocess, sys, threading, time, zipfile
import struct, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "3.9.0"
OURS = ("ml-check-rust", 19231)
STOCK = ("ml-check-stock", 19232)
SERVE = 8766
TOLERANCE = 2e-5
# Names carry the run's time, so that a reused --stock node's earlier models and groups do not count.
RUN = str(int(time.time()))

spec = importlib.util.spec_from_file_location("golden", os.path.join(ROOT, "scripts", "ml-golden.py"))
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

SETTINGS = {"persistent": {
    "plugins.ml_commons.only_run_on_ml_node": False,
    "plugins.ml_commons.allow_registering_model_via_url": True,
    "plugins.ml_commons.native_memory_threshold": 100,
    "plugins.ml_commons.jvm_heap_memory_threshold": 100,
    "plugins.ml_commons.max_model_on_node": 100}}


class Node:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def call(self, method, path, body=None):
        """(status, json) of one call."""
        r = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                                   method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=300) as f:
                return f.status, json.loads(f.read() or b"null")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, raw.decode(errors="replace")

    def ok(self, method, path, body=None):
        status, r = self.call(method, path, body)
        if status != 200:
            raise RuntimeError(f"{self.url} {method} {path}: {status} {r}")
        return r

    def task(self, task_id):
        for _ in range(600):
            t = self.ok("GET", f"/_plugins/_ml/tasks/{task_id}")
            if t["state"] in ("COMPLETED", "FAILED", "COMPLETED_WITH_ERROR"):
                return t
            time.sleep(0.2)
        raise RuntimeError(f"task {task_id} did not finish")


def start(name, port, image, env=(), mounts=()):
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    args = ["docker", "run", "-d", "--name", name, "--network", "host",
            "-e", "discovery.type=single-node", "-e", "DISABLE_SECURITY_PLUGIN=true",
            "-e", "DISABLE_INSTALL_DEMO_CONFIG=true", "-e", "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g",
            "-e", f"http.port={port}", "-e", f"transport.port={port + 100}",
            "-e", "cluster.routing.allocation.disk.threshold_enabled=false"]
    for e in env:
        args += ["-e", e]
    for m in mounts:
        args += ["-v", m]
    subprocess.run(args + [image], check=True, capture_output=True)


def wait(node):
    for _ in range(150):
        try:
            if node.ok("GET", "/_cluster/health")["status"] in ("green", "yellow"):
                return
        except Exception:
            pass
        time.sleep(2)
    sys.exit(f"node {node.url} did not start")


def serve(zips):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = zips[self.path.strip("/").removesuffix(".zip")]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", SERVE), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


class Check:
    def __init__(self):
        self.rows, self.failed = 0, []

    def same(self, what, ours, stock):
        self.rows += 1
        if ours != stock:
            self.failed.append(f"{what}: ours {ours!r} stock {stock!r}")

    def true(self, what, cond, detail=""):
        self.rows += 1
        if not cond:
            self.failed.append(f"{what}: {detail}")


def reason(r):
    return r["error"]["reason"] if isinstance(r, dict) and "error" in r else r


# Model document fields both nodes must write alike (times and ids differ).
MODEL_FIELDS = [ "algorithm", "model_version", "model_format", "model_state", "model_content_hash_value",
                "model_content_size_in_bytes", "total_chunks", "model_config", "planning_worker_node_count",
                "current_worker_node_count", "deploy_to_all_nodes", "is_hidden"]
TASK_FIELDS = ["task_type", "function_name", "state", "is_async", "error"]


def pick(doc, fields):
    return {f: doc.get(f) for f in fields if f in doc}


def register(node, name, z, model_type, pooling, normalize, group=None):
    body = {"name": f"{name}-{RUN}", "version": "1.0.0", "model_format": "ONNX",
            "model_content_hash_value": hashlib.sha256(z).hexdigest(), "model_content_size_in_bytes": len(z),
            "url": f"http://localhost:{SERVE}/{name}.zip",
            "model_config": {"model_type": model_type, "embedding_dimension": 32,
                             "framework_type": "sentence_transformers", "pooling_mode": pooling,
                             "normalize_result": normalize}}
    if group:
        body["model_group_id"] = group
    return node.ok("POST", "/_plugins/_ml/models/_register", body)


def close(a, b):
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= TOLERANCE * max(1.0, abs(y)) for x, y in zip(a, b))


def approx(a, b):
    """Equal, floats within TOLERANCE, byte buffers compared as their little-endian floats."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        for k in a:
            if k == "byte_buffer" and a.get("data_type") == "INT64":
                if a[k] != b[k]:
                    return False
            elif k == "byte_buffer":
                fa = struct.unpack(f"<{len(base64.b64decode(a[k]['array'])) // 4}f", base64.b64decode(a[k]["array"]))
                fb = struct.unpack(f"<{len(base64.b64decode(b[k]['array'])) // 4}f", base64.b64decode(b[k]["array"]))
                if a[k]["order"] != b[k]["order"] or not close(fa, fb):
                    return False
            elif not approx(a[k], b[k]):
                return False
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(approx(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and close([a], [b])
    return a == b


def answer(r):
    """A call's status with its result, or with its error's type and reason."""
    s, o = r
    if s != 200 and isinstance(o, dict) and "error" in o:
        return s, o["error"].get("type"), o["error"].get("reason")
    return s, o


def onnx_model(n, name, function, z, config=None, fmt="ONNX"):
    body = {"name": f"{name}-{RUN}", "version": "1.0.0", "model_format": fmt, "function_name": function,
            "model_content_hash_value": hashlib.sha256(z).hexdigest(), "model_content_size_in_bytes": len(z),
            "url": f"http://localhost:{SERVE}/{name}.zip"}
    if config:
        body["model_config"] = config
    t = n.task(n.ok("POST", "/_plugins/_ml/models/_register", body)["task_id"])
    mid = t["model_id"]
    dt = n.task(n.ok("POST", f"/_plugins/_ml/models/{mid}/_deploy")["task_id"])
    return mid, [pick(t, TASK_FIELDS), pick(dt, TASK_FIELDS), pick(n.ok("GET", f"/_plugins/_ml/models/{mid}"), MODEL_FIELDS)]


def compare_calls(c, label, ours, stock, z, function, config, calls, fmt="ONNX"):
    out = {}
    for side, n in (("ours", ours), ("stock", stock)):
        mid, rows = onnx_model(n, label, function, z, config, fmt)
        for _, path, body in calls:
            rows.append(answer(n.call("POST", path.replace("<id>", mid), body)))
        n.ok("POST", f"/_plugins/_ml/models/{mid}/_undeploy")
        n.call("DELETE", f"/_plugins/_ml/models/{mid}")
        out[side] = [r if not isinstance(r, tuple) else tuple(x.replace(mid, "<id>") if isinstance(x, str) else x for x in r)
                     for r in rows]
    for i, what in enumerate(["register task", "deploy task", "model"] + [w for w, _, _ in calls]):
        a, b = out["ours"][i], out["stock"][i]
        c.true(f"{label} {what}", approx(a, b), f"ours {json.dumps(a)[:400]} stock {json.dumps(b)[:400]}")


def encoding(c, ours, stock, z):
    p = "/_plugins/_ml/_predict/sparse_encoding/<id>"
    calls = [(f"text {i}", p, {"text_docs": [t]}) for i, t in enumerate(golden.TEXTS)]
    calls += [(f"text {i} by id", p, {"text_docs": [t], "parameters": {"sparse_embedding_format": "TOKEN_ID"}})
              for i, t in enumerate(golden.TEXTS)]
    calls += [
        ("two texts", p, {"text_docs": ["hello world", "the model"]}),
        ("lower-case format", p, {"text_docs": ["hello"], "parameters": {"sparse_embedding_format": "token_id"}}),
        ("bad format", p, {"text_docs": ["hello"], "parameters": {"sparse_embedding_format": "BYTES"}}),
        ("content type", p, {"text_docs": ["hello"], "parameters": {"content_type": "query"}}),
        ("target position", p, {"text_docs": ["hello"], "target_response_positions": [0]}),
        ("target name", p, {"text_docs": ["hello"], "target_response": ["output"]}),
        ("no texts", p, {}),
        ("null text", p, {"text_docs": [None]}),
        ("result filter", p, {"text_docs": ["hello"], "result_filter": {"target_response_positions": [1]}}),
        ("text embedding route", "/_plugins/_ml/_predict/text_embedding/<id>", {"text_docs": ["hello"]}),
        ("similarity route", "/_plugins/_ml/_predict/text_similarity/<id>", {"query_text": "q", "text_docs": ["a"]}),
        ("model route", "/_plugins/_ml/models/<id>/_predict", {"text_docs": ["search documents"]}),
    ]
    compare_calls(c, "tiny-sparse", ours, stock, z, "SPARSE_ENCODING", None, calls)


def similarity(c, ours, stock, z):
    p = "/_plugins/_ml/_predict/text_similarity/<id>"
    calls = [("all texts", p, {"query_text": golden.QUERY, "text_docs": golden.TEXTS}),
             ("empty query", p, {"query_text": "", "text_docs": ["hello"]}),
             ("no query", p, {"text_docs": ["hello"]}),
             ("no docs", p, {"query_text": "q", "text_docs": []}),
             ("filters ignored", p, {"query_text": "q", "text_docs": ["a"], "return_number": False, "target_response": ["x"]}),
             ("model route", "/_plugins/_ml/models/<id>/_predict", {"query_text": "q", "text_docs": ["a b"]}),
             ("model route without docs", "/_plugins/_ml/models/<id>/_predict", {"query_text": "q"}),
             ("text embedding route", "/_plugins/_ml/_predict/text_embedding/<id>", {"text_docs": ["a"]})]
    config = {"model_type": "bert", "embedding_dimension": 1, "framework_type": "huggingface_transformers"}
    compare_calls(c, "tiny-cross-encoder", ours, stock, z, "TEXT_SIMILARITY", config, calls)


def asymmetric(c, ours, stock, z):
    p = "/_plugins/_ml/_predict/text_embedding/<id>"
    config = {"model_type": "bert", "embedding_dimension": 32, "framework_type": "sentence_transformers",
              "pooling_mode": "mean", "normalize_result": True}
    prefixed = dict(config, query_prefix="query: ", passage_prefix="passage: ")
    calls = [("no content type", p, {"text_docs": ["hello"]}),
             ("query", p, {"text_docs": ["hello", "world"], "parameters": {"content_type": "query"}}),
             ("passage", p, {"text_docs": ["hello"], "parameters": {"content_type": "PASSAGE"}}),
             ("no type in parameters", p, {"text_docs": ["hello"], "parameters": {}}),
             ("bad type", p, {"text_docs": ["hello"], "parameters": {"content_type": "doc"}})]
    compare_calls(c, "tiny-bert-prefixed", ours, stock, z, "TEXT_EMBEDDING", prefixed, calls)
    calls = [("parameters without prefixes", p, {"text_docs": ["hello"], "parameters": {"content_type": "query"}}),
             ("no numbers", p, {"text_docs": ["hello"], "return_number": False}),
             ("bytes", p, {"text_docs": ["hello"], "return_bytes": True}),
             ("target position", p, {"text_docs": ["hello"], "target_response_positions": [0]}),
             ("target position missing", p, {"text_docs": ["hello"], "target_response_positions": [3]}),
             ("target other", p, {"text_docs": ["hello"], "target_response": ["other"]}),
             ("null text", p, {"text_docs": ["hello", None]}),
             ("result filter", p, {"text_docs": ["hello"], "result_filter": {"return_number": False}}),
             ("result filter wins", p, {"text_docs": ["hello"], "return_number": False, "result_filter": {}})]
    compare_calls(c, "tiny-bert-plain", ours, stock, z, "TEXT_EMBEDDING", config, calls)


def torch_zip(name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(os.path.join(golden.HERE, f"{name}.pt"), f"{name}.pt")
        f.write(os.path.join(golden.HERE, "tokenizer.json"), "tokenizer.json")
    return buf.getvalue()


TORCH_MODELS = ("tiny-st", "tiny-hf", "tiny-sparse-ts", "tiny-cross-ts")


def torchscript(c, ours, stock, zips):
    p = "/_plugins/_ml/_predict/text_embedding/<id>"
    calls = [(f"text {i}", p, {"text_docs": [t]}) for i, t in enumerate(golden.TEXTS)]
    calls += [("bytes", p, {"text_docs": ["hello"], "return_bytes": True}),
              ("sentence embedding only", p, {"text_docs": ["hello"], "target_response": ["sentence_embedding"]}),
              ("positions", p, {"text_docs": ["hello"], "target_response_positions": [0, 3]}),
              ("no numbers", p, {"text_docs": ["hello"], "return_number": False}),
              ("two texts", p, {"text_docs": ["hello world", "the model"]}),
              ("model route", "/_plugins/_ml/models/<id>/_predict", {"text_docs": ["search documents"]})]
    config = {"model_type": "bert", "embedding_dimension": 32, "framework_type": "sentence_transformers"}
    compare_calls(c, "tiny-st", ours, stock, zips["tiny-st"], "TEXT_EMBEDDING", config, calls, "TORCH_SCRIPT")
    for pooling, normalize in golden.CASES:
        calls = [(f"text {i}", p, {"text_docs": [t]}) for i, t in enumerate(golden.TEXTS)]
        calls.append(("bytes", p, {"text_docs": ["hello"], "return_bytes": True}))
        config = {"model_type": "bert", "embedding_dimension": 32, "framework_type": "huggingface_transformers",
                  "pooling_mode": pooling, "normalize_result": normalize}
        compare_calls(c, f"tiny-hf-{pooling}-{normalize}", ours, stock, zips["tiny-hf"], "TEXT_EMBEDDING", config,
                      calls, "TORCH_SCRIPT")
    p = "/_plugins/_ml/_predict/sparse_encoding/<id>"
    calls = [(f"text {i}", p, {"text_docs": [t]}) for i, t in enumerate(golden.TEXTS)]
    calls += [(f"text {i} by id", p, {"text_docs": [t], "parameters": {"sparse_embedding_format": "TOKEN_ID"}})
              for i, t in enumerate(golden.TEXTS)]
    calls += [("target name", p, {"text_docs": ["hello"], "target_response": ["output"]}),
              ("text embedding route", "/_plugins/_ml/_predict/text_embedding/<id>", {"text_docs": ["hello"]})]
    compare_calls(c, "tiny-sparse-ts", ours, stock, zips["tiny-sparse-ts"], "SPARSE_ENCODING", None, calls, "TORCH_SCRIPT")
    p = "/_plugins/_ml/_predict/text_similarity/<id>"
    calls = [("all texts", p, {"query_text": golden.QUERY, "text_docs": golden.TEXTS}),
             ("empty query", p, {"query_text": "", "text_docs": ["hello"]}),
             ("model route", "/_plugins/_ml/models/<id>/_predict", {"query_text": "q", "text_docs": ["a b"]})]
    config = {"model_type": "bert", "embedding_dimension": 1, "framework_type": "huggingface_transformers"}
    compare_calls(c, "tiny-cross-ts", ours, stock, zips["tiny-cross-ts"], "TEXT_SIMILARITY", config, calls, "TORCH_SCRIPT")


def groups(c, ours, stock):
    rs = {}
    for label, n in (("ours", ours), ("stock", stock)):
        s1, r1 = n.call("POST", "/_plugins/_ml/model_groups/_register", {"name": f"check-group-{RUN}", "description": "d"})
        s2, r2 = n.call("POST", "/_plugins/_ml/model_groups/_register", {"name": f"check-group-{RUN}"})
        gid = r1.get("model_group_id")
        g = n.ok("GET", f"/_plugins/_ml/model_groups/{gid}")
        s3, r3 = n.call("PUT", f"/_plugins/_ml/model_groups/{gid}", {"description": "e"})
        g2 = n.ok("GET", f"/_plugins/_ml/model_groups/{gid}")
        hits = n.ok("POST", "/_plugins/_ml/model_groups/_search", {"query": {"term": {"name.keyword": f"check-group-{RUN}"}}})["hits"]["total"]["value"]
        s4, r4 = n.call("GET", "/_plugins/_ml/model_groups/missing-group")
        s5, _ = n.call("DELETE", f"/_plugins/_ml/model_groups/{gid}")
        s6, _ = n.call("GET", f"/_plugins/_ml/model_groups/{gid}")
        rs[label] = {
            "register": (s1, r1.get("status")), "duplicate": (s2, reason(r2).replace(gid, "<id>")),
            "get": pick(g, ["name", "description", "latest_version", "access"]), "update": (s3, r3),
            "after_update": g2.get("description"), "search": hits, "missing": (s4, reason(r4)),
            "delete": s5, "get_deleted": s6}
    for k in rs["ours"]:
        c.same(f"groups {k}", rs["ours"][k], rs["stock"][k])


def models(c, ours, stock, zips):
    texts = golden.TEXTS
    for name, model_type in golden.MODELS:
        z = zips[name]
        for pooling, normalize in golden.CASES:
            label = f"{name} {pooling} {normalize}"
            state = {}
            for side, n in (("ours", ours), ("stock", stock)):
                r = register(n, name, z, model_type, pooling, normalize)
                t = n.task(r["task_id"])
                mid = t.get("model_id")
                m = n.ok("GET", f"/_plugins/_ml/models/{mid}")
                before = n.call("POST", f"/_plugins/_ml/_predict/text_embedding/{mid}", {"text_docs": ["hi"]})
                d = n.ok("POST", f"/_plugins/_ml/models/{mid}/_deploy")
                dt = n.task(d["task_id"])
                md = n.ok("GET", f"/_plugins/_ml/models/{mid}")
                outs = []
                for text in texts:
                    s, o = n.call("POST", f"/_plugins/_ml/_predict/text_embedding/{mid}", {"text_docs": [text]})
                    outs.append((s, o))
                s_t, o_t = n.call("POST", f"/_plugins/_ml/models/{mid}/_predict",
                                  {"text_docs": ["hello world"], "target_response": ["sentence_embedding"]})
                s_del, r_del = n.call("DELETE", f"/_plugins/_ml/models/{mid}")
                u = n.ok("POST", f"/_plugins/_ml/models/{mid}/_undeploy")
                _, mu = n.call("GET", f"/_plugins/_ml/models/{mid}")
                s_d2, _ = n.call("DELETE", f"/_plugins/_ml/models/{mid}")
                s_g, r_g = n.call("GET", f"/_plugins/_ml/models/{mid}")
                state[side] = dict(register=(r.get("status"), pick(t, TASK_FIELDS)), model=pick(m, MODEL_FIELDS),
                                   before=(before[0], reason(before[1])), deploy=(d.get("status"), d.get("task_type")),
                                   deploy_task=pick(dt, TASK_FIELDS), deployed=pick(md, MODEL_FIELDS), outs=outs,
                                   target=(s_t, o_t), delete_deployed=(s_del, reason(r_del)),
                                   undeploy=sorted(v["stats"][mid] for v in u.values()),
                                   undeployed=pick(mu if "error" not in mu else {}, ["model_state", "planning_worker_node_count", "current_worker_node_count"]),
                                   delete=s_d2, get_deleted=(s_g, reason(r_g).replace(mid, "<id>")))
            a, b = state["ours"], state["stock"]
            for k in ("register", "model", "before", "deploy", "deploy_task", "deployed", "delete_deployed",
                      "undeploy", "undeployed", "delete", "get_deleted"):
                c.same(f"{label} {k}", a[k], b[k])
            for i, ((sa, oa), (sb, ob)) in enumerate(zip(a["outs"], b["outs"])):
                if sa != 200 or sb != 200:
                    c.same(f"{label} text {i} status", sa, sb)
                    continue
                ta, tb = oa["inference_results"][0]["output"][0], ob["inference_results"][0]["output"][0]
                c.same(f"{label} text {i} tensor", {k: ta[k] for k in ("name", "data_type", "shape")},
                       {k: tb[k] for k in ("name", "data_type", "shape")})
                c.true(f"{label} text {i} vector", close(ta["data"], tb["data"]), f"{ta['data'][:4]} vs {tb['data'][:4]}")
            c.same(f"{label} target_response status", a["target"][0], b["target"][0])


IDF = {"search": 3.25, "documents": 1.5, "model": 0.75, "##s": 2.0}
SPARSE_TEXTS = ["search documents models search", "", "the model", "Neural search finds documents by meaning",
                "café über straße naïve"]


def sparse_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(os.path.join(golden.HERE, "tokenizer.json"), "tokenizer.json")
        f.writestr("idf.json", json.dumps(IDF))
    return buf.getvalue()


def sparse(c, ours, stock, z):
    out = {}
    for side, n in (("ours", ours), ("stock", stock)):
        r = n.ok("POST", "/_plugins/_ml/models/_register", {
            "name": f"sparse-{RUN}", "version": "1.0.0", "model_format": "TORCH_SCRIPT", "function_name": "SPARSE_TOKENIZE",
            "model_content_hash_value": hashlib.sha256(z).hexdigest(), "model_content_size_in_bytes": len(z),
            "url": f"http://localhost:{SERVE}/sparse.zip"})
        t = n.task(r["task_id"])
        mid = t["model_id"]
        dt = n.task(n.ok("POST", f"/_plugins/_ml/models/{mid}/_deploy")["task_id"])
        m = n.ok("GET", f"/_plugins/_ml/models/{mid}")
        rows = [pick(t, TASK_FIELDS), pick(dt, TASK_FIELDS), pick(m, MODEL_FIELDS)]
        for fmt in (None, "TOKEN_ID"):
            body = {"text_docs": SPARSE_TEXTS}
            if fmt:
                body["parameters"] = {"sparse_embedding_format": fmt}
            rows.append(n.call("POST", f"/_plugins/_ml/_predict/sparse_tokenize/{mid}", body))
        n.ok("POST", f"/_plugins/_ml/models/{mid}/_undeploy")
        out[side] = rows
    for i, what in enumerate(["register task", "deploy task", "model", "predict by token", "predict by id"]):
        c.same(f"sparse {what}", out["ours"][i], out["stock"][i])


def tasks(c, ours, stock):
    for path in ("/_plugins/_ml/tasks/missing-task",):
        sa, ra = ours.call("GET", path)
        sb, rb = stock.call("GET", path)
        c.same(f"GET {path}", (sa, reason(ra)), (sb, reason(rb)))
    for side in (ours, stock):
        side.ok("POST", "/_plugins/_ml/tasks/_search", {"query": {"match_all": {}}})
    sa, ra = ours.call("POST", "/_plugins/_ml/models/_register", {"name": "x"})
    sb, rb = stock.call("POST", "/_plugins/_ml/models/_register", {"name": "x"})
    c.same("register without format", (sa, reason(ra)), (sb, reason(rb)))
    sa, ra = ours.call("POST", "/_plugins/_ml/_predict/text_embedding/missing-model", {"text_docs": ["a"]})
    sb, rb = stock.call("POST", "/_plugins/_ml/_predict/text_embedding/missing-model", {"text_docs": ["a"]})
    c.same("predict missing model", (sa, reason(ra)), (sb, reason(rb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock")
    ap.add_argument("--libtorch")
    ap.add_argument("--our-libtorch")
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
    if a.our_libtorch:
        start(OURS[0], OURS[1], image, ("PYTORCH_LIBRARY_PATH=/opt/libtorch",),
              (f"{os.path.abspath(a.our_libtorch)}:/opt/libtorch",))
    else:
        start(OURS[0], OURS[1], image)
    if a.stock:
        stock = Node(a.stock)
    else:
        env, mounts = (), ()
        if a.libtorch:
            env = ("PYTORCH_LIBRARY_PATH=/opt/libtorch", "PYTORCH_VERSION=2.5.1", "PYTORCH_FLAVOR=cpu")
            mounts = (f"{os.path.abspath(a.libtorch)}:/opt/libtorch",)
        start(STOCK[0], STOCK[1], f"opensearchproject/opensearch:{VERSION}", env, mounts)
        stock = Node(f"http://localhost:{STOCK[1]}")
    ours = Node(f"http://localhost:{OURS[1]}")
    zips = {name: golden.model_zip(name) for name, _ in golden.MODELS}
    zips["sparse"] = sparse_zip()
    for name in (golden.SPARSE, golden.CROSS):
        zips[name] = golden.model_zip(name)
    zips["tiny-bert-prefixed"] = zips["tiny-bert-plain"] = zips["tiny-bert"]
    for name in TORCH_MODELS:
        zips[name] = torch_zip(name)
    for pooling, normalize in golden.CASES:
        zips[f"tiny-hf-{pooling}-{normalize}"] = zips["tiny-hf"]
    serve(zips)
    c = Check()
    try:
        for n in (ours, stock):
            wait(n)
            n.ok("PUT", "/_cluster/settings", SETTINGS)
        sections = {
            "groups": lambda: groups(c, ours, stock),
            "sparse": lambda: sparse(c, ours, stock, zips["sparse"]),
            "encoding": lambda: encoding(c, ours, stock, zips[golden.SPARSE]),
            "similarity": lambda: similarity(c, ours, stock, zips[golden.CROSS]),
            "asymmetric": lambda: asymmetric(c, ours, stock, zips["tiny-bert"]),
            "torchscript": lambda: torchscript(c, ours, stock, zips),
            "models": lambda: models(c, ours, stock, zips),
            "tasks": lambda: tasks(c, ours, stock),
        }
        for name in (a.only.split(",") if a.only else sections):
            sections[name]()
    finally:
        if not a.keep:
            subprocess.run(["docker", "rm", "-f", OURS[0]] + ([] if a.stock else [STOCK[0]]), capture_output=True)
    for f in c.failed:
        print("MISMATCH", f)
    print(f"check-ml-plugin: {c.rows - len(c.failed)}/{c.rows} rows match stock")
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
