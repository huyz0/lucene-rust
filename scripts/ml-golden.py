#!/usr/bin/env python3
"""Records, from a stock OpenSearch node with ml-commons, the vectors each
fixtures/ml model gives under each pooling mode: fixtures/ml/expected.json,
which lucene-ml's tests compare against.

The node must reach this machine's HTTP server for the model zips (the script
serves fixtures/ml on --port) and have libtorch available to DJL.

Usage: scripts/ml-golden.py http://localhost:9200 [--port 8765]
"""
import argparse
import hashlib
import http.server
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
import zipfile

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "ml")
TEXTS = [
    "hello world",
    "",
    "Neural search finds documents by meaning",
    "café über straße naïve",
    "unknownwordzzz qqq",
    " ".join(["opensearch indexes documents into shards"] * 120),
]
MODELS = [
    ("tiny-bert", "bert"),
    ("tiny-bert-pooled", "bert"),
    ("tiny-distilbert", "distilbert"),
]
CASES = [
    ("mean", True), ("mean", False), ("mean_sqrt_len", True), ("max", True), ("max", False),
    ("weighted_mean", True), ("cls", False), ("last_token", True), ("none", False),
]


def model_zip(name):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as f:
        f.write(os.path.join(HERE, f"{name}.onnx"), f"{name}.onnx")
        f.write(os.path.join(HERE, "tokenizer.json"), "tokenizer.json")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    zips = {name: model_zip(name) for name, _ in MODELS}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = zips[self.path.strip("/").removesuffix(".zip")]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def req(method, path, body=None):
        r = urllib.request.Request(a.host + path, data=None if body is None else json.dumps(body).encode(),
                                   method=method, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r) as f:
            return json.loads(f.read())

    req("PUT", "/_cluster/settings", {"persistent": {
        "plugins.ml_commons.only_run_on_ml_node": False,
        "plugins.ml_commons.allow_registering_model_via_url": True,
        "plugins.ml_commons.native_memory_threshold": 100,
        "plugins.ml_commons.jvm_heap_memory_threshold": 100,
        "plugins.ml_commons.max_model_on_node": 100}})

    def wait(task):
        while True:
            t = req("GET", f"/_plugins/_ml/tasks/{task}")
            if t["state"] in ("COMPLETED", "FAILED", "COMPLETED_WITH_ERROR"):
                if t["state"] != "COMPLETED":
                    raise SystemExit(f"task {task}: {t}")
                return t
            time.sleep(0.3)

    expected = {"texts": TEXTS, "cases": []}
    for name, model_type in MODELS:
        z = zips[name]
        for pooling, normalize in CASES:
            r = req("POST", "/_plugins/_ml/models/_register", {
                "name": f"{name}-{pooling}-{normalize}", "version": "1.0.0", "model_format": "ONNX",
                "model_content_hash_value": hashlib.sha256(z).hexdigest(), "model_content_size_in_bytes": len(z),
                "url": f"http://localhost:{a.port}/{name}.zip",
                "model_config": {"model_type": model_type, "embedding_dimension": 32,
                                 "framework_type": "sentence_transformers", "pooling_mode": pooling,
                                 "normalize_result": normalize}})
            mid = wait(r["task_id"])["model_id"]
            wait(req("POST", f"/_plugins/_ml/models/{mid}/_deploy")["task_id"])
            vectors = []
            for text in TEXTS:
                try:
                    o = req("POST", f"/_plugins/_ml/_predict/text_embedding/{mid}", {"text_docs": [text]})
                    o = o["inference_results"][0]["output"][0]
                    vectors.append({"shape": o["shape"], "data": o["data"]})
                except urllib.error.HTTPError as e:
                    vectors.append({"error": e.code, "reason": json.loads(e.read())["error"]["reason"]})
            expected["cases"].append({"model": name, "model_type": model_type, "pooling": pooling,
                                      "normalize": normalize, "outputs": vectors})
            req("POST", f"/_plugins/_ml/models/{mid}/_undeploy")
            print(name, pooling, normalize, [v.get("shape", v.get("reason")) for v in vectors])
    with open(os.path.join(HERE, "expected.json"), "w") as f:
        json.dump(expected, f, separators=(",", ":"))
    server.shutdown()


if __name__ == "__main__":
    main()
