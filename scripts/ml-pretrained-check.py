"""The `pretrained` section of check-ml-plugin.py: models registered by name.

Registers OpenSearch pre-trained models (name, version and format, no url)
on both nodes and compares the task, the model document, the vectors and the
refusals. The nodes download from artifacts.opensearch.org, so this section
needs network access from both of them (see --proxy-truststore) and is not
part of the default run.
"""
import json

SENTENCE = "huggingface/sentence-transformers/all-MiniLM-L6-v2"
CROSS = "huggingface/cross-encoders/ms-marco-MiniLM-L-6-v2"
MODEL = ["algorithm", "model_version", "model_format", "model_state", "model_content_hash_value",
         "model_content_size_in_bytes", "total_chunks", "model_config", "description", "is_hidden"]
TASK = ["task_type", "function_name", "state", "is_async", "error"]


def pick(doc, fields):
    return {f: doc.get(f) for f in fields if f in doc}


def register(n, body):
    status, r = n.call("POST", "/_plugins/_ml/models/_register", body)
    if status != 200:
        return {"status": status, "reason": r["error"]["reason"] if isinstance(r, dict) and "error" in r else r}
    t = n.task(r["task_id"])
    out = {"status": status, "created": r.get("status"), "task": pick(t, TASK)}
    if t.get("model_id"):
        out["model"] = pick(n.ok("GET", f"/_plugins/_ml/models/{t['model_id']}"), MODEL)
        out["id"] = t["model_id"]
    return out


def pretrained(c, ours, stock, close):
    cases = {
        "sentence embedding": {"name": SENTENCE, "version": "1.0.2", "model_format": "ONNX"},
        "lower case format": {"name": SENTENCE, "version": "1.0.2", "model_format": "onnx", "function_name": "SPARSE_ENCODING"},
        "description ignored": {"name": SENTENCE, "version": "1.0.2", "model_format": "ONNX", "description": "mine"},
        "cross encoder": {"name": CROSS, "version": "1.0.2", "model_format": "ONNX"},
        "no version": {"name": SENTENCE, "model_format": "ONNX"},
        "no format": {"name": SENTENCE, "version": "1.0.2"},
        "unknown version": {"name": SENTENCE, "version": "9.9.9", "model_format": "ONNX"},
        "unknown name": {"name": "huggingface/sentence-transformers/nosuch", "version": "1.0.2", "model_format": "ONNX"},
        "unknown group": {"name": SENTENCE, "version": "1.0.2", "model_format": "ONNX", "model_group_id": "nope"},
        "highlighter by name": {"name": "amazon/sentence-highlighting/opensearch-semantic-highlighter-v1",
                                "version": "1.0.0", "model_format": "TORCH_SCRIPT"},
    }
    ids = {}
    for label, body in cases.items():
        got = {}
        for side, n in (("ours", ours), ("stock", stock)):
            got[side] = register(n, body)
            ids[side, label] = got[side].pop("id", None)
        c.same(f"pretrained {label}", got["ours"], got["stock"])
    # URL registrations: the config checks stock makes before it downloads anything.
    u = {"name": "p", "url": "http://127.0.0.1:1/x.zip", "model_format": "ONNX"}
    for label, extra in {
        "url without config": {},
        "similarity without config": {"function_name": "TEXT_SIMILARITY"},
        "question answering without config": {"function_name": "QUESTION_ANSWERING"},
        "config without type": {"model_config": {"embedding_dimension": 4, "framework_type": "sentence_transformers"}},
        "sparse without config": {"function_name": "SPARSE_ENCODING"},
    }.items():
        got = {side: register(n, {**u, **extra}) for side, n in (("ours", ours), ("stock", stock))}
        c.same(f"pretrained {label}", got["ours"], got["stock"])
    for label in ("sentence embedding", "cross encoder"):
        outs = {}
        for side, n in (("ours", ours), ("stock", stock)):
            mid = ids[side, label]
            d = n.ok("POST", f"/_plugins/_ml/models/{mid}/_deploy")
            n.task(d["task_id"])
            if label == "sentence embedding":
                body, path = {"text_docs": ["hello world", "pretrained models"]}, f"/_plugins/_ml/_predict/text_embedding/{mid}"
            else:
                body, path = {"text_docs": ["a cat", "a dog"], "query_text": "feline"}, f"/_plugins/_ml/_predict/text_similarity/{mid}"
            s, o = n.call("POST", path, body)
            outs[side] = (s, [r["output"][0]["data"] for r in o["inference_results"]] if s == 200 else o)
        c.same(f"pretrained {label} predict status", outs["ours"][0], outs["stock"][0])
        if outs["ours"][0] == 200:
            ok = all(close(a, b) for a, b in zip(outs["ours"][1], outs["stock"][1]))
            c.true(f"pretrained {label} predict vectors", ok, f"{outs['ours'][1]} vs {outs['stock'][1]}")
