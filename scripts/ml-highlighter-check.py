"""The `highlighter` section of check-ml-plugin.py: the sentence highlighting model.

Registers OpenSearch's semantic highlighter (by its published URL; by name stock refuses it) on both nodes, deploys
it and compares the answers to `question_answering` predict calls: the
highlighted sentences with their positions and offsets, contexts with
abbreviations, quotes and non-ASCII text, a context longer than one chunk,
and the refusals. The nodes download the 439 MB model from
artifacts.opensearch.org, so this section needs network access from both
(see --proxy-truststore) and is not part of the default run.
"""
import json

URL = ("https://artifacts.opensearch.org/models/ml-models/amazon/sentence-highlighting/"
       "opensearch-semantic-highlighter-v1/1.0.0/torch_script/"
       "sentence-highlighting_opensearch-semantic-highlighter-v1-1.0.0-torch_script.zip")
REGISTER = {
    "name": "opensearch-semantic-highlighter-v1", "version": "1.0.0", "function_name": "QUESTION_ANSWERING",
    "model_format": "TORCH_SCRIPT", "url": URL,
    "model_content_hash_value": "bb647ba4eaace03b3441059d372050a0786c9e063af50b2e3a113fd401760a86",
    "model_config": {"model_type": "sentence_highlighting", "framework_type": "huggingface_transformers"},
}

WEATHER = "The weather is nice, it is beautiful day. The sun is shining. The sky is blue."
LONG = " ".join(
    f"Sentence number {i} talks about {t}." for i, t in enumerate(
        ["cats", "the weather", "engines", "gardens", "the stock market", "rivers", "bread", "music"] * 40))

CASES = {
    "weather": {"question": "How is the weather?", "context": WEATHER},
    "unrelated": {"question": "What is the capital of France?", "context": WEATHER},
    "abbreviations": {"question": "Who visited the office?",
                      "context": "Dr. Smith visited the office on Monday. Mrs. Jones stayed home. "
                                 "The visit, i.e. the tour, ended at noon. Prof. Lee called e.g. at five."},
    "quotes": {"question": "What did he say?",
               "context": "He said \"we should go.\" Then he left the room. \"Why?\" she asked. It was late!"},
    "unicode": {"question": "Où est le chat ?",
                "context": "Le chat dort sur le canapé. 猫はソファで寝ています。 The dog barks \U0001F600 loudly. Café noir."},
    "initials": {"question": "Who is J.R. Smith?", "context": "It was J.R. Smith who came. He is a baker. U.S.A. is far."},
    "one sentence": {"question": "weather", "context": "no end mark here"},
    "whitespace": {"question": "weather", "context": "  The sun.\n\n  The   rain?  \t Yes! "},
    "empty context": {"question": "weather", "context": ""},
    "long context": {"question": "What is said about the stock market?", "context": LONG},
    "long context other": {"question": "Tell me about bread", "context": LONG},
    "no question": {"context": WEATHER},
    "no context": {"question": "weather"},
    "null question": {"question": None, "context": WEATHER},
    "null context": {"question": "weather", "context": None},
    "empty question": {"question": "", "context": WEATHER},
    "blank question": {"question": "  ", "context": WEATHER},
    "blank context": {"question": "weather", "context": "   "},
    "numeric question": {"question": 5, "context": WEATHER},
    "numeric context": {"question": "weather", "context": 5},
    "list context": {"question": "weather", "context": ["a"]},
    "boolean question": {"question": True, "context": WEATHER},
    "map context": {"question": "weather", "context": {"a": 1}},
    "list question only": {"question": ["a"]},
    "both empty": {"question": "", "context": ""},
    "both null": {"question": None, "context": None},
    "float question": {"question": 1.5, "context": "The sun is out. Rain."},
    "extra field": {"question": "How is the weather?", "context": WEATHER, "extra": 1},
}


def answer(n, mid, path, body):
    """The status and the body (or error reason), with the model's id masked."""
    s, o = n.call("POST", path.replace("{id}", mid), body)
    if s != 200:
        o = o.get("error", {}).get("reason", o) if isinstance(o, dict) else o
    return {"status": s, "body": json.loads(json.dumps(o).replace(mid, "<id>"))}


def highlighter(c, ours, stock, close):
    ids = {}
    for side, n in (("ours", ours), ("stock", stock)):
        s, r = n.call("POST", "/_plugins/_ml/models/_register", REGISTER)
        assert s == 200, r
        t = n.task(r["task_id"])
        assert t.get("model_id"), t
        ids[side] = t["model_id"]
        d = n.ok("POST", f"/_plugins/_ml/models/{ids[side]}/_deploy")
        t = n.task(d["task_id"])
        assert t["state"] == "COMPLETED", t
    paths = {
        "predict": "/_plugins/_ml/_predict/question_answering/{id}",
        "predict by model path": "/_plugins/_ml/models/{id}/_predict",
        "wrong algorithm": "/_plugins/_ml/_predict/text_embedding/{id}",
    }
    for label, body in CASES.items():
        got = {side: answer(n, ids[side], paths["predict"], body) for side, n in (("ours", ours), ("stock", stock))}
        c.same(f"highlighter {label}", got["ours"], got["stock"])
    body = {"question": "How is the weather?", "context": WEATHER}
    for label in ("predict by model path",):
        got = {side: answer(n, ids[side], paths[label], body) for side, n in (("ours", ours), ("stock", stock))}
        c.same(f"highlighter {label}", got["ours"], got["stock"])
    got = {side: answer(n, ids[side], paths["wrong algorithm"], {"text_docs": ["a"]}) for side, n in (("ours", ours), ("stock", stock))}
    c.same("highlighter wrong algorithm", got["ours"], got["stock"])
