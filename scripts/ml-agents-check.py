"""The `agents` section of check-ml-plugin.py: the agent registry and the tool catalogue.

Registers, reads, updates, searches and deletes agents on both nodes and
compares the answers (agent ids, times and index versions masked), then
lists and reads the tools ml-commons itself registers. A stock node with
other plugins lists their tools too; only the names both sides know from
ml-commons are compared. Then runs flow and conversational_flow agents with
the tools that need no language model, against indices both nodes hold, a
remote model and a connector on the mock model server of the `remote`
section, sync and with `?async=true`. Last, the conversation memory API:
memories, messages and traces, their listing, search and deletion.
"""
import importlib.util, json, os, re, time

ML_TOOLS = ["ConnectorTool", "IndexInsightTool", "QueryPlanningTool", "VisualizationTool", "McpSseTool", "AgentTool",
            "WriteToScratchPadTool", "ReadFromScratchPadTool", "McpStreamableHttpTool", "SearchIndexTool", "MLModelTool",
            "IndexMappingTool", "ListIndexTool"]
DROP = {"created_time", "create_time", "updated_time", "last_updated_time", "_version", "_seq_no", "_primary_term", "took"}

FULL = {"name": "full", "type": "conversational", "description": "d",
        "llm": {"model_id": "m", "parameters": {"x": "y", "n": 3}},
        "tools": [{"type": "ListIndexTool", "name": "li", "description": "D", "parameters": {"p": 1, "q": {"a": 2}, "b": True},
                   "include_output_in_agent_response": True, "attributes": {"k": "v"}, "config": {"c": 1}}],
        "memory": {"type": "conversation_index", "session_id": "s", "window_size": 4},
        "parameters": {"n": 3, "b": True, "o": {"a": [1, 2]}}, "app_type": "A"}

REGISTER = {
    "flow": {"name": "r-flow", "type": "flow", "tools": [{"type": "ListIndexTool"}]},
    "flow no tools": {"name": "r-flow2", "type": "flow"},
    "full": FULL,
    "upper case type": {"name": "r-up", "type": "FLOW"},
    "all types": {"name": "r-cf", "type": "conversational_flow", "memory": {"type": "conversation_index"}},
    "ag ui": {"name": "r-ag", "type": "ag_ui"},
    "plan": {"name": "r-plan", "type": "plan_execute_and_reflect", "llm": {"model_id": "m", "parameters": {"q": 1}},
             "parameters": {"a": "b"}, "tools": [{"type": "ListIndexTool"}], "memory": {"type": "conversation_index"}},
    "plan without parameters": {"name": "r-plan2", "type": "plan_execute_and_reflect", "llm": {"model_id": "m"}},
    "v2 without memory": {"name": "r-v2", "type": "conversational_v2", "llm": {"model_id": "m"}},
    "same name twice": {"name": "r-flow", "type": "flow"},
    "extra field": {"name": "r-extra", "type": "flow", "zzz": 1},
    "hidden ignored": {"name": "r-hid", "type": "flow", "is_hidden": True},
    "no name": {"type": "flow"},
    "blank name": {"name": " ", "type": "flow"},
    "long name": {"name": "x" * 129, "type": "flow"},
    "name at the limit": {"name": "x" * 128, "type": "flow"},
    "no type": {"name": "r-nt"},
    "bad type": {"name": "r-bt", "type": "zzz"},
    "conversational without llm": {"name": "r-c", "type": "conversational"},
    "llm without model": {"name": "r-l", "type": "conversational", "llm": {}},
    "model alias": {"name": "r-ma", "type": "conversational", "model": {"model_id": "m"}},
    "tool without type": {"name": "r-t", "type": "flow", "tools": [{"name": "x"}]},
    "duplicate tool": {"name": "r-d", "type": "flow", "tools": [{"type": "A", "name": "n"}, {"type": "A", "name": "n"}]},
    "same tool other name": {"name": "r-d2", "type": "flow", "tools": [{"type": "A", "name": "n1"}, {"type": "A", "name": "n2"}]},
    "unknown tool": {"name": "r-u", "type": "flow", "tools": [{"type": "Nope"}]},
    "bad memory": {"name": "r-bm", "type": "flow", "memory": {"type": "bogus"}},
    "name object": {"name": {"a": "b"}, "type": "flow"},
    "name list": {"name": [1], "type": "flow"},
    "name number": {"name": 5, "type": "flow"},
    "type object": {"name": "r-to", "type": {"a": 1}},
    "llm string": {"name": "r-ls", "type": "flow", "llm": "x"},
    "tools string": {"name": "r-ts", "type": "flow", "tools": "x"},
    "memory number": {"name": "r-mn", "type": "flow", "memory": 5},
    "window size text": {"name": "r-ws", "type": "flow", "memory": {"type": "conversation_index", "window_size": "4"}},
    "description object": {"name": "r-do", "type": "flow", "description": {"a": 1}},
}

UPDATES = [
    ("name", {"name": "u2"}), ("description", {"description": "d1"}), ("type", {"type": "flow"}),
    ("same type", {"type": "conversational"}), ("llm model", {"llm": {"model_id": "m2"}}),
    ("llm parameters", {"llm": {"parameters": {"z": "y"}}}), ("tools", {"tools": [{"type": "ListIndexTool"}]}),
    ("empty tools", {"tools": []}), ("bad tool", {"tools": [{"name": "x"}]}), ("parameters", {"parameters": {"n": 1}}),
    ("app type", {"app_type": "A1"}), ("memory", {"memory": {"type": "conversation_index", "session_id": "s2"}}),
    ("memory window", {"memory": {"window_size": 9}}), ("bad memory", {"memory": {"type": "bogus"}}),
    ("agentic memory", {"memory": {"type": "agentic_memory"}}), ("blank name", {"name": ""}),
    ("upper case type", {"type": "CONVERSATIONAL"}), ("hidden", {"is_hidden": True}), ("unknown", {"zzz": 1}), ("empty", {}),
]


def mask(o, ids):
    text = json.dumps(o, sort_keys=True)
    for i in ids:
        text = text.replace(i, "<id>")
    o = json.loads(text)

    def drop(x):
        if isinstance(x, dict):
            return {k: drop(v) for k, v in x.items() if k not in DROP}
        if isinstance(x, list):
            return [drop(v) for v in x]
        return x
    return drop(o)


def answer(n, method, path, body, ids):
    s, o = n.call(method, path, body)
    if s != 200 and isinstance(o, dict) and "error" in o:
        o = o["error"].get("reason", o["error"])
    return {"status": s, "body": mask(o, ids)}


def agents(c, ours, stock):
    registry(c, ours, stock)
    execution(c, ours, stock)
    memory(c, ours, stock)


def registry(c, ours, stock):
    ids = {"ours": [], "stock": []}
    made = {}
    for label, body in REGISTER.items():
        got = {}
        for side, n in (("ours", ours), ("stock", stock)):
            s, o = n.call("POST", "/_plugins/_ml/agents/_register", body)
            got[side] = {"status": s, "body": "agent_id" if s == 200 and list(o) == ["agent_id"] else o}
            if s == 200:
                made[side, label] = o["agent_id"]
                ids[side].append(o["agent_id"])
            elif isinstance(o, dict) and "error" in o:
                got[side]["body"] = o["error"].get("reason", o["error"])
        c.same(f"agents register {label}", got["ours"], got["stock"])
    for side, n in (("ours", ours), ("stock", stock)):
        if (side, "plan") in made:
            s, o = n.call("GET", f"/_plugins/_ml/agents/{made[side, 'plan']}")
            ids[side].append(o["parameters"]["executor_agent_id"])
    for label in REGISTER:
        if ("ours", label) in made and ("stock", label) in made:
            got = {side: answer(n, "GET", f"/_plugins/_ml/agents/{made[side, label]}", None, ids[side])
                   for side, n in (("ours", ours), ("stock", stock))}
            c.same(f"agents get {label}", got["ours"], got["stock"])
    target = {side: made[side, "full"] for side in ("ours", "stock")}
    for label, body in UPDATES:
        got = {}
        for side, n in (("ours", ours), ("stock", stock)):
            s, o = n.call("PUT", f"/_plugins/_ml/agents/{target[side]}", body)
            got[side] = {"status": s, "body": o["error"].get("reason") if s != 200 else o.get("result")}
        c.same(f"agents update {label}", got["ours"], got["stock"])
        got = {side: answer(n, "GET", f"/_plugins/_ml/agents/{target[side]}", None, ids[side])
               for side, n in (("ours", ours), ("stock", stock))}
        c.same(f"agents after update {label}", got["ours"], got["stock"])
    up = {}
    for side, n in (("ours", ours), ("stock", stock)):
        up[side] = made[side, "upper case type"]
    for body in ({"type": "flow"}, {"type": "FLOW"}, {"type": "conversational"}):
        got = {}
        for side, n in (("ours", ours), ("stock", stock)):
            s, o = n.call("PUT", f"/_plugins/_ml/agents/{up[side]}", body)
            got[side] = {"status": s, "body": o["error"].get("reason") if s != 200 else o.get("result")}
        c.same(f"agents update upper case agent to {body['type']}", got["ours"], got["stock"])
    got = {side: answer(n, "PUT", "/_plugins/_ml/agents/nope", {"description": "x"}, ids[side])
           for side, n in (("ours", ours), ("stock", stock))}
    c.same("agents update missing", got["ours"], got["stock"])
    got = {side: answer(n, "GET", "/_plugins/_ml/agents/nope", None, ids[side]) for side, n in (("ours", ours), ("stock", stock))}
    c.same("agents get missing", got["ours"], got["stock"])
    for label, q in {"all": {"query": {"match_all": {}}, "size": 100, "sort": [{"name.keyword": "asc"}]},
                     "by name": {"query": {"term": {"name.keyword": "r-flow"}}},
                     "by type": {"query": {"term": {"type": "flow"}}, "size": 0}}.items():
        got = {}
        for side, n in (("ours", ours), ("stock", stock)):
            s, o = n.call("POST", "/_plugins/_ml/agents/_search", q)
            hits = o["hits"]
            got[side] = {"status": s, "total": hits["total"],
                         "sources": sorted(json.dumps(mask(h["_source"], ids[side]), sort_keys=True) for h in hits["hits"])}
        c.same(f"agents search {label}", got["ours"], got["stock"])
    for side, n in (("ours", ours), ("stock", stock)):
        pass
    got = {side: answer(n, "DELETE", f"/_plugins/_ml/agents/{made[side, 'flow']}", None, ids[side])
           for side, n in (("ours", ours), ("stock", stock))}
    for g in got.values():
        g["body"] = {k: v for k, v in g["body"].items() if k in ("result", "forced_refresh")} if isinstance(g["body"], dict) else g["body"]
    c.same("agents delete", got["ours"], got["stock"])
    got = {side: answer(n, "DELETE", f"/_plugins/_ml/agents/{made[side, 'flow']}", None, ids[side])
           for side, n in (("ours", ours), ("stock", stock))}
    c.same("agents delete again", got["ours"], got["stock"])

    def mine(n):
        s, o = n.call("GET", "/_plugins/_ml/tools")
        return {t["name"]: t for t in o if t["name"] in ML_TOOLS}
    c.same("tools list", mine(ours), mine(stock))
    for name in ML_TOOLS + ["Nope"]:
        got = {side: answer(n, "GET", f"/_plugins/_ml/tools/{name}", None, []) for side, n in (("ours", ours), ("stock", stock))}
        c.same(f"tools get {name}", got["ours"], got["stock"])


DOCS = [("a", {"t": "hello world", "n": 1}), ("b", {"t": "second doc", "n": 2}), ("c", {"t": "third \"quoted\" <doc> & more", "n": 3, "o": {"p": [1, 2.5, True, None]}})]
INDICES = {
    "ag1": {"settings": {"number_of_replicas": 1}},
    "ag2": {"settings": {"number_of_replicas": 0}, "mappings": {"properties": {"k": {"type": "keyword"}, "d": {"type": "date"}}}},
    "ag3": {"settings": {"number_of_replicas": 0, "refresh_interval": "5s"},
            "mappings": {"dynamic": "strict", "_meta": {"v": 1}, "properties": {"o": {"properties": {"z": {"type": "keyword"}, "y": {"type": "date"}}},
                                                                                "a": {"type": "text", "analyzer": "standard"}}}},
}


def text_mask(t, ids):
    """A tool's text without what differs between nodes: ids, uuids, times, sizes."""
    for i in ids:
        t = t.replace(i, "<id>")
    t = re.sub(r"START_OBJECT at 1:\d+", "START_OBJECT at 1:N", t)
    t = re.sub(r"index\.(creation_date|uuid)=[A-Za-z0-9_-]*", r"index.\1=<x>", t)
    lines = []
    for line in t.split("\n"):
        cols = line.split(",")
        if len(cols) == 11 and cols[0].isdigit():
            cols[0], cols[4], cols[9], cols[10] = "<row>", "<uuid>", "<size>", "<size>"
        lines.append(",".join(cols))
    if lines and lines[0].startswith("row,health"):
        rows = sorted(x for x in lines[1:] if x)
        lines = [lines[0]] + rows + [""]
    return "\n".join(lines)


def deep(x, ids):
    if isinstance(x, str):
        return text_mask(x, ids)
    if isinstance(x, list):
        return [deep(v, ids) for v in x]
    if isinstance(x, dict):
        return {k: deep(v, ids) for k, v in x.items()}
    return x


def run_mask(o, ids):
    if isinstance(o, dict) and "inference_results" in o:
        o = deep(o, ids)
    return mask(o, ids)


def mock_server():
    spec = importlib.util.spec_from_file_location("remote_check", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml-remote-check.py"))
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    try:
        rc.serve_mock()
    except OSError:
        pass  # the remote section's mock is still serving
    return rc


def execution(c, ours, stock):
    rc = mock_server()
    side = (("ours", ours), ("stock", stock))
    ids = {"ours": [], "stock": []}
    fx = {"ours": {}, "stock": {}}
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.trusted_connector_endpoints_regex": [".*"],
                                                          "plugins.ml_commons.connector.private_ip_enabled": True}})
        for ix, body in INDICES.items():
            n.call("DELETE", f"/{ix}")
            n.ok("PUT", f"/{ix}", body)
        for ix in INDICES:
            for did, doc in DOCS:
                n.call("PUT", f"/{ix}/_doc/{did}?refresh=true", doc)
        con = rc.connector("agent-conn", "http", {"k": "v"}, [rc.act(f"http://127.0.0.1:{rc.MOCK}/gen/x", "{\"q\": \"${parameters.question}\"}")], {"a": "b"})
        fx[name]["connector"] = n.ok("POST", "/_plugins/_ml/connectors/_create", con)["connector_id"]
        fx[name]["model"] = n.ok("POST", "/_plugins/_ml/models/_register?deploy=true", {"name": "agent-model", "function_name": "remote", "connector": con})["model_id"]
        n.ok("POST", "/_plugins/_ml/agents/_register", {"name": "inner", "type": "flow", "tools": [{"type": "ListIndexTool", "parameters": {"indices": ["ag2"]}}]})
        s, o = n.call("POST", "/_plugins/_ml/agents/_search", {"query": {"term": {"name.keyword": "inner"}}})
        fx[name]["inner"] = o["hits"]["hits"][0]["_id"]
        ids[name] += list(fx[name].values())
    time.sleep(3)

    def agent(label, body, tools=None):
        got = {}
        for name, n in side:
            b = json.loads(json.dumps(body).replace("<connector>", fx[name]["connector"]).replace("<model>", fx[name]["model"]).replace("<inner>", fx[name]["inner"]))
            s, o = n.call("POST", "/_plugins/_ml/agents/_register", b)
            got[name] = o["agent_id"] if s == 200 else None
            if s == 200:
                ids[name].append(o["agent_id"])
        return got

    def execute(label, body, request, query=""):
        made = agent(label, body)
        if None in made.values():
            return None
        got = {}
        for name, n in side:
            s, o = n.call("POST", f"/_plugins/_ml/agents/{made[name]}/_execute{query}", json.loads(json.dumps(request).replace("<connector>", fx[name]["connector"]).replace("<model>", fx[name]["model"])))
            got[name] = {"status": s, "body": run_mask(o, ids[name])}
        c.same(f"agents execute {label}", got["ours"], got["stock"])
        return made

    counter = [0]

    def flow(label, tools, request, type="flow", **extra):
        counter[0] += 1
        return execute(label, dict({"name": f"e{counter[0]}", "type": type, "tools": tools}, **extra), request)

    def one(label, tool, request, **kw):
        return flow(label, [tool], request, **kw)

    LI, IM, SI = "ListIndexTool", "IndexMappingTool", "SearchIndexTool"
    for label, p in {"explicit": {"indices": ["ag1"]}, "two": {"indices": ["ag1", "ag2"]}, "wildcard": {"indices": ["ag*"]}, "missing": {"indices": ["nope"]},
                     "missing wildcard": {"indices": ["nope*"]}, "bare": {"indices": "ag1"}, "text array": {"indices": "[\"ag2\"]"},
                     "local": {"indices": ["ag1"], "local": "true"}, "number": {"indices": 5}, "object": {"indices": {"a": 1}}}.items():
        one(f"list {label}", {"type": LI}, {"parameters": p})
    for label, p in {"explicit": {"index": ["ag1"]}, "strict": {"index": ["ag3"]}, "keyword": {"index": ["ag2"]}, "two": {"index": ["ag1", "ag2"]},
                     "wildcard": {"index": ["ag*"]}, "missing": {"index": ["nope"]}, "empty": {"index": []}, "absent": {}, "bare": {"index": "ag1"}}.items():
        one(f"mapping {label}", {"type": IM}, {"parameters": p})
    q = lambda index, query: json.dumps({"index": index, "query": query})
    for label, p in {
        "match all": {"input": q("ag1", {"query": {"match_all": {}}})},
        "size zero": {"input": q("ag1", {"query": {"match_all": {}}, "size": 0})},
        "size one": {"input": q("ag1", {"query": {"match_all": {}}, "size": 1})},
        "match": {"input": q("ag1", {"query": {"match": {"t": "doc"}}})},
        "sorted": {"input": q("ag1", {"query": {"match_all": {}}, "sort": [{"n": "desc"}]})},
        "aggs only": {"input": q("ag1", {"size": 1, "aggs": {"a": {"max": {"field": "n"}}}})},
        "source filter": {"input": q("ag1", {"query": {"match_all": {}}, "_source": ["n"]})},
        "unknown query": {"input": q("ag1", {"query": {"zzz": {}}})},
        "missing index": {"input": q("nope", {"query": {"match_all": {}}})},
        "wildcard": {"input": q("ag*", {"query": {"term": {"n": 2}}})},
        "no index": {"input": json.dumps({"query": {"query": {"match_all": {}}}})},
        "no query": {"input": json.dumps({"index": "ag1"})},
        "neither": {}, "bare word": {"input": "x"}, "array": {"input": "[1]"}, "number": {"input": "5"}, "null": {"input": "null"},
        "query text": {"input": json.dumps({"index": "ag1", "query": "{\"query\":{\"match_all\":{}},\"size\":1}"})},
    }.items():
        one(f"search {label}", {"type": SI}, {"parameters": p})
    one("model", {"type": "MLModelTool", "parameters": {"model_id": "<model>"}}, {"parameters": {"question": "hi <there> & 'you'"}})
    one("model tool parameter", {"type": "MLModelTool", "parameters": {"model_id": "<model>", "question": "tool"}}, {"parameters": {}})
    one("model response field", {"type": "MLModelTool", "parameters": {"model_id": "<model>", "response_field": "echo"}}, {"parameters": {"question": "hi"}})
    one("model missing response field", {"type": "MLModelTool", "parameters": {"model_id": "<model>", "response_field": "nope"}}, {"parameters": {"question": "hi"}})
    one("model nested value", {"type": "MLModelTool", "parameters": {"model_id": "<model>", "response_field": "nested"}}, {"parameters": {"question": "hi"}})
    one("model no id", {"type": "MLModelTool"}, {"parameters": {"question": "hi"}})
    one("model blank id", {"type": "MLModelTool", "parameters": {"model_id": " "}}, {"parameters": {}})
    one("model unknown", {"type": "MLModelTool", "parameters": {"model_id": "zz"}}, {"parameters": {"question": "hi"}})
    one("model id in request", {"type": "MLModelTool"}, {"parameters": {"model_id": "<model>", "question": "hi"}})
    CT = {"type": "ConnectorTool", "parameters": {"connector_id": "<connector>", "connector_action": "predict"}}
    one("connector", CT, {"parameters": {"question": "hi"}})
    one("connector newline", CT, {"parameters": {"question": "a\nb\"c\\d"}})
    one("connector bool and number", CT, {"parameters": {"question": True}})
    one("connector number", CT, {"parameters": {"question": 1.50}})
    one("connector null value", CT, {"parameters": {"question": None}})
    one("connector object value", CT, {"parameters": {"question": {"a": 1}}})
    one("connector unfilled", CT, {"parameters": {"question": "a ${parameters.nope} b"}})
    one("connector no action", {"type": "ConnectorTool", "parameters": {"connector_id": "<connector>"}}, {"parameters": {"question": "hi"}})
    one("connector bad action", {"type": "ConnectorTool", "parameters": {"connector_id": "<connector>", "connector_action": "zz"}}, {"parameters": {}})
    one("connector missing", {"type": "ConnectorTool", "parameters": {"connector_id": "zz", "connector_action": "predict"}}, {"parameters": {}})
    one("connector blank", {"type": "ConnectorTool", "parameters": {"connector_id": " "}}, {"parameters": {}})
    one("agent tool", {"type": "AgentTool", "parameters": {"agent_id": "<inner>"}}, {"parameters": {}})
    one("agent tool missing", {"type": "AgentTool", "parameters": {"agent_id": "zz"}}, {"parameters": {}})
    one("agent tool blank", {"type": "AgentTool"}, {"parameters": {}})
    one("scratch write", {"type": "WriteToScratchPadTool"}, {"parameters": {"notes": "a\nb"}})
    one("scratch write blank", {"type": "WriteToScratchPadTool"}, {"parameters": {"notes": "  "}})
    one("scratch write missing", {"type": "WriteToScratchPadTool"}, {"parameters": {}})
    one("scratch read", {"type": "ReadFromScratchPadTool"}, {"parameters": {"notes": "a"}})
    flow("scratch write then read", [{"type": "WriteToScratchPadTool", "name": "w"}, {"type": "ReadFromScratchPadTool", "name": "r"}], {"parameters": {"notes": "x"}})
    flow("two unnamed", [{"type": LI, "parameters": {"indices": ["ag1"]}}, {"type": IM, "parameters": {"index": ["ag2"]}}], {"parameters": {}})
    flow("two named", [{"type": LI, "name": "li", "parameters": {"indices": ["ag1"]}}, {"type": IM, "name": "im", "parameters": {"index": ["ag2"]}}], {"parameters": {}})
    flow("included", [{"type": LI, "name": "li", "include_output_in_agent_response": True, "parameters": {"indices": ["ag1"]}},
                      {"type": IM, "name": "im", "parameters": {"index": ["ag2"]}}], {"parameters": {}})
    flow("included last only", [{"type": LI, "name": "li", "parameters": {"indices": ["ag1"]}},
                                {"type": IM, "name": "im", "include_output_in_agent_response": True, "parameters": {"index": ["ag2"]}}], {"parameters": {}})
    flow("all included", [{"type": LI, "name": "li", "include_output_in_agent_response": True, "parameters": {"indices": ["ag1"]}},
                          {"type": IM, "name": "im", "include_output_in_agent_response": True, "parameters": {"index": ["ag2"]}},
                          {"type": SI, "name": "si", "include_output_in_agent_response": True,
                           "parameters": {"input": q("ag2", {"query": {"match_all": {}}})}}], {"parameters": {}})
    flow("first fails", [{"type": LI, "parameters": {"indices": ["nope"]}}, {"type": IM, "parameters": {"index": ["ag2"]}}], {"parameters": {}})
    flow("chain output", [{"type": IM, "name": "im", "parameters": {"index": ["ag2"]}},
                          {"type": "ConnectorTool", "name": "ct", "parameters": {"connector_id": "<connector>", "connector_action": "predict", "question": "got:${parameters.im.output}"}}],
         {"parameters": {}})
    flow("chain search output", [{"type": SI, "name": "si", "parameters": {"input": q("ag1", {"query": {"match_all": {}}})}},
                                 {"type": "ConnectorTool", "name": "ct", "parameters": {"connector_id": "<connector>", "connector_action": "predict", "question": "${parameters.si.output}"}}],
         {"parameters": {}})
    flow("chain request wins", [{"type": LI, "name": "li", "parameters": {"indices": ["ag1"]}},
                                {"type": "ConnectorTool", "parameters": {"connector_id": "<connector>", "connector_action": "predict", "question": "tool"}}], {"parameters": {"question": "request"}})
    one("request over tool", {"type": LI, "parameters": {"indices": ["nope"]}}, {"parameters": {"indices": ["ag1"]}})
    one("tool over agent", {"type": LI, "parameters": {"indices": ["nope"]}}, {"parameters": {}}, parameters={"indices": ["ag1"]})
    one("request over agent", {"type": LI}, {"parameters": {"indices": ["ag1"]}}, parameters={"indices": ["nope"]})
    one("prefix by name", {"type": LI, "name": "li"}, {"parameters": {"li.indices": ["ag1"]}})
    one("prefix by type", {"type": LI}, {"parameters": {"ListIndexTool.indices": ["ag1"]}})
    one("prefix other tool", {"type": LI, "name": "li", "parameters": {"indices": ["ag1"]}}, {"parameters": {"other.indices": ["nope"]}})
    one("parameters null", {"type": IM, "parameters": {"index": ["ag1"]}}, {"parameters": None})
    one("parameters text", {"type": IM, "parameters": {"index": ["ag1"]}}, {"parameters": "x"})
    one("parameters number", {"type": IM, "parameters": {"index": ["ag1"]}}, {"parameters": 5})
    one("no parameters key", {"type": IM, "parameters": {"index": ["ag1"]}}, {})
    one("other input", {"type": IM, "parameters": {"index": ["ag1"]}}, {"text_docs": ["a"]})
    one("conversational flow", {"type": IM, "parameters": {"index": ["ag2"]}}, {"parameters": {"question": "q"}}, type="conversational_flow")
    one("conversational flow with memory", {"type": IM, "name": "im", "parameters": {"index": ["ag2"]}}, {"parameters": {"question": "q"}}, type="conversational_flow",
        memory={"type": "conversation_index"})
    one("conversational flow without question", {"type": IM, "parameters": {"index": ["ag2"]}}, {"parameters": {}}, type="conversational_flow")
    one("conversational flow connector", CT, {"parameters": {"question": "q"}}, type="conversational_flow")
    flow("no tools", [], {"parameters": {}})
    execute("no tools field", {"name": "e-nt", "type": "flow"}, {"parameters": {}})
    execute("no tools field conversational flow", {"name": "e-nt2", "type": "conversational_flow"}, {"parameters": {"question": "q"}})
    got = {}
    for name, n in side:
        s, o = n.call("POST", "/_plugins/_ml/agents/nope/_execute", {"parameters": {}})
        got[name] = {"status": s, "body": o}
    c.same("agents execute missing agent", got["ours"], got["stock"])
    got = {}
    for name, n in side:
        s, o = n.call("POST", "/_plugins/_ml/agents/nope/_execute", {})
        got[name] = {"status": s, "body": o}
    c.same("agents execute missing agent without parameters", got["ours"], got["stock"])
    made = agent("body required", {"name": "e-br", "type": "flow", "tools": [{"type": IM, "parameters": {"index": ["ag1"]}}]})
    got = {}
    for name, n in side:
        s, o = n.call("POST", f"/_plugins/_ml/agents/{made[name]}/_execute", None)
        got[name] = {"status": s, "body": o}
    c.same("agents execute without a body", got["ours"], got["stock"])
    # async: a task, then the task's document.
    for label, tools in {"async ok": [{"type": IM, "parameters": {"index": ["ag2"]}}],
                         "async fails": [{"type": SI, "parameters": {}}],
                         "async missing index": [{"type": LI, "parameters": {"indices": ["nope"]}}],
                         "async two tools": [{"type": IM, "name": "im", "parameters": {"index": ["ag2"]}}, {"type": "ConnectorTool", "parameters": {"connector_id": "<connector>", "connector_action": "predict", "question": "z"}}]}.items():
        counter[0] += 1
        made = agent(label, {"name": f"e{counter[0]}", "type": "flow", "tools": tools})
        got = {}
        for name, n in side:
            s, o = n.call("POST", f"/_plugins/_ml/agents/{made[name]}/_execute?async=true", {"parameters": {}})
            body = {"status": s, "keys": sorted(o), "state": o.get("status")}
            task = None
            if s == 200:
                task = n.task(o["task_id"])
                ids[name].append(o["task_id"])
                task = {k: v for k, v in task.items() if k not in ("create_time", "last_update_time", "worker_node")}
                task["response"] = run_mask(task.get("response"), ids[name])
            got[name] = {"start": body, "task": task}
        c.same(f"agents execute {label}", got["ours"], got["stock"])
    got = {}
    for name, n in side:
        s, o = n.call("POST", "/_plugins/_ml/agents/nope/_execute?async=true", {"parameters": {}})
        got[name] = {"status": s, "body": o}
    c.same("agents execute async missing agent", got["ours"], got["stock"])


def memory(c, ours, stock):
    side = (("ours", ours), ("stock", stock))
    ids = {"ours": [], "stock": []}
    M = "/_plugins/_ml/memory"
    for name, n in side:
        s, o = n.call("POST", M + "/_search", {"query": {"match_all": {}}, "size": 100})
        for h in (o.get("hits", {}).get("hits", []) if s == 200 else []):
            n.call("DELETE", f"{M}/{h['_id']}")

    def both(label, method, path, body=None, keep=True):
        """Runs one call on both nodes (path may use {m0}, {m1}, ... for memories, {g0}... for messages)."""
        got = {}
        for name, n in side:
            p = path.format(**{k: v[name] for k, v in made.items()})
            b = body
            if body is not None:
                b = json.loads(json.dumps(body).replace("{m2}", made["m2"][name]) if "m2" in made else json.dumps(body))
            s, o = n.call(method, p, b)
            if isinstance(o, dict):
                for key in ("memory_id", "message_id"):
                    if key in o and keep:
                        ids[name].append(o[key])
                if "hits" in o:
                    o = {"total": o["hits"]["total"], "hits": [{k: (v is not None if k == "_score" else v) for k, v in h.items() if k not in ("_version", "_seq_no", "_primary_term", "sort")} for h in o["hits"]["hits"]]}
            got[name] = {"status": s, "body": run_mask(o, ids[name])}
            got[name] = json.loads(re.sub(r"START_OBJECT at 1:\d+", "START_OBJECT at 1:N", json.dumps(got[name])))
        c.same(f"memory {label}", got["ours"], got["stock"])
        return got

    made = {}

    def make(key, kind, path, body):
        got = {}
        for name, n in side:
            p = path.format(**{k: v[name] for k, v in made.items()})
            s, o = n.call("POST", p, body)
            if s != 200:
                return False
            got[name] = o[kind]
            ids[name].append(o[kind])
        made[key] = got
        return True

    for label, body in {"empty": None, "named": {"name": "c1"}, "application": {"name": "c2", "application_type": "chat", "additional_info": {"a": 1, "b": {"c": "d"}}},
                        "unknown field": {"zzz": 1}, "name number": {"name": 5}, "name object": {"name": {"a": 1}}, "info text": {"additional_info": "x"},
                        "application number": {"application_type": 5}}.items():
        both(f"create {label}", "POST", M, body)
        # Only the answers' shape is compared: each node makes its own ids.
    for n_ in ("m0", "m1", "m2"):
        make(n_, "memory_id", M, {"name": n_, "additional_info": {"k": n_}})
    both("get", "GET", M + "/{m0}")
    both("get with info", "GET", M + "/{m1}")
    both("get missing", "GET", M + "/nope")
    for label, body in [("name", {"name": "renamed"}), ("empty", {}), ("unknown", {"zzz": 1}), ("info", {"additional_info": {"x": 1}}), ("number", {"name": 5}),
                        ("application", {"application_type": "chat"}), ("blank name", {"name": ""})]:
        both(f"update {label}", "PUT", M + "/{m0}", body)
        both(f"get after update {label}", "GET", M + "/{m0}")
    both("update missing", "PUT", M + "/nope", {"name": "x"})
    both("update no body", "PUT", M + "/{m0}")
    # messages
    msgs = [("input", {"input": "hi"}), ("template", {"prompt_template": "t"}), ("response", {"response": "r"}), ("origin", {"origin": "o"}),
            ("info", {"additional_info": {"k": "v", "n": 1}}), ("all", {"input": "hi", "prompt_template": "t", "response": "hello", "origin": "o", "additional_info": {"k": "v"}}),
            ("unicode", {"input": "h\u00e9llo \u4e16\u754c \"q\" <b>", "response": "x"})]
    for i, (label, body) in enumerate(msgs):
        if make(f"g{i}", "message_id", M + "/{m0}/messages", body):
            both(f"message {label}", "GET", M + "/message/{g%d}" % i)
    for label, body in {"empty": {}, "blank strings": {"input": "", "response": ""}, "empty info": {"additional_info": {}}, "unknown only": {"zzz": 1},
                        "input number": {"input": 5}, "input object": {"input": {"a": 1}}, "info text": {"additional_info": "x"}, "trace text": {"input": "a", "trace_number": "1"},
                        "trace number float": {"input": "a", "trace_number": 1.5}, "parent number": {"input": "a", "parent_message_id": 5}}.items():
        both(f"add message {label}", "POST", M + "/{m1}/messages", body)
    both("add message to missing memory", "POST", M + "/nope/messages", {"input": "x"})
    both("add message without body", "POST", M + "/{m1}/messages")
    # traces
    make("t0", "message_id", M + "/{m2}/messages", {"input": "q", "response": "r"})
    for i, (num, extra) in enumerate([(2, "b"), (1, "a"), (3, "c"), (1, "again"), (None, "no number")]):
        body = {"input": extra, "parent_message_id": made["t0"]["ours"] if False else "PARENT"}
        got = {}
        for name, n in side:
            b = {"input": extra, "parent_message_id": made["t0"][name]}
            if num is not None:
                b["trace_number"] = num
            s, o = n.call("POST", f"{M}/{made['m2'][name]}/messages", b)
            ids[name].append(o.get("message_id"))
            got[name] = s
        c.same(f"memory add trace {i}", got["ours"], got["stock"])
    both("traces", "GET", M + "/message/{t0}/traces")
    both("traces max", "GET", M + "/message/{t0}/traces?max_results=2")
    both("traces next", "GET", M + "/message/{t0}/traces?max_results=2&next_token=2")
    both("traces of a message without any", "GET", M + "/message/{g0}/traces")
    both("traces missing", "GET", M + "/message/nope/traces")
    both("trace message get", "GET", M + "/message/{t0}")
    both("messages list excludes traces", "GET", M + "/{m2}/messages")
    # list
    both("messages list", "GET", M + "/{m0}/messages")
    both("messages list max", "GET", M + "/{m0}/messages?max_results=2")
    both("messages list next", "GET", M + "/{m0}/messages?max_results=2&next_token=2")
    both("messages list exact page", "GET", M + "/{m0}/messages?max_results=7")
    both("messages list past end", "GET", M + "/{m0}/messages?next_token=50")
    both("messages list zero", "GET", M + "/{m0}/messages?max_results=0")
    both("messages list negative start", "GET", M + "/{m0}/messages?next_token=-1")
    both("messages list both bad", "GET", M + "/{m0}/messages?max_results=-1&next_token=-1")
    both("messages list text", "GET", M + "/{m0}/messages?max_results=x")
    both("messages list missing memory", "GET", M + "/nope/messages")
    both("memories list", "GET", M)
    both("memories list max", "GET", M + "?max_results=2")
    both("memories list next", "GET", M + "?max_results=2&next_token=2")
    both("memories list zero", "GET", M + "?max_results=0")
    both("memories list negative", "GET", M + "?next_token=-1")
    # message update and get
    for label, body in [("response", {"response": "new"}), ("info", {"additional_info": {"z": 1}}), ("input", {"input": "changed"}), ("template", {"prompt_template": "pt"}),
                        ("origin", {"origin": "oo"}), ("empty", {}), ("unknown", {"zzz": 1}), ("number", {"response": 5}), ("trace", {"trace_number": 4})]:
        both(f"update message {label}", "PUT", M + "/message/{g0}", body)
        both(f"message after update {label}", "GET", M + "/message/{g0}")
    both("update message info merges", "PUT", M + "/message/{g0}", {"additional_info": {"q": "w"}})
    both("message after info merge", "GET", M + "/message/{g0}")
    both("update message without body", "PUT", M + "/message/{g0}")
    both("update missing message", "PUT", M + "/message/nope", {"response": "x"})
    both("get missing message", "GET", M + "/message/nope")
    both("get message after memory update", "GET", M + "/{m0}")
    # search
    both("search memories", "POST", M + "/_search", {"query": {"match_all": {}}, "sort": [{"name.keyword": "asc"}] if False else [{"create_time": "asc"}], "size": 100})
    both("search memories by name", "POST", M + "/_search", {"query": {"match": {"name": "m1"}}})
    both("search memories by info", "POST", M + "/_search", {"query": {"term": {"additional_info.k": "m1"}}})
    both("search memories without body", "POST", M + "/_search")
    both("search messages", "POST", M + "/{m0}/_search", {"query": {"match_all": {}}, "sort": [{"create_time": "asc"}], "size": 100})
    both("search messages by text", "POST", M + "/{m0}/_search", {"query": {"match": {"input": "hi"}}, "sort": [{"create_time": "asc"}]})
    both("search messages scored", "POST", M + "/{m0}/_search", {"query": {"match": {"input": "hi"}}})
    both("search messages no query", "POST", M + "/{m0}/_search", {"size": 100, "sort": [{"create_time": "asc"}]})
    both("search messages includes traces", "POST", M + "/{m2}/_search", {"query": {"match_all": {}}, "sort": [{"create_time": "asc"}], "size": 100})
    both("search messages without body", "POST", M + "/{m0}/_search")
    both("search messages get without body", "GET", M + "/{m0}/_search")
    both("search messages missing memory", "POST", M + "/nope/_search", {"query": {"match_all": {}}})
    # delete
    both("delete", "DELETE", M + "/{m2}")
    both("delete again", "DELETE", M + "/{m2}")
    both("get deleted", "GET", M + "/{m2}")
    both("message of a deleted memory", "GET", M + "/message/{t0}")
    both("trace of a deleted memory", "GET", M + "/message/{t0}/traces")
    both("memory messages index after delete", "POST", "/.plugins-ml-memory-message/_search", {"query": {"term": {"memory_id": "{m2}"}}, "size": 0})
    both("delete with messages", "DELETE", M + "/{m0}")
    both("delete missing", "DELETE", M + "/nope")
    both("memories list after delete", "GET", M)
