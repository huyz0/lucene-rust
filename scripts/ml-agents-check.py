"""The `agents` section of check-ml-plugin.py: the agent registry and the tool catalogue.

Registers, reads, updates, searches and deletes agents on both nodes and
compares the answers (agent ids, times and index versions masked), then
lists and reads the tools ml-commons itself registers. A stock node with
other plugins lists their tools too; only the names both sides know from
ml-commons are compared. Then runs flow and conversational_flow agents with
the tools that need no language model, against indices both nodes hold, a
remote model and a connector on the mock model server of the `remote`
section, sync and with `?async=true`. Then the conversation memory API:
memories, messages and traces, their listing, search and deletion. Last, conversational agents
on a scripted model: what they ask it, the tools they run, and what they write to memory.
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
    containers(c, ours, stock)
    chat(c, ours, stock)
    agui(c, ours, stock)


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


LLM_PORT = 8768


STYLES = {
    "openai": "openai/v1/chat/completions",
    "bedrock": "bedrock/converse/claude",
    "gemini": "gemini/v1beta/generatecontent",
}
BODIES = {
    "openai": ('{ "model": "${parameters.model}", "messages": [{"role":"developer","content":"${parameters.system_prompt}"},${parameters._chat_history:-}'
               '{"role":"user","content":"${parameters.question}"}${parameters._interactions:-}]${parameters.tool_configs:-} }'),
    "bedrock": ('{"system":[{"text":"${parameters.system_prompt}"}],"messages":[${parameters._chat_history:-}'
                '{"role":"user","content":[{"text":"${parameters.question}"}]}${parameters._interactions:-}]${parameters.tool_configs:-}}'),
    "gemini": ('{"system_instruction":{"parts":[{"text":"${parameters.system_prompt}"}]},"contents":[${parameters._chat_history:-}'
               '{"role":"user","parts":[{"text":"${parameters.question}"}]}${parameters._interactions:-}]${parameters.tool_configs:-}}'),
}


CONTAINERS = {
    "plain": {"name": "mc"}, "description": {"name": "mc", "description": "d"}, "no name": {"description": "d"}, "empty name": {"name": ""},
    "name number": {"name": 5}, "prefix": {"name": "mc", "configuration": {"index_prefix": "pfx"}}, "blank prefix": {"name": "mc", "configuration": {"index_prefix": " "}},
    "bad prefix": {"name": "mc", "configuration": {"index_prefix": "a/b"}}, "user index": {"name": "mc", "configuration": {"index_prefix": "usr", "use_system_index": False}},
    "no session": {"name": "mc", "configuration": {"index_prefix": "nos", "disable_session": True}}, "no history": {"name": "mc", "configuration": {"disable_history": True}},
    "infer size": {"name": "mc", "configuration": {"max_infer_size": 5}}, "big infer size": {"name": "mc", "configuration": {"max_infer_size": 11}},
    "zero infer size": {"name": "mc", "configuration": {"max_infer_size": 0}}, "roles": {"name": "mc", "backend_roles": ["a", "b"]},
    "strategies": {"name": "mc", "configuration": {"strategies": [{"type": "SEMANTIC", "namespace": ["user_id"]}]}},
    "strategies bad type": {"name": "mc", "configuration": {"strategies": [{"type": "zzz", "namespace": ["u"]}]}}, "strategies no namespace": {"name": "mc", "configuration": {"strategies": [{"type": "SEMANTIC"}]}},
    "llm missing": {"name": "mc", "configuration": {"llm_id": "nope"}}, "embedding no dimension": {"name": "mc", "configuration": {"embedding_model_id": "x", "embedding_model_type": "TEXT_EMBEDDING"}},
    "embedding missing": {"name": "mc", "configuration": {"embedding_model_id": "x", "embedding_model_type": "TEXT_EMBEDDING", "embedding_dimension": 4}},
    "extra field": {"name": "mc", "zzz": 1}, "configuration extra": {"name": "mc", "configuration": {"zzz": 1, "parameters": {"a": 1}}},
}

CONTAINER_UPDATES = [
    ("name", {"name": "n2"}), ("name number", {"name": 7}), ("description", {"description": "d2"}), ("roles", {"backend_roles": ["x"]}),
    ("infer size", {"configuration": {"max_infer_size": 3}}), ("llm", {"configuration": {"llm_id": "m9"}}), ("llm infer size", {"configuration": {"llm_id": "m8", "max_infer_size": 4}}), ("llm again", {"configuration": {"llm_id": "m7"}}), ("prefix ignored", {"configuration": {"index_prefix": "zz"}}),
    ("disable ignored", {"configuration": {"disable_history": True}}), ("retention", {"retention_policy": {"x": 1}}),
    ("strategies", {"configuration": {"strategies": [{"type": "SEMANTIC"}]}}), ("strategies namespace", {"configuration": {"strategies": [{"type": "SEMANTIC", "namespace": ["u"]}]}}), ("strategies bad type", {"configuration": {"strategies": [{"type": "zzz", "namespace": ["u"]}]}}), ("empty", {}), ("unknown", {"zzz": 1}),
]

MEMORIES = {
    "messages": {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    "messages and session": {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "namespace": {"session_id": "s1", "user_id": "u"}},
    "second in session": {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "yo"}]}], "namespace": {"session_id": "s1"}},
    "data": {"structured_data": {"a": 1}, "payload_type": "DATA"}, "lower payload": {"structured_data": {"a": 1}, "payload_type": "data"},
    "bad payload": {"structured_data": {"a": 1}, "payload_type": "zzz"}, "data string": {"structured_data": "x"},
    "binary": {"binary_data": "AAEC"}, "tags": {"messages": [{"role": "user", "content": []}], "tags": {"a": 1, "b": {"c": [1]}}},
    "metadata": {"structured_data": {"k": 1}, "metadata": {"m": 5, "n": True}}, "namespace values": {"structured_data": {"k": 1}, "namespace": {"n": 5, "o": {"p": 1}}},
    "no role": {"messages": [{"content": []}]}, "no content": {"messages": [{"role": "user"}]}, "messages string": {"messages": "x"},
    "empty messages": {"messages": []}, "empty": {}, "infer": {"messages": [{"role": "user", "content": []}], "infer": True},
    "infer false": {"messages": [{"role": "user", "content": []}], "infer": False}, "infer text": {"messages": [{"role": "user", "content": []}], "infer": "x"},
    "top level session": {"structured_data": {"a": 1}, "session_id": "s7", "owner": "o"}, "extra": {"structured_data": {"a": 1}, "zzz": 1},
}

MEMORY_UPDATES = [
    ("tags", {"tags": {"t": "1"}}), ("metadata", {"metadata": {"m": "2"}}), ("structured data", {"structured_data": {"z": 9}}), ("messages", {"messages": [{"role": "user", "content": []}]}),
    ("bad messages", {"messages": "x"}), ("binary", {"binary_data": "AQ=="}), ("namespace ignored", {"namespace": {"q": "1"}}), ("unknown", {"zzz": 1}), ("empty", {}),
]


def containers(c, ours, stock):
    """Agentic memory containers, the working memories and sessions they hold."""
    side = (("ours", ours), ("stock", stock))
    ids = {"ours": [], "stock": []}
    C = "/_plugins/_ml/memory_containers"

    def both(label, method, path, body=None, made=None, keep=True):
        got = {}
        for name, n in side:
            p = path.format(**{k: v[name] for k, v in (made or {}).items()})
            if "_search" in p:
                n.call("POST", "/.plugins-ml-am-*,default-memory-*,pfx-*,usr-*,nos-*/_refresh")
            s, o = n.call(method, p, body)
            if isinstance(o, dict) and "memory_container_id" in o and keep:
                ids[name].append(o["memory_container_id"])
            if isinstance(o, dict) and "working_memory_id" in o and keep:
                ids[name].append(o["working_memory_id"])
            if isinstance(o, dict) and "hits" in o:
                o = {"total": o["hits"]["total"], "hits": [{k: v for k, v in h.items() if k not in ("_version", "_seq_no", "_primary_term", "sort", "_score")} for h in o["hits"]["hits"]]}
            if s != 200 and isinstance(o, dict) and "error" in o:
                o = {"type": o["error"].get("type"), "reason": o["error"].get("reason"), "caused_by": o["error"].get("caused_by", {}).get("type")}
            got[name] = {"status": s, "body": mask(o, ids[name])}
            if isinstance(got[name]["body"], dict) and isinstance(got[name]["body"].get("hits"), list):
                got[name]["body"]["hits"].sort(key=lambda h: json.dumps({k: v for k, v in h.items() if k != "_source"} | {"_source": {k: v for k, v in h["_source"].items() if k not in ("created_time", "last_updated_time")}}, sort_keys=True))
            got[name] = json.loads(re.sub(r"at 1:\d+", "at 1:N", json.dumps(got[name])))
        c.same(f"containers {label}", got["ours"], got["stock"])
        return got

    def reset():
        for name, n in side:
            s, o = n.call("POST", C + "/_search", {"query": {"match_all": {}}, "size": 100})
            for h in (o.get("hits", {}).get("hits", []) if s == 200 else []):
                n.call("DELETE", f"{C}/{h['_id']}")
            n.call("DELETE", "/.plugins-ml-am-*,default-memory-*,pfx-*,usr-*,nos-*")
            ids[name].clear()

    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.agentic_memory_enabled": False}})
    both("disabled create", "POST", C + "/_create", {"name": "x"})
    both("disabled get", "GET", C + "/x")
    both("disabled search", "POST", C + "/_search", {})
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.agentic_memory_enabled": True}})
    reset()

    for label, body in CONTAINERS.items():
        both(f"create {label}", "POST", C + "/_create", body)
    reset()
    made = {}
    for name, n in side:
        made.setdefault("c", {})[name] = n.ok("POST", C + "/_create", {"name": "base", "description": "d", "backend_roles": ["r"]})["memory_container_id"]
        ids[name].append(made["c"][name])
    both("get", "GET", C + "/{c}", made=made)
    both("get unknown", "GET", C + "/nope")
    for name, n in side:
        made.setdefault("u", {})[name] = n.ok("POST", C + "/_create", {"name": "upd", "description": "d", "backend_roles": ["r"]})["memory_container_id"]
        ids[name].append(made["u"][name])
    for label, body in CONTAINER_UPDATES:
        both(f"update {label}", "PUT", C + "/{u}", body, made=made)
        both(f"get after update {label}", "GET", C + "/{u}", made=made)
    both("update unknown", "PUT", C + "/nope", {"name": "x"})
    both("search", "POST", C + "/_search", {"query": {"match_all": {}}}, made=made)
    both("search get", "GET", C + "/_search", {"query": {"match_all": {}}}, made=made)
    both("search no body", "POST", C + "/_search")
    both("search sorted", "POST", C + "/_search", {"query": {"match": {"name": "n2"}}, "_source": ["name"]}, made=made)

    W = C + "/{c}/memories"
    for label, body in MEMORIES.items():
        got = both(f"add {label}", "POST", W, body, made=made)
        if got["ours"]["status"] == 200:
            both(f"search working after {label}", "POST", W + "/working/_search", {"query": {"match_all": {}}, "size": 100}, made=made)
            both(f"search sessions after {label}", "POST", W + "/sessions/_search", {"query": {"match_all": {}}, "size": 100}, made=made)
    both("add unknown container", "POST", C + "/nope/memories", {"structured_data": {"a": 1}})
    both("add no body", "POST", W, made=made)

    one = {}
    for name, n in side:
        one[name] = n.ok("POST", f"{C}/{made['c'][name]}/memories", {"structured_data": {"a": 1}, "tags": {"t": "0"}, "namespace": {"session_id": "sess9"}})
        ids[name].append(one[name]["working_memory_id"])
    made["m"] = {name: one[name]["working_memory_id"] for name in one}
    made["s"] = {name: "sess9" for name in one}
    both("get working", "GET", W + "/working/{m}", made=made)
    both("get session", "GET", W + "/sessions/{s}", made=made)
    both("get type invalid", "GET", W + "/zzz/{m}", made=made)
    both("get long term", "GET", W + "/long-term/{m}", made=made)
    both("get history", "GET", W + "/history/{m}", made=made)
    both("get missing", "GET", W + "/working/nope", made=made)
    both("get unknown container", "GET", C + "/nope/memories/working/{m}", made=made)
    for label, body in MEMORY_UPDATES:
        both(f"update working {label}", "PUT", W + "/working/{m}", body, made=made)
        both(f"get working after {label}", "GET", W + "/working/{m}", made=made)
    for label, body in [("summary", {"summary": "s"}), ("metadata", {"metadata": {"a": "1"}}), ("agents", {"agents": {"a": 1}}), ("additional", {"additional_info": {"x": "y"}}), ("tags", {"tags": {}}), ("empty", {})]:
        both(f"update session {label}", "PUT", W + "/sessions/{s}", body, made=made)
        both(f"get session after {label}", "GET", W + "/sessions/{s}", made=made)
    both("update missing", "PUT", W + "/working/nope", {"tags": {"a": "1"}}, made=made)
    both("update long term", "PUT", W + "/long-term/x", {"tags": {"a": "1"}}, made=made)
    both("search no body", "POST", W + "/working/_search", made=made)
    both("search long term", "POST", W + "/long-term/_search", {"query": {"match_all": {}}}, made=made)
    both("search invalid type", "POST", W + "/zzz/_search", {"query": {"match_all": {}}}, made=made)
    both("search query", "POST", W + "/working/_search", {"query": {"term": {"tags.t": "0"}}}, made=made)
    both("delete working", "DELETE", W + "/working/{m}", made=made)
    both("delete working again", "DELETE", W + "/working/{m}", made=made)
    both("delete session", "DELETE", W + "/sessions/{s}", made=made)
    both("delete session again", "DELETE", W + "/sessions/{s}", made=made)
    both("delete long term", "DELETE", W + "/long-term/x", made=made)
    both("delete container", "DELETE", C + "/{c}", made=made)
    both("delete container again", "DELETE", C + "/{c}", made=made)
    both("get after delete", "GET", C + "/{c}", made=made)

    for name, n in side:
        n.call("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.agentic_memory_enabled": None}})
    reset()


def agui(c, ours, stock):
    """The agent framework switch, AG-UI run inputs and the streaming endpoint."""
    side = (("ours", ours), ("stock", stock))
    ids = {"ours": [], "stock": []}
    A = "/_plugins/_ml/agents"
    agent = {}
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.ag_ui_enabled": True, "plugins.ml_commons.stream_enabled": True}})
        agent[name] = n.ok("POST", A + "/_register", {"name": "agui", "type": "ag_ui", "llm": {"model_id": "m"}, "tools": [{"type": "ListIndexTool"}]})["agent_id"]
        agent[name + "c"] = n.ok("POST", A + "/_register", {"name": "conv", "type": "conversational", "llm": {"model_id": "m"}, "tools": [{"type": "ListIndexTool"}]})["agent_id"]
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.trusted_connector_endpoints_regex": [".*"], "plugins.ml_commons.connector.private_ip_enabled": True}})
        con = {"name": "llm", "description": "d", "version": "1", "protocol": "http", "parameters": {"endpoint": "x"}, "credential": {"k": "v"},
               "actions": [{"action_type": "predict", "method": "POST", "url": "http://127.0.0.1:1/x", "headers": {"a": "b"}, "request_body": "{}"}]}
        model = n.ok("POST", "/_plugins/_ml/models/_register", {"name": "m", "function_name": "remote", "connector": con})["model_id"]
        agent[name + "r"] = n.ok("POST", A + "/_register", {"name": "real", "type": "ag_ui", "llm": {"model_id": model}, "tools": [{"type": "ListIndexTool"}]})["agent_id"]
        ids[name] += [agent[name], agent[name + "c"], agent[name + "r"], model]

    def both(label, method, path, body=None, conv=False, real=False):
        got = {}
        for name, n in side:
            p = path.replace("{a}", agent[name + ("c" if conv else "r" if real else "")])
            s, o = n.call(method, p, body)
            if isinstance(o, dict) and "error" in o and isinstance(o["error"], dict):
                o = {"type": o["error"].get("type"), "reason": o["error"].get("reason"), "details": o["error"].get("details"), "status": o.get("status")}
            elif s == 200:
                o = "ok" if "tools" in p else o
            got[name] = {"status": s, "body": mask(o, ids[name])}
        c.same(f"agents agui {label}", got["ours"], got["stock"])

    full = {"threadId": "t1", "runId": "r1", "messages": [{"id": "m1", "role": "user", "content": "hi"}], "tools": [], "context": [], "state": {}, "forwardedProps": {}}
    both("input", "POST", A + "/{a}/_execute", full)
    both("input conversational agent", "POST", A + "/{a}/_execute", full, conv=True)
    both("input unknown agent", "POST", A + "/nope/_execute", full)
    both("input with parameters", "POST", A + "/{a}/_execute", dict(full, parameters={"question": "q"}))
    both("input nulls", "POST", A + "/{a}/_execute", dict(full, threadId=None, messages=[1]))
    for k in full:
        both(f"input without {k}", "POST", A + "/{a}/_execute", {x: v for x, v in full.items() if x != k})
    both("stream", "POST", A + "/{a}/_execute/stream", full)
    both("stream existing model", "POST", A + "/{a}/_execute/stream", full, real=True)
    both("stream existing model no body", "POST", A + "/{a}/_execute/stream", real=True)
    both("stream conversational agent", "POST", A + "/{a}/_execute/stream", full, conv=True)
    both("stream parameters", "POST", A + "/{a}/_execute/stream", {"parameters": {"question": "hi"}})
    both("stream no body", "POST", A + "/{a}/_execute/stream")
    both("stream unknown agent", "POST", A + "/nope/_execute/stream", full)
    both("stream get", "GET", A + "/{a}/_execute/stream")
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.ag_ui_enabled": False}})
    both("input disabled", "POST", A + "/{a}/_execute", full)
    both("input disabled unknown agent", "POST", A + "/nope/_execute", full)
    both("input disabled conversational agent", "POST", A + "/{a}/_execute", full, conv=True)
    both("stream with input disabled", "POST", A + "/{a}/_execute/stream", full)
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.stream_enabled": False}})
    both("stream disabled", "POST", A + "/{a}/_execute/stream", full)
    both("stream disabled existing model", "POST", A + "/{a}/_execute/stream", full, real=True)
    both("stream disabled unknown agent", "POST", A + "/nope/_execute/stream", full)
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.stream_enabled": True, "plugins.ml_commons.agent_framework_enabled": False}})
    both("framework off register", "POST", A + "/_register", {"name": "x", "type": "flow", "tools": [{"type": "ListIndexTool"}]})
    both("framework off get", "GET", A + "/{a}")
    both("framework off update", "PUT", A + "/{a}", {"name": "z"})
    both("framework off delete", "DELETE", A + "/nope")
    both("framework off search", "POST", A + "/_search", {"query": {"match_all": {}}})
    both("framework off execute", "POST", A + "/{a}/_execute", {"parameters": {"question": "x"}})
    both("framework off execute no parameters", "POST", A + "/{a}/_execute", {})
    both("framework off execute unknown agent", "POST", A + "/nope/_execute", {"parameters": {"question": "x"}})
    both("framework off input", "POST", A + "/{a}/_execute", full)
    both("framework off stream", "POST", A + "/{a}/_execute/stream", full)
    both("framework off tools", "GET", "/_plugins/_ml/tools")
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.agent_framework_enabled": None, "plugins.ml_commons.ag_ui_enabled": None, "plugins.ml_commons.stream_enabled": None}})
        for a in (agent[name], agent[name + "c"], agent[name + "r"]):
            n.call("DELETE", f"{A}/{a}")


def final(text):
    return ("final", text)


def calls(*specs):
    return ("calls", specs)


def shape(style, step):
    """A scripted step as the model of that interface answers it."""
    if isinstance(step, dict):
        return step
    kind, arg = step
    if style == "openai":
        if kind == "final":
            return {"choices": [{"message": {"role": "assistant", "content": arg}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7}}
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": i, "type": "function", "function": {"name": n, "arguments": a}} for i, n, a in arg]}, "finish_reason": "tool_calls"}]}
    if style == "bedrock":
        if kind == "final":
            return {"output": {"message": {"role": "assistant", "content": [{"text": arg}]}}, "stopReason": "end_turn", "usage": {"inputTokens": 1}}
        return {"output": {"message": {"role": "assistant", "content": [{"toolUse": {"toolUseId": i, "name": n, "input": json.loads(a) if a.startswith("{") else {}}} for i, n, a in arg]}},
                "stopReason": "tool_use"}
    if kind == "final":
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": arg}]}, "finishReason": "STOP"}]}
    return {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": n, "args": json.loads(a) if a.startswith("{") else {}}} for i, n, a in arg]}, "finishReason": "STOP"}]}


LISTING = ("c1", "ListIndexTool", "{\"indices\":[\"ag1\"]}")
SCENARIOS = {
    "plain": [final("hello")],
    "quoted": [final("a \"quoted\"\nline \\ end")],
    "tool": [calls(LISTING), final("the index is ag1")],
    "loop": [calls(LISTING)],
    "summary": [calls(LISTING), calls(LISTING), final("steps one and two")],
    "badargs": [calls(("c1", "IndexMappingTool", "not json")), final("done")],
    "unknown": [calls(("c1", "Nope", "{}")), final("done")],
    "toolerr": [calls(("c1", "SearchIndexTool", "{}")), final("after error")],
    "two": [calls(LISTING, ("c2", "ListIndexTool", "{\"indices\":[\"ag2\"]}")), final("end")],
    "weird": [{"weird": 1}],
    "nullcontent": [{"choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}]}],
    "pe_tool": [final('{"steps":["list indices","summarize"],"result":""}'), calls(LISTING), final("step done"), final('{"steps":["summarize"],"result":""}'),
                final("sum done"), final('{"steps":[],"result":"final!"}')],
    "pe_fenced": [final('```json\n{"steps":[],"result":"fenced ok"}\n```')],
    "pe_invalid": [final("not json")],
    "pe_missing": [final('{"foo":1}')],
    "pe_max": [final('{"steps":["a"],"result":""}'), final("done a"), final('{"steps":["b"],"result":""}'), final("done b"), final('{"steps":["c"],"result":""}')],
    "pe_empty": [final('{"steps":[],"result":""}')],
    "mcp_add": [calls(("c1", "add", "{\"a\":2,\"b\":3}")), final("done")],
    "mcp_boom": [calls(("c1", "boom", "{}")), final("done")],
    "mcp_echo": [calls(("c1", "echo", "{\"text\":\"hi \\\"x\\\"\"}")), final("done")],
    "mcp_bad": [calls(("c1", "echo", "not json")), final("done")],
    "mcp_pic": [calls(("c1", "pic", "{}")), final("done")],
    "mcp_shout": [calls(("c1", "shout", "{\"text\":\"loud\"}")), final("done")],
    "scratch": [calls(("c1", "WriteToScratchPadTool", "{\"notes\":\"n1\"}")), final("saved")],
}


def llm_server():
    import http.server, threading
    seen = {}
    state = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n).decode("utf-8") if n else ""
            scenario, style, node, run = self.path.strip("/").split("/")
            key = (scenario, node, run)
            seen.setdefault(key, []).append(body)
            steps = SCENARIOS[scenario]
            out = shape(style, steps[min(len(seen[key]) - 1, len(steps) - 1)])
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    try:
        server = http.server.ThreadingHTTPServer(("0.0.0.0", LLM_PORT), Handler)
    except OSError:
        return seen
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return seen


MCP_PORT = 8769
MCP_TOOLS = [
    {"name": "add", "description": "Add two numbers", "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}},
    {"name": "echo", "description": "Echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "boom", "description": "Always fails", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "shout", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "pic", "description": "Answers an image and text", "inputSchema": {"type": "object", "properties": {}}},
]


def mcp_server():
    """An MCP server on streamable HTTP and on SSE; `/<node>/...` keeps what each node sent it."""
    import http.server, queue, threading
    log = {"ours": [], "stock": []}
    streams = {}

    def respond(body):
        """The JSON-RPC answer to `body`, with the headers it carries; None for a notification."""
        method, i = body.get("method"), body.get("id")
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": i, "result": {"protocolVersion": body["params"]["protocolVersion"], "capabilities": {"tools": {}},
                                                          "serverInfo": {"name": "mock", "version": "1"}}}, {"Mcp-Session-Id": "s1"}
        if method and method.startswith("notifications/"):
            return None, {}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": i, "result": {"tools": MCP_TOOLS}}, {}
        if method == "tools/call":
            name, a = body["params"]["name"], body["params"].get("arguments") or {}
            if name == "add":
                r = {"content": [{"type": "text", "text": str(a.get("a", 0) + a.get("b", 0))}], "isError": False}
            elif name in ("echo", "shout"):
                r = {"content": [{"type": "text", "text": a.get("text", "")}], "isError": False}
            elif name == "pic":
                r = {"content": [{"type": "image", "data": "aGk=", "mimeType": "image/png"}, {"type": "text", "text": "caption"}], "isError": False}
            else:
                r = {"content": [{"type": "text", "text": "kaboom"}], "isError": True}
            return {"jsonrpc": "2.0", "id": i, "result": r}, {}
        return {"jsonrpc": "2.0", "id": i, "error": {"code": -32601, "message": "no such method"}}, {}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, code, obj=None, headers=None):
            data = json.dumps(obj).encode() if obj is not None else b""
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.endswith("/sse"):
                node = self.path.strip("/").split("/")[0]
                out = streams[node] = queue.Queue()
                log.setdefault(node, []).append({"path": "GET /sse", "accept": self.headers.get("Accept"), "authorization": self.headers.get("Authorization")})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    self.wfile.write(f"event: endpoint\ndata: /{node}/message?sessionId=sid\n\n".encode())
                    self.wfile.flush()
                    while True:
                        try:
                            item = out.get(timeout=1)
                        except queue.Empty:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            continue
                        self.wfile.write(f"event: message\ndata: {json.dumps(item)}\n\n".encode())
                        self.wfile.flush()
                except OSError:
                    return
                return
            self.reply(405)

        def do_DELETE(self):
            self.reply(200)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            node = self.path.strip("/").split("/")[0]
            log.setdefault(node, []).append({"path": self.path.replace("/" + node, ""), "body": body, "authorization": self.headers.get("Authorization"),
                                             "session": self.headers.get("Mcp-Session-Id")})
            out, headers = respond(body)
            if "/message" in self.path:
                if out is not None:
                    streams[node].put(out)
                return self.reply(202)
            if out is None:
                return self.reply(202)
            self.reply(200, out, headers)

        def log_message(self, *args):
            pass

    try:
        server = http.server.ThreadingHTTPServer(("0.0.0.0", MCP_PORT), Handler)
    except OSError:
        return log
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return log


def mcp_traffic(entries):
    """What a node sent the MCP server, with the request ids a client makes up masked."""
    out = []
    for e in entries:
        text = re.sub(r"[0-9a-f]{8}-(\d+)", r"<c>-\1", json.dumps(e, sort_keys=True))
        out.append(json.loads(text))
    return out


def chat(c, ours, stock):
    seen = llm_server()
    mcp_log = mcp_server()
    side = (("ours", ours), ("stock", stock))
    ids = {"ours": [], "stock": []}
    counter = [0]

    def setup(name, n, scenario, run, extra=None, tools=None, memory=True, parameters=None, style="openai", atype="conversational"):
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.trusted_connector_endpoints_regex": [".*"],
                                                          "plugins.ml_commons.connector.private_ip_enabled": True}})
        con = {"name": "llm", "description": "d", "version": "1", "protocol": "http", "parameters": {"endpoint": "x", "model": "gpt-x"}, "credential": {"k": "v"},
               "actions": [{"action_type": "predict", "method": "POST", "url": f"http://127.0.0.1:{LLM_PORT}/{scenario}/{style}/{name}/{run}", "headers": {"a": "b"}, "request_body": BODIES[style].replace("parameters.question", "parameters.prompt") if atype not in ("conversational", "ag_ui") else BODIES[style]}]}
        model = n.ok("POST", "/_plugins/_ml/models/_register?deploy=true", {"name": "m", "function_name": "remote", "connector": con})["model_id"]
        time.sleep(2)
        agent = {"name": f"chat{run}", "type": atype, "llm": {"model_id": model, "parameters": {"max_iteration": "3", "system_prompt": "You are X"}},
                 "app_type": "chat", "tools": tools if tools is not None else [{"type": "ListIndexTool"}], "parameters": dict({"_llm_interface": STYLES[style]}, **((parameters(name) if callable(parameters) else parameters) or {}))}
        if memory:
            agent["memory"] = {"type": "conversation_index"}
        agent.update(extra or {})
        s, o = n.call("POST", "/_plugins/_ml/agents/_register", agent)
        if s != 200:
            return None, s, o
        ids[name] += [model, o["agent_id"]]
        return o["agent_id"], s, o

    def requests(name, scenario, run):
        out = []
        for b in seen.get((scenario, name, run), []):
            out.append(deep(json.loads(b), ids[name]))
        return out

    def case(label, scenario, request, again=None, style="openai", atype="conversational", mcp=False, **kw):
        counter[0] += 1
        run = str(counter[0])
        got = {}
        for name, n in side:
            agent, s, o = setup(name, n, scenario, run, style=style, atype=atype, **kw)
            mcp_log[name].clear()
            if agent is None:
                got[name] = {"register": s, "body": run_mask(o, ids[name])}
                continue
            s, o = n.call("POST", f"/_plugins/_ml/agents/{agent}/_execute", request)
            result = {"status": s}
            if s == 200 and again:
                first = next(x["result"] for x in o["inference_results"][0]["output"] if x["name"] == "memory_id")
                ids[name].append(first)
                s, o = n.call("POST", f"/_plugins/_ml/agents/{agent}/_execute", {"parameters": dict(again["parameters"], memory_id=first)})
                result["status"] = s
            if s == 200:
                out = o["inference_results"][0]["output"]
                mem = next((x["result"] for x in out if x["name"] == "memory_id"), None)
                par = next((x["result"] for x in out if x["name"] == "parent_interaction_id"), None)
                ids[name] += [x for x in (mem, par) if x]
                ids[name] += [str(x["result"]) for x in out if x["name"].endswith("_id") and x["name"] not in ("memory_id", "parent_interaction_id")]
                result["body"] = run_mask(o, ids[name])
                if mem:
                    s2, tr = n.call("GET", f"/_plugins/_ml/memory/message/{par}/traces")
                    ids[name] += [t["message_id"] for t in tr.get("traces", [])] if s2 == 200 else []
                    result["traces"] = [{k: v for k, v in t.items() if k not in ("create_time", "updated_time")} for t in deep(run_mask(tr, ids[name]), ids[name])["traces"]] if s2 == 200 else s2
                    result["traces"] = json.loads(json.dumps(result["traces"]))
                    s3, msgs = n.call("GET", f"/_plugins/_ml/memory/{mem}/messages")
                    if s3 == 200:
                        ids[name] += [m["message_id"] for m in msgs["messages"]]
                    result["messages"] = deep(run_mask(msgs, ids[name]), ids[name]) if s3 == 200 else s3
                    s4, meta = n.call("GET", f"/_plugins/_ml/memory/{mem}")
                    result["memory"] = run_mask(meta, ids[name]) if s4 == 200 else s4
            else:
                result["body"] = run_mask(o, ids[name])
            result["llm"] = requests(name, scenario, run)
            if mcp:
                result["mcp"] = mcp_traffic(mcp_log[name])
            got[name] = result
        c.same(f"agents chat {style} {label}" if style != "openai" else f"agents chat {label}", got["ours"], got["stock"])

    q = lambda t, **p: {"parameters": dict({"question": t}, **p)}
    case("plain", "plain", q("hi"))
    case("ag ui plain", "plain", q("hi"), atype="ag_ui")
    case("ag ui tool", "tool", q("which index?"), atype="ag_ui")
    case("ag ui loop", "loop", q("loop"), atype="ag_ui")
    case("ag ui history", "quoted", q("first \"turn\""), again=q("second"), atype="ag_ui")
    case("ag ui unknown tool", "unknown", q("unk"), atype="ag_ui")
    case("ag ui tool error", "toolerr", q("err"), tools=[{"type": "ListIndexTool"}, {"type": "SearchIndexTool"}], atype="ag_ui")
    case("quoted question", "plain", q("say \"x\"\nnow \\ y"))
    case("quoted answer", "quoted", q("hi"))
    case("tool", "tool", q("which index?"))
    case("loop", "loop", q("loop"))
    case("summary", "summary", q("sum"))
    case("unknown tool", "unknown", q("unk"))
    case("bad arguments", "badargs", q("bad"), tools=[{"type": "IndexMappingTool"}])
    case("tool error", "toolerr", q("err"), tools=[{"type": "ListIndexTool"}, {"type": "SearchIndexTool"}])
    case("two calls", "two", q("two"))
    case("weird answer", "weird", q("w"))
    case("null content", "nullcontent", q("n"))
    case("no question", "plain", {"parameters": {}})
    case("no memory", "tool", q("nomem"), memory=False)
    case("scratchpad", "scratch", q("note"), tools=[{"type": "WriteToScratchPadTool"}, {"type": "ReadFromScratchPadTool"}])
    case("renamed tool", "tool", q("rename"), tools=[{"type": "ListIndexTool", "name": "ListIndexTool", "description": "custom \"desc\""}])
    case("request overrides", "plain", q("hi", system_prompt="Be brief", max_iteration="1"))
    case("history", "quoted", q("first \"turn\""), again=q("second"))
    case("unknown memory", "plain", q("hi", memory_id="nope"))
    for style in ("bedrock", "gemini"):
        for label, scenario, tools in [("plain", "plain", None), ("tool", "tool", None), ("loop", "loop", None), ("unknown tool", "unknown", None),
                                       ("tool error", "toolerr", [{"type": "ListIndexTool"}, {"type": "SearchIndexTool"}]), ("two calls", "two", None),
                                       ("weird answer", "weird", None), ("quoted answer", "quoted", None),
                                       ("all tools", "plain", [{"type": "ListIndexTool"}, {"type": "SearchIndexTool"}, {"type": "IndexMappingTool"}])]:
            case(label, scenario, q("which?"), style=style, tools=tools)
        case("history", "quoted", q("first \"turn\""), again=q("second"), style=style)
    flow_memory(c, ours, stock)
    pe = dict(atype="plan_execute_and_reflect")
    q = lambda t, **p: {"parameters": dict({"question": t}, **p)}
    case("plan tool", "pe_tool", q("q"), **pe)
    case("plan fenced", "pe_fenced", q("q"), **pe)
    case("plan invalid json", "pe_invalid", q("q"), **pe)
    case("plan missing fields", "pe_missing", q("q"), **pe)
    case("plan max steps", "pe_max", q("q"), parameters={"max_steps": "2"}, **pe)
    case("plan empty", "pe_empty", q("q"), **pe)
    for label, extra in [("prefix", {"planner_system_prompt_prefix": "PFX"}), ("important rules", {"important_rules_expand": "RULE ONE"}),
                         ("result expand", {"result_expand_and_override": "RESULT TEXT"}), ("response format", {"plan_execute_reflect_response_format": "FORMAT"}),
                         ("system prompt", {"system_prompt": "SYS"}), ("executor system prompt", {"executor_system_prompt": "EXEC SYS"}),
                         ("history limits", {"message_history_limit": "1", "executor_message_history_limit": "1"}),
                         ("datetime", {"inject_datetime": "true", "datetime_format": "yyyy"}),
                         ("all prompts", {"planner_system_prompt_prefix": "P", "important_rules_expand": "I", "result_expand_and_override": "R", "plan_execute_reflect_response_format": "F"})]:
        case(f"plan {label}", "pe_tool", q("q"), parameters=extra, **pe)
    case("plan no question", "pe_tool", {"parameters": {}}, **pe)
    case("plan unknown memory", "pe_tool", q("q", memory_id="nope"), **pe)
    mcp_rows(c, ours, stock, case, ids, mcp_log)


def mcp_rows(c, ours, stock, case, ids, mcp_log):
    """Agents that use the tools of an MCP server."""
    side = (("ours", ours), ("stock", stock))
    cid = {}
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.mcp_connector_enabled": True}})
        cid[name] = n.ok("POST", "/_plugins/_ml/connectors/_create", {
            "name": "mcpc", "description": "d", "version": "1", "protocol": "mcp_streamable_http", "url": f"http://127.0.0.1:{MCP_PORT}",
            "credential": {"k": "v"}, "parameters": {"x": "y", "endpoint": f"/{name}/mcp/"}, "headers": {"Authorization": "Bearer ${credential.k}"}})["connector_id"]
        ids[name].append(cid[name])

    base = {"name": "c", "description": "d", "version": "1", "protocol": "mcp_streamable_http", "url": f"http://127.0.0.1:{MCP_PORT}",
            "credential": {"k": "v"}, "parameters": {"x": "y"}, "headers": {"A": "${credential.k}"}}
    variants = {"plain": {}, "sse": {"protocol": "mcp_sse"}, "no url": {"url": None}, "blank url": {"url": " "}, "url number": {"url": 5},
                "no credential": {"credential": None}, "empty credential": {"credential": {}}, "no headers": {"headers": None}, "no parameters": {"parameters": None},
                "with actions": {"actions": [{"action_type": "predict", "method": "POST", "url": "http://127.0.0.1:1/x", "request_body": "{}"}]},
                "client config": {"client_config": {"read_timeout": 5, "zzz": 1}}, "no name": {"name": None}, "no version": {"version": None},
                "extra field": {"zzz": 1}, "header number": {"headers": {"A": 5}}, "access control": {"access": "public"}}
    for label, change in variants.items():
        got = {}
        for name, n in side:
            body = {k: v for k, v in dict(base, **change).items() if v is not None}
            s, o = n.call("POST", "/_plugins/_ml/connectors/_create", body)
            if s == 200:
                ids[name].append(o["connector_id"])
                s2, g = n.call("GET", f"/_plugins/_ml/connectors/{o['connector_id']}")
                got[name] = {"status": s, "body": mask(g, ids[name])}
                s3, u = n.call("PUT", f"/_plugins/_ml/connectors/{o['connector_id']}", {"url": f"http://127.0.0.1:{MCP_PORT}/b", "headers": {"Z": "1"}, "parameters": {"q": "1"}})
                s4, g = n.call("GET", f"/_plugins/_ml/connectors/{o['connector_id']}")
                got[name]["updated"] = [s3, mask(g, ids[name]) if s4 == 200 else s4]
            else:
                got[name] = {"status": s, "body": o["error"]["reason"] if "error" in o else o}
        c.same(f"agents mcp connector {label}", got["ours"], got["stock"])

    sse = {}
    for name, n in side:
        sse[name] = n.ok("POST", "/_plugins/_ml/connectors/_create", dict(base, protocol="mcp_sse", parameters={"sse_endpoint": f"/{name}/sse"}))["connector_id"]
        ids[name].append(sse[name])
    got = {}
    for name, n in side:
        mcp_log[name].clear()
        s, o = n.call("GET", f"/_plugins/_ml/connectors/{sse[name]}/tools")
        got[name] = {"status": s, "body": mask(o, ids[name]) if s == 200 else o["error"]["reason"], "mcp": mcp_traffic(mcp_log[name])}
    c.same("agents mcp sse list tools", got["ours"], got["stock"])

    def entries(*specs):
        return lambda name: {"mcp_connectors": json.dumps([dict(s, mcp_connector_id=cid[name]) for s in specs])}

    q = lambda t, **p: {"parameters": dict({"question": t}, **p)}
    plain = entries({})
    got = {}
    for name, n in side:
        mcp_log[name].clear()
        s, o = n.call("GET", f"/_plugins/_ml/connectors/{cid[name]}/tools")
        got[name] = {"status": s, "body": mask(o, ids[name]) if s == 200 else o["error"]["reason"], "mcp": mcp_traffic(mcp_log[name])}
    c.same("agents mcp list tools", got["ours"], got["stock"])
    got = {}
    for name, n in side:
        s, o = n.call("GET", "/_plugins/_ml/connectors/nope/tools")
        got[name] = {"status": s, "body": o["error"]["reason"]}
    c.same("agents mcp list tools unknown connector", got["ours"], got["stock"])
    case("mcp tool", "mcp_add", q("2+3?"), parameters=plain, mcp=True)
    case("mcp sse tool", "mcp_add", q("2+3?"), parameters=lambda name: {"mcp_connectors": json.dumps([{"mcp_connector_id": sse[name]}])}, mcp=True)
    case("mcp sse tool error result", "mcp_boom", q("b"), parameters=lambda name: {"mcp_connectors": json.dumps([{"mcp_connector_id": sse[name]}])}, mcp=True)
    case("mcp tool openai tools", "mcp_add", q("2+3?"), parameters=plain, tools=[], mcp=True)
    case("mcp tool error result", "mcp_boom", q("b"), parameters=plain, mcp=True)
    case("mcp quoted arguments", "mcp_echo", q("e"), parameters=plain, mcp=True)
    case("mcp bad arguments", "mcp_bad", q("e"), parameters=plain, mcp=True)
    case("mcp image result", "mcp_pic", q("p"), parameters=plain, mcp=True)
    case("mcp tool filters", "mcp_add", q("f"), parameters=entries({"tool_filters": ["^a.*", "echo"]}), mcp=True)
    case("mcp filters match whole name", "mcp_add", q("f"), parameters=entries({"tool_filters": ["ad"]}), mcp=True)
    case("mcp tool descriptions", "mcp_add", q("d"), parameters=entries({"tool_descriptions": [{"add": "ADDER", "nope": "x"}, {"echo": ""}]}), mcp=True)
    case("mcp tool description map", "mcp_add", q("d"), parameters=entries({"tool_descriptions": {"add": "ADDER"}}), mcp=True)
    case("mcp tool without description", "mcp_shout", q("s"), parameters=plain, mcp=True)
    case("mcp declared tool", "mcp_add", q("x"), parameters=plain, tools=[{"type": "McpStreamableHttpTool", "name": "add"}], mcp=True)
    case("mcp declared unknown tool", "mcp_add", q("x"), parameters=plain, tools=[{"type": "McpStreamableHttpTool", "name": "nope"}], mcp=True)
    case("mcp declared without name", "mcp_add", q("x"), parameters=plain, tools=[{"type": "McpStreamableHttpTool"}], mcp=True)
    case("mcp no memory", "mcp_add", q("m"), parameters=plain, memory=False, mcp=True)
    for style in ("bedrock", "gemini"):
        case("mcp tool", "mcp_add", q("2+3?"), style=style, parameters=plain, mcp=True)
        case("mcp tool error result", "mcp_boom", q("b"), style=style, parameters=plain, mcp=True)
    case("mcp connector not found", "mcp_add", q("n"), parameters=lambda name: {"mcp_connectors": json.dumps([{"mcp_connector_id": "nope"}])}, mcp=True)
    case("mcp connector without id", "mcp_add", q("n"), parameters=lambda name: {"mcp_connectors": json.dumps([{}])}, mcp=True)

    def flow(label, tools, request, params=None, atype="flow"):
        got = {}
        for name, n in side:
            agent = {"name": "mcp-flow", "type": atype, "tools": tools, "parameters": dict(entries({})(name), **(params or {}))}
            s, o = n.call("POST", "/_plugins/_ml/agents/_register", agent)
            if s != 200:
                got[name] = {"register": s, "body": run_mask(o, ids[name])}
                continue
            ids[name].append(o["agent_id"])
            mcp_log[name].clear()
            s, o = n.call("POST", f"/_plugins/_ml/agents/{o['agent_id']}/_execute", request)
            got[name] = {"status": s, "body": run_mask(o, ids[name]), "mcp": mcp_traffic(mcp_log[name])}
        c.same(f"agents mcp flow {label}", got["ours"], got["stock"])

    M = lambda n: {"type": "McpStreamableHttpTool", "name": n}
    flow("add", [M("add")], {"parameters": {"input": "{\"a\":1,\"b\":2}"}})
    flow("two tools", [M("add"), M("echo")], {"parameters": {"input": "{\"text\":\"hi\"}"}})
    flow("fixed input", [{"type": "McpStreamableHttpTool", "name": "add", "parameters": {"input": "{\"a\":5,\"b\":6}"}}], {"parameters": {"question": "x"}})
    flow("no input", [M("add")], {"parameters": {"a": "1"}})
    flow("invalid input", [M("add")], {"parameters": {"input": "not json"}})
    flow("unknown tool", [M("nope")], {"parameters": {"input": "{}"}})
    flow("error result", [M("boom")], {"parameters": {"input": "{}"}})
    flow("mixed with a standard tool", [{"type": "ListIndexTool", "parameters": {"indices": ["ag1"]}}, M("echo")], {"parameters": {"input": "{\"text\":\"x\"}"}})
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.mcp_connector_enabled": False}})
    got = {}
    for name, n in side:
        s, o = n.call("GET", f"/_plugins/_ml/connectors/{cid[name]}/tools")
        s2, o2 = n.call("POST", "/_plugins/_ml/agents/_register", {"name": "off", "type": "conversational", "llm": {"model_id": "x"}, "parameters": {"mcp_connectors": "[]"}})
        got[name] = {"list": [s, o["error"]["reason"]], "register": [s2, o2["error"]["reason"] if s2 != 200 else "ok"]}
    c.same("agents mcp disabled", got["ours"], got["stock"])
    for name, n in side:
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.mcp_connector_enabled": True}})


def flow_memory(c, ours, stock):
    """conversational_flow agents: what they write to a memory named by `memory_id`."""
    side = (("ours", ours), ("stock", stock))
    ids = {"ours": [], "stock": []}
    tools = {"one": [{"type": "ListIndexTool", "parameters": {"indices": ["ag1"]}}],
             "two": [{"type": "ListIndexTool", "parameters": {"indices": ["ag1"]}}, {"type": "ListIndexTool", "name": "second", "parameters": {"indices": ["ag2"]}}]}
    for label, kind, memory, request in [
        ("one tool", "one", True, {"question": "what?"}),
        ("two tools", "two", True, {"question": "what?"}),
        ("no memory id", "one", False, {"question": "what?"}),
        ("no question", "one", True, {}),
        ("unknown memory", "one", True, {"question": "q", "memory_id": "nope"}),
        ("two runs", "one", True, {"question": "again"}),
    ]:
        got = {}
        for name, n in side:
            s, o = n.call("POST", "/_plugins/_ml/agents/_register", {"name": f"cf-{label}", "type": "conversational_flow", "memory": {"type": "conversation_index"}, "tools": tools[kind]})
            agent = o["agent_id"]
            ids[name].append(agent)
            req = dict(request)
            mem = None
            if memory and label != "unknown memory":
                s, o = n.call("POST", "/_plugins/_ml/memory", {"name": "m1"})
                mem = o["memory_id"]
                ids[name].append(mem)
                req["memory_id"] = mem
            runs = 2 if label == "two runs" else 1
            for _ in range(runs):
                s, o = n.call("POST", f"/_plugins/_ml/agents/{agent}/_execute", {"parameters": req})
            result = {"status": s, "body": run_mask(o, ids[name])}
            if mem:
                s2, msgs = n.call("GET", f"/_plugins/_ml/memory/{mem}/messages")
                if s2 == 200:
                    ids[name] += [m["message_id"] for m in msgs["messages"]]
                    result["messages"] = [{k: v for k, v in m.items() if k not in ("create_time", "updated_time")} for m in deep(run_mask(msgs, ids[name]), ids[name])["messages"]]
            got[name] = result
        c.same(f"agents flow memory {label}", got["ours"], got["stock"])
