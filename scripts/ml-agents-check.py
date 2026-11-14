"""The `agents` section of check-ml-plugin.py: the agent registry and the tool catalogue.

Registers, reads, updates, searches and deletes agents on both nodes and
compares the answers (agent ids, times and index versions masked), then
lists and reads the tools ml-commons itself registers. A stock node with
other plugins lists their tools too; only the names both sides know from
ml-commons are compared.
"""
import json

ML_TOOLS = ["ConnectorTool", "IndexInsightTool", "QueryPlanningTool", "VisualizationTool", "McpSseTool", "AgentTool",
            "WriteToScratchPadTool", "ReadFromScratchPadTool", "McpStreamableHttpTool", "SearchIndexTool", "MLModelTool",
            "IndexMappingTool", "ListIndexTool"]
DROP = {"created_time", "last_updated_time", "_version", "_seq_no", "_primary_term", "took"}

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
}

UPDATES = [
    ("name", {"name": "u2"}), ("description", {"description": "d1"}), ("type", {"type": "flow"}),
    ("same type", {"type": "conversational"}), ("llm model", {"llm": {"model_id": "m2"}}),
    ("llm parameters", {"llm": {"parameters": {"z": "y"}}}), ("tools", {"tools": [{"type": "ListIndexTool"}]}),
    ("empty tools", {"tools": []}), ("bad tool", {"tools": [{"name": "x"}]}), ("parameters", {"parameters": {"n": 1}}),
    ("app type", {"app_type": "A1"}), ("memory", {"memory": {"type": "conversation_index", "session_id": "s2"}}),
    ("memory window", {"memory": {"window_size": 9}}), ("bad memory", {"memory": {"type": "bogus"}}),
    ("agentic memory", {"memory": {"type": "agentic_memory"}}), ("blank name", {"name": ""}),
    ("hidden", {"is_hidden": True}), ("unknown", {"zzz": 1}), ("empty", {}),
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
