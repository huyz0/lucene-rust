"""The `remote` section of check-ml-plugin.py: connectors and remote models.

A mock model server (this module, on MOCK) answers as each provider's API
does, for the stock blueprint of each: OpenAI and Azure OpenAI, Bedrock
(Titan embeddings and Claude), Anthropic, Cohere (embed and rerank), Gemini,
DeepSeek, Mistral, plus Painless pre/post-process scripts, error statuses and
throttling retries. Both nodes run the same calls one after the other; each
answer is compared, and so is the request the mock received (method, path,
headers but those the HTTP clients choose themselves, body). A SigV4
request's signature is checked by recomputing it, since the signing time
differs. Then the connector API: create, get, search, update, delete and
their refusals, and the Vertex AI (google_cloud) connector's checks.
"""
import hashlib, hmac, http.server, json, os, subprocess, threading, time, urllib.parse

MOCK = 8767
# Headers each HTTP client picks for itself.
VOLATILE = {"host", "user-agent", "connection", "accept", "accept-encoding", "keep-alive", "x-amz-date",
            "authorization", "amz-sdk-invocation-id", "amz-sdk-request", "amz-sdk-retry", "expect", "upgrade",
            "http2-settings", "transfer-encoding"}
SECRET = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
log, lock = [], threading.Lock()
throttled = {}


def respond(path, body):
    """The provider's answer to a request: status and JSON."""
    try:
        req = json.loads(body) if body else {}
    except ValueError:
        req = {}
    vec = lambda i: [0.125 * (i + 1), -0.5, 1.0e-3 * (i + 1), 2]
    if path.startswith("/status/"):
        return int(path.split("/")[2]), {"error": {"message": "mock " + path.split("/")[2]}}
    if path.startswith("/slow/"):
        time.sleep(int(path.split("/")[2]) / 1000)
        return 200, {"ok": True}
    if path.startswith("/fail/"):
        # /fail/<status>/<times>/<side>: fails <times> times, then answers.
        n = throttled.get(path, 0)
        throttled[path] = n + 1
        if n < int(path.split("/")[3]):
            return int(path.split("/")[2]), {"message": "unavailable"}
        return 200, {"ok": True, "attempts": n + 1}
    if path.startswith("/amzthrottle/"):
        n = throttled.get(path, 0)
        throttled[path] = n + 1
        if n < int(path.split("/")[2]):
            return 400, {"message": "Rate exceeded"}, {"x-amzn-ErrorType": "ThrottlingException:http://internal.amazon.com/coral/com.amazon.coral.availability/"}
        return 200, {"ok": True, "attempts": n + 1}
    if path.startswith("/throttle/"):
        key = path
        n = throttled.get(key, 0)
        throttled[key] = n + 1
        if n < int(path.split("/")[2]):
            return 429, {"message": "Too many requests"}
        return 200, {"ok": True, "attempts": n + 1}
    if path.endswith("/embeddings") and "openai" in path or path.startswith("/mistral/v1/embeddings"):
        inputs = req.get("input", [])
        return 200, {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": vec(i)} for i in range(len(inputs))],
                     "model": req.get("model"), "usage": {"prompt_tokens": 3, "total_tokens": 3}}
    if "amazon.titan-embed" in path:
        return 200, {"embedding": vec(len(req.get("inputText", ""))), "inputTextTokenCount": 3}
    if "anthropic.claude" in path or path.startswith("/anthropic/v1/messages"):
        return 200, {"id": "msg_1", "type": "message", "role": "assistant", "model": req.get("model", "claude"),
                     "content": [{"type": "text", "text": "echo: " + json.dumps(req.get("messages"))}],
                     "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 7}}
    if path.startswith("/cohere/v1/embed"):
        return 200, {"id": "c", "embeddings": [vec(i) for i in range(len(req.get("texts", [])))], "texts": req.get("texts")}
    if path.startswith("/cohere/v1/rerank"):
        docs = req.get("documents", [])
        return 200, {"results": [{"index": i, "relevance_score": 1.0 / (i + 2)} for i in reversed(range(len(docs)))]}
    if ":generateContent" in path:
        text = req.get("contents", [{}])[0].get("parts", [{}])[0].get("text")
        return 200, {"candidates": [{"content": {"parts": [{"text": f"gemini: {text}"}], "role": "model"}, "finishReason": "STOP"}],
                     "usageMetadata": {"promptTokenCount": 2}}
    if path.endswith("/chat/completions") or "/chat/completions?" in path:
        return 200, {"id": "c1", "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "chat: " + json.dumps(req.get("messages"))}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 9}}
    return 200, {"echo": body, "nested": {"vals": [1, 2.5, "s"]}}


def serve_mock():
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def handle_one(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n).decode("utf-8") if n else ""
            with lock:
                log.append({"method": self.command, "path": self.path, "headers": dict(self.headers.items()), "body": body})
            status, out, *extra = respond(self.path, body)
            data = json.dumps(out).encode()
            self.send_response(status)
            for k, v in (extra[0] if extra else {}).items():
                self.send_header(k, v)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except BrokenPipeError:
                pass  # the client gave up waiting (the read timeout rows)

        do_POST = do_GET = do_PUT = do_DELETE = handle_one

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", MOCK), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()


def take():
    with lock:
        out = list(log)
        log.clear()
    return out


def sigv4_ok(e):
    """Whether a received request's SigV4 signature is the one its headers and body make."""
    auth = {k.lower(): v for k, v in e["headers"].items()}.get("authorization", "")
    if not auth.startswith("AWS4-HMAC-SHA256 "):
        return False
    fields = dict(p.strip().split("=", 1) for p in auth[len("AWS4-HMAC-SHA256 "):].split(","))
    akid, date, region, service, _ = fields["Credential"].split("/")
    signed = fields["SignedHeaders"].split(";")
    h = {k.lower(): " ".join(v.split()) for k, v in e["headers"].items()}
    path, _, query = e["path"].partition("?")
    # The SDK encodes the already-encoded path once more.
    canon_path = urllib.parse.quote(path, safe="/-_.~")
    pairs = sorted(urllib.parse.parse_qsl(query, keep_blank_values=True))
    canon_query = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}" for k, v in pairs)
    payload = h.get("x-amz-content-sha256") or hashlib.sha256(e["body"].encode()).hexdigest()
    canonical = "\n".join([e["method"], canon_path, canon_query, "".join(f"{k}:{h[k]}\n" for k in signed), ";".join(signed), payload])
    amz_date = h["x-amz-date"]
    scope = f"{date}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = ("AWS4" + SECRET).encode()
    for part in (date, region, service, "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    return hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest() == fields["Signature"]


def seen(entries):
    """What the mock received, without what the HTTP clients choose themselves."""
    out = []
    for e in entries:
        headers = sorted((k.lower(), v) for k, v in e["headers"].items() if k.lower() not in VOLATILE)
        auth = {k.lower(): v for k, v in e["headers"].items()}.get("authorization")
        if auth and auth.startswith("AWS4-HMAC-SHA256"):
            auth = ("sigv4", auth.split("SignedHeaders=")[1].split(",")[0], sigv4_ok(e))
        out.append((e["method"], e["path"], headers, auth, e["body"]))
    # Stock sends a split input's requests concurrently, so in no set order.
    return sorted(out, key=json.dumps)


def connector(name, protocol, params, actions, credential=None, **extra):
    c = {"name": name, "description": "check", "version": "1", "protocol": protocol, "parameters": params, "actions": actions}
    if credential is not None:
        c["credential"] = credential
    c.update(extra)
    return c


def act(url, body, headers=None, pre=None, post=None, method="POST"):
    a = {"action_type": "predict", "method": method, "url": url, "request_body": body}
    if headers:
        a["headers"] = headers
    if pre:
        a["pre_process_function"] = pre
    if post:
        a["post_process_function"] = post
    return a


H = f"localhost:{MOCK}"
AWS = {"access_key": "AKIDEXAMPLE", "secret_key": SECRET}

BLUEPRINTS = {
    "openai embedding": (connector("openai", "http", {"endpoint": H + "/openai", "model": "text-embedding-ada-002"},
        [act("http://${parameters.endpoint}/v1/embeddings", "{ \"input\": ${parameters.input}, \"model\": \"${parameters.model}\" }",
             {"Authorization": "Bearer ${credential.openAI_key}"}, "connector.pre_process.openai.embedding",
             "connector.post_process.openai.embedding")], {"openAI_key": "sk-o"}),
        [("text_embedding", {"text_docs": ["hello world", "b \"q\" é"]}),
         ("remote", {"parameters": {"input": ["x", "y"]}})]),
    "bedrock titan": (connector("titan", "aws_sigv4", {"region": "us-east-1", "service_name": "bedrock", "endpoint": H},
        [act("http://${parameters.endpoint}/model/amazon.titan-embed-text-v1/invoke", "{ \"inputText\": \"${parameters.inputText}\" }",
             {"content-type": "application/json", "x-amz-content-sha256": "required"}, "connector.pre_process.bedrock.embedding",
             "connector.post_process.bedrock.embedding")], AWS),
        [("text_embedding", {"text_docs": ["hello \"q\" world", "b"]})]),
    "bedrock claude": (connector("claude-bedrock", "aws_sigv4", {"region": "us-east-1", "service_name": "bedrock", "endpoint": H,
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 8000, "temperature": 0.0001, "response_filter": "$.content[0].text"},
        [act("http://${parameters.endpoint}/model/anthropic.claude-3-sonnet-20240229-v1:0/invoke",
             "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"${parameters.inputs}\"}]}],"
             "\"anthropic_version\":\"${parameters.anthropic_version}\",\"max_tokens\":${parameters.max_tokens},\"temperature\":${parameters.temperature}}",
             {"content-type": "application/json"})], dict(AWS, session_token="tok")),
        [("remote", {"parameters": {"inputs": "What is \"k-NN\"?\nShort."}})]),
    "anthropic messages": (connector("anthropic", "http", {"endpoint": H + "/anthropic", "model": "claude-sonnet-4-5", "max_tokens": 1024,
            "anthropic_version": "2023-06-01", "response_filter": "$.content[0].text"},
        [act("http://${parameters.endpoint}/v1/messages",
             "{\"model\":\"${parameters.model}\",\"max_tokens\":${parameters.max_tokens},\"messages\":${parameters.messages}}",
             {"x-api-key": "${credential.anthropic_key}", "anthropic-version": "${parameters.anthropic_version}", "content-type": "application/json"})],
        {"anthropic_key": "sk-ant"}),
        [("remote", {"parameters": {"messages": [{"role": "user", "content": "hi"}]}}),
         ("remote", {"parameters": {"messages": "[{\"role\":\"user\",\"content\":\"as string\"}]", "response_filter": "$.usage"}})]),
    "cohere embed": (connector("cohere", "http", {"endpoint": H + "/cohere", "model": "embed-english-v3.0", "input_type": "search_document", "truncate": "END"},
        [act("http://${parameters.endpoint}/v1/embed",
             "{ \"texts\": ${parameters.texts}, \"truncate\": \"${parameters.truncate}\", \"model\": \"${parameters.model}\", \"input_type\": \"${parameters.input_type}\" }",
             {"Authorization": "Bearer ${credential.cohere_key}", "Request-Source": "unspecified:opensearch"},
             "connector.pre_process.cohere.embedding", "connector.post_process.cohere.embedding")], {"cohere_key": "ck"}),
        [("text_embedding", {"text_docs": ["x", "y"]})]),
    "cohere rerank": (connector("rerank", "http", {"endpoint": H + "/cohere", "model": "rerank-english-v3.0"},
        [act("http://${parameters.endpoint}/v1/rerank",
             "{ \"documents\": ${parameters.documents}, \"query\": \"${parameters.query}\", \"model\": \"${parameters.model}\", \"top_n\": ${parameters.top_n} }",
             {"Authorization": "Bearer ${credential.cohere_key}"}, "connector.pre_process.cohere.rerank", "connector.post_process.cohere.rerank")],
        {"cohere_key": "ck"}),
        [("text_similarity", {"query_text": "q", "text_docs": ["a", "b", "c"]})]),
    "gemini": (connector("gemini", "http", {"endpoint": H + "/gemini", "model": "gemini-2.0-flash", "response_filter": "$.candidates[0].content.parts[0].text"},
        [act("http://${parameters.endpoint}/v1beta/models/${parameters.model}:generateContent",
             "{\"contents\":[{\"parts\":[{\"text\":\"${parameters.prompt}\"}]}]}",
             {"x-goog-api-key": "${credential.gemini_key}"})], {"gemini_key": "gk"}),
        [("remote", {"parameters": {"prompt": "tab\there"}})]),
    "azure openai": (connector("azure", "http", {"endpoint": H + "/azure", "deploy-name": "gpt", "api-version": "2024-02-01", "temperature": 0.7},
        [act("http://${parameters.endpoint}/openai/deployments/${parameters.deploy-name}/chat/completions?api-version=${parameters.api-version}",
             "{ \"messages\": ${parameters.messages}, \"temperature\": ${parameters.temperature} }",
             {"api-key": "${credential.openAI_key}"})], {"openAI_key": "az"}),
        [("remote", {"parameters": {"messages": [{"role": "user", "content": "x"}]}})]),
    "deepseek": (connector("deepseek", "http", {"endpoint": H + "/deepseek", "model": "deepseek-chat"},
        [act("http://${parameters.endpoint}/v1/chat/completions", "{ \"model\": \"${parameters.model}\", \"messages\": ${parameters.messages} }",
             {"Authorization": "Bearer ${credential.deepSeek_key}", "Content-Type": "application/json"})], {"deepSeek_key": "ds"}),
        [("remote", {"parameters": {"messages": [{"role": "user", "content": "y"}], "response_filter": "$.choices[0].message.content"}})]),
    "mistral": (connector("mistral", "http", {"endpoint": H + "/mistral", "model": "mistral-embed"},
        [act("http://${parameters.endpoint}/v1/embeddings", "{ \"model\": \"${parameters.model}\", \"input\": ${parameters.input} }",
             {"Authorization": "Bearer ${credential.mistralAPIKey}"}, "connector.pre_process.openai.embedding",
             "connector.post_process.openai.embedding")], {"mistralAPIKey": "mk"}),
        [("text_embedding", {"text_docs": ["m"]})]),
    "painless": (connector("painless", "http", {"endpoint": H},
        [act("http://${parameters.endpoint}/gen/pl", "{\"q\": ${parameters.q}}", None,
             "\n    StringBuilder builder = new StringBuilder();\n    builder.append(\"\\\"\");\n    builder.append(escape(params.text_docs[0]));\n"
             "    builder.append(\"\\\"\");\n    def parameters = \"{\" +\"\\\"q\\\":\" + builder + \"}\";\n    return  \"{\" +\"\\\"parameters\\\":\" + parameters + \"}\";",
             "\n  def name = \"sentence_embedding\";\n  def dataType = \"FLOAT32\";\n  def v = params.nested.vals;\n"
             "  return \"[{\\\"name\\\":\\\"\" + name + \"\\\",\\\"data_type\\\":\\\"\" + dataType + \"\\\",\\\"shape\\\":[2],\\\"data\\\":[\" + v[0] + \",\" + v[1] + \"]}]\";")],
        {"k": "v"}),
        [("text_embedding", {"text_docs": ["hi \"there\""]})]),
    "generic": (connector("generic", "http", {"endpoint": H},
        [act("http://${parameters.endpoint}/gen/x",
             "{ \"q\": \"${parameters.q}\", \"n\": ${parameters.n:-null}, \"d\": \"${parameters.d:-dflt}\" }")], {"k": "v"}),
        [("remote", {"parameters": {"q": "a \"b\"\nc/é "}}),
         ("remote", {"parameters": {"q": "x", "response_filter": "$.nested.vals"}}),
         ("remote", {"parameters": {"q": "x", "response_filter": "$.nope"}}),
         ("remote", {"parameters": {}}),
         ("remote", {"parameters": {"q": "x y", "no_escape_params": "q"}}),
         ("remote", {"parameters": {"q": "x", "skip_validating_missing_parameters": "true"}}),
         ("remote", {}),
         ("text_embedding", {"text_docs": ["a"]}),
         ("text_similarity", {"query_text": "q", "text_docs": ["a"]})]),
    "status 500": (connector("s500", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/status/500", "{\"q\":\"${parameters.q}\"}")], {"k": "v"}),
        [("remote", {"parameters": {"q": "x"}})]),
    "status 401": (connector("s401", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/status/401", "{\"q\":\"${parameters.q}\"}")], {"k": "v"}),
        [("remote", {"parameters": {"q": "x"}})]),
    "throttled, no retry": (connector("t0", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/throttle/1/<side>", "{}")], {"k": "v"}),
        [("remote", {"parameters": {}})]),
    "throttled, retried": (connector("t2", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/throttle/2/<side>", "{}")], {"k": "v"},
        client_config={"max_retry_times": 3, "retry_backoff_millis": 10}),
        [("remote", {"parameters": {}})]),
    "503, retried": (connector("f503", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/fail/503/2/<side>", "{}")], {"k": "v"},
        client_config={"max_retry_times": 3, "retry_backoff_millis": 10}),
        [("remote", {"parameters": {}})]),
    "400, not retried": (connector("f400", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/fail/400/1/<side>", "{}")], {"k": "v"},
        client_config={"max_retry_times": 3, "retry_backoff_millis": 10}),
        [("remote", {"parameters": {}})]),
    # Attempts at about 0, 0.4, 0.8 and 1.2 s: the fourth is past retry_timeout_seconds.
    "unlimited retries, until retry_timeout_seconds": (connector("fmax", "http", {"endpoint": H},
        [act("http://${parameters.endpoint}/fail/502/100/<side>", "{}")], {"k": "v"},
        client_config={"max_retry_times": -1, "retry_backoff_millis": 400, "retry_timeout_seconds": 1}),
        [("remote", {"parameters": {}})]),
    "aws throttling header, no retry": (connector("a0", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/amzthrottle/1/<side>", "{}")],
        {"k": "v"}),
        [("remote", {"parameters": {}})]),
    "aws throttling header, retried": (connector("a1", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/amzthrottle/1/<side>", "{}")],
        {"k": "v"}, client_config={"max_retry_times": 2, "retry_backoff_millis": 10}),
        [("remote", {"parameters": {}})]),
    "read_timeout in seconds": (connector("slow", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/slow/1500", "{}")], {"k": "v"},
        client_config={"read_timeout": 3, "connection_timeout": 2}),
        [("remote", {"parameters": {}})]),
    "read_timeout exceeded": (connector("slower", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/slow/2500", "{}")], {"k": "v"},
        client_config={"read_timeout": 1}),
        [("remote", {"parameters": {}})]),
}


def answer(r):
    s, o = r
    if s != 200 and isinstance(o, dict) and "error" in o:
        # Netty adds the channel's diagnostics to a timeout.
        reason = str(o["error"].get("reason")).split(" Channel Information: ")[0]
        return s, o["error"].get("type"), reason
    return s, o


def scrub(v, ids):
    s = json.dumps(v)
    for i in ids:
        if i:
            s = s.replace(i, "<id>")
    return json.loads(s)


MODEL_FIELDS = ["name", "algorithm", "model_version", "description", "model_state", "is_hidden", "connector",
                "planning_worker_node_count", "current_worker_node_count", "deploy_to_all_nodes", "auto_redeploy_retry_times"]
TASK_FIELDS = ["task_type", "function_name", "state", "is_async", "error"]


def scrub_seq(r):
    if len(r) == 2 and isinstance(r[1], dict):
        return r[0], {k: v for k, v in r[1].items() if k not in ("_seq_no", "_primary_term")}
    return r


def pick(doc, fields):
    return {f: doc.get(f) for f in fields if f in doc} if isinstance(doc, dict) else doc


def settled(n, mid):
    """The model once a deploy has finished: stock answers a remote deploy before the model's state is written."""
    for _ in range(50):
        m = n.ok("GET", f"/_plugins/_ml/models/{mid}")
        if m.get("model_state") not in ("DEPLOYING", "REGISTERED"):
            return m
        time.sleep(0.2)
    return m


def blueprints(c, ours, stock, run):
    for label, (conn, calls) in BLUEPRINTS.items():
        rows = {}
        for side, n in (("ours", ours), ("stock", stock)):
            body = json.loads(json.dumps(conn).replace("<side>", side))
            body["name"] = f"{body['name']}-{run}"
            take()
            s, r = n.call("POST", "/_plugins/_ml/connectors/_create", body)
            cid = r.get("connector_id")
            s2, reg = n.call("POST", "/_plugins/_ml/models/_register", {"name": f"rm-{label}-{run}", "function_name": "remote", "connector_id": cid})
            mid = reg.get("model_id")
            out = [(s, sorted(r)), (s2, sorted(reg))]
            time.sleep(0.5)
            out.append(pick(n.ok("GET", f"/_plugins/_ml/tasks/{reg.get('task_id')}"), TASK_FIELDS))
            # The first call deploys the model, as stock does for an undeployed remote model.
            for route, payload in calls:
                path = f"/_plugins/_ml/models/{mid}/_predict" if route == "remote" else f"/_plugins/_ml/_predict/{route}/{mid}"
                out.append((answer(n.call("POST", path, payload)), seen(take())))
            out.append(pick(settled(n, mid), MODEL_FIELDS))
            n.call("POST", f"/_plugins/_ml/models/{mid}/_undeploy")
            n.call("DELETE", f"/_plugins/_ml/models/{mid}")
            n.call("DELETE", f"/_plugins/_ml/connectors/{cid}")
            rows[side] = scrub(out, [mid, cid, side])
        names = ["create", "register", "register task"] + [f"{r} {i}" for i, (r, _) in enumerate(calls)] + ["model"]
        for i, what in enumerate(names):
            a, b = rows["ours"][i], rows["stock"][i]
            c.true(f"remote {label}: {what}", a == b, f"\n   ours  {json.dumps(a)[:1500]}\n   stock {json.dumps(b)[:1500]}")
        for side in ("ours", "stock"):
            for row in rows[side][3:-1]:
                for req in row[1]:
                    if isinstance(req[3], list) and req[3][0] == "sigv4":
                        c.true(f"remote {label}: {side} SigV4 signature verifies", req[3][2], json.dumps(req)[:400])


def lifecycle(c, ours, stock, run):
    gen = connector(f"life-{run}", "http", {"endpoint": H}, [act("http://${parameters.endpoint}/gen/x", "{ \"q\": \"${parameters.q}\" }",
                    {"Authorization": "Bearer ${credential.key}"})], {"key": "sk"})
    rows = {}
    for side, n in (("ours", ours), ("stock", stock)):
        out, ids = [], []
        s, r = n.call("POST", "/_plugins/_ml/connectors/_create", gen)
        cid = r["connector_id"]
        ids.append(cid)
        g = n.ok("GET", f"/_plugins/_ml/connectors/{cid}")
        out.append({k: v for k, v in g.items() if not k.endswith("_time")})
        hits = n.ok("POST", "/_plugins/_ml/connectors/_search", {"query": {"term": {"_id": cid}}})["hits"]["hits"]
        out.append([{k: v for k, v in h["_source"].items() if not k.endswith("_time")} for h in hits])
        s, r = n.call("PUT", f"/_plugins/_ml/connectors/{cid}", {"description": "e", "credential": {"key": "sk2"}})
        out.append((s, r.get("result") if s == 200 else answer((s, r))))
        out.append(n.ok("GET", f"/_plugins/_ml/connectors/{cid}").get("description"))
        reg = n.ok("POST", "/_plugins/_ml/models/_register?deploy=true", {"name": f"life-{run}", "function_name": "remote", "connector_id": cid})
        mid = reg["model_id"]
        ids.append(mid)
        time.sleep(2)
        take()
        out.append(answer(n.call("POST", f"/_plugins/_ml/models/{mid}/_predict", {"parameters": {"q": "x"}})))
        out.append(seen(take()))
        out.append(answer(n.call("DELETE", f"/_plugins/_ml/connectors/{cid}")))
        out.append(answer(n.call("PUT", f"/_plugins/_ml/connectors/{cid}", {"description": "f"})))
        s, r = n.call("POST", f"/_plugins/_ml/models/{mid}/_deploy")
        out.append((s, sorted(r) if s == 200 else r, r.get("status")))
        un = n.ok("POST", f"/_plugins/_ml/models/{mid}/_undeploy")
        out.append([list(v["stats"].values()) for v in un.values()])
        out.append(pick(n.ok("GET", f"/_plugins/_ml/models/{mid}"), MODEL_FIELDS))
        out.append(n.call("DELETE", f"/_plugins/_ml/models/{mid}")[0])
        out.append(n.call("DELETE", f"/_plugins/_ml/connectors/{cid}")[0])
        out.append(answer(n.call("GET", f"/_plugins/_ml/connectors/{cid}")))
        inline = n.ok("POST", "/_plugins/_ml/models/_register?deploy=true",
                      {"name": f"inline-{run}", "function_name": "remote", "description": "x", "connector": gen})
        ids.append(inline["model_id"])
        time.sleep(2)
        out.append(pick(n.ok("GET", f"/_plugins/_ml/models/{inline['model_id']}"), MODEL_FIELDS))
        take()
        out.append(answer(n.call("POST", f"/_plugins/_ml/models/{inline['model_id']}/_predict", {"parameters": {"q": "y"}})))
        out.append(seen(take()))
        n.call("POST", f"/_plugins/_ml/models/{inline['model_id']}/_undeploy")
        n.call("DELETE", f"/_plugins/_ml/models/{inline['model_id']}")
        raw = n.call("GET", "/.plugins-ml-connector/_search")
        out.append(raw[0])
        rows[side] = scrub(out, ids)
    for i, (a, b) in enumerate(zip(rows["ours"], rows["stock"])):
        c.true(f"remote lifecycle {i}", a == b, f"\n   ours  {json.dumps(a)[:1500]}\n   stock {json.dumps(b)[:1500]}")


def refusals(c, ours, stock, run, pem):
    base = connector(f"bad-{run}", "http", {"p": "v"}, [act(f"http://{H}/gen/x", "{}")], {"k": "v"})
    without = lambda k: {x: y for x, y in base.items() if x != k}
    cases = [
        ("bad protocol", "/_plugins/_ml/connectors/_create", dict(base, protocol="ftp")),
        ("no actions", "/_plugins/_ml/connectors/_create", without("actions")),
        ("no name", "/_plugins/_ml/connectors/_create", without("name")),
        ("no version", "/_plugins/_ml/connectors/_create", without("version")),
        ("no credential", "/_plugins/_ml/connectors/_create", without("credential")),
        ("empty credential", "/_plugins/_ml/connectors/_create", dict(base, credential={})),
        ("no method", "/_plugins/_ml/connectors/_create", dict(base, actions=[{"action_type": "predict", "url": f"http://{H}/x"}])),
        ("bad action type", "/_plugins/_ml/connectors/_create", dict(base, actions=[dict(base["actions"][0], action_type="nope")])),
        ("missing connector", "/_plugins/_ml/models/_register", {"name": "m", "function_name": "remote", "connector_id": "nope"}),
    ]
    vertex = connector(f"vx-{run}", "google_cloud", {"project": "p"}, [act(f"http://{H}/vertex/x", "{\"q\":\"${parameters.q}\"}")],
                       {"client_email": "a@b.iam.gserviceaccount.com", "private_key": pem})
    for label, path, body in cases:
        c.same(f"remote refusal: {label}", answer(ours.call("POST", path, body)), answer(stock.call("POST", path, body)))
    for label, n in (("missing get", "GET"), ("missing delete", "DELETE")):
        # A delete's sequence number is the node's own.
        a, b = (scrub_seq(answer(x.call(n, "/_plugins/_ml/connectors/nope"))) for x in (ours, stock))
        c.same(f"remote refusal: {label}", a, b)
    for n in (ours, stock):
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.connector.private_ip_enabled": False}})
    try:
        got = []
        for n in (ours, stock):
            local = connector(f"ip-{run}", "http", {}, [act(f"http://127.0.0.1:{MOCK}/gen/x", "{}")], {"k": "v"})
            mid = n.ok("POST", "/_plugins/_ml/models/_register?deploy=true", {"name": f"ip-{run}", "function_name": "remote", "connector": local})["model_id"]
            time.sleep(2)
            got.append(answer(n.call("POST", f"/_plugins/_ml/models/{mid}/_predict", {"parameters": {}})))
            n.call("POST", f"/_plugins/_ml/models/{mid}/_undeploy")
            n.call("DELETE", f"/_plugins/_ml/models/{mid}")
        c.same("remote refusal: private ip", got[0], got[1])
    finally:
        for n in (ours, stock):
            n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.connector.private_ip_enabled": True}})
    c.same("remote refusal: vertex disabled", answer(ours.call("POST", "/_plugins/_ml/connectors/_create", vertex)),
           answer(stock.call("POST", "/_plugins/_ml/connectors/_create", vertex)))
    for n in (ours, stock):
        n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.connector.vertexai_enabled": True}})
    try:
        got = []
        for n in (ours, stock):
            cid = n.ok("POST", "/_plugins/_ml/connectors/_create", vertex)["connector_id"]
            got.append(answer(n.call("PUT", f"/_plugins/_ml/connectors/{cid}", {"description": "changed"}))[0])
            got.append(n.ok("GET", f"/_plugins/_ml/connectors/{cid}").get("description"))
            n.call("DELETE", f"/_plugins/_ml/connectors/{cid}")
        c.same("remote refusal: vertex update without credential", got[:2], got[2:])
        for label, body in (("vertex missing key", dict(vertex, credential={"x": "y"})),
                            ("vertex adc with key", dict(vertex, parameters={"auth_mode": "adc"}))):
            c.same(f"remote refusal: {label}", answer(ours.call("POST", "/_plugins/_ml/connectors/_create", body)),
                   answer(stock.call("POST", "/_plugins/_ml/connectors/_create", body)))
        for label, cred, params in (
            ("vertex bad key", {"client_email": "a@b", "private_key": "x"}, None),
            ("vertex http token_uri", {"token_uri": "http://oauth2.googleapis.com/token"}, None),
            ("vertex token_uri host", {"token_uri": "https://localhost:8791/token"}, None),
            ("vertex token_uri port", {"token_uri": "https://oauth2.googleapis.com:8443/token"}, None),
            ("vertex token_uri invalid", {"token_uri": "::"}, None),
            ("vertex adc not found", {"k": "v"}, {"auth_mode": "adc"}),
        ):
            got = []
            for n in (ours, stock):
                cred_all = cred if "k" in cred or "private_key" in cred and cred["private_key"] == "x" else dict(vertex["credential"], **cred)
                body = dict(vertex, credential=cred_all, parameters=params or vertex["parameters"])
                mid = n.ok("POST", "/_plugins/_ml/models/_register?deploy=true", {"name": f"vx-{run}", "function_name": "remote", "connector": body})["model_id"]
                time.sleep(2)
                got.append(answer(n.call("POST", f"/_plugins/_ml/models/{mid}/_predict", {"parameters": {"q": "x"}})))
                n.call("POST", f"/_plugins/_ml/models/{mid}/_undeploy")
                n.call("DELETE", f"/_plugins/_ml/models/{mid}")
            c.same(f"remote refusal: {label}", got[0], got[1])
    finally:
        for n in (ours, stock):
            n.ok("PUT", "/_cluster/settings", {"persistent": {"plugins.ml_commons.connector.vertexai_enabled": None}})


def remote(c, ours, stock, run):
    serve_mock()
    for n in (ours, stock):
        n.ok("PUT", "/_cluster/settings", {"persistent": {
            "plugins.ml_commons.trusted_connector_endpoints_regex": [".*"],
            "plugins.ml_commons.connector.private_ip_enabled": True}})
    pem = subprocess.run(["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048"],
                         check=True, capture_output=True, text=True).stdout
    blueprints(c, ours, stock, run)
    lifecycle(c, ours, stock, run)
    refusals(c, ours, stock, run, pem)
