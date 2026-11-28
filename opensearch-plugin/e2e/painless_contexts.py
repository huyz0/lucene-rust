#!/usr/bin/env python3
"""Painless in the contexts beyond search scripts, answered by a plugin node and a stock node: ingest `script`
processors (`_ingest/pipeline/_simulate`), update scripts (`_update`), `scripted_metric`, `moving_fn`,
`bucket_script` and `bucket_selector`. Every response must be the same JSON (error types for refusals).

    python3 opensearch-plugin/e2e/painless_contexts.py [--plugin URL] [--stock URL]

Needs `script.disable_max_compilations_rate=true` on both nodes."""
import argparse
import json
import sys
import urllib.error
import urllib.request

DOCS = [
    {"name": "Alpha Beta", "n": 5, "tags": ["x", "y"], "nested": {"a": 1, "b": [1, 2, 3]}, "price": 9.99, "flag": True, "s": None},
    {"name": "gamma", "n": -3, "tags": [], "nested": {}, "price": 0.1, "flag": False},
    {"name": "", "n": 2147483647, "tags": ["z"], "price": 1e10, "t": "2024-02-29T12:30:00Z", "list": [3, 1, 2]},
    {"n": 7, "name": "O'Brien \"q\" ü😀", "m": {"k1": 1, "k2": "v"}},
]

INGEST = [
    "ctx.n2 = ctx.n * 2",
    "ctx.upper = ctx.name?.toUpperCase()",
    "ctx.len = ctx.name == null ? -1 : ctx.name.length()",
    "ctx.remove('tags')",
    "ctx.tags.add('new')",
    "ctx.tags = ctx.tags.stream().map(t -> t + '!').collect(Collectors.toList())",
    "ctx.sum = 0; for (def t : ctx.tags) { ctx.sum += t.length(); }",
    "ctx['nested'] = ['x': 1, 'y': [1, 2]]",
    "ctx.nested.c = 'new'",
    "ctx.nested.b.add(4)",
    "ctx.nested.remove('a')",
    "ctx.price = Math.round(ctx.price * 100) / 100.0",
    "ctx.price2 = ctx.price * 3",
    "ctx.flag = !ctx.flag",
    "ctx.when = Instant.parse(ctx.t).toEpochMilli()",
    "ctx.parts = ctx.name.splitOnToken(' ')",
    "ctx.idx = ctx.list.indexOf(1)",
    "Collections.sort(ctx.list); ctx.list.add(ctx.list.size())",
    "ctx.list = ctx.list.stream().sorted((a, b) -> b - a).collect(Collectors.toList())",
    "ctx.m.k3 = ctx.m.k1 + 1; ctx.m.remove('k2')",
    "ctx.keys = new ArrayList(ctx.m.keySet())",
    "ctx.big = ctx.n + 1",
    "ctx.big = (long) ctx.n + 1",
    "ctx.ratio = ctx.n / 2",
    "ctx.ratio = ctx.n / 2.0",
    "ctx.r = ctx.n % 3",
    "ctx.c = 'x' + ctx.n + 1.5 + true + null",
    "ctx.s2 = ctx.s == null ? 'none' : ctx.s",
    "ctx._id = 'set-' + ctx.n",
    "ctx._index = 'other'",
    "ctx._routing = 'r'",
    "ctx.nums = [1, 2.5, 3L, 'x', null, [1], ['a': 1]]",
    "ctx.f = (float) ctx.price",
    "ctx.d = ctx.price / 3",
    "ctx.str = String.valueOf(ctx.price) + '|' + ctx.n.toString()",
    "ctx.bool = ctx.containsKey('nested')",
    "ctx.nested = null",
    "ctx.nested.c = 1",
    "ctx.tags.get(5)",
    "ctx.x = ctx.nope.length()",
    "throw new IllegalArgumentException('boom ' + ctx.n)",
    "ctx.out = new HashMap(); ctx.out.put('b', 1); ctx.out.put('a', 2); ctx.out.put('c', 3)",
    "ctx.hs = new HashSet(ctx.tags)",
    "ctx.re = /[aeiou]/.matcher(ctx.name).replaceAll('*')",
    "ctx.fmt = String.format('%05d|%.2f', new def[] {ctx.n, ctx.price})",
]

UPDATES = [
    "ctx._source.n += params.d",
    "ctx._source.n = ctx._source.n * 2 + 1",
    "ctx._source.tags.add(params.t)",
    "ctx._source.remove('tags')",
    "ctx._source.nested.z = [1, 2]",
    "if (ctx._source.n > 0) { ctx.op = 'delete' } else { ctx.op = 'noop' }",
    "ctx.op = 'none'",
    "ctx._source.name = ctx._source.name + '!' ",
    "ctx._source.counter = (ctx._source.counter == null ? 0 : ctx._source.counter) + 1",
    "ctx._source.price = ctx._source.price * 1.1",
    "ctx._source.list = new ArrayList(); ctx._source.list.add(ctx._source.n)",
    "ctx._source.m = ['a': 1, 'b': ['c': 2]]",
    "ctx._source.big = Long.MAX_VALUE; ctx._source.big++",
    "ctx._source.x = 1 / ctx._source.n",
]

AGG_SCRIPTS = [
    ("scripted_metric", {"init_script": "state.sum = 0L; state.cnt = 0", "map_script": "state.sum += doc['n'].value; state.cnt++",
                         "combine_script": "return state", "reduce_script": "long s = 0; int c = 0; for (a in states) { s += a.sum; c += a.cnt; } return [s, c]"}),
    ("scripted_metric", {"init_script": "state.m = new HashMap()", "map_script": "state.m.merge(doc['tag'].value, 1, (a, b) -> a + b)",
                         "combine_script": "return state.m", "reduce_script": "Map r = new HashMap(); for (a in states) { for (e in a.entrySet()) { r.merge(e.getKey(), e.getValue(), (x, y) -> x + y); } } return r"}),
    ("scripted_metric", {"init_script": "state.l = []", "map_script": "state.l.add(doc['n'].value)",
                         "combine_script": "Collections.sort(state.l); return state.l", "reduce_script": "def all = []; for (a in states) { all.addAll(a); } Collections.sort(all); return all.subList(0, 5)"}),
    ("scripted_metric", {"init_script": "state.d = 0.0", "map_script": "state.d += doc['price'].value / 3",
                         "combine_script": "return state.d", "reduce_script": "double t = 0; for (a in states) { t += a; } return t"}),
    ("scripted_metric", {"map_script": "state.x = 1", "combine_script": "return state.x", "reduce_script": "return states.size()"}),
]

MOVING = ["MovingFunctions.max(values)", "MovingFunctions.min(values)", "MovingFunctions.sum(values)", "MovingFunctions.unweightedAvg(values)",
          "MovingFunctions.linearWeightedAvg(values)", "MovingFunctions.ewma(values, 0.3)", "MovingFunctions.holt(values, 0.3, 0.1)",
          "MovingFunctions.holtWinters(values, 0.3, 0.1, 0.2, 2, true)", "MovingFunctions.stdDev(values, MovingFunctions.unweightedAvg(values))",
          "MovingFunctions.sum(values) / values.length", "values.length == 0 ? 0 : values[values.length - 1]",
          "MovingFunctions.holtWinters(values, 0.3, 0.1, 0.2, 3, false)"]


def req(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(base + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, {}


def clean(x):
    """Responses minus what differs by node (timing, ids of nodes, ephemeral identity hashes)."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "_ingest":
                continue
            if k in ("took", "node", "_seq_no", "_primary_term", "script_stack", "position", "stack_trace", "header", "root_cause", "reason") and isinstance(v, (int, str, list, dict)):
                if k in ("reason",) and isinstance(v, str):
                    out[k] = v if len(v) < 200 and "Object@" not in v else v[:60]
                continue
            out[k] = clean(v)
        return out
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def error_sig(r):
    e = r.get("error", {})
    c = e
    chain = []
    while isinstance(c, dict):
        chain.append(c.get("type"))
        c = c.get("caused_by")
    return ("error", tuple(chain))


def outcome(st, r):
    if st >= 400:
        return error_sig(r)
    return ("ok", clean(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="http://localhost:9250")
    ap.add_argument("--stock", default="http://localhost:9251")
    args = ap.parse_args()
    nodes = (args.plugin, args.stock)
    bad = total = 0

    def compare(label, calls):
        nonlocal bad, total
        total += 1
        a, b = (outcome(*calls(n)) for n in nodes)
        if a != b:
            bad += 1
            print("DIFF", label, "\n   plugin:", json.dumps(a)[:500], "\n   stock: ", json.dumps(b)[:500])

    for src in INGEST:
        def sim(n, src=src):
            return req(n, "POST", "/_ingest/pipeline/_simulate", {
                "pipeline": {"processors": [{"script": {"source": src}}]},
                "docs": [{"_index": "ix", "_id": str(i), "_source": d} for i, d in enumerate(DOCS)]})
        compare("ingest: " + src, sim)

    for n in nodes:
        req(n, "DELETE", "/up")
        req(n, "PUT", "/up", {"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0,
                                           **({"index.engine.accelerated": True} if n == args.plugin else {})}})
        for i, d in enumerate(DOCS):
            req(n, "PUT", "/up/_doc/%d?refresh=true" % i, d)
    for src in UPDATES:
        for i in range(len(DOCS)):
            def upd(n, src=src, i=i):
                st, r = req(n, "POST", "/up/_update/%d?_source=true" % i, {"script": {"source": src, "params": {"d": 3, "t": "added"}}})
                st2, g = req(n, "GET", "/up/_doc/%d" % i)
                return st, {"upd": {k: v for k, v in r.items() if k in ("result", "error", "get", "_version")}, "doc": g.get("_source")}
            compare("update: %s on %d" % (src, i), upd)
            for n in nodes:  # put the document back
                req(n, "PUT", "/up/_doc/%d?refresh=true" % i, DOCS[i])

    for n in nodes:
        req(n, "DELETE", "/ag")
        req(n, "PUT", "/ag", {"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0,
                                           **({"index.engine.accelerated": True} if n == args.plugin else {})},
                              "mappings": {"properties": {"tag": {"type": "keyword"}, "n": {"type": "long"}, "price": {"type": "double"}, "ts": {"type": "date"}}}})
        lines = []
        for i in range(200):
            lines += [json.dumps({"index": {"_id": str(i)}}), json.dumps({"tag": "t%d" % (i % 7), "n": (i * 37) % 101 - 20, "price": (i % 13) * 1.37, "ts": 1700000000000 + i * 3600_000})]
        data = ("\n".join(lines) + "\n").encode()
        r = urllib.request.Request(n + "/ag/_bulk?refresh=true", data=data, method="POST", headers={"content-type": "application/x-ndjson"})
        urllib.request.urlopen(r).read()
    for kind, parts in AGG_SCRIPTS:
        compare("scripted_metric " + json.dumps(parts)[:80], lambda n, parts=parts: req(n, "POST", "/ag/_search", {"size": 0, "aggs": {"m": {kind: {**{k: {"source": v} for k, v in parts.items()}}}}}))
        compare("scripted_metric filtered " + json.dumps(parts)[:60], lambda n, parts=parts: req(n, "POST", "/ag/_search", {"size": 0, "query": {"term": {"tag": "t3"}}, "aggs": {"m": {kind: {**{k: {"source": v} for k, v in parts.items()}}}}}))
    for fn in MOVING:
        for window in (1, 3, 5):
            compare("moving_fn %s window %d" % (fn, window), lambda n, fn=fn, window=window: req(n, "POST", "/ag/_search", {"size": 0, "aggs": {"h": {
                "date_histogram": {"field": "ts", "fixed_interval": "6h"},
                "aggs": {"s": {"sum": {"field": "n"}}, "mv": {"moving_fn": {"buckets_path": "s", "window": window, "script": fn}}}}}}))
    for scr in ["params.s * 2", "params.s / (params.c == 0 ? 1 : params.c)", "params.s > 10 ? params.s : 0", "Math.abs(params.s) + params.c", "(double) params.s / 3"]:
        compare("bucket_script " + scr, lambda n, scr=scr: req(n, "POST", "/ag/_search", {"size": 0, "aggs": {"h": {
            "terms": {"field": "tag"}, "aggs": {"s": {"sum": {"field": "n"}}, "c": {"value_count": {"field": "n"}},
                                                "bs": {"bucket_script": {"buckets_path": {"s": "s", "c": "c"}, "script": scr}}}}}}))
    for scr in ["params.s > 100", "params.s < 0 || params.c > 28", "params.s % 2 == 0"]:
        compare("bucket_selector " + scr, lambda n, scr=scr: req(n, "POST", "/ag/_search", {"size": 0, "aggs": {"h": {
            "terms": {"field": "tag"}, "aggs": {"s": {"sum": {"field": "n"}}, "c": {"value_count": {"field": "n"}},
                                                "bs": {"bucket_selector": {"buckets_path": {"s": "s", "c": "c"}, "script": scr}}}}}}))

    print("%d cases, %d differ" % (total, bad))
    sys.exit(1 if bad else 0)


main()
