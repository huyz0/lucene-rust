#!/usr/bin/env python3
"""Rows updated again and again on a primary with a replica (two plugin nodes, `os1` and `os2`), against a standalone stock
node: the seeded operation stream of verify_hot_rows.py, answered identically; then the node holding the primary is
SIGKILLed mid-stream (the replica is promoted and writes continue), restarted (peer recovery), and every copy of the
shard, read with preference=_only_nodes, must hold what the stock node holds.

    nodectl run os1 plugin 9201 --transport 9301 --setting cluster.name=hc --setting discovery.seed_hosts=127.0.0.1:9301,127.0.0.1:9302 --setting cluster.initial_cluster_manager_nodes=os1,os2   (and os2 on 9202/9302)
    python3 opensearch-plugin/e2e/verify_hot_cluster.py SEED OPS [--hot]   (stock on :9251)"""
import json
import random
import sys
import urllib.error
import urllib.request

NODES = {"os1": 9201, "os2": 9202}
DOWN = set()
S = "http://localhost:9251"
P = "plugin-cluster"
IDS = ["r%d" % i for i in range(12)]
MAPPING = {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}, "t": {"type": "text"}}}


def req1(base, method, path, body=None, nd=False):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    r = urllib.request.Request(base + path, data=data, method=method,
                               headers={"content-type": "application/x-ndjson" if nd else "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, {}


def req(base, method, path, body=None, nd=False):
    if base != P:
        return req1(base, method, path, body, nd)
    last = None
    for n, port in NODES.items():
        if n in DOWN:
            continue
        try:
            return req1("http://localhost:%d" % port, method, path, body, nd)
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last = e
    raise RuntimeError("no node answered: %s" % last)


def reset(index):
    for base in (P, S):
        req(base, "DELETE", "/" + index)
        s = {"index.number_of_shards": 1, "index.number_of_replicas": 1, "refresh_interval": "-1"}
        if base == P:
            s["index.engine.accelerated"] = True
        else:
            s["index.number_of_replicas"] = 0
        st, r = req(base, "PUT", "/" + index, {"settings": s, "mappings": MAPPING})
        assert st == 200, r


def item(r):
    if "error" in r:
        return ("error", r.get("status"), r["error"].get("type") if isinstance(r["error"], dict) else r["error"])
    return tuple(r.get(k) for k in ("result", "_version", "_seq_no", "_primary_term", "status"))


def result(st, r):
    if st >= 300 and "items" not in r:
        return ("http", st, (r.get("error") or {}).get("type") if isinstance(r.get("error"), dict) else None)
    if "items" in r:
        return [item(list(i.values())[0]) for i in r["items"]]
    return item(r)


def doc(rnd):
    return {"k": "k%d" % rnd.randint(0, 3), "n": rnd.randint(-5, 50), "t": " ".join(rnd.choice(["red", "blue", "fox", "dog"]) for _ in range(3))}


def state(index, ids):
    out = {}
    for base, name in ((P, "plugin"), (S, "stock")):
        got = {}
        st, r = req(base, "POST", "/%s/_mget" % index, {"ids": ids})
        got["mget"] = [(d["_id"], d.get("found"), d.get("_version"), d.get("_seq_no"), d.get("_source")) for d in r["docs"]]
        out[name] = got
    return out


def refreshed_state(index):
    out = {}
    for base, name in ((P, "plugin"), (S, "stock")):
        req(base, "POST", "/%s/_refresh" % index)
        st, r = req(base, "POST", "/%s/_search" % index, {"size": 100, "query": {"match_all": {}}, "sort": [{"_id": "asc"}], "track_total_hits": True,
                                                         "seq_no_primary_term": True, "version": True})
        hits = [(h["_id"], h.get("_version"), h.get("_seq_no"), h["_source"]) for h in r["hits"]["hits"]]
        st, c = req(base, "POST", "/%s/_count" % index, {"query": {"term": {"k": "k1"}}})
        st, a = req(base, "POST", "/%s/_search" % index, {"size": 0, "aggs": {"n": {"sum": {"field": "n"}}, "k": {"terms": {"field": "k"}}}})
        out[name] = (hits, c.get("count"), a["aggregations"])
    return out


def copies(index):
    """Every copy of the shard, read on its own node, against the stock node."""
    st, stock = req(S, "POST", "/%s/_search" % index, {"size": 100, "query": {"match_all": {}}, "sort": [{"_id": "asc"}], "track_total_hits": True})
    want = [(h["_id"], h["_source"]) for h in stock["hits"]["hits"]]
    global bad_copies
    for n, port in NODES.items():
        if n in DOWN:
            continue
        req1("http://localhost:%d" % port, "POST", "/%s/_refresh" % index)
        st, r = req1("http://localhost:%d" % port, "POST", "/%s/_search?preference=_only_nodes:%s" % (index, n),
                     {"size": 100, "query": {"match_all": {}}, "sort": [{"_id": "asc"}], "track_total_hits": True})
        got = [(h["_id"], h["_source"]) for h in r["hits"]["hits"]] if st == 200 else r
        if got != want:
            bad_copies += 1
            print("FAIL copy on %s differs from stock\n  copy:  %s\n  stock: %s" % (n, str(got)[:400], str(want)[:400]))
        else:
            print("ok   copy on %s holds the %d documents stock holds" % (n, len(want)))


bad_copies = 0


def main():
    seed, ops = int(sys.argv[1]), int(sys.argv[2])
    kill = "--kill" in sys.argv
    rnd = random.Random(seed)
    index = "hot"
    reset(index)
    bad = 0
    seen = {}
    for step in range(ops):
        kinds = ["index", "index", "index", "update", "scripted", "delete", "create", "get", "bulk", "versioned", "if_seq", "refresh", "flush", "mget"]
        if "--hot" in sys.argv:
            kinds = ["index"] * 12 + ["update", "scripted", "delete", "create", "get", "bulk", "versioned", "if_seq", "mget"] + ["refresh"] * (1 if step % 40 == 0 else 0)
        k = rnd.choice(kinds)
        i = rnd.choice(IDS)
        d = doc(rnd)
        call = None
        if k == "index":
            call = ("PUT", "/%s/_doc/%s" % (index, i), d)
        elif k == "create":
            call = ("PUT", "/%s/_create/%s" % (index, i), d)
        elif k == "update":
            call = ("POST", "/%s/_update/%s" % (index, i), {"doc": {"n": d["n"]}, "doc_as_upsert": rnd.random() < .5})
        elif k == "scripted":
            call = ("POST", "/%s/_update/%s" % (index, i),
                    {"script": {"source": "ctx._source.n = (ctx._source.n == null ? 0 : ctx._source.n) + params.x", "params": {"x": d["n"]}},
                     "upsert": d})
        elif k == "delete":
            call = ("DELETE", "/%s/_doc/%s" % (index, i), None)
        elif k == "get":
            call = ("GET", "/%s/_doc/%s" % (index, i), None)
        elif k == "versioned":
            call = ("PUT", "/%s/_doc/%s?version=%d&version_type=external" % (index, i, rnd.randint(1, 3 * step + 5)), d)
        elif k == "if_seq":
            st, r = req(S, "GET", "/%s/_doc/%s" % (index, i))
            seq = r.get("_seq_no", 0) + rnd.choice([0, 0, 1])
            call = ("PUT", "/%s/_doc/%s?if_seq_no=%d&if_primary_term=1" % (index, i, seq), d)
        elif k == "bulk":
            lines = []
            for _ in range(rnd.randint(2, 8)):
                j = rnd.choice(IDS)
                op = rnd.choice(["index", "index", "update", "delete", "create"])
                lines.append(json.dumps({op: {"_id": j}}))
                if op in ("index", "create"):
                    lines.append(json.dumps(doc(rnd)))
                elif op == "update":
                    lines.append(json.dumps({"doc": {"n": rnd.randint(0, 9)}, "doc_as_upsert": True}))
            call = ("POST", "/%s/_bulk" % index, ("\n".join(lines) + "\n").encode())
        elif k == "refresh":
            call = ("POST", "/%s/_refresh" % index, None)
        elif k == "flush":
            call = ("POST", "/%s/_flush" % index, None)
        elif k == "mget":
            call = ("POST", "/%s/_mget" % index, {"ids": rnd.sample(IDS, 5)})
        method, path, body = call
        nd = k == "bulk"
        a = req(P, method, path, body, nd=nd)
        b = req(S, method, path, body, nd=nd)
        if k == "get":
            ra = (a[0], a[1].get("found"), a[1].get("_version"), a[1].get("_seq_no"), a[1].get("_source"))
            rb = (b[0], b[1].get("found"), b[1].get("_version"), b[1].get("_seq_no"), b[1].get("_source"))
        elif k == "mget":
            ra = [(x["_id"], x.get("found"), x.get("_version"), x.get("_seq_no"), x.get("_source")) for x in a[1]["docs"]]
            rb = [(x["_id"], x.get("found"), x.get("_version"), x.get("_seq_no"), x.get("_source")) for x in b[1]["docs"]]
        elif k in ("refresh", "flush"):
            ra = rb = a[0] == b[0]
        else:
            ra, rb = result(*a), result(*b)
        seen[k] = seen.get(k, 0) + 1
        if ra != rb:
            bad += 1
            print("FAIL step %d %s %s %s\n  plugin: %s\n  stock:  %s" % (step, k, method, path, str(ra)[:300], str(rb)[:300]))
        if step % 50 == 49:
            s = state(index, IDS)
            if s["plugin"] != s["stock"]:
                bad += 1
                print("FAIL step %d realtime mget differs\n  plugin: %s\n  stock:  %s" % (step, str(s["plugin"])[:400], str(s["stock"])[:400]))
        if step % 200 == 199:
            s = refreshed_state(index)
            if s["plugin"] != s["stock"]:
                bad += 1
                print("FAIL step %d searches differ\n  plugin: %s\n  stock:  %s" % (step, str(s["plugin"])[:600], str(s["stock"])[:600]))
        if step in (ops // 3, 2 * ops // 3):
            sys.path.insert(0, __file__.rsplit("/", 1)[0])
            import nodectl
            if step == ops // 3:
                st, shards = req(P, "GET", "/_cat/shards/%s?format=json&h=node,prirep,state" % index)
                holder = [x["node"] for x in shards if x["prirep"] == "p"][0]
                nodectl.kill(holder)
                DOWN.add(holder)
                print("killed the primary's node", holder)
                for _ in range(120):
                    st, h = req(P, "GET", "/_cluster/health/%s?wait_for_status=yellow&timeout=1s" % index)
                    if st == 200 and not h.get("timed_out"):
                        break
            else:
                nodectl.start(holder)
                DOWN.discard(holder)
                st, h = req(P, "GET", "/_cluster/health/%s?wait_for_status=green&timeout=180s" % index)
                print("restarted", holder, "health", h.get("status"), "timed_out", h.get("timed_out"))
                copies(index)
            s = state(index, IDS)
            if s["plugin"] != s["stock"]:
                bad += 1
                print("FAIL after the node change at step %d\n  plugin: %s\n  stock:  %s" % (step, str(s["plugin"])[:400], str(s["stock"])[:400]))
    s = refreshed_state(index)
    if s["plugin"] != s["stock"]:
        bad += 1
        print("FAIL final searches differ\n  plugin: %s\n  stock:  %s" % (str(s["plugin"])[:600], str(s["stock"])[:600]))
    copies(index)
    bad += bad_copies
    print("operations", seen)
    print("%d differences" % bad)
    sys.exit(1 if bad else 0)


main()
