#!/usr/bin/env python3
"""The Rust engine (M5) in a three-node cluster, driven by scripts/verify-opensearch-cluster.sh.

os1 and os2 serve Rust-engine indices with the Rust writer; os3 runs with
lucene_rust.engine.node_enabled: false and serves them with OpenSearch's own engine. Every write
the test makes is recorded in a model of what was acknowledged, and every copy of the shard --
read with preference=_only_nodes -- must hold exactly that, after each of:

  1. document replication, a Rust primary and a Rust replica;
  2. peer recovery of a new replica on the Java node: a Rust primary with a Java replica;
  3. failover: the primary's node stops, a replica is promoted, writes continue;
  4. the stopped node rejoins and recovers as a replica;
  5. relocation Rust -> Java: the primary moves to the Java node, whose engine then writes into
     the Rust-written segments -- a Java primary with Rust replicas once they return;
  6. relocation Java -> Rust: back to a Rust node, whose writer then soft-deletes documents in
     Java-written (compound) segments;
  7. segment replication: a Java primary whose replicas open its segments and answer the plugin's
     native queries -- and a segment-replicated index asking for the Rust writer is refused, since
     OpenSearch 3.8 cannot use a plugin engine as a segment-replication primary.

Standard library only.
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request

NODES = {"os1": 9201, "os2": 9202, "os3": 9203}
FAILURES = []
CHECKS = [0]
DOWN = set()


def req(method, path, body=None, ndjson=False, node=None):
    targets = [node] if node else [n for n in NODES if n not in DOWN]
    last = None
    for n in targets:
        data = None
        headers = {}
        if body is not None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
        r = urllib.request.Request(f"http://localhost:{NODES[n]}{path}", data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                return resp.status, json.load(resp)
        except urllib.error.HTTPError as e:
            text = e.read().decode()
            try:
                return e.code, json.loads(text)
            except ValueError:
                return e.code, {"raw": text}
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last = e
    raise RuntimeError(f"{method} {path}: no node answered ({last})")


def must(method, path, body=None, ndjson=False, node=None):
    status, out = req(method, path, body, ndjson, node)
    if status not in (200, 201):
        raise RuntimeError(f"{method} {path}: {status} {json.dumps(out)[:600]}")
    return out


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(what)
        print("FAIL:", what, flush=True)


def wait(pred, what, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if pred():
                return True
        except RuntimeError:
            pass
        time.sleep(1)
    check(False, f"timed out waiting for {what}")
    return False


def health(index, status):
    st, h = req("GET", f"/_cluster/health/{index}?wait_for_status={status}&timeout=5s")
    return st == 200 and h.get("status") in (("green",) if status == "green" else ("green", status)) \
        and not h.get("timed_out", False) and h.get("relocating_shards", 0) == 0 and h.get("initializing_shards", 0) == 0


def copies(index):
    """{node name: primary?} for shard 0."""
    rows = must("GET", f"/_cat/shards/{index}?format=json&h=shard,prirep,state,node")
    return {r["node"]: r["prirep"] == "p" for r in rows if r["state"] == "STARTED"}


def primary(index):
    return next(n for n, p in copies(index).items() if p)


def engines(node):
    return must("GET", "/_plugins/lucene_rust/stats", node=node).get("engines", {})


WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()


class Model:
    def __init__(self, index, seed):
        self.index = index
        self.docs = {}
        self.r = random.Random(seed)

    def source(self, i):
        r = self.r
        return {"title": " ".join(r.choice(WORDS) for _ in range(1 + r.randrange(5))),
                "n": r.randrange(100000), "tag": f"t{r.randrange(8)}"}

    def ops(self, count, ids=400):
        lines, expect = [], []
        present = set(self.docs)  # as the bulk will leave it, operation by operation
        for _ in range(count):
            doc_id = f"d{self.r.randrange(ids)}"
            k = self.r.random()
            if k < 0.6 or doc_id not in present:
                present.add(doc_id)
                src = self.source(doc_id)
                lines += [{"index": {"_id": doc_id}}, src]
                expect.append((doc_id, "index", src))
            elif k < 0.8:
                patch = {"n": self.r.randrange(100), "tag": "upd"}
                lines += [{"update": {"_id": doc_id}}, {"doc": patch}]
                expect.append((doc_id, "update", patch))
            else:
                present.discard(doc_id)
                lines += [{"delete": {"_id": doc_id}}]
                expect.append((doc_id, "delete", None))
        out = must("POST", f"/{self.index}/_bulk", "\n".join(json.dumps(l) for l in lines) + "\n", ndjson=True)
        for item, (doc_id, kind, payload) in zip(out["items"], expect):
            (_, res), = item.items()
            status = res.get("status", 500)
            if status >= 300:
                check(status == 404 and kind == "delete", f"{self.index}: {kind} {doc_id} failed: {res}")
                continue
            if kind == "index":
                self.docs[doc_id] = dict(payload)
            elif kind == "update":
                self.docs[doc_id].update(payload)
            else:
                self.docs.pop(doc_id, None)
        must("POST", f"/{self.index}/_refresh")

    def copy_matches(self, node):
        out = must("POST", f"/{self.index}/_search?preference=_only_nodes:{node}&request_cache=false",
                   {"size": 10000, "query": {"match_all": {}}})
        got = {h["_id"]: h["_source"] for h in out["hits"]["hits"]}
        self.last_diff = {
            "missing": sorted(set(self.docs) - set(got))[:10],
            "extra": sorted(set(got) - set(self.docs))[:10],
            "different": sorted(k for k in set(got) & set(self.docs) if got[k] != self.docs[k])[:10],
        }
        return got == self.docs

    def verify(self, label):
        """Every started copy holds exactly the acknowledged documents, and a query agrees."""
        nodes = sorted(copies(self.index))
        for node in nodes:
            ok = wait(lambda: self.copy_matches(node), f"{label}: copy on {node} to catch up", 120)
            check(ok, f"{label}: copy on {node} holds the acknowledged documents {'' if ok else self.last_diff}")
        query = {"size": 50, "query": {"match": {"title": "alpha gamma"}}, "sort": [{"_id": "asc"}]}
        results = {}
        for node in nodes:
            out = must("POST", f"/{self.index}/_search?preference=_only_nodes:{node}&request_cache=false", query)
            results[node] = ([h["_id"] for h in out["hits"]["hits"]], out["hits"]["total"]["value"])
        check(len(set(map(json.dumps, results.values()))) == 1, f"{label}: a query agrees on every copy: {results}")
        print(f"{label}: {len(self.docs)} documents on {nodes}", flush=True)


def docker(*args):
    import subprocess
    subprocess.run(["docker", *args], check=True, capture_output=True)


def settings(index, s):
    must("PUT", f"/{index}/_settings", s)


def main():
    wait(lambda: must("GET", "/_cluster/health")["number_of_nodes"] == 3, "three nodes", 300)
    for idx in ("docrep", "segrep", "segrep_rust"):
        req("DELETE", f"/{idx}")

    # 1. Document replication: a Rust primary and a Rust replica.
    must("PUT", "/docrep", {"settings": {
        "number_of_shards": 1, "number_of_replicas": 1, "index.replication.type": "DOCUMENT",
        "index.lucene_rust.engine": True, "index.routing.allocation.include._name": "os1,os2"}})
    wait(lambda: health("docrep", "green"), "docrep green")
    m = Model("docrep", 1)
    for _ in range(5):
        m.ops(300)
    m.verify("document replication (Rust primary, Rust replica)")
    for node in ("os1", "os2"):
        check(engines(node).get("rust", 0) >= 1, f"{node} runs the Rust engine: {engines(node)}")

    # 2. Peer recovery onto the Java node.
    settings("docrep", {"index.number_of_replicas": 2, "index.routing.allocation.include._name": "os1,os2,os3"})
    wait(lambda: health("docrep", "green"), "three copies")
    check(engines("os3").get("java", 0) >= 1, f"os3 runs Java's engine: {engines('os3')}")
    m.verify("peer recovery (Rust primary, new Java replica)")
    m.ops(300)
    m.verify("writes to a Rust primary with a Java replica")

    # 3. Failover: stop the primary's node.
    p = primary("docrep")
    docker("stop", p)
    DOWN.add(p)
    wait(lambda: health("docrep", "yellow") and primary("docrep") != p, "a replica promoted")
    q = primary("docrep")
    print(f"primary {p} stopped; {q} promoted", flush=True)
    m.ops(300)
    m.verify(f"writes after failover to {q}")

    # 4. The stopped node rejoins as a replica.
    docker("start", p)
    DOWN.discard(p)
    wait(lambda: must("GET", "/_cluster/health", node=q)["number_of_nodes"] == 3, "the node rejoins", 300)
    wait(lambda: health("docrep", "green"), "green again", 300)
    m.verify("the stopped node recovered as a replica")

    # 5. Relocation Rust -> Java: one copy, pinned to os3; then Rust replicas return.
    settings("docrep", {"index.number_of_replicas": 0})
    wait(lambda: health("docrep", "green"), "one copy")
    if primary("docrep") == "os3":
        settings("docrep", {"index.routing.allocation.include._name": "os1"})
        wait(lambda: health("docrep", "green") and primary("docrep") == "os1", "the primary on os1")
        m.ops(300)
    settings("docrep", {"index.routing.allocation.include._name": "os3"})
    wait(lambda: health("docrep", "green") and primary("docrep") == "os3", "the primary relocated to os3")
    m.ops(300)
    m.verify("relocated Rust -> Java; Java's engine writes into Rust-written segments")
    settings("docrep", {"index.number_of_replicas": 2, "index.routing.allocation.include._name": "os1,os2,os3"})
    wait(lambda: health("docrep", "green"), "Rust replicas of a Java primary", 300)
    if primary("docrep") == "os3":
        m.ops(300)
        m.verify("a Java primary with Rust replicas")

    # 6. Relocation Java -> Rust: the Rust writer soft-deletes into Java's compound segments.
    settings("docrep", {"index.number_of_replicas": 0, "index.routing.allocation.include._name": "os1"})
    wait(lambda: health("docrep", "green") and primary("docrep") == "os1", "the primary relocated to os1")
    m.ops(600)
    m.verify("relocated Java -> Rust; the Rust writer updates Java-written segments")
    settings("docrep", {"index.number_of_replicas": 2, "index.routing.allocation.include._name": "os1,os2,os3"})
    wait(lambda: health("docrep", "green"), "three copies again", 300)
    m.verify("replicas recovered from the Rust primary")
    must("POST", "/docrep/_forcemerge?max_num_segments=1")
    must("POST", "/docrep/_refresh")
    m.verify("force-merged")

    # 7. Segment replication: a Java primary, replicas opening its segments natively...
    must("PUT", "/segrep", {"settings": {
        "number_of_shards": 1, "number_of_replicas": 2, "index.replication.type": "SEGMENT"}})
    wait(lambda: health("segrep", "green"), "segrep green")
    sr = Model("segrep", 2)
    for _ in range(4):
        sr.ops(300)
    sr.verify("segment replication (Java primary, replicas read natively)")
    replica = next(n for n, prim in copies("segrep").items() if not prim)
    before = must("GET", "/_plugins/lucene_rust/stats", node=replica)["native_queries"]
    must("POST", f"/segrep/_search?preference=_only_nodes:{replica}&request_cache=false",
         {"query": {"match": {"title": "alpha"}}})
    after = must("GET", "/_plugins/lucene_rust/stats", node=replica)["native_queries"]
    check(after > before, f"the replica on {replica} answered a match query natively: {before} -> {after}")
    # ...and the Rust writer refused as a segment-replication primary.
    must("PUT", "/segrep_rust", {"settings": {
        "number_of_shards": 1, "number_of_replicas": 0, "index.replication.type": "SEGMENT",
        "index.lucene_rust.engine": True,
        # On os3 (node_enabled: false) OpenSearch's engine serves it, and there is nothing to refuse.
        "index.routing.allocation.include._name": "os1,os2"}})
    time.sleep(5)
    explain = must("GET", "/_cluster/allocation/explain", {"index": "segrep_rust", "shard": 0, "primary": True})
    check("segment-replication primary" in json.dumps(explain), f"the refusal says why: {json.dumps(explain)[:400]}")

    print(f"verify_cluster: {CHECKS[0]} checks, {len(FAILURES)} failures", flush=True)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
