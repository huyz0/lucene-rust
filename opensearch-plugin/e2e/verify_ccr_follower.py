"""Cross-cluster replication with followers on the accelerated engine.

Needs two running nodes, each with this plugin and a replication plugin that has
``plugins.replication.follower.engine``: a leader (LEADER, transport LEADER_TRANSPORT)
and a follower (FOLLOWER). The follower node runs with
``plugins.replication.follower.engine: delegate`` when MODE=delegate, without it when
MODE=replication.

For each of a plain index and one with an ``extended`` k-NN field it follows the leader,
indexes, updates and deletes on the leader while following, and checks that the follower
ends with the leader's documents at the leader's sequence numbers and the same k-NN hits.
It prints the time the follower takes to catch up with a bulk load, and which engine
served the follower.
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

LEADER = os.environ.get("LEADER", "http://127.0.0.1:9200")
FOLLOWER = os.environ.get("FOLLOWER", "http://127.0.0.1:9201")
LEADER_TRANSPORT = os.environ.get("LEADER_TRANSPORT", "127.0.0.1:9300")
MODE = os.environ.get("MODE", "delegate")
DOCS = int(os.environ.get("DOCS", "50000"))
DIM = 32


def req(base, method, path, body=None, ok=(200, 201)):
    data = None
    headers = {}
    if body is not None:
        if isinstance(body, str):
            data = body.encode()
            headers["Content-Type"] = "application/x-ndjson"
        else:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
    r = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        text = e.read().decode()
        if e.code in ok:
            return json.loads(text)
        raise RuntimeError(f"{method} {base}{path}: {e.code} {text[:800]}")


def vec(rng):
    return [round(rng.gauss(0, 1), 4) for _ in range(DIM)]


def bulk(index, start, n, rng):
    lines = []
    for i in range(start, start + n):
        lines.append(json.dumps({"index": {"_index": index, "_id": str(i)}}))
        lines.append(json.dumps({"n": i, "tag": f"t{i % 17}", "body": f"doc {i} word{i % 101}", "v": vec(rng)}))
    out = req(LEADER, "POST", "/_bulk", "\n".join(lines) + "\n")
    assert not out["errors"], json.dumps(out)[:800]


def count(base, index):
    req(base, "POST", f"/{index}/_refresh")
    return req(base, "GET", f"/{index}/_count")["count"]


def wait_until(what, fn, timeout=300):
    t0 = time.time()
    while True:
        try:
            if fn():
                return time.time() - t0
        except RuntimeError:
            pass
        if time.time() - t0 > timeout:
            raise RuntimeError(f"timed out waiting for {what}")
        time.sleep(0.2)


def seqnos(base, index):
    out = req(base, "POST", f"/{index}/_search", {
        "size": 10000, "seq_no_primary_term": True, "_source": False,
        "query": {"range": {"n": {"lt": 2000}}}, "sort": [{"n": "asc"}]})
    return {h["_id"]: h["_seq_no"] for h in out["hits"]["hits"]}


def knn_ids(base, index, q):
    out = req(base, "POST", f"/{index}/_search", {"size": 10, "_source": False,
                                                  "query": {"knn": {"v": {"vector": q, "k": 10}}}})
    return [h["_id"] for h in out["hits"]["hits"]]


def all_vectors(base, index):
    out = req(base, "POST", f"/{index}/_search?scroll=1m", {"size": 5000, "_source": ["v"]})
    vectors = {}
    while out["hits"]["hits"]:
        for h in out["hits"]["hits"]:
            vectors[h["_id"]] = h["_source"]["v"]
        out = req(base, "POST", "/_search/scroll", {"scroll": "1m", "scroll_id": out["_scroll_id"]})
    return vectors


def stats(base):
    try:
        return req(base, "GET", "/_plugins/lucene_rust/stats").get("engines", {})
    except RuntimeError as e:
        return str(e)


def run(name, method):
    rng = random.Random(7)
    index = f"ccr_{name}"
    for base in (FOLLOWER, LEADER):
        try:
            if base == FOLLOWER:
                req(base, "POST", f"/_plugins/_replication/{index}/_stop", {})
        except RuntimeError:
            pass
        try:
            req(base, "DELETE", f"/{index}")
        except RuntimeError:
            pass
    field = {"type": "knn_vector", "dimension": DIM}
    field.update(method)
    req(LEADER, "PUT", f"/{index}", {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": True,
                               "engine.accelerated": True}},
        "mappings": {"properties": {"n": {"type": "long"}, "tag": {"type": "keyword"},
                                    "body": {"type": "text"}, "v": field}}})
    bulk(index, 0, 2000, rng)
    req(LEADER, "POST", f"/{index}/_flush")
    before = stats(FOLLOWER)
    try:
        req(FOLLOWER, "PUT", f"/_plugins/_replication/{index}/_start",
            {"leader_alias": "leader", "leader_index": index})
    except RuntimeError as e:
        print(f"{name}: start refused: {e}")
        return {"name": name, "started": False, "error": str(e)[:300]}

    def syncing():
        s = req(FOLLOWER, "GET", f"/_plugins/_replication/{index}/_status")
        if s.get("status") == "FAILED":
            raise SystemExit(f"{name}: replication failed: {s}")
        return s.get("status") == "SYNCING"
    try:
        wait_until("syncing", syncing, 120)
    except RuntimeError as e:
        status = req(FOLLOWER, "GET", f"/_plugins/_replication/{index}/_status")
        health = req(FOLLOWER, "GET", f"/_cluster/health/{index}")
        print(f"{name}: not syncing: {status} {health}")
        return {"name": name, "started": False, "error": json.dumps(status)[:300]}
    after_start = stats(FOLLOWER)

    # Changes while following: a bulk load, updates, deletes.
    t0 = time.time()
    batch = 2000
    for s in range(2000, 2000 + DOCS, batch):
        bulk(index, s, batch, rng)
    load = time.time() - t0
    req(LEADER, "POST", f"/{index}/_refresh")
    lines = []
    for i in range(0, 2000, 7):
        lines.append(json.dumps({"update": {"_index": index, "_id": str(i)}}))
        lines.append(json.dumps({"doc": {"tag": "updated", "v": vec(rng)}}))
    for i in range(3, 2000, 11):
        lines.append(json.dumps({"delete": {"_index": index, "_id": str(i)}}))
    out = req(LEADER, "POST", "/_bulk", "\n".join(lines) + "\n")
    assert not out["errors"]
    leader_count = count(LEADER, index)
    lag = wait_until("follower catch-up", lambda: count(FOLLOWER, index) == leader_count
                     and seqnos(FOLLOWER, index) == seqnos(LEADER, index), 600)
    total = time.time() - t0

    lseq, fseq = seqnos(LEADER, index), seqnos(FOLLOWER, index)
    assert lseq == fseq, f"{name}: seq_no differ on {sum(1 for k in lseq if lseq[k] != fseq.get(k))} docs"
    upd_l = req(LEADER, "POST", f"/{index}/_count", {"query": {"term": {"tag": "updated"}}})["count"]
    upd_f = req(FOLLOWER, "POST", f"/{index}/_count", {"query": {"term": {"tag": "updated"}}})["count"]
    assert upd_l == upd_f, (upd_l, upd_f)
    # Recall@10 against exact neighbours, on each side: the follower builds its own graphs from the
    # replayed operations, so its hits are as approximate as the leader's, not identical to them.
    vectors = all_vectors(LEADER, index)
    qs = 20
    qrng = random.Random(11)
    rl = rf = overlap = 0
    for _ in range(qs):
        q = vec(qrng)
        exact = set(sorted(vectors, key=lambda k: sum((a - b) ** 2 for a, b in zip(vectors[k], q)))[:10])
        hl, hf = knn_ids(LEADER, index, q), knn_ids(FOLLOWER, index, q)
        rl += len(exact & set(hl)) / 10
        rf += len(exact & set(hf)) / 10
        overlap += len(set(hl) & set(hf)) / 10
    # Follower search latency on the k-NN query.
    q = vec(qrng)
    t1 = time.time()
    for _ in range(50):
        knn_ids(FOLLOWER, index, q)
    knn_ms = (time.time() - t1) * 1000 / 50
    settings = req(FOLLOWER, "GET", f"/{index}/_settings")[index]["settings"]["index"]
    res = {"name": name, "started": True, "docs": leader_count, "leader_load_s": round(load, 2),
           "follower_caught_up_s": round(total, 2), "lag_after_load_s": round(lag, 2),
           "recall10_leader": round(rl / qs, 3), "recall10_follower": round(rf / qs, 3),
           "leader_follower_overlap": round(overlap / qs, 3), "follower_knn_ms": round(knn_ms, 2),
           "engines_before": before, "engines_after_start": after_start}
    print(json.dumps(res))
    req(FOLLOWER, "POST", f"/_plugins/_replication/{index}/_stop", {})
    return res


def main():
    req(FOLLOWER, "PUT", "/_cluster/settings",
        {"persistent": {"cluster.remote.leader.seeds": [LEADER_TRANSPORT]}})
    results = [run("plain", {"method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"}}),
               run("extended", {"method": {"name": "hnsw", "engine": "extended", "space_type": "l2",
                                           "parameters": {"quantizer": "rabitq"}}})]
    print("MODE", MODE, json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
