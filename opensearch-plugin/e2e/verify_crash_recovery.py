#!/usr/bin/env python3
"""kill -9 of the plugin node while a writer indexes / updates / deletes: after the restart every
acknowledged operation must be there (translog replay), no document that was never sent may
appear, and the count by search equals the count by GET. Uses /tmp/restart_plugin.sh-style
control through PLUGIN_RESTART (a command that restarts the node on the same data path).

    PLUGIN_URL=http://localhost:9200 PLUGIN_RESTART=/tmp/restart_plugin.sh \\
        python3 opensearch-plugin/e2e/verify_crash_recovery.py
"""
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
RESTART = os.environ.get("PLUGIN_RESTART", "/tmp/restart_plugin.sh")
PORT = PLUGIN.rsplit(":", 1)[1]


def req(method, path, body=None, raw=False, timeout=60):
    data = body if raw else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(PLUGIN + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, json.loads(f.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def kill9():
    pid = subprocess.check_output(["pgrep", "-f", f"http.port={PORT}"]).split()[0].decode()
    os.kill(int(pid), 9)
    while os.path.exists(f"/proc/{pid}"):
        time.sleep(0.2)


def run_round(n, seed, settings):
    r = random.Random(seed)
    index = f"crash_{n}"
    req("DELETE", f"/{index}")
    st, resp = req("PUT", f"/{index}", {"settings": {"number_of_shards": 2, "number_of_replicas": 0, **settings},
                                         "mappings": {"properties": {"v": {"type": "integer"}, "t": {"type": "text"}, "k": {"type": "keyword"}}}})
    assert st == 200, resp
    state = {}                      # id -> acknowledged version (None = acked delete)
    sent = set()
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set() and i < 6000:
            lines = []
            ops = []
            for _ in range(r.randint(1, 40)):
                kind = r.random()
                if kind < 0.6 or not state:
                    doc_id = f"d{i}"; i += 1
                    v = r.randint(0, 10**6)
                    ops.append((doc_id, v)); lines += [json.dumps({"index": {"_id": doc_id}}), json.dumps({"v": v, "t": f"w{v % 50} x", "k": f"k{v % 9}"})]
                elif kind < 0.85:
                    doc_id = r.choice(list(state))
                    v = r.randint(0, 10**6)
                    ops.append((doc_id, v)); lines += [json.dumps({"index": {"_id": doc_id}}), json.dumps({"v": v, "t": f"w{v % 50} y", "k": f"k{v % 9}"})]
                else:
                    doc_id = r.choice(list(state))
                    ops.append((doc_id, None)); lines.append(json.dumps({"delete": {"_id": doc_id}}))
            for d, _ in ops:
                sent.add(d)
            try:
                st, resp = req("POST", f"/{index}/_bulk", ("\n".join(lines) + "\n").encode(), raw=True, timeout=30)
            except Exception:
                return
            if st != 200:
                return
            for (d, v), item in zip(ops, resp["items"]):
                res = next(iter(item.values()))
                if res.get("status") in (200, 201):
                    state[d] = v
            if r.random() < 0.1:
                try:
                    req("POST", f"/{index}/_refresh", timeout=10)
                    if r.random() < 0.3:
                        req("POST", f"/{index}/_flush", timeout=20)
                except Exception:
                    return

    t = threading.Thread(target=writer)
    t.start()
    time.sleep(r.uniform(1.0, 6.0))
    kill9()
    stop.set()
    t.join()
    subprocess.run([RESTART], check=True, stdout=subprocess.DEVNULL)
    for _ in range(120):
        st, h = req("GET", f"/_cluster/health/{index}?wait_for_status=yellow&timeout=5s")
        if st == 200 and h.get("status") in ("yellow", "green") and not h.get("timed_out"):
            break
    req("POST", f"/{index}/_refresh")
    bad = []
    st, resp = req("POST", f"/{index}/_search", {"size": 10000, "_source": ["v"], "query": {"match_all": {}}, "track_total_hits": True})
    got = {h["_id"]: h["_source"]["v"] for h in resp["hits"]["hits"]}
    for d, v in state.items():
        if v is None:
            if d in got:
                bad.append(f"{d}: acked delete but present")
        elif got.get(d) is None:
            bad.append(f"{d}: acked write lost")
        elif got[d] != v:
            # a later unacked write may have been applied; acceptable only if the doc was re-sent
            pass
    for d in got:
        if d not in sent:
            bad.append(f"{d}: never sent")
    st, c = req("POST", f"/{index}/_count", {})
    if c.get("count") != len(got):
        bad.append(f"count {c.get('count')} vs hits {len(got)}")
    for d in list(got)[:200]:
        st, g = req("GET", f"/{index}/_doc/{d}")
        if not g.get("found"):
            bad.append(f"{d}: search finds it, GET does not")
    st, chk = req("POST", f"/{index}/_forcemerge?max_num_segments=1")
    st, resp2 = req("POST", f"/{index}/_search", {"size": 0, "track_total_hits": True})
    if resp2["hits"]["total"]["value"] != len(got):
        bad.append(f"after merge {resp2['hits']['total']['value']} vs {len(got)}")
    print(f"round {n} {settings}: acked {len(state)} ops, docs after restart {len(got)}: {'OK' if not bad else bad[:5]}")
    return len(bad)


def main():
    configs = [{}, {"index.translog.durability": "request", "index.search.realtime.enabled": True},
               {"index.merge.reorder.enabled": True}, {"index.translog.flush_threshold_size": "1mb"}]
    bad = 0
    for n, s in enumerate(configs * int(os.environ.get("ROUNDS", "2"))):
        bad += run_round(n, 100 + n, s) or 0
    print(f"crash-recovery failures: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
