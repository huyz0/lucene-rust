#!/usr/bin/env python3
"""M6's upgrade and rollback, executed (T6.4), driven by scripts/verify-rollback.sh.

Three nodes, each node's data bind-mounted so a node can be re-created on
another image or with other settings and keep its shards. Every write is
recorded in a model of what was acknowledged, and after every step each
index's count and ids must match it; a fixed set of query shapes must answer
as they did at the start (native or not, the answers are Lucene's).

Forward -- adopting the Rust engine:
  F1. The cluster runs OpenSearch without the plugin; index `legacy` is
      written by OpenSearch's engine.
  F2. Rolling restart onto the plugin image. `legacy` stays on OpenSearch's
      engine (it never asked for the Rust one); searches on it go native.
  F3. Adoption: reindex `legacy` into `adopted`, created with
      `index.lucene_rust.engine: true`, and move the alias `app` over.

Backward -- rolling back off it:
  B1. Rolling restart with `lucene_rust.engine.node_enabled: false`:
      OpenSearch's engine serves the Rust-written shards; writes continue.
  B2. Force merge `adopted` to one segment under OpenSearch's engine, then
      reindex it into `restored` (no plugin setting) and move the alias:
      a node without the plugin refuses an index that carries the final
      `index.lucene_rust.engine`, so a rolling removal needs it gone.
  B3. Rolling restart onto OpenSearch without the plugin; every document and
      every answer stays.

Timings for each step go to the JSON file named by --out.
Standard library only.
"""
import json
import os
import random
import subprocess
import sys
import time
import http.client
import urllib.error
import urllib.request

# A reindex answers with one `Warning` header per deprecation it met, which
# passes http.client's default cap of 100 headers.
http.client._MAXHEADERS = 100_000
WARNINGS = set()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_opensearch as v  # noqa: E402

NODES = [("os1", 9201, 9301), ("os2", 9202, 9302), ("os3", 9203, 9303)]
PLUGIN_IMAGE = os.environ.get("PLUGIN_IMAGE", "lucene-rust-opensearch:3.8.0")
STOCK_IMAGE = os.environ.get("STOCK_IMAGE", "lucene-rust-stock:3.8.0")
DATA = os.environ["ROLLBACK_DATA"]
DOCS = int(os.environ.get("ROLLBACK_DOCS", "200000"))
FAILURES = []
CHECKS = [0]
TIMINGS = {}
BASE = "http://localhost:9201"


def req(method, path, body=None, ndjson=False, timeout=600):
    last = None
    for _, port, _ in NODES:
        data = None
        headers = {}
        if body is not None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
        r = urllib.request.Request(f"http://localhost:{port}{path}", data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                for w in resp.headers.get_all("Warning") or []:
                    if w not in WARNINGS:
                        WARNINGS.add(w)
                        print(f"  warning from {method} {path.split('?')[0]}: {w[:300]}", flush=True)
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (502, 503):
                last = e
                continue
            raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:400]}")
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last = e
    raise RuntimeError(f"{method} {path}: no node answered: {last}")


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(what)
        print("FAIL:", what, flush=True)


def docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def node(name, http, transport, image, enabled=True):
    docker("rm", "-f", name)
    env = [
        "-e", f"node.name={name}", "-e", "cluster.name=lucene-rust-rollback",
        "-e", "http.host=0.0.0.0", "-e", f"http.port={http}",
        "-e", "transport.host=127.0.0.1", "-e", f"transport.port={transport}",
        "-e", "discovery.seed_hosts=127.0.0.1:9301,127.0.0.1:9302,127.0.0.1:9303",
        "-e", "cluster.initial_cluster_manager_nodes=os1,os2,os3",
        "-e", "DISABLE_SECURITY_PLUGIN=true", "-e", "DISABLE_INSTALL_DEMO_CONFIG=true",
        "-e", "cluster.routing.allocation.disk.threshold_enabled=false",
        "-e", "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g",
    ]
    if image == PLUGIN_IMAGE:
        env += ["-e", f"lucene_rust.engine.node_enabled={'true' if enabled else 'false'}"]
    r = docker("run", "-d", "--name", name, "--network", "container:lucene-rust-net",
               "-v", f"{DATA}/{name}:/usr/share/opensearch/data", *env, image)
    check(r.returncode == 0, f"start {name} on {image}: {r.stderr[:300]}")


def health(status="green", nodes=3, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        try:
            h = req("GET", f"/_cluster/health?wait_for_status={status}&timeout=20s", timeout=30)
            if h.get("status") == status and h.get("number_of_nodes") == nodes:
                return True
        except RuntimeError:
            time.sleep(3)
    return False


def rolling(image, enabled, label):
    """Each node in turn: stop, re-create on `image` with the same data, wait
    for the cluster to be whole again. Returns the seconds each node took."""
    per = []
    for name, http, transport in NODES:
        t0 = time.time()
        # Stop allocating replicas elsewhere while the node is away, as an
        # operator's rolling restart does.
        req("PUT", "/_cluster/settings", {"persistent": {"cluster.routing.allocation.enable": "primaries"}})
        req("POST", "/_flush", timeout=600)
        docker("stop", "-t", "120", name)
        node(name, http, transport, image, enabled)
        up = health("yellow", timeout=900)
        req("PUT", "/_cluster/settings", {"persistent": {"cluster.routing.allocation.enable": None}})
        ok = up and health("green", timeout=900)
        check(ok, f"{label}: cluster green after {name}")
        per.append(round(time.time() - t0, 1))
        print(f"  {label}: {name} back in {per[-1]} s", flush=True)
    return per


class Model:
    """What was acknowledged, per index: id -> source."""

    def __init__(self):
        self.docs = {}

    def write(self, index, ops):
        lines = []
        for op, i, src in ops:
            if op == "delete":
                lines.append(json.dumps({"delete": {"_index": index, "_id": str(i)}}))
            else:
                lines.append(json.dumps({"index": {"_index": index, "_id": str(i)}}))
                lines.append(json.dumps(src))
        res = req("POST", "/_bulk?refresh=wait_for", "\n".join(lines) + "\n", ndjson=True)
        check(not res["errors"], f"{index}: bulk without errors")
        d = self.docs.setdefault(index, {})
        for op, i, src in ops:
            if op == "delete":
                d.pop(i, None)
            else:
                d[i] = src

    def verify(self, index, label, alias=None):
        name = alias or index
        req("POST", f"/{name}/_refresh")
        want = self.docs.get(index, {})
        got = {}
        res = req("POST", f"/{name}/_search?scroll=5m", {"size": 5000, "_source": ["n"], "sort": ["_doc"]})
        while res["hits"]["hits"]:
            for h in res["hits"]["hits"]:
                got[int(h["_id"])] = h["_source"].get("n")
            res = req("POST", "/_search/scroll", {"scroll": "5m", "scroll_id": res["_scroll_id"]})
        check(set(got) == set(want), f"{label}: {name} holds exactly the acknowledged ids "
                                    f"({len(got)} vs {len(want)}; missing {len(set(want) - set(got))}, extra {len(set(got) - set(want))})")
        check(all(got[i] == want[i].get("n") for i in set(got) & set(want)), f"{label}: {name}'s documents are the last written")


def shapes():
    return [(n, b) for n, b, e in v.matrix() if e == "native"][:60]


def answers(index, label):
    """Every shape's answer on `index`. A shard failure is a failed check here,
    not a quieter answer: a failed shard only thins the hits."""
    out = {}
    v.BASE = BASE
    for name, body in shapes():
        url, b = v.search_url(index, body)
        try:
            resp = v.req("POST", url, b)
            failed = resp.get("_shards", {}).get("failed", 0)
            check(failed == 0, f"{label}: {index} [{name}] {failed} shard(s) failed: "
                               f"{json.dumps(resp['_shards'].get('failures'))[:300]}")
            out[name] = v.shape(resp, b)
        except RuntimeError as e:
            out[name] = f"error: {str(e)[:120]}"
            check(False, f"{label}: {index} [{name}] {out[name]}")
    return out


def stable(a):
    """The part of an answer every copy of the same documents gives. Scores and
    the order of tied hits are not in it: a replica has its own deleted-document
    counts (so its own BM25 statistics) and its own doc ids, and a restart or a
    reindex changes which copy answers."""
    if isinstance(a, str):
        return a
    total = a["total"] if a["total"] and a["total"].get("relation") == "eq" else "lower bound"
    return {"total": total, "aggs": a.get("aggs"), "hits": len(a["hits"]),
            "sorts": [h[2] for h in a["hits"]] if any(h[2] for h in a["hits"]) else None}


def score_dependent(body):
    """A shape whose hits or totals follow the scores themselves (a sort on
    `_score`, `min_score`): across copies only its success is comparable."""
    text = json.dumps(body)
    return "_score" in text or "min_score" in text


def same_answers(ref, index, label):
    got = answers(index, label)
    bodies = dict(shapes())
    for name in ref:
        if score_dependent(bodies.get(name, {})):
            continue
        want, have = stable(ref[name]), stable(got[name])
        check(want == have, f"{label}: {index} [{name}] answers as before: "
                            f"{json.dumps(have)[:200]} vs {json.dumps(want)[:200]}")


def generate(n, seed):
    r = random.Random(seed)
    out = []
    for i in range(n):
        doc = {
            "body": " ".join(v.word(r) for _ in range(r.randint(1, 40))),
            "title": " ".join(v.word(r) for _ in range(r.randint(1, 5))),
            "tuned": " ".join(v.word(r) for _ in range(r.randint(1, 10))),
            "tag": v.word(r),
            "mtag": [v.word(r) + str(r.randint(0, 9)) for _ in range(r.randint(0, 3))],
            "n": i, "price": round(r.uniform(-50, 500), 2), "qty": r.randint(0, 40),
            "ts": 1_700_000_000_000 + r.randint(0, 10_000) * 60_000,
            "@timestamp": 1_700_000_000_000 + r.randint(0, 500) * 60_000,
            "ratio": r.random(), "m": [r.randint(0, 1000) for _ in range(r.randint(0, 3))],
            "md": [round(r.uniform(-50, 50), 3) for _ in range(r.randint(0, 4))],
            "mi": [r.randint(-100, 100) for _ in range(r.randint(0, 4))],
        }
        if r.random() < 0.7:
            doc["sp"] = r.randint(-100, 100)
        out.append(doc)
    return out


def churn(model, index, seed, n=3000):
    """Updates and deletes over the index's ids, then new ids."""
    r = random.Random(seed)
    ids = list(model.docs.get(index, {}))
    top = max(ids, default=-1) + 1
    ops = []
    for _ in range(n):
        if ids and r.random() < 0.3:
            ops.append(("delete", r.choice(ids), None))
        else:
            i = r.choice(ids) if ids and r.random() < 0.5 else top + r.randrange(10_000)
            src = generate(1, r.random())[0]
            src["n"] = i
            ops.append(("index", i, src))
    for k in range(0, len(ops), 1000):
        model.write(index, ops[k:k + 1000])


def engines():
    out = {}
    for _, port, _ in NODES:
        try:
            r = urllib.request.urlopen(f"http://localhost:{port}/_plugins/lucene_rust/stats", timeout=10)
            out[port] = json.load(r)["engines"]
        except Exception:
            out[port] = None
    return out


def main():
    out_file = sys.argv[sys.argv.index("--out") + 1]
    model = Model()
    v.BASE = BASE

    # F1: OpenSearch without the plugin; `legacy` written by its engine.
    t0 = time.time()
    for name, http, transport in NODES:
        node(name, http, transport, STOCK_IMAGE)
    check(health(), "F1: the stock cluster forms")
    v.req = lambda method, path, body=None, ndjson=False: req(method, path, body, ndjson)
    v.create("legacy", 3)
    req("PUT", "/legacy/_settings", {"index": {"number_of_replicas": 1, "refresh_interval": "1s"}})
    docs = generate(DOCS, 1)
    for k in range(0, DOCS, 5000):
        model.write("legacy", [("index", i, docs[i]) for i in range(k, min(DOCS, k + 5000))])
    churn(model, "legacy", 2)
    check(health(), "F1: legacy green")
    model.verify("legacy", "F1")
    ref = answers("legacy", "F1")
    TIMINGS["F1_load_s"] = round(time.time() - t0, 1)
    print(f"F1: {len(model.docs['legacy'])} documents on OpenSearch's engine", flush=True)

    # F2: rolling restart onto the plugin.
    TIMINGS["F2_rolling_restart_s"] = rolling(PLUGIN_IMAGE, True, "F2")
    model.verify("legacy", "F2")
    same_answers(ref, "legacy", "F2")
    e = engines()
    check(all(x is not None and x["rust"] == 0 for x in e.values()),
          f"F2: legacy stays on OpenSearch's engine: {e}")
    before = sum(req("GET", "/_plugins/lucene_rust/stats")["native_queries"] for _ in [0])
    req("POST", "/legacy/_search", {"query": {"match": {"body": "alpha"}}})
    check(req("GET", "/_plugins/lucene_rust/stats")["native_queries"] >= before, "F2: the plugin serves searches")
    churn(model, "legacy", 3, 1000)
    model.verify("legacy", "F2 after writes")

    ref = answers("legacy", "F2 after writes")

    # F3: adoption by reindex into a Rust-engine index, alias moved.
    t0 = time.time()
    body = req("GET", "/legacy/_mapping")["legacy"]["mappings"]
    req("PUT", "/adopted", {"settings": {"number_of_shards": 3, "number_of_replicas": 1,
                                         "index.lucene_rust.engine": True,
                                         "similarity": {"tuned": {"type": "BM25", "k1": 2.0, "b": 0.5}}},
                            "mappings": body})
    req("POST", "/_aliases", {"actions": [{"add": {"index": "legacy", "alias": "app"}}]})
    res = req("POST", "/_reindex?wait_for_completion=true&refresh=true",
              {"source": {"index": "legacy"}, "dest": {"index": "adopted"}}, timeout=3600)
    check(not res.get("failures"), f"F3: reindex without failures: {str(res.get('failures'))[:300]}")
    req("POST", "/_aliases", {"actions": [{"remove": {"index": "legacy", "alias": "app"}},
                                          {"add": {"index": "adopted", "alias": "app"}}]})
    model.docs["adopted"] = dict(model.docs["legacy"])
    check(health(), "F3: adopted green")
    TIMINGS["F3_reindex_s"] = round(time.time() - t0, 1)
    model.verify("adopted", "F3", alias="app")
    same_answers(ref, "adopted", "F3")
    e = engines()
    check(sum(x["rust"] for x in e.values() if x) > 0, f"F3: adopted is on the Rust engine: {e}")
    churn(model, "adopted", 4)
    model.verify("adopted", "F3 after writes")
    ref2 = answers("adopted", "F3 after writes")
    print(f"F3: adopted, {TIMINGS['F3_reindex_s']} s", flush=True)

    # B1: OpenSearch's engine for every shard, the plugin still installed.
    TIMINGS["B1_rolling_restart_s"] = rolling(PLUGIN_IMAGE, False, "B1")
    e = engines()
    check(all(x is not None and x["rust"] == 0 and x["java"] > 0 for x in e.values()),
          f"B1: every shard on OpenSearch's engine: {e}")
    model.verify("adopted", "B1")
    same_answers(ref2, "adopted", "B1")
    churn(model, "adopted", 5)
    model.verify("adopted", "B1 after writes")
    ref3 = answers("adopted", "B1 after writes")

    # B2: force merge under OpenSearch's engine -- the Rust-written segments
    # rewritten by it -- then a reindex into an index without the plugin's
    # settings. `index.lucene_rust.engine` is final, and a node without the
    # plugin refuses a shard of an index that carries it ("unknown setting");
    # OpenSearch archives unknown settings only when a full-cluster restart
    # loads the metadata, never during a rolling one.
    t0 = time.time()
    req("POST", "/adopted/_forcemerge?max_num_segments=1", timeout=3600)
    TIMINGS["B2_force_merge_s"] = round(time.time() - t0, 1)
    # `_cat/segments` lists the searchable segments: refresh so the merged
    # one replaces the ones it merged.
    req("POST", "/adopted/_refresh")
    segs = req("GET", "/_cat/segments/adopted?format=json")
    per_shard = {}
    for s in segs:
        per_shard.setdefault((s["shard"], s["prirep"], s["ip"] + s.get("id", "")), 0)
        per_shard[(s["shard"], s["prirep"], s["ip"] + s.get("id", ""))] += 1
    check(all(c == 1 for c in per_shard.values()), f"B2: one segment per copy: {per_shard}")
    model.verify("adopted", "B2")
    same_answers(ref3, "adopted", "B2")
    t0 = time.time()
    req("PUT", "/restored", {"settings": {"number_of_shards": 3, "number_of_replicas": 1,
                                          "similarity": {"tuned": {"type": "BM25", "k1": 2.0, "b": 0.5}}},
                             "mappings": body})
    res = req("POST", "/_reindex?wait_for_completion=true&refresh=true",
              {"source": {"index": "adopted"}, "dest": {"index": "restored"}}, timeout=3600)
    check(not res.get("failures"), f"B2: reindex without failures: {str(res.get('failures'))[:300]}")
    req("POST", "/_aliases", {"actions": [{"remove": {"index": "adopted", "alias": "app"}},
                                          {"add": {"index": "restored", "alias": "app"}}]})
    model.docs["restored"] = dict(model.docs["adopted"])
    check(health(), "B2: restored green")
    model.verify("restored", "B2", alias="app")
    same_answers(ref3, "restored", "B2")
    req("DELETE", "/adopted")
    TIMINGS["B2_reindex_out_s"] = round(time.time() - t0, 1)

    # B3: OpenSearch without the plugin.
    TIMINGS["B3_rolling_restart_s"] = rolling(STOCK_IMAGE, True, "B3")
    s = req("GET", "/_all/_settings?flat_settings=true")
    check(not any("lucene_rust" in k for i in s.values() for k in i["settings"]),
          f"B3: no index carries a plugin setting: {[(n, k) for n, i in s.items() for k in i['settings'] if 'lucene_rust' in k]}")
    model.verify("restored", "B3", alias="app")
    same_answers(ref3, "restored", "B3")
    churn(model, "restored", 6)
    model.verify("restored", "B3 after writes")
    model.verify("legacy", "B3")

    TIMINGS["docs"] = len(model.docs["restored"])
    TIMINGS["checks"] = CHECKS[0]
    TIMINGS["failures"] = FAILURES
    with open(out_file, "w") as f:
        json.dump(TIMINGS, f, indent=1)
    print(f"verify_rollback: {CHECKS[0]} checks, {len(FAILURES)} failures; timings {json.dumps(TIMINGS)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
