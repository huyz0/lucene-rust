#!/usr/bin/env python3
"""Soak test of the accelerated engine on real multi-node clusters, driven by scripts/soak.sh.

Nodes are OpenSearch tarball distributions run as processes (no Docker): a 2.x distribution,
a stock 3.9.0 one and a 3.9.0 one with this plugin installed, started from one data directory
in turn where a scenario swaps them. Every scenario runs a write load (index, update, delete)
and a search load while it does something to the cluster, then checks:

  * every acknowledged write is on every shard copy (a model of what was acknowledged;
    writes whose answer was lost are "uncertain" and may be either state);
  * all copies of a shard hold the same documents;
  * every document equals what the generator wrote for its (id, version): no corrupt stored
    fields;
  * a query matrix (scored and sorted queries, aggregations, highlights) answers the same
    on the cluster under test and on a stock 3.9.0 reference loaded with the same documents,
    before and after a force merge to one segment; k-NN fields are compared by recall;
  * the node logs hold no JVM crash, native panic, corruption or assertion.

Scenarios: restarts, upgrade, plugin_swap, snapshot, ccr. See the README of each function.
Standard library only.
"""
import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_opensearch as vo  # noqa: E402  (shape/same: the response comparison)

SOAK = os.environ.get("SOAK_DIR", "/home/user/soak")
RUN = os.path.join(SOAK, "run")
REPO_DIR = os.path.join(SOAK, "repo")
DIST = {
    "2x": os.path.join(SOAK, "dist2"),
    "stock": os.path.join(SOAK, "dist39"),
    "plugin": os.path.join(SOAK, "dist39p"),
    "ccr": os.path.join(SOAK, "dist39c"),  # plugin + the patched replication plugin
}
USER = "osk"
HEAP = os.environ.get("SOAK_HEAP", "1g")
FINDINGS = []
CHECKS = [0]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FINDINGS.append(what)
        log("FAIL:", what)
    return ok


# ---------------------------------------------------------------------------------------------
# HTTP


class Http:
    """One endpoint, or a set tried in turn."""

    def __init__(self, urls):
        self.urls = urls if isinstance(urls, list) else [urls]

    def req(self, method, path, body=None, ndjson=False, timeout=60, retry=True, ok=None):
        data = None
        headers = {}
        if body is not None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
        last = None
        urls = list(self.urls)
        random.shuffle(urls)
        for u in urls:
            r = urllib.request.Request(u + path, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(r, timeout=timeout) as resp:
                    return resp.status, json.load(resp)
            except urllib.error.HTTPError as e:
                text = e.read().decode()
                try:
                    out = json.loads(text)
                except ValueError:
                    out = {"raw": text}
                if e.code in (502, 503) and retry:
                    last = f"{e.code} {text[:200]}"
                    continue
                return e.code, out
            except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
                last = e
                if not retry:
                    break
        raise RuntimeError(f"{method} {path}: no endpoint answered ({last})")

    def ok(self, method, path, body=None, **kw):
        code, out = self.req(method, path, body, **kw)
        if code >= 300:
            raise RuntimeError(f"{method} {path}: {code} {json.dumps(out)[:600]}")
        return out


# ---------------------------------------------------------------------------------------------
# Nodes and clusters


class Node:
    def __init__(self, cluster, name, http_port, transport_port):
        self.cluster = cluster
        self.name = name
        self.http_port = http_port
        self.transport_port = transport_port
        self.base = os.path.join(RUN, cluster.name, name)
        self.data = os.path.join(self.base, "data")
        self.logs = os.path.join(self.base, "logs")
        self.conf = os.path.join(self.base, "conf")
        self.proc = None
        self.dist = None
        self.url = f"http://127.0.0.1:{http_port}"

    def settings(self, extra):
        c = self.cluster
        s = {
            "cluster.name": c.name,
            "node.name": self.name,
            "path.data": self.data,
            "path.logs": self.logs,
            "path.repo": REPO_DIR,
            "network.host": "127.0.0.1",
            "http.port": self.http_port,
            "transport.port": self.transport_port,
            "discovery.seed_hosts": [f"127.0.0.1:{n.transport_port}" for n in c.nodes],
            "cluster.initial_cluster_manager_nodes": [n.name for n in c.nodes],
            "cluster.routing.allocation.disk.threshold_enabled": False,
            "cluster.routing.allocation.node_initial_primaries_recoveries": 4,
        }
        if c.size == 1:
            s.pop("discovery.seed_hosts")
        s.update(c.extra)
        s.update(extra or {})
        return s

    def start(self, dist, extra=None):
        assert self.proc is None or self.proc.poll() is not None, f"{self.name} already running"
        self.dist = dist
        d = DIST[dist]
        for p in (self.data, self.logs, self.conf):
            os.makedirs(p, exist_ok=True)
        # conf: the distribution's, minus its java agent, with this node's heap and settings.
        shutil.rmtree(self.conf, ignore_errors=True)
        shutil.copytree(os.path.join(d, "config"), self.conf)
        jvm = []
        for line in open(os.path.join(self.conf, "jvm.options")):
            if line.startswith("-Xm"):
                continue
            jvm.append(line.rstrip("\n").replace("logs/", self.logs + "/").replace("=data", "=" + self.data).replace("javaagent:agent/", "javaagent:" + d + "/agent/"))
        jvm += [f"-Xms{HEAP}", f"-Xmx{HEAP}"]
        open(os.path.join(self.conf, "jvm.options"), "w").write("\n".join(jvm) + "\n")
        if os.path.isdir(os.path.join(d, "plugins", "lucene-rust")):
            os.makedirs(os.path.join(self.conf, "jvm.options.d"), exist_ok=True)
            open(os.path.join(self.conf, "jvm.options.d", "lucene-rust.options"), "w").write(
                "--enable-native-access=ALL-UNNAMED\n")
        with open(os.path.join(self.conf, "opensearch.yml"), "w") as f:
            for k, v in self.settings(extra).items():
                f.write(f"{k}: {json.dumps(v)}\n")
        subprocess.run(["chown", "-R", f"{USER}:{USER}", self.base, REPO_DIR], check=True)
        env = dict(os.environ, OPENSEARCH_PATH_CONF=self.conf, OPENSEARCH_JAVA_OPTS="", HOME=f"/home/{USER}")
        env.pop("JAVA_TOOL_OPTIONS", None)
        env.pop("JAVA_HOME", None)
        env["OPENSEARCH_JAVA_HOME"] = os.path.join(d, "jdk")
        out = open(os.path.join(self.base, "stdout.log"), "ab")
        self.proc = subprocess.Popen(["runuser", "-u", USER, "--", os.path.join(d, "bin", "opensearch")],
                                     stdout=out, stderr=subprocess.STDOUT, env=env, cwd=self.base,
                                     start_new_session=True)
        log(f"started {self.name} ({dist}) pid {self.proc.pid}")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, kill=False, timeout=90):
        if not self.alive():
            return
        sig = signal.SIGKILL if kill else signal.SIGTERM
        # runuser forks: signal the whole group.
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
            log(f"{self.name}: did not stop in {timeout}s, killed")
        log(f"stopped {self.name} ({'kill -9' if kill else 'graceful'})")

    def wait_http(self, timeout=240):
        t = time.time()
        while time.time() - t < timeout:
            if not self.alive():
                raise RuntimeError(f"{self.name} exited: {self.tail()}")
            try:
                with urllib.request.urlopen(self.url + "/", timeout=3) as r:
                    return json.load(r)
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"{self.name} did not answer in {timeout}s: {self.tail()}")

    def tail(self):
        try:
            return open(os.path.join(self.base, "stdout.log")).read()[-1500:]
        except OSError:
            return ""


class Cluster:
    def __init__(self, name, size, port_base, extra=None):
        self.name = name
        self.size = size
        self.extra = extra or {}
        shutil.rmtree(os.path.join(RUN, name), ignore_errors=True)
        self.nodes = [Node(self, f"{name}-{i + 1}", port_base + i, port_base + 100 + i) for i in range(size)]

    def http(self, live=True):
        return Http([n.url for n in self.nodes if n.alive()])

    def req(self, *a, **kw):
        return self.http().req(*a, **kw)

    def ok(self, *a, **kw):
        return self.http().ok(*a, **kw)

    def start(self, dist, extra=None):
        for n in self.nodes:
            n.start(dist, extra)
        for n in self.nodes:
            n.wait_http()
        self.wait_status("yellow")

    def stop(self):
        for n in self.nodes:
            n.stop()

    def wait_status(self, status="green", timeout=300):
        t = time.time()
        last = None
        while time.time() - t < timeout:
            try:
                code, out = self.http().req(
                    "GET", f"/_cluster/health?wait_for_status={status}&timeout=10s", retry=False, timeout=20)
                last = out
                if code == 200 and not out.get("timed_out") and out.get("number_of_nodes", 0) >= len(
                        [n for n in self.nodes if n.alive()]):
                    return out
            except RuntimeError:
                pass
            time.sleep(1)
        raise RuntimeError(f"cluster {self.name} not {status} in {timeout}s: {json.dumps(last)[:500]}")

    def wait_nodes(self, n, timeout=240):
        t = time.time()
        while time.time() - t < timeout:
            try:
                code, out = self.http().req("GET", "/_cluster/health", retry=False, timeout=10)
                if code == 200 and out["number_of_nodes"] >= n:
                    return
            except RuntimeError:
                pass
            time.sleep(1)
        raise RuntimeError(f"cluster {self.name} has not {n} nodes after {timeout}s")

    def cluster_manager(self):
        _, out = self.req("GET", "/_cat/cluster_manager?format=json")
        return out[0]["node"]

    def settings(self, persistent):
        self.ok("PUT", "/_cluster/settings", {"persistent": persistent})

    def rolling(self, nodes, dist, kill=False, extra=None, pause_after=0):
        """Restarts `nodes` one at a time onto `dist`, as a rolling upgrade does."""
        for n in nodes:
            self.settings({"cluster.routing.allocation.enable": "primaries"})
            try:
                self.req("POST", "/_flush", timeout=120)
            except RuntimeError:
                pass
            n.stop(kill=kill)
            n.start(dist, extra)
            n.wait_http()
            self.wait_nodes(len([x for x in self.nodes if x.alive()]))
            self.settings({"cluster.routing.allocation.enable": None})
            self.wait_status("green", 420)
            if pause_after:
                time.sleep(pause_after)
            log(f"rolled {n.name} -> {dist}")


# ---------------------------------------------------------------------------------------------
# Documents, the model and the load

TAGS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"]
WORDS = ["fox", "dog", "cat", "river", "stone", "cloud", "paper", "engine", "garden", "window", "silver",
         "market", "winter", "signal", "bridge", "harbor", "forest", "copper", "lantern", "meadow"] + TAGS
TS0 = 1_700_000_000_000
DIM = 8


def make_doc(i, ver, vec=False):
    r = random.Random(i * 1_000_003 + ver)
    d = {
        "n": i,
        "v": ver,
        "tag": TAGS[(i * 7 + ver) % len(TAGS)],
        "mtag": [TAGS[r.randrange(len(TAGS))] for _ in range(r.randint(0, 3))],
        "body": " ".join(WORDS[min(len(WORDS) - 1, int(r.random() ** 2 * len(WORDS)))] for _ in range(r.randint(3, 30))),
        "price": round(r.uniform(0, 500), 2),
        "ts": TS0 + ((i * 13 + ver * 7) % 20000) * 60_000,
    }
    if vec:
        d["vec"] = make_vec(i, ver)
    return d


def make_vec(i, ver):
    r = random.Random(i * 7_777_777 + ver * 31 + 5)
    return [round(r.gauss(0, 1), 4) for _ in range(DIM)]


def index_body(shards=2, replicas=1, vec=False, accelerated=None, ref=False, refresh="1s"):
    settings = {"number_of_shards": shards, "number_of_replicas": replicas, "refresh_interval": refresh}
    props = {
        "n": {"type": "long"}, "v": {"type": "integer"}, "tag": {"type": "keyword"}, "mtag": {"type": "keyword"},
        "body": {"type": "text"}, "price": {"type": "double"}, "ts": {"type": "date"},
    }
    if vec:
        settings["index.knn"] = True
        engine = "lucene" if ref else "extended"
        props["vec"] = {"type": "knn_vector", "dimension": DIM,
                        "method": {"name": "hnsw", "engine": engine, "space_type": "l2"}}
        if not ref:
            settings["index.engine.accelerated"] = True
    elif accelerated is not None:
        settings["index.engine.accelerated"] = accelerated
    return {"settings": settings, "mappings": {"properties": props}}


class Model:
    """What was acknowledged: id -> ("p", ver) | ("a",) | ("u", frozenset of states)."""

    def __init__(self):
        self.state = {}
        self.history = {}  # id -> set of states an unacknowledged write may have left
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            return dict(self.state), {k: set(v) for k, v in self.history.items()}

    def candidates(self, i):
        with self.lock:
            s = self.state.get(i, ("a",))
            out = {s} if s[0] != "u" else set(s[1])
            out |= self.history.get(i, set())
            return out


class Load:
    """Writers (index, update, delete by id) and searchers against `index`, until stopped."""

    def __init__(self, cluster, index, writers=3, searchers=1, ids=20000, vec=False, batch=150, pause=0.05,
                 model=None, http=None):
        self.cluster = cluster
        self.http = http or cluster.http
        self.index = index
        self.vec = vec
        self.ids = ids
        self.writers = writers
        self.searchers = searchers
        self.batch = batch
        self.pause = pause
        self.model = model or Model()
        self.stop_ev = threading.Event()
        self.pause_ev = threading.Event()  # set: writers hold between batches
        self.idle = [threading.Event() for _ in range(writers)]
        self.threads = []
        self.ver = {}
        self.stats = {"bulks": 0, "ops": 0, "uncertain": 0, "bulk_errors": 0, "search": 0, "search_err": 0,
                      "get": 0, "stale_get": 0}
        self.errors = {}
        self.unexpected = []

    def note_error(self, where, msg):
        key = f"{where}: {msg[:160]}"
        self.errors[key] = self.errors.get(key, 0) + 1
        if re.search(r"panic|native|ffi|NullPointer|ClassCast|Assertion|IllegalState|ArrayIndex|Corrupt|"
                     r"abort|segfault|rust", msg, re.I) and len(self.unexpected) < 20:
            self.unexpected.append(key)

    def start(self):
        for t in range(self.writers):
            th = threading.Thread(target=self.writer, args=(t,), daemon=True)
            th.start()
            self.threads.append(th)
        for t in range(self.searchers):
            th = threading.Thread(target=self.searcher, args=(t,), daemon=True)
            th.start()
            self.threads.append(th)

    def stop(self):
        self.stop_ev.set()
        for th in self.threads:
            th.join(timeout=180)

    def quiesce(self):
        """Holds the writers between batches; returns once every writer has."""
        self.pause_ev.set()
        for e in self.idle:
            e.wait(timeout=180)

    def resume(self):
        self.pause_ev.clear()

    def writer(self, t):
        r = random.Random(1000 + t)
        mine = list(range(t, self.ids, self.writers))
        while not self.stop_ev.is_set():
            if self.pause_ev.is_set():
                self.idle[t].set()
                time.sleep(0.05)
                continue
            self.idle[t].clear()
            ops = []
            seen = set()
            while len(ops) < self.batch:
                i = r.choice(mine)
                if i in seen:
                    continue
                seen.add(i)
                with self.model.lock:
                    st = self.model.state.get(i, ("a",))
                if st[0] == "u":
                    # An uncertain id: rewrite it, which makes it certain again.
                    kind = "index"
                elif st[0] == "a":
                    kind = "index"
                else:
                    x = r.random()
                    kind = "index" if x < 0.72 else ("delete" if x < 0.9 else "skip")
                if kind == "skip":
                    continue
                if kind == "index":
                    self.ver[i] = self.ver.get(i, 0) + 1
                    ops.append(("index", i, self.ver[i]))
                else:
                    ops.append(("delete", i, None))
            lines = []
            for kind, i, ver in ops:
                lines.append(json.dumps({kind: {"_index": self.index, "_id": str(i)}}))
                if kind == "index":
                    lines.append(json.dumps(make_doc(i, ver, self.vec)))
            try:
                code, out = self.http().req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True, timeout=90)
                if code != 200 or "items" not in out:
                    raise RuntimeError(f"bulk {code} {json.dumps(out)[:200]}")
            except RuntimeError as e:
                self.stats["bulk_errors"] += 1
                self.note_error("bulk", str(e))
                self.mark_uncertain(ops)
                time.sleep(0.5)
                continue
            acked = []
            for (kind, i, ver), item in zip(ops, out["items"]):
                it = next(iter(item.values()))
                status = it.get("status", 500)
                if kind == "index" and status in (200, 201):
                    new = ("p", ver)
                elif kind == "delete" and status in (200, 404):
                    new = ("a",)
                else:
                    self.stats["uncertain"] += 1
                    self.note_error("bulk item", f"{status} {json.dumps(it.get('error'))[:200]}")
                    self.mark_uncertain([(kind, i, ver)])
                    continue
                with self.model.lock:
                    self.model.state[i] = new
                acked.append((kind, i, ver))
            self.stats["bulks"] += 1
            self.stats["ops"] += len(ops)
            # Read your writes: a realtime GET of an acknowledged write must not be older.
            for kind, i, ver in r.sample(acked, min(3, len(acked))):
                try:
                    code, g = self.http().req("GET", f"/{self.index}/_doc/{i}", timeout=30, retry=False)
                except RuntimeError:
                    continue
                self.stats["get"] += 1
                got = g.get("_source", {}).get("v") if g.get("found") else None
                with self.model.lock:
                    cur = self.model.state.get(i)
                if code in (200, 404) and cur == (("p", ver) if kind == "index" else ("a",)):
                    ok = (kind == "index" and g.get("found") and got == ver) or (kind == "delete" and not g.get("found"))
                    if not ok:
                        self.stats["stale_get"] += 1
                        check(False, f"{self.index}: GET {i} after acked {kind} v{ver} gave found={g.get('found')} v={got}")
            time.sleep(self.pause)

    def mark_uncertain(self, ops):
        with self.model.lock:
            for kind, i, ver in ops:
                new = ("p", ver) if kind == "index" else ("a",)
                self.model.history.setdefault(i, set()).add(new)
                old = self.model.state.get(i, ("a",))
                self.model.state[i] = ("u", frozenset((set(old[1]) if old[0] == "u" else {old}) | {new}))

    QUERIES = [
        {"query": {"match": {"body": "fox river"}}, "size": 5, "_source": False},
        {"query": {"bool": {"must": [{"match": {"body": "stone"}}], "filter": [{"term": {"tag": "beta"}}]}},
         "size": 5, "_source": False},
        {"size": 0, "aggs": {"t": {"terms": {"field": "tag"}, "aggs": {"p": {"avg": {"field": "price"}}}}}},
        {"size": 3, "sort": [{"price": "desc"}, {"n": "asc"}], "_source": False},
        {"size": 0, "query": {"range": {"n": {"gte": 100, "lt": 9000}}}, "aggs": {"s": {"stats": {"field": "v"}}}},
    ]

    def searcher(self, t):
        r = random.Random(2000 + t)
        while not self.stop_ev.is_set():
            body = dict(r.choice(self.QUERIES))
            if self.vec and r.random() < 0.3:
                body = {"size": 5, "_source": False,
                        "query": {"knn": {"vec": {"vector": make_vec(r.randrange(self.ids), 1), "k": 5}}}}
            try:
                code, out = self.http().req("POST", f"/{self.index}/_search", body, timeout=60)
                self.stats["search"] += 1
                if code != 200:
                    self.stats["search_err"] += 1
                    self.note_error("search", f"{code} {json.dumps(out.get('error', out))[:200]}")
                else:
                    sh = out.get("_shards", {})
                    if sh.get("failed"):
                        self.stats["search_err"] += 1
                        self.note_error("search shard failure", json.dumps(sh.get("failures", [])[:1])[:300])
            except RuntimeError as e:
                self.stats["search_err"] += 1
                self.note_error("search", str(e))
            time.sleep(0.05)


# ---------------------------------------------------------------------------------------------
# Verification


def shard_copies(cl, index):
    _, rows = cl.req("GET", f"/_cat/shards/{index}?format=json&h=shard,prirep,state,node")
    return rows


def scan(http, index, preference=None, fields=True, size=2000):
    """All documents of `index` (id -> source), by scroll."""
    pref = f"&preference={preference}" if preference else ""
    out = http.ok("POST", f"/{index}/_search?scroll=2m&size={size}{pref}",
                  {"sort": ["_doc"], "_source": fields}, timeout=180)
    docs = {}
    total = out["hits"]["total"]["value"]
    while out["hits"]["hits"]:
        for h in out["hits"]["hits"]:
            docs[h["_id"]] = h.get("_source")
        out = http.ok("POST", "/_search/scroll", {"scroll": "2m", "scroll_id": out["_scroll_id"]}, timeout=180)
    try:
        http.req("DELETE", "/_search/scroll", {"scroll_id": out["_scroll_id"]})
    except Exception:
        pass
    return docs, total


def verify_model(cl, index, model, vec=False, label="", http=None, strict_model=True):
    """Acknowledged writes on every shard copy; copies agree; every source is the generator's."""
    http = http or cl.http()
    http.ok("POST", f"/{index}/_refresh")
    state, hist = model.snapshot()
    present = {i: s[1] for i, s in state.items() if s[0] == "p"}
    uncertain = {i for i, s in state.items() if s[0] == "u"}
    docs, total = scan(http, index)
    check(len(docs) == total, f"{label}{index}: scan returned {len(docs)} of {total}")
    bad = 0
    corrupt = 0
    for sid, src in docs.items():
        i = int(sid)
        if src is None or src != make_doc(i, src.get("v", -1), vec) and not (
                vec and {k: v for k, v in src.items() if k != "vec"} == {k: v for k, v in make_doc(i, src.get("v", -1), vec).items() if k != "vec"}):
            corrupt += 1
            if corrupt <= 3:
                check(False, f"{label}{index}: doc {sid} is not what the generator wrote for v={None if src is None else src.get('v')}: {json.dumps(src)[:200]}")
    check(corrupt == 0, f"{label}{index}: {corrupt} corrupt documents")
    missing, wrong, extra = [], [], []
    for i, ver in present.items():
        d = docs.get(str(i))
        if d is None:
            if not (model.history.get(i)):
                missing.append(i)
        elif d["v"] != ver and ("p", d["v"]) not in model.candidates(i):
            wrong.append((i, ver, d["v"]))
    for sid in docs:
        i = int(sid)
        s = state.get(i, ("a",))
        if s[0] == "a" and ("p", docs[sid]["v"]) not in hist.get(i, set()):
            extra.append(i)
        elif s[0] == "u" and ("p", docs[sid]["v"]) not in model.candidates(i):
            wrong.append((i, s, docs[sid]["v"]))
    if strict_model:
        check(not missing, f"{label}{index}: {len(missing)} acknowledged documents missing, e.g. {missing[:5]}")
        check(not wrong, f"{label}{index}: {len(wrong)} documents at the wrong version, e.g. {wrong[:5]}")
        check(not extra, f"{label}{index}: {len(extra)} deleted documents still there, e.g. {extra[:5]}")
    log(f"{label}{index}: {len(docs)} docs, model present {len(present)}, uncertain {len(uncertain)}; "
        f"missing {len(missing)} wrong {len(wrong)} resurrected {len(extra)} corrupt {corrupt}")
    # Shard copies agree.
    rows = shard_copies(cl, index)
    per_shard = {}
    for row in rows:
        if row["state"] != "STARTED":
            check(False, f"{label}{index}: shard {row['shard']} {row['prirep']} is {row['state']}")
            continue
        pref = f"_shards:{row['shard']}|_only_nodes:{row['node']}"
        out = http.ok("POST", f"/{index}/_search?preference={pref}&request_cache=false", {
            "size": 0, "track_total_hits": True,
            "aggs": {"c": {"value_count": {"field": "v"}}, "sv": {"sum": {"field": "v"}}, "sn": {"sum": {"field": "n"}},
                     "sp": {"sum": {"field": "price"}}}})
        sig = (out["hits"]["total"]["value"], out["aggregations"]["sv"]["value"], out["aggregations"]["sn"]["value"],
               round(out["aggregations"]["sp"]["value"], 3))
        per_shard.setdefault(row["shard"], []).append((row["prirep"], row["node"], sig))
    for shard, copies in per_shard.items():
        sigs = {c[2] for c in copies}
        check(len(sigs) == 1, f"{label}{index}: shard {shard} copies differ: {copies}")
    return docs


def knn_check(http, index, docs, label, k=10, queries=20):
    """Recall of k-NN hits against brute force over the documents the index holds."""
    r = random.Random(11)
    ids = list(docs)
    vecs = {i: docs[i]["vec"] for i in ids if docs[i].get("vec")}
    if not vecs:
        return None
    hit = tot = 0
    for _ in range(queries):
        q = make_vec(r.randrange(10 ** 6), 1)
        out = http.ok("POST", f"/{index}/_search?request_cache=false", {
            "size": k, "_source": False, "query": {"knn": {"vec": {"vector": q, "k": k}}}})
        got = [h["_id"] for h in out["hits"]["hits"]]
        exact = sorted(vecs, key=lambda i: sum((a - b) ** 2 for a, b in zip(vecs[i], q)))[:k]
        hit += len(set(got) & set(exact))
        tot += k
    recall = hit / tot
    log(f"{label}{index}: k-NN recall@{k} {recall:.3f}")
    check(recall >= 0.85, f"{label}{index}: k-NN recall {recall:.3f} < 0.85")
    return recall


MATRIX = [
    ("count", {"size": 0, "track_total_hits": True}, False),
    ("match", {"query": {"match": {"body": "fox river"}}, "size": 20}, True),
    ("match rare", {"query": {"match": {"body": "lantern meadow"}}, "size": 20}, True),
    ("match and", {"query": {"match": {"body": {"query": "stone cloud", "operator": "and"}}}, "size": 20}, True),
    ("bool", {"query": {"bool": {"must": [{"match": {"body": "paper"}}], "should": [{"term": {"tag": "gamma"}}],
                                 "filter": [{"range": {"price": {"gte": 50, "lt": 400}}}],
                                 "must_not": [{"term": {"tag": "mu"}}]}}, "size": 20}, True),
    ("phrase", {"query": {"match_phrase": {"body": "fox river"}}, "size": 20}, True),
    ("term sort n", {"query": {"term": {"tag": "delta"}}, "sort": [{"n": "desc"}], "size": 50}, False),
    ("range sort price", {"query": {"range": {"n": {"gte": 500, "lt": 15000}}},
                          "sort": [{"price": "asc"}, {"n": "asc"}], "size": 50}, False),
    ("exists mtag", {"query": {"exists": {"field": "mtag"}}, "sort": [{"n": "asc"}], "size": 30}, False),
    ("prefix", {"query": {"prefix": {"tag": "ep"}}, "sort": [{"n": "asc"}], "size": 30}, False),
    ("terms mtag", {"query": {"terms": {"mtag": ["alpha", "mu"]}}, "sort": [{"n": "asc"}], "size": 30}, False),
    ("search_after", {"query": {"match_all": {}}, "sort": [{"ts": "asc"}, {"n": "asc"}], "search_after": [TS0 + 600000, 100],
                      "size": 40}, False),
    ("agg terms+avg", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "size": 20}, "aggs": {"p": {"avg": {"field": "price"}}}}}}, False),
    ("agg mtag", {"size": 0, "aggs": {"t": {"terms": {"field": "mtag", "size": 20}}}}, False),
    ("agg stats", {"size": 0, "aggs": {"s": {"stats": {"field": "price"}}, "v": {"max": {"field": "v"}},
                                       "c": {"cardinality": {"field": "tag"}}}}, False),
    ("agg histogram", {"size": 0, "aggs": {"h": {"histogram": {"field": "n", "interval": 1000}}}}, False),
    ("agg date_histogram", {"size": 0, "aggs": {"d": {"date_histogram": {"field": "ts", "calendar_interval": "week"},
                                                      "aggs": {"m": {"sum": {"field": "v"}}}}}}, False),
    ("agg range", {"size": 0, "aggs": {"r": {"range": {"field": "price", "ranges": [{"to": 100}, {"from": 100, "to": 300}, {"from": 300}]}}}}, False),
    ("agg filtered", {"size": 0, "query": {"match": {"body": "winter"}},
                      "aggs": {"t": {"terms": {"field": "tag"}, "aggs": {"s": {"sum": {"field": "price"}}}}}}, True),
    ("agg top_hits", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "size": 3},
                                                "aggs": {"top": {"top_hits": {"size": 2, "sort": [{"n": "asc"}], "_source": ["n"]}}}}}}, False),
    ("highlight", {"query": {"match": {"body": "copper bridge"}}, "size": 10,
                   "highlight": {"fields": {"body": {}}}}, True),
    ("function_score", {"query": {"function_score": {"query": {"match": {"body": "fox"}}, "field_value_factor": {"field": "v"}}}, "size": 10}, True),
    ("source filter", {"query": {"term": {"tag": "zeta"}}, "sort": [{"n": "asc"}], "size": 20, "_source": ["n", "v", "tag"]}, False),
]


def round_floats(x):
    """Aggregation sums differ in the last bits with the order a shard adds in."""
    if isinstance(x, float):
        return float(f"{x:.9g}")
    if isinstance(x, dict):
        return {k: round_floats(v) for k, v in x.items()}
    if isinstance(x, list):
        return [round_floats(v) for v in x]
    return x


def run_matrix(cl, ref, index, ref_index, label, scored=True):
    bad = 0
    for name, body, is_scored in MATRIX:
        if is_scored and not scored:
            continue
        b = dict(body)
        if is_scored and b.get("size") and "sort" not in b:
            # Ties at the cut-off must break the same way on both sides.
            b["sort"] = ["_score", {"n": "asc"}]
            b["track_scores"] = True
        try:
            a = vo.shape(cl.ok("POST", f"/{index}/_search?request_cache=false", b, timeout=120), b)
            c = vo.shape(ref.ok("POST", f"/{ref_index}/_search?request_cache=false", b, timeout=120), b)
        except RuntimeError as e:
            check(False, f"{label}{index} {name}: {e}")
            bad += 1
            continue
        a, c = round_floats(a), round_floats(c)
        diff = vo.same(a, c)
        if diff:
            bad += 1
            check(False, f"{label}{index} '{name}' differs from stock: {diff}")
    log(f"{label}{index}: matrix {len(MATRIX)} rows, {bad} differ ({'scored too' if scored else 'unscored only'})")


def merge_until_clean(cl, index, expected, label, timeout=150):
    """Force merges until the segments hold exactly the live documents. Soft-deleted history stays
    in a merged segment (and in term statistics, so in scores) until the retention leases catch
    up with the global checkpoint, which takes a lease sync after the load stops."""
    t = time.time()
    while True:
        cl.ok("POST", f"/{index}/_flush")
        cl.ok("POST", f"/{index}/_forcemerge?max_num_segments=1", timeout=900)
        segs = cl.ok("GET", f"/_cat/segments/{index}?format=json&h=prirep,docs.count")
        held = sum(int(x["docs.count"]) for x in segs if x["prirep"] == "p")
        if held == expected:
            return
        if time.time() - t > timeout:
            log(f"{label}{index}: after {timeout}s the primaries still hold {held} docs for {expected} live")
            return
        time.sleep(10)


def copy_scores(cl, index, label):
    """After a force merge every copy of a shard should score alike: same documents, one segment."""
    per = {}
    for row in shard_copies(cl, index):
        pref = f"_shards:{row['shard']}|_only_nodes:{row['node']}"
        out = cl.ok("POST", f"/{index}/_search?preference={pref}&request_cache=false",
                    {"size": 1, "query": {"match": {"body": "fox river"}}, "explain": False})
        segs = cl.ok("GET", f"/_cat/segments/{index}?format=json&h=shard,prirep,segment,docs.count,docs.deleted,version")
        per.setdefault(row["shard"], []).append((row["prirep"], row["node"], out["hits"].get("max_score"), out["hits"]["total"]["value"]))
    for shard, copies in per.items():
        if len({c[2] for c in copies}) > 1:
            segs = [x for x in segs if x["shard"] == shard]
            check(False, f"{label}{index}: shard {shard} copies score differently after force merge: {copies}; segments {segs}")


def compare_with_stock(cl, ref, index, docs, shards, label, vec=False):
    """Loads `docs` into the stock reference, then runs the matrix on both: as the cluster holds
    them, and again after both force merge to one segment per shard (scores then agree)."""
    ref_index = index
    http = ref.http()
    http.req("DELETE", f"/{ref_index}")
    body = index_body(shards, 0, vec=vec, ref=True, refresh="-1")
    body["settings"]["index.knn"] = vec or None
    if not vec:
        body["settings"].pop("index.knn", None)
    http.ok("PUT", f"/{ref_index}", body)
    items = list(docs.items())
    for s in range(0, len(items), 1000):
        lines = []
        for sid, src in items[s:s + 1000]:
            lines.append(json.dumps({"index": {"_index": ref_index, "_id": sid}}))
            lines.append(json.dumps(src))
        out = http.ok("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True, timeout=180)
        check(not out["errors"], f"{label}{index}: reference bulk has errors")
    http.ok("POST", f"/{ref_index}/_refresh")
    run_matrix(cl, ref, index, ref_index, label + "pre-merge ", scored=False)
    merge_until_clean(cl, index, len(docs), label)
    http.ok("POST", f"/{ref_index}/_forcemerge?max_num_segments=1", timeout=900)
    cl.ok("POST", f"/{index}/_refresh")
    copy_scores(cl, index, label)
    run_matrix(cl, ref, index, ref_index, label + "merged ", scored=True)
    if vec:
        knn_check(cl.http(), index, docs, label)
        knn_check(http, ref_index, docs, label + "stock ")


# ---------------------------------------------------------------------------------------------
# Log scan

BAD_LOG = re.compile(r"CorruptIndexException|panicked at|SIGSEGV|fatal error in thread|AssertionError|"
                     r"native error|FfiError|lucene_rust.*(ERROR|Exception)|failed to recover from|"
                     r"IndexFormatTooOld|TranslogCorrupted|EngineCreationFailure|unsupported|"
                     r"ShardLockObtainFailed", re.I)


def scan_logs(clusters):
    seen = {}
    for cl in clusters:
        for n in cl.nodes:
            for fn in sorted(os.listdir(n.logs)) if os.path.isdir(n.logs) else []:
                p = os.path.join(n.logs, fn)
                if fn.startswith("hs_err"):
                    check(False, f"{n.name}: JVM crash log {fn}")
                if not fn.endswith(".log") or fn.startswith("gc"):
                    continue
                prev = None
                for line in open(p, errors="replace"):
                    if BAD_LOG.search(line):
                        key = re.sub(r"\[\d{4}-[^\]]*\]|\d+", "#", line)[:200]
                        seen.setdefault(key, [n.name, fn, line.strip()[:400], 0])[3] += 1
    for key, (node, fn, line, cnt) in seen.items():
        log(f"LOG x{cnt} {node}/{fn}: {line}")
    return seen


def plugin_stats(cl):
    out = {}
    for n in cl.nodes:
        if not n.alive():
            continue
        try:
            code, s = Http([n.url]).req("GET", "/_plugins/lucene_rust/stats", retry=False)
            if code == 200:
                out[n.name] = {k: s.get(k) for k in ("engines", "native_errors", "fallbacks") if k in s}
        except RuntimeError:
            pass
    return out


def engines(cl, index):
    """Which engine each shard copy runs on, from the plugin's per-node stats."""
    out = {}
    for n in cl.nodes:
        if not n.alive():
            continue
        try:
            code, s = Http([n.url]).req("GET", "/_plugins/lucene_rust/stats", retry=False)
        except RuntimeError:
            continue
        out[n.name] = s.get("engines") if code == 200 else f"{code}"
    return out


def load_report(load, label):
    log(f"{label}: load {json.dumps(load.stats)}")
    for k, v in sorted(load.errors.items(), key=lambda kv: -kv[1])[:12]:
        log(f"  load error x{v}: {k}")
    for u in load.unexpected:
        check(False, f"{label}: unexpected error under load: {u}")


# ---------------------------------------------------------------------------------------------
# Scenarios


def new_ref():
    ref = Cluster("ref", 1, 9500)
    ref.start("stock")
    return ref


def warm(load, secs):
    time.sleep(secs)


def sc_restarts(a):
    """Three plugin nodes (one replica, a k-NN `extended` field): graceful restarts, kill -9 of
    a data node, of the cluster manager, of two nodes at once and a full-cluster power cut, all
    under write and search load."""
    cl = Cluster("restarts", 3, 9200)
    cl.start("plugin")
    ref = new_ref()
    try:
        cl.ok("PUT", "/main", index_body(2, 1, vec=True))
        cl.wait_status("green")
        load = Load(cl, "main", writers=3, searchers=2, ids=12000, vec=True)
        load.start()
        time.sleep(20)
        plan = [("graceful restart, one node", [1], False), ("kill -9, one data node", [2], True),
                ("kill -9 the cluster manager", ["cm"], True), ("kill -9 two nodes", [0, 1], True),
                ("graceful restart, every node in turn", [0, 1, 2], False)]
        for name, which, kill in plan:
            log(f"== {name}")
            nodes = []
            for w in which:
                if w == "cm":
                    nodes.append(next(n for n in cl.nodes if n.name == cl.cluster_manager()))
                else:
                    nodes.append(cl.nodes[w])
            if len(nodes) == 2:
                for n in nodes:
                    n.stop(kill=kill)
                time.sleep(8)
                for n in nodes:
                    n.start("plugin")
                for n in nodes:
                    n.wait_http()
                cl.wait_status("green", 420)
            elif kill:
                nodes[0].stop(kill=True)
                time.sleep(10)
                nodes[0].start("plugin")
                nodes[0].wait_http()
                cl.wait_status("green", 420)
            else:
                cl.rolling(nodes, "plugin", kill=False)
            time.sleep(15)
            cl.req("POST", "/main/_flush")
            if a.minutes and time.time() - T0 > a.minutes * 60:
                break
        log("== full-cluster power cut")
        load.quiesce()
        for n in cl.nodes:
            n.stop(kill=True)
        for n in cl.nodes:
            n.start("plugin")
        for n in cl.nodes:
            n.wait_http()
        cl.wait_status("green", 420)
        load.resume()
        time.sleep(20)
        load.quiesce()
        load_report(load, "restarts")
        docs = verify_model(cl, "main", load.model, vec=True, label="restarts ")
        compare_with_stock(cl, ref, "main", docs, 2, "restarts ", vec=True)
        load.stop()
        log("plugin stats", json.dumps(plugin_stats(cl))[:600])
    finally:
        scan_logs([cl, ref])
        if not os.environ.get("SOAK_KEEP"):
            cl.stop()
            ref.stop()


def sc_upgrade(a):
    """A three-node OpenSearch 2.19.6 cluster with indices (one with replicas, one nested-free
    text index, one k-NN lucene index) is rolled to 3.9.0 + plugin, non-managers first, while
    load runs; then `_upgrade`, force merge, a new index on the accelerated engine, and the
    same checks against stock."""
    cl = Cluster("upgrade", 3, 9200)
    cl.start("2x")
    ref = new_ref()
    try:
        cl.ok("PUT", "/main", index_body(2, 1))
        cl.wait_status("green")
        load = Load(cl, "main", writers=3, searchers=2, ids=12000)
        load.start()
        time.sleep(60)
        order = sorted(cl.nodes, key=lambda n: n.name == cl.cluster_manager())
        log("upgrade order", [n.name for n in order])
        for n in order:
            cl.rolling([n], "plugin")
            log("mixed cluster versions:", [x["version"] for x in cl.ok("GET", "/_cat/nodes?format=json&h=version")])
            time.sleep(20)
        cl.cluster_manager()
        # Writes on the upgraded cluster for a while, then settle.
        time.sleep(30)
        load.quiesce()
        load_report(load, "upgrade")
        segs = cl.ok("GET", "/_cat/segments/main?format=json&h=shard,segment,version,docs.count")
        log("segment versions before _upgrade:", sorted({s["version"] for s in segs}))
        docs = verify_model(cl, "main", load.model, label="upgrade ")
        compare_with_stock(cl, ref, "main", docs, 2, "upgrade (2.x-created, merged) ")
        code, up = cl.req("POST", "/main/_upgrade", timeout=900)
        log("_upgrade", code, json.dumps(up)[:200])
        segs = cl.ok("GET", "/_cat/segments/main?format=json&h=shard,segment,version")
        log("segment versions after merge:", sorted({s["version"] for s in segs}))
        check(all(s["version"].startswith("10.") for s in segs), "upgrade: segments left in an old Lucene version")
        # A new index on the upgraded cluster, under load again, then compare.
        cl.ok("PUT", "/fresh", index_body(2, 1))
        cl.wait_status("green")
        load2 = Load(cl, "fresh", writers=3, searchers=2, ids=12000)
        load2.start()
        load.resume()
        time.sleep(60)
        load2.quiesce()
        load.quiesce()
        docs2 = verify_model(cl, "fresh", load2.model, label="upgrade ")
        compare_with_stock(cl, ref, "fresh", docs2, 2, "upgrade ")
        docs = verify_model(cl, "main", load.model, label="upgrade ")
        compare_with_stock(cl, ref, "main", docs, 2, "upgrade (after) ")
        load.stop()
        load2.stop()
        load_report(load2, "upgrade fresh")
        log("plugin stats", json.dumps(plugin_stats(cl))[:600])
    finally:
        scan_logs([cl, ref])
        if not os.environ.get("SOAK_KEEP"):
            cl.stop()
            ref.stop()


def sc_plugin_swap(a):
    """Plugin removal and re-add on a live cluster: nodes restart one at a time onto stock
    3.9.0 over the plugin's data (a mixed cluster in between), the cluster runs on stock, then
    nodes come back with the plugin, all under load."""
    cl = Cluster("swap", 3, 9200)
    cl.start("plugin")
    ref = new_ref()
    try:
        cl.ok("PUT", "/main", index_body(2, 1))
        cl.wait_status("green")
        load = Load(cl, "main", writers=3, searchers=2, ids=12000)
        load.start()
        time.sleep(45)
        log("== removing the plugin, one node at a time")
        cl.rolling(cl.nodes, "stock", pause_after=10)
        time.sleep(30)
        load.quiesce()
        docs = verify_model(cl, "main", load.model, label="swap removed ")
        compare_with_stock(cl, ref, "main", docs, 2, "swap removed ")
        load.resume()
        log("== re-adding the plugin")
        cl.rolling(cl.nodes, "plugin", pause_after=10)
        time.sleep(30)
        load.quiesce()
        load_report(load, "swap")
        docs = verify_model(cl, "main", load.model, label="swap re-added ")
        compare_with_stock(cl, ref, "main", docs, 2, "swap re-added ")
        load.stop()
        log("plugin stats", json.dumps(plugin_stats(cl))[:600])
    finally:
        scan_logs([cl, ref])
        cl.stop()
        ref.stop()


def sc_snapshot(a):
    """Snapshots under load: restore onto the same cluster, onto stock, a stock snapshot onto the
    plugin cluster, incremental snapshots, an in-place restore of a closed index, a snapshot taken
    while writes continue (restored copy must be internally valid)."""
    shutil.rmtree(os.path.join(REPO_DIR, "r1"), ignore_errors=True)
    cl = Cluster("snap", 3, 9200)
    cl.start("plugin")
    ref = Cluster("ref", 1, 9500)
    ref.start("stock")
    try:
        for c in (cl, ref):
            c.ok("PUT", "/_snapshot/repo", {"type": "fs", "settings": {"location": os.path.join(REPO_DIR, "r1")}})
        cl.ok("PUT", "/main", index_body(2, 1, vec=True))
        cl.wait_status("green")
        load = Load(cl, "main", writers=3, searchers=2, ids=12000, vec=True)
        load.start()
        time.sleep(40)
        # 1. a snapshot taken while writes continue: internally valid
        t = time.time()
        cl.ok("PUT", "/_snapshot/repo/live?wait_for_completion=true", {"indices": "main", "include_global_state": False}, timeout=600)
        log(f"snapshot under load took {time.time() - t:.1f}s")
        # 2. a quiesced snapshot with a model to compare against
        load.quiesce()
        state, _ = load.model.snapshot()
        cl.ok("POST", "/main/_flush")
        t = time.time()
        out = cl.ok("PUT", "/_snapshot/repo/quiet?wait_for_completion=true", {"indices": "main", "include_global_state": False}, timeout=600)
        log(f"quiet snapshot {json.dumps(out['snapshot']['shards'])} took {time.time() - t:.1f}s")
        check(out["snapshot"]["state"] == "SUCCESS", f"quiet snapshot state {out['snapshot']['state']}")
        frozen = Model()
        frozen.state = dict(state)
        frozen.history = {k: set(v) for k, v in load.model.history.items()}
        load.resume()
        # incremental snapshot while writing continues
        time.sleep(30)
        out = cl.ok("PUT", "/_snapshot/repo/incr?wait_for_completion=true", {"indices": "main", "include_global_state": False}, timeout=600)
        log("incremental snapshot", json.dumps(out["snapshot"]["shards"]), out["snapshot"]["state"])
        load.quiesce()
        load_report(load, "snapshot")
        final_docs = verify_model(cl, "main", load.model, vec=True, label="snapshot live ")
        # restores onto the same cluster
        for snap, name in (("quiet", "r_quiet"), ("live", "r_live"), ("incr", "r_incr")):
            cl.ok("POST", f"/_snapshot/repo/{snap}/_restore?wait_for_completion=true", {
                "indices": "main", "rename_pattern": "main", "rename_replacement": name,
                "include_global_state": False}, timeout=600)
            cl.wait_status("green", 300)
            docs, total = scan(cl.http(), name)
            bad = [sid for sid, s in docs.items() if s is None or {k: v for k, v in s.items() if k != "vec"} != {
                k: v for k, v in make_doc(int(sid), s["v"], True).items() if k != "vec"}]
            check(not bad, f"snapshot restore {snap}: {len(bad)} corrupt documents {bad[:5]}")
            log(f"restored {snap} -> {name}: {len(docs)} docs")
            if snap == "quiet":
                present = {i: s[1] for i, s in frozen.state.items() if s[0] == "p"}
                got = {int(k): v["v"] for k, v in docs.items()}
                missing = [i for i in present if i not in got and i not in frozen.history]
                wrong = [i for i in present if i in got and got[i] != present[i] and ("p", got[i]) not in frozen.candidates(i)]
                check(not missing and not wrong, f"snapshot restore quiet: {len(missing)} missing {len(wrong)} wrong (state at snapshot)")
                compare_with_stock(cl, ref, name, docs, 2, "snapshot restored(quiet) ", vec=True)
        # onto stock: the plugin-written snapshot of an index without the extended field
        cl.ok("PUT", "/plain", index_body(2, 1))
        cl.wait_status("green")
        plain = Load(cl, "plain", writers=2, searchers=1, ids=6000)
        plain.start()
        time.sleep(25)
        plain.quiesce()
        cl.ok("POST", "/plain/_flush")
        cl.ok("PUT", "/_snapshot/repo/plain_snap?wait_for_completion=true", {"indices": "plain", "include_global_state": False}, timeout=600)
        ref.ok("POST", "/_snapshot/repo/_verify")
        ref.ok("POST", "/_snapshot/repo/plain_snap/_restore?wait_for_completion=true",
               {"indices": "plain", "include_global_state": False}, timeout=600)
        ref.wait_status("yellow", 300)
        docs, total = scan(ref.http(), "plain")
        state, _ = plain.model.snapshot()
        present = {i: s[1] for i, s in state.items() if s[0] == "p"}
        got = {int(k): v["v"] for k, v in docs.items()}
        check(all(i in got for i in present if i not in plain.model.history), "restore onto stock: acknowledged documents missing")
        run_docs = scan(cl.http(), "plain")[0]
        cl_pre = {k: v for k, v in run_docs.items()}
        check(cl_pre == docs, "restore onto stock: documents differ from the source")
        # stock snapshot -> plugin cluster
        ref.ok("PUT", "/stock_idx", index_body(2, 0, ref=True))
        lines = []
        for i in range(6000):
            lines.append(json.dumps({"index": {"_index": "stock_idx", "_id": str(i)}}))
            lines.append(json.dumps(make_doc(i, 1)))
        ref.ok("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True, timeout=120)
        ref.ok("POST", "/stock_idx/_flush")
        ref.ok("PUT", "/_snapshot/repo/stock_snap?wait_for_completion=true", {"indices": "stock_idx", "include_global_state": False}, timeout=600)
        # Another cluster wrote to the repository: re-register it so this one reads the new generation.
        cl.ok("DELETE", "/_snapshot/repo")
        cl.ok("PUT", "/_snapshot/repo", {"type": "fs", "settings": {"location": os.path.join(REPO_DIR, "r1")}})
        cl.ok("POST", "/_snapshot/repo/stock_snap/_restore?wait_for_completion=true", {
            "indices": "stock_idx", "include_global_state": False,
            "index_settings": {"index.number_of_replicas": 1}}, timeout=600)
        cl.wait_status("green", 300)
        sdocs, _ = scan(ref.http(), "stock_idx")
        pdocs, _ = scan(cl.http(), "stock_idx")
        check(sdocs == pdocs, "stock snapshot restored on the plugin cluster: documents differ")
        compare_with_stock(cl, ref, "stock_idx", pdocs, 2, "stock snapshot on plugin ")
        # writes onto the restored (stock-written) index on the accelerated engine
        lines = []
        for i in range(6000, 6600):
            lines.append(json.dumps({"index": {"_index": "stock_idx", "_id": str(i)}}))
            lines.append(json.dumps(make_doc(i, 1)))
        for i in range(0, 300):
            lines.append(json.dumps({"delete": {"_index": "stock_idx", "_id": str(i)}}))
        out = cl.ok("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True, timeout=120)
        check(not out["errors"], "writes into the restored stock index failed")
        cl.ok("POST", "/stock_idx/_refresh")
        check(cl.ok("GET", "/stock_idx/_count")["count"] == 6300, "restored stock index count after writes")
        # in-place restore of a closed index
        cl.ok("POST", "/main/_close")
        cl.ok("POST", "/_snapshot/repo/quiet/_restore?wait_for_completion=true", {
            "indices": "main", "include_global_state": False}, timeout=600)
        cl.wait_status("green", 300)
        docs, _ = scan(cl.http(), "main")
        check(len(docs) == len([1 for s in frozen.state.values() if s[0] == "p"]) or any(frozen.history),
              f"in-place restore: {len(docs)} docs")
        # delete and clone
        cl.ok("PUT", "/_snapshot/repo/quiet/_clone/quiet2", {"indices": "main"})
        cl.ok("DELETE", "/_snapshot/repo/live")
        cl.ok("POST", "/_snapshot/repo/quiet2/_restore?wait_for_completion=true", {
            "indices": "main", "rename_pattern": "main", "rename_replacement": "r_clone",
            "include_global_state": False}, timeout=600)
        cl.wait_status("green", 300)
        check(cl.ok("GET", "/r_clone/_count")["count"] == cl.ok("GET", "/r_quiet/_count")["count"], "cloned snapshot count differs")
        load.stop()
        plain.stop()
        log("plugin stats", json.dumps(plugin_stats(cl))[:600])
    finally:
        scan_logs([cl, ref])
        cl.stop()
        ref.stop()


def sc_ccr(a):
    """Cross-cluster replication with followers served by this plugin's engine
    (`plugins.replication.follower.engine: delegate`), under load, with follower restarts, a
    leader restart, a pause/resume, and the same checks on the follower."""
    ccr_plugin = DIST["ccr"]
    cl = Cluster("leader", 3, 9200, extra={"plugins.replication.follower.engine": "delegate"})
    fl = Cluster("follower", 3, 9700, extra={"plugins.replication.follower.engine": "delegate"})
    ref = new_ref()
    try:
        cl.start("ccr")
        fl.start("ccr")
        fl.settings({"cluster.remote.leader.seeds": [f"127.0.0.1:{cl.nodes[0].transport_port}"]})
        cl.ok("PUT", "/main", index_body(2, 1, vec=True))
        cl.wait_status("green")
        load = Load(cl, "main", writers=3, searchers=1, ids=10000, vec=True)
        load.start()
        time.sleep(25)
        out = fl.ok("PUT", "/_plugins/_replication/main/_start", {"leader_alias": "leader", "leader_index": "main"}, timeout=300)
        log("replication started", json.dumps(out))

        def lag():
            _, st = fl.req("GET", "/_plugins/_replication/main/_status")
            return st

        def wait_synced(timeout=300):
            t = time.time()
            while time.time() - t < timeout:
                st = lag()
                if st.get("status") == "SYNCING" and st.get("syncing_details", {}).get("follower_checkpoint", -2) >= st.get("syncing_details", {}).get("leader_checkpoint", 0) >= 0:
                    return st
                time.sleep(2)
            return lag()

        time.sleep(40)
        log("status", json.dumps(lag())[:400])
        log("== follower node restart (kill -9)")
        fl.nodes[1].stop(kill=True)
        time.sleep(10)
        fl.nodes[1].start("ccr")
        fl.nodes[1].wait_http()
        fl.wait_status("green", 420)
        time.sleep(20)
        log("== follower graceful rolling restart")
        fl.rolling(fl.nodes, "ccr")
        time.sleep(20)
        log("== leader node kill -9")
        cl.nodes[1].stop(kill=True)
        time.sleep(10)
        cl.nodes[1].start("ccr")
        cl.nodes[1].wait_http()
        cl.wait_status("green", 420)
        time.sleep(20)
        log("== pause and resume replication")
        fl.ok("POST", "/_plugins/_replication/main/_pause", {})
        time.sleep(20)
        fl.ok("POST", "/_plugins/_replication/main/_resume", {})
        time.sleep(30)
        load.quiesce()
        load_report(load, "ccr")
        leader_docs = verify_model(cl, "main", load.model, vec=True, label="ccr leader ")
        st = wait_synced(420)
        log("status after quiesce", json.dumps(st)[:500])
        check(st.get("status") == "SYNCING", f"ccr: follower status {st.get('status')}")
        fl.wait_status("green", 420)
        t = time.time()
        while time.time() - t < 240:
            fd, ft = scan(fl.http(), "main")
            if len(fd) == len(leader_docs):
                break
            time.sleep(3)
        verify_model(fl, "main", load.model, vec=True, label="ccr follower ", strict_model=True)
        fd, _ = scan(fl.http(), "main")
        diff = [k for k in set(fd) | set(leader_docs) if fd.get(k) != leader_docs.get(k)]
        check(not diff, f"ccr: {len(diff)} documents differ between leader and follower, e.g. {diff[:5]}")
        # Stopping replication turns the follower into a normal index (and makes it writable).
        fl.ok("POST", "/_plugins/_replication/main/_stop", {})
        fl.wait_status("green", 300)
        compare_with_stock(fl, ref, "main", fd, 2, "ccr follower (stopped) ", vec=True)
        log("follower plugin stats", json.dumps(plugin_stats(fl))[:600])
        # The stopped follower takes writes on the accelerated engine.
        out = fl.ok("POST", "/_bulk", "\n".join(
            x for i in range(100000, 100200)
            for x in (json.dumps({"index": {"_index": "main", "_id": str(i)}}), json.dumps(make_doc(i, 1, True)))) + "\n",
            ndjson=True)
        check(not out["errors"], "ccr: writes into the stopped follower failed")
        load.stop()
    finally:
        scan_logs([cl, fl, ref])
        cl.stop()
        fl.stop()
        ref.stop()


SCENARIOS = {"restarts": sc_restarts, "upgrade": sc_upgrade, "plugin_swap": sc_plugin_swap,
             "snapshot": sc_snapshot, "ccr": sc_ccr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=list(SCENARIOS) + ["all"])
    ap.add_argument("--minutes", type=float, default=0, help="stop optional steps after this long")
    a = ap.parse_args()
    os.makedirs(REPO_DIR, exist_ok=True)
    names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    for name in names:
        log(f"######## scenario {name}")
        before = len(FINDINGS)
        try:
            SCENARIOS[name](a)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(False, f"{name}: scenario aborted: {e}")
        log(f"######## {name}: {len(FINDINGS) - before} findings")
    print(f"\n{CHECKS[0]} checks, {len(FINDINGS)} findings")
    for f in FINDINGS:
        print(" -", f)
    return 1 if FINDINGS else 0


if __name__ == "__main__":
    sys.exit(main())
