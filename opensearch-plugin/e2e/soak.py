"""M6's soak (T6.1): a three-node cluster on the Rust engine under mixed index
and search load for days, with random restarts and SIGKILLs, instrumented
every minute.

    soak.py run DIR [--hours H] [--resume]   the soak, against the nodes
                                             scripts/soak-opensearch.sh starts
    soak.py report DIR                       the verdict from DIR's traces

What it holds the cluster to, and where each check lives:

- Acknowledged writes survive. Every write carries `sv`, a version from one
  counter; the model keeps each id's last acknowledged version (or its
  deletion). A bulk item without a clear answer -- a node dying under it --
  makes its id uncertain until a later write to it is acknowledged. Checks:
  sampled realtime GETs every few minutes, and every hour a quiesce (writes
  paused, refresh) comparing the exact count and every id's version with the
  model.
- The native search path answers as Lucene does: at each quiesce a set of
  query shapes runs with the native path on and off, responses compared.
- Nothing grows without bound: RSS, open file descriptors, mapped regions,
  threads, heap after GC, data on disk and segment counts, per node, every
  minute (`metrics.jsonl`); search and indexing latency per minute
  (`latency.jsonl`).
- No unexplained shard failures: node logs are scanned for shard failures and
  panics (`incidents.jsonl`), each classified against the chaos that was
  running at the time.

The environment this runs in can be recycled while idle, which kills every
process at once. `--resume` treats that as what it is -- a power loss of the
whole cluster: it checks the saved model against what the cluster kept, then
rebuilds the model from the index and carries on. The model is saved once a
minute; every bulk since is in `journal.jsonl` (its items before it is sent,
their outcomes after, fsynced), which `--resume` replays onto the saved model
first. Without it a delete made after the last save reads as a lost document,
and an acknowledged write lost after it is not seen at all. Only time under
load counts toward the soak's hours.
"""
import http.client
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_opensearch as v  # noqa: E402

# A response can carry one `Warning` header per deprecation it met, past
# http.client's default cap of 100 headers.
http.client._MAXHEADERS = 100_000

NODES = [("os1", 9201), ("os2", 9202), ("os3", 9203)]
INDEX = "soak"
# Seconds between chaos events (uniform in [x, 3x]) and between quiesces; the
# environment variables shorten both for a smoke run of the harness itself.
CHAOS_MIN_S = float(os.environ.get("SOAK_CHAOS_MIN_S", 30 * 60))
QUIESCE_S = float(os.environ.get("SOAK_QUIESCE_S", 3600))
IDS = 250_000  # the id space writes cycle over: bounded data, constant merging
BASES = [f"http://localhost:{p}" for _, p in NODES]


def now():
    return time.time()


class Soak:
    def __init__(self, d, hours):
        self.dir = d
        self.target = hours * 3600
        self.lock = threading.Lock()
        self.model = {}  # id -> (sv, deleted)
        self.uncertain = {}  # id -> set of (sv, deleted) it may hold
        self.counter = 0
        self.paused = threading.Event()  # set while writes are paused
        self.writing = threading.Lock()  # held by the writer for a bulk
        self.stop = threading.Event()
        self.chaos_quiet = threading.Lock()  # chaos and quiesce exclude each other
        self.lat = {"search": [], "bulk": [], "search_err": 0, "bulk_err": 0, "bulk_items": 0}
        self.lat_lock = threading.Lock()
        self.chaos_log = []  # (start, end, node, action)
        self.state = {"active_s": 0.0, "started": now(), "resumes": 0, "quiesces": 0,
                      "quiesce_failures": 0, "get_checks": 0, "get_failures": 0,
                      "chaos": 0, "native_compared": 0, "native_mismatch": 0,
                      "counter": 0, "lost": 0, "target_h": hours}
        self.rng = random.Random()
        self.base_i = 0

    # -- plumbing ---------------------------------------------------------

    def path(self, name):
        return os.path.join(self.dir, name)

    def append(self, name, rec):
        with open(self.path(name), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def req(self, method, p, body=None, ndjson=False, timeout=60, base=None):
        """A request to a live node, trying each in turn."""
        order = [base] if base else [BASES[(self.base_i + k) % 3] for k in range(3)]
        last = None
        for b in order:
            data = None
            headers = {}
            if body is not None:
                data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
                headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
            r = urllib.request.Request(b + p, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(r, timeout=timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code in (502, 503):
                    last = e
                    continue
                raise RuntimeError(f"{method} {p}: {e.code} {e.read().decode()[:300]}")
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                last = e
                self.base_i += 1
                continue
        raise RuntimeError(f"{method} {p}: no node answered: {last}")

    def green(self, timeout=900):
        end = now() + timeout
        while now() < end:
            try:
                h = self.req("GET", "/_cluster/health?wait_for_status=green&timeout=30s", timeout=40)
                if h.get("status") == "green" and h.get("number_of_nodes") == 3:
                    return True
            except RuntimeError:
                time.sleep(5)
        return False

    def journal(self, rec):
        """One line of the write-ahead journal, on disk before this returns:
        the model is saved once a minute, and an environment restart must not
        lose what was written since."""
        f = self.jfile
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())

    def save(self):
        # No bulk in flight: the snapshot and the empty journal agree.
        with self.writing:
            self._save()

    def _save(self):
        with self.lock:
            snap = {"model": {str(k): list(v2) for k, v2 in self.model.items()},
                    "uncertain": {str(k): [list(x) for x in s] for k, s in self.uncertain.items()},
                    "state": dict(self.state, counter=self.counter)}
        tmp = self.path("model.json.tmp")
        with open(tmp, "w") as f:
            json.dump(snap, f)
        os.replace(tmp, self.path("model.json"))
        self.jfile.truncate(0)
        self.jfile.seek(0)
        os.fsync(self.jfile.fileno())

    # -- the index --------------------------------------------------------

    def create(self):
        v.BASE = BASES[0]
        v.create(INDEX, 3)
        self.req("PUT", f"/{INDEX}/_settings", {"index": {
            "number_of_replicas": 1, "refresh_interval": "1s"}})
        self.req("PUT", f"/{INDEX}/_mapping", {"properties": {"sv": {"type": "long"}}})

    def doc(self, i, sv, r):
        d = {
            "body": " ".join(v.word(r) for _ in range(r.randint(1, 40))),
            "title": " ".join(v.word(r) for _ in range(r.randint(1, 5))),
            "tuned": " ".join(v.word(r) for _ in range(r.randint(1, 10))),
            "tag": v.word(r),
            "mtag": [v.word(r) + str(r.randint(0, 9)) for _ in range(r.randint(0, 3))],
            "n": i,
            "price": round(r.uniform(-50, 500), 2),
            "qty": r.randint(0, 40),
            "ts": 1_700_000_000_000 + r.randint(0, 10_000) * 60_000,
            "@timestamp": 1_700_000_000_000 + r.randint(0, 500) * 60_000,
            "ratio": r.random(),
            "m": [r.randint(0, 1000) for _ in range(r.randint(0, 3))],
            "md": [round(r.uniform(-50, 50), 3) for _ in range(r.randint(0, 4))],
            "mi": [r.randint(-100, 100) for _ in range(r.randint(0, 4))],
            "sv": sv,
        }
        if r.random() < 0.7:
            d["sp"] = r.randint(-100, 100)
        return d

    # -- load ---------------------------------------------------------------

    def writer(self):
        r = random.Random(7)
        while not self.stop.is_set():
            with self.writing:
                # Checked under the lock: a quiesce sets `paused` and then takes
                # this lock once, so no bulk may start after that. Checked
                # before it, a bulk slipped into a quiesce's scan (the smoke
                # run's last bulk: acknowledged, not scanned).
                wrote = not self.paused.is_set() and self.bulk(r)
            time.sleep(0.4 if wrote else 0.2)

    def bulk(self, r):
        """One bulk of 100 writes (caller holds `writing`)."""
        ops = []
        lines = []
        for _ in range(100):
            i = r.randrange(IDS)
            with self.lock:
                self.counter += 1
                sv = self.counter
            delete = r.random() < 0.08
            if delete:
                lines.append(json.dumps({"delete": {"_index": INDEX, "_id": str(i), "version": sv, "version_type": "external_gte"}}))
            else:
                lines.append(json.dumps({"index": {"_index": INDEX, "_id": str(i), "version": sv, "version_type": "external_gte"}}))
                lines.append(json.dumps(self.doc(i, sv, r)))
            ops.append((i, sv, delete))
        self.journal({"ops": ops})
        t0 = now()
        try:
            res = self.req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True, timeout=60)
            items = res["items"]
            ok = True
        except RuntimeError:
            items = [None] * len(ops)
            ok = False
        dt = now() - t0
        outcome = []
        with self.lock:
            for (i, sv, delete), it in zip(ops, items):
                st = None
                if it is not None:
                    body = it.get("delete") or it.get("index")
                    st = body.get("status")
                # 404 on a delete of an absent document is an
                # acknowledged delete too; 409 means a later version
                # is already there (the id's own history moved on).
                if st in (200, 201) or (delete and st == 404):
                    outcome.append("a")
                elif st == 409:
                    outcome.append("-")
                else:
                    outcome.append("?")
            self.apply(ops, outcome)
        self.journal({"outcome": "".join(outcome)})
        with self.lat_lock:
            if ok:
                self.lat["bulk"].append(dt)
                self.lat["bulk_items"] += len(ops)
            else:
                self.lat["bulk_err"] += 1
        return True

    def apply(self, ops, outcome):
        """A bulk's resolved items into the model (caller holds the lock):
        `a` acknowledged, `-` superseded by a later version (409), `?` unknown."""
        for (i, sv, delete), o in zip(ops, outcome):
            if o == "a":
                self.model[i] = (sv, delete)
                self.uncertain.pop(i, None)
            elif o == "?":
                s = self.uncertain.setdefault(i, set())
                if i in self.model:
                    s.add(self.model[i])
                s.add((sv, delete))

    def replay(self):
        """The journal since the last save, applied as the writer applied it;
        a bulk with no outcome was in flight when everything stopped."""
        n = 0
        pending = None
        with open(self.path("journal.jsonl")) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    break  # a torn last line
                if "ops" in rec:
                    if pending is not None:
                        self.apply(pending, "?" * len(pending))
                    pending = [tuple(x) for x in rec["ops"]]
                    n += 1
                elif pending is not None:
                    self.apply(pending, rec["outcome"])
                    pending = None
        if pending is not None:
            self.apply(pending, "?" * len(pending))
        return n

    def searcher(self, seed):
        r = random.Random(seed)
        shapes = [(n, b) for n, b, e in v.matrix() if e == "native"]
        while not self.stop.is_set():
            name, body = r.choice(shapes)
            b = dict(body)
            t0 = now()
            try:
                self.req("POST", f"/{INDEX}/_search?request_cache=false", b, timeout=60)
                dt = now() - t0
                with self.lat_lock:
                    self.lat["search"].append(dt)
            except RuntimeError:
                with self.lat_lock:
                    self.lat["search_err"] += 1
            time.sleep(0.05)

    # -- checks -------------------------------------------------------------

    def acceptable(self, i, found):
        """Whether `found` -- (sv, deleted) as the cluster holds id `i` -- is
        one the model allows: at least the last acknowledged write, or one of
        the writes whose outcome is not known."""
        with self.lock:
            acked = self.model.get(i)
            unsure = set(self.uncertain.get(i, ()))
        if found[1]:
            # Absent: an acknowledged or possible delete (its version is not
            # visible), or an id never acknowledged at all.
            return acked is None or acked[1] or any(d for _, d in unsure)
        if found in unsure or found == acked:
            return True
        if acked is None:
            return False
        # A later write in flight right now; the check runs alongside the writer.
        return found[0] > acked[0]

    def get_checks(self, n=200):
        fails = []
        r = self.rng
        with self.lock:
            ids = list(self.model.keys())
        for i in r.sample(ids, min(n, len(ids))):
            try:
                g = self.req("GET", f"/{INDEX}/_doc/{i}", timeout=30)
                found = (g["_source"]["sv"], False)
            except RuntimeError as e:
                if " 404 " in str(e) or ": 404" in str(e):
                    found = (0, True)
                else:
                    continue
            self.state["get_checks"] += 1
            if not self.acceptable(i, found):
                fails.append({"id": i, "found": found, "model": self.model.get(i),
                              "unsure": sorted(self.uncertain.get(i, ()))})
        if fails:
            self.state["get_failures"] += len(fails)
            self.append("incidents.jsonl", {"t": now(), "kind": "get_mismatch", "detail": fails[:20]})

    def scan(self):
        """Every live document's (id, sv), by a sliced-free scroll."""
        out = {}
        res = self.req("POST", f"/{INDEX}/_search?scroll=5m", {
            "size": 5000, "_source": ["sv"], "sort": ["_doc"]}, timeout=300)
        while True:
            hits = res["hits"]["hits"]
            if not hits:
                break
            for h in hits:
                out[int(h["_id"])] = h["_source"]["sv"]
            res = self.req("POST", "/_search/scroll", {"scroll": "5m", "scroll_id": res["_scroll_id"]}, timeout=300)
        return out

    def compare_native(self, rounds=1):
        shapes = [(n, b) for n, b, e in v.matrix() if e == "native"]
        r = self.rng
        v.BASE = BASES[self.base_i % 3]
        picked = r.sample(shapes, min(25, len(shapes)))
        mism = []
        for name, body in picked:
            try:
                url, b = v.search_url(INDEX, body)
                # Both requests on the same copies: a replica keeps its own
                # deleted-document counts, so its own BM25 statistics, and its
                # own doc ids for tie order.
                url += ("&" if "?" in url else "?") + "preference=soak-compare"
                v.set_native(INDEX, False)
                ref = v.shape(v.req("POST", url, b), b)
                v.set_native(INDEX, True)
                got = v.shape(v.req("POST", url, b), b)
                diff = v.same(got, ref)
                self.state["native_compared"] += 1
                if diff is not None:
                    mism.append({"shape": name, "diff": str(diff)[:300]})
            except Exception as e:  # a node restarting under the comparison
                self.append("incidents.jsonl", {"t": now(), "kind": "compare_error", "shape": name, "err": str(e)[:300]})
        v.set_native(INDEX, True)
        if mism:
            self.state["native_mismatch"] += len(mism)
            self.append("incidents.jsonl", {"t": now(), "kind": "native_mismatch", "detail": mism})

    def quiesce(self):
        with self.chaos_quiet:
            self.paused.set()
            with self.writing:
                pass
            try:
                if not self.green():
                    self.append("incidents.jsonl", {"t": now(), "kind": "not_green_at_quiesce"})
                    return
                self.req("POST", f"/{INDEX}/_refresh", timeout=300)
                live = self.scan()
                cnt = self.req("GET", f"/{INDEX}/_count", timeout=120)["count"]
                bad = []
                with self.lock:
                    ids = set(self.model) | set(self.uncertain) | set(live)
                for i in ids:
                    found = (live[i], False) if i in live else (0, True)
                    if not self.acceptable(i, found):
                        bad.append({"id": i, "found": found, "model": self.model.get(i),
                                    "unsure": sorted(self.uncertain.get(i, ()))})
                self.state["quiesces"] += 1
                if bad or cnt != len(live):
                    self.state["quiesce_failures"] += 1
                    self.append("incidents.jsonl", {"t": now(), "kind": "quiesce_mismatch",
                                                    "count": cnt, "scanned": len(live), "bad": bad[:50], "n_bad": len(bad)})
                # What the cluster holds is now known exactly: it is the model.
                with self.lock:
                    self.model = {i: (s, False) for i, s in live.items()}
                    for i in ids - set(live):
                        self.model[i] = (0, True)
                    self.uncertain.clear()
                self.compare_native()
                self.append("events.jsonl", {"t": now(), "kind": "quiesce", "count": cnt, "bad": len(bad)})
            finally:
                self.paused.clear()
        self.save()

    # -- chaos ----------------------------------------------------------------

    def chaos(self):
        r = random.Random()
        while not self.stop.is_set():
            wait = r.uniform(CHAOS_MIN_S, 3 * CHAOS_MIN_S)
            end = now() + wait
            while now() < end and not self.stop.is_set():
                time.sleep(5)
            if self.stop.is_set():
                return
            with self.chaos_quiet:
                name, _ = r.choice(NODES)
                action = r.choice(["restart", "kill"])
                t0 = now()
                if action == "restart":
                    subprocess.run(["docker", "restart", "-t", "60", name], capture_output=True)
                else:
                    subprocess.run(["docker", "kill", "-s", "KILL", name], capture_output=True)
                    time.sleep(3)
                    subprocess.run(["docker", "start", name], capture_output=True)
                ok = self.green(timeout=1800)
                t1 = now()
                self.chaos_log.append((t0, t1, name, action))
                self.state["chaos"] += 1
                self.append("events.jsonl", {"t": t0, "kind": "chaos", "node": name, "action": action,
                                             "green_after_s": round(t1 - t0, 1), "green": ok})
                if not ok:
                    self.append("incidents.jsonl", {"t": t1, "kind": "not_green_after_chaos", "node": name, "action": action})

    # -- instrumentation ------------------------------------------------------

    def proc(self, name):
        pid = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", name],
                             capture_output=True, text=True).stdout.strip()
        if not pid or pid == "0":
            return None
        # The java process: the container's init after `exec`, or its child.
        cands = [pid]
        try:
            with open(f"/proc/{pid}/task/{pid}/children") as f:
                cands += f.read().split()
        except OSError:
            pass
        for p in cands:
            try:
                with open(f"/proc/{p}/comm") as f:
                    if f.read().strip() == "java":
                        return p
            except OSError:
                pass
        return None

    def node_metrics(self, name):
        p = self.proc(name)
        m = {"node": name, "up": p is not None}
        if p is None:
            return m
        try:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith(("VmRSS", "Threads", "VmSwap")):
                        k, val = line.split(":")
                        m[k] = int(val.split()[0])
            m["fds"] = len(os.listdir(f"/proc/{p}/fd"))
            with open(f"/proc/{p}/maps") as f:
                m["maps"] = sum(1 for _ in f)
        except OSError:
            pass
        data = self.path(f"data/{name}")
        du = subprocess.run(["du", "-sb", data], capture_output=True, text=True).stdout.split()
        m["data_bytes"] = int(du[0]) if du else None
        return m

    def metrics(self):
        rec = {"t": now(), "active_s": round(self.state["active_s"]),
               "load": float(open("/proc/loadavg").read().split()[0]),
               "nodes": [self.node_metrics(n) for n, _ in NODES]}
        try:
            ns = self.req("GET", "/_nodes/stats/jvm,indices,process", timeout=30)
            for n in ns["nodes"].values():
                for nm in rec["nodes"]:
                    if nm["node"] == n["name"]:
                        nm["heap_used"] = n["jvm"]["mem"]["heap_used_in_bytes"]
                        nm["gc_old"] = n["jvm"]["gc"]["collectors"]["old"]["collection_count"]
                        nm["segments"] = n["indices"]["segments"]["count"]
                        nm["merges_current"] = n["indices"]["merges"]["current"]
                        nm["merges_total"] = n["indices"]["merges"]["total"]
                        nm["open_fds_os"] = n["process"]["open_file_descriptors"]
            st = self.req("GET", f"/{INDEX}/_stats/docs,store", timeout=30)["_all"]["total"]
            rec["docs"] = st["docs"]["count"]
            rec["deleted"] = st["docs"]["deleted"]
            rec["store"] = st["store"]["size_in_bytes"]
            ps = [self.req("GET", "/_plugins/lucene_rust/stats", timeout=30, base=b) for b in BASES]
            rec["native_queries"] = sum(p["native_queries"] for p in ps)
            rec["native_errors"] = sum(p["native_errors"] for p in ps)
            rec["engines"] = [p["engines"] for p in ps]
        except (RuntimeError, KeyError) as e:
            rec["err"] = str(e)[:200]
        self.append("metrics.jsonl", rec)
        with self.lat_lock:
            lat, self.lat = self.lat, {"search": [], "bulk": [], "search_err": 0, "bulk_err": 0, "bulk_items": 0}

        def pct(xs, q):
            return round(sorted(xs)[min(len(xs) - 1, int(q * len(xs)))] * 1000, 2) if xs else None
        self.append("latency.jsonl", {
            "t": now(), "active_s": round(self.state["active_s"]),
            "search_n": len(lat["search"]), "search_p50": pct(lat["search"], 0.5),
            "search_p90": pct(lat["search"], 0.9), "search_p99": pct(lat["search"], 0.99),
            "search_err": lat["search_err"],
            "bulk_n": len(lat["bulk"]), "bulk_items": lat["bulk_items"], "bulk_p50": pct(lat["bulk"], 0.5),
            "bulk_p99": pct(lat["bulk"], 0.99), "bulk_err": lat["bulk_err"],
            "chaos_active": any(a <= now() <= b + 60 for a, b, _, _ in self.chaos_log[-3:]),
        })

    def scan_logs(self, since):
        for name, _ in NODES:
            out = subprocess.run(["docker", "logs", "--since", str(int(since)), name],
                                 capture_output=True, text=True)
            text = out.stdout + out.stderr
            for line in text.splitlines():
                low = line.lower()
                if ("shard-failed" in low or "failing shard" in low or "shard failed" in low
                        or "panicked" in low or "rust panic" in low or "fatal error" in low
                        or "java.lang.outofmemoryerror" in low) and "jvm arguments" not in low:
                    self.append("incidents.jsonl", {"t": now(), "kind": "log", "node": name, "line": line[:500]})

    # -- the run ------------------------------------------------------------

    def resume(self):
        """After the whole environment went down: every acknowledged write in
        the last saved model must still be there."""
        try:
            with open(self.path("model.json")) as f:
                snap = json.load(f)
        except OSError:
            return
        self.state.update(snap["state"])
        self.counter = snap["state"].get("counter", 0) + 10_000_000  # past anything in flight
        self.state["resumes"] += 1
        self.model = {int(k): tuple(x) for k, x in snap["model"].items()}
        self.uncertain = {int(k): {tuple(x) for x in s} for k, s in snap["uncertain"].items()}
        replayed = self.replay()
        if not self.green():
            self.append("incidents.jsonl", {"t": now(), "kind": "not_green_after_resume"})
        self.req("POST", f"/{INDEX}/_refresh", timeout=300)
        live = self.scan()
        lost = []
        for i, (sv, deleted) in self.model.items():
            found = (live[i], False) if i in live else (0, True)
            if not self.acceptable(i, found):
                lost.append({"id": i, "found": found, "model": [sv, deleted]})
        self.state["lost"] += len(lost)
        self.append("events.jsonl", {"t": now(), "kind": "resume", "checked": len(self.model),
                                     "journaled_bulks": replayed, "lost": len(lost)})
        if lost:
            self.append("incidents.jsonl", {"t": now(), "kind": "lost_after_environment_restart", "n": len(lost), "detail": lost[:50]})
        self.model = {i: (s, False) for i, s in live.items()}
        self.uncertain = {}
        self._save()  # the rebuilt model, and an empty journal to go with it

    def run(self, resume):
        os.makedirs(self.dir, exist_ok=True)
        # Opened for appending, not truncated: `resume` replays it first.
        self.jfile = open(self.path("journal.jsonl"), "a+")
        v.BASE = BASES[0]
        if resume:
            self.resume()
        else:
            self.create()
            self.append("events.jsonl", {"t": now(), "kind": "start", "target_s": self.target})
        threads = [threading.Thread(target=self.writer, daemon=True),
                   threading.Thread(target=self.searcher, args=(1,), daemon=True),
                   threading.Thread(target=self.searcher, args=(2,), daemon=True),
                   threading.Thread(target=self.chaos, daemon=True)]
        for t in threads:
            t.start()
        last = now()
        last_log = now()
        next_get = now() + 300
        next_quiesce = now() + QUIESCE_S
        while self.state["active_s"] < self.target:
            time.sleep(60)
            t = now()
            # A gap longer than a few minutes is the environment having been
            # suspended, not time under load.
            self.state["active_s"] += min(t - last, 180)
            last = t
            try:
                self.metrics()
                self.scan_logs(last_log)
                last_log = t
                if t >= next_get:
                    self.get_checks()
                    next_get = t + 300
                if t >= next_quiesce:
                    self.quiesce()
                    next_quiesce = t + QUIESCE_S
                self.save()
            except Exception as e:  # never let one bad minute end the soak
                self.append("incidents.jsonl", {"t": now(), "kind": "harness_error", "err": repr(e)[:500]})
        self.stop.set()
        self.quiesce()
        self.append("events.jsonl", {"t": now(), "kind": "end", "active_s": self.state["active_s"]})
        self.save()


def report(d):
    def load(name):
        try:
            with open(os.path.join(d, name)) as f:
                return [json.loads(line) for line in f if line.strip()]
        except OSError:
            return []
    metrics, lat, events, incidents = (load(n) for n in ("metrics.jsonl", "latency.jsonl", "events.jsonl", "incidents.jsonl"))
    with open(os.path.join(d, "model.json")) as f:
        state = json.load(f)["state"]
    hours = state["active_s"] / 3600
    print(f"soak: {hours:.1f} h under load, {state['chaos']} restarts/kills, {state['resumes']} environment restarts")
    print(f"  quiesces {state['quiesces']} ({state['quiesce_failures']} mismatched), GET checks {state['get_checks']} "
          f"({state['get_failures']} failed), native-vs-Lucene {state['native_compared']} ({state['native_mismatch']} differ), "
          f"writes lost across environment restarts {state['lost']}")
    kinds = {}
    for i in incidents:
        kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
    print("  incidents:", kinds or "none")

    # Plateaus: each day's max per node, after the first day's warm-up.
    def daily(key):
        days = {}
        for m in metrics:
            day = int(m["active_s"] // 86400)
            for nm in m["nodes"]:
                if nm.get(key) is not None:
                    days.setdefault(day, {}).setdefault(nm["node"], []).append(nm[key])
        return {day: {n: max(xs) for n, xs in per.items()} for day, per in sorted(days.items())}
    for key, unit, div in (("VmRSS", "MB", 1024), ("fds", "", 1), ("maps", "", 1), ("Threads", "", 1),
                           ("data_bytes", "MB", 1 << 20), ("segments", "", 1)):
        rows = daily(key)
        print(f"  {key} daily max per node{(' (' + unit + ')') if unit else ''}:")
        for day, per in rows.items():
            print(f"    day {day + 1}: " + "  ".join(f"{n} {per[n] / div:.0f}" for n in sorted(per)))
    # Latency: the first and last full day's median of per-minute medians,
    # minutes without chaos only.
    quiet = [x for x in lat if not x.get("chaos_active") and x.get("search_p50") is not None]
    if quiet:
        first = [x["search_p50"] for x in quiet if x["active_s"] < 86400 and x["active_s"] >= 3600]
        last_day = max(x["active_s"] for x in quiet) - 86400
        last = [x["search_p50"] for x in quiet if x["active_s"] >= last_day]
        f99 = [x["search_p99"] for x in quiet if x["active_s"] < 86400 and x["active_s"] >= 3600]
        l99 = [x["search_p99"] for x in quiet if x["active_s"] >= last_day]
        if first and last:
            print(f"  search latency, median of per-minute p50: first day {statistics.median(first):.1f} ms, "
                  f"last day {statistics.median(last):.1f} ms; p99 {statistics.median(f99):.1f} -> {statistics.median(l99):.1f} ms")
        per_min = [x["search_n"] for x in quiet]
        print(f"  searches per minute: median {statistics.median(per_min):.0f}; bulk items per minute: "
              f"median {statistics.median([x['bulk_items'] for x in quiet]):.0f}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    cmd, d = sys.argv[1], sys.argv[2]
    if cmd == "run":
        hours = float(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else 168.0
        Soak(d, hours).run("--resume" in sys.argv)
    elif cmd == "report":
        report(d)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
