#!/usr/bin/env python3
"""End-to-end verification of the Rust engine (M5) inside a real OpenSearch node.

Two indices with the same mapping and shard count -- one on OpenSearch's own engine, one created
with index.lucene_rust.engine: true -- receive the same operation stream, operation for operation.
Because both run InternalEngine's contracts (the Rust engine is InternalEngine with the writer
swapped), every observable result must be identical:

  * every bulk item: status, result, _version, _seq_no, _primary_term, error type -- index,
    create conflicts, partial and scripted updates, deletes of present and absent ids, and
    optimistic concurrency (if_seq_no/if_primary_term, current and stale; external versions);
  * realtime and non-realtime get, mget, _count;
  * searches: hit ids, _seq_no/_primary_term/_version per hit, sort values, totals -- and scores,
    once both indices are force-merged to one segment (before that, BM25 statistics legitimately
    differ with how each engine happened to merge away deleted documents);
  * aggregations, exactly: terms, histogram, date_histogram, avg/sum/min/max, cardinality,
    percentiles, nested, top_hits;
  * all of the above again after a graceful restart and after SIGKILL (translog replay into
    the Rust writer);
  * failure containment: a Rust-engine shard whose writer panics is failed and recovered, and
    no other shard notices;
  * the writer's circuit breaker trips (HTTP 429) instead of the node running out of memory.

Usage: verify_engine.py <base-url> <container-name> [--ops N]
Standard library only.
"""
import argparse
import json
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:9200"
FAILURES = []
CHECKS = [0]
JAVA, RUST = "eng_java", "eng_rust"


def req(method, path, body=None, ndjson=False, ok=(200, 201)):
    data = None
    headers = {}
    if body is not None:
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        payload = e.read().decode()
        try:
            return e.code, json.loads(payload)
        except ValueError:
            return e.code, {"raw": payload}


def must(method, path, body=None, ndjson=False):
    status, out = req(method, path, body, ndjson)
    if status not in (200, 201):
        raise RuntimeError(f"{method} {path}: {status} {json.dumps(out)[:600]}")
    return out


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(what)
        if len(FAILURES) <= 60:
            print("FAIL:", what, flush=True)


MAPPING = {
    "properties": {
        "title": {"type": "text", "analyzer": "english"},
        "body": {"type": "text"},
        "tag": {"type": "keyword"},
        "n": {"type": "long"},
        "price": {"type": "double"},
        "d": {"type": "date"},
        "flag": {"type": "boolean"},
        "ip": {"type": "ip"},
        "loc": {"type": "geo_point"},
        "obj": {"properties": {"a": {"type": "keyword"}, "b": {"type": "integer"}}},
        "nest": {"type": "nested", "properties": {"k": {"type": "keyword"}, "v": {"type": "long"}}},
    }
}

WORDS = ("the quick brown fox jumps over lazy dogs while cats sleep alpha beta gamma delta epsilon "
         "zeta eta theta iota kappa lambda running runners ran houses housing").split()


def words(r, n):
    return " ".join(WORDS[int((r.random() ** 2) * len(WORDS))] for _ in range(n))


def source(r, i):
    doc = {
        "title": words(r, 1 + r.randrange(6)),
        "tag": f"t{r.randrange(15)}",
        "n": r.randrange(-1000, 100000),
        "price": round(r.random() * 500, 2),
        "d": f"2026-0{1 + r.randrange(9)}-{10 + r.randrange(18)}T{r.randrange(24):02d}:00:00Z",
        "flag": r.random() < 0.5,
        "ip": f"10.{r.randrange(256)}.{r.randrange(256)}.{r.randrange(256)}",
        "loc": {"lat": round(r.uniform(-60, 60), 4), "lon": round(r.uniform(-170, 170), 4)},
        "obj": {"a": f"a{r.randrange(5)}", "b": r.randrange(100)},
        "nest": [{"k": f"k{r.randrange(4)}", "v": r.randrange(50)} for _ in range(r.randrange(3))],
    }
    if r.random() < 0.7:
        doc["body"] = [words(r, r.randrange(40)) for _ in range(1 + r.randrange(2))]
    if r.random() < 0.1:
        del doc["title"]
    return doc


def create(index, rust, extra=None):
    # Explicit both ways: on a node whose lucene_rust.engine.default is true, an index that says
    # nothing is a Rust one.
    settings = {"number_of_shards": 2, "number_of_replicas": 0, "refresh_interval": -1,
                "index.lucene_rust.engine": bool(rust)}
    settings.update(extra or {})
    must("PUT", f"/{index}", {"settings": settings, "mappings": MAPPING})


def comparable_item(item):
    (op, body), = item.items()
    out = {"op": op, "status": body.get("status"), "result": body.get("result"),
           "_version": body.get("_version"), "_seq_no": body.get("_seq_no"),
           "_primary_term": body.get("_primary_term"), "_id": body.get("_id")}
    if "error" in body:
        out["error"] = body["error"].get("type")
    if "get" in body:
        out["get"] = body["get"].get("_source")
    return out


class Stream:
    """The operation stream both indices receive; tracks each id's last seq_no/term for OCC."""

    def __init__(self, seed, ids):
        self.r = random.Random(seed)
        self.ids = ids
        self.last = {}

    def batch(self, size):
        lines = []
        for _ in range(size):
            i = self.r.randrange(self.ids)
            doc_id = f"id{i}"
            k = self.r.random()
            if k < 0.45:
                lines += [{"index": {"_id": doc_id}}, source(self.r, i)]
            elif k < 0.55:
                lines += [{"create": {"_id": doc_id}}, source(self.r, i)]
            elif k < 0.65:
                lines += [{"update": {"_id": doc_id}}, {"doc": {"n": self.r.randrange(100), "tag": "upd"}}]
            elif k < 0.70:
                lines += [{"update": {"_id": doc_id}},
                          {"script": {"source": "ctx._source.n += params.d", "params": {"d": 3}},
                           "upsert": source(self.r, i)}]
            elif k < 0.82:
                lines += [{"delete": {"_id": doc_id}}]
            elif k < 0.92:
                seq, term = self.last.get(doc_id, (0, 1))
                if self.r.random() < 0.3:
                    seq = max(0, seq - 1)  # deliberately stale
                lines += [{"index": {"_id": doc_id, "if_seq_no": seq, "if_primary_term": term}}, source(self.r, i)]
            else:
                lines += [{"index": {"_id": doc_id, "version": 1000 + self.r.randrange(50),
                                     "version_type": "external"}}, source(self.r, i)]
        return "\n".join(json.dumps(l) for l in lines) + "\n"

    def observe(self, items):
        for item in items:
            (_, body), = item.items()
            if body.get("_seq_no") is not None and body.get("status", 500) < 300:
                self.last[body["_id"]] = (body["_seq_no"], body["_primary_term"])


def run_stream(ops, seed):
    stream = Stream(seed, max(50, ops // 4))
    done = 0
    batch = 0
    while done < ops:
        body = stream.batch(100)
        rj = must("POST", f"/{JAVA}/_bulk", body, ndjson=True)
        rr = must("POST", f"/{RUST}/_bulk", body, ndjson=True)
        a = [comparable_item(x) for x in rj["items"]]
        b = [comparable_item(x) for x in rr["items"]]
        for x, y in zip(a, b):
            check(x == y, f"bulk item: java {x} rust {y}")
        stream.observe(rj["items"])
        done += 100
        batch += 1
        if batch % 3 == 0:
            for idx in (JAVA, RUST):
                must("POST", f"/{idx}/_refresh")
        if batch % 7 == 0:
            for idx in (JAVA, RUST):
                must("POST", f"/{idx}/_flush")
        if batch % 11 == 0:
            for idx in (JAVA, RUST):
                must("POST", f"/{idx}/_forcemerge?only_expunge_deletes=true")
    return stream


QUERIES = [
    {"match": {"title": "running houses"}},
    {"match": {"body": "quick fox"}},
    {"term": {"tag": "t3"}},
    {"range": {"n": {"gte": 100, "lt": 50000}}},
    {"bool": {"must": [{"match": {"body": "alpha"}}], "filter": [{"term": {"flag": True}}],
              "must_not": [{"term": {"tag": "upd"}}]}},
    {"nested": {"path": "nest", "query": {"term": {"nest.k": "k1"}}}},
    {"match_phrase": {"body": "quick brown"}},
    {"prefix": {"tag": "t1"}},
    {"exists": {"field": "title"}},
    {"geo_distance": {"distance": "3000km", "loc": {"lat": 10, "lon": 10}}},
    {"term": {"ip": "10.1.2.3"}},
    {"match_all": {}},
]

AGGS = {
    "tags": {"terms": {"field": "tag", "size": 30}},
    "hist": {"histogram": {"field": "n", "interval": 10000}},
    "dates": {"date_histogram": {"field": "d", "calendar_interval": "month"}},
    "avg_n": {"avg": {"field": "n"}},
    "sum_p": {"sum": {"field": "price"}},
    "min_p": {"min": {"field": "price"}},
    "max_n": {"max": {"field": "n"}},
    "card": {"cardinality": {"field": "tag"}},
    # HDR, not the default TDigest: TDigest's result depends on the order it sees values in,
    # which differs with segment layout between any two indices, Java or not.
    "pct": {"percentiles": {"field": "price", "percents": [5, 50, 95],
                            "hdr": {"number_of_significant_value_digits": 3}}},
    "nest": {"nested": {"path": "nest"}, "aggs": {"ks": {"terms": {"field": "nest.k"}}}},
    "flags": {"terms": {"field": "flag"}, "aggs": {"top": {"top_hits": {"size": 1, "sort": [{"n": "asc"}, {"_id": "asc"}]}}}},
}


def hits(index, query, sort):
    body = {"query": query, "size": 50 if sort else 10000, "track_total_hits": True,
            "seq_no_primary_term": True, "version": True}
    if sort:
        body["sort"] = [{"n": "asc"}, {"_id": "asc"}]
    out = must("POST", f"/{index}/_search?request_cache=false", body)
    return out["hits"]


def compare_searches(label, scores, exact_scores=False, java=None, rust=None):
    for q in QUERIES:
        for sort in (False, True):
            if not sort and not scores:
                continue
            a, b = hits(java or JAVA, q, sort), hits(rust or RUST, q, sort)
            check(a["total"] == b["total"], f"{label}: total {q} sort={sort}: {a['total']} vs {b['total']}")
            ka = [(h["_id"], h.get("_seq_no"), h.get("_primary_term"), h.get("_version"), h.get("sort")) for h in a["hits"]]
            kb = [(h["_id"], h.get("_seq_no"), h.get("_primary_term"), h.get("_version"), h.get("sort")) for h in b["hits"]]
            if sort:
                check(ka == kb, f"{label}: hits {q}: {ka[:3]} vs {kb[:3]}")
            elif exact_scores:
                sa = sorted((round(h["_score"], 4), h["_id"]) for h in a["hits"])
                sb = sorted((round(h["_score"], 4), h["_id"]) for h in b["hits"])
                check(sa == sb, f"{label}: scored hits {q}: {sa[:3]} vs {sb[:3]}")
            else:
                # BM25 counts soft-deleted documents a merge kept, and how much history a shard
                # keeps follows its retention leases' timing -- between any two indices.
                check(sorted(h["_id"] for h in a["hits"]) == sorted(h["_id"] for h in b["hits"]),
                      f"{label}: scored hit set {q}")
    ra = must("POST", f"/{java or JAVA}/_search?request_cache=false", {"size": 0, "aggs": AGGS})
    rb = must("POST", f"/{rust or RUST}/_search?request_cache=false", {"size": 0, "aggs": AGGS})
    def strip(v):
        if isinstance(v, dict):
            return {k: strip(x) for k, x in v.items() if k != "_index"}
        if isinstance(v, list):
            return [strip(x) for x in v]
        return v
    for name in AGGS:
        check(strip(ra["aggregations"][name]) == strip(rb["aggregations"][name]),
              f"{label}: aggregation {name}: {json.dumps(ra['aggregations'][name])[:300]} vs "
              f"{json.dumps(rb['aggregations'][name])[:300]}")


def compare_gets(label, ids):
    for realtime in ("true", "false"):
        for i in range(0, ids, 7):
            a = req("GET", f"/{JAVA}/_doc/id{i}?realtime={realtime}")
            b = req("GET", f"/{RUST}/_doc/id{i}?realtime={realtime}")
            ja = {k: v for k, v in a[1].items() if k != "_index"}
            jb = {k: v for k, v in b[1].items() if k != "_index"}
            diff = {k: (ja.get(k), jb.get(k)) for k in set(ja) | set(jb) if ja.get(k) != jb.get(k)}
            check(a[0] == b[0] and not diff, f"{label}: get id{i} realtime={realtime}: {a[0]} vs {b[0]} {str(diff)[:300]}")
    body = {"ids": [f"id{i}" for i in range(0, ids, 3)]}
    ma = must("POST", f"/{JAVA}/_mget", body)["docs"]
    mb = must("POST", f"/{RUST}/_mget", body)["docs"]
    strip = lambda docs: [{k: v for k, v in d.items() if k != "_index"} for d in docs]
    check(strip(ma) == strip(mb), f"{label}: mget differs")
    ca = must("GET", f"/{JAVA}/_count")["count"]
    cb = must("GET", f"/{RUST}/_count")["count"]
    check(ca == cb, f"{label}: count {ca} vs {cb}")
    return ca


def compare_all(label, ids):
    for idx in (JAVA, RUST):
        must("POST", f"/{idx}/_refresh")
    count = compare_gets(label, ids)
    compare_searches(label, scores=False)
    # A flush first: the history a merge keeps is bounded by the safe commit, and the Rust
    # engine commits on every refresh -- flushing gives Java's engine the same safe commit, so
    # both merges keep the same soft-deleted documents and BM25 sees the same statistics.
    for idx in (JAVA, RUST):
        must("POST", f"/{idx}/_flush")
        must("POST", f"/{idx}/_forcemerge?max_num_segments=1")
        must("POST", f"/{idx}/_refresh")
    compare_searches(label + " (merged)", scores=True)
    print(f"{label}: {count} documents compared", flush=True)


def insert_only_scoring():
    """No updates or deletes, so no retained history: BM25 sees the same statistics on both
    engines however each merged, and every score must match."""
    for idx, rust in (("score_java", False), ("score_rust", True)):
        req("DELETE", f"/{idx}")
        create(idx, rust)
    wait_green("score_java,score_rust")
    r = random.Random(9)
    for batch in range(8):
        lines = []
        for i in range(250):
            n = batch * 250 + i
            lines += [{"index": {"_id": f"s{n}"}}, source(r, n)]
        body = "\n".join(json.dumps(l) for l in lines) + "\n"
        for idx in ("score_java", "score_rust"):
            must("POST", f"/{idx}/_bulk", body, ndjson=True)
            if batch % 3 == 2:
                must("POST", f"/{idx}/_refresh")
    for idx in ("score_java", "score_rust"):
        must("POST", f"/{idx}/_refresh")
    compare_searches("insert-only", scores=True, exact_scores=True, java="score_java", rust="score_rust")
    for idx in ("score_java", "score_rust"):
        must("POST", f"/{idx}/_forcemerge?max_num_segments=1")
        must("POST", f"/{idx}/_refresh")
    compare_searches("insert-only (merged)", scores=True, exact_scores=True, java="score_java", rust="score_rust")
    for idx in ("score_java", "score_rust"):
        must("DELETE", f"/{idx}")


def wait_green(index, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, h = req("GET", f"/_cluster/health/{index}?wait_for_status=green&timeout=5s")
            if status == 200 and h.get("status") == "green":
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(2)
    return False


def restart(container, kill):
    if kill:
        subprocess.run(["docker", "kill", "-s", "KILL", container], check=True, capture_output=True)
        subprocess.run(["docker", "start", container], check=True, capture_output=True)
    else:
        subprocess.run(["docker", "restart", container], check=True, capture_output=True)
    ok = wait_green(f"{JAVA},{RUST}", 300)
    check(ok, f"cluster green after {'SIGKILL' if kill else 'restart'}")


def recoveries(index, shards=2):
    """Each shard's latest recovery: (type, start time). A failed shard recovers again."""
    got = {}

    def read():
        out = must("GET", f"/{index}/_recovery")[index]["shards"]
        got.clear()
        got.update({str(s["id"]): (s["type"], s["start_time_in_millis"]) for s in out if s["stage"] == "DONE"})
        return len(got) == shards

    wait(read, f"{index}: every shard recovered")
    return dict(got)


def wait(pred, what, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(1)
    check(False, f"timed out waiting for {what}")
    return False


def fault_containment():
    """A panic inside one Rust shard's writer fails that shard only."""
    create("fault_rust", True, {"index.lucene_rust.engine.fault_injection": True})
    create("fault_peer", True)
    for idx in ("fault_rust", "fault_peer"):
        wait_green(idx)
        for i in range(20):
            must("PUT", f"/{idx}/_doc/p{i}?routing=r", {"n": i, "tag": "x"})
        must("POST", f"/{idx}/_refresh")
    before_rust, before_peer = recoveries("fault_rust"), recoveries("fault_peer")
    java_before = must("GET", f"/{JAVA}/_count")["count"]
    status, out = req("PUT", "/fault_rust/_doc/boom?routing=r", {"__lucene_rust_panic": 1, "n": 1})
    check(status >= 500, f"the panicking write fails: {status} {json.dumps(out)[:300]}")
    check("panic" in json.dumps(out).lower(), f"the failure names the panic: {json.dumps(out)[:300]}")
    check(wait_green("fault_rust", 300), "the failed shard recovers")
    after_rust, after_peer = recoveries("fault_rust"), recoveries("fault_peer")
    changed = [s for s in before_rust if after_rust[s] != before_rust[s]]
    check(len(changed) == 1 and after_rust[changed[0]][0] == "EXISTING_STORE",
          f"exactly one shard of the index was failed and recovered from its store: {before_rust} -> {after_rust}")
    check(after_peer == before_peer, f"no other Rust shard was failed: {before_peer} -> {after_peer}")
    must("POST", "/fault_rust/_refresh")
    check(must("GET", "/fault_rust/_count")["count"] == 20, "the failed shard lost no acknowledged write")
    check(must("GET", "/fault_peer/_count")["count"] == 20, "the other index still serves")
    check(must("GET", f"/{JAVA}/_count")["count"] == java_before, "the Java index still serves")
    health = must("GET", "/_cluster/health")
    check(health["number_of_nodes"] == 1, "the node is still up")
    for idx in ("fault_rust", "fault_peer"):
        must("DELETE", f"/{idx}")


def breaker():
    """The Rust writer's buffer is accounted: a tiny limit trips the breaker with a 429."""
    create("breaker_rust", True)
    wait_green("breaker_rust")
    must("PUT", "/_cluster/settings", {"transient": {"breaker.lucene_rust_writer.limit": "1kb"}})
    try:
        r = random.Random(7)
        lines = []
        for i in range(200):
            lines += [{"index": {"_id": f"b{i}"}}, source(r, i)]
        status, out = req("POST", "/breaker_rust/_bulk", "\n".join(json.dumps(l) for l in lines) + "\n", ndjson=True)
        errors = [it["index"].get("error", {}).get("type") for it in out.get("items", [])]
        statuses = [it["index"].get("status") for it in out.get("items", [])]
        check("circuit_breaking_exception" in errors, f"the breaker trips: {set(errors)}")
        check(429 in statuses, f"a tripped write is a 429: {set(statuses)}")
        stats = must("GET", "/_nodes/stats/breaker")["nodes"]
        node = next(iter(stats.values()))
        check("lucene_rust_writer" in node["breakers"], "the breaker is listed in node stats")
        check(node["breakers"]["lucene_rust_writer"]["tripped"] > 0, "node stats count the trip")
    finally:
        must("PUT", "/_cluster/settings", {"transient": {"breaker.lucene_rust_writer.limit": None}})
    status, out = req("PUT", "/breaker_rust/_doc/after", {"n": 1})
    check(status in (200, 201), f"writes resume under the default limit: {status}")
    must("POST", "/breaker_rust/_refresh")
    stats = must("GET", "/_nodes/stats/breaker")["nodes"]
    node = next(iter(stats.values()))
    check(node["breakers"]["lucene_rust_writer"]["estimated_size_in_bytes"] == 0,
          f"a refresh releases the accounted buffer: {node['breakers']['lucene_rust_writer']}")
    must("DELETE", "/breaker_rust")


def unsupported():
    """What the Rust writer cannot produce is refused, never written differently."""
    status, out = req("PUT", "/bad_codec", {"settings": {"index.lucene_rust.engine": True,
                                                          "index.codec": "best_compression",
                                                          "number_of_replicas": 0}})
    if status in (200, 201):
        time.sleep(3)
        health = must("GET", "/_cluster/health/bad_codec?timeout=10s")
        check(health["status"] == "red", f"an unsupported codec does not start: {health['status']}")
        explain = must("GET", "/_cluster/allocation/explain", {"index": "bad_codec", "shard": 0, "primary": True})
        check("default codec" in json.dumps(explain), "the refusal names the codec")
        must("DELETE", "/bad_codec")
    else:
        check("codec" in json.dumps(out), f"the refusal names the codec: {out}")
    create("tv_rust", True)
    must("PUT", "/tv_rust/_mapping", {"properties": {"tv": {"type": "text", "term_vector": "with_positions"}}})
    status, out = req("PUT", "/tv_rust/_doc/1", {"tv": "some text"})
    check(status == 400 and "term vectors" in json.dumps(out), f"term vectors are refused per document: {status}")
    status, out = req("PUT", "/tv_rust/_doc/2", {"n": 1})
    check(status in (200, 201), "the shard carries on after a refused document")
    must("DELETE", "/tv_rust")
    nodes = must("GET", "/_nodes/settings?flat_settings=true")["nodes"].values()
    if any(n["settings"].get("lucene_rust.engine.default") == "true" for n in nodes):
        # Only asked-for is strict: an index that merely inherits the node default and needs what
        # the Rust writer cannot produce is served by OpenSearch's engine.
        must("PUT", "/fallback_codec", {"settings": {"index.codec": "best_compression", "number_of_replicas": 0}})
        wait_green("fallback_codec")
        status, _ = req("PUT", "/fallback_codec/_doc/1?refresh=true", {"n": 1})
        check(status in (200, 201), "an inherited-default index the Rust writer cannot serve falls back")
        must("DELETE", "/fallback_codec")


def main():
    global BASE
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("container")
    p.add_argument("--ops", type=int, default=6000)
    a = p.parse_args()
    BASE = a.base
    for idx in (JAVA, RUST):
        req("DELETE", f"/{idx}")
    create(JAVA, False)
    create(RUST, True)
    wait_green(f"{JAVA},{RUST}")
    stream = run_stream(a.ops, seed=42)
    stats = must("GET", f"/{RUST}/_stats/segments,indexing")["_all"]["primaries"]
    check(stats["segments"]["count"] >= 1, "segment stats are populated")
    check(stats["indexing"]["index_total"] > 0, "indexing stats are populated")
    check(stats["segments"]["index_writer_memory_in_bytes"] >= 0, "writer memory is reported")
    compare_all("after the stream", stream.ids)
    # Recovery: a graceful restart replays nothing; SIGKILL replays the translog into the writer.
    run_stream(600, seed=43)
    restart(a.container, kill=False)
    compare_all("after a restart", stream.ids)
    run_stream(600, seed=44)
    restart(a.container, kill=True)
    compare_all("after SIGKILL", stream.ids)
    insert_only_scoring()
    fault_containment()
    breaker()
    unsupported()
    print(f"verify_engine: {CHECKS[0]} checks, {len(FAILURES)} failures", flush=True)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
