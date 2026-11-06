#!/usr/bin/env python3
"""End-to-end verification of the Rust engine inside a real OpenSearch node.

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
  * the writer's circuit breaker trips (HTTP 429) instead of the node running out of memory;
  * the rust_encrypted store (encryption at rest) on both engines: plaintext-equal results, every
    shard file encrypted on disk, recovery after SIGKILL.

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
        # Custom analyzers built from Lucene components: analyzed in Rust by the Rust engine.
        "note": {"type": "text", "analyzer": "fancy"},
        "code": {"type": "text", "analyzer": "parts", "index_options": "offsets"},
        "syn": {"type": "text", "analyzer": "syn"},
        "hun": {"type": "text", "analyzer": "hun"},
        # The plugin's rust_thai analyzer, and its tokenizer in a custom chain.
        "thai": {"type": "text", "analyzer": "rust_thai"},
        "thai_html": {"type": "text", "analyzer": "thai_html"},
        "tag": {"type": "keyword"},
        "n": {"type": "long"},
        "price": {"type": "double"},
        "d": {"type": "date"},
        # OpenSearch gives @timestamp a doc-values skip index.
        "@timestamp": {"type": "date"},
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


ANALYSIS = {
    "analyzer": {
        "fancy": {"type": "custom", "char_filter": ["html_strip"], "tokenizer": "standard",
                  "filter": ["lowercase", "keep_both", "stop", "snowball"]},
        "parts": {"type": "custom", "tokenizer": "whitespace",
                  "filter": ["word_delimiter", "lowercase", "kstem", "remove_duplicates"]},
        "syn": {"type": "custom", "tokenizer": "standard",
                "filter": ["lowercase", "overrides", "synonyms", "porter_stem"]},
        # config/hunspell/en_US (opensearch-plugin/e2e/hunspell), two .dic files.
        "hun": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "en_stem"]},
        "thai_html": {"type": "custom", "char_filter": ["html_strip"], "tokenizer": "rust_thai",
                      "filter": ["lowercase"]},
    },
    "filter": {
        "keep_both": {"type": "asciifolding", "preserve_original": True},
        "overrides": {"type": "stemmer_override", "rules": ["runners => runner", "houses' => home"]},
        "en_stem": {"type": "hunspell", "locale": "en_US"},
        "synonyms": {"type": "synonym", "synonyms": ["quick, fast, speedy", "foxes => vixens, foxes",
                                                    "jumping cats => leaping felines", "café, coffee shop"]},
    },
}

NOTES = ["<p>Caf\u00e9 na\u00efve</p>", "r\u00e9sum\u00e9 &amp; <b>runners</b>", "The QUICK foxes",
         "houses'", "Stra\u00dfe \u00fcber", "jumping <i>cats</i>"]
THAI = ["ภาษาไทยง่ายนิดเดียว", "ประเทศไทยมีประชากรประมาณหกสิบหกล้านคน", "<b>กรุงเทพมหานคร</b>เป็นเมืองหลวง",
        "ฉันชอบกินข้าวผัดกับไข่ดาว", "นักเรียนกำลังอ่านหนังสือในห้องสมุด", "Thai ไทย 123 ๑๒๓"]
CODES = ["PowerShot500", "wi-fi_router", "parseHTTPRequest", "x86-64 ARMv8", "SD-500 sd500", "o'neil's"]


def source(r, i):
    doc = {
        "title": words(r, 1 + r.randrange(6)),
        "tag": f"t{r.randrange(15)}",
        "n": r.randrange(-1000, 100000),
        "price": round(r.random() * 500, 2),
        "d": f"2026-0{1 + r.randrange(9)}-{10 + r.randrange(18)}T{r.randrange(24):02d}:00:00Z",
        "@timestamp": 1_767_225_600_000 + i * 60_000 + r.randrange(60_000),
        "flag": r.random() < 0.5,
        "ip": f"10.{r.randrange(256)}.{r.randrange(256)}.{r.randrange(256)}",
        "loc": {"lat": round(r.uniform(-60, 60), 4), "lon": round(r.uniform(-170, 170), 4)},
        "obj": {"a": f"a{r.randrange(5)}", "b": r.randrange(100)},
        "nest": [{"k": f"k{r.randrange(4)}", "v": r.randrange(50)} for _ in range(r.randrange(3))],
    }
    if r.random() < 0.7:
        doc["body"] = [words(r, r.randrange(40)) for _ in range(1 + r.randrange(2))]
    if r.random() < 0.6:
        doc["note"] = [" ".join(r.choice(NOTES) for _ in range(1 + r.randrange(4))) for _ in range(1 + r.randrange(2))]
        doc["syn"] = doc["note"]
    if r.random() < 0.6:
        doc["hun"] = words(r, 1 + r.randrange(12))
    if r.random() < 0.5:
        doc["thai"] = " ".join(r.choice(THAI) for _ in range(1 + r.randrange(3)))
        doc["thai_html"] = doc["thai"]
    if r.random() < 0.6:
        doc["code"] = " ".join(r.choice(CODES) for _ in range(1 + r.randrange(5)))
    if r.random() < 0.1:
        del doc["title"]
    return doc


def create(index, rust, extra=None, mapping=None):
    # Explicit both ways: on a node whose lucene_rust.engine.default is true, an index that says
    # nothing is a Rust one.
    settings = {"number_of_shards": 2, "number_of_replicas": 0, "refresh_interval": -1,
                "index.lucene_rust.engine": bool(rust)}
    settings["analysis"] = ANALYSIS
    settings.update(extra or {})
    must("PUT", f"/{index}", {"settings": settings, "mappings": mapping or MAPPING})


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
    {"match": {"note": "cafe runner"}},
    {"match": {"note": "café uber"}},
    {"match_phrase": {"note": "quick fox"}},
    {"match": {"code": "shot 500"}},
    {"match": {"syn": "fast vixens"}},
    {"match_phrase": {"syn": "leaping felines"}},
    {"match": {"syn": "coffee runner"}},
    {"match": {"hun": "runners housing"}},
    {"match_phrase": {"hun": "lazy dogs"}},
    {"match": {"hun": "ran"}},
    {"match": {"thai": "ประเทศไทย"}},
    {"match_phrase": {"thai": "ข้าวผัด"}},
    {"match": {"thai_html": "กรุงเทพมหานคร ๑๒๓"}},
    {"match_phrase": {"code": "http request"}},
    {"prefix": {"tag": "t1"}},
    {"exists": {"field": "title"}},
    {"geo_distance": {"distance": "3000km", "loc": {"lat": 10, "lon": 10}}},
    {"term": {"ip": "10.1.2.3"}},
    {"match_all": {}},
    {"range": {"@timestamp": {"gte": 1_767_225_600_000 + 1000 * 60_000, "lt": 1_767_225_600_000 + 3000 * 60_000}}},
]

AGGS = {
    "tags": {"terms": {"field": "tag", "size": 30}},
    "hist": {"histogram": {"field": "n", "interval": 10000}},
    "dates": {"date_histogram": {"field": "d", "calendar_interval": "month"}},
    "ts": {"date_histogram": {"field": "@timestamp", "fixed_interval": "1d"}},
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
        # A double sum depends on the order it adds values in, which follows segment layout --
        # different between any two indices, Java or not. Twelve significant digits, not exact.
        if isinstance(v, float):
            return float(f"{v:.12g}")
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
    # Adjacent doc ids in one fetch take OpenSearch's sequential stored-fields reader, which
    # needs a `CodecReader` under every leaf wrapper -- a soft-deletes wrapper that was not one
    # failed the whole shard's fetch once a segment had deletes.
    seq = {"query": {"match_all": {}}, "size": 30, "sort": ["_doc"], "track_total_hits": True}
    for idx in (JAVA, RUST):
        r = must("POST", f"/{idx}/_search?request_cache=false", seq)
        check(r["_shards"]["failed"] == 0 and len(r["hits"]["hits"]) == min(30, r["hits"]["total"]["value"]),
              f"{label}: sequential fetch on {idx}: {json.dumps(r['_shards'])[:300]}")
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


def realtime():
    """index.lucene_rust.realtime_search: a plain scored search sees what was indexed before it,
    with no refresh, and scores it as a refreshed index scores the same documents. Aggregations
    and sorted searches see the same documents too, through the buffers written as segments."""
    rt, ref = "rt_live", "rt_ref"
    for idx in (rt, ref):
        req("DELETE", f"/{idx}")
    # Nested documents stay on the last refresh: no nested field here.
    flat = {"properties": {k: v for k, v in MAPPING["properties"].items() if v.get("type") != "nested"}}
    create(rt, True, {"index.lucene_rust.realtime_search": True}, flat)
    create(ref, True, None, flat)
    wait_green(f"{rt},{ref}")
    before = must("GET", "/_plugins/lucene_rust/stats").get("live_queries", 0)
    r = random.Random(11)
    queries = [{"match": {"body": "fox"}}, {"match": {"title": "quick dogs"}},
               {"bool": {"must": [{"match": {"body": "alpha"}}], "should": [{"term": {"tag": "t3"}}]}},
               {"match_phrase": {"body": "quick brown"}}, {"term": {"tag": "t3"}}]

    def scored(index, q):
        out = must("POST", f"/{index}/_search?request_cache=false", {"query": q, "size": 10000, "track_total_hits": True})
        return out["hits"]["total"]["value"], sorted((round(h["_score"], 4), h["_id"], json.dumps(h["_source"], sort_keys=True))
                                                      for h in out["hits"]["hits"])

    aggs = {"tags": {"terms": {"field": "tag", "size": 20}},
            "n": {"stats": {"field": "n"}},
            "range": {"range": {"field": "n", "ranges": [{"to": 0}, {"from": 0, "to": 50000}, {"from": 50000}]}},
            "obj": {"terms": {"field": "obj.a"}, "aggs": {"b": {"avg": {"field": "obj.b"}}}},
            "price": {"histogram": {"field": "price", "interval": 100}},
            "when": {"date_histogram": {"field": "d", "calendar_interval": "month"}}}
    sorts = [[{"n": "desc"}, {"@timestamp": "asc"}], [{"price": "asc"}, {"@timestamp": "asc"}],
             [{"tag": "asc"}, {"d": "desc"}, {"@timestamp": "asc"}]]

    def same_views(where):
        """Aggregations and sorted searches on the realtime index equal the refreshed index's."""
        for q in queries + [{"match_all": {}}]:
            body = {"query": q, "size": 0, "track_total_hits": True, "aggs": aggs}
            a = must("POST", f"/{rt}/_search?request_cache=false", body)
            b = must("POST", f"/{ref}/_search?request_cache=false", body)
            check(a["hits"]["total"]["value"] == b["hits"]["total"]["value"], f"{where}: aggregation total {q}")
            check(json.dumps(a["aggregations"], sort_keys=True) == json.dumps(b["aggregations"], sort_keys=True),
                  f"{where}: aggregations {q}: {json.dumps(a['aggregations'])[:300]} vs {json.dumps(b['aggregations'])[:300]}")
        for sort in sorts:
            body = {"query": {"match": {"body": "fox"}}, "size": 50, "sort": sort, "track_total_hits": True}
            a = must("POST", f"/{rt}/_search?request_cache=false", body)
            b = must("POST", f"/{ref}/_search?request_cache=false", body)
            ha = [(h["_id"], h["sort"]) for h in a["hits"]["hits"]]
            hb = [(h["_id"], h["sort"]) for h in b["hits"]["hits"]]
            check(a["hits"]["total"]["value"] == b["hits"]["total"]["value"], f"{where}: sorted total {sort}")
            check(ha == hb, f"{where}: sorted hits {sort}: {ha[:3]} vs {hb[:3]}")

    for rnd in range(3):
        lines = []
        for i in range(300):
            n = rnd * 300 + i
            lines += [{"index": {"_id": f"r{n}"}}, source(r, n)]
        body = "\n".join(json.dumps(l) for l in lines) + "\n"
        must("POST", f"/{rt}/_bulk", body, ndjson=True)
        must("POST", f"/{ref}/_bulk", body, ndjson=True)
        must("POST", f"/{ref}/_refresh")
        for q in queries:
            a, b = scored(rt, q), scored(ref, q)
            check(a[0] == b[0], f"realtime round {rnd}: total {q}: {a[0]} vs {b[0]}")
            check(a[1] == b[1], f"realtime round {rnd}: scored hits {q}: {a[1][:2]} vs {b[1][:2]}")
        same_views(f"realtime round {rnd}")
        if rnd == 0:
            # A flush publishes the buffers without refreshing the searcher: its view reads
            # them as their segments.
            must("POST", f"/{rt}/_flush")
            same_views("realtime after a flush")
        if rnd == 1:
            must("POST", f"/{rt}/_refresh")
    after = must("GET", "/_plugins/lucene_rust/stats").get("live_queries", 0)
    check(after - before >= 12 + 3 * (len(queries) + 1 + len(sorts)), f"realtime: live searches counted ({after - before})")
    # Updates and deletes of refreshed documents, and of buffered ones: the live search sees the
    # new versions only, as a refreshed index does.
    for idx in (rt, ref):
        must("POST", f"/{idx}/_doc/r5", {"body": "fox fox fox", "tag": "t3"})
        must("POST", f"/{idx}/_doc/r700", {"body": "fox alpha", "tag": "t3"})
        must("DELETE", f"/{idx}/_doc/r6")
        must("DELETE", f"/{idx}/_doc/r701")
    must("POST", f"/{ref}/_refresh")
    before = must("GET", "/_plugins/lucene_rust/stats").get("live_queries", 0)
    for q in queries:
        a, b = scored(rt, q), scored(ref, q)
        check(a[0] == b[0], f"realtime after updates: total {q}: {a[0]} vs {b[0]}")
        check(a[1] == b[1], f"realtime after updates: scored hits {q}: {a[1][:2]} vs {b[1][:2]}")
        ids = [h[1] for h in a[1]]
        check(ids.count("r5") <= 1 and "r6" not in ids, f"realtime after updates: {q} finds r5 once, r6 never")
    same_views("realtime after updates")
    after = must("GET", "/_plugins/lucene_rust/stats").get("live_queries", 0)
    check(after - before >= 4 + len(queries) + 1 + len(sorts), f"realtime: searches after updates stay live ({after - before})")
    for idx in (rt, ref):
        must("DELETE", f"/{idx}")


def bp_reorder():
    """index.lucene_rust.bp_reorder.*: a forced merge renumbers the documents, and searches find and
    score the same documents as on an index merged in order."""
    bp, ref = "bp_on", "bp_ref"
    for idx in (bp, ref):
        req("DELETE", f"/{idx}")
    flat = {"properties": {k: v for k, v in MAPPING["properties"].items() if v.get("type") != "nested"}}
    create(bp, True, {"index.lucene_rust.bp_reorder.enabled": True, "index.lucene_rust.bp_reorder.min_doc_freq": 2,
                      "index.lucene_rust.bp_reorder.min_partition_size": 4,
                      "index.lucene_rust.bp_reorder.min_merge_docs": 1}, flat)
    create(ref, True, None, flat)
    wait_green(f"{bp},{ref}")
    r = random.Random(23)
    for rnd in range(4):
        lines = []
        for i in range(400):
            n = rnd * 400 + i
            lines += [{"index": {"_id": f"b{n}"}}, source(r, n)]
        body = "\n".join(json.dumps(l) for l in lines) + "\n"
        for idx in (bp, ref):
            must("POST", f"/{idx}/_bulk", body, ndjson=True)
            must("POST", f"/{idx}/_refresh")
    for idx in (bp, ref):
        must("POST", f"/{idx}/_forcemerge?max_num_segments=1")
        must("POST", f"/{idx}/_refresh")
    queries = [{"match": {"body": "fox"}}, {"match": {"title": "quick dogs"}}, {"match_phrase": {"body": "quick brown"}},
               {"bool": {"must": [{"match": {"body": "alpha"}}], "filter": [{"term": {"tag": "t3"}}]}}]
    for q in queries:
        a = must("POST", f"/{bp}/_search?request_cache=false", {"query": q, "size": 10000, "track_total_hits": True})
        b = must("POST", f"/{ref}/_search?request_cache=false", {"query": q, "size": 10000, "track_total_hits": True})
        check(a["hits"]["total"]["value"] == b["hits"]["total"]["value"], f"bp_reorder: total {q}")
        hits = lambda out: sorted((round(h["_score"], 4), h["_id"]) for h in out["hits"]["hits"])
        check(hits(a) == hits(b), f"bp_reorder: scored hits {q}")
    order = lambda idx: [h["_id"] for h in must("POST", f"/{idx}/_search?request_cache=false",
                                                    {"size": 10000, "sort": ["_doc"], "query": {"match_all": {}}})["hits"]["hits"]]
    on, off = order(bp), order(ref)
    check(sorted(on) == sorted(off) and len(on) == 1600, "bp_reorder: every document survives the merge")
    check(on != off, "bp_reorder: the forced merge renumbered the documents")
    must("PUT", f"/{bp}/_settings", {"index.lucene_rust.bp_reorder.enabled": False})
    for idx in (bp, ref):
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
    # The 5xx comes back before the shard is failed: the engine fails itself and
    # IndexShard reports it to the cluster manager asynchronously, so the shard can
    # still be green here and its recovery record unchanged. Wait for a shard's
    # latest finished recovery to move (recoveries() returns only once every shard's
    # latest recovery is DONE), then for green.
    wait(lambda: recoveries("fault_rust") != before_rust, "the panicking shard to be failed and recovered", 300)
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


def recovery_types(index):
    """How each shard of `index` last recovered: EMPTY_STORE until a failed shard is recovered."""
    return {s["type"] for s in must("GET", f"/{index}/_recovery")[index]["shards"]}


def breaker():
    """The Rust writer's buffer is accounted: a tiny limit trips the breaker with a 429, and only
    the refused documents fail: the no-ops recording them are not refused, so no shard fails."""
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
    wait_green("breaker_rust")
    types = recovery_types("breaker_rust")
    check(types == {"EMPTY_STORE"}, f"a refused document fails no shard (recoveries: {types})")
    must("DELETE", "/breaker_rust")


def unsupported():
    """What the Rust writer cannot produce is refused, never written differently."""
    refused = (("bad_codec", {"index.codec": "best_compression"}, "default codec"),
               ("bad_fuzzy_set", {"index.optimize_doc_id_lookup.fuzzy_set.enabled": True}, "fuzzy_set"))
    for index, extra, named in refused:
        status, out = req("PUT", f"/{index}", {"settings": {"index.lucene_rust.engine": True,
                                                            "number_of_replicas": 0, **extra}})
        if status in (200, 201):
            time.sleep(3)
            health = must("GET", f"/_cluster/health/{index}?timeout=10s")
            check(health["status"] == "red", f"{index}: an unsupported setting does not start: {health['status']}")
            explain = must("GET", "/_cluster/allocation/explain", {"index": index, "shard": 0, "primary": True})
            check(named in json.dumps(explain), f"{index}: the refusal names [{named}]")
            must("DELETE", f"/{index}")
        else:
            check(named in json.dumps(out), f"{index}: the refusal names [{named}]: {out}")
    create("fuzzy_later", True)
    status, out = req("PUT", "/fuzzy_later/_settings", {"index.optimize_doc_id_lookup.fuzzy_set.enabled": True})
    check(status == 400 and "fuzzy_set" in json.dumps(out), f"turning fuzzy_set on later is refused: {status} {out}")
    must("DELETE", "/fuzzy_later")
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


def encrypted(container):
    """The rust_encrypted store (BYOK encryption at rest) under both engines: the same results as a
    plaintext index, every shard file encrypted on disk, and both survive SIGKILL."""
    enc = {"index.store.type": "rust_encrypted",
           "index.store.rust_encrypted.key_provider_type": "lucene_rust_local",
           "index.store.rust_encrypted.chunk_size": "16kb"}
    names = (("enc_plain", False, None), ("enc_java", False, enc), ("enc_rust", True, enc))
    for idx, rust, extra in names:
        req("DELETE", f"/{idx}")
        create(idx, rust, extra)
    # A shard that fails to start leaves its half of every bulk unindexed, and the searches below
    # then count only the other shard's documents.
    check(wait_green("enc_plain,enc_java,enc_rust"), "encrypted indices start green")
    r = random.Random(11)
    for batch in range(8):
        lines = []
        for i in range(250):
            n = batch * 250 + i
            lines += [{"index": {"_id": f"e{n}"}}, source(r, n)]
        if batch == 5:
            lines += [{"delete": {"_id": f"e{n}"}} for n in range(0, 1000, 9)]
        body = "\n".join(json.dumps(l) for l in lines) + "\n"
        for idx, _, _ in names:
            out = must("POST", f"/{idx}/_bulk", body, ndjson=True)
            errors = [v.get("error") for it in out["items"] for v in it.values() if "error" in v]
            check(not errors, f"{idx}: bulk {batch} applies every operation: {len(errors)} errors, {json.dumps(errors[:1])[:300]}")
            if batch % 3 == 2:
                must("POST", f"/{idx}/_refresh")
    for idx, _, _ in names:
        must("POST", f"/{idx}/_flush")
        must("POST", f"/{idx}/_refresh")
    compare_searches("encrypted, Java engine", scores=False, java="enc_plain", rust="enc_java")
    compare_searches("encrypted, Rust engine", scores=False, java="enc_plain", rust="enc_rust")

    for idx in ("enc_java", "enc_rust"):
        uuid = must("GET", f"/{idx}/_settings")[idx]["settings"]["index"]["uuid"]
        script = (f"cd /usr/share/opensearch/data/nodes/0/indices/{uuid} && "
                  "for f in */index/*; do [ \"${f##*/}\" = write.lock ] && continue; "
                  "printf '%s %s\\n' \"$(head -c4 \"$f\")\" \"$f\"; done")
        out = subprocess.run(["docker", "exec", container, "sh", "-c", script],
                             capture_output=True, text=True, errors="replace").stdout.splitlines()
        # Commit metadata (segments_N, .si) stays plaintext: OpenSearch's gateway reads it without
        # the store's directory (RustEncryptedDirectory.isCommitMetadata).
        def commit(line):
            name = line.rsplit("/", 1)[-1]
            return "segments_" in name or name.endswith(".si") or name.startswith("corrupted_")
        wrong = [l for l in out if l.startswith("LRSE ") == commit(l)]
        check(len(out) > 4 and not wrong, f"{idx}: every shard file but commit metadata is encrypted ({len(out)} files): {wrong[:5]}")
    stats = must("GET", "/_plugins/lucene_rust/stats")
    check(stats.get("native_errors", 0) == 0, f"no native errors on encrypted indices: {stats.get('native_errors')}")

    subprocess.run(["docker", "kill", "-s", "KILL", container], check=True, capture_output=True)
    subprocess.run(["docker", "start", container], check=True, capture_output=True)
    check(wait_green("enc_plain,enc_java,enc_rust", 300), "encrypted indices green after SIGKILL")
    compare_searches("encrypted after SIGKILL, Rust engine", scores=False, java="enc_plain", rust="enc_rust")
    for idx in ("enc_java", "enc_rust"):
        must("POST", f"/{idx}/_forcemerge?max_num_segments=1")
    compare_searches("encrypted, force-merged", scores=False, java="enc_plain", rust="enc_rust")

    # A store type the Rust engine cannot write through is refused, not bypassed.
    status, out = req("PUT", "/enc_bad", {"settings": {"index.lucene_rust.engine": True,
                                                        "index.store.type": "rust_encrypted",
                                                        "index.store.rust_encrypted.key_provider_type": "aws-kms",
                                                        "number_of_replicas": 0}})
    if status in (200, 201):
        time.sleep(3)
        health = must("GET", "/_cluster/health/enc_bad?timeout=10s")
        check(health["status"] == "red", f"an encrypted index without a key provider does not start: {health['status']}")
        must("DELETE", "/enc_bad")
    for idx, _, _ in names:
        must("DELETE", f"/{idx}")


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
    analyzed = must("GET", "/_plugins/lucene_rust/stats").get("analyzed_values", {})
    # title (english), body (standard), note and code (custom) are all Lucene components.
    check(analyzed.get("native", 0) > a.ops // 2, f"text values analyzed in Rust: {analyzed}")
    compare_all("after the stream", stream.ids)
    # Recovery: a graceful restart replays nothing; SIGKILL replays the translog into the writer.
    run_stream(600, seed=43)
    restart(a.container, kill=False)
    compare_all("after a restart", stream.ids)
    run_stream(600, seed=44)
    restart(a.container, kill=True)
    compare_all("after SIGKILL", stream.ids)
    insert_only_scoring()
    realtime()
    bp_reorder()
    fault_containment()
    breaker()
    unsupported()
    encrypted(a.container)
    print(f"verify_engine: {CHECKS[0]} checks, {len(FAILURES)} failures", flush=True)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
