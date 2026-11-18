#!/usr/bin/env python3
"""Differential fuzzer: random corpus, queries, sorts and aggregations run on a plugin node and
a stock OpenSearch node; any difference in hits, scores, sort order, aggregations or errors is a
finding.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/fuzz_diff.py --seed 1 --docs 400 --queries 400 --shards 1

The corpus (text with several analyzers, keyword, numeric, date, boolean, ip, geo_point, nested,
multi-fields; nulls, arrays, missing fields, unicode, extreme numbers) is indexed the same way on
both sides, with random deletes and updates, then optionally force-merged.  Hits are compared as
id -> score maps over the whole corpus (size = docs), ordering only where a sort with an _id
tie-break is given.  Findings are printed one per line as JSON and appended to --out.
"""
import argparse
import json
import math
import random
import sys
import urllib.request
import urllib.error

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma "
         "tau upsilon phi chi psi omega running runs ran jumped jumping quick brown fox lazy dog "
         "the a of and to in is it").split()
UNI = ["café", "naïve", "日本語", "مرحبا", "\U0001f600smile", "straße",
       "über", "Ω", "é", "İstanbul", "ǅ", "ﬁne", "x​y"]
TAGS = ["red", "green", "blue", "Red", "RED", "a b", "", "x*y", "x.y", "UPPER", "lower", "10", "-1"]


_tie_rng = random.Random(4242)


def TIE():
    """The unique sort tie-break: `_seq_no` runs on the native path, `_id` falls back to Lucene
    (the `_id` sort has no native comparator), so most cases use the former."""
    return {"_seq_no": "asc"} if _tie_rng.random() < 0.8 else {"_id": "asc"}


class Node:
    def __init__(self, url, timeout=300):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def req(self, method, path, body=None, raw=False):
        data = None
        hdr = {}
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            hdr["Content-Type"] = "application/x-ndjson" if raw else "application/json"
        r = urllib.request.Request(self.url + path, data=data, method=method, headers=hdr)
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as f:
                return f.status, json.loads(f.read() or b"null")
        except urllib.error.HTTPError as e:
            txt = e.read()
            try:
                return e.code, json.loads(txt)
            except Exception:
                return e.code, {"raw": txt.decode("utf8", "replace")}
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            return -1, {"error": {"type": "connection", "reason": str(e)[:200]}}


def mapping():
    return {"properties": {
        "title": {"type": "text", "fields": {"raw": {"type": "keyword", "ignore_above": 40}}},
        "body": {"type": "text", "analyzer": "english"},
        "ws": {"type": "text", "analyzer": "whitespace"},
        "tag": {"type": "keyword"},
        "ltag": {"type": "keyword", "normalizer": "lc"},
        "tags": {"type": "keyword"},
        "n": {"type": "long"},
        "i": {"type": "integer"},
        "sh": {"type": "short"},
        "price": {"type": "double"},
        "f": {"type": "float"},
        "hf": {"type": "half_float"},
        "sf": {"type": "scaled_float", "scaling_factor": 100},
        "ts": {"type": "date"},
        "flag": {"type": "boolean"},
        "ip": {"type": "ip"},
        "loc": {"type": "geo_point"},
        "nodv": {"type": "keyword", "doc_values": False},
        "noidx": {"type": "long", "index": False},
        "comments": {"type": "nested", "properties": {
            "author": {"type": "keyword"}, "text": {"type": "text"}, "stars": {"type": "integer"}}},
    }}


def settings(shards, shapes="fast"):
    return {"settings": {"number_of_shards": shards, "number_of_replicas": 0, **({"index.search.accelerated_shapes": "all"} if shapes == "all" else {}),
                         "analysis": {"normalizer": {"lc": {"type": "custom", "filter": ["lowercase"]}}}},
            "mappings": mapping()}


VOCAB = []
WEIGHTS = []


def set_vocab(n):
    """Zipf vocabulary: the fixed words first (most frequent), then n synthetic ones, so posting
    lists range from one document to nearly every document (multi-block, bitset, skip data)."""
    global VOCAB, WEIGHTS
    VOCAB = WORDS + ["w%d" % i for i in range(n)]
    WEIGHTS = [1.0 / (i + 1) for i in range(len(VOCAB))]


def sentence(r, lo=1, hi=8):
    if VOCAB:
        ws = r.choices(VOCAB, WEIGHTS, k=r.randint(lo, hi) * (3 if hi > 10 else 1))
        return " ".join(w if r.random() > 0.02 else r.choice(UNI) for w in ws)
    ws = [r.choice(WORDS) if r.random() > 0.08 else r.choice(UNI) for _ in range(r.randint(lo, hi))]
    return " ".join(ws)


def make_doc(r, i):
    d = {}
    big = len(VOCAB) > 0
    def maybe(p):
        return r.random() < p
    if maybe(0.9):
        d["title"] = sentence(r, 1, 6) if maybe(0.97) else None
    if maybe(0.85):
        d["body"] = sentence(r, 3, 25)
    if maybe(0.5):
        d["ws"] = sentence(r, 1, 5).upper() if maybe(0.5) else sentence(r, 1, 5)
    if maybe(0.8):
        d["tag"] = r.choice(TAGS) if maybe(0.97) else None
    if maybe(0.6):
        d["ltag"] = r.choice(TAGS)
    if maybe(0.6):
        d["tags"] = [r.choice(TAGS) for _ in range(r.randint(0, 4))]
    if maybe(0.85):
        d["n"] = r.choice([0, 1, -1, 7, 42, 2**31, -2**31, 2**53, -2**62, 2**63 - 1, r.randint(-1000, 1000)])
    if maybe(0.8):
        d["i"] = r.choice([0, 5, -5, 2**31 - 1, -2**31, r.randint(-50, 50)])
    if maybe(0.5):
        d["sh"] = r.randint(-100, 100)
    if maybe(0.85):
        d["price"] = r.choice([0.0, -0.0, 1.5, -1.5, 1e6, -1e6, 1e-9, r.uniform(-100, 100), round(r.uniform(0, 50), 1)])
    if maybe(0.5):
        d["f"] = r.choice([0.1, 0.5, 1.0, r.uniform(-10, 10)])
    if maybe(0.4):
        d["hf"] = round(r.uniform(-10, 10), 2)
    if maybe(0.5):
        d["sf"] = round(r.uniform(-20, 20), 3)
    if maybe(0.85):
        y = r.randint(1960, 2030)
        d["ts"] = r.choice([f"{y}-{r.randint(1,12):02d}-{r.randint(1,28):02d}T{r.randint(0,23):02d}:{r.randint(0,59):02d}:00Z",
                            r.randint(0, 2_000_000_000_000), f"{y}-01-01", "1969-12-31T23:59:59.999Z"])
    if maybe(0.7):
        d["flag"] = r.choice([True, False, "true", "false", ""]) if maybe(0.95) else None
    if maybe(0.6):
        d["ip"] = r.choice(["10.0.0.1", "192.168.1.5", "::1", "2001:db8::1", f"10.{r.randint(0,3)}.{r.randint(0,255)}.{r.randint(0,255)}"])
    if maybe(0.6):
        d["loc"] = r.choice([{"lat": r.uniform(-85, 85), "lon": r.uniform(-180, 180)},
                             [r.uniform(-180, 180), r.uniform(-85, 85)], "41.12,-71.34", {"lat": 0, "lon": 0}])
    if maybe(0.3):
        d["nodv"] = r.choice(TAGS)
    if maybe(0.3):
        d["noidx"] = r.randint(0, 10)
    if maybe(0.5):
        d["comments"] = [{"author": r.choice(TAGS[:5]), "text": sentence(r, 1, 6), "stars": r.randint(1, 5)}
                         for _ in range(r.randint(0, 4))]
    if maybe(0.05):
        d["unmapped"] = {"x": r.randint(0, 3)}
    return d


def ingest(nodes, index, shards, seed, ndocs, merge, log, shapes="fast"):
    r = random.Random(seed)
    for n in nodes:
        n.req("DELETE", f"/{index}")
        st, resp = n.req("PUT", f"/{index}", settings(shards, shapes if n.url.endswith("9200") or n is nodes[0] else "fast"))
        if st != 200:
            raise SystemExit(f"create index failed on {n.url}: {resp}")
    docs = {i: make_doc(r, i) for i in range(ndocs)}
    batch = 100
    ids = list(docs)
    ops = []
    for i in ids:
        ops.append(("index", i, docs[i]))
    # updates / deletes interleaved
    for _ in range(ndocs // 8):
        i = r.choice(ids)
        ops.append(("index", i, make_doc(r, i)))
    for _ in range(ndocs // 10):
        ops.append(("delete", r.choice(ids), None))
    for s in range(0, len(ops), batch):
        lines = []
        for kind, i, doc in ops[s:s + batch]:
            lines.append(json.dumps({kind: {"_id": f"d{i}"}}))
            if doc is not None:
                lines.append(json.dumps(doc))
        body = ("\n".join(lines) + "\n").encode()
        results = []
        for n in nodes:
            st, resp = n.req("POST", f"/{index}/_bulk?refresh=true", body, raw=True)
            results.append((st, [(list(it.values())[0].get("status"), list(it.values())[0].get("error", {}).get("type"))
                                 for it in resp.get("items", [])] if isinstance(resp, dict) else resp))
        if results[0] != results[1]:
            log({"kind": "bulk-diff", "batch": s, "plugin": str(results[0])[:600], "stock": str(results[1])[:600]})
    for n in nodes:
        if merge:
            n.req("POST", f"/{index}/_forcemerge?max_num_segments=1")
        n.req("POST", f"/{index}/_refresh")


# --------------------------------------------------------------------------- queries

TEXT_FIELDS = ["title", "body", "ws"]
KW_FIELDS = ["tag", "ltag", "tags", "title.raw", "nodv"]
NUM_FIELDS = ["n", "i", "sh", "price", "f", "hf", "sf"]
ALL_FIELDS = TEXT_FIELDS + KW_FIELDS + NUM_FIELDS + ["ts", "flag", "ip", "loc", "noidx", "missing_field", "unmapped.x"]


def word(r):
    if VOCAB and r.random() < 0.7:
        return r.choices(VOCAB, WEIGHTS)[0] if r.random() < 0.6 else r.choice(VOCAB)
    return r.choice(WORDS + UNI + ["Alpha", "RUNNING", "zzz", "al*", "run"])


def numval(r, field):
    if field in ("price", "f", "hf", "sf"):
        return r.choice([0, 1.5, -1.5, 3.25, r.uniform(-50, 50), 1e10, "2.5"])
    return r.choice([0, 1, -1, 7, 42, 2**31, r.randint(-100, 100), "5"])


def leaf(r):
    k = r.choice(["term", "term", "terms", "match", "match", "match_phrase", "prefix", "wildcard", "regexp", "range",
                  "range", "exists", "ids", "match_all", "match_none", "query_string", "simple_query_string",
                  "multi_match", "fuzzy", "match_phrase_prefix", "match_bool_prefix", "geo_distance", "geo_bbox", "intervals",
                  "term_ci", "span_term", "combined_fields", "script", "wrapper_term", "nested", "nested", "regexp_ci", "wildcard_ci"])
    tf = r.choice(TEXT_FIELDS)
    kf = r.choice(KW_FIELDS)
    nf = r.choice(NUM_FIELDS)
    if k == "term":
        f = r.choice(ALL_FIELDS)
        if f in NUM_FIELDS:
            v = numval(r, f)
        elif f == "ts":
            v = r.choice(["2000-01-01", "1969-12-31T23:59:59.999Z", 0])
        elif f == "flag":
            v = r.choice([True, False, "true"])
        elif f == "ip":
            v = r.choice(["10.0.0.1", "::1", "10.0.0.0/8", "192.168.0.0/16"])
        elif f == "loc":
            v = "41.12,-71.34"
        else:
            v = r.choice(TAGS + WORDS + UNI)
        return {"term": {f: {"value": v, **({"boost": r.choice([0.5, 2])} if r.random() < .2 else {})}}}
    if k == "terms":
        f = r.choice(KW_FIELDS + NUM_FIELDS + TEXT_FIELDS)
        vs = [numval(r, f) if f in NUM_FIELDS else word(r) if f in TEXT_FIELDS else r.choice(TAGS) for _ in range(r.randint(0, 5))]
        return {"terms": {f: vs}}
    if k == "match":
        f = r.choice(TEXT_FIELDS + ["tag", "n", "flag", "ts"])
        q = sentence(r, 1, 4) if f in TEXT_FIELDS else (str(numval(r, "n")) if f == "n" else r.choice(["true", "2000-01-01", "red"]))
        m = {"query": q}
        if r.random() < .4:
            m["operator"] = r.choice(["and", "or"])
        if r.random() < .3:
            m["minimum_should_match"] = r.choice([1, 2, "50%", "-1", "75%", "2<80%"])
        if r.random() < .3:
            m["fuzziness"] = r.choice(["AUTO", 0, 1, 2, "AUTO:2,6"])
            if r.random() < .5:
                m["prefix_length"] = r.randint(0, 2)
            if r.random() < .3:
                m["transpositions"] = r.choice([True, False])
            if r.random() < .3:
                m["max_expansions"] = r.choice([1, 3, 50])
        if r.random() < .15:
            m["zero_terms_query"] = r.choice(["all", "none"])
        if r.random() < .15:
            m["analyzer"] = r.choice(["whitespace", "keyword", "english", "standard", "simple", "stop"])
        if r.random() < .15:
            m["lenient"] = True
        if r.random() < .15:
            m["auto_generate_synonyms_phrase_query"] = r.choice([True, False])
        return {"match": {f: m}}
    if k == "match_phrase":
        m = {"query": sentence(r, 1, 3)}
        if r.random() < .4:
            m["slop"] = r.randint(0, 4)
        if r.random() < .2:
            m["analyzer"] = r.choice(["whitespace", "simple", "english"])
        return {"match_phrase": {tf: m}}
    if k == "match_phrase_prefix":
        return {"match_phrase_prefix": {tf: {"query": sentence(r, 1, 2)[:r.randint(1, 8)], **({"slop": r.randint(0, 3)} if r.random() < .3 else {}), **({"max_expansions": r.choice([1, 3, 50])} if r.random() < .3 else {})}}}
    if k == "match_bool_prefix":
        return {"match_bool_prefix": {tf: {"query": sentence(r, 1, 3)[:r.randint(2, 15)], **({"operator": "and"} if r.random() < .3 else {})}}}
    if k == "prefix":
        f = r.choice(KW_FIELDS + TEXT_FIELDS)
        return {"prefix": {f: {"value": word(r)[:r.randint(0, 3)], **({"case_insensitive": True} if r.random() < .3 else {})}}}
    if k in ("wildcard", "wildcard_ci"):
        f = r.choice(KW_FIELDS + TEXT_FIELDS)
        w = word(r)
        pat = r.choice([w[:2] + "*", "*" + w[-2:], "?" + w[1:], "*" + w[1:3] + "*", w[:1] + "*" + w[-1:], "*", "?*", "\\*", w + "?"])
        return {"wildcard": {f: {"value": pat, **({"case_insensitive": True} if k == "wildcard_ci" or r.random() < .2 else {})}}}
    if k in ("regexp", "regexp_ci"):
        f = r.choice(KW_FIELDS + TEXT_FIELDS)
        w = word(r)
        pat = r.choice([w[:2] + ".*", ".*" + w[-2:], "[a-c].*", w + "|" + word(r), "(" + w[:2] + ")+.*", ".{2,4}", "~" + w, "<1-9>", "@", "&", w[:1] + "[^a]*", "a{1,2}b?"])
        return {"regexp": {f: {"value": pat, **({"case_insensitive": True} if k == "regexp_ci" or r.random() < .2 else {}), **({"flags": r.choice(["ALL", "NONE", "INTERSECTION", "COMPLEMENT"])} if r.random() < .2 else {})}}}
    if k == "range":
        f = r.choice(NUM_FIELDS + ["ts", "ip"] + KW_FIELDS + TEXT_FIELDS)
        if f in NUM_FIELDS:
            vals = [v for v in (numval(r, f) for _ in range(6)) if isinstance(v, (int, float))][:2]
            a, b = sorted(vals + [0] * (2 - len(vals)))
        elif f == "ts":
            a, b = sorted(r.choice(["1990-01-01", "2000-01-01", "2010-06-01T12:00:00Z", "1000000000000", "2015-03-01", "2020"]) for _ in range(2)) if r.random() < .5 else ("2000-01-01", "2024-01-01")
        elif f == "ip":
            a, b = "10.0.0.0", "10.1.255.255"
        else:
            a, b = sorted([r.choice(TAGS + WORDS), r.choice(TAGS + WORDS)])
        rng = {}
        ops = r.sample([r.choice(["gte", "gt"]), r.choice(["lte", "lt"])], r.randint(1, 2))
        for o in ops:
            rng[o] = a if o in ("gte", "gt") else b
        if f == "ts" and r.random() < .3:
            rng["format"] = r.choice(["strict_date_optional_time", "epoch_millis", "yyyy"])
        if f == "ts" and r.random() < .2:
            rng["time_zone"] = r.choice(["+01:00", "-05:30", "UTC", "America/New_York"])
        if r.random() < .15:
            rng["boost"] = 2.0
        return {"range": {f: rng}}
    if k == "exists":
        return {"exists": {"field": r.choice(ALL_FIELDS + ["comments", "comments.author", "loc", "title.raw", "ghost*", "t*"])}}
    if k == "ids":
        return {"ids": {"values": [f"d{r.randint(0, 500)}" for _ in range(r.randint(0, 6))]}}
    if k == "match_all":
        return {"match_all": {**({"boost": 1.7} if r.random() < .3 else {})}}
    if k == "match_none":
        return {"match_none": {}}
    if k == "query_string":
        parts = []
        for _ in range(r.randint(1, 3)):
            t = r.choice([word(r), f"{word(r)}*", f"{word(r)}~{r.randint(0,2)}", f'"{sentence(r, 2, 3)}"', f'"{sentence(r, 2, 3)}"~{r.randint(1,3)}',
                          f"{r.choice(KW_FIELDS + TEXT_FIELDS)}:{word(r)}", f"n:[{r.randint(-5,0)} TO {r.randint(1,50)}]", f"n:>{r.randint(0,30)}",
                          f"_exists_:{r.choice(ALL_FIELDS)}", f"/{word(r)[:2]}.*/", f"({word(r)} OR {word(r)})", f"-{word(r)}", f"+{word(r)}",
                          f"{word(r)}^{r.randint(1,5)}", "NOT " + word(r), "title:(a b)", f"ts:[2000 TO 2010]", "tag:*", "?lpha", f"price:[0 TO *]"])
            parts.append(t)
        body = {"query": (" " + r.choice(["AND", "OR", ""]) + " ").join(parts)}
        if r.random() < .5:
            body["default_field"] = r.choice(TEXT_FIELDS + ["tag"])
        if r.random() < .3:
            body["default_operator"] = r.choice(["AND", "OR"])
        if r.random() < .3:
            body["fields"] = r.sample(TEXT_FIELDS + ["tag", "tags"], 2)
            body.pop("default_field", None)
            if r.random() < .5:
                body["type"] = r.choice(["best_fields", "most_fields", "cross_fields", "phrase", "phrase_prefix", "bool_prefix"])
            if r.random() < .3:
                body["tie_breaker"] = 0.3
        if r.random() < .3:
            body["lenient"] = True
        if r.random() < .2:
            body["analyze_wildcard"] = True
        if r.random() < .2:
            body["minimum_should_match"] = r.choice([1, "50%"])
        return {"query_string": body}
    if k == "simple_query_string":
        q = " ".join(r.choice([word(r), f"{word(r)}*", f'"{sentence(r,2,3)}"', f"-{word(r)}", f"+{word(r)}", f"{word(r)}~1", "|", "(a|b)", "foo~2"]) for _ in range(r.randint(1, 4)))
        return {"simple_query_string": {"query": q, "fields": r.sample(TEXT_FIELDS + ["tag"], r.randint(1, 3)), "default_operator": r.choice(["and", "or"]),
                                        **({"flags": r.choice(["ALL", "NONE", "AND|OR|NOT", "PREFIX|PHRASE", "FUZZY|SLOP"])} if r.random() < .3 else {})}}
    if k == "multi_match":
        body = {"query": sentence(r, 1, 3), "fields": r.sample(TEXT_FIELDS + ["tag"], r.randint(1, 3)), "type": r.choice(["best_fields", "most_fields", "cross_fields", "phrase", "phrase_prefix", "bool_prefix"])}
        if r.random() < .3:
            body["tie_breaker"] = r.choice([0, 0.3, 1])
        if r.random() < .3:
            body["operator"] = "and"
        if r.random() < .3:
            body["fields"] = [f + "^" + str(r.randint(1, 4)) for f in body["fields"]]
        if r.random() < .2:
            body["fuzziness"] = "AUTO"
        if r.random() < .2:
            body["minimum_should_match"] = "50%"
        return {"multi_match": body}
    if k == "combined_fields":
        return {"combined_fields": {"query": sentence(r, 1, 3), "fields": ["title"] if r.random() < .5 else ["body"], **({"operator": "and"} if r.random() < .3 else {})}}
    if k == "fuzzy":
        f = r.choice(["tag", "ltag", "tags", "title", "body"])
        return {"fuzzy": {f: {"value": word(r), "fuzziness": r.choice([0, 1, 2, "AUTO"]), **({"prefix_length": r.randint(0, 3)} if r.random() < .3 else {}), **({"transpositions": False} if r.random() < .2 else {}), **({"max_expansions": r.choice([1, 5])} if r.random() < .2 else {})}}}
    if k == "geo_distance":
        return {"geo_distance": {"distance": r.choice(["10km", "500km", "5000km", "100000km", "1m"]), "loc": {"lat": r.uniform(-60, 60), "lon": r.uniform(-170, 170)}, **({"distance_type": r.choice(["arc", "plane"])} if r.random() < .3 else {})}}
    if k == "geo_bbox":
        a, b = sorted([r.uniform(-80, 80), r.uniform(-80, 80)])
        c, d = sorted([r.uniform(-170, 170), r.uniform(-170, 170)])
        if r.random() < .3:
            c, d = d, c
        return {"geo_bounding_box": {"loc": {"top_left": {"lat": b, "lon": c}, "bottom_right": {"lat": a, "lon": d}}}}
    if k == "intervals":
        return {"intervals": {tf: {"match": {"query": sentence(r, 2, 3), "max_gaps": r.choice([0, 1, 3, -1]), "ordered": r.choice([True, False])}}}}
    if k == "term_ci":
        return {"term": {r.choice(["tag", "ltag", "tags"]): {"value": r.choice(TAGS), "case_insensitive": True}}}
    if k == "span_term":
        return {"span_near": {"clauses": [{"span_term": {tf: word(r)}}, {"span_term": {tf: word(r)}}], "slop": r.randint(0, 3), "in_order": r.choice([True, False])}}
    if k == "script":
        return {"script": {"script": {"source": r.choice(["doc['n'].size() > 0 && doc['n'].value > params.x", "doc['price'].size() > 0 && doc['price'].value < params.x", "doc['tag'].size() > 0 && doc['tag'].value.length() > params.x"]), "params": {"x": r.randint(-2, 20)}}}}
    if k == "wrapper_term":
        import base64
        return {"wrapper": {"query": base64.b64encode(json.dumps({"term": {"tag": r.choice(TAGS)}}).encode()).decode()}}
    if k == "nested":
        inner = r.choice([{"term": {"comments.author": r.choice(TAGS[:5])}}, {"range": {"comments.stars": {"gte": r.randint(1, 5)}}},
                          {"match": {"comments.text": word(r)}}, {"exists": {"field": "comments.stars"}},
                          {"bool": {"must": [{"term": {"comments.author": r.choice(TAGS[:5])}}, {"range": {"comments.stars": {"gt": r.randint(1, 4)}}}]}}])
        return {"nested": {"path": "comments", "query": inner, "score_mode": r.choice(["avg", "sum", "max", "min", "none"]), **({"inner_hits": {"size": 3}} if r.random() < .3 else {})}}
    return {"match_all": {}}


def query(r, depth=0):
    if depth >= 3 or r.random() < 0.45:
        return leaf(r)
    k = r.choice(["bool", "bool", "constant_score", "dis_max", "boosting", "function_score", "filter_bool", "script_score", "bool_msm"])
    if k in ("bool", "filter_bool", "bool_msm"):
        b = {}
        for occ in ("must", "should", "filter", "must_not"):
            if r.random() < (0.5 if occ in ("must", "should") else 0.3):
                b[occ] = [query(r, depth + 1) for _ in range(r.randint(1, 3))]
        if not b:
            b["must"] = [query(r, depth + 1)]
        if k == "bool_msm" or r.random() < .2:
            b["minimum_should_match"] = r.choice([0, 1, 2, "50%", "-1", "100%"])
        if r.random() < .1:
            b["boost"] = 2.5
        return {"bool": b}
    if k == "constant_score":
        return {"constant_score": {"filter": query(r, depth + 1), "boost": r.choice([1, 2.5])}}
    if k == "dis_max":
        return {"dis_max": {"queries": [query(r, depth + 1) for _ in range(r.randint(1, 3))], "tie_breaker": r.choice([0, 0.3, 1])}}
    if k == "boosting":
        return {"boosting": {"positive": query(r, depth + 1), "negative": query(r, depth + 1), "negative_boost": r.choice([0.1, 0.5])}}
    if k == "script_score":
        return {"script_score": {"query": query(r, depth + 1), "script": {"source": r.choice(["_score * 2", "Math.log(2 + _score)", "doc['i'].size() > 0 ? Math.abs(doc['i'].value) : 1", "_score + params.p"]), "params": {"p": 1.5}}}}
    fns = [r.choice([{"weight": 2}, {"field_value_factor": {"field": r.choice(["i", "price", "n"]), "missing": 1, "factor": 1.2, "modifier": r.choice(["none", "log1p", "sqrt", "square", "ln2p"])}},
                     {"random_score": {"seed": 5, "field": "_seq_no"}}, {"gauss": {"price": {"origin": 0, "scale": 20}}}, {"exp": {"ts": {"origin": "2000-01-01", "scale": "3650d"}}},
                     {"linear": {"n": {"origin": 0, "scale": 100, "decay": 0.3}}}, {"script_score": {"script": {"source": "doc['i'].size() > 0 ? doc['i'].value * 0.1 : 1"}}}])
           for _ in range(r.randint(1, 3))]
    return {"function_score": {"query": query(r, depth + 1), "functions": fns, "score_mode": r.choice(["multiply", "sum", "avg", "max", "min", "first"]),
                               "boost_mode": r.choice(["multiply", "replace", "sum", "avg", "max", "min"]), **({"min_score": 0.2} if r.random() < .1 else {}), **({"max_boost": 5} if r.random() < .2 else {})}}


# --------------------------------------------------------------------------- aggregations / sorts

def metric(r):
    f = r.choice(NUM_FIELDS + ["ts"])
    t = r.choice(["avg", "sum", "min", "max", "stats", "extended_stats", "value_count", "cardinality", "percentiles", "percentile_ranks", "median_absolute_deviation", "weighted_avg", "geo_bounds", "geo_centroid", "top_hits", "min_str"])
    if t in ("avg", "sum", "min", "max", "stats", "extended_stats", "value_count"):
        spec = {"field": f}
        if r.random() < .2:
            spec["missing"] = r.choice([0, 1])
        if r.random() < .15 and f != "ts":
            spec = {"script": {"source": "doc['%s'].size() > 0 ? doc['%s'].value * 2 : 0" % (f, f)}}
        return {t: spec}
    if t == "cardinality":
        return {"cardinality": {"field": r.choice(KW_FIELDS + NUM_FIELDS + ["ip", "ts", "flag"]), **({"precision_threshold": 100} if r.random() < .3 else {})}}
    if t == "percentiles":
        return {"percentiles": {"field": f, "percents": [1, 25, 50, 90, 99], "tdigest": {"compression": 50}, **({"hdr": {"number_of_significant_value_digits": 2}} if r.random() < .15 and f not in ("price", "f", "hf", "sf") else {})}}
    if t == "percentile_ranks":
        return {"percentile_ranks": {"field": f, "values": [0, 10, 100], "tdigest": {"compression": 100}}}
    if t == "median_absolute_deviation":
        return {"median_absolute_deviation": {"field": f, "compression": 1000}}
    if t == "weighted_avg":
        return {"weighted_avg": {"value": {"field": "price"}, "weight": {"field": "i", "missing": 1}}}
    if t == "geo_bounds":
        return {"geo_bounds": {"field": "loc", "wrap_longitude": r.choice([True, False])}}
    if t == "geo_centroid":
        return {"geo_centroid": {"field": "loc"}}
    if t == "top_hits":
        return {"top_hits": {"size": r.randint(1, 3), "sort": [{r.choice(["n", "price", "i"]): {"order": r.choice(["asc", "desc"]), "missing": "_last"}}, TIE()], "_source": r.choice([True, False, ["tag"], ["n", "title"]])}}
    if t == "min_str":
        return {"max": {"field": "n", "format": "000"}}
    return {"avg": {"field": "n"}}


def bucket(r, depth):
    t = r.choice(["terms", "terms", "histogram", "date_histogram", "range", "date_range", "filter", "filters", "missing", "global_skip", "ip_range", "composite", "nested", "sampler_skip", "rare_terms", "significant_terms", "auto_date_histogram", "geohash_grid", "geotile_grid", "adjacency_matrix", "multi_terms", "significant_text", "reverse_nested_skip", "diversified_skip", "terms_script"])
    sub = {}
    if depth < 2 and r.random() < .5:
        for i in range(r.randint(1, 2)):
            sub[f"s{i}"] = bucket(r, depth + 1) if r.random() < .4 else metric(r)
    out = None
    if t == "terms":
        f = r.choice(KW_FIELDS + ["n", "i", "flag", "ip", "price", "ts", "sh"])
        out = {"terms": {"field": f, "size": r.choice([1, 3, 10, 100]), "order": r.choice([[{"_count": "desc"}, {"_key": "asc"}], [{"_key": "desc"}], [{"_count": "asc"}, {"_key": "asc"}]]),
                         **({"min_doc_count": r.choice([0, 1, 2, 5])} if r.random() < .3 else {}), **({"missing": "N/A" if f in KW_FIELDS else 0} if r.random() < .2 else {}),
                         **({"include": ".*e.*"} if r.random() < .1 and f in KW_FIELDS else {}), **({"exclude": ["red", "blue"]} if r.random() < .1 and f in KW_FIELDS else {}),
                         **({"execution_hint": r.choice(["map", "global_ordinals"])} if r.random() < .15 and f in KW_FIELDS else {}), **({"shard_size": 200} if r.random() < .2 else {}),
                         **({"show_term_doc_count_error": True} if r.random() < .15 else {}), **({"collect_mode": r.choice(["depth_first", "breadth_first"])} if r.random() < .2 else {})}}
    elif t == "terms_script":
        out = {"terms": {"script": {"source": "doc['tag'].size() > 0 ? doc['tag'].value.length() : -1"}, "size": 20, "order": [{"_key": "asc"}]}}
    elif t == "histogram":
        f_ = r.choice(["n", "i", "sf", "sh", "f", "hf"])
        out = {"histogram": {"field": f_, "interval": r.choice([1, 2.5, 10, 100, 1000000]) if f_ in ("sh", "sf", "f", "hf") else r.choice([2**24, 10**9, 2**40]), **({"offset": r.choice([0.5, 1, -1])} if r.random() < .3 else {}), "min_doc_count": r.choice([0, 1, 3]) if f_ in ("sh", "sf", "f", "hf") else r.choice([1, 3]),
                             **({"extended_bounds": {"min": -10, "max": 10}} if r.random() < .2 and f_ in ("sh", "sf", "f", "hf") else {}), **({"hard_bounds": {"min": -50, "max": 50}} if r.random() < .15 else {}), **({"missing": 0} if r.random() < .15 else {}), **({"order": {"_key": "desc"}} if r.random() < .2 else {})}}
    elif t == "date_histogram":
        small = r.random() < .35
        iv = ({"calendar_interval": r.choice(["hour", "minute"])} if r.random() < .5 else {"fixed_interval": r.choice(["1h", "90m", "5000ms"])}) if small else \
            ({"calendar_interval": r.choice(["week", "month", "quarter", "year", "1M"])} if r.random() < .6 else {"fixed_interval": r.choice(["7d", "30d", "365d"])})
        out = {"date_histogram": {"field": "ts", **iv,
                                  **({"time_zone": r.choice(["+01:00", "America/New_York", "Asia/Kolkata", "-08:00", "Europe/London"])} if r.random() < .4 else {}),
                                  **({"offset": r.choice(["+6h", "-1d", "1h"])} if r.random() < .2 else {}), "min_doc_count": r.choice([0, 1, 2]) if not small else r.choice([1, 2]), **({"format": "yyyy-MM-dd HH:mm"} if r.random() < .3 else {}),
                                  **({"order": {"_count": "desc"}} if r.random() < .15 else {}), **({"missing": "2000-01-01"} if r.random() < .15 else {}), **({"keyed": True} if r.random() < .1 else {}), **({"extended_bounds": {"min": "1999-01-01", "max": "2001-01-01"}} if r.random() < .15 and not small else {})}}
    elif t == "auto_date_histogram":
        out = {"auto_date_histogram": {"field": "ts", "buckets": r.choice([1, 5, 10, 50]), **({"minimum_interval": r.choice(["day", "month", "year", "hour"])} if r.random() < .3 else {}), **({"time_zone": "+02:00"} if r.random() < .2 else {})}}
    elif t == "variable_width_histogram":
        out = {"variable_width_histogram": {"field": r.choice(["n", "i", "price"]), "buckets": r.choice([1, 3, 10])}}
    elif t == "range":
        out = {"range": {"field": r.choice(["n", "i", "price", "f", "sh"]), "ranges": [{"to": 0}, {"from": 0, "to": 10}, {"from": 10}, {"key": "k", "from": -5, "to": 5}], **({"keyed": True} if r.random() < .2 else {})}}
    elif t == "date_range":
        out = {"date_range": {"field": "ts", "ranges": [{"to": "2000-01-01"}, {"from": "2000-01-01", "to": "2015-03-01"}, {"from": "2010"}], **({"time_zone": "+05:00"} if r.random() < .2 else {}), **({"format": "yyyy-MM-dd"} if r.random() < .3 else {})}}
    elif t == "ip_range":
        out = {"ip_range": {"field": "ip", "ranges": [{"mask": "10.0.0.0/8"}, {"from": "10.0.0.5", "to": "10.1.0.0"}, {"mask": "::/0"}]}}
    elif t == "filter":
        out = {"filter": leaf(r)}
    elif t == "filters":
        out = {"filters": {"filters": {"a": leaf(r), "b": leaf(r)}, **({"other_bucket_key": "o"} if r.random() < .4 else {})}}
    elif t == "adjacency_matrix":
        out = {"adjacency_matrix": {"filters": {"a": leaf(r), "b": leaf(r), "c": leaf(r)}}}
    elif t == "missing":
        out = {"missing": {"field": r.choice(ALL_FIELDS)}}
    elif t == "composite":
        srcs = []
        for j in range(r.randint(1, 3)):
            kind = r.choice(["terms", "histogram", "date_histogram"])
            if kind == "terms":
                srcs.append({f"c{j}": {"terms": {"field": r.choice(KW_FIELDS + ["n", "i", "flag", "ts"]), **({"order": r.choice(["asc", "desc"])} if r.random() < .5 else {}), **({"missing_bucket": True} if r.random() < .4 else {})}}})
            elif kind == "histogram":
                srcs.append({f"c{j}": {"histogram": {"field": r.choice(["sh", "sf"]), "interval": r.choice([5, 50]), **({"order": r.choice(["asc", "desc"])} if r.random() < .3 else {}), **({"missing_bucket": True} if r.random() < .3 else {})}}})
            else:
                srcs.append({f"c{j}": {"date_histogram": {"field": "ts", "calendar_interval": r.choice(["year", "month"]), **({"missing_bucket": True} if r.random() < .3 else {}), **({"time_zone": "+03:00"} if r.random() < .2 else {})}}})
        out = {"composite": {"sources": srcs, "size": r.choice([2, 5, 100])}}
        sub = {k: v for k, v in sub.items() if True}
    elif t == "nested":
        out = {"nested": {"path": "comments"}}
        sub = {"inner": {"terms": {"field": "comments.author", "size": 10, "order": [{"_count": "desc"}, {"_key": "asc"}]}}, "st": {"avg": {"field": "comments.stars"}}}
    elif t == "rare_terms":
        out = {"rare_terms": {"field": r.choice(["tag", "n", "i"]), "max_doc_count": r.choice([1, 3])}}
    elif t == "significant_terms":
        out = {"significant_terms": {"field": r.choice(["tag", "tags"]), "size": 5, **({"background_filter": {"term": {"flag": True}}} if r.random() < .3 else {}), **({r.choice(["jlh", "mutual_information", "chi_square", "gnd", "percentage"]): {}} if r.random() < .5 else {})}}
    elif t == "significant_text":
        out = {"significant_text": {"field": r.choice(["title", "body"]), "size": 5, "filter_duplicate_text": False}}
        sub = {}
    elif t == "geohash_grid":
        out = {"geohash_grid": {"field": "loc", "precision": r.randint(1, 4)}}
    elif t == "geotile_grid":
        out = {"geotile_grid": {"field": "loc", "precision": r.randint(1, 6)}}
    elif t == "multi_terms":
        out = {"multi_terms": {"terms": [{"field": "tag"}, {"field": r.choice(["i", "flag", "n"])}], "size": 10, "order": [{"_count": "desc"}, {"_key": "asc"}]}}
    else:
        out = {"terms": {"field": "tag", "size": 5, "order": [{"_key": "asc"}]}}
    if sub:
        out["aggs"] = sub
    return out


def aggs(r):
    out = {}
    for i in range(r.randint(1, 3)):
        out[f"a{i}"] = bucket(r, 0) if r.random() < .65 else metric(r)
    if r.random() < .2:
        out["pipe_sum"] = {"sum_bucket": {"buckets_path": "a0>_count"}} if "terms" in json.dumps(out.get("a0", {}))[:20] else {"max": {"field": "n"}}
    if r.random() < .1:
        out["g"] = {"global": {}, "aggs": {"c": {"value_count": {"field": "_id"}}}}
    return out


def sort_spec(r):
    fs = []
    for _ in range(r.randint(1, 3)):
        f = r.choice(["n", "i", "price", "f", "sf", "ts", "tag", "ltag", "tags", "flag", "ip", "title.raw", "_score", "_doc_skip", "sh", "hf", "missing_field"])
        if f == "_doc_skip":
            f = "_score"
        spec = {"order": r.choice(["asc", "desc"])}
        if f not in ("_score",):
            if r.random() < .5:
                spec["missing"] = r.choice(["_first", "_last", 0])
            if f in ("tags", "n", "price") and r.random() < .3:
                spec["mode"] = r.choice(["min", "max", "avg", "sum", "median"] if f != "tags" else ["min", "max"])
            if f == "missing_field":
                spec["unmapped_type"] = "long"
            if f in ("n", "i", "price") and r.random() < .1:
                spec["numeric_type"] = r.choice(["double", "long", "date"])
        fs.append({f: spec})
    if r.random() < .1:
        fs = [{"_geo_distance": {"loc": {"lat": 10, "lon": 20}, "order": "asc", "unit": "km"}}]
    if r.random() < .1:
        fs = [{"_script": {"type": "number", "script": {"source": "doc['n'].size() > 0 ? doc['n'].value % 7 : -1"}, "order": r.choice(["asc", "desc"])}}]
    fs.append(TIE())
    return fs


# --------------------------------------------------------------------------- compare

def close(a, b, rel=1e-4):
    if a == b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9) or abs(a - b) < 1e-9
    return False


IGNORE = {"took", "timed_out", "_shards", "_version", "profile", "took_in_millis", "max_score", "_seq_no", "_primary_term", "doc_count_error_upper_bound", "sum_other_doc_count"}


def diff(a, b, path="", rel=1e-4, ignore=IGNORE):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ignore:
                continue
            if k not in a:
                return f"{path}/{k}: missing on plugin (stock={str(b[k])[:200]})"
            if k not in b:
                return f"{path}/{k}: missing on stock (plugin={str(a[k])[:200]})"
            d = diff(a[k], b[k], f"{path}/{k}", rel, ignore)
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length plugin={len(a)} stock={len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = diff(x, y, f"{path}[{i}]", rel, ignore)
            if d:
                return d
        return None
    if "/values/" in path and path.endswith("_as_string"):
        return None
    if "/values/" in path and close(a, b, 0.3 if isinstance(a, (int, float)) and abs(a) > 1e-3 else 1e9):
        return None   # t-digest estimates depend on the order the values arrive in, and a merge reorders documents
    if close(a, b, rel):
        return None
    return f"{path}: plugin={str(a)[:200]} stock={str(b)[:200]}"


def hits_map(resp):
    out = {}
    for n, h in enumerate(resp.get("hits", {}).get("hits", [])):
        out[h.get("_id", f"#{n}")] = h.get("_score")
    return out


def cmp_search(p, s, ordered):
    if p[0] != s[0] and p[0] != 200 and s[0] != 200 and ("function score query returned" in str(p[1]) + str(s[1])):
        return None
    if p[0] != s[0]:
        return f"status plugin={p[0]} stock={s[0]}: {str(p[1])[:300]} || {str(s[1])[:300]}"
    if p[0] != 200:
        pe = p[1].get("error", {}) if isinstance(p[1], dict) else {}
        se = s[1].get("error", {}) if isinstance(s[1], dict) else {}
        pt = _root_type(pe)
        st = _root_type(se)
        if pt != st:
            if "function score query returned an invalid score" in str(pe) + str(se) or "must not produce negative" in str(pe) + str(se):
                return None  # which document fails first depends on doc order after a merge
            return f"error type plugin={pt} stock={st}: {str(pe)[:300]} || {str(se)[:300]}"
        return None
    pr, sr = p[1], s[1]
    if pr["hits"].get("total") != sr["hits"].get("total"):
        return f"total plugin={pr['hits'].get('total')} stock={sr['hits'].get('total')}"
    pm, sm = hits_map(pr), hits_map(sr)
    nh = len(sr["hits"]["hits"])
    if not ordered and nh and (sr["hits"].get("total") or {"value": nh})["value"] > nh and all(v is not None for v in sm.values()):
        cut = min(sm.values()) * (1 + 1e-4) + 1e-9   # ties at the cut-off may pick different documents
        pm = {k: v for k, v in pm.items() if v is None or v > cut}
        sm = {k: v for k, v in sm.items() if v > cut}
    score_page = ordered and nh and (sr["hits"].get("total") or {"value": nh})["value"] > nh and sr["hits"]["hits"][0].get("sort") \
        and sr["hits"]["hits"][0]["sort"][0] == sr["hits"]["hits"][0].get("_score")
    if set(pm) != set(sm) and not score_page:
        return f"hit ids differ: only plugin={sorted(set(pm) - set(sm))[:6]} only stock={sorted(set(sm) - set(pm))[:6]}"
    for k in pm:
        if k in sm and not close(pm[k], sm[k], 1e-4):
            return f"score of {k}: plugin={pm[k]} stock={sm[k]}"
    if ordered and nh and (sr["hits"].get("total") or {"value": nh})["value"] > nh and all(h.get("_score") is not None for h in sr["hits"]["hits"]) \
            and all(h.get("sort") and h["sort"][0] == h["_score"] for h in sr["hits"]["hits"][:1]):
        # score-sorted page: hits tied with the last one within float error may be swapped for others
        cut = sr["hits"]["hits"][-1]["_score"]
        keep = lambda hs: [h for h in hs if h["_score"] is None or abs(h["_score"] - cut) > 1e-4 * max(abs(cut), 1e-9)]
        pr = {**pr, "hits": {**pr["hits"], "hits": keep(pr["hits"]["hits"])}}
        sr = {**sr, "hits": {**sr["hits"], "hits": keep(sr["hits"]["hits"])}}
        pm, sm = hits_map(pr), hits_map(sr)
        if set(pm) != set(sm):
            return f"hit ids differ: only plugin={sorted(set(pm) - set(sm))[:6]} only stock={sorted(set(sm) - set(pm))[:6]}"
    if ordered:
        po = [h.get("_id") for h in pr["hits"]["hits"]]
        so = [h.get("_id") for h in sr["hits"]["hits"]]
        if po != so:
            return f"order differs: plugin={po[:8]} stock={so[:8]}"
        for hp, hs in zip(pr["hits"]["hits"], sr["hits"]["hits"]):
            d = diff(hp.get("sort"), hs.get("sort"))
            if d:
                return f"sort values of {hp['_id']}: {d}"
    if "aggregations" in pr or "aggregations" in sr:
        d = diff(pr.get("aggregations"), sr.get("aggregations"), "aggs", 1e-6 if False else 1e-4)
        if d:
            return "aggs " + d
    for hp in pr["hits"]["hits"]:
        if "inner_hits" in hp:
            hs = next((x for x in sr["hits"]["hits"] if x["_id"] == hp["_id"]), None)
            if hs is not None:
                d = diff(hp["inner_hits"], hs.get("inner_hits"), "inner_hits")
                if d:
                    return d
    return None


def _resource(resp):
    if resp[0] in (-1, 429):
        return True
    t = json.dumps(resp[1])[:3000] if resp[0] != 200 else ""
    return "circuit_breaking" in t or "too_many_buckets" in t or "OutOfMemory" in t


def _root_type(e):
    if not isinstance(e, dict):
        return str(e)[:80]
    cause = e.get("root_cause") or [e]
    t = cause[0].get("type") if cause else e.get("type")
    return t


def run_case(plugin, stock, index, body, ordered, params=""):
    path = f"/{index}/_search{params}"
    p = plugin.req("POST", path, body)
    s = stock.req("POST", path, body)
    return cmp_search(p, s, ordered), p, s


def plugin_stats(node):
    st, r = node.req("GET", "/_plugins/accelerated/stats")
    return r if st == 200 and isinstance(r, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="http://localhost:9200")
    ap.add_argument("--stock", default="http://localhost:9201")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--merge", type=int, default=1, help="force-merge to one segment (stats then agree)")
    ap.add_argument("--modes", default="query,sort,agg,aggq", help="comma list")
    ap.add_argument("--index", default="fz")
    ap.add_argument("--out", default="fuzz-findings.jsonl")
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--shapes", default="all", choices=["fast", "all"])
    ap.add_argument("--vocab", type=int, default=0, help="synthetic Zipf vocabulary size (0: small fixed word list)")
    ap.add_argument("--keep", action="store_true", help="reuse the existing index")
    a = ap.parse_args()
    plugin, stock = Node(a.plugin), Node(a.stock)
    if a.vocab:
        set_vocab(a.vocab)
    out = open(a.out, "a")
    seen = set()
    errs = {}
    fb = {}
    recent = []
    stats = {"cases": 0, "findings": 0, "both_error": 0}

    def log(rec):
        rec.update({"seed": a.seed, "shards": a.shards, "merge": a.merge})
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(json.dumps(rec)[:900], flush=True)
        stats["findings"] += 1

    if not a.keep:
        ingest([plugin, stock], a.index, a.shards, a.seed, a.docs, a.merge, log, a.shapes)
    r = random.Random(a.seed * 7919 + 1)
    modes = a.modes.split(",")
    for qi in range(a.queries):
        mode = r.choice(modes)
        q = query(r)
        body = {"size": min(a.docs + 50, 500), "query": q, "track_total_hits": True}
        ordered = False
        if mode == "sort":
            body["sort"] = sort_spec(r)
            body["size"] = r.choice([5, 20, 100])
            body["track_scores"] = r.random() < .3
            ordered = True
            if r.random() < .2:
                body["search_after"] = None
                del body["search_after"]
            if r.random() < .3:
                body["from"] = r.randint(0, 10)
        elif mode == "agg":
            body = {"size": 0, "query": r.choice([{"match_all": {}}, q]), "aggs": aggs(r), "track_total_hits": True}
        elif mode == "aggq":
            body["aggs"] = aggs(r)
            body["size"] = 5
            body["sort"] = [{"_score": "desc"}, TIE()]
            ordered = True
        if r.random() < .1 and mode != "agg":
            body["post_filter"] = leaf(r)
        if r.random() < .1:
            body["min_score"] = r.choice([0.1, 1.0])
        if r.random() < .1 and mode in ("query",):
            body["terminate_after"] = r.choice([1, 10])
            ordered = False
        if r.random() < .15 and mode in ("query", "sort"):
            body["_source"] = r.choice([False, ["title", "n"], {"includes": ["t*"], "excludes": ["tags"]}])
            body["docvalue_fields"] = r.sample(["n", "tag", "price", "ts", "flag", "ip", "tags"], 2)
            body["stored_fields"] = "_none_" if r.random() < .0 else "_source"
        if r.random() < .08:
            body["highlight"] = {"fields": {"title": {}, "body": {}}}
        if r.random() < .05 and mode == "query":
            body["rescore"] = {"window_size": 20, "query": {"rescore_query": leaf(r), "query_weight": 0.7, "rescore_query_weight": 1.2}}
        stats["cases"] += 1
        st0 = plugin_stats(plugin)
        d, p, s = run_case(plugin, stock, a.index, body, ordered)
        st1 = plugin_stats(plugin)
        if st0 and st1:
            if st1.get("accelerated_errors", 0) > st0.get("accelerated_errors", 0) or st1.get("script_errors", 0) > st0.get("script_errors", 0):
                if not (p[0] != 200 and s[0] != 200):
                    log({"kind": "native-error", "mode": mode, "body": body})
            for k, v in st1.get("fallbacks", {}).items():
                dv = v - st0.get("fallbacks", {}).get(k, 0)
                if dv:
                    fb[k] = fb.get(k, 0) + 1
            if st1.get("accelerated_queries", 0) > st0.get("accelerated_queries", 0):
                stats["native"] = stats.get("native", 0) + 1
        if _resource(p) or _resource(s):
            stats["resource"] = stats.get("resource", 0) + 1
            if a.errors and stats["resource"] == 1:
                for qj, bj in recent:
                    print("RECENT", qj, json.dumps(bj))
            if a.errors:
                print("RESOURCE", qi, json.dumps(p[1])[:200], "||", json.dumps(s[1])[:200], json.dumps(body)[:300])
            continue
        if a.errors and qi % 25 == 0:
            print("cases", qi, "last bodies kept", flush=True)
        recent.append((qi, body))
        del recent[:-4]
        if "terminate_after" in body and d and d.startswith("hit ids differ"):
            continue
        if p[0] != 200 and s[0] != 200:
            stats["both_error"] += 1
            e = s[1].get("error", {}) if isinstance(s[1], dict) else {}
            k = (e.get("root_cause") or [e])[0].get("reason", str(e))[:90]
            errs[k] = errs.get(k, 0) + 1
        if d is not None and mode in ("query",) and "terminate_after" not in body:
            pass
        if d:
            sig = (mode, d.split(":")[0][:60])
            first = sig not in seen
            seen.add(sig)
            log({"kind": "diff", "mode": mode, "detail": d[:700], "body": body, "first_of_kind": first})
    print(json.dumps(stats))
    print("fallback reasons (cases):", json.dumps(dict(sorted(fb.items(), key=lambda x: -x[1])[:25])))
    if a.errors:
        for k, v in sorted(errs.items(), key=lambda x: -x[1])[:15]:
            print(v, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
