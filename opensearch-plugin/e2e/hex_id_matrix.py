"""The hex_id half of verify_opensearch.py: one index whose documents carry the
same values in a `keyword` field and in a `hex_id` field, and every request
shape run once on each -- term, terms, exists, case-insensitive term, sorts,
search_after, terms/composite/cardinality/significant aggregations, collapse,
docvalue_fields, the fields API and scripts -- which must answer identically:
the same hits, scores, sort values, keys and counts, and natively on as many
shards as the keyword request.

The values are mostly lowercase hex ids (the bytes-stored canonical form) with
every other kind mixed in: uppercase and mixed-case hex, odd length, a
leading `ff` byte, dashed UUIDs, the empty string, non-hex and non-ASCII
text, arrays, nulls and missing fields. Non-canonical values sort after the
canonical ones on hex_id, so rows that order by the value itself filter to
canonical documents (`canon: true`); every other row covers all of them.

Run standalone against a node: python3 hex_id_matrix.py [http://localhost:9200]
"""
import json
import random
import sys
import urllib.error
import urllib.request

INDEX = "hexid"
SHARDS = 2
DOCS = 3000


# (label, index settings): OpenSearch's engine, the Rust engine, and the Rust engine with
# _source derived from doc values (where ignore_above, copy_to and text fields have no place).
VARIANTS = (
    ("opensearch engine", {}),
    ("rust engine", {"index.engine.accelerated": True}),
    ("derived source", {"index.engine.accelerated": True, "index.derived_source.enabled": True}),
)


def mapping(ignore_above=True):
    props = {"canon": {"type": "boolean"}, "single": {"type": "boolean"}, "n": {"type": "long"}}
    for pre, typ in (("k", "keyword"), ("h", "hex_id")):
        props[pre] = {"type": typ, "copy_to": pre + "c"} if ignore_above else {"type": typ}
        props[pre + "c"] = {"type": typ}
        props[pre + "dv"] = {"type": typ, "index": False}
        props[pre + "nv"] = {"type": typ, "null_value": "0badc0de"}
        if ignore_above:
            props[pre + "ia"] = {"type": typ, "ignore_above": 20}
            props[pre + "t"] = {"type": "text", "fields": {"x": {"type": typ}}}
    return {"properties": props}


def hex_id(r, n=16):
    return "".join(r.choice("0123456789abcdef") for _ in range(2 * n))


def value(r, pool):
    """One value and whether it is canonical (lowercase, even-length hex)."""
    k = r.random()
    if k < 0.70:
        return r.choice(pool), True
    if k < 0.74:
        return "ff" + r.choice(pool)[2:], True
    if k < 0.78:
        return r.choice(pool).upper(), False
    if k < 0.81:
        v = r.choice(pool)
        return "".join(c.upper() if r.random() < 0.5 else c for c in v), None
    if k < 0.84:
        return r.choice(pool)[:31], False
    if k < 0.87:
        v = r.choice(pool)
        return f"{v[:8]}-{v[8:12]}-{v[12:16]}-{v[16:20]}-{v[20:]}", False
    if k < 0.89:
        return "", False
    if k < 0.92:
        return r.choice(["req-" + hex_id(r, 4), "not hex", "zz", "g0", "deadbeef!"]), False
    if k < 0.94:
        return r.choice(["é0", "日本", "aé"]), False
    if k < 0.97:
        return hex_id(r, r.choice([1, 2, 4, 8])), True
    return r.choice(pool)[:2], True


def canonical(v):
    return len(v) > 0 and len(v) % 2 == 0 and all(c in "0123456789abcdef" for c in v)


def docs(seed, ignore_above=True):
    r = random.Random(seed)
    pool = [hex_id(r) for _ in range(400)]
    out = []
    for i in range(DOCS):
        d = {"n": i}
        k = r.random()
        if k < 0.03:
            v = None
        elif k < 0.05:
            v = "null"
        elif k < 0.10:
            v = [value(r, pool)[0] for _ in range(r.randint(2, 3))]
        else:
            v = value(r, pool)[0]
        vals = [] if v is None or v == "null" else (v if isinstance(v, list) else [v])
        d["canon"] = bool(vals) and all(canonical(x) for x in vals)
        d["single"] = len(vals) == 1
        for pre in ("k", "h"):
            if v is not None:
                val = None if v == "null" else v
                d[pre] = val
                d[pre + "dv"] = val
                d[pre + "nv"] = val
                if ignore_above:
                    d[pre + "ia"] = val
                    d[pre + "t"] = val
        out.append(d)
    return out, pool


def pool_ff(pool):
    """Values whose first byte is the codec's marker."""
    return ["ff" + v[2:] for v in pool[:5]]


def rows(pool, ignore_above=True):
    """(name, body with field placeholder F, order-sensitive on the value)."""
    a, b, c = pool[0], pool[1], pool[2]
    out = []

    def add(name, body):
        out.append((name, body))

    for v in (a, a.upper(), a[:31], "", "ff00", "zz", "日本", f"{a[:8]}-{a[8:12]}-{a[12:16]}-{a[16:20]}-{a[20:]}", "0badc0de", "absent00"):
        add(f"term {v!r}", {"query": {"term": {"F": v}}, "size": 100, "sort": ["n"]})
    add("term on copy_to target", {"query": {"term": {"Fc": a}}, "size": 100, "sort": ["n"]})
    add("term on text multi-field", {"query": {"term": {"Ft.x": a.upper()}}, "size": 100, "sort": ["n"]})
    add("terms agg on text multi-field", {"size": 0, "aggs": {"t": {"terms": {"field": "Ft.x", "size": 5000}}}})
    ffs = sorted({v for v in pool_ff(pool)})[:3]
    add("terms ff-prefixed", {"query": {"terms": {"F": ffs + ["ff", "ff00", "FF"]}}, "size": 100, "sort": ["n"]})
    add("term on index:false", {"query": {"term": {"Fdv": a}}, "size": 100, "sort": ["n"]})
    add("term null_value", {"query": {"term": {"Fnv": "0badc0de"}}, "size": 100, "sort": ["n"]})
    add("term over ignore_above", {"query": {"term": {"Fia": a}}, "size": 100, "sort": ["n"]})
    add("term scored", {"query": {"term": {"F": b}}, "size": 50})
    add("term boosted in bool", {"query": {"bool": {"should": [{"term": {"F": {"value": a, "boost": 2}}}, {"term": {"F": b}}]}}, "size": 50})
    add("terms mixed", {"query": {"terms": {"F": [a, b.upper(), c[:31], "", "zz", "absent00"]}}, "size": 200, "sort": ["n"]})
    add("terms on index:false", {"query": {"terms": {"Fdv": [a, b, c]}}, "size": 200, "sort": ["n"]})
    add("exists", {"query": {"exists": {"field": "F"}}, "size": 0, "track_total_hits": True})
    add("exists null_value", {"query": {"exists": {"field": "Fnv"}}, "size": 0, "track_total_hits": True})
    add("exists ignore_above", {"query": {"exists": {"field": "Fia"}}, "size": 0, "track_total_hits": True})
    add("must_not exists", {"query": {"bool": {"must_not": {"exists": {"field": "F"}}}}, "size": 200, "sort": ["n"]})
    for v in (a.upper(), a, a.title(), "ZZ", "NOT HEX", ""):
        add(f"term case_insensitive {v!r}", {"query": {"term": {"F": {"value": v, "case_insensitive": True}}}, "size": 200, "sort": ["n"]})
    add("term case_insensitive on index:false", {"query": {"term": {"Fdv": {"value": a.upper(), "case_insensitive": True}}}, "size": 200, "sort": ["n"]})
    add("bool filter term", {"query": {"bool": {"filter": [{"term": {"F": a}}, {"range": {"n": {"gte": 100}}}]}}, "size": 100, "sort": ["n"]})
    add("sort by value asc (canonical)", {"query": {"term": {"canon": True}}, "size": 300, "sort": [{"F": "asc"}, "n"]})
    add("sort by value desc (canonical)", {"query": {"term": {"canon": True}}, "size": 300, "sort": [{"F": {"order": "desc"}}, "n"]})
    add("sort min mode (canonical)", {"query": {"term": {"canon": True}}, "size": 300, "sort": [{"F": {"order": "asc", "mode": "max"}}, "n"]})
    add("sort missing _first (canonical)", {"query": {"bool": {"should": [{"term": {"canon": True}}, {"bool": {"must_not": {"exists": {"field": "F"}}}}]}},
                                            "size": 300, "sort": [{"F": {"order": "asc", "missing": "_first"}}, "n"]})
    add("sort missing value (canonical)", {"query": {"bool": {"should": [{"term": {"canon": True}}, {"bool": {"must_not": {"exists": {"field": "F"}}}}]}},
                                           "size": 400, "sort": [{"F": {"order": "asc", "missing": "deadbeef"}}, "n"]})
    add("sort search_after ff (canonical)", {"query": {"term": {"canon": True}}, "size": 100, "sort": [{"F": "asc"}, "n"], "search_after": ["ff", 0]})
    add("sort search_after (canonical)", {"query": {"term": {"canon": True}}, "size": 100, "sort": [{"F": "asc"}, "n"], "search_after": [b, 0]})
    add("docvalue_fields", {"query": {"range": {"n": {"lt": 400}}}, "size": 400, "sort": ["n"], "docvalue_fields": ["F", "Fdv", "Fnv", "Fia"]})
    add("fields api", {"query": {"range": {"n": {"lt": 400}}}, "size": 400, "sort": ["n"], "fields": ["F", "Fnv", "Fia"], "_source": False})
    # `.value` is a document's first value in term order: single-valued documents only.
    add("script_fields", {"query": {"bool": {"filter": [{"range": {"n": {"lt": 600}}}, {"term": {"single": True}}]}}, "size": 600, "sort": ["n"],
                          "script_fields": {"v": {"script": {"source": "doc['F'].size() == 0 ? null : doc['F'].value"}},
                                            "all": {"script": {"source": "def l = []; for (def x : doc['F']) { l.add(x) } return l"}}}})
    add("terms agg", {"size": 0, "aggs": {"t": {"terms": {"field": "F", "size": 5000}}}})
    add("terms agg top 10 (canonical)", {"size": 0, "query": {"term": {"canon": True}}, "aggs": {"t": {"terms": {"field": "F", "size": 10}}}})
    add("terms agg index:false", {"size": 0, "aggs": {"t": {"terms": {"field": "Fdv", "size": 5000}}}})
    add("terms agg missing", {"size": 0, "aggs": {"t": {"terms": {"field": "F", "size": 5000, "missing": "deadbeef"}}}})
    add("terms agg include", {"size": 0, "aggs": {"t": {"terms": {"field": "F", "size": 100, "include": [a, b, a.upper(), "zz", ""]}}}})
    add("terms agg exclude", {"size": 0, "aggs": {"t": {"terms": {"field": "F", "size": 5000, "exclude": [a, c]}}}})
    add("terms agg sub max", {"size": 0, "aggs": {"t": {"terms": {"field": "F", "size": 5000}, "aggs": {"m": {"max": {"field": "n"}}}}}})
    add("terms agg under filter", {"size": 0, "query": {"range": {"n": {"lt": 1500}}}, "aggs": {"t": {"terms": {"field": "F", "size": 5000}}}})
    add("value_count", {"size": 0, "aggs": {"c": {"value_count": {"field": "F"}}}})
    add("missing agg", {"size": 0, "aggs": {"m": {"missing": {"field": "F"}}}})
    add("composite", {"size": 0, "aggs": {"c": {"composite": {"size": 10000, "sources": [{"v": {"terms": {"field": "F"}}}]}}}})
    add("composite after (canonical)", {"size": 0, "query": {"term": {"canon": True}},
                                        "aggs": {"c": {"composite": {"size": 50, "sources": [{"v": {"terms": {"field": "F"}}}], "after": {"v": b}}}}})
    add("significant_terms", {"size": 0, "query": {"range": {"n": {"lt": 600}}}, "aggs": {"s": {"significant_terms": {"field": "F", "min_doc_count": 2, "size": 5000}}}})
    add("collapse (canonical)", {"query": {"bool": {"filter": [{"term": {"canon": True}}, {"term": {"single": True}}]}},
                                 "size": 200, "sort": ["n"], "collapse": {"field": "F"}})
    add("collapse inner_hits (canonical)", {"query": {"bool": {"filter": [{"term": {"canon": True}}, {"term": {"single": True}}]}},
                                            "size": 50, "sort": ["n"], "collapse": {"field": "F", "inner_hits": {"name": "i", "size": 3, "sort": ["n"]}}})
    if not ignore_above:
        out = [(n, b) for n, b in out if "Fia" not in json.dumps(b) and "Ft.x" not in json.dumps(b) and '"Fc"' not in json.dumps(b)]
    return out


def fields(hit):
    """A hit's `fields`, each list sorted: a document's values come in term order, where
    non-canonical values follow the canonical ones on hex_id."""
    f = hit.get("fields")
    return None if f is None else {k: sorted(json.dumps(x) for x in v) for k, v in f.items()}


def normalise(name, resp):
    """The parts both fields must agree on; bucket order is compared as a mapping where the
    terms aggregation breaks count ties by key, which non-canonical values order differently."""
    hits = resp["hits"]
    out = {
        "total": hits.get("total"),
        "max_score": hits.get("max_score"),
        "hits": [(h["_id"], h.get("_score"), h.get("sort"), fields(h), h.get("_source")) for h in hits["hits"]],
    }
    aggs = resp.get("aggregations")
    if aggs:
        for k, agg in aggs.items():
            if "buckets" in agg:
                agg = dict(agg)
                agg["buckets"] = sorted((json.dumps(b, sort_keys=True) for b in agg["buckets"]))
                if "after_key" in agg and "after" not in name:
                    agg.pop("after_key")
            aggs[k] = agg
        out["aggs"] = aggs
    return out


def rename(body, f):
    s = json.dumps({"_source": False, **body})
    for suffix in ("", "dv", "nv", "ia", "c", "t.x"):
        s = s.replace(f'"F{suffix}"', f'"{f}{suffix}"')
    s = s.replace("doc['F']", f"doc['{f}']")
    return json.loads(s)


def rename_back(resp, f):
    """`resp` with the field's own names put back as the placeholder, so the two compare."""
    s = json.dumps(resp)
    for suffix in ("dv", "nv", "ia", "c", "t.x", ""):
        s = s.replace(f'"{f}{suffix}"', f'"F{suffix}"')
    return json.loads(s)


def norm(v):
    return sorted(v) if isinstance(v, list) else v


def run(req, check, stats):
    """Every variant; returns the number of shard queries the hex_id side ran natively."""
    return sum(run_variant(req, check, stats, label, settings) for label, settings in VARIANTS)


def run_variant(req, check, stats, label, settings):
    derived = settings.get("index.derived_source.enabled", False)
    try:
        req("DELETE", f"/{INDEX}")
    except RuntimeError:
        pass
    req("PUT", f"/{INDEX}", {"settings": {"number_of_shards": SHARDS, "number_of_replicas": 0, "refresh_interval": -1, **settings},
                             "mappings": mapping(not derived)})
    req("GET", f"/_cluster/health/{INDEX}?wait_for_status=green&timeout=60s")
    data, pool = docs(31, not derived)
    for start in range(0, len(data), 1000):
        lines = []
        for i, d in enumerate(data[start:start + 1000], start):
            lines.append(json.dumps({"index": {"_index": INDEX, "_id": str(i)}}))
            lines.append(json.dumps(d))
        r = req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        check(not r.get("errors"), f"hex_id: bulk errors {json.dumps(r)[:500]}")
        # A few segments per shard.
        req("POST", f"/{INDEX}/_refresh")
    native_total = 0
    for name, body in rows(pool, not derived):
        name = f"{label}: {name}"
        got = {}
        ran = {}
        for f in ("k", "h"):
            before = stats()
            try:
                resp = req("POST", f"/{INDEX}/_search?request_cache=false", rename(body, f))
            except RuntimeError as e:
                resp = {"error": str(e)[:300]}
            after = stats()
            ran[f] = after["accelerated_queries"] - before["accelerated_queries"]
            errors = after["accelerated_errors"] - before["accelerated_errors"]
            check(errors == 0, f"hex_id [{name}] {f}: {errors} native errors")
            got[f] = resp
        if "error" in got["k"] or "error" in got["h"]:
            check(False, f"hex_id [{name}]: keyword {got['k'].get('error')} / hex_id {got['h'].get('error')}")
            continue
        shards = got["h"].get("_shards", {})
        check(shards.get("failed", 0) == 0, f"hex_id [{name}]: failed shards {shards}")
        a, b = normalise(name, rename_back(got["k"], "k")), normalise(name, rename_back(got["h"], "h"))
        if a != b:
            check(False, f"hex_id [{name}]: hex_id differs from keyword:\n  keyword {json.dumps(a, sort_keys=True)[:1200]}\n  hex_id  {json.dumps(b, sort_keys=True)[:1200]}")
        else:
            check(True, name)
        check(ran["h"] == ran["k"], f"hex_id [{name}]: native on {ran['h']} shards, keyword on {ran['k']}")
        native_total += ran["h"]
    # Refused queries: as on an ip field, with the reason.
    for q in ({"prefix": {"h": "ab"}}, {"wildcard": {"h": "ab*"}}, {"regexp": {"h": "ab.*"}},
              {"fuzzy": {"h": "abcd"}}, {"range": {"h": {"gte": "00", "lt": "80"}}}):
        try:
            req("POST", f"/{INDEX}/_search", {"query": q})
            check(False, f"hex_id: {json.dumps(q)} should be refused")
        except RuntimeError as e:
            check("does not support" in str(e), f"hex_id: {json.dumps(q)} refused for another reason: {str(e)[:300]}")
    # Every value round-trips through the bytes: the terms aggregation over all docs gives back
    # exactly the strings indexed.
    want = {}
    for d in data:
        v = d.get("h")
        for x in set([] if v is None else (v if isinstance(v, list) else [v])):
            want[x] = want.get(x, 0) + 1
    got = req("POST", f"/{INDEX}/_search", {"size": 0, "aggs": {"t": {"terms": {"field": "h", "size": 10000}}}})
    have = {b["key"]: b["doc_count"] for b in got["aggregations"]["t"]["buckets"]}
    diff = sorted(set(have.items()) ^ set(want.items()))[:10]
    check(have == want, f"hex_id: terms keys differ from the values indexed: {diff}")
    # cardinality hashes the stored bytes, so the estimate is not keyword's to the unit: both
    # must be within the HyperLogLog++ linear-counting error of the true count.
    for f in ("k", "h"):
        got = req("POST", f"/{INDEX}/_search", {"size": 0, "aggs": {"c": {"cardinality": {"field": f, "precision_threshold": 40000}}}})
        est = got["aggregations"]["c"]["value"]
        check(abs(est - len(want)) <= max(2, len(want) // 200), f"hex_id: cardinality on {f} {est}, {len(want)} distinct values")
    if derived:
        # The _source rebuilt from the bytes is the one rebuilt from the keyword's doc values.
        got = req("POST", f"/{INDEX}/_search", {"size": DOCS, "sort": ["n"], "query": {"match_all": {}}})
        bad = []
        for hit in got["hits"]["hits"]:
            src = hit["_source"]
            for suffix in ("", "dv", "nv"):
                kv, hv = src.get("k" + suffix), src.get("h" + suffix)
                if norm(kv) != norm(hv):
                    bad.append((hit["_id"], suffix, kv, hv))
        check(len(got["hits"]["hits"]) == DOCS and not bad, f"hex_id {label}: derived _source differs: {bad[:5]}")
    return native_total


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9200"
    checks, failures = [0], []

    def req(method, path, body=None, ndjson=False):
        data = None
        headers = {}
        if body is not None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
        r = urllib.request.Request(base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:500]}")

    def check(ok, what):
        checks[0] += 1
        if not ok:
            failures.append(what)
            print("FAIL:", what)

    native = run(req, check, lambda: req("GET", "/_plugins/lucene_rust/stats"))
    print(f"hex_id_matrix: {checks[0]} checks, {len(failures)} failures, {native} hex_id shard queries native")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
