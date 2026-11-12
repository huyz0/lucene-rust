"""The rust_wildcard half of verify_opensearch.py: one index whose documents carry
the same value in a `wildcard` field (`w`) and in a `rust_wildcard` field (`s`),
and every query shape the wildcard field takes run once on each -- term, terms,
prefix, wildcard and regexp (case sensitive and not), match, exists, bools of
them, a filter aggregation over one, and count -- which must match the same
documents. Under the Rust engine every rust_wildcard query runs natively on
every shard (the wildcard field's run on Lucene); under OpenSearch's engine the
rust_wildcard field is plain BINARY doc values and must answer the same.

The values are log-like lines with every character class the patterns care
about: upper and lower case, digits, `* ? \\ / [ ] . |`, non-ASCII (`İ`, the
Kelvin sign, `Ü`), the empty string, and missing fields.

Run standalone against a node: python3 rust_wildcard_matrix.py [http://localhost:9200]
"""
import json
import random
import sys
import urllib.error
import urllib.request

INDEX = "rustwildcard"
SHARDS = 2
DOCS = 4000

VARIANTS = (
    ("opensearch engine", {}),
    ("rust engine", {"index.lucene_rust.engine": True}),
)

WORDS = ("Connection timeout java.net.SocketTimeoutException: read timed out ERROR error WARN disk full "
         "blk_-16089 10.250.19.102 postdrop İstanbul Kelvin parity corrected a*b what? back\\slash "
         "/path/ [x] a|b ab Ünïcødé sshd[2420] Failed password for root").split(" ")


def docs(seed):
    r = random.Random(seed)
    out = []
    for i in range(DOCS):
        d = {"n": i}
        if i % 13 == 3:
            out.append(d)
            continue
        if i % 41 == 7:
            v = ""
        else:
            v = " ".join([str(i % 97)] + [r.choice(WORDS) for _ in range(r.randrange(8))])
        d["w"] = v
        d["s"] = v
        out.append(d)
    return out


# How a row's two answers must relate. "equal": same hits, totals and filter counts. "ids": same
# hits; the wildcard field's reported total and filter count can leave out the documents whose
# value is "" (its count shortcut reads the trigram postings, which have no entry for ""), so the
# rust_wildcard side's total must equal its hit count instead. "superset": the wildcard field's
# trigram prefilter misses documents its own test accepts (a case-insensitive pattern with
# non-ASCII letters, or "" in terms), so rust_wildcard must find every one of its hits and more.
EQUAL, IDS, SUPERSET = "equal", "ids", "superset"


def rows():
    """(name, query on field F, whether the rust_wildcard side runs natively, relation)."""
    out = []

    def add(name, q, native=True, relation=EQUAL):
        out.append((name, q, native, relation))

    for ci in (False, True):
        for p in ("*timeout*", "*TIMEOUT*", "1? *", "*corrected", "*a\\*b*", "*what\\?*", "*back\\\\slash*", "*", "?", "??",
                  "*istanbul*", "*kelvin*", "*conn*time*", "4? *out*", "*/path/*", "*[x]*", "*a|b*", "5", "ünï*", "*Ü*",
                  "*sshd[2420]*", "*10.250.19.102*"):
            relation = IDS if p == "*" else SUPERSET if ci and p in ("*kelvin*", "*Ü*") else EQUAL
            add(f"wildcard {p} ci={ci}", {"wildcard": {"F": {"value": p, "case_insensitive": ci}}}, relation=relation)
        for p in ("1", "Connection", "conn", "9 ERROR"):
            add(f"prefix {p} ci={ci}", {"prefix": {"F": {"value": p, "case_insensitive": ci}}})
    for t in ("5", "5 timeout", "", "a*b", "1 ERROR", "12 error"):
        add(f"term {t!r}", {"term": {"F": t}})
        add(f"term ci {t!r}", {"term": {"F": {"value": t, "case_insensitive": True}}})
    add("terms", {"terms": {"F": ["5", "6", "7 error", "5"]}})
    add("terms with empty", {"terms": {"F": ["5", "6", "", "7 error", "5"]}}, relation=SUPERSET)
    for re in (".*[Tt]ime(d|out).*", "1[0-9] .*error.*", ".*blk_-?[0-9]+.*", "5", ".*", ".*a\\|b.*", "[0-9]+ (WARN|ERROR).*",
               ".*~(x).*", ".*<1-20> .*", ".*failed PASSWORD.*"):
        relation = IDS if re == ".*" else EQUAL
        add(f"regexp {re}", {"regexp": {"F": re}}, relation=relation)
        add(f"regexp ci {re}", {"regexp": {"F": {"value": re, "case_insensitive": True}}}, relation=relation)
    add("regexp flags NONE", {"regexp": {"F": {"value": ".*time.*", "flags": "NONE"}}}, native=False)
    add("match", {"match": {"F": "5 timeout"}})
    add("exists", {"exists": {"field": "F"}}, relation=IDS)
    add("bool", {"bool": {"must": [{"wildcard": {"F": {"value": "*error*", "case_insensitive": True}}}],
                          "must_not": [{"wildcard": {"F": "*timeout*"}}]}})
    add("bool filter", {"bool": {"filter": [{"wildcard": {"F": "*timed*"}}, {"range": {"n": {"gte": 1000}}}]}})
    return out


def rename(q, f):
    return json.loads(json.dumps(q).replace('"F"', json.dumps(f)))


def run(req, check, stats):
    """Every variant; returns the number of shard queries the rust_wildcard side ran natively."""
    return sum(run_variant(req, check, stats, label, settings) for label, settings in VARIANTS)


def run_variant(req, check, stats, label, settings):
    rust = settings.get("index.lucene_rust.engine", False)
    try:
        req("DELETE", f"/{INDEX}")
    except RuntimeError:
        pass
    req("PUT", f"/{INDEX}", {
        "settings": {"number_of_shards": SHARDS, "number_of_replicas": 0, "refresh_interval": -1, **settings},
        "mappings": {"properties": {"n": {"type": "long"}, "w": {"type": "wildcard"}, "s": {"type": "rust_wildcard"}}}})
    req("GET", f"/_cluster/health/{INDEX}?wait_for_status=green&timeout=60s")
    data = docs(43)
    for start in range(0, len(data), 1000):
        lines = []
        for i, d in enumerate(data[start:start + 1000], start):
            lines.append(json.dumps({"index": {"_index": INDEX, "_id": str(i)}}))
            lines.append(json.dumps(d))
        r = req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        check(not r.get("errors"), f"rust_wildcard: bulk errors {json.dumps(r)[:500]}")
        # A few segments per shard, then a merge of some of them below.
        req("POST", f"/{INDEX}/_refresh")
    native_total = 0
    for phase in ("segments", "merged"):
        if phase == "merged":
            req("POST", f"/{INDEX}/_forcemerge?max_num_segments=1")
            req("POST", f"/{INDEX}/_refresh")
        for name, q, native, relation in rows():
            name = f"{label} {phase}: {name}"
            got = {}
            ran = {}
            for f in ("w", "s"):
                body = {"size": DOCS, "track_total_hits": True, "_source": False, "sort": ["n"],
                        "query": rename(q, f), "aggs": {"f": {"filter": rename(q, f)}}}
                before = stats()
                try:
                    resp = req("POST", f"/{INDEX}/_search?request_cache=false", body)
                except RuntimeError as e:
                    resp = {"error": str(e)[:300]}
                after = stats()
                ran[f] = after["native_queries"] - before["native_queries"]
                errors = after["native_errors"] - before["native_errors"]
                check(errors == 0, f"rust_wildcard [{name}] {f}: {errors} native errors")
                got[f] = resp
            if "error" in got["w"] or "error" in got["s"]:
                check(False, f"rust_wildcard [{name}]: wildcard {got['w'].get('error')} / rust_wildcard {got['s'].get('error')}")
                continue
            a = ([h["_id"] for h in got["w"]["hits"]["hits"]], got["w"]["hits"]["total"], got["w"]["aggregations"]["f"]["doc_count"])
            b = ([h["_id"] for h in got["s"]["hits"]["hits"]], got["s"]["hits"]["total"], got["s"]["aggregations"]["f"]["doc_count"])
            what = (f"rust_wildcard [{name}]: {a[1]} / {a[2]} hits on wildcard, {b[1]} / {b[2]} on rust_wildcard; "
                    f"first differing ids {sorted(set(a[0]) ^ set(b[0]))[:10]}")
            if relation == EQUAL:
                check(a == b, what)
            elif relation == IDS:
                check(a[0] == b[0], what)
            else:
                check(set(a[0]) < set(b[0]), what)
            if relation != EQUAL:
                check(b[1]["value"] == len(b[0]) == b[2], f"{what}; rust_wildcard's total and filter count are not its hit count")
            if rust and native:
                check(ran["s"] >= SHARDS, f"rust_wildcard [{name}]: native on {ran['s']} shards, wanted {SHARDS}")
            native_total += ran["s"]
            c = req("POST", f"/{INDEX}/_count", {"query": rename(q, "s")})
            check(c["count"] == b[1]["value"], f"rust_wildcard [{name}]: _count {c['count']}, search {b[1]['value']}")
    # Refusals: an array value, sorting and aggregating on the field, fuzzy and range.
    try:
        r = req("POST", f"/{INDEX}/_doc/x?refresh=true", {"s": ["a", "b"]})
        check(False, f"rust_wildcard: an array value should be refused: {r}")
    except RuntimeError as e:
        check("one value per document" in str(e), f"rust_wildcard: array refused for another reason: {str(e)[:300]}")
    for body in ({"sort": ["s"]}, {"aggs": {"t": {"terms": {"field": "s"}}}}, {"query": {"fuzzy": {"s": "abcd"}}},
                 {"query": {"range": {"s": {"gte": "a"}}}}):
        try:
            req("POST", f"/{INDEX}/_search", body)
            check(False, f"rust_wildcard: {json.dumps(body)} should be refused")
        except RuntimeError:
            check(True, "refused")
    # The fields API reads the value from _source, as the wildcard field's would without doc values.
    got = req("POST", f"/{INDEX}/_search", {"size": 50, "sort": ["n"], "_source": False, "fields": ["w", "s"]})
    same = all(h.get("fields", {}).get("w") == h.get("fields", {}).get("s") for h in got["hits"]["hits"])
    check(same, f"rust_wildcard {label}: the fields API differs")
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
    print(f"rust_wildcard_matrix: {checks[0]} checks, {len(failures)} failures, {native} rust_wildcard shard queries native")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
