#!/usr/bin/env python3
"""`POST /_plugins/accelerated/sql` against the same node's search API: every statement's rows must
be what the equivalent query returns, deletions and updates must show, and what SQL does not run
(several shards, a filtered alias, writes, an unknown column or index) must be refused with a clear
status.

    PLUGIN_URL=http://localhost:9200 python3 opensearch-plugin/e2e/verify_sql.py
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchhttp import Client  # noqa: E402

BASE = os.environ.get("PLUGIN_URL", "http://localhost:9200")
C = Client(BASE)
MAP = {"properties": {"k": {"type": "keyword"}, "n": {"type": "long"}, "d": {"type": "date"},
                      "t": {"type": "text"}, "f": {"type": "double"}, "b": {"type": "boolean"}}}
WORDS = ["quick", "brown", "fox", "lazy", "dog", "river", "stone", "cloud", "ember", "north"]
failures = []


def req(method, path, body=None, raw=False):
    try:
        return 200, C.json(method, path, body)
    except RuntimeError as e:
        text = str(e)
        found = re.search(r": (\d{3}) ", text)
        return (int(found.group(1)) if found else 0), text


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
        print(f"FAIL {label} {detail}")
    else:
        print(f"ok   {label}")


def sql(statement, expect=200):
    status, out = req("POST", "/_plugins/accelerated/sql", {"query": statement})
    if expect != status:
        return None, f"{status}: {str(out)[:300]}"
    return out, None


def make(index, shards=1, extra=None):
    req("DELETE", f"/{index}")
    status, out = req("PUT", f"/{index}", {"settings": {"number_of_shards": shards, "number_of_replicas": 0,
                                                       **(extra or {})}, "mappings": MAP})
    assert status == 200, out


def load(index, n, seed=7):
    rnd = random.Random(seed)
    lines = []
    for i in range(n):
        doc = {"k": f"c{rnd.randrange(8)}", "n": i, "d": 1_700_000_000_000 + i * 1000,
               "t": " ".join(rnd.choice(WORDS) for _ in range(6)), "f": rnd.random(), "b": i % 2 == 0}
        lines += [json.dumps({"index": {"_id": f"d{i}"}}), json.dumps(doc)]
    body = ("\n".join(lines) + "\n").encode()
    status, out = req("POST", f"/{index}/_bulk?refresh=true", body, raw=True)
    assert status == 200, out


def search(index, body):
    status, out = req("POST", f"/{index}/_search", body)
    assert status == 200, out
    return out


def main():
    make("sqlt")
    load("sqlt", 3000)
    req("POST", "/sqlt/_forcemerge?max_num_segments=1")
    for i in (5, 11, 2999):
        req("DELETE", f"/sqlt/_doc/d{i}")
    req("POST", "/sqlt/_refresh")

    out, err = sql("SELECT count(*) FROM sqlt")
    check("count(*) sees the deletions", out and out["datarows"] == [[2997]], err or out)
    out, err = sql("SELECT n FROM sqlt WHERE n = 11")
    check("a deleted document is not read", out and out["datarows"] == [], err or out)
    check("schema survives an empty result", out and out["schema"][0] == {"name": "n", "type": "long"}, out)

    out, err = sql("SELECT k, count(*) AS c, sum(n) AS s, min(n) AS lo, max(n) AS hi FROM sqlt GROUP BY k ORDER BY k")
    agg = search("sqlt", {"size": 0, "aggs": {"k": {"terms": {"field": "k", "size": 20, "order": {"_key": "asc"}},
                                                    "aggs": {"s": {"sum": {"field": "n"}}, "lo": {"min": {"field": "n"}},
                                                             "hi": {"max": {"field": "n"}}}}}})
    expect = [[b["key"], b["doc_count"], int(b["s"]["value"]), int(b["lo"]["value"]), int(b["hi"]["value"])]
              for b in agg["aggregations"]["k"]["buckets"]]
    check("group by equals the terms aggregation", out and out["datarows"] == expect, err or (out and out["datarows"][:2]))

    out, err = sql("SELECT n, k FROM sqlt WHERE k = 'c3' ORDER BY n LIMIT 7")
    hits = search("sqlt", {"query": {"term": {"k": "c3"}}, "sort": [{"n": "asc"}], "size": 7, "_source": ["n", "k"]})
    expect = [[h["_source"]["n"], h["_source"]["k"]] for h in hits["hits"]["hits"]]
    check("equality and order by equal the search", out and out["datarows"] == expect, err or out)

    out, err = sql("SELECT count(*) FROM sqlt WHERE n > 1000 AND k = 'c1'")
    count = req("POST", "/sqlt/_count", {"query": {"bool": {"filter": [{"range": {"n": {"gt": 1000}}}, {"term": {"k": "c1"}}]}}})[1]["count"]
    check("a range and a term equal _count", out and out["datarows"] == [[count]], err or out)

    out, err = sql("SELECT n, _score FROM sqlt WHERE match_text(t, 'Quick FOX') ORDER BY _score DESC LIMIT 10")
    hits = search("sqlt", {"query": {"match": {"t": "Quick FOX"}}, "size": 10, "_source": ["n"]})
    got = [r[0] for r in out["datarows"]] if out else None
    expect = [h["_source"]["n"] for h in hits["hits"]["hits"]]
    scores = [r[1] for r in out["datarows"]] if out else []
    best = [h["_score"] for h in hits["hits"]["hits"]]
    check("match_text finds the match query's top hits with its scores", out and got == expect
          and all(abs(a - b) < 1e-4 for a, b in zip(scores, best)), err or (got, expect, scores[:3], best[:3]))

    out, err = sql("SELECT d, n FROM sqlt WHERE n = 42")
    check("a date column reads as epoch milliseconds", out and out["datarows"] == [[1_700_000_042_000, 42]], err or out)

    # updates show after a refresh
    req("PUT", "/sqlt/_doc/d1?refresh=true", {"k": "updated", "n": 1, "d": 1, "t": "x", "f": 0.5, "b": True})
    out, err = sql("SELECT k FROM sqlt WHERE n = 1")
    check("an updated document shows its new value", out and out["datarows"] == [["updated"]], err or out)

    # refusals
    _, err = sql("SELECT nope FROM sqlt", expect=400)
    check("an unknown column is a 400", err is None, err)
    _, err = sql("DELETE FROM sqlt", expect=400)
    check("a write is a 400", err is None, err)
    _, err = sql("DROP TABLE sqlt", expect=400)
    check("DDL is a 400", err is None, err)
    _, err = sql("SELECT * FROM sqlt a JOIN sqlt b ON a.n = b.n", expect=400)
    check("a join is a 400", err is None, err)
    _, err = sql("SELEC", expect=400)
    check("a statement that does not parse is a 400", err is None, err)
    status, out = req("POST", "/_plugins/accelerated/sql", {"nothing": 1})
    check("a request without a query is a 400", status == 400, (status, out))
    status, _ = req("POST", "/_plugins/accelerated/sql?format=csv", {"query": "SELECT 1 FROM sqlt"})
    check("another format is a 400", status == 400, status)
    _, err = sql("SELECT * FROM does_not_exist", expect=404)
    check("an unknown index is a 404", err is None, err)

    make("sqlt_multi", shards=2)
    load("sqlt_multi", 100)
    _, err = sql("SELECT count(*) FROM sqlt_multi", expect=400)
    check("several shards are refused with the reason", err is None, err)

    req("POST", "/_aliases", {"actions": [{"add": {"index": "sqlt", "alias": "sqlt_all"}}]})
    out, err = sql("SELECT count(*) FROM sqlt_all")
    check("an alias resolves to its index", out and out["datarows"] == [[2997]], err or out)
    req("POST", "/_aliases", {"actions": [{"add": {"index": "sqlt", "alias": "sqlt_c1", "filter": {"term": {"k": "c1"}}}}]})
    _, err = sql("SELECT count(*) FROM sqlt_c1", expect=400)
    check("a filtered alias is refused", err is None, err)

    stats = req("GET", "/_plugins/accelerated/stats")[1]
    check("stats count the statements", stats.get("sql_statements", 0) > 5
          and stats.get("sql_router_statements", 0) > 0 and stats.get("sql_standard_statements", 0) > 0, stats)

    if failures:
        print(f"{len(failures)} failed: {failures}")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
