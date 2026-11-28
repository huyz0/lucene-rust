#!/usr/bin/env python3
"""SQL over REST against the same shapes through the search API, on one node, one shard.

Loads a synthetic index (keyword `uid` unique, keyword `k` of 12 values, long `n`, text `t` of Zipf
words), then times each shape on both endpoints with one persistent connection (benchhttp), reporting
p50/p90/p99 in microseconds and the ratio. `took` of the plugin's counters gives the native share:
the rest of a SQL request's time is REST parsing, the transport action and the response.

    PLUGIN_URL=http://localhost:9200 DOCS=1000000 python3 opensearch-plugin/e2e/bench_sql.py
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchhttp as bh  # noqa: E402

BASE = os.environ.get("PLUGIN_URL", "http://localhost:9200")
DOCS = int(os.environ.get("DOCS", "1000000"))
ROUNDS = int(os.environ.get("ROUNDS", "600"))
INDEX = os.environ.get("INDEX", "sqlbench")
C = bh.Client(BASE)


def load():
    if os.environ.get("REUSE") and C.json("GET", f"/{INDEX}/_count")["count"] == DOCS:
        return
    try:
        C.json("DELETE", f"/{INDEX}")
    except RuntimeError:
        pass
    C.json("PUT", f"/{INDEX}", {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"properties": {"uid": {"type": "keyword"}, "k": {"type": "keyword"},
                                    "n": {"type": "long"}, "t": {"type": "text"}}}})
    rnd = random.Random(11)
    batch = 20000
    t0 = time.time()
    for start in range(0, DOCS, batch):
        lines = []
        for i in range(start, min(start + batch, DOCS)):
            words = " ".join(f"w{int(400 ** rnd.random())}" for _ in range(12))
            lines.append('{"index":{"_id":"%d"}}' % i)
            lines.append(json.dumps({"uid": f"u{i}", "k": f"c{i % 12:02d}", "n": rnd.randrange(1000), "t": words}))
        C.json("POST", f"/{INDEX}/_bulk", ("\n".join(lines) + "\n").encode(), ndjson=True)
        if (start // batch) % 12 == 11:
            C.json("POST", f"/{INDEX}/_refresh")
    C.json("POST", f"/{INDEX}/_refresh")
    C.json("POST", f"/{INDEX}/_forcemerge?max_num_segments=4")
    print(f"loaded {DOCS} docs in {time.time() - t0:.0f}s", flush=True)


def sql(statement):
    return lambda: C.json("POST", "/_plugins/accelerated/sql", {"query": statement})


def dsl(body):
    return lambda: C.json("POST", f"/{INDEX}/_search", body)


SHAPES = [
    ("point read", sql(f"SELECT uid, n FROM {INDEX} WHERE uid = 'u424242'"),
     dsl({"query": {"term": {"uid": "u424242"}}, "size": 1, "_source": ["uid", "n"]})),
    ("text top-10", sql(f"SELECT n, _score FROM {INDEX} WHERE match_text(t, 'w3 w17') ORDER BY _score DESC LIMIT 10"),
     dsl({"query": {"match": {"t": "w3 w17"}}, "size": 10, "_source": ["n"]})),
    ("text rare top-10", sql(f"SELECT n, _score FROM {INDEX} WHERE match_text(t, 'w390') ORDER BY _score DESC LIMIT 10"),
     dsl({"query": {"match": {"t": "w390"}}, "size": 10, "_source": ["n"]})),
    ("group by keyword", sql(f"SELECT k, count(*), sum(n), min(n), max(n) FROM {INDEX} GROUP BY k"),
     dsl({"size": 0, "aggs": {"k": {"terms": {"field": "k", "size": 20}, "aggs": {
         "s": {"sum": {"field": "n"}}, "lo": {"min": {"field": "n"}}, "hi": {"max": {"field": "n"}}}}}})),
    ("filtered group by (DataFusion)", sql(f"SELECT k, count(*) FROM {INDEX} WHERE n > 900 GROUP BY k"),
     dsl({"size": 0, "query": {"range": {"n": {"gt": 900}}}, "aggs": {"k": {"terms": {"field": "k", "size": 20}}}})),
    ("range count (DataFusion)", sql(f"SELECT count(*) FROM {INDEX} WHERE n > 900 AND k = 'c03'"),
     dsl({"size": 0, "track_total_hits": True, "query": {"bool": {"filter": [{"range": {"n": {"gt": 900}}}, {"term": {"k": "c03"}}]}}})),
]


def main():
    load()
    print(f"{'shape':34s} {'sql p50':>9s} {'p90':>8s} {'p99':>8s} | {'dsl p50':>9s} {'p90':>8s} {'p99':>8s} | sql/dsl")
    for name, s, d in SHAPES:
        a = bh.summarize(bh.measure(s, rounds=ROUNDS, warmup=100, warmup_s=3))
        b = bh.summarize(bh.measure(d, rounds=ROUNDS, warmup=100, warmup_s=3))
        print(f"{name:34s} {a['median_ms'] * 1000:9.0f} {a['p90_ms'] * 1000:8.0f} {a['p99_ms'] * 1000:8.0f} | "
              f"{b['median_ms'] * 1000:9.0f} {b['p90_ms'] * 1000:8.0f} {b['p99_ms'] * 1000:8.0f} | {a['median_ms'] / b['median_ms']:.2f}x", flush=True)
    stats = C.json("GET", "/_plugins/accelerated/stats")
    print("sql statements", stats["sql_statements"], "router", stats["sql_router_statements"], "standard", stats["sql_standard_statements"],
          "native sql ms", stats.get("sql_time_in_nanos", 0) / 1e6)


if __name__ == "__main__":
    main()
