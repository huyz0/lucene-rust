#!/usr/bin/env python3
"""Stock OpenSearch SQL and PPL against our SQL endpoint and `_search`, same data and statements.

Needs the plugin node (NODE_PLUGIN, URL PLUGIN_URL) and a stock node with opensearch-sql (NODE_STOCK,
STOCK_URL), both holding index `sqlbench` from bench_sql.py. Each endpoint runs alone on an otherwise
idle node; CPU is the node process's user+system time per request. The `_explain` rows are stock's
statement-to-DSL translation, run without searching.

    python3 opensearch-plugin/e2e/bench_stock_sql.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchhttp as bh  # noqa: E402

PLUGIN_URL = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK_URL = os.environ.get("STOCK_URL", "http://localhost:9201")
NODE_PLUGIN = os.environ.get("NODE_PLUGIN", "n1i")
NODE_STOCK = os.environ.get("NODE_STOCK", "n2")
ROUNDS = int(os.environ.get("ROUNDS", "500"))
HERE = os.path.dirname(os.path.abspath(__file__))
TICK_US = 1e6 / os.sysconf("SC_CLK_TCK")

OURS = "/_plugins/accelerated/sql"
SHAPES = [
    ("point read",
     "SELECT uid, n FROM sqlbench WHERE uid = 'u424242'",
     "source=sqlbench | where uid = 'u424242' | fields uid, n",
     {"query": {"term": {"uid": "u424242"}}, "size": 1, "_source": ["uid", "n"]}),
    ("text top-10",
     "SELECT n, _score FROM sqlbench WHERE match(t, 'w3 w17') ORDER BY _score DESC LIMIT 10",
     "source=sqlbench | where match(t, 'w3 w17') | head 10 | fields n",
     {"query": {"match": {"t": "w3 w17"}}, "size": 10, "_source": ["n"]}),
    ("group by keyword",
     "SELECT k, count(*), sum(n), min(n), max(n) FROM sqlbench GROUP BY k",
     "source=sqlbench | stats count(), sum(n), min(n), max(n) by k",
     {"size": 0, "aggs": {"k": {"terms": {"field": "k", "size": 20}, "aggs": {
         "s": {"sum": {"field": "n"}}, "lo": {"min": {"field": "n"}}, "hi": {"max": {"field": "n"}}}}}}),
    ("filtered group by",
     "SELECT k, count(*) FROM sqlbench WHERE n > 900 GROUP BY k",
     "source=sqlbench | where n > 900 | stats count() by k",
     {"size": 0, "query": {"range": {"n": {"gt": 900}}}, "aggs": {"k": {"terms": {"field": "k", "size": 20}}}}),
    ("range count",
     "SELECT count(*) FROM sqlbench WHERE n > 900 AND k = 'c03'",
     "source=sqlbench | where n > 900 and k = 'c03' | stats count()",
     {"size": 0, "track_total_hits": True,
      "query": {"bool": {"filter": [{"range": {"n": {"gt": 900}}}, {"term": {"k": "c03"}}]}}}),
]


def pid(name):
    return int(subprocess.check_output([sys.executable, os.path.join(HERE, "nodectl.py"), "pid", name]).split()[-1])


def cpu_ticks(p):
    with open(f"/proc/{p}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    return int(parts[11]) + int(parts[12])


def run(client, p, fn):
    ms = bh.measure(fn, rounds=ROUNDS, warmup=200, warmup_s=3)
    # CPU over a second, separate pass so warm-up work is excluded
    t0 = cpu_ticks(p)
    for _ in range(ROUNDS):
        fn()
    cpu = (cpu_ticks(p) - t0) * TICK_US / ROUNDS
    s = bh.summarize(ms)
    return s["median_ms"] * 1000, cpu


def main():
    plugin, stock = bh.Client(PLUGIN_URL), bh.Client(STOCK_URL)
    pp, ps = pid(NODE_PLUGIN), pid(NODE_STOCK)
    rows = []
    for name, sql, ppl, dsl in SHAPES:
        cells = {}
        ours = sql.replace("match(", "match_text(")
        cells["ours sql"] = run(plugin, pp, lambda: plugin.json("POST", OURS, {"query": ours}))
        cells["ours _search"] = run(plugin, pp, lambda: plugin.json("POST", "/sqlbench/_search", dsl))
        cells["stock sql"] = run(stock, ps, lambda: stock.json("POST", "/_plugins/_sql", {"query": sql}))
        cells["stock ppl"] = run(stock, ps, lambda: stock.json("POST", "/_plugins/_ppl", {"query": ppl}))
        cells["stock _search"] = run(stock, ps, lambda: stock.json("POST", "/sqlbench/_search", dsl))
        cells["stock sql explain"] = run(stock, ps, lambda: stock.json("POST", "/_plugins/_sql/_explain", {"query": sql}))
        cells["stock ppl explain"] = run(stock, ps, lambda: stock.json("POST", "/_plugins/_ppl/_explain", {"query": ppl}))
        rows.append((name, cells))
        print(f"\n{name}   (p50 us / cpu us per request)", flush=True)
        for k, (p50, cpu) in cells.items():
            print(f"  {k:20s} {p50:9.0f} {cpu:9.0f}", flush=True)
    print("\nsummary p50 us")
    keys = list(rows[0][1])
    print(f"{'shape':20s}" + "".join(f"{k:>18s}" for k in keys))
    for name, cells in rows:
        print(f"{name:20s}" + "".join(f"{cells[k][0]:18.0f}" for k in keys))
    print("\nsummary cpu us per request")
    for name, cells in rows:
        print(f"{name:20s}" + "".join(f"{cells[k][1]:18.0f}" for k in keys))


if __name__ == "__main__":
    main()
