"""Shard query-phase time per request shape, native vs Lucene, from the plugin's
query_phase_nanos counters (no HTTP or fetch time in either).

Usage: phase_bench.py ROUNDS INDEX [PATTERN], against a node left running by
`scripts/verify-opensearch.sh --keep` (BASE, default http://localhost:9200).
GEO=1 measures the geo rows (geo_matrix.py) instead, on a geo index
(`geo_matrix.py`'s mapping and loader; `GEO_LOAD=DOCS` loads one first).

Every shape first runs WARMUP times on each path: the plugin's own code is
only as fast as the JIT has made it, and a node that has served a few
thousand requests has not compiled it yet (the query phase of a term query
measured 133 us before and 66 us after, the Lucene path barely moving).
Then six turns alternate the two paths, ROUNDS/6 requests per shape per
turn after three warm-up requests; the median per shape is reported.
Aggregations finish in postProcess, outside the counters, so their Lucene
times are understated: compare those over REST instead -- REST=1 times each
request's round trip (the median of the turn's requests) in place of the
counters."""
import os, sys, statistics, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_opensearch as v
import geo_matrix
v.BASE = os.environ.get("BASE", "http://localhost:9200")
WARMUP = int(os.environ.get("WARMUP", "40"))
REST = os.environ.get("REST") == "1"

rounds = int(sys.argv[1]); index = sys.argv[2]
if os.environ.get("GEO_LOAD"):
    try:
        v.req("DELETE", f"/{index}")
    except RuntimeError:
        pass
    v.req("PUT", f"/{index}", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1},
                               "mappings": geo_matrix.mapping()})
    geo_matrix.create_shapes(v.req)
    geo_matrix.load(v.req, v.check, index, int(os.environ["GEO_LOAD"]), 21)
pat = sys.argv[3] if len(sys.argv) > 3 else ""
# GEO=1: geo_matrix.py's rows (M9 T9.6), against an index loaded as its geo indices are.
rows = geo_matrix.rows() if os.environ.get("GEO") == "1" else v.matrix()
qs = [(n, b) for n, b, e in rows if e in ("native", "slow") and pat in n]


def phase():
    s = v.stats()["query_phase_nanos"]
    return s["native"], s["native_count"], s["lucene"], s["lucene_count"]


for mode in (True, False):
    v.set_native(index, mode)
    for name, body in qs:
        url, b = v.search_url(index, body)
        try:
            for _ in range(WARMUP):
                v.req("POST", url, b)
        except RuntimeError:
            pass  # a row the stock engine cannot answer

res = {}
for rep in range(6):
    for mode in ((True, False) if rep % 2 else (False, True)):
        v.set_native(index, mode)
        for name, body in qs:
            url, b = v.search_url(index, body)
            try:
                for _ in range(3):
                    v.req("POST", url, b)
                n0 = phase()
                walls = []
                for _ in range(rounds // 6):
                    t = time.perf_counter()
                    v.req("POST", url, b)
                    walls.append(time.perf_counter() - t)
                n1 = phase()
            except RuntimeError:
                continue  # a row the stock engine cannot answer
            if REST:
                res.setdefault(name, {}).setdefault(mode, []).append(statistics.median(walls) * 1e6)
                continue
            if mode:
                d, c = n1[0] - n0[0], n1[1] - n0[1]
            else:
                d, c = n1[2] - n0[2], n1[3] - n0[3]
            if c:
                res.setdefault(name, {}).setdefault(mode, []).append(d / c / 1000)
v.set_native(index, True)
ratios = []
for name, d in res.items():
    if True not in d or False not in d:
        print(f"{name:40s} (one path only: {list(d)})")
        continue
    lu = statistics.median(d[False]); na = statistics.median(d[True])
    ratios.append(lu / na)
    print(f"{name:40s} lucene {lu:9.1f} us  native {na:9.1f} us  {lu/na:6.2f}x")
print(f"rows {len(ratios)}  min {min(ratios):.2f}x  median {statistics.median(ratios):.2f}x  below 1.0: {sum(r < 1 for r in ratios)}")
