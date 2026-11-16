"""NYC taxis (the Rally nyc_taxis track's mapping and request shapes) on a node started by native-node.sh.

  taxis.py prep SRC.json DIR [DOCS]        split taxi documents (one JSON per line) into bulk-*.ndjson files
  taxis.py load BASE DIR [SETTINGS_JSON]   index `bench`, bulk-load, force-merge to one segment, refresh
  taxis.py time BASE LABEL OUT             date_histogram, auto_date_histogram, sorts, search_after, scroll, ...
  taxis.py report OUT_STOCK OUT_PLUGIN...  as ab.py report

SRC is the head or a sample of https://rally-tracks.elastic.co/nyc_taxis/documents.json.bz2.
Shape filter: SHAPES (substrings). ROUNDS/WARMUP/WARMUP_S as in ab.py; SCROLL_ROUNDS (default 10)."""
import glob
import json
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../../opensearch-plugin/e2e"))
import bench_nodes as bn  # noqa: E402
import ab  # noqa: E402

ROUNDS = int(os.environ.get("ROUNDS", "100"))
WARMUP = int(os.environ.get("WARMUP", "30"))
WARMUP_S = float(os.environ.get("WARMUP_S", "1.5"))
SCROLL_ROUNDS = int(os.environ.get("SCROLL_ROUNDS", "10"))
DATE = {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"}
HALF = {"type": "half_float"}
MAPPING = {"dynamic": "strict", "_source": {"enabled": True}, "properties": {
    "cab_color": {"type": "keyword"}, "dropoff_datetime": DATE, "dropoff_location": {"type": "geo_point"},
    "ehail_fee": {"type": "scaled_float", "scaling_factor": 100}, "extra": HALF, "fare_amount": HALF,
    "improvement_surcharge": HALF, "mta_tax": HALF, "passenger_count": {"type": "integer"},
    "payment_type": {"type": "keyword"}, "pickup_datetime": DATE, "pickup_location": {"type": "geo_point"},
    "rate_code_id": {"type": "keyword"}, "store_and_fwd_flag": {"type": "keyword"}, "surcharge": HALF,
    "tip_amount": HALF, "tolls_amount": HALF, "total_amount": HALF,
    "trip_distance": {"type": "scaled_float", "scaling_factor": 100}, "trip_type": {"type": "keyword"},
    "vendor_id": {"type": "keyword"}, "vendor_name": {"type": "text"}}}
PER_FILE = 20000


def prep(src, out, docs):
    os.makedirs(out, exist_ok=True)
    n = f = 0
    buf = []
    with open(src) as fh:
        for line in fh:
            buf.append('{"index":{}}\n' + line.rstrip("\n"))
            n += 1
            if len(buf) == PER_FILE or n == docs:
                open(os.path.join(out, "bulk-%05d.ndjson" % f), "w").write("\n".join(buf) + "\n")
                f, buf = f + 1, []
            if n == docs:
                break
    if buf:
        open(os.path.join(out, "bulk-%05d.ndjson" % f), "w").write("\n".join(buf) + "\n")
    print(f"prep: {n} documents in {f + 1} files under {out}")


def load(base, corpus, settings):
    bn.BASE = base
    try:
        bn.req("DELETE", "/bench")
    except RuntimeError:
        pass
    st = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    st.update(json.loads(settings))
    bn.req("PUT", "/bench", {"settings": st, "mappings": MAPPING})
    files = sorted(glob.glob(os.path.join(corpus, "bulk-*.ndjson")))
    indexed, _ = bn.load(files)
    bn.req("POST", "/bench/_forcemerge?max_num_segments=1", timeout=3600)
    bn.req("POST", "/bench/_refresh", timeout=3600)
    idx = bn.req("GET", "/_cat/indices/bench?format=json&bytes=b")[0]
    print(f"{bn.req('GET', '/bench/_count')['count']} docs in {indexed:.1f}s, store {int(idx['store.size']) / 2**20:.0f} MiB")


def rng(field, lo, hi):
    return {"range": {field: {"gte": lo, "lt": hi, "format": "yyyy-MM-dd"}}}


YEAR = rng("dropoff_datetime", "2015-01-01", "2016-01-01")
AFTER_PU = {"size": 100, "query": {"match_all": {}}, "sort": [{"pickup_datetime": "asc"}], "search_after": ["2015-03-01 00:00:00"]}


def shapes():
    return [
        ("match_all size 12", {"size": 12, "query": {"match_all": {}}}),
        ("range total_amount", {"size": 0, "query": {"range": {"total_amount": {"gte": 5, "lt": 15}}}}),
        ("distance_amount_agg", {"size": 0, "query": {"bool": {"filter": [{"range": {"trip_distance": {"lt": 50, "gte": 0}}}, rng("pickup_datetime", "2015-01-01", "2015-01-21")]}},
                                 "aggs": {"distance_histo": {"histogram": {"field": "trip_distance", "interval": 1}, "aggs": {"total_amount_stats": {"stats": {"field": "total_amount"}}}}}}),
        ("autohisto_agg", {"size": 0, "query": YEAR, "aggs": {"a": {"auto_date_histogram": {"field": "dropoff_datetime", "buckets": 20}}}}),
        ("autohisto_agg, whole index", {"size": 0, "aggs": {"a": {"auto_date_histogram": {"field": "dropoff_datetime", "buckets": 20}}}}),
        ("date_histogram_agg month", {"size": 0, "query": YEAR, "aggs": {"a": {"date_histogram": {"field": "dropoff_datetime", "calendar_interval": "month"}}}}),
        ("date_histogram_agg day", {"size": 0, "query": YEAR, "aggs": {"a": {"date_histogram": {"field": "dropoff_datetime", "calendar_interval": "day"}}}}),
        ("date_histogram_agg hour, whole index", {"size": 0, "aggs": {"a": {"date_histogram": {"field": "pickup_datetime", "fixed_interval": "1h"}}}}),
        ("date_histogram + avg", {"size": 0, "query": YEAR, "aggs": {"a": {"date_histogram": {"field": "dropoff_datetime", "calendar_interval": "day"}, "aggs": {"t": {"avg": {"field": "total_amount"}}}}}}),
        ("sort pickup asc, match_all", {"size": 10, "query": {"match_all": {}}, "sort": [{"pickup_datetime": "asc"}]}),
        ("sort pickup desc, match_all", {"size": 10, "query": {"match_all": {}}, "sort": [{"pickup_datetime": "desc"}]}),
        ("sort pickup asc, size 100", {"size": 100, "query": {"match_all": {}}, "sort": [{"pickup_datetime": "asc"}]}),
        ("sort pickup asc, range filter", {"size": 10, "query": rng("pickup_datetime", "2015-03-01", "2015-04-01"), "sort": [{"pickup_datetime": "asc"}]}),
        ("sort pickup desc, range filter", {"size": 10, "query": rng("pickup_datetime", "2015-03-01", "2015-04-01"), "sort": [{"pickup_datetime": "desc"}]}),
        ("asc sort + search_after pickup", AFTER_PU),
        ("sort tip_amount desc", {"size": 10, "query": {"match_all": {}}, "sort": [{"tip_amount": "desc"}]}),
        ("sort tip_amount asc", {"size": 10, "query": {"match_all": {}}, "sort": [{"tip_amount": "asc"}]}),
        ("sort tip desc + after", {"size": 10, "query": {"match_all": {}}, "sort": [{"tip_amount": "desc"}, {"pickup_datetime": "asc"}], "search_after": [5.0, "2015-03-01 00:00:00"]}),
        ("terms payment_type", {"size": 0, "aggs": {"a": {"terms": {"field": "payment_type"}}}}),
    ]


def scroll_run(pages, size, sort=None):
    """One scroll through `pages` pages (0 = all) of match_all, ms; clears the scroll."""
    body = {"size": size, "query": {"match_all": {}}}
    if sort:
        body["sort"] = sort
    t0 = time.perf_counter()
    r = bn.req("POST", "/bench/_search?scroll=1m", body)
    sid, got, n = r["_scroll_id"], len(r["hits"]["hits"]), 1
    while r["hits"]["hits"] and (not pages or n < pages):
        r = bn.req("POST", "/_search/scroll", {"scroll": "1m", "scroll_id": sid})
        sid = r["_scroll_id"]
        got += len(r["hits"]["hits"])
        n += 1
    ms = (time.perf_counter() - t0) * 1000
    bn.req("DELETE", "/_search/scroll", {"scroll_id": sid})
    return ms, got


SCROLLS = [("scroll 25 pages x 1000", 25, 1000, None), ("scroll 25 pages x 1000, sort _doc", 25, 1000, ["_doc"]),
           ("scroll all, size 5000", 0, 5000, None), ("scroll 25 pages x 100", 25, 100, None)]


def pstats():
    try:
        return bn.req("GET", "/_plugins/lucene_rust/stats")
    except Exception:
        return None


def time_all(base, label, out):
    bn.BASE = base
    want = [w for w in os.environ.get("SHAPES", "").split(",") if w]
    res = {}
    for name, body in shapes():
        if want and not any(w in name for w in want):
            continue
        last = []

        def one():
            last[:] = [bn.req("POST", "/bench/_search?request_cache=false", body)]

        wall = bn.bh.measure(one, ROUNDS, WARMUP, WARMUP_S)
        s0 = pstats()
        n = 30
        for _ in range(n):
            one()
        s1 = pstats()
        r = last[0]
        res[name] = {**bn.bh.summarize(wall), "hits": r["hits"]["total"]["value"], "took": r.get("took")}
        if s0 and s1:  # the plugin's own time in the query and fetch phases, per request
            res[name]["query_native_ms"] = round((s1["query_phase_nanos"]["accelerated"] - s0["query_phase_nanos"]["accelerated"]) / n / 1e6, 3)
            res[name]["fetch_native_ms"] = round((s1["fetch_nanos"]["accelerated"] - s0["fetch_nanos"]["accelerated"]) / n / 1e6, 3)
    for name, pages, size, sort in SCROLLS:
        if want and not any(w in name for w in want):
            continue
        scroll_run(pages, size, sort)  # warm
        ms = sorted(scroll_run(pages, size, sort)[0] for _ in range(SCROLL_ROUNDS))
        res[name] = {"n": len(ms), "min_ms": ms[0], "median_ms": ms[len(ms) // 2], "p90_ms": ms[-1], "hits": scroll_run(pages, size, sort)[1]}
    res["_stats"] = bn.req("GET", "/_nodes/stats/os,process")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "prep":
        prep(a[2], a[3], int(a[4]) if len(a) > 4 else 10**12)
    elif a[1] == "load":
        load(a[2], a[3], a[4] if len(a) > 4 else "{}")
    elif a[1] == "time":
        time_all(a[2], a[3], a[4])
    elif a[1] == "report":
        ab.report(a[2], a[3:])
    else:
        sys.exit(__doc__)
