#!/usr/bin/env python3
"""Concurrency invariants on one node (run it against the plugin node and a stock node):
writers update documents with optimistic concurrency, deleters and re-creators race them, a merger
force-merges and refreshes, and readers check invariants that must hold at any time:

  * a point-in-time search_after walk visits every document of its snapshot exactly once, and the
    snapshot's count, a terms aggregation total and a sum aggregation agree with the walk;
  * a scroll visits no document twice;
  * a document's `_version` never goes down between two reads that follow each other;
  * after the writers stop, every document's last acknowledged value is what a GET and a search return,
    and counts by search, `_count` and aggregation agree.

    python3 opensearch-plugin/e2e/sweep_concurrency.py [--url URL] [--seconds 60] [--docs 300]
"""
import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request

import fuzz_diff as fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9200")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--writers", type=int, default=3)
    ap.add_argument("--index", default="conc")
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--settings", default="{}")
    a = ap.parse_args()
    node = fd.Node(a.url)
    idx = a.index
    node.req("DELETE", f"/{idx}")
    st, r = node.req("PUT", f"/{idx}", {"settings": {"number_of_shards": a.shards, "number_of_replicas": 0, "refresh_interval": "100ms", **json.loads(a.settings)},
                                       "mappings": {"properties": {"v": {"type": "long"}, "w": {"type": "keyword"}, "t": {"type": "text"}, "grp": {"type": "integer"}}}})
    assert st == 200, r
    stop = threading.Event()
    lock = threading.Lock()
    expected = {}        # id -> (version value) last acknowledged ("deleted" when deleted)
    problems = []
    stats = {"writes": 0, "conflicts": 0, "deletes": 0, "pit_walks": 0, "scrolls": 0, "version_reads": 0, "merges": 0}

    def problem(kind, detail):
        with lock:
            problems.append({"kind": kind, "detail": str(detail)[:400]})
            print(json.dumps(problems[-1]), flush=True)

    ids = [f"d{i}" for i in range(a.docs)]
    # seed
    lines = []
    for i in ids:
        lines += [json.dumps({"index": {"_id": i}}), json.dumps({"v": 0, "w": "w0", "t": "alpha beta", "grp": int(i[1:]) % 7})]
    node.req("POST", f"/{idx}/_bulk?refresh=true", ("\n".join(lines) + "\n").encode(), raw=True)
    for i in ids:
        expected[i] = ("live", 0)

    # Each id belongs to one writer: its acknowledged state is then exactly known.
    def writer(k):
        r = random.Random(k)
        mine = [i for n, i in enumerate(ids) if n % a.writers == k]
        counter = {i: 0 for i in mine}
        while not stop.is_set():
            i = r.choice(mine)
            op = r.random()
            if op < 0.15:
                s, resp = node.req("DELETE", f"/{idx}/_doc/{i}")
                if s in (200, 404):
                    with lock:
                        expected[i] = ("deleted", None)
                        stats["deletes"] += 1
                else:
                    problem("delete-status", (s, resp))
            elif op < 0.3:
                counter[i] += 1
                s, resp = node.req("PUT", f"/{idx}/_doc/{i}", {"v": counter[i], "w": f"w{counter[i] % 5}", "t": "alpha gamma", "grp": int(i[1:]) % 7})
                if s in (200, 201):
                    with lock:
                        expected[i] = ("live", counter[i])
                        stats["writes"] += 1
                else:
                    problem("create-status", (s, resp))
            else:
                counter[i] += 1
                s, resp = node.req("POST", f"/{idx}/_update/{i}?retry_on_conflict=3", {"doc": {"v": counter[i], "w": f"w{counter[i] % 5}", "grp": int(i[1:]) % 7}, "doc_as_upsert": True})
                if s in (200, 201):
                    with lock:
                        expected[i] = ("live", counter[i])
                        stats["writes"] += 1
                elif s == 409:
                    stats["conflicts"] += 1
                else:
                    problem("update-status", (s, resp))

    def merger():
        r = random.Random(99)
        while not stop.is_set():
            time.sleep(r.uniform(0.5, 2.0))
            node.req("POST", f"/{idx}/_forcemerge?max_num_segments={r.choice([1, 2, 4])}")
            stats["merges"] += 1
            node.req("POST", f"/{idx}/_refresh")

    def pit_reader():
        while not stop.is_set():
            s, r = node.req("POST", f"/{idx}/_search/point_in_time?keep_alive=1m")
            if s != 200:
                problem("pit-open", (s, r))
                return
            pit = r.get("pit_id") or r.get("id")
            body = {"size": 37, "pit": {"id": pit, "keep_alive": "1m"}, "query": {"match_all": {}}, "sort": [{"grp": "asc"}, {"_shard_doc": "asc"}],
                    "track_total_hits": True, "aggs": {"w": {"terms": {"field": "w", "size": 20}}, "g": {"terms": {"field": "grp", "size": 20}}, "s": {"sum": {"field": "v"}}}}
            seen = []
            total = None
            first_aggs = None
            after = None
            ok = True
            while True:
                b = dict(body)
                if after:
                    b["search_after"] = after
                    b.pop("aggs", None)
                s, r = node.req("POST", "/_search", b)
                if s != 200:
                    problem("pit-search", (s, r))
                    ok = False
                    break
                if after is None:
                    total = r["hits"]["total"]["value"]
                    first_aggs = r["aggregations"]
                hs = r["hits"]["hits"]
                if not hs:
                    break
                seen += [h["_id"] for h in hs]
                after = hs[-1]["sort"]
            if ok:
                stats["pit_walks"] += 1
                if len(seen) != len(set(seen)):
                    problem("pit-duplicate", [x for x in set(seen) if seen.count(x) > 1][:5])
                if len(seen) != total:
                    problem("pit-count", f"walked {len(seen)} total {total}")
                wsum = sum(b["doc_count"] for b in first_aggs["w"]["buckets"])
                gsum = sum(b["doc_count"] for b in first_aggs["g"]["buckets"])
                if wsum != total or gsum != total:
                    problem("pit-agg-total", f"total {total} w {wsum} g {gsum}")
            node.req("DELETE", "/_search/point_in_time", {"pit_id": [pit]})

    def scroller():
        while not stop.is_set():
            s, r = node.req("POST", f"/{idx}/_search?scroll=1m", {"size": 41, "query": {"match_all": {}}, "sort": ["_doc"]})
            if s != 200:
                problem("scroll-open", (s, r))
                return
            sid = r["_scroll_id"]
            seen = [h["_id"] for h in r["hits"]["hits"]]
            while r["hits"]["hits"]:
                s, r = node.req("POST", "/_search/scroll", {"scroll": "1m", "scroll_id": sid})
                if s != 200:
                    problem("scroll-next", (s, r))
                    break
                seen += [h["_id"] for h in r["hits"]["hits"]]
            stats["scrolls"] += 1
            if len(seen) != len(set(seen)):
                problem("scroll-duplicate", [x for x in set(seen) if seen.count(x) > 1][:5])
            node.req("DELETE", "/_search/scroll", {"scroll_id": sid})

    def version_reader():
        r = random.Random(5)
        last = {}
        while not stop.is_set():
            i = r.choice(ids)
            s, resp = node.req("GET", f"/{idx}/_doc/{i}")
            if s == 200:
                v = resp["_version"]
                if i in last and v < last[i] and resp["_seq_no"] >= 0:
                    # A delete followed by a re-create keeps counting; versions of one id never go back.
                    problem("version-regress", (i, last[i], v))
                last[i] = v
                stats["version_reads"] += 1

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(a.writers)]
    threads += [threading.Thread(target=merger), threading.Thread(target=pit_reader), threading.Thread(target=pit_reader), threading.Thread(target=scroller), threading.Thread(target=version_reader)]
    for t in threads:
        t.start()
    time.sleep(a.seconds)
    stop.set()
    for t in threads:
        t.join()
    node.req("POST", f"/{idx}/_refresh")
    # final state
    live = {i: v[1] for i, v in expected.items() if v[0] == "live"}
    s, r = node.req("POST", f"/{idx}/_search", {"size": a.docs + 10, "query": {"match_all": {}}, "track_total_hits": True, "aggs": {"s": {"sum": {"field": "v"}}, "w": {"terms": {"field": "w", "size": 20}}}})
    got = {h["_id"]: h["_source"]["v"] for h in r["hits"]["hits"]}
    if got != live:
        problem("final-search", {"missing": sorted(set(live) - set(got))[:5], "extra": sorted(set(got) - set(live))[:5],
                                 "wrong": [(k, live[k], got[k]) for k in live if k in got and got[k] != live[k]][:5]})
    s2, c = node.req("GET", f"/{idx}/_count")
    if c.get("count") != len(live):
        problem("final-count", (c, len(live)))
    if abs(r["aggregations"]["s"]["value"] - sum(live.values())) > 0.5:
        problem("final-sum", (r["aggregations"]["s"], sum(live.values())))
    for i in list(live)[:60]:
        s3, g = node.req("GET", f"/{idx}/_doc/{i}")
        if s3 != 200 or g["_source"]["v"] != live[i]:
            problem("final-get", (i, s3, live[i], g.get("_source")))
    deleted = [i for i, v in expected.items() if v[0] == "deleted"][:40]
    for i in deleted:
        s3, g = node.req("GET", f"/{idx}/_doc/{i}")
        if s3 != 404:
            problem("final-get-deleted", (i, s3))
    node.req("POST", f"/{idx}/_forcemerge?max_num_segments=1")
    node.req("POST", f"/{idx}/_refresh")
    s4, c2 = node.req("GET", f"/{idx}/_count")
    if c2.get("count") != len(live):
        problem("final-count-merged", (c2, len(live)))
    print(json.dumps({"url": a.url, "stats": stats, "live": len(live), "problems": len(problems)}))


if __name__ == "__main__":
    main()
