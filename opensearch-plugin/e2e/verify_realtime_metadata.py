#!/usr/bin/env python3
"""Versions, sequence numbers and explanations under the realtime view.

    PLUGIN_URL=http://localhost:9200 python3 opensearch-plugin/e2e/verify_realtime_metadata.py

Two indexes get the same writes: one with `index.search.realtime.enabled` (no refresh before the
searches), one refreshed. A search with `version`, `seq_no_primary_term` or `explain` must return
the same hits, versions, sequence numbers and explanation values on both, and the realtime index
must answer it natively (no `fetch_metadata` or `realtime_view` fallback).
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchhttp as bh  # noqa: E402

BASE = os.environ.get("PLUGIN_URL", "http://localhost:9200")
RT, REF = "rt_meta", "rt_meta_ref"
WORDS = ["fox", "quick", "brown", "dogs", "alpha", "beta", "gamma", "delta"]
MAPPING = {"properties": {"body": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "long"}, "u": {"type": "long"}}}


def req(method, path, body=None, ndjson=False):
    return bh.req(BASE, method, path, body, ndjson=ndjson)


def stats():
    return req("GET", "/_plugins/accelerated/stats")


def main():
    for idx in (RT, REF):
        try:
            req("DELETE", f"/{idx}")
        except RuntimeError:
            pass
    live = {"index.number_of_shards": 1, "index.number_of_replicas": 0, "index.search.realtime.enabled": True,
            "index.search.realtime.max_staleness": "0ms", "refresh_interval": "30s"}
    req("PUT", f"/{RT}", {"settings": live, "mappings": MAPPING})
    req("PUT", f"/{REF}", {"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0}, "mappings": MAPPING})
    req("GET", f"/_cluster/health/{RT},{REF}?wait_for_status=green&timeout=60s")
    r = random.Random(5)
    fails = []
    checked = 0
    before = stats()
    for rnd in range(4):
        lines = []
        for i in range(200):
            # Ids repeat, so versions go past 1 and some documents were updated since the commit.
            lines += [{"index": {"_id": f"d{r.randrange(300)}"}},
                      {"body": " ".join(r.choice(WORDS) for _ in range(r.randint(2, 8))), "tag": f"t{r.randrange(5)}",
                       "n": r.randrange(1000), "u": rnd * 1000 + len(lines)}]
        body = "\n".join(json.dumps(x) for x in lines) + "\n"
        req("POST", f"/{RT}/_bulk", body, ndjson=True)
        req("POST", f"/{REF}/_bulk?refresh=true", body, ndjson=True)
        bodies = [
            {"query": {"match": {"body": "fox"}}, "size": 500, "version": True, "seq_no_primary_term": True},
            {"query": {"match": {"body": "quick brown"}}, "size": 50, "version": True, "sort": [{"n": "asc"}, {"u": "asc"}]},
            {"query": {"match": {"body": "alpha"}}, "size": 30, "explain": True},
            {"query": {"bool": {"must": [{"match": {"body": "dogs"}}], "filter": [{"term": {"tag": "t2"}}]}},
             "size": 40, "explain": True, "version": True, "seq_no_primary_term": True,
             "sort": [{"n": "desc"}, {"u": "asc"}]},
        ]
        for b in bodies:
            a = req("POST", f"/{RT}/_search?request_cache=false", b)
            c = req("POST", f"/{REF}/_search?request_cache=false", b)

            def seen(h):
                e = h.get("_explanation", {}).get("value")
                return (h["_id"], h.get("sort"), h.get("_version"), h.get("_seq_no"), h.get("_primary_term"),
                        None if h.get("_score") is None else round(h["_score"], 4),
                        None if e is None else round(e, 4))

            ha, hc = [seen(h) for h in a["hits"]["hits"]], [seen(h) for h in c["hits"]["hits"]]
            if "sort" not in b:
                ha.sort(key=str)
                hc.sort(key=str)
            checked += 1
            if a["hits"]["total"] != c["hits"]["total"] or ha != hc:
                fails.append(f"round {rnd} {json.dumps(b)[:100]}: {ha[:2]} vs {hc[:2]}")
    after = stats()
    fell = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()}
    fell = {k: v for k, v in fell.items() if v}
    live_q = after.get("live_queries", 0) - before.get("live_queries", 0)
    print(f"searches checked: {checked}, mismatches: {len(fails)}, live queries: {live_q}, new fallbacks: {fell}")
    for f in fails[:10]:
        print("FAIL:", f)
    for idx in (RT, REF):
        req("DELETE", f"/{idx}")
    if fails or any(k in fell for k in ("fetch_metadata", "realtime_view")):
        sys.exit(1)


if __name__ == "__main__":
    main()
