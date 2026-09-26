#!/usr/bin/env python3
"""End-to-end verification of the lucene-rust plugin inside a real OpenSearch node.

Driven by scripts/verify-opensearch.sh, which builds the plugin, bakes it into
the pinned OpenSearch image and starts a node. Standard library only.

Every query in the matrix runs twice through the REST API: once with the
plugin's per-index switch off (stock Lucene -- the reference) and once with it
on. The two responses must agree -- hit ids, scores to 1e-5, total hits,
max_score -- and the plugin's own stats must show the query ran where the
matrix says it runs: native for the supported shapes, fallback (with the
expected reason) for the rest. Then:

  * lifecycle: a force merge plus refresh must leave no native reader open
    beyond the live searchers, and no mapping of a deleted index file in the
    node's address space;
  * crash: SIGKILL the node, restart it, and require the same document count
    and native/Lucene agreement on the recovered shards.

Usage: verify_opensearch.py <base-url> <container-name> [--docs N] [--bench-out FILE]
"""
import argparse
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:9200"
FAILURES = []
CHECKS = [0]


def req(method, path, body=None, ndjson=False):
    data = None
    headers = {}
    if body is not None:
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        headers["content-type"] = "application/x-ndjson" if ndjson else "application/json"
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:500]}")


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(what)
        print("FAIL:", what)


def stats():
    return req("GET", "/_plugins/lucene_rust/stats")


def set_native(index, on):
    req("PUT", f"/{index}/_settings", {"index.lucene_rust.search.enabled": on})


def set_shapes(index, mode):
    req("PUT", f"/{index}/_settings", {"index.lucene_rust.search.native_shapes": mode})


WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
         "sigma tau upsilon phi chi psi omega").split()


def word(r):
    return WORDS[min(len(WORDS) - 1, int((r.random() ** 2.2) * len(WORDS)))]


def create(index, shards):
    req("PUT", f"/{index}", {
        "settings": {
            "number_of_shards": shards,
            "number_of_replicas": 0,
            "refresh_interval": -1,
            # A field with non-default BM25 parameters: must fall back.
            "similarity": {"tuned": {"type": "BM25", "k1": 2.0, "b": 0.5}},
        },
        "mappings": {"properties": {
            "body": {"type": "text"},
            "title": {"type": "text"},
            "tuned": {"type": "text", "similarity": "tuned"},
            "tag": {"type": "keyword"},
            "mtag": {"type": "keyword"},
            "n": {"type": "long"},
            # Sort fields (read path R4): ties, gaps, several values per document.
            "price": {"type": "double"},
            "qty": {"type": "integer"},
            "ts": {"type": "date"},
            "@timestamp": {"type": "date"},
            "ratio": {"type": "float"},
            "m": {"type": "long"},
            "sp": {"type": "long"},
        }},
    })


def load(index, docs, seed, deletes=True):
    """Indexes `docs` documents in batches with a refresh after each (many
    segments), then deletes and updates some (hard and soft deletes) unless
    `deletes` is false."""
    r = random.Random(seed)
    batch = max(1, docs // 8)
    for start in range(0, docs, batch):
        lines = []
        for i in range(start, min(docs, start + batch)):
            lines.append(json.dumps({"index": {"_index": index, "_id": str(i)}}))
            doc = {
                "body": " ".join(word(r) for _ in range(r.randint(1, 40))),
                "title": " ".join(word(r) for _ in range(r.randint(1, 5))),
                "tuned": " ".join(word(r) for _ in range(r.randint(1, 10))),
                "tag": word(r),
                "mtag": [word(r) + str(r.randint(0, 9)) for _ in range(r.randint(0, 3))],
                "n": i,
                "price": round(r.uniform(-50, 500), 2),
                "qty": r.randint(0, 40),
                "ts": 1_700_000_000_000 + r.randint(0, 10_000) * 60_000,
                "@timestamp": 1_700_000_000_000 + r.randint(0, 500) * 60_000,
                "ratio": r.random(),
                "m": [r.randint(0, 1000) for _ in range(r.randint(0, 3))],
            }
            if r.random() < 0.7:
                doc["sp"] = r.randint(-100, 100)
            lines.append(json.dumps(doc))
        res = req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        check(not res["errors"], f"{index}: bulk batch at {start} has no errors")
        req("POST", f"/{index}/_refresh")
    if not deletes:
        return
    lines = []
    for i in r.sample(range(docs), docs // 40):
        lines.append(json.dumps({"delete": {"_index": index, "_id": str(i)}}))
    for i in r.sample(range(docs), docs // 60):
        lines.append(json.dumps({"index": {"_index": index, "_id": str(i)}}))
        lines.append(json.dumps({"body": "updated " + word(r), "title": "updated", "tuned": "x", "tag": "updated", "n": i, "qty": 3, "m": [5, 1]}))
    req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
    req("POST", f"/{index}/_refresh")


# (name, request body, expected): "native" runs native in both routing modes
# (since read path R1 no encodable shape is routed to Lucene as slower, so
# `native_shapes` fast and all agree); anything else is the fallback reason
# expected in both modes ("query_*" matches any query class).
def matrix():
    q = []
    add = lambda name, body, expect: q.append((name, body, expect))
    add("match one term", {"query": {"match": {"body": "alpha"}}}, "native")
    add("match rare term", {"query": {"match": {"body": "omega"}}}, "native")
    add("match two terms", {"query": {"match": {"body": "alpha kappa"}}}, "native")
    add("match four terms", {"query": {"match": {"body": "beta mu sigma omega"}}}, "native")
    add("match operator and", {"query": {"match": {"body": {"query": "alpha beta", "operator": "and"}}}}, "native")
    add("match minimum_should_match", {"query": {"match": {"body": {"query": "alpha beta gamma delta", "minimum_should_match": 2}}}}, "native")
    add("term keyword", {"query": {"term": {"tag": "gamma"}}}, "native")
    add("term missing", {"query": {"term": {"tag": "no-such-tag"}}}, "native")
    add("bool must+should", {"query": {"bool": {"must": [{"match": {"body": "alpha"}}], "should": [{"match": {"title": "beta"}}]}}}, "native")
    add("bool must_not", {"query": {"bool": {"must": [{"match": {"body": "beta"}}], "must_not": [{"term": {"tag": "alpha"}}]}}}, "native")
    add("bool filter", {"query": {"bool": {"must": [{"match": {"body": "gamma"}}], "filter": [{"term": {"tag": "alpha"}}]}}}, "native")
    add("bool filter only", {"query": {"bool": {"filter": [{"term": {"tag": "beta"}}]}}}, "native")
    add("bool nested", {"query": {"bool": {"should": [
        {"bool": {"must": [{"match": {"body": "alpha"}}, {"match": {"title": "gamma"}}]}},
        {"match": {"body": "omega"}}]}}}, "native")
    add("query_string OR", {"query": {"query_string": {"query": "body:(delta OR theta)"}}}, "native")
    add("size 0 count", {"size": 0, "query": {"match": {"body": "alpha"}}}, "native")
    add("track_total_hits false", {"track_total_hits": False, "query": {"match": {"body": "beta"}}}, "native")
    add("track_total_hits true", {"track_total_hits": True, "query": {"match": {"body": "alpha beta"}}}, "native")
    add("track_total_hits 100", {"track_total_hits": 100, "query": {"match": {"body": "alpha"}}}, "native")
    add("from 20 size 15", {"from": 20, "size": 15, "query": {"match": {"body": "delta"}}}, "native")
    add("size 200", {"size": 200, "query": {"match": {"body": "zeta eta"}}}, "native")
    # Totals below the threshold, where the coordinator's clamp to track_total_hits cannot hide a
    # shard-level count difference.
    add("size 0 below threshold", {"size": 0, "track_total_hits": 1000000, "query": {"term": {"tag": "omega"}}}, "native")
    add("rare term, exact total", {"track_total_hits": 1000000, "query": {"match": {"body": "omega psi"}}}, "native")
    add("highlight (fetch phase)", {"query": {"match": {"body": "alpha"}}, "highlight": {"fields": {"body": {}}}}, "native")
    # Outside the matrix: each must fall back, for the stated reason.
    add("match_phrase", {"query": {"match_phrase": {"body": "alpha beta"}}}, "native")
    add("prefix keyword", {"query": {"prefix": {"tag": "ga"}}}, "native")
    add("prefix text", {"query": {"bool": {"must": [{"match": {"body": "alpha"}}], "filter": [{"prefix": {"body": "be"}}]}}}, "native")
    add("wildcard", {"query": {"wildcard": {"tag": "*ta"}}}, "native")
    add("terms keyword", {"query": {"terms": {"tag": ["alpha", "beta", "omega"]}}}, "native")
    add("bool must_not terms", {"query": {"bool": {"must": [{"match": {"body": "gamma"}}], "must_not": [{"terms": {"tag": ["beta", "delta"]}}]}}}, "native")
    add("match_phrase slop", {"query": {"match_phrase": {"body": {"query": "beta alpha", "slop": 2}}}}, "native")
    add("bool must + phrase", {"query": {"bool": {"must": [{"match": {"body": "gamma"}}], "should": [{"match_phrase": {"body": "alpha beta"}}]}}}, "native")
    add("bool must_not phrase", {"query": {"bool": {"must": [{"match": {"body": "beta"}}], "must_not": [{"match_phrase": {"body": "alpha gamma"}}]}}}, "native")
    add("range", {"query": {"range": {"n": {"gte": 10, "lte": 500}}}}, "approximate")
    add("range, exact total", {"track_total_hits": True, "query": {"range": {"n": {"gte": 10, "lte": 500}}}}, "native")
    add("bool must + range filter", {"query": {"bool": {"must": [{"match": {"body": "alpha"}}], "filter": [{"range": {"n": {"gte": 1000, "lt": 30000}}}]}}}, "native")
    add("bool must_not range", {"query": {"bool": {"must": [{"match": {"body": "beta"}}], "must_not": [{"range": {"n": {"lte": 5000}}}]}}}, "native")
    add("prefix", {"query": {"prefix": {"tag": "al"}}}, "native")
    add("match_all", {"query": {"match_all": {}}}, "native")
    add("boosted match", {"query": {"match": {"body": {"query": "alpha", "boost": 2}}}}, "native")
    add("bool with boosted clause", {"query": {"bool": {"should": [{"match": {"body": {"query": "alpha", "boost": 3}}}, {"term": {"tag": "beta"}}]}}}, "native")
    add("constant_score", {"query": {"constant_score": {"filter": {"match": {"body": "gamma"}}, "boost": 1.5}}}, "native")
    add("multi_match", {"query": {"multi_match": {"query": "alpha", "fields": ["body", "title"]}}}, "native")
    add("dis_max", {"query": {"dis_max": {"tie_breaker": 0.3, "queries": [{"match": {"body": "alpha"}}, {"match": {"title": "gamma"}}]}}}, "native")
    # OpenSearch counts these without collecting (shortcutTotalHitCount): a
    # match-all always, a term when nothing is deleted (the "clean" index).
    # These rows prove the responses agree; they cannot prove the shortcut is
    # taken -- a response caps hits.total at track_total_hits either way, and
    # the matrix passed with the plugin's shortcut disabled.
    add("match_all size 0", {"size": 0, "query": {"match_all": {}}}, "native")
    add("term keyword size 0", {"size": 0, "query": {"term": {"tag": "beta"}}}, "native")
    add("constant_score match_all", {"query": {"constant_score": {"filter": {"match_all": {}}, "boost": 2}}}, "native")
    add("match_all + filter", {"query": {"bool": {"must": [{"match_all": {}}], "filter": [{"term": {"tag": "beta"}}]}}}, "native")
    add("bool msm with must", {"query": {"bool": {"must": [{"match": {"body": "alpha"}}], "should": [{"match": {"title": "beta"}}, {"match": {"title": "gamma"}}], "minimum_should_match": 1}}}, "native")
    add("custom similarity", {"query": {"match": {"tuned": "alpha"}}}, "field_similarity")
    # Sorted searches (read path R4): numeric keys of every type, missing values, several keys,
    # the score and _doc, search_after, and the ones that stay on Lucene.
    add("sort long desc", {"query": {"match": {"body": "alpha"}}, "sort": [{"n": "desc"}]}, "native")
    add("sort double asc", {"query": {"match": {"body": "beta"}}, "sort": [{"price": "asc"}]}, "native")
    add("sort integer desc, long asc", {"query": {"match_all": {}}, "sort": [{"qty": "desc"}, {"n": "asc"}]}, "native")
    add("sort date desc", {"query": {"bool": {"filter": [{"term": {"tag": "gamma"}}]}}, "sort": [{"ts": "desc"}]}, "native")
    add("sort float asc", {"query": {"match": {"body": "delta"}}, "sort": [{"ratio": {"order": "asc"}}]}, "native")
    add("sort multi-valued max", {"query": {"match_all": {}}, "sort": [{"m": {"order": "desc", "mode": "max"}}]}, "approximate")
    add("sort multi-valued min", {"query": {"match": {"body": "alpha"}}, "sort": [{"m": {"order": "asc", "mode": "min"}}]}, "native")
    add("sort sparse missing first", {"query": {"match": {"body": "gamma"}}, "sort": [{"sp": {"order": "asc", "missing": "_first"}}, {"n": "asc"}]}, "native")
    add("sort sparse missing last", {"query": {"match_all": {}}, "sort": [{"sp": {"order": "desc", "missing": "_last"}}]}, "native")
    add("sort sparse missing value", {"query": {"match_all": {}}, "sort": [{"sp": {"order": "asc", "missing": 7}}, {"n": "desc"}]}, "native")
    add("sort score then field", {"query": {"match": {"body": "alpha beta"}}, "sort": ["_score", {"qty": "asc"}]}, "native")
    add("sort field then score", {"query": {"match": {"body": "alpha beta"}}, "sort": [{"qty": "asc"}, "_score"]}, "native")
    add("sort _doc", {"query": {"match": {"body": "alpha"}}, "sort": ["_doc"]}, "native")
    add("sort from 20 size 15", {"from": 20, "size": 15, "query": {"match": {"body": "beta"}}, "sort": [{"qty": "asc"}, {"n": "desc"}]}, "native")
    add("sort track_total_hits true", {"track_total_hits": True, "query": {"match_all": {}}, "sort": [{"price": "desc"}]}, "native")
    add("sort size 0", {"size": 0, "query": {"match": {"body": "alpha"}}, "sort": [{"n": "desc"}]}, "native")
    add("search_after", {"query": {"match_all": {}}, "sort": [{"qty": "asc"}, {"n": "asc"}], "search_after": [10, 500]}, "native")
    add("search_after desc", {"query": {"match": {"body": "alpha"}}, "sort": [{"price": "desc"}, {"n": "asc"}], "search_after": [250.0, 100]}, "native")
    # OpenSearch answers a top-level range, and a match_all sorted by one numeric field with no
    # `missing`, approximately (ApproximateScoreQuery): its ties follow BKD order, so these stay on
    # OpenSearch's own path.
    add("sort keyword", {"query": {"match": {"body": "alpha"}}, "sort": [{"tag": "asc"}]}, "native")
    add("sort keyword desc then n", {"query": {"match_all": {}}, "sort": [{"tag": "desc"}, {"n": "asc"}]}, "native")
    add("sort keyword missing first max", {"query": {"match": {"body": "beta"}}, "sort": [{"tag": {"order": "asc", "missing": "_first", "mode": "max"}}, "_score"]}, "native")
    add("sort keyword search_after", {"query": {"match_all": {}}, "sort": [{"tag": "asc"}, {"n": "asc"}], "search_after": ["gamma", 100]}, "native")
    add("sort mode avg", {"query": {"match": {"body": "alpha"}}, "sort": [{"m": {"order": "asc", "mode": "avg"}}]}, "sort_*")
    add("sort track_scores", {"query": {"match": {"body": "alpha"}}, "sort": [{"n": "desc"}], "track_scores": True}, "native")
    add("sort keyword track_scores", {"query": {"match": {"body": "beta gamma"}}, "sort": [{"tag": "asc"}, "_doc"], "track_scores": True}, "native")
    # Metric aggregations (read path R5): top level, numeric or date fields, no missing/script.
    add("agg min max size 0", {"size": 0, "query": {"match": {"body": "alpha"}}, "aggs": {"lo": {"min": {"field": "price"}}, "hi": {"max": {"field": "n"}}}}, "native")
    add("agg stats + hits", {"query": {"match": {"body": "beta"}}, "aggs": {"s": {"stats": {"field": "qty"}}}}, "native")
    add("agg sum avg multi-valued", {"size": 0, "query": {"match_all": {}}, "aggs": {"s": {"sum": {"field": "m"}}, "a": {"avg": {"field": "m"}}, "c": {"value_count": {"field": "m"}}}}, "native")
    add("agg no query min max", {"size": 0, "aggs": {"hi": {"max": {"field": "n"}}, "t": {"max": {"field": "ts"}}, "lo": {"min": {"field": "qty"}}}}, "native")
    add("agg no query min double (points)", {"size": 0, "aggs": {"lo": {"min": {"field": "price"}}, "lof": {"min": {"field": "ratio"}}}}, "native")
    add("agg explicit match_all min max", {"size": 0, "query": {"match_all": {}}, "aggs": {"lo": {"min": {"field": "price"}}, "hi": {"max": {"field": "ratio"}}}}, "native")
    # A time-series shard sorted by @timestamp ascending: OpenSearch visits segments last first.
    add("agg sort @timestamp asc", {"query": {"match": {"body": "alpha"}}, "sort": [{"@timestamp": "asc"}], "aggs": {"s": {"sum": {"field": "price"}}}}, "time_series_order")
    add("agg date min max", {"size": 0, "query": {"bool": {"filter": [{"term": {"tag": "gamma"}}]}}, "aggs": {"first": {"min": {"field": "ts"}}, "last": {"max": {"field": "ts"}}}}, "native")
    add("agg sparse, nothing matches", {"size": 0, "query": {"term": {"tag": "no-such-tag"}}, "aggs": {"s": {"stats": {"field": "sp"}}, "a": {"avg": {"field": "sp"}}}}, "native")
    add("agg float sum with sort", {"query": {"match": {"body": "gamma"}}, "sort": [{"n": "desc"}], "aggs": {"r": {"sum": {"field": "ratio"}}}}, "native")
    add("agg with meta", {"size": 0, "query": {"match": {"body": "delta"}}, "aggs": {"lo": {"min": {"field": "qty"}, "meta": {"k": "v"}}}}, "native")
    add("agg missing value", {"size": 0, "query": {"match": {"body": "alpha"}}, "aggs": {"s": {"sum": {"field": "sp", "missing": 1}}}}, "aggregations")
    add("agg sub-aggregation", {"size": 0, "query": {"match": {"body": "alpha"}}, "aggs": {"g": {"global": {}, "aggs": {"lo": {"min": {"field": "n"}}}}}}, "aggregations")
    # terms on keyword fields: the default order, min_doc_count >= 1.
    add("aggregation", {"query": {"match": {"body": "alpha"}}, "aggs": {"tags": {"terms": {"field": "tag"}}}}, "native")
    add("agg terms size 0 match_all", {"size": 0, "aggs": {"tags": {"terms": {"field": "tag", "size": 5}}}}, "native")
    add("agg terms multi-valued", {"size": 0, "query": {"match": {"body": "beta"}}, "aggs": {"m": {"terms": {"field": "mtag", "size": 20}}}}, "native")
    add("agg terms shard_size", {"size": 0, "query": {"match": {"body": "gamma delta"}}, "aggs": {"t": {"terms": {"field": "mtag", "size": 3, "shard_size": 4}}}}, "native")
    add("agg terms + metrics + hits", {"size": 5, "query": {"match": {"body": "delta"}}, "aggs": {"t": {"terms": {"field": "tag", "size": 3}}, "p": {"avg": {"field": "price"}}}}, "native")
    add("agg terms with sort", {"query": {"match": {"body": "alpha"}}, "sort": [{"n": "desc"}], "aggs": {"t": {"terms": {"field": "tag"}}}}, "native")
    add("agg terms min_doc_count 3", {"size": 0, "query": {"match": {"body": "omega"}}, "aggs": {"t": {"terms": {"field": "mtag", "min_doc_count": 3}}}}, "native")
    add("agg terms order _key", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "order": {"_key": "asc"}}}}}, "aggregations")
    add("agg terms min_doc_count 0", {"size": 0, "query": {"match": {"body": "omega"}}, "aggs": {"t": {"terms": {"field": "tag", "min_doc_count": 0}}}}, "aggregations")
    add("agg terms include", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "include": "a.*"}}}}, "aggregations")
    add("agg terms numeric", {"size": 0, "aggs": {"t": {"terms": {"field": "qty"}}}}, "aggregations")
    # post_filter (read path R7): the hits are "query AND filter", scored by the query alone;
    # aggregations see every match of the query.
    add("post_filter", {"query": {"match": {"body": "alpha"}}, "post_filter": {"term": {"tag": "beta"}}}, "native")
    add("post_filter + aggs", {"query": {"match": {"body": "alpha"}}, "post_filter": {"term": {"tag": "beta"}}, "aggs": {"t": {"terms": {"field": "tag"}}, "p": {"sum": {"field": "price"}}}}, "native")
    add("post_filter size 0", {"size": 0, "query": {"match_all": {}}, "post_filter": {"range": {"n": {"gte": 100, "lt": 900}}}}, "native")
    add("post_filter sort", {"query": {"match": {"body": "gamma"}}, "sort": [{"n": "desc"}], "post_filter": {"bool": {"should": [{"term": {"tag": "alpha"}}, {"term": {"tag": "delta"}}]}}}, "native")
    add("post_filter bool query paged", {"from": 5, "size": 7, "query": {"bool": {"must": [{"match": {"body": "alpha"}}], "should": [{"match": {"title": "beta"}}]}}, "post_filter": {"range": {"sp": {"gte": 0}}}}, "native")
    add("post_filter no total", {"track_total_hits": False, "query": {"match": {"body": "beta delta"}}, "post_filter": {"term": {"tag": "gamma"}}}, "native")
    add("post_filter nothing", {"query": {"match": {"body": "alpha"}}, "post_filter": {"term": {"tag": "no-such-tag"}}}, "native")
    # min_score (R7): by score natively, OpenSearch's MinimumScoreCollector around every collector
    # (aggregations too); behind a sort it stays on Lucene.
    add("min_score", {"query": {"match": {"body": "alpha"}}, "min_score": 0.3}, "native")
    add("min_score disjunction", {"query": {"match": {"body": "alpha beta gamma"}}, "min_score": 1.2}, "native")
    add("min_score + aggs", {"query": {"match": {"body": "beta delta"}}, "min_score": 0.8, "aggs": {"t": {"terms": {"field": "tag"}}, "s": {"stats": {"field": "qty"}}}}, "native")
    add("min_score size 0", {"size": 0, "query": {"match": {"body": "gamma"}}, "min_score": 0.5}, "native")
    add("min_score size 0 aggs", {"size": 0, "query": {"match": {"body": "alpha omega"}}, "min_score": 0.9, "aggs": {"m": {"terms": {"field": "mtag", "size": 5}}, "p": {"avg": {"field": "price"}}}}, "native")
    add("min_score + post_filter", {"query": {"match": {"body": "alpha beta"}}, "min_score": 0.6, "post_filter": {"term": {"tag": "gamma"}}}, "native")
    add("min_score nothing passes", {"query": {"match": {"body": "alpha"}}, "min_score": 1000}, "native")
    add("min_score everything passes", {"query": {"match_all": {}}, "min_score": 0}, "native")
    add("min_score no total", {"track_total_hits": False, "query": {"match": {"body": "delta"}}, "min_score": 0.4}, "native")
    add("min_score sort", {"query": {"match": {"body": "alpha"}}, "min_score": 0.3, "sort": [{"n": "desc"}]}, "min_score")
    # terminate_after (R7): native where Lucene collects document by document (a scored search
    # over a query whose bulk scorer hands out no ranges, anything under a post_filter) and for
    # a size-0 count over a term or match-all; Lucene's otherwise.
    add("terminate_after", {"query": {"match": {"body": "alpha"}}, "terminate_after": 5}, "native")
    add("terminate_after disjunction", {"size": 3, "query": {"match": {"body": "alpha beta"}}, "terminate_after": 50}, "native")
    add("terminate_after size 0 term", {"size": 0, "query": {"term": {"tag": "beta"}}, "terminate_after": 10}, "native")
    add("terminate_after size 0 match_all", {"size": 0, "query": {"match_all": {}}, "terminate_after": 100}, "native")
    add("terminate_after size 0 no query", {"size": 0, "terminate_after": 3}, "native")
    add("terminate_after size 0 bool filter", {"size": 0, "query": {"bool": {"filter": [{"term": {"tag": "gamma"}}]}}, "terminate_after": 20}, "native")
    add("terminate_after size 0 no total", {"size": 0, "track_total_hits": False, "query": {"match": {"body": "beta gamma"}}, "terminate_after": 7}, "native")
    add("terminate_after sort track_scores", {"query": {"match": {"body": "gamma"}}, "sort": [{"n": "desc"}], "track_scores": True, "terminate_after": 40}, "native")
    add("terminate_after sort score then field", {"query": {"match": {"body": "delta"}}, "sort": ["_score", {"n": "asc"}], "terminate_after": 25}, "native")
    add("terminate_after post_filter", {"query": {"match_all": {}}, "post_filter": {"term": {"tag": "alpha"}}, "terminate_after": 7}, "native")
    add("terminate_after not reached", {"track_total_hits": True, "query": {"match": {"body": "omega"}}, "terminate_after": 100000}, "native")
    add("terminate_after field sort", {"query": {"match": {"body": "alpha"}}, "sort": [{"n": "desc"}], "terminate_after": 5}, "terminate_after")
    add("terminate_after match_all hits", {"query": {"match_all": {}}, "terminate_after": 5}, "terminate_after")
    add("terminate_after aggs", {"size": 0, "query": {"match": {"body": "alpha"}}, "terminate_after": 5, "aggs": {"t": {"terms": {"field": "tag"}}}}, "terminate_after")
    add("profile", {"query": {"match": {"body": "alpha"}}, "profile": True}, "profile")
    # timeout (R7): native, checked before and after the native search as ContextIndexSearcher
    # checks it before each segment; none of these reaches it.
    add("timeout", {"query": {"match": {"body": "alpha"}}, "timeout": "10s"}, "native")
    add("timeout sort aggs", {"query": {"match": {"body": "beta"}}, "timeout": "30s", "sort": [{"n": "asc"}], "aggs": {"t": {"terms": {"field": "tag"}}}}, "native")
    add("collapse", {"query": {"match": {"body": "alpha"}}, "collapse": {"field": "tag"}}, "collapse")
    # Scored with statistics the native engine does not have: cross-shard (dfs) or blended
    # TermStates (cross_fields; tie_breaker 1 makes Lucene rewrite the dismax to a boolean).
    add("cross_fields", {"query": {"multi_match": {"query": "alpha", "type": "cross_fields", "fields": ["body", "title"], "tie_breaker": 1}}}, "term_states")
    add("dfs_query_then_fetch", {"_params": "&search_type=dfs_query_then_fetch", "query": {"match": {"body": "alpha"}}}, "dfs_multi")
    add("rescore", {"query": {"match": {"body": "alpha"}}, "rescore": {"window_size": 20, "query": {"rescore_query": {"match": {"title": "beta"}}}}}, "rescore")
    return q


def shape(resp, body):
    """The part of a search response both engines must agree on."""
    hits = resp["hits"]
    total = hits.get("total")
    out = {
        "total": total,
        "max_score": hits.get("max_score"),
        "hits": [(h["_id"], h.get("_score"), h.get("sort")) for h in hits["hits"]],
    }
    if "aggregations" in resp:
        out["aggs"] = resp["aggregations"]
    out["terminated_early"] = resp.get("terminated_early")
    out["timed_out"] = resp.get("timed_out")
    if "highlight" in json.dumps(body):
        out["highlight"] = [h.get("highlight") for h in hits["hits"]]
    return out


def same(a, b):
    """Equal, with scores compared to 1e-5; hits whose scores tie may swap."""
    if a["total"] != b["total"]:
        return f"total {a['total']} vs {b['total']}"
    for k in ("aggs", "highlight", "terminated_early", "timed_out"):
        if a.get(k) != b.get(k):
            return f"{k} differ"
    ma, mb = a["max_score"], b["max_score"]
    if (ma is None) != (mb is None) or (ma is not None and abs(ma - mb) > 1e-5 * max(1, abs(mb))):
        return f"max_score {ma} vs {mb}"
    if len(a["hits"]) != len(b["hits"]):
        return f"{len(a['hits'])} hits vs {len(b['hits'])}"
    for i, (x, y) in enumerate(zip(a["hits"], b["hits"])):
        sx, sy = x[1], y[1]
        if (sx is None) != (sy is None) or (sx is not None and abs(sx - sy) > 1e-5 * max(1, abs(sy))):
            return f"hit {i}: {x} vs {y}"
        if x[0] != y[0]:
            ties = [h for h in b["hits"] if h[1] is not None and sx is not None and abs(h[1] - sx) <= 1e-5 * max(1, abs(sx))]
            if x[0] not in [h[0] for h in ties]:
                return f"hit {i}: {x} vs {y}"
        if x[2] != y[2]:
            return f"hit {i}: sort {x[2]} vs {y[2]}"
    return None


def search_url(index, body):
    """`_search`, with a matrix row's `_params` (URL parameters) moved to the query string."""
    params = body.get("_params", "")
    return f"/{index}/_search?request_cache=false{params}", {k: v for k, v in body.items() if k != "_params"}


def run_matrix(index, shards, label, shapes="fast"):
    """The whole matrix against the Lucene reference, under one routing mode."""
    label = f"{label}/{shapes}"
    set_shapes(index, shapes)
    queries = matrix()
    set_native(index, False)
    reference = {}
    for name, body, _ in queries:
        url, b = search_url(index, body)
        reference[name] = shape(req("POST", url, b), b)
    set_native(index, True)
    native_total = 0
    for name, body, expect in queries:
        if expect == "slow":
            expect = "native" if shapes == "all" else "slower_shape"
        if expect == "dfs_multi":
            # OpenSearch runs a one-shard dfs search as query_then_fetch
            # (TransportSearchAction), so only a multi-shard index gets the dfs phase.
            expect = "dfs" if shards > 1 else "native"
        before = stats()
        url, b = search_url(index, body)
        got = shape(req("POST", url, b), b)
        after = stats()
        diff = same(got, reference[name])
        check(diff is None, f"{label} {index} [{name}]: native-enabled response differs from Lucene: {diff}")
        ran_native = after["native_queries"] - before["native_queries"]
        errors = after["native_errors"] - before["native_errors"]
        check(errors == 0, f"{label} {index} [{name}]: {errors} native errors")
        if expect == "native":
            check(ran_native == shards, f"{label} {index} [{name}]: ran native on {ran_native} of {shards} shards; fallbacks {fallback_delta(before, after)}")
            native_total += ran_native
        else:
            delta = fallback_delta(before, after)
            if expect.endswith("*"):
                matched = sum(v for k, v in delta.items() if k.startswith(expect[:-1]))
            else:
                matched = delta.get(expect, 0)
            print(f"  {label} {index} [{name}]: fallback {delta}")
            check(ran_native == 0 and matched == shards,
                  f"{label} {index} [{name}]: expected fallback '{expect}' on {shards} shards, got native={ran_native} fallbacks={delta}")
    return native_total


SCROLLS = [
    ("scroll by score", {"size": 7, "query": {"match": {"body": "alpha beta"}}}),
    ("scroll by _doc", {"size": 50, "query": {"match_all": {}}, "sort": ["_doc"]}),
    ("scroll sorted", {"size": 9, "query": {"match": {"body": "gamma"}}, "sort": [{"n": "desc"}, {"tag": "asc"}]}),
    ("scroll score track_scores", {"size": 11, "query": {"bool": {"should": [{"match": {"body": "delta"}}, {"term": {"tag": "beta"}}]}}, "track_scores": True}),
    ("scroll + post_filter", {"size": 6, "query": {"match": {"body": "omega"}}, "post_filter": {"term": {"tag": "alpha"}}}),
]


def scroll_pages(index, body, limit=40):
    """Every page of a scroll (at most `limit`), each as `shape` gives it."""
    resp = req("POST", f"/{index}/_search?scroll=1m", body)
    pages = [shape(resp, body)]
    sid = resp["_scroll_id"]
    while resp["hits"]["hits"] and len(pages) < limit:
        resp = req("POST", "/_search/scroll", {"scroll": "1m", "scroll_id": sid})
        sid = resp["_scroll_id"]
        pages.append(shape(resp, body))
    req("DELETE", "/_search/scroll", {"scroll_id": sid})
    return pages


def run_scroll(index, shards, label):
    """Scrolls to the end with the native engine off, then on: page for page the same, and every
    page's query phase native on every shard."""
    for name, body in SCROLLS:
        set_native(index, False)
        reference = scroll_pages(index, body)
        set_native(index, True)
        before = stats()
        got = scroll_pages(index, body)
        after = stats()
        check(len(got) == len(reference), f"{label} {index} [{name}]: {len(got)} pages vs {len(reference)}")
        for i, (g, r) in enumerate(zip(got, reference)):
            diff = same(g, r)
            check(diff is None, f"{label} {index} [{name}]: page {i} differs from Lucene: {diff}")
        ran_native = after["native_queries"] - before["native_queries"]
        check(ran_native == len(got) * shards,
              f"{label} {index} [{name}]: ran native {ran_native} times for {len(got)} pages on {shards} shards; fallbacks {fallback_delta(before, after)}")


def fallback_delta(before, after):
    out = {}
    for k, v in after["fallbacks"].items():
        d = v - before["fallbacks"].get(k, 0)
        if d:
            out[k] = d
    return out


def java_pid(container):
    """The node's JVM (the image has no pgrep)."""
    script = 'for p in /proc/[0-9]*; do grep -qa org.opensearch.bootstrap.OpenSearch $p/cmdline 2>/dev/null && echo ${p#/proc/}; done'
    return subprocess.check_output(["docker", "exec", container, "sh", "-c", script]).decode().split()[0]


def deleted_index_mappings(container):
    """Mappings of index files that were deleted from disk but are still mapped."""
    pid = java_pid(container)
    maps = subprocess.check_output(["docker", "exec", container, "cat", f"/proc/{pid}/maps"]).decode()
    return [l for l in maps.splitlines() if "/indices/" in l and "(deleted)" in l]


def lifecycle(index, container):
    """Force-merge down to one segment, twice: the native readers over the
    old segments must close with their Java readers, and nothing may keep a
    merged-away file mapped.

    Two rounds because the native reader copies compound (.cfs) segments into
    memory rather than mapping them, and OpenSearch's small flushed segments
    are compound -- so a leak of the first round's readers is visible only to
    the open-reader count. The first merge produces a large non-compound
    segment, which the native reader does map; the second merge deletes it,
    and a leaked reader then shows as a mapping of a deleted file."""
    s0 = None
    for rnd in (1, 2):
        req("POST", f"/{index}/_search?request_cache=false", {"query": {"match": {"body": "alpha beta"}}})
        s0 = stats()
        check(s0["open_native_readers"] >= 1, f"lifecycle {rnd}: a native reader is open before the merge")
        segs_before = len(req("GET", f"/_cat/segments/{index}?format=json"))
        req("POST", f"/{index}/_forcemerge?max_num_segments=1")
        req("POST", f"/{index}/_refresh")
        req("POST", f"/{index}/_search?request_cache=false", {"query": {"match": {"body": "alpha beta"}}})
        # Old searchers are released asynchronously once no search holds them.
        deadline = time.time() + 30
        leftover = None
        while time.time() < deadline:
            leftover = deleted_index_mappings(container)
            if not leftover and stats()["open_native_readers"] <= s0["open_native_readers"]:
                break
            time.sleep(1)
        segs_after = len(req("GET", f"/_cat/segments/{index}?format=json"))
        print(f"lifecycle {rnd}: {segs_before} segments -> {segs_after}; open native readers "
              f"{s0['open_native_readers']} -> {stats()['open_native_readers']}")
        check(segs_after == 1, f"lifecycle {rnd}: force merge left {segs_after} segments")
        check(not leftover, f"lifecycle {rnd}: deleted index files still mapped after the merge: {leftover[:3]}")
        check(stats()["open_native_readers"] <= s0["open_native_readers"],
              f"lifecycle {rnd}: native readers leaked: {stats()['open_native_readers']}")
        if rnd == 1:
            # New segments on top of the merged one, for the second round.
            load_more(index, 2000)


def load_more(index, n):
    r = random.Random(7)
    lines = []
    for i in range(n):
        lines.append(json.dumps({"index": {"_index": index, "_id": f"more-{i}"}}))
        lines.append(json.dumps({"body": " ".join(word(r) for _ in range(r.randint(1, 40))),
                                 "title": word(r), "tuned": word(r), "tag": word(r), "n": 10_000_000 + i}))
    req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
    req("POST", f"/{index}/_refresh")


def unsupported_format():
    """An index with a field in a postings format the Rust reader does not
    decode (a completion field): every search must fall back cleanly, with
    the reason, and none may reach the native engine to fail there."""
    index = "completion"
    try:
        req("DELETE", f"/{index}")
    except RuntimeError:
        pass
    req("PUT", f"/{index}", {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
                             "mappings": {"properties": {"body": {"type": "text"}, "suggest": {"type": "completion"}}}})
    lines = []
    for i in range(50):
        lines.append(json.dumps({"index": {"_index": index, "_id": str(i)}}))
        lines.append(json.dumps({"body": "alpha beta" if i % 2 else "gamma", "suggest": f"word{i}"}))
    req("POST", "/_bulk?refresh=true", "\n".join(lines) + "\n", ndjson=True)
    before = stats()
    r = req("POST", f"/{index}/_search?request_cache=false", {"query": {"match": {"body": "alpha"}}})
    after = stats()
    check(r["hits"]["total"]["value"] == 25, f"completion index: {r['hits']['total']}")
    delta = fallback_delta(before, after)
    check(delta.get("postings_format") == 1 and after["native_errors"] == before["native_errors"],
          f"completion index: expected a clean 'postings_format' fallback, got {delta}, errors "
          f"{after['native_errors'] - before['native_errors']}")
    req("DELETE", f"/{index}")


def wait_up(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            h = req("GET", "/_cluster/health?wait_for_status=yellow&timeout=5s")
            if h["status"] in ("yellow", "green") and h["initializing_shards"] == 0 and h["unassigned_shards"] == 0:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("node did not come back")


def crash(container, indices):
    counts = {i: req("GET", f"/{i}/_count")["count"] for i in indices}
    # Unrefreshed, uncommitted writes in flight at the kill: recovery replays
    # them from the translog.
    lines = []
    for n in range(200):
        lines.append(json.dumps({"index": {"_index": indices[0], "_id": f"late-{n}"}}))
        lines.append(json.dumps({"body": "late arrival alpha", "title": "late", "tuned": "late", "tag": "late", "n": -n}))
    req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
    counts[indices[0]] += 200
    subprocess.check_call(["docker", "kill", "--signal", "KILL", container], stdout=subprocess.DEVNULL)
    time.sleep(2)
    subprocess.check_call(["docker", "start", container], stdout=subprocess.DEVNULL)
    wait_up()
    # Read before the counts below: a _count is a match_all, which runs natively.
    s = stats()
    check(s["native_queries"] == 0, f"crash: stats reset with the process ({s['native_queries']})")
    for i in indices:
        req("POST", f"/{i}/_refresh")
        got = req("GET", f"/{i}/_count")["count"]
        check(got == counts[i], f"crash: {i} has {got} docs after SIGKILL + restart, expected {counts[i]}")


def bench(index, out, rounds):
    """REST latency, native vs Lucene, same node, same index, same queries --
    every shape the engine answers natively (native_shapes=all), so the
    routing policy's "slow" shapes are measured too."""
    set_shapes(index, "all")
    queries = [(n, b) for n, b, e in matrix() if e in ("native", "slow") and "highlight" not in n]
    kinds = {n: e for n, _, e in matrix()}
    results = {}
    for mode in (False, True, False, True):
        set_native(index, mode)
        for name, body in queries:
            for _ in range(3):
                req("POST", f"/{index}/_search?request_cache=false", body)
            took = []
            wall = []
            for _ in range(rounds):
                t = time.perf_counter()
                r = req("POST", f"/{index}/_search?request_cache=false", body)
                wall.append((time.perf_counter() - t) * 1000)
                took.append(r["took"])
            results.setdefault(name, {}).setdefault("native" if mode else "lucene", []).extend(wall)
    set_native(index, True)
    set_shapes(index, "fast")
    docs = req("GET", f"/{index}/_count")["count"]
    segs = len(req("GET", f"/_cat/segments/{index}?format=json"))
    with open(out, "w") as f:
        json.dump({
            "index": {"docs": docs, "segments": segs, "rounds": rounds},
            "queries": {k: {"routing": kinds[k], **{m: statistics.median(v) for m, v in d.items()}} for k, d in results.items()},
        }, f, indent=1)
    print(f"bench: wrote {out}")


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("container")
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--bench-out")
    ap.add_argument("--bench-rounds", type=int, default=30)
    a = ap.parse_args()
    BASE = a.base.rstrip("/")
    wait_up()
    for index in ("single", "multi"):
        try:
            req("DELETE", f"/{index}")
        except RuntimeError:
            pass
    create("single", 1)
    create("multi", 3)
    load("single", a.docs, 1)
    load("multi", a.docs, 2)
    # No deletions, and more than 10,000 documents: every shortcut total applies.
    try:
        req("DELETE", "/clean")
    except RuntimeError:
        pass
    create("clean", 1)
    load("clean", max(a.docs, 12000), 3, deletes=False)
    native = 0
    for shapes in ("fast", "all"):
        native += run_matrix("single", 1, "initial", shapes) + run_matrix("multi", 3, "initial", shapes)
    native += run_matrix("clean", 1, "no deletions")
    print(f"matrix: {len(matrix())} request shapes x 3 indices; {native} shard queries ran native")
    run_scroll("single", 1, "scroll")
    run_scroll("multi", 3, "scroll")
    unsupported_format()
    lifecycle("single", a.container)
    run_matrix("single", 1, "after merge")
    run_matrix("single", 1, "after merge", "all")
    crash(a.container, ["single", "multi"])
    run_matrix("single", 1, "after crash")
    run_matrix("multi", 3, "after crash")
    if a.bench_out:
        bench("single", a.bench_out, a.bench_rounds)
    print(f"verify_opensearch: {CHECKS[0]} checks, {len(FAILURES)} failures")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
