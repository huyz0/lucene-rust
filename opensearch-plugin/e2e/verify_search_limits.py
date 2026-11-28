#!/usr/bin/env python3
"""Index and request limits, answered by a plugin node (direct search on, the default) and a stock node:
every request must give the same status, the same error type and, when it succeeds, the same hits and total.

    python3 opensearch-plugin/e2e/verify_search_limits.py        (plugin on :9250, stock on :9251)

Each case builds a small one-shard index on both nodes with the case's index settings, runs the case's
requests on both and compares. Limits are where a request that skips OpenSearch's search action can
answer what the action refuses: `index.max_result_window`, `index.max_docvalue_fields_search`,
`index.max_terms_count`, `index.max_script_fields`, `search.allow_expensive_queries`, read blocks,
closed indices, wildcard and alias resolution options."""
import json
import sys
import urllib.error
import urllib.request

P, S = "http://localhost:9250", "http://localhost:9251"
DOCS = 150
MAPPING = {"properties": {"body": {"type": "text"}, "tag": {"type": "keyword"}, "n": {"type": "long"}}}


def req(base, method, path, body=None, nd=False):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    r = urllib.request.Request(base + path, data=data, method=method,
                               headers={"content-type": "application/x-ndjson" if nd else "application/json"})
    try:
        with urllib.request.urlopen(r) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, {}


def build(index, settings):
    for base in (P, S):
        req(base, "DELETE", "/" + index)
        s = {"index.number_of_shards": 1, "index.number_of_replicas": 0}
        s.update(settings)
        if base == P:
            s["index.engine.accelerated"] = True
        st, r = req(base, "PUT", "/" + index, {"settings": s, "mappings": MAPPING})
        assert st == 200, r
        lines = []
        for i in range(DOCS):
            lines += [json.dumps({"index": {"_id": str(i)}}),
                      json.dumps({"body": "alpha beta %d" % (i % 7), "tag": "t%d" % (i % 5), "n": i})]
        req(base, "POST", "/%s/_bulk?refresh=true" % index, ("\n".join(lines) + "\n").encode(), nd=True)


def shape(status, r):
    if status != 200:
        err = r.get("error", {})
        return (status, err.get("type") if isinstance(err, dict) else err)
    hits = r["hits"]
    return (status, hits.get("total"), [(h["_id"], h.get("_score"), h.get("sort"), h.get("fields")) for h in hits["hits"]])


failures = []


def check(label, path, body=None, method="POST"):
    a = shape(*req(P, method, path, body))
    b = shape(*req(S, method, path, body))
    if a != b:
        failures.append(label)
        print("FAIL", label, "\n  plugin:", str(a)[:300], "\n  stock: ", str(b)[:300])
    else:
        print("ok  ", label, str(a)[:70])


def case(name, settings, cluster=None):
    build("lim", settings)
    if cluster:
        for base in (P, S):
            req(base, "PUT", "/_cluster/settings", {"transient": cluster})
    return name


def reset_cluster(keys):
    for base in (P, S):
        req(base, "PUT", "/_cluster/settings", {"transient": {k: None for k in keys}})


Q = {"query": {"match": {"body": "alpha"}}}

case("max_result_window", {"index.max_result_window": 10})
check("window: size 50", "/lim/_search", {**Q, "size": 50})
check("window: from 5 size 20", "/lim/_search", {**Q, "from": 5, "size": 20})
check("window: size 10", "/lim/_search", {**Q, "size": 10})
check("window: size param", "/lim/_search?size=30", Q)
check("window: sorted", "/lim/_search", {**Q, "size": 30, "sort": [{"n": "desc"}]})
check("window: size 0 with aggs", "/lim/_search", {"size": 0, "aggs": {"t": {"terms": {"field": "tag"}}}})
check("window: search_after", "/lim/_search", {**Q, "size": 30, "sort": [{"n": "asc"}], "search_after": [3]})

case("docvalue", {"index.max_docvalue_fields_search": 1})
check("docvalue: two fields", "/lim/_search", {**Q, "size": 3, "docvalue_fields": ["n", "tag"]})
check("docvalue: one field", "/lim/_search", {**Q, "size": 3, "docvalue_fields": ["n"]})

case("terms", {"index.max_terms_count": 3})
check("terms: four", "/lim/_search", {"query": {"terms": {"tag": ["t0", "t1", "t2", "t3"]}}})
check("terms: three", "/lim/_search", {"query": {"terms": {"tag": ["t0", "t1", "t2"]}}})

case("expensive", {}, {"search.allow_expensive_queries": False})
check("expensive: prefix on text", "/lim/_search", {"query": {"prefix": {"body": "al"}}})
check("expensive: wildcard on keyword", "/lim/_search", {"query": {"wildcard": {"tag": "t*"}}})
check("expensive: range on keyword", "/lim/_search", {"query": {"range": {"tag": {"gte": "t1"}}}})
check("expensive: fuzzy", "/lim/_search", {"query": {"fuzzy": {"body": "alpa"}}})
check("expensive: range on long", "/lim/_search", {"query": {"range": {"n": {"gte": 5}}}})
reset_cluster(["search.allow_expensive_queries"])

case("clause_count", {})
for base in (P, S):
    req(base, "PUT", "/_cluster/settings", {"transient": {"indices.query.bool.max_clause_count": 2}})
check("clauses: three", "/lim/_search", {"query": {"bool": {"should": [{"term": {"tag": "t0"}}, {"term": {"tag": "t1"}}, {"term": {"tag": "t2"}}]}}})
reset_cluster(["indices.query.bool.max_clause_count"])

case("read_block", {"index.blocks.read": True})
check("read block", "/lim/_search", Q)
check("read block count", "/lim/_count", Q)

case("closed", {})
for base in (P, S):
    req(base, "POST", "/lim/_close")
check("closed index", "/lim/_search", Q)
check("closed index wildcard", "/li*/_search", Q)
check("closed index wildcard all", "/li*/_search?expand_wildcards=all", Q)
for base in (P, S):
    req(base, "POST", "/lim/_open")
    req(base, "GET", "/_cluster/health/lim?wait_for_status=green&timeout=30s")

case("resolution", {})
for base in (P, S):
    req(base, "POST", "/_aliases", {"actions": [{"add": {"index": "lim", "alias": "lima"}}]})
check("wildcard none", "/li*/_search?expand_wildcards=none", Q)
check("missing index", "/nope/_search", Q)
check("missing ignore_unavailable", "/nope/_search?ignore_unavailable=true", Q)
check("missing allow_no_indices false", "/nope*/_search?allow_no_indices=false", Q)
check("missing wildcard", "/nope*/_search", Q)
check("list with missing", "/lim,nope/_search", Q)
check("list with missing ignored", "/lim,nope/_search?ignore_unavailable=true", Q)
check("alias", "/lima/_search", Q)
check("exclusion", "/lim*,-lima/_search", Q)
check("date math", "/<lim>/_search", Q)
check("routing", "/lim/_search?routing=x", Q)
check("preference custom", "/lim/_search?preference=abc", Q)
check("preference replica", "/lim/_search?preference=_replica", Q)
check("preference shards", "/lim/_search?preference=_shards:0", Q)
check("size negative", "/lim/_search?size=-1", Q)
check("from negative", "/lim/_search?from=-1", Q)
check("body size negative", "/lim/_search", {**Q, "size": -1})
check("track_total_hits string", "/lim/_search?track_total_hits=abc", Q)
check("track_total_hits 5", "/lim/_search?track_total_hits=5", {**Q, "size": 2})
check("empty body object", "/lim/_search", {})
check("source false", "/lim/_search", {**Q, "size": 2, "_source": False})
check("unknown body key", "/lim/_search", {**Q, "bogus": 1})
check("unknown query", "/lim/_search", {"query": {"bogus": {}}})
check("size over 100", "/lim/_search", {**Q, "size": 101})
check("hidden", "/lim/_search?expand_wildcards=hidden", Q)
check("ignore_throttled", "/lim/_search?ignore_throttled=false", Q)
check("allow_partial", "/lim/_search?allow_partial_search_results=false", Q)
check("batched_reduce_size 1", "/lim/_search?batched_reduce_size=1", Q)
check("max_concurrent_shard_requests 0", "/lim/_search?max_concurrent_shard_requests=0", Q)
check("pre_filter_shard_size 0", "/lim/_search?pre_filter_shard_size=0", Q)
check("typed_keys false", "/lim/_search?typed_keys=false", {"size": 0, "aggs": {"t": {"terms": {"field": "tag"}}}})
check("request_cache bad", "/lim/_search?request_cache=maybe", Q)

if failures:
    print("\n%d differences:" % len(failures), *failures, sep="\n  ")
    sys.exit(1)
print("\nall equal")
