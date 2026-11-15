#!/usr/bin/env python3
"""Document and search API scenarios against a stock node: every request below runs on both nodes
against the same data and its response (minus timings) must be identical -- explain, collapse,
inner_hits, highlight variants, suggesters, scroll and point-in-time paging, more_like_this,
percolate, term vectors, updates / deletes / versions / routing, update_by_query /
delete_by_query / reindex, mget, bulk error statuses, count, validate, field_caps, msearch.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_api_stock.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node, diff  # noqa: E402

plugin = Node(os.environ.get("PLUGIN_URL", "http://localhost:9200"), 120)
stock = Node(os.environ.get("STOCK_URL", "http://localhost:9201"), 120)
bad = 0
total = 0
IGN = {"took", "timed_out", "_shards", "took_in_millis", "time_in_nanos", "_scroll_id", "pit_id", "id", "start_time_in_millis", "running_time_in_nanos", "time", "failures_total", "node",
       "breakdown", "total_time_in_millis", "_seq_no", "_primary_term", "tasks", "task", "response_processors", "_index_uuid", "index_uuid", "uuid", "creation_date"}


def both(method, path, body=None, raw=False, ign=IGN, label=None, tol=1e-4):
    global bad, total
    total += 1
    p = plugin.req(method, path, body, raw)
    s = stock.req(method, path, body, raw)
    if p[0] != s[0]:
        bad += 1
        print(f"[{label or path}] status plugin={p[0]} stock={s[0]}: {json.dumps(p[1])[:200]} || {json.dumps(s[1])[:200]}")
        return p, s
    d = diff(p[1], s[1], "", tol, ign)
    if d:
        bad += 1
        print(f"[{label or path}] {d[:400]}")
    return p, s


def setup():
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    r = random.Random(3)
    for n in (plugin, stock):
        n.req("DELETE", "/api,api2,api_dst,percol")
        n.req("PUT", "/api", {"settings": {"number_of_shards": 2, "number_of_replicas": 0, "index.number_of_routing_shards": 2},
                              "mappings": {"properties": {"title": {"type": "text", "fields": {"raw": {"type": "keyword"}, "trigram": {"type": "text", "analyzer": "standard"}}},
                                                          "body": {"type": "text", "term_vector": "with_positions_offsets_payloads"}, "tag": {"type": "keyword"}, "grp": {"type": "keyword"}, "n": {"type": "long"},
                                                          "suggest": {"type": "completion"}, "ts": {"type": "date"},
                                                          "kids": {"type": "nested", "properties": {"name": {"type": "keyword"}, "score": {"type": "integer"}, "txt": {"type": "text"}}}}}})
    lines = []
    for i in range(300):
        d = {"title": " ".join(r.choice(words) for _ in range(r.randint(1, 4))), "body": " ".join(r.choice(words) for _ in range(r.randint(5, 30))),
             "tag": r.choice(words[:5]), "grp": f"g{i % 7}", "n": r.randint(0, 100), "ts": f"20{10 + i % 12}-0{1 + i % 9}-1{i % 9}",
             "suggest": [r.choice(words) + r.choice(["", "x", "y"]) for _ in range(2)],
             "kids": [{"name": r.choice(words[:4]), "score": r.randint(1, 5), "txt": " ".join(r.choice(words) for _ in range(3))} for _ in range(r.randint(0, 3))]}
        lines += [json.dumps({"index": {"_id": f"d{i}", "routing": f"r{i % 3}"}}), json.dumps(d)]
    payload = ("\n".join(lines) + "\n").encode()
    for n in (plugin, stock):
        n.req("POST", "/api/_bulk?refresh=true", payload, raw=True)
        n.req("POST", "/api/_forcemerge?max_num_segments=1")
        n.req("POST", "/api/_refresh")


def main():
    setup()
    # scoring explanations, for several query shapes
    for q in [{"match": {"title": "alpha beta"}}, {"match_phrase": {"body": "alpha beta"}}, {"bool": {"must": [{"match": {"title": "gamma"}}], "should": [{"term": {"tag": "alpha"}}], "filter": [{"range": {"n": {"gte": 10}}}]}},
              {"multi_match": {"query": "alpha delta", "fields": ["title^2", "body"], "type": "best_fields", "tie_breaker": 0.3}}, {"function_score": {"query": {"match": {"title": "mu"}}, "functions": [{"field_value_factor": {"field": "n", "modifier": "log1p"}}]}},
              {"query_string": {"query": "title:alpha~1 OR body:be*"}}, {"dis_max": {"queries": [{"match": {"title": "eta"}}, {"match": {"body": "eta"}}]}}, {"constant_score": {"filter": {"term": {"tag": "beta"}}, "boost": 3}},
              {"nested": {"path": "kids", "query": {"term": {"kids.name": "alpha"}}, "score_mode": "avg"}}, {"match_phrase_prefix": {"title": "alpha b"}}, {"match": {"title": {"query": "alpah", "fuzziness": 1}}}]:
        p, s = both("POST", "/api/_search", {"query": q, "size": 3, "explain": True, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}]}, label=f"explain {json.dumps(q)[:60]}", ign=IGN | {"_shard", "_node"})
    # explain API, validate, count
    both("POST", "/api/_explain/d3?routing=r0", {"query": {"match": {"title": "alpha"}}}, label="_explain doc")
    both("POST", "/api/_validate/query?explain=true&rewrite=true", {"query": {"query_string": {"query": "title:(alpha OR beta*) AND n:[1 TO 5]"}}}, label="_validate")
    both("POST", "/api/_count", {"query": {"range": {"n": {"gte": 50}}}}, label="_count")
    # collapse + inner_hits
    both("POST", "/api/_search", {"query": {"match": {"body": "alpha"}}, "collapse": {"field": "grp", "inner_hits": {"name": "top", "size": 2, "sort": [{"n": "desc"}, {"_seq_no": "asc"}]}}, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}], "size": 5}, label="collapse inner_hits")
    both("POST", "/api/_search", {"query": {"nested": {"path": "kids", "query": {"range": {"kids.score": {"gte": 3}}}, "inner_hits": {"size": 5, "sort": [{"kids.score": "desc"}], "_source": ["kids.name"]}}}, "sort": [{"_seq_no": "asc"}], "size": 5}, label="nested inner_hits")
    # highlighting
    for hl in [{"type": "unified"}, {"type": "plain"}, {"type": "fvh"}, {"type": "unified", "fragment_size": 40, "number_of_fragments": 2}, {"type": "plain", "number_of_fragments": 0}, {"pre_tags": ["<x>"], "post_tags": ["</x>"]},
               {"type": "unified", "boundary_scanner": "sentence"}, {"type": "unified", "order": "score", "no_match_size": 20}, {"type": "fvh", "boundary_chars": ".,!?", "phrase_limit": 5}, {"type": "unified", "highlight_query": {"match": {"body": "beta"}}}]:
        both("POST", "/api/_search", {"query": {"match": {"body": "alpha beta"}}, "size": 3, "sort": [{"_seq_no": "asc"}], "highlight": {"fields": {"body": hl, "title": {}}, "require_field_match": False}}, label=f"highlight {json.dumps(hl)}")
    both("POST", "/api/_search", {"query": {"match_phrase": {"body": "alpha beta"}}, "size": 3, "sort": [{"_seq_no": "asc"}], "highlight": {"fields": {"body": {"type": "unified"}}}}, label="highlight phrase")
    # suggesters
    both("POST", "/api/_search", {"size": 0, "suggest": {"s": {"text": "alpah betta", "term": {"field": "title"}}}}, label="term suggest")
    both("POST", "/api/_search", {"size": 0, "suggest": {"s": {"text": "alpah betta gama", "phrase": {"field": "title.trigram", "size": 3, "direct_generator": [{"field": "title.trigram", "suggest_mode": "always"}], "highlight": {"pre_tag": "<em>", "post_tag": "</em>"}}}}}, label="phrase suggest")
    both("POST", "/api/_search", {"size": 0, "suggest": {"s": {"prefix": "al", "completion": {"field": "suggest", "size": 5}}}}, label="completion suggest")
    both("POST", "/api/_search", {"size": 0, "suggest": {"s": {"prefix": "alpah", "completion": {"field": "suggest", "size": 5, "fuzzy": {"fuzziness": 1}}}}}, label="completion fuzzy")
    # scroll paging
    for n in (plugin, stock):
        pass
    outs = []
    for n in (plugin, stock):
        st, r = n.req("POST", "/api/_search?scroll=1m", {"size": 40, "query": {"match": {"body": "alpha"}}, "sort": [{"n": "asc"}, {"_seq_no": "asc"}], "_source": False})
        ids = [h["_id"] for h in r["hits"]["hits"]]
        sid = r.get("_scroll_id")
        while True:
            st, r = n.req("POST", "/_search/scroll", {"scroll": "1m", "scroll_id": sid})
            if st != 200 or not r["hits"]["hits"]:
                break
            ids += [h["_id"] for h in r["hits"]["hits"]]
        n.req("DELETE", "/_search/scroll", {"scroll_id": sid})
        outs.append(ids)
    global total, bad
    total += 1
    if outs[0] != outs[1]:
        bad += 1
        print(f"[scroll] ids differ: {outs[0][:8]} vs {outs[1][:8]}")
    # PIT + search_after
    outs = []
    for n in (plugin, stock):
        st, r = n.req("POST", "/api/_search/point_in_time?keep_alive=1m")
        pit = r.get("pit_id")
        ids, after = [], None
        for _ in range(20):
            q = {"size": 50, "pit": {"id": pit, "keep_alive": "1m"}, "sort": [{"n": "desc"}, {"_seq_no": "asc"}], "_source": False}
            if after:
                q["search_after"] = after
            st, r = n.req("POST", "/_search", q)
            if st != 200 or not r["hits"]["hits"]:
                break
            ids += [h["_id"] for h in r["hits"]["hits"]]
            after = r["hits"]["hits"][-1]["sort"]
        n.req("DELETE", "/_search/point_in_time", {"pit_id": [pit]})
        outs.append(ids)
    total += 1
    if outs[0] != outs[1] or len(outs[0]) != 300:
        bad += 1
        print(f"[pit] differ or incomplete: {len(outs[0])} vs {len(outs[1])}")
    # more_like_this, percolate
    both("POST", "/api/_search", {"query": {"more_like_this": {"fields": ["body"], "like": [{"_index": "api", "_id": "d5", "routing": "r2"}], "min_term_freq": 1, "min_doc_freq": 1, "max_query_terms": 12}}, "size": 5, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}]}, label="more_like_this by doc")
    both("POST", "/api/_search", {"query": {"more_like_this": {"fields": ["body", "title"], "like": "alpha beta gamma delta", "min_term_freq": 1, "min_doc_freq": 1}}, "size": 5, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}]}, label="more_like_this text")
    for n in (plugin, stock):
        n.req("PUT", "/percol", {"mappings": {"properties": {"q": {"type": "percolator"}, "body": {"type": "text"}, "n": {"type": "long"}}}})
        for i, q in enumerate([{"match": {"body": "alpha"}}, {"bool": {"must": [{"match": {"body": "beta"}}, {"range": {"n": {"gte": 5}}}]}}, {"match_phrase": {"body": "gamma delta"}}, {"term": {"n": 7}}]):
            n.req("PUT", f"/percol/_doc/q{i}?refresh=true", {"q": q})
    both("POST", "/percol/_search", {"query": {"percolate": {"field": "q", "document": {"body": "alpha gamma delta", "n": 7}}}, "sort": [{"_id": "asc"}]}, label="percolate")
    both("POST", "/percol/_search", {"query": {"percolate": {"field": "q", "documents": [{"body": "beta", "n": 9}, {"body": "gamma delta", "n": 1}]}}, "sort": [{"_id": "asc"}], "highlight": {"fields": {"body": {}}}}, label="percolate multi + highlight")
    # term vectors, mtermvectors
    for doc in ("d1", "d2", "d9"):
        both("GET", f"/api/_termvectors/{doc}?routing=r{int(doc[1:]) % 3}&fields=body,title&positions=true&offsets=true&payloads=true&term_statistics=true&field_statistics=true", label=f"termvectors {doc}")
    both("POST", "/_termvectors", {"doc": {"body": "alpha beta alpha", "title": "x"}}, label="termvectors artificial") if False else None
    both("POST", "/api/_termvectors", {"doc": {"body": "alpha beta alpha gamma"}, "fields": ["body"], "term_statistics": True, "field_statistics": True, "positions": True, "offsets": True, "per_field_analyzer": {"body": "whitespace"}}, label="termvectors artificial")
    both("POST", "/api/_mtermvectors", {"docs": [{"_id": "d1", "routing": "r1", "fields": ["body"]}, {"_id": "d2", "routing": "r2", "fields": ["body"], "term_statistics": True}]}, label="mtermvectors")
    # _analyze, field_caps, mget, msearch
    both("POST", "/api/_analyze", {"analyzer": "standard", "text": "The QUICK brown-fox jumped 3.5 times"}, label="_analyze")
    both("POST", "/api/_analyze", {"field": "title", "text": "Alpha BETA"}, label="_analyze field")
    both("POST", "/api/_analyze", {"tokenizer": "standard", "filter": ["lowercase", {"type": "stop"}], "text": "The Alpha", "explain": True}, label="_analyze explain")
    both("GET", "/api/_field_caps?fields=*", label="field_caps")
    both("POST", "/api/_mget", {"docs": [{"_id": "d1", "routing": "r1"}, {"_id": "d2", "routing": "r2", "_source": ["n"]}, {"_id": "nope"}, {"_id": "d4", "routing": "r1", "stored_fields": ["title"]}]}, label="mget")
    lines = "\n".join([json.dumps({"index": "api"}), json.dumps({"query": {"term": {"tag": "alpha"}}, "size": 0}), json.dumps({"index": "api"}), json.dumps({"query": {"bad": {}}}), json.dumps({"index": "missing_idx"}), json.dumps({"query": {"match_all": {}}})]) + "\n"
    both("POST", "/_msearch", lines.encode(), raw=True, label="msearch")
    # write APIs: create / update / delete / versions / routing
    both("PUT", "/api/_create/new1?routing=r1", {"title": "created", "n": 1}, label="create")
    both("PUT", "/api/_create/new1?routing=r1", {"title": "created again"}, label="create conflict")
    both("PUT", "/api/_doc/new1?routing=r1&if_seq_no=0&if_primary_term=1", {"title": "occ"}, label="occ conflict", ign=IGN | {"reason"})
    both("POST", "/api/_update/new1?routing=r1", {"doc": {"n": 2, "extra": "x"}}, label="update doc")
    both("POST", "/api/_update/new1?routing=r1", {"doc": {"n": 2, "extra": "x"}}, label="update noop")
    both("POST", "/api/_update/new1?routing=r1", {"script": {"source": "ctx._source.n += params.d; ctx._source.tags = [ctx._source.n, 'a']", "params": {"d": 5}}}, label="update script")
    both("POST", "/api/_update/new2?routing=r1", {"doc": {"n": 9}, "doc_as_upsert": True}, label="upsert doc")
    both("POST", "/api/_update/new3?routing=r1", {"script": {"source": "ctx._source.n = 1"}, "upsert": {"n": 100}, "scripted_upsert": False}, label="upsert scripted")
    both("POST", "/api/_update/new1?routing=r1", {"script": {"source": "ctx.op = 'delete'"}}, label="update delete")
    both("POST", "/api/_update/none?routing=r1", {"doc": {"n": 1}}, label="update missing")
    both("PUT", "/api/_doc/ext1?routing=r1&version=5&version_type=external", {"n": 1}, label="external version")
    both("PUT", "/api/_doc/ext1?routing=r1&version=5&version_type=external", {"n": 2}, label="external version conflict")
    both("PUT", "/api/_doc/ext1?routing=r1&version=7&version_type=external_gte", {"n": 3}, label="external_gte")
    both("DELETE", "/api/_doc/ext1?routing=r1&version=6&version_type=external", label="delete ext version conflict")
    both("DELETE", "/api/_doc/ext1?routing=r1", label="delete")
    both("DELETE", "/api/_doc/ext1?routing=r1", label="delete again")
    both("GET", "/api/_doc/d3?routing=r0&_source_includes=title,n", label="get source filter")
    both("GET", "/api/_doc/d3", label="get wrong routing")
    both("HEAD", "/api/_doc/d3?routing=r0", label="head")
    both("POST", "/api/_refresh", label="refresh")
    # bulk with mixed errors
    lines = "\n".join([json.dumps({"index": {"_id": "b1", "routing": "r1"}}), json.dumps({"n": 1}), json.dumps({"create": {"_id": "b1", "routing": "r1"}}), json.dumps({"n": 2}),
                       json.dumps({"update": {"_id": "b1", "routing": "r1"}}), json.dumps({"doc": {"n": 3}}), json.dumps({"delete": {"_id": "nope", "routing": "r1"}}),
                       json.dumps({"index": {"_id": "b2", "routing": "r1"}}), json.dumps({"n": "notanumber"}), json.dumps({"update": {"_id": "nodoc", "routing": "r1"}}), json.dumps({"doc": {"n": 3}}),
                       json.dumps({"index": {"_id": "b3", "routing": "r1", "version": 3, "version_type": "external"}}), json.dumps({"n": 5})]) + "\n"
    both("POST", "/api/_bulk?refresh=true", lines.encode(), raw=True, label="bulk mixed", ign=IGN | {"reason", "caused_by"})
    # by-query operations
    both("POST", "/api/_update_by_query?refresh=true&conflicts=proceed", {"query": {"term": {"tag": "alpha"}}, "script": {"source": "ctx._source.n += 1000"}}, label="update_by_query", ign=IGN | {"retries", "throttled_millis", "requests_per_second", "throttled_until_millis"})
    both("POST", "/api/_search", {"size": 0, "aggs": {"s": {"sum": {"field": "n"}}, "c": {"filter": {"range": {"n": {"gte": 1000}}}}}}, label="after update_by_query")
    both("POST", "/api/_delete_by_query?refresh=true&conflicts=proceed", {"query": {"range": {"n": {"gte": 1000}}}}, label="delete_by_query", ign=IGN | {"retries", "throttled_millis", "requests_per_second", "throttled_until_millis"})
    both("POST", "/api/_count", label="count after delete_by_query")
    both("POST", "/_reindex?refresh=true", {"source": {"index": "api", "query": {"term": {"tag": "beta"}}}, "dest": {"index": "api_dst"}}, label="reindex", ign=IGN | {"retries", "throttled_millis", "requests_per_second", "throttled_until_millis"})
    both("POST", "/api_dst/_search", {"size": 0, "aggs": {"t": {"terms": {"field": "grp", "size": 10, "order": [{"_key": "asc"}]}}}}, label="reindexed content")
    # search misc: source filtering, fields API, script_fields, docvalue formats, stored fields, runtime fields, search templates
    both("POST", "/api/_search", {"size": 3, "sort": [{"_seq_no": "asc"}], "_source": {"includes": ["ti*", "kids.name"], "excludes": ["title.raw"]}, "fields": [{"field": "ts", "format": "yyyy/MM"}, "n"], "docvalue_fields": [{"field": "ts", "format": "epoch_second"}, "tag"]}, label="source/fields/docvalue")
    both("POST", "/api/_search", {"size": 3, "sort": [{"_seq_no": "asc"}], "runtime_mappings": {"n2": {"type": "long", "script": {"source": "emit(doc['n'].value * 2)"}}}, "fields": ["n2"], "query": {"range": {"n2": {"gte": 100}}}, "aggs": {"m": {"max": {"field": "n2"}}}}, label="runtime_mappings")
    both("POST", "/api/_search/template", {"source": {"query": {"match": {"title": "{{q}}"}}, "size": "{{sz}}"}, "params": {"q": "alpha", "sz": 2}}, label="search template")
    both("POST", "/api/_search", {"size": 2, "query": {"match": {"title": "alpha"}}, "profile": True, "sort": [{"_seq_no": "asc"}]}, label="profile (shape only)", ign=IGN | {"profile"})
    both("POST", "/api/_search", {"size": 0, "track_total_hits": 10, "query": {"match_all": {}}}, label="track_total_hits cap")
    both("POST", "/api/_search?preference=_shards:0", {"size": 0, "query": {"match_all": {}}}, label="preference shard")
    both("POST", "/api/_search", {"size": 5, "query": {"match_all": {}}, "sort": [{"_seq_no": "asc"}], "slice": {"id": 0, "max": 2}}, label="slice") if False else None
    both("POST", "/api/_search", {"size": 0, "query": {"match_all": {}}, "aggs": {"a": {"terms": {"field": "grp", "size": 3}, "meta": {"m": 1}}}, "stats": ["g1"]}, label="agg meta")
    both("POST", "/api/_search?typed_keys=true", {"size": 0, "aggs": {"a": {"terms": {"field": "grp", "size": 3, "order": [{"_key": "asc"}]}, "aggs": {"p": {"percentiles": {"field": "n"}}, "s": {"stats": {"field": "n"}}}}}}, label="typed_keys")
    print(f"scenarios: {total}, mismatches: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
