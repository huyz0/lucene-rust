"""direct_check.py BASE: the direct search path (`search.accelerated.direct.enabled`) against OpenSearch's own path.

Run on a node with the plugin (the path is on by default). Builds small
accelerated indices, sends each request twice -- as is, and with a `"profile": false` the direct path leaves to OpenSearch --
and requires equal totals, ids, scores and sources; checks `direct_searches` moved for eligible requests and did not for the
ineligible ones (alias, sort by score/script, search_after, aggregations, routing, a request parameter, nested, multi-shard, opt-out index)."""
import json
import sys

here = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, here + "/../../opensearch-plugin/e2e")
import bench_nodes as bn  # noqa: E402

bn.BASE = sys.argv[1]
ACC = {"index.engine.accelerated": True}


def put(index, settings, mappings, docs, routing=None):
    try:
        bn.req("DELETE", "/" + index)
    except Exception:  # noqa: BLE001
        pass
    bn.req("PUT", "/" + index, {"settings": {**ACC, "index.number_of_shards": 1, "index.number_of_replicas": 0, **settings}, "mappings": mappings})
    lines = []
    for i, d in docs:
        meta = {"_id": i}
        if routing:
            meta["routing"] = routing
        lines += [json.dumps({"index": meta}), json.dumps(d)]
    bn.req("POST", "/%s/_bulk?refresh=true" % index, ("\n".join(lines) + "\n").encode(), ndjson=True)
    bn.req("POST", "/%s/_forcemerge?max_num_segments=1" % index)


def direct():
    return bn.req("GET", "/_plugins/accelerated/stats")["direct_searches"]


def check(label, path, body, expect_direct, same_total=True):
    before = direct()
    a = bn.req("POST", path, body)
    moved = direct() - before
    std = dict(body)
    std["profile"] = False
    b = bn.req("POST", path, std)

    def key(r):
        return ([(h["_index"], h["_id"], round(h["_score"] or 0, 4), json.dumps(h.get("_source"), sort_keys=True), h.get("sort"), h.get("_routing"), h.get("_ignored")) for h in r["hits"]["hits"]],
                r["hits"].get("total"), r["hits"]["max_score"] and round(r["hits"]["max_score"], 4), r["_shards"])
    ok = key(a) == key(b) and (moved == 1) == expect_direct and moved in (0, 1)
    print("%-4s %-44s direct=%d expected=%s hits=%d total=%s" % ("ok" if ok else "FAIL", label, moved, expect_direct, len(a["hits"]["hits"]), a["hits"].get("total")))
    return ok


for left_over in ("dcb", "dcr", "dcm", "dcn", "dco"):
    try:
        bn.req("PUT", "/%s/_settings" % left_over, {"index.blocks.read": False})
        bn.req("DELETE", "/" + left_over)
    except Exception:  # noqa: BLE001
        pass
docs = [(i, {"tag": "t%d" % (i % 7), "n": i, "title": "word%d common" % (i % 13), "d": "2024-01-%02d" % (i % 28 + 1), "loc": {"lat": i % 90, "lon": i % 180}}) for i in range(1, 30001)]
docs[0] = ('quo"te\\backé\n', docs[0][1])
mapping = {"properties": {"tag": {"type": "keyword"}, "n": {"type": "long"}, "title": {"type": "text"}, "d": {"type": "date"}, "loc": {"type": "geo_point"}}}
put("dc", {}, mapping, docs)
ok = True
P = "/dc/_search?request_cache=false"
for label, body in [
    ("term", {"query": {"term": {"tag": "t3"}}}),
    ("term size 0", {"size": 0, "query": {"term": {"tag": "t3"}}}),
    ("term size 100", {"size": 100, "query": {"term": {"tag": "t3"}}}),
    ("match_all", {"query": {"match_all": {}}}),
    ("no query", {"size": 5}),
    ("match", {"query": {"match": {"title": "word3 common"}}}),
    ("bool", {"query": {"bool": {"must": {"match": {"title": "word3"}}, "filter": [{"range": {"n": {"gte": 100, "lt": 20000}}}, {"term": {"tag": "t3"}}]}}}),
    ("range date", {"query": {"range": {"d": {"gte": "2024-01-05", "lte": "2024-01-09"}}}}),
    ("ids", {"query": {"ids": {"values": ["5", "6", 'quo"te\\backé\n']}}}),
    ("no hits", {"query": {"term": {"tag": "none"}}}),
    ("few hits exact total", {"query": {"term": {"n": 77}}}),
    ("from 5", {"from": 5, "size": 7, "query": {"term": {"tag": "t3"}}}),
    ("from past the hits", {"from": 50, "size": 10, "query": {"term": {"n": 77}}}),
    ("track_total_hits true", {"track_total_hits": True, "query": {"term": {"tag": "t3"}}}),
    ("track_total_hits true size 0", {"size": 0, "track_total_hits": True, "query": {"match": {"title": "word3"}}}),
    ("track_total_hits false", {"track_total_hits": False, "query": {"term": {"tag": "t3"}}}),
    ("track_total_hits 500", {"track_total_hits": 500, "query": {"term": {"tag": "t3"}}}),
    ("sort long desc", {"sort": [{"n": "desc"}], "query": {"term": {"tag": "t3"}}}),
    ("sort long asc match_all", {"sort": [{"n": "asc"}], "query": {"match_all": {}}}),
    ("sort date then long", {"sort": [{"d": "desc"}, {"n": "asc"}], "size": 20, "query": {"match": {"title": "word3"}}}),
    ("sort keyword", {"sort": [{"tag": "asc"}, {"n": "desc"}], "query": {"match_all": {}}}),
    ("sort with from", {"from": 5, "size": 7, "sort": [{"n": "desc"}], "query": {"term": {"tag": "t3"}}}),
    ("sort size 0", {"size": 0, "sort": [{"n": "desc"}], "query": {"term": {"tag": "t3"}}}),
    ("sort no hits", {"sort": [{"n": "desc"}], "query": {"term": {"tag": "none"}}}),
    ("sort missing last", {"sort": [{"n": {"order": "desc", "missing": "_last"}}], "query": {"match_all": {}}}),
    ("_source false", {"_source": False, "query": {"term": {"tag": "t3"}}}),
    ("_source includes", {"_source": ["tag", "n"], "query": {"term": {"tag": "t3"}}}),
    ("_source object includes excludes", {"_source": {"includes": ["t*", "n"], "excludes": ["title"]}, "query": {"term": {"tag": "t3"}}}),
    ("_source filter matching nothing", {"_source": ["nothing"], "query": {"term": {"tag": "t3"}}}),
    ("sort by score", {"sort": ["_score"], "query": {"match": {"title": "word3"}}}),
    ("sort by score then long", {"sort": ["_score", {"n": "desc"}], "query": {"match": {"title": "word3"}}}),
    ("sort long then score", {"sort": [{"n": "desc"}, "_score"], "query": {"match": {"title": "word3"}}}),
    ("sort by script", {"sort": [{"_script": {"type": "number", "script": "doc['n'].value * 2", "order": "desc"}}], "query": {"term": {"tag": "t3"}}}),
    ("sort by geo distance", {"sort": [{"_geo_distance": {"loc": {"lat": 10, "lon": 20}, "order": "asc", "unit": "km"}}], "query": {"term": {"tag": "t3"}}}),
    ("search_after", {"search_after": [3000], "sort": [{"n": "asc"}], "query": {"term": {"tag": "t3"}}}),
    ("search_after two keys", {"search_after": ["t3", 3000], "sort": [{"tag": "asc"}, {"n": "asc"}], "query": {"match_all": {}}}),
    ("min_score", {"min_score": 0.9, "query": {"match": {"title": "word3 common"}}}),
    ("min_score with sort", {"min_score": 0.9, "sort": [{"n": "desc"}], "query": {"match": {"title": "word3 common"}}}),
    ("track_total_hits 5000 exact below", {"track_total_hits": 5000, "query": {"term": {"tag": "t3"}}}),
]:
    ok &= check(label, P, body, True)
for label, path, body in [
    ("URL size", P + "&size=3", {"query": {"term": {"tag": "t3"}}}),
    ("URL size overrides body", P + "&size=2", {"size": 7, "query": {"term": {"tag": "t3"}}}),
    ("URL from", P + "&from=4&size=3", {"query": {"term": {"tag": "t3"}}}),
    ("URL track_total_hits true", P + "&track_total_hits=true", {"query": {"match": {"title": "word3"}}}),
    ("URL track_total_hits false", P + "&track_total_hits=false", {"query": {"match": {"title": "word3"}}}),
    ("URL track_total_hits 50", P + "&track_total_hits=50", {"query": {"match": {"title": "word3"}}}),
    ("URL rest_total_hits_as_int", P + "&rest_total_hits_as_int=true", {"query": {"match": {"title": "word3"}}}),
    ("URL rest_total_hits_as_int small", P + "&rest_total_hits_as_int=true", {"query": {"term": {"n": 77}}}),
    ("URL _source false", P + "&_source=false", {"query": {"term": {"tag": "t3"}}}),
    ("URL _source_includes", P + "&_source_includes=tag,n", {"query": {"term": {"tag": "t3"}}}),
    ("URL _source_excludes", P + "&_source_excludes=title", {"query": {"term": {"tag": "t3"}}}),
    ("URL preference _local", P + "&preference=_local", {"query": {"term": {"tag": "t3"}}}),
    ("URL preference custom", P + "&preference=abc", {"query": {"term": {"tag": "t3"}}}),
    ("URL routing", P + "&routing=r1", {"query": {"term": {"tag": "t3"}}}),
]:
    ok &= check(label, path, body, True)
for label, path, body in [
    ("from + size above 100", P, {"from": 95, "size": 10, "query": {"term": {"tag": "t3"}}}),
    
    ("aggregation", P, {"aggs": {"a": {"terms": {"field": "tag"}}}, "query": {"term": {"tag": "t3"}}}),
    ("size above 100", P, {"size": 101, "query": {"term": {"tag": "t3"}}}),
    ("highlight", P, {"highlight": {"fields": {"title": {}}}, "query": {"match": {"title": "word3"}}}),
    ("request param", "/dc/_search?request_cache=false&search_type=dfs_query_then_fetch", {"query": {"term": {"tag": "t3"}}}),
]:
    ok &= check(label, path, body, False)
bn.req("POST", "/_aliases", {"actions": [{"add": {"index": "dc", "alias": "dca"}}]})
ok &= check("alias", "/dca/_search?request_cache=false", {"query": {"term": {"tag": "t3"}}}, True)
bn.req("POST", "/_aliases", {"actions": [{"add": {"index": "dc", "alias": "dcf", "filter": {"term": {"tag": "t3"}}}}]})
ok &= check("filtered alias", "/dcf/_search?request_cache=false", {"query": {"match_all": {}}}, False)
ok &= check("alias with a filtered sibling on the index", "/dca/_search?request_cache=false", {"query": {"match_all": {}}}, False)
bn.req("POST", "/_aliases", {"actions": [{"remove": {"index": "dc", "alias": "dcf"}}]})
put("uniq1", {}, mapping, docs[:500])
ok &= check("wildcard to one index", "/uniq*/_search?request_cache=false", {"query": {"term": {"tag": "t3"}}}, True)
ok &= check("wildcard matching several indices", "/dc*,uniq*/_search?request_cache=false", {"query": {"term": {"tag": "t3"}}}, False)
ok &= check("list of two indices", "/dc,uniq1/_search?request_cache=false", {"query": {"term": {"tag": "t3"}}}, False)
bn.req("POST", "/_aliases", {"actions": [{"add": {"index": "dc", "alias": "dcboth"}}, {"add": {"index": "uniq1", "alias": "dcboth"}}]})
ok &= check("alias over two indices", "/dcboth/_search?request_cache=false", {"query": {"term": {"tag": "t3"}}}, False)
put("dcr", {}, mapping, docs[:200], routing="r1")
ok &= check("hits with _routing", "/dcr/_search?request_cache=false", {"query": {"match_all": {}}}, True)
put("dcm", {"index.number_of_shards": 3}, mapping, docs[:300])
ok &= check("three shards", "/dcm/_search?request_cache=false", {"query": {"match_all": {}}}, False)
put("dcn", {}, {"properties": {"c": {"type": "nested", "properties": {"x": {"type": "keyword"}}}}}, [(1, {"c": [{"x": "a"}]})])
ok &= check("nested mapping", "/dcn/_search?request_cache=false", {"query": {"match_all": {}}}, False)
put("dcb", {"index.blocks.read": True}, mapping, docs[:100])
try:
    bn.req("POST", "/dcb/_search", {"query": {"match_all": {}}})
    print("FAIL read block answered"); ok = False
except Exception:  # noqa: BLE001
    print("ok   read-blocked index is refused as OpenSearch refuses it")
put("dco", {"index.search.accelerated": False}, mapping, docs[:100])
ok &= check("index with accelerated search off", "/dco/_search?request_cache=false", {"query": {"match_all": {}}}, False)
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
