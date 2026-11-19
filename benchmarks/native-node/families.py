"""Feature-family sweep: every search family, stock against the plugin, through REST on a node started by
native-node.sh. Complements ab.py (the 63 core shapes on index `bench`) with the families that file does not cover.

  families.py load BASE CORPUS [N] [SETTINGS_JSON]  index `fam` (N docs, default 300000: nested, completion, ip,
                                                     boolean, date, geo) and `jn` (join, N/3 parents, 3 children each)
                                                     from the bench corpus (bench_nodes.py gen)
  families.py time BASE LABEL OUT                    every shape below on `bench`, `fam` and `jn`
  families.py report 'OUT_STOCK-r*.json' 'OUT_NEW-r*.json'   per family: stock, new, ratio on p50 and p99

Environment: ROUNDS (timed requests per shape, 100), WARMUP (30), WARMUP_S (1.5), FAMILIES (substrings)."""
import glob
import json
import os
import re
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../../opensearch-plugin/e2e"))
import bench_nodes as bn  # noqa: E402

ROUNDS = int(os.environ.get("ROUNDS", "100"))
WARMUP = int(os.environ.get("WARMUP", "30"))
WARMUP_S = float(os.environ.get("WARMUP_S", "1.5"))

FAM_MAPPING = {
    "properties": {
        "body": {"type": "text"},
        "title": {"type": "text", "fields": {"raw": {"type": "keyword"}, "sayt": {"type": "search_as_you_type"}}},
        "tag": {"type": "keyword"}, "category": {"type": "keyword"},
        "num": {"type": "long"}, "price": {"type": "double"}, "ts": {"type": "date"},
        "loc": {"type": "geo_point"}, "ip": {"type": "ip"}, "flag": {"type": "boolean"},
        "comments": {"type": "nested", "properties": {
            "author": {"type": "keyword"}, "text": {"type": "text"}, "stars": {"type": "integer"}}},
    }
}
SG_MAPPING = {"properties": {"title": {"type": "text"}, "sugg": {"type": "completion"}}}
JN_MAPPING = {"properties": {
    "rel": {"type": "join", "relations": {"parent": "child"}},
    "title": {"type": "text"}, "tag": {"type": "keyword"}, "num": {"type": "long"},
    "who": {"type": "keyword"}, "stars": {"type": "integer"}}}


def derive(doc, i):
    words = doc["body"].split()
    doc["comments"] = [{"author": "u%d" % ((i * 7 + k * 13) % 100), "text": " ".join(words[k * 3:k * 3 + 6]),
                        "stars": (i + k) % 5 + 1} for k in range(3)]
    doc["ip"] = "10.%d.%d.%d" % ((i >> 16) & 255, (i >> 8) & 255, i & 255)
    doc["flag"] = (i % 3) == 0
    return doc


def bulk(index, lines):
    res = bn.req("POST", "/%s/_bulk" % index, ("\n".join(lines) + "\n").encode(), ndjson=True, timeout=600)
    if res["errors"]:
        raise RuntimeError("bulk errors on " + index)


def load(base, corpus, n, settings):
    bn.BASE = base
    base_st = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}
    base_st.update(json.loads(settings))
    # the accelerated engine does not write completion fields: `sg` stays on the standard engine on both nodes
    sg_st = {k: v for k, v in base_st.items() if k != "index.engine.accelerated"}
    for idx, mp in (("fam", FAM_MAPPING), ("jn", JN_MAPPING), ("sg", SG_MAPPING)):
        try:
            bn.req("DELETE", "/" + idx)
        except RuntimeError:
            pass
        bn.req("PUT", "/" + idx, {"settings": sg_st if idx == "sg" else base_st, "mappings": mp})
    sbuf = []
    t0 = time.perf_counter()
    files = sorted(glob.glob(os.path.join(corpus, "bulk-*.ndjson")))
    i = 0
    buf = []
    jbuf = []
    jparents = n // 3
    for f in files:
        with open(f) as fh:
            ls = fh.read().splitlines()
        for a in range(0, len(ls), 2):
            if i >= n:
                break
            doc = derive(json.loads(ls[a + 1]), i)
            buf.append(json.dumps({"index": {"_id": str(i)}}))
            buf.append(json.dumps(doc, separators=(",", ":")))
            if i < jparents:
                jbuf.append(json.dumps({"index": {"_id": "p%d" % i, "routing": "p%d" % i}}))
                jbuf.append(json.dumps({"rel": "parent", "title": doc["title"], "tag": doc["tag"], "num": doc["num"]}))
                for k in range(3):
                    jbuf.append(json.dumps({"index": {"_id": "c%d_%d" % (i, k), "routing": "p%d" % i}}))
                    jbuf.append(json.dumps({"rel": {"name": "child", "parent": "p%d" % i},
                                            "who": "u%d" % ((i + k) % 100), "stars": (i + k) % 5 + 1,
                                            "title": " ".join(doc["body"].split()[k:k + 4])}))
            if i < 100000:
                sbuf.append(json.dumps({"index": {"_id": str(i)}}))
                sbuf.append(json.dumps({"title": doc["title"], "sugg": {"input": [doc["title"].split()[0] + "x" + str(i % 50)], "weight": i % 100}}))
                if len(sbuf) >= 4000:
                    bulk("sg", sbuf)
                    sbuf = []
            i += 1
            if len(buf) >= 4000:
                bulk("fam", buf)
                buf = []
            if len(jbuf) >= 4000:
                bulk("jn", jbuf)
                jbuf = []
    if buf:
        bulk("fam", buf)
    if jbuf:
        bulk("jn", jbuf)
    if sbuf:
        bulk("sg", sbuf)
    for idx in ("fam", "jn", "sg"):
        bn.req("POST", "/%s/_forcemerge?max_num_segments=1" % idx, timeout=3600)
        bn.req("POST", "/%s/_refresh" % idx, timeout=3600)
    print("%d docs (+%d join parents) in %.0fs" % (i, jparents, time.perf_counter() - t0))


def Q(q, **kw):
    return {"query": q, **kw}


def sh():
    """(family, name, index, body, params)"""
    S = []

    def add(fam, name, body, idx="bench", params=""):
        S.append((fam, name, idx, body, params))

    # --- term-level and multi-term queries on `bench` (1M docs)
    add("wildcard", "wildcard keyword g1*", Q({"wildcard": {"tag": "g1*"}}))
    add("wildcard", "wildcard keyword *5", Q({"wildcard": {"tag": "*5"}}))
    add("wildcard", "wildcard text title t1z*", Q({"wildcard": {"title": "t1z*"}}))
    add("wildcard", "wildcard keyword c?3", Q({"wildcard": {"category": "c?3"}}))
    add("wildcard", "wildcard case-insensitive", Q({"wildcard": {"tag": {"value": "G1*", "case_insensitive": True}}}))
    add("regexp", "regexp keyword c[1-3][0-9]", Q({"regexp": {"category": "c[1-3][0-9]"}}))
    add("regexp", "regexp text title t1z[a-c].*", Q({"regexp": {"title": "t1z[a-c].*"}}))
    add("regexp", "regexp keyword g.5.*", Q({"regexp": {"tag": "g.5.*"}}))
    add("fuzzy", "fuzzy body t0x AUTO", Q({"fuzzy": {"body": {"value": "t01", "fuzziness": "AUTO"}}}))
    add("fuzzy", "fuzzy body t2s fuzziness 1", Q({"fuzzy": {"body": {"value": "t2t", "fuzziness": 1}}}))
    add("fuzzy", "match fuzziness AUTO 2 terms", Q({"match": {"body": {"query": "t1y t2s", "fuzziness": "AUTO"}}}))
    add("prefix", "prefix keyword tag g1", Q({"prefix": {"tag": "g1"}}))
    add("prefix", "prefix text title t1z (size 100)", Q({"prefix": {"title": "t1z"}}, size=100))
    add("terms", "terms keyword 20 values", Q({"terms": {"tag": ["g%x" % i for i in range(20)]}}))
    add("terms", "terms long 50 values", Q({"terms": {"num": list(range(1000, 51000, 1000))}}))
    add("exists", "exists price", Q({"exists": {"field": "price"}}))
    add("ids", "ids 20", Q({"ids": {"values": [str(i * 997) for i in range(20)]}}))
    add("range", "range date 1 month", Q({"range": {"ts": {"gte": 1_700_000_000_000, "lt": 1_702_592_000_000}}}))
    add("range", "range double", Q({"range": {"price": {"gte": 100, "lt": 110}}}))
    add("range", "range keyword", Q({"range": {"category": {"gte": "c10", "lt": "c20"}}}))
    add("range", "bool filter date + term", Q({"bool": {"filter": [{"range": {"ts": {"gte": 1_705_000_000_000}}}, {"term": {"category": "c7"}}]}}))
    # --- query_string and friends
    add("query_string", "query_string t1 AND t2", Q({"query_string": {"query": "t1 AND t2", "default_field": "body"}}))
    add("query_string", "query_string t1 OR t2 -t3", Q({"query_string": {"query": "t1 OR t2 -t3", "default_field": "body"}}))
    add("query_string", "query_string field:term + range", Q({"query_string": {"query": "category:c7 AND num:[1000 TO 200000]"}}))
    add("query_string", "query_string phrase", Q({"query_string": {"query": "\"t0 t1\"", "default_field": "body"}}))
    add("query_string", "query_string wildcard+fuzzy", Q({"query_string": {"query": "t1z* OR t2s~1", "default_field": "title"}}))
    add("query_string", "query_string multi-field", Q({"query_string": {"query": "t3 t7", "fields": ["body", "title^2"]}}))
    add("query_string", "simple_query_string", Q({"simple_query_string": {"query": "t1 +t2 -t3", "fields": ["body"]}}))
    # --- scoring / compound
    add("bool", "bool must+should+must_not+filter", Q({"bool": {"must": {"match": {"body": "t1"}}, "should": [{"match": {"title": "t2"}}], "must_not": {"term": {"category": "c9"}}, "filter": {"range": {"num": {"lt": 500000}}}}}))
    add("bool", "bool 4 should minimum_should_match 2", Q({"bool": {"should": [{"term": {"body": "t%d" % i}} for i in range(1, 5)], "minimum_should_match": 2}}))
    add("bool", "dis_max 2 fields", Q({"dis_max": {"queries": [{"match": {"body": "t2"}}, {"match": {"title": "t2"}}], "tie_breaker": 0.3}}))
    add("bool", "boosting", Q({"boosting": {"positive": {"match": {"body": "t1"}}, "negative": {"match": {"body": "t2"}}, "negative_boost": 0.2}}))
    add("bool", "constant_score filter", Q({"constant_score": {"filter": {"term": {"category": "c3"}}}}))
    add("bool", "multi_match most_fields", Q({"multi_match": {"query": "t3 t7", "fields": ["body", "title"], "type": "most_fields"}}))
    add("bool", "multi_match cross_fields", Q({"multi_match": {"query": "t3 t7", "fields": ["body", "title"], "type": "cross_fields"}}))
    add("bool", "multi_match phrase_prefix", Q({"multi_match": {"query": "t0 t1", "fields": ["body", "title"], "type": "phrase_prefix"}}))
    add("bool", "combined_fields", Q({"combined_fields": {"query": "t3 t7", "fields": ["body", "title"]}}))
    add("bool", "match bool_prefix", Q({"match_bool_prefix": {"body": "t1 t2z"}}))
    add("bool", "more_like_this", Q({"more_like_this": {"fields": ["body"], "like": "t1 t2 t3 t5 t8 t13", "min_term_freq": 1, "min_doc_freq": 5}}))
    add("bool", "common-term OR 8 terms", Q({"match": {"body": "t0 t1 t2 t3 t4 t5 t6 t7"}}))
    add("function_score", "function_score field_value_factor", Q({"function_score": {"query": {"match": {"body": "t1 t2"}}, "field_value_factor": {"field": "num", "modifier": "log1p", "missing": 1}}}))
    add("function_score", "function_score gauss decay on price", Q({"function_score": {"query": {"match": {"body": "t1 t2"}}, "gauss": {"price": {"origin": 500, "scale": 100}}}}))
    add("function_score", "function_score weight+filter x3", Q({"function_score": {"query": {"match": {"body": "t1"}}, "functions": [{"filter": {"term": {"category": "c1"}}, "weight": 2}, {"filter": {"term": {"category": "c2"}}, "weight": 3}, {"filter": {"range": {"num": {"lt": 1000}}}, "weight": 5}], "score_mode": "sum", "boost_mode": "multiply"}}))
    add("function_score", "function_score random_score", Q({"function_score": {"query": {"match": {"body": "t1"}}, "random_score": {"seed": 7, "field": "_seq_no"}}}))
    add("rescore", "rescore phrase top 50", {"query": {"match": {"body": "t1 t2"}}, "rescore": {"window_size": 50, "query": {"rescore_query": {"match_phrase": {"body": {"query": "t1 t2", "slop": 2}}}, "query_weight": 0.7, "rescore_query_weight": 1.2}}})
    # --- scripts (Painless)
    add("script", "script_score doc['num']", Q({"script_score": {"query": {"match": {"body": "t1"}}, "script": {"source": "_score * Math.log(2 + doc['num'].value)"}}}))
    add("script", "script query (filter)", Q({"bool": {"filter": {"script": {"script": {"source": "doc['num'].value % 7 == 3"}}}}}))
    add("script", "script_fields x2", {"query": {"match": {"body": "t5"}}, "script_fields": {"a": {"script": {"source": "doc['price'].value * 2"}}, "b": {"script": {"source": "doc['num'].value + params.k", "params": {"k": 5}}}}})
    add("script", "sort by script", {"query": {"match": {"body": "t5"}}, "sort": {"_script": {"type": "number", "script": {"source": "doc['price'].value - doc['num'].value / 1000.0"}, "order": "asc"}}})
    add("script", "terms agg with script", {"size": 0, "aggs": {"a": {"terms": {"script": {"source": "doc['category'].value + '-' + (doc['num'].value % 3)"}, "size": 10}}}})
    add("script", "avg agg with script", {"size": 0, "aggs": {"a": {"avg": {"script": {"source": "doc['price'].value * 2"}}}}})
    add("script", "scripted_metric", {"size": 0, "query": {"match": {"body": "t8"}}, "aggs": {"a": {"scripted_metric": {"init_script": "state.s = 0.0", "map_script": "state.s += doc['price'].value", "combine_script": "return state.s", "reduce_script": "double t = 0; for (s in states) { t += s } return t"}}}})
    # --- sort, paging, totals
    add("sort", "sort keyword desc + num asc", {"query": {"match_all": {}}, "sort": [{"category": "desc"}, {"num": "asc"}], "size": 20})
    add("sort", "sort date desc size 100", {"query": {"match": {"body": "t3"}}, "sort": [{"ts": "desc"}], "size": 100})
    add("sort", "sort _score + _doc tiebreak", {"query": {"match": {"body": "t1 t2"}}, "sort": ["_score", {"num": "asc"}]})
    add("sort", "search_after by num", {"query": {"match_all": {}}, "sort": [{"num": "asc"}, {"_doc": "asc"}], "search_after": [500000, 12345], "size": 20})
    add("sort", "from 10000 size 10 by score", {"query": {"match": {"body": "t1 t2"}}, "from": 10000, "size": 10})
    add("sort", "size 1000 by score", {"query": {"match": {"body": "t1 t2"}}, "size": 1000})
    add("totals", "track_total_hits false", {"query": {"match": {"body": "t1 t2"}}, "track_total_hits": False})
    add("totals", "track_total_hits true (default 10k)", {"query": {"match": {"body": "t1 t2"}}, "track_total_hits": 10000})
    add("totals", "collapse on category", {"query": {"match": {"body": "t3"}}, "collapse": {"field": "category"}})
    add("totals", "post_filter + terms agg", {"query": {"match": {"body": "t3"}}, "post_filter": {"term": {"category": "c4"}}, "aggs": {"a": {"terms": {"field": "category"}}}})
    # --- fetch
    add("fetch", "fetch 10 _source", {"query": {"match": {"body": "t3"}}, "size": 10})
    add("fetch", "fetch 100 _source", {"query": {"match": {"body": "t3"}}, "size": 100})
    add("fetch", "fetch 100 _source includes 2 fields", {"query": {"match": {"body": "t3"}}, "size": 100, "_source": ["title", "num"]})
    add("fetch", "fetch 100 stored fields none + docvalue", {"query": {"match": {"body": "t3"}}, "size": 100, "_source": False, "docvalue_fields": ["num", "tag", "price"]})
    add("fetch", "fetch 10 explain", {"query": {"match": {"body": "t3"}}, "size": 10, "explain": True})
    add("fetch", "fetch 100 + inner source (fields)", {"query": {"match": {"body": "t3"}}, "size": 100, "fields": ["num", "ts"]})
    # --- highlighting (on `fam`: body has term vectors; `bench` body has none)
    add("highlight", "unified highlight body top 10", {"query": {"match": {"body": "t1 t2"}}, "highlight": {"fields": {"body": {}}}}, "fam")
    add("highlight", "unified highlight 100 hits", {"query": {"match": {"body": "t3"}}, "size": 100, "highlight": {"fields": {"body": {}}}}, "fam")
    add("highlight", "plain highlight body", {"query": {"match": {"body": "t1 t2"}}, "highlight": {"fields": {"body": {"type": "plain"}}}}, "fam")
    add("highlight", "unified phrase highlight", {"query": {"match_phrase": {"body": "t0 t1"}}, "highlight": {"fields": {"body": {}}}}, "fam")
    # --- suggest (on `fam`)
    add("suggest", "completion prefix", {"size": 0, "suggest": {"s": {"prefix": "t1x", "completion": {"field": "sugg", "size": 5}}}}, "sg")
    add("suggest", "completion fuzzy", {"size": 0, "suggest": {"s": {"prefix": "t1y", "completion": {"field": "sugg", "size": 5, "fuzzy": {"fuzziness": 1}}}}}, "sg")
    add("suggest", "term suggester", {"size": 0, "suggest": {"s": {"text": "t1z", "term": {"field": "title"}}}}, "sg")
    add("suggest", "phrase suggester", {"size": 0, "suggest": {"s": {"text": "t1 t2z", "phrase": {"field": "title", "size": 3}}}}, "sg")
    add("suggest", "search_as_you_type", {"query": {"multi_match": {"query": "t1z", "type": "bool_prefix", "fields": ["title.sayt", "title.sayt._2gram", "title.sayt._3gram"]}}}, "fam")
    # --- aggregations beyond the core set (on bench)
    A = lambda a, **kw: {"size": 0, "aggs": {"a": a}, **kw}  # noqa: E731
    add("agg", "value_count + min + max + avg", A({"value_count": {"field": "num"}}, aggs=None) if False else {"size": 0, "aggs": {"a": {"value_count": {"field": "num"}}, "b": {"min": {"field": "price"}}, "c": {"max": {"field": "price"}}, "d": {"avg": {"field": "price"}}}})
    add("agg", "extended_stats price", A({"extended_stats": {"field": "price"}}))
    add("agg", "percentile_ranks price", A({"percentile_ranks": {"field": "price", "values": [100, 500, 900]}}))
    add("agg", "median_absolute_deviation", A({"median_absolute_deviation": {"field": "price"}}))
    add("agg", "weighted_avg", A({"weighted_avg": {"value": {"field": "price"}, "weight": {"field": "num"}}}))
    add("agg", "top_metrics", A({"top_metrics": {"metrics": {"field": "price"}, "sort": {"num": "desc"}}}))
    add("agg", "string_stats tag", A({"string_stats": {"field": "tag"}}))
    add("agg", "matrix_stats", A({"matrix_stats": {"fields": ["num", "price"]}}))
    add("agg", "geo_bounds + geo_centroid", {"size": 0, "aggs": {"a": {"geo_bounds": {"field": "loc"}}, "b": {"geo_centroid": {"field": "loc"}}}})
    add("agg", "histogram num interval 10000", A({"histogram": {"field": "num", "interval": 10000}}))
    add("agg", "date_histogram day", A({"date_histogram": {"field": "ts", "calendar_interval": "day"}}))
    add("agg", "date_histogram hour fixed + tz", A({"date_histogram": {"field": "ts", "fixed_interval": "6h", "time_zone": "America/New_York"}}))
    add("agg", "auto_date_histogram 50", A({"auto_date_histogram": {"field": "ts", "buckets": 50}}))
    add("agg", "variable_width_histogram", A({"variable_width_histogram": {"field": "price", "buckets": 10}}))
    add("agg", "range price 5 ranges", A({"range": {"field": "price", "ranges": [{"to": 100}, {"from": 100, "to": 300}, {"from": 300, "to": 600}, {"from": 600, "to": 900}, {"from": 900}]}}))
    add("agg", "date_range 4", A({"date_range": {"field": "ts", "ranges": [{"to": 1_710_000_000_000}, {"from": 1_710_000_000_000, "to": 1_720_000_000_000}, {"from": 1_720_000_000_000, "to": 1_730_000_000_000}, {"from": 1_730_000_000_000}]}}))
    add("agg", "filter + filters", {"size": 0, "aggs": {"a": {"filter": {"term": {"category": "c1"}}}, "b": {"filters": {"filters": {"x": {"term": {"tag": "g1"}}, "y": {"range": {"num": {"lt": 1000}}}}}}}})
    add("agg", "missing + global", {"size": 0, "query": {"match": {"body": "t8"}}, "aggs": {"m": {"missing": {"field": "price"}}, "g": {"global": {}, "aggs": {"c": {"value_count": {"field": "num"}}}}}})
    add("agg", "terms include/exclude regex", A({"terms": {"field": "tag", "include": "g1.*", "exclude": "g1[0-4].*", "size": 20}}))
    add("agg", "terms order by sub-agg avg", A({"terms": {"field": "category", "order": {"p": "desc"}}, "aggs": {"p": {"avg": {"field": "price"}}}}))
    add("agg", "terms size 1000 on tag", A({"terms": {"field": "tag", "size": 1000}}))
    add("agg", "terms min_doc_count + shard_size", A({"terms": {"field": "tag", "size": 10, "shard_size": 100, "min_doc_count": 1000}}))
    add("agg", "rare_terms tag", A({"rare_terms": {"field": "tag", "max_doc_count": 20}}))
    add("agg", "multi_terms category+tag", A({"multi_terms": {"terms": [{"field": "category"}, {"field": "tag"}], "size": 20}}))
    add("agg", "significant_terms tag (fg t8)", {"size": 0, "query": {"match": {"body": "t8"}}, "aggs": {"a": {"significant_terms": {"field": "tag", "size": 10}}}})
    add("agg", "significant_text body (fg t8)", {"size": 0, "query": {"match": {"body": "t8"}}, "aggs": {"a": {"significant_text": {"field": "body", "size": 10, "filter_duplicate_text": False}}}})
    add("agg", "sampler > terms", {"size": 0, "query": {"match": {"body": "t3"}}, "aggs": {"s": {"sampler": {"shard_size": 1000}, "aggs": {"a": {"terms": {"field": "tag"}}}}}})
    add("agg", "diversified_sampler", {"size": 0, "query": {"match": {"body": "t3"}}, "aggs": {"s": {"diversified_sampler": {"shard_size": 1000, "field": "category"}, "aggs": {"a": {"terms": {"field": "tag"}}}}}})
    add("agg", "adjacency_matrix", A({"adjacency_matrix": {"filters": {"a": {"term": {"category": "c1"}}, "b": {"term": {"category": "c2"}}, "c": {"range": {"num": {"lt": 5000}}}}}}))
    add("agg", "geo_distance agg 3 rings", A({"geo_distance": {"field": "loc", "origin": {"lat": 10, "lon": 20}, "unit": "km", "ranges": [{"to": 500}, {"from": 500, "to": 2000}, {"from": 2000}]}}))
    add("agg", "geohash_grid precision 3", A({"geohash_grid": {"field": "loc", "precision": 3}}))
    add("agg", "geotile_grid zoom 5", A({"geotile_grid": {"field": "loc", "precision": 5}}))
    add("agg", "pipeline date_hist > sum > cumulative_sum", A({"date_histogram": {"field": "ts", "calendar_interval": "month"}, "aggs": {"s": {"sum": {"field": "price"}}, "c": {"cumulative_sum": {"buckets_path": "s"}}, "d": {"derivative": {"buckets_path": "s"}}}}))
    add("agg", "terms > avg + sibling avg_bucket", {"size": 0, "aggs": {"a": {"terms": {"field": "category", "size": 50}, "aggs": {"p": {"avg": {"field": "price"}}}}, "m": {"avg_bucket": {"buckets_path": "a>p"}}}})
    add("agg", "terms > bucket_sort", A({"terms": {"field": "tag", "size": 200}, "aggs": {"p": {"sum": {"field": "price"}}, "b": {"bucket_sort": {"sort": [{"p": "desc"}], "size": 10}}}}))
    add("agg", "terms on ip-less num > cardinality", A({"terms": {"field": "category"}, "aggs": {"c": {"cardinality": {"field": "tag"}}}}))
    add("agg", "agg on query t0 (dense)", {"size": 0, "query": {"match": {"body": "t0"}}, "aggs": {"a": {"terms": {"field": "category"}}, "b": {"avg": {"field": "price"}}}})
    # --- nested / join / ip / boolean (on fam, jn)
    add("nested", "nested term", Q({"nested": {"path": "comments", "query": {"term": {"comments.author": "u7"}}}}), "fam")
    add("nested", "nested bool author+stars", Q({"nested": {"path": "comments", "query": {"bool": {"must": [{"term": {"comments.author": "u7"}}, {"range": {"comments.stars": {"gte": 4}}}]}}}}), "fam")
    add("nested", "nested match text", Q({"nested": {"path": "comments", "query": {"match": {"comments.text": "t1 t2"}}}}), "fam")
    add("nested", "nested score_mode max + inner_hits", Q({"nested": {"path": "comments", "score_mode": "max", "query": {"match": {"comments.text": "t1"}}, "inner_hits": {"size": 3}}}), "fam")
    add("nested", "nested agg terms > reverse_nested", {"size": 0, "aggs": {"n": {"nested": {"path": "comments"}, "aggs": {"a": {"terms": {"field": "comments.author", "size": 10}, "aggs": {"r": {"reverse_nested": {}}}}}}}}, "fam")
    add("nested", "nested agg avg stars", {"size": 0, "aggs": {"n": {"nested": {"path": "comments"}, "aggs": {"a": {"avg": {"field": "comments.stars"}}}}}}, "fam")
    add("join", "has_child term", Q({"has_child": {"type": "child", "query": {"term": {"who": "u7"}}}}), "jn")
    add("join", "has_child score_mode sum", Q({"has_child": {"type": "child", "score_mode": "sum", "query": {"match": {"title": "t1"}}}}), "jn")
    add("join", "has_parent", Q({"has_parent": {"parent_type": "parent", "query": {"term": {"tag": "g1"}}}}), "jn")
    add("join", "parent_id", Q({"parent_id": {"type": "child", "id": "p77"}}), "jn")
    add("join", "children agg", {"size": 0, "aggs": {"c": {"children": {"type": "child"}, "aggs": {"a": {"terms": {"field": "who", "size": 10}}}}}}, "jn")
    add("join", "parent agg", {"size": 0, "query": {"term": {"who": "u7"}}, "aggs": {"p": {"parent": {"type": "child"}, "aggs": {"a": {"terms": {"field": "tag", "size": 10}}}}}}, "jn")
    add("other-types", "ip term", Q({"term": {"ip": "10.0.1.7"}}), "fam")
    add("other-types", "ip cidr", Q({"term": {"ip": "10.0.0.0/20"}}), "fam")
    add("other-types", "boolean term", Q({"term": {"flag": True}}), "fam")
    add("other-types", "ip range agg", {"size": 0, "aggs": {"a": {"ip_range": {"field": "ip", "ranges": [{"mask": "10.0.0.0/16"}, {"mask": "10.1.0.0/16"}]}}}}, "fam")
    # --- geo extras
    add("geo", "geo_polygon", Q({"bool": {"filter": {"geo_polygon": {"loc": {"points": [{"lat": 0, "lon": 0}, {"lat": 30, "lon": 10}, {"lat": 20, "lon": 40}, {"lat": -5, "lon": 25}]}}}}}))
    add("geo", "geo_distance sort", {"query": {"match": {"body": "t5"}}, "sort": [{"_geo_distance": {"loc": {"lat": 10, "lon": 20}, "order": "asc", "unit": "km"}}]})
    add("geo", "geo_distance 50km filter", Q({"bool": {"filter": {"geo_distance": {"distance": "50km", "loc": {"lat": 10, "lon": 20}}}}}))
    return S


def stats_snapshot():
    try:
        return bn.req("GET", "/_nodes/stats/os,process")
    except Exception:  # noqa: BLE001
        return {}


def time_all(base, label, out):
    bn.BASE = base
    want = [w for w in os.environ.get("FAMILIES", "").split(",") if w]
    res = {}
    for fam, name, idx, body, params in sh():
        if want and not any(w in fam or w in name for w in want):
            continue
        last = []
        path = "/%s/_search?request_cache=false%s" % (idx, params)

        def one():
            last[:] = [bn.req("POST", path, body)]

        try:
            wall = bn.bh.measure(one, ROUNDS, WARMUP, WARMUP_S)
        except Exception as e:  # noqa: BLE001
            res[name] = {"family": fam, "error": str(e)[:300]}
            print("  ERR %-44s %s" % (name, str(e)[:120]), flush=True)
            continue
        r = last[0]
        tot = r.get("hits", {}).get("total", {})
        res[name] = {"family": fam, **bn.bh.summarize(wall), "hits": tot.get("value") if isinstance(tot, dict) else tot, "took": r.get("took")}
        print("  %-12s %-44s %8.2f ms (p99 %.2f)" % (fam, name, res[name]["median_ms"], res[name]["p99_ms"]), flush=True)
    res["_stats"] = stats_snapshot()
    json.dump(res, open(out, "w"), indent=1)


def best(files):
    runs = [json.load(open(f)) for f in files]
    out = {}
    for k in runs[0]:
        if k.startswith("_"):
            continue
        rs = [r[k] for r in runs if k in r and "error" not in r[k]]
        if not rs:
            out[k] = {"error": runs[0][k].get("error"), "family": runs[0][k]["family"]}
            continue
        out[k] = {"family": rs[0]["family"], "p50": min(x["median_ms"] for x in rs), "p99": min(x["p99_ms"] for x in rs), "hits": rs[0]["hits"]}
    return out


def group(pattern):
    return sorted(glob.glob(pattern))


def report(stock_pat, new_pats):
    s = best(group(stock_pat))
    cols = [(p, best(group(p))) for p in new_pats]
    for p, c in cols:
        print("\n=== %s vs stock (best median of rounds; ratio = stock / new, >1 faster) ===" % p)
        print("%-12s %-44s %8s %8s %6s | %8s %8s %6s %s" % ("family", "shape", "stock50", "new50", "x50", "stock99", "new99", "x99", "note"))
        fams = {}
        for k, v in s.items():
            if "error" in v or "error" in c.get(k, {"error": 1}):
                print("%-12s %-44s  error: stock=%s new=%s" % (v["family"], k, v.get("error"), c.get(k, {}).get("error")))
                continue
            n = c[k]
            r50, r99 = v["p50"] / n["p50"], v["p99"] / n["p99"]
            note = "HITS DIFFER %s vs %s" % (v["hits"], n["hits"]) if v["hits"] != n["hits"] else ""
            print("%-12s %-44s %8.2f %8.2f %5.2fx | %8.2f %8.2f %5.2fx %s" % (v["family"], k, v["p50"], n["p50"], r50, v["p99"], n["p99"], r99, note))
            fams.setdefault(v["family"], []).append(r50)
        print("\nfamily geometric mean of x50:")
        import math
        for f, rs in fams.items():
            g = math.exp(sum(math.log(x) for x in rs) / len(rs))
            print("  %-12s n=%2d  geomean %.2fx  min %.2fx  >=1.5x: %d  1.0-1.5x: %d  <1.0x: %d" % (f, len(rs), g, min(rs), sum(x >= 1.5 for x in rs), sum(1.0 <= x < 1.5 for x in rs), sum(x < 1.0 for x in rs)))


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "load":
        load(a[2], a[3], int(a[4]) if len(a) > 4 else 300000, a[5] if len(a) > 5 else "{}")
    elif a[1] == "time":
        time_all(a[2], a[3], a[4])
    elif a[1] == "report":
        report(a[2], a[3:])
    else:
        sys.exit(__doc__)
