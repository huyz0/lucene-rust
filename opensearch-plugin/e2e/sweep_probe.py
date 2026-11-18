#!/usr/bin/env python3
"""Hand-written differential catalog for areas the random fuzzers do not generate: per-field
similarities, pipeline and rarely used aggregations, scoring wrappers, request-level options
(rescore, collapse, post_filter, docvalue/stored fields, search_after, pagination), and concurrent
segment search. Same requests go to a plugin node and a stock node; any difference is printed.

    python3 opensearch-plugin/e2e/sweep_probe.py [--plugin URL] [--stock URL] [--seed N] [--concurrent]
"""
import argparse
import json
import random
import sys

import fuzz_diff as fd

N = 1500


def Q(**k):
    return {"query": k.pop("q"), **k}


def cases(r):
    c = []
    add = lambda name, body, ordered=False, idx="sw", params="": c.append((name, idx, body, ordered, params))
    W = fd.WORDS
    w = lambda: r.choice(W[:20])
    big = {"size": N}
    # ---- scoring wrappers
    for sm in ("multiply", "sum", "avg", "first", "max", "min"):
        for bm in ("multiply", "sum", "avg", "max", "min", "replace"):
            add(f"function_score {sm}/{bm}", {**big, "query": {"function_score": {
                "query": {"match": {"body": w()}},
                "functions": [{"filter": {"term": {"tag": "red"}}, "weight": 3},
                              {"field_value_factor": {"field": "i", "missing": 1, "modifier": "log2p", "factor": 2}, "filter": {"range": {"n": {"gt": 0}}}},
                              {"gauss": {"price": {"origin": 10, "scale": 20, "decay": 0.3, "offset": 1}}},
                              {"exp": {"ts": {"origin": "2000-01-01", "scale": "3650d"}}},
                              {"linear": {"loc": {"origin": "0,0", "scale": "5000km"}}},
                              {"script_score": {"script": "Math.log(2 + doc['sh'].size() > 0 ? doc['sh'].value + 101 : 1)"}}],
                "score_mode": sm, "boost_mode": bm, "max_boost": 100, "min_score": 0.0, "boost": 1.5}}})
    add("function_score random_score seed", {**big, "query": {"function_score": {"query": {"match_all": {}}, "random_score": {"seed": 7, "field": "_seq_no"}}}})
    add("function_score fvf sqrt neg", {**big, "query": {"function_score": {"field_value_factor": {"field": "price", "modifier": "sqrt", "missing": 0}}}})
    add("script_score doc", {**big, "query": {"script_score": {"query": {"match": {"title": w()}}, "script": {"source": "_score * 2 + doc['i'].size()"}}}})
    add("script_score params", {**big, "query": {"script_score": {"query": {"match_all": {}}, "script": {"source": "params.a * (doc['n'].size() > 0 ? 1 : 0) + 1", "params": {"a": 2.5}}, "min_score": 2}}})
    add("distance_feature date", {**big, "query": {"bool": {"must": {"match_all": {}}, "should": {"distance_feature": {"field": "ts", "origin": "2000-01-01", "pivot": "100d"}}}}})
    add("distance_feature geo", {**big, "query": {"bool": {"must": {"match_all": {}}, "should": {"distance_feature": {"field": "loc", "origin": [10, 10], "pivot": "500km", "boost": 2}}}}})
    add("boosting", {**big, "query": {"boosting": {"positive": {"match": {"body": w()}}, "negative": {"match": {"body": w()}}, "negative_boost": 0.3}}})
    add("constant_score", {**big, "query": {"constant_score": {"filter": {"match": {"body": w()}}, "boost": 2.2}}})
    add("terms_set field", {**big, "query": {"terms_set": {"tags": {"terms": ["red", "blue", "green", "UPPER"], "minimum_should_match_field": "sh"}}}})
    add("terms_set script", {**big, "query": {"terms_set": {"tags": {"terms": ["red", "blue", "green"], "minimum_should_match_script": {"source": "Math.min(params.num_terms, 2)"}}}}})
    add("terms lookup-like many", {**big, "query": {"terms": {"tag": ["red", "Red", "RED", "a b"] * 50}}})
    add("terms boost", {**big, "query": {"terms": {"n": [0, 1, 7, 42], "boost": 3}}})
    add("rank_feature-less exists", {**big, "query": {"exists": {"field": "comments"}}})
    add("mlt", {**big, "query": {"more_like_this": {"fields": ["body"], "like": [{"_index": "sw", "_id": "d5"}, "alpha beta gamma"], "min_term_freq": 1, "min_doc_freq": 1, "max_query_terms": 12, "minimum_should_match": "30%"}}})
    for mm in ("best_fields", "most_fields", "cross_fields", "phrase", "phrase_prefix", "bool_prefix"):
        for tb in (0, 0.3):
            add(f"multi_match {mm} tb{tb}", {**big, "query": {"multi_match": {"query": f"{w()} {w()}", "fields": ["title^2", "body", "ws"], "type": mm, "tie_breaker": tb, "operator": "or"}}})
    add("multi_match fuzziness", {**big, "query": {"multi_match": {"query": "alpa betta", "fields": ["title", "body"], "fuzziness": "AUTO", "prefix_length": 1, "fuzzy_transpositions": False}}})
    add("multi_match zero_terms", {**big, "query": {"multi_match": {"query": "the of", "fields": ["body"], "zero_terms_query": "all"}}})
    add("match cutoff_frequency-ish msm", {**big, "query": {"match": {"body": {"query": f"{w()} {w()} {w()} {w()}", "minimum_should_match": "-1"}}}})
    add("match msm combo", {**big, "query": {"match": {"body": {"query": f"{w()} {w()} {w()} {w()} {w()}", "minimum_should_match": "2<-25% 9<-3"}}}})
    add("match synonyms-free fuzzy rewrite", {**big, "query": {"match": {"body": {"query": "quikc brwn", "fuzziness": 2, "fuzzy_rewrite": "top_terms_blended_freqs_3", "max_expansions": 10}}}})
    add("fuzzy rewrite constant", {**big, "query": {"fuzzy": {"body": {"value": "alpa", "fuzziness": 1, "rewrite": "constant_score"}}}})
    add("prefix rewrite scoring_boolean", {**big, "query": {"prefix": {"title": {"value": "a", "rewrite": "scoring_boolean"}}}})
    add("wildcard rewrite top_terms", {**big, "query": {"wildcard": {"title": {"value": "a*a", "rewrite": "top_terms_5"}}}})
    add("regexp flags", {**big, "query": {"regexp": {"tag": {"value": "r.*|b<1-3>", "flags": "ALL", "case_insensitive": True}}}})
    add("regexp interval", {**big, "query": {"regexp": {"n": {"value": "1"}}}})
    add("query_string complex", {**big, "query": {"query_string": {"query": "(alpha AND beta) OR title:\"quick brown\"~2 OR body:fox^2 -lazy +dog tag:[a TO r} n:>=3 price:(1.5 OR 0) title:alp* ws:/ALPH.*/", "default_operator": "AND", "fields": ["title^3", "body"], "analyze_wildcard": True, "lenient": True, "fuzzy_max_expansions": 5, "tie_breaker": 0.2}}})
    add("query_string fuzzy proximity", {**big, "query": {"query_string": {"query": "alpa~1 AND \"beta gamma\"~3 AND (delta OR NOT eta)", "default_field": "body", "phrase_slop": 1, "auto_generate_synonyms_phrase_query": False}}})
    add("query_string dates/ranges", {**big, "query": {"query_string": {"query": "ts:[2000-01-01 TO 2010-12-31] OR ts:>2025 OR _exists_:loc OR flag:true", "lenient": True, "time_zone": "+05:30"}}})
    add("query_string minimum_should_match", {**big, "query": {"query_string": {"query": "alpha beta gamma delta", "fields": ["body", "title"], "minimum_should_match": "75%", "type": "cross_fields"}}})
    add("simple_query_string flags", {**big, "query": {"simple_query_string": {"query": "alpha + beta | \"gamma delta\" -eta qu* fo~1", "fields": ["body", "title^2"], "flags": "AND|OR|NOT|PHRASE|PREFIX|FUZZY|NEAR|SLOP|WHITESPACE|PRECEDENCE", "default_operator": "and"}}})
    add("combined_fields", {**big, "query": {"combined_fields": {"query": f"{w()} {w()}", "fields": ["title^2", "body"], "operator": "and"}}})
    add("intervals nested", {**big, "query": {"intervals": {"body": {"all_of": {"ordered": True, "max_gaps": 2, "intervals": [{"match": {"query": w(), "max_gaps": 1}}, {"any_of": {"intervals": [{"match": {"query": w()}}, {"prefix": {"prefix": "a"}}]}}]}}}}})
    add("span_containing", {**big, "query": {"span_containing": {"little": {"span_term": {"body": w()}}, "big": {"span_near": {"clauses": [{"span_term": {"body": w()}}, {"span_term": {"body": w()}}], "slop": 5, "in_order": False}}}}})
    add("span_within/not/first/multi", {**big, "query": {"bool": {"should": [
        {"span_not": {"include": {"span_term": {"body": w()}}, "exclude": {"span_term": {"body": w()}}, "pre": 1, "post": 1}},
        {"span_first": {"match": {"span_term": {"body": w()}}, "end": 3}},
        {"span_multi": {"match": {"prefix": {"body": "al"}}}},
        {"span_or": {"clauses": [{"span_term": {"body": w()}}, {"span_term": {"body": w()}}]}},
        {"field_masking_span": {"query": {"span_term": {"title": w()}}, "field": "body"}}]}}})
    add("geo_distance sorted", {**big, "query": {"geo_distance": {"distance": "3000km", "loc": {"lat": 10, "lon": 10}, "distance_type": "plane", "validation_method": "COERCE"}}})
    add("geo_polygon", {**big, "query": {"geo_polygon": {"loc": {"points": [[-60, -30], [60, -30], [60, 50], [-60, 50]]}}}})
    add("ip cidr/range", {**big, "query": {"bool": {"should": [{"term": {"ip": "10.0.0.0/8"}}, {"range": {"ip": {"gte": "192.168.0.0", "lt": "2001:db8::ff"}}}]}}})
    add("date range tz/format/rel", {**big, "query": {"range": {"ts": {"gte": "01/06/2000", "lt": "now/y", "format": "dd/MM/yyyy||epoch_millis", "time_zone": "-08:00"}}}})
    add("date range round gt/lte", {**big, "query": {"range": {"ts": {"gt": "2000-01-31||/M", "lte": "2010-06-30||/d"}}}})
    add("range keyword", {**big, "query": {"range": {"tag": {"gte": "a", "lt": "r", "boost": 2}}}})
    add("range float bounds", {**big, "query": {"range": {"f": {"gt": 0.1, "lte": 1.0}}}})
    add("range long edges", {**big, "query": {"range": {"n": {"gte": 9223372036854775807}}}})
    add("range long float bound", {**big, "query": {"range": {"n": {"gt": 41.5, "lt": 1e30}}}})
    add("term numeric string", {**big, "query": {"term": {"n": "42"}}})
    add("term boolean", {**big, "query": {"terms": {"flag": [True, "false"]}}})
    add("match numeric float on long", {**big, "query": {"match": {"n": 42.0}}})
    add("ids", {**big, "query": {"ids": {"values": ["d1", "d2", "d3", "nope"]}}})
    add("nested score modes", {**big, "query": {"bool": {"should": [{"nested": {"path": "comments", "query": {"match": {"comments.text": w()}}, "score_mode": m}} for m in ("avg",)]}}})
    for m in ("sum", "max", "min", "none"):
        add(f"nested score {m}", {**big, "query": {"nested": {"path": "comments", "query": {"function_score": {"query": {"match": {"comments.text": w()}}, "boost": 1}}, "score_mode": m, "ignore_unmapped": True}}})
    # ---- request level
    add("rescore sum", {**big, "query": {"match": {"body": w()}}, "rescore": {"window_size": 50, "query": {"rescore_query": {"match_phrase": {"body": f"{w()} {w()}"}}, "query_weight": 0.7, "rescore_query_weight": 1.3, "score_mode": "total"}}}, ordered=False)
    for sm in ("avg", "max", "min", "multiply"):
        add(f"rescore {sm}", {"size": 100, "query": {"match": {"body": w()}}, "rescore": [{"window_size": 30, "query": {"rescore_query": {"match": {"title": w()}}, "score_mode": sm}}, {"window_size": 10, "query": {"rescore_query": {"term": {"tag": "red"}}, "score_mode": "total"}}]})
    add("post_filter+aggs", {**big, "query": {"match": {"body": w()}}, "post_filter": {"range": {"n": {"gt": 0}}}, "aggs": {"t": {"terms": {"field": "tag", "size": 5}}}})
    add("min_score", {**big, "query": {"match": {"body": f"{w()} {w()}"}}, "min_score": 1.2})
    add("track_total_hits false", {"size": 20, "track_total_hits": False, "query": {"match": {"body": w()}}, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}]}, ordered=True)
    add("track_total_hits 17", {"size": 20, "track_total_hits": 17, "query": {"match": {"body": w()}}})
    add("track_scores sort", {"size": 30, "track_scores": True, "query": {"match": {"body": w()}}, "sort": [{"n": "asc"}, {"_seq_no": "asc"}]}, ordered=True)
    add("docvalue_fields", {"size": 40, "query": {"match_all": {}}, "docvalue_fields": ["n", "price", "tag", "tags", {"field": "ts", "format": "yyyy-MM-dd"}, "ip", "loc", "flag", "sf", "hf", "f"], "sort": [{"_seq_no": "asc"}]}, ordered=True)
    add("fields api", {"size": 40, "query": {"match_all": {}}, "fields": ["title", "tag*", {"field": "ts", "format": "epoch_millis"}, "loc", "comments.*"], "_source": False, "sort": [{"_seq_no": "asc"}]}, ordered=True)
    add("source filter", {"size": 40, "query": {"match_all": {}}, "_source": {"includes": ["t*", "comments.a*"], "excludes": ["title"]}, "sort": [{"_seq_no": "asc"}]}, ordered=True)
    add("stored_fields none", {"size": 40, "query": {"match_all": {}}, "stored_fields": "_none_", "sort": [{"_seq_no": "asc"}]}, ordered=True)
    add("script_fields", {"size": 40, "query": {"match_all": {}}, "script_fields": {"x": {"script": "doc['n'].size() > 0 ? doc['n'].value * 2 : -1"}, "y": {"script": "params._source.title"}}, "sort": [{"_seq_no": "asc"}]}, ordered=True)
    add("version/seqno", {"size": 20, "query": {"match_all": {}}, "version": True, "seq_no_primary_term": True, "sort": [{"_seq_no": "asc"}]}, ordered=True)
    add("explain flag", {"size": 10, "explain": True, "query": {"bool": {"must": {"match": {"body": w()}}, "filter": {"term": {"tag": "red"}}}}, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}]}, ordered=True)
    add("from/size deep", {"from": 700, "size": 60, "query": {"match": {"body": w()}}, "sort": [{"price": "desc"}, {"_seq_no": "asc"}]}, ordered=True)
    add("collapse", {"size": 30, "query": {"match": {"body": w()}}, "collapse": {"field": "tag", "inner_hits": {"name": "x", "size": 2, "sort": [{"n": "desc"}, {"_seq_no": "asc"}]}}, "sort": [{"tag": "asc"}, {"_seq_no": "asc"}]}, ordered=True)
    add("collapse numeric", {"size": 30, "query": {"match_all": {}}, "collapse": {"field": "sh"}, "sort": [{"sh": "desc"}, {"_seq_no": "asc"}]}, ordered=True)
    add("search_after score", {"size": 25, "query": {"match": {"body": w()}}, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}], "search_after": [1.0, 50]}, ordered=True)
    add("terminate_after", {"size": 5, "terminate_after": 7, "query": {"match_all": {}}, "sort": [{"_seq_no": "asc"}]}, ordered=True)
    for mode in ("min", "max", "avg", "sum", "median"):
        add(f"sort mode {mode}", {"size": 50, "query": {"match_all": {}}, "sort": [{"tags": {"order": "asc", "mode": mode}} if mode in ("min", "max") else {"n": {"order": "desc"}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort nested", {"size": 50, "query": {"match_all": {}}, "sort": [{"comments.stars": {"order": "desc", "mode": "avg", "nested": {"path": "comments", "filter": {"range": {"comments.stars": {"gte": 2}}}}, "missing": 0}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort nested sum", {"size": 50, "query": {"match_all": {}}, "sort": [{"comments.stars": {"order": "asc", "mode": "sum", "nested": {"path": "comments"}}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort geo", {"size": 50, "query": {"exists": {"field": "loc"}}, "sort": [{"_geo_distance": {"loc": [10, 10], "order": "asc", "unit": "km", "distance_type": "arc", "mode": "min"}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort script", {"size": 50, "query": {"match_all": {}}, "sort": [{"_script": {"type": "number", "script": "doc['i'].size() > 0 ? doc['i'].value : 0", "order": "desc"}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort script string", {"size": 50, "query": {"match_all": {}}, "sort": [{"_script": {"type": "string", "script": "doc['tag'].size() > 0 ? doc['tag'].value : 'zzz'", "order": "asc"}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort missing custom", {"size": 50, "query": {"match_all": {}}, "sort": [{"price": {"order": "asc", "missing": 5.5}}, {"tag": {"order": "desc", "missing": "zz"}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort numeric_type", {"size": 50, "query": {"match_all": {}}, "sort": [{"i": {"order": "asc", "numeric_type": "double"}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort date_nanos type", {"size": 50, "query": {"match_all": {}}, "sort": [{"ts": {"order": "desc", "numeric_type": "date_nanos"}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort unmapped", {"size": 20, "query": {"match_all": {}}, "sort": [{"nofield": {"order": "asc", "unmapped_type": "long"}}, {"_seq_no": "asc"}]}, ordered=True)
    add("sort _doc _shard_doc", {"size": 20, "query": {"match_all": {}}, "sort": [{"_shard_doc": "asc"}]}, ordered=False)
    # ---- aggregations: pipeline & rarely used
    hist = {"date_histogram": {"field": "ts", "calendar_interval": "year", "min_doc_count": 0}, "aggs": {"s": {"sum": {"field": "price"}}, "a": {"avg": {"field": "i"}}, "c": {"value_count": {"field": "tag"}}}}
    P = {
        "derivative": {"derivative": {"buckets_path": "s", "unit": "1d"}},
        "second derivative": {"derivative": {"buckets_path": "d1"}},
        "cumulative_sum": {"cumulative_sum": {"buckets_path": "s"}},
        "cumulative_cardinality": {"cumulative_cardinality": {"buckets_path": "c"}},
        "moving_fn": {"moving_fn": {"buckets_path": "s", "window": 3, "script": "MovingFunctions.unweightedAvg(values)", "shift": 1}},
        "moving_fn holt": {"moving_fn": {"buckets_path": "s", "window": 4, "script": "MovingFunctions.holt(values, 0.3, 0.1)"}},
        "serial_diff": {"serial_diff": {"buckets_path": "s", "lag": 2}},
        "bucket_script": {"bucket_script": {"buckets_path": {"x": "s", "y": "_count"}, "script": "params.x * 2 + params.y", "gap_policy": "insert_zeros"}},
        "bucket_selector": {"bucket_selector": {"buckets_path": {"x": "_count"}, "script": "params.x > 5"}},
        "bucket_sort": {"bucket_sort": {"sort": [{"s": {"order": "desc"}}], "from": 1, "size": 4}},
    }
    for name, agg in P.items():
        sub = {"s": {"sum": {"field": "price"}}, "a": {"avg": {"field": "i"}}, "c": {"value_count": {"field": "tag"}}, "d1": {"derivative": {"buckets_path": "s"}}}
        sub["p"] = agg
        add(f"pipeline {name}", {"size": 0, "aggs": {"h": {"date_histogram": {"field": "ts", "calendar_interval": "year", "min_doc_count": 0}, "aggs": sub}}})
    for name in ("avg_bucket", "sum_bucket", "min_bucket", "max_bucket", "stats_bucket", "extended_stats_bucket", "percentiles_bucket"):
        add(f"sibling {name}", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "size": 20}, "aggs": {"m": {"avg": {"field": "price"}}}}, "p": {name: {"buckets_path": "t>m", **({"percents": [10, 50, 99.5]} if name == "percentiles_bucket" else {})}}}})
    add("scripted_metric", {"size": 0, "query": {"match": {"body": w()}}, "aggs": {"sm": {"scripted_metric": {"init_script": "state.t = []", "map_script": "if (doc['n'].size() > 0) state.t.add(doc['n'].value)", "combine_script": "double s = 0; for (x in state.t) { s += x } return s", "reduce_script": "double s = 0; for (x in states) { s += x } return s"}}}})
    add("top_metrics", {"size": 0, "aggs": {"tm": {"top_metrics": {"metrics": [{"field": "price"}, {"field": "n"}], "sort": {"_seq_no": "desc"}, "size": 3}}}})
    add("top_metrics terms", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "size": 8}, "aggs": {"tm": {"top_metrics": {"metrics": {"field": "i"}, "sort": {"n": "asc"}}}}}}})
    add("boxplot", {"size": 0, "aggs": {"b": {"boxplot": {"field": "price", "compression": 200}}}})
    add("string_stats", {"size": 0, "aggs": {"s": {"string_stats": {"field": "tag", "show_distribution": True}}}})
    add("matrix_stats", {"size": 0, "aggs": {"m": {"matrix_stats": {"fields": ["i", "sh", "price"], "mode": "avg"}}}})
    add("t_test", {"size": 0, "aggs": {"t": {"t_test": {"a": {"field": "price"}, "b": {"field": "f"}, "type": "heteroscedastic"}}}})
    add("rate", {"size": 0, "aggs": {"h": {"date_histogram": {"field": "ts", "calendar_interval": "year"}, "aggs": {"r": {"rate": {"field": "price", "unit": "month"}}, "r2": {"rate": {"unit": "day"}}}}}})
    add("sampler", {"size": 0, "query": {"match": {"body": w()}}, "aggs": {"s": {"sampler": {"shard_size": 30}, "aggs": {"t": {"terms": {"field": "tag"}}}}}})
    add("diversified_sampler", {"size": 0, "query": {"match": {"body": w()}}, "aggs": {"s": {"diversified_sampler": {"shard_size": 40, "field": "tag", "max_docs_per_value": 2}, "aggs": {"a": {"avg": {"field": "price"}}}}}})
    add("significant_terms params", {"size": 0, "query": {"term": {"tag": "red"}}, "aggs": {"s": {"significant_terms": {"field": "tags", "min_doc_count": 1, "chi_square": {"include_negatives": True}, "size": 5}}}})
    add("significant_terms gnd", {"size": 0, "query": {"term": {"tag": "red"}}, "aggs": {"s": {"significant_terms": {"field": "tags", "min_doc_count": 1, "gnd": {"background_is_superset": False}}}}})
    add("significant_terms jlh bg", {"size": 0, "query": {"term": {"tag": "red"}}, "aggs": {"s": {"significant_terms": {"field": "tags", "min_doc_count": 1, "background_filter": {"range": {"n": {"gt": 0}}}, "mutual_information": {}}}}})
    add("significant_text filter dup", {"size": 0, "query": {"match": {"body": w()}}, "aggs": {"s": {"significant_text": {"field": "body", "filter_duplicate_text": True, "size": 6, "percentage": {}}}}})
    add("geo_distance agg", {"size": 0, "aggs": {"g": {"geo_distance": {"field": "loc", "origin": "0,0", "unit": "km", "ranges": [{"to": 3000}, {"from": 3000, "to": 8000, "key": "mid"}, {"from": 8000}]}}}})
    add("histogram bounds", {"size": 0, "aggs": {"h": {"histogram": {"field": "i", "interval": 7, "offset": 2, "extended_bounds": {"min": -100, "max": 100}, "hard_bounds": {"min": -40, "max": 60}, "min_doc_count": 0, "keyed": True, "order": {"_count": "desc"}}}}})
    add("histogram missing", {"size": 0, "aggs": {"h": {"histogram": {"field": "price", "interval": 25, "missing": 1000}}}})
    for tz in ("America/New_York", "+05:45", "Australia/Lord_Howe", "UTC"):
        for ci in ("day", "week", "month", "quarter", "1h", "30d"):
            ky = "calendar_interval" if ci in ("day", "week", "month", "quarter") else "fixed_interval"
            add(f"date_histogram {ci} {tz}", {"size": 0, "query": {"range": {"ts": {"gte": "2015-01-01", "lt": "2016-06-01"}}}, "aggs": {"h": {"date_histogram": {"field": "ts", "time_zone": tz, ky: ci, "format": "yyyy-MM-dd HH:mm:ss ZZZZ", "offset": "+3h", "min_doc_count": 0, "extended_bounds": {"min": "2015-03-01", "max": "2015-06-01"}}}}})
    add("date_histogram hard bounds order", {"size": 0, "aggs": {"h": {"date_histogram": {"field": "ts", "calendar_interval": "year", "hard_bounds": {"min": "1990", "max": "2010"}, "order": {"_count": "desc"}}}}})
    add("date_range keyed", {"size": 0, "aggs": {"d": {"date_range": {"field": "ts", "format": "yyyy", "keyed": True, "time_zone": "+01:00", "ranges": [{"to": "1980"}, {"from": "1980", "to": "2000", "key": "x"}, {"from": "now-100y/y"}]}}}})
    add("range agg keyed", {"size": 0, "aggs": {"r": {"range": {"field": "price", "keyed": True, "ranges": [{"to": 0}, {"from": 0, "to": 10}, {"from": 10, "key": "big"}, {"from": 5, "to": 8}]}, "aggs": {"s": {"stats": {"field": "n"}}}}}})
    add("filters other_bucket", {"size": 0, "aggs": {"f": {"filters": {"other_bucket_key": "rest", "filters": {"a": {"term": {"tag": "red"}}, "b": {"range": {"n": {"gt": 5}}}}}, "aggs": {"m": {"max": {"field": "price"}}}}}})
    add("filters anon", {"size": 0, "aggs": {"f": {"filters": {"other_bucket": True, "filters": [{"term": {"tag": "red"}}, {"match": {"body": w()}}]}}}})
    add("missing + global", {"size": 0, "query": {"term": {"tag": "red"}}, "aggs": {"g": {"global": {}, "aggs": {"m": {"missing": {"field": "price"}}, "c": {"cardinality": {"field": "tag", "precision_threshold": 5}}}}}})
    add("terms order multi", {"size": 0, "aggs": {"t": {"terms": {"field": "tags", "size": 6, "order": [{"m": "desc"}, {"_key": "asc"}], "shard_size": 100}, "aggs": {"m": {"max": {"field": "n"}}}}}})
    add("terms include/exclude regex", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "include": "r.*|b.*", "exclude": "blue", "size": 50}}}})
    add("terms include partition", {"size": 0, "aggs": {"t": {"terms": {"field": "tags", "include": {"partition": 1, "num_partitions": 3}, "size": 50}}}})
    add("terms exact arrays", {"size": 0, "aggs": {"t": {"terms": {"field": "tags", "include": ["red", "blue", "x.y"], "min_doc_count": 0, "size": 50}}}})
    add("terms numeric key as string", {"size": 0, "aggs": {"t": {"terms": {"field": "sh", "size": 10, "show_term_doc_count_error": True, "format": "000", "order": {"_key": "desc"}}}}})
    add("terms on ip", {"size": 0, "aggs": {"t": {"terms": {"field": "ip", "size": 10, "order": {"_key": "asc"}}}}})
    add("terms on boolean", {"size": 0, "aggs": {"t": {"terms": {"field": "flag", "missing": "n/a"}}}})
    add("terms on date", {"size": 0, "aggs": {"t": {"terms": {"field": "ts", "size": 5, "order": {"_key": "desc"}, "format": "yyyy"}}}})
    add("terms breadth_first", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "collect_mode": "breadth_first", "size": 3}, "aggs": {"h": {"terms": {"field": "tags", "size": 3}}}}}})
    add("terms script", {"size": 0, "aggs": {"t": {"terms": {"script": {"source": "doc['tag'].size() > 0 ? doc['tag'].value.length() : -1"}, "size": 20}}}})
    add("terms value_type", {"size": 0, "aggs": {"t": {"terms": {"field": "n", "value_type": "double", "size": 5, "order": {"_key": "asc"}}}}})
    add("percentiles hdr", {"size": 0, "aggs": {"p": {"percentiles": {"field": "i", "percents": [1, 50, 99.9], "hdr": {"number_of_significant_value_digits": 2}, "keyed": False}}}})
    add("percentile_ranks tdigest", {"size": 0, "aggs": {"p": {"percentile_ranks": {"field": "price", "values": [0, 10, 40], "tdigest": {"compression": 50}}}}})
    add("stats missing script", {"size": 0, "aggs": {"s": {"extended_stats": {"field": "i", "sigma": 3, "missing": 7}}, "t": {"stats": {"script": {"source": "doc['sh'].size() > 0 ? doc['sh'].value * 2 : 0"}}}}})
    add("cardinality script", {"size": 0, "aggs": {"c": {"cardinality": {"script": "doc['tag'].size() > 0 ? doc['tag'].value + 'x' : 'none'"}}, "d": {"cardinality": {"field": "ip"}}, "e": {"cardinality": {"field": "loc"}}}})
    add("avg unit/format", {"size": 0, "aggs": {"a": {"avg": {"field": "ts", "format": "yyyy-MM"}}, "mx": {"max": {"field": "ts"}}, "mn": {"min": {"field": "sf"}}, "su": {"sum": {"field": "hf"}}}})
    add("weighted_avg", {"size": 0, "aggs": {"w": {"weighted_avg": {"value": {"field": "price", "missing": 0}, "weight": {"field": "i", "missing": 1}, "format": "0.000"}}}})
    add("top_hits nested", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "size": 3}, "aggs": {"th": {"top_hits": {"size": 2, "sort": [{"n": "desc"}, {"_seq_no": "asc"}], "_source": ["title"], "docvalue_fields": ["price"], "highlight": {"fields": {"body": {}}}}}}}}})
    add("nested + reverse_nested", {"size": 0, "aggs": {"n": {"nested": {"path": "comments"}, "aggs": {"a": {"terms": {"field": "comments.author", "size": 5}, "aggs": {"r": {"reverse_nested": {}, "aggs": {"t": {"terms": {"field": "tag", "size": 3}}}}, "st": {"avg": {"field": "comments.stars"}}}}}}}})
    add("composite after", {"size": 0, "aggs": {"c": {"composite": {"size": 7, "sources": [{"a": {"terms": {"field": "tag", "order": "desc", "missing_bucket": True}}}, {"b": {"histogram": {"field": "sh", "interval": 40}}}, {"d": {"date_histogram": {"field": "ts", "calendar_interval": "decade"}}}], "after": {"a": "red", "b": 0, "d": 631152000000}}}}})
    add("composite geotile", {"size": 0, "aggs": {"c": {"composite": {"size": 5, "sources": [{"g": {"geotile_grid": {"field": "loc", "precision": 3}}}]}}}})
    add("auto_date_histogram", {"size": 0, "aggs": {"a": {"auto_date_histogram": {"field": "ts", "buckets": 12, "minimum_interval": "month", "time_zone": "+02:00"}, "aggs": {"s": {"sum": {"field": "n"}}}}}})
    add("multi_terms", {"size": 0, "aggs": {"m": {"multi_terms": {"terms": [{"field": "tag"}, {"field": "sh", "missing": -1}], "size": 8, "order": [{"_count": "desc"}, {"_key": "asc"}]}}}})
    add("rare_terms", {"size": 0, "aggs": {"r": {"rare_terms": {"field": "tags", "max_doc_count": 3, "precision": 0.001}}}})
    add("adjacency_matrix", {"size": 0, "aggs": {"a": {"adjacency_matrix": {"filters": {"x": {"term": {"tag": "red"}}, "y": {"range": {"n": {"gt": 0}}}, "z": {"exists": {"field": "loc"}}}, "separator": "&"}}}})
    add("ip_prefix", {"size": 0, "aggs": {"a": {"ip_prefix": {"field": "ip", "prefix_length": 8, "min_doc_count": 1}}}})
    add("variable_width", {"size": 0, "aggs": {"v": {"variable_width_histogram": {"field": "price", "buckets": 4}}}})
    add("sum of bucket_script nested", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "size": 5}, "aggs": {"s": {"sum": {"field": "n"}}, "c": {"bucket_script": {"buckets_path": {"s": "s", "n": "_count"}, "script": "params.s / params.n"}}}}, "mx": {"max_bucket": {"buckets_path": "t>c"}}}})
    add("agg errors: bad path", {"size": 0, "aggs": {"t": {"terms": {"field": "tag"}, "aggs": {"d": {"derivative": {"buckets_path": "nope"}}}}}})
    add("agg errors: text field", {"size": 0, "aggs": {"t": {"terms": {"field": "title"}}}})
    add("agg errors: bad interval", {"size": 0, "aggs": {"t": {"histogram": {"field": "n", "interval": 0}}}})
    add("agg meta", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "size": 2}, "meta": {"x": [1, {"y": 2}]}}}})
    add("typed_keys", {"size": 0, "aggs": {"t": {"terms": {"field": "tag", "size": 2}, "aggs": {"p": {"percentiles": {"field": "n"}}, "e": {"extended_stats": {"field": "n"}}}}}}, params="?typed_keys=true")
    # ---- _count, _validate, _explain style APIs via search parameters
    add("search_type dfs", {**big, "query": {"match": {"body": f"{w()} {w()}"}}}, params="?search_type=dfs_query_then_fetch")
    add("request_cache + size0", {"size": 0, "query": {"match": {"body": w()}}, "aggs": {"c": {"cardinality": {"field": "tag"}}}}, params="?request_cache=true")
    add("preference+routing", {**big, "query": {"match_all": {}}}, params="?routing=a")
    add("allow_partial/ignore", {**big, "query": {"match_all": {}}}, params="?ignore_unavailable=true&expand_wildcards=all")
    add("pre_filter_shard_size", {**big, "query": {"range": {"n": {"gt": 10**6}}}}, params="?pre_filter_shard_size=1")
    add("indices_boost", {**big, "query": {"match": {"body": w()}}, "indices_boost": [{"sw": 2.5}]})
    add("stats groups", {**big, "query": {"match_all": {}}, "stats": ["g1"]})
    add("seq_no_primary_term sort _id", {"size": 30, "query": {"match_all": {}}, "sort": [{"_id": "desc"}]}, ordered=True)
    add("sort _index _score", {"size": 30, "query": {"match": {"body": w()}}, "sort": [{"_index": "asc"}, {"_score": "desc"}, {"_seq_no": "asc"}]}, ordered=True)
    add("source false+hits only", {"size": 5, "_source": False, "query": {"match": {"body": w()}}, "sort": [{"_seq_no": "asc"}]}, ordered=True)
    add("highlight variety", {"size": 15, "query": {"bool": {"should": [{"match": {"body": w()}}, {"match_phrase": {"title": f"{w()} {w()}"}}, {"prefix": {"title": "al"}}]}}, "sort": [{"_seq_no": "asc"}], "highlight": {"pre_tags": ["<b>"], "post_tags": ["</b>"], "fields": {"body": {"type": "unified", "fragment_size": 40, "number_of_fragments": 2, "no_match_size": 20}, "title": {"type": "plain", "number_of_fragments": 0}, "ws": {"type": "fvh"}}, "require_field_match": False}}, ordered=True)
    return c


def sim_cases():
    """Per-field similarity indices: same data, same mapping, score every document."""
    sims = {
        "bm25_k": {"type": "BM25", "k1": 2.0, "b": 0.3, "discount_overlaps": False},
        
        "boolean": {"type": "boolean"},
        "dfr": {"type": "DFR", "basic_model": "g", "after_effect": "l", "normalization": "h2", "normalization.h2.c": 2.0},
        "dfr2": {"type": "DFR", "basic_model": "in", "after_effect": "b", "normalization": "z", "normalization.z.z": 0.4},
        "ib": {"type": "IB", "distribution": "spl", "lambda": "ttf", "normalization": "h3", "normalization.h3.c": 1.5},
        "ib2": {"type": "IB", "distribution": "ll", "lambda": "df", "normalization": "no"},
        "lmd": {"type": "LMDirichlet", "mu": 500},
        "lmj": {"type": "LMJelinekMercer", "lambda": 0.4},
        "scripted": {"type": "scripted", "script": {"source": "double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; double norm = 1/Math.sqrt(doc.length); return query.boost * tf * idf * norm;"}},
    }
    return sims


QUERIES_SIM = [
    {"match": {"body": "alpha beta"}},
    {"match": {"body": {"query": "alpha beta gamma", "operator": "and"}}},
    {"match_phrase": {"body": "quick brown"}},
    {"match_phrase": {"body": {"query": "quick fox", "slop": 3}}},
    {"multi_match": {"query": "alpha fox", "fields": ["body", "title"], "type": "most_fields"}},
    {"query_string": {"query": "alpha OR (beta AND fox*)", "default_field": "body"}},
    {"bool": {"must": {"term": {"body": "alpha"}}, "should": [{"match": {"body": "dog"}}, {"fuzzy": {"body": "gama"}}]}},
    {"combined_fields": {"query": "alpha beta", "fields": ["body", "title"]}},
    {"span_near": {"clauses": [{"span_term": {"body": "quick"}}, {"span_term": {"body": "fox"}}], "slop": 4, "in_order": False}},
    {"match_all": {}},
    {"intervals": {"body": {"all_of": {"ordered": True, "intervals": [{"match": {"query": "alpha"}}, {"match": {"query": "beta"}}]}}}},
    {"more_like_this": {"fields": ["body"], "like": "alpha quick fox", "min_term_freq": 1, "min_doc_freq": 1}},
]


def build_sim(nodes, log):
    sims = sim_cases()
    for name, sim in sims.items():
        idx = f"sim_{name}"
        # normalization.* keys flatten into nested settings
        s = dict(sim)
        mp = {"properties": {"body": {"type": "text", "similarity": "custom"}, "title": {"type": "text", "similarity": "custom"}}}
        r = random.Random(3)
        lines = []
        for i in range(600):
            d = {"body": fd.sentence(r, 2, 30), "title": fd.sentence(r, 1, 5)}
            if r.random() < 0.05:
                d.pop("title")
            lines.append(json.dumps({"index": {"_id": f"d{i}"}}))
            lines.append(json.dumps(d))
        body = ("\n".join(lines) + "\n").encode()
        for n in nodes:
            n.req("DELETE", f"/{idx}")
            st, resp = n.req("PUT", f"/{idx}", {"settings": {"number_of_shards": 2, "number_of_replicas": 0, "similarity": {"custom": s}}, "mappings": mp})
            if st != 200:
                log({"kind": "sim-create", "sim": name, "status": st, "resp": str(resp)[:300], "node": n.url})
                break
        else:
            for n in nodes:
                n.req("POST", f"/{idx}/_bulk?refresh=true", body, raw=True)
                n.req("POST", f"/{idx}/_refresh")
            yield idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="http://localhost:9200")
    ap.add_argument("--stock", default="http://localhost:9201")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--merge", type=int, default=0)
    ap.add_argument("--concurrent", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-sim", action="store_true")
    a = ap.parse_args()
    plugin, stock = fd.Node(a.plugin), fd.Node(a.stock)
    nodes = [plugin, stock]
    mode = "all" if a.concurrent else "none"
    for n in nodes:
        n.req("PUT", "/_cluster/settings", {"transient": {"search.concurrent_segment_search.mode": mode}})
    findings = []

    def log(f):
        findings.append(f)
        print(json.dumps(f), flush=True)

    fd.set_vocab(0)
    fd.ingest(nodes, "sw", a.shards, a.seed, N, a.merge, log)
    r = random.Random(a.seed)
    cs = cases(r)
    n_run = 0
    for name, idx, body, ordered, params in cs:
        if a.only and a.only not in name:
            continue
        d, p, s = fd.run_case(plugin, stock, idx, body, ordered, params)
        n_run += 1
        if d and not fd._resource(p) and not fd._resource(s):
            log({"kind": "diff", "case": name, "diff": d, "body": json.dumps(body)[:500], "params": params})
    if not a.skip_sim and not a.only:
        for idx in build_sim(nodes, log):
            for qi, q in enumerate(QUERIES_SIM):
                body = {"size": 600, "query": q, "explain": False}
                d, p, s = fd.run_case(plugin, stock, idx, body, False)
                n_run += 1
                if d:
                    log({"kind": "diff", "case": f"{idx} q{qi}", "diff": d, "body": json.dumps(q)[:300]})
    print(json.dumps({"cases": n_run, "findings": len(findings), "concurrent": a.concurrent}))


if __name__ == "__main__":
    main()
