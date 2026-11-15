#!/usr/bin/env python3
"""Field-type differential: one index per mapping type (including exotic ones), random values
(with nulls, arrays, malformed and boundary values), and the type's own queries and aggregations,
on a plugin node and a stock node; the same documents must be accepted or rejected, and every
query and aggregation must answer the same.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/fuzz_types.py --seed 1 --docs 300
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node, diff, cmp_search  # noqa: E402


def ints(lo, hi):
    return lambda r: r.choice([r.randint(lo, hi), lo, hi, 0, str(r.randint(lo, hi)), float(r.randint(lo, hi)) if hi < 2**40 else 0, "x" if r.random() < .03 else 1, r.randint(lo, hi) + 0.5 if hi < 2**40 else 1])


def dbl(r):
    return r.choice([round(r.uniform(-100, 100), 4), 0.0, -0.0, 1e-5, 1e5, "3.5", "abc" if r.random() < .05 else 2, 2**40, -2**40 + 0.25])


def dates(r):
    return r.choice(["2020-02-29T12:00:00Z", "1999-12-31", "2000-01-01T00:00:00.123Z", r.randint(0, 2 * 10**12), str(r.randint(0, 2 * 10**12)), "2021-13-45" if r.random() < .1 else "2001-01-01", "1970-01-01T00:00:00Z", "2262-04-11T23:47:16.854Z", "0001-01-01T00:00:00Z"])


def rng_int(r):
    a = r.randint(-50, 50)
    return r.choice([{"gte": a, "lte": a + r.randint(0, 20)}, {"gt": a, "lt": a + 30}, {"gte": a}, {"lte": a}, {}, a if r.random() < .1 else {"gte": 1, "lte": 2}])


def rng_date(r):
    y = r.randint(1990, 2030)
    return r.choice([{"gte": f"{y}-01-01", "lte": f"{y + 3}-12-31"}, {"gt": f"{y}-06-01", "lt": f"{y + 1}-06-01"}, {"gte": f"{y}-01-01"}, f"{y}-01-01/{y + 1}-01-01"])


def rng_ip(r):
    return r.choice([{"gte": "10.0.0.1", "lte": "10.0.0.200"}, "192.168.0.0/16", {"gte": "::1", "lte": "::ff"}, {"gt": "1.1.1.1"}, "10.1.0.0/24"])


def geo(r):
    return r.choice([{"lat": r.uniform(-80, 80), "lon": r.uniform(-179, 179)}, [r.uniform(-179, 179), r.uniform(-80, 80)], f"{r.uniform(-80, 80):.4f},{r.uniform(-179, 179):.4f}",
                     {"lat": 90, "lon": 180}, {"lat": -90, "lon": -180}, "POINT (3.5 4.5)", "u0tkq" if r.random() < .3 else "gcpvj0", {"lat": 91, "lon": 0} if r.random() < .05 else {"lat": 0, "lon": 0}])


def shape(r):
    x, y = r.uniform(-60, 60), r.uniform(-60, 60)
    return r.choice([{"type": "point", "coordinates": [x, y]}, {"type": "polygon", "coordinates": [[[x, y], [x + 5, y], [x + 5, y + 5], [x, y + 5], [x, y]]]},
                     {"type": "linestring", "coordinates": [[x, y], [x + 10, y + 3]]}, {"type": "envelope", "coordinates": [[x, y + 8], [x + 8, y]]},
                     f"POINT ({x:.3f} {y:.3f})", {"type": "multipoint", "coordinates": [[x, y], [y, x]]}, {"type": "circle", "coordinates": [x, y], "radius": "100km"} if r.random() < .0 else {"type": "point", "coordinates": [0, 0]}])


WORDS = "alpha beta gamma delta epsilon the quick brown fox 2020 x-y a_b".split()


def text(r):
    return " ".join(r.choice(WORDS) for _ in range(r.randint(1, 6)))


def kw(r):
    return r.choice(["a", "B", "ab", "abc", "ABC", "", "x y", "0", "true", "\u00e9\u00e8", "\U0001f600", "a" * 300, "foo-bar", "10.0.0.1"])


# type name -> (mapping, value generator, queries, aggregations)
def cases():
    T = {}
    T["byte"] = ({"type": "byte"}, ints(-128, 127))
    T["short"] = ({"type": "short"}, ints(-32768, 32767))
    T["integer"] = ({"type": "integer"}, ints(-2**31, 2**31 - 1))
    T["long"] = ({"type": "long"}, ints(-2**63, 2**63 - 1))
    T["unsigned_long"] = ({"type": "unsigned_long"}, lambda r: r.choice([r.randint(0, 100), 2**64 - 1, 2**63, str(2**64 - 1), "18446744073709551615", -1 if r.random() < .05 else 5]))
    T["long_ignore_malformed"] = ({"type": "long", "ignore_malformed": True}, ints(-1000, 1000))
    T["long_null_value"] = ({"type": "long", "null_value": -7}, lambda r: r.choice([None, 3, 4, -7]))
    T["long_coerce_off"] = ({"type": "integer", "coerce": False}, ints(-100, 100))
    T["long_noindex"] = ({"type": "long", "index": False}, ints(-100, 100))
    T["long_nodv"] = ({"type": "long", "doc_values": False}, ints(-100, 100))
    T["float"] = ({"type": "float"}, dbl)
    T["half_float"] = ({"type": "half_float"}, dbl)
    T["double"] = ({"type": "double"}, dbl)
    T["scaled_float"] = ({"type": "scaled_float", "scaling_factor": 100}, dbl)
    T["scaled_float_ign"] = ({"type": "scaled_float", "scaling_factor": 10, "ignore_malformed": True}, dbl)
    T["date"] = ({"type": "date"}, dates)
    T["date_fmt"] = ({"type": "date", "format": "yyyy/MM/dd HH:mm||epoch_second"}, lambda r: r.choice(["2020/01/02 03:04", str(r.randint(0, 2 * 10**9)), r.randint(0, 2 * 10**9), "2020-01-02", "bad"]))
    T["date_nanos"] = ({"type": "date_nanos"}, lambda r: r.choice(["2020-01-01T00:00:00.123456789Z", "1970-01-01T00:00:00.000000001Z", "2100-01-01", r.randint(0, 2 * 10**12), "2262-04-11T23:47:16.854775807Z"]))
    T["boolean"] = ({"type": "boolean"}, lambda r: r.choice([True, False, "true", "false", "", "yes" if r.random() < .05 else True, None]))
    T["ip"] = ({"type": "ip"}, lambda r: r.choice(["10.0.0.1", "10.0.0.200", "192.168.5.5", "::1", "2001:db8::ff", "255.255.255.255", "0.0.0.0", "::ffff:10.0.0.1", "bad" if r.random() < .05 else "1.1.1.1"]))
    T["keyword"] = ({"type": "keyword"}, kw)
    T["keyword_ignore_above"] = ({"type": "keyword", "ignore_above": 3}, kw)
    T["keyword_normalizer"] = ({"type": "keyword", "normalizer": "lc"}, kw)
    T["keyword_null_value"] = ({"type": "keyword", "null_value": "NULL"}, lambda r: r.choice([None, "a", "b", "NULL"]))
    T["keyword_nodv"] = ({"type": "keyword", "doc_values": False}, kw)
    T["keyword_noindex"] = ({"type": "keyword", "index": False}, kw)
    T["constant_keyword"] = ({"type": "constant_keyword", "value": "fixed"}, lambda r: r.choice(["fixed", None]))
    T["wildcard"] = ({"type": "wildcard"}, lambda r: r.choice(["abc", "abcdef", "xyz.log", "ABC", "a*b", "", "hello world"]) + r.choice(["", "123", ".txt"]))
    T["text"] = ({"type": "text"}, text)
    T["text_norms_off"] = ({"type": "text", "norms": False}, text)
    T["text_docs"] = ({"type": "text", "index_options": "docs"}, text)
    T["text_freqs"] = ({"type": "text", "index_options": "freqs"}, text)
    T["text_offsets"] = ({"type": "text", "index_options": "offsets"}, text)
    T["text_prefixes"] = ({"type": "text", "index_prefixes": {"min_chars": 2, "max_chars": 4}}, text)
    T["text_phrases"] = ({"type": "text", "index_phrases": True}, text)
    T["text_boolsim"] = ({"type": "text", "similarity": "boolean"}, text)
    T["text_lmd"] = ({"type": "text", "similarity": "LMDirichlet"}, text)
    T["text_slop_gap"] = ({"type": "text", "position_increment_gap": 5}, lambda r: [text(r) for _ in range(r.randint(1, 3))])
    T["match_only_text"] = ({"type": "match_only_text"}, text)
    T["search_as_you_type"] = ({"type": "search_as_you_type"}, text)
    T["token_count"] = ({"type": "token_count", "analyzer": "standard"}, text)
    T["binary"] = ({"type": "binary", "doc_values": True}, lambda r: r.choice(["U29tZSBiaW5hcnkgYmxvYg==", "AAECAw==", ""]))
    T["geo_point"] = ({"type": "geo_point"}, geo)
    T["geo_point_ign"] = ({"type": "geo_point", "ignore_malformed": True, "ignore_z_value": True}, geo)
    T["geo_shape"] = ({"type": "geo_shape"}, shape)
    T["integer_range"] = ({"type": "integer_range"}, rng_int)
    T["long_range"] = ({"type": "long_range"}, rng_int)
    T["float_range"] = ({"type": "float_range"}, rng_int)
    T["double_range"] = ({"type": "double_range"}, rng_int)
    T["date_range"] = ({"type": "date_range"}, rng_date)
    T["ip_range"] = ({"type": "ip_range"}, rng_ip)
    T["rank_feature"] = ({"type": "rank_feature"}, lambda r: r.choice([1, 2.5, 10, 0.01, 100, 0 if r.random() < .05 else 3]))
    T["rank_features"] = ({"type": "rank_features"}, lambda r: {"a": r.randint(1, 10), "b": r.randint(1, 5), **({"c": 2} if r.random() < .5 else {})})
    T["flat_object"] = ({"type": "flat_object"}, lambda r: {"a": r.choice(["x", "y", 3]), "b": {"c": r.choice(["p", "q"]), "d": [1, 2]}})
    T["object_dynamic"] = ({"type": "object", "dynamic": True}, lambda r: {"a": r.choice(["x", "y", 3]), "b": {"c": r.choice(["p", 1.5, True]), "d": [r.randint(1, 5), 2]}})
    T["object_strict_ok"] = ({"type": "object", "dynamic": "strict", "properties": {"a": {"type": "keyword"}}}, lambda r: {"a": r.choice(["x", "y"]), **({"zz": 1} if r.random() < .1 else {})})
    T["nested"] = ({"type": "nested", "properties": {"a": {"type": "keyword"}, "n": {"type": "long"}}}, lambda r: [{"a": r.choice(["x", "y", "z"]), "n": r.randint(0, 9)} for _ in range(r.randint(0, 3))])
    T["completion"] = ({"type": "completion"}, lambda r: r.choice(["alpha", "alphabet", "beta", {"input": ["gamma", "gam"], "weight": 5}, ["delta", "del"]]))
    T["version"] = ({"type": "version"}, lambda r: r.choice(["1.0.0", "1.10.0", "2.0.0-rc1", "1.2.3+build", "10.0.1", "bad" if r.random() < .05 else "1.0.0"]))
    T["xy_point"] = ({"type": "xy_point"}, lambda r: r.choice([{"x": r.uniform(-100, 100), "y": r.uniform(-100, 100)}, [r.uniform(-100, 100), r.uniform(-100, 100)], "1.5, 2.5"]))
    return T


# queries per type family: (name, builder(field, r))
def queries(name, r):
    t = name
    q = [{"exists": {"field": "f"}}, {"match_all": {}}]
    numeric = t.split("_")[0] in ("byte", "short", "integer", "long", "unsigned", "float", "half", "double", "scaled") or t.startswith("long")
    if numeric:
        v = r.choice([0, 1, -1, 5, 3.5, 100, 2**31, -7])
        q += [{"term": {"f": v}}, {"terms": {"f": [v, 2, 3]}}, {"range": {"f": {"gte": -5, "lte": 50}}}, {"range": {"f": {"gt": v}}}, {"range": {"f": {"lt": v}}},
              {"match": {"f": v}}, {"query_string": {"query": "f:[-5 TO 50]"}}, {"query_string": {"query": f"f:{v}"}}, {"bool": {"must_not": {"range": {"f": {"gte": 0}}}}}]
    elif t.startswith("date"):
        d = r.choice(["2020-02-29", "2000-01-01", "1970-01-01T00:00:00Z", "2020-02-29T12:00:00Z"])
        q += [{"term": {"f": d}}, {"range": {"f": {"gte": "1999-01-01", "lte": "2021-01-01"}}}, {"range": {"f": {"gt": d}}}, {"range": {"f": {"lt": d, "format": "strict_date_optional_time"}}},
              {"range": {"f": {"gte": 0, "lte": 10**12, "format": "epoch_millis"}}}, {"match": {"f": d}}, {"query_string": {"query": "f:[2000 TO 2020]"}},
              {"range": {"f": {"gte": "2000-01-01", "time_zone": "+05:00"}}}]
    elif t == "boolean":
        q += [{"term": {"f": True}}, {"term": {"f": "false"}}, {"terms": {"f": [True, False]}}, {"match": {"f": "true"}}, {"range": {"f": {"gte": "false"}}}]
    elif t == "ip" or t == "ip_range":
        q += [{"term": {"f": "10.0.0.1"}}, {"term": {"f": "10.0.0.0/24"}}, {"range": {"f": {"gte": "10.0.0.1", "lte": "10.0.0.250"}}}, {"terms": {"f": ["::1", "1.1.1.1"]}}, {"query_string": {"query": "f:\"10.0.0.0/8\""}}]
    elif t.startswith("keyword") or t in ("constant_keyword", "wildcard", "version"):
        v = r.choice(["a", "B", "abc", "ABC", "fixed", "x y", "1.0.0", "abcdef", "0", "NULL", "true"])
        q += [{"term": {"f": v}}, {"terms": {"f": [v, "ab"]}}, {"prefix": {"f": v[:2]}}, {"wildcard": {"f": f"*{v[-2:]}*"}}, {"wildcard": {"f": {"value": f"{v[:1]}*", "case_insensitive": True}}},
              {"regexp": {"f": f"{v[:1]}.*"}}, {"fuzzy": {"f": v}}, {"range": {"f": {"gte": "a", "lt": "b"}}}, {"match": {"f": v}}, {"term": {"f": {"value": v, "case_insensitive": True}}},
              {"range": {"f": {"gt": "1.0.0", "lte": "2.0.0"}}} if t == "version" else {"exists": {"field": "f"}}]
    elif t.startswith("text") or t in ("match_only_text", "search_as_you_type", "token_count"):
        w = r.choice(WORDS)
        if t == "token_count":
            q += [{"term": {"f": r.randint(1, 6)}}, {"range": {"f": {"gte": 2, "lte": 4}}}]
        else:
            q += [{"match": {"f": w}}, {"match": {"f": {"query": f"{w} {r.choice(WORDS)}", "operator": "and"}}}, {"match_phrase": {"f": f"{w} {r.choice(WORDS)}"}}, {"prefix": {"f": w[:2]}},
                  {"match_phrase_prefix": {"f": w[:2]}}, {"term": {"f": w}}, {"query_string": {"query": f"f:{w}* OR f:\"{w} {r.choice(WORDS)}\"~2"}}, {"match": {"f": {"query": w, "fuzziness": 1}}},
                  {"wildcard": {"f": f"*{w[1:3]}*"}}, {"regexp": {"f": f"{w[:1]}.*"}}, {"simple_query_string": {"query": f"{w} -{r.choice(WORDS)}", "fields": ["f"]}},
                  {"intervals": {"f": {"match": {"query": f"{w} {r.choice(WORDS)}", "max_gaps": 2}}}}, {"span_near": {"clauses": [{"span_term": {"f": w}}, {"span_term": {"f": r.choice(WORDS)}}], "slop": 2, "in_order": False}}]
            if t == "search_as_you_type":
                q += [{"multi_match": {"query": w[:3], "type": "bool_prefix", "fields": ["f", "f._2gram", "f._3gram"]}}, {"match": {"f._index_prefix": w[:2]}}]
    elif t == "geo_point" or t == "geo_point_ign":
        q += [{"geo_distance": {"distance": f"{r.choice([10, 1000, 8000])}km", "f": {"lat": r.uniform(-60, 60), "lon": r.uniform(-170, 170)}}},
              {"geo_bounding_box": {"f": {"top_left": {"lat": 60, "lon": -100}, "bottom_right": {"lat": -10, "lon": 60}}}},
              {"geo_polygon": {"f": {"points": [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 60}, {"lat": 50, "lon": 30}]}}},
              {"geo_shape": {"f": {"shape": {"type": "envelope", "coordinates": [[-50, 50], [50, -50]]}, "relation": "within"}}} if False else {"exists": {"field": "f"}}]
    elif t == "geo_shape":
        env = {"type": "envelope", "coordinates": [[r.uniform(-90, 0), r.uniform(0, 60)], [r.uniform(0, 90), r.uniform(-60, 0)]]}
        q += [{"geo_shape": {"f": {"shape": env, "relation": rel}}} for rel in ("intersects", "disjoint", "within", "contains")]
        q += [{"geo_shape": {"f": {"shape": {"type": "point", "coordinates": [0, 0]}, "relation": "intersects"}}}]
    elif t.endswith("_range"):
        if t == "date_range":
            q += [{"range": {"f": {"gte": "2000-01-01", "lte": "2010-01-01", "relation": rel}}} for rel in ("intersects", "within", "contains")] + [{"term": {"f": "2005-05-05"}}]
        elif t == "ip_range":
            q += [{"range": {"f": {"gte": "10.0.0.1", "lte": "10.0.0.250", "relation": rel}}} for rel in ("intersects", "within", "contains")] + [{"term": {"f": "10.0.0.5"}}]
        else:
            q += [{"range": {"f": {"gte": -10, "lte": 10, "relation": rel}}} for rel in ("intersects", "within", "contains")] + [{"term": {"f": 5}}]
    elif t in ("rank_feature",):
        q += [{"rank_feature": {"field": "f"}}, {"rank_feature": {"field": "f", "saturation": {"pivot": 5}}}, {"rank_feature": {"field": "f", "log": {"scaling_factor": 2}}}, {"rank_feature": {"field": "f", "sigmoid": {"pivot": 5, "exponent": 0.6}}}, {"rank_feature": {"field": "f", "linear": {}}}]
    elif t == "rank_features":
        q += [{"rank_feature": {"field": "f.a"}}, {"rank_feature": {"field": "f.b", "log": {"scaling_factor": 4}}}, {"exists": {"field": "f.c"}}]
    elif t == "flat_object" or t == "object_dynamic":
        q += [{"term": {"f.a": "x"}}, {"term": {"f.b.c": "p"}}, {"exists": {"field": "f.b.d"}}, {"range": {"f.a": {"gte": "x"}}}, {"match": {"f.b.c": "q"}}, {"term": {"f": "x"}}, {"term": {"f.b.d": 2}}]
    elif t == "nested":
        q += [{"nested": {"path": "f", "query": {"term": {"f.a": r.choice(["x", "y", "z"])}}, "score_mode": m}} for m in ("avg", "sum", "max", "min", "none")]
        q += [{"nested": {"path": "f", "query": {"bool": {"must": [{"term": {"f.a": "x"}}, {"range": {"f.n": {"gte": 3}}}]}}}}]
    elif t == "completion":
        pass
    elif t == "binary":
        q += [{"exists": {"field": "f"}}]
    elif t == "xy_point":
        q += [{"xy_shape": {"f": {"shape": {"type": "envelope", "coordinates": [[-50, 50], [50, -50]]}, "relation": "intersects"}}}] if False else []
    return q


def aggs(name, r):
    a = []
    t = name
    numeric = t.split("_")[0] in ("byte", "short", "integer", "long", "unsigned", "float", "half", "double", "scaled") or t.startswith("long")
    dv = not t.endswith("_nodv")
    if numeric and dv and not t.endswith("noindex") or (numeric and t.endswith("noindex")):
        a += [{"stats": {"field": "f"}}, {"terms": {"field": "f", "size": 50, "order": [{"_key": "asc"}]}}, {"histogram": {"field": "f", "interval": 10, "min_doc_count": 1}} if t not in ("unsigned_long",) else {"max": {"field": "f"}},
              {"percentiles": {"field": "f", "percents": [50]}} if False else {"cardinality": {"field": "f"}}, {"range": {"field": "f", "ranges": [{"to": 0}, {"from": 0}]}}, {"avg": {"field": "f", "missing": 1}} if t != "unsigned_long" else {"min": {"field": "f"}}]
    elif t.startswith("date"):
        a += [{"date_histogram": {"field": "f", "calendar_interval": "year", "min_doc_count": 1}}, {"min": {"field": "f"}}, {"max": {"field": "f", "format": "yyyy"}}, {"terms": {"field": "f", "size": 50, "order": [{"_key": "asc"}]}},
              {"date_range": {"field": "f", "ranges": [{"to": "2000-01-01"}, {"from": "2000-01-01"}]}}, {"cardinality": {"field": "f"}}]
    elif t == "boolean":
        a += [{"terms": {"field": "f", "size": 5, "order": [{"_key": "asc"}]}}, {"cardinality": {"field": "f"}}, {"sum": {"field": "f"}}]
    elif t in ("ip",):
        a += [{"terms": {"field": "f", "size": 50, "order": [{"_key": "asc"}]}}, {"ip_range": {"field": "f", "ranges": [{"mask": "10.0.0.0/8"}, {"mask": "::/0"}]}}, {"cardinality": {"field": "f"}}, {"max": {"field": "f"}} if False else {"value_count": {"field": "f"}}]
    elif (t.startswith("keyword") or t in ("constant_keyword", "wildcard", "version")) and dv and not t.endswith("noindex"):
        a += [{"terms": {"field": "f", "size": 50, "order": [{"_key": "asc"}]}}, {"cardinality": {"field": "f"}}, {"value_count": {"field": "f"}}, {"missing": {"field": "f"}},
              {"terms": {"field": "f", "size": 5, "include": "a.*", "order": [{"_count": "desc"}, {"_key": "asc"}]}}, {"rare_terms": {"field": "f", "max_doc_count": 2}}, {"significant_terms": {"field": "f", "size": 5}}]
    elif t in ("geo_point", "geo_point_ign"):
        a += [{"geo_bounds": {"field": "f"}}, {"geo_centroid": {"field": "f"}}, {"geohash_grid": {"field": "f", "precision": 2}}, {"geotile_grid": {"field": "f", "precision": 3}}, {"geo_distance": {"field": "f", "origin": "0,0", "unit": "km", "ranges": [{"to": 5000}, {"from": 5000}]}}]
    elif t == "geo_shape":
        a += [{"geo_bounds": {"field": "f"}}, {"geo_centroid": {"field": "f"}}, {"geohash_grid": {"field": "f", "precision": 1}}]
    elif t in ("integer_range", "long_range", "float_range", "double_range", "date_range", "ip_range"):
        a += [{"range": {"field": "f", "ranges": [{"from": 0}]}}] if False else [{"value_count": {"field": "f"}}] if False else []
    elif t in ("text", "text_norms_off", "text_docs"):
        a += [{"significant_text": {"field": "f", "size": 5}}]
    elif t == "token_count":
        a += [{"stats": {"field": "f"}}, {"terms": {"field": "f", "size": 10, "order": [{"_key": "asc"}]}}]
    elif t == "rank_feature":
        a += []
    elif t == "nested":
        a += [{"nested": {"path": "f"}, "aggs": {"t": {"terms": {"field": "f.a", "size": 5, "order": [{"_key": "asc"}]}}, "s": {"stats": {"field": "f.n"}}}}]
    elif t == "version":
        a += [{"terms": {"field": "f", "size": 10, "order": [{"_key": "asc"}]}}]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="http://localhost:9200")
    ap.add_argument("--stock", default="http://localhost:9201")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--docs", type=int, default=250)
    ap.add_argument("--only", default="")
    ap.add_argument("--merge", type=int, default=1)
    a = ap.parse_args()
    plugin, stock = Node(a.plugin), Node(a.stock)
    r = random.Random(a.seed)
    bad = 0
    checked = 0
    for name, (mapping, gen) in cases().items():
        if a.only and name not in a.only.split(","):
            continue
        idx = "ft_" + name.lower()
        props = {"f": mapping, "tag": {"type": "keyword"}}
        if name == "text_slop_gap":
            pass
        for n in (plugin, stock):
            n.req("DELETE", f"/{idx}")
            st, resp = n.req("PUT", f"/{idx}", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "analysis": {"normalizer": {"lc": {"type": "custom", "filter": ["lowercase"]}}}}, "mappings": {"properties": props}})
            if st != 200:
                print(f"{name}: create failed on {n.url}: {json.dumps(resp)[:200]}")
        lines = []
        rr = random.Random(a.seed * 31 + len(name))
        for i in range(a.docs):
            d = {"tag": f"t{i % 5}"}
            if rr.random() > 0.12:
                v = gen(rr)
                if rr.random() < .2 and name not in ("nested", "flat_object", "object_dynamic", "object_strict_ok", "rank_features", "completion") and not name.endswith("_range"):
                    v = [v, gen(rr)]
                d["f"] = v
            lines += [json.dumps({"index": {"_id": f"d{i}"}}), json.dumps(d)]
        payload = ("\n".join(lines) + "\n").encode()
        res = []
        for n in (plugin, stock):
            st, resp = n.req("POST", f"/{idx}/_bulk?refresh=true", payload, raw=True)
            res.append([(it["index"].get("status"), it["index"].get("error", {}).get("type")) for it in resp.get("items", [])] if st == 200 else [("http", st)])
            n.req("POST", f"/{idx}/_forcemerge?max_num_segments=1") if a.merge else None
            n.req("POST", f"/{idx}/_refresh")
        if res[0] != res[1]:
            k = next((j for j, (x, y) in enumerate(zip(res[0], res[1])) if x != y), 0)
            bad += 1
            print(f"{name}: indexing outcomes differ at doc {k}: plugin={res[0][k:k+1]} stock={res[1][k:k+1]} doc={lines[2 * k + 1][:150]}")
        cs = [(("query", q), {"size": a.docs + 10, "query": q, "track_total_hits": True, "sort": [{"_seq_no": "asc"}] if False else None}) for q in queries(name, r)]
        for kind, q in [("query", q) for q in queries(name, r)]:
            body = {"size": a.docs + 10, "query": q, "track_total_hits": True}
            checked += 1
            p = plugin.req("POST", f"/{idx}/_search", body)
            s = stock.req("POST", f"/{idx}/_search", body)
            d = cmp_search(p, s, False)
            if d:
                bad += 1
                print(f"{name}: query {json.dumps(q)[:200]} -> {d[:300]}")
        for ag in aggs(name, r):
            body = {"size": 0, "aggs": {"x": ag}}
            checked += 1
            p = plugin.req("POST", f"/{idx}/_search", body)
            s = stock.req("POST", f"/{idx}/_search", body)
            d = cmp_search(p, s, False)
            if d:
                bad += 1
                print(f"{name}: agg {json.dumps(ag)[:200]} -> {d[:300]}")
        # sort by the field, and docvalue_fields / fields API
        for body in [{"size": 20, "sort": [{"f": {"order": o, "missing": m}}, {"_seq_no": "asc"}], "_source": False} for o in ("asc", "desc") for m in ("_first", "_last")] + \
                    [{"size": 30, "docvalue_fields": ["f"], "_source": False, "sort": [{"_seq_no": "asc"}]}, {"size": 30, "fields": ["f"], "_source": False, "sort": [{"_seq_no": "asc"}]}]:
            checked += 1
            p = plugin.req("POST", f"/{idx}/_search", body)
            s = stock.req("POST", f"/{idx}/_search", body)
            if p[0] != s[0]:
                bad += 1
                print(f"{name}: {json.dumps(body)[:150]} status plugin={p[0]} stock={s[0]}")
            elif p[0] == 200:
                dd = diff(p[1]["hits"]["hits"], s[1]["hits"]["hits"], "hits", ignore={"_score", "_index", "_seq_no", "_primary_term", "_version"})
                if dd:
                    bad += 1
                    print(f"{name}: {json.dumps(body)[:150]} -> {dd[:300]}")
    print(f"checked {checked} requests, mismatches: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
