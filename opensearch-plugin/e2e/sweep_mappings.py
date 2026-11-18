#!/usr/bin/env python3
"""Mapping-option and index-setting matrix against stock: each variant index gets the same documents
on a plugin node and a stock node, then a battery of queries, aggregations, sorts and `_source` /
`fields` reads built from its field list. Any difference in hits, scores, order, aggregations,
sort values or fetched fields is printed.

    python3 opensearch-plugin/e2e/sweep_mappings.py [--plugin URL] [--stock URL] [--seed N] [--only NAME]
"""
import argparse
import json
import random

import fuzz_diff as fd

WORDS = fd.WORDS
N = 500


def txt(r, lo=1, hi=12):
    return " ".join(r.choice(WORDS[:30] + ["Café", "NAÏVE", "foo-bar", "x_y", "3.14", "e-mail", "O'Neil", "日本"]) for _ in range(r.randint(lo, hi)))


def variants():
    v = {}
    text_opts = {}
    for io in ("docs", "freqs", "positions", "offsets"):
        text_opts[f"t_{io}"] = {"type": "text", "index_options": io}
    text_opts.update({
        "t_nonorms": {"type": "text", "norms": False},
        "t_phrases": {"type": "text", "index_phrases": True},
        "t_prefixes": {"type": "text", "index_prefixes": {"min_chars": 1, "max_chars": 4}},
        "t_gap": {"type": "text", "position_increment_gap": 50},
        "t_gap0": {"type": "text", "position_increment_gap": 0},
        "t_store": {"type": "text", "store": True, "term_vector": "with_positions_offsets_payloads"},
        "t_ws": {"type": "text", "analyzer": "whitespace", "search_analyzer": "standard"},
        "t_fielddata": {"type": "text", "fielddata": True},
        "t_sim": {"type": "text", "analyzer": "english", "search_analyzer": "simple", "search_quote_analyzer": "standard"},
        "t_stop": {"type": "text", "analyzer": "stop"},
        "t_shingle": {"type": "text", "analyzer": "sh"},
        "t_ngram": {"type": "text", "analyzer": "ng"},
        "t_syn": {"type": "text", "analyzer": "std_syn", "search_analyzer": "std_syn"},
        "t_mot": {"type": "match_only_text"},
        "t_copy": {"type": "text", "copy_to": ["t_copy_target"]},
        "t_copy_target": {"type": "text"},
        "t_sayt": {"type": "search_as_you_type", "max_shingle_size": 3},
        "t_tokcount": {"type": "text", "fields": {"count": {"type": "token_count", "analyzer": "standard"}, "kw": {"type": "keyword", "ignore_above": 20}, "en": {"type": "text", "analyzer": "english"}}},
    })
    kw_opts = {
        "k_plain": {"type": "keyword"},
        "k_norm": {"type": "keyword", "normalizer": "lc", "ignore_above": 12, "null_value": "NULL"},
        "k_norms": {"type": "keyword", "norms": True},
        "k_nodv": {"type": "keyword", "doc_values": False},
        "k_noidx": {"type": "keyword", "index": False},
        "k_eager": {"type": "keyword", "eager_global_ordinals": True},
        "k_split": {"type": "keyword", "split_queries_on_whitespace": True},
        "k_const": {"type": "constant_keyword", "value": "const"},
        "k_wild": {"type": "wildcard"},
        "k_flat": {"type": "flat_object"},
        "k_store": {"type": "keyword", "store": True},
    }
    num_opts = {}
    for t in ("long", "integer", "short", "byte", "double", "float", "half_float", "unsigned_long"):
        num_opts[f"n_{t}"] = {"type": t}
        num_opts[f"n_{t}_nv"] = {"type": t, "null_value": 3, "ignore_malformed": True, "coerce": False}
        num_opts[f"n_{t}_nodv"] = {"type": t, "doc_values": False}
        num_opts[f"n_{t}_noidx"] = {"type": t, "index": False}
    num_opts["n_scaled"] = {"type": "scaled_float", "scaling_factor": 10}
    num_opts["n_scaled_nv"] = {"type": "scaled_float", "scaling_factor": 1000, "null_value": 1.5, "ignore_malformed": True}
    date_opts = {
        "d_def": {"type": "date"},
        "d_fmt": {"type": "date", "format": "yyyy/MM/dd HH:mm||epoch_second||strict_date"},
        "d_nanos": {"type": "date_nanos"},
        "d_nv": {"type": "date", "null_value": "1999-01-01", "ignore_malformed": True},
        "d_nodv": {"type": "date", "doc_values": False},
        "d_millis": {"type": "date", "format": "epoch_millis"},
    }
    misc = {
        "b_plain": {"type": "boolean"}, "b_nv": {"type": "boolean", "null_value": True}, "b_nodv": {"type": "boolean", "doc_values": False},
        "ip_plain": {"type": "ip"}, "ip_nv": {"type": "ip", "null_value": "0.0.0.1", "ignore_malformed": True},
        "bin": {"type": "binary", "doc_values": True},
        "r_int": {"type": "integer_range"}, "r_date": {"type": "date_range"}, "r_ip": {"type": "ip_range"}, "r_dbl": {"type": "double_range"},
        "g_pt": {"type": "geo_point"}, "g_shape": {"type": "geo_shape"},
        "alias_n": {"type": "alias", "path": "n_long"}, "alias_t": {"type": "alias", "path": "t_positions"},
        "rf": {"type": "rank_feature"}, "rfs": {"type": "rank_features"},
    }
    base = {}
    for d in (text_opts, kw_opts, num_opts, date_opts, misc):
        base.update(d)
    analysis = {
        "normalizer": {"lc": {"type": "custom", "filter": ["lowercase", "asciifolding"]}},
        "filter": {"syn": {"type": "synonym", "synonyms": ["alpha, a1", "quick brown => fast", "xi, ksi, chi"]},
                   "shg": {"type": "shingle", "min_shingle_size": 2, "max_shingle_size": 3, "output_unigrams": True}},
        "tokenizer": {"ngt": {"type": "ngram", "min_gram": 2, "max_gram": 3}},
        "analyzer": {"sh": {"tokenizer": "standard", "filter": ["lowercase", "shg"]},
                     "ng": {"tokenizer": "ngt", "filter": ["lowercase"]},
                     "std_syn": {"tokenizer": "standard", "filter": ["lowercase", "syn"]}},
    }
    v["base"] = ({"number_of_shards": 2}, {"properties": base}, analysis)
    v["sorted"] = ({"number_of_shards": 1, "index.sort.field": ["n_long", "k_plain"], "index.sort.order": ["desc", "asc"], "index.sort.missing": ["_last", "_first"]},
                   {"properties": base}, analysis)
    v["sorted_mode"] = ({"number_of_shards": 2, "index.sort.field": ["n_double"], "index.sort.order": ["asc"], "index.sort.mode": ["max"]}, {"properties": base}, analysis)
    v["strict_dyn_off"] = ({"number_of_shards": 1}, {"dynamic": "false", "properties": base}, analysis)
    v["source_excl"] = ({"number_of_shards": 1}, {"_source": {"excludes": ["t_*", "k_flat"]}, "properties": base}, analysis)
    v["codec_best"] = ({"number_of_shards": 1, "codec": "best_compression"}, {"properties": base}, analysis)
    v["codec_zstd"] = ({"number_of_shards": 1, "codec": "zstd_no_dict"}, {"properties": base}, analysis)
    v["many_seg"] = ({"number_of_shards": 1, "refresh_interval": "-1", "index.merge.policy.segments_per_tier": 2}, {"properties": base}, analysis)
    return v


def make_doc(r, i):
    d = {}
    m = lambda p: r.random() < p
    for k in ("t_docs", "t_freqs", "t_positions", "t_offsets", "t_nonorms", "t_phrases", "t_prefixes", "t_store", "t_ws", "t_fielddata", "t_sim",
              "t_stop", "t_shingle", "t_ngram", "t_syn", "t_mot", "t_copy", "t_sayt", "t_tokcount", "t_gap", "t_gap0"):
        if m(0.8):
            d[k] = txt(r) if m(0.85) else [txt(r, 1, 4), txt(r, 1, 4)]
    for k in ("k_plain", "k_norm", "k_norms", "k_nodv", "k_noidx", "k_eager", "k_split", "k_wild", "k_store"):
        if m(0.8):
            d[k] = r.choice(["red", "Green", "BLUE", "a b", "", "x*y", "x.y", "long-" * 5, "Café", "10", None])
            if m(0.2):
                d[k] = [r.choice(["red", "green"]), "blue"]
    if m(0.7):
        d["k_const"] = "const"
    if m(0.6):
        d["k_flat"] = {"a": r.choice(["x", "y", "z"]), "b": {"c": r.randint(0, 3)}, "d": [1, 2]}
    for t in ("long", "integer", "short", "byte", "double", "float", "half_float", "unsigned_long"):
        for suffix in ("", "_nv", "_nodv", "_noidx"):
            if m(0.75):
                hi = {"long": 2**62, "integer": 2**31 - 1, "short": 32767, "byte": 127, "unsigned_long": 2**64 - 1}.get(t, 1000)
                pool = [0, 1, -1, 5, 42, 100, "7", "abc" if suffix == "_nv" else "8", None, 3.7 if suffix == "_nv" else 4, hi]
                if t == "unsigned_long":
                    pool = [0, 1, 5, 42, 2**63, 2**64 - 1, "9", None]
                if t in ("double", "float", "half_float"):
                    pool += [0.5, -2.25, 1e5, 1e-5]
                d[f"n_{t}{suffix}"] = r.choice(pool)
                if m(0.1):
                    d[f"n_{t}{suffix}"] = [r.choice(pool[:5] if t != "unsigned_long" else [1, 2, 3]) for _ in range(3)]
    for k in ("n_scaled", "n_scaled_nv"):
        if m(0.7):
            d[k] = r.choice([0, 1.23, -4.56, 100.001, None, "x" if k.endswith("nv") else 2.2])
    ts = lambda: f"{r.randint(1990, 2030)}-{r.randint(1,12):02d}-{r.randint(1,28):02d}"
    if m(0.8):
        d["d_def"] = r.choice([ts(), ts() + "T10:20:30Z", r.randint(0, 2 * 10**12), ts() + "T10:20:30.123+02:00"])
    if m(0.8):
        d["d_fmt"] = r.choice([ts().replace("-", "/") + " 10:20", r.randint(0, 2 * 10**9), ts()])
    if m(0.8):
        d["d_nanos"] = r.choice([ts() + "T10:20:30.123456789Z", ts(), r.randint(0, 2 * 10**12)])
    if m(0.8):
        d["d_nv"] = r.choice([ts(), None, "garbage"])
    if m(0.7):
        d["d_nodv"] = ts()
    if m(0.7):
        d["d_millis"] = r.randint(0, 2 * 10**12)
    if m(0.7):
        d["b_plain"] = r.choice([True, False, "true", "false", ""])
    if m(0.7):
        d["b_nv"] = r.choice([True, False, None])
    if m(0.5):
        d["b_nodv"] = r.choice([True, False])
    if m(0.7):
        d["ip_plain"] = r.choice(["10.0.0.1", "192.168.1.5", "::1", "2001:db8::1", "::ffff:1.2.3.4"])
    if m(0.7):
        d["ip_nv"] = r.choice(["10.0.0.9", None, "bad"])
    if m(0.5):
        d["bin"] = r.choice(["U29tZSBiaW5hcnk=", "AAAA", "/w=="])
    if m(0.5):
        a = r.randint(-50, 50)
        d["r_int"] = {"gte": a, "lte": a + r.randint(0, 30)}
    if m(0.5):
        d["r_date"] = {"gte": ts(), "lt": "2031-01-01"}
    if m(0.5):
        d["r_ip"] = r.choice(["10.0.0.0/24", {"gte": "10.0.0.5", "lte": "10.0.1.5"}, "192.168.0.0/16"])
    if m(0.5):
        a = r.uniform(-5, 5)
        d["r_dbl"] = {"gte": a, "lte": a + r.uniform(0, 3)}
    if m(0.6):
        d["g_pt"] = r.choice([{"lat": r.uniform(-85, 85), "lon": r.uniform(-180, 180)}, [r.uniform(-180, 180), r.uniform(-85, 85)], "41.12,-71.34"])
    if m(0.5):
        x, y = r.uniform(-100, 100), r.uniform(-50, 50)
        d["g_shape"] = r.choice([{"type": "point", "coordinates": [x, y]}, {"type": "envelope", "coordinates": [[x, y + 5], [x + 5, y]]},
                                 {"type": "polygon", "coordinates": [[[x, y], [x + 3, y], [x + 3, y + 3], [x, y + 3], [x, y]]]}])
    if m(0.6):
        d["rf"] = r.choice([1, 2.5, 10, 0.1])
    if m(0.6):
        d["rfs"] = {"a": r.randint(1, 5), "b": r.random() + 0.1}
    if m(0.1):
        d["extra"] = {"dyn": r.randint(0, 3), "s": "text"}
    return d


def queries(r):
    q = []
    w = lambda: r.choice(WORDS[:30])
    for f in ("t_docs", "t_freqs", "t_positions", "t_offsets", "t_nonorms", "t_phrases", "t_prefixes", "t_store", "t_ws", "t_sim", "t_stop", "t_shingle", "t_ngram", "t_syn", "t_mot", "t_copy_target", "alias_t", "t_sayt", "t_gap", "t_gap0", "t_fielddata", "t_tokcount", "t_tokcount.en"):
        q.append((f"match {f}", {"match": {f: f"{w()} {w()}"}}))
        q.append((f"match and {f}", {"match": {f: {"query": f"{w()} {w()}", "operator": "and"}}}))
        q.append((f"phrase {f}", {"match_phrase": {f: f"{w()} {w()}"}}))
        q.append((f"phrase slop {f}", {"match_phrase": {f: {"query": f"{w()} {w()}", "slop": 2}}}))
        q.append((f"phrase_prefix {f}", {"match_phrase_prefix": {f: f"{w()} {w()[:2]}"}}))
        q.append((f"prefix {f}", {"prefix": {f: w()[:3]}}))
        q.append((f"wildcard {f}", {"wildcard": {f: f"*{w()[1:4]}*"}}))
        q.append((f"fuzzy {f}", {"fuzzy": {f: {"value": w(), "fuzziness": 1}}}))
        q.append((f"qs {f}", {"query_string": {"query": f"{w()} AND \"{w()} {w()}\" OR {w()[:2]}*", "default_field": f}}))
        q.append((f"mbp {f}", {"match_bool_prefix": {f: f"{w()} {w()[:2]}"}}))
        q.append((f"exists {f}", {"exists": {"field": f}}))
        q.append((f"span {f}", {"span_near": {"clauses": [{"span_term": {f: w()}}, {"span_term": {f: w()}}], "slop": 3, "in_order": False}}))
        q.append((f"intervals {f}", {"intervals": {f: {"match": {"query": f"{w()} {w()}", "max_gaps": 3, "ordered": False}}}}))
    q.append(("sayt 2gram", {"match": {"t_sayt._2gram": f"{w()} {w()}"}}))
    q.append(("sayt 3gram", {"match": {"t_sayt._3gram": f"{w()} {w()} {w()}"}}))
    q.append(("sayt index_prefix", {"match": {"t_sayt._index_prefix": w()[:3]}}))
    q.append(("sayt bool_prefix multi", {"multi_match": {"query": f"{w()} {w()[:2]}", "type": "bool_prefix", "fields": ["t_sayt", "t_sayt._2gram", "t_sayt._3gram"]}}))
    q.append(("tokcount range", {"range": {"t_tokcount.count": {"gte": 3, "lt": 8}}}))
    q.append(("copy_to target", {"match": {"t_copy_target": w()}}))
    for f in ("k_plain", "k_norm", "k_norms", "k_nodv", "k_noidx", "k_eager", "k_split", "k_const", "k_wild", "k_store", "t_tokcount.kw"):
        for val in ("red", "Red", "green", "BLUE", "a b", "", "x*y", "NULL", "const", "Café", "10", "long-long-long-long-long-"):
            q.append((f"term {f}={val}", {"term": {f: val}}))
        q.append((f"terms {f}", {"terms": {f: ["red", "blue", "a b", "NULL", "const"]}}))
        q.append((f"prefix {f}", {"prefix": {f: "r"}}))
        q.append((f"prefix ci {f}", {"prefix": {f: {"value": "R", "case_insensitive": True}}}))
        q.append((f"wildcard {f}", {"wildcard": {f: "*e*"}}))
        q.append((f"wildcard ci {f}", {"wildcard": {f: {"value": "*E?", "case_insensitive": True}}}))
        q.append((f"regexp {f}", {"regexp": {f: "r.*|(bl|gr)e+n?"}}))
        q.append((f"range {f}", {"range": {f: {"gte": "a", "lte": "m"}}}))
        q.append((f"fuzzy {f}", {"fuzzy": {f: {"value": "gren", "fuzziness": 1}}}))
        q.append((f"match {f}", {"match": {f: "a b"}}))
        q.append((f"exists {f}", {"exists": {"field": f}}))
        q.append((f"qs {f}", {"query_string": {"query": f"{f}:red OR {f}:(green OR blue)"}}))
    q.append(("flat_object term", {"term": {"k_flat.a": "x"}}))
    q.append(("flat_object nested term", {"term": {"k_flat.b.c": 2}}))
    q.append(("flat_object exists", {"exists": {"field": "k_flat.b.c"}}))
    q.append(("flat_object prefix", {"prefix": {"k_flat": "x"}}))
    for t in ("long", "integer", "short", "byte", "double", "float", "half_float", "unsigned_long"):
        for suf in ("", "_nv", "_nodv", "_noidx"):
            f = f"n_{t}{suf}"
            q.append((f"term {f}", {"term": {f: 5}}))
            q.append((f"terms {f}", {"terms": {f: [0, 1, 42, 3, 7]}}))
            q.append((f"range {f}", {"range": {f: {"gte": -1, "lt": 50}}}))
            q.append((f"range open {f}", {"range": {f: {"gt": 3.5}}}))
            q.append((f"range str {f}", {"range": {f: {"gte": "0", "lte": "10"}}}))
            q.append((f"exists {f}", {"exists": {"field": f}}))
            q.append((f"match {f}", {"match": {f: "42"}}))
            q.append((f"qs {f}", {"query_string": {"query": f"{f}:[0 TO 10] OR {f}:>=100"}}))
    for f in ("n_scaled", "n_scaled_nv"):
        q.append((f"term {f}", {"term": {f: 1.23}}))
        q.append((f"range {f}", {"range": {f: {"gt": -5, "lte": 1.5}}}))
        q.append((f"exists {f}", {"exists": {"field": f}}))
    q.append(("alias numeric", {"range": {"alias_n": {"gte": 0, "lt": 100}}}))
    for f in ("d_def", "d_fmt", "d_nanos", "d_nv", "d_nodv", "d_millis"):
        q.append((f"range {f}", {"range": {f: {"gte": "2000-01-01", "lt": "2015-06-30"}}}))
        q.append((f"range epoch {f}", {"range": {f: {"gte": 0, "lte": 10**9}}}))
        q.append((f"range fmt {f}", {"range": {f: {"gte": "01/01/2000", "format": "dd/MM/yyyy", "time_zone": "+09:00"}}}))
        q.append((f"term {f}", {"term": {f: "2000-01-01"}}))
        q.append((f"exists {f}", {"exists": {"field": f}}))
        q.append((f"round {f}", {"range": {f: {"gte": "2010-05||/M", "lt": "2010-05||+1M/d"}}}))
    for f in ("b_plain", "b_nv", "b_nodv"):
        q.append((f"term {f}", {"term": {f: True}}))
        q.append((f"term str {f}", {"term": {f: "false"}}))
        q.append((f"exists {f}", {"exists": {"field": f}}))
    for f in ("ip_plain", "ip_nv"):
        q.append((f"term {f}", {"term": {f: "10.0.0.1"}}))
        q.append((f"cidr {f}", {"term": {f: "10.0.0.0/8"}}))
        q.append((f"range {f}", {"range": {f: {"gte": "10.0.0.0", "lte": "192.168.255.255"}}}))
        q.append((f"v6 {f}", {"range": {f: {"gte": "::", "lte": "ffff::"}}}))
        q.append((f"exists {f}", {"exists": {"field": f}}))
    q.append(("binary exists", {"exists": {"field": "bin"}}))
    for f, vals in (("r_int", [5, 30]), ("r_date", ["2010-01-01", "2020-01-01"]), ("r_ip", ["10.0.0.7", "10.0.1.1"]), ("r_dbl", [0.5, 2.0])):
        for rel in ("intersects", "contains", "within"):
            q.append((f"range field {f} {rel}", {"range": {f: {"gte": vals[0], "lte": vals[1], "relation": rel}}}))
        q.append((f"term {f}", {"term": {f: vals[0]}}))
        q.append((f"exists {f}", {"exists": {"field": f}}))
    q.append(("geo bbox", {"geo_bounding_box": {"g_pt": {"top_left": {"lat": 60, "lon": -100}, "bottom_right": {"lat": -10, "lon": 60}}}}))
    q.append(("geo bbox dateline", {"geo_bounding_box": {"g_pt": {"top_left": {"lat": 60, "lon": 150}, "bottom_right": {"lat": -10, "lon": -150}}}}))
    q.append(("geo distance", {"geo_distance": {"distance": "2500km", "g_pt": [10, 10]}}))
    q.append(("geo shape intersects", {"geo_shape": {"g_shape": {"shape": {"type": "envelope", "coordinates": [[-50, 40], [50, -20]]}, "relation": "intersects"}}}))
    q.append(("geo shape within", {"geo_shape": {"g_shape": {"shape": {"type": "envelope", "coordinates": [[-90, 50], [90, -50]]}, "relation": "within"}}}))
    q.append(("geo shape disjoint", {"geo_shape": {"g_shape": {"shape": {"type": "circle", "coordinates": [0, 0], "radius": "3000km"}, "relation": "disjoint"}}}))
    q.append(("rank_feature", {"rank_feature": {"field": "rf"}}))
    q.append(("rank_feature sigmoid", {"rank_feature": {"field": "rf", "sigmoid": {"pivot": 2, "exponent": 0.6}}}))
    q.append(("rank_feature log", {"rank_feature": {"field": "rfs.a", "log": {"scaling_factor": 4}}}))
    q.append(("rank_feature in bool", {"bool": {"must": {"match": {"t_docs": w()}}, "should": {"rank_feature": {"field": "rfs.b", "saturation": {"pivot": 0.5}}}}}))
    q.append(("exists dyn", {"exists": {"field": "extra.dyn"}}))
    q.append(("terms extra", {"term": {"extra.s": "text"}}))
    q.append(("match extra", {"match": {"extra.s": "text"}}))
    q.append(("exists wildcard field", {"exists": {"field": "extra.*"}}))
    q.append(("exists object", {"exists": {"field": "extra"}}))
    q.append(("exists prefix field", {"exists": {"field": "n_*"}}))
    return q


def aggs(r):
    a = []
    for f in ("k_plain", "k_norm", "k_nodv", "k_eager", "k_split", "k_const", "k_wild", "t_fielddata", "t_tokcount.kw", "n_long", "n_integer_nv", "n_unsigned_long", "n_double", "n_float_nv", "n_half_float", "n_scaled", "n_byte", "b_plain", "b_nv", "ip_plain", "ip_nv", "d_def", "d_nanos", "d_nv", "d_fmt", "alias_n", "n_long_nodv", "d_nodv", "k_flat.a", "t_tokcount.count"):
        if not f.startswith("k_flat"):   # stock renders flat_object bucket keys as Object@identityHash
            a.append((f"terms {f}", {"t": {"terms": {"field": f, "size": 50, "order": [{"_count": "desc"}, {"_key": "asc"}]}}}))
        if not f.startswith("k_flat"):
          a.append((f"terms missing {f}", {"t": {"terms": {"field": f, "size": 5, "missing": "zz" if f.startswith(("k", "t", "ip")) and f != "t_tokcount.count" else 0 if not f.startswith("d") else "2000-01-01", "order": {"_key": "desc"}}}}))
        a.append((f"cardinality {f}", {"c": {"cardinality": {"field": f}}}))
        a.append((f"value_count {f}", {"c": {"value_count": {"field": f}}}))
        a.append((f"missing {f}", {"m": {"missing": {"field": f}}}))
    for f in ("n_long", "n_integer", "n_short", "n_byte", "n_double", "n_float", "n_half_float", "n_unsigned_long", "n_scaled", "n_scaled_nv", "n_long_nv", "d_def", "d_nanos", "d_fmt", "d_nv", "d_millis", "alias_n", "n_double_nodv", "b_plain", "b_nv"):
        a.append((f"stats {f}", {"s": {"stats": {"field": f}}}))
        a.append((f"ext stats {f}", {"s": {"extended_stats": {"field": f}}}))
        a.append((f"percentiles {f}", {"s": {"percentiles": {"field": f, "percents": [5, 50, 95]}}}))
        a.append((f"min/max {f}", {"mn": {"min": {"field": f}}, "mx": {"max": {"field": f}}}))
        if not f.startswith("b_"):
            a.append((f"histogram {f}", {"h": {"histogram": {"field": f, "interval": 10 ** 8 if f.startswith("d") else 25 if f.startswith("n_") else 1, "min_doc_count": 1}}}) if not f.startswith("d") else
                     (f"date_hist {f}", {"h": {"date_histogram": {"field": f, "calendar_interval": "year", "format": "yyyy", "min_doc_count": 1}}}))
            a.append((f"range {f}", {"r": {"range": {"field": f, "ranges": [{"to": 10}, {"from": 10, "to": 1000}, {"from": 1000}]}}}) if not f.startswith("d") else
                     (f"date_range {f}", {"r": {"date_range": {"field": f, "ranges": [{"to": "2000"}, {"from": "2000", "to": "2020"}, {"from": "2020"}]}}}))
    a.append(("ip_range agg", {"r": {"ip_range": {"field": "ip_plain", "ranges": [{"mask": "10.0.0.0/8"}, {"from": "192.168.0.0", "to": "192.169.0.0"}]}}}))
    a.append(("geo_bounds", {"g": {"geo_bounds": {"field": "g_pt", "wrap_longitude": True}}}))
    a.append(("geo_centroid", {"g": {"geo_centroid": {"field": "g_pt"}}}))
    a.append(("geohash grid", {"g": {"geohash_grid": {"field": "g_pt", "precision": 2}}}))
    a.append(("geotile grid", {"g": {"geotile_grid": {"field": "g_pt", "precision": 4}}}))
    a.append(("range field histogram-free terms", {"t": {"terms": {"script": "doc['r_int'].size()"}}}))
    a.append(("significant_terms kw", {"s": {"significant_terms": {"field": "k_plain", "min_doc_count": 1}}}))
    a.append(("composite mixed", {"c": {"composite": {"size": 30, "sources": [{"a": {"terms": {"field": "k_norm", "missing_bucket": True}}}, {"b": {"terms": {"field": "n_short"}}}, {"d": {"date_histogram": {"field": "d_def", "calendar_interval": "year"}}}]}}}))
    a.append(("composite ip/bool", {"c": {"composite": {"size": 30, "sources": [{"a": {"terms": {"field": "ip_plain"}}}, {"b": {"terms": {"field": "b_plain"}}}]}}}))
    a.append(("composite nanos", {"c": {"composite": {"size": 30, "sources": [{"a": {"date_histogram": {"field": "d_nanos", "fixed_interval": "365d"}}}]}}}))
    a.append(("composite desc unsigned", {"c": {"composite": {"size": 30, "sources": [{"a": {"terms": {"field": "n_unsigned_long", "order": "desc"}}}]}}}))
    a.append(("composite half_float", {"c": {"composite": {"size": 30, "sources": [{"a": {"terms": {"field": "n_half_float"}}}, {"b": {"histogram": {"field": "n_float", "interval": 3}}}]}}}))
    return a


def sorts():
    s = []
    for f in ("k_plain", "k_norm", "k_nodv", "k_const", "k_wild", "n_long", "n_integer_nv", "n_short", "n_byte", "n_double", "n_float", "n_half_float", "n_unsigned_long", "n_scaled", "d_def", "d_nanos", "d_fmt", "d_nv", "d_millis", "b_plain", "b_nv", "ip_plain", "ip_nv", "alias_n", "t_fielddata", "t_tokcount.count", "n_long_nodv"):
        for order in ("asc", "desc"):
            for missing in (None, "_first", "_last"):
                spec = {"order": order}
                if missing:
                    spec["missing"] = missing
                s.append((f"sort {f} {order} {missing}", [{f: spec}, {"_seq_no": "asc"}]))
    for f in ("n_long", "k_plain", "n_double", "d_def", "d_nanos"):
        s.append((f"sort {f} mode", [{f: {"order": "asc", "mode": "max"}}, {"_seq_no": "asc"}]))
    s.append(("sort geo", [{"_geo_distance": {"g_pt": [0, 0], "order": "asc", "unit": "km"}}, {"_seq_no": "asc"}]))
    s.append(("sort 3 keys", [{"k_norm": "asc"}, {"n_short": "desc"}, {"d_def": "asc"}, {"_seq_no": "asc"}]))
    s.append(("sort _doc", [{"_doc": "asc"}]))
    return s


def hits_equal(p, s, ordered):
    return fd.cmp_search(p, s, ordered)


def fetch_cmp(p, s):
    """_source / fields / docvalue_fields / stored_fields of every hit."""
    if p[0] != 200 or s[0] != 200:
        return None if p[0] == s[0] else f"status {p[0]} vs {s[0]}"
    d = fd.diff([h for h in p[1]["hits"]["hits"]], [h for h in s[1]["hits"]["hits"]], "hits", 1e-6)
    return d


def run(a):
    plugin, stock = fd.Node(a.plugin), fd.Node(a.stock)
    nodes = [plugin, stock]
    for n in nodes:
        n.req("PUT", "/_cluster/settings", {"transient": {"search.concurrent_segment_search.mode": "none"}})
    findings = []
    count = [0]

    def log(f):
        findings.append(f)
        print(json.dumps(f)[:900], flush=True)

    for name, (settings, mappings, analysis) in variants().items():
        if a.only and a.only != name:
            continue
        idx = f"mx_{name}"
        for n in nodes:
            n.req("DELETE", f"/{idx}")
            st, resp = n.req("PUT", f"/{idx}", {"settings": {"number_of_replicas": 0, "analysis": analysis, **settings}, "mappings": mappings})
            if st != 200:
                log({"kind": "create", "variant": name, "node": n.url, "resp": str(resp)[:300]})
        r = random.Random(a.seed)
        ops = []
        for i in range(N):
            ops.append((i, make_doc(r, i)))
        for _ in range(N // 6):
            i = r.randrange(N)
            ops.append((i, make_doc(r, i)))
        dels = [r.randrange(N) for _ in range(N // 10)]
        for s0 in range(0, len(ops), 100):
            lines = []
            for i, d in ops[s0:s0 + 100]:
                lines.append(json.dumps({"index": {"_id": f"d{i}"}}))
                lines.append(json.dumps(d))
            body = ("\n".join(lines) + "\n").encode()
            res = []
            for n in nodes:
                st, resp = n.req("POST", f"/{idx}/_bulk?refresh=true", body, raw=True)
                res.append([(list(it.values())[0].get("status"), (list(it.values())[0].get("error") or {}).get("type")) for it in resp.get("items", [])] if isinstance(resp, dict) else resp)
            if res[0] != res[1]:
                bad = [(k, x, y) for k, (x, y) in enumerate(zip(*res)) if x != y][:3]
                log({"kind": "bulk-diff", "variant": name, "batch": s0, "diff": str(bad)})
        res = []
        for n in nodes:
            lines = []
            for i in dels:
                lines.append(json.dumps({"delete": {"_id": f"d{i}"}}))
            n.req("POST", f"/{idx}/_bulk?refresh=true", ("\n".join(lines) + "\n").encode(), raw=True)
            if name != "many_seg":
                n.req("POST", f"/{idx}/_forcemerge?max_num_segments=1")
            n.req("POST", f"/{idx}/_refresh")
        rr = random.Random(a.seed + 1)
        for qn, qb in queries(rr):
            body = {"size": N, "query": qb}
            d, p, s = fd.run_case(plugin, stock, idx, body, False)
            count[0] += 1
            if d and not fd._resource(p) and not fd._resource(s):
                log({"kind": "query", "variant": name, "case": qn, "diff": d, "body": json.dumps(qb)[:300]})
        for an, ab in aggs(rr):
            body = {"size": 0, "aggs": ab}
            d, p, s = fd.run_case(plugin, stock, idx, body, False)
            count[0] += 1
            if d and not fd._resource(p) and not fd._resource(s):
                log({"kind": "agg", "variant": name, "case": an, "diff": d, "body": json.dumps(ab)[:300]})
            body = {"size": 0, "query": {"range": {"n_long": {"gte": 0}}}, "aggs": ab}
            d, p, s = fd.run_case(plugin, stock, idx, body, False)
            count[0] += 1
            if d and not fd._resource(p) and not fd._resource(s):
                log({"kind": "agg+q", "variant": name, "case": an, "diff": d, "body": json.dumps(ab)[:300]})
        for sn, sb in sorts():
            body = {"size": 40, "query": {"match_all": {}}, "sort": sb, "track_total_hits": True}
            d, p, s = fd.run_case(plugin, stock, idx, body, True)
            count[0] += 1
            if d and not fd._resource(p) and not fd._resource(s):
                log({"kind": "sort", "variant": name, "case": sn, "diff": d})
        fetches = [
            ("source", {"size": 60, "query": {"match_all": {}}, "sort": [{"_seq_no": "asc"}]}),
            ("fields star", {"size": 60, "query": {"match_all": {}}, "fields": ["*"], "_source": False, "sort": [{"_seq_no": "asc"}]}),
            ("docvalue star", {"size": 60, "query": {"match_all": {}}, "docvalue_fields": ["n_*", "d_*", "k_*", "b_*", "ip_*", "g_pt"], "_source": False, "sort": [{"_seq_no": "asc"}]}),
            ("stored", {"size": 60, "query": {"match_all": {}}, "stored_fields": ["t_store", "k_store"], "sort": [{"_seq_no": "asc"}]}),
            ("fields fmt", {"size": 60, "query": {"match_all": {}}, "fields": [{"field": "d_def", "format": "yyyy-MM-dd"}, {"field": "d_nanos", "format": "strict_date_optional_time_nanos"}, "g_pt", "n_scaled", "n_unsigned_long", "r_int", "r_date", "ip_nv"], "_source": False, "sort": [{"_seq_no": "asc"}]}),
            ("script_fields", {"size": 60, "query": {"match_all": {}}, "script_fields": {"a": {"script": "doc['n_long'].size() > 0 ? doc['n_long'].value : -1"}, "b": {"script": "doc['k_plain'].size() > 0 ? doc['k_plain'].value : 'none'"}, "c": {"script": "doc['d_def'].size() > 0 ? doc['d_def'].value.toString() : 'none'"}}, "_source": False, "sort": [{"_seq_no": "asc"}]}),
            ("highlight", {"size": 40, "query": {"bool": {"should": [{"match": {"t_offsets": "alpha beta"}}, {"match": {"t_positions": "alpha"}}, {"match_phrase": {"t_store": "alpha beta"}}, {"prefix": {"t_docs": "al"}}]}}, "highlight": {"fields": {"t_offsets": {"type": "unified"}, "t_positions": {"type": "unified"}, "t_store": {"type": "fvh"}, "t_docs": {"type": "plain"}}, "require_field_match": False}, "sort": [{"_seq_no": "asc"}]}),
            ("termvectors-free explain", {"size": 5, "explain": True, "query": {"bool": {"must": {"match": {"t_phrases": "alpha beta"}}, "should": [{"term": {"k_norm": "red"}}]}}, "sort": [{"_score": "desc"}, {"_seq_no": "asc"}]}),
        ]
        for fn, fb in fetches:
            p = plugin.req("POST", f"/{idx}/_search", fb)
            s = stock.req("POST", f"/{idx}/_search", fb)
            count[0] += 1
            d = fetch_cmp(p, s)
            if d:
                log({"kind": "fetch", "variant": name, "case": fn, "diff": d})
        # term vectors and mget on a sample
        for i in range(0, 60, 7):
            p = plugin.req("GET", f"/{idx}/_termvectors/d{i}?fields=t_store,t_docs,t_positions,t_offsets&positions=true&offsets=true&payloads=true&term_statistics=true&field_statistics=true")
            s = stock.req("GET", f"/{idx}/_termvectors/d{i}?fields=t_store,t_docs,t_positions,t_offsets&positions=true&offsets=true&payloads=true&term_statistics=true&field_statistics=true")
            count[0] += 1
            d = fd.diff(p[1], s[1], "tv", 1e-5) if p[0] == s[0] else f"status {p[0]} {s[0]}"
            if d:
                log({"kind": "termvectors", "variant": name, "doc": i, "diff": d})
        print(json.dumps({"variant": name, "cases_so_far": count[0], "findings": len(findings)}), flush=True)
    print(json.dumps({"cases": count[0], "findings": len(findings)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="http://localhost:9200")
    ap.add_argument("--stock", default="http://localhost:9201")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
