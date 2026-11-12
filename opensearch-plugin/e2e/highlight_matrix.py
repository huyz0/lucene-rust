"""Highlighting on the plugin node against stock OpenSearch: every highlight of
every hit of a matrix of indices (analyzers, offset sources, field types), query
shapes and highlight options must be identical on both.

Usage: highlight_matrix.py [load|check|all] [DOCS]
  PLUGIN (default http://localhost:9200) is a node with the plugin, STOCK
  (default http://localhost:9201) a stock one. `load` creates and fills the
  indices on both, `check` runs the matrix, `all` (the default) does both.
  HL_TYPES (default unified,plain,fvh) picks the highlighters whose option sets run; the fvh
  sets run on the indices with term vectors (and one without, for the error).

The corpus is generated (seeded): sentences of English words with
abbreviations, numbers, punctuation, HTML-special characters, accented and
CJK text, emoji (surrogate pairs), newlines, multi-valued fields and long
values, so passages break on every kind of boundary the break iterators
know. A row prints `ok` or the first differing hit; the plugin's stats show
how many fields ran natively and every fallback reason."""
import json, os, random, sys
import requests

PLUGIN = os.environ.get("PLUGIN", "http://localhost:9200")
STOCK = os.environ.get("STOCK", "http://localhost:9201")
S = requests.Session()

WORDS = ("the quick brown fox jumps over lazy dog search engine lucene rust "
         "highlight passage sentence fragment query match phrase index shard "
         "node cluster vector token analyzer stemming running runs ran foxes "
         "dogs jumping searched searches engines indexing nodes clusters "
         "apple banana cherry orange grape lemon melon peach plum pear "
         "red green blue yellow black white silver gold copper iron "
         "alpha beta gamma delta epsilon zeta theta kappa lambda sigma").split()
EXTRA = ["Dr. Smith", "e.g.", "U.S.A.", "3.14", "1,000", "C++", "<b>", "&amp;", "a&b", "x<y", "'quoted'",
         "\"double\"", "café", "naïve", "Zürich", "東京", "検索エンジン", "😀", "🦀 crab", "self-driving",
         "foo_bar", "www.example.com", "user@example.com", "A/B", "50%"]


def sentence(rnd):
    n = rnd.randint(3, 25)
    out = []
    for i in range(n):
        if rnd.random() < 0.08:
            out.append(rnd.choice(EXTRA))
        else:
            out.append(rnd.choice(WORDS))
    s = " ".join(out)
    s = s[0].upper() + s[1:]
    end = rnd.choice([".", ".", ".", "!", "?", "...", ";", ""])
    if rnd.random() < 0.1:
        s += "\n"
    return s + end


def text(rnd, lo, hi):
    return " ".join(sentence(rnd) for _ in range(rnd.randint(lo, hi)))


def doc(rnd, i):
    body = text(rnd, 1, 12)
    if i % 17 == 0:
        body = text(rnd, 40, 80)  # long values: many passages
    d = {
        "body": body,
        "title": sentence(rnd),
        "tags": [rnd.choice(WORDS) for _ in range(rnd.randint(0, 4))],
        "multi": [sentence(rnd) for _ in range(rnd.randint(1, 4))],
        "kw": rnd.choice(WORDS) + " " + rnd.choice(WORDS),
        "num": rnd.randint(0, 1000),
        "date": f"2026-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}",
    }
    extra = random.Random(i)  # its own stream, so the fields above stay as they were
    d["comments"] = [{"text": sentence(extra)} for _ in range(extra.randint(0, 3))]
    d["loc"] = {"lat": extra.uniform(-60, 60), "lon": extra.uniform(-170, 170)}
    if i % 11 == 0:
        d["body"] = ""  # empty value
    if i % 13 == 0:
        del d["title"]  # missing field
    return d


ANALYSIS = {
    "analyzer": {
        "ws_lower": {"type": "custom", "tokenizer": "whitespace", "filter": ["lowercase"]},
        "stop_stem": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "stop", "porter_stem"]},
        "syn": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "syns"]},
        "shingles": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "shingle"]},
        "ngram": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "edge"]},
    },
    "filter": {
        "syns": {"type": "synonym", "synonyms": ["fox, dog", "quick, fast", "search engine, retrieval system"]},
        "edge": {"type": "edge_ngram", "min_gram": 2, "max_gram": 5},
    },
}


def text_field(analyzer=None, **kw):
    f = {"type": "text", **kw}
    if analyzer:
        f["analyzer"] = analyzer
    return f


def mapping(body):
    return {
        "properties": {
            "body": body,
            "title": {"type": "text", "fields": {"en": {"type": "text", "analyzer": "english"}}},
            "tags": {"type": "keyword"},
            "multi": body,
            "kw": {"type": "keyword"},
            "num": {"type": "integer"},
            "date": {"type": "date"},
            "comments": {"type": "nested", "properties": {"text": {"type": "text"}}},
            "loc": {"type": "geo_point"},
        }
    }


INDICES = {
    "hl_standard": mapping(text_field()),
    "hl_english": mapping(text_field("english")),
    "hl_offsets": mapping(text_field(index_options="offsets")),
    "hl_tv": mapping(text_field(term_vector="with_positions_offsets")),
    "hl_offsets_tv": mapping(text_field(index_options="offsets", term_vector="with_positions_offsets")),
    "hl_ws": mapping(text_field("ws_lower")),
    "hl_stop": mapping(text_field("stop_stem")),
    "hl_syn": mapping(text_field("syn")),
    "hl_ngram": mapping(text_field("ngram")),
    "hl_stored": mapping(text_field(store=True)),
}

TV = "with_positions_offsets"


def fvh_mapping(analyzer=None, **kw):
    """Term vectors on every text field, and an english subfield for matched_fields."""
    en = {"type": "text", "analyzer": "english", "term_vector": TV}
    m = mapping(text_field(analyzer, term_vector=TV, fields={"en": en}, **kw))
    m["properties"]["title"] = {"type": "text", "term_vector": TV, "fields": {"en": en}}
    return m


# The fvh highlighter's own indices: stored or not, and analyzers that break it (edge_ngram).
FVH_INDICES = {
    "hl_fvh": fvh_mapping(),
    "hl_fvh_stored": fvh_mapping(store=True),
    "hl_fvh_ngram": fvh_mapping("ngram"),
    "hl_fvh_ngram_stored": fvh_mapping("ngram", store=True),
    "hl_fvh_syn": fvh_mapping("syn"),
    "hl_fvh_english": fvh_mapping("english"),
}
TYPES = os.environ.get("HL_TYPES", "unified,plain,fvh").split(",")
if "fvh" in TYPES:
    INDICES.update(FVH_INDICES)


def load(docs):
    for base in (PLUGIN, STOCK):
        for name, m in INDICES.items():
            S.delete(f"{base}/{name}")
            r = S.put(f"{base}/{name}", json={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0, "analysis": ANALYSIS},
                "mappings": m,
            })
            assert r.ok, r.text
            rnd = random.Random(42)
            lines = []
            for i in range(docs):
                lines.append(json.dumps({"index": {"_id": str(i)}}))
                lines.append(json.dumps(doc(rnd, i)))
                if len(lines) >= 2000:
                    r = S.post(f"{base}/{name}/_bulk", data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"})
                    assert r.ok and not r.json()["errors"], r.text[:500]
                    lines = []
            if lines:
                r = S.post(f"{base}/{name}/_bulk", data="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"})
                assert r.ok and not r.json()["errors"], r.text[:500]
            S.post(f"{base}/{name}/_refresh")
            # Several segments: a merge would make the two nodes' segment layouts differ anyway.
    print("loaded", docs, "docs into", len(INDICES), "indices on both nodes")


QUERIES = {
    "match": {"match": {"body": "quick fox"}},
    "match_and": {"match": {"body": {"query": "lucene rust", "operator": "and"}}},
    "match_extra": {"match": {"body": "café 東京 crab"}},
    "match_phrase": {"match_phrase": {"body": "brown fox"}},
    "match_phrase_slop": {"match_phrase": {"body": {"query": "quick fox", "slop": 3}}},
    "match_phrase_stop": {"match_phrase": {"body": "over the lazy"}},
    "phrase_and_term": {"bool": {"must": [{"match_phrase": {"body": "search engine"}}], "should": [{"match": {"body": "fox"}}]}},
    "multi_match": {"multi_match": {"query": "fox gold", "fields": ["body", "title"]}},
    "multi_match_phrase": {"multi_match": {"query": "lazy dog", "fields": ["body", "title"], "type": "phrase"}},
    "bool": {"bool": {"must": [{"match": {"body": "fox"}}], "should": [{"match": {"title": "dog"}}],
                      "must_not": [{"match": {"body": "melon"}}], "filter": [{"range": {"num": {"gte": 100}}}]}},
    "dis_max": {"dis_max": {"queries": [{"match": {"body": "apple"}}, {"match": {"title": "apple"}}]}},
    "constant_score": {"constant_score": {"filter": {"match": {"body": "banana"}}}},
    "boosted": {"match": {"body": {"query": "cherry", "boost": 3}}},
    "term": {"term": {"body": "fox"}},
    "terms": {"terms": {"body": ["fox", "dog", "gold"]}},
    "prefix": {"prefix": {"body": "sea"}},
    "wildcard": {"wildcard": {"body": "r*n*"}},
    "regexp": {"regexp": {"body": "gr[a-z]+"}},
    "fuzzy": {"fuzzy": {"body": "lucine"}},
    "match_fuzzy": {"match": {"body": {"query": "qiuck brwn", "fuzziness": "AUTO"}}},
    "query_string": {"query_string": {"query": "body:(fox OR do*) AND NOT body:melon"}},
    "query_string_phrase": {"query_string": {"query": "\"lazy dog\" OR gold~1", "default_field": "body"}},
    "query_string_range": {"query_string": {"query": "body:[apple TO cherry] AND fox", "default_field": "body"}},
    "simple_query_string": {"simple_query_string": {"query": "fox + (dog | gold) -melon", "fields": ["body"]}},
    "span_near": {"span_near": {"clauses": [{"span_term": {"body": "quick"}}, {"span_term": {"body": "fox"}}], "slop": 5, "in_order": False}},
    "span_or": {"span_or": {"clauses": [{"span_term": {"body": "red"}}, {"span_term": {"body": "blue"}}]}},
    "span_first": {"span_first": {"match": {"span_term": {"body": "quick"}}, "end": 6}},
    "span_not": {"span_not": {"include": {"span_term": {"body": "fox"}}, "exclude": {"span_near": {"clauses": [{"span_term": {"body": "red"}}, {"span_term": {"body": "fox"}}], "slop": 0, "in_order": True}}}},
    "span_containing": {"span_containing": {"big": {"span_near": {"clauses": [{"span_term": {"body": "quick"}}, {"span_term": {"body": "dog"}}], "slop": 8, "in_order": False}}, "little": {"span_term": {"body": "fox"}}}},
    "span_multi": {"span_near": {"clauses": [{"span_multi": {"match": {"prefix": {"body": "qui"}}}}, {"span_term": {"body": "fox"}}], "slop": 4, "in_order": False}},
    "phrase_syn": {"match_phrase": {"body": "quick fox"}},
    "function_score_phrase": {"function_score": {"query": {"match_phrase": {"body": "lazy dog"}}, "field_value_factor": {"field": "num"}}},
    "kw_term": {"term": {"kw": "fox dog"}},
    "tags": {"terms": {"tags": ["fox", "gold"]}},
    "range_only": {"range": {"num": {"gte": 500}}},
    "match_all": {"match_all": {}},
    "function_score": {"function_score": {"query": {"match": {"body": "fox"}}, "field_value_factor": {"field": "num"}}},
    "phrase_prefix": {"match_phrase_prefix": {"body": "quick br"}},
    "phrase_prefix_one": {"match_phrase_prefix": {"body": "qui"}},
    "phrase_prefix_slop": {"match_phrase_prefix": {"body": {"query": "brown qu", "slop": 2}}},
    "phrase_prefix_stop": {"match_phrase_prefix": {"body": "over the la"}},
    "phrase_prefix_fo": {"match_phrase_prefix": {"body": "quick fo"}},
    "phrase_prefix_none": {"match_phrase_prefix": {"body": "quick zzz"}},
    "phrase_prefix_and_term": {"bool": {"must": [{"match_phrase_prefix": {"body": "lazy d"}}], "should": [{"match": {"body": "gold"}}]}},
    "phrase_prefix_multi_match": {"multi_match": {"query": "lazy do", "fields": ["body", "title"], "type": "phrase_prefix"}},
    "title_en": {"match": {"title.en": "jumping foxes"}},
    "nested": {"nested": {"path": "comments", "query": {"match": {"comments.text": "fox dog"}}}},
    "nested_and_body": {"bool": {"must": [{"match": {"body": "lazy"}}], "should": [{"nested": {"path": "comments", "query": {"match_phrase": {"comments.text": "quick brown"}}}}]}},
    "script_filter": {"bool": {"must": [{"match": {"body": "fox"}}], "filter": [{"script": {"script": "doc['num'].value > 300"}}]}},
    "script_score_phrase": {"script_score": {"query": {"match_phrase": {"body": "lazy dog"}}, "script": {"source": "_score * 2"}}},
    "intervals": {"intervals": {"body": {"match": {"query": "quick fox", "max_gaps": 3}}}},
    "combined_fields": {"combined_fields": {"query": "fox dog", "fields": ["body", "title"]}},
    "match_bool_prefix": {"match_bool_prefix": {"body": "quick br"}},
    "geo_filter": {"bool": {"must": [{"match": {"body": "lazy"}}], "filter": [{"geo_distance": {"distance": "3000km", "loc": {"lat": 10, "lon": 20}}}]}},
    "more_like_this": {"more_like_this": {"fields": ["body"], "like": "quick brown fox jumps over the lazy dog", "min_term_freq": 1, "min_doc_freq": 1}},
    "exists_filter": {"bool": {"must": [{"match": {"body": "gold"}}], "filter": [{"exists": {"field": "title"}}]}},
    "ids_filter": {"bool": {"must": [{"match": {"body": "fox"}}], "filter": [{"ids": {"values": [str(i) for i in range(0, 3000, 3)]}}]}},
    "terms_set": {"terms_set": {"body": {"terms": ["fox", "dog", "gold"], "minimum_should_match_script": {"source": "1"}}}},
    "distance_feature": {"bool": {"must": [{"match": {"body": "fox"}}], "should": [{"distance_feature": {"field": "date", "pivot": "30d", "origin": "2026-06-01"}}]}},
    "boosting": {"boosting": {"positive": {"match": {"body": "fox"}}, "negative": {"match": {"body": "dog"}}, "negative_boost": 0.5}},
    "should_then_filter": {"bool": {"should": [{"match": {"body": "fox"}}], "filter": [{"intervals": {"body": {"match": {"query": "dog"}}}}]}},
    "boost_twice": {"bool": {"should": [{"term": {"body": {"value": "fox", "boost": 5}}}, {"term": {"body": {"value": "dog", "boost": 2}}}], "filter": [{"term": {"body": "fox"}}]}},
    "function_score_nested": {"function_score": {"query": {"nested": {"path": "comments", "query": {"match_phrase": {"comments.text": "quick brown"}}}}, "field_value_factor": {"field": "num"}}},
    "nested_function_score": {"nested": {"path": "comments", "query": {"function_score": {"query": {"match_phrase": {"comments.text": "quick brown"}}}}}},
    "query_string_mixed": {"query_string": {"query": "fox OR (lazy AND dog) OR gold", "default_field": "body"}},
    "query_string_proximity": {"query_string": {"query": "\"quick fox\"~3 OR lazy", "default_field": "body"}},
}

OPTIONS = {
    "default": {"fields": {"body": {}}},
    "frags1": {"fields": {"body": {"number_of_fragments": 1}}},
    "frags0": {"fields": {"body": {"number_of_fragments": 0}}},
    "size20": {"fields": {"body": {"fragment_size": 20, "number_of_fragments": 3}}},
    "size200": {"fields": {"body": {"fragment_size": 200}}},
    "size0": {"fields": {"body": {"fragment_size": 0}}},
    "no_match": {"fields": {"body": {"no_match_size": 40}, "title": {"no_match_size": 15}}},
    "word": {"fields": {"body": {"boundary_scanner": "word"}}},
    "sentence": {"fields": {"body": {"boundary_scanner": "sentence", "fragment_size": 60}}},
    "html": {"encoder": "html", "fields": {"body": {}}},
    "tags": {"pre_tags": ["<em class=\"x\">"], "post_tags": ["</em>"], "fields": {"body": {}}},
    "score": {"order": "score", "fields": {"body": {"number_of_fragments": 3, "fragment_size": 40}}},
    "no_require": {"require_field_match": False, "fields": {"body": {}, "title": {}, "multi": {}}},
    "star": {"fields": {"*": {}}},
    "multi": {"fields": {"multi": {"number_of_fragments": 2}}},
    "kw": {"fields": {"kw": {}, "tags": {}}},
    "hq": {"fields": {"body": {"highlight_query": {"match": {"body": "gold silver"}}}}},
    "max_offset": {"fields": {"body": {"max_analyzer_offset": 50}}},
    "unified_named": {"fields": {"body": {"type": "unified"}}},
    "title_en": {"fields": {"title.en": {}, "title": {}}},
    "nested_field": {"fields": {"comments.text": {}, "body": {}}},
}

# The plain highlighter: each option set above as `type: plain`, and its own fragmenters.
PLAIN_OPTIONS = {f"plain_{k}": {**v, "type": "plain"} for k, v in OPTIONS.items() if k != "unified_named"}
PLAIN_OPTIONS.update({
    "plain_simple": {"type": "plain", "fragmenter": "simple", "fields": {"body": {"fragment_size": 30, "number_of_fragments": 4}}},
    "plain_simple_score": {"type": "plain", "fragmenter": "simple", "order": "score", "fields": {"body": {"fragment_size": 25, "number_of_fragments": 3}}},
    "plain_span": {"type": "plain", "fragmenter": "span", "fields": {"body": {"fragment_size": 50, "number_of_fragments": 2}}},
    "plain_frags0_multi": {"type": "plain", "fields": {"multi": {"number_of_fragments": 0}}},
    "plain_simple_frags0": {"type": "plain", "fragmenter": "simple", "fields": {"body": {"number_of_fragments": 0, "no_match_size": 30}}},
    "plain_field_type": {"fields": {"body": {"type": "plain", "fragment_size": 60}, "title": {}}},
    "plain_html_score": {"type": "plain", "encoder": "html", "order": "score", "fields": {"body": {"number_of_fragments": 4, "fragment_size": 30}}},
})

# The fvh highlighter: each option set above as `type: fvh`, and its own options.
FVH_OPTIONS = {f"fvh_{k}": {**v, "type": "fvh"} for k, v in OPTIONS.items() if k != "unified_named"}
FVH_OPTIONS.update({
    "fvh_matched": {"type": "fvh", "fields": {"body": {"matched_fields": ["body", "body.en"]}}},
    "fvh_matched_score": {"type": "fvh", "order": "score", "fields": {"body": {"matched_fields": ["body.en", "body"], "number_of_fragments": 3, "fragment_size": 30}}},
    "fvh_offset": {"type": "fvh", "fields": {"body": {"fragment_offset": 5, "fragment_size": 40}}},
    "fvh_phrase_limit": {"type": "fvh", "fields": {"body": {"phrase_limit": 1}}},
    "fvh_chars": {"type": "fvh", "fields": {"body": {"boundary_chars": ".", "boundary_max_scan": 5, "fragment_size": 30}}},
    "fvh_force_source": {"type": "fvh", "force_source": True, "fields": {"body": {"number_of_fragments": 2, "fragment_size": 30}}},
    "fvh_frags0_multi": {"type": "fvh", "fields": {"multi": {"number_of_fragments": 0}}},
    "fvh_multi_score": {"type": "fvh", "order": "score", "fields": {"multi": {"number_of_fragments": 3, "fragment_size": 20}}},
    "fvh_no_match_multi": {"type": "fvh", "fields": {"multi": {"no_match_size": 20}}},
    "fvh_word_locale": {"type": "fvh", "boundary_scanner": "word", "boundary_scanner_locale": "en-US", "fields": {"body": {"fragment_size": 30}}},
    "fvh_sentence_frags": {"type": "fvh", "boundary_scanner": "sentence", "fields": {"body": {"fragment_size": 30, "number_of_fragments": 4}}},
    "fvh_highlight_filter": {"type": "fvh", "highlight_filter": True, "fields": {"body": {}}},
    "fvh_field_type": {"fields": {"body": {"type": "fvh", "fragment_size": 60}, "title": {}}},
})
# fvh needs term vectors: its sets run on these (hl_standard for the refusal).
FVH_RUN_ON = set(FVH_INDICES) | {"hl_tv", "hl_offsets_tv", "hl_standard"}


def option_sets(index):
    out = {}
    if "unified" in TYPES and index not in FVH_INDICES:
        out.update(OPTIONS)
    if "plain" in TYPES and index not in FVH_INDICES:
        out.update(PLAIN_OPTIONS)
    if "fvh" in TYPES and index in FVH_RUN_ON:
        out.update(FVH_OPTIONS)
    return out


# HL_QUERIES=a,b runs only those queries; HL_TRACE=1 prints each query's fallbacks.
ONLY = [q for q in os.environ.get("HL_QUERIES", "").split(",") if q]
TRACE = os.environ.get("HL_TRACE") == "1"


def fallbacks():
    stats = S.get(f"{PLUGIN}/_plugins/lucene_rust/stats").json()
    return {k: v for k, v in stats.get("fallbacks", {}).items() if "highlight" in k}


def search(base, index, query, hl, size):
    body = {"query": query, "highlight": hl, "size": size, "sort": ["_score", {"num": "asc"}, "_doc"]}
    r = S.post(f"{base}/{index}/_search", json=body)
    if not r.ok:
        return {"error": r.json().get("error", {}).get("root_cause", [{}])[0].get("type"), "status": r.status_code}
    hits = r.json()["hits"]["hits"]
    return [(h["_id"], h.get("highlight")) for h in hits]


def check(size):
    rows = bad = 0
    for index in INDICES:
        for qn, q in QUERIES.items():
            if ONLY and qn not in ONLY:
                continue
            before = fallbacks() if TRACE else None
            for on, o in option_sets(index).items():
                rows += 1
                a = search(PLUGIN, index, q, o, size)
                b = search(STOCK, index, q, o, size)
                if a != b:
                    bad += 1
                    if isinstance(a, dict) or isinstance(b, dict):
                        print(f"DIFF {index} {qn} {on}: plugin {str(a)[:200]} stock {str(b)[:200]}")
                        continue
                    for (ia, ha), (ib, hb) in zip(a, b):
                        if ia != ib or ha != hb:
                            print(f"DIFF {index} {qn} {on} id {ia}/{ib}:\n  plugin {json.dumps(ha, ensure_ascii=False)[:600]}\n  stock  {json.dumps(hb, ensure_ascii=False)[:600]}")
                            break
                    else:
                        print(f"DIFF {index} {qn} {on}: {len(a)} vs {len(b)} hits")
            if TRACE:
                after = fallbacks()
                delta = {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
                if delta:
                    print(f"FALLBACK {index} {qn} {delta}")
    stats = S.get(f"{PLUGIN}/_plugins/lucene_rust/stats").json()
    print(f"{rows - bad}/{rows} rows identical")
    print("native_highlights", stats.get("native_highlights"), "native_errors", stats.get("native_errors"))
    print("highlight fallbacks", {k: v for k, v in stats.get("fallbacks", {}).items() if "highlight" in k})
    return bad + (stats.get("native_errors") or 0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    docs = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    if mode in ("load", "all"):
        load(docs)
    if mode in ("check", "all"):
        sys.exit(1 if check(int(os.environ.get("SIZE", "50"))) else 0)
