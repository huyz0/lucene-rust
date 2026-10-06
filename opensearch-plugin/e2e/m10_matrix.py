"""M10 T10.7's rows of the native-vs-Lucene REST matrix (verify_opensearch.py).

`rows()` runs on the main indices (`single`, `multi`: body/title text, tag
keyword, n long, price double, qty integer, ratio float, ts date, m multi-valued
long); `nested_rows()` on the nested indices (`nest`, `nestm`: root body/tag/n,
nested `props` with v long, w double, i integer, f float, k keyword);
`join_rows()` on `pj`, a parent-join index (`create_join`, `load_join`).

Each row is (name, body, expected): "native" must run native on every shard;
a reason must fall back for it on every shard ("prefix*" matches any reason
with the prefix); "fallback" any reason; "either" only requires the two
engines' responses to agree (a shape whose rewritten form this file does not
pin). Responses must agree in every case.
"""

import json
import random


def rows():
    q = []
    add = lambda name, body, expect: q.append((name, body, expect))

    def span(t, field="body"):
        return {"span_term": {field: t}}

    # Spans (`lucene-queries`' spans, node 21).
    add("span_term", {"query": span("alpha")}, "native")
    add("span_near ordered", {"query": {"span_near": {"clauses": [span("alpha"), span("beta")], "slop": 3, "in_order": True}}}, "native")
    add("span_near unordered", {"query": {"span_near": {"clauses": [span("gamma"), span("delta"), span("alpha")], "slop": 8, "in_order": False}}}, "native")
    add("span_or", {"query": {"span_or": {"clauses": [span("epsilon"), span("zeta")]}}}, "native")
    add("span_not", {"query": {"span_not": {"include": {"span_near": {"clauses": [span("alpha"), span("beta")], "slop": 4, "in_order": False}}, "exclude": span("gamma"), "pre": 1, "post": 1}}}, "native")
    add("span_first", {"query": {"span_first": {"match": span("alpha"), "end": 3}}}, "native")
    add("span_containing", {"query": {"span_containing": {"big": {"span_near": {"clauses": [span("alpha"), span("beta")], "slop": 6, "in_order": True}}, "little": span("gamma")}}}, "native")
    add("span_within", {"query": {"span_within": {"big": {"span_near": {"clauses": [span("alpha"), span("delta")], "slop": 6, "in_order": False}}, "little": span("beta")}}}, "native")
    add("field_masking_span", {"query": {"span_near": {"clauses": [span("alpha"), {"field_masking_span": {"query": span("beta", "title"), "field": "body"}}], "slop": 10, "in_order": False}}}, "native")
    add("span_multi prefix", {"query": {"span_multi": {"match": {"prefix": {"body": {"value": "al"}}}}}}, "native")
    add("span_near boosted in bool", {"query": {"bool": {"must": [{"match": {"title": "alpha"}}], "should": [{"span_near": {"clauses": [span("beta"), span("gamma")], "slop": 2, "in_order": True, "boost": 2}}]}}}, "native")
    add("span_near with span_gap", {"query": {"span_near": {"clauses": [span("alpha"), {"span_gap": {"body": 1}}, span("beta")], "slop": 0, "in_order": True}}}, "span_gap")

    # Intervals (`IntervalQuery`, node 22).
    add("intervals match ordered", {"query": {"intervals": {"body": {"match": {"query": "alpha beta", "max_gaps": 3, "ordered": True}}}}}, "native")
    add("intervals match unordered", {"query": {"intervals": {"body": {"match": {"query": "gamma delta epsilon", "max_gaps": 10}}}}}, "native")
    add("intervals any_of", {"query": {"intervals": {"body": {"any_of": {"intervals": [{"match": {"query": "alpha"}}, {"match": {"query": "zeta eta", "ordered": True}}]}}}}}, "native")
    add("intervals all_of filter", {"query": {"intervals": {"body": {"all_of": {"ordered": False, "max_gaps": 8, "intervals": [{"match": {"query": "alpha"}}, {"match": {"query": "beta"}}],
                                                               "filter": {"not_containing": {"match": {"query": "gamma"}}}}}}}}, "native")
    add("intervals contained_by", {"query": {"intervals": {"body": {"match": {"query": "beta", "filter": {"contained_by": {"match": {"query": "alpha delta", "max_gaps": 5}}}}}}}}, "native")
    add("intervals before", {"query": {"intervals": {"body": {"match": {"query": "alpha", "filter": {"before": {"match": {"query": "omega"}}}}}}}}, "native")
    add("intervals prefix", {"query": {"intervals": {"body": {"prefix": {"prefix": "ga"}}}}}, "native")
    add("intervals wildcard", {"query": {"intervals": {"body": {"wildcard": {"pattern": "*ta"}}}}}, "native")
    add("intervals fuzzy", {"query": {"intervals": {"body": {"fuzzy": {"term": "alpah"}}}}}, "native")
    add("intervals script filter", {"query": {"intervals": {"body": {"match": {"query": "alpha beta", "filter": {"script": {"source": "interval.start > 1"}}}}}}}, "interval_*")

    # combined_fields (`CombinedFieldQuery`, node 23).
    add("combined_fields", {"query": {"combined_fields": {"query": "alpha beta", "fields": ["body", "title^2"]}}}, "native")
    add("combined_fields and", {"query": {"combined_fields": {"query": "gamma delta", "fields": ["body", "title"], "operator": "and"}}}, "native")

    # OpenSearch's function_score (node 24).
    def fs(functions, **extra):
        return {"query": {"function_score": {"query": {"match": {"body": "alpha beta"}}, "functions": functions, **extra}}}

    add("function_score weight filter", fs([{"filter": {"term": {"tag": "alpha"}}, "weight": 3}, {"filter": {"term": {"tag": "beta"}}, "weight": 2}]), "native")
    add("function_score field_value_factor", fs([{"field_value_factor": {"field": "qty", "factor": 1.5, "modifier": "log1p", "missing": 1}}]), "native")
    add("function_score fvf float square", fs([{"field_value_factor": {"field": "ratio", "modifier": "square", "missing": 0.5}}], boost_mode="sum"), "native")
    add("function_score fvf long ln2p", fs([{"field_value_factor": {"field": "n", "modifier": "ln2p"}}], boost_mode="replace"), "native")
    add("function_score random seeded", fs([{"random_score": {"seed": 42, "field": "_seq_no"}}]), "native")
    add("function_score random keyword", fs([{"random_score": {"seed": 7, "field": "tag"}}], boost_mode="avg"), "native")
    add("function_score gauss long", fs([{"gauss": {"n": {"origin": 5000, "scale": 1000, "decay": 0.3}}}], score_mode="max"), "native")
    add("function_score exp double", fs([{"exp": {"price": {"origin": 100, "scale": 50, "offset": 5}}}]), "native")
    add("function_score linear date", fs([{"linear": {"ts": {"origin": "2023-11-18", "scale": "3d"}}}], boost_mode="max"), "native")
    add("function_score decay multi avg", fs([{"gauss": {"m": {"origin": 500, "scale": 200}, "multi_value_mode": "avg"}}], boost_mode="min"), "native")
    add("function_score many", fs([
        {"filter": {"term": {"tag": "gamma"}}, "weight": 2},
        {"field_value_factor": {"field": "qty", "modifier": "sqrt", "missing": 0}, "weight": 0.5},
        {"gauss": {"n": {"origin": 100, "scale": 5000}}},
    ], score_mode="sum", max_boost=3), "native")
    add("function_score avg", fs([{"weight": 2}, {"filter": {"term": {"tag": "delta"}}, "weight": 4}], score_mode="avg"), "native")
    add("function_score boosted", {"query": {"function_score": {"query": {"match": {"body": "gamma"}}, "weight": 2, "boost": 3}}}, "native")
    add("function_score under a boosted bool", {"query": {"bool": {"boost": 2, "must": [{"function_score": {"query": {"match": {"body": "alpha"}}, "weight": 3, "boost_mode": "sum"}}],
                                                                 "should": [{"match": {"title": "beta"}}]}}}, "native")
    add("function_score replace under a boosted bool", {"query": {"bool": {"boost": 2, "must": [{"function_score": {"query": {"match": {"body": "alpha"}}, "weight": 3, "boost_mode": "replace"}}],
                                                                         "should": [{"match": {"title": "beta"}}]}}}, "native")
    add("function_score in bool", {"query": {"bool": {"must": [{"match": {"title": "alpha"}}],
                                                      "should": [{"function_score": {"query": {"match": {"body": "beta"}}, "functions": [{"field_value_factor": {"field": "qty", "missing": 2}}]}}]}}}, "native")
    add("function_score min_score", fs([{"weight": 2}], min_score=1), "function_score_min_score")
    add("function_score script", fs([{"script_score": {"script": {"source": "doc['qty'].value + 1"}}}]), "function_score_script")
    add("script_score query", {"query": {"script_score": {"query": {"match": {"body": "alpha"}}, "script": {"source": "_score * 2"}}}}, "query_*")

    # Rewritten before the query phase sees them: more_like_this to a boolean of
    # term queries, common to blended term queries.
    add("more_like_this text", {"query": {"more_like_this": {"fields": ["body"], "like": "alpha beta gamma delta epsilon", "min_term_freq": 1, "min_doc_freq": 1}}}, "native")
    add("more_like_this doc", {"query": {"more_like_this": {"fields": ["body", "title"], "like": [{"_id": "1"}], "min_term_freq": 1, "min_doc_freq": 1}}}, "native")
    add("common", {"query": {"common": {"body": {"query": "alpha beta omega", "cutoff_frequency": 0.01}}}}, "native")
    return q


def nested_rows():
    q = []
    add = lambda name, body, expect: q.append((name, body, expect))

    def nested(query, mode="avg", **extra):
        return {"query": {"nested": {"path": "props", "query": query, "score_mode": mode, **extra}}}

    add("nested term avg", nested({"term": {"props.k": "alpha"}}), "native")
    for mode in ("max", "min", "sum", "none"):
        add(f"nested range {mode}", nested({"range": {"props.v": {"gte": -10, "lte": 30}}}, mode), "native")
    add("nested bool", nested({"bool": {"must": [{"term": {"props.k": "beta"}}], "filter": [{"range": {"props.w": {"gte": 0}}}]}}, "max"), "native")
    add("nested match_all", nested({"match_all": {}}, "sum"), "native")
    add("nested in bool", {"query": {"bool": {"must": [{"match": {"body": "alpha"}}],
                                               "should": [{"nested": {"path": "props", "query": {"term": {"props.k": "gamma"}}, "score_mode": "max"}}]}}}, "native")
    add("nested inner_hits", nested({"term": {"props.k": "delta"}}, "avg", inner_hits={"size": 2}), "native")
    add("nested boosted", {"query": {"nested": {"path": "props", "query": {"range": {"props.v": {"gte": 0}}}, "score_mode": "avg", "boost": 2}}}, "native")
    # A range on a 4-byte field (`integer`) is a points range the native side does not take, in a
    # nested query as anywhere.
    add("nested integer range", {"query": {"nested": {"path": "props", "query": {"range": {"props.i": {"gte": 0}}}, "score_mode": "avg"}}}, "points_width")
    add("nested function_score", nested({"function_score": {"query": {"term": {"props.k": "alpha"}}, "functions": [{"field_value_factor": {"field": "props.i", "missing": 1, "modifier": "square"}}]}}, "sum"), "native")
    return q


def create_join(req, index):
    req("PUT", f"/{index}", {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1},
        "mappings": {"properties": {
            "body": {"type": "text"},
            "rel": {"type": "join", "relations": {"question": "answer"}},
        }},
    })


def load_join(req, check, index, questions, seed):
    """Questions with zero to three answers each, routed to their question."""
    r = random.Random(seed)
    words = "alpha beta gamma delta epsilon zeta eta theta".split()
    lines = []
    aid = 0
    for qn in range(questions):
        lines.append(json.dumps({"index": {"_index": index, "_id": f"q{qn}"}}))
        lines.append(json.dumps({"body": " ".join(r.choice(words) for _ in range(r.randint(1, 8))), "rel": "question"}))
        for _ in range(r.randint(0, 3)):
            lines.append(json.dumps({"index": {"_index": index, "_id": f"a{aid}", "routing": f"q{qn}"}}))
            lines.append(json.dumps({"body": " ".join(r.choice(words) for _ in range(r.randint(1, 8))), "rel": {"name": "answer", "parent": f"q{qn}"}}))
            aid += 1
    res = req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
    check(not res["errors"], f"{index}: parent-join bulk has no errors")
    req("POST", f"/{index}/_refresh")


def join_rows():
    """parent-join: has_child/has_parent rewrite to JoinUtil's global-ordinal
    joins in Java (the from side searched during the rewrite, over OpenSearch's
    own global ordinals), which the native side does not take over; parent_id
    is a boolean of term queries."""
    q = []
    add = lambda name, body, expect: q.append((name, body, expect))
    add("has_child", {"query": {"has_child": {"type": "answer", "query": {"match": {"body": "alpha"}}, "score_mode": "max"}}}, "query_GlobalOrdinalsWithScoreQuery")
    add("has_parent", {"query": {"has_parent": {"parent_type": "question", "query": {"match": {"body": "beta"}}, "score": True}}}, "query_GlobalOrdinalsWithScoreQuery")
    add("parent_id", {"query": {"parent_id": {"type": "answer", "id": "q3"}}}, "native")
    return q
