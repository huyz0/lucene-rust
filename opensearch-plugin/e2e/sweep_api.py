#!/usr/bin/env python3
"""Index-management and request-API differential: filtered and routed aliases, clone / shrink / split,
close and open, search templates, msearch, mget, field_caps, rank_eval, URI search, by-query writes,
reindex with scripts, validate and explain. The same steps run on a plugin node and a stock node; each
step's response (minus volatile fields) is compared.

    python3 opensearch-plugin/e2e/sweep_api.py [--plugin URL] [--stock URL]
"""
import argparse
import json
import random

import fuzz_diff as fd

VOLATILE = fd.IGNORE | {"_scroll_id", "pit_id", "task", "took", "taskId", "id", "uuid", "creation_date", "index_uuid", "_shards", "total_time_in_millis",
                        "time_in_millis", "search_time", "indexing_time", "size_in_bytes", "store_size", "bytes", "memory_in_bytes", "version", "ms", "searchable_snapshot"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="http://localhost:9200")
    ap.add_argument("--stock", default="http://localhost:9201")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    P, S = fd.Node(a.plugin), fd.Node(a.stock)
    findings = []
    steps = [0]

    def both(method, path, body=None, raw=False, name=None, ignore=(), keys=None):
        steps[0] += 1
        p = P.req(method, path, body, raw)
        s = S.req(method, path, body, raw)
        if p[0] != s[0]:
            findings.append({"step": name or path, "diff": f"status plugin={p[0]} stock={s[0]}", "p": str(p[1])[:300], "s": str(s[1])[:300]})
            print(json.dumps(findings[-1]), flush=True)
            return p, s
        if p[0] >= 400:
            pt, st = fd._root_type(p[1].get("error", {})) if isinstance(p[1], dict) else None, fd._root_type(s[1].get("error", {})) if isinstance(s[1], dict) else None
            if pt != st:
                findings.append({"step": name or path, "diff": f"error type {pt} vs {st}", "p": str(p[1])[:300], "s": str(s[1])[:300]})
                print(json.dumps(findings[-1]), flush=True)
            return p, s
        pb, sb = p[1], s[1]
        if keys:
            pb = {k: pb.get(k) for k in keys}
            sb = {k: sb.get(k) for k in keys}
        d = fd.diff(pb, sb, "", 1e-4, VOLATILE | set(ignore))
        if d:
            findings.append({"step": name or path, "diff": d})
            print(json.dumps(findings[-1]), flush=True)
        return p, s

    r = random.Random(a.seed)
    fd.set_vocab(0)
    mapping = {"properties": {"title": {"type": "text"}, "body": {"type": "text", "analyzer": "english"}, "tag": {"type": "keyword"}, "n": {"type": "long"},
                              "price": {"type": "double"}, "ts": {"type": "date"}, "loc": {"type": "geo_point"}, "tags": {"type": "keyword"}}}
    for n in (P, S):
        for i in ("api_a", "api_b", "api_c", "api_clone", "api_shrink", "api_split", "api_re", "api_re2", "api_closed"):
            n.req("DELETE", f"/{i}")
        n.req("DELETE", "/_template/api_t")
    both("PUT", "/api_a", {"settings": {"number_of_shards": 4, "number_of_replicas": 0, "index.routing_partition_size": 2}, "mappings": {**mapping, "_routing": {"required": True}}}, name="create routed index")
    both("PUT", "/api_b", {"settings": {"number_of_shards": 2, "number_of_replicas": 0}, "mappings": mapping}, name="create b")
    for idx in ("api_a", "api_b"):
        lines = []
        for i in range(400):
            doc = {"title": fd.sentence(r, 1, 6), "body": fd.sentence(r, 3, 20), "tag": r.choice(["red", "green", "blue", "x"]), "n": r.randint(-50, 50),
                   "price": round(r.uniform(0, 100), 2), "ts": f"20{r.randint(10, 25)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}",
                   "loc": {"lat": r.uniform(-80, 80), "lon": r.uniform(-170, 170)}, "tags": [r.choice(["p", "q", "r"]) for _ in range(r.randint(0, 3))]}
            meta = {"index": {"_id": f"{idx[-1]}{i}"}}
            if idx == "api_a":
                meta["index"]["routing"] = f"r{i % 5}"
            lines += [json.dumps(meta), json.dumps(doc)]
        both("POST", f"/{idx}/_bulk?refresh=true", ("\n".join(lines) + "\n").encode(), raw=True, name=f"bulk {idx}", keys=["errors"])
    # aliases
    both("POST", "/_aliases", {"actions": [
        {"add": {"index": "api_a", "alias": "al_red", "filter": {"term": {"tag": "red"}}, "routing": "r1"}},
        {"add": {"index": "api_a", "alias": "al_all"}}, {"add": {"index": "api_b", "alias": "al_all", "filter": {"range": {"n": {"gte": 0}}}}},
        {"add": {"index": "api_a", "alias": "al_search_route", "search_routing": "r1,r2", "index_routing": "r3"}},
        {"add": {"index": "api_b", "alias": "al_b_filtered", "filter": {"bool": {"must": [{"match": {"body": "alpha"}}], "filter": {"range": {"price": {"lt": 50}}}}}}}]}, name="aliases")
    q = lambda body, name, params="", path="al_all": both("POST", f"/{path}/_search{params}", body, name=name, ignore=("_index",))
    for al in ("al_red", "al_all", "al_search_route", "al_b_filtered"):
        q({"size": 500, "query": {"match_all": {}}, "sort": [{"_seq_no": "asc"}, {"_index": "asc"}] if False else [{"n": "asc"}, {"_id": "asc"}]}, f"alias {al} match_all", path=al)
        q({"size": 500, "query": {"match": {"body": "alpha beta"}}, "aggs": {"t": {"terms": {"field": "tag"}}, "s": {"stats": {"field": "price"}}}}, f"alias {al} scored+aggs", path=al)
        q({"size": 0, "aggs": {"h": {"histogram": {"field": "n", "interval": 10}}, "c": {"cardinality": {"field": "tags"}}}}, f"alias {al} aggs only", path=al)
        both("GET", f"/{al}/_count", name=f"count {al}")
        both("POST", f"/{al}/_count", {"query": {"term": {"tag": "green"}}}, name=f"count q {al}")
    both("POST", "/al_all/_search?routing=r1", {"size": 500, "query": {"match_all": {}}, "sort": [{"n": "asc"}, {"_id": "asc"}]}, name="alias+routing param", ignore=("_index",))
    both("POST", "/api_a,api_b/_search?preference=_shards:1", {"size": 500, "query": {"match_all": {}}, "sort": [{"n": "asc"}, {"_id": "asc"}]}, name="preference shards", ignore=("_index",))
    both("POST", "/al_*/_search", {"size": 0, "aggs": {"i": {"terms": {"field": "_index"}}}}, name="terms _index over wildcard alias")
    both("POST", "/api_*/_search", {"size": 5, "query": {"term": {"_index": "api_b"}}, "sort": [{"_id": "asc"}]}, name="_index term")
    both("POST", "/api_*/_search", {"size": 5, "query": {"terms": {"_index": ["api_a", "nope*"]}}, "sort": [{"_id": "asc"}]}, name="_index terms")
    both("POST", "/api_*/_search", {"size": 5, "query": {"prefix": {"_index": "api_"}}, "sort": [{"_id": "asc"}]}, name="_index prefix")
    both("GET", "/_alias/al_*", name="get aliases")
    both("POST", "/_aliases", {"actions": [{"remove": {"index": "api_b", "alias": "al_all"}}, {"add": {"index": "api_b", "alias": "al_all", "is_write_index": True}}]}, name="write index alias")
    # field caps / mapping / settings
    both("GET", "/api_a,api_b/_field_caps?fields=*", name="field_caps", keys=["fields"])
    both("GET", "/api_a/_mapping", name="mapping")
    # msearch / mget / templates
    ms = []
    for k in range(6):
        ms += [json.dumps({"index": r.choice(["api_a", "api_b", "al_all", "nope"]), "ignore_unavailable": True}), json.dumps({"size": 3, "query": {"match": {"body": fd.WORDS[k]}}, "sort": [{"n": "asc"}, {"_id": "asc"}]})]
    both("POST", "/_msearch", ("\n".join(ms) + "\n").encode(), raw=True, name="msearch", ignore=("_index",))
    both("POST", "/api_b/_mget", {"docs": [{"_id": "b1"}, {"_id": "b2", "_source": ["title"]}, {"_id": "nope"}, {"_id": "b3", "stored_fields": ["title"]}]}, name="mget")
    both("POST", "/api_b/_mget?_source_includes=tag,n", {"ids": ["b4", "b5", "b6"]}, name="mget source filter")
    both("POST", "/_scripts/api_tpl", {"script": {"lang": "mustache", "source": '{"size": {{size}}, "query": {"match": {"body": "{{q}}"}}, "sort": [{"n": "asc"}, {"_id": "asc"}]}'}}, name="put template")
    both("POST", "/api_b/_search/template", {"id": "api_tpl", "params": {"size": 4, "q": "alpha"}}, name="search template")
    both("POST", "/api_b/_search/template", {"source": '{"query": {"terms": {"tag": {{#toJson}}tags{{/toJson}}}}, "size": 3, "sort": [{"_id": "asc"}]}', "params": {"tags": ["red", "blue"]}}, name="inline template toJson")
    both("POST", "/api_b/_render/template", {"id": "api_tpl", "params": {"size": 2, "q": "beta"}}, name="render template")
    both("POST", "/api_b/_rank_eval", {"requests": [{"id": "q1", "request": {"query": {"match": {"body": "alpha"}}}, "ratings": [{"_index": "api_b", "_id": "b1", "rating": 3}, {"_index": "api_b", "_id": "b2", "rating": 1}]}],
                                       "metric": {"dcg": {"k": 10, "normalize": True}}}, name="rank_eval dcg", keys=["metric_score"])
    both("POST", "/api_b/_rank_eval", {"requests": [{"id": "q1", "request": {"query": {"match": {"body": "alpha"}}}, "ratings": [{"_index": "api_b", "_id": "b1", "rating": 1}]}], "metric": {"precision": {"k": 20, "relevant_rating_threshold": 1}}}, name="rank_eval precision", keys=["metric_score"])
    both("GET", "/api_b/_search?q=body:alpha+AND+tag:red&sort=n:asc,_id:asc&size=7&_source=title&default_operator=OR", name="URI search")
    both("GET", "/api_b/_search?q=alpha&df=body&analyze_wildcard=true&lenient=true&size=3&sort=_id", name="URI df")
    both("POST", "/api_b/_validate/query?explain=true&rewrite=true", {"query": {"bool": {"must": {"match": {"body": "alpha beta"}}, "filter": {"range": {"n": {"gte": 3}}}}}}, name="validate rewrite", keys=["valid", "explanations"])
    both("POST", "/api_b/_validate/query?explain=true", {"query": {"match": {"n": "notanumber"}}}, name="validate invalid", keys=["valid"])
    both("POST", "/api_b/_explain/b1", {"query": {"match": {"body": "alpha"}}}, name="explain doc")
    both("POST", "/api_b/_analyze", {"field": "body", "text": "The Quick brown FOXES jumped"}, name="analyze field")
    # clone / shrink / split / close
    both("PUT", "/api_b/_settings", {"index.blocks.write": True}, name="block writes")
    both("POST", "/api_b/_clone/api_clone", {"settings": {"index.number_of_replicas": 0}}, name="clone")
    both("POST", "/api_b/_shrink/api_shrink", {"settings": {"index.number_of_shards": 1, "index.number_of_replicas": 0, "index.blocks.write": None}}, name="shrink")
    both("POST", "/api_b/_split/api_split", {"settings": {"index.number_of_shards": 6, "index.number_of_replicas": 0, "index.blocks.write": None}}, name="split")
    for n in (P, S):
        n.req("GET", "/_cluster/health?wait_for_status=green&timeout=30s")
        for i in ("api_clone", "api_shrink", "api_split"):
            n.req("POST", f"/{i}/_refresh")
    for idx in ("api_clone", "api_shrink", "api_split"):
        both("POST", f"/{idx}/_search", {"size": 500, "query": {"match": {"body": "alpha beta"}}, "aggs": {"t": {"terms": {"field": "tag"}}, "p": {"percentiles": {"field": "price"}}}, "sort": [{"n": "asc"}, {"_id": "asc"}]}, name=f"search {idx}", ignore=("_index",))
        both("GET", f"/{idx}/_count", name=f"count {idx}")
        both("POST", f"/{idx}/_forcemerge?max_num_segments=1", name=f"forcemerge {idx}", keys=["_shards"])
        both("POST", f"/{idx}/_search", {"size": 500, "query": {"range": {"n": {"gte": 0}}}, "sort": [{"price": "desc"}, {"_id": "asc"}]}, name=f"search merged {idx}", ignore=("_index",))
    both("PUT", "/api_b/_settings", {"index.blocks.write": None}, name="unblock")
    both("POST", "/api_b/_close", name="close", keys=["acknowledged"])
    both("POST", "/api_b/_search", {"query": {"match_all": {}}}, name="search closed")
    both("POST", "/api_b/_open", name="open", keys=["acknowledged"])
    for n in (P, S):
        n.req("GET", "/_cluster/health?wait_for_status=green&timeout=30s")
    both("POST", "/api_b/_search", {"size": 300, "query": {"match": {"body": "alpha"}}, "sort": [{"n": "asc"}, {"_id": "asc"}]}, name="search after reopen", ignore=("_index",))
    # by-query writes and reindex
    both("POST", "/api_b/_update_by_query?refresh=true&conflicts=proceed", {"script": {"source": "ctx._source.n = ctx._source.n * 2; ctx._source.tags.add('z')"}, "query": {"term": {"tag": "red"}}}, name="update_by_query", keys=["updated", "total", "failures", "version_conflicts"])
    both("POST", "/api_b/_delete_by_query?refresh=true", {"query": {"range": {"price": {"lt": 10}}}}, name="delete_by_query", keys=["deleted", "total", "failures"])
    both("POST", "/api_b/_search", {"size": 500, "query": {"match_all": {}}, "sort": [{"n": "asc"}, {"_id": "asc"}], "_source": ["n", "tags"]}, name="after by-query", ignore=("_index",))
    both("POST", "/_reindex?refresh=true", {"source": {"index": "api_b", "query": {"term": {"tag": "green"}}, "size": 50}, "dest": {"index": "api_re"}, "script": {"source": "ctx._source.copied = true; ctx._id = 'c' + ctx._id"}}, name="reindex script", keys=["created", "total", "failures"])
    both("POST", "/_reindex?refresh=true&slices=3", {"source": {"index": "api_a"}, "dest": {"index": "api_re2"}}, name="reindex sliced from routed", keys=["created", "total", "failures"])
    for idx in ("api_re", "api_re2"):
        both("POST", f"/{idx}/_search", {"size": 500, "query": {"match_all": {}}, "sort": [{"n": "asc"}, {"_id": "asc"}]}, name=f"search {idx}", ignore=("_index",))
    both("POST", "/api_a/_search", {"size": 500, "query": {"match_all": {}}, "sort": [{"n": "asc"}, {"_id": "asc"}], "_source": False}, name="routed index no routing search", ignore=("_index",))
    both("GET", "/api_a/_doc/a1", name="routed get without routing")
    both("GET", "/api_a/_doc/a1?routing=r1", name="routed get with routing")
    # stats-like APIs that must stay consistent
    for n in (P, S):
        n.req("POST", "/api_b/_refresh")
    both("GET", "/api_b/_stats/docs", name="stats docs", keys=["_all"], ignore=("size_in_bytes", "total_size_in_bytes", "deleted"))
    both("GET", "/api_b/_search?size=0&track_total_hits=true&terminate_after=20", name="terminate_after", keys=["terminated_early"])
    print(json.dumps({"steps": steps[0], "findings": len(findings)}))


if __name__ == "__main__":
    main()
