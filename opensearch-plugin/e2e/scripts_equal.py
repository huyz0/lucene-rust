"""Scripts and function_score shapes, answered by a plugin node and by a stock node on the same corpus.

  scripts_equal.py load BASE            creates index `sc` and the stored scripts
  scripts_equal.py dump BASE OUT.json   writes {shape: {hits: [[id, score]], total, error}}
  scripts_equal.py diff A.json B.json   exits 1 on any difference
  scripts_equal.py stats BASE           prints the plugin's fallback counters

Every shape here must run natively: after a dump on the plugin node, `stats` shows what fell back."""
import base64
import json
import random
import sys

import benchhttp as bh

N = 2400
MAPPING = {
    "properties": {
        "tag": {"type": "keyword"},
        "body": {"type": "text"},
        "l": {"type": "long"},
        "lm": {"type": "long"},
        "d": {"type": "double"},
        "f": {"type": "float"},
        "h": {"type": "half_float"},
        "sf": {"type": "scaled_float", "scaling_factor": 100},
        "ul": {"type": "unsigned_long"},
        "dt": {"type": "date"},
        "dn": {"type": "date_nanos"},
        "gp": {"type": "geo_point"},
        "gpm": {"type": "geo_point"},
        "bin": {"type": "binary", "doc_values": True},
        "hl": {"type": "long"},
        "b": {"type": "boolean"},
    }
}
WORDS = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"]


def docs():
    r = random.Random(42)
    for i in range(N):
        d = {"tag": "t%d" % (i % 13), "body": " ".join(r.choice(WORDS) for _ in range(6))}
        if i % 11:
            d["l"] = r.randint(1, 5000)
        if i % 5:
            d["lm"] = [r.randint(1, 900) for _ in range(r.randint(1, 4))]
        if i % 7:
            d["d"] = round(r.uniform(0.5, 800), 3)
        if i % 9:
            d["f"] = round(r.uniform(0.5, 100), 2)
        if i % 8:
            d["h"] = round(r.uniform(0.5, 60), 1)
        if i % 6:
            d["sf"] = round(r.uniform(0.01, 90), 2)
        if i % 10:
            d["ul"] = r.choice([r.randint(0, 10**6), r.randint(2**63, 2**64 - 1), r.randint(0, 2**63 - 1)])
        if i % 12:
            d["dt"] = 1_700_000_000_000 + r.randint(0, 10**9)
            d["dn"] = "2024-%02d-%02dT%02d:%02d:%02d.%06d123Z" % (
                r.randint(1, 12), r.randint(1, 28), r.randint(0, 23), r.randint(0, 59), r.randint(0, 59), r.randint(0, 999999))
        if i % 4:
            d["gp"] = {"lat": round(r.uniform(-60, 60), 4), "lon": round(r.uniform(-170, 170), 4)}
        if i % 6 < 3:
            d["gpm"] = [{"lat": round(r.uniform(40, 41), 4), "lon": round(r.uniform(-74, -73), 4)} for _ in range(r.randint(1, 3))]
        if i % 3:
            d["bin"] = base64.b64encode(bytes(r.randint(0, 255) for _ in range(r.choice([4, 4, 4, 3, 6])))).decode()
        if i % 13:
            d["hl"] = r.getrandbits(40)
        if i % 2:
            d["b"] = bool(r.getrandbits(1))
        yield i, d


def load(base):
    try:
        bh.req(base, "DELETE", "/sc")
    except Exception:
        pass
    bh.req(base, "PUT", "/sc", {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1}, "mappings": MAPPING})
    lines = []
    for i, d in docs():
        lines.append(json.dumps({"index": {"_id": str(i)}}))
        lines.append(json.dumps(d))
        if (i + 1) % 800 == 0:
            bh.req(base, "POST", "/sc/_bulk?refresh=true", body="\n".join(lines) + "\n", ndjson=True)
            lines = []
    scripts = {
        "s_score": {"lang": "painless", "source": "doc['l'].size() == 0 ? 0.5 : doc['l'].value * params.k"},
        "s_filter": {"lang": "painless", "source": "doc['l'].size() > 0 && doc['l'].value % params.m == 0"},
        "s_sort": {"lang": "painless", "source": "doc['d'].size() == 0 ? 0 : doc['d'].value * params.k"},
        "s_agg": {"lang": "painless", "source": "doc['l'].size() == 0 ? 0 : doc['l'].value * params.k"},
        "s_min": {"lang": "painless", "source": "params.n"},
        "s_loop": {"lang": "painless", "source": "int s = 0; for (int i = 0; i < 3; i++) { s += i; } return s + doc['f'].size()"},
    }
    for k, v in scripts.items():
        bh.req(base, "PUT", "/_scripts/" + k, {"script": v})
    bh.req(base, "POST", "/sc/_forcemerge?max_num_segments=1") if False else None


def knn(field, q, space="hammingbit"):
    return {"source": "knn_score", "lang": "knn", "params": {"field": field, "query_value": q, "space_type": space}}


def shapes():
    s = {}
    ma = {"match_all": {}}
    body = {"match": {"body": "alpha beta"}}
    # min_score
    s["fs min_score weight"] = {"query": {"function_score": {"query": ma, "functions": [{"weight": 2}], "min_score": 1.5}}}
    s["fs min_score fvf"] = {"query": {"function_score": {"query": body, "field_value_factor": {"field": "l", "missing": 1, "modifier": "log1p"}, "boost_mode": "multiply", "min_score": 6.0}}}
    s["fs min_score replace"] = {"query": {"function_score": {"query": body, "functions": [{"filter": {"term": {"tag": "t3"}}, "weight": 4}], "boost_mode": "replace", "min_score": 1}}}
    s["fs min_score count"] = {"size": 0, "track_total_hits": True, "query": {"function_score": {"query": body, "functions": [{"weight": 3}], "min_score": 2.0}}}
    s["fs min_score sorted"] = {"size": 30, "sort": [{"l": "asc"}, {"_doc": "asc"}], "query": {"function_score": {"query": body, "functions": [{"weight": 3}], "min_score": 2.0}}}
    s["fs min_score aggs"] = {"size": 0, "aggs": {"t": {"terms": {"field": "tag"}}, "s": {"sum": {"field": "l"}}}, "query": {"function_score": {"query": body, "functions": [{"weight": 3}], "min_score": 2.0}}}
    s["fs min_score no functions"] = {"query": {"function_score": {"query": body, "min_score": 1.0}}}
    s["fs min_score in bool"] = {"query": {"bool": {"must": [{"function_score": {"query": body, "functions": [{"weight": 2}], "min_score": 2.0}}], "should": [{"term": {"tag": "t1"}}]}}}
    s["script_score min_score"] = {"query": {"script_score": {"query": body, "script": {"source": "doc['l'].size() == 0 ? 0 : doc['l'].value / 100.0"}, "min_score": 20}}}
    s["script_score min_score count"] = {"size": 0, "track_total_hits": True, "query": {"script_score": {"query": ma, "script": {"source": "doc['l'].size() == 0 ? 0 : doc['l'].value / 100.0"}, "min_score": 20}}}
    s["script_score min_score boost"] = {"query": {"script_score": {"query": body, "script": {"source": "_score + 1"}, "min_score": 2.5, "boost": 2}}}
    # stored scripts
    s["stored script_score"] = {"query": {"script_score": {"query": body, "script": {"id": "s_score", "params": {"k": 0.01}}}}}
    s["stored fs script"] = {"query": {"function_score": {"query": body, "functions": [{"script_score": {"script": {"id": "s_score", "params": {"k": 0.02}}}}], "boost_mode": "sum"}}}
    s["stored script query"] = {"size": 30, "query": {"bool": {"filter": [{"script": {"script": {"id": "s_filter", "params": {"m": 7}}}}]}}, "sort": ["_doc"]}
    s["stored sort"] = {"size": 30, "query": ma, "sort": [{"_script": {"type": "number", "script": {"id": "s_sort", "params": {"k": 2}}, "order": "desc"}}, "_doc"]}
    s["stored agg"] = {"size": 0, "aggs": {"a": {"sum": {"script": {"id": "s_agg", "params": {"k": 3}}}}, "m": {"max": {"script": {"id": "s_agg", "params": {"k": 3}}}}}}
    s["stored terms agg"] = {"size": 0, "aggs": {"a": {"terms": {"script": {"id": "s_agg", "params": {"k": 0.001}}, "size": 5}}}}
    s["stored min_score"] = {"query": {"script_score": {"query": body, "script": {"id": "s_min", "params": {"n": 3}}, "min_score": 2}}}
    s["stored terms_set"] = {"query": {"terms_set": {"tag": {"terms": ["t1", "t2", "t3"], "minimum_should_match_script": {"id": "s_min", "params": {"n": 1}}}}}}
    # field_value_factor over every numeric type
    for f in ["l", "d", "f", "h", "sf", "ul", "dt", "dn"]:
        s["fvf " + f] = {"size": 30, "query": {"function_score": {"query": body, "field_value_factor": {"field": f, "missing": 2, "modifier": "log2p", "factor": 1.5}, "boost_mode": "sum"}}}
        s["random " + f] = {"size": 30, "query": {"function_score": {"query": ma, "random_score": {"seed": 11, "field": f}, "boost_mode": "replace"}}}
    s["fvf lm"] = {"size": 30, "query": {"function_score": {"query": body, "field_value_factor": {"field": "lm", "missing": 0, "modifier": "sqrt"}}}}
    s["fvf missing error"] = {"size": 30, "query": {"function_score": {"query": ma, "field_value_factor": {"field": "ul"}}}}
    # decay
    for kind in ["gauss", "exp", "linear"]:
        dec = {"origin": 2000, "scale": 800, "offset": 100, "decay": 0.4}
        for f in ["l", "d", "f", "h", "sf", "ul"]:
            o = {"l": 2000, "d": 400, "f": 50, "h": 30, "sf": 45, "ul": 500000}[f]
            s["decay %s %s" % (kind, f)] = {"size": 25, "query": {"function_score": {"query": body, kind: {f: dict(dec, origin=o, scale=o / 2.5)}, "boost_mode": "multiply"}}}
        s["decay %s lm modes" % kind] = {"size": 25, "query": {"function_score": {"query": body, kind: {"lm": {"origin": 400, "scale": 200}, "multi_value_mode": "avg"}}}}
        s["decay %s lm median" % kind] = {"size": 25, "query": {"function_score": {"query": body, kind: {"lm": {"origin": 400, "scale": 200}, "multi_value_mode": "median"}}}}
        s["decay %s date" % kind] = {"size": 25, "query": {"function_score": {"query": body, kind: {"dt": {"origin": 1_700_500_000_000, "scale": "3d", "offset": "1d"}}}}}
        s["decay %s date_nanos" % kind] = {"size": 25, "query": {"function_score": {"query": body, kind: {"dn": {"origin": "2024-06-15T00:00:00Z", "scale": "20d"}}}}}
        s["decay %s geo" % kind] = {"size": 25, "query": {"function_score": {"query": body, kind: {"gp": {"origin": {"lat": 10, "lon": 20}, "scale": "800km", "offset": "50km"}}}}}
        s["decay %s geo multi" % kind] = {"size": 25, "query": {"function_score": {"query": ma, kind: {"gpm": {"origin": "40.5,-73.5", "scale": "10km"}, "multi_value_mode": "max"}}}}
        s["decay %s geo min" % kind] = {"size": 25, "query": {"function_score": {"query": ma, kind: {"gpm": {"origin": "40.5,-73.5", "scale": "10km"}, "multi_value_mode": "min"}}}}
    # hammingbit
    s["hammingbit long"] = {"size": 20, "query": {"script_score": {"query": {"exists": {"field": "hl"}}, "script": knn("hl", 123456789012)}}}
    s["hammingbit long all"] = {"size": 20, "query": {"script_score": {"query": ma, "script": knn("hl", 5)}}}
    s["hammingbit binary"] = {"size": 20, "query": {"script_score": {"query": ma, "script": knn("bin", base64.b64encode(bytes([1, 2, 3, 4])).decode())}}}
    s["hammingbit binary short"] = {"size": 20, "query": {"script_score": {"query": ma, "script": knn("bin", base64.b64encode(bytes([0, 0, 7])).decode())}}}
    s["hammingbit binary long"] = {"size": 20, "query": {"script_score": {"query": ma, "script": knn("bin", base64.b64encode(bytes([255, 1, 2, 3, 4, 5, 6])).decode())}}}
    s["hammingbit min_score"] = {"size": 20, "query": {"script_score": {"query": ma, "script": knn("hl", 5), "min_score": 0.06}}}
    # the expression language
    def ex(source, **params):
        d = {"lang": "expression", "source": source}
        if params:
            d["params"] = params
        return d

    for name, src in [
        ("long", "doc['l'].value * 2"),
        ("score", "_score * doc['d'].value / 100"),
        ("multi", "doc['lm'].avg() + doc['lm'].median() + doc['lm'].min() + doc['lm'].max() + doc['lm'].sum()"),
        ("empty", "doc['l'].empty ? 1 : doc['l'].value / 1000"),
        ("length", "doc['lm'].length + doc['lm'].count()"),
        ("float", "doc['f'].value + doc['h'].value + doc['d'].value"),
        ("date", "doc['dt'].value / 1000000000000"),
        ("fns", "sqrt(doc['d'].value) + ln(doc['l'].value + 1) + min(doc['f'].value, 5) + max(doc['h'].value, 1) + pow(2, 3) + abs(0 - doc['l'].value)"),
        ("div0", "doc['l'].value / (doc['h'].value - doc['h'].value)"),
        ("methods", "doc['l'].getValue() + doc['lm'].size() + doc['d'].isEmpty()"),
    ]:
        s["expr score " + name] = {"size": 30, "query": {"script_score": {"query": body, "script": ex(src)}}}
        s["expr fs " + name] = {"size": 30, "query": {"function_score": {"query": body, "functions": [{"script_score": {"script": ex(src)}}], "boost_mode": "sum"}}}
        s["expr sort " + name] = {"size": 30, "query": ma, "sort": [{"_script": {"type": "number", "script": ex(src), "order": "desc"}}, "_doc"]}
    s["expr score params"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("doc['l'].value * k + j", k=0.5, j=3)}}}
    s["expr score min_score"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("doc['l'].value / 100"), "min_score": 20}}}
    s["expr score negative"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("0 - doc['l'].value")}}}
    s["expr score missing field"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("doc['nope'].value")}}}
    s["expr script query"] = {"size": 30, "query": {"bool": {"filter": [{"script": {"script": ex("doc['l'].value > 2500")}}]}}, "sort": ["_doc"]}
    s["expr script query count"] = {"size": 0, "track_total_hits": True, "query": {"script": {"script": ex("doc['l'].value > 100 && doc['d'].value < 400")}}}
    s["expr agg sum"] = {"size": 0, "aggs": {"a": {"sum": {"script": ex("doc['l'].value * 3")}}, "m": {"max": {"script": ex("doc['d'].value + 1")}}, "av": {"avg": {"script": ex("doc['lm'].avg()")}}}}
    s["expr terms agg"] = {"size": 0, "aggs": {"a": {"terms": {"script": ex("doc['l'].value % 10"), "size": 5}}}}
    s["expr histogram agg"] = {"size": 0, "aggs": {"a": {"histogram": {"script": ex("doc['l'].value"), "interval": 500}}}}
    s["expr terms_set"] = {"size": 30, "query": {"terms_set": {"tag": {"terms": ["t1", "t2", "t3"], "minimum_should_match_script": ex("1 + doc['l'].empty")}}}}
    s["expr min_score fs"] = {"size": 30, "query": {"function_score": {"query": body, "functions": [{"script_score": {"script": ex("doc['l'].value / 100")}}], "min_score": 20, "boost_mode": "replace"}}}
    s["expr date year"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("doc['dt'].date.year")}}}
    s["expr date method"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("doc['dt'].date.getHourOfDay() + doc['dt'].date.dayOfWeek")}}}
    s["expr date short"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("doc['dt'].getYear() + doc['dt'].getMonth()")}}}
    s["expr geo"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("abs(doc['gp'].lat) + abs(doc['gp'].lon) + doc['gp'].empty")}}}
    s["expr geo method"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("abs(doc['gp'].getLat()) + abs(doc['gp'].getLon()) + doc['gp'].isEmpty()")}}}
    s["expr haversin"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("1000 / (1 + haversin(10, 20, doc['gp'].lat, doc['gp'].lon))")}}}
    s["expr sf"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("doc['sf'].value")}}}
    s["expr dn"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("doc['dn'].value / 1000000000000")}}}
    s["expr ul"] = {"size": 30, "query": {"script_score": {"query": body, "script": ex("doc['ul'].value / 1000000")}}}
    return s


def run(base, name, body):
    try:
        r = bh.req(base, "POST", "/sc/_search?request_cache=false&rest_total_hits_as_int=true", dict(body, size=body.get("size", 10)))
    except Exception as e:  # an error is an answer too
        return {"error": str(e)[:400]}
    out = {"total": r["hits"]["total"], "hits": [[h["_id"], h["_score"]] for h in r["hits"]["hits"]], "sort": [h.get("sort") for h in r["hits"]["hits"]]}
    if "aggregations" in r:
        out["aggs"] = r["aggregations"]
    return out


def dump(base):
    return {name: run(base, name, body) for name, body in shapes().items()}


def norm(x):
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v) for v in x]
    if isinstance(x, float):
        return float("%.5g" % x)
    return x


def error_class(e):
    # the exception type and reason, without node-specific text
    import re
    return re.sub(r"@[0-9a-f]+", "@", e.split("reason")[-1] if "reason" in e else e)[:40]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "load":
        load(sys.argv[2])
    elif cmd == "dump":
        json.dump(dump(sys.argv[2]), open(sys.argv[3], "w"))
    elif cmd == "stats":
        st = bh.req(sys.argv[2], "GET", "/_plugins/lucene_rust/stats")
        print(json.dumps({k: v for k, v in st.get("fallbacks", {}).items()}, indent=1, sort_keys=True))
        print("native_errors", st.get("native_errors"))
    else:
        a, b = (norm(json.load(open(p))) for p in sys.argv[2:4])
        bad = 0
        for k in a:
            if "error" in a[k] or "error" in b.get(k, {}):
                same = ("error" in a[k]) == ("error" in b.get(k, {})) and error_class(a[k].get("error", "")) == error_class(b[k].get("error", ""))
            else:
                same = a[k] == b.get(k)
            if not same:
                bad += 1
            print(("DIFF " if not same else "same ") + k)
            if not same:
                print("   stock :", json.dumps(a[k])[:400])
                print("   plugin:", json.dumps(b.get(k))[:400])
        sys.exit(1 if bad else 0)
