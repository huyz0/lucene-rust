"""How far past a search `timeout` a request runs: heavy requests over
memory_bounds.py's corpus (index `membounds`, loaded by it), each with a
TIMEOUT (default 50ms) and without, on the accelerated path and on
OpenSearch's own (index.search.accelerated false). Per case and path: took
(the server's), timed_out, hits returned and status, ROUNDS (default 5)
times.

Usage:
  timeout_bench.py BASE LABEL OUT
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_bounds as mb  # noqa: E402

ROUNDS = int(os.environ.get("ROUNDS", "5"))
TIMEOUT = os.environ.get("TIMEOUT", "50ms")
INDEX = "membounds"


def cases():
    q = {"size": 0, "track_total_hits": False}
    return {
        "terms_ids_avg": dict(q, aggs={"t": {"terms": {"field": "id", "size": 10},
                                              "aggs": {"a": {"avg": {"field": "n"}}}}}),
        "date_histogram_70ms_max": dict(q, aggs={"h": {"date_histogram": {"field": "ts", "fixed_interval": "70ms"},
                                                       "aggs": {"m": {"max": {"field": "n"}}}}}),
        "cardinality_ids": dict(q, aggs={"c": {"cardinality": {"field": "id", "precision_threshold": 40000}}}),
        "sort_all_by_n": {"size": 100, "track_total_hits": True, "sort": [{"n": "asc"}], "_source": False},
        "wildcard_msg_count": {"size": 10, "track_total_hits": True, "query": {"wildcard": {"msg": "w1*"}}},
        "match_count": {"size": 10, "track_total_hits": True, "query": {"match": {"msg": "w1 w2 w3 w4 w5 w6"}}},
    }


def run(base, body):
    out = []
    for _ in range(ROUNDS):
        st, r = mb.req(base, "POST", "/%s/_search?request_cache=false" % INDEX, body)
        out.append({"status": st, "took": r.get("took"), "timed_out": r.get("timed_out"),
                    "hits": len(r.get("hits", {}).get("hits", [])),
                    "total": (r.get("hits", {}).get("total") or {}).get("value"),
                    "error": (r.get("error") or {}).get("type")})
    return out


def main():
    base, label, out_path = sys.argv[1:4]
    result = {"label": label, "timeout": TIMEOUT, "cases": {}}
    for path in ("accelerated", "standard"):
        mb.req(base, "PUT", "/%s/_settings" % INDEX, {"index.search.accelerated": path == "accelerated"})
        for name, body in cases().items():
            for t in (None, TIMEOUT):
                b = dict(body, timeout=t) if t else body
                r = run(base, b)
                key = "%s/%s/%s" % (path, name, "timeout" if t else "none")
                result["cases"][key] = r
                took = sorted(x["took"] or -1 for x in r)
                print(label, key, "took", took, "timed_out", [x["timed_out"] for x in r],
                      "hits", r[-1]["hits"], "total", r[-1]["total"], "status", r[-1]["status"], r[-1]["error"] or "",
                      flush=True)
    mb.req(base, "PUT", "/%s/_settings" % INDEX, {"index.search.accelerated": True})
    stats = mb.req(base, "GET", "/_plugins/lucene_rust/stats")[1]
    result["fallbacks"] = stats.get("fallbacks")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
