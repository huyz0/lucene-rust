#!/usr/bin/env python3
"""Sort and search_after shapes that used to run on Lucene, against a stock node: each search is
sent to both, the hits' ids, sort values and totals must be identical, and the plugin must report
no new fallback for it (`fallbacks` in `GET _plugins/accelerated/stats`).

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_sort_shapes.py [--docs 30000] [--only substring]

The index holds more documents than the total-hits threshold (10000) in several segments, because
Lucene's comparators skip with the field's points past it and the answer must match there too.
"""
import argparse
import json
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node  # noqa: E402

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
INDEX = "sort_shapes"


def mapping():
    return {"properties": {
        "sf": {"type": "scaled_float", "scaling_factor": 100},
        "sf3": {"type": "scaled_float", "scaling_factor": 1000},
        "sfm": {"type": "scaled_float", "scaling_factor": 100},
        "n": {"type": "long"},
        "d": {"type": "double"},
        "dm": {"type": "double"},
        "lm": {"type": "long"},
        "hf": {"type": "half_float"},
        "ul": {"type": "unsigned_long"},
        "kw": {"type": "keyword"},
        "loc": {"type": "geo_point"},
        "locs": {"type": "geo_point"},
        "title": {"type": "text"},
        "items": {"type": "nested", "properties": {
            "p": {"type": "long"}, "sf": {"type": "scaled_float", "scaling_factor": 100},
            "hf": {"type": "half_float"}, "ul": {"type": "unsigned_long"}, "tag": {"type": "keyword"},
            "d": {"type": "double"}}},
    }}


def doc(r, i):
    d = {"n": i, "title": r.choice(["alpha beta", "beta", "gamma alpha", "delta"]) + " w%d" % (i % 7)}
    if r.random() < 0.9:
        d["sf"] = round(r.uniform(-500, 500), r.choice([0, 1, 2, 3]))
    if r.random() < 0.9:
        d["sf3"] = round(r.uniform(-50, 50), 3)
    if r.random() < 0.8:
        d["sfm"] = [round(r.uniform(-100, 100), 2) for _ in range(r.randint(1, 4))]
    if r.random() < 0.9:
        d["d"] = round(r.uniform(-100, 100), 3)
    if r.random() < 0.8:
        d["dm"] = [round(r.uniform(-100, 100), 2) for _ in range(r.randint(1, 4))]
    if r.random() < 0.8:
        d["lm"] = [r.randint(-100, 100) for _ in range(r.randint(1, 4))]
    if r.random() < 0.8:
        d["hf"] = round(r.uniform(-60, 60), 1)
    if r.random() < 0.8:
        d["ul"] = r.choice([r.randint(0, 5000), 2**63 + r.randint(0, 1000)])
    if r.random() < 0.9:
        d["kw"] = r.choice(["a", "b", "c", "d", "e", "f", "g"])
    if r.random() < 0.9:
        d["loc"] = {"lat": r.uniform(-60, 60), "lon": r.uniform(-170, 170)}
    if r.random() < 0.7:
        d["locs"] = [{"lat": r.uniform(-60, 60), "lon": r.uniform(-170, 170)} for _ in range(r.randint(1, 3))]
    if r.random() < 0.85:
        d["items"] = [
            {k: v for k, v in {
                "p": r.randint(0, 50),
                "sf": round(r.uniform(-20, 20), 2) if r.random() < 0.9 else None,
                "hf": round(r.uniform(-20, 20), 1) if r.random() < 0.9 else None,
                "ul": r.randint(0, 100) if r.random() < 0.9 else None,
                "d": round(r.uniform(-20, 20), 2) if r.random() < 0.9 else None,
                "tag": r.choice(["x", "y", "z"])}.items() if v is not None}
            for _ in range(r.randint(1, 4))]
    return d


TIE = {"_seq_no": "asc"}
SCALE = {"sf": 100, "sf3": 1000, "sfm": 100}


def java_round(x):
    return int(math.floor(x + 0.5))


def truth(docs, field, spec, size):
    """The first `size` ids of an exact sort. Stock's own answer is wrong once more documents than
    the total-hits threshold match, for a scaled_float (its comparator skips with the field's points,
    which hold the scaled longs, as if they were doubles) and for mode sum (a sum passes the values
    the points bound): it drops documents that belong in the top hits."""
    factor = SCALE.get(field)
    order, missing, mode = spec.get("order", "asc"), spec.get("missing", "_last"), spec.get("mode")
    rows = []
    for i, d in enumerate(docs):
        vals = d.get(field)
        vals = None if vals is None else sorted(java_round(v * factor) if factor else v for v in (vals if isinstance(vals, list) else [vals]))
        if not vals:
            m = missing
            k = (math.inf if (m == "_last") == (order == "asc") else -math.inf) if isinstance(m, str) else float(m)
            rows.append((k, i))
            continue
        ds = [v * (1.0 / factor) if factor else v for v in vals]
        n = len(ds)
        mid = (n - 1) // 2
        pick = {None: ds[-1] if order == "desc" else ds[0], "min": ds[0], "max": ds[-1], "sum": sum(ds), "avg": sum(ds) / n,
                "median": ds[mid] if n % 2 else (ds[mid] + ds[mid + 1]) / 2}[mode]
        rows.append((pick, i))
    rows.sort(key=lambda t: (-t[0] if order == "desc" else t[0], t[1]))
    return [f"d{i}" for _, i in rows[:size]]
MODES = [None, "min", "max", "sum", "avg", "median"]


def nested_truth(docs, field, spec, size):
    """The first `size` ids of an exact nested sort over `items` (fields p, sf, d)."""
    order, missing, mode = spec.get("order", "asc"), spec.get("missing", "_last"), spec.get("mode")
    nest = spec["nested"]
    sub = field.split(".")[1]
    rows = []
    for i, d in enumerate(docs):
        kids = [k for k in d.get("items", []) if "filter" not in nest or k.get("tag") == nest["filter"]["term"]["items.tag"]]
        vals = [k[sub] for k in kids if sub in k][:nest.get("max_children", len(kids))]
        if sub == "sf":
            vals = [java_round(v * 100) * (1.0 / 100) for v in vals]
        if sub == "hf":
            vals = [struct.unpack("e", struct.pack("e", v))[0] for v in vals]
        if not vals:
            k = (math.inf if (missing == "_last") == (order == "asc") else -math.inf)
        else:
            n = len(vals)
            m = mode or ("max" if order == "desc" else "min")
            k = {"min": min(vals), "max": max(vals), "sum": sum(vals), "avg": sum(vals) / n}[m]
        rows.append((k, i))
    rows.sort(key=lambda t: (-t[0] if order == "desc" else t[0], t[1]))
    return [f"d{i}" for _, i in rows[:size]]


def cases():
    """(name, body, oracle): oracle is (field, spec) where stock's own answer cannot be the reference."""
    out = []
    for f in ("sf", "sf3", "sfm", "d", "dm", "lm", "hf", "ul"):
        for order in ("asc", "desc"):
            for missing in (None, "_first", "_last", -3.5):
                if f == "ul" and missing is not None:
                    continue  # stock raises UnsupportedOperationException for most of these
                for mode in MODES:
                    if f == "ul" and mode == "median":
                        continue
                    spec = {"order": order}
                    if missing is not None:
                        spec["missing"] = missing
                    if mode:
                        spec["mode"] = mode
                    out.append((f"{f} {json.dumps(spec)}", {"size": 25, "sort": [{f: spec}, TIE], "_source": False},
                                (f, spec) if f in SCALE or spec.get("mode") == "sum" and f in ("dm", "lm") else None))
                    out.append((f"{f} {json.dumps(spec)} filtered", {
                        "size": 12, "query": {"range": {"n": {"lt": 6000}}}, "sort": [{f: spec}, TIE], "_source": False}, None))
    for f in ("items.p", "items.sf", "items.hf", "items.d"):
        for order in ("asc", "desc"):
            for mode in (None, "min", "max", "sum", "avg", "median"):
                if mode == "median":
                    continue  # stock refuses a nested median
                spec = {"order": order, "nested": {"path": "items"}}
                if mode:
                    spec["mode"] = mode
                out.append((f"{f} nested {json.dumps(spec)}", {"size": 25, "sort": [{f: dict(spec)}, TIE], "_source": False}, ("nested", f, spec) if f in ("items.p", "items.sf", "items.d", "items.hf") else None))
                f2 = dict(spec, nested={"path": "items", "filter": {"term": {"items.tag": "x"}}, "max_children": 2})
                out.append((f"{f} nested filtered {json.dumps(f2)}", {"size": 25, "sort": [{f: f2}, TIE], "_source": False}, ("nested", f, f2) if f in ("items.p", "items.sf", "items.d", "items.hf") else None))
    out.append(("sf multi key", {"size": 40, "query": {"range": {"n": {"lt": 9000}}}, "sort": [{"sf": "desc"}, {"sf3": "asc"}, {"kw": "asc"}, TIE], "_source": False}, None))
    out.append(("sf from", {"size": 10, "from": 5000, "query": {"range": {"n": {"lt": 9000}}}, "sort": [{"sf": "asc"}, TIE], "_source": False}, None))
    return out


def walk(node, sort, query=None, size=500, pages=6):
    """A search_after walk: the ids and sort values of `pages` pages, or the error."""
    seen, after = [], None
    for _ in range(pages):
        q = {"size": size, "sort": sort, "_source": False, "track_total_hits": True}
        if query:
            q["query"] = query
        if after is not None:
            q["search_after"] = after
        st, resp = node.req("POST", f"/{INDEX}/_search", q)
        if st != 200:
            return seen, (st, resp.get("error", {}).get("type"))
        hits = resp["hits"]["hits"]
        if not hits:
            break
        seen += [(h["_id"], h["sort"]) for h in hits]
        after = hits[-1]["sort"]
    return seen, None


def digest(resp):
    h = resp["hits"]
    return {"total": h["total"], "hits": [(x["_id"], x.get("sort"), x.get("_score")) for x in h["hits"]],
            "terminated_early": resp.get("terminated_early"), "timed_out": resp.get("timed_out")}


def fallbacks(node):
    st, r = node.req("GET", "/_plugins/accelerated/stats")
    return dict(r.get("fallbacks", {})) if st == 200 else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=30000)
    ap.add_argument("--only", default="")
    ap.add_argument("--keep", action="store_true", help="reuse the index if it exists")
    args = ap.parse_args()
    plugin, stock = Node(PLUGIN, 120), Node(STOCK, 120)
    r = random.Random(11)
    docs = [doc(r, i) for i in range(args.docs)]
    for n in (plugin, stock):
        if args.keep and n.req("GET", f"/{INDEX}/_count")[0] == 200:
            continue
        n.req("DELETE", f"/{INDEX}")
        st, resp = n.req("PUT", f"/{INDEX}", {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
                                               "mappings": mapping()})
        if st != 200:
            raise SystemExit(f"create failed: {json.dumps(resp)[:300]}")
    if not (args.keep and plugin.req("GET", f"/{INDEX}/_count")[1].get("count") == args.docs):
        for n in (plugin, stock):
            for b in range(0, len(docs), 5000):
                lines = []
                for i in range(b, min(b + 5000, len(docs))):
                    lines += [json.dumps({"index": {"_id": f"d{i}"}}), json.dumps(docs[i])]
                st, resp = n.req("POST", f"/{INDEX}/_bulk", ("\n".join(lines) + "\n").encode(), raw=True)
                assert st == 200 and not resp.get("errors"), json.dumps(resp)[:300]
                n.req("POST", f"/{INDEX}/_refresh")
            n.req("POST", f"/{INDEX}/_forcemerge?max_num_segments=3")
    bad = total = 0
    before = fallbacks(plugin)
    for name, body, oracle in cases():
        if args.only and args.only not in name:
            continue
        total += 1
        if oracle:
            sp, rp = plugin.req("POST", f"/{INDEX}/_search", body)
            want = nested_truth(docs, oracle[1], oracle[2], body["size"]) if oracle[0] == "nested" else truth(docs, oracle[0], oracle[1], body["size"])
            got = [h["_id"] for h in rp["hits"]["hits"]] if sp == 200 else None
            if got != want:
                bad += 1
                print(f"{name}: plugin {got[:4] if got else sp} truth {want[:4]}")
            continue
        sp, rp = plugin.req("POST", f"/{INDEX}/_search", body)
        ss, rs = stock.req("POST", f"/{INDEX}/_search", body)
        if sp == 200 and ss == 500 and "unsupported_operation_exception" in json.dumps(rs):
            # unsigned_long's comparator throws while it skips over a missing-value bound; the
            # plugin answers the sort instead
            continue
        if sp != ss:
            bad += 1
            print(f"{name}: status plugin={sp} stock={ss} {json.dumps(rp)[:200]} / {json.dumps(rs)[:200]}")
            continue
        if sp != 200:
            continue
        a, b = digest(rp), digest(rs)
        if a != b:
            bad += 1
            k = next((j for j, (x, y) in enumerate(zip(a["hits"], b["hits"])) if x != y), None)
            print(f"{name}: differs total {a['total']}/{b['total']} hit {k}: "
                  f"plugin={a['hits'][k] if k is not None else None} stock={b['hits'][k] if k is not None else None}")
    for label, sort, query in [
        ("walk sf desc", [{"sf": {"order": "desc"}}, TIE], None),
        ("walk geo asc m", [{"_geo_distance": {"loc": [10.5, 20.5], "order": "asc", "unit": "m"}}, TIE], None),
        ("walk geo asc km", [{"_geo_distance": {"loc": {"lat": -30.0, "lon": 100.0}, "order": "asc", "unit": "km"}}, TIE], None),
        ("walk geo desc", [{"_geo_distance": {"loc": [10.5, 20.5], "order": "desc", "unit": "m"}}, TIE], None),
        ("walk geo multi max", [{"_geo_distance": {"locs": [10.5, 20.5], "order": "asc", "mode": "max", "unit": "m"}}, TIE], None),
        ("walk geo two keys", [{"_geo_distance": {"loc": [0.5, 0.5], "order": "asc", "unit": "m"}}, {"kw": "asc"}, TIE], None),
        ("walk sfm avg", [{"sfm": {"order": "asc", "mode": "avg"}}, TIE], None),
    ]:
        if args.only and args.only not in label:
            continue
        total += 1
        (ps, pe), (ss, se) = walk(plugin, sort, query), walk(stock, sort, query)
        if pe != se or ps != ss:
            bad += 1
            print(f"{label}: plugin err={pe} n={len(ps)} stock err={se} n={len(ss)}")
    # Requests Lucene's collector refuses: the same error from both nodes.
    refused = [
        ("search_after null on a long", {"size": 2, "sort": [{"n": "asc"}], "search_after": [None]}),
        ("search_after null on a double", {"size": 2, "sort": [{"d": "desc"}, TIE], "search_after": [None, 5]}),
        ("search_after null after a keyword", {"size": 2, "sort": [{"kw": "asc"}, {"n": "asc"}], "search_after": ["a", None]}),
        ("search_after null on the score", {"size": 2, "sort": ["_score", {"n": "asc"}], "search_after": [None, 5]}),
        ("search_after without a sort", {"size": 2, "query": {"match": {"title": "alpha"}}, "search_after": [1.5, 3]}),
    ]
    for name, body in refused:
        if args.only and args.only not in name:
            continue
        total += 1
        sp, rp = plugin.req("POST", f"/{INDEX}/_search", body)
        ss, rs = stock.req("POST", f"/{INDEX}/_search", body)
        err = lambda r: json.dumps(r.get("error", {}).get("root_cause")) + json.dumps(r.get("error", {}).get("caused_by", {}).get("type"))
        if sp != ss or err(rp) != err(rs):
            bad += 1
            print(f"{name}: plugin {sp} {err(rp)[:160]} stock {ss} {err(rs)[:160]}")
    after = fallbacks(plugin)
    new = {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0) > 0}
    print(f"searches checked: {total}, mismatches: {bad}, new plugin fallbacks: {json.dumps(new)}")
    return 1 if bad or new else 0


if __name__ == "__main__":
    sys.exit(main())
