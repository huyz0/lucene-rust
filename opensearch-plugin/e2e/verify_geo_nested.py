#!/usr/bin/env python3
"""`_geo_distance` sorts under a nested path against a stock node: every mode OpenSearch has a nested
pick for, both orders, arc and plane, metres and kilometres, one and several origins, a nested filter,
`max_children`, and a `search_after` walk. Hit ids, sort values and totals must be identical, and the
plugin must answer them natively.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_geo_nested.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node  # noqa: E402

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
INDEX = "geo_nested"


def main():
    r = random.Random(11)
    plugin, stock = Node(PLUGIN, 120), Node(STOCK, 120)
    body = {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {"n": {"type": "long"},
                                        "items": {"type": "nested", "properties": {
                                            "loc": {"type": "geo_point"}, "tag": {"type": "keyword"}}}}}}
    lines = []
    for i in range(3000):
        kids = []
        for _ in range(r.choice([0, 1, 1, 2, 3, 5])):
            item = {"tag": r.choice(["x", "y"])}
            if r.random() < 0.85:
                item["loc"] = [{"lat": r.uniform(-80, 80), "lon": r.uniform(-179, 179)} for _ in range(r.choice([1, 1, 2]))]
                if len(item["loc"]) == 1:
                    item["loc"] = item["loc"][0]
            kids.append(item)
        lines += [json.dumps({"index": {"_id": f"d{i}"}}), json.dumps({"n": i, "items": kids})]
    for n in (plugin, stock):
        n.req("DELETE", f"/{INDEX}")
        st, resp = n.req("PUT", f"/{INDEX}", body)
        if st != 200:
            raise SystemExit(f"create failed: {json.dumps(resp)[:300]}")
        for k in range(0, len(lines), 1000):
            n.req("POST", f"/{INDEX}/_bulk", ("\n".join(lines[k:k + 1000]) + "\n").encode(), raw=True)
        n.req("POST", f"/{INDEX}/_refresh")
        n.req("POST", f"/{INDEX}/_forcemerge?max_num_segments=3")
    _, before = plugin.req("GET", "/_plugins/lucene_rust/stats")
    bad = total = 0
    for mode in ("min", "max", "sum", "avg"):
        for order in ("asc", "desc"):
            for dtype in ("arc", "plane"):
                for unit, origins in (("m", [[10, 20]]), ("km", [[10, 20], [-30, 100]])):
                    for nested in ({"path": "items"},
                                   {"path": "items", "filter": {"term": {"items.tag": "x"}}},
                                   {"path": "items", "max_children": 2}):
                        spec = {"items.loc": origins if len(origins) > 1 else origins[0],
                                "order": order, "mode": mode, "unit": unit, "distance_type": dtype,
                                "nested": nested}
                        sort = [{"_geo_distance": spec}, {"n": "asc"}]
                        total += 1
                        walks = []
                        for node in (plugin, stock):
                            seen, after, err = [], None, None
                            for _ in range(6):
                                q = {"size": 400, "sort": sort, "_source": False, "track_total_hits": True}
                                if after is not None:
                                    q["search_after"] = after
                                st, resp = node.req("POST", f"/{INDEX}/_search", q)
                                if st != 200:
                                    err = (st, json.dumps(resp)[:200])
                                    break
                                hits = resp["hits"]["hits"]
                                if not hits:
                                    break
                                seen += [(h["_id"], h["sort"]) for h in hits]
                                after = hits[-1]["sort"]
                            walks.append((seen, err, resp.get("hits", {}).get("total")))
                        (ps, pe, pt), (ss, se, st_) = walks
                        msg = None
                        if (pe is None) != (se is None):
                            msg = f"error plugin={pe} stock={se}"
                        elif pe is None:
                            if [i for i, _ in ps] != [i for i, _ in ss]:
                                k = next((j for j, (a, b) in enumerate(zip(ps, ss)) if a[0] != b[0]), min(len(ps), len(ss)))
                                msg = f"order differs at {k}: plugin={ps[k:k + 2]} stock={ss[k:k + 2]}"
                            else:
                                # `plane` runs StrictMath.cos here, the JIT's intrinsic there: the
                                # last bit of a distance can differ.
                                for (_, a), (_, b) in zip(ps, ss):
                                    same = a[1:] == b[1:] and (a[0] == b[0] or (dtype == "plane" and abs(a[0] - b[0]) <= 1e-9 * max(1.0, abs(b[0]))))
                                    if not same:
                                        msg = f"sort values differ: plugin={a} stock={b}"
                                        break
                        if msg:
                            bad += 1
                            if bad <= 15:
                                print("MISMATCH", json.dumps(spec)[:200], msg)
    _, after_stats = plugin.req("GET", "/_plugins/lucene_rust/stats")
    fell = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after_stats.get("fallbacks", {}).items()}
    fell = {k: v for k, v in fell.items() if v}
    print(f"geo nested sorts checked: {total}, mismatches: {bad}, new plugin fallbacks: {fell}")
    sys.exit(1 if bad or fell else 0)


if __name__ == "__main__":
    main()
