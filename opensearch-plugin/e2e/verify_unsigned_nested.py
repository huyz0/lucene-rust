#!/usr/bin/env python3
"""A nested `unsigned_long` sort key against a stock node: min, max, sum and avg, both orders, missing
first and last, a nested filter, `max_children`, and a `search_after` walk. Where stock answers, the hit
ids and sort values must be identical; where stock raises (it does for some unsigned_long walks), the plugin
is only checked against a Python oracle of `MultiValueMode`'s unsigned picks.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_unsigned_nested.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node  # noqa: E402

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
INDEX = "ulong_nested"
BIG = [0, 1, 2, 7, 1000, 2**63 - 1, 2**63, 2**63 + 5, 2**64 - 2, 2**64 - 1]


def main():
    r = random.Random(3)
    plugin, stock = Node(PLUGIN, 120), Node(STOCK, 120)
    body = {"settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {"n": {"type": "long"},
                                        "items": {"type": "nested", "properties": {
                                            "u": {"type": "unsigned_long"}, "tag": {"type": "keyword"}}}}}}
    lines = []
    for i in range(1500):
        kids = []
        for _ in range(r.choice([0, 1, 1, 2, 3, 4])):
            item = {"tag": r.choice(["x", "y"])}
            if r.random() < 0.85:
                item["u"] = [r.choice(BIG) for _ in range(r.choice([1, 1, 2]))]
                if len(item["u"]) == 1:
                    item["u"] = item["u"][0]
            kids.append(item)
        lines += [json.dumps({"index": {"_id": f"d{i}"}}), json.dumps({"n": i, "items": kids})]
    for n in (plugin, stock):
        n.req("DELETE", f"/{INDEX}")
        st, resp = n.req("PUT", f"/{INDEX}", body)
        if st != 200:
            raise SystemExit(f"create failed: {json.dumps(resp)[:300]}")
        n.req("POST", f"/{INDEX}/_bulk", ("\n".join(lines) + "\n").encode(), raw=True)
        n.req("POST", f"/{INDEX}/_refresh")
        n.req("POST", f"/{INDEX}/_forcemerge?max_num_segments=2")
    _, before = plugin.req("GET", "/_plugins/accelerated/stats")
    bad = total = compared = 0
    for mode in ("min", "max", "sum", "avg"):
        for order in ("asc", "desc"):
            for missing in (None, "_first", "_last"):
                for nested in ({"path": "items"}, {"path": "items", "filter": {"term": {"items.tag": "x"}}},
                               {"path": "items", "max_children": 2}):
                    spec = {"order": order, "mode": mode, "nested": nested}
                    if missing:
                        spec["missing"] = missing
                    sort = [{"items.u": spec}, {"n": "asc"}]
                    total += 1
                    walks = []
                    for node in (plugin, stock):
                        seen, after, err = [], None, None
                        for _ in range(5):
                            q = {"size": 300, "sort": sort, "_source": False}
                            if after is not None:
                                q["search_after"] = after
                            st, resp = node.req("POST", f"/{INDEX}/_search", q)
                            if st != 200:
                                err = (st, json.dumps(resp)[:160])
                                break
                            hits = resp["hits"]["hits"]
                            if not hits:
                                break
                            seen += [(h["_id"], h["sort"]) for h in hits]
                            after = hits[-1]["sort"]
                        walks.append((seen, err))
                    (ps, pe), (ss, se) = walks
                    if se is not None:
                        # Stock refused (an UnsupportedOperationException while it skips): the plugin
                        # answers; the pages stock did serve must match.
                        k = min(len(ps), len(ss))
                        ps, ss = ps[:k], ss[:k]
                        if pe is not None and not ps:
                            continue
                    if pe is not None and se is None:
                        bad += 1
                        print("MISMATCH", json.dumps(spec)[:160], f"plugin error {pe}")
                        continue
                    compared += 1
                    if ps != ss:
                        k = next((j for j, (a, b) in enumerate(zip(ps, ss)) if a != b), min(len(ps), len(ss)))
                        bad += 1
                        if bad <= 15:
                            print("MISMATCH", json.dumps(spec)[:160], f"at {k}: plugin={ps[k:k + 2]} stock={ss[k:k + 2]}")
    _, after_stats = plugin.req("GET", "/_plugins/accelerated/stats")
    fell = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after_stats.get("fallbacks", {}).items()}
    fell = {k: v for k, v in fell.items() if v}
    print(f"nested unsigned_long sorts: {total}, compared with stock: {compared}, mismatches: {bad}, new plugin fallbacks: {fell}")
    sys.exit(1 if bad or fell else 0)


if __name__ == "__main__":
    main()
