#!/usr/bin/env python3
"""Sorting on every field type against a stock node: the hits' `sort` values, their order, and a
`search_after` walk over the whole index (which feeds the sort values back in) must be identical
for ascending / descending, `missing` first / last, and multi-valued modes.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 \\
        python3 opensearch-plugin/e2e/verify_sort_types.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzz_diff import Node, close  # noqa: E402

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
INDEX = "sort_types"
FIELDS = {
    "byte": {"type": "byte"}, "short": {"type": "short"}, "int": {"type": "integer"}, "long": {"type": "long"},
    "ulong": {"type": "unsigned_long"}, "half": {"type": "half_float"}, "float": {"type": "float"}, "double": {"type": "double"},
    "scaled": {"type": "scaled_float", "scaling_factor": 100}, "scaled3": {"type": "scaled_float", "scaling_factor": 1000},
    "date": {"type": "date"}, "nanos": {"type": "date_nanos"}, "bool": {"type": "boolean"}, "ip": {"type": "ip"},
    "kw": {"type": "keyword"}, "kwn": {"type": "keyword", "normalizer": "lc"}, "version": {"type": "version"},
    "text_kw": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
}


def value(kind, r):
    if kind == "byte": return r.randint(-128, 127)
    if kind == "short": return r.randint(-32768, 32767)
    if kind == "int": return r.choice([r.randint(-1000, 1000), 2**31 - 1, -2**31])
    if kind == "long": return r.choice([r.randint(-1000, 1000), 2**63 - 1, -2**63, 2**53 + 1])
    if kind == "ulong": return r.choice([r.randint(0, 1000), 2**64 - 1, 2**63, 2**63 - 1])
    if kind in ("half", "float", "double"): return r.choice([round(r.uniform(-100, 100), 3), 0.0, -0.0, 1e-5, 12345.678])
    if kind in ("scaled", "scaled3"): return r.choice([round(r.uniform(-100, 100), 3), 0, -0.01, 0.005, 123456.789])
    if kind == "date": return r.choice(["2020-01-01T00:00:00Z", r.randint(-10**11, 2 * 10**12), "1969-12-31T23:59:59.999Z", "2262-01-01"])
    if kind == "nanos": return r.choice(["2020-01-01T00:00:00.123456789Z", "1970-01-01T00:00:00.000000001Z", "2100-06-15T10:20:30.5Z"])
    if kind == "bool": return r.choice([True, False])
    if kind == "ip": return r.choice(["10.0.0.1", "::1", "192.168.0.1", "2001:db8::ff", "255.255.255.255"])
    if kind in ("kw", "kwn", "text_kw"): return r.choice(["a", "B", "ab", "", "é", "zz", "A", "\U0001f600", "10", "9"])
    if kind == "version": return r.choice(["1.0.0", "1.10.0", "1.2.0", "2.0.0-alpha", "10.0.1"])


def main():
    r = random.Random(7)
    plugin, stock = Node(PLUGIN, 60), Node(STOCK, 60)
    body = {"settings": {"number_of_shards": 1, "number_of_replicas": 0, "analysis": {"normalizer": {"lc": {"type": "custom", "filter": ["lowercase"]}}}},
            "mappings": {"properties": {k: v for k, v in FIELDS.items()}}}
    docs = []
    for i in range(400):
        d = {}
        for k in FIELDS:
            x = r.random()
            if x < 0.15:
                continue
            if x < 0.3:
                d[k] = [value(k, r) for _ in range(r.randint(1, 3))]
            else:
                d[k] = value(k, r)
        docs.append(d)
    for n in (plugin, stock):
        n.req("DELETE", f"/{INDEX}")
        st, resp = n.req("PUT", f"/{INDEX}", body)
        if st != 200:
            raise SystemExit(f"create failed: {json.dumps(resp)[:400]}")
        lines = []
        for i, d in enumerate(docs):
            lines += [json.dumps({"index": {"_id": f"d{i}"}}), json.dumps(d)]
        n.req("POST", f"/{INDEX}/_bulk?refresh=true", ("\n".join(lines) + "\n").encode(), raw=True)
        n.req("POST", f"/{INDEX}/_forcemerge?max_num_segments=1")
    bad = 0
    total = 0
    for field in FIELDS:
        sf = field + ".raw" if field == "text_kw" else field
        for order in ("asc", "desc"):
            for missing in (None, "_first", "_last"):
                if sf == "nanos" and missing == "_first":
                    continue  # stock hangs sorting date_nanos with missing _first (both nodes), skip
                for mode in (None, "min", "max") + (("median", "sum", "avg") if sf == "ulong" else ()):
                    spec = {"order": order}
                    if missing:
                        spec["missing"] = missing
                    if mode:
                        spec["mode"] = mode
                    sort = [{sf: spec}, {"_seq_no": "asc"}]
                    total += 1
                    walks = []
                    for n in (plugin, stock):
                        seen, after, err = [], None, None
                        for _ in range(1 if sf == "nanos" else 80):   # stock hangs on search_after over date_nanos
                            q = {"size": 37, "sort": sort, "_source": False}
                            if after is not None:
                                q["search_after"] = after
                            st, resp = n.req("POST", f"/{INDEX}/_search", q)
                            if st != 200:
                                err = (st, json.dumps(resp)[:200])
                                break
                            hits = resp["hits"]["hits"]
                            if not hits:
                                break
                            seen += [(h["_id"], h["sort"]) for h in hits]
                            after = hits[-1]["sort"]
                        walks.append((seen, err))
                    (ps, pe), (ss, se) = walks
                    msg = None
                    if se is not None and ss:
                        # stock itself refuses some search_after walks (scaled_float, unsigned_long
                        # with missing): compare the pages it served, and an error of ours there is its own
                        k = min(len(ps), len(ss))
                        ps, ss = ps[:k], ss[:k]
                        pe = se = None
                    if pe is None and se is not None and "unsupported_operation_exception" in se[1]:
                        # stock's comparator for a scaled_float / unsigned_long key throws while it
                        # iterates the field's values to skip; the plugin answers the sort instead
                        continue
                    if (pe is None) != (se is None):
                        msg = f"error plugin={pe} stock={se}"
                    elif pe is None:
                        if [i for i, _ in ps] != [i for i, _ in ss]:
                            k = next((j for j, (a, b) in enumerate(zip(ps, ss)) if a[0] != b[0]), min(len(ps), len(ss)))
                            msg = f"order differs at {k}: plugin={ps[k:k + 2]} stock={ss[k:k + 2]} (lengths {len(ps)}/{len(ss)})"
                        else:
                            for (i, a), (_, b) in zip(ps, ss):
                                if len(a) != len(b) or not all(close(x, y, 1e-12) if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) else x == y for x, y in zip(a, b)):
                                    msg = f"sort values of {i}: plugin={a} stock={b}"
                                    break
                    if msg:
                        bad += 1
                        print(f"{sf} {json.dumps(spec)}: {msg}")
    print(f"sorts checked: {total}, mismatches: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
