#!/usr/bin/env python3
"""Sort shapes against stock, keep-alive client: median ms per shape on the index verify_sort_shapes.py builds.

    PLUGIN_URL=http://localhost:9200 STOCK_URL=http://localhost:9201 python3 opensearch-plugin/e2e/bench_sort_shapes.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchhttp as bh  # noqa: E402

PLUGIN = os.environ.get("PLUGIN_URL", "http://localhost:9200")
STOCK = os.environ.get("STOCK_URL", "http://localhost:9201")
TIE = {"_seq_no": "asc"}
F = {"range": {"n": {"lt": 6000}}}
NEST = {"path": "items"}
SHAPES = {
    "scaled_float asc, 6k matches": {"query": F, "sort": [{"sf": "asc"}, TIE]},
    "scaled_float desc multi, 6k": {"query": F, "sort": [{"sfm": {"order": "desc", "mode": "max"}}, TIE]},
    "scaled_float asc, 30k": {"sort": [{"sf": "asc"}, TIE]},
    "double sum, 30k": {"sort": [{"dm": {"order": "asc", "mode": "sum"}}, TIE]},
    "nested long min, 30k": {"sort": [{"items.p": {"order": "asc", "nested": NEST}}, TIE]},
    "nested double max filtered, 30k": {"sort": [{"items.d": {"order": "desc", "mode": "max", "nested": {"path": "items", "filter": {"term": {"items.tag": "x"}}}}}, TIE]},
    "nested scaled_float avg, 30k": {"sort": [{"items.sf": {"order": "asc", "mode": "avg", "nested": NEST}}, TIE]},
    "nested half_float min, 30k": {"sort": [{"items.hf": {"order": "asc", "nested": NEST}}, TIE]},
}


def main():
    print(f"{'shape':40} {'stock ms':>9} {'plugin ms':>10} {'x':>6}")
    for name, body in SHAPES.items():
        body = dict(body, size=25, _source=False)
        res = {}
        for label, url in (("stock", STOCK), ("plugin", PLUGIN)):
            ms = bh.measure(lambda: bh.req(url, "POST", "/sort_shapes/_search", body), rounds=100)
            res[label] = bh.summarize(ms)["median_ms"]
        print(f"{name:40} {res['stock']:9.2f} {res['plugin']:10.2f} {res['stock'] / res['plugin']:6.2f}")


if __name__ == "__main__":
    main()
