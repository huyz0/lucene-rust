#!/usr/bin/env python3
"""How much of a cluster's search traffic the lucene-rust plugin runs native,
and why the rest falls back -- from every node's
`GET /_plugins/lucene_rust/stats` (docs/feature-matrix.md).

    scripts/fallback-report.py [URL] [--save FILE] [--since FILE]

URL defaults to http://localhost:9200. The counters are totals since each
node started, so a workload's mix is the difference between two readings:
`--save` writes this reading to FILE, `--since FILE` reports the difference
from a saved one. Standard library only.
"""
import json
import sys
import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def reading(base):
    """Per node: native queries and fallbacks by reason."""
    nodes = get(f"{base}/_nodes/http?filter_path=nodes.*.name,nodes.*.http.publish_address")["nodes"]
    out = {}
    for n in nodes.values():
        addr = n["http"]["publish_address"].split("/")[-1]
        try:
            s = get(f"http://{addr}/_plugins/lucene_rust/stats")
        except OSError:
            # A publish address the caller cannot reach (a container's):
            # the node the caller named answers for itself only.
            s = get(f"{base}/_plugins/lucene_rust/stats") if len(nodes) == 1 else None
        if s is None:
            print(f"warning: node {n['name']} at {addr} unreachable; left out", file=sys.stderr)
            continue
        out[n["name"]] = {"native": s["native_queries"], "fallbacks": s.get("fallbacks", {})}
    return out


def minus(now, before):
    out = {}
    for node, r in now.items():
        b = before.get(node, {"native": 0, "fallbacks": {}})
        if r["native"] < b["native"]:
            b = {"native": 0, "fallbacks": {}}  # the node restarted: its counters did too
        out[node] = {"native": r["native"] - b["native"],
                     "fallbacks": {k: v - b["fallbacks"].get(k, 0) for k, v in r["fallbacks"].items()}}
    return out


def main():
    args = sys.argv[1:]
    base = next((a for a in args if a.startswith("http")), "http://localhost:9200").rstrip("/")
    now = reading(base)
    if "--save" in args:
        with open(args[args.index("--save") + 1], "w") as f:
            json.dump(now, f)
    r = now
    if "--since" in args:
        with open(args[args.index("--since") + 1]) as f:
            r = minus(now, json.load(f))
    native = sum(x["native"] for x in r.values())
    reasons = {}
    for x in r.values():
        for k, v in x["fallbacks"].items():
            reasons[k] = reasons.get(k, 0) + v
    # `disabled` is an index turned off on purpose, not a shape the plugin missed.
    eligible = native + sum(v for k, v in reasons.items() if k != "disabled")
    total = native + sum(reasons.values())
    print(f"{len(r)} nodes, {total} searches: {native} native "
          f"({100 * native / total:.1f}%)" if total else f"{len(r)} nodes, no searches counted")
    if eligible and "disabled" in reasons:
        print(f"  of those on indices with native search enabled: {100 * native / eligible:.1f}% native")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"  {k:28} {v:10}  {100 * v / total:5.1f}%")


if __name__ == "__main__":
    main()
