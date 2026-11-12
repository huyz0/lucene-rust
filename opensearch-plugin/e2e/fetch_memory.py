"""Fetch-phase memory with very large documents: a few documents of SIZE_MB
(default 100) of `_source` each among small ones, fetched whole, filtered
and not at all, each request sampled for the container's anonymous memory
while it runs (memory_bounds.py's Sampler). Also a concurrent burst of whole
fetches, which a node must answer or refuse with OpenSearch's own
circuit_breaking_exception -- never by dying.

Usage:
  fetch_memory.py BASE CONTAINER LABEL OUT [load]

`load` indexes BIG (default 3) large documents and 1,000 small ones into
index `bigdocs` (one shard, refreshed and flushed). The node needs
http.max_content_length above SIZE_MB. The JSON at OUT holds, per case:
status codes, error types, took, the peak rss while it ran, and the plugin's
off-heap stats after it.
"""
import concurrent.futures
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_bounds as mb  # noqa: E402

SIZE_MB = int(os.environ.get("SIZE_MB", "100"))
BIG = int(os.environ.get("BIG", "3"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
INDEX = "bigdocs"


def load(base):
    mb.req(base, "DELETE", "/" + INDEX)
    st, r = mb.req(base, "PUT", "/" + INDEX, {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
        "mappings": {"dynamic": False, "properties": {"k": {"type": "keyword"}, "big": {"type": "boolean"}}}})
    assert st == 200, r
    rnd = random.Random(11)
    # Hex digits: compressible about 2x, as text is, so the stored block is
    # still tens of MB.
    payload = "%x" % rnd.getrandbits(SIZE_MB * 2**20 * 4)
    for i in range(BIG):
        doc = json.dumps({"k": "big%d" % i, "big": True, "payload": payload[i:] + payload[:i]}).encode()
        st, r = mb.req(base, "PUT", "/%s/_doc/big%d" % (INDEX, i), doc, timeout=1200)
        assert st in (200, 201), str(r)[:500]
    lines = []
    for i in range(1000):
        lines.append('{"index":{"_index":"%s","_id":"s%d"}}' % (INDEX, i))
        lines.append(json.dumps({"k": "s%d" % i, "big": False, "payload": "x" * 100}))
    st, r = mb.req(base, "POST", "/_bulk", ("\n".join(lines) + "\n").encode(), ndjson=True)
    assert st == 200 and not r.get("errors"), str(r)[:500]
    mb.req(base, "POST", "/%s/_refresh" % INDEX)
    mb.req(base, "POST", "/%s/_flush" % INDEX)
    print("loaded %d x %d MB documents" % (BIG, SIZE_MB), flush=True)


def cases():
    big = {"term": {"big": True}}
    return {
        "whole_source": {"size": BIG, "query": big},
        "filtered_source": {"size": BIG, "query": big, "_source": ["k"]},
        "no_source": {"size": BIG, "query": big, "_source": False},
        "get_one": None,
    }


def run(base, container, name, body):
    if body is not None:
        mb.INDEX = INDEX
        return mb.run_case(base, container, name, body, ROUNDS)
    out = {"status": [], "errors": [], "took_ms": []}
    import time
    with mb.Sampler(container) as s:
        for _ in range(ROUNDS):
            t = time.time()
            st, r = mb.req(base, "GET", "/%s/_doc/big0?realtime=false" % INDEX)
            out["status"].append(st)
            out["took_ms"].append(round((time.time() - t) * 1000))
            if st != 200:
                out["errors"].append(r.get("error", {}).get("type", "?"))
    out["peak_rss_mb"] = round(s.peak / 2**20)
    return out


def main():
    base, container, label, out_path = sys.argv[1:5]
    if len(sys.argv) > 5 and sys.argv[5] == "load":
        load(base)
    # FETCH=standard reads stored fields with Lucene (index.fetch.accelerated false), for the
    # stock comparison on the same node.
    standard = os.environ.get("FETCH") == "standard"
    mb.req(base, "PUT", "/%s/_settings" % INDEX, {"index.fetch.accelerated": not standard})
    result = {"label": label, "fetch": "standard" if standard else "accelerated",
              "baseline_rss_mb": round(mb.cgroup_rss(container) / 2**20), "cases": {}}
    for name, body in cases().items():
        r = run(base, container, name, body)
        r["off_heap"] = mb.plugin_stats(base)
        result["cases"][name] = r
        print(label, name, json.dumps({k: v for k, v in r.items() if k != "off_heap"}), flush=True)
    mb.INDEX = INDEX
    r = mb.concurrent_case(base, container, "concurrent", cases()["whole_source"], CONCURRENCY)
    r["off_heap"] = mb.plugin_stats(base)
    result["cases"]["concurrent_%d" % CONCURRENCY] = r
    print(label, "concurrent", json.dumps({k: v for k, v in r.items() if k != "off_heap"}), flush=True)
    result["breakers"] = mb.breaker_stats(base)
    result["final_rss_mb"] = round(mb.cgroup_rss(container) / 2**20)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    allowed = {"circuit_breaking_exception", "shard:circuit_breaking_exception"}
    # Filtering a 100 MB _source parses it, past OpenSearch's own string length limit: stock
    # answers it the same way.
    expected = {"filtered_source": {"parse_exception"}}
    bad = []
    for name, c in result["cases"].items():
        statuses = c["status"] if isinstance(c["status"], list) else [int(k) for k in c["status"]]
        if -1 in statuses:
            bad.append("%s: the node did not answer" % name)
        errs = c["errors"] if isinstance(c["errors"], list) else list(c["errors"])
        bad += ["%s: %s" % (name, e) for e in errs if e not in allowed | expected.get(name, set())]
    for b in bad:
        print(label, "FAIL", b, flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
