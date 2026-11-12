"""Highlighting latency on the plugin node against stock OpenSearch.

Usage: highlight_bench.py [load|run|all] [DOCS] [SECONDS]
  PLUGIN (default http://localhost:9200) and STOCK (default
  http://localhost:9201) as in highlight_matrix.py. `load` fills one index
  per offset source (postings offsets, term vectors, re-analysis) on both
  nodes with highlight_matrix.py's corpus; `run` times each case for SECONDS
  (default 12) a node, the two nodes alternating in rounds, and prints the
  median and p90 milliseconds and the stock/plugin ratio of the medians. The
  `no_highlight` rows are the same searches without highlighting: the cost
  around the highlighter. CASES (comma-separated) runs only those cases."""
import json, os, random, statistics, sys, time
import requests

sys.path.insert(0, os.path.dirname(__file__))
import highlight_matrix as m

PLUGIN, STOCK = m.PLUGIN, m.STOCK
S = requests.Session()

INDICES = {
    "hlb_offsets": m.mapping(m.text_field(index_options="offsets")),
    "hlb_tv": m.mapping(m.text_field(term_vector="with_positions_offsets")),
    "hlb_analysis": m.mapping(m.text_field()),
}

CASES = {
    "term": ({"match": {"body": "fox"}}, {"fields": {"body": {}}}),
    "terms": ({"match": {"body": "quick brown gold"}}, {"fields": {"body": {}}}),
    "phrase": ({"match_phrase": {"body": "brown fox"}}, {"fields": {"body": {}}}),
    "prefix": ({"prefix": {"body": "sea"}}, {"fields": {"body": {}}}),
    "phrase_prefix": ({"match_phrase_prefix": {"body": "quick br"}}, {"fields": {"body": {}}}),
    "all_fragments": ({"match": {"body": "fox"}}, {"fields": {"body": {"number_of_fragments": 0}}}),
}


def load(docs):
    for base in (PLUGIN, STOCK):
        for name, mapping in INDICES.items():
            S.delete(f"{base}/{name}")
            r = S.put(f"{base}/{name}", json={
                "settings": {"number_of_shards": 1, "number_of_replicas": 0, "analysis": m.ANALYSIS},
                "mappings": mapping,
            })
            assert r.ok, r.text
            rnd = random.Random(7)
            lines = []
            for i in range(docs):
                lines.append(json.dumps({"index": {"_id": str(i)}}))
                lines.append(json.dumps(m.doc(rnd, i)))
                if len(lines) >= 4000 or i == docs - 1:
                    r = S.post(f"{base}/{name}/_bulk", data="\n".join(lines) + "\n",
                               headers={"Content-Type": "application/x-ndjson"})
                    assert r.ok and not r.json()["errors"], r.text[:500]
                    lines = []
            S.post(f"{base}/{name}/_forcemerge?max_num_segments=1")
            S.post(f"{base}/{name}/_refresh")
    print("loaded", docs, "docs into", len(INDICES), "indices on both nodes")


def timed(base, index, body):
    t = time.perf_counter()
    r = S.post(f"{base}/{index}/_search", json=body)
    dt = (time.perf_counter() - t) * 1000
    assert r.ok, r.text[:300]
    return dt


def run(seconds, size=50):
    print(f"{'index':14} {'case':16} {'plugin p50':>10} {'p90':>7} {'stock p50':>10} {'p90':>7} {'stock/plugin':>12}")
    for index in INDICES:
        only = os.environ.get("CASES")
        for case, (query, highlight) in list(CASES.items()) + [("no_highlight", (CASES["terms"][0], None))]:
            if only and case not in only.split(","):
                continue
            body = {"query": query, "size": size, "_source": False}
            if highlight is not None:
                body["highlight"] = highlight
            for base in (PLUGIN, STOCK):
                for _ in range(20):
                    timed(base, index, body)
            samples = {PLUGIN: [], STOCK: []}
            end = time.time() + 2 * seconds
            while time.time() < end:
                for base in (PLUGIN, STOCK):
                    for _ in range(10):
                        samples[base].append(timed(base, index, body))
            p = sorted(samples[PLUGIN])
            s = sorted(samples[STOCK])
            p50, s50 = statistics.median(p), statistics.median(s)
            print(f"{index:14} {case:16} {p50:10.2f} {p[int(len(p) * .9)]:7.2f} {s50:10.2f} "
                  f"{s[int(len(s) * .9)]:7.2f} {s50 / p50:11.2f}x", flush=True)
    stats = S.get(f"{PLUGIN}/_plugins/lucene_rust/stats").json()
    print("accelerated_highlights", stats.get("accelerated_highlights"), "accelerated_errors", stats.get("accelerated_errors"))
    print("highlight fallbacks", {k: v for k, v in stats.get("fallbacks", {}).items() if "highlight" in k})


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    docs = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 12
    if mode in ("load", "all"):
        load(docs)
    if mode in ("run", "all"):
        run(seconds)
