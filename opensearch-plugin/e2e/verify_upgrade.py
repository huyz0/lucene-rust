#!/usr/bin/env python3
"""M8 acceptance criterion 3: an index an OpenSearch 2.x node wrote, served
natively after the upgrade to OpenSearch 3.8.0 with the lucene-rust plugin.

Driven by scripts/verify-opensearch-upgrade.sh, in two phases:

  load <base>
      On the OpenSearch 2.19 node (Lucene 9.12 segments): a representative
      corpus -- text, keyword and numeric fields with deletes and updates
      (verify_opensearch.py's `single`/`multi` indices, one and three shards),
      a `knn_vector` field in both (the k-NN plugin's flat doc-values field,
      `index.knn` off), nested documents (`nest`), and a k-NN index proper
      (`knn`: `index.knn: true`, a Lucene HNSW field) -- then an fs snapshot
      of all of it, and a flush, so the data directory the upgraded nodes
      open holds every document in a Lucene 9.12 segment.

  verify <plugin-base> <stock-base>
      Two 3.8.0 nodes opened the 2.19 data directory in place (the plugin node
      the original, the stock node a copy of it). Checks, per index:

        * every index opens (green) and every segment is still a Lucene 9.12
          one -- the old format is what gets served;
        * every request of verify_opensearch.py's matrix (plus aggregations
          and vector requests) gives the plugin node the stock node's hits,
          scores (1e-5), totals and aggregations, and the plugin's stats show
          it ran natively on every shard (or fell back for the stated reason);
        * the same after restoring the 2.19 snapshot into both nodes;
        * a force merge rewrites every segment as Lucene 10.5.0 on both nodes,
          and the answers still agree, still natively.

Standard library only; reuses verify_opensearch.py's matrix and comparison.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_opensearch as vo  # noqa: E402

OLD_LUCENE = "9.12."
NEW_LUCENE = "10.5.0"
REPO = "upgrade_repo"
SNAPSHOT = "from-2x"
DIM = 8
# index -> shards. `knn` carries the k-NN plugin's own codec (KNN9120Codec),
# which the Rust reader does not know: it falls back as a whole, before and
# after the upgrade, and is here to prove that fallback is clean and correct.
INDICES = {"single": 1, "multi": 3, "nest": 1, "knn": 1}


def on(base):
    vo.BASE = base.rstrip("/")


def vec(r):
    return [round(r.uniform(-1, 1), 4) for _ in range(DIM)]


def add_vectors(index, docs, seed):
    """`docs` more documents with an `emb` vector (and text, tag and numbers
    the matrix queries), in two refreshes."""
    r = vo.random.Random(seed)
    for half in (0, 1):
        lines = []
        for i in range(half * docs // 2, (half + 1) * docs // 2):
            lines.append(json.dumps({"index": {"_index": index, "_id": f"v{i}"}}))
            lines.append(json.dumps({
                "body": " ".join(vo.word(r) for _ in range(r.randint(1, 20))),
                "title": vo.word(r), "tag": vo.word(r), "n": 1_000_000 + i,
                "price": round(r.uniform(-50, 500), 2), "qty": r.randint(0, 40),
                "emb": vec(r)}))
        res = vo.req("POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
        vo.check(not res["errors"], f"{index}: vector bulk has no errors")
        vo.req("POST", f"/{index}/_refresh")


def create_knn():
    vo.req("PUT", "/knn", {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1,
                     "index.knn": True},
        "mappings": {"properties": {
            "body": {"type": "text"},
            "tag": {"type": "keyword"},
            "n": {"type": "long"},
            "emb": {"type": "knn_vector", "dimension": DIM,
                    "method": {"name": "hnsw", "engine": "lucene", "space_type": "l2"}},
        }},
    })


def load(base, docs):
    on(base)
    vo.wait_up()
    for index in ("single", "multi"):
        vo.create(index, INDICES[index])
        vo.req("PUT", f"/{index}/_mapping", {"properties": {"emb": {"type": "knn_vector", "dimension": DIM}}})
        vo.load(index, docs, 11 + INDICES[index])
        add_vectors(index, 400, 21 + INDICES[index])
    vo.create_nested("nest", 1)
    vo.load_nested("nest", 1500, 31)
    create_knn()
    add_vectors("knn", 1000, 41)
    vo.req("PUT", f"/_snapshot/{REPO}", {"type": "fs", "settings": {"location": "/usr/share/opensearch/snap"}})
    snap = vo.req("PUT", f"/_snapshot/{REPO}/{SNAPSHOT}?wait_for_completion=true",
                  {"indices": ",".join(INDICES), "include_global_state": False})
    vo.check(snap["snapshot"]["state"] == "SUCCESS", f"snapshot on 2.x: {snap['snapshot']['state']}")
    vo.req("POST", "/_flush")
    for index in INDICES:
        versions = segment_versions(index)
        vo.check(versions and all(v.startswith(OLD_LUCENE) for v in versions),
                 f"2.x node: {index} segments are Lucene {OLD_LUCENE}x: {sorted(versions)}")
    node = vo.req("GET", "/")
    counts = ", ".join("%s %d docs" % (i, vo.req("GET", f"/{i}/_count")["count"]) for i in INDICES)
    print(f"load: OpenSearch {node['version']['number']} (Lucene {node['version']['lucene_version']}), "
          f"{counts}; snapshot {SNAPSHOT}")


def segment_versions(index):
    return {s["version"] for s in vo.req("GET", f"/_cat/segments/{index}?format=json")}


def vector_rows(knn_index):
    """Vector requests: none is a native shape (the plugin encodes no vector
    query), so each falls back with its reason and must still agree."""
    r = vo.random.Random(99)
    target = vec(r)
    rows = [("script knn", {"size": 10, "query": {"script_score": {
        "query": {"exists": {"field": "emb"}},
        "script": {"source": "knn_score", "lang": "knn",
                   "params": {"field": "emb", "query_value": target, "space_type": "l2"}}}}}, "query_*")]
    if knn_index:
        rows.append(("knn query", {"size": 10, "query": {"knn": {"emb": {"vector": target, "k": 10}}}}, "query_*"))
        rows.append(("knn filtered", {"size": 10, "query": {"knn": {"emb": {"vector": target, "k": 10,
                                                                             "filter": {"term": {"tag": "alpha"}}}}}},
                     "query_*"))
    return rows


def fell_back(delta, expect):
    """Fallbacks in `delta` for any reason `expect` allows: `a|b`, `query_*`."""
    n = 0
    for reason, count in delta.items():
        for want in expect.split("|"):
            if reason == want or (want.endswith("*") and reason.startswith(want[:-1])):
                n += count
                break
    return n


def run_rows(index, rows, label, plugin, stock, shards, expect_override=None):
    """Each row on the stock node (the reference) and on the plugin node: same
    answer, and the plugin's counters show where it ran."""
    on(stock)
    reference = {}
    for name, body, _ in rows:
        url, b = vo.search_url(index, body)
        try:
            reference[name] = vo.shape(vo.req("POST", url, b), b)
        except RuntimeError as e:
            vo.UNVERIFIABLE.append(f"{label} {index} [{name}]")
            print(f"UNVERIFIABLE: {label} {index} [{name}]: the stock node failed: {str(e)[:300]}")
    on(plugin)
    native = 0
    for name, body, expect in rows:
        if name not in reference:
            continue
        expect = expect_override or expect
        if expect == "dfs_multi":
            expect = "dfs" if shards > 1 else "native"
        before = vo.stats()
        url, b = vo.search_url(index, body)
        got = vo.shape(vo.req("POST", url, b), b)
        after = vo.stats()
        diff = vo.same(got, reference[name])
        vo.check(diff is None, f"{label} {index} [{name}]: the plugin node differs from the stock node: {diff}")
        ran = after["native_queries"] - before["native_queries"]
        errors = after["native_errors"] - before["native_errors"]
        delta = vo.fallback_delta(before, after)
        vo.check(errors == 0, f"{label} {index} [{name}]: {errors} native errors")
        if expect == "native":
            vo.check(ran == shards, f"{label} {index} [{name}]: ran native on {ran} of {shards} shards; fallbacks {delta}")
            native += ran
        elif expect.startswith("native|"):
            # Native, or the one fallback the shard's shape may legitimately force.
            vo.check(ran + fell_back(delta, expect[len("native|"):]) == shards,
                     f"{label} {index} [{name}]: expected native or '{expect}' on {shards} shards, got native={ran} fallbacks={delta}")
            native += ran
        else:
            vo.check(ran == 0 and fell_back(delta, expect) == shards,
                     f"{label} {index} [{name}]: expected fallback '{expect}' on {shards} shards, got native={ran} fallbacks={delta}")
    return native


def serve_all(label, plugin, stock, prefix="", merged=False):
    native = 0
    for base, shards in (("single", 1), ("multi", 3)):
        index = prefix + base
        native += run_rows(index, vo.matrix() + vector_rows(False), label, plugin, stock, shards)
    # Once a force merge has dropped every deletion, OpenSearch answers a term's total without
    # counting (shortcutTotalHitCount), which leaves a count limit of `size`: past it Lucene's
    # comparator skips, which a nested key does not natively, so such a sort falls back.
    native += run_rows(prefix + "nest", vo.nested_rows(), label, plugin, stock, 1,
                       expect_override="native|sort_nested" if merged else None)
    # The k-NN plugin's codec: the whole reader falls back, text queries included.
    text = [(n, b, e) for n, b, e in vo.matrix()[:8]]
    native += run_rows(prefix + "knn", text, label, plugin, stock, 1, expect_override="native_open_failed")
    native += run_rows(prefix + "knn", vector_rows(True), label, plugin, stock, 1,
                       expect_override="native_open_failed|query_*")
    print(f"{label}: {native} shard queries ran native")
    return native


def check_versions(label, bases, want, indices):
    for base in bases:
        on(base)
        for index in indices:
            v = segment_versions(index)
            ok = v and all(x.startswith(want) for x in v)
            vo.check(ok, f"{label} {base} {index}: segments should all be Lucene {want}*, are {sorted(v)}")


def green(base, indices):
    on(base)
    vo.wait_up()
    h = vo.req("GET", f"/_cluster/health/{','.join(indices)}?wait_for_status=green&timeout=60s")
    vo.check(h["status"] == "green", f"{base}: {','.join(indices)} are {h['status']}")


def verify(plugin, stock):
    for base in (plugin, stock):
        green(base, list(INDICES))
        on(base)
        created = vo.req("GET", "/single/_settings/index.version.created?flat_settings=true")
        print(f"{base}: OpenSearch {vo.req('GET', '/')['version']['number']}, single created by version id "
              f"{created['single']['settings']['index.version.created']}")
    for base in (plugin, stock):
        on(base)
        vo.check(vo.req("GET", "/_cat/indices?format=json") is not None, f"{base}: _cat/indices")
    check_versions("in place", (plugin, stock), OLD_LUCENE, INDICES)
    total = serve_all("in-place upgrade", plugin, stock)

    # The 2.x snapshot, restored next to the in-place copy on both nodes.
    for base in (plugin, stock):
        on(base)
        vo.req("PUT", f"/_snapshot/{REPO}", {"type": "fs", "settings": {"location": "/usr/share/opensearch/snap"}})
        vo.req("POST", f"/_snapshot/{REPO}/{SNAPSHOT}/_restore?wait_for_completion=true",
               {"indices": ",".join(INDICES), "rename_pattern": "(.+)", "rename_replacement": "restored_$1",
                "include_global_state": False})
        green(base, [f"restored_{i}" for i in INDICES])
    check_versions("restored", (plugin, stock), OLD_LUCENE, [f"restored_{i}" for i in INDICES])
    total += serve_all("snapshot restore", plugin, stock, "restored_")

    # A force merge on both nodes: Lucene 10.5.0 rewrites every segment.
    for base in (plugin, stock):
        on(base)
        for index in INDICES:
            vo.req("POST", f"/{index}/_forcemerge?max_num_segments=1")
            vo.req("POST", f"/{index}/_refresh")
    check_versions("force-merged", (plugin, stock), NEW_LUCENE, INDICES)
    total += serve_all("after force merge", plugin, stock, merged=True)

    on(plugin)
    s = vo.stats()
    print(f"plugin stats: {json.dumps(s, sort_keys=True)}")
    print(f"verify_upgrade: {vo.CHECKS[0]} checks, {len(vo.FAILURES)} failures, {len(vo.UNVERIFIABLE)} rows "
          f"the stock node failed; {total} shard queries ran native")
    return total


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "load":
        docs = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
        load(sys.argv[2], docs)
    elif len(sys.argv) == 4 and sys.argv[1] == "verify":
        if verify(sys.argv[2], sys.argv[3]) == 0:
            vo.check(False, "nothing ran native")
    else:
        sys.exit("usage: verify_upgrade.py load <base> [docs] | verify <plugin-base> <stock-base>")
    if vo.FAILURES:
        print(f"{len(vo.FAILURES)} failures", file=sys.stderr)
    sys.exit(1 if vo.FAILURES else 0)


if __name__ == "__main__":
    main()
