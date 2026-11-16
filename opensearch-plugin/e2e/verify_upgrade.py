#!/usr/bin/env python3
"""An index an OpenSearch 2.x node wrote, served
natively after the upgrade to OpenSearch 3.9.0 with the lucene-rust plugin.

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
      Two 3.9.0 nodes opened the 2.19 data directory in place (the plugin node
      the original, the stock node a copy of it). Checks, per index:

        * every index opens (green) and every segment is still a Lucene 9.12
          one -- the old format is what gets served;
        * every request of verify_opensearch.py's matrix (plus aggregations
          and vector requests) gives the plugin node the stock node's hits,
          scores (1e-5), totals and aggregations, and the plugin's stats show
          it ran natively on every shard (or fell back for the stated reason);
        * the same after restoring the 2.19 snapshot into both nodes;
        * the upgrade API (`_upgrade?only_ancient_segments=true`) on the restored
          copies rewrites each Lucene 9.12 segment on its own on both nodes,
          and the answers still agree;
        * a force merge rewrites every segment as Lucene 10.5.x on both nodes,
          and the answers still agree, still natively;
        * the 2.x-created indices are on the accelerated engine, as any index
          that inherits the node default.

  removed <stock-base> <record>
      A stock 3.9.0 node, the plugin removed, opens the plugin node's data
      directory: every index green with the documents `verify` recorded.

Standard library only; reuses verify_opensearch.py's matrix and comparison.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_opensearch as vo  # noqa: E402

OLD_LUCENE = "9.12."
NEW_LUCENE = "10.5."
# The snapshot repository directory (the nodes' path.repo), set by scripts/verify-opensearch-upgrade.sh.
SNAP_DIR = os.environ.get("UPGRADE_SNAP", "/usr/share/opensearch/snap")
REPO = "upgrade_repo"
SNAPSHOT = "from-2x"
DIM = 8
# index -> shards. `knn` carries the k-NN plugin's own codec (KNN9120Codec),
# which the Rust reader reads as its Lucene delegate, before and after the
# upgrade.
INDICES = {"single": 1, "multi": 3, "nest": 1, "knn": 1}


# 2.x creates these indices and knows no `LegacyBM25` similarity.
vo.SIMILARITIES = tuple(k for k in vo.SIMILARITIES if k != "legacy")


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
    vo.req("PUT", f"/_snapshot/{REPO}", {"type": "fs", "settings": {"location": SNAP_DIR}})
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


def segment_count(index):
    return len(vo.req("GET", f"/_cat/segments/{index}?format=json"))


def removed(base, record):
    """After the plugin is removed, a stock node opens the plugin node's data directory as it is:
    every index green, the same documents, every segment the Lucene 10.5 one the plugin's force
    merge wrote."""
    green(base, list(INDICES))
    with open(record) as f:
        counts = json.load(f)
    for index, count in counts.items():
        got = vo.req("GET", f"/{index}/_count")["count"]
        vo.check(got == count, f"plugin removed: {index} holds {got} documents, {count} before")
    check_versions("plugin removed", (base,), NEW_LUCENE, INDICES)
    print(f"plugin removed: {vo.CHECKS[0]} checks, {len(vo.FAILURES)} failures")


def segment_versions(index):
    return {s["version"] for s in vo.req("GET", f"/_cat/segments/{index}?format=json")}


def vector_rows(knn_index):
    """Vector requests: none is a native shape here (stock k-NN serves the
    script and the queries, so the plugin encodes none), so each falls back
    with its reason and must still agree."""
    r = vo.random.Random(99)
    target = vec(r)
    rows = [("script knn", {"size": 10, "query": {"script_score": {
        "query": {"exists": {"field": "emb"}},
        "script": {"source": "knn_score", "lang": "knn",
                   "params": {"field": "emb", "query_value": target, "space_type": "l2"}}}}},
             "script_score_script")]
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


def _children(source, path, key_field, flt):
    """The values of `key_field` on the `path` children that pass `flt`, in child order."""
    short = key_field.split(".", 1)[1]
    out = []
    for child in source.get(path, []):
        if flt is not None:
            if "term" in flt:
                (f, want), = flt["term"].items()
                if child.get(f.split(".", 1)[1]) != want:
                    continue
            elif "range" in flt:
                (f, bounds), = flt["range"].items()
                vals = child.get(f.split(".", 1)[1])
                vals = vals if isinstance(vals, list) else [] if vals is None else [vals]
                if not any(v >= bounds.get("gte", float("-inf")) for v in vals):
                    continue
        v = child.get(short)
        if v is None:
            continue
        out.extend(v if isinstance(v, list) else [v])
    return out


def _f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


def check_nested_oracle(index, name, body, label):
    """A nested sort whose reference is the answer computed from `_source`, not the stock node.

    On an index created by 2.x and then rewritten (`_upgrade`, force merge), stock returns top
    hits that are not the top (checked against the documents' own children; the same index read
    by stock with the plugin's native search off gives the same wrong answer, a 3.9-created index
    does not). The plugin's answer is the exact one (#392), so stock is not the reference.
    The answer is recomputed from each document's children:
        the first key's value per mode, filter and missing, then the keys after it; the plugin's hits
    and sort values must equal it exactly, ids included where the keys are unique."""
    first = body["sort"][0]
    (field, spec), = first.items()
    order, nested = spec["order"], spec.get("nested", {})
    mode = spec.get("mode", "min" if order == "asc" else "max")
    missing = spec.get("missing", "_last")
    rest = body["sort"][1:]
    all_hits = vo.req("POST", f"/{index}/_search", {"query": body.get("query", {"match_all": {}}), "size": 10000, "_source": ["props", "n"]})["hits"]["hits"]
    float32 = field.endswith(".f")
    rows = []
    for h in all_hits:
        vals = _children(h["_source"], "props", field, nested.get("filter"))
        if float32:
            vals = [_f32(v) for v in vals]
        if not vals:
            key = None
        elif mode == "min":
            key = min(vals)
        elif mode == "max":
            key = max(vals)
        else:
            total = 0.0
            for v in vals:  # in child order, uncompensated, as the comparator adds them
                total += v
            key = total if mode == "sum" else total / len(vals)
        rows.append((key, int(h["_source"]["n"]), h["_id"]))
    sign = 1 if order == "asc" else -1

    real = [r for r in rows if r[0] is not None or isinstance(missing, (int, float))]
    second = None
    if rest and isinstance(rest[0], dict) and "n" in rest[0]:
        spec2 = rest[0]["n"]
        second = -1 if (spec2["order"] if isinstance(spec2, dict) else spec2) == "desc" else 1
    real.sort(key=lambda r: (sign * (r[0] if r[0] is not None else missing), second * r[1] if second else 0))
    got = vo.req("POST", f"/{index}/_search", body)["hits"]["hits"]
    check_n = len(got)
    for i, h in enumerate(got[:check_n]):
        if i >= len(real):
            break
        want = real[i]
        value = want[0] if want[0] is not None else missing
        reported = h["sort"][0]
        same = _f32(reported) == _f32(value) if float32 else reported == value
        vo.check(same, f"{label} {index} [{name}]: hit {i} sort value {reported} is not {value} computed from _source")
        keys_equal = [r for r in real if r[0] == want[0]]
        if rest and isinstance(rest[0], dict) and "n" in rest[0]:
            vo.check(h["_source"]["n"] == want[1] if "_source" in h else h["sort"][1] == want[1],
                     f"{label} {index} [{name}]: hit {i} second key {h['sort'][1]} is not {want[1]}")
        elif len(keys_equal) == 1:
            vo.check(h["_id"] == want[2], f"{label} {index} [{name}]: hit {i} is {h['_id']}, _source says {want[2]}")



def run_rows(index, rows, label, plugin, stock, shards, expect_override=None):
    """Each row on the stock node (the reference) and on the plugin node: same
    answer, and the plugin's counters show where it ran."""
    on(stock)
    reference = {}
    for name, body, _ in rows:
        url, b = vo.search_url(index, body)
        try:
            reference[name] = vo.shape(vo.complete(vo.search(index, body)), b)
        except RuntimeError as e:
            vo.UNVERIFIABLE.append(f"{label} {index} [{name}]")
            print(f"UNVERIFIABLE: {label} {index} [{name}]: the stock node failed: {str(e)[:300]}")
    on(plugin)
    native = 0
    for name, body, expect in rows:
        if name not in reference:
            continue
        expect = expect_override or expect
        # The plugin node runs the default accelerated_shapes: fast.
        if expect == "slow":
            expect = "slower_shape"
        if expect == "slow_agg":
            expect = "aggregations"
        if expect == "aggregations":
            expect = "aggregation*"  # the reason names the type, or "aggregation_shape"
        # The plugin node runs the default exact approximation: a sorted range or match_all
        # OpenSearch approximates runs natively, a post_filter one still falls back.
        if expect == "approximate":
            expect = "native|approximate"
        before = vo.stats()
        url, b = vo.search_url(index, body)
        got = vo.shape(vo.search(index, body), b)
        after = vo.stats()
        diff = vo.same(got, reference[name])
        vo.check(diff is None, f"{label} {index} [{name}]: the plugin node differs from the stock node: {diff}")
        ran = after["accelerated_queries"] - before["accelerated_queries"]
        errors = after["accelerated_errors"] - before["accelerated_errors"]
        delta = vo.fallback_delta(before, after)
        vo.check(errors == 0, f"{label} {index} [{name}]: {errors} native errors")
        if expect == "either":
            # The two nodes' answers agree (checked above); where it ran is not pinned.
            native += ran
        elif expect == "native":
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
    # Rows that stay in Java anyway (an aggregation it cannot run) keep their own reason.
    nest = [(n, b, e) for n, b, e in vo.nested_rows()]
    if merged:
        # A sort that sums, averages or filters the children: stock is not its reference here.
        oracle = {"nested double sum", "nested double avg missing", "nested float sum",
                  "nested keyword filter", "nested then n"}
        sums = [r for r in nest if r[0] in oracle]
        nest = [r for r in nest if r not in sums]
        on(plugin)
        for name, body, _ in sums:
            before = vo.stats()
            check_nested_oracle(prefix + "nest", name, body, label)
            ran = vo.stats()["accelerated_queries"] - before["accelerated_queries"]
            vo.check(ran >= 1, f"{label} {prefix}nest [{name}]: did not run native")
    native += run_rows(prefix + "nest", nest, label, plugin, stock, 1)
    # The k-NN plugin's codec reads natively; with the k-NN plugin installed, its own
    # vector queries stay in Java.
    text = [(n, b, e) for n, b, e in vo.matrix()[:8]]
    native += run_rows(prefix + "knn", text, label, plugin, stock, 1)
    native += run_rows(prefix + "knn", vector_rows(True), label, plugin, stock, 1)
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
        vo.req("PUT", f"/_snapshot/{REPO}", {"type": "fs", "settings": {"location": SNAP_DIR}})
        vo.req("POST", f"/_snapshot/{REPO}/{SNAPSHOT}/_restore?wait_for_completion=true",
               {"indices": ",".join(INDICES), "rename_pattern": "(.+)", "rename_replacement": "restored_$1",
                "include_global_state": False})
        green(base, [f"restored_{i}" for i in INDICES])
    check_versions("restored", (plugin, stock), OLD_LUCENE, [f"restored_{i}" for i in INDICES])
    total += serve_all("snapshot restore", plugin, stock, "restored_")

    # The upgrade API on the restored copies: each Lucene 9 segment rewritten on its own (no
    # merging down), on both nodes alike.
    restored = [f"restored_{i}" for i in INDICES]
    after = {}
    for base in (plugin, stock):
        on(base)
        res = vo.req("POST", f"/{','.join(restored)}/_upgrade?only_ancient_segments=true")
        vo.check(res["_shards"]["failed"] == 0, f"{base}: _upgrade failed on a shard: {res['_shards']}")
        vo.req("POST", f"/{','.join(restored)}/_refresh")
        after[base] = {i: segment_count(i) for i in restored}
    vo.check(after[plugin] == after[stock],
             f"an upgrade leaves the same segment counts on both nodes: plugin {after[plugin]}, stock {after[stock]}")
    check_versions("upgraded", (plugin, stock), NEW_LUCENE, restored)
    total += serve_all("after upgrade", plugin, stock, "restored_", merged=True)

    # A force merge on both nodes: Lucene 10.5.x rewrites every segment.
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
    # Indices created on 2.x inherit the accelerated engine default like any other.
    vo.check(s["engines"]["accelerated"] > 0, f"2.x-created indices move onto the accelerated engine: {s['engines']}")
    record = os.environ.get("UPGRADE_RECORD")
    if record:
        with open(record, "w") as f:
            json.dump({i: vo.req("GET", f"/{i}/_count")["count"] for i in INDICES}, f)
    print(f"verify_upgrade: {vo.CHECKS[0]} checks, {len(vo.FAILURES)} failures, {len(vo.UNVERIFIABLE)} rows "
          f"the stock node failed; {total} shard queries ran native")
    return total


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "load":
        docs = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
        load(sys.argv[2], docs)
    elif len(sys.argv) == 4 and sys.argv[1] == "removed":
        removed(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "verify":
        if verify(sys.argv[2], sys.argv[3]) == 0:
            vo.check(False, "nothing ran native")
    else:
        sys.exit("usage: verify_upgrade.py load <base> [docs] | verify <plugin-base> <stock-base> | removed <base> <record>")
    if vo.FAILURES:
        print(f"{len(vo.FAILURES)} failures", file=sys.stderr)
    sys.exit(1 if vo.FAILURES else 0)


if __name__ == "__main__":
    main()
