#!/usr/bin/env python3
"""lucene-engine knn_vector indexes moved between stock k-NN and the lucene-rust plugin, both ways.

Stock k-NN's lucene engine writes Lucene's own vector formats, and so does the
plugin; each must read what the other writes. Every lucene-engine mapping of
scripts/check-knn-plugin.py is indexed (three flushed segments, some deletes)
and searched on the writing node, and the answers are recorded. Then the other
node must answer the same queries over the same files, two ways:

  in place   the writing node stops and the other starts on its data directory;
  snapshot   the other node restores the writer's snapshot of every index.

Exact mappings must return the same hits and scores (1e-4). Quantized ones:
the plugin searches full precision, so where it reads it must return the
exhaustive scan's top 10, and the two nodes' top 10 must overlap by MIN_RECALL.

  stock -> plugin   stock writes; the plugin reads, then takes new documents
                    and deletes into the old index, force-merges it to one
                    segment and still returns the exhaustive scan's top k.
  plugin -> stock   the plugin writes, once with the Rust engine
                    (index.engine.accelerated) and once with OpenSearch's;
                    stock reads.

Every plugin search must run in Rust (its stats show no fallback).

Stock writes its indexes with derived source on (index.knn.derived_source.enabled,
its default): the stored _source leaves the vectors out, and the reader puts
them back. Every live document's _source must hold its vector wherever it is
read, through a search (the plugin's native fetch) and the multi-get API
(Lucene's codec). After the plugin's writes and merge into stock's indexes,
stock reads them again in place (stock -> plugin -> stock).

  scripts/check-knn-upgrade.py [--keep]

Needs Docker and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import glob, importlib.util, json, os, random, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("check_knn", os.path.join(ROOT, "scripts", "check-knn-plugin.py"))
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

NODE = ("knn-upgrade", 19223)
DATA, REPO = "knn-upgrade-data", "knn-upgrade-repo"
DELETED = 40
ADDED = 120
STOCK_IMAGE = f"opensearchproject/opensearch:{ck.VERSION}"


def start(image, *env):
    subprocess.run(["docker", "rm", "-f", NODE[0]], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", NODE[0], "-p", f"{NODE[1]}:9200",
                    "-v", f"{DATA}:/usr/share/opensearch/data", "-v", f"{REPO}:/usr/share/opensearch/repo",
                    "-e", "path.repo=/usr/share/opensearch/repo",
                    "-e", "discovery.type=single-node", "-e", "DISABLE_SECURITY_PLUGIN=true",
                    "-e", "DISABLE_INSTALL_DEMO_CONFIG=true", "-e", "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g",
                    "-e", "cluster.routing.allocation.disk.threshold_enabled=false",
                    *[a for e in env for a in ("-e", e)], image],
                   check=True, capture_output=True)
    ck.wait(NODE[1])


def stop():
    subprocess.run(["docker", "stop", "-t", "60", NODE[0]], check=True, capture_output=True)


def fresh_volumes():
    subprocess.run(["docker", "rm", "-f", NODE[0]], capture_output=True)
    for v in (DATA, REPO):
        subprocess.run(["docker", "volume", "rm", "-f", v], capture_output=True)
        subprocess.run(["docker", "volume", "create", v], check=True, capture_output=True)
    # The image runs as uid 1000; a new volume's root is root's.
    subprocess.run(["docker", "run", "--rm", "-u", "0", "-v", f"{REPO}:/repo", "--entrypoint", "chown",
                    STOCK_IMAGE, "1000:1000", "/repo"], check=True, capture_output=True)


def queries(t):
    k = 10
    return [
        ("top k", {"size": 50, "query": ck.knn(t, k=k)}),
        ("filtered", {"size": 50, "query": ck.knn(t, k=k, filter={"term": {"tag": "even"}})}),
        ("range filter", {"size": 50, "query": ck.knn(t, k=5, filter={"range": {"n": {"lt": 300}}})}),
        ("ef_search", {"size": 50, "query": ck.knn(t, k=k, method_parameters={"ef_search": 200})}),
        ("in bool", {"size": 20, "query": {"bool": {"must": [ck.knn(t, k=k)],
                                                    "filter": [{"term": {"tag": "odd"}}]}}}),
    ]


def lucene_engine(spec):
    """Stock k-NN writes Lucene's formats only for the lucene engine: its default is faiss, except at 4x."""
    return spec.get("engine") == "lucene" or spec.get("compression_level") == "4x"


def write(prefix, rust, plugin=True):
    """Every lucene-engine mapping indexed and searched on the running node; the recorded cases.

    The plugin resolves a mapping without an engine to lucene and persists that, so its indexes of
    every mapping are lucene ones a stock node reads."""
    cases = {}
    for group, specs in (("exact", ck.EXACT), ("quantized", ck.QUANTIZED)):
        for name, spec in specs.items():
            if not plugin and not lucene_engine(spec):
                continue
            index = prefix + name
            rng = random.Random(f"{ck.SEED}-{name}")
            docs = [{"_id": f"d{i}", "tag": "even" if i % 2 == 0 else "odd", "n": i, "emb": ck.vector_for(spec, rng)}
                    for i in range(ck.DOCS)]
            ck.create(NODE[1], index, spec, docs, rust)
            if not plugin:
                # Stock turns derived source on for every k-NN index: its _source leaves the vectors out.
                got = ck.req(NODE[1], "GET", f"/{index}/_settings/index.knn.derived_source.enabled?include_defaults=true")
                flat = json.dumps(got[index])
                assert '"enabled": "true"' in flat, f"{index}: stock wrote no derived source: {flat}"
            deleted = {f"d{i}" for i in rng.sample(range(ck.DOCS), DELETED)}
            lines = [json.dumps({"delete": {"_index": index, "_id": i}}) for i in sorted(deleted)]
            ck.req(NODE[1], "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True)
            ck.req(NODE[1], "POST", f"/{index}/_flush")
            ck.req(NODE[1], "POST", f"/{index}/_refresh")
            targets = [ck.vector_for(spec, rng) for _ in range(6)]
            recorded = [[ck.search(NODE[1], index, body) for _, body in queries(t)] for t in targets]
            cases[index] = (group, spec, [d for d in docs if d["_id"] not in deleted], targets, recorded, rng)
    return cases


def snapshot(name, indices):
    ck.req(NODE[1], "PUT", "/_snapshot/repo", {"type": "fs", "settings": {"location": "/usr/share/opensearch/repo"}})
    r = ck.req(NODE[1], "PUT", f"/_snapshot/repo/{name}?wait_for_completion=true",
               {"indices": ",".join(indices), "include_global_state": False}, timeout=600)
    assert r["snapshot"]["state"] == "SUCCESS", json.dumps(r)[:500]


def restore(name, failures, settings=None):
    ck.req(NODE[1], "PUT", "/_snapshot/repo", {"type": "fs", "settings": {"location": "/usr/share/opensearch/repo"}})
    body = {"rename_pattern": "(.+)", "rename_replacement": "restored-$1", "include_global_state": False}
    if settings:
        body["index_settings"] = settings
    r = ck.req(NODE[1], "POST", f"/_snapshot/repo/{name}/_restore?wait_for_completion=true", body,
               timeout=600, ok_errors=True)
    if "error" in r or r.get("snapshot", {}).get("shards", {}).get("failed", 0):
        failures.append(f"restore of {name} failed: {json.dumps(r)[:600]}")


def green(failures, what):
    health = ck.req(NODE[1], "GET", "/_cluster/health?wait_for_status=green&timeout=60s", ok_errors=True)
    if health.get("status") != "green":
        explain = ck.req(NODE[1], "GET", "/_cluster/allocation/explain", ok_errors=True)
        failures.append(f"{what}: cluster {health.get('status')}: {json.dumps(explain)[:800]}")


def read(cases, rename, plugin_reads, failures):
    """The recorded queries answered by the running node over index `rename + index`; the count."""
    checked = 0
    for index, (group, spec, alive, targets, recorded, _) in cases.items():
        overlaps = []
        for qi, t in enumerate(targets):
            for (label, body), want in zip(queries(t), recorded[qi]):
                r = ck.req(NODE[1], "POST", f"/{rename}{index}/_search", body, ok_errors=True)
                if "error" in r:
                    failures.append(f"{rename}{index} {label}: {json.dumps(r['error'])[:300]}")
                    continue
                got = [(h["_id"], h["_score"]) for h in r["hits"]["hits"]]
                checked += 1
                if group == "exact":
                    if not (ck.close(got, want) if ck.half_cosine(spec) else ck.same(got, want, ties=ck.ties(spec))):
                        failures.append(f"{rename}{index} {label} query {qi}: {got[:3]} vs writer's {want[:3]}")
                elif label == "top k":
                    if plugin_reads:
                        scan = ck.exact_top(spec, alive, t, 10)
                        if not same_top(got[:10], scan):
                            failures.append(f"{rename}{index}: top 10 is not the exhaustive scan's")
                    overlaps.append(len({i for i, _ in got[:10]} & {i for i, _ in want[:10]}) / 10)
        if overlaps:
            recall = sum(overlaps) / len(overlaps)
            print(f"  {rename}{index}: top 10 overlap the writer's by {recall:.3f}")
            if recall < ck.MIN_RECALL:
                failures.append(f"{rename}{index}: overlap with the writer's {recall:.3f} < {ck.MIN_RECALL}")
    return checked


def sources(docs_by_index, rename, failures):
    """Every live document's _source vector, by search and by multi-get, against what was indexed."""
    checked = 0
    for index, docs in docs_by_index.items():
        want = {d["_id"]: d["emb"] for d in docs}
        r = ck.req(NODE[1], "POST", f"/{rename}{index}/_search",
                   {"size": len(want) + 10, "query": {"match_all": {}}}, ok_errors=True)
        got = {h["_id"]: h["_source"].get("emb") for h in r.get("hits", {}).get("hits", [])}
        m = ck.req(NODE[1], "POST", f"/{rename}{index}/_mget", {"ids": sorted(want)}, ok_errors=True)
        by_get = {d["_id"]: d.get("_source", {}).get("emb") for d in m.get("docs", []) if d.get("found")}
        for how, seen in (("search", got), ("mget", by_get)):
            if set(seen) != set(want):
                failures.append(f"{rename}{index} {how}: {len(seen)} documents, {len(want)} live")
                continue
            # A half_float field's derived source is its half floats, as stock's: within a half-float
            # step of this script's, whose normalization rounds differently from VectorUtil's.
            spec = spec_of(index)
            bad = [i for i, v in want.items()
                   if not same_vector(seen[i], v) and not same_vector(seen[i], ck.indexed(spec, v), 1e-3)]
            checked += len(seen)
            if bad:
                failures.append(f"{rename}{index} {how}: _source of {bad[:3]} is {seen[bad[0]]!r:.80}, "
                                f"indexed {want[bad[0]]!r:.80}")
    return checked


def spec_of(index):
    """The mapping case an index (a prefix and the case's name) was created from."""
    specs = {**ck.EXACT, **ck.QUANTIZED}
    return specs[max((n for n in specs if index.endswith(n)), key=len)]


def same_vector(got, want, tol=1e-6):
    if not isinstance(got, list) or len(got) != len(want):
        return False
    return all(isinstance(g, (int, float)) and abs(g - w) <= tol * max(1.0, abs(w)) for g, w in zip(got, want))


def same_top(got, scan, eps=1e-5):
    """`got` (id, score) is the exhaustive `scan` (score, id), letting equal scores trade places."""
    if len(got) < len(scan):
        return False
    want = {i for _, i in scan}
    cutoff = scan[-1][0]
    for (i, score), (s, _) in zip(got, scan):
        if abs(score - s) > eps * max(1.0, abs(s)):
            return False
        if i not in want and abs(score - cutoff) > eps * max(1.0, abs(cutoff)):
            return False
    return True


def write_into(cases, failures, live, rename=""):
    """New documents and deletes into each old index, one merged segment, the exhaustive top k;
    each index's live documents into `live`."""
    checked = 0
    for name, (group, spec, alive, targets, _, rng) in cases.items():
        index = rename + name
        rng = random.Random(f"{rng.random()}-{rename}")
        added = [{"_id": f"a{i}", "tag": "even" if i % 2 == 0 else "odd", "n": i, "emb": ck.vector_for(spec, rng)}
                 for i in range(ADDED)]
        gone = {d["_id"] for d in rng.sample(alive, 20)}
        lines = []
        for d in added:
            lines.append(json.dumps({"index": {"_index": index, "_id": d["_id"]}}))
            lines.append(json.dumps({"tag": d["tag"], "n": d["n"], "emb": d["emb"]}))
        lines += [json.dumps({"delete": {"_index": index, "_id": i}}) for i in sorted(gone)]
        r = ck.req(NODE[1], "POST", "/_bulk", "\n".join(lines) + "\n", ndjson=True, ok_errors=True)
        if r.get("errors") or "error" in r:
            failures.append(f"{index}: writes refused: {json.dumps(r)[:400]}")
            continue
        ck.req(NODE[1], "POST", f"/{index}/_forcemerge?max_num_segments=1", timeout=600)
        ck.req(NODE[1], "POST", f"/{index}/_refresh")
        now = [d for d in alive + added if d["_id"] not in gone]
        live[index] = now
        for t in targets[:3]:
            # A beam wide enough that the merged graph's search is exhaustive.
            got = ck.search(NODE[1], index, {"size": 10, "query": ck.knn(
                t, k=10, method_parameters={"ef_search": len(now)})})
            scan = ck.exact_top(spec, now, t, 10)
            checked += 1
            if not same_top(got, scan):
                failures.append(f"{index} after writes and merge: {[i for i, _ in got]} vs scan {[i for _, i in scan]}")
    return checked


def fallbacks_since(before):
    after = ck.req(NODE[1], "GET", "/_plugins/lucene_rust/stats")
    return after["accelerated_queries"] - before["accelerated_queries"], {
        k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
        if v != before.get("fallbacks", {}).get(k, 0)}


def main():
    keep = "--keep" in sys.argv
    zips = glob.glob(os.path.join(ROOT, "opensearch-plugin", "build", "distributions", "lucene-rust-*.zip"))
    if not zips:
        sys.exit("no plugin zip: run gradle -p opensearch-plugin bundlePlugin")
    docker_dir = os.path.join(ROOT, "opensearch-plugin", "docker")
    for z in glob.glob(os.path.join(docker_dir, "*.zip")):
        os.remove(z)
    shutil.copy(zips[0], docker_dir)
    image = f"lucene-rust-opensearch:{ck.VERSION}"
    subprocess.run(["docker", "build", "-q", "-t", image, docker_dir], check=True, capture_output=True)

    failures = []
    try:
        print("stock -> plugin")
        fresh_volumes()
        start(STOCK_IMAGE)
        cases = write("", False, plugin=False)
        snapshot("from-stock", list(cases))
        stop()
        # The stock indexes in place inherit the Rust engine; the restored ones take OpenSearch's.
        start(image, "indices.engine.accelerated.default=true")
        before = ck.req(NODE[1], "GET", "/_plugins/lucene_rust/stats")
        green(failures, "stock indexes in place")
        n = read(cases, "", True, failures)
        restore("from-stock", failures, {"index.engine.accelerated": False})
        green(failures, "stock snapshot restored")
        n += read(cases, "restored-", True, failures)
        alive = {index: c[2] for index, c in cases.items()}
        m = sources(alive, "", failures) + sources(alive, "restored-", failures)
        live = {}
        n += write_into(cases, failures, live)
        n += write_into(cases, failures, live, "restored-")
        m += sources(live, "", failures)
        ran, fell = fallbacks_since(before)
        after = ck.req(NODE[1], "GET", "/_plugins/lucene_rust/stats")
        engines = after["engines"]
        fetched = after["fetch_nanos"]["accelerated_count"] - before["fetch_nanos"]["accelerated_count"]
        errors = after["accelerated_errors"] - before["accelerated_errors"]
        print(f"  {n} queries compared, {m} sources; {ran} shard queries in Rust, fallbacks {fell}; "
              f"engines {engines}; {fetched} documents fetched natively, {errors} native errors")
        if fell:
            failures.append(f"stock -> plugin: queries fell back to Java: {fell}")
        if engines["accelerated"] < len(cases):
            failures.append(f"stock -> plugin: the {len(cases)} indexes in place are not on the accelerated engine: {engines}")
        if errors or fetched == 0:
            failures.append(f"stock -> plugin: native fetch read {fetched} documents with {errors} errors")
        stop()

        print("stock -> plugin -> stock")
        start(STOCK_IMAGE)
        green(failures, "plugin-merged stock indexes in place")
        print(f"  {sources(live, '', failures)} sources")
        stop()

        print("plugin -> stock")
        fresh_volumes()
        start(image)
        cases = {**write("rust-", True), **write("java-", False)}
        snapshot("from-plugin", list(cases))
        stop()
        start(STOCK_IMAGE)
        green(failures, "plugin indexes in place")
        n = read(cases, "", False, failures)
        restore("from-plugin", failures)
        green(failures, "plugin snapshot restored")
        n += read(cases, "restored-", False, failures)
        alive = {index: c[2] for index, c in cases.items()}
        m = sources(alive, "", failures) + sources(alive, "restored-", failures)
        print(f"  {n} queries compared, {m} sources")
    finally:
        if failures:
            logs = subprocess.run(["docker", "logs", NODE[0]], capture_output=True, text=True)
            print("\n".join(l for l in (logs.stdout + logs.stderr).splitlines()
                            if "WARN" in l or "ERROR" in l or "Exception" in l)[-6000:])
        if not keep:
            subprocess.run(["docker", "rm", "-f", NODE[0]], capture_output=True)
            for v in (DATA, REPO):
                subprocess.run(["docker", "volume", "rm", "-f", v], capture_output=True)
    for f in failures:
        print("FAIL:", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
