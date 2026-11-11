#!/usr/bin/env python3
"""pattern_text fields through the lucene-rust plugin.

Starts one node of the pinned OpenSearch image with the plugin and checks a
`pattern_text` field on an index the Rust engine writes and on one
OpenSearch's own engine writes (both with index.derived_source.enabled, two
flushed segments, some documents deleted), against a `text` field holding
the same messages:

  phrase   `match_phrase` (with and without slop) and `match_phrase_prefix`
           hit exactly the documents a positional phrase on `text` hits,
           each scoring 0 as on `match_only_text`; `match` hits as on text.
  native   every exact phrase on the Rust engine ran in Rust (the plugin's
           stats count no fallback for them).
  source   `_source` rebuilds every message exactly, from the template doc
           value and the stored variables.
  errors   mappings the field refuses (positions, norms, index_phrases) and
           a document with two values, each a 400.
  size     the store size of each index after a force merge, printed.

  scripts/check-pattern-text-plugin.py [--keep]

Needs Docker and the plugin zip in opensearch-plugin/build/distributions
(`gradle -p opensearch-plugin bundlePlugin`).
"""
import glob, json, os, shutil, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "3.9.0"
NAME, PORT = "pattern-text-check-rust", 19217
DOCS = 3000

PHRASES = [
    "connection closed by", "failed password for root", "failed password",
    "instruction cache parity error corrected", "parity error", "closed by peer",
    "receiving block", "spaced message", "password for invalid user",
    "no such phrase anywhere", "user admin0 from", "admin1 from", "port 40009",
    "connection 3 closed", "connection 3 closed by peer", "closed by 10.0.1.1",
]


def message(i):
    ip = f"10.0.{i % 7}.{i % 13}"
    k = i % 8
    if k == 0:
        return f"Dec 10 06:55:{i % 60:02d} sshd[{24000 + i}]: Connection closed by {ip} [preauth]"
    if k == 1:
        return f"Failed password for root from {ip} port {40000 + i} ssh2"
    if k == 2:
        return f"Failed password for invalid user admin{i % 3} from {ip}"
    if k == 3:
        return "instruction cache parity error corrected"
    if k == 4:
        return "instruction cache parity error"
    if k == 5:
        return f"connection {i % 5} closed by peer"
    if k == 6:
        return f"Receiving block blk_{i * 31} src: /{ip}:50010"
    return f"  spaced   message {i}  "


def req(method, path, body=None, ndjson=False, timeout=120):
    data = None
    if body is not None:
        data = body.encode() if ndjson else json.dumps(body).encode()
    r = urllib.request.Request(f"http://localhost:{PORT}{path}", data=data, method=method,
                               headers={"Content-Type": "application/x-ndjson" if ndjson else "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def must(method, path, body=None, ndjson=False):
    code, r = req(method, path, body, ndjson)
    if code not in (200, 201):
        raise RuntimeError(f"{method} {path}: {code} {str(r)[:2000]}")
    return r


def start(image, heap="1g"):
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", NAME, "-p", f"{PORT}:9200",
                    "-e", "discovery.type=single-node", "-e", "DISABLE_SECURITY_PLUGIN=true",
                    "-e", "DISABLE_INSTALL_DEMO_CONFIG=true", "-e", f"OPENSEARCH_JAVA_OPTS=-Xms{heap} -Xmx{heap}",
                    "-e", "cluster.routing.allocation.disk.threshold_enabled=false", image],
                   check=True, capture_output=True)
    for _ in range(150):
        try:
            if must("GET", "/_cluster/health")["status"] in ("green", "yellow"):
                return
        except Exception:
            pass
        time.sleep(2)
    sys.exit("node did not start")


def build_image():
    """The pinned OpenSearch image with the built plugin installed."""
    zips = glob.glob(os.path.join(ROOT, "opensearch-plugin", "build", "distributions", "lucene-rust-*.zip"))
    if not zips:
        sys.exit("no plugin zip: run gradle -p opensearch-plugin bundlePlugin")
    docker_dir = os.path.join(ROOT, "opensearch-plugin", "docker")
    for z in glob.glob(os.path.join(docker_dir, "*.zip")):
        os.remove(z)
    shutil.copy(zips[0], docker_dir)
    image = f"lucene-rust-opensearch:{VERSION}"
    subprocess.run(["docker", "build", "-q", "-t", image, docker_dir], check=True, capture_output=True)
    return image


def stop():
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)


def index(name, engine, field_type, docs, deleted):
    settings = {"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": -1,
                "index.derived_source.enabled": True}
    if engine:
        settings["index.lucene_rust.engine"] = True
    must("PUT", f"/{name}", {"settings": settings, "mappings": {"properties": {
        "message": {"type": field_type}, "n": {"type": "long"}}}})
    half = len(docs) // 2
    for part in (docs[:half], docs[half:]):
        lines = []
        for i, m in part:
            lines.append(json.dumps({"index": {"_id": str(i)}}))
            lines.append(json.dumps({"message": m, "n": i}))
        r = must("POST", f"/{name}/_bulk", "\n".join(lines) + "\n", ndjson=True)
        if r["errors"]:
            raise RuntimeError(f"bulk into {name}: {json.dumps(r)[:1000]}")
        must("POST", f"/{name}/_flush")
    for i in deleted:
        must("DELETE", f"/{name}/_doc/{i}")
    must("POST", f"/{name}/_refresh")


def hits(name, query):
    r = must("POST", f"/{name}/_search", {"query": query, "size": 10000, "_source": False})
    return {h["_id"]: h["_score"] for h in r["hits"]["hits"]}


def main():
    keep = "--keep" in sys.argv
    start(build_image())
    failures = []
    docs = [(i, message(i)) for i in range(DOCS)]
    deleted = list(range(5, DOCS, 97))
    try:
        index("text", True, "text", docs, deleted)
        for name, engine in (("rust", True), ("lucene", False)):
            index(name, engine, "pattern_text", docs, deleted)
        before = must("GET", "/_plugins/lucene_rust/stats")
        for name in ("rust", "lucene"):
            for p in PHRASES:
                for slop in (0, 1):
                    q = {"match_phrase": {"message": {"query": p, "slop": slop}}}
                    want = hits("text", q)
                    pre = must("GET", "/_plugins/lucene_rust/stats")
                    got = hits(name, q)
                    post = must("GET", "/_plugins/lucene_rust/stats")
                    if name == "rust" and slop == 0:
                        new = {k: v for k, v in post.get("fallbacks", {}).items() if v != pre.get("fallbacks", {}).get(k, 0)}
                        if post["native_queries"] - pre["native_queries"] != 1 or new:
                            failures.append(f"exact phrase {p!r} did not run in Rust: fallbacks {new}")
                    if set(got) != set(want):
                        failures.append(f"{name} {p!r}~{slop}: {len(got)} hits, text {len(want)}")
                    if any(s != 0 for s in got.values()):
                        failures.append(f"{name} {p!r}~{slop}: a phrase hit scored above 0")
                q = {"match_phrase_prefix": {"message": {"query": p[:-1], "max_expansions": 10000}}}
                if set(hits(name, q)) != set(hits("text", q)):
                    failures.append(f"{name} prefix {p[:-1]!r}: differs from text")
            for word in ("failed", "parity", "40009", "peer"):
                q = {"match": {"message": word}}
                if set(hits(name, q)) != set(hits("text", q)):
                    failures.append(f"{name} match {word!r}: differs from text")
        after = must("GET", "/_plugins/lucene_rust/stats")
        fallbacks = {k: v - before.get("fallbacks", {}).get(k, 0) for k, v in after.get("fallbacks", {}).items()
                     if v != before.get("fallbacks", {}).get(k, 0)}
        print(f"plugin: {after['native_queries'] - before['native_queries']} shard queries in Rust, fallbacks {fallbacks}")
        if any(k.startswith("pattern_phrase_") and k not in ("pattern_phrase_slop", "pattern_phrase_prefix") for k in fallbacks):
            failures.append(f"phrases fell back to Java for another reason: {fallbacks}")

        # _source, rebuilt
        for name in ("rust", "lucene"):
            r = must("POST", f"/{name}/_search", {"query": {"match_all": {}}, "size": 10000})
            got = {h["_id"]: h["_source"].get("message") for h in r["hits"]["hits"]}
            want = {str(i): m for i, m in docs if i not in deleted}
            bad = [k for k in want if got.get(k) != want[k]]
            if bad or len(got) != len(want):
                failures.append(f"{name}: {len(bad)} messages not rebuilt, e.g. {bad[:1]} -> {got.get(bad[0]) if bad else None!r}")

        # a document is visible to a realtime get before a refresh, its message rebuilt
        for name in ("rust", "lucene"):
            must("PUT", f"/{name}/_doc/rt1", {"message": "late line 42 arrived", "n": -1})
            src = must("GET", f"/{name}/_doc/rt1")["_source"]
            if src.get("message") != "late line 42 arrived":
                failures.append(f"{name}: realtime get returned {src}")

        # mappings the field takes: its round trip, match_only_text's parameters, index: false, copy_to
        mapping = must("GET", "/rust/_mapping")["rust"]["mappings"]
        must("PUT", "/copy", {"settings": {"index.lucene_rust.engine": True, "index.derived_source.enabled": True},
                              "mappings": mapping})
        must("PUT", "/copy/_mapping", mapping)
        must("PUT", "/params", {"settings": {"index.lucene_rust.engine": True}, "mappings": {"properties": {"m": {
            "type": "pattern_text", "position_increment_gap": 10, "fielddata": True, "index_prefixes": None,
            "index_options": "docs", "norms": False, "fields": {"raw": {"type": "keyword"}}}}}})
        must("PUT", "/unindexed", {"settings": {"index.lucene_rust.engine": True, "index.derived_source.enabled": True},
                                   "mappings": {"properties": {"m": {"type": "pattern_text", "index": False}}}})
        must("PUT", "/unindexed/_doc/1?refresh=true", {"m": "kept 7 times"})
        if must("GET", "/unindexed/_doc/1")["_source"].get("m") != "kept 7 times":
            failures.append("index: false did not keep the message")
        must("PUT", "/copied", {"settings": {"index.lucene_rust.engine": True}, "mappings": {"properties": {
            "a": {"type": "keyword", "copy_to": "m"}, "m": {"type": "pattern_text"}}}})

        # errors
        bad = [
            ("PUT", "/copied/_doc/1", {"a": "x", "m": "y 1"}, "one value per document"),
            ("PUT", "/bad4", {"mappings": {"properties": {"m": {"type": "pattern_text", "fields": {"template": {"type": "keyword"}}}}}}, "multi-field"),
            ("POST", "/unindexed/_search", {"query": {"match_phrase": {"m": "kept times"}}}, "not indexed"),
            ("PUT", "/bad1", {"mappings": {"properties": {"m": {"type": "pattern_text", "index_options": "positions"}}}}, "index_options"),
            ("PUT", "/bad2", {"mappings": {"properties": {"m": {"type": "pattern_text", "norms": True}}}}, "norms"),
            ("PUT", "/bad3", {"mappings": {"properties": {"m": {"type": "pattern_text", "index_phrases": True}}}}, "index_phrases"),
            ("PUT", "/rust/_doc/x1", {"message": ["a 1", "b 2"]}, "one value per document"),
        ]
        for method, path, body, text in bad:
            code, r = req(method, path, body)
            if code != 400 or text not in str(r):
                failures.append(f"{method} {path}: expected a 400 naming {text!r}, got {code} {str(r)[:300]}")
        print(f"errors: {len(bad)} refusals checked")

        for name in ("text", "rust", "lucene"):
            must("POST", f"/{name}/_forcemerge?max_num_segments=1")
            size = must("GET", f"/{name}/_stats/store")["indices"][name]["primaries"]["store"]["size_in_bytes"]
            print(f"size {name}: {size} bytes")
    finally:
        if failures:
            logs = subprocess.run(["docker", "logs", NAME], capture_output=True, text=True)
            print("\n".join(l for l in (logs.stdout + logs.stderr).splitlines() if "WARN" in l or "ERROR" in l)[-5000:])
        if not keep:
            stop()
    for f in failures[:40]:
        print("FAIL:", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
