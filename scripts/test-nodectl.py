#!/usr/bin/env python3
"""Checks opensearch-plugin/e2e/nodectl.py against a fake OpenSearch distribution (a shell script
that serves HTTP and records its arguments), so the process manager is tested without a 1 GB
tarball or a JVM: variants, hard-linked trees, plugin keep lists, node lifecycle (run, kill, start,
stop, restart, rm), exec path translation, settings assembly and the library swap."""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

work = tempfile.mkdtemp(prefix="nodectl-test-")
FAKE = os.path.join(work, "fake")
os.makedirs(FAKE)


def write(path, text, mode=0o755):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, mode)


# bin/opensearch: records its arguments and the environment it was given, then serves HTTP on http.port.
write(f"{FAKE}/bin/opensearch", """#!/bin/bash
home=$(cd "$(dirname "$0")/.." && pwd)
port=9200; data=""
for a in "$@"; do case "$a" in http.port=*) port=${a#http.port=};; path.data=*) data=${a#path.data=};; esac; done
mkdir -p "$data"
printf '%s\\n' "$@" > "$data/args.txt"
printf 'JAVA_OPTS=%s\\nJAVA_HOME_SET=%s\\nEXTRA=%s\\n' "$OPENSEARCH_JAVA_OPTS" "${JAVA_HOME:-}" "${NODECTL_TEST_ENV:-}" > "$data/env.txt"
exec python3 -c "
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')
    def log_message(self, *a): pass
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
" "$port" org.opensearch.bootstrap.OpenSearch
""")
# the plugin tool: `install --batch file://ZIP` creates the plugin directory with a native library in it
write(f"{FAKE}/bin/opensearch-plugin", """#!/bin/bash
home=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$home/plugins/lucene-rust/native/linux-x86_64"
echo original > "$home/plugins/lucene-rust/native/linux-x86_64/liblucene_ffi.so"
""")
write(f"{FAKE}/bin/opensearch-keystore", """#!/bin/bash
home=$(cd "$(dirname "$0")/.." && pwd)
case "$1" in create) : > "$home/config/opensearch.keystore";; add) cat > "$home/config/key.txt";; esac
""")
for d in ("jdk/bin", "lib", "modules", "config/jvm.options.d"):
    os.makedirs(f"{FAKE}/{d}", exist_ok=True)
write(f"{FAKE}/lib/a.jar", "jar", 0o644)
for p in ("opensearch-knn", "opensearch-custom-codecs", "opensearch-neural-search"):
    write(f"{FAKE}/plugins/{p}/plugin.jar", "x", 0o644)
tgz = os.path.join(work, "fake.tar.gz")
subprocess.run(["tar", "czf", tgz, "-C", os.path.dirname(FAKE), "--transform", "s|^fake|opensearch-test|", "fake"], check=True)
zipf = os.path.join(work, "lucene-rust-test.zip")
write(zipf, "zip", 0o644)
libf = os.path.join(work, "new-lib.so")
write(libf, "replacement", 0o644)

os.environ.update(NODE_HOME=os.path.join(work, "home"), OPENSEARCH_TARBALL=tgz, OPENSEARCH_VERSION="3.9.0",
                  NODE_USER=os.environ.get("NODE_USER", "osbench"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "opensearch-plugin", "e2e"))
import nodectl  # noqa: E402

failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
        print("FAIL:", what)


def get(port):
    with urllib.request.urlopen(f"http://localhost:{port}/", timeout=5) as r:
        return r.status


def args_of(name):
    return open(os.path.join(nodectl.data_dir(name), "args.txt")).read().split("\n")


try:
    # variants
    stock = nodectl.dist("stock")
    check(sorted(os.listdir(f"{stock}/plugins")) == ["opensearch-custom-codecs", "opensearch-knn", "opensearch-neural-search"],
          "stock keeps every bundled plugin")
    mn = nodectl.dist("stock-min")
    check(os.listdir(f"{mn}/plugins") == [], "stock-min has no bundled plugin")
    plugin = nodectl.dist("plugin", zips=[zipf])
    check(sorted(os.listdir(f"{plugin}/plugins")) == ["lucene-rust", "opensearch-custom-codecs"], "plugin keeps custom-codecs and installs the zip")
    check(open(f"{plugin}/config/jvm.options.d/lucene-rust.options").read().strip() == "--enable-native-access=ALL-UNNAMED", "native access option written")
    check(os.path.isfile(f"{plugin}/config/key.txt"), "encrypted-store test key added to the keystore")
    check(os.stat(f"{plugin}/lib/a.jar").st_ino == os.stat(f"{nodectl.pristine()}/lib/a.jar").st_ino, "lib is hard-linked, not copied")
    marker = os.path.join(plugin, "marker")
    open(marker, "w").close()
    check(nodectl.dist("plugin", zips=[zipf]) == plugin and os.path.isfile(marker), "an unchanged variant is not rebuilt")
    knn = nodectl.dist("plugin-knn", zips=[zipf])
    check("opensearch-knn" in os.listdir(f"{knn}/plugins"), "plugin-knn keeps opensearch-knn")
    old = nodectl.dist("old", keep=["opensearch-knn"], store_key=False)
    check(os.listdir(f"{old}/plugins") == ["opensearch-knn"] and nodectl.dist("old") == old, "a custom variant is reused without respecifying it")

    # lifecycle
    nodectl.run("n1", "plugin", 19200, settings=["indices.engine.accelerated.default=false"], heap="256m",
                env=["NODECTL_TEST_ENV=yes"], zips=[zipf], timeout=60)
    check(get(19200) == 200, "node answers on its port")
    pid = nodectl.pid("n1")
    check(pid is not None and nodectl.rss("n1") > 0, "pid and rss of a running node")
    a = args_of("n1")
    check("indices.engine.accelerated.default=false" in a and "node.name=n1" in a and "discovery.type=single-node" in a
          and "network.host=127.0.0.1" in a, f"settings reach the node: {a}")
    envs = open(os.path.join(nodectl.data_dir("n1"), "env.txt")).read()
    check("-Xms256m -Xmx256m" in envs and "EXTRA=yes" in envs, f"heap and env reach the node: {envs}")
    nodectl.kill("n1")
    check(nodectl.pid("n1") is None, "kill stops the node")
    nodectl.start("n1", timeout=60)
    check(get(19200) == 200 and nodectl.pid("n1") != pid, "start relaunches over the same state")
    nodectl.restart("n1")
    check(get(19200) == 200, "restart")
    out = nodectl.exec_("n1", "ls /usr/share/opensearch/data && echo $NODE_PID && echo $OPENSEARCH_HOME").stdout.split()
    check("args.txt" in out and str(nodectl.pid("n1")) in out and nodectl.home("n1") in out, f"exec translates container paths and sets NODE_PID: {out}")
    nodectl.stop("n1", timeout=10)
    check(nodectl.pid("n1") is None, "stop")
    nodectl.rm("n1")
    check("n1" not in nodectl.names(), "rm forgets the node")

    # a second node on another port, with a data directory outside the node's state, and the library swap
    data = os.path.join(work, "data2")
    nodectl.run("n2", "plugin", 19201, data=data, transport=19301, lib=libf, zips=[zipf], timeout=60)
    check("transport.port=19301" in args_of("n2"), "transport port")
    so = f"{nodectl.home('n2')}/plugins/lucene-rust/native/linux-x86_64/liblucene_ffi.so"
    check(open(so).read() == "replacement", "a library replaces the zip's")
    check(open(f"{plugin}/plugins/lucene-rust/native/linux-x86_64/liblucene_ffi.so").read().strip() == "original", "the variant itself is untouched by a library swap")
    nodectl.rm("n2")
    check(os.path.isdir(data), "rm keeps an external data directory")
finally:
    for n in nodectl.names():
        nodectl.rm(n)
    subprocess.run(["rm", "-rf", work])

if failures:
    print(f"{len(failures)} failures")
    sys.exit(1)
print("test-nodectl: ok")
