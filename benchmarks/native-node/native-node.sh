#!/bin/bash
# Run OpenSearch 3.9.0 (stock, or with the accelerated-search plugin) as plain processes, no Docker:
# for hosts without a container runtime (cloud sandboxes, CI boxes) and for A/B runs that need a
# quiet machine. Source it, then call the functions.
#
#   source benchmarks/native-node/native-node.sh
#   native_setup                       # once: download the tarball, build the plugin zip, install it
#   native_start stock  /data/stock    # or: native_start plugin /data/plugin [/path/other.so]
#   ... drive http://localhost:${PORT:-9250} with the e2e scripts ...
#   native_stop
#
# Environment (all optional):
#   NODE_HOME   work dir for the unpacked distributions (default $HOME/os-native)
#   PORT        HTTP port (9250, what the e2e scripts use)
#   HEAP        JVM heap (2g)
#   NODE_CPUS   taskset cpu list for the node (0-2); run the driver on another core
#   MEM_LIMIT   cgroup memory limit for the node, e.g. 4G: page cache counts against it, so it
#               is real memory pressure (needs root and a writable /sys/fs/cgroup/memory; cgroup v1)
#   OPENSEARCH_VERSION  (3.9.0)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OPENSEARCH_VERSION=${OPENSEARCH_VERSION:-3.9.0}
NODE_HOME=${NODE_HOME:-$HOME/os-native}
PORT=${PORT:-9250}; HEAP=${HEAP:-2g}; NODE_CPUS=${NODE_CPUS:-0-2}
U=http://localhost:$PORT
TARBALL_URL=https://artifacts.opensearch.org/releases/bundle/opensearch/$OPENSEARCH_VERSION/opensearch-$OPENSEARCH_VERSION-linux-x64.tar.gz

native_setup() {
  mkdir -p "$NODE_HOME"
  if [ ! -d "$NODE_HOME/stock" ]; then
    [ -f "$NODE_HOME/os.tgz" ] || curl -fsSL -o "$NODE_HOME/os.tgz" "$TARBALL_URL"
    mkdir -p "$NODE_HOME/stock.tmp" && tar xzf "$NODE_HOME/os.tgz" -C "$NODE_HOME/stock.tmp" --strip-components=1
    mv "$NODE_HOME/stock.tmp" "$NODE_HOME/stock"
  fi
  local d=$REPO/target/opensearch-dist/$OPENSEARCH_VERSION   # what opensearch-plugin/build.gradle compiles against
  if [ ! -d "$d/lib" ]; then
    mkdir -p "$d/modules"; cp -r "$NODE_HOME/stock/lib" "$d/lib"; cp -r "$NODE_HOME/stock/modules/lang-painless" "$d/modules/"
  fi
  # Gradle needs the bundled JDK 25 as a toolchain
  grep -qs "installations.paths" ~/.gradle/gradle.properties 2>/dev/null ||
    { mkdir -p ~/.gradle; echo "org.gradle.java.installations.paths=$NODE_HOME/stock/jdk" >> ~/.gradle/gradle.properties; }
  # the library file must be named liblucene_ffi.so inside the zip
  [ -n "${NATIVE_LIB:-}" ] && { mkdir -p "$NODE_HOME/lib"; cp "$NATIVE_LIB" "$NODE_HOME/lib/liblucene_ffi.so"; GRADLE_ARGS="-PnativeLib=$NODE_HOME/lib/liblucene_ffi.so ${GRADLE_ARGS:-}"; }
  (cd "$REPO" && gradle -p opensearch-plugin bundlePlugin -x torchShim ${GRADLE_ARGS:-})
  rm -rf "$NODE_HOME/plugin" && cp -rL "$NODE_HOME/stock" "$NODE_HOME/plugin"
  # as in opensearch-plugin/e2e/nodectl.py (variant `plugin`): the bundled plugins go (neural-search registers its own
  # QueryPhaseSearcher and the node allows one); KEEP_PLUGINS names the ones to keep
  local p; for p in "$NODE_HOME"/plugin/plugins/*; do
    case " ${KEEP_PLUGINS:-opensearch-custom-codecs} " in *" $(basename "$p") "*) ;; *) rm -rf "$p" ;; esac
  done
  "$NODE_HOME/plugin/bin/opensearch-plugin" install --batch "file://$(ls "$REPO"/opensearch-plugin/build/distributions/accelerated-search-0.1.0.zip)"
  echo "--enable-native-access=ALL-UNNAMED" > "$NODE_HOME/plugin/config/jvm.options.d/accelerated-search.options"
}

# native_start stock|plugin DATA_DIR [LIB.so]  (a library replaces the installed one: A/B of builds)
native_start() {
  local kind=$1 data=$2 so=${3:-}
  native_stop
  local home=$NODE_HOME/$kind; mkdir -p "$data"
  [ -n "$so" ] && cp "$so" "$home/plugins/accelerated-search/native/linux-x86_64/liblucene_ffi.so"
  # OpenSearch refuses to run as root: drop to a bench user (created on first use)
  local drop=""
  if [ "$(id -u)" = 0 ]; then
    id osbench >/dev/null 2>&1 || useradd -m osbench
    chown -R osbench "$NODE_HOME" "$data"; drop="setpriv --reuid=osbench --regid=osbench --init-groups"
  fi
  local sec=""; [ -d "$home/plugins/opensearch-security" ] && sec="-E plugins.security.disabled=true"
  local pre=(taskset -c "$NODE_CPUS")
  [ -n "$drop" ] && pre=($drop taskset -c "$NODE_CPUS")
  if [ -n "${MEM_LIMIT:-}" ]; then
    local cg=/sys/fs/cgroup/memory/benchnode; mkdir -p $cg
    echo 0 > $cg/memory.swappiness 2>/dev/null || true
    echo "$MEM_LIMIT" > $cg/memory.limit_in_bytes; echo 1 > $cg/memory.force_empty 2>/dev/null || true
    pre=(bash -c "echo \$\$ > $cg/cgroup.procs; exec $drop taskset -c $NODE_CPUS \"\$@\"" --)
  fi
  env -u JAVA_HOME -u JAVA_TOOL_OPTIONS OPENSEARCH_JAVA_HOME="$home/jdk" OPENSEARCH_JAVA_OPTS="-Xms$HEAP -Xmx$HEAP" nohup "${pre[@]}" "$home/bin/opensearch" \
    -E discovery.type=single-node -E http.port=$PORT -E path.data="$data" -E path.logs="$NODE_HOME/logs-$kind" \
    -E cluster.routing.allocation.disk.threshold_enabled=false $sec ${NODE_ARGS:-} > "$NODE_HOME/$kind.out" 2>&1 &
  echo $! > "$NODE_HOME/node.pid"
  for i in $(seq 120); do curl -sf "$U/_cluster/health?wait_for_status=yellow" >/dev/null 2>&1 && return 0; sleep 2; done
  echo "node did not start, see $NODE_HOME/$kind.out" >&2; return 1
}

native_stop() {
  local pids; pids=$(pgrep -f "org.opensearch.bootstrap.OpenSearch" || true)
  [ -n "$pids" ] && { kill $pids; for i in $(seq 60); do pgrep -f "org.opensearch.bootstrap.OpenSearch" >/dev/null || break; sleep 1; done; }
  pgrep -f "org.opensearch.bootstrap.OpenSearch" >/dev/null && kill -9 $(pgrep -f "org.opensearch.bootstrap.OpenSearch") || true
  rm -f "$NODE_HOME/node.pid"
}

# Resident memory of the node, the shape bench_nodes.py reads from `docker stats`.
native_rss() { ps -o rss= -p "$(pgrep -f org.opensearch.bootstrap.OpenSearch | head -1)" | awk '{print $1*1024}'; }
