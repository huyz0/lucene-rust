#!/usr/bin/env bash
# M2's end-to-end proof: builds the lucene-rust OpenSearch plugin, installs it
# into the pinned OpenSearch image, starts a node, and runs
# opensearch-plugin/e2e/verify_opensearch.py against it -- the query matrix
# native vs Lucene, force-merge reader release, and SIGKILL recovery.
#
# With --yaml it then runs OpenSearch's own REST YAML suites (YAML_SUITES)
# against that node *and* against a stock node built from the same image
# without the plugin, and requires the two failure sets to be identical and
# the plugin to report no native error: the suites' own pass/fail is
# OpenSearch's business, a difference between the two nodes is ours.
#
#   scripts/verify-opensearch.sh [--docs N] [--bench-out FILE] [--yaml] [--keep]
#
# Needs Docker, a JDK 21, Gradle and cargo. The node listens on
# localhost:${OS_PORT:-9200}.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION=3.8.0
IMAGE="lucene-rust-opensearch:${VERSION}"
NAME="${OS_CONTAINER:-lucene-rust-verify}"
PORT="${OS_PORT:-9200}"
KEEP=0
YAML=0
YAML_SUITES="${YAML_SUITES:-search,search.highlight,search.inner_hits,msearch,scroll,count,explain,suggest,get,index,delete,bulk,update,mget,exists}"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP=1; shift ;;
        --yaml) YAML=1; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

scripts/opensearch-dist.sh
gradle --no-daemon -q -p opensearch-plugin clean bundlePlugin check
rm -f opensearch-plugin/docker/*.zip
cp opensearch-plugin/build/distributions/lucene-rust-*.zip opensearch-plugin/docker/

# The library must load in the image: its newest required glibc symbol
# version may not exceed the image's glibc.
need=$(objdump -T "${CARGO_TARGET_DIR:-target}/release/liblucene_ffi.so" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
have=$(docker run --rm --entrypoint ldd "opensearchproject/opensearch:${VERSION}" --version 2>/dev/null | sed -n 1p | grep -o '[0-9.]*$')
if [[ "$(printf '%s\n%s\n' "${need#GLIBC_}" "$have" | sort -V | tail -1)" != "$have" ]]; then
    echo "verify-opensearch: liblucene_ffi.so needs ${need}, the image has glibc ${have}" >&2
    exit 1
fi

docker build -q -t "$IMAGE" opensearch-plugin/docker >/dev/null
# The disk-watermark block is off: a CI runner's or container's disk can sit
# past the flood stage, and a node refusing to create indices tests nothing.
start_node() { # name port image
    docker rm -f "$1" >/dev/null 2>&1 || true
    docker run -d --name "$1" -p "$2:9200" \
        -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false \
        -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms1g -Xmx1g" \
        "$3" >/dev/null
}
start_node "$NAME" "$PORT" "$IMAGE"
if [[ $KEEP == 0 ]]; then
    trap 'docker rm -f "$NAME" "$NAME-stock" >/dev/null 2>&1 || true' EXIT
fi

status=0
python3 opensearch-plugin/e2e/verify_opensearch.py "http://localhost:${PORT}" "$NAME" "${ARGS[@]}" || status=$?

# The YAML runner: OpenSearch's test framework refuses to run as root, so a
# root shell runs it as `nobody` from a staged classpath.
run_yaml() { # port out-dir
    local cp="$PWD/opensearch-plugin/build/yaml-classpath"
    local as=()
    [[ $(id -u) == 0 ]] && as=(runuser -u nobody --)
    mkdir -p "$2" && chmod 777 "$2"
    (cd "$2" && "${as[@]}" java -ea -cp "$cp:$cp/*" \
        -Dtests.rest.cluster="localhost:$1" -Dtests.cluster="localhost:$1" \
        -Dtests.clustername=docker-cluster -Dtests.rest.suite="$YAML_SUITES" \
        -Dtests.security.manager=false -Djava.io.tmpdir="$2" \
        org.junit.runner.JUnitCore org.lucenerust.opensearch.LuceneRustYamlIT >"$2/out.txt" 2>&1 || true)
    grep -E '^[0-9]+\) ' "$2/out.txt" | sed 's/^[0-9]*) //' | sort >"$2/failures.txt" || true
    grep -E '^(OK|Tests run)' "$2/out.txt" | tail -1
}
if [[ $YAML == 1 && $status == 0 ]]; then
    gradle --no-daemon -q -p opensearch-plugin yamlRestTestClasspath
    work=$(mktemp -d) && chmod 755 "$work"
    # A fresh node: the e2e above killed and restarted this one.
    start_node "$NAME" "$PORT" "$IMAGE"
    printf 'FROM opensearchproject/opensearch:%s\nRUN rm -rf /usr/share/opensearch/plugins/*\n' "$VERSION" \
        | docker build -q -t "opensearch-stock-min:${VERSION}" - >/dev/null
    start_node "$NAME-stock" "$((PORT + 2))" "opensearch-stock-min:${VERSION}"
    for p in "$PORT" "$((PORT + 2))"; do
        for _ in $(seq 1 90); do curl -sf "localhost:$p" >/dev/null && break; sleep 2; done
    done
    echo "yaml (lucene-rust): $(run_yaml "$PORT" "$work/rust")"
    echo "yaml (stock):       $(run_yaml "$((PORT + 2))" "$work/stock")"
    stats=$(curl -s "localhost:${PORT}/_plugins/lucene_rust/stats")
    echo "yaml: plugin stats $stats"
    if ! diff "$work/stock/failures.txt" "$work/rust/failures.txt"; then
        echo "verify-opensearch: the YAML suites fail differently with the plugin" >&2
        status=1
    fi
    if ! grep -q '"native_errors":0' <<<"$stats" || grep -q '"native_queries":0' <<<"$stats"; then
        echo "verify-opensearch: native errors during the YAML suites, or nothing ran native" >&2
        status=1
    fi
    echo "yaml: $(wc -l <"$work/rust/failures.txt") failures on both nodes; outputs in $work"
fi
if docker logs "$NAME" 2>&1 | grep -E "A fatal error has been detected|SIGSEGV \(0xb\)" >/dev/null; then
    echo "verify-opensearch: the JVM crashed" >&2
    status=1
fi
exit $status
