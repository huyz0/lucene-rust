#!/usr/bin/env bash
# the end-to-end proof: builds the accelerated-search OpenSearch plugin, installs it
# into the pinned OpenSearch distribution, starts a node (a plain process, no Docker), and runs
# opensearch-plugin/e2e/verify_opensearch.py against it -- the query matrix
# native vs Lucene, force-merge reader release, and SIGKILL recovery.
#
# With --yaml it then runs OpenSearch's own REST YAML suites (YAML_SUITES)
# against that node *and* against a stock node built from the same image
# without the plugin (`stock-min`), and requires the two failure sets to be identical and
# the plugin to report no native error: the suites' own pass/fail is
# OpenSearch's business, a difference between the two nodes is ours. Then
# e2e/highlight_matrix.py requires every highlight on the two to be identical.
#
# Without --engine the plugin node runs with indices.engine.accelerated.default: false,
# the opt-out: OpenSearch's engine for every index that does not ask, native
# search on top. With --engine it keeps the default (true), so it serves every
# index -- system indices and every index the YAML suites create included --
# with the accelerated engine, and the e2e stage is
# opensearch-plugin/e2e/verify_engine.py: the Rust engine against OpenSearch's
# own, operation for operation, through restarts, SIGKILL, a writer panic and
# a tripped breaker.
#
# With --js the plugin is built with V8 (-Pjs) and the node serves
# `"lang": "js"` scripts (script.js.enabled), which the e2e
# checks against the same scripts in Painless.
#
#   scripts/verify-opensearch.sh [--docs N] [--bench-out FILE] [--yaml] [--engine] [--js] [--keep]
#
# Needs a JDK 25 (Gradle's toolchain; scripts/opensearch-dist.sh registers the tarball's own),
# Gradle and cargo; nodes are tarball distributions started by
# opensearch-plugin/e2e/nodectl.py (state and data under $NODE_HOME, ~/os-native). The node
# listens on localhost:${OS_PORT:-9200}.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION=3.9.0
nodectl() { python3 opensearch-plugin/e2e/nodectl.py "$@"; }
NAME="${OS_CONTAINER:-lucene-rust-verify}"
PORT="${OS_PORT:-9200}"
KEEP=0
YAML=0
ENGINE=0
JS=0
YAML_SUITES="${YAML_SUITES:-search,search.aggregation,search.highlight,search.inner_hits,mlt,msearch,scroll,count,explain,suggest,get,index,delete,bulk,update,mget,exists}"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP=1; shift ;;
        --yaml) YAML=1; shift ;;
        --engine) ENGINE=1; shift ;;
        --js) JS=1; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

scripts/opensearch-dist.sh
if [[ $JS == 1 ]]; then
    export ORG_GRADLE_PROJECT_js=1 E2E_JS=1
fi
gradle --no-daemon -q -p opensearch-plugin clean bundlePlugin check
# The disk-watermark block is off in nodectl (a CI runner's or container's disk can sit past
# the flood stage, and a node refusing to create indices tests nothing).
NODE_ENV=(indices.engine.accelerated.default=false)
if [[ $ENGINE == 1 ]]; then
    NODE_ENV=()
fi
if [[ $JS == 1 ]]; then
    NODE_ENV+=(script.js.enabled=true script.js.timeout=1s)
fi
start_node() { # name port variant [setting ...]; the e2e hunspell dictionary is on every node, stock ones too
    local n=$1 p=$2 v=$3; shift 3
    local a=() s
    for s in "$@"; do a+=(--setting "$s"); done
    nodectl run "$n" "$v" "$p" --setting cluster.name=accelerated-search-e2e "${a[@]}" >/dev/null
}
start_node "$NAME" "$PORT" plugin ${NODE_ENV[@]+"${NODE_ENV[@]}"}
if [[ $KEEP == 0 ]]; then
    trap 'nodectl rm "$NAME" >/dev/null 2>&1; nodectl rm "$NAME-stock" >/dev/null 2>&1 || true' EXIT
fi

status=0
if [[ $ENGINE == 1 ]]; then
    for _ in $(seq 1 90); do curl -sf "localhost:$PORT" >/dev/null && break; sleep 2; done
    python3 opensearch-plugin/e2e/verify_engine.py "http://localhost:${PORT}" "$NAME" "${ARGS[@]}" || status=$?
    if [[ $status != 0 ]]; then
        # The node's own account of a failed check (a shard that failed to start, say); the
        # container is gone once the script exits.
        echo "verify-opensearch: the node's last warnings and errors:" >&2
        nodectl logs "$NAME" 2>&1 | grep -E -A12 '^\[[^]]*\]\[(WARN|ERROR)' | tail -n 200 >&2 || true
    fi
else
    python3 opensearch-plugin/e2e/verify_opensearch.py "http://localhost:${PORT}" "$NAME" "${ARGS[@]}" || status=$?
fi

# The YAML runner: OpenSearch's test framework refuses to run as root, so a
# root shell runs it as `nobody` from a staged classpath.
run_yaml() { # port out-dir
    local cp="$PWD/opensearch-plugin/build/yaml-classpath"
    # The JDK 25 the classes were compiled for, not whatever `java` is on PATH.
    local java_bin
    java_bin=$(cat "$PWD/opensearch-plugin/build/java-launcher.txt")
    local as=()
    [[ $(id -u) == 0 ]] && as=(runuser -u nobody --)
    mkdir -p "$2" && chmod 777 "$2"
    (cd "$2" && "${as[@]}" "$java_bin" -ea -cp "$cp:$cp/*" \
        -Dtests.rest.cluster="localhost:$1" -Dtests.cluster="localhost:$1" \
        -Dtests.clustername=accelerated-search-e2e -Dtests.rest.suite="$YAML_SUITES" \
        -Dtests.security.manager=false -Djava.io.tmpdir="$2" \
        org.junit.runner.JUnitCore org.lucenerust.opensearch.LuceneRustYamlIT >"$2/out.txt" 2>&1 || true)
    grep -E '^[0-9]+\) ' "$2/out.txt" | sed 's/^[0-9]*) //' | sort >"$2/failures.txt" || true
    grep -E '^(OK|Tests run)' "$2/out.txt" | tail -1
}
# Checked now: the YAML stage below replaces this container, and its log with it.
check_jvm() {
    if nodectl logs "$1" 2>&1 | grep -E "A fatal error has been detected|SIGSEGV \(0xb\)" >/dev/null; then
        echo "verify-opensearch: the JVM of $1 crashed" >&2
        status=1
    fi
}
check_jvm "$NAME"
if [[ $YAML == 1 && $status == 0 ]]; then
    gradle --no-daemon -q -p opensearch-plugin yamlRestTestClasspath
    work=$(mktemp -d) && chmod 755 "$work"
    # A fresh node: the e2e above killed and restarted this one.
    start_node "$NAME" "$PORT" plugin ${NODE_ENV[@]+"${NODE_ENV[@]}"}
    start_node "$NAME-stock" "$((PORT + 2))" stock-min
    for p in "$PORT" "$((PORT + 2))"; do
        for _ in $(seq 1 90); do curl -sf "localhost:$p" >/dev/null && break; sleep 2; done
    done
    echo "yaml (lucene-rust): $(run_yaml "$PORT" "$work/rust")"
    echo "yaml (stock):       $(run_yaml "$((PORT + 2))" "$work/stock")"
    stats=$(curl -s "localhost:${PORT}/_plugins/accelerated/stats")
    echo "yaml: plugin stats $stats"
    # Failures that are the plugin's design, each with its reason in the file. An entry that no
    # longer fails is an error: the list can only name tests that still differ.
    known="opensearch-plugin/e2e/yaml-known-differences.txt"
    { grep -v '^#' "$known" || true; } | { grep -v '^$' || true; } | sort >"$work/known.txt"
    comm -13 "$work/rust/failures.txt" "$work/known.txt" >"$work/stale.txt"
    if [[ -s "$work/stale.txt" ]]; then
        echo "verify-opensearch: $known lists tests that pass now:" >&2
        cat "$work/stale.txt" >&2
        status=1
    fi
    comm -23 "$work/rust/failures.txt" "$work/known.txt" >"$work/rust/failures-unexpected.txt"
    if ! diff "$work/stock/failures.txt" "$work/rust/failures-unexpected.txt"; then
        echo "verify-opensearch: the YAML suites fail differently with the plugin" >&2
        status=1
    fi
    if ! grep -q '"accelerated_errors":0' <<<"$stats" || grep -q '"accelerated_queries":0' <<<"$stats"; then
        echo "verify-opensearch: native errors during the YAML suites, or nothing ran native" >&2
        status=1
    fi
    echo "yaml: $(wc -l <"$work/rust/failures.txt") failures on both nodes; outputs in $work"
    # Every highlight of the highlight matrix, plugin against stock (the native `unified`).
    if ! PLUGIN="http://localhost:${PORT}" STOCK="http://localhost:$((PORT + 2))" \
        python3 opensearch-plugin/e2e/highlight_matrix.py all; then
        echo "verify-opensearch: highlights differ from stock" >&2
        status=1
    fi
fi
if [[ $YAML == 1 ]]; then
    check_jvm "$NAME"
fi
exit $status
