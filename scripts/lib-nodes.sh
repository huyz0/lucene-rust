#!/usr/bin/env bash
# Shared helpers for scripts that start OpenSearch nodes: plain processes through
# opensearch-plugin/e2e/nodectl.py, no Docker. Source it from the repo root.
#
#   os_up NAME VARIANT PORT [-e setting=value]... [-e ENV=value]...   # start; waits until it answers
#   os_down NAME            # kill and forget (data directories outside the node's own stay)
#   os_data VOLUME          # a persistent data directory (what a docker named volume was)
#   os_has_data VOLUME      # whether that directory exists and has data
#
# VARIANT: plugin, stock-min, stock (nodectl.py documents them). Environment read by os_up:
# HEAP (default 1g), NODE_CPUS (taskset list), MEM_LIMIT (cgroup memory cap, no swap; root),
# DATA_DIR (data directory, default under the node's own state). `-e` takes the docker-run
# spelling the old scripts used: a dotted name is an OpenSearch setting, anything else an
# environment variable of the node.
nodectl() { python3 "${REPO_ROOT:-.}/opensearch-plugin/e2e/nodectl.py" "$@"; }

os_up() {
    local n=$1 v=$2 p=$3; shift 3
    local a=() k
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -e) k=${2%%=*}
                if [[ $k == *.* ]]; then a+=(--setting "$2"); else a+=(--env "$2"); fi
                shift 2 ;;
            *) a+=("$1"); shift ;;
        esac
    done
    nodectl rm "$n" >/dev/null 2>&1 || true
    nodectl run "$n" "$v" "$p" --heap "${HEAP:-1g}" ${NODE_CPUS:+--cpus "$NODE_CPUS"} \
        ${MEM_LIMIT:+--mem-limit "$MEM_LIMIT"} ${DATA_DIR:+--data "$DATA_DIR"} \
        ${a[@]+"${a[@]}"} >/dev/null
}

os_down() { nodectl rm "$1" >/dev/null 2>&1 || true; }

os_data() { echo "${NODE_HOME:-$HOME/os-native}/data/$1"; }

os_has_data() { [[ -d "$(os_data "$1")/nodes" ]]; }
