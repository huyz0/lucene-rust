#!/usr/bin/env bash
# Runs a command under a hard memory (and pid) cap without Docker: a cgroup, so the kernel kills
# the runaway build or test inside it and the host session survives (what scripts/docker-test.sh
# does with a container, for boxes that have none).
#
#   MEM=16g PIDS=4096 scripts/capped.sh cargo nextest run --workspace
#
# Needs root with a writable cgroup tree (v1 memory controller or v2), or systemd-run for a user
# scope; with neither it says so and runs the command uncapped only when CAPPED_OPTIONAL=1.
set -euo pipefail
MEM="${MEM:-16g}"
PIDS="${PIDS:-4096}"
[[ $# -gt 0 ]] || { echo "usage: scripts/capped.sh COMMAND..." >&2; exit 2; }
name="capped-$$"
if [[ $(id -u) == 0 && -f /sys/fs/cgroup/cgroup.controllers ]]; then
    cg="/sys/fs/cgroup/$name"
    echo +memory +pids > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true
    mkdir "$cg"
    numfmt --from=iec "${MEM^^}" > "$cg/memory.max"
    echo 0 > "$cg/memory.swap.max" 2>/dev/null || true
    echo "$PIDS" > "$cg/pids.max" 2>/dev/null || true
    trap 'rmdir "$cg" 2>/dev/null || true' EXIT
    bash -c 'echo $$ > "$0/cgroup.procs"; exec "${@:1}"' "$cg" "$@"
elif [[ $(id -u) == 0 && -d /sys/fs/cgroup/memory ]]; then
    cg="/sys/fs/cgroup/memory/$name"
    mkdir "$cg"
    numfmt --from=iec "${MEM^^}" > "$cg/memory.limit_in_bytes"
    echo 0 > "$cg/memory.swappiness" 2>/dev/null || true
    trap 'rmdir "$cg" 2>/dev/null || true' EXIT
    bash -c 'echo $$ > "$0/cgroup.procs"; exec "${@:1}"' "$cg" "$@"
elif command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope -p MemoryMax=1G true >/dev/null 2>&1; then
    systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 -p TasksMax="$PIDS" "$@"
elif [[ "${CAPPED_OPTIONAL:-0}" == 1 ]]; then
    echo "capped: no cgroup access, running uncapped" >&2
    exec "$@"
else
    echo "capped: needs root with cgroups or systemd-run --user (CAPPED_OPTIONAL=1 runs uncapped)" >&2
    exit 1
fi
