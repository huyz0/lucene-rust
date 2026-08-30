#!/usr/bin/env bash
# Runs a command against this repo inside a resource-capped container.
#
# The point is containment, not convenience. On WSL2 a runaway cargo/test
# process that exhausts the VM takes the Claude Code session with it and needs
# a manual `wsl --shutdown` to recover. Inside these limits the kernel kills
# something in the container instead, and the host is never at risk.
#
#   scripts/docker-test.sh gate                     # THE gate (see AGENTS.md)
#   scripts/docker-test.sh                          # cargo test --workspace
#   scripts/docker-test.sh cargo test -p lucene-codecs
#   scripts/docker-test.sh scripts/verify-write-path.sh
#   scripts/docker-test.sh bash                     # interactive shell
#
# This is the official way to run the gate in local development. CI does not
# use it: GitHub Actions runners are already isolated VMs, so containing them
# again would only cost build time -- CI runs the same commands natively
# (.github/workflows/ci.yml).
#
# Env knobs (all have defensible defaults, none need setting):
#   MEM=8g CPUS=8 PIDS=512 TMPFS=1g  scripts/docker-test.sh ...
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lucene-rust-test}"

# Hard ceilings. `--memory-swap` equal to `--memory` disables swap for the
# container: without it, a runaway allocates into swap and drags the host down
# slowly instead of failing fast, which is the exact behaviour this script
# exists to prevent.
MEM="${MEM:-8g}"
CPUS="${CPUS:-8}"
PIDS="${PIDS:-512}"
# /tmp is a tmpfs and therefore RAM. The suites used to leak a directory per
# test into it and filled a 16 GB tmpfs; that leak is fixed
# (`lucene_util::test_support::TempDir`), but a bounded mount means a
# regression costs a failed test rather than the machine.
TMPFS="${TMPFS:-1g}"

# `gate` expands to the whole pre-commit gate, so that the definition lives in
# exactly one place and a developer cannot accidentally run a subset of it.
if [ "${1:-}" = "gate" ]; then
  CMD=(bash /work/scripts/gate.sh)
else
  CMD=("${@:-cargo test --workspace}")
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "docker-test: building $IMAGE (first run only)" >&2
  docker build -t "$IMAGE" -f "$REPO/docker/Dockerfile" "$REPO/docker"
fi

# Named volumes for the cargo registry and target dir: without them every run
# recompiles the world, which is itself a memory spike worth avoiding. They are
# separate from the host's ./target so a container build and a host build never
# fight over the same lock or mix incompatible artifacts.
docker volume create lucene-rust-cargo >/dev/null
docker volume create lucene-rust-target >/dev/null

# `-it` only when there is a terminal: agents and CI invoke this with no TTY,
# and `docker run -it` fails outright there rather than degrading.
TTY_FLAGS=()
[ -t 0 ] && [ -t 1 ] && TTY_FLAGS=(-it)

exec docker run --rm "${TTY_FLAGS[@]}" \
  --memory="$MEM" --memory-swap="$MEM" \
  --cpus="$CPUS" --pids-limit="$PIDS" \
  --tmpfs "/tmp:rw,size=$TMPFS,mode=1777" \
  -v "$REPO:/work" \
  -v "$HOME/work/lucene-10.5.0:/lucene-10.5.0:ro" \
  -v lucene-rust-cargo:/usr/local/cargo/registry \
  -v lucene-rust-target:/work/target-docker \
  -e CARGO_TARGET_DIR=/work/target-docker \
  -e JARS=/opt/lucene-jars \
  -e CARGO_BUILD_JOBS="$CPUS" \
  -w /work \
  "$IMAGE" \
  "${CMD[@]}"
