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
#   MEM=16g CPUS=8 PIDS=512 TMPFS=1g  scripts/docker-test.sh ...
#   DOCKER_LOCK=0 skips the one-job-at-a-time lock (DOCKER_TEST_LOCK,
#   DOCKER_LOCK_STALE_S: where it lives, when a dead holder's lock is reclaimed)
#   PREFLIGHT=0 skips the engine and disk checks (scripts/preflight.sh lists
#   their own knobs: MIN_FREE_GB, DOCKER_VHDX_MAX_GB, BUILD_CACHE_MAX_GB, ...)
#   LUCENE_JAVA_SRC selects the pinned Lucene 10.5.2 checkout (read-only)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Tagged by the Dockerfile's content, so a checkout whose Dockerfile adds a jar
# or bumps a toolchain builds its own image instead of reusing a stale one (and
# two checkouts on different Dockerfiles never overwrite each other's image).
DOCKERFILE_TAG="$(sha256sum "$REPO/docker/Dockerfile" | cut -c1-12)"
IMAGE="${IMAGE:-lucene-rust-test:$DOCKERFILE_TAG}"
JAVA_SOURCE="${LUCENE_JAVA_SRC:-$HOME/work/lucene-10.5.2}"

# Hard ceilings. `--memory-swap` equal to `--memory` disables swap for the
# container: without it, a runaway allocates into swap and drags the host down
# slowly instead of failing fast, which is the exact behaviour this script
# exists to prevent.
MEM="${MEM:-16g}"
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
  # Every run trims this checkout's target volume, not only the gate: coverage,
  # fixture and bench runs grew it to 146 GB between gates (gate.sh runs the
  # same check itself, so it is not wrapped twice).
  if [ "${PREFLIGHT:-1}" != "0" ]; then
    CMD=(bash -c 'scripts/preflight.sh container && exec "$@"' preflight "${CMD[@]}")
  fi
fi

# Refuse to start on a wedged engine or a nearly full drive: a build that fills
# the disk kills rustc and then Docker itself (scripts/preflight.sh says how).
"$REPO/scripts/preflight.sh" host

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "docker-test: building $IMAGE (first run for this Dockerfile)" >&2
  # Relative paths from the repo: under Git Bash with MSYS_NO_PATHCONV=1 (how
  # agents run this) an absolute /e/... path reaches docker.exe unconverted
  # and the build fails with "path not found".
  (cd "$REPO" && docker build -t "$IMAGE" -f docker/Dockerfile docker)
fi

# Named volumes for the cargo registry and target dir: without them every run
# recompiles the world, which is itself a memory spike worth avoiding. They are
# separate from the host's ./target so a container build and a host build never
# fight over the same lock or mix incompatible artifacts.
docker volume create lucene-rust-cargo >/dev/null
# Every checkout is mounted at /work. Sharing Cargo artifacts between those
# checkouts lets an older source mtime reuse a different worktree's artifact:
# a linked-worktree test actually resolved an obsolete lucene-codecs export.
# Keep dependency downloads shared, but fingerprint each checkout's builds in
# a separate volume. The machine-wide lock below still bounds total build load.
TARGET_ID="$(printf '%s' "$REPO" | sha256sum | cut -c1-12)"
TARGET_VOLUME="lucene-rust-target-$TARGET_ID"
docker volume create "$TARGET_VOLUME" >/dev/null
echo "docker-test: target cache $TARGET_VOLUME" >&2
# The benchmark corpus lives on a volume too, not on the bind mount: on Docker
# Desktop a bind-mounted host drive turns every mmap page fault into a
# filesystem round trip, which taxes both engines and measures the mount.
docker volume create lucene-rust-corpus >/dev/null

# `-it` only when there is a terminal: agents and CI invoke this with no TTY,
# and `docker run -it` fails outright there rather than degrading.
TTY_FLAGS=()
[ -t 0 ] && [ -t 1 ] && TTY_FLAGS=(-it)

# One heavy job at a time, across every checkout and session on this machine.
# Two containers for one checkout share its target volume, and each may take
# 16 GiB (two can exhaust the VM). The lock is a directory -- `mkdir` is
# atomic and Git Bash has no `flock` -- under $HOME, so worktrees share it. A
# holder that died without its EXIT trap (kill -9) leaves a stale lock; it is
# reclaimed once it is older than DOCKER_LOCK_STALE_S, no lucene-rust-test
# container is running, *and* its holder is gone: the owner file's pid fails
# `kill -0` (MSYS pids are machine-wide, so another session's shell is seen),
# or, for an owner file without a pid, the engine answers. "No container
# running" alone is not enough: with the engine stopped (a vhdx compaction
# holding the lock) it is trivially true, and a queued run once stole the lock.
# After taking the lock we also wait out any running lucene-rust-test
# container, which covers a checkout whose script predates the lock.
# DOCKER_LOCK=0 skips all of this (CI does not use this script anyway).
LOCK="${DOCKER_TEST_LOCK:-$HOME/.lucene-rust-docker-test.lock}"
LOCK_STALE_S="${DOCKER_LOCK_STALE_S:-300}"
running_jobs() {
  docker ps --format '{{.Image}}' 2>/dev/null | grep -c '^lucene-rust-test' || true
}
holder_gone() {
  local pid
  pid="$(sed -n 's/.* pid \([0-9][0-9]*\) since .*/\1/p' "$LOCK/owner" 2>/dev/null)"
  if [ -n "$pid" ]; then
    ! kill -0 "$pid" 2>/dev/null
  else
    docker info >/dev/null 2>&1
  fi
}
if [ "${DOCKER_LOCK:-1}" != "0" ]; then
  waited=0
  me="pid $$ nonce $RANDOM$RANDOM"
  while :; do
  until mkdir "$LOCK" 2>/dev/null; do
    age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || date +%s) ))
    stale_owner="$(cat "$LOCK/owner" 2>/dev/null)"
    if [ "$age" -gt "$LOCK_STALE_S" ] && [ "$(running_jobs)" = "0" ] && holder_gone \
      && [ "$(cat "$LOCK/owner" 2>/dev/null)" = "$stale_owner" ]; then
      echo "docker-test: reclaiming stale lock $LOCK (${age}s old, holder gone, no job running)" >&2
      rm -rf "$LOCK"
      continue
    fi
    if [ "$waited" = 0 ]; then
      echo "docker-test: waiting for the other job to finish ($(cat "$LOCK/owner" 2>/dev/null || echo unknown owner))" >&2
      waited=1
    fi
    sleep 15
  done
  echo "$REPO pid $$ since $(date '+%F %T') ($me): ${CMD[*]}" > "$LOCK/owner"
  # Waiters that judged the same dead lock stale at the same moment each
  # `rm -rf` it and `mkdir` their own, the later ones deleting the earlier
  # ones' fresh lock: four runs once all started at one second this way.
  # Settle, then keep the lock only if the owner file is still ours.
  sleep 3
  if grep -qF "($me)" "$LOCK/owner" 2>/dev/null; then break; fi
  echo "docker-test: lost the lock to a concurrent reclaim, waiting again" >&2
  done
  trap 'rm -rf "$LOCK"' EXIT
  while [ "$(running_jobs)" != "0" ]; do
    if [ "$waited" != 2 ]; then
      echo "docker-test: lock held, waiting for a running lucene-rust-test container to exit" >&2
      waited=2
    fi
    sleep 15
  done
fi

# Linked worktrees carry a gitfile with a host-absolute metadata path. That
# path is unavailable at /work in the container. Overlay only the gitfile
# with a temporary container path and mount the common metadata read-only;
# Git can then resolve the worktree's index and HEAD without changing the
# host checkout or overriding Git's environment for unrelated repositories.
GIT_FLAGS=()
if [ -f "$REPO/.git" ]; then
  GIT_REPO="$REPO"
  if command -v cygpath >/dev/null 2>&1; then
    GIT_REPO="$(cygpath -m "$REPO")"
  fi
  GIT_COMMON="$(git -c safe.directory="$GIT_REPO" -C "$GIT_REPO" rev-parse --path-format=absolute --git-common-dir)"
  GIT_META="$(git -c safe.directory="$GIT_REPO" -C "$GIT_REPO" rev-parse --absolute-git-dir)"
  GIT_REL="${GIT_META#"$GIT_COMMON"/}"
  if [ "$GIT_REL" = "$GIT_META" ]; then
    echo "docker-test: worktree metadata is outside the common Git directory" >&2
    exit 1
  fi
  GIT_FILE="$(mktemp)"
  if [ "${DOCKER_LOCK:-1}" != "0" ]; then
    trap 'rm -f "$GIT_FILE"; rm -rf "$LOCK"' EXIT
  else
    trap 'rm -f "$GIT_FILE"' EXIT
  fi
  printf 'gitdir: /work-git/%s\n' "$GIT_REL" > "$GIT_FILE"
  GIT_FILE_HOST="$GIT_FILE"
  if command -v cygpath >/dev/null 2>&1; then
    GIT_FILE_HOST="$(cygpath -m "$GIT_FILE")"
  fi
  GIT_FLAGS=(-v "$GIT_COMMON:/work-git:ro" -v "$GIT_FILE_HOST:/work/.git:ro")
fi

# Not `exec`: the shell must stay to release the lock on exit.
docker run --rm "${TTY_FLAGS[@]}" \
  --memory="$MEM" --memory-swap="$MEM" \
  --cpus="$CPUS" --pids-limit="$PIDS" \
  --tmpfs "/tmp:rw,size=$TMPFS,mode=1777" \
  -v "$REPO:/work" \
  "${GIT_FLAGS[@]}" \
  -v "$JAVA_SOURCE:/lucene-10.5.2:ro" \
  -v lucene-rust-cargo:/usr/local/cargo/registry \
  -v "$TARGET_VOLUME:/work/target-docker" \
  -v lucene-rust-corpus:/work/benchmarks/.corpus \
  -e CARGO_TARGET_DIR=/work/target-docker \
  -e DOCKER_TARGET_VOLUME="$TARGET_VOLUME" \
  -e JARS=/opt/lucene-jars \
  -e CARGO_BUILD_JOBS="$CPUS" \
  -w /work \
  "$IMAGE" \
  "${CMD[@]}"
