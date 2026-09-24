#!/usr/bin/env bash
# M4's interoperability matrix (T4.6): one index written, appended to and
# merged by real Lucene and by this port, in every order an incremental
# OpenSearch adoption produces -- a shard holds segments from both engines at
# once.
#
#   1. Java writes          -> Rust reads
#   2. Rust writes          -> Java reads
#   3. Java writes -> Rust appends -> Java reads (and Rust reads)
#   4. Rust writes -> Java appends -> Rust reads (and Java reads)
#   5. Java writes -> Rust merges  -> Java reads (and Rust reads)
#   6. Rust + Java segments -> Rust merges -> both read
#   7. Java writes -> Rust deletes by term -> both read
#   8. Rust writes -> Java deletes by term -> both read
#   9. Java writes -> Rust deletes -> Rust merges -> both read
#  10. Java writes -> Java deletes -> Rust merges -> both read
#
# Directions 9 and 10 merge compound segments that carry deletions: the
# `.liv` sits beside the archive, not in it.
#
# Java's half is fixtures/src/InteropIndex.java, Rust's is
# crates/lucene-search/examples/interop.rs; both carry the same document
# table and each checks every document's fields semantically (doc values,
# points, term and phrase counts; Java also stored fields) plus its own
# CheckIndex. Java writes with a default IndexWriterConfig, so its flushed
# segments are compound files.
#
# Usage: scripts/verify-interop.sh [--jars DIR] [--keep]
set -euo pipefail

LUCENE_MODULES=(lucene-core lucene-analysis-common lucene-queries)

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
JARS="$FIXTURES/.jars"
KEEP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --jars) JARS="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "verify-interop: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath "${LUCENE_MODULES[@]}")

WORK=$(mktemp -d)
CLASSES=$(mktemp -d)
cleanup() { [ "$KEEP" -eq 1 ] && echo "verify-interop: indexes kept in $WORK" || rm -rf "$WORK"; rm -rf "$CLASSES"; }
trap cleanup EXIT

echo "verify-interop: compiling InteropIndex and building the interop example"
javac -nowarn -cp "$CP" -d "$CLASSES" "$FIXTURES/src/InteropIndex.java"
cargo build --quiet -p lucene-search --example interop
RUST="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/debug/examples/interop"

java_() { java --enable-native-access=ALL-UNNAMED -cp "$CLASSES:$CP" InteropIndex "$@"; }
rust_() { "$RUST" "$@"; }

failed=0
passed=0
# step <label> <engine> <args...>: run one step, print its output on failure.
step() {
  local label="$1" engine="$2"; shift 2
  local out
  if out=$("${engine}_" "$@" 2>&1); then
    return 0
  fi
  echo "  FAIL  $label: $engine $*"
  echo "$out" | sed 's/^/      /' | tail -20
  return 1
}

direction() {
  local name="$1"; shift
  if "$@"; then
    echo "  ok    $name"; passed=$((passed+1))
  else
    failed=$((failed+1))
  fi
}

d1() { local d="$WORK/1"
  step "1" java write "$d" 0 3000 1000 && step "1" rust verify "$d" 3000; }
d2() { local d="$WORK/2"
  step "2" rust write "$d" 0 3000 1000 && step "2" java verify "$d" 3000 3; }
d3() { local d="$WORK/3"
  step "3" java write "$d" 0 3000 1000 && step "3" rust write "$d" 3000 5000 1000 \
    && step "3" java verify "$d" 5000 5 && step "3" rust verify "$d" 5000; }
d4() { local d="$WORK/4"
  step "4" rust write "$d" 0 3000 1000 && step "4" java write "$d" 3000 5000 1000 \
    && step "4" rust verify "$d" 5000 && step "4" java verify "$d" 5000 5; }
d5() { local d="$WORK/5"
  step "5" java write "$d" 0 4000 1000 && step "5" rust merge "$d" \
    && step "5" java verify "$d" 4000 1 && step "5" rust verify "$d" 4000; }
d6() { local d="$WORK/6"
  step "6" rust write "$d" 0 2000 1000 && step "6" java write "$d" 2000 4000 1000 \
    && step "6" rust merge "$d" && step "6" java verify "$d" 4000 1 && step "6" rust verify "$d" 4000; }

d7() { local d="$WORK/7"
  step "7" java write "$d" 0 3000 1000 && step "7" rust delete "$d" w7 \
    && step "7" java verify "$d" 3000 3 w7 && step "7" rust verify "$d" 3000 w7; }
d8() { local d="$WORK/8"
  step "8" rust write "$d" 0 3000 1000 && step "8" java delete "$d" w7 \
    && step "8" java verify "$d" 3000 3 w7 && step "8" rust verify "$d" 3000 w7; }

d9() { local d="$WORK/9"
  step "9" java write "$d" 0 4000 1000 && step "9" rust delete "$d" w7 && step "9" rust merge "$d" \
    && step "9" java verify "$d" 4000 1 w7 && step "9" rust verify "$d" 4000 w7; }
d10() { local d="$WORK/10"
  step "10" java write "$d" 0 4000 1000 && step "10" java delete "$d" w7 && step "10" rust merge "$d" \
    && step "10" java verify "$d" 4000 1 w7 && step "10" rust verify "$d" 4000 w7; }

echo "verify-interop: Lucene $LUCENE_VERSION and this port on one index"
direction "1. Java writes -> Rust reads" d1
direction "2. Rust writes -> Java reads" d2
direction "3. Java writes -> Rust appends -> both read" d3
direction "4. Rust writes -> Java appends -> both read" d4
direction "5. Java writes -> Rust merges -> both read" d5
direction "6. Rust + Java segments -> Rust merges -> both read" d6
direction "7. Java writes -> Rust deletes by term -> both read" d7
direction "8. Rust writes -> Java deletes by term -> both read" d8
direction "9. Java writes -> Rust deletes -> Rust merges -> both read" d9
direction "10. Java writes -> Java deletes -> Rust merges -> both read" d10

echo
if [ "$failed" -eq 0 ]; then
  echo "verify-interop: ok ($passed/$passed passed)"
else
  echo "verify-interop: $failed of $((passed+failed)) failed"
  exit 1
fi
