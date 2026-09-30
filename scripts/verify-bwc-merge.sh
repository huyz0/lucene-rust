#!/usr/bin/env bash
# M8 T8.4: this port merges segments an older Lucene wrote into Lucene104
# ones, and real Lucene 10.5.0 checks the result.
#
# For every fixtures/data/bwc/<version> index (one per codec era, 9.0.0 to
# 10.4.0), two merges, each checked by fixtures/bwc/BwcMergeCheck.java on
# 10.5.0 + backward-codecs:
#
#   force   the old index alone, IndexWriter::force_merge(1)
#   mixed   the old index plus three segments Lucene 10.5.0 appends to it
#           (fixtures/bwc/BwcAppend.java: an upgraded shard), merged by an
#           ordinary TieredMergePolicy merge at commit that takes old and new
#           segments together
#   upgrade the same mixed index through IndexUpgrader, which rewrites only
#           the segments the older Lucene wrote and keeps the new ones
#
# The same two merges run over every fixtures/data/bwc-quantized/<version>
# index (9.9.2, 9.12.2, 10.2.2: per-field Lucene99 scalar- and Lucene102
# binary-quantized vector fields, flat and HNSW), whose raw vectors the merge
# carries into Lucene99HnswVectorsFormat.
#
# BwcMergeCheck requires CheckIndex to be clean, every segment the merge wrote
# to be Lucene104 (postings Lucene104, vectors Lucene99HnswVectorsFormat), and
# every live document of the input to be in the output with all of its
# content: postings with positions, offsets and payloads, norms, all five
# doc-values types, points, stored fields, term vectors and vectors.
#
# Usage: scripts/verify-bwc-merge.sh [--only <version>] [--keep]
#
# The merge binary is built with cargo's `dev` profile, or $CARGO_BWC_PROFILE.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

VERSIONS=(9.0.0 9.1.0 9.3.0 9.4.2 9.8.0 9.11.1 9.12.2 10.0.0 10.2.2 10.4.0)
QUANTIZED_VERSIONS=(9.9.2 9.12.2 10.2.2)
ONLY=""
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "verify-bwc-merge: unknown argument: $1" >&2; exit 2 ;;
  esac
done

JARS="$PWD/fixtures/.jars"
# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core lucene-backward-codecs)

WORK=$(mktemp -d)
cleanup() { [ "$KEEP" -eq 1 ] && echo "verify-bwc-merge: kept $WORK" || rm -rf "$WORK"; }
trap cleanup EXIT

javac -nowarn -proc:none -cp "$CP" -d "$WORK/classes" \
  fixtures/bwc/BwcMergeCheck.java fixtures/bwc/BwcAppend.java
PROFILE="${CARGO_BWC_PROFILE:-dev}"
PROFILE_DIR="$PROFILE"
[ "$PROFILE" = dev ] && PROFILE_DIR=debug
cargo build --quiet --profile "$PROFILE" -p lucene-search --example bwc_merge
MERGE="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/$PROFILE_DIR/examples/bwc_merge"

failed=0
passed=0
check() {
  local label="$1" input="$2" output="$3" out
  if out=$(java -cp "$WORK/classes:$CP" BwcMergeCheck "$input" "$output" 2>&1); then
    passed=$((passed + 1))
    echo "  ok    $label"
  else
    failed=$((failed + 1))
    echo "  FAIL  $label"
    echo "$out" | grep -v '^ok ' | sed 's/^/      /' | tail -20
  fi
}

runs=()
for v in "${VERSIONS[@]}"; do runs+=("bwc/$v"); done
for v in "${QUANTIZED_VERSIONS[@]}"; do runs+=("bwc-quantized/$v"); done
for run in "${runs[@]}"; do
  v="${run#*/}"
  [ -z "$ONLY" ] || [ "$v" = "$ONLY" ] || continue
  src="fixtures/data/$run"
  echo "verify-bwc-merge: $run"
  v="${run/\//-}"

  "$MERGE" "$src" "$WORK/$v-force" force > /dev/null
  check "$v force_merge(1)" "$src" "$WORK/$v-force"

  mkdir -p "$WORK/$v-mixed"
  find "$src" -maxdepth 1 -type f ! -name '*.txt' -exec cp {} "$WORK/$v-mixed/" \;
  java -cp "$WORK/classes:$CP" BwcAppend "$WORK/$v-mixed" 3 2>/dev/null
  "$MERGE" "$WORK/$v-mixed" "$WORK/$v-mixed-merged" policy > /dev/null
  check "$v + 3 Lucene 10.5.0 segments, ordinary merge" "$WORK/$v-mixed" "$WORK/$v-mixed-merged"

  "$MERGE" "$WORK/$v-mixed" "$WORK/$v-mixed-upgraded" upgrade > /dev/null
  check "$v + 3 Lucene 10.5.0 segments, IndexUpgrader" "$WORK/$v-mixed" "$WORK/$v-mixed-upgraded"
done

echo "verify-bwc-merge: $passed passed, $failed failed"
[ "$failed" -eq 0 ]
