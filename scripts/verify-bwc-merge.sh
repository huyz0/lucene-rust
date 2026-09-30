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
#
# BwcMergeCheck requires CheckIndex to be clean, every segment the merge wrote
# to be Lucene104 (postings Lucene104, vectors Lucene99HnswVectorsFormat), and
# every live document of the input to be in the output with all of its
# content: postings with positions, offsets and payloads, norms, all five
# doc-values types, points, stored fields, term vectors and vectors.
#
# Usage: scripts/verify-bwc-merge.sh [--only <version>] [--keep]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

VERSIONS=(9.0.0 9.1.0 9.3.0 9.4.2 9.8.0 9.11.1 9.12.2 10.0.0 10.2.2 10.4.0)
ONLY=""
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
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
cargo build --quiet -p lucene-search --example bwc_merge
MERGE="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/debug/examples/bwc_merge"

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

for v in "${VERSIONS[@]}"; do
  [ -z "$ONLY" ] || [ "$v" = "$ONLY" ] || continue
  src="fixtures/data/bwc/$v"
  echo "verify-bwc-merge: $v"

  "$MERGE" "$src" "$WORK/$v-force" force > /dev/null
  check "$v force_merge(1)" "$src" "$WORK/$v-force"

  mkdir -p "$WORK/$v-mixed"
  find "$src" -maxdepth 1 -type f ! -name '*.txt' -exec cp {} "$WORK/$v-mixed/" \;
  java -cp "$WORK/classes:$CP" BwcAppend "$WORK/$v-mixed" 3 2>/dev/null
  "$MERGE" "$WORK/$v-mixed" "$WORK/$v-mixed-merged" policy > /dev/null
  check "$v + 3 Lucene 10.5.0 segments, ordinary merge" "$WORK/$v-mixed" "$WORK/$v-mixed-merged"
done

echo "verify-bwc-merge: $passed passed, $failed failed"
[ "$failed" -eq 0 ]
