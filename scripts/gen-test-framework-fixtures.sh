#!/usr/bin/env bash
# Test-framework fixtures (M14.h): the Java twins in fixtures/src/testframework/
# run against Lucene 10.5.0's lucene-test-framework jar (with lucene-misc,
# whose SweetSpotSimilarity GenTestFrameworkSearch explains; plus the JUnit,
# Hamcrest and randomizedtesting jars it needs on the classpath -- generator
# only, never shipped; docs/licences.md) and record what Lucene's test
# framework draws from seeded randoms. The Rust tests in
# crates/lucene-test-framework/tests/ replay the same seeds and compare.
#
# Output: fixtures/testframework/data/ (outside fixtures/data, whose own
# --check flags files it did not generate).
#
# Usage: scripts/gen-test-framework-fixtures.sh --only <Gen name>   (repeatable)
#        scripts/gen-test-framework-fixtures.sh --check
#        scripts/gen-test-framework-fixtures.sh --list
#
# --check regenerates every generator into a scratch directory and diffs it
# against the committed data: the generators are deterministic (fixed seeds,
# no segment ids), so any difference means the committed fixture is not what
# Lucene 10.5.0 produces.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=fixtures/src/testframework
DATA=fixtures/testframework/data
EXCERPT=crates/lucene-test-framework/data/europarl.lines.excerpt.txt.gz

ONLY=()
CHECK=0
LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY+=("${2#Gen}"); shift 2 ;;
    --check) CHECK=1; shift ;;
    --list) LIST=1; shift ;;
    *) echo "gen-test-framework-fixtures: unknown argument $1" >&2; exit 2 ;;
  esac
done

ALL=()
for f in "$SRC"/Gen*.java; do
  n=$(basename "$f" .java)
  ALL+=("${n#Gen}")
done
if [ "$LIST" = 1 ]; then
  printf 'Gen%s\n' "${ALL[@]}"
  exit 0
fi
if [ "$CHECK" = 1 ] && [ ${#ONLY[@]} -gt 0 ]; then
  echo "gen-test-framework-fixtures: --check verifies every generator; it cannot be combined with --only" >&2
  exit 2
fi
if [ "$CHECK" = 0 ] && [ ${#ONLY[@]} -eq 0 ]; then
  echo "gen-test-framework-fixtures: pass --only <name> (one of: ${ALL[*]}) or --check" >&2
  exit 2
fi
for n in "${ONLY[@]}"; do
  [ -f "$SRC/Gen$n.java" ] || { echo "gen-test-framework-fixtures: no generator Gen$n" >&2; exit 2; }
done

JARS="${JARS:-$PWD/fixtures/.jars}"
# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework lucene-misc):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
# Let javac resolve helpers from the source path for selected generators.
# A full check still compiles every source, including unselected failures.
SOURCES=()
if [ "$CHECK" = 1 ]; then
  SOURCES=("$SRC"/*.java)
else
  for n in "${ONLY[@]}"; do SOURCES+=("$SRC/Gen$n.java"); done
fi
javac -nowarn -proc:none -cp "$CP" -sourcepath "$SRC" -d "$WORK/classes" "${SOURCES[@]}"

run() { # <generator> <outdir>
  local assertion_opts=()
  # These generators record assertions in Lucene's asserting wrappers.
  case "$1" in
    AssertingCodec|AssertingImpacts|AssertingDocValues|AssertingDocValuesIterators|AssertingKnnVectors|AssertingLiveDocs|LiveDocsWriterRouting|AssertingNorms|AssertingPoints|AssertingPostings|AssertingStoredFields|StoredFieldsWriterRouting|AssertingTermVectors|CompressingCodec|ConfigurableM|CorruptUnknownFiles|CrankyDocValues|CrankyPostings|ScorerIndexSearcher|NoUnreferencedFilesOnClose) assertion_opts=(-ea) ;;
    FstTesterLists) assertion_opts=(-ea -Dtests.multiplier=1) ;;
  esac
  java "${LUCENE_FIXTURE_JVM_OPTS[@]}" "${assertion_opts[@]}" -cp "$WORK/classes:$CP" "Gen$1" "$2" "$EXCERPT"
}

if [ "$CHECK" = 1 ]; then
  status=0
  for n in "${ALL[@]}"; do
    run "$n" "$WORK/check/data"
  done
  if ! diff -r "$WORK/check/data" "$DATA" > "$WORK/diff.txt"; then
    echo "gen-test-framework-fixtures: $DATA differs from what Lucene 10.5.0 generates:"
    head -40 "$WORK/diff.txt"
    status=1
  fi
  [ "$status" = 0 ] && echo "gen-test-framework-fixtures: ok (${#ALL[@]} generators)"
  exit "$status"
fi

for n in "${ONLY[@]}"; do
  run "$n" "$DATA"
  echo "gen-test-framework-fixtures: Gen$n -> $DATA"
done
