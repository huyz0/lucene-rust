#!/usr/bin/env bash
# Regenerate the Java-produced differential-testing fixtures under fixtures/data.
#
# Two modes:
#   (default)  regenerate in place into fixtures/data
#   --check    regenerate into temp dirs and verify the committed fixtures are
#              still what Lucene 10.5.0 actually produces (see "What --check
#              can and cannot prove" below)
#
# Usage:
#   scripts/gen-fixtures.sh [--out DIR] [--check] [--jars DIR]
#
#   --out DIR    write fixtures here (default: fixtures/data)
#   --check      verification mode; implies a temp --out, leaves the tree alone
#   --jars DIR   look for the Lucene jars here before the Gradle cache;
#                also where they are downloaded to (default: fixtures/.jars)
#
# ---------------------------------------------------------------------------
# What --check can and cannot prove
#
# Fixtures written through a real Lucene IndexWriter are NOT byte-reproducible:
# Lucene stamps a random segment ID (StringHelper.randomId()) into every index
# header, so 25 of the 48 fixture entries differ on every run by design. Only
# the 23 entries written as raw bytes (primitives, FSTs, analysis) are stable.
#
# So --check does not diff everything against the committed tree. It:
#   1. generates twice, and calls a file deterministic iff the two runs agree;
#   2. asserts every deterministic file matches the committed copy byte for
#      byte -- this is what catches a hand-edit;
#   3. asserts the generated file tree matches the committed tree, so a
#      generator that silently stops emitting a file is caught even where the
#      bytes cannot be compared.
#
# Deriving the deterministic set by generating twice, rather than maintaining a
# hand-written list, means the check stays correct as fixtures are added.
#
# Not yet checkable: running the Rust suite against freshly generated fixtures.
# crates/lucene-ffi/src/segment.rs hardcodes the committed blocktree fixture's
# segment ID, so fresh bytes fail those tests. See docs/milestones/
# m0-ci-and-green-tree.md ("Findings") for the follow-up.
# ---------------------------------------------------------------------------
set -euo pipefail

# Modules the generators actually need. lucene-queries is required by
# GenBlockTree (org.apache.lucene.queries.spans); the fixtures README used to
# document only lucene-core + lucene-analysis-common, which no longer compiles.
LUCENE_MODULES=(lucene-core lucene-analysis-common lucene-queries)

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
OUT="$FIXTURES/data"
JARS="$FIXTURES/.jars"
CHECK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out)   OUT="$2"; shift 2 ;;
    --jars)  JARS="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "gen-fixtures: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --- resolve the Lucene jars -------------------------------------------------
# Prefer --jars, then the Gradle cache (fast, local), then Maven Central (CI).
# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath "${LUCENE_MODULES[@]}")

# --- compile -----------------------------------------------------------------
CLASSES=$(mktemp -d)
trap 'rm -rf "$CLASSES" ${TMP_A:-} ${TMP_B:-}' EXIT
javac -nowarn -cp "$CP" -d "$CLASSES" "$FIXTURES"/src/*.java

# Generators, then manifest appenders. Both lists come from the filesystem so a
# newly added program is picked up without editing this script.
#
# Order matters: Append*Manifest programs open an already-generated index
# read-only and append cross-engine ground truth to its manifest.properties
# (they deliberately do not regenerate the index). Running only the Gen*
# programs yields an incomplete manifest -- 239 keys instead of 402 for
# blocktree_index.
mapfile -t GENERATORS < <(cd "$FIXTURES/src" && ls Gen*.java | sed 's/\.java$//' | sort)
mapfile -t APPENDERS  < <(cd "$FIXTURES/src" && ls Append*.java | sed 's/\.java$//' | sort)

generate_into() {
  local dest="$1"
  mkdir -p "$dest"
  for cls in "${GENERATORS[@]}" "${APPENDERS[@]}"; do
    java --enable-native-access=ALL-UNNAMED -cp "$CLASSES:$CP" "$cls" "$dest" >/dev/null
  done
  # IndexWriter leaves a zero-byte write.lock behind in every index it creates.
  # It is a lock artifact, not a fixture, and nothing in crates/ reads it --
  # drop it so an in-place regeneration leaves a git-clean tree.
  find "$dest" -name 'write.lock' -type f -delete
}

if [ "$CHECK" -eq 0 ]; then
  echo "gen-fixtures: generating ${#GENERATORS[@]} generators + ${#APPENDERS[@]} appenders into $OUT"
  generate_into "$OUT"
  echo "gen-fixtures: ok"
  exit 0
fi

# --- check mode --------------------------------------------------------------
TMP_A=$(mktemp -d); TMP_B=$(mktemp -d)
echo "gen-fixtures: generating twice to derive the deterministic subset"
generate_into "$TMP_A"
generate_into "$TMP_B"

status=0
deterministic=0; nondeterministic=0; missing=0; mismatched=0; extra=0

# write.lock is an IndexWriter lock artifact, not a fixture: generators leave it
# behind and it is deliberately not committed.
GENERATED_NOISE='(^|/)write\.lock$'

while IFS= read -r rel; do
  if [[ "$rel" =~ $GENERATED_NOISE ]]; then continue; fi
  a="$TMP_A/$rel"; b="$TMP_B/$rel"; c="$OUT/$rel"
  if [ ! -e "$c" ]; then
    echo "  MISSING from committed fixtures: $rel"; missing=$((missing+1)); status=1; continue
  fi
  if cmp -s "$a" "$b"; then
    deterministic=$((deterministic+1))
    if ! cmp -s "$a" "$c"; then
      echo "  MISMATCH (deterministic file differs from committed): $rel"
      mismatched=$((mismatched+1)); status=1
    fi
  else
    nondeterministic=$((nondeterministic+1))
  fi
done < <(cd "$TMP_A" && find . -type f | sed 's|^\./||' | sort)

# Files committed under fixtures/data that no Java generator produces. These are
# written by Rust examples (see scripts/verify-write-path.sh) and are expected.
RUST_WRITTEN='^sparse_numeric_doc_values/'
while IFS= read -r rel; do
  [ -e "$TMP_A/$rel" ] && continue
  if [[ "$rel" =~ $GENERATED_NOISE ]]; then continue; fi
  if [[ "$rel" =~ $RUST_WRITTEN ]]; then continue; fi
  echo "  EXTRA in committed fixtures, produced by no generator: $rel"
  extra=$((extra+1)); status=1
done < <(cd "$OUT" && find . -type f | sed 's|^\./||' | sort)

echo
echo "gen-fixtures --check summary:"
echo "  deterministic files verified byte-identical : $((deterministic - mismatched))"
echo "  non-deterministic files (random segment id) : $nondeterministic"
echo "  deterministic mismatches                    : $mismatched"
echo "  missing from committed tree                 : $missing"
echo "  unexplained extras in committed tree        : $extra"
[ "$status" -eq 0 ] && echo "gen-fixtures: ok" || echo "gen-fixtures: FAILED"
exit "$status"
