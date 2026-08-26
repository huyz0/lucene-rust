#!/usr/bin/env bash
# Run the reverse-direction differential tests: Rust writes real Lucene bytes,
# and a Java program confirms unmodified Lucene 10.5.0 can read them back.
#
# Every fixture under fixtures/data is Java-writes/Rust-reads. These verifiers
# are the opposite direction, and they are the only automated evidence that this
# port's write path emits bytes real Lucene accepts. Round-tripping through this
# port's own reader cannot catch a misreading of the spec that the reader and
# writer share; these can.
#
# Usage: scripts/verify-write-path.sh [--jars DIR] [--keep]
#   --jars DIR  look for the Lucene jars here (default: fixtures/.jars)
#   --keep      keep the generated fixtures instead of deleting them on exit
#
# Coverage gap, deliberate: there is no verifier for the postings / term
# dictionary (.doc/.tim/.tip/.tmd) -- the most important format this port
# writes. Closing it is task T3.1, see docs/milestones/m3-write-path-proven.md.
set -euo pipefail

LUCENE_VERSION="10.5.0"
LUCENE_MODULES=(lucene-core lucene-analysis-common lucene-queries)
MAVEN_BASE="https://repo1.maven.org/maven2/org/apache/lucene"

cd "$(git rev-parse --show-toplevel)"
FIXTURES="$PWD/fixtures"
JARS="$FIXTURES/.jars"
KEEP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --jars) JARS="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "verify-write-path: unknown argument: $1" >&2; exit 2 ;;
  esac
done

resolve_jar() {
  local module="$1"
  local jar="$module-$LUCENE_VERSION.jar"
  local found=""
  if [ -f "$JARS/$jar" ]; then echo "$JARS/$jar"; return; fi
  found=$(find "$HOME/.gradle/caches" -name "$jar" ! -name '*sources*' ! -name '*javadoc*' 2>/dev/null | head -1 || true)
  if [ -n "$found" ]; then echo "$found"; return; fi
  mkdir -p "$JARS"
  echo "verify-write-path: downloading $jar from Maven Central" >&2
  curl -fsSL -o "$JARS/$jar" "$MAVEN_BASE/$module/$LUCENE_VERSION/$jar"
  echo "$JARS/$jar"
}

CP=""
for m in "${LUCENE_MODULES[@]}"; do CP="$CP${CP:+:}$(resolve_jar "$m")"; done

WORK=$(mktemp -d)
CLASSES=$(mktemp -d)
cleanup() { [ "$KEEP" -eq 1 ] && echo "verify-write-path: fixtures kept in $WORK" || rm -rf "$WORK"; rm -rf "$CLASSES"; }
trap cleanup EXIT

# crate | example | output subdir | verifier class
CASES=(
  "lucene-codecs|write_stored_fields_fixture|stored-fields|VerifyStoredFields"
  "lucene-codecs|write_field_infos_fixture|field-infos|VerifyFieldInfos"
  "lucene-index|write_segment_info_fixture|segment-info|VerifySegmentInfo"
  "lucene-index|write_segment_infos_fixture|segment-infos|VerifySegmentInfos"
  "lucene-index|write_multi_segment_commit_fixture|multi-segment|VerifySegmentInfos"
  "lucene-codecs|write_points_fixture|points|VerifyPoints"
  "lucene-codecs|write_term_vectors_fixture|term-vectors|VerifyTermVectors"
  "lucene-codecs|write_doc_values_fixture|doc-values|VerifyDocValues"
  "lucene-codecs|write_sparse_numeric_doc_values_fixture|sparse-numeric-dv|VerifySparseNumericDocValues"
  "lucene-codecs|write_norms_fixture|norms|VerifyNorms"
  "lucene-codecs|write_live_docs_fixture|live-docs|VerifyLiveDocs"
  "lucene-codecs|write_compound_format_fixture|compound-format|VerifyCompoundFormat"
  "lucene-codecs|write_fst_fixture|fst|VerifyFst"
)

echo "verify-write-path: compiling verifiers"
javac -nowarn -cp "$CP" -d "$CLASSES" "$FIXTURES"/src/Verify*.java

echo "verify-write-path: writing fixtures from Rust and verifying with Lucene $LUCENE_VERSION"
failed=0
for case in "${CASES[@]}"; do
  IFS='|' read -r crate example subdir verifier <<<"$case"
  out="$WORK/$subdir"
  if ! cargo run --quiet --release -p "$crate" --example "$example" -- "$out" >/dev/null; then
    echo "  FAIL (rust write)   $example"; failed=$((failed+1)); continue
  fi
  if java --enable-native-access=ALL-UNNAMED -cp "$CLASSES:$CP" "$verifier" "$out" >/dev/null 2>&1; then
    echo "  ok                  $verifier <- $example"
  else
    echo "  FAIL (java verify)  $verifier <- $example"
    java --enable-native-access=ALL-UNNAMED -cp "$CLASSES:$CP" "$verifier" "$out" 2>&1 | sed 's/^/      /' | tail -15
    failed=$((failed+1))
  fi
done

echo
if [ "$failed" -eq 0 ]; then
  echo "verify-write-path: ok (${#CASES[@]}/${#CASES[@]} passed)"
else
  echo "verify-write-path: FAILED ($failed of ${#CASES[@]} cases)"
fi
exit "$failed"
