#!/usr/bin/env bash
# Backward-codecs fixtures (M8): one index per Lucene release whose default
# codec an OpenSearch 2.x/3.x cluster can hold, each written by *that
# release's own jar*, then described by Lucene 10.5.0 + backward-codecs.
#
#   fixtures/bwc/BwcWrite.java  compiled and run against lucene-core-<v>.jar
#   fixtures/bwc/BwcDump.java   compiled and run against 10.5.0 + backward-codecs
#
# Output: fixtures/data/bwc/<version>/ (the index, written_by.txt,
# expected.txt). The versions are one per codec era: Lucene90 (9.0), 91
# (9.1), 92 (9.3), 94 (9.4), 95 (9.8), 99 (9.11), 912 (9.12), 100 (10.0),
# 101 (10.2), 104 (10.4 -- the current codec already, a cross-check).
#
# Usage: scripts/gen-bwc-fixtures.sh [--only <version>]
#
# A regenerated index differs from the committed one byte for byte (segment
# ids are random), and so do its digests of file-level content; regenerate
# deliberately, one version at a time, like scripts/gen-fixtures.sh --only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

VERSIONS=(9.0.0 9.1.0 9.3.0 9.4.2 9.8.0 9.11.1 9.12.2 10.0.0 10.2.2 10.4.0)
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "gen-bwc-fixtures: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -z "$ONLY" ]; then
  echo "gen-bwc-fixtures: refusing to regenerate every version; pass --only <version> (one of ${VERSIONS[*]})" >&2
  exit 2
fi

JARS="$PWD/fixtures/.jars"
# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
DUMP_CP=$(lucene_classpath lucene-core lucene-backward-codecs)

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
javac -nowarn -proc:none -cp "$DUMP_CP" -d "$WORK/dump" fixtures/bwc/BwcDump.java

for v in "${VERSIONS[@]}"; do
  [ "$v" = "$ONLY" ] || continue
  jar="$JARS/bwc/lucene-core-$v.jar"
  if [ ! -s "$jar" ]; then
    mkdir -p "$JARS/bwc"
    curl -fsSL --retry 5 -o "$jar" "$MAVEN_BASE/lucene-core/$v/lucene-core-$v.jar"
  fi
  out="fixtures/data/bwc/$v"
  rm -rf "${out:?}"
  javac -nowarn -proc:none -cp "$jar" -d "$WORK/$v" fixtures/bwc/BwcWrite.java
  java -cp "$WORK/$v:$jar" BwcWrite "$out" 2>/dev/null
  rm -f "$out/write.lock"
  java -cp "$WORK/dump:$DUMP_CP" BwcDump "$out" 2>/dev/null
  echo "gen-bwc-fixtures: $v -> $out ($(wc -l < "$out/expected.txt") lines)"
done
