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
# Usage: scripts/gen-bwc-fixtures.sh --only <version>
#        scripts/gen-bwc-fixtures.sh --check
#        scripts/gen-bwc-fixtures.sh --big <version>
#        scripts/gen-bwc-fixtures.sh --quantized <version>
#        scripts/gen-bwc-fixtures.sh --dump <fixture dir>
#
# --big writes fixtures/data/bwc-big/<version>/ instead: fixtures/bwc/BwcBig.java,
# one 20,000-document segment whose terms need every level of the retired
# postings formats' skip data (trailing multi-level lists, inline level 1),
# which the 3,000-document corpus never reaches. Versions: 9.0.0 (Lucene90),
# 9.11.1 (Lucene99), 9.12.2 (Lucene912), 10.2.2 (Lucene101).
#
# --quantized writes fixtures/data/bwc-quantized/<version>/ instead:
# fixtures/bwc/BwcQuantized.java, per-field quantized vector fields --
# Lucene99(Hnsw)ScalarQuantizedVectorsFormat and, from 10.2,
# Lucene102(Hnsw)BinaryQuantizedVectorsFormat -- then BwcDump. Versions:
# 9.9.2 (the scalar format's version 0), 9.12.2 (version 1: 4-bit, compressed,
# flat), 10.2.2 (all of it plus the binary formats).
#
# --dump re-runs only BwcDump over an existing fixture directory, rewriting
# its expected.txt and nothing else (the index is not touched, so every line
# BwcDump already wrote comes back byte for byte) -- for a new line kind.
#
# A regenerated index differs from the committed one byte for byte (segment
# ids are random), and so do its digests of file-level content; regenerate
# deliberately, one version at a time, like scripts/gen-fixtures.sh --only.
#
# --check regenerates nothing. For every version it re-runs BwcDump (10.5.0 +
# backward-codecs) over a copy of the committed index and asserts the fresh
# expected.txt is byte-identical to the committed one, that written_by.txt
# names that version, and that the directory holds exactly the version
# subdirectories listed here. It proves expected.txt is what Lucene reads out
# of the committed bytes. It cannot prove the bytes were written by the old
# jar -- that needs a regeneration, whose output is random -- and it cannot see
# an index regenerated in place: fixtures/segment-ids.txt (which covers bwc/,
# checked by scripts/gen-fixtures.sh --check) is what catches that.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

VERSIONS=(9.0.0 9.1.0 9.3.0 9.4.2 9.8.0 9.11.1 9.12.2 10.0.0 10.2.2 10.4.0)
QUANTIZED_VERSIONS=(9.9.2 9.12.2 10.2.2)
ONLY=""
CHECK=0
BIG=""
QUANTIZED=""
DUMP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    --big) BIG="$2"; shift 2 ;;
    --quantized) QUANTIZED="$2"; shift 2 ;;
    --dump) DUMP="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "gen-bwc-fixtures: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ -n "$BIG" ]; then
  JARS="$PWD/fixtures/.jars"
  jar="$JARS/bwc/lucene-core-$BIG.jar"
  [ -s "$jar" ] || { echo "gen-bwc-fixtures: no $jar (run --only $BIG once first)" >&2; exit 2; }
  WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
  out="fixtures/data/bwc-big/$BIG"
  rm -rf "${out:?}"
  javac -nowarn -proc:none -cp "$jar" -d "$WORK/big" fixtures/bwc/BwcBig.java
  java -cp "$WORK/big:$jar" BwcBig "$out" 2>/dev/null
  rm -f "$out/write.lock"
  python3 scripts/fixture-segment-ids.py fixtures/data > fixtures/segment-ids.txt
  echo "gen-bwc-fixtures: $BIG -> $out (segment-ids.txt refreshed)"
  exit 0
fi
if [ -n "$DUMP" ] || [ -n "$QUANTIZED" ]; then
  JARS="$PWD/fixtures/.jars"
  # shellcheck source=scripts/lib-lucene-jars.sh
  source "$(dirname "$0")/lib-lucene-jars.sh"
  DUMP_CP=$(lucene_classpath lucene-core lucene-backward-codecs)
  WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
  javac -nowarn -proc:none -cp "$DUMP_CP" -d "$WORK/dump" fixtures/bwc/BwcDump.java
  if [ -n "$QUANTIZED" ]; then
    case " ${QUANTIZED_VERSIONS[*]} " in
      *" $QUANTIZED "*) ;;
      *) echo "gen-bwc-fixtures: --quantized takes one of ${QUANTIZED_VERSIONS[*]}" >&2; exit 2 ;;
    esac
    jar="$JARS/bwc/lucene-core-$QUANTIZED.jar"
    if [ ! -s "$jar" ]; then
      mkdir -p "$JARS/bwc"
      curl -fsSL --retry 5 -o "$jar" "$MAVEN_BASE/lucene-core/$QUANTIZED/lucene-core-$QUANTIZED.jar"
    fi
    DUMP="fixtures/data/bwc-quantized/$QUANTIZED"
    rm -rf "${DUMP:?}"
    javac -nowarn -proc:none -cp "$jar" -d "$WORK/q" fixtures/bwc/BwcQuantized.java
    java -cp "$WORK/q:$jar" BwcQuantized "$DUMP" 2>/dev/null
    rm -f "$DUMP/write.lock"
  fi
  java "${LUCENE_FIXTURE_JVM_OPTS[@]}" -cp "$WORK/dump:$DUMP_CP" BwcDump "$DUMP" 2>/dev/null
  [ -n "$QUANTIZED" ] && python3 scripts/fixture-segment-ids.py fixtures/data > fixtures/segment-ids.txt
  echo "gen-bwc-fixtures: $DUMP ($(wc -l < "$DUMP/expected.txt") lines)"
  exit 0
fi
if [ "$CHECK" = 1 ] && [ -n "$ONLY" ]; then
  echo "gen-bwc-fixtures: --check verifies every version; it cannot be combined with --only" >&2
  exit 2
fi
if [ "$CHECK" = 0 ] && [ -z "$ONLY" ]; then
  echo "gen-bwc-fixtures: refusing to regenerate every version; pass --only <version> (one of ${VERSIONS[*]})" >&2
  exit 2
fi

JARS="$PWD/fixtures/.jars"
# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
DUMP_CP=$(lucene_classpath lucene-core lucene-backward-codecs)

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
javac -nowarn -proc:none -cp "$DUMP_CP" -d "$WORK/dump" fixtures/bwc/BwcDump.java

if [ "$CHECK" = 1 ]; then
  status=0
  committed=$(cd fixtures/data/bwc && find . -mindepth 1 -maxdepth 1 -type d | sed 's|^\./||' | sort)
  listed=$(printf '%s\n' "${VERSIONS[@]}" | sort)
  if [ "$committed" != "$listed" ]; then
    echo "  VERSION SET: fixtures/data/bwc holds [$(echo $committed)], this script lists [$(echo $listed)]"
    status=1
  fi
  committed=$(cd fixtures/data/bwc-quantized && find . -mindepth 1 -maxdepth 1 -type d | sed 's|^\./||' | sort)
  listed=$(printf '%s\n' "${QUANTIZED_VERSIONS[@]}" | sort)
  if [ "$committed" != "$listed" ]; then
    echo "  VERSION SET: fixtures/data/bwc-quantized holds [$(echo $committed)], this script lists [$(echo $listed)]"
    status=1
  fi
  for src in "${VERSIONS[@]/#/fixtures/data/bwc/}" "${QUANTIZED_VERSIONS[@]/#/fixtures/data/bwc-quantized/}"; do
    v=${src##*/}
    [ -d "$src" ] || { echo "  MISSING: $src"; status=1; continue; }
    if [ "$(cat "$src/written_by.txt")" != "$v" ]; then
      echo "  WRITTEN_BY: $src/written_by.txt says '$(cat "$src/written_by.txt")', not $v"
      status=1
    fi
    chk="$WORK/check-${src//\//_}"
    cp -r "$src" "$chk"
    rm -f "$chk/expected.txt"
    java "${LUCENE_FIXTURE_JVM_OPTS[@]}" -cp "$WORK/dump:$DUMP_CP" BwcDump "$chk" 2>/dev/null
    if cmp -s "$chk/expected.txt" "$src/expected.txt"; then
      echo "gen-bwc-fixtures: $src ok ($(wc -l < "$src/expected.txt") lines)"
    else
      echo "  MISMATCH: $src/expected.txt is not what BwcDump reads from the committed index"
      diff "$src/expected.txt" "$chk/expected.txt" | head -10 | sed 's/^/      /' || true
      status=1
    fi
  done
  [ "$status" = 0 ] && echo "gen-bwc-fixtures: ok" || echo "gen-bwc-fixtures: FAILED"
  exit "$status"
fi

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
  java "${LUCENE_FIXTURE_JVM_OPTS[@]}" -cp "$WORK/dump:$DUMP_CP" BwcDump "$out" 2>/dev/null
  echo "gen-bwc-fixtures: $v -> $out ($(wc -l < "$out/expected.txt") lines)"
done
# The regenerated index has fresh segment ids; record them, as
# scripts/gen-fixtures.sh does after a write-mode run.
python3 scripts/fixture-segment-ids.py fixtures/data > fixtures/segment-ids.txt
