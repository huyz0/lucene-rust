#!/usr/bin/env bash
# Build the benchmark corpus: a real Lucene 10.5.0 index that both engines read.
#
# Two variants are generated, because a port can plausibly win one and lose the
# other and only measuring one would hide it:
#   segmented/  -- many small segments, as a shard looks after refreshes
#   merged/     -- force-merged to one segment, raw decode throughput
#
# The corpus is NOT checked in (it is gigabytes). This script and the manifest
# it writes are what make a published number reproducible.
#
# Usage: scripts/bench-corpus.sh [--docs N] [--seed S] [--out DIR] [--jars DIR]
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
DOCS=5000000
SEED=20260827
OUT="$PWD/benchmarks/.corpus"
JARS="$PWD/fixtures/.jars"

while [ $# -gt 0 ]; do
  case "$1" in
    --docs) DOCS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --out)  OUT="$2";  shift 2 ;;
    --jars) JARS="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "bench-corpus: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core lucene-analysis-common)

CLASSES=$(mktemp -d)
trap 'rm -rf "$CLASSES"' EXIT
javac -nowarn -cp "$CP" -d "$CLASSES" benchmarks/corpus/src/GenCorpus.java

for variant in segmented merged; do
  args=("$OUT/$variant" "$DOCS" "$SEED")
  [ "$variant" = merged ] && args+=(--force-merge)
  echo "bench-corpus: building $variant ($(printf "%'d" "$DOCS") docs)"
  java --enable-native-access=ALL-UNNAMED -Xmx4g \
       -cp "$CLASSES:$CP" GenCorpus "${args[@]}"
done

echo "bench-corpus: corpus in $OUT"
