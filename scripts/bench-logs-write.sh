#!/usr/bin/env bash
# Write-path cost on log-shaped documents, this port against stock Lucene:
# indexing with a 1 s refresh, then a force merge to one segment. See
# benchmarks/rust-runner/src/logs_write.rs for the document and writer setup.
#
#   scripts/bench-logs-write.sh [docs] [out dir]
#
# RUNS (default 2) runs of each engine, alternating. Rows: engine, docs,
# index s, docs/s, total s (commit and merges included), CPU s, refreshes,
# refresh mean ms, MB on disk; then the force merge: took s, CPU s, MB.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DOCS="${1:-2000000}"
OUT="${2:-target/bench-logs-write}"
THREADS="${THREADS:-4}"
REFRESH_MS="${REFRESH_MS:-1000}"
RUNS="${RUNS:-2}"
mkdir -p "$OUT/classes" "$OUT/idx"
OUT=$(cd "$OUT" && pwd)

JARS="$PWD/fixtures/.jars"
source scripts/lib-lucene-jars.sh
CP=$(lucene_classpath lucene-core lucene-analysis-common)
javac -nowarn -cp "$CP" -d "$OUT/classes" benchmarks/java-runner/src/LogsWriteBench.java
JVM=(java -Xmx6g --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp "$OUT/classes:$CP")
cargo build --release -q --manifest-path benchmarks/rust-runner/Cargo.toml --bin logs-write
RUST="$PWD/benchmarks/rust-runner/target/release/logs-write"
[[ -n "${CARGO_TARGET_DIR:-}" ]] && RUST="$CARGO_TARGET_DIR/release/logs-write"

corpus="$OUT/logs-$DOCS.tsv"
[[ -f "$corpus" ]] || "$RUST" corpus "$corpus" "$DOCS" 7

echo "engine,docs,index_s,docs_per_s,total_s,cpu_s,refreshes,refresh_mean_ms,mb" | tee "$OUT/index.csv"
for _ in $(seq "$RUNS"); do
    "$RUST" index "$corpus" "$OUT/idx/rust" "$THREADS" "$REFRESH_MS" | tee -a "$OUT/index.csv"
    rm -rf "$OUT/idx/java"
    "${JVM[@]}" LogsWriteBench index "$corpus" "$OUT/idx/java" "$THREADS" "$REFRESH_MS" 2>/dev/null \
        | grep '^java' | tee -a "$OUT/index.csv"
done
echo "engine,forcemerge,took_s,cpu_s,mb" | tee "$OUT/forcemerge.csv"
"$RUST" forcemerge "$OUT/idx/rust" 1 | tee -a "$OUT/forcemerge.csv"
"${JVM[@]}" LogsWriteBench forcemerge "$OUT/idx/java" 1 2>/dev/null | grep '^java' | tee -a "$OUT/forcemerge.csv"
echo "bench-logs-write: results in $OUT"
