#!/usr/bin/env bash
# What partitioning a shard by tenant costs and buys on the write side:
# indexing throughput, refresh and commit times, natural merges, a forced
# merge, then query latency, per layout -- arrival order (`plain`), index
# sort on the tenant (`tenant`), tenant then time (`composite`) and, Rust
# only, each tenant's documents in segments of their own (`segments`) --
# for stock Lucene and this port over one corpus.
#
#   scripts/bench-partition-cost.sh fts [DOCS] [TENANTS] [OUT_DIR]       # default 2000000, 1000
#   scripts/bench-partition-cost.sh vectors [DOCS] [TENANTS] [OUT_DIR]   # default 500000, 1000
#
# The corpus is PartitionBench's (Zipf tenant sizes, documents interleaved
# across tenants on arrival). Every run indexes from THREADS threads (default
# 4) with a reader refreshed every REFRESH_MS (default 1000), merges on the
# default TieredMergePolicy and ConcurrentMergeScheduler, then commits; the
# index is then force-merged to one segment (one per tenant for `segments`).
# `fts` queries the natural and the merged index with
# scripts/bench-fts-partition.sh's query set; `vectors` (HNSW for both
# engines, IVF and IVF with clusters per tenant for this port) runs
# tenant-filtered top-10 searches on this port. LAYOUTS and FORMATS narrow a
# run; RAM_MB sets the indexing buffer (default 256). Never run cargo or gradle while it runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

KIND="${1:?fts or vectors}"
DEFAULT_DOCS=2000000
[[ "$KIND" == vectors ]] && DEFAULT_DOCS=500000
DOCS="${2:-$DEFAULT_DOCS}"
TENANTS="${3:-1000}"
OUT="${4:-target/bench-partition-cost}"
THREADS="${THREADS:-4}"
REFRESH_MS="${REFRESH_MS:-1000}"
PASSES="${PASSES:-5}"
mkdir -p "$OUT/classes" "$OUT/idx"
OUT=$(cd "$OUT" && pwd)

JARS="$PWD/fixtures/.jars"
# shellcheck source=scripts/lib-lucene-jars.sh
source scripts/lib-lucene-jars.sh
CP=$(lucene_classpath lucene-core lucene-analysis-common)
javac -nowarn -cp "$CP" -d "$OUT/classes" benchmarks/java-runner/src/PartitionBench.java \
    benchmarks/java-runner/src/PartitionCostBench.java
JVM=(java -Xmx4g --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp "$OUT/classes:$CP")
cargo build --release -q -p lucene-search --example partition_cost --example fts_partition
RUST="$PWD/target/release/examples/partition_cost"
FTS="$PWD/target/release/examples/fts_partition"

corpus="$OUT/corpus-$DOCS-$TENANTS.tsv"
queries="$OUT/queries-$DOCS-$TENANTS.tsv"
[[ -f "$corpus" ]] || "${JVM[@]}" PartitionCostBench corpus "$corpus" "$queries" "$DOCS" "$TENANTS"

INDEX_HEADER="engine,layout,format,docs,index_s,docs_per_s,commit_s,refreshes,refresh_mean_ms,refresh_p50_ms,refresh_p99_ms,merges,merge_s,segments,total_s,mb"
FM_HEADER="engine,layout,forcemerge,segments,took_s,mb"

if [[ "$KIND" == fts ]]; then
    echo "$INDEX_HEADER" > "$OUT/fts-index.csv"
    echo "$FM_HEADER,segments_after" > "$OUT/fts-forcemerge.csv"
    for engine in ${ENGINES:-rust java}; do
        layouts="${LAYOUTS:-plain tenant composite segments}"
        for layout in $layouts; do
            [[ "$engine" == java && "$layout" == segments ]] && continue
            idx="$OUT/idx/fts-$engine-$layout"
            if [[ "$engine" == rust ]]; then
                "$RUST" index "$corpus" "$idx" "$layout" "$THREADS" "$REFRESH_MS" | tee -a "$OUT/fts-index.csv"
                q=("$FTS" query "$idx" "$queries" "$PASSES")
            else
                rm -rf "$idx"
                "${JVM[@]}" PartitionCostBench index "$corpus" "$idx" "$layout" "$THREADS" "$REFRESH_MS" | tee -a "$OUT/fts-index.csv"
                q=("${JVM[@]}" PartitionBench query "$idx" "$queries" "$PASSES")
            fi
            "${q[@]}" > "$OUT/fts-query-$engine-$layout-natural.csv"
            if [[ "$engine" == rust ]]; then
                "$RUST" forcemerge "$idx" "$layout" 1 | tee -a "$OUT/fts-forcemerge.csv"
            else
                "${JVM[@]}" PartitionCostBench forcemerge "$idx" "$layout" 1 | tee -a "$OUT/fts-forcemerge.csv"
            fi
            "${q[@]}" > "$OUT/fts-query-$engine-$layout-merged.csv"
        done
    done
else
    dim=96
    vecs="$OUT/vectors-$DOCS-$dim.fbin"
    qvecs="$OUT/queries-$dim.fbin"
    [[ -f "$vecs" ]] || "$RUST" vectors "$vecs" "$DOCS" "$dim" 11
    [[ -f "$qvecs" ]] || "$RUST" vectors "$qvecs" 2000 "$dim" 12
    echo "$INDEX_HEADER" > "$OUT/vec-index.csv"
    echo "$FM_HEADER,segments_after" > "$OUT/vec-forcemerge.csv"
    for fmt in ${FORMATS:-hnsw ivf ivf-p}; do
        for layout in ${LAYOUTS:-plain tenant composite}; do
            # Clusters per tenant need the tenant sort.
            [[ "$fmt" == ivf-p && "$layout" == plain ]] && continue
            for engine in ${ENGINES:-rust java}; do
                [[ "$engine" == java && "$fmt" != hnsw ]] && continue
                idx="$OUT/idx/vec-$engine-$fmt-$layout"
                if [[ "$engine" == rust ]]; then
                    "$RUST" index "$corpus" "$idx" "$layout" "$THREADS" "$REFRESH_MS" "$vecs" "$fmt" | tee -a "$OUT/vec-index.csv"
                    "$RUST" knn "$idx" "$corpus" "$vecs" "$qvecs" > "$OUT/vec-knn-$fmt-$layout-natural.csv"
                    "$RUST" forcemerge "$idx" "$layout" 1 "$vecs" "$fmt" | tee -a "$OUT/vec-forcemerge.csv"
                    "$RUST" knn "$idx" "$corpus" "$vecs" "$qvecs" > "$OUT/vec-knn-$fmt-$layout-merged.csv"
                else
                    rm -rf "$idx"
                    "${JVM[@]}" PartitionCostBench index "$corpus" "$idx" "$layout" "$THREADS" "$REFRESH_MS" "$vecs" | tee -a "$OUT/vec-index.csv"
                    "${JVM[@]}" PartitionCostBench forcemerge "$idx" "$layout" 1 | tee -a "$OUT/vec-forcemerge.csv"
                fi
            done
        done
    done
fi
echo "bench-partition-cost: results in $OUT"
