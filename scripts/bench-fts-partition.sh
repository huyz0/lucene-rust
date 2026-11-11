#!/usr/bin/env bash
# Full-text search in one shard shared by many tenants, every query filtered
# by its tenant: stock Lucene (Java) against this port over the same three
# indexes -- unsorted, sorted by tenant then time (newest first), and that
# sort with nested-style document blocks -- written once by stock Lucene.
#
#   scripts/bench-fts-partition.sh [DOCS] [TENANTS] [OUT_DIR]   # default 2000000, 1000, target/bench-fts-partition
#
# Per index and engine: a cold pass on a dropped page cache (mean latency and
# bytes read from the device; needs a writable /proc/sys/vm/drop_caches),
# then hot latency per tenant class and query kind, and every query's top
# hits, which must agree across engines and indexes (scores bit for bit,
# the tenant's statistics being the shard's in both). BASELINE=<git dir>
# also runs the Rust side built from that checkout (the example copied in).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DOCS="${1:-2000000}"
TENANTS="${2:-1000}"
OUT="${3:-target/bench-fts-partition}"
PASSES="${PASSES:-5}"
mkdir -p "$OUT/classes"
OUT=$(cd "$OUT" && pwd)

JARS="$PWD/fixtures/.jars"
# shellcheck source=scripts/lib-lucene-jars.sh
source scripts/lib-lucene-jars.sh
CP=$(lucene_classpath lucene-core lucene-analysis-common)
javac -nowarn -cp "$CP" -d "$OUT/classes" benchmarks/java-runner/src/PartitionBench.java
JAVA=(java -Xmx3g --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED -cp "$OUT/classes:$CP" PartitionBench)

cargo build --release -q -p lucene-search --example fts_partition
RUST=("$PWD/target/release/examples/fts_partition")
if [[ -n "${BASELINE:-}" ]]; then
    # A checkout without `pinned_sort_keys` never ends a segment early.
    grep -q "pub use exec::pinned_sort_keys" "$BASELINE/crates/lucene-search/src/lib.rs" \
        && cp crates/lucene-search/examples/fts_partition.rs "$BASELINE/crates/lucene-search/examples/" \
        || sed 's/lucene_search::pinned_sort_keys(s, q).unwrap() >= 1/{ let _ = q; false }/' \
            crates/lucene-search/examples/fts_partition.rs > "$BASELINE/crates/lucene-search/examples/fts_partition.rs"
    (cd "$BASELINE" && cargo build --release -q -p lucene-search --example fts_partition)
fi

cold() {
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "bench-fts-partition: cannot drop the page cache; cold numbers are warm" >&2
}

for mode in plain composite blocks; do
    idx="$OUT/idx/$mode"
    [[ -f "$idx/queries.tsv" ]] || "${JAVA[@]}" build "$idx" "$DOCS" "$TENANTS" "$mode"
    engines=(java rust)
    [[ -n "${BASELINE:-}" ]] && engines+=(base)
    for engine in "${engines[@]}"; do
        case "$engine" in
            java) run=("${JAVA[@]}") ;;
            rust) run=("${RUST[@]}") ;;
            base) run=("$BASELINE/target/release/examples/fts_partition") ;;
        esac
        cold
        "${run[@]}" query "$idx" "$idx/queries.tsv" "$PASSES" > "$OUT/$mode-$engine.csv"
        "${run[@]}" hits "$idx" "$idx/queries.tsv" > "$OUT/$mode-$engine.hits"
    done
done
python3 - "$OUT" <<'PY'
import sys, os, collections
out = sys.argv[1]
def hits(p):
    r = {}
    for line in open(p):
        f = line.rstrip("\n").split("\t")
        r[f[0]] = (f[1], [x.split(":") for x in f[2:]])
    return r
ref = hits(f"{out}/plain-java.hits")
for name in sorted(os.listdir(out)):
    if not name.endswith(".hits"):
        continue
    h = hits(f"{out}/{name}")
    bad = 0
    for q, (total, top) in ref.items():
        t2, top2 = h[q]
        # Values (scores or times) must match in order; ids may differ only
        # among ties, which the doc order breaks.
        if [v for _, v in top] != [v for _, v in top2]:
            bad += 1
            continue
        groups = collections.defaultdict(set)
        groups2 = collections.defaultdict(set)
        for (i, v), (i2, v2) in zip(top, top2):
            groups[v].add(i); groups2[v].add(i2)
        last = top[-1][1] if top else None
        if any(groups[v] != groups2[v] for v in groups if v != last):
            bad += 1
    print(f"{name}: {len(ref) - bad}/{len(ref)} queries agree with plain-java")
PY
echo "bench-fts-partition: results in $OUT"
