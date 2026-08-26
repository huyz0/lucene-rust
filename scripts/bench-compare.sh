#!/usr/bin/env bash
# Run both engines over the same index and query set, join on query id, and
# report the ratio per query plus the M1 gate verdict.
#
# Cross-checks recall BEFORE comparing timings: if the two engines disagree on
# hit counts or top-10 doc ids, the timings are measuring different work and the
# ratio is meaningless. A speedup obtained by returning fewer results is a bug
# report, not a benchmark result.
#
# Usage: scripts/bench-compare.sh [--index DIR] [--queries F] [--warmup N] [--iters N] [--pin CPUS]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

INDEX="$PWD/benchmarks/.corpus/merged"
QUERIES="$PWD/benchmarks/queries.tsv"
WARMUP=200
ITERS=200
JARS="$PWD/fixtures/.jars"
# i5-13600KF is a hybrid part: P-cores 0-11 (SMT), E-cores 12-19. Pin to a
# single P-core pair so the scheduler cannot migrate a run onto an E-core
# mid-measurement, which would otherwise dominate the variance.
PIN="0,1"

while [ $# -gt 0 ]; do
  case "$1" in
    --index)   INDEX="$2";   shift 2 ;;
    --queries) QUERIES="$2"; shift 2 ;;
    --warmup)  WARMUP="$2";  shift 2 ;;
    --iters)   ITERS="$2";   shift 2 ;;
    --pin)     PIN="$2";     shift 2 ;;
    --jars)    JARS="$2";    shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "bench-compare: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# shellcheck source=scripts/lib-lucene-jars.sh
source "$(dirname "$0")/lib-lucene-jars.sh"
CP=$(lucene_classpath lucene-core lucene-analysis-common)

OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

echo "bench-compare: building runners"
( cd benchmarks/rust-runner && cargo build --release --quiet )
javac -nowarn -cp "$CP" -d "$OUT/classes" benchmarks/java-runner/src/BenchRunner.java

command -v taskset >/dev/null && PINCMD=(taskset -c "$PIN") || PINCMD=()

echo "bench-compare: index=$INDEX warmup=$WARMUP iters=$ITERS pinned=${PIN:-none}"
"${PINCMD[@]}" ./benchmarks/rust-runner/target/release/bench-runner \
    "$INDEX" "$QUERIES" "$WARMUP" "$ITERS" > "$OUT/rust.tsv"
"${PINCMD[@]}" java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
    -Xmx4g -cp "$OUT/classes:$CP" BenchRunner \
    "$INDEX" "$QUERIES" "$WARMUP" "$ITERS" > "$OUT/java.tsv" 2>"$OUT/java.err" || {
      echo "bench-compare: java runner failed"; cat "$OUT/java.err"; exit 1; }

python3 - "$OUT/rust.tsv" "$OUT/java.tsv" <<'PY'
import sys, csv

def load(p):
    with open(p) as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}

rust, java = load(sys.argv[1]), load(sys.argv[2])
ids = [i for i in java if i in rust]

# --- recall cross-check first -------------------------------------------------
mismatch, tie_only = [], []
for i in ids:
    r, j = rust[i], java[i]
    if r["hits"] != j["hits"]:
        mismatch.append((i, "hits", r["hits"], j["hits"]))
    elif r["topset"] != j["topset"]:
        mismatch.append((i, "topset", r["topset"][:48], j["topset"][:48]))
    elif abs(float(r["top1score"]) - float(j["top1score"])) > 1e-5:
        mismatch.append((i, "top1score", r["top1score"], j["top1score"]))
    elif r["top1doc"] != j["top1doc"]:
        tie_only.append(i)          # same set, same score, different tie-break

print(f"{'query':<6} {'rust qps':>11} {'java qps':>11} {'ratio':>7}  {'rust p99':>9} {'java p99':>9}  recall")
print("-" * 76)
ratios = []
bad = {m[0] for m in mismatch}
for i in ids:
    r, j = rust[i], java[i]
    rq, jq = float(r["qps"]), float(j["qps"])
    ratio = rq / jq if jq else float("inf")
    ratios.append((i, ratio))
    ok = "MISMATCH" if i in bad else ("tie" if i in tie_only else "ok")
    print(f"{i:<6} {rq:>11,.1f} {jq:>11,.1f} {ratio:>6.2f}x  {r['p99_us']:>9} {j['p99_us']:>9}  {ok}")

print()
if tie_only:
    print(f"note: {len(tie_only)} queries agree on hit set and top score but order ties "
          f"differently ({', '.join(tie_only)}) -- not a recall defect.")
if mismatch:
    print(f"RECALL MISMATCH on {len(mismatch)} of {len(ids)} queries -- their timings are NOT comparable:")
    for m in mismatch[:12]:
        print(f"  {m[0]}: {m[1]}\n    rust={m[2]}\n    java={m[3]}")
    print()

wins = [i for i, r in ratios if r >= 1.5]
losses = [i for i, r in ratios if r < 1.0]
pct = 100.0 * len(wins) / len(ratios) if ratios else 0
print(f"M1 gate:")
print(f"  >=1.5x on {len(wins)}/{len(ratios)} queries ({pct:.0f}%)   [criterion: >=80%]")
print(f"  slower than Java on {len(losses)} queries: {', '.join(losses) or 'none'}   [criterion: none]")
print(f"  recall mismatches: {len(mismatch)}   [criterion: 0]")
passed = pct >= 80 and not losses and not mismatch
print(f"  => {'PASS' if passed else 'FAIL'}")
PY
