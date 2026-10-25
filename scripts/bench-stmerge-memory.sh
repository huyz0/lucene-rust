#!/usr/bin/env bash
# Peak process RSS for the paired shared-terms merge case. Run after the
# timing benchmark has built the Rust runner, through scripts/docker-test.sh.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP=$(lucene_classpath lucene-core lucene-codecs)
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
javac -nowarn -cp "$CP" -d "$OUT" benchmarks/micro/java/StMergeMicro.java
ROOT="$PWD/fixtures/data/codecs_sharedterms_merge/unsorted"
RUST_TARGET_DIR="${CARGO_TARGET_DIR:-benchmarks/rust-runner/target}"
# A fresh Python parent for each command prevents RUSAGE_CHILDREN's
# cumulative maximum from carrying the first engine's RSS into the second.
peak_rss() {
  python3 - "$@" <<'PY'
import resource
import subprocess
import sys

engine = sys.argv[1]
subprocess.run(sys.argv[2:], check=True)
print(f"{engine}\tpeak_process_rss_kib\t{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}")
PY
}
peak_rss rust env MICRO_WARMUP_MS=5000 MICRO_MEASURE_MS=5000 \
  "$RUST_TARGET_DIR/release/micro" stmerge "$ROOT"
peak_rss java java --add-modules jdk.incubator.vector \
  -DwarmupMs=5000 -DmeasureMs=5000 -cp "$CP:$OUT" StMergeMicro "$ROOT"
