#!/usr/bin/env bash
# Peak process RSS for native SimpleText phrase scoring.
# Run after bench-micro.sh --bench stphrase, through scripts/docker-test.sh.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP=$(lucene_classpath lucene-core lucene-codecs)
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
javac -nowarn -cp "$CP" -d "$OUT" benchmarks/micro/java/SimpleTextPhraseMicro.java
ROOT="$PWD/fixtures/data/simpletext_phrase_scorer/deep/merged/index"
RUST_TARGET_DIR="${CARGO_TARGET_DIR:-benchmarks/rust-runner/target}"
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
peak_rss rust env MICRO_WARMUP_MS=3000 MICRO_MEASURE_MS=3000 \
  "$RUST_TARGET_DIR/release/micro" stphrase "$ROOT"
peak_rss java java --add-modules jdk.incubator.vector \
  -DwarmupMs=3000 -DmeasureMs=3000 -cp "$CP:$OUT" SimpleTextPhraseMicro "$ROOT"
