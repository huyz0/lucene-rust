#!/usr/bin/env bash
# Substring search over log lines: the Substring90 BINARY doc values field
# (this port) against stock Lucene's text field and OpenSearch's `wildcard`
# field layout over the same corpus and queries.
#
#   scripts/bench-substring-field.sh <loghub dir> [OUT_DIR] [BLOCK_SHIFT]   # default target/bench-substring-field, 5
#
# <loghub dir> holds bgl.txt, hdfs.txt, openssh.txt and thunderbird.txt (the
# engine study used the first 250 MB of each from Loghub). Prints every
# index's size, indexing and merge seconds, then per query the median
# latency of each engine and whether this port's hits (count and a hash of
# the hits' line numbers) equal the wildcard field's.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SRC="$1"
OUT="${2:-target/bench-substring-field}"
SHIFT="${3:-5}"
SECS="${SECS:-2}"
mkdir -p "$OUT/classes"
OUT=$(cd "$OUT" && pwd)

JARS="$PWD/fixtures/.jars"
# shellcheck source=scripts/lib-lucene-jars.sh
source scripts/lib-lucene-jars.sh
CP=$(lucene_classpath lucene-core lucene-analysis-common)
javac -nowarn -cp "$CP" -d "$OUT/classes" \
  benchmarks/java-runner/src/SubstringBench.java benchmarks/java-runner/src/SubstringCorpus.java
JAVA=(java -Xmx6g --enable-native-access=ALL-UNNAMED -cp "$OUT/classes:$CP")

[ -f "$OUT/corpus.txt" ] || "${JAVA[@]}" SubstringCorpus "$SRC" "$OUT/corpus.txt"
cargo build --profile dist -p lucene-search --example substring_bench
RUST=target/dist/examples/substring_bench

"${JAVA[@]}" SubstringBench build-text "$OUT/corpus.txt" "$OUT/text"
"${JAVA[@]}" SubstringBench build-os "$OUT/corpus.txt" "$OUT/wildcard"
"$RUST" build "$OUT/corpus.txt" "$OUT/substring" "$SHIFT"

"${JAVA[@]}" SubstringBench query "$OUT/text" "$OUT/wildcard" benchmarks/substring-queries.tsv "$SECS" \
  | grep -v '^WARNING' > "$OUT/java.tsv"
"$RUST" query "$OUT/substring" benchmarks/substring-queries.tsv "$SECS" > "$OUT/rust.tsv"

# Join on the query id; the Rust hash is signed, the Java one hex.
python3 - "$OUT/java.tsv" "$OUT/rust.tsv" <<'EOF'
import sys
java = {l.split('\t')[0]: l.rstrip('\n').split('\t') for l in open(sys.argv[1])}
print('id\ttext_ms\twildcard_ms\tsubstring_ms\thits\tsame_hits')
for line in open(sys.argv[2]):
    c = line.rstrip('\n').split('\t')
    if c[0] == 'id':
        continue
    j = java[c[0]]
    same = j[4] == c[2] and j[5] == format(int(c[3]) & (2**64 - 1), 'x')
    print('\t'.join([c[0], j[1], j[3], c[1], c[2], 'yes' if same else 'NO']))
EOF
