#!/usr/bin/env bash
# Native Lucene reads all 44 explicit and 12 configured Rust final indexes.
# Run through scripts/docker-test.sh, after generating the selected fixture.
set -euo pipefail
cd "$(dirname "$0")/.."
JARS="${JARS:-$PWD/fixtures/.jars}"
source scripts/lib-lucene-jars.sh
CP="$(lucene_classpath lucene-core lucene-codecs lucene-test-framework):$(thirdparty_classpath "${TEST_FRAMEWORK_DEPS[@]}")"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
javac -nowarn -proc:none -cp "$CP" -d "$OUT/classes" \
  fixtures/src/testframework/TermVectorsRoutingSupport.java fixtures/src/testframework/GenTermVectorsWriterRouting.java \
  fixtures/src/VerifyConfiguredTvGaps.java
LUCENE_TV_ROUTING_EXPORT="$OUT/indexes" cargo test -p lucene-test-framework \
  --test term_vectors_writer_routing_fixtures
count=0
for directory in "$OUT/indexes"/*; do
  name="${directory##*/}"
  name="${name%-configured}"
  java --add-modules jdk.incubator.vector -ea -Xms256m -Xmx256m \
    -cp "$OUT/classes:$CP" GenTermVectorsWriterRouting --verify-dir "$directory" "$name"
  count=$((count + 1))
done
if [ "$count" != 56 ]; then echo "expected 56 Rust indexes; found $count" >&2; exit 1; fi
echo "Native Java term-vector readback passed for all $count Rust indexes."

# Validate fixed-schema gaps before Rust merge can normalize source metadata.
cargo run --quiet -p lucene-index --example write_configured_tv_gaps -- "$OUT/schema-gaps"
java --add-modules jdk.incubator.vector -ea -Xms256m -Xmx256m \
  -cp "$OUT/classes:$CP" VerifyConfiguredTvGaps "$OUT/schema-gaps"
