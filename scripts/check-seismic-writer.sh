#!/usr/bin/env bash
# Checks that neural-search reads the sparse indexes lucene-sparse writes and finds the same hits:
# runs crates/lucene-sparse/tests/seismic_writer.rs, keeping its indexes and hits, then runs
# scripts/seismic/Check.java over the indexes with the jars and JDK of the OpenSearch distribution (no Docker; nodectl.py
# downloads the tarball once) and diffs the two hit lists.
set -euo pipefail

VERSION="${OPENSEARCH_VERSION:-3.9.0}"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/idx" "$work/check"
cp "$root/scripts/seismic/Check.java" "$work/check/"

SEISMIC_WRITER_OUT="$work/idx" SEISMIC_PRINT="$work/rust.txt" \
    cargo test -q --manifest-path "$root/Cargo.toml" -p lucene-sparse --test seismic_writer
chmod -R a+rwX "$work"

home="$(python3 "$root/opensearch-plugin/e2e/nodectl.py" dist stock)"
cp="$home/lib/*:$home/plugins/opensearch-neural-search/*:$home/plugins/opensearch-knn/*:$home/plugins/opensearch-ml/*"
(
    set -e
    cd "$work/check"
    "$home/jdk/bin/javac" -nowarn -cp "$cp" Check.java
    for d in flushed merged; do "$home/jdk/bin/java" -cp "$cp:." Check "$work/idx/$d"; done > java.txt
) 2> >(grep -v -e WARNING -e VectorizationProvider -e '^$' >&2)

if diff "$work/rust.txt" "$work/check/java.txt"; then
    echo "neural-search $VERSION returned the same $(wc -l < "$work/rust.txt") hits"
else
    echo "hits differ" >&2
    exit 1
fi
