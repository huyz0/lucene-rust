#!/usr/bin/env bash
# Checks that the derived engine under opensearch-plugin/.../engine is exactly what
# opensearch-plugin/tools/derive_engine.py produces from OpenSearch 3.9.0's InternalEngine and
# helper sources, so a hand edit of AcceleratedEngine.java (or of a copied helper), or a script
# change nobody re-ran, fails here. The sibling of check-derived-engine-tests.sh.
#
#   scripts/check-derived-engine.sh [SOURCES_JAR]
#
# SOURCES_JAR, when given, is opensearch-3.9.0-sources.jar already (offline); otherwise it is
# downloaded. It is checked against its pinned SHA-256 either way: a republished jar is a failure,
# not a new baseline. Needs python3, unzip, and (without SOURCES_JAR) curl.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION=3.9.0
URL="https://repo1.maven.org/maven2/org/opensearch/opensearch/${VERSION}/opensearch-${VERSION}-sources.jar"
SHA256=eb77911b062935c41513e29b6fb3c9da1aeceab7b33cefc85db864f0a8c42abf
ENGINE=opensearch-plugin/src/main/java/org/lucenerust/opensearch/engine

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
jar="$work/sources.jar"
if [[ $# -ge 1 ]]; then
    cp "$1" "$jar"
else
    curl -sSfL --retry 4 --retry-delay 2 -o "$jar" "$URL"
fi
echo "$SHA256  $jar" | sha256sum -c --quiet - || {
    echo "check-derived-engine: the sources jar does not match its pinned SHA-256" >&2
    exit 1
}
mkdir -p "$work/src" "$work/out"
unzip -q "$jar" 'org/opensearch/index/engine/*' -d "$work/src"
python3 opensearch-plugin/tools/derive_engine.py "$work/src" "$work/out"

status=0
for f in "$work"/out/*.java; do
    name=$(basename "$f")
    if ! diff -u "$f" "$ENGINE/$name" >"$work/diff" 2>&1; then
        echo "check-derived-engine: $name differs from the script's output:" >&2
        head -40 "$work/diff" >&2
        status=1
    fi
done
[[ $status == 0 ]] && echo "check-derived-engine: $(ls "$work"/out | wc -l) derived files match derive_engine.py's output"
exit $status
