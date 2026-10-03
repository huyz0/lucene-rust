#!/usr/bin/env bash
# Checks that the derived engine tests under opensearch-plugin/src/engineTest
# are exactly what opensearch-plugin/tools/derive_engine_tests.py produces from
# OpenSearch 3.8.0's sources -- InternalEngineTests.java and its three server
# test helpers (pinned by tag), and the test framework's sources jar -- so a
# hand edit of a derived file, or a script change nobody re-ran, fails here.
#
#   scripts/check-derived-engine-tests.sh [SOURCES_DIR]
#
# SOURCES_DIR, when given, holds the five inputs already (offline); otherwise
# they are downloaded. Each is checked against its pinned SHA-256 either way:
# a moved tag or a republished jar is a failure, not a new baseline.
# Needs python3, unzip, and (without SOURCES_DIR) curl.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION=3.8.0
SERVER="https://raw.githubusercontent.com/opensearch-project/OpenSearch/${VERSION}/server/src/test/java/org/opensearch/index"
FRAMEWORK="https://repo1.maven.org/maven2/org/opensearch/test/framework/${VERSION}/framework-${VERSION}-sources.jar"
# name  sha256  url
INPUTS=(
    "InternalEngineTests.java df84a9f83a1dc8d402bf6e62da162b9d53c3337e5b0a28ce4dea6bf458024657 $SERVER/engine/InternalEngineTests.java"
    "EngineSearcherTotalHitsMatcher.java f8c7c12ccef11d83748a4d88bae3ca4d11d555d60d3d81ee94406f90e33a601e $SERVER/engine/EngineSearcherTotalHitsMatcher.java"
    "TestTranslog.java 5a422aa0085eec939215075298e1791bd9cafd6d1a4538203fd0fc42f2085e02 $SERVER/translog/TestTranslog.java"
    "SnapshotMatchers.java a86936154897809f0c1c88d5142e0d643b52fca9d595a751693a00c733ba0722 $SERVER/translog/SnapshotMatchers.java"
    "framework-${VERSION}-sources.jar 883588bde6c80075eb34e3ed2ee5f8620ec6759503ebad435bac089d9c7aec89 $FRAMEWORK"
)
# The files the script writes, relative to src/engineTest/java. The others
# there (RustIndexerFactoryTests, EngineTestAccess) are written by hand.
DERIVED=(
    org/lucenerust/opensearch/engine/RustEngineTestCase.java
    org/lucenerust/opensearch/engine/RustTestEngine.java
    org/lucenerust/opensearch/engine/RustEngineTests.java
    org/opensearch/index/engine/EngineSearcherTotalHitsMatcher.java
    org/opensearch/index/translog/TestTranslog.java
    org/opensearch/index/translog/SnapshotMatchers.java
)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
src="$work/src"
mkdir -p "$src"
for entry in "${INPUTS[@]}"; do
    read -r name sum url <<<"$entry"
    if [[ $# -ge 1 ]]; then
        cp "$1/$name" "$src/$name"
    else
        curl -sSfL --retry 4 --retry-delay 2 -o "$src/$name" "$url"
    fi
    echo "$sum  $src/$name" | sha256sum -c --quiet - || {
        echo "check-derived-engine-tests: $name does not match its pinned SHA-256" >&2
        exit 1
    }
done
mkdir -p "$work/framework"
unzip -q "$src/framework-${VERSION}-sources.jar" -d "$work/framework"
out="$work/out/org/lucenerust/opensearch/engine"
python3 opensearch-plugin/tools/derive_engine_tests.py "$src" "$work/framework" "$out"

status=0
for f in "${DERIVED[@]}"; do
    if ! diff -u "$work/out/$f" "opensearch-plugin/src/engineTest/java/$f" >"$work/diff" 2>&1; then
        echo "check-derived-engine-tests: $f differs from the script's output:" >&2
        head -40 "$work/diff" >&2
        status=1
    fi
done
# Nothing derived that the list above does not name.
extra=$(cd "$work/out" && find . -name '*.java' | sed 's|^\./||' | sort)
want=$(printf '%s\n' "${DERIVED[@]}" | sort)
if [[ "$extra" != "$want" ]]; then
    echo "check-derived-engine-tests: the script now writes other files than DERIVED lists:" >&2
    diff <(echo "$want") <(echo "$extra") >&2 || true
    status=1
fi
[[ $status == 0 ]] && echo "check-derived-engine-tests: ${#DERIVED[@]} derived files match derive_engine_tests.py's output"
exit $status
