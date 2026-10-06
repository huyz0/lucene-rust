#!/usr/bin/env bash
# Checks that crates/lucene-analysis/src/snowball/algorithms/ is exactly what
# crates/lucene-analysis/tools/gen_snowball.sh produces from the Snowball
# sources Lucene 10.5.0's stemmers were generated from, with Lucene's own
# stemmers as the cross-check -- so a hand edit of a generated stemmer, or a
# generator change nobody re-ran, fails here.
#
#   scripts/check-snowball-generated.sh [SNOWBALL_CHECKOUT [LUCENE_SOURCES_JAR]]
#
# Inputs, each pinned by SHA-256 (a moved input is a failure, not a new
# baseline): the Snowball compiler and algorithms at gen_snowball.sh's
# SNOWBALL_COMMIT (fetched with git, or copied from a checkout of it; hashed
# file by file, .git excluded), and lucene-analysis-common 10.5.0's sources
# jar from Maven Central (or the given file), with which gen_snowball.sh first
# checks that the compiler's Java backend reproduces Lucene's stemmers.
# Needs git (without a checkout), curl (without a jar), gcc, make, perl,
# python3 and unzip.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SNOWBALL_COMMIT=34f3612e5e8c48975243bc2e87561abdac5aa9bb
SNOWBALL_TREE_SHA256=30402baf3c9d496ceb7a0007ca80cc440f9ec992321de2d42705b0e58d42f527
JAR_URL=https://repo1.maven.org/maven2/org/apache/lucene/lucene-analysis-common/10.5.0/lucene-analysis-common-10.5.0-sources.jar
JAR_SHA256=bf873863f14c621ae0a8b13d9530e269a778f7552c47c05a00b85ca18f248df7
GENERATED=crates/lucene-analysis/src/snowball/algorithms

grep -q "^SNOWBALL_COMMIT=$SNOWBALL_COMMIT\$" crates/lucene-analysis/tools/gen_snowball.sh || {
    echo "check-snowball-generated: gen_snowball.sh pins another Snowball commit" >&2
    exit 1
}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
if [[ $# -ge 1 ]]; then
    mkdir "$work/snowball"
    (cd "$1" && tar --exclude=.git -cf - .) | (cd "$work/snowball" && tar -xf -)
else
    git init -q "$work/snowball"
    git -C "$work/snowball" fetch -q --depth 1 https://github.com/snowballstem/snowball "$SNOWBALL_COMMIT"
    git -C "$work/snowball" checkout -q FETCH_HEAD
    rm -rf "$work/snowball/.git"
fi
tree=$(cd "$work/snowball" && find . -type f -print0 | LC_ALL=C sort -z \
    | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
if [[ $tree != "$SNOWBALL_TREE_SHA256" ]]; then
    echo "check-snowball-generated: the Snowball sources do not match their pinned SHA-256 ($tree)" >&2
    exit 1
fi
if [[ $# -ge 2 ]]; then
    cp "$2" "$work/sources.jar"
else
    curl -sSfL --retry 4 --retry-delay 2 -o "$work/sources.jar" "$JAR_URL"
fi
echo "$JAR_SHA256  $work/sources.jar" | sha256sum -c --quiet - || {
    echo "check-snowball-generated: the Lucene sources jar does not match its pinned SHA-256" >&2
    exit 1
}

# gen_snowball.sh writes next to itself (../src/snowball/algorithms), so it
# runs from a copy of the tools directory; the committed tree is untouched.
mkdir -p "$work/tree/crates/lucene-analysis"
cp -r crates/lucene-analysis/tools "$work/tree/crates/lucene-analysis/tools"
"$work/tree/crates/lucene-analysis/tools/gen_snowball.sh" "$work/snowball" "$work/sources.jar"
if ! diff -r "$work/tree/$GENERATED" "$GENERATED" >"$work/diff"; then
    echo "check-snowball-generated: $GENERATED differs from gen_snowball.sh's output:" >&2
    head -40 "$work/diff" >&2
    exit 1
fi
echo "check-snowball-generated: $(ls "$GENERATED" | wc -l) files match gen_snowball.sh's output"
