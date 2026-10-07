#!/usr/bin/env bash
# Regenerates fixtures/data/analysis_date_locales/dates_jdk25.txt under JDK 25
# and diffs it with the committed file (--write replaces it instead).
#
#   scripts/check-date-locales-jdk25.sh [--write]
#
# miscellaneous/date_locales.rs is read off JDK 25, and its parse follows JDK
# 25 where JDK 21 differs (JDK 23+'s lenient space-separator matching, CLDR 47
# names). gen-fixtures.sh's dates.txt keeps to what both JDKs agree on, so it
# cannot see that behaviour; dates_jdk25.txt is GenAnalysisDateLocales --jdk25:
# the same batteries over all 1,151 locales of the table, no text left out
# for its spaces. `java` must be a JDK 25 (the generator refuses any other);
# CI's `fixtures` job runs this after switching to one. gen-fixtures.sh
# --check skips the file (no JDK 21 generator writes it).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

JARS="${JARS:-$PWD/fixtures/.jars}"
# shellcheck source=scripts/lib-lucene-jars.sh
source scripts/lib-lucene-jars.sh
CP=$(lucene_classpath lucene-core lucene-analysis-common)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
javac -nowarn -cp "$CP" -sourcepath fixtures/src -d "$WORK/classes" \
    fixtures/src/GenAnalysisDateLocales.java
FIXTURES_CORPUS="$PWD/fixtures/corpus" java -cp "$WORK/classes:$CP" GenAnalysisDateLocales "$WORK/out" --jdk25
FILE=analysis_date_locales/dates_jdk25.txt
if [ "${1:-}" = "--write" ]; then
    cp "$WORK/out/$FILE" "fixtures/data/$FILE"
    echo "check-date-locales-jdk25: wrote fixtures/data/$FILE ($(wc -l < "fixtures/data/$FILE") rows)"
elif diff -q "$WORK/out/$FILE" "fixtures/data/$FILE" >/dev/null; then
    echo "check-date-locales-jdk25: ok ($(wc -l < "fixtures/data/$FILE") rows)"
else
    diff "$WORK/out/$FILE" "fixtures/data/$FILE" | head -20 || true
    echo "check-date-locales-jdk25: fixtures/data/$FILE is not what JDK 25 writes" >&2
    exit 1
fi
