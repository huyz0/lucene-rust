#!/usr/bin/env bash
# Opt-in differential: Lucene 10.5.0's own Hunspell test dictionaries through
# fixtures/src/GenHunspell.java and crates/lucene-analysis/tests/hunspell_fixtures.rs.
#
#   scripts/check-hunspell-lucene-dictionaries.sh [LUCENE_CHECKOUT]
#
# Lucene's test dictionaries (lucene/analysis/common/src/test/.../hunspell)
# are mostly Hunspell's own test suite, which this repository does not
# redistribute (docs/licences.md); the committed fixtures use dictionaries
# written here instead. This fetches Lucene's at the 10.5.0 release commit
# (sparse, with git; or copies them from a checkout of it), checks them
# against a pinned SHA-256 (over each .aff/.dic/.good/.wrong file, sorted),
# and in a temporary directory only: copies each .aff and .dic, joins its
# .good and .wrong into the .words GenHunspell reads, runs GenHunspell over
# them with Lucene 10.5.0, and runs hunspell_fixtures.rs on the result
# (HUNSPELL_CORPUS and HUNSPELL_DATA name the two directories). Nothing is
# written to the repository.
#
# Every file is checked and every difference listed (the test does not stop
# at the first). All 97 dictionaries agree with Lucene. Needs git (without a
# checkout), a JDK and cargo.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LUCENE_COMMIT=f6eaee8148b7569e83c433feacc4f624608188fd # releases/lucene/10.5.0
DICTIONARIES_SHA256=10aea8ea9eb86410689a14b7182e355f0d8d1cc4eaaf474f720b8378e92d0d4f
DIR=lucene/analysis/common/src/test/org/apache/lucene/analysis/hunspell

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
if [[ $# -ge 1 ]]; then
    src="$1/$DIR"
else
    git init -q "$work/lucene"
    git -C "$work/lucene" sparse-checkout set --no-cone "/$DIR/"
    git -C "$work/lucene" fetch -q --depth 1 --filter=blob:none https://github.com/apache/lucene "$LUCENE_COMMIT"
    git -C "$work/lucene" checkout -q FETCH_HEAD
    src="$work/lucene/$DIR"
fi
sum=$(cd "$src" && find . -maxdepth 1 -type f \( -name '*.aff' -o -name '*.dic' -o -name '*.good' -o -name '*.wrong' \) -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
if [[ $sum != "$DICTIONARIES_SHA256" ]]; then
    echo "check-hunspell-lucene-dictionaries: the dictionaries do not match their pinned SHA-256 ($sum)" >&2
    exit 1
fi

corpus="$work/corpus/hunspell"
mkdir -p "$corpus" "$work/data"
for aff in "$src"/*.aff; do
    name=$(basename "$aff" .aff)
    cp "$aff" "$corpus/"
    [[ -f "$src/$name.dic" ]] && cp "$src/$name.dic" "$corpus/"
    for list in good wrong; do
        [[ -f "$src/$name.$list" ]] && cat "$src/$name.$list" >> "$corpus/$name.words"
    done
done

JARS=fixtures/.jars
# shellcheck source=scripts/lib-lucene-jars.sh
source scripts/lib-lucene-jars.sh
CP=$(lucene_classpath lucene-core lucene-analysis-common)
javac -nowarn -cp "$CP" -d "$work/classes" fixtures/src/GenHunspell.java
FIXTURES_CORPUS="$work/corpus" java "${LUCENE_FIXTURE_JVM_OPTS[@]}" -cp "$work/classes:$CP" GenHunspell "$work/data"
echo "check-hunspell-lucene-dictionaries: $(ls "$corpus"/*.aff | wc -l) of Lucene's dictionaries generated"
HUNSPELL_CORPUS="$corpus/" HUNSPELL_DATA="$work/data/hunspell/" \
    cargo test --release -p lucene-analysis --test hunspell_fixtures -- --nocapture
