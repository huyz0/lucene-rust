#!/usr/bin/env bash
# Builds the documentation site into site/book/.
#
# The hand-written pages live in site/src. The reference pages are the files
# under docs/ that the repository already keeps true (the same ones its gates
# read), copied in here with their links to source files pointed at GitHub, so
# there is one copy of each fact. Needs `mdbook` (cargo install mdbook).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=https://github.com/huyz0/lucene-rust/blob/main

rm -rf src/reference src/contributing/gates
mkdir -p src/reference src/contributing/gates src/reference/benchmarks

fix_links() { # rewrite links that leave docs/ so they land on the source tree
    sed -E \
        -e "s#\]\((\.\./)+(scripts|crates|opensearch-plugin|fixtures|benchmarks|docker|\.github|\.githooks)/#](${REPO}/\2/#g" \
        -e "s#\]\(lucene-port-anchor\.json\)#](${REPO}/docs/lucene-port-anchor.json)#g" \
        -e "s#\]\((\.\./)?(parity/[^)]*|parity\.md)#](${REPO}/docs/\2#g" \
        -e "s#\]\((\.\./)+(LICENSE|NOTICE|AGENTS\.md|README\.md|Cargo\.toml|rust-toolchain\.toml)\)#](${REPO}/\2)#g" \
        "$1"
}

for f in operations feature-matrix compatibility accelerated-queries opensearch-engine \
         realtime-search pluggable-storage logs-preset ml-commons-remote licences upstream-forks; do
    fix_links "../docs/$f.md" > "src/reference/$f.md"
done
for f in ci testing-without-docker mechanical-gates arithmetic-gate porting-workflow \
         lucene-port-anchor; do
    fix_links "../docs/$f.md" > "src/contributing/gates/$f.md"
done
for f in $(grep -o 'reference/benchmarks/[^)]*\.md' src/SUMMARY.md | sed 's#reference/benchmarks/##'); do
    fix_links "../docs/benchmarks/$f" > "src/reference/benchmarks/$f"
done

mdbook build
