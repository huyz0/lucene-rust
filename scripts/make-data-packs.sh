#!/usr/bin/env bash
# Writes the language modules' data packs (lucene_analysis::data_pack): one
# directory per pack, holding the files its crate's `src/pack.rs` lists,
# byte for byte its src/resources copies. A build without a module's
# `embedded-data` feature reads them from a directory on LUCENE_RUST_PACKS
# (or lucene_analysis::data_pack::add_pack_dir).
#
#   scripts/make-data-packs.sh [out-dir]     # default target/data-packs
#   scripts/make-data-packs.sh --check DIR   # DIR matches the sources
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

check=0
if [[ "${1:-}" == "--check" ]]; then
    check=1
    shift
fi
out="${1:-target/data-packs}"

status=0
for crate in crates/lucene-analysis-*/; do
    pack_rs="$crate/src/pack.rs"
    [[ -f "$pack_rs" ]] || continue
    name=$(sed -n 's/^pub const NAME: &str = "\(.*\)";/\1/p' "$pack_rs")
    files=$(grep -o 'pack_resource!("[^"]*", "[^"]*")' "$pack_rs" | sed 's/.*", "\(.*\)")/\1/')
    mkdir -p "$out/$name"
    for f in $files; do
        src="$crate/src/$f"
        dst="$out/$name/$(basename "$f")"
        if (( check )); then
            cmp -s "$src" "$dst" || { echo "data pack $name: $dst differs from $src" >&2; status=1; }
        else
            cp "$src" "$dst"
        fi
    done
    (( check )) || echo "data pack $name: $(echo $files | wc -w) files, $(du -sh "$out/$name" | cut -f1)"
done
exit $status
