#!/usr/bin/env bash
# Checks OpenSearch's custom-codecs zstd stored fields against lucene-rust in both directions:
# runs crates/lucene-search/tests/opensearch_zstd_stored_fields.rs, keeping the `zstd` and
# `zstd_no_dict` indexes it writes (flushed and force-merged), then has the plugin's own
# Zstd104Codec/ZstdNoDict104Codec read them (scripts/zstd/ZstdStoredFields.java dump) and compares
# every document with the dump of the fixture OpenSearch wrote. `--fixture` instead rewrites
# fixtures/data/opensearch_zstd with the plugin's writer.
#
# Runs inside the opensearchproject/opensearch image (needs Docker), or against a local classpath
# holding OpenSearch's lib/ jars and the custom-codecs plugin's jars: OPENSEARCH_CP="lib/*:cc/*".
set -euo pipefail

VERSION="${OPENSEARCH_VERSION:-3.9.0}"
root="$(cd "$(dirname "$0")/.." && pwd)"
fixtures="$root/fixtures/data/opensearch_zstd"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/idx" "$work/check"
cp "$root/scripts/zstd/ZstdStoredFields.java" "$work/check/"

# Runs `ZstdStoredFields.java` with its arguments, the index directory under /idx.
java_run() {
    if [[ -n "${OPENSEARCH_CP:-}" ]]; then
        local classes="$work/check/classes"
        [[ -d "$classes" ]] || javac -nowarn -d "$classes" -cp "$OPENSEARCH_CP" "$work/check/ZstdStoredFields.java"
        local args=("${@//\/idx/$work/idx}")
        java -cp "$OPENSEARCH_CP:$classes" ZstdStoredFields "${args[@]}"
    else
        docker run --rm --entrypoint bash -v "$work/check:/check" -v "$work/idx:/idx" \
            "opensearchproject/opensearch:$VERSION" -c '
set -e
home=/usr/share/opensearch
cp="$home/lib/*:$home/plugins/opensearch-custom-codecs/*"
cd /tmp && cp /check/ZstdStoredFields.java . && "$home/jdk/bin/javac" -nowarn -cp "$cp" ZstdStoredFields.java
"$home/jdk/bin/java" -cp "$cp:/tmp" ZstdStoredFields "$@"
' bash "$@"
    fi 2> >(grep -v -e WARNING -e VectorizationProvider -e '^$' >&2)
}

if [[ "${1:-}" == "--fixture" ]]; then
    chmod -R a+rwX "$work"
    for m in zstd zstd_no_dict; do
        level=3; [[ "$m" == zstd_no_dict ]] && level=6
        java_run write "/idx/$m" "$m" "$level"
        java_run dump "/idx/$m" > "$work/$m.tsv"
        rm -rf "${fixtures:?}/$m"
        mkdir -p "$fixtures/$m"
        find "$work/idx/$m" -type f ! -name write.lock -exec cp {} "$fixtures/$m/" \;
        cp "$work/$m.tsv" "$fixtures/$m.tsv"
    done
    echo "rewrote $fixtures"
    exit 0
fi

ZSTD_WRITER_OUT="$work/idx" \
    cargo test -q --manifest-path "$root/Cargo.toml" -p lucene-search --test opensearch_zstd_stored_fields
chmod -R a+rwX "$work"

status=0
for m in zstd zstd_no_dict; do
    java_run dump "/idx/$m" > "$work/$m.java.tsv"
    if ! diff -q "$work/$m.java.tsv" "$fixtures/$m.tsv" >/dev/null; then
        echo "$m: the plugin reads the Rust-written index differently" >&2
        diff "$work/$m.java.tsv" "$fixtures/$m.tsv" | head >&2
        status=1
    fi
    # A merge renumbers segments and documents; the documents stay in order.
    java_run dump "/idx/${m}_merged" > "$work/${m}_merged.java.tsv"
    if ! diff -q <(grep -v '^segment' "$work/${m}_merged.java.tsv" | cut -f2-) \
                 <(grep -v '^segment' "$fixtures/$m.tsv" | cut -f2-) >/dev/null; then
        echo "${m}_merged: the plugin reads the merged Rust index differently" >&2
        status=1
    fi
    echo "$m: $(head -1 "$work/${m}_merged.java.tsv" | cut -f3,4) -- $(grep -vc '^segment' "$work/$m.java.tsv") documents match"
done
exit "$status"
