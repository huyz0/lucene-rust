#!/usr/bin/env bash
# Fetches the ONNX Runtime library lucene-ml's tests load (LUCENE_ML_ONNXRUNTIME):
# Microsoft's Maven jar, pinned by SHA-256, the same one the plugin zip bundles
# (opensearch-plugin/build.gradle, onnxRuntimeVersion).
#
#   scripts/fetch-onnxruntime.sh DIR    -> DIR/libonnxruntime.so, path on stdout
set -euo pipefail
VERSION=1.23.2
SHA256=e0ab4a1af57d2da09097202f2dfd691e390c82e81183314788ccde4cf7c3cc38
dir=${1:?usage: fetch-onnxruntime.sh DIR}
mkdir -p "$dir"
jar="$dir/onnxruntime-$VERSION.jar"
case "$(uname -m)" in
    x86_64) platform=linux-x64 ;;
    aarch64 | arm64) platform=linux-aarch64 ;;
    *) echo "fetch-onnxruntime: no ONNX Runtime build for $(uname -m)" >&2; exit 1 ;;
esac
if ! echo "$SHA256  $jar" | sha256sum -c --status 2>/dev/null; then
    url="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/$VERSION/onnxruntime-$VERSION.jar"
    for delay in 2 4 8 16 32 0; do
        curl -sSfL -o "$jar.part" "$url" && break
        [ "$delay" = 0 ] && { echo "fetch-onnxruntime: cannot download $url" >&2; exit 1; }
        sleep "$delay"
    done
    echo "$SHA256  $jar.part" | sha256sum -c --status || { echo "fetch-onnxruntime: checksum mismatch" >&2; exit 1; }
    mv "$jar.part" "$jar"
fi
unzip -o -j -q "$jar" "ai/onnxruntime/native/$platform/libonnxruntime.so" -d "$dir"
echo "$dir/libonnxruntime.so"
