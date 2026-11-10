#!/usr/bin/env bash
# Fetches the libtorch build lucene-ffi's TorchScript tests load
# (LUCENE_ML_LIBTORCH): DJL's PyTorch 2.5.1 CPU libraries for this machine,
# from Maven Central, pinned by SHA-256. The pre-C++11-ABI build, the one
# ml-commons downloads by default; it bundles its own libstdc++.
#
#   scripts/fetch-libtorch.sh DIR    -> the libtorch directory on stdout
set -euo pipefail
VERSION=2.5.1
dir=${1:?usage: fetch-libtorch.sh DIR}
case "$(uname -m)" in
    x86_64) arch=x86_64; SHA256=5503bb5addcdf54498f53520a410bb53c15b781531c14b15b0c49bc8520db753 ;;
    aarch64 | arm64) arch=aarch64; SHA256=3722754bb5b0ae41f4fdfad00460af922f1eb3dbaea250b17152bf3711a9737b ;;
    *) echo "fetch-libtorch: no libtorch build for $(uname -m)" >&2; exit 1 ;;
esac
mkdir -p "$dir"
lib="$dir/$VERSION-cpu-precxx11-linux-$arch"
if [ ! -f "$lib/libtorch.so" ]; then
    jar="$dir/pytorch-native-cpu-precxx11-$VERSION-linux-$arch.jar"
    url="https://repo1.maven.org/maven2/ai/djl/pytorch/pytorch-native-cpu-precxx11/$VERSION/$(basename "$jar")"
    for delay in 2 4 8 16 32 0; do
        curl -sSfL -o "$jar.part" "$url" && break
        [ "$delay" = 0 ] && { echo "fetch-libtorch: cannot download $url" >&2; exit 1; }
        sleep "$delay"
    done
    echo "$SHA256  $jar.part" | sha256sum -c --status || { echo "fetch-libtorch: checksum mismatch" >&2; exit 1; }
    rm -rf "$lib.part" && mkdir -p "$lib.part"
    unzip -o -j -q "$jar.part" "pytorch/cpu-precxx11/linux-$arch/*" -d "$lib.part"
    rm -f "$jar.part"
    mv "$lib.part" "$lib"
fi
echo "$lib"
