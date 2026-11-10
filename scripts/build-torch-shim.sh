#!/usr/bin/env bash
# Builds the TorchScript shim (crates/lucene-ffi/torch/lucene_torch.cpp) the
# plugin runs TorchScript models through, once per libtorch C++ string ABI:
#
#   DIR/liblucene_torch_cxx11.so      for libtorch's C++11-ABI builds
#   DIR/liblucene_torch_precxx11.so   for its pre-C++11-ABI builds (DJL's default)
#
# libtorch's headers come from PyTorch's 2.5.1 wheel on PyPI (the version
# ml-commons runs), pinned by SHA-256 and cached in DIR; the aarch64 wheel is
# used on every platform for being the smallest, and its headers are the
# same. Nothing links against libtorch at build time. Needs g++ (C++17) and
# python3.
#
#   scripts/build-torch-shim.sh DIR    -> the two library paths on stdout
set -euo pipefail
WHEEL=torch-2.5.1-cp312-cp312-manylinux2014_aarch64.whl
URL=https://files.pythonhosted.org/packages/6d/69/d8ada8b6e0a4257556d5b4ddeb4345ea8eeaaef3c98b60d1cca197c7ad8e/$WHEEL
SHA256=3f4b7f10a247e0dcd7ea97dc2d3bfbfc90302ed36d7f3952b0008d0df264e697
dir=${1:?usage: build-torch-shim.sh DIR}
src=$(cd "$(dirname "$0")/.." && pwd)/crates/lucene-ffi/torch/lucene_torch.cpp
mkdir -p "$dir"
dir=$(cd "$dir" && pwd)
include="$dir/torch-2.5.1-include"
if [ ! -f "$include/.complete" ]; then
    wheel="$dir/$WHEEL"
    if ! echo "$SHA256  $wheel" | sha256sum -c --status 2>/dev/null; then
        for delay in 2 4 8 16 32 0; do
            curl -sSfL -o "$wheel.part" "$URL" && break
            [ "$delay" = 0 ] && { echo "build-torch-shim: cannot download $URL" >&2; exit 1; }
            sleep "$delay"
        done
        echo "$SHA256  $wheel.part" | sha256sum -c --status || { echo "build-torch-shim: checksum mismatch" >&2; exit 1; }
        mv "$wheel.part" "$wheel"
    fi
    rm -rf "$include"
    python3 -I - "$wheel" "$include" <<'PY'
import os, sys, zipfile
wheel, out = sys.argv[1], sys.argv[2]
prefix = "torch/include/"
with zipfile.ZipFile(wheel) as z:
    for name in z.namelist():
        if name.startswith(prefix) and not name.endswith("/"):
            rel = os.path.normpath(name[len(prefix):])
            if os.path.isabs(rel) or rel.startswith(".."):
                sys.exit(f"unsafe path in wheel: {name}")
            target = os.path.join(out, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(name) as f, open(target, "wb") as o:
                o.write(f.read())
PY
    touch "$include/.complete"
    rm -f "$wheel"
fi
for abi in cxx11 precxx11; do
    flag=$([ "$abi" = cxx11 ] && echo 1 || echo 0)
    out="$dir/liblucene_torch_$abi.so"
    g++ -O2 -shared -fPIC -std=c++17 -fvisibility=hidden -D_GLIBCXX_USE_CXX11_ABI="$flag" \
        -isystem "$include" -isystem "$include/torch/csrc/api/include" \
        -Wl,-soname,"liblucene_torch_$abi.so" "$src" -o "$out.part"
    # The newest runtime versions it may need: GCC 9's libstdc++ and glibc 2.17.
    too_new=$(objdump -T "$out.part" | grep -oE 'GLIBCXX_3\.4\.[0-9]+|GLIBC_2\.[0-9]+' | sort -u \
        | awk -F'[._]' '/GLIBCXX/ && $4 > 26 {print} /^GLIBC_/ && $3 > 17 {print}')
    [ -z "$too_new" ] || { echo "build-torch-shim: $abi needs $too_new" >&2; exit 1; }
    mv "$out.part" "$out"
    echo "$out"
done
