#!/usr/bin/env bash
# Extracts lib/ of the pinned OpenSearch distribution from its Docker image,
# for opensearch-plugin/'s build to compile against.
#
# The plugin must match its node's exact version, and the image is the
# distribution the end-to-end tests install into -- so compiling against the
# image's own jars means the build and the node cannot disagree, and the build
# needs no Maven repository.
#
#   scripts/opensearch-dist.sh            # -> target/opensearch-dist/3.8.0/lib
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="${OPENSEARCH_VERSION:-3.8.0}"
IMAGE="opensearchproject/opensearch:${VERSION}"
OUT="target/opensearch-dist/${VERSION}"

if [[ -f "$OUT/lib/opensearch-${VERSION}.jar" ]]; then
    echo "opensearch-dist: $OUT/lib already present"
    exit 0
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
mkdir -p "$OUT"
cid=$(docker create "$IMAGE")
trap 'docker rm -f "$cid" >/dev/null' EXIT
docker cp "$cid:/usr/share/opensearch/lib" "$OUT/lib"
echo "opensearch-dist: extracted $(ls "$OUT/lib" | wc -l) jars to $OUT/lib"
