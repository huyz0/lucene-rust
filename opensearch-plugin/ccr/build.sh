#!/usr/bin/env bash
# Builds the cross-cluster replication plugin 3.9.0.0 with the configurable follower engine
# (0001-configurable-follower-engine.patch): opensearch-cross-cluster-replication-3.9.0.0.zip
# in ./build. Install it on the leader and follower clusters in place of the bundled one.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${1:-$here/build}"
rm -rf "$work/src" && mkdir -p "$work"
git clone -q --depth 1 --branch 3.9.0.0 https://github.com/opensearch-project/cross-cluster-replication.git "$work/src"
git -C "$work/src" apply "$here/0001-configurable-follower-engine.patch"
(cd "$work/src" && ./gradlew -q -Dopensearch.version=3.9.0 -Dbuild.snapshot=false assemble -x test)
cp "$work/src/build/distributions/opensearch-cross-cluster-replication-3.9.0.0.zip" "$work/"
echo "built $work/opensearch-cross-cluster-replication-3.9.0.0.zip"
