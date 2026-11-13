#!/usr/bin/env bash
# Soak test of the accelerated engine on real multi-node clusters, no Docker needed:
# opensearch-plugin/e2e/soak.py runs 3-node clusters made of tarball distributions under
# write+search load and checks acknowledged writes, shard copies and a query matrix against
# stock OpenSearch. Scenarios: restarts, upgrade (2.19.6 -> 3.9.0 rolling), plugin_swap
# (remove and re-add the plugin), snapshot, ccr (follower on this plugin's engine).
#
#   scripts/soak.sh prepare          # download tarballs, build plugin + patched CCR, lay out $SOAK_DIR
#   scripts/soak.sh run [scenario]   # scenario or `all` (default); each takes 5-14 minutes
#
# Needs root (nodes run as user `osk`, created if missing), 4 cores and 12 GB free, cargo,
# gradle. Each scenario prints findings and exits non-zero if any.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SOAK_DIR="${SOAK_DIR:-$HOME/soak}"
export SOAK_DIR
OS=3.9.0
OLD=2.19.6
KEEP="opensearch-job-scheduler opensearch-knn opensearch-cross-cluster-replication"

lean() { # dir
    for p in "$1"/plugins/*; do case " $KEEP " in *" $(basename "$p") "*) ;; *) rm -rf "$p" ;; esac; done
}

prepare() {
    id osk >/dev/null 2>&1 || useradd -m -s /bin/bash osk
    mkdir -p "$SOAK_DIR" && cd "$SOAK_DIR"
    for v in $OS $OLD; do
        [[ -f opensearch-$v-linux-x64.tar.gz ]] || curl -sSfLO "https://artifacts.opensearch.org/releases/bundle/opensearch/$v/opensearch-$v-linux-x64.tar.gz"
    done
    mkdir -p dist2 dist39
    tar xzf opensearch-$OLD-linux-x64.tar.gz -C dist2 --strip-components=1
    tar xzf opensearch-$OS-linux-x64.tar.gz -C dist39 --strip-components=1
    lean dist2; lean dist39
    tar xzf opensearch-$OS-linux-x64.tar.gz -C dist39 --strip-components=1 opensearch-$OS/agent
    cd - >/dev/null
    export JAVA_HOME_25_X64="$SOAK_DIR/dist39/jdk"
    D=target/opensearch-dist/$OS
    mkdir -p $D/modules && cp -r "$SOAK_DIR/dist39/lib" $D/lib && cp -r "$SOAK_DIR/dist39/modules/lang-painless" $D/modules/
    gradle --no-daemon -q -p opensearch-plugin bundlePlugin -PcargoProfile=release
    ZIP=$(ls opensearch-plugin/build/distributions/lucene-rust-*.zip)
    cp -r "$SOAK_DIR/dist39" "$SOAK_DIR/dist39p"
    rm -rf "$SOAK_DIR/dist39p/plugins/opensearch-knn"   # the plugin serves k-NN itself (engine `extended`)
    JAVA_HOME="$SOAK_DIR/dist39p/jdk" "$SOAK_DIR/dist39p/bin/opensearch-plugin" install --batch "file://$PWD/$ZIP"
    JAVA_HOME_25_X64="$SOAK_DIR/dist39/jdk" JAVA_HOME="$SOAK_DIR/dist39/jdk" opensearch-plugin/ccr/build.sh "$SOAK_DIR/ccrbuild"
    cp -r "$SOAK_DIR/dist39p" "$SOAK_DIR/dist39c"
    rm -rf "$SOAK_DIR/dist39c/plugins/opensearch-cross-cluster-replication"
    JAVA_HOME="$SOAK_DIR/dist39c/jdk" "$SOAK_DIR/dist39c/bin/opensearch-plugin" install --batch \
        "file://$SOAK_DIR/ccrbuild/opensearch-cross-cluster-replication-3.9.0.0.zip"
    chmod -R go+rX "$SOAK_DIR"/dist*
}

case "${1:-run}" in
    prepare) prepare ;;
    run) exec python3 -u opensearch-plugin/e2e/soak.py "${2:-all}" ;;
    *) echo "usage: $0 prepare | run [scenario]" >&2; exit 2 ;;
esac
