#!/usr/bin/env bash
# M6's soak (T6.1): three OpenSearch 3.8.0 nodes, every index on the Rust
# engine, under opensearch-plugin/e2e/soak.py's mixed load with random
# restarts and SIGKILLs, instrumented every minute; at the end real Lucene's
# CheckIndex over every shard copy.
#
#   scripts/soak-opensearch.sh start DIR [--hours H] [--no-build]
#   scripts/soak-opensearch.sh resume DIR    after the environment went down
#   scripts/soak-opensearch.sh check DIR     stop the nodes, CheckIndex them
#   scripts/soak-opensearch.sh report DIR
#
# Each node's data lives under DIR/data/<node>, bind-mounted, so it outlives
# the containers -- and the environment, which may be recycled while idle
# (see soak.py). Needs Docker, a JDK and what scripts/verify-opensearch.sh
# needs to build the plugin image.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION=3.8.0
IMAGE="lucene-rust-opensearch:${VERSION}"
CMD="${1:-}"
DIR="${2:-}"
[[ -n "$CMD" && -n "$DIR" ]] || { sed -n '2,15p' "$0"; exit 2; }
shift 2
HOURS=168
BUILD=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hours) HOURS="$2"; shift 2 ;;
        --no-build) BUILD=0; shift ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$DIR"
DIR="$(cd "$DIR" && pwd)"

ensure_docker() {
    docker info >/dev/null 2>&1 && return
    # The environment's daemon does not survive a restart of the environment.
    (setsid nohup dockerd >"$DIR/dockerd.log" 2>&1 </dev/null &)
    for _ in $(seq 60); do docker info >/dev/null 2>&1 && return; sleep 1; done
    echo "soak: docker did not come up" >&2; exit 1
}

node() { # name http-port transport-port
    docker rm -f "$1" >/dev/null 2>&1 || true
    mkdir -p "$DIR/data/$1"
    chown -R 1000:1000 "$DIR/data/$1"
    docker run -d --name "$1" --network container:lucene-rust-net \
        -v "$DIR/data/$1:/usr/share/opensearch/data" \
        -e node.name="$1" -e cluster.name=lucene-rust-soak \
        -e http.host=0.0.0.0 -e http.port="$2" \
        -e transport.host=127.0.0.1 -e transport.port="$3" \
        -e discovery.seed_hosts=127.0.0.1:9301,127.0.0.1:9302,127.0.0.1:9303 \
        -e cluster.initial_cluster_manager_nodes=os1,os2,os3 \
        -e lucene_rust.engine.default=true \
        -e DISABLE_SECURITY_PLUGIN=true -e DISABLE_INSTALL_DEMO_CONFIG=true \
        -e cluster.routing.allocation.disk.threshold_enabled=false \
        -e OPENSEARCH_JAVA_OPTS="-Xms1g -Xmx1g" \
        "$IMAGE" >/dev/null
}

cluster() {
    docker rm -f lucene-rust-net >/dev/null 2>&1 || true
    docker run -d --name lucene-rust-net -p 9201:9201 -p 9202:9202 -p 9203:9203 \
        --entrypoint sleep "opensearchproject/opensearch:${VERSION}" infinity >/dev/null
    node os1 9201 9301
    node os2 9202 9302
    node os3 9203 9303
    for _ in $(seq 180); do
        curl -s localhost:9201/_cluster/health 2>/dev/null | grep -q '"number_of_nodes":3' && return
        sleep 2
    done
    echo "soak: the cluster did not form" >&2; exit 1
}

run() { # extra soak.py arguments
    (setsid nohup python3 opensearch-plugin/e2e/soak.py run "$DIR" --hours "$HOURS" "$@" \
        >>"$DIR/soak.log" 2>&1 </dev/null &)
    echo "soak: running; logs and traces in $DIR"
}

case "$CMD" in
    start)
        ensure_docker
        if [[ $BUILD == 1 ]]; then
            scripts/opensearch-dist.sh
            gradle --no-daemon -q -p opensearch-plugin bundlePlugin
            rm -f opensearch-plugin/docker/*.zip
            cp opensearch-plugin/build/distributions/lucene-rust-*.zip opensearch-plugin/docker/
            docker build -q -t "$IMAGE" opensearch-plugin/docker >/dev/null
        fi
        [[ -e "$DIR/model.json" ]] && { echo "soak: $DIR holds a soak; use resume" >&2; exit 1; }
        git rev-parse HEAD >"$DIR/commit"
        cluster
        run
        ;;
    resume)
        ensure_docker
        pgrep -f "soak.py run $DIR" >/dev/null && { echo "soak: already running"; exit 0; }
        HOURS=$(python3 -c "import json;print(json.load(open('$DIR/model.json'))['state'].get('target_h', $HOURS))" 2>/dev/null || echo "$HOURS")
        cluster
        run --resume
        ;;
    check)
        ensure_docker
        pkill -f "soak.py run $DIR" || true
        docker stop -t 120 os1 os2 os3 >/dev/null 2>&1 || true
        CP="$(ls target/opensearch-dist/${VERSION}/lib/lucene-core-*.jar)"
        fail=0
        n=0
        for idx in "$DIR"/data/os*/nodes/0/indices/*/*/index; do
            [[ -e "$idx/write.lock" ]] && rm -f "$idx/write.lock"
            n=$((n + 1))
            if java -cp "$CP" org.apache.lucene.index.CheckIndex "$idx" >"$DIR/checkindex.$n.log" 2>&1; then
                echo "CheckIndex ok: $idx"
            else
                echo "CheckIndex FAILED: $idx (see $DIR/checkindex.$n.log)"
                fail=1
            fi
        done
        echo "soak: CheckIndex over $n shard copies, $([[ $fail == 0 ]] && echo 'all clean' || echo 'FAILURES')"
        exit $fail
        ;;
    report)
        python3 opensearch-plugin/e2e/soak.py report "$DIR"
        ;;
    *)
        sed -n '2,15p' "$0"; exit 2 ;;
esac
