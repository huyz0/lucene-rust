#!/usr/bin/env bash
# node.sh NAME IMAGE DATADIR [extra docker args]
set -e
docker rm -f "$1" >/dev/null 2>&1 || true
docker run -d --name "$1" -p 9250:9200 -v "$3":/usr/share/opensearch/data "${@:4}" \
  -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
  -e cluster.routing.allocation.disk.threshold_enabled=false \
  -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms4g -Xmx4g" "$2" >/dev/null
for _ in $(seq 1 120); do curl -sf localhost:9250 >/dev/null && echo UP && exit 0; sleep 2; done
docker logs --tail 50 "$1"; exit 1
