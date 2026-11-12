#!/bin/bash
# up.sh NAME PORT IMAGE [docker args]: start a node, wait until it answers or exits.
docker rm -f $1 >/dev/null 2>&1
docker run -d --name $1 -p $2:9200 "${@:4}" -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true -e cluster.routing.allocation.disk.threshold_enabled=false -e DISABLE_INSTALL_DEMO_CONFIG=true -e OPENSEARCH_JAVA_OPTS="-Xms${HEAP:-4g} -Xmx${HEAP:-4g}" $3 >/dev/null
for _ in $(seq 1 90); do curl -sf localhost:$2 >/dev/null && exit 0; [ "$(docker inspect -f '{{.State.Running}}' $1)" = true ] || { docker logs $1 2>&1 | grep -m3 "root cause\|Exception"; exit 1; }; sleep 2; done; exit 1
