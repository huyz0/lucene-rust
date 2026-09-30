# A single OpenSearch node with the lucene-rust plugin installed, for
# scripts/verify-opensearch.sh.
#
# The bundled plugins are removed: neural-search registers its own
# QueryPhaseSearcher, and OpenSearch allows exactly one. An operator running
# both must choose -- see docs/opensearch-native-queries.md. KEEP_PLUGINS
# names bundled plugins to keep (space-separated; scripts/verify-opensearch-upgrade.sh
# keeps opensearch-knn, which the upgraded 2.x index's knn_vector fields need).
ARG OPENSEARCH_VERSION=3.8.0
FROM opensearchproject/opensearch:${OPENSEARCH_VERSION}
ARG KEEP_PLUGINS=""
RUN cd /usr/share/opensearch/plugins && for p in *; do \
        case " ${KEEP_PLUGINS} " in *" $p "*) ;; *) rm -rf "$p" ;; esac; \
    done
COPY lucene-rust-*.zip /tmp/lucene-rust.zip
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch file:///tmp/lucene-rust.zip
# The plugin reaches liblucene_ffi through the Foreign Function & Memory API,
# whose downcalls are restricted methods: without this the JVM warns on first
# use, and a later JDK refuses. (OpenSearch's own JNA needs it too.)
RUN echo '--enable-native-access=ALL-UNNAMED' > /usr/share/opensearch/config/jvm.options.d/lucene-rust.options
