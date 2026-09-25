# A single OpenSearch node with the lucene-rust plugin installed, for
# scripts/verify-opensearch.sh.
#
# The bundled plugins are removed: neural-search registers its own
# QueryPhaseSearcher, and OpenSearch allows exactly one. An operator running
# both must choose -- see docs/opensearch-native-queries.md.
ARG OPENSEARCH_VERSION=3.8.0
FROM opensearchproject/opensearch:${OPENSEARCH_VERSION}
RUN rm -rf /usr/share/opensearch/plugins/*
COPY lucene-rust-*.zip /tmp/lucene-rust.zip
RUN /usr/share/opensearch/bin/opensearch-plugin install --batch file:///tmp/lucene-rust.zip
