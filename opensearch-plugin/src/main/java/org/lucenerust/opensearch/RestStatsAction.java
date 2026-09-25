/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestRequest;
import org.opensearch.transport.client.node.NodeClient;

import java.util.List;
import java.util.Map;

/**
 * {@code GET /_plugins/lucene_rust/stats}: this node's native and fallback query counts by reason,
 * native errors, and open native readers -- the numbers that say whether the supported matrix fits
 * the traffic, and whether native readers are being released.
 */
public final class RestStatsAction extends BaseRestHandler {
    private final SearchStats stats;
    private final NativeReaders readers;
    private final String libraryPath;

    RestStatsAction(SearchStats stats, NativeReaders readers, String libraryPath) {
        this.stats = stats;
        this.readers = readers;
        this.libraryPath = libraryPath;
    }

    @Override
    public String getName() {
        return "lucene_rust_stats";
    }

    @Override
    public List<Route> routes() {
        return List.of(new Route(RestRequest.Method.GET, "/_plugins/lucene_rust/stats"));
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
        return channel -> {
            XContentBuilder b = channel.newBuilder();
            b.startObject();
            b.field("library", libraryPath);
            b.field("abi_version", NativeBridge.abiVersion());
            b.field("native_queries", stats.nativeCount());
            b.field("native_errors", stats.errorCount());
            b.startObject("fallbacks");
            for (Map.Entry<String, Long> e : stats.fallbackCounts().entrySet()) {
                b.field(e.getKey(), e.getValue());
            }
            b.endObject();
            b.field("open_native_readers", readers.openCount());
            b.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, b));
        };
    }
}
