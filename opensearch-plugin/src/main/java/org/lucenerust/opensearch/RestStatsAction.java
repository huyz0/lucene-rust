/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.lucenerust.opensearch.engine.RustEngineFactory;

import org.opensearch.core.rest.RestStatus;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
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
            b.field("live_queries", stats.liveCount());
            b.field("native_errors", stats.errorCount());
            b.field("script_errors", stats.scriptErrorCount());
            b.field("native_fetches", stats.fetchCount());
            b.field("native_sequential_fetches", stats.sequentialFetchCount());
            b.field("native_doc_values", stats.docValuesCount());
            b.field("native_highlights", stats.highlightCount());
            b.field("native_sources_without_vectors", stats.sourcesWithoutVectorsCount());
            // k-NN hits a request's later phases (fetch, scroll pages) took from its first rewrite.
            b.startObject("knn_hits");
            b.field("kept", org.lucenerust.opensearch.knn.KnnHitsMemo.keptCount());
            b.field("reused", org.lucenerust.opensearch.knn.KnnHitsMemo.reusedCount());
            b.field("contexts", org.lucenerust.opensearch.knn.KnnHitsMemo.contexts());
            b.endObject();
            b.startObject("fallbacks");
            for (Map.Entry<String, Long> e : stats.fallbackCounts().entrySet()) {
                b.field(e.getKey(), e.getValue());
            }
            b.endObject();
            b.field("open_native_readers", readers.openCount());
            long[] parents = NativeBridge.parentCacheStats();
            b.field("nested_parent_sets", parents[0]);
            b.field("nested_parent_set_bytes", parents[1]);
            // searchWith's own time per path (SearchStats.nativeTime/luceneTime).
            b.startObject("query_phase_nanos");
            b.field("native", stats.nativeNanos());
            b.field("native_count", stats.nativeCount());
            b.field("lucene", stats.luceneNanos());
            b.field("lucene_count", stats.luceneCount());
            b.endObject();
            // Realtime views' readers built for aggregations, sorts and the like (SearchStats.liveReaderTime).
            b.startObject("live_view_reader_nanos");
            b.field("total", stats.liveReaderNanos());
            b.field("count", stats.liveReaderCount());
            b.endObject();
            // StoredFields.document's time per path (SearchStats.fetchTime).
            b.startObject("fetch_nanos");
            b.field("native", stats.fetchNativeNanos());
            b.field("native_count", stats.fetchCount());
            b.field("lucene", stats.fetchLuceneNanos());
            b.field("lucene_count", stats.fetchLuceneCount());
            b.endObject();
            // Engines this node has created for Rust-engine indices, by kind.
            b.startObject("engines");
            b.field("rust", RustEngineFactory.RUST.get());
            b.field("java", RustEngineFactory.JAVA.get());
            b.field("nrt_replica", RustEngineFactory.NRT_REPLICA.get());
            b.field("rust_indexer", org.lucenerust.opensearch.engine.RustIndexerFactory.INDEXERS.get());
            b.field("segrep_primary", org.lucenerust.opensearch.engine.RustIndexerFactory.NATIVE ? "native" : "indexer_factory");
            b.endObject();
            // Text field values the Rust engine's writer analyzed in Rust and in Java.
            long[] analyzed = RustEngineFactory.analyzedValues();
            b.startObject("analyzed_values");
            b.field("native", analyzed[0]);
            b.field("java", analyzed[1]);
            b.endObject();
            // Memory the library holds outside the heap, its bound, and the requests it stopped.
            long[] memory = NativeMemory.stats();
            if (memory != null) {
                long[] requests = stats.requestMemory();
                b.startObject("off_heap_memory");
                b.field("used_in_bytes", memory[0]);
                b.field("peak_in_bytes", memory[1]);
                b.field("limit_in_bytes", memory[2]);
                b.field("request_breaker_trips", memory[3]);
                b.field("limit_trips", memory[4]);
                b.field("cancelled_mid_call", memory[5]);
                b.field("request_peak_in_bytes", requests[0]);
                b.field("request_peaks_sum_in_bytes", requests[1]);
                b.field("requests", requests[2]);
                b.startObject("query_cache");
                b.field("memory_size_in_bytes", memory[6]);
                b.field("limit_in_bytes", memory[7]);
                b.field("evictions", memory[8]);
                b.endObject();
                b.field("sort_columns_in_bytes", memory[9]);
                b.field("global_ordinals_in_bytes", memory[10]);
                b.endObject();
            }
            // Recall-targeted k-NN tuners (index.lucene_rust.knn.target_recall), by index/shard/field.
            String recall = NativeBridge.knnRecallStats();
            if (recall != null) {
                try (
                    XContentParser p = JsonXContent.jsonXContent.createParser(
                        NamedXContentRegistry.EMPTY,
                        DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                        recall
                    )
                ) {
                    b.field("knn_recall", p.map());
                }
            }
            b.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, b));
        };
    }
}
