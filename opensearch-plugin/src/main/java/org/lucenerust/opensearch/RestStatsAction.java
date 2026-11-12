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
            b.field("accelerated_queries", stats.nativeCount());
            b.field("live_queries", stats.liveCount());
            b.field("accelerated_errors", stats.errorCount());
            b.field("script_errors", stats.scriptErrorCount());
            b.field("accelerated_fetches", stats.fetchCount());
            b.field("accelerated_sequential_fetches", stats.sequentialFetchCount());
            b.field("accelerated_doc_values", stats.docValuesCount());
            b.field("accelerated_highlights", stats.highlightCount());
            b.field("accelerated_sources_without_vectors", stats.sourcesWithoutVectorsCount());
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
            b.field("open_accelerated_readers", readers.openCount());
            long[] parents = NativeBridge.parentCacheStats();
            b.field("nested_parent_sets", parents[0]);
            b.field("nested_parent_set_bytes", parents[1]);
            long[] agg = new long[NativeBridge.AGG_CACHE_STATS];
            if (NativeBridge.aggCacheStats(agg) == NativeBridge.OK) {
                b.startObject("aggregation_segment_cache");
                b.field("hits", agg[0]);
                b.field("misses", agg[1]);
                b.field("evictions", agg[2]);
                b.field("entries", agg[3]);
                b.field("bytes", agg[4]);
                b.field("budget_bytes", agg[5]);
                b.endObject();
            }
            // searchWith's own time per path (SearchStats.nativeTime/luceneTime).
            b.startObject("query_phase_nanos");
            b.field("accelerated", stats.nativeNanos());
            b.field("accelerated_count", stats.nativeCount());
            b.field("fallback", stats.luceneNanos());
            b.field("fallback_count", stats.luceneCount());
            b.endObject();
            // Realtime views' readers built for aggregations, sorts and the like (SearchStats.liveReaderTime).
            b.startObject("live_view_reader_nanos");
            b.field("total", stats.liveReaderNanos());
            b.field("count", stats.liveReaderCount());
            b.endObject();
            // StoredFields.document's time per path (SearchStats.fetchTime).
            b.startObject("fetch_nanos");
            b.field("accelerated", stats.fetchNativeNanos());
            b.field("accelerated_count", stats.fetchCount());
            b.field("fallback", stats.fetchLuceneNanos());
            b.field("fallback_count", stats.fetchLuceneCount());
            b.endObject();
            // Engines this node has created for accelerated-engine indices, by kind.
            b.startObject("engines");
            b.field("accelerated", RustEngineFactory.RUST.get());
            b.field("standard", RustEngineFactory.JAVA.get());
            b.field("nrt_replica", RustEngineFactory.NRT_REPLICA.get());
            b.field("accelerated_indexer", org.lucenerust.opensearch.engine.RustIndexerFactory.INDEXERS.get());
            b.field("segrep_primary", org.lucenerust.opensearch.engine.RustIndexerFactory.NATIVE ? "accelerated" : "indexer_factory");
            b.endObject();
            // Text field values the accelerated writer analyzed on its own path and on the fallback.
            long[] analyzed = RustEngineFactory.analyzedValues();
            b.startObject("analyzed_values");
            b.field("accelerated", analyzed[0]);
            b.field("fallback", analyzed[1]);
            b.endObject();
            // The page cache (page_cache and encrypted shards): its memory is native, outside
            // the heap OpenSearch measures; resident bytes are also charged to the indexing_buffers
            // breaker (PageCacheAccounting).
            long[] pages = NativeBridge.storePageCacheStats();
            b.startObject("page_cache");
            b.field("budget_in_bytes", pages[0]);
            b.field("resident_in_bytes", pages[1]);
            b.field("whole_files_in_bytes", pages[2]);
            b.field("faults", pages[3]);
            b.field("refaults", pages[4]);
            b.field("evictions", pages[5]);
            b.field("read_errors", pages[6]);
            b.field("fault_time_in_nanos", pages[7]);
            b.endObject();
            // Recall-targeted k-NN tuners (index.knn.target_recall), by index/shard/field.
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
