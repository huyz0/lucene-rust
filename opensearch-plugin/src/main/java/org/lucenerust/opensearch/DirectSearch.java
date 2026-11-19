/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DocumentStoredFieldVisitor;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexService;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.Uid;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryShardContext;
import org.opensearch.index.query.Rewriteable;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestRequest;
import org.opensearch.search.builder.SearchSourceBuilder;

/**
 * Answers a plain search on an accelerated-engine index straight from the REST layer: the request's query is
 * parsed and built by OpenSearch's own code, encoded and run natively, and the hits' {@code _id} and
 * {@code _source} read natively and written as JSON, without the coordinating search action, the shard search
 * service, its thread-pool hand-off and the response objects. That machinery costs about 0.6 ms per request on
 * every engine, which is most of a cheap search's time.
 *
 * <p>It is a request-level shortcut for what the search would anyway do natively. Anything else is left to
 * OpenSearch untouched: the request's parameters other than {@code request_cache}, an alias, wildcard or list of
 * indices, an index that is not a single-shard accelerated index, one with realtime search, derived source, nested
 * fields, per-index accelerated search off, a query the native encoder refuses (or a shape slower than Lucene's),
 * and every body feature other than {@code query}, {@code size}, {@code from} and {@code track_total_hits} with
 * {@code from + size} up to 100 (sort, aggregations, highlight, source filtering ...), as well as hits that carry {@code _routing} or
 * {@code _ignored}.
 *
 * <p>What it does not do, which is why it is off by default ({@link #ENABLED}) and never on with the security
 * plugin installed: it does not go through the request's authorization or the task manager (no cancel, no task
 * listing), the search slow log, the index's search statistics, the request circuit breaker or a timeout; its
 * counts are in {@code direct_searches} of the plugin's stats instead.
 */
final class DirectSearch {
    private static final Logger logger = LogManager.getLogger(DirectSearch.class);

    /** Node setting: answer eligible searches directly (see the class comment for what that bypasses). */
    static final Setting<Boolean> ENABLED = Setting.boolSetting("search.accelerated.direct.enabled", false, Setting.Property.NodeScope);

    private static final java.util.Set<String> PLAIN_NAMES = java.util.Set.of("query", "size", "from", "track_total_hits");
    /** The most hits a direct search collects (from + size). */
    private static final int MAX_SIZE = 100;
    private static final long TOTAL_HITS_THRESHOLD = 10_000;

    private final NativeReaders readers;
    private final SearchStats stats;
    private final Map<String, IndexService> services = new ConcurrentHashMap<>();
    private final Map<String, IndexShard> shards = new ConcurrentHashMap<>();

    DirectSearch(NativeReaders readers, SearchStats stats) {
        this.readers = readers;
        this.stats = stats;
    }

    /** Tracks the index's service, and its one shard once started, while it is eligible. */
    void track(org.opensearch.index.IndexModule indexModule) {
        indexModule.addIndexEventListener(new org.opensearch.index.shard.IndexEventListener() {
            @Override
            public void afterIndexCreated(IndexService service) {
                if (eligibleIndex(service)) {
                    services.put(service.index().getName(), service);
                }
            }

            @Override
            public void afterIndexShardStarted(IndexShard shard) {
                IndexService service = services.get(shard.shardId().getIndexName());
                if (service != null && shard.routingEntry().primary()) {
                    shards.put(shard.shardId().getIndexName(), shard);
                }
            }

            @Override
            public void beforeIndexShardClosed(org.opensearch.core.index.shard.ShardId id, IndexShard shard, org.opensearch.common.settings.Settings settings) {
                shards.remove(id.getIndexName(), shard);
            }

            @Override
            public void afterIndexRemoved(org.opensearch.core.index.Index index, org.opensearch.index.IndexSettings indexSettings, org.opensearch.indices.cluster.IndicesClusterStateService.AllocatedIndices.IndexRemovalReason reason) {
                services.remove(index.getName());
                shards.remove(index.getName());
            }
        });
    }

    private static boolean eligibleIndex(IndexService service) {
        org.opensearch.index.IndexSettings settings = service.getIndexSettings();
        return org.lucenerust.opensearch.engine.AcceleratedEngineSupport.servedByRust(settings.getSettings())
            && settings.getNumberOfShards() == 1
            && org.lucenerust.opensearch.engine.AcceleratedEngineSupport.REALTIME_SEARCH.get(settings.getSettings()) == false
            && settings.isDerivedSourceEnabled() == false
            && service.mapperService().hasNested() == false;
    }

    /** Whether it answered the request; false leaves it, untouched, to OpenSearch. */
    boolean handle(RestRequest request, RestChannel channel) {
        try {
            return answer(request, channel);
        } catch (Exception | LinkageError e) {
            logger.debug("accelerated-search: direct search left to OpenSearch", e);
            return false;
        }
    }

    private boolean answer(RestRequest request, RestChannel channel) throws Exception {
        String path = request.rawPath();
        if ((request.method() != RestRequest.Method.POST && request.method() != RestRequest.Method.GET)
            || path.endsWith("/_search") == false
            || path.length() <= "/_search".length() + 1
            || request.hasContent() == false) {
            return false;
        }
        String index = path.substring(1, path.length() - "/_search".length());
        IndexShard shard = shards.get(index);
        IndexService service = services.get(index);
        if (shard == null || service == null || shard.state() != org.opensearch.index.shard.IndexShardState.STARTED) {
            return false;
        }
        for (String key : request.params().keySet()) {
            if (key.equals("index") == false && key.equals("request_cache") == false) {
                return false;
            }
        }
        long start = System.nanoTime();
        SearchFlags.Flags flags = SearchFlags.of(service.getIndexSettings());
        if (flags.enabled() == false || service.mapperService().hasNested()) {
            return false;
        }
        // A request that is not a plain query goes back to OpenSearch having cost a scan of the body's top-level
        // names, not a full parse.
        try (XContentParser names = request.contentParser()) {
            if (names.nextToken() != XContentParser.Token.START_OBJECT) {
                return false;
            }
            while (names.nextToken() == XContentParser.Token.FIELD_NAME) {
                String name = names.currentName();
                if (PLAIN_NAMES.contains(name) == false) {
                    return false;
                }
                names.nextToken();
                names.skipChildren();
            }
        }
        SearchSourceBuilder source;
        try (XContentParser parser = request.contentParser()) {
            source = SearchSourceBuilder.fromXContent(parser);
        }
        if (plain(source) == false) {
            return false;
        }
        int size = source.size() < 0 ? 10 : source.size();
        int from = Math.max(source.from(), 0);
        if (from + size > MAX_SIZE) {
            return false;
        }
        Integer tracked = source.trackTotalHitsUpTo();
        long countLimit = tracked == null ? TOTAL_HITS_THRESHOLD : tracked < 0 ? 0 : tracked == Integer.MAX_VALUE ? Long.MAX_VALUE : tracked;
        try (Engine.Searcher searcher = shard.acquireSearcher("direct_search")) {
            IndexReader reader = searcher.getIndexReader();
            NativeReaders.Acquired acquired = readers.acquire(reader);
            if (acquired.handle() == 0) {
                return false;
            }
            IndexSearcher indexSearcher = new IndexSearcher(reader);
            indexSearcher.setSimilarity(service.similarityService().similarity(service.mapperService()));
            QueryShardContext context = service.newQueryShardContext(0, indexSearcher, System::currentTimeMillis, null);
            QueryBuilder builder = source.query() == null ? new MatchAllQueryBuilder() : source.query();
            Query query = indexSearcher.rewrite(Rewriteable.rewrite(builder, context, true).toQuery(context));
            SimilarityEncoder similarities = new SimilarityEncoder(indexSearcher.getSimilarity());
            QueryEncoder.Encoded encoded = QueryEncoder.encode(
                query,
                similarities,
                new EncodeContext(reader.getContext(), service.mapperService()::fieldType, false, EncodeContext.Reader.SEARCH, false)
            );
            if (encoded.fallbackReason() != null || encoded.fast() == false && flags.allShapes() == false) {
                return false;
            }
            int[] docs = new int[from + size];
            float[] scores = new float[from + size];
            long[] counts = new long[3];
            NativeMemory.queryStarted(0);
            if (NativeSearch.search(acquired.handle(), similarities.prefix(encoded.blob()), from + size, countLimit, docs, scores, counts) != NativeBridge.OK) {
                return false;
            }
            String json = response(index, reader, start, from, countLimit, docs, scores, counts);
            if (json == null) {
                return false;
            }
            stats.nativeQuery();
            stats.directSearch();
            stats.nativeTime(System.nanoTime() - start);
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, "application/json", json.getBytes(StandardCharsets.UTF_8)));
            return true;
        }
    }

    /** The search response's JSON, or null when a hit's fields are ones a direct search leaves to OpenSearch. */
    private static String response(String index, IndexReader reader, long start, int from, long countLimit, int[] docs, float[] scores, long[] counts)
        throws java.io.IOException {
        int found = (int) counts[0];
        StringBuilder out = new StringBuilder(512 + Math.max(found - from, 0) * 512);
        out.append("{\"took\":").append((System.nanoTime() - start) / 1_000_000)
            .append(",\"timed_out\":false,\"_shards\":{\"total\":1,\"successful\":1,\"skipped\":0,\"failed\":0},\"hits\":{");
        if (countLimit > 0) {
            boolean lowerBound = counts[2] == 1;
            out.append("\"total\":{\"value\":").append(lowerBound ? countLimit : counts[1])
                .append(",\"relation\":\"").append(lowerBound ? "gte" : "eq").append("\"},");
        }
        out.append("\"max_score\":").append(found == 0 ? "null" : Float.toString(scores[0])).append(",\"hits\":[");
        for (int i = from; i < found; i++) {
            DocumentStoredFieldVisitor visitor = new DocumentStoredFieldVisitor("_id", "_source", "_routing", "_ignored");
            reader.storedFields().document(docs[i], visitor);
            Document doc = visitor.getDocument();
            BytesRef id = doc.getBinaryValue("_id");
            BytesRef sourceBytes = doc.getBinaryValue("_source");
            if (id == null || sourceBytes == null || doc.getField("_routing") != null || doc.getField("_ignored") != null) {
                return null;
            }
            if (i > from) {
                out.append(',');
            }
            out.append("{\"_index\":");
            json(out, index);
            out.append(",\"_id\":");
            json(out, Uid.decodeId(id.bytes, id.offset, id.length));
            out.append(",\"_score\":").append(scores[i]).append(",\"_source\":")
                .append(new String(sourceBytes.bytes, sourceBytes.offset, sourceBytes.length, StandardCharsets.UTF_8)).append('}');
        }
        out.append("]}}");
        return out.toString();
    }

    /** A body of only a query, paging and a total-hits setting: nothing else changes what the search does or returns. */
    private static boolean plain(SearchSourceBuilder source) {
        return source.aggregations() == null
            && source.postFilter() == null
            && source.highlighter() == null
            && source.suggest() == null
            && source.collapse() == null
            && source.profile() == false
            && source.sorts() == null
            && (source.rescores() == null || source.rescores().isEmpty())
            && source.fetchSource() == null
            && source.storedFields() == null
            && source.docValueFields() == null
            && (source.scriptFields() == null || source.scriptFields().isEmpty())
            && source.minScore() == null
            && source.searchAfter() == null
            && source.timeout() == null
            && source.terminateAfter() == 0
            && source.slice() == null
            && Boolean.TRUE.equals(source.explain()) == false
            && Boolean.TRUE.equals(source.version()) == false
            && Boolean.TRUE.equals(source.seqNoAndPrimaryTerm()) == false
            && source.trackScores() == false
            && source.pointInTimeBuilder() == null
            && (source.ext() == null || source.ext().isEmpty())
            && (source.indexBoosts() == null || source.indexBoosts().isEmpty())
            && source.size() <= MAX_SIZE
            && (source.getDerivedFieldsObject() == null || source.getDerivedFieldsObject().isEmpty())
            && (source.stats() == null || source.stats().isEmpty())
            && (source.searchPipelineSource() == null || source.searchPipelineSource().isEmpty());
    }

    /** {@code s} as a JSON string literal. */
    private static void json(StringBuilder out, String s) {
        out.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        out.append('"');
    }
}
