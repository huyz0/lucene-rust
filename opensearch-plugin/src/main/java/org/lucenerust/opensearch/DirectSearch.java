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
import org.opensearch.search.sort.SortAndFormats;
import org.opensearch.search.sort.SortBuilder;

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
 * and every body feature other than {@code query}, {@code size}, {@code from}, {@code track_total_hits} and field
 * {@code sort} with {@code from + size} up to 100 (aggregations, highlight, source filtering, script or score sorts ...), as well as
 * hits that carry {@code _routing} or {@code _ignored}, an index with a read block or a default search pipeline.
 *
 * <p>It is on by default ({@link #ENABLED}) and never on with the security plugin installed. What it does not go
 * through: the task manager (no cancel, no task listing), the index's search statistics ({@code query_total} and the
 * like), the request circuit breaker and a cluster default timeout; the search slow log is written for it. Its counts
 * are in {@code direct_searches} of the plugin's stats instead.
 */
final class DirectSearch {
    private static final Logger logger = LogManager.getLogger(DirectSearch.class);

    /** Node setting: answer eligible searches directly (see the class comment for what that bypasses). */
    static final Setting<Boolean> ENABLED = Setting.boolSetting("search.accelerated.direct.enabled", true, Setting.Property.NodeScope);

    private static final java.util.Set<String> PLAIN_NAMES = java.util.Set.of("query", "size", "from", "track_total_hits", "sort");
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
        if (flags.enabled() == false || service.mapperService().hasNested() || open(service) == false) {
            return false;
        }
        SearchSourceBuilder source = plainBody(request);
        if (source == null) {
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
            SortAndFormats sorted = source.sorts() == null ? null : SortBuilder.buildSort(source.sorts(), context).orElse(null);
            if (source.sorts() != null && (sorted == null || directSort(sorted) == false)) {
                return false;
            }
            SimilarityEncoder similarities = new SimilarityEncoder(indexSearcher.getSimilarity());
            EncodeContext encodeContext = new EncodeContext(reader.getContext(), service.mapperService()::fieldType, false, EncodeContext.Reader.SEARCH, false);
            QueryEncoder.Encoded encoded = QueryEncoder.encode(query, similarities, encodeContext);
            SortEncoder.Encoded sortBlob = sorted == null ? null
                : SortEncoder.encode(sorted.sort, null, false, new int[0][], 0, false, SortEncoder.indexSorted(reader, sorted.sort), encodeContext);
            if (sortBlob != null && sortBlob.fallbackReason() != null) {
                return false;
            }
            if (encoded.fallbackReason() != null || encoded.fast() == false && flags.allShapes() == false) {
                return false;
            }
            int[] docs = new int[Math.max(from + size, 1)];
            float[] scores = new float[Math.max(from + size, 1)];
            long[] counts = new long[5];
            String[] sortJson = null;
            NativeMemory.queryStarted(0);
            if (sorted == null) {
                if (NativeSearch.search(acquired.handle(), similarities.prefix(encoded.blob()), from + size, countLimit, docs, scores, counts) != NativeBridge.OK) {
                    return false;
                }
            } else {
                sortJson = sortedHits(acquired.handle(), similarities.prefix(encoded.blob()), sorted, sortBlob.blob(), from + size, countLimit, docs, counts);
                if (sortJson == null) {
                    return false;
                }
            }
            String json = response(index, reader, start, from, size, countLimit, docs, scores, counts, sortJson);
            if (json == null) {
                return false;
            }
            reply(channel, index, source, json, start, counts[1]);
            return true;
        }
    }

    /** The search response's JSON, or null when a hit's fields are ones a direct search leaves to OpenSearch. */
    private static String response(String index, IndexReader reader, long start, int from, int size, long countLimit, int[] docs, float[] scores, long[] counts, String[] sortJson)
        throws java.io.IOException {
        int found = (int) counts[0];
        StringBuilder out = new StringBuilder(512 + Math.max(found - from, 0) * 512);
        out.append("{\"took\":").append((System.nanoTime() - start) / 1_000_000)
            .append(",\"timed_out\":false,\"_shards\":{\"total\":1,\"successful\":1,\"skipped\":0,\"failed\":0},\"hits\":{");
        if (countLimit > 0) {
            boolean lowerBound = counts[2] == 1 || counts[1] > countLimit;
            out.append("\"total\":{\"value\":").append(lowerBound ? countLimit : counts[1])
                .append(",\"relation\":\"").append(lowerBound ? "gte" : "eq").append("\"},");
        }
        out.append("\"max_score\":").append(found == 0 || size == 0 || sortJson != null ? "null" : Float.toString(scores[0])).append(",\"hits\":[");
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
            out.append(",\"_score\":").append(sortJson == null ? Float.toString(scores[i]) : "null").append(",\"_source\":")
                .append(new String(sourceBytes.bytes, sourceBytes.offset, sourceBytes.length, StandardCharsets.UTF_8));
            if (sortJson != null) {
                out.append(',').append(sortJson[i]);
            }
            out.append('}');
        }
        out.append("]}}");
        return out.toString();
    }

    private void reply(RestChannel channel, String index, SearchSourceBuilder source, String json, long start, long totalHits) {
        long took = System.nanoTime() - start;
        stats.nativeQuery();
        stats.directSearch();
        stats.nativeTime(took);
        slowLog(index, took, source, totalHits);
        channel.sendResponse(new BytesRestResponse(RestStatus.OK, "application/json", json.getBytes(StandardCharsets.UTF_8)));
    }

    /** The query slow log, as the index's thresholds have it: the first level (warn down to trace) whose threshold the search took. */
    private void slowLog(String index, long tookNanos, SearchSourceBuilder source, long totalHits) {
        org.opensearch.common.settings.Settings settings = services.get(index).getIndexSettings().getSettings();
        Logger slow = LogManager.getLogger("index.search.slowlog.query");
        for (org.apache.logging.log4j.Level level : new org.apache.logging.log4j.Level[] {
            org.apache.logging.log4j.Level.WARN, org.apache.logging.log4j.Level.INFO,
            org.apache.logging.log4j.Level.DEBUG, org.apache.logging.log4j.Level.TRACE }) {
            org.opensearch.common.unit.TimeValue threshold = settings.getAsTime(
                "index.search.slowlog.threshold.query." + level.name().toLowerCase(java.util.Locale.ROOT),
                org.opensearch.common.unit.TimeValue.MINUS_ONE
            );
            if (threshold.nanos() >= 0 && tookNanos >= threshold.nanos()) {
                slow.log(level, "[{}][0] took[{}], took_millis[{}], total_hits[{}], search_type[QUERY_THEN_FETCH], total_shards[1], source[{}]",
                    index, org.opensearch.common.unit.TimeValue.timeValueNanos(tookNanos), tookNanos / 1_000_000, totalHits, source);
                return;
            }
        }
    }

    /** Whether the index, as it is now, lets a search through: no read block and no default search pipeline. */
    private static boolean open(IndexService service) {
        org.opensearch.common.settings.Settings settings = service.getIndexSettings().getSettings();
        String pipeline = settings.get("index.search.default_pipeline");
        return settings.getAsBoolean("index.blocks.read", false) == false && (pipeline == null || pipeline.equals("_none"));
    }

    /** The request's body when it is a plain query, else null: the cost of leaving a request is a scan of the body's top-level names. */
    private static SearchSourceBuilder plainBody(RestRequest request) throws java.io.IOException {
        try (XContentParser names = request.contentParser()) {
            if (names.nextToken() != XContentParser.Token.START_OBJECT) {
                return null;
            }
            while (names.nextToken() == XContentParser.Token.FIELD_NAME) {
                if (PLAIN_NAMES.contains(names.currentName()) == false) {
                    return null;
                }
                names.nextToken();
                names.skipChildren();
            }
        }
        SearchSourceBuilder source;
        try (XContentParser parser = request.contentParser()) {
            source = SearchSourceBuilder.fromXContent(parser);
        }
        return plain(source) && fieldSorts(source) ? source : null;
    }

    /** Whether every sort key is a plain field key: a script, geo-distance or score key is left to OpenSearch. */
    private static boolean fieldSorts(SearchSourceBuilder source) {
        if (source.sorts() == null) {
            return true;
        }
        for (org.opensearch.search.sort.SortBuilder<?> sort : source.sorts()) {
            if (sort instanceof org.opensearch.search.sort.FieldSortBuilder field) {
                if (field.getNestedSort() != null || field.getFieldName().equals("_score")) {
                    return false;
                }
            } else {
                return false;
            }
        }
        return true;
    }

    /** Whether the native engine can answer {@code sort} for a direct search: field keys only, no score. */
    private static boolean directSort(SortAndFormats sorted) {
        if (sorted.sort.needsScores()) {
            return false;
        }
        for (org.apache.lucene.search.SortField field : sorted.sort.getSort()) {
            if (field.getType() == org.apache.lucene.search.SortField.Type.SCORE) {
                return false;
            }
        }
        return true;
    }

    /**
     * The sorted search: fills {@code docs} and {@code counts}, and returns each hit's {@code "sort":[...]} member as
     * OpenSearch writes it, or null when the native engine does not answer.
     */
    private static String[] sortedHits(long handle, byte[] blob, SortAndFormats sorted, byte[] sortBlob, int room, long countLimit, int[] docs, long[] counts)
        throws java.io.IOException {
        org.apache.lucene.search.SortField[] fields = sorted.sort.getSort();
        long[] values = new long[Math.max(room, 1) * fields.length];
        byte[][] terms = new byte[1][];
        if (NativeSearch.searchSorted(handle, blob, sortBlob, Math.max(room, 1), countLimit, docs, values, counts, terms) != NativeBridge.OK) {
            return null;
        }
        int n = room == 0 ? 0 : (int) counts[0];
        counts[0] = n;
        org.apache.lucene.search.FieldDoc[] hits = SortEncoder.hits(fields, n, docs, values, terms[0]);
        String[] out = new String[n];
        for (int i = 0; i < n; i++) {
            try (org.opensearch.core.xcontent.XContentBuilder b = org.opensearch.common.xcontent.json.JsonXContent.contentBuilder()) {
                b.startObject();
                new org.opensearch.search.SearchSortValues(hits[i].fields, sorted.formats).toXContent(b, org.opensearch.core.xcontent.ToXContent.EMPTY_PARAMS);
                b.endObject();
                String json = b.toString();
                out[i] = json.substring(1, json.length() - 1);
            }
        }
        return out;
    }

    /** A body of only a query, paging and a total-hits setting: nothing else changes what the search does or returns. */
    private static boolean plain(SearchSourceBuilder source) {
        return source.aggregations() == null
            && source.postFilter() == null
            && source.highlighter() == null
            && source.suggest() == null
            && source.collapse() == null
            && source.profile() == false
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
