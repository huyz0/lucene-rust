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
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.searchafter.SearchAfterBuilder;
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
 * OpenSearch untouched: query-string parameters that could change the answer, an alias, wildcard or list that
 * resolves to several indices or has a filtered alias, an index that is not a single-shard accelerated index, one with realtime search, derived source, k-NN vectors (the k-NN source-excludes processor is part of its search), nested
 * fields, per-index accelerated search off, a query the native encoder refuses (or a shape slower than Lucene's),
 * and every body feature other than {@code query}, {@code size}, {@code from}, {@code track_total_hits}, {@code
 * _source}, {@code min_score}, {@code search_after} and field, score, script or geo-distance {@code sort} keys with {@code from + size} up
 * to 100 (aggregations, highlight, stored or doc-value fields, nested sorts ...), an index with a read block or a default search
 * pipeline.
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

    private static final java.util.Set<String> PLAIN_NAMES = java.util.Set.of(
        "query", "size", "from", "track_total_hits", "sort", "_source", "search_after", "min_score", "aggs", "aggregations", "docvalue_fields", "stored_fields"
    );
    private static final java.util.Set<String> PLAIN_PARAMS = java.util.Set.of(
        "index", "request_cache", "preference", "routing", "ignore_unavailable", "allow_no_indices", "expand_wildcards",
        "ccs_minimize_roundtrips", "max_concurrent_shard_requests", "pre_filter_shard_size", "batched_reduce_size", "typed_keys",
        "rest_total_hits_as_int", "size", "from", "track_total_hits", "_source", "_source_includes", "_source_excludes",
        "_source_include", "_source_exclude"
    );
    /** The most hits a direct search collects (from + size). */
    private static final int MAX_SIZE = 100;
    private static final long TOTAL_HITS_THRESHOLD = 10_000;

    private final NativeReaders readers;
    private final SearchStats stats;
    private final Map<String, IndexService> services = new ConcurrentHashMap<>();
    private final Map<String, IndexShard> shards = new ConcurrentHashMap<>();

    private final java.util.function.Supplier<org.opensearch.cluster.service.ClusterService> cluster;
    private final java.util.function.Supplier<org.opensearch.cluster.metadata.IndexNameExpressionResolver> resolver;

    DirectSearch(
        NativeReaders readers,
        SearchStats stats,
        java.util.function.Supplier<org.opensearch.cluster.service.ClusterService> cluster,
        java.util.function.Supplier<org.opensearch.cluster.metadata.IndexNameExpressionResolver> resolver
    ) {
        this.readers = readers;
        this.stats = stats;
        this.cluster = cluster;
        this.resolver = resolver;
    }

    /**
     * The one index a path's index expression stands for: a concrete name, or an alias, wildcard or list that
     * resolves to exactly one index with no filtered alias on it (a filter would narrow the documents searched).
     * Null leaves the request to OpenSearch.
     */
    private String concrete(String expression) {
        if (shards.containsKey(expression)) {
            return expression;
        }
        org.opensearch.cluster.service.ClusterService clusterService = cluster.get();
        org.opensearch.cluster.metadata.IndexNameExpressionResolver names = resolver.get();
        if (expression.indexOf(':') >= 0 || clusterService == null || names == null) {
            return null;
        }
        org.opensearch.cluster.ClusterState state = clusterService.state();
        String[] concrete = names.concreteIndexNames(state, org.opensearch.action.support.IndicesOptions.strictExpandOpenAndForbidClosedIgnoreThrottled(), expression);
        org.opensearch.cluster.metadata.IndexMetadata meta = concrete.length == 1 ? state.metadata().index(concrete[0]) : null;
        if (meta == null) {
            return null;
        }
        for (org.opensearch.cluster.metadata.AliasMetadata alias : meta.getAliases().values()) {
            if (alias.filteringRequired()) {
                return null;
            }
        }
        return concrete[0];
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
            && settings.getSettings().getAsBoolean("index.knn", false) == false
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
        String index = concrete(path.substring(1, path.length() - "/_search".length()));
        IndexShard shard = index == null ? null : shards.get(index);
        IndexService service = services.get(index);
        if (shard == null || service == null || shard.state() != org.opensearch.index.shard.IndexShardState.STARTED || plainParams(request) == false) {
            return false;
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
        if (source.aggregations() != null
            && DirectAggs.slicing(cluster.get().getSettings(), cluster.get().state().metadata().settings(), service.getIndexSettings().getSettings())) {
            return false;
        }
        DirectResponse.Shape shape = applyParams(request, source);
        if (shape == null || DirectFields.supported(source, service.mapperService()) == false) {
            return false;
        }
        if (shape.totalAsInt() && source.trackTotalHitsUpTo() != Integer.MAX_VALUE) {
            return false;
        }
        int size = source.size() < 0 ? 10 : source.size();
        int from = Math.max(source.from(), 0);
        if (from + size > MAX_SIZE || source.searchAfter() != null && (source.sorts() == null || from > 0)) {
            return false;
        }
        Integer tracked = source.trackTotalHitsUpTo();
        long countLimit = tracked == null ? TOTAL_HITS_THRESHOLD : tracked < 0 ? 0 : tracked == Integer.MAX_VALUE ? Long.MAX_VALUE : tracked;
        try (Engine.Searcher searcher = shard.acquireSearcher("direct_search")) {
            DirectResponse.Hits hits = search(service, searcher.getIndexReader(), source, from + size, countLimit, request);
            if (hits == null) {
                return false;
            }
            if (source.docValueFields() != null && source.docValueFields().isEmpty() == false) {
                QueryShardContext fieldContext = service.newQueryShardContext(0, new IndexSearcher(searcher.getIndexReader()), System::currentTimeMillis, null);
                DirectFields fields = DirectFields.of(source.docValueFields(), service.mapperService(), fieldContext);
                if (fields == null) {
                    return false;
                }
                hits = new DirectResponse.Hits(hits.docs(), hits.scores(), hits.counts(), hits.sort(), hits.maxScore(), hits.aggregations(), fields);
            }
            String json = DirectResponse.write(index, searcher.getIndexReader(), start, from, size, countLimit, hits, shape);
            if (json == null) {
                return false;
            }
            reply(channel, index, source, json, start, hits.counts()[1]);
            return true;
        }
    }

    /** Runs the query (and sort) natively over the reader; null when the native engine does not answer it. */
    private DirectResponse.Hits search(IndexService service, IndexReader reader, SearchSourceBuilder source, int room, long countLimit, RestRequest params) throws Exception {
        NativeReaders.Acquired acquired = readers.acquire(reader);
        if (acquired.handle() == 0) {
            return null;
        }
        IndexSearcher indexSearcher = new IndexSearcher(reader);
        indexSearcher.setSimilarity(service.similarityService().similarity(service.mapperService()));
        QueryShardContext context = service.newQueryShardContext(0, indexSearcher, System::currentTimeMillis, null);
        QueryBuilder builder = source.query() == null ? new MatchAllQueryBuilder() : source.query();
        Query query = indexSearcher.rewrite(Rewriteable.rewrite(builder, context, true).toQuery(context));
        SortAndFormats sorted = source.sorts() == null ? null : SortBuilder.buildSort(source.sorts(), context).orElse(null);
        SimilarityEncoder similarities = new SimilarityEncoder(indexSearcher.getSimilarity());
        EncodeContext encodeContext = new EncodeContext(reader.getContext(), service.mapperService()::fieldType, false, EncodeContext.Reader.SEARCH, false);
        QueryEncoder.Encoded encoded = QueryEncoder.encode(query, similarities, encodeContext);
        SortEncoder.Encoded sortBlob = sorted == null ? null : SortEncoder.encode(
            sorted.sort,
            source.searchAfter() == null ? null : SearchAfterBuilder.buildFieldDoc(sorted, source.searchAfter()),
            false,
            new int[0][],
            0,
            false,
            SortEncoder.indexSorted(reader, sorted.sort),
            encodeContext
        );
        if (encoded.fallbackReason() != null || encoded.fast() == false && SearchFlags.of(service.getIndexSettings()).allShapes() == false
            || sortBlob != null && sortBlob.fallbackReason() != null) {
            return null;
        }
        byte[] blob = encoded.blob();
        if (source.minScore() != null) {
            blob = QueryEncoder.withMinScore(blob, source.minScore());
        }
        blob = similarities.prefix(blob);
        if (source.aggregations() != null) {
            return aggregated(acquired.handle(), blob, service, reader, source, countLimit, params);
        }
        int[] docs = new int[Math.max(room, 1)];
        float[] scores = new float[Math.max(room, 1)];
        long[] counts = new long[5];
        NativeMemory.queryStarted(0);
        if (sorted == null) {
            return NativeSearch.search(acquired.handle(), blob, room, countLimit, docs, scores, counts) == NativeBridge.OK
                ? new DirectResponse.Hits(docs, scores, counts, null, null) : null;
        }
        return sortedHits(acquired.handle(), blob, sorted, sortBlob.blob(), room, countLimit, docs, counts);
    }

    /** The aggregations' pass over the query: no hits (size 0), the total it counted, the aggregations as written. */
    private DirectResponse.Hits aggregated(long handle, byte[] blob, IndexService service, IndexReader reader, SearchSourceBuilder source, long countLimit, RestRequest params) throws java.io.IOException {
        if (source.minScore() != null || source.sorts() != null) {
            return null;
        }
        NativeAggregations.Plan plan = DirectAggs.plan(source, service.mapperService(), reader);
        if (plan == null) {
            return null;
        }
        NativeMemory.queryStarted(0);
        DirectAggs.Result result = DirectAggs.run(handle, blob, plan, reader, countLimit, params);
        if (result == null) {
            return null;
        }
        long[] counts = new long[] { 0, result.total(), result.lowerBound() ? 1 : 0, 0, 0 };
        return new DirectResponse.Hits(new int[1], new float[1], counts, null, null, result.json());
    }

    /** Whether the request's query-string parameters leave the answer as the body alone gives it. */
    private static boolean plainParams(RestRequest request) {
        for (String key : request.params().keySet()) {
            if (PLAIN_PARAMS.contains(key) == false) {
                return false;
            }
        }
        String preference = request.param("preference");
        return preference == null || preference.startsWith("_") == false || preference.equals("_local") || preference.startsWith("_primary");
    }

    /** The query-string parameters that override the body (size, from, track_total_hits, _source) and the response's shape. */
    private static DirectResponse.Shape applyParams(RestRequest request, SearchSourceBuilder source) {
        if (request.hasParam("size")) {
            source.size(request.paramAsInt("size", -1));
        }
        if (request.hasParam("from")) {
            source.from(request.paramAsInt("from", 0));
        }
        String tracked = request.param("track_total_hits");
        if (tracked != null) {
            if (tracked.equals("true") || tracked.equals("false")) {
                source.trackTotalHits(tracked.equals("true"));
            } else {
                source.trackTotalHitsUpTo(Integer.parseInt(tracked));
            }
        }
        boolean totalAsInt = request.paramAsBoolean("rest_total_hits_as_int", false);
        if (totalAsInt && source.trackTotalHitsUpTo() == null) {
            source.trackTotalHits(true);
        }
        FetchSourceContext fetch = FetchSourceContext.parseFromRestRequest(request);
        // A body's own `_source` next to `stored_fields` is left to OpenSearch (plainBody), so only the URL's is explicit here.
        boolean explicit = fetch != null;
        if (fetch == null) {
            fetch = source.fetchSource();
        }
        boolean none = DirectFields.none(source);
        if (none && explicit) {
            return null;
        }
        if (source.storedFields() != null && explicit == false && DirectFields.names(source).contains("_source") == false) {
            fetch = FetchSourceContext.DO_NOT_FETCH_SOURCE;
        }
        return new DirectResponse.Shape(fetch, totalAsInt, none == false);
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
            boolean stored = false;
            boolean sourced = false;
            while (names.nextToken() == XContentParser.Token.FIELD_NAME) {
                if (PLAIN_NAMES.contains(names.currentName()) == false) {
                    return null;
                }
                stored |= names.currentName().equals("stored_fields");
                sourced |= names.currentName().equals("_source");
                if (stored && sourced) {
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

    /** Whether every sort key is one the native sort reads: a field, score, script or geo-distance key, not a nested one. */
    private static boolean fieldSorts(SearchSourceBuilder source) {
        if (source.sorts() == null) {
            return true;
        }
        for (org.opensearch.search.sort.SortBuilder<?> sort : source.sorts()) {
            boolean nested = sort instanceof org.opensearch.search.sort.FieldSortBuilder f ? f.getNestedSort() != null
                : sort instanceof org.opensearch.search.sort.ScriptSortBuilder sc ? sc.getNestedSort() != null
                : sort instanceof org.opensearch.search.sort.GeoDistanceSortBuilder g ? g.getNestedSort() != null
                : sort instanceof org.opensearch.search.sort.ScoreSortBuilder == false;
            if (nested) {
                return false;
            }
        }
        return true;
    }

    /**
     * The sorted search: fills {@code docs} and {@code counts}, and returns each hit's {@code "sort":[...]} member as
     * OpenSearch writes it (null when the native engine does not answer), with the max score a leading score key gives.
     */
    private static DirectResponse.Hits sortedHits(long handle, byte[] blob, SortAndFormats sorted, byte[] sortBlob, int room, long countLimit, int[] docs, long[] counts)
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
        int score = -1;
        for (int k = fields.length - 1; k >= 0; k--) {
            if (fields[k].getType() == org.apache.lucene.search.SortField.Type.SCORE) {
                score = k;
            }
        }
        float[] scores = score < 0 ? null : new float[n];
        for (int i = 0; scores != null && i < n; i++) {
            scores[i] = (float) hits[i].fields[score];
        }
        String lead = n > 0 && fields[0].getType() == org.apache.lucene.search.SortField.Type.SCORE ? Float.toString((float) hits[0].fields[0]) : "null";
        return new DirectResponse.Hits(docs, scores, counts, out, lead);
    }

    /** A body of only a query, paging and a total-hits setting: nothing else changes what the search does or returns. */
    private static boolean plain(SearchSourceBuilder source) {
        return DirectAggs.shape(source)
            && source.postFilter() == null
            && source.highlighter() == null
            && source.suggest() == null
            && source.collapse() == null
            && source.profile() == false
            && (source.rescores() == null || source.rescores().isEmpty())
            && DirectFields.shape(source)
            && (source.scriptFields() == null || source.scriptFields().isEmpty())
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
}
