/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.action.ActionType;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.single.shard.SingleShardRequest;
import org.opensearch.action.support.single.shard.TransportSingleShardAction;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.block.ClusterBlockException;
import org.opensearch.cluster.block.ClusterBlockLevel;
import org.opensearch.cluster.metadata.AliasMetadata;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.routing.ShardsIterator;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexService;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.mapper.DocumentMapper;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.indices.IndicesService;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestActionListener;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;
import org.opensearch.transport.client.node.NodeClient;

/**
 * Read-only SQL over an index, native: {@code POST /_plugins/accelerated/sql} with {@code {"query": "SELECT ..."}}.
 *
 * <p>The statement runs through a transport action named {@code indices:data/read/accelerated_sql}, not from the REST
 * layer, so the security plugin authenticates the caller and checks their index permissions for it exactly as it
 * does for a search, and a request for an index on another node is routed to a node holding a copy. On that node
 * the statement runs over the searcher's reader, which the security plugin's field- and document-level security
 * wraps: a user sees the documents and fields (masked as configured) that a search of theirs sees, or the
 * statement is refused when the native side cannot honour the wrapper.
 *
 * <p>The table is the index (or an alias that resolves to one unfiltered index) named in {@code FROM}, quoted when
 * its name is not a plain identifier ({@code FROM "logs-2026"}). Its columns are the index's keyword, integer
 * and date fields and its text fields ({@link SqlTables}); {@code match_text(column, 'words')} searches a text
 * column and orders by {@code _score}. The index must have one shard. A statement that returns more than {@link
 * #MAX_ROWS} rows fails. The response has the shape of the SQL plugin's ({@code schema}, {@code datarows}, {@code total},
 * {@code size}, {@code status}), which serves {@code /_plugins/_sql} when it is installed: the two do not share a path.
 */
public final class AcceleratedSql {
    private AcceleratedSql() {}

    private static final org.apache.logging.log4j.Logger logger = org.apache.logging.log4j.LogManager.getLogger(AcceleratedSql.class);

    /** The path of the endpoint. */
    public static final String PATH = "/_plugins/accelerated/sql";

    /** Node setting: serve {@link #PATH}. */
    public static final Setting<Boolean> ENABLED = Setting.boolSetting("plugins.accelerated.sql.enabled", true, Setting.Property.NodeScope);

    /** Node setting: answer a statement from the REST layer when this node holds the index (never with the security plugin). */
    public static final Setting<Boolean> DIRECT = Setting.boolSetting("plugins.accelerated.sql.direct", true, Setting.Property.NodeScope);

    /** Node setting: the most rows a statement may return. */
    public static final Setting<Integer> MAX_ROWS = Setting.intSetting("plugins.accelerated.sql.max_rows", 10_000, 1, Setting.Property.NodeScope);

    /** Node setting: the memory one table's operators (sorts, aggregations) may hold at once. */
    public static final Setting<org.opensearch.core.common.unit.ByteSizeValue> MEMORY_LIMIT = Setting.byteSizeSetting(
        "plugins.accelerated.sql.memory_limit",
        new org.opensearch.core.common.unit.ByteSizeValue(512, org.opensearch.core.common.unit.ByteSizeUnit.MB),
        Setting.Property.NodeScope
    );

    /** Node setting: how long a statement may run before it is stopped; 0 for no limit. */
    public static final Setting<org.opensearch.common.unit.TimeValue> TIMEOUT = Setting.timeSetting(
        "plugins.accelerated.sql.timeout",
        org.opensearch.common.unit.TimeValue.timeValueSeconds(30),
        org.opensearch.common.unit.TimeValue.ZERO,
        Setting.Property.NodeScope
    );

    private static volatile long timeoutNanos;

    /** Sets the statement time limit from {@link #TIMEOUT}. */
    static void configureTimeout(org.opensearch.common.unit.TimeValue timeout) {
        timeoutNanos = timeout.nanos();
    }

    /** The table each recent statement reads, so a repeated statement is not parsed twice. */
    private static final Map<String, String> TABLE_NAMES = new ConcurrentHashMap<>();

    /** Longer statements are not remembered, so the map holds at most 4096 short strings. */
    private static final int MAX_REMEMBERED_STATEMENT = 2048;

    private static volatile NativeReaders readers;
    private static volatile SearchStats stats;

    /** What the transport action reads, set by the plugin before the node serves requests. */
    static void install(NativeReaders nativeReaders, SearchStats searchStats) {
        readers = nativeReaders;
        stats = searchStats;
    }

    public static final class Action extends ActionType<Response> {
        public static final Action INSTANCE = new Action();
        public static final String NAME = "indices:data/read/accelerated_sql";

        private Action() {
            super(NAME, Response::new);
        }
    }

    public static final class Request extends SingleShardRequest<Request> {
        private String sql;
        private String table;

        Request(String table, String sql) {
            super(table);
            this.table = table;
            this.sql = sql;
        }

        Request(StreamInput in) throws IOException {
            super(in);
            sql = in.readString();
            table = in.readString();
        }

        @Override
        public ActionRequestValidationException validate() {
            ActionRequestValidationException e = validateNonNullIndex();
            if (sql == null || sql.isBlank()) {
                e = org.opensearch.action.ValidateActions.addValidationError("query is missing", e);
            }
            return e;
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeString(sql);
            out.writeString(table);
        }

        String sql() {
            return sql;
        }

        /** The name the statement gives the table: what the index or alias was called in {@code FROM}. */
        String table() {
            return table;
        }
    }

    public static final class Response extends ActionResponse {
        private final byte[] json;

        Response(byte[] json) {
            this.json = json;
        }

        Response(StreamInput in) throws IOException {
            super(in);
            json = in.readByteArray();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeByteArray(json);
        }

        byte[] json() {
            return json;
        }
    }

    public static final class TransportAction extends TransportSingleShardAction<Request, Response> {
        private final IndicesService indices;

        @Inject
        public TransportAction(
            ThreadPool threadPool,
            ClusterService clusterService,
            TransportService transportService,
            ActionFilters actionFilters,
            IndexNameExpressionResolver resolver,
            IndicesService indices
        ) {
            super(Action.NAME, threadPool, clusterService, transportService, actionFilters, resolver, Request::new, ThreadPool.Names.SEARCH);
            this.indices = indices;
            NativeMemory.rememberBreakers(indices.getCircuitBreakerService());
        }

        @Override
        protected boolean resolveIndex(Request request) {
            return true;
        }

        @Override
        protected ShardsIterator shards(ClusterState state, InternalRequest request) {
            return clusterService.operationRouting().getShards(state, request.concreteIndex(), 0, null);
        }

        @Override
        protected ClusterBlockException checkRequestBlock(ClusterState state, InternalRequest request) {
            return state.blocks().indexBlockedException(ClusterBlockLevel.READ, request.concreteIndex());
        }

        @Override
        protected Writeable.Reader<Response> getResponseReader() {
            return Response::new;
        }

        @Override
        protected Response shardOperation(Request request, ShardId shardId) throws IOException {
            IndexService service = indices.indexServiceSafe(shardId.getIndex());
            IndexShard shard = service.getShard(shardId.id());
            Runner.refuse(request.table(), service, clusterService.state());
            return new Response(Runner.run(shard, service, request.table(), request.sql()));
        }
    }

    /** Runs a statement over a local shard: what the transport action and the direct path share. */
    static final class Runner {
        private Runner() {}

        private record Cached(DocumentMapper mapper, SqlTables.Table table) {}

        private static final Map<String, Cached> TABLES = new ConcurrentHashMap<>();

        /** Throws when {@code service}'s index is not one SQL runs over, or {@code table} reaches it by a filtered alias. */
        static void refuse(String table, IndexService service, ClusterState state) {
            if (unsupported(table, service, state) instanceof String reason) {
                throw new OpenSearchStatusException(reason, RestStatus.BAD_REQUEST);
            }
        }

        /** Why SQL does not run over the index {@code table} names, or null when it does. */
        static String unsupported(String table, IndexService service, ClusterState state) {
            org.opensearch.index.IndexSettings settings = service.getIndexSettings();
            if (settings.getNumberOfShards() != 1) {
                return "SQL runs over an index with one shard; [" + service.index().getName() + "] has " + settings.getNumberOfShards();
            }
            if (org.lucenerust.opensearch.engine.AcceleratedEngineSupport.REALTIME_SEARCH.get(settings.getSettings())
                || service.mapperService().hasNested()) {
                return "SQL does not run over an index with realtime search or nested fields";
            }
            if (table.equals(service.index().getName()) == false) {
                IndexMetadata meta = state.metadata().index(service.index());
                for (AliasMetadata alias : meta.getAliases().values()) {
                    if (alias.filteringRequired()) {
                        return "[" + table + "] resolves to an index with a filtered alias, which SQL does not apply";
                    }
                }
            }
            return null;
        }

        /** The index as a table: its columns follow its mapping, found again when the mapping changes. */
        static SqlTables.Table table(String index, String name, IndexService service) {
            DocumentMapper mapper = service.mapperService().documentMapper();
            String key = index + '\0' + name;
            Cached cached = TABLES.get(key);
            if (cached != null && cached.mapper() == mapper) {
                return cached.table();
            }
            SqlTables.Table built = SqlTables.of(name, service.mapperService());
            TABLES.put(key, new Cached(mapper, built));
            return built;
        }

        static byte[] run(IndexShard shard, IndexService service, String name, String sql) throws IOException {
            SqlTables.Table table = table(service.index().getName(), name, service);
            long start = System.nanoTime();
            try (Engine.Searcher searcher = shard.acquireSearcher("accelerated_sql")) {
                NativeReaders.Acquired acquired = readers.acquire(searcher.getIndexReader());
                if (acquired.handle() == 0) {
                    throw new OpenSearchStatusException(
                        "the native reader cannot serve this index to this user: " + acquired.fallbackReason(),
                        RestStatus.BAD_REQUEST
                    );
                }
                // The statement's native memory is charged to the request breaker, and it stops at
                // the time limit.
                NativeMemory.queryStarted(timeoutNanos);
                byte[] json;
                try (NativeMemory.Scope memory = NativeMemory.open(NativeMemory.breakerService(), () -> false, "accelerated_sql")) {
                    try {
                        json = NativeSql.query(acquired.handle(), table.blob(), sql);
                    } finally {
                        NativeMemory.queryStarted(0);
                    }
                } 
                stats.sqlStatement(System.nanoTime() - start);
                return json;
            } catch (NativeSql.Failure e) {
                throw new OpenSearchStatusException(e.getMessage(), status(e));
            }
        }

        private static RestStatus status(NativeSql.Failure e) {
            return switch (e.status()) {
                case NativeBridge.MEMORY_LIMIT, NativeBridge.TOO_MANY_BUCKETS -> RestStatus.TOO_MANY_REQUESTS;
                case NativeBridge.CANCELLED -> RestStatus.REQUEST_TIMEOUT;
                case NativeBridge.INVALID_ARGUMENT, NativeBridge.SEARCH_FAILED, NativeBridge.DECODE -> RestStatus.BAD_REQUEST;
                default -> RestStatus.INTERNAL_SERVER_ERROR;
            };
        }
    }

    /**
     * Answers a statement from the REST layer when this node holds a copy of the index: the statement runs on the
     * search thread pool without the transport action's routing, filters and serialisation. Never with the security
     * plugin installed, whose index permission check is the transport action's: the plugin does not create it then.
     */
    static final class Direct {
        private final Map<String, IndexService> services = new ConcurrentHashMap<>();
        private final java.util.function.Supplier<ClusterService> cluster;
        private final java.util.function.Supplier<IndexNameExpressionResolver> resolver;

        Direct(java.util.function.Supplier<ClusterService> cluster, java.util.function.Supplier<IndexNameExpressionResolver> resolver) {
            this.cluster = cluster;
            this.resolver = resolver;
        }

        /** Tracks the index's service while it lives. */
        void track(org.opensearch.index.IndexModule indexModule) {
            indexModule.addIndexEventListener(new org.opensearch.index.shard.IndexEventListener() {
                @Override
                public void afterIndexCreated(IndexService service) {
                    services.put(service.index().getName(), service);
                }

                @Override
                public void afterIndexRemoved(
                    org.opensearch.core.index.Index index,
                    org.opensearch.index.IndexSettings indexSettings,
                    org.opensearch.indices.cluster.IndicesClusterStateService.AllocatedIndices.IndexRemovalReason reason
                ) {
                    services.remove(index.getName());
                }
            });
        }

        /** The consumer that runs the statement here, or null when it is the transport action's to run or refuse. */
        org.opensearch.common.CheckedConsumer<org.opensearch.rest.RestChannel, Exception> consumer(NodeClient client, String table, String sql) {
            ClusterService clusterService = cluster.get();
            IndexNameExpressionResolver names = resolver.get();
            if (clusterService == null || names == null || table.indexOf(':') >= 0) {
                return null;
            }
            ClusterState state = clusterService.state();
            String[] concrete;
            try {
                concrete = names.concreteIndexNames(state, org.opensearch.action.support.IndicesOptions.strictExpandOpenAndForbidClosedIgnoreThrottled(), table);
            } catch (RuntimeException e) {
                return null;
            }
            IndexService service = concrete.length == 1 ? services.get(concrete[0]) : null;
            IndexShard shard = service == null ? null : service.getShardOrNull(0);
            if (shard == null
                || shard.state() != org.opensearch.index.shard.IndexShardState.STARTED
                || state.blocks().indexBlockedException(ClusterBlockLevel.READ, concrete[0]) != null
                || Runner.unsupported(table, service, state) != null) {
                return null;
            }
            return channel -> client.threadPool().executor(ThreadPool.Names.SEARCH).execute(() -> {
                try {
                    byte[] json = Runner.run(shard, service, table, sql);
                    stats.sqlDirect();
                    channel.sendResponse(new BytesRestResponse(RestStatus.OK, "application/json", json));
                } catch (Exception | LinkageError e) {
                    try {
                        channel.sendResponse(new BytesRestResponse(channel, e instanceof Exception ex ? ex : new IllegalStateException(e)));
                    } catch (IOException io) {
                        logger.warn("accelerated-search: could not report a SQL failure", io);
                    }
                }
            });
        }
    }

    /** {@code POST} or {@code GET} {@link #PATH}, the statement in the body's {@code query}. */
    public static final class Rest extends BaseRestHandler {
        private final Direct direct;

        /** {@code direct} answers from this node when it can; null leaves every statement to the transport action. */
        Rest(Direct direct) {
            this.direct = direct;
        }

        @Override
        public String getName() {
            return "accelerated_sql";
        }

        @Override
        public List<Route> routes() {
            return List.of(new Route(RestRequest.Method.POST, PATH), new Route(RestRequest.Method.GET, PATH));
        }

        @Override
        protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
            String format = request.param("format", "jdbc");
            if (format.equals("jdbc") == false && format.equals("json") == false) {
                throw new IllegalArgumentException("format [" + format + "] is not supported; the response is the SQL plugin's jdbc format");
            }
            String sql = request.param("query");
            if (sql == null && request.hasContent()) {
                try (XContentParser parser = request.contentParser()) {
                    Object value = parser.map().get("query");
                    sql = value == null ? null : value.toString();
                }
            }
            if (sql == null || sql.isBlank()) {
                throw new IllegalArgumentException("the request needs a [query]");
            }
            String table = sql.length() <= MAX_REMEMBERED_STATEMENT ? TABLE_NAMES.get(sql) : null;
            if (table == null) {
                try {
                    table = NativeSql.table(sql);
                } catch (NativeSql.Failure e) {
                    throw new IllegalArgumentException(e.getMessage());
                }
                if (sql.length() <= MAX_REMEMBERED_STATEMENT) {
                    if (TABLE_NAMES.size() >= 4096) {
                        TABLE_NAMES.clear();
                    }
                    TABLE_NAMES.put(sql, table);
                }
            }
            if (direct != null) {
                org.opensearch.common.CheckedConsumer<org.opensearch.rest.RestChannel, Exception> local = direct.consumer(client, table, sql);
                if (local != null) {
                    return local::accept;
                }
            }
            Request execute = new Request(table, sql);
            return channel -> client.execute(Action.INSTANCE, execute, new RestActionListener<>(channel) {
                @Override
                protected void processResponse(Response response) {
                    channel.sendResponse(new BytesRestResponse(RestStatus.OK, "application/json", response.json()));
                }
            });
        }
    }
}
