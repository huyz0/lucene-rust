/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.rest;

import org.opensearch.action.ActionType;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.broadcast.BroadcastRequest;
import org.opensearch.action.support.broadcast.BroadcastResponse;
import org.opensearch.action.support.broadcast.node.TransportBroadcastByNodeAction;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.block.ClusterBlockException;
import org.opensearch.cluster.block.ClusterBlockLevel;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.routing.ShardRouting;
import org.opensearch.cluster.routing.ShardsIterator;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.action.support.DefaultShardOperationFailedException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.indices.IndicesService;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.util.List;

/**
 * k-NN's per-shard actions, {@code KNNWarmupAction} and {@code ClearCacheAction}, under the stock
 * plugin's action names and responses ({@code _shards}). The stock plugin loads native-engine graphs
 * into its cache and evicts them; a lucene-engine field has none, and the stock plugin does nothing
 * for one but acquire a searcher. Here warmup opens each shard's searcher, so its segments are open
 * before the first query, and clearing the cache has nothing to evict.
 */
public final class KnnShardsTransport {
    private KnnShardsTransport() {}

    /** {@code KNNWarmupRequest} and {@code ClearCacheRequest}. */
    public static final class Request extends BroadcastRequest<Request> {
        public Request(String... indices) {
            super(indices);
        }

        public Request(StreamInput in) throws IOException {
            super(in);
        }
    }

    /** {@code KNNWarmupResponse} and {@code ClearCacheResponse}. */
    public static final class Response extends BroadcastResponse {
        public Response(StreamInput in) throws IOException {
            super(in);
        }

        Response(int total, int successful, int failed, List<DefaultShardOperationFailedException> failures) {
            super(total, successful, failed, failures);
        }
    }

    public static final class WarmupAction extends ActionType<Response> {
        public static final WarmupAction INSTANCE = new WarmupAction();
        public static final String NAME = "cluster:admin/knn_warmup_action";

        private WarmupAction() {
            super(NAME, Response::new);
        }
    }

    public static final class ClearCacheAction extends ActionType<Response> {
        public static final ClearCacheAction INSTANCE = new ClearCacheAction();
        public static final String NAME = "cluster:admin/clear_cache_action";

        private ClearCacheAction() {
            super(NAME, Response::new);
        }
    }

    /** One operation on every shard of the requested indices, as the stock transport actions run it. */
    abstract static class Base extends TransportBroadcastByNodeAction<Request, Response, TransportBroadcastByNodeAction.EmptyResult> {
        private final IndicesService indices;
        private final ClusterBlockLevel level;

        Base(
            String name,
            ClusterBlockLevel level,
            ClusterService clusterService,
            TransportService transportService,
            ActionFilters actionFilters,
            IndexNameExpressionResolver resolver,
            IndicesService indices
        ) {
            super(name, clusterService, transportService, actionFilters, resolver, Request::new, ThreadPool.Names.SEARCH);
            this.indices = indices;
            this.level = level;
        }

        abstract void onShard(IndexShard shard) throws IOException;

        @Override
        protected EmptyResult readShardResult(StreamInput in) throws IOException {
            return EmptyResult.readEmptyResultFrom(in);
        }

        @Override
        protected Response newResponse(
            Request request,
            int totalShards,
            int successfulShards,
            int failedShards,
            List<EmptyResult> results,
            List<DefaultShardOperationFailedException> failures,
            ClusterState state
        ) {
            return new Response(totalShards, successfulShards, failedShards, failures);
        }

        @Override
        protected Request readRequestFrom(StreamInput in) throws IOException {
            return new Request(in);
        }

        @Override
        protected EmptyResult shardOperation(Request request, ShardRouting routing) throws IOException {
            onShard(indices.indexServiceSafe(routing.shardId().getIndex()).getShard(routing.shardId().id()));
            return EmptyResult.INSTANCE;
        }

        @Override
        protected ShardsIterator shards(ClusterState state, Request request, String[] concreteIndices) {
            return state.routingTable().allShards(concreteIndices);
        }

        @Override
        protected ClusterBlockException checkGlobalBlock(ClusterState state, Request request) {
            return state.blocks().globalBlockedException(level);
        }

        @Override
        protected ClusterBlockException checkRequestBlock(ClusterState state, Request request, String[] concreteIndices) {
            return state.blocks().indicesBlockedException(level, concreteIndices);
        }
    }

    /** {@code KNNWarmupTransportAction}. */
    public static final class WarmupTransportAction extends Base {
        @Inject
        public WarmupTransportAction(
            ClusterService clusterService,
            TransportService transportService,
            ActionFilters actionFilters,
            IndexNameExpressionResolver resolver,
            IndicesService indices
        ) {
            super(WarmupAction.NAME, ClusterBlockLevel.METADATA_READ, clusterService, transportService, actionFilters, resolver, indices);
        }

        @Override
        void onShard(IndexShard shard) {
            if (shard.indexSettings().isWarmIndex()) {
                // Read on demand from the remote store: nothing to load ahead.
                return;
            }
            try (Engine.Searcher searcher = shard.acquireSearcher("knn-warmup")) {
                searcher.getIndexReader().leaves();
            }
        }
    }

    /** {@code ClearCacheTransportAction}. */
    public static final class ClearCacheTransportAction extends Base {
        @Inject
        public ClearCacheTransportAction(
            ClusterService clusterService,
            TransportService transportService,
            ActionFilters actionFilters,
            IndexNameExpressionResolver resolver,
            IndicesService indices
        ) {
            super(ClearCacheAction.NAME, ClusterBlockLevel.METADATA_WRITE, clusterService, transportService, actionFilters, resolver, indices);
        }

        @Override
        void onShard(IndexShard shard) {}
    }
}
