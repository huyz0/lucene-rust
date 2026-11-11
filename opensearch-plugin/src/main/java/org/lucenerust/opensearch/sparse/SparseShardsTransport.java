/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.lucenerust.opensearch.knn.rest.KnnShardsTransport;
import org.lucenerust.opensearch.neural.NeuralSearchSettings;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.ActionType;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.block.ClusterBlockLevel;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.common.Strings;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.indices.IndicesService;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.TransportService;
import org.opensearch.transport.client.node.NodeClient;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

/**
 * neural-search's {@code _plugins/_neural/warmup/{index}} and {@code clear_cache/{index}} on
 * {@code index.sparse} indexes, under its action names, errors and {@code _shards} responses.
 * neural-search loads forward indexes and clustered postings into JVM caches and evicts them; the
 * plugin keeps none, so both open each shard's searcher (failing a shard that cannot serve one, as
 * neural-search does) and evict nothing.
 */
public final class SparseShardsTransport {
    private SparseShardsTransport() {}

    /** neural-search's {@code seismic_thread_pool}, with its {@code seismic_thread_pool.*} settings. */
    public static final String THREAD_POOL = "seismic_thread_pool";

    public static final class WarmupAction extends ActionType<KnnShardsTransport.Response> {
        public static final WarmupAction INSTANCE = new WarmupAction();
        public static final String NAME = "cluster:admin/neural_sparse_warmup_action";

        private WarmupAction() {
            super(NAME, KnnShardsTransport.Response::new);
        }
    }

    public static final class ClearCacheAction extends ActionType<KnnShardsTransport.Response> {
        public static final ClearCacheAction INSTANCE = new ClearCacheAction();
        public static final String NAME = "cluster:admin/neural_sparse_clear_cache_action";

        private ClearCacheAction() {
            super(NAME, KnnShardsTransport.Response::new);
        }
    }

    /** {@code TransportUtils.validateSparseIndices}. */
    static void validateSparseIndices(ClusterState state, String[] concreteIndices, String operationName) {
        List<String> invalid = Arrays.stream(concreteIndices)
            .filter(
                name -> Optional.ofNullable(state.metadata().index(name))
                    .map(IndexMetadata::getSettings)
                    .map(SparseSettings.SPARSE_INDEX::get)
                    .orElse(false) == false
            )
            .toList();
        if (invalid.isEmpty() == false) {
            throw new OpenSearchStatusException(
                String.format(Locale.ROOT, "Request rejected. Indices [%s] don't support %s operation.", String.join(", ", invalid), operationName),
                RestStatus.BAD_REQUEST
            );
        }
    }

    /** {@code NeuralSparseWarmupTransportAction}. */
    public static final class WarmupTransportAction extends KnnShardsTransport.Base {
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
        protected void validate(ClusterState state, String[] concreteIndices) {
            validateSparseIndices(state, concreteIndices, "neural_sparse_warmup_action");
        }

        @Override
        protected void onShard(IndexShard shard) {
            try (Engine.Searcher searcher = shard.acquireSearcher("neural-sparse-warmup")) {
                searcher.getIndexReader().leaves();
            }
        }
    }

    /** {@code NeuralSparseClearCacheTransportAction}. */
    public static final class ClearCacheTransportAction extends KnnShardsTransport.Base {
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
        protected void validate(ClusterState state, String[] concreteIndices) {
            validateSparseIndices(state, concreteIndices, "neural_sparse_clear_cache_action");
        }

        @Override
        protected void onShard(IndexShard shard) {
            try (Engine.Searcher searcher = shard.acquireSearcher("neural-sparse-clear-cache")) {
                searcher.getIndexReader().leaves();
            }
        }
    }

    /** {@code RestNeuralSparseWarmupHandler} ({@code warmup} true) and {@code RestNeuralSparseClearCacheHandler}. */
    public static final class Rest extends BaseRestHandler {
        private final boolean warmup;

        public Rest(boolean warmup) {
            this.warmup = warmup;
        }

        @Override
        public String getName() {
            return warmup ? "neural_sparse_warmup_action" : "neural_sparse_clear_cache_action";
        }

        @Override
        public List<Route> routes() {
            return List.of(
                new Route(RestRequest.Method.POST, NeuralSearchSettings.NEURAL_BASE_URI + (warmup ? "/warmup/{index}" : "/clear_cache/{index}"))
            );
        }

        @Override
        protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
            KnnShardsTransport.Request r = new KnnShardsTransport.Request(Strings.splitStringByCommaToArray(request.param("index")));
            return channel -> client.execute(
                warmup ? WarmupAction.INSTANCE : ClearCacheAction.INSTANCE,
                r,
                new RestToXContentListener<>(channel)
            );
        }
    }
}
