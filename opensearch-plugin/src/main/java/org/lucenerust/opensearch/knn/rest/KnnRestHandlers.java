/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.rest;

import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.common.Strings;
import org.opensearch.core.index.Index;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestActions;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Supplier;

import static org.opensearch.action.support.IndicesOptions.strictExpandOpen;

/**
 * k-NN's REST endpoints under {@code _plugins/_knn}: {@code stats} ({@code RestKNNStatsHandler}),
 * {@code warmup/{index}} ({@code RestKNNWarmupHandler}) and {@code clear_cache/{index}}
 * ({@code RestClearCacheHandler}), with the stock plugin's routes, parameters and errors.
 */
public final class KnnRestHandlers {
    static final String BASE = "/_plugins/_knn";

    private KnnRestHandlers() {}

    /** Every handler. */
    public static List<RestHandler> all(Supplier<ClusterService> clusterService, IndexNameExpressionResolver resolver) {
        return List.of(new Stats(), new Shards(true, clusterService, resolver), new Shards(false, clusterService, resolver));
    }

    /** {@code RestKNNStatsHandler}. */
    static final class Stats extends BaseRestHandler {
        @Override
        public String getName() {
            return "knn_stats_action";
        }

        @Override
        public List<Route> routes() {
            return List.of(
                new Route(RestRequest.Method.GET, BASE + "/{nodeId}/stats/"),
                new Route(RestRequest.Method.GET, BASE + "/{nodeId}/stats/{stat}"),
                new Route(RestRequest.Method.GET, BASE + "/stats/"),
                new Route(RestRequest.Method.GET, BASE + "/stats/{stat}")
            );
        }

        @Override
        protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
            String nodes = request.param("nodeId");
            KnnStatsTransport.Request r = new KnnStatsTransport.Request(
                nodes == null || nodes.isEmpty() ? null : nodes.split(",")
            );
            r.timeout(request.param("timeout"));
            String stats = request.param("stat");
            Set<String> wanted = stats == null || stats.isEmpty() ? null : new HashSet<>(Arrays.asList(stats.split(",")));
            if (wanted == null || (wanted.size() == 1 && wanted.contains(KnnStatsTransport.Request.ALL))) {
                r.all();
            } else if (wanted.contains(KnnStatsTransport.Request.ALL)) {
                throw new IllegalArgumentException("Request " + request.path() + " contains _all and individual stats");
            } else {
                Set<String> invalid = new TreeSet<>();
                for (String stat : wanted) {
                    if (r.add(stat) == false) {
                        invalid.add(stat);
                    }
                }
                if (invalid.isEmpty() == false) {
                    throw new IllegalArgumentException(unrecognized(request, invalid, KnnStats.names(), "stat"));
                }
            }
            return channel -> client.execute(KnnStatsTransport.Action.INSTANCE, r, new RestActions.NodesResponseRestListener<>(channel));
        }
    }

    /** {@code RestKNNWarmupHandler} and {@code RestClearCacheHandler}. */
    static final class Shards extends BaseRestHandler {
        private final boolean warmup;
        private final Supplier<ClusterService> clusterService;
        private final IndexNameExpressionResolver resolver;

        Shards(boolean warmup, Supplier<ClusterService> clusterService, IndexNameExpressionResolver resolver) {
            this.warmup = warmup;
            this.clusterService = clusterService;
            this.resolver = resolver;
        }

        @Override
        public String getName() {
            return warmup ? "knn_warmup_action" : "knn_clear_cache_action";
        }

        @Override
        public List<Route> routes() {
            return warmup
                ? List.of(new Route(RestRequest.Method.GET, BASE + "/warmup/{index}"))
                : List.of(new Route(RestRequest.Method.POST, BASE + "/clear_cache/{index}"));
        }

        @Override
        protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
            String[] names = Strings.splitStringByCommaToArray(request.param("index"));
            ClusterService cluster = clusterService.get();
            Index[] indices = resolver.concreteIndices(cluster.state(), strictExpandOpen(), names);
            List<String> invalid = Arrays.stream(indices)
                .filter(i -> "true".equals(cluster.state().metadata().getIndexSafe(i).getSettings().get("index.knn")) == false)
                .map(Index::getName)
                .toList();
            if (invalid.isEmpty() == false) {
                throw new KNNInvalidIndicesException(
                    invalid,
                    (warmup ? "Warm up" : "ClearCache") + " request rejected. One or more indices have 'index.knn' set to false."
                );
            }
            KnnShardsTransport.Request r = new KnnShardsTransport.Request(names);
            return channel -> client.execute(
                warmup ? KnnShardsTransport.WarmupAction.INSTANCE : KnnShardsTransport.ClearCacheAction.INSTANCE,
                r,
                new RestToXContentListener<>(channel)
            );
        }
    }
}
