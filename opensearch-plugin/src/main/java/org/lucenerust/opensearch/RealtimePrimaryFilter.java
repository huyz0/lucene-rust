/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.lucenerust.opensearch.engine.RustEngineSupport;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.search.CreatePitAction;
import org.opensearch.action.search.CreatePitRequest;
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.support.ActionFilter;
import org.opensearch.action.support.ActionFilterChain;
import org.opensearch.action.support.ActionRequestMetadata;
import org.opensearch.action.support.IndicesOptions;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.routing.Preference;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.indices.replication.common.ReplicationType;
import org.opensearch.tasks.Task;

import java.util.function.Supplier;

/**
 * Searches of an index whose searches read the writer's buffered documents ({@code
 * index.search.realtime.enabled}) go to its primaries when its replicas copy segments (segment
 * replication, remote store): such a replica has no writer, so it would see only what the primary
 * last refreshed. A request naming its own {@code preference} keeps it. As OpenSearch routes a
 * realtime get to the primary under segment replication.
 */
final class RealtimePrimaryFilter implements ActionFilter {
    private final Supplier<ClusterService> clusterService;
    private final Supplier<IndexNameExpressionResolver> resolver;

    RealtimePrimaryFilter(Supplier<ClusterService> clusterService, Supplier<IndexNameExpressionResolver> resolver) {
        this.clusterService = clusterService;
        this.resolver = resolver;
    }

    @Override
    public int order() {
        return 0;
    }

    @Override
    public <Request extends ActionRequest, Response extends ActionResponse> void apply(
        Task task,
        String action,
        Request request,
        ActionRequestMetadata<Request, Response> metadata,
        ActionListener<Response> listener,
        ActionFilterChain<Request, Response> chain
    ) {
        if (SearchAction.NAME.equals(action) && request instanceof SearchRequest s) {
            if (s.preference() == null && needsPrimary(s.indices(), s.indicesOptions())) {
                s.preference(Preference.PRIMARY.type());
            }
        } else if (CreatePitAction.NAME.equals(action) && request instanceof CreatePitRequest p) {
            if (p.getPreference() == null && needsPrimary(p.getIndices(), p.getIndicesOptions())) {
                p.setPreference(Preference.PRIMARY.type());
            }
        }
        chain.proceed(task, action, request, listener);
    }

    /** Whether any index {@code indices} names reads the buffers and has replicas that copy segments. */
    boolean needsPrimary(String[] indices, IndicesOptions options) {
        ClusterService cluster = clusterService.get();
        IndexNameExpressionResolver names = resolver.get();
        if (cluster == null || names == null) {
            return false;
        }
        ClusterState state = cluster.state();
        String[] concrete;
        try {
            concrete = names.concreteIndexNames(state, options, indices);
        } catch (RuntimeException e) {
            // Missing or remote indices: the search reports them itself.
            return false;
        }
        for (String name : concrete) {
            IndexMetadata index = state.metadata().index(name);
            if (index != null && segmentCopies(index)) {
                return true;
            }
        }
        return false;
    }

    static boolean segmentCopies(IndexMetadata index) {
        Settings settings = index.getSettings();
        return RustEngineSupport.REALTIME_SEARCH.get(settings)
            && index.getNumberOfReplicas() > 0
            && (IndexMetadata.INDEX_REPLICATION_TYPE_SETTING.get(settings) == ReplicationType.SEGMENT
                || settings.getAsBoolean(IndexMetadata.SETTING_REMOTE_STORE_ENABLED, false));
    }
}
