/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.join;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.support.ActionFilter;
import org.opensearch.action.support.ActionFilterChain;
import org.opensearch.action.support.ActionRequestMetadata;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.tasks.Task;

/**
 * Wraps the {@code has_child} and {@code has_parent} queries of a search request in {@link
 * NativeJoinQueryBuilder}s, so the shards searched natively run the joins natively. A request
 * that names a remote cluster's index is left alone: that cluster may not have this plugin to read
 * the wrapper.
 */
public final class JoinRequestFilter implements ActionFilter {
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
            wrap(s);
        }
        chain.proceed(task, action, request, listener);
    }

    static void wrap(SearchRequest s) {
        SearchSourceBuilder source = s.source();
        if (source == null || source.query() == null || remote(s.indices())) {
            return;
        }
        if (JoinQueries.holdsJoin(source.query())) {
            source.query(JoinQueries.wrap(source.query()));
        }
    }

    private static boolean remote(String[] indices) {
        for (String index : indices) {
            if (index.indexOf(':') >= 0) {
                return true;
            }
        }
        return false;
    }
}
