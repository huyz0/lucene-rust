/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestHandler;
import org.opensearch.rest.RestHandler.Route;
import org.opensearch.rest.RestRequest;
import org.opensearch.transport.client.node.NodeClient;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.opensearch.rest.RestRequest.Method.DELETE;
import static org.opensearch.rest.RestRequest.Method.GET;
import static org.opensearch.rest.RestRequest.Method.POST;
import static org.opensearch.rest.RestRequest.Method.PUT;

/**
 * ml-commons' REST endpoints under {@code _plugins/_ml}, with the stock routes, parameters and
 * handler names, each running the transport action stock runs.
 */
public final class MlRestHandlers {
    static final String BASE = "/_plugins/_ml";

    private MlRestHandlers() {}

    /** Every handler. */
    public static List<RestHandler> all() {
        return List.of(
            new Handler(
                "ml_register_model_action",
                MlActions.REGISTER_MODEL,
                Set.of("deploy"),
                List.of(new Route(POST, BASE + "/models/_register"))
            ),
            new Handler(
                "ml_deploy_model_action",
                MlActions.DEPLOY_MODEL,
                Set.of(),
                List.of(new Route(POST, BASE + "/models/{model_id}/_deploy"))
            ),
            new Handler(
                "ml_undeploy_model_action",
                MlActions.UNDEPLOY_MODEL,
                Set.of(),
                List.of(new Route(POST, BASE + "/models/{model_id}/_undeploy"))
            ),
            new Handler(
                "ml_predict_action",
                MlActions.PREDICT,
                Set.of(),
                List.of(new Route(POST, BASE + "/_predict/{algorithm}/{model_id}"), new Route(POST, BASE + "/models/{model_id}/_predict"))
            ),
            new Handler("ml_get_model_action", MlActions.GET_MODEL, Set.of(), List.of(new Route(GET, BASE + "/models/{model_id}"))),
            new Handler("ml_delete_model_action", MlActions.DELETE_MODEL, Set.of(), List.of(new Route(DELETE, BASE + "/models/{model_id}"))),
            new Handler(
                "ml_search_model_action",
                MlActions.SEARCH_MODELS,
                Set.of(),
                List.of(new Route(POST, BASE + "/models/_search"), new Route(GET, BASE + "/models/_search"))
            ),
            new Handler(
                "ml_register_model_group_action",
                MlActions.REGISTER_MODEL_GROUP,
                Set.of(),
                List.of(new Route(POST, BASE + "/model_groups/_register"))
            ),
            new Handler(
                "ml_update_model_group_action",
                MlActions.UPDATE_MODEL_GROUP,
                Set.of(),
                List.of(new Route(PUT, BASE + "/model_groups/{model_group_id}"))
            ),
            new Handler(
                "ml_get_model_group_action",
                MlActions.GET_MODEL_GROUP,
                Set.of(),
                List.of(new Route(GET, BASE + "/model_groups/{model_group_id}"))
            ),
            new Handler(
                "ml_delete_model_group_action",
                MlActions.DELETE_MODEL_GROUP,
                Set.of(),
                List.of(new Route(DELETE, BASE + "/model_groups/{model_group_id}"))
            ),
            new Handler(
                "ml_search_model_group_action",
                MlActions.SEARCH_MODEL_GROUPS,
                Set.of(),
                List.of(new Route(POST, BASE + "/model_groups/_search"), new Route(GET, BASE + "/model_groups/_search"))
            ),
            new Handler("ml_get_task_action", MlActions.GET_TASK, Set.of(), List.of(new Route(GET, BASE + "/tasks/{task_id}"))),
            new Handler("ml_delete_task_action", MlActions.DELETE_TASK, Set.of(), List.of(new Route(DELETE, BASE + "/tasks/{task_id}"))),
            new Handler(
                "ml_search_task_action",
                MlActions.SEARCH_TASKS,
                Set.of(),
                List.of(new Route(POST, BASE + "/tasks/_search"), new Route(GET, BASE + "/tasks/_search"))
            )
        );
    }

    /** Hands the call's path parameters, the {@code query} parameters it reads, and its body to {@code action}. */
    static final class Handler extends BaseRestHandler {
        private static final Set<String> PATH = Set.of("model_id", "algorithm", "model_group_id", "task_id");

        private final String name;
        private final MlActions.Type action;
        private final Set<String> query;
        private final List<Route> routes;

        Handler(String name, MlActions.Type action, Set<String> query, List<Route> routes) {
            this.name = name;
            this.action = action;
            this.query = query;
            this.routes = routes;
        }

        @Override
        public String getName() {
            return name;
        }

        @Override
        public List<Route> routes() {
            return routes;
        }

        @Override
        protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
            Map<String, String> params = new LinkedHashMap<>();
            for (String p : PATH) {
                String v = request.param(p);
                if (v != null) {
                    params.put(p, p.equals("algorithm") ? v.toUpperCase(Locale.ROOT) : v);
                }
            }
            for (String p : query) {
                String v = request.param(p);
                if (v != null) {
                    params.put(p, v);
                }
            }
            byte[] body = request.hasContentOrSourceParam() ? BytesReference.toBytes(request.contentOrSourceParam().v2()) : new byte[0];
            MlActions.Request r = new MlActions.Request(params, body);
            return channel -> client.execute(action, r, new ActionListener<>() {
                @Override
                public void onResponse(MlActions.Response response) {
                    channel.sendResponse(new BytesRestResponse(RestStatus.OK, "application/json; charset=UTF-8", response.json));
                }

                @Override
                public void onFailure(Exception e) {
                    try {
                        channel.sendResponse(new BytesRestResponse(channel, e));
                    } catch (IOException inner) {
                        inner.addSuppressed(e);
                        channel.sendResponse(new BytesRestResponse(RestStatus.INTERNAL_SERVER_ERROR, inner.toString()));
                    }
                }
            });
        }
    }
}
