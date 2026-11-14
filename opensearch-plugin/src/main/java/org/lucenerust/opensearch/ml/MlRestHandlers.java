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
            new Handler("ml_update_model_action", MlActions.UPDATE_MODEL, Set.of(), List.of(new Route(PUT, BASE + "/models/{model_id}"))),
            new Handler("ml_register_model_meta_action", MlActions.REGISTER_MODEL_META, Set.of(), List.of(new Route(POST, BASE + "/models/meta"))),
            new Handler(
                "ml_upload_model_chunk_action",
                MlActions.UPLOAD_MODEL_CHUNK,
                Set.of(),
                List.of(new Route(POST, BASE + "/models/{model_id}/upload_chunk/{chunk_number}"))
            ),
            new Handler(
                "ml_stats_action",
                MlActions.STATS,
                Set.of(),
                List.of(
                    new Route(GET, BASE + "/{nodeId}/stats/"),
                    new Route(GET, BASE + "/{nodeId}/stats/{stat}"),
                    new Route(GET, BASE + "/stats/{stat}"),
                    new Route(GET, BASE + "/stats")
                )
            ),
            new Handler("ml_profile_action", MlActions.PROFILE, Set.of("view"), Map.of("scope", "all"), List.of(new Route(GET, BASE + "/profile"))),
            new Handler(
                "ml_profile_models_action",
                MlActions.PROFILE,
                Set.of("view"),
                Map.of("scope", "models"),
                List.of(new Route(GET, BASE + "/profile/models"), new Route(GET, BASE + "/profile/models/{model_id}"))
            ),
            new Handler(
                "ml_profile_tasks_action",
                MlActions.PROFILE,
                Set.of("view"),
                Map.of("scope", "tasks"),
                List.of(new Route(GET, BASE + "/profile/tasks"), new Route(GET, BASE + "/profile/tasks/{task_id}"))
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
            ),
            new Handler(
                "ml_create_connector_action",
                MlActions.CREATE_CONNECTOR,
                Set.of(),
                List.of(new Route(POST, BASE + "/connectors/_create"))
            ),
            new Handler(
                "ml_get_connector_action",
                MlActions.GET_CONNECTOR,
                Set.of(),
                List.of(new Route(GET, BASE + "/connectors/{connector_id}"))
            ),
            new Handler(
                "ml_update_connector_action",
                MlActions.UPDATE_CONNECTOR,
                Set.of(),
                List.of(new Route(PUT, BASE + "/connectors/{connector_id}"))
            ),
            new Handler(
                "ml_delete_connector_action",
                MlActions.DELETE_CONNECTOR,
                Set.of(),
                List.of(new Route(DELETE, BASE + "/connectors/{connector_id}"))
            ),
            new Handler(
                "ml_search_connector_action",
                MlActions.SEARCH_CONNECTORS,
                Set.of(),
                List.of(new Route(POST, BASE + "/connectors/_search"), new Route(GET, BASE + "/connectors/_search"))
            ),
            new Handler("ml_register_agent_action", MlActions.REGISTER_AGENT, Set.of(), List.of(new Route(POST, BASE + "/agents/_register"))),
            new Handler("ml_get_agent_action", MlActions.GET_AGENT, Set.of(), List.of(new Route(GET, BASE + "/agents/{agent_id}"))),
            new Handler("ml_update_agent_action", MlActions.UPDATE_AGENT, Set.of(), List.of(new Route(PUT, BASE + "/agents/{agent_id}"))),
            new Handler("ml_delete_agent_action", MlActions.DELETE_AGENT, Set.of(), List.of(new Route(DELETE, BASE + "/agents/{agent_id}"))),
            new Handler(
                "ml_search_agent_action",
                MlActions.SEARCH_AGENTS,
                Set.of(),
                List.of(new Route(POST, BASE + "/agents/_search"), new Route(GET, BASE + "/agents/_search"))
            ),
            new Handler("ml_list_tools_action", MlActions.LIST_TOOLS, Set.of(), List.of(new Route(GET, BASE + "/tools"))),
            new Handler("ml_get_tool_action", MlActions.GET_TOOL, Set.of(), List.of(new Route(GET, BASE + "/tools/{tool_name}"))),
            new Handler("ml_create_memory_action", MlActions.CREATE_MEMORY, Set.of(), List.of(new Route(POST, BASE + "/memory"))),
            new Handler("ml_list_memories_action", MlActions.LIST_MEMORIES, Set.of("max_results", "next_token"), List.of(new Route(GET, BASE + "/memory"))),
            new Handler("ml_get_memory_action", MlActions.GET_MEMORY, Set.of(), List.of(new Route(GET, BASE + "/memory/{memory_id}"))),
            new Handler("ml_update_memory_action", MlActions.UPDATE_MEMORY, Set.of(), List.of(new Route(PUT, BASE + "/memory/{memory_id}"))),
            new Handler("ml_delete_memory_action", MlActions.DELETE_MEMORY, Set.of(), List.of(new Route(DELETE, BASE + "/memory/{memory_id}"))),
            new Handler(
                "ml_search_memories_action",
                MlActions.SEARCH_MEMORIES,
                Set.of(),
                List.of(new Route(POST, BASE + "/memory/_search"), new Route(GET, BASE + "/memory/_search"))
            ),
            new Handler(
                "ml_search_messages_action",
                MlActions.SEARCH_MESSAGES,
                Set.of(),
                List.of(new Route(POST, BASE + "/memory/{memory_id}/_search"), new Route(GET, BASE + "/memory/{memory_id}/_search"))
            ),
            new Handler("ml_create_message_action", MlActions.CREATE_MESSAGE, Set.of(), List.of(new Route(POST, BASE + "/memory/{memory_id}/messages"))),
            new Handler(
                "ml_list_messages_action",
                MlActions.LIST_MESSAGES,
                Set.of("max_results", "next_token"),
                List.of(new Route(GET, BASE + "/memory/{memory_id}/messages"))
            ),
            new Handler("ml_get_message_action", MlActions.GET_MESSAGE, Set.of(), List.of(new Route(GET, BASE + "/memory/message/{message_id}"))),
            new Handler("ml_update_message_action", MlActions.UPDATE_MESSAGE, Set.of(), List.of(new Route(PUT, BASE + "/memory/message/{message_id}"))),
            new Handler(
                "ml_get_traces_action",
                MlActions.GET_TRACES,
                Set.of("max_results", "next_token"),
                List.of(new Route(GET, BASE + "/memory/message/{message_id}/traces"))
            ),
            new Handler("ml_execute_agent_action", MlActions.EXECUTE_AGENT, Set.of("async"), Map.of(), true, List.of(new Route(POST, BASE + "/agents/{agent_id}/_execute")))
        );
    }

    /** Hands the call's path parameters, the {@code query} parameters it reads, and its body to {@code action}. */
    static final class Handler extends BaseRestHandler {
        private static final Set<String> PATH = Set.of("model_id", "algorithm", "model_group_id", "task_id", "connector_id", "chunk_number", "nodeId", "stat", "agent_id", "tool_name", "memory_id", "message_id");

        private final String name;
        private final MlActions.Type action;
        private final Set<String> query;
        private final Map<String, String> fixed;
        private final List<Route> routes;
        private final boolean envelope;

        Handler(String name, MlActions.Type action, Set<String> query, List<Route> routes) {
            this(name, action, query, Map.of(), routes);
        }

        Handler(String name, MlActions.Type action, Set<String> query, Map<String, String> fixed, List<Route> routes) {
            this(name, action, query, fixed, false, routes);
        }

        /** {@code envelope}: failures are answered in the shape ml-commons' execute endpoints use. */
        Handler(String name, MlActions.Type action, Set<String> query, Map<String, String> fixed, boolean envelope, List<Route> routes) {
            this.envelope = envelope;
            this.name = name;
            this.action = action;
            this.query = query;
            this.fixed = fixed;
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
            params.putAll(fixed);
            byte[] body = request.hasContentOrSourceParam() ? BytesReference.toBytes(request.contentOrSourceParam().v2()) : new byte[0];
            MlActions.Request r = new MlActions.Request(params, body);
            return channel -> client.execute(action, r, new ActionListener<>() {
                @Override
                public void onResponse(MlActions.Response response) {
                    channel.sendResponse(new BytesRestResponse(RestStatus.OK, "application/json; charset=UTF-8", response.json));
                }

                @Override
                public void onFailure(Exception e) {
                    if (envelope && e instanceof org.opensearch.OpenSearchParseException == false) {
                        Throwable cause = e instanceof org.opensearch.transport.RemoteTransportException && e.getCause() != null ? e.getCause() : e;
                        RestStatus status = cause instanceof IllegalArgumentException ? RestStatus.BAD_REQUEST : org.opensearch.ExceptionsHelper.status(cause);
                        Map<String, Object> error = new LinkedHashMap<>();
                        error.put("type", cause.getClass().getSimpleName());
                        error.put("reason", status == RestStatus.BAD_REQUEST ? "Invalid Request" : "System Error");
                        error.put("details", cause.getMessage() == null ? "" : cause.getMessage());
                        Map<String, Object> body = new LinkedHashMap<>();
                        body.put("status", status.getStatus());
                        body.put("error", error);
                        channel.sendResponse(new BytesRestResponse(status, "application/json; charset=UTF-8", MlCommons.json(body).getBytes(java.nio.charset.StandardCharsets.UTF_8)));
                        return;
                    }
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
