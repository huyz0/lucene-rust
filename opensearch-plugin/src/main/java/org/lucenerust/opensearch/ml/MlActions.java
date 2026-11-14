/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.action.ActionType;
import org.opensearch.action.DocWriteResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.action.ActionResponse;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.plugins.ActionPlugin.ActionHandler;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * ml-commons' transport actions, under the stock action names so that the security plugin's ml
 * roles ({@code ml_full_access}, {@code ml_readonly_access}) authorize them as they do on stock.
 * Every action takes the REST call's path parameters and JSON body and answers with JSON.
 */
public final class MlActions {
    private MlActions() {}

    private static final String ML = "cluster:admin/opensearch/ml/";

    public static final Type REGISTER_MODEL = new Type(ML + "register_model");
    public static final Type DEPLOY_MODEL = new Type(ML + "deploy_model");
    public static final Type UNDEPLOY_MODEL = new Type(ML + "undeploy_models");
    public static final Type PREDICT = new Type(ML + "predict");
    public static final Type GET_MODEL = new Type(ML + "models/get");
    public static final Type DELETE_MODEL = new Type(ML + "models/delete");
    public static final Type SEARCH_MODELS = new Type(ML + "models/search");
    public static final Type UPDATE_MODEL = new Type(ML + "models/update");
    public static final Type REGISTER_MODEL_META = new Type(ML + "register_model_meta");
    public static final Type UPLOAD_MODEL_CHUNK = new Type(ML + "upload_model_chunk");
    public static final Type STATS = new Type(ML + "stats/nodes");
    public static final Type PROFILE = new Type(ML + "profile/nodes");
    public static final Type REGISTER_MODEL_GROUP = new Type(ML + "register_model_group");
    public static final Type UPDATE_MODEL_GROUP = new Type(ML + "update_model_group");
    public static final Type GET_MODEL_GROUP = new Type(ML + "model_groups/get");
    public static final Type DELETE_MODEL_GROUP = new Type(ML + "model_groups/delete");
    public static final Type SEARCH_MODEL_GROUPS = new Type(ML + "model_groups/search");
    public static final Type GET_TASK = new Type(ML + "tasks/get");
    public static final Type DELETE_TASK = new Type(ML + "tasks/delete");
    public static final Type SEARCH_TASKS = new Type(ML + "tasks/search");
    public static final Type CREATE_CONNECTOR = new Type(ML + "create_connector");
    public static final Type GET_CONNECTOR = new Type(ML + "connectors/get");
    public static final Type UPDATE_CONNECTOR = new Type(ML + "update_connector");
    public static final Type DELETE_CONNECTOR = new Type(ML + "connectors/delete");
    public static final Type LIST_MCP_TOOLS = new Type(ML + "mcp/connector/list_tools");
    public static final Type SEARCH_CONNECTORS = new Type(ML + "connectors/search");
    public static final Type REGISTER_AGENT = new Type(ML + "agents/register");
    public static final Type GET_AGENT = new Type(ML + "agents/get");
    public static final Type UPDATE_AGENT = new Type(ML + "agents/update");
    public static final Type DELETE_AGENT = new Type(ML + "agents/delete");
    public static final Type SEARCH_AGENTS = new Type(ML + "agents/search");
    public static final Type CREATE_CONTAINER = new Type(ML + "memory_containers/create");
    public static final Type GET_CONTAINER = new Type(ML + "memory_containers/get");
    public static final Type UPDATE_CONTAINER = new Type(ML + "memory_containers/update");
    public static final Type DELETE_CONTAINER = new Type(ML + "memory_containers/delete");
    public static final Type SEARCH_CONTAINERS = new Type(ML + "memory_containers/search");
    public static final Type ADD_AM_MEMORY = new Type(ML + "memory_containers/memories/add");
    public static final Type GET_AM_MEMORY = new Type(ML + "memory_containers/memories/get");
    public static final Type UPDATE_AM_MEMORY = new Type(ML + "memory_containers/memories/update");
    public static final Type DELETE_AM_MEMORY = new Type(ML + "memory_containers/memories/delete");
    public static final Type SEARCH_AM_MEMORIES = new Type(ML + "memory_containers/memories/search");
    public static final Type LIST_TOOLS = new Type(ML + "tools/list");
    public static final Type GET_TOOL = new Type(ML + "tools/get");
    public static final Type EXECUTE_AGENT = new Type(ML + "agents/execute");
    public static final Type CREATE_MEMORY = new Type(ML + "memory/create");
    public static final Type LIST_MEMORIES = new Type(ML + "memory/list");
    public static final Type GET_MEMORY = new Type(ML + "memory/get");
    public static final Type UPDATE_MEMORY = new Type(ML + "memory/update");
    public static final Type DELETE_MEMORY = new Type(ML + "memory/delete");
    public static final Type SEARCH_MEMORIES = new Type(ML + "memory/search");
    public static final Type SEARCH_MESSAGES = new Type(ML + "memory/messages/search");
    public static final Type CREATE_MESSAGE = new Type(ML + "memory/message/create");
    public static final Type LIST_MESSAGES = new Type(ML + "memory/message/list");
    public static final Type GET_MESSAGE = new Type(ML + "memory/message/get");
    public static final Type UPDATE_MESSAGE = new Type(ML + "memory/message/update");
    public static final Type GET_TRACES = new Type(ML + "memory/message/traces");

    /** The action handlers to register, with the node-to-node actions. */
    public static List<ActionHandler<?, ?>> handlers() {
        return List.of(
            new ActionHandler<>(REGISTER_MODEL, RegisterModel.class),
            new ActionHandler<>(DEPLOY_MODEL, DeployModel.class),
            new ActionHandler<>(UNDEPLOY_MODEL, UndeployModel.class),
            new ActionHandler<>(PREDICT, Predict.class),
            new ActionHandler<>(GET_MODEL, GetModel.class),
            new ActionHandler<>(DELETE_MODEL, DeleteModel.class),
            new ActionHandler<>(SEARCH_MODELS, SearchModels.class),
            new ActionHandler<>(UPDATE_MODEL, UpdateModel.class),
            new ActionHandler<>(REGISTER_MODEL_META, RegisterModelMeta.class),
            new ActionHandler<>(UPLOAD_MODEL_CHUNK, UploadModelChunk.class),
            new ActionHandler<>(STATS, Stats.class),
            new ActionHandler<>(PROFILE, Profile.class),
            new ActionHandler<>(REGISTER_MODEL_GROUP, RegisterModelGroup.class),
            new ActionHandler<>(UPDATE_MODEL_GROUP, UpdateModelGroup.class),
            new ActionHandler<>(GET_MODEL_GROUP, GetModelGroup.class),
            new ActionHandler<>(DELETE_MODEL_GROUP, DeleteModelGroup.class),
            new ActionHandler<>(SEARCH_MODEL_GROUPS, SearchModelGroups.class),
            new ActionHandler<>(GET_TASK, GetTask.class),
            new ActionHandler<>(DELETE_TASK, DeleteTask.class),
            new ActionHandler<>(SEARCH_TASKS, SearchTasks.class),
            new ActionHandler<>(CREATE_CONNECTOR, CreateConnector.class),
            new ActionHandler<>(GET_CONNECTOR, GetConnector.class),
            new ActionHandler<>(UPDATE_CONNECTOR, UpdateConnector.class),
            new ActionHandler<>(DELETE_CONNECTOR, DeleteConnector.class),
            new ActionHandler<>(LIST_MCP_TOOLS, ListMcpTools.class),
            new ActionHandler<>(SEARCH_CONNECTORS, SearchConnectors.class),
            new ActionHandler<>(REGISTER_AGENT, RegisterAgent.class),
            new ActionHandler<>(GET_AGENT, GetAgent.class),
            new ActionHandler<>(UPDATE_AGENT, UpdateAgent.class),
            new ActionHandler<>(DELETE_AGENT, DeleteAgent.class),
            new ActionHandler<>(SEARCH_AGENTS, SearchAgents.class),
            new ActionHandler<>(CREATE_CONTAINER, CreateContainer.class),
            new ActionHandler<>(GET_CONTAINER, GetContainer.class),
            new ActionHandler<>(UPDATE_CONTAINER, UpdateContainer.class),
            new ActionHandler<>(DELETE_CONTAINER, DeleteContainer.class),
            new ActionHandler<>(SEARCH_CONTAINERS, SearchContainers.class),
            new ActionHandler<>(ADD_AM_MEMORY, AddAmMemory.class),
            new ActionHandler<>(GET_AM_MEMORY, GetAmMemory.class),
            new ActionHandler<>(UPDATE_AM_MEMORY, UpdateAmMemory.class),
            new ActionHandler<>(DELETE_AM_MEMORY, DeleteAmMemory.class),
            new ActionHandler<>(SEARCH_AM_MEMORIES, SearchAmMemories.class),
            new ActionHandler<>(LIST_TOOLS, ListTools.class),
            new ActionHandler<>(GET_TOOL, GetTool.class),
            new ActionHandler<>(EXECUTE_AGENT, ExecuteAgent.class),
            new ActionHandler<>(CREATE_MEMORY, CreateMemory.class),
            new ActionHandler<>(LIST_MEMORIES, ListMemories.class),
            new ActionHandler<>(GET_MEMORY, GetMemory.class),
            new ActionHandler<>(UPDATE_MEMORY, UpdateMemory.class),
            new ActionHandler<>(DELETE_MEMORY, DeleteMemory.class),
            new ActionHandler<>(SEARCH_MEMORIES, SearchMemories.class),
            new ActionHandler<>(SEARCH_MESSAGES, SearchMessages.class),
            new ActionHandler<>(CREATE_MESSAGE, CreateMessage.class),
            new ActionHandler<>(LIST_MESSAGES, ListMessages.class),
            new ActionHandler<>(GET_MESSAGE, GetMessage.class),
            new ActionHandler<>(UPDATE_MESSAGE, UpdateMessage.class),
            new ActionHandler<>(GET_TRACES, GetTraces.class),
            new ActionHandler<>(MlNodes.Action.INSTANCE, MlNodes.TransportAction.class),
            new ActionHandler<>(MlNodes.PredictAction.INSTANCE, MlNodes.PredictTransportAction.class),
            new ActionHandler<>(MlNodes.RemotePredictAction.INSTANCE, MlNodes.RemotePredictTransportAction.class)
        );
    }

    public static final class Type extends ActionType<Response> {
        Type(String name) {
            super(name, Response::new);
        }
    }

    /** A REST call's path and query parameters, and its body (empty when it has none). */
    public static final class Request extends ActionRequest {
        final Map<String, String> params;
        final byte[] body;

        public Request(Map<String, String> params, byte[] body) {
            this.params = params;
            this.body = body;
        }

        public Request(StreamInput in) throws IOException {
            super(in);
            params = in.readMap(StreamInput::readString, StreamInput::readString);
            body = in.readByteArray();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            super.writeTo(out);
            out.writeMap(params, StreamOutput::writeString, StreamOutput::writeString);
            out.writeByteArray(body);
        }

        @Override
        public ActionRequestValidationException validate() {
            return null;
        }

        String param(String name) {
            return params.get(name);
        }

        Map<String, Object> json() {
            return body.length == 0 ? new LinkedHashMap<>() : new LinkedHashMap<>(MlCommons.parse(body));
        }
    }

    /** The JSON answer. */
    public static final class Response extends ActionResponse implements ToXContentObject {
        public final byte[] json;

        Response(byte[] json) {
            this.json = json;
        }

        public Response(StreamInput in) throws IOException {
            super(in);
            json = in.readByteArray();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeByteArray(json);
        }

        @Override
        public XContentBuilder toXContent(XContentBuilder builder, ToXContent.Params params) throws IOException {
            try (
                XContentParser p = JsonXContent.jsonXContent.createParser(NamedXContentRegistry.EMPTY, DeprecationHandler.IGNORE_DEPRECATIONS, json)
            ) {
                return builder.copyCurrentStructure(p);
            }
        }

        static Response of(Object value) {
            return new Response(MlCommons.json(value).getBytes(StandardCharsets.UTF_8));
        }

        static Response of(ToXContent value) {
            try (XContentBuilder b = JsonXContent.contentBuilder()) {
                if (value.isFragment()) {
                    b.startObject();
                    value.toXContent(b, ToXContent.EMPTY_PARAMS);
                    b.endObject();
                } else {
                    value.toXContent(b, ToXContent.EMPTY_PARAMS);
                }
                return new Response(b.toString().getBytes(StandardCharsets.UTF_8));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    /** Runs one action on this node with the ml-commons service. */
    abstract static class Base extends HandledTransportAction<Request, Response> {
        Base(Type type, TransportService transportService, ActionFilters filters) {
            super(type.name(), transportService, filters, Request::new);
        }

        @Override
        protected final void doExecute(Task task, Request request, ActionListener<Response> listener) {
            try {
                run(MlCommons.get(), request, listener);
            } catch (Exception e) {
                listener.onFailure(e);
            }
        }

        abstract void run(MlCommons ml, Request request, ActionListener<Response> listener) throws Exception;
    }

    private static <T> ActionListener<T> map(ActionListener<Response> listener) {
        return ActionListener.wrap(r -> listener.onResponse(Response.of(r)), listener::onFailure);
    }

    private static <T extends DocWriteResponse> ActionListener<T> written(ActionListener<Response> listener) {
        return ActionListener.wrap(r -> listener.onResponse(Response.of((ToXContent) r)), listener::onFailure);
    }

    /** A search action: the request body is a search source, as on stock's {@code _search} endpoints. */
    abstract static class Search extends Base {
        private final NamedXContentRegistry registry;
        private final String index;

        Search(Type type, String index, TransportService transportService, ActionFilters filters, NamedXContentRegistry registry) {
            super(type, transportService, filters);
            this.registry = registry;
            this.index = index;
        }

        @Override
        void run(MlCommons ml, Request request, ActionListener<Response> listener) throws IOException {
            SearchSourceBuilder source = parseSource(registry, request.body);
            if (index.equals(MlIndices.MODEL)) {
                // Chunks hold the model bytes; stock leaves them out of every model search.
                source.fetchSource(source.fetchSource() == null ? null : source.fetchSource().includes(), exclude(source, "model_content", "connector.credential"));
            } else if (index.equals(MlIndices.CONNECTOR)) {
                source.fetchSource(source.fetchSource() == null ? null : source.fetchSource().includes(), exclude(source, "credential"));
            }
            ml.search(index, source, ActionListener.wrap(r -> listener.onResponse(Response.of((ToXContent) r)), listener::onFailure));
        }

        static SearchSourceBuilder parseSource(NamedXContentRegistry registry, byte[] body) throws IOException {
            SearchSourceBuilder source = new SearchSourceBuilder();
            if (body.length > 0) {
                try (XContentParser p = JsonXContent.jsonXContent.createParser(registry, DeprecationHandler.THROW_UNSUPPORTED_OPERATION, body)) {
                    source = SearchSourceBuilder.fromXContent(p);
                }
            }
            return source;
        }

        private static String[] exclude(SearchSourceBuilder source, String... fields) {
            String[] given = source.fetchSource() == null ? new String[0] : source.fetchSource().excludes();
            String[] all = Arrays.copyOf(given, given.length + fields.length);
            System.arraycopy(fields, 0, all, given.length, fields.length);
            return all;
        }
    }

    public static final class RegisterModel extends Base {
        @Inject
        public RegisterModel(TransportService t, ActionFilters f) {
            super(REGISTER_MODEL, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.registerModel(r.json(), Boolean.parseBoolean(r.param("deploy")), map(l));
        }
    }

    public static final class UpdateModel extends Base {
        @Inject
        public UpdateModel(TransportService t, ActionFilters f) {
            super(UPDATE_MODEL, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.updateModel(r.param("model_id"), r.json(), written(l));
        }
    }

    public static final class RegisterModelMeta extends Base {
        @Inject
        public RegisterModelMeta(TransportService t, ActionFilters f) {
            super(REGISTER_MODEL_META, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.registerModelMeta(r.json(), map(l));
        }
    }

    public static final class UploadModelChunk extends Base {
        @Inject
        public UploadModelChunk(TransportService t, ActionFilters f) {
            super(UPLOAD_MODEL_CHUNK, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.uploadModelChunk(r.param("model_id"), r.param("chunk_number"), r.body, map(l));
        }
    }

    public static final class Stats extends Base {
        @Inject
        public Stats(TransportService t, ActionFilters f) {
            super(STATS, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.stats(r.param("nodeId"), r.param("stat"), map(l));
        }
    }

    public static final class Profile extends Base {
        @Inject
        public Profile(TransportService t, ActionFilters f) {
            super(PROFILE, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.profile(r.param("scope"), r.param("view"), r.param("model_id"), r.param("task_id"), map(l));
        }
    }

    public static final class DeployModel extends Base {
        @Inject
        public DeployModel(TransportService t, ActionFilters f) {
            super(DEPLOY_MODEL, t, f);
        }

        @Override
        @SuppressWarnings("unchecked")
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            Object nodes = r.json().get("node_ids");
            ml.deployModel(r.param("model_id"), nodes instanceof List<?> n ? (List<String>) n : null, map(l));
        }
    }

    public static final class UndeployModel extends Base {
        @Inject
        public UndeployModel(TransportService t, ActionFilters f) {
            super(UNDEPLOY_MODEL, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.undeployModel(r.param("model_id"), map(l));
        }
    }

    public static final class Predict extends Base {
        @Inject
        public Predict(TransportService t, ActionFilters f) {
            super(PREDICT, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.predict(r.param("algorithm"), r.param("model_id"), r.json(), map(l));
        }
    }

    public static final class GetModel extends Base {
        @Inject
        public GetModel(TransportService t, ActionFilters f) {
            super(GET_MODEL, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.getModel(r.param("model_id"), ActionListener.wrap(m -> l.onResponse(Response.of(withoutCredential(m))), l::onFailure));
        }
    }

    /** A model as GET shows it: an inline connector's credential is left out, as stock does. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> withoutCredential(Map<String, Object> model) {
        if (model.get("connector") instanceof Map<?, ?> c && c.containsKey("credential")) {
            Map<String, Object> connector = new LinkedHashMap<>((Map<String, Object>) c);
            connector.remove("credential");
            Map<String, Object> out = new LinkedHashMap<>(model);
            out.put("connector", connector);
            return out;
        }
        return model;
    }

    public static final class DeleteModel extends Base {
        @Inject
        public DeleteModel(TransportService t, ActionFilters f) {
            super(DELETE_MODEL, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.deleteModel(r.param("model_id"), written(l));
        }
    }

    public static final class SearchModels extends Search {
        @Inject
        public SearchModels(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(SEARCH_MODELS, MlIndices.MODEL, t, f, x);
        }
    }

    public static final class RegisterModelGroup extends Base {
        @Inject
        public RegisterModelGroup(TransportService t, ActionFilters f) {
            super(REGISTER_MODEL_GROUP, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.registerModelGroup(r.json(), ActionListener.wrap(id -> {
                Map<String, Object> response = new LinkedHashMap<>();
                response.put("model_group_id", id);
                response.put("status", "CREATED");
                l.onResponse(Response.of(response));
            }, l::onFailure));
        }
    }

    public static final class UpdateModelGroup extends Base {
        @Inject
        public UpdateModelGroup(TransportService t, ActionFilters f) {
            super(UPDATE_MODEL_GROUP, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.updateModelGroup(
                r.param("model_group_id"),
                r.json(),
                ActionListener.wrap(u -> l.onResponse(Response.of(Map.of("status", "Updated"))), l::onFailure)
            );
        }
    }

    public static final class GetModelGroup extends Base {
        @Inject
        public GetModelGroup(TransportService t, ActionFilters f) {
            super(GET_MODEL_GROUP, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.getModelGroup(r.param("model_group_id"), map(l));
        }
    }

    public static final class DeleteModelGroup extends Base {
        @Inject
        public DeleteModelGroup(TransportService t, ActionFilters f) {
            super(DELETE_MODEL_GROUP, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.deleteModelGroup(r.param("model_group_id"), written(l));
        }
    }

    public static final class SearchModelGroups extends Search {
        @Inject
        public SearchModelGroups(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(SEARCH_MODEL_GROUPS, MlIndices.MODEL_GROUP, t, f, x);
        }
    }

    public static final class GetTask extends Base {
        @Inject
        public GetTask(TransportService t, ActionFilters f) {
            super(GET_TASK, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.getTask(r.param("task_id"), map(l));
        }
    }

    public static final class DeleteTask extends Base {
        @Inject
        public DeleteTask(TransportService t, ActionFilters f) {
            super(DELETE_TASK, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.deleteTask(r.param("task_id"), written(l));
        }
    }

    public static final class SearchTasks extends Search {
        @Inject
        public SearchTasks(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(SEARCH_TASKS, MlIndices.TASK, t, f, x);
        }
    }

    public static final class CreateConnector extends Base {
        @Inject
        public CreateConnector(TransportService t, ActionFilters f) {
            super(CREATE_CONNECTOR, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.createConnector(r.json(), map(l));
        }
    }

    public static final class GetConnector extends Base {
        @Inject
        public GetConnector(TransportService t, ActionFilters f) {
            super(GET_CONNECTOR, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.getConnector(r.param("connector_id"), map(l));
        }
    }

    public static final class ListMcpTools extends Base {
        @Inject
        public ListMcpTools(TransportService t, ActionFilters f) {
            super(LIST_MCP_TOOLS, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            MlMcp.listTools(ml, r.param("connector_id"), map(l));
        }
    }

    public static final class UpdateConnector extends Base {
        @Inject
        public UpdateConnector(TransportService t, ActionFilters f) {
            super(UPDATE_CONNECTOR, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.updateConnector(r.param("connector_id"), r.json(), written(l));
        }
    }

    public static final class DeleteConnector extends Base {
        @Inject
        public DeleteConnector(TransportService t, ActionFilters f) {
            super(DELETE_CONNECTOR, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.deleteConnector(r.param("connector_id"), written(l));
        }
    }

    public static final class SearchConnectors extends Search {
        @Inject
        public SearchConnectors(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(SEARCH_CONNECTORS, MlIndices.CONNECTOR, t, f, x);
        }
    }

    public static final class RegisterAgent extends Base {
        @Inject
        public RegisterAgent(TransportService t, ActionFilters f) {
            super(REGISTER_AGENT, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.agents().register(r.json(), new String(r.body, StandardCharsets.UTF_8), ActionListener.wrap(id -> l.onResponse(Response.of(Map.of("agent_id", id))), l::onFailure));
        }
    }

    public static final class GetAgent extends Base {
        @Inject
        public GetAgent(TransportService t, ActionFilters f) {
            super(GET_AGENT, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.agents().get(r.param("agent_id"), map(l));
        }
    }

    public static final class UpdateAgent extends Base {
        @Inject
        public UpdateAgent(TransportService t, ActionFilters f) {
            super(UPDATE_AGENT, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.agents().update(r.param("agent_id"), r.json(), written(l));
        }
    }

    public static final class DeleteAgent extends Base {
        @Inject
        public DeleteAgent(TransportService t, ActionFilters f) {
            super(DELETE_AGENT, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.agents().delete(r.param("agent_id"), written(l));
        }
    }

    public static final class SearchAgents extends Search {
        @Inject
        public SearchAgents(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(SEARCH_AGENTS, MlIndices.AGENT, t, f, x);
        }
    }

    public static final class CreateContainer extends Base {
        @Inject
        public CreateContainer(TransportService t, ActionFilters f) {
            super(CREATE_CONTAINER, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().create(r.json(), map(l));
        }
    }

    public static final class GetContainer extends Base {
        @Inject
        public GetContainer(TransportService t, ActionFilters f) {
            super(GET_CONTAINER, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().get(r.param("memory_container_id"), map(l));
        }
    }

    public static final class UpdateContainer extends Base {
        @Inject
        public UpdateContainer(TransportService t, ActionFilters f) {
            super(UPDATE_CONTAINER, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().update(r.param("memory_container_id"), r.json(), written(l));
        }
    }

    public static final class DeleteContainer extends Base {
        @Inject
        public DeleteContainer(TransportService t, ActionFilters f) {
            super(DELETE_CONTAINER, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().delete(r.param("memory_container_id"), written(l));
        }
    }

    public static final class SearchContainers extends Base {
        private final NamedXContentRegistry registry;

        @Inject
        public SearchContainers(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(SEARCH_CONTAINERS, t, f);
            this.registry = x;
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().search(r.body, Search.parseSource(registry, r.body), ActionListener.wrap(s -> l.onResponse(Response.of((ToXContent) s)), l::onFailure));
        }
    }

    public static final class AddAmMemory extends Base {
        @Inject
        public AddAmMemory(TransportService t, ActionFilters f) {
            super(ADD_AM_MEMORY, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().add(r.param("memory_container_id"), r.body, map(l));
        }
    }

    public static final class GetAmMemory extends Base {
        @Inject
        public GetAmMemory(TransportService t, ActionFilters f) {
            super(GET_AM_MEMORY, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().getMemory(r.param("memory_container_id"), r.param("memory_type"), r.param("memory_id"), map(l));
        }
    }

    public static final class UpdateAmMemory extends Base {
        @Inject
        public UpdateAmMemory(TransportService t, ActionFilters f) {
            super(UPDATE_AM_MEMORY, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().updateMemory(r.param("memory_container_id"), r.param("memory_type"), r.param("memory_id"), r.json(), written(l));
        }
    }

    public static final class DeleteAmMemory extends Base {
        @Inject
        public DeleteAmMemory(TransportService t, ActionFilters f) {
            super(DELETE_AM_MEMORY, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().deleteMemory(r.param("memory_container_id"), r.param("memory_type"), r.param("memory_id"), written(l));
        }
    }

    public static final class SearchAmMemories extends Base {
        private final NamedXContentRegistry registry;

        @Inject
        public SearchAmMemories(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(SEARCH_AM_MEMORIES, t, f);
            this.registry = x;
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws Exception {
            ml.containers().searchMemories(r.param("memory_container_id"), r.param("memory_type"), r.body, Search.parseSource(registry, r.body), ActionListener.wrap(s -> l.onResponse(Response.of((ToXContent) s)), l::onFailure));
        }
    }

    public static final class ListTools extends Base {
        @Inject
        public ListTools(TransportService t, ActionFilters f) {
            super(LIST_TOOLS, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            l.onResponse(Response.of(MlAgents.tools()));
        }
    }

    public static final class GetTool extends Base {
        @Inject
        public GetTool(TransportService t, ActionFilters f) {
            super(GET_TOOL, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            l.onResponse(Response.of(MlAgents.tool(r.param("tool_name"))));
        }
    }

    public static final class ExecuteAgent extends Base {
        private final NamedXContentRegistry registry;

        @Inject
        public ExecuteAgent(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(EXECUTE_AGENT, t, f);
            this.registry = x;
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            if (r.body.length == 0) {
                throw new org.opensearch.OpenSearchParseException("request body is required");
            }
            ml.agents().registry(registry);
            if (Boolean.parseBoolean(r.param("async"))) {
                ml.agents().executeAsync(r.param("agent_id"), r.json(), map(l));
            } else {
                ml.agents().execute(r.param("agent_id"), r.json(), map(l));
            }
        }
    }

    public static final class CreateMemory extends Base {
        @Inject
        public CreateMemory(TransportService t, ActionFilters f) {
            super(CREATE_MEMORY, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().create(r.json(), ActionListener.wrap(id -> l.onResponse(Response.of(Map.of("memory_id", id))), l::onFailure));
        }
    }

    public static final class ListMemories extends Base {
        @Inject
        public ListMemories(TransportService t, ActionFilters f) {
            super(LIST_MEMORIES, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().list(r.param("max_results"), r.param("next_token"), map(l));
        }
    }

    public static final class GetMemory extends Base {
        @Inject
        public GetMemory(TransportService t, ActionFilters f) {
            super(GET_MEMORY, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().get(r.param("memory_id"), map(l));
        }
    }

    public static final class UpdateMemory extends Base {
        @Inject
        public UpdateMemory(TransportService t, ActionFilters f) {
            super(UPDATE_MEMORY, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().update(r.param("memory_id"), r.body.length == 0 ? null : r.json(), written(l));
        }
    }

    public static final class DeleteMemory extends Base {
        @Inject
        public DeleteMemory(TransportService t, ActionFilters f) {
            super(DELETE_MEMORY, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().delete(r.param("memory_id"), map(l));
        }
    }

    public static final class CreateMessage extends Base {
        @Inject
        public CreateMessage(TransportService t, ActionFilters f) {
            super(CREATE_MESSAGE, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().addMessage(r.param("memory_id"), r.body.length == 0 ? null : r.json(), ActionListener.wrap(id -> l.onResponse(Response.of(Map.of("message_id", id))), l::onFailure));
        }
    }

    public static final class ListMessages extends Base {
        @Inject
        public ListMessages(TransportService t, ActionFilters f) {
            super(LIST_MESSAGES, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().messages(r.param("memory_id"), r.param("max_results"), r.param("next_token"), map(l));
        }
    }

    public static final class GetMessage extends Base {
        @Inject
        public GetMessage(TransportService t, ActionFilters f) {
            super(GET_MESSAGE, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().message(r.param("message_id"), map(l));
        }
    }

    public static final class UpdateMessage extends Base {
        @Inject
        public UpdateMessage(TransportService t, ActionFilters f) {
            super(UPDATE_MESSAGE, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().updateMessage(r.param("message_id"), r.body.length == 0 ? null : r.json(), written(l));
        }
    }

    public static final class GetTraces extends Base {
        @Inject
        public GetTraces(TransportService t, ActionFilters f) {
            super(GET_TRACES, t, f);
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) {
            ml.memory().traces(r.param("message_id"), r.param("max_results"), r.param("next_token"), map(l));
        }
    }

    public static final class SearchMemories extends Search {
        @Inject
        public SearchMemories(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(SEARCH_MEMORIES, MlIndices.MEMORY_META, t, f, x);
        }

        @Override
        void run(MlCommons ml, Request request, ActionListener<Response> listener) throws IOException {
            if (request.body.length == 0) {
                throw new org.opensearch.OpenSearchParseException("request body or source parameter is required");
            }
            super.run(ml, request, listener);
        }
    }

    public static final class SearchMessages extends Base {
        private final NamedXContentRegistry registry;

        @Inject
        public SearchMessages(TransportService t, ActionFilters f, NamedXContentRegistry x) {
            super(SEARCH_MESSAGES, t, f);
            this.registry = x;
        }

        @Override
        void run(MlCommons ml, Request r, ActionListener<Response> l) throws IOException {
            if (r.body.length == 0) {
                throw new org.opensearch.OpenSearchParseException("request body or source parameter is required");
            }
            ml.memory().searchMessages(r.param("memory_id"), Search.parseSource(registry, r.body), ActionListener.wrap(res -> l.onResponse(Response.of((ToXContent) res)), l::onFailure));
        }
    }
}
