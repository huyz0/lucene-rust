/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.rest;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.ResourceNotFoundException;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.action.delete.DeleteRequest;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.CheckedConsumer;
import org.opensearch.common.ValidationException;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestStatusToXContentListener;
import org.opensearch.rest.action.search.RestSearchAction;
import org.opensearch.transport.client.node.NodeClient;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;

import static org.lucenerust.opensearch.knn.rest.KnnRestHandlers.BASE;

/**
 * k-NN's model endpoints under {@code _plugins/_knn/models}: get ({@code RestGetModelHandler}),
 * delete ({@code RestDeleteModelHandler}), search ({@code RestSearchModelHandler}) and train
 * ({@code RestTrainModelHandler}), with the stock plugin's routes and errors.
 *
 * <p>Every model stock trains is built by the faiss engine, which this node does not run, so
 * training is refused once the request passes stock's request checks (parameters, method, training
 * index). Models in a model index restored from a stock cluster can still be read, searched and
 * deleted, with the caller's own permissions on that index.
 */
public final class KnnModelHandlers {
    private static final Logger LOGGER = LogManager.getLogger(KnnModelHandlers.class);
    /** {@code KNNConstants.MODEL_INDEX_NAME}. */
    static final String MODEL_INDEX = ".opensearch-knn-models";
    static final String MODELS = BASE + "/models";

    private KnnModelHandlers() {}

    /** Every handler. */
    public static List<RestHandler> all(Supplier<ClusterService> clusterService) {
        return List.of(new Get(), new Delete(clusterService), new Search(), new Train(clusterService));
    }

    /** The model as {@code Model.toXContent} prints it, from its document in the model index. */
    static void model(XContentBuilder b, String id, Map<String, Object> source) throws IOException {
        b.startObject();
        b.field("model_id", id);
        for (String key : List.of(
            "model_blob",
            "state",
            "timestamp",
            "description",
            "error",
            "space_type",
            "dimension",
            "engine",
            "training_node_assignment",
            "model_definition",
            "data_type",
            "mode",
            "compression_level",
            "model_version"
        )) {
            Object value = source.get(key);
            if (key.equals("model_blob") && value == null) {
                value = "";
            }
            if (value == null) {
                continue;
            }
            if (key.equals("model_definition") && value instanceof String s) {
                value = XContentHelper.convertToMap(JsonXContent.jsonXContent, s, false);
            } else if (key.equals("data_type") && value instanceof String s) {
                value = s.toLowerCase(Locale.ROOT);
            }
            b.field(key, value);
        }
        b.endObject();
    }

    /** {@code onResponse}, with any failure (its own included) sent as the error response. */
    static <T> ActionListener<T> answer(RestChannel channel, CheckedConsumer<T, Exception> onResponse) {
        return ActionListener.wrap(onResponse, e -> {
            try {
                channel.sendResponse(new BytesRestResponse(channel, e));
            } catch (IOException io) {
                io.addSuppressed(e);
                LOGGER.warn("failed to send a k-NN model error response", io);
            }
        });
    }

    /** {@code RestGetModelHandler}. */
    static final class Get extends BaseRestHandler {
        @Override
        public String getName() {
            return "knn_get_model_action";
        }

        @Override
        public List<Route> routes() {
            return List.of(new Route(RestRequest.Method.GET, MODELS + "/{model_id}"));
        }

        @Override
        protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
            String id = request.param("model_id");
            return channel -> client.get(new GetRequest(MODEL_INDEX, id), answer(channel, (GetResponse r) -> {
                if (r.isExists() == false) {
                    throw new ResourceNotFoundException(id);
                }
                XContentBuilder b = channel.newBuilder();
                model(b, id, r.getSourceAsMap());
                channel.sendResponse(new BytesRestResponse(RestStatus.OK, b));
            }));
        }
    }

    /** {@code RestDeleteModelHandler}. */
    static final class Delete extends BaseRestHandler {
        private final Supplier<ClusterService> clusterService;

        Delete(Supplier<ClusterService> clusterService) {
            this.clusterService = clusterService;
        }

        @Override
        public String getName() {
            return "knn_delete_model_action";
        }

        @Override
        public List<Route> routes() {
            return List.of(new Route(RestRequest.Method.DELETE, MODELS + "/{model_id}"));
        }

        @Override
        protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
            String id = request.param("model_id");
            if (clusterService.get().state().metadata().hasIndex(MODEL_INDEX) == false) {
                throw new ResourceNotFoundException(
                    "Cannot delete model [" + id + "]. Model index [" + MODEL_INDEX + "] does not exist"
                );
            }
            return channel -> client.get(new GetRequest(MODEL_INDEX, id), answer(channel, (GetResponse r) -> {
                if (r.isExists() == false) {
                    throw new ResourceNotFoundException("Unable to delete model [" + id + "]. Model does not exist");
                }
                if ("training".equals(r.getSourceAsMap().get("state"))) {
                    throw new OpenSearchStatusException(
                        "Cannot delete model [" + id + "]. Model is still in training",
                        RestStatus.CONFLICT
                    );
                }
                DeleteRequest delete = new DeleteRequest(MODEL_INDEX, id).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                client.delete(delete, answer(channel, d -> {
                    XContentBuilder b = channel.newBuilder();
                    b.startObject().field("model_id", id).field("result", "deleted").endObject();
                    channel.sendResponse(new BytesRestResponse(RestStatus.OK, b));
                }));
            }));
        }
    }

    /** {@code RestSearchModelHandler}: a search of the model index. */
    static final class Search extends BaseRestHandler {
        @Override
        public String getName() {
            return "knn_search_model_action";
        }

        @Override
        public List<Route> routes() {
            return List.of(new Route(RestRequest.Method.GET, MODELS + "/_search"), new Route(RestRequest.Method.POST, MODELS + "/_search"));
        }

        @Override
        protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
            SearchRequest search = new SearchRequest();
            request.withContentOrSourceParamParserOrNull(
                p -> RestSearchAction.parseSearchRequest(search, request, p, client.getNamedWriteableRegistry(), size -> search.source().size(size))
            );
            search.indices(MODEL_INDEX);
            return channel -> client.search(search, new RestStatusToXContentListener<>(channel));
        }
    }

    /** {@code RestTrainModelHandler}: the request checked as stock checks it, then refused. */
    static final class Train extends BaseRestHandler {
        private final Supplier<ClusterService> clusterService;

        Train(Supplier<ClusterService> clusterService) {
            this.clusterService = clusterService;
        }

        private static final Set<String> PARAMETERS = Set.of(
            "training_index",
            "training_field",
            "dimension",
            "max_training_vector_count",
            "search_size",
            "description",
            "method",
            "mode",
            "compression_level",
            "space_type",
            "preference"
        );

        @Override
        public String getName() {
            return "knn_train_model_action";
        }

        @Override
        public List<Route> routes() {
            return List.of(
                new Route(RestRequest.Method.POST, MODELS + "/{model_id}/_train"),
                new Route(RestRequest.Method.POST, MODELS + "/_train")
            );
        }

        @Override
        protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
            request.param("model_id");
            Map<String, Object> body;
            try (XContentParser p = request.contentOrSourceParamParser()) {
                body = p.map();
            }
            for (String key : body.keySet()) {
                if (PARAMETERS.contains(key) == false) {
                    throw new IllegalArgumentException("Unable to parse token. \"" + key + "\" is not a valid parameter.");
                }
            }
            if (body.get("method") == null && body.get("mode") == null && body.get("compression_level") == null) {
                throw new IllegalArgumentException("At least \"[method]\", \"[mode]\" or \"[compression_level]\" needs to be set");
            }
            for (String key : List.of("training_index", "training_field", "dimension")) {
                if (body.get(key) == null) {
                    throw new IllegalArgumentException("Request did not set \"" + key + ".");
                }
            }
            Map<?, ?> method = body.get("method") instanceof Map<?, ?> m ? m : Map.of();
            Object engine = method.get("engine");
            if (engine != null && "lucene".equalsIgnoreCase(engine.toString())) {
                ValidationException e = new ValidationException();
                e.addValidationError("Cannot use \"lucene\" engine from training context");
                throw e;
            }
            Object name = method.get("name");
            if (name != null && List.of("hnsw", "ivf").contains(name.toString()) == false) {
                throw new IllegalArgumentException("Invalid method name: " + name);
            }
            ActionRequestValidationException invalid = new ActionRequestValidationException();
            if ("hnsw".equals(name)) {
                invalid.addValidationError("Method does not require training.");
            }
            String index = body.get("training_index").toString();
            if (clusterService.get().state().metadata().hasIndex(index) == false) {
                invalid.addValidationError("Index \"" + index + "\" does not exist.");
            }
            if (invalid.validationErrors().isEmpty() == false) {
                throw invalid;
            }
            throw new IllegalArgumentException(
                "Engine [faiss] is not supported: models are trained by the faiss engine, which this node does not run"
            );
        }
    }
}
