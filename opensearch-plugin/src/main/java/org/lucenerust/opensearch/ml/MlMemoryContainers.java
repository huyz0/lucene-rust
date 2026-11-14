package org.lucenerust.opensearch.ml;

import org.opensearch.OpenSearchException;
import org.opensearch.OpenSearchParseException;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.ResourceAlreadyExistsException;
import org.opensearch.action.admin.indices.create.CreateIndexRequest;
import org.opensearch.action.admin.indices.create.CreateIndexResponse;
import org.opensearch.action.DocWriteResponse;
import org.opensearch.action.delete.DeleteRequest;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.Strings;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.core.xcontent.XContentParserUtils;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.search.builder.SearchSourceBuilder;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * ml-commons' agentic memory: memory containers, and the working memories and sessions they hold
 * ({@code _plugins/_ml/memory_containers}). A container names the indices its memories live in;
 * adding a memory stores it as given. Containers with a language model or an embedding model, and
 * the long-term and history memories those make, are not supported.
 */
final class MlMemoryContainers {
    static final String CONTAINER = ".plugins-ml-am-memory-container";
    private static final List<String> TYPES = List.of("sessions", "working", "long-term", "history");
    private static final String NOT_FOUND = "Memory container not found";
    private static final Set<String> INVALID_PREFIX = Set.of(" ", "\"", "*", "\\", "<", "|", ",", ">", "/", "?");
    private static final String PREFIX_MESSAGE = "must not contain the following characters [ , \", *, \\, <, |, ,, >, /, ?]";

    private final MlCommons ml;

    MlMemoryContainers(MlCommons ml) {
        this.ml = ml;
    }

    private void enabled() {
        if (ml.setting(MlSettings.ML_COMMONS_AGENTIC_MEMORY_ENABLED) == false) {
            throw new OpenSearchStatusException(MlSettings.ML_COMMONS_AGENTIC_MEMORY_DISABLED_MESSAGE, RestStatus.FORBIDDEN);
        }
    }

    private static String resource(String file) {
        try (InputStream in = MlMemoryContainers.class.getResourceAsStream("mappings/" + file)) {
            if (in == null) {
                throw new IllegalStateException("no mapping resource " + file);
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** The name of the index holding {@code type} memories of a container with {@code configuration}. */
    private static String index(Map<String, Object> configuration, String type) {
        String prefix = configuration.get("index_prefix") == null ? "default" : String.valueOf(configuration.get("index_prefix"));
        boolean system = Boolean.FALSE.equals(configuration.get("use_system_index")) == false;
        return (system ? ".plugins-ml-am-" : "") + (system ? prefix + "-memory-" : prefix + "-memory-") + type;
    }

    @SuppressWarnings("deprecation")
    private void ensure(String index, String mapping, String expand, ActionListener<Void> then) {
        if (ml.clusterState().metadata().hasIndex(index)) {
            then.onResponse(null);
            return;
        }
        CreateIndexRequest request = new CreateIndexRequest(index).mapping(resource(mapping), XContentType.JSON)
            .settings(Settings.builder().put("index.number_of_shards", 1).put("index.auto_expand_replicas", expand).build());
        MlIndices.system(ml.client(), l -> ml.client().admin().indices().create(request, l), ActionListener.<CreateIndexResponse>wrap(r -> then.onResponse(null), e -> {
            if (e instanceof ResourceAlreadyExistsException) {
                then.onResponse(null);
            } else {
                then.onFailure(e);
            }
        }));
    }

    // ---- Containers ---------------------------------------------------------------------------

    /** {@code POST memory_containers/_create}. */
    @SuppressWarnings("unchecked")
    void create(Map<String, Object> body, ActionListener<Map<String, Object>> listener) {
        enabled();
        if (body.get("name") == null) {
            throw new IllegalArgumentException("name is null");
        }
        Map<String, Object> given = body.get("configuration") instanceof Map<?, ?> c ? (Map<String, Object>) c : Map.of();
        String prefix = given.get("index_prefix") == null || String.valueOf(given.get("index_prefix")).isBlank() ? null : String.valueOf(given.get("index_prefix"));
        if (prefix != null) {
            for (String bad : INVALID_PREFIX) {
                if (prefix.contains(bad)) {
                    throw new OpenSearchParseException("missing or invalid index prefix [" + prefix + "] " + PREFIX_MESSAGE);
                }
            }
        }
        strategies(given, null, false);
        Object maxInfer = given.get("max_infer_size");
        if (maxInfer instanceof Number n && n.intValue() > 10) {
            throw new IllegalArgumentException("Maximum infer size cannot exceed 10");
        }
        String embeddingType = given.get("embedding_model_type") == null ? null : String.valueOf(given.get("embedding_model_type"));
        if (embeddingType != null && given.get("embedding_dimension") == null && "TEXT_EMBEDDING".equals(embeddingType)) {
            throw new IllegalArgumentException("Dimension is required for TEXT_EMBEDDING");
        }
        models(given, ActionListener.wrap(v -> {
            Map<String, Object> configuration = new LinkedHashMap<>();
            configuration.put("index_prefix", prefix == null ? "default" : prefix);
            for (String f : List.of("llm_id", "embedding_model_id", "embedding_model_type", "embedding_dimension", "strategies", "index_settings", "retention_policy", "parameters")) {
                if (given.get(f) != null) {
                    configuration.put(f, given.get(f));
                }
            }
            if (maxInfer instanceof Number n && n.intValue() > 0) {
                configuration.put("max_infer_size", n.intValue());
            }
            configuration.put("disable_history", Boolean.TRUE.equals(given.get("disable_history")));
            configuration.put("disable_session", Boolean.TRUE.equals(given.get("disable_session")));
            configuration.put("use_system_index", Boolean.FALSE.equals(given.get("use_system_index")) == false);
            if (configuration.containsKey("llm_id") || configuration.containsKey("embedding_model_id")) {
                throw new OpenSearchStatusException(
                    "Memory containers with a language model or an embedding model are not supported",
                    RestStatus.NOT_IMPLEMENTED
                );
            }
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put("name", String.valueOf(body.get("name")));
            if (body.get("description") != null) {
                doc.put("description", String.valueOf(body.get("description")));
            }
            long t = MlCommons.now();
            doc.put("created_time", t);
            doc.put("last_updated_time", t);
            doc.put("configuration", configuration);
            if (body.get("backend_roles") instanceof List<?> roles) {
                doc.put("backend_roles", roles);
            }
            boolean system = (boolean) configuration.get("use_system_index");
            ml.indices().ensure(CONTAINER, ActionListener.wrap(a -> ensure(index(configuration, "working"), "am_working.json", system ? "0-all" : "0-all", ActionListener.wrap(b -> {
                ActionListener<Void> store = ActionListener.wrap(c -> {
                    IndexRequest request = new IndexRequest(CONTAINER).source(doc).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                    MlIndices.system(ml.client(), l -> ml.client().index(request, l), ActionListener.<IndexResponse>wrap(r -> {
                        Map<String, Object> out = new LinkedHashMap<>();
                        out.put("memory_container_id", r.getId());
                        out.put("status", "created");
                        listener.onResponse(out);
                    }, listener::onFailure));
                }, listener::onFailure);
                if (Boolean.TRUE.equals(configuration.get("disable_session"))) {
                    store.onResponse(null);
                } else {
                    ensure(index(configuration, "sessions"), "am_sessions.json", "0-all", store);
                }
            }, listener::onFailure)), listener::onFailure));
        }, listener::onFailure));
    }

    /** The refusal of strategies without both models; {@code current} holds the configuration they join. */
    private static void strategies(Map<String, Object> given, Map<String, Object> current, boolean namespaces) {
        if (given.get("strategies") instanceof List<?> l && l.isEmpty() == false) {
            for (Object o : l) {
                if (o instanceof Map<?, ?> strategy) {
                    Object type = strategy.get("type");
                    if (type != null && Set.of("SEMANTIC", "USER_PREFERENCE", "SUMMARY").contains(String.valueOf(type).toUpperCase(Locale.ROOT)) == false) {
                        throw new IllegalArgumentException("Invalid memory type: " + type + ". Must be SEMANTIC, USER_PREFERENCE, or SUMMARY");
                    }
                }
            }
            if (namespaces) {
                for (Object o : l) {
                    if (!(o instanceof Map<?, ?> strategy) || !(strategy.get("namespace") instanceof List<?> n) || n.isEmpty()) {
                        throw new IllegalArgumentException("Strategy namespace is required. Please provide a non-empty namespace array.");
                    }
                }
            }
            Map<String, Object> merged = new LinkedHashMap<>(current == null ? Map.of() : current);
            merged.putAll(given);
            boolean llm = merged.get("llm_id") != null;
            boolean embedding = merged.get("embedding_model_id") != null && merged.get("embedding_model_type") != null && merged.get("embedding_dimension") != null;
            if (llm == false || embedding == false) {
                String missing = llm == false && embedding == false
                    ? "LLM model and embedding model"
                    : llm == false ? "LLM model" : "embedding model (embedding_model_id, embedding_model_type, dimension)";
                throw new IllegalArgumentException(
                    "Strategies require both an LLM model and embedding model to be configured. Missing: " + missing
                        + ". Strategies use LLM for fact extraction and embedding model for semantic search."
                );
            }
        }
    }

    private void models(Map<String, Object> given, ActionListener<Void> then) {
        String llm = given.get("llm_id") == null ? null : String.valueOf(given.get("llm_id"));
        String embedding = given.get("embedding_model_id") == null ? null : String.valueOf(given.get("embedding_model_id"));
        ActionListener<Void> embeddings = ActionListener.wrap(v -> {
            if (embedding == null) {
                then.onResponse(null);
                return;
            }
            exists(embedding, ActionListener.wrap(found -> {
                if (found == false) {
                    throw new IllegalArgumentException("Embedding model with ID " + embedding + " not found");
                }
                then.onResponse(null);
            }, then::onFailure));
        }, then::onFailure);
        if (llm == null) {
            embeddings.onResponse(null);
            return;
        }
        exists(llm, ActionListener.wrap(found -> {
            if (found == false) {
                throw new IllegalArgumentException("LLM model with ID " + llm + " not found");
            }
            embeddings.onResponse(null);
        }, then::onFailure));
    }

    private void exists(String modelId, ActionListener<Boolean> listener) {
        ml.getDoc(MlIndices.MODEL, modelId, ActionListener.wrap(r -> listener.onResponse(r.isExists()), e -> {
            if (e instanceof IndexNotFoundException) {
                listener.onResponse(false);
            } else {
                listener.onFailure(e);
            }
        }));
    }

    /** {@code GET memory_containers/<id>}. */
    void get(String id, ActionListener<Map<String, Object>> listener) {
        enabled();
        String missing = "Failed to find memory container with the provided memory container id: " + id;
        ml.getDoc(CONTAINER, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw new OpenSearchStatusException(missing, RestStatus.NOT_FOUND);
            }
            Map<String, Object> doc = r.getSourceAsMap();
            doc.remove("tenant_id");
            doc.remove("owner");
            listener.onResponse(doc);
        }, e -> listener.onFailure(e instanceof IndexNotFoundException ? new OpenSearchStatusException(missing, RestStatus.NOT_FOUND) : e)));
    }

    /** The stored container {@code id}, or the refusal stock gives for a missing one. */
    private void container(String id, ActionListener<Map<String, Object>> listener) {
        ml.getDoc(CONTAINER, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw new OpenSearchStatusException(NOT_FOUND, RestStatus.NOT_FOUND);
            }
            listener.onResponse(r.getSourceAsMap());
        }, e -> listener.onFailure(e instanceof IndexNotFoundException ? new OpenSearchStatusException(NOT_FOUND, RestStatus.NOT_FOUND) : e)));
    }

    /** {@code PUT memory_containers/<id>}. */
    @SuppressWarnings("unchecked")
    void update(String id, Map<String, Object> body, ActionListener<UpdateResponse> listener) {
        enabled();
        container(id, ActionListener.wrap(current -> {
            Map<String, Object> fields = new LinkedHashMap<>();
            if (body.get("name") != null) {
                fields.put("name", String.valueOf(body.get("name")));
            }
            if (body.get("description") != null) {
                fields.put("description", String.valueOf(body.get("description")));
            }
            if (body.get("backend_roles") instanceof List<?> roles) {
                fields.put("backend_roles", roles);
            }
            if (body.get("configuration") instanceof Map<?, ?> c) {
                Map<String, Object> given = (Map<String, Object>) c;
                if (given.get("retention_policy") != null && ml.setting(MlSettings.ML_COMMONS_MEMORY_RETENTION_ENABLED) == false) {
                    throw new OpenSearchStatusException(
                        "Cannot set retention_policy: the memory retention feature is not enabled. To enable it, please update the cluster setting "
                            + MlSettings.ML_COMMONS_MEMORY_RETENTION_ENABLED.getKey(),
                        RestStatus.FORBIDDEN
                    );
                }
                Map<String, Object> existing = current.get("configuration") instanceof Map<?, ?> e ? (Map<String, Object>) e : Map.of();
                strategies(given, existing, true);
                Map<String, Object> configuration = new LinkedHashMap<>();
                for (String f : List.of("llm_id", "strategies", "retention_policy")) {
                    if (given.get(f) != null) {
                        configuration.put(f, given.get(f));
                    }
                }
                if (given.get("llm_id") != null) {
                    Object max = given.get("max_infer_size") != null ? given.get("max_infer_size") : 5;
                    configuration.put("max_infer_size", max);
                }
                if (configuration.isEmpty() == false) {
                    fields.put("configuration", configuration);
                }
            }
            fields.put("last_updated_time", MlCommons.now());
            UpdateRequest request = new UpdateRequest(CONTAINER, id).doc(fields).retryOnConflict(5);
            MlIndices.system(ml.client(), l -> ml.client().update(request, l), listener);
        }, listener::onFailure));
    }

    /** {@code DELETE memory_containers/<id>}. */
    void delete(String id, ActionListener<DeleteResponse> listener) {
        enabled();
        container(id, ActionListener.wrap(current -> {
            DeleteRequest request = new DeleteRequest(CONTAINER, id).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            MlIndices.system(ml.client(), l -> ml.client().delete(request, l), listener);
        }, listener::onFailure));
    }

    /** {@code POST memory_containers/_search}. */
    void search(byte[] body, SearchSourceBuilder source, ActionListener<SearchResponse> listener) {
        enabled();
        if (body.length == 0) {
            throw new OpenSearchParseException("request body or source parameter is required");
        }
        ml.search(CONTAINER, source, listener);
    }

    // ---- Memories -----------------------------------------------------------------------------

    private static String type(String type) {
        if (TYPES.contains(type) == false) {
            throw new IllegalArgumentException("Invalid memory type: " + type + ". Must be one of: " + String.join(", ", TYPES));
        }
        return type;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> configuration(Map<String, Object> container) {
        return container.get("configuration") instanceof Map<?, ?> c ? (Map<String, Object>) c : Map.of();
    }

    /** The working memory a request body describes, as it is stored. */
    private static Map<String, Object> working(byte[] body, boolean[] infer) throws IOException {
        Map<String, Object> doc = new LinkedHashMap<>();
        Map<String, String> namespace = new LinkedHashMap<>();
        String payload = "conversational";
        boolean inferred = false;
        List<Object> messages = new ArrayList<>();
        try (XContentParser p = JsonXContent.jsonXContent.createParser(NamedXContentRegistry.EMPTY, DeprecationHandler.IGNORE_DEPRECATIONS, body.length == 0 ? "{}".getBytes(StandardCharsets.UTF_8) : body)) {
            XContentParserUtils.ensureExpectedToken(XContentParser.Token.START_OBJECT, p.nextToken(), p);
            while (p.nextToken() != XContentParser.Token.END_OBJECT) {
                String field = p.currentName();
                p.nextToken();
                switch (field) {
                    case "messages" -> {
                        XContentParserUtils.ensureExpectedToken(XContentParser.Token.START_ARRAY, p.currentToken(), p);
                        while (p.nextToken() != XContentParser.Token.END_ARRAY) {
                            messages.add(message(p));
                        }
                    }
                    case "message_id" -> doc.put("message_id", p.intValue());
                    case "structured_data" -> {
                        if (p.currentToken() == XContentParser.Token.START_OBJECT) {
                            doc.put("structured_data", p.map());
                        } else {
                            p.skipChildren();
                            doc.put("structured_data", new LinkedHashMap<>());
                        }
                    }
                    case "binary_data" -> doc.put("binary_data", p.text());
                    case "namespace" -> {
                        namespace.putAll(strings(p));
                        infer[1] = true;
                    }
                    case "metadata", "tags" -> doc.put(field, strings(p));
                    case "infer" -> inferred = p.booleanValue();
                    case "payload_type" -> {
                        String given = p.text();
                        if (given.equalsIgnoreCase("conversational") == false && given.equalsIgnoreCase("data") == false) {
                            throw new IllegalArgumentException("Invalid working memory type: " + given + ". Must be one of: CONVERSATIONAL, DATA");
                        }
                        payload = given.toLowerCase(Locale.ROOT);
                    }
                    case "checkpoint_id" -> doc.put("checkpoint_id", p.text());
                    default -> p.skipChildren();
                }
            }
        }
        infer[0] = inferred;
        doc.put("payload_type", payload);
        long t = MlCommons.now();
        doc.put("created_time", t);
        doc.put("last_updated_time", t);
        doc.put("infer", inferred);
        if (namespace.isEmpty() == false) {
            doc.put("namespace_size", namespace.size());
            doc.put("namespace", namespace);
            if (namespace.get("session_id") != null) {
                doc.put("session_id", namespace.get("session_id"));
            }
        }
        if (messages.isEmpty() == false) {
            doc.put("messages", messages);
        }
        return doc;
    }

    private static Map<String, Object> message(XContentParser p) throws IOException {
        XContentParserUtils.ensureExpectedToken(XContentParser.Token.START_OBJECT, p.currentToken(), p);
        String role = null;
        List<Object> content = null;
        while (p.nextToken() != XContentParser.Token.END_OBJECT) {
            String field = p.currentName();
            p.nextToken();
            switch (field) {
                case "role" -> role = p.text();
                case "content" -> {
                    XContentParserUtils.ensureExpectedToken(XContentParser.Token.START_ARRAY, p.currentToken(), p);
                    content = new ArrayList<>();
                    while (p.nextToken() != XContentParser.Token.END_ARRAY) {
                        content.add(p.map());
                    }
                }
                default -> p.skipChildren();
            }
        }
        if (role == null || content == null) {
            throw new IllegalArgumentException("Message must have role and content");
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("role", role);
        out.put("content", content);
        return out;
    }

    /** An object whose values are kept as text: numbers as written, objects and arrays as their JSON. */
    private static Map<String, String> strings(XContentParser p) throws IOException {
        Map<String, String> out = new LinkedHashMap<>();
        if (p.currentToken() != XContentParser.Token.START_OBJECT) {
            p.skipChildren();
            return out;
        }
        while (p.nextToken() != XContentParser.Token.END_OBJECT) {
            String key = p.currentName();
            XContentParser.Token token = p.nextToken();
            switch (token) {
                case VALUE_NULL -> {
                }
                case START_OBJECT, START_ARRAY -> {
                    try (XContentBuilder b = XContentFactory.jsonBuilder()) {
                        b.copyCurrentStructure(p);
                        out.put(key, b.toString());
                    }
                }
                default -> out.put(key, p.text());
            }
        }
        return out;
    }

    /** {@code POST memory_containers/<id>/memories}. */
    void add(String containerId, byte[] body, ActionListener<Map<String, Object>> listener) throws IOException {
        enabled();
        if (body.length == 0) {
            throw new IOException("Add memory request has empty body");
        }
        boolean[] infer = new boolean[2];
        Map<String, Object> doc = working(body, infer);
        container(containerId, ActionListener.wrap(container -> {
            Map<String, Object> configuration = configuration(container);
            if (infer[0] && configuration.get("llm_id") == null) {
                throw new IllegalArgumentException("infer=true requires llm_model_id to be configured in memory storage");
            }
            doc.put("memory_container_id", containerId);
            String workingIndex = index(configuration, "working");
            boolean generated = infer[1] && configuration.get("llm_id") != null && doc.get("session_id") == null && Boolean.TRUE.equals(configuration.get("disable_session")) == false;
            Object given = doc.get("namespace");
            if (generated) {
                String id = org.opensearch.common.UUIDs.base64UUID();
                Map<String, Object> namespace = new LinkedHashMap<>();
                if (given instanceof Map<?, ?> m) {
                    m.forEach((k, v) -> namespace.put(String.valueOf(k), v));
                }
                namespace.put("session_id", id);
                doc.put("session_id", id);
                doc.put("namespace", namespace);
                doc.put("namespace_size", namespace.size());
            }
            Object session = doc.get("session_id");
            ActionListener<Void> store = ActionListener.wrap(v -> ensure(workingIndex, "am_working.json", "0-all", ActionListener.wrap(w -> {
                IndexRequest request = new IndexRequest(workingIndex).source(doc).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                MlIndices.system(ml.client(), l -> ml.client().index(request, l), ActionListener.<IndexResponse>wrap(r -> {
                    Map<String, Object> out = new LinkedHashMap<>();
                    if (session != null) {
                        out.put("session_id", session);
                    }
                    out.put("working_memory_id", r.getId());
                    listener.onResponse(out);
                }, listener::onFailure));
            }, listener::onFailure)), listener::onFailure);
            if (session == null || Boolean.TRUE.equals(configuration.get("disable_session"))) {
                store.onResponse(null);
            } else {
                session(containerId, configuration, String.valueOf(session), generated ? (given == null ? Map.of() : given) : doc.get("namespace"), generated, store);
            }
        }, listener::onFailure));
    }

    /** Makes sure session {@code id} exists and notes it was used. */
    private void session(String containerId, Map<String, Object> configuration, String id, Object namespace, boolean generated, ActionListener<Void> then) {
        String sessions = index(configuration, "sessions");
        ensure(sessions, "am_sessions.json", "0-all", ActionListener.wrap(v -> {
            Map<String, Object> doc = new LinkedHashMap<>();
            long t = MlCommons.now();
            if (generated) {
                doc.put("summary", "");
            }
            doc.put("created_time", t);
            doc.put("last_updated_time", t);
            doc.put("namespace", namespace);
            doc.put("memory_container_id", containerId);
            IndexRequest create = new IndexRequest(sessions).id(id).source(doc).opType(org.opensearch.action.DocWriteRequest.OpType.CREATE);
            MlIndices.system(ml.client(), l -> ml.client().index(create, l), ActionListener.<IndexResponse>wrap(r -> {
                if (generated) {
                    then.onResponse(null);
                } else {
                    touch(sessions, id, then);
                }
            }, e -> {
                if (e instanceof org.opensearch.index.engine.VersionConflictEngineException) {
                    touch(sessions, id, then);
                } else {
                    then.onFailure(e);
                }
            }));
        }, then::onFailure));
    }

    private void touch(String sessions, String id, ActionListener<Void> then) {
        UpdateRequest update = new UpdateRequest(sessions, id).doc(Map.of("last_updated_time", MlCommons.now())).retryOnConflict(5);
        MlIndices.system(ml.client(), l -> ml.client().update(update, l), ActionListener.<UpdateResponse>wrap(r -> then.onResponse(null), then::onFailure));
    }

    /** The index holding {@code type} memories of {@code container}, or the refusal for one not there. */
    private String existing(Map<String, Object> container, String type) {
        String index = index(configuration(container), type);
        if (ml.clusterState().metadata().hasIndex(index) == false) {
            throw new OpenSearchStatusException("Memory index not found", RestStatus.NOT_FOUND);
        }
        return index;
    }

    /** {@code GET memory_containers/<id>/memories/<type>/<memory_id>}. */
    void getMemory(String containerId, String type, String id, ActionListener<Map<String, Object>> listener) {
        enabled();
        type(type);
        container(containerId, ActionListener.wrap(container -> {
            String index = existing(container, type);
            ml.getDoc(index, id, ActionListener.wrap(r -> {
                if (r.isExists() == false) {
                    throw new OpenSearchStatusException("Memory not found", RestStatus.NOT_FOUND);
                }
                listener.onResponse(shape(type, r.getSourceAsMap()));
            }, listener::onFailure));
        }, listener::onFailure));
    }

    /** A stored memory as the API reads it back. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> shape(String type, Map<String, Object> stored) {
        Map<String, Object> out = new LinkedHashMap<>();
        List<String> fields = switch (type) {
            case "working" -> List.of("memory_container_id", "payload_type", "messages", "binary_data", "structured_data", "namespace", "namespace_size", "infer", "metadata", "tags", "message_id", "checkpoint_id", "created_time", "last_updated_time");
            case "sessions" -> List.of("memory_container_id", "summary", "created_time", "last_updated_time", "metadata", "agents", "additional_info", "namespace");
            default -> List.copyOf(stored.keySet());
        };
        if (type.equals("working") && stored.get("messages") instanceof List<?> messages) {
            for (Object m : messages) {
                Map<String, Object> message = m instanceof Map<?, ?> mm ? (Map<String, Object>) mm : Map.of();
                if (message.get("role") == null || message.get("content") == null) {
                    throw new IllegalArgumentException("Message must have role and content");
                }
            }
        }
        for (String f : fields) {
            if (stored.get(f) != null) {
                out.put(f, stored.get(f));
            }
        }
        return out;
    }

    /** {@code PUT memory_containers/<id>/memories/<type>/<memory_id>}. */
    void updateMemory(String containerId, String type, String id, Map<String, Object> body, ActionListener<UpdateResponse> listener) {
        enabled();
        type(type);
        if (body.isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }
        container(containerId, ActionListener.wrap(container -> {
            String index = existing(container, type);
            Map<String, Object> fields = new LinkedHashMap<>();
            boolean working = type.equals("working");
            for (String f : working ? List.of("tags", "metadata", "structured_data", "binary_data", "messages") : List.of("summary", "metadata", "agents", "additional_info")) {
                if (body.get(f) != null) {
                    fields.put(f, body.get(f));
                }
            }
            fields.put("last_updated_time", MlCommons.now());
            replace(index, id, fields, listener);
        }, listener::onFailure));
    }

    /** Sets {@code fields} on document {@code id}, each replacing the stored value as a whole, and answers as an update does. */
    private void replace(String index, String id, Map<String, Object> fields, ActionListener<UpdateResponse> listener) {
        ml.getDoc(index, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw new OpenSearchStatusException("Memory not found", RestStatus.NOT_FOUND);
            }
            Map<String, Object> source = new LinkedHashMap<>(r.getSourceAsMap());
            source.putAll(fields);
            IndexRequest request = new IndexRequest(index).id(id)
                .source(source)
                .setIfSeqNo(r.getSeqNo())
                .setIfPrimaryTerm(r.getPrimaryTerm())
                .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            MlIndices.system(ml.client(), l -> ml.client().index(request, l), ActionListener.<IndexResponse>wrap(w -> {
                UpdateResponse out = new UpdateResponse(w.getShardInfo(), w.getShardId(), w.getId(), w.getSeqNo(), w.getPrimaryTerm(), w.getVersion(), DocWriteResponse.Result.UPDATED);
                out.setForcedRefresh(w.forcedRefresh());
                listener.onResponse(out);
            }, listener::onFailure));
        }, listener::onFailure));
    }

    /** {@code DELETE memory_containers/<id>/memories/<type>/<memory_id>}. */
    void deleteMemory(String containerId, String type, String id, ActionListener<DeleteResponse> listener) {
        enabled();
        type(type);
        container(containerId, ActionListener.wrap(container -> {
            String index = existing(container, type);
            DeleteRequest request = new DeleteRequest(index, id);
            MlIndices.system(ml.client(), l -> ml.client().delete(request, l), ActionListener.<DeleteResponse>wrap(r -> {
                if (r.getResult() == DocWriteResponse.Result.NOT_FOUND) {
                    throw new OpenSearchStatusException("Memory not found", RestStatus.NOT_FOUND);
                }
                listener.onResponse(r);
            }, listener::onFailure));
        }, listener::onFailure));
    }

    /** {@code POST memory_containers/<id>/memories/<type>/_search}. */
    void searchMemories(String containerId, String type, byte[] body, SearchSourceBuilder source, ActionListener<SearchResponse> listener) {
        enabled();
        type(type);
        if (body.length == 0) {
            throw new IllegalArgumentException("Search memories request has empty body");
        }
        container(containerId, ActionListener.wrap(container -> {
            String index = index(configuration(container), type);
            if (ml.clusterState().metadata().hasIndex(index) == false) {
                throw new OpenSearchException("Search execution failed: index must not be null", new NullPointerException("index must not be null"));
            }
            MlIndices.system(ml.client(), l -> ml.client().search(new SearchRequest(index).source(source), l), listener);
        }, listener::onFailure));
    }

    static boolean blank(String s) {
        return Strings.isNullOrEmpty(s);
    }
}
