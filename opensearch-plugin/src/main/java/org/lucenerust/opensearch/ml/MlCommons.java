/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.DocWriteResponse;
import org.opensearch.action.delete.DeleteRequest;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.PlainActionFuture;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.cluster.ClusterChangedEvent;
import org.opensearch.cluster.ClusterStateListener;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.node.DiscoveryNodeRole;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.util.concurrent.OpenSearchExecutors;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.reindex.BulkByScrollResponse;
import org.opensearch.index.reindex.DeleteByQueryAction;
import org.opensearch.index.reindex.DeleteByQueryRequest;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.threadpool.ExecutorBuilder;
import org.opensearch.threadpool.FixedExecutorBuilder;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.client.Client;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

/**
 * The plugin's port of OpenSearch ml-commons: local models registered, deployed and run for
 * prediction with the stock REST API and system index documents. Inference runs in Rust
 * ({@code lucene-ml}); this class keeps ml-commons' bookkeeping (model groups, models and their
 * chunks, tasks) in the stock indices and formats.
 *
 * <p>It is installed only when the stock ml-commons plugin is not: with both, stock serves
 * {@code _plugins/_ml}.
 */
public final class MlCommons {
    private static final Logger LOGGER = LogManager.getLogger(MlCommons.class);

    public static final String BASE = "/_plugins/_ml";
    public static final String GENERAL_THREAD_POOL = "opensearch_ml_general";
    public static final String REGISTER_THREAD_POOL = "opensearch_ml_register";
    public static final String DEPLOY_THREAD_POOL = "opensearch_ml_deploy";
    public static final String PREDICT_THREAD_POOL = "opensearch_ml_predict";
    static final String THREAD_POOL_PREFIX = "thread_pool.ml_commons.";
    /** {@code ModelHelper.CHUNK_SIZE}. */
    static final int CHUNK_SIZE = 10_000_000;
    private static final String LOCAL_MODEL_DISABLED =
        "Local Model is disabled. To enable it, update the setting plugins.ml_commons.local_model.enabled to true.";

    /** ml-commons' {@code ml} node role, so nodes configured with {@code node.roles: [ml]} start. */
    public static final DiscoveryNodeRole ML_ROLE = new DiscoveryNodeRole("ml", "ml") {
        @Override
        public org.opensearch.common.settings.Setting<Boolean> legacySetting() {
            return null;
        }
    };

    private static volatile MlCommons instance;

    private final Client client;
    private final ClusterService clusterService;
    private final ThreadPool threadPool;
    private final MlIndices indices;
    private final LocalModels local;
    private final Path cacheDir;
    private final AtomicInteger roundRobin = new AtomicInteger();
    private final AtomicBoolean redeployed = new AtomicBoolean();

    private MlCommons(Client client, ClusterService clusterService, ThreadPool threadPool, Path cacheDir, Path runtimeLibrary) {
        this.client = client;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.indices = new MlIndices(client, clusterService);
        this.local = new LocalModels(client, cacheDir, runtimeLibrary);
        this.cacheDir = cacheDir;
    }

    /** Whether the stock ml-commons plugin is installed beside this one. */
    public static boolean stockInstalled(Path pluginDir) {
        return Files.isDirectory(pluginDir.resolveSibling("opensearch-ml"));
    }

    /** Sets up the node's instance; {@code dataDir} holds {@code ml_cache} as on stock. */
    public static MlCommons initialize(Client client, ClusterService clusterService, ThreadPool threadPool, Path dataDir, Path runtimeLibrary) {
        MlCommons ml = new MlCommons(client, clusterService, threadPool, dataDir.resolve("ml_cache").resolve("models_cache"), runtimeLibrary);
        instance = ml;
        clusterService.addListener(ml.redeployListener());
        return ml;
    }

    /** The node's instance. */
    public static MlCommons get() {
        MlCommons ml = instance;
        if (ml == null) {
            throw new IllegalStateException("ml-commons is not initialized on this node");
        }
        return ml;
    }

    /** Undeploys every local model of the node's instance, if there is one. */
    public static void closeIfInitialized() {
        MlCommons ml = instance;
        if (ml != null) {
            ml.close();
        }
    }

    /** Undeploys every local model, for node shutdown. */
    public void close() {
        local.closeAll();
    }

    /** ml-commons' thread pools, under their stock names and settings. */
    public static List<ExecutorBuilder<?>> executorBuilders(Settings settings) {
        int processors = OpenSearchExecutors.allocatedProcessors(settings);
        return List.of(
            new FixedExecutorBuilder(settings, GENERAL_THREAD_POOL, Math.max(1, processors - 1), 100, THREAD_POOL_PREFIX + GENERAL_THREAD_POOL, false),
            new FixedExecutorBuilder(settings, REGISTER_THREAD_POOL, Math.max(4, processors - 1), 10, THREAD_POOL_PREFIX + REGISTER_THREAD_POOL, false),
            new FixedExecutorBuilder(settings, DEPLOY_THREAD_POOL, Math.max(4, processors - 1), 10, THREAD_POOL_PREFIX + DEPLOY_THREAD_POOL, false),
            new FixedExecutorBuilder(settings, PREDICT_THREAD_POOL, processors * 2, 10000, THREAD_POOL_PREFIX + PREDICT_THREAD_POOL, false)
        );
    }

    LocalModels local() {
        return local;
    }

    Client client() {
        return client;
    }

    private static long now() {
        return Instant.now().toEpochMilli();
    }

    private <T> T setting(org.opensearch.common.settings.Setting<T> s) {
        return clusterService.getClusterSettings().get(s);
    }

    // ---- Documents --------------------------------------------------------------------------

    private void getDoc(String index, String id, ActionListener<GetResponse> listener) {
        MlIndices.system(client, l -> client.get(new GetRequest(index, id), l), listener);
    }

    private void indexDoc(String index, String id, Map<String, Object> source, ActionListener<IndexResponse> listener) {
        IndexRequest request = new IndexRequest(index).source(source).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
        if (id != null) {
            request.id(id);
        }
        indices.ensure(index, ActionListener.wrap(v -> MlIndices.system(client, l -> client.index(request, l), listener), listener::onFailure));
    }

    private void updateDoc(String index, String id, Map<String, Object> fields, ActionListener<UpdateResponse> listener) {
        UpdateRequest request = new UpdateRequest(index, id).doc(fields).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
        MlIndices.system(client, l -> client.update(request, l), listener);
    }

    private void searchDocs(SearchRequest request, ActionListener<SearchResponse> listener) {
        String index = request.indices()[0];
        if (clusterService.state().metadata().hasIndex(index) == false) {
            // ml-commons answers a search of a missing system index with no hits.
            request.indicesOptions(org.opensearch.action.support.IndicesOptions.lenientExpandOpen());
        }
        MlIndices.system(client, l -> client.search(request, l), listener);
    }

    private static <T> ActionListener<T> logged(String what) {
        return ActionListener.wrap(r -> {}, e -> LOGGER.warn("lucene-rust ml: {} failed", what, e));
    }

    // ---- Tasks -----------------------------------------------------------------------------

    private void createTask(String type, String function, String modelId, List<String> workers, ActionListener<String> listener) {
        Map<String, Object> task = new LinkedHashMap<>();
        if (modelId != null) {
            task.put("model_id", modelId);
        }
        task.put("task_type", type);
        task.put("function_name", function);
        task.put("state", "CREATED");
        task.put("worker_node", workers);
        long t = now();
        task.put("create_time", t);
        task.put("last_update_time", t);
        task.put("is_async", true);
        indexDoc(MlIndices.TASK, null, task, ActionListener.wrap(r -> listener.onResponse(r.getId()), listener::onFailure));
    }

    private void finishTask(String taskId, String state, Map<String, Object> extra) {
        Map<String, Object> fields = new LinkedHashMap<>(extra);
        fields.put("state", state);
        fields.put("last_update_time", now());
        updateDoc(MlIndices.TASK, taskId, fields, logged("task update"));
    }

    private void failTask(String taskId, Exception e) {
        finishTask(taskId, "FAILED", Map.of("error", e.getMessage() == null ? "" : e.getMessage()));
    }

    /** {@code GET _plugins/_ml/tasks/<id>}. */
    public void getTask(String taskId, ActionListener<Map<String, Object>> listener) {
        getDoc(MlIndices.TASK, taskId, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw new OpenSearchStatusException("Failed to find task", RestStatus.NOT_FOUND);
            }
            listener.onResponse(r.getSourceAsMap());
        }, e -> listener.onFailure(missingIndex(e, "Failed to find task"))));
    }

    /** {@code DELETE _plugins/_ml/tasks/<id>}. */
    public void deleteTask(String taskId, ActionListener<DeleteResponse> listener) {
        getTask(taskId, ActionListener.wrap(task -> {
            if ("RUNNING".equals(task.get("state"))) {
                throw new IllegalArgumentException("Task cannot be deleted in running state. Try after sometime");
            }
            DeleteRequest request = new DeleteRequest(MlIndices.TASK, taskId).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            MlIndices.system(client, l -> client.delete(request, l), listener);
        }, listener::onFailure));
    }

    private static Exception missingIndex(Exception e, String message) {
        if (e instanceof org.opensearch.index.IndexNotFoundException) {
            return new OpenSearchStatusException(message, RestStatus.NOT_FOUND);
        }
        return e;
    }

    // ---- Model groups ----------------------------------------------------------------------

    /** {@code POST _plugins/_ml/model_groups/_register}: {@code model_group_id} of the new group. */
    public void registerModelGroup(Map<String, Object> body, ActionListener<String> listener) {
        Object name = body.get("name");
        if (name == null || String.valueOf(name).isBlank()) {
            listener.onFailure(new IllegalArgumentException("model group name is null"));
            return;
        }
        findGroupByName(String.valueOf(name), ActionListener.wrap(existing -> {
            if (existing != null) {
                throw new IllegalArgumentException(
                    "The name you provided is already being used by a model group with ID: " + existing.getKey() + "."
                );
            }
            createGroup(String.valueOf(name), (String) body.get("description"), listener);
        }, listener::onFailure));
    }

    private void createGroup(String name, String description, ActionListener<String> listener) {
        Map<String, Object> group = new LinkedHashMap<>();
        group.put("name", name);
        if (description != null) {
            group.put("description", description);
        }
        group.put("latest_version", 0);
        group.put("access", "public");
        long t = now();
        group.put("created_time", t);
        group.put("last_updated_time", t);
        indexDoc(MlIndices.MODEL_GROUP, null, group, ActionListener.wrap(r -> listener.onResponse(r.getId()), listener::onFailure));
    }

    private void findGroupByName(String name, ActionListener<Map.Entry<String, Map<String, Object>>> listener) {
        if (clusterService.state().metadata().hasIndex(MlIndices.MODEL_GROUP) == false) {
            listener.onResponse(null);
            return;
        }
        SearchRequest request = new SearchRequest(MlIndices.MODEL_GROUP).source(
            new SearchSourceBuilder().query(QueryBuilders.termQuery("name.keyword", name)).size(1)
        );
        searchDocs(request, ActionListener.wrap(r -> {
            if (r.getHits().getHits().length == 0) {
                listener.onResponse(null);
            } else {
                var hit = r.getHits().getHits()[0];
                listener.onResponse(Map.entry(hit.getId(), hit.getSourceAsMap()));
            }
        }, listener::onFailure));
    }

    /** {@code GET _plugins/_ml/model_groups/<id>}. */
    public void getModelGroup(String id, ActionListener<Map<String, Object>> listener) {
        String missing = "Failed to find model group with the provided model group id: " + id;
        getDoc(MlIndices.MODEL_GROUP, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw new OpenSearchStatusException(missing, RestStatus.NOT_FOUND);
            }
            listener.onResponse(r.getSourceAsMap());
        }, e -> listener.onFailure(missingIndex(e, missing))));
    }

    /** {@code PUT _plugins/_ml/model_groups/<id>}: name and description. */
    public void updateModelGroup(String id, Map<String, Object> body, ActionListener<UpdateResponse> listener) {
        getModelGroup(id, ActionListener.wrap(group -> {
            Map<String, Object> fields = new LinkedHashMap<>();
            if (body.get("name") != null) {
                fields.put("name", body.get("name"));
            }
            if (body.get("description") != null) {
                fields.put("description", body.get("description"));
            }
            fields.put("last_updated_time", now());
            updateDoc(MlIndices.MODEL_GROUP, id, fields, listener);
        }, listener::onFailure));
    }

    /** {@code DELETE _plugins/_ml/model_groups/<id>}: refused while the group has models. */
    public void deleteModelGroup(String id, ActionListener<DeleteResponse> listener) {
        getModelGroup(id, ActionListener.wrap(group -> {
            SearchRequest models = new SearchRequest(MlIndices.MODEL).source(
                new SearchSourceBuilder().query(QueryBuilders.termQuery("model_group_id", id)).size(0)
            );
            searchDocs(models, ActionListener.wrap(r -> {
                if (r.getHits().getTotalHits() != null && r.getHits().getTotalHits().value() > 0) {
                    throw new OpenSearchStatusException("Cannot delete the model group when it has associated model versions", RestStatus.CONFLICT);
                }
                DeleteRequest request = new DeleteRequest(MlIndices.MODEL_GROUP, id).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                MlIndices.system(client, l -> client.delete(request, l), listener);
            }, listener::onFailure));
        }, listener::onFailure));
    }

    /** A search of {@code index}, run as ml-commons runs it for its search endpoints. */
    public void search(String index, SearchSourceBuilder source, ActionListener<SearchResponse> listener) {
        if (index.equals(MlIndices.MODEL)) {
            BoolQueryBuilder visible = QueryBuilders.boolQuery()
                .should(QueryBuilders.termQuery("is_hidden", false))
                .should(QueryBuilders.boolQuery().mustNot(QueryBuilders.existsQuery("is_hidden")))
                .minimumShouldMatch(1);
            BoolQueryBuilder query = QueryBuilders.boolQuery().filter(visible);
            if (source.query() != null) {
                query.must(source.query());
            }
            source.query(query);
        }
        searchDocs(new SearchRequest(index).source(source), listener);
    }

    // ---- Models ----------------------------------------------------------------------------

    /** {@code GET _plugins/_ml/models/<id>}. */
    public void getModel(String id, ActionListener<Map<String, Object>> listener) {
        String missing = "Failed to find model with the provided model id: " + id;
        getDoc(MlIndices.MODEL, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw new OpenSearchStatusException(missing, RestStatus.NOT_FOUND);
            }
            Map<String, Object> model = r.getSourceAsMap();
            model.remove("model_content");
            listener.onResponse(model);
        }, e -> listener.onFailure(missingIndex(e, missing))));
    }

    /**
     * {@code POST _plugins/_ml/models/_register} for a local model at a URL: answers with the task
     * id; the download, chunking and model document follow on the register pool. With {@code deploy}
     * the model is deployed once registered.
     */
    public void registerModel(Map<String, Object> body, boolean deploy, ActionListener<Map<String, Object>> listener) {
        String name = (String) body.get("name");
        if (name == null) {
            listener.onFailure(new IllegalArgumentException("model name is null"));
            return;
        }
        String format = (String) body.get("model_format");
        if (format == null && body.get("connector") == null && body.get("connector_id") == null) {
            listener.onFailure(new IllegalArgumentException("model format is null"));
            return;
        }
        if (body.get("connector") != null || body.get("connector_id") != null) {
            listener.onFailure(new IllegalArgumentException("lucene-rust does not run remote models yet"));
            return;
        }
        if (setting(MlSettings.ML_COMMONS_LOCAL_MODEL_ENABLED) == false) {
            listener.onFailure(new IllegalArgumentException(LOCAL_MODEL_DISABLED));
            return;
        }
        String url = (String) body.get("url");
        if (url == null) {
            // Pre-trained models (downloaded by name and version) and chunk uploads are not ported.
            listener.onFailure(new IllegalArgumentException("lucene-rust registers local models from a url only"));
            return;
        }
        if (setting(MlSettings.ML_COMMONS_ALLOW_MODEL_URL) == false) {
            listener.onFailure(
                new IllegalArgumentException(
                    "To upload custom model user needs to enable allow_registering_model_via_url settings. Otherwise please use OpenSearch pre-trained models."
                )
            );
            return;
        }
        if (Pattern.compile(setting(MlSettings.ML_COMMONS_TRUSTED_URL_REGEX)).matcher(url).find() == false) {
            listener.onFailure(new IllegalArgumentException("URL can't match trusted url regex"));
            return;
        }
        String function = ((String) body.getOrDefault("function_name", "TEXT_EMBEDDING")).toUpperCase(Locale.ROOT);
        String workers = clusterService.localNode().getId();
        createTask("REGISTER_MODEL", function, null, List.of(workers), ActionListener.wrap(taskId -> {
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("task_id", taskId);
            response.put("status", "CREATED");
            listener.onResponse(response);
            threadPool.executor(REGISTER_THREAD_POOL).execute(() -> register(taskId, function, body, deploy));
        }, listener::onFailure));
    }

    private void register(String taskId, String function, Map<String, Object> body, boolean deploy) {
        String name = (String) body.get("name");
        Object groupId = body.get("model_group_id");
        ActionListener<String> withGroup = ActionListener.wrap(group -> {
            nextVersion(group, ActionListener.wrap(version -> writeModel(taskId, function, body, group, version, deploy), e -> failTask(taskId, e)));
        }, e -> failTask(taskId, e));
        if (groupId != null) {
            withGroup.onResponse(String.valueOf(groupId));
        } else {
            findGroupByName(name, ActionListener.wrap(existing -> {
                if (existing != null) {
                    withGroup.onResponse(existing.getKey());
                } else {
                    createGroup(name, (String) body.get("description"), withGroup);
                }
            }, withGroup::onFailure));
        }
    }

    /** Takes the group's next version number, as {@code MLModelGroupManager} does. */
    private void nextVersion(String groupId, ActionListener<String> listener) {
        nextVersion(groupId, 10, listener);
    }

    /** Bumps {@code latest_version} only if the group is unchanged since read, so concurrent registers get distinct versions. */
    private void nextVersion(String groupId, int attempts, ActionListener<String> listener) {
        String missing = "Failed to find model group with the provided model group id: " + groupId;
        getDoc(MlIndices.MODEL_GROUP, groupId, ActionListener.wrap(g -> {
            if (g.isExists() == false) {
                throw new OpenSearchStatusException(missing, RestStatus.NOT_FOUND);
            }
            int version = ((Number) g.getSourceAsMap().getOrDefault("latest_version", 0)).intValue() + 1;
            UpdateRequest request = new UpdateRequest(MlIndices.MODEL_GROUP, groupId).doc(
                Map.of("latest_version", version, "last_updated_time", now())
            ).setIfSeqNo(g.getSeqNo()).setIfPrimaryTerm(g.getPrimaryTerm()).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            MlIndices.system(client, l -> client.update(request, l), ActionListener.<UpdateResponse>wrap(
                r -> listener.onResponse(Integer.toString(version)),
                e -> {
                    if (org.opensearch.ExceptionsHelper.unwrapCause(e) instanceof org.opensearch.index.engine.VersionConflictEngineException && attempts > 1) {
                        nextVersion(groupId, attempts - 1, listener);
                    } else {
                        listener.onFailure(e);
                    }
                }
            ));
        }, e -> listener.onFailure(missingIndex(e, missing))));
    }

    @SuppressWarnings("unchecked")
    private void writeModel(String taskId, String function, Map<String, Object> body, String groupId, String version, boolean deploy) {
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("name", body.get("name"));
        model.put("model_group_id", groupId);
        if (body.get("description") != null) {
            model.put("description", body.get("description"));
        }
        model.put("algorithm", function);
        model.put("model_version", version);
        model.put("model_format", String.valueOf(body.get("model_format")).toUpperCase(Locale.ROOT));
        model.put("model_state", "REGISTERING");
        Object config = body.get("model_config");
        if (config instanceof Map<?, ?> c) {
            model.put("model_config", modelConfig((Map<String, Object>) c));
        }
        if (body.get("model_content_hash_value") != null) {
            model.put("model_content_hash_value", body.get("model_content_hash_value"));
        }
        long t = now();
        model.put("created_time", t);
        model.put("last_updated_time", t);
        model.put("is_hidden", Boolean.TRUE.equals(body.get("is_hidden")));
        indexDoc(MlIndices.MODEL, null, model, ActionListener.wrap(r -> {
            String modelId = r.getId();
            updateDoc(MlIndices.TASK, taskId, Map.of("model_id", modelId, "state", "RUNNING", "last_update_time", now()), logged("task update"));
            threadPool.executor(REGISTER_THREAD_POOL).execute(() -> {
                try {
                    downloadAndChunk(modelId, model, (String) body.get("url"));
                    finishTask(taskId, "COMPLETED", Map.of("model_id", modelId));
                    if (deploy) {
                        deployModel(modelId, null, logged("deploy after register"));
                    }
                } catch (Exception e) {
                    LOGGER.warn("lucene-rust ml: registering model [{}] failed", modelId, e);
                    updateDoc(MlIndices.MODEL, modelId, Map.of("model_state", "REGISTER_FAILED", "last_updated_time", now()), logged("model update"));
                    failTask(taskId, e);
                }
            });
        }, e -> failTask(taskId, e)));
    }

    /** {@code model_config} as {@code TextEmbeddingModelConfig} writes it: enum values upper case. */
    private static Map<String, Object> modelConfig(Map<String, Object> config) {
        // TextEmbeddingModelConfig.toXContent's fields and order; normalize_result only when true.
        Map<String, Object> out = new LinkedHashMap<>();
        for (String key : List.of(
            "model_type",
            "embedding_dimension",
            "framework_type",
            "all_config",
            "additional_config",
            "model_max_length",
            "pooling_mode",
            "normalize_result",
            "query_prefix",
            "passage_prefix"
        )) {
            Object v = config.get(key);
            if (v == null || key.equals("normalize_result") && Boolean.TRUE.equals(v) == false) {
                continue;
            }
            if ((key.equals("framework_type") || key.equals("pooling_mode")) && v instanceof String s) {
                v = s.toUpperCase(Locale.ROOT);
            }
            out.put(key, v);
        }
        return out;
    }

    /** Downloads the zip, checks its hash, stores it as base64 chunks and marks the model registered. */
    private void downloadAndChunk(String modelId, Map<String, Object> model, String url) throws Exception {
        Path dir = cacheDir.resolve("register");
        Files.createDirectories(dir);
        Path tmp = Files.createTempFile(dir, modelId + "-", ".zip");
        try {
            URL u = URI.create(url).toURL();
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            try (InputStream in = new java.security.DigestInputStream(u.openStream(), sha)) {
                Files.copy(in, tmp, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            }
            long size = Files.size(tmp);
            String hash = (String) model.get("model_content_hash_value");
            String got = HexFormat.of().formatHex(sha.digest());
            // Checked before any chunk is written, so a refused zip leaves no chunks behind.
            if (hash != null && hash.equals(got) == false) {
                throw new IllegalArgumentException("model content changed");
            }
            int chunks = (int) ((size + CHUNK_SIZE - 1) / CHUNK_SIZE);
            try (InputStream in = Files.newInputStream(tmp)) {
                for (int i = 0; i < chunks; i++) {
                    byte[] chunk = in.readNBytes(CHUNK_SIZE);
                    Map<String, Object> doc = new LinkedHashMap<>();
                    doc.put("model_id", modelId);
                    doc.put("model_version", model.get("model_version"));
                    doc.put("name", model.get("name"));
                    doc.put("algorithm", model.get("algorithm"));
                    doc.put("model_format", model.get("model_format"));
                    doc.put("chunk_number", i);
                    doc.put("total_chunks", chunks);
                    doc.put("model_content", Base64.getEncoder().encodeToString(chunk));
                    long t = now();
                    doc.put("created_time", t);
                    doc.put("last_updated_time", t);
                    doc.put("is_hidden", model.get("is_hidden"));
                    PlainActionFuture<IndexResponse> f = PlainActionFuture.newFuture();
                    indexDoc(MlIndices.MODEL, modelId + "_" + i, doc, f);
                    f.actionGet(TimeValue.timeValueMinutes(1));
                }
            }
            Map<String, Object> fields = new LinkedHashMap<>();
            fields.put("model_state", "REGISTERED");
            fields.put("total_chunks", chunks);
            fields.put("model_content_size_in_bytes", size);
            if (hash == null) {
                fields.put("model_content_hash_value", got);
            }
            long t = now();
            fields.put("last_registered_time", t);
            fields.put("last_updated_time", t);
            PlainActionFuture<UpdateResponse> f = PlainActionFuture.newFuture();
            updateDoc(MlIndices.MODEL, modelId, fields, f);
            f.actionGet(TimeValue.timeValueMinutes(1));
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    /** The nodes a local model may run on ({@code MLTaskDispatcher}). */
    private List<DiscoveryNode> eligibleNodes() {
        boolean onlyMl = setting(MlSettings.ML_COMMONS_ONLY_RUN_ON_ML_NODE);
        List<String> roles = setting(MlSettings.ML_COMMONS_LOCAL_MODEL_ELIGIBLE_NODE_ROLES);
        List<DiscoveryNode> nodes = new ArrayList<>();
        for (DiscoveryNode n : clusterService.state().nodes()) {
            boolean ml = n.getRoles().stream().anyMatch(r -> r.roleName().equals("ml"));
            if (onlyMl ? ml : n.getRoles().stream().anyMatch(r -> roles.contains(r.roleName()))) {
                nodes.add(n);
            }
        }
        if (nodes.isEmpty()) {
            throw new IllegalArgumentException(
                "No eligible node found to execute this request. It's best practice to"
                    + " provision ML nodes to serve your models. You can disable this setting to serve the model on your data"
                    + " node for development purposes by disabling the \"plugins.ml_commons.only_run_on_ml_node\" "
                    + "configuration using the _cluster/settings api"
            );
        }
        return nodes;
    }

    /**
     * {@code POST _plugins/_ml/models/<id>/_deploy}: answers with the task id, then deploys on every
     * eligible node (or {@code nodeIds}) and records the outcome in the model and task documents.
     */
    public void deployModel(String modelId, List<String> nodeIds, ActionListener<Map<String, Object>> listener) {
        if (setting(MlSettings.ML_COMMONS_LOCAL_MODEL_ENABLED) == false) {
            listener.onFailure(new IllegalArgumentException(LOCAL_MODEL_DISABLED));
            return;
        }
        getModel(modelId, ActionListener.wrap(model -> {
            List<String> targets = new ArrayList<>();
            List<DiscoveryNode> eligible = eligibleNodes();
            for (DiscoveryNode n : eligible) {
                if (nodeIds == null || nodeIds.contains(n.getId())) {
                    targets.add(n.getId());
                }
            }
            if (targets.isEmpty()) {
                throw new IllegalArgumentException("no eligible node found");
            }
            String function = String.valueOf(model.get("algorithm"));
            createTask("DEPLOY_MODEL", function, modelId, targets, ActionListener.wrap(taskId -> {
                Map<String, Object> response = new LinkedHashMap<>();
                response.put("task_id", taskId);
                response.put("task_type", "DEPLOY_MODEL");
                response.put("status", "CREATED");
                Map<String, Object> fields = new LinkedHashMap<>();
                fields.put("model_state", "DEPLOYING");
                fields.put("planning_worker_node_count", targets.size());
                fields.put("planning_worker_nodes", targets);
                fields.put("deploy_to_all_nodes", nodeIds == null);
                fields.put("last_updated_time", now());
                updateDoc(MlIndices.MODEL, modelId, fields, ActionListener.wrap(u -> {
                    listener.onResponse(response);
                    runDeploy(modelId, taskId, targets);
                }, listener::onFailure));
            }, listener::onFailure));
        }, listener::onFailure));
    }

    private void runDeploy(String modelId, String taskId, List<String> targets) {
        MlNodes.Request request = new MlNodes.Request(MlNodes.Op.DEPLOY, modelId, new byte[0], targets.toArray(String[]::new));
        client.execute(MlNodes.Action.INSTANCE, request, ActionListener.wrap(r -> {
            Map<String, Object> errors = new LinkedHashMap<>();
            int ok = 0;
            for (MlNodes.NodeResponse n : r.getNodes()) {
                if (n.error == null) {
                    ok++;
                } else {
                    errors.put(n.getNode().getId(), n.error);
                }
            }
            r.failures().forEach(f -> errors.put(f.nodeId(), f.getDetailedMessage()));
            String modelState = ok == targets.size() ? "DEPLOYED" : ok == 0 ? "DEPLOY_FAILED" : "PARTIALLY_DEPLOYED";
            Map<String, Object> fields = new LinkedHashMap<>();
            fields.put("model_state", modelState);
            long t = now();
            if (ok > 0) {
                fields.put("last_deployed_time", t);
            }
            fields.put("last_updated_time", t);
            fields.put("current_worker_node_count", ok);
            fields.put("auto_redeploy_retry_times", 0);
            updateDoc(MlIndices.MODEL, modelId, fields, logged("model update"));
            if (errors.isEmpty()) {
                finishTask(taskId, "COMPLETED", Map.of());
            } else {
                finishTask(taskId, ok == 0 ? "FAILED" : "COMPLETED_WITH_ERROR", Map.of("error", json(errors)));
            }
        }, e -> {
            updateDoc(MlIndices.MODEL, modelId, Map.of("model_state", "DEPLOY_FAILED", "last_updated_time", now()), logged("model update"));
            failTask(taskId, e);
        }));
    }

    /** {@code POST _plugins/_ml/models/<id>/_undeploy}: per node, the model's state there. */
    public void undeployModel(String modelId, ActionListener<Map<String, Object>> listener) {
        String[] nodes = clusterService.state().nodes().getNodes().keySet().toArray(String[]::new);
        MlNodes.Request request = new MlNodes.Request(MlNodes.Op.UNDEPLOY, modelId, new byte[0], nodes);
        client.execute(MlNodes.Action.INSTANCE, request, ActionListener.wrap(r -> {
            Map<String, Object> response = new LinkedHashMap<>();
            for (MlNodes.NodeResponse n : r.getNodes()) {
                if (n.payload != null) {
                    String status = new String(n.payload, java.nio.charset.StandardCharsets.UTF_8);
                    response.put(n.getNode().getId(), Map.of("stats", Map.of(modelId, status)));
                }
            }
            Map<String, Object> fields = new LinkedHashMap<>();
            fields.put("model_state", "UNDEPLOYED");
            fields.put("planning_worker_nodes", List.of());
            fields.put("planning_worker_node_count", 0);
            fields.put("last_updated_time", now());
            fields.put("current_worker_node_count", 0);
            getDoc(MlIndices.MODEL, modelId, ActionListener.wrap(g -> {
                if (g.isExists()) {
                    updateDoc(MlIndices.MODEL, modelId, fields, ActionListener.wrap(u -> listener.onResponse(response), listener::onFailure));
                } else {
                    listener.onResponse(response);
                }
            }, listener::onFailure));
        }, listener::onFailure));
    }

    /** {@code DELETE _plugins/_ml/models/<id>}: the model document and its chunks. */
    public void deleteModel(String modelId, ActionListener<DeleteResponse> listener) {
        getModel(modelId, ActionListener.wrap(model -> {
            String state = String.valueOf(model.get("model_state"));
            if (state.equals("DEPLOYED") || state.equals("DEPLOYING") || state.equals("PARTIALLY_DEPLOYED")) {
                throw new OpenSearchStatusException(
                    "Model cannot be deleted in deploying or deployed state. Try undeploy model first then delete",
                    RestStatus.BAD_REQUEST
                );
            }
            DeleteByQueryRequest chunks = new DeleteByQueryRequest(MlIndices.MODEL).setQuery(
                QueryBuilders.boolQuery().must(QueryBuilders.termQuery("model_id", modelId)).must(QueryBuilders.existsQuery("chunk_number"))
            ).setRefresh(true);
            MlIndices.system(client, l -> client.execute(DeleteByQueryAction.INSTANCE, chunks, l), ActionListener.<BulkByScrollResponse>wrap(d -> {
                DeleteRequest request = new DeleteRequest(MlIndices.MODEL, modelId).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                MlIndices.system(client, l -> client.delete(request, l), listener);
            }, listener::onFailure));
        }, listener::onFailure));
    }

    // ---- Prediction ------------------------------------------------------------------------

    /** The texts of a {@code text_docs} predict input. */
    @SuppressWarnings("unchecked")
    static List<String> textDocs(Map<String, Object> body) {
        Object docs = body.get("text_docs");
        if (docs == null && body.get("parameters") instanceof Map<?, ?> p) {
            docs = ((Map<String, Object>) p).get("texts");
        }
        if (docs == null || docs instanceof List<?> l && l.isEmpty()) {
            throw new IllegalArgumentException("Empty text docs");
        }
        if (docs instanceof List<?> list) {
            List<String> texts = new ArrayList<>(list.size());
            for (Object o : list) {
                texts.add(o == null ? null : String.valueOf(o));
            }
            return texts;
        }
        throw new IllegalArgumentException("input data set should not be null");
    }

    /**
     * {@code TextSimilarityMLInput}: the query and the documents of a {@code text_similarity}
     * predict input.
     */
    @SuppressWarnings("unchecked")
    static Map.Entry<String, List<String>> similarityInput(Map<String, Object> body) {
        List<String> docs = new ArrayList<>();
        if (body.get("text_docs") instanceof List<?> list) {
            for (Object o : list) {
                docs.add(String.valueOf(o));
            }
        }
        if (docs.isEmpty()) {
            throw new IllegalArgumentException("No text documents were provided");
        }
        Object query = body.get("query_text");
        if (query == null) {
            throw new IllegalArgumentException("No query text was provided");
        }
        return Map.entry(String.valueOf(query), docs);
    }

    /**
     * {@code POST _plugins/_ml/_predict/<algorithm>/<model_id>} and {@code models/<id>/_predict}: run
     * where the model is deployed, here if possible.
     */
    public void predict(String algorithm, String modelId, Map<String, Object> body, ActionListener<Map<String, Object>> listener) {
        try {
            String route = algorithm == null ? null : algorithm.toUpperCase(Locale.ROOT);
            String function = route != null ? route : local.function(modelId);
            if (function != null) {
                validateInput(function, body);
            }
            if (local.isDeployed(modelId)) {
                threadPool.executor(PREDICT_THREAD_POOL).execute(() -> {
                    try {
                        listener.onResponse(predictLocal(route, modelId, body));
                    } catch (Exception e) {
                        listener.onFailure(e);
                    }
                });
                return;
            }
            getModel(modelId, ActionListener.wrap(model -> {
                if (function == null) {
                    validateInput(String.valueOf(model.get("algorithm")), body);
                }
                List<String> workers = workers(model);
                if (workers.isEmpty()) {
                    throw new IllegalArgumentException("Model not ready yet. Please deploy the model first.");
                }
                String node = workers.get(Math.floorMod(roundRobin.getAndIncrement(), workers.size()));
                Map<String, Object> call = new LinkedHashMap<>();
                call.put("route", route);
                call.put("body", body);
                byte[] payload = json(call).getBytes(java.nio.charset.StandardCharsets.UTF_8);
                MlNodes.Request request = new MlNodes.Request(MlNodes.Op.PREDICT, modelId, payload, node);
                client.execute(MlNodes.PredictAction.INSTANCE, request, ActionListener.wrap(r -> {
                    if (r.getNodes().isEmpty()) {
                        throw new IllegalStateException("prediction on node [" + node + "] failed: " + r.failures());
                    }
                    MlNodes.NodeResponse n = r.getNodes().get(0);
                    if (n.error != null) {
                        throw n.badRequest ? new IllegalArgumentException(n.error) : new MLException(n.error);
                    }
                    listener.onResponse(parse(n.payload));
                }, listener::onFailure));
            }, e -> {
                if (e instanceof OpenSearchStatusException s && s.status() == RestStatus.NOT_FOUND) {
                    // The predict path's own wording (MLModelCacheHelper).
                    listener.onFailure(new OpenSearchStatusException("Failed to find model", RestStatus.NOT_FOUND));
                } else {
                    listener.onFailure(e);
                }
            }));
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }

    @SuppressWarnings("unchecked")
    private List<String> workers(Map<String, Object> model) {
        Object state = model.get("model_state");
        List<String> live = new ArrayList<>();
        if (model.get("planning_worker_nodes") instanceof List<?> nodes && ("DEPLOYED".equals(state) || "PARTIALLY_DEPLOYED".equals(state))) {
            for (String id : (List<String>) nodes) {
                if (clusterService.state().nodes().nodeExists(id)) {
                    live.add(id);
                }
            }
        }
        return live;
    }

    /** Parses {@code body} as the input of {@code function}, as stock's REST layer does before running anything. */
    private static void validateInput(String function, Map<String, Object> body) {
        if ("TEXT_SIMILARITY".equals(function)) {
            similarityInput(body);
        } else {
            textDocs(body);
        }
    }

    /**
     * Runs {@code body}'s prediction with {@code modelId}, deployed on this node. {@code route} is the
     * path's algorithm ({@code null} on {@code models/<id>/_predict}): the input is parsed as that
     * algorithm's, which a model of the other input kind (text docs or query and docs) cannot run.
     */
    private Map<String, Object> predictLocal(String route, String modelId, Map<String, Object> body) {
        String function = local.function(modelId);
        boolean similarity = "TEXT_SIMILARITY".equals(function);
        if (route != null && "TEXT_SIMILARITY".equals(route) != similarity) {
            throw new MLException("Failed to inference " + function + " model: " + modelId);
        }
        if (similarity) {
            return similarity(modelId, similarityInput(body));
        }
        List<String> texts = textDocs(body);
        Map<String, String> params = embeddingParameters(body);
        try {
            if (texts.contains(null)) {
                throw new LocalModels.InferenceException(new NullPointerException("a text of text_docs is null"));
            }
            return switch (function == null ? "" : function) {
                case "SPARSE_TOKENIZE" -> sparseTokenize(modelId, texts, params);
                case "SPARSE_ENCODING" -> sparseEncode(modelId, texts, params, body);
                default -> embedLocal(modelId, prefixed(modelId, texts, params), body);
            };
        } catch (LocalModels.InferenceException e) {
            // The model's own predict failed: stock wraps every such failure the same way.
            throw new MLException("Failed to inference " + function + " model: " + modelId, e.getCause());
        }
    }

    /**
     * {@code AsymmetricTextEmbeddingParameters} of a text-docs input: {@code content_type} and {@code
     * sparse_embedding_format}, each upper-cased and checked; {@code null} when the input has no
     * {@code parameters}.
     */
    private static Map<String, String> embeddingParameters(Map<String, Object> body) {
        if (body.get("parameters") instanceof Map<?, ?> p) {
            Map<String, String> out = new LinkedHashMap<>();
            Object contentType = p.get("content_type");
            if (contentType != null) {
                out.put("content_type", constant(contentType, List.of("QUERY", "PASSAGE"), "AsymmetricTextEmbeddingParameters.EmbeddingContentType"));
            }
            Object format = p.get("sparse_embedding_format");
            if (format != null) {
                out.put("sparse_embedding_format", constant(format, List.of("WORD", "TOKEN_ID"), "SparseEmbeddingFormat"));
            }
            return out;
        }
        return null;
    }

    /** {@code Enum.valueOf} of an upper-cased name, with the JDK's message for an unknown one. */
    private static String constant(Object value, List<String> names, String enumClass) {
        String name = String.valueOf(value).toUpperCase(Locale.ROOT);
        if (names.contains(name) == false) {
            throw new IllegalArgumentException("No enum constant org.opensearch.ml.common.input.parameter.textembedding." + enumClass + "." + name);
        }
        return name;
    }

    /**
     * {@code TextEmbeddingModel.isAsymmetricModel} and {@code addPrefixesToData}: a model registered
     * with a {@code query_prefix} or {@code passage_prefix} needs the content type, and its texts get
     * the prefix of that type.
     */
    @SuppressWarnings("unchecked")
    private List<String> prefixed(String modelId, List<String> texts, Map<String, String> params) {
        Map<String, Object> model = local.model(modelId);
        Map<String, Object> config = model != null && model.get("model_config") instanceof Map<?, ?> c ? (Map<String, Object>) c : Map.of();
        Object queryPrefix = config.get("query_prefix");
        Object passagePrefix = config.get("passage_prefix");
        if (params == null) {
            if (queryPrefix != null || passagePrefix != null) {
                throw new LocalModels.InferenceException(
                    new IllegalArgumentException(
                        "The embedding model chosen is asymmetric. To use it, you must declare whether the input is of type `QUERY` or of type `PASSAGE`."
                    )
                );
            }
            return texts;
        }
        if (queryPrefix == null && passagePrefix == null) {
            throw new LocalModels.InferenceException(
                new IllegalArgumentException(
                    "When passing AsymmetricTextEmbeddingParameters, the model requires to be "
                        + "registered with at least one of `query_prefix` or `passage_prefix`."
                )
            );
        }
        Object prefix = "PASSAGE".equals(params.get("content_type")) ? passagePrefix : queryPrefix;
        if (prefix == null) {
            return texts;
        }
        List<String> out = new ArrayList<>(texts.size());
        for (String t : texts) {
            out.add(prefix + t);
        }
        return out;
    }

    /**
     * {@code ModelTensors.filter}, by the input's {@code result_filter} or its top-level fields: with
     * {@code target_response} or {@code target_response_positions}
     * only the tensors they name (by name or position) stay; {@code return_number: false} drops each
     * tensor's {@code data}, and {@code byte_buffer} stays only with {@code return_bytes: true}.
     */
    @SuppressWarnings("unchecked")
    private static List<Object> filter(List<Map<String, Object>> tensors, Map<String, Object> input) {
        // A result_filter object takes the place of the four top-level fields.
        Map<String, Object> body = input.get("result_filter") instanceof Map<?, ?> f ? (Map<String, Object>) f : input;
        boolean returnBytes = Boolean.parseBoolean(String.valueOf(body.getOrDefault("return_bytes", false)));
        boolean returnNumber = Boolean.parseBoolean(String.valueOf(body.getOrDefault("return_number", true)));
        List<Object> names = body.get("target_response") instanceof List<?> l ? (List<Object>) l : List.of();
        List<Object> positions = body.get("target_response_positions") instanceof List<?> l ? (List<Object>) l : List.of();
        List<Object> out = new ArrayList<>(tensors.size());
        for (int i = 0; i < tensors.size(); i++) {
            Map<String, Object> t = tensors.get(i);
            boolean all = names.isEmpty() && positions.isEmpty();
            boolean named = t.get("name") != null && names.stream().anyMatch(n -> String.valueOf(n).equals(t.get("name")));
            int at = i;
            boolean placed = positions.stream().anyMatch(p -> p instanceof Number n && n.intValue() == at);
            if (all || named || placed) {
                if (returnBytes == false) {
                    t.remove("byte_buffer");
                }
                if (returnNumber == false) {
                    t.remove("data");
                }
                out.add(t);
            }
        }
        return out;
    }

    /**
     * {@code SparseTokenizerModel.predict}: per text one tensor whose {@code dataAsMap} holds the token
     * weights, keyed by token, or by token id with {@code parameters.sparse_embedding_format: TOKEN_ID}.
     */
    private Map<String, Object> sparseTokenize(String modelId, List<String> texts, Map<String, String> params) {
        boolean byId = params != null && "TOKEN_ID".equals(params.get("sparse_embedding_format"));
        List<Object> results = new ArrayList<>(texts.size());
        for (List<LocalModels.TokenWeight> weights : local.sparseTokenize(modelId, texts)) {
            results.add(Map.of("output", List.of(Map.of("dataAsMap", Map.of("response", List.of(weightMap(weights, byId)))))));
        }
        return Map.of("inference_results", results);
    }

    private static Map<String, Object> weightMap(List<LocalModels.TokenWeight> weights, boolean byId) {
        Map<String, Object> map = new LinkedHashMap<>();
        for (LocalModels.TokenWeight w : weights) {
            map.put(byId ? Integer.toUnsignedString(w.id()) : w.token(), w.weight());
        }
        return map;
    }

    /**
     * {@code TextEmbeddingSparseEncodingModel} with its {@code SparseEncodingTranslator}: per text, each
     * model output's non-zero weights as a {@code dataAsMap} tensor (ONNX outputs carry no tensor name
     * there), keyed as for {@link #sparseTokenize}, then filtered as text embeddings are.
     */
    private Map<String, Object> sparseEncode(String modelId, List<String> texts, Map<String, String> params, Map<String, Object> body) {
        boolean byId = params != null && "TOKEN_ID".equals(params.get("sparse_embedding_format"));
        List<Object> results = new ArrayList<>(texts.size());
        for (String text : texts) {
            List<Map<String, Object>> tensors = new ArrayList<>();
            for (List<LocalModels.SparseOutput> outputs : local.sparseEncode(modelId, List.of(text), byId)) {
                for (LocalModels.SparseOutput o : outputs) {
                    Map<String, Object> tensor = new LinkedHashMap<>();
                    tensor.put("dataAsMap", Map.of("response", List.of(weightMap(o.weights(), byId))));
                    tensors.add(tensor);
                }
            }
            results.add(outputs(filter(tensors, body)));
        }
        return Map.of("inference_results", results);
    }

    /** One {@code ModelTensors}: its {@code output} list, left out when the filter kept nothing. */
    private static Map<String, Object> outputs(List<Object> tensors) {
        return tensors.isEmpty() ? Map.of() : Map.of("output", tensors);
    }

    /** {@code ModelTensorOutput} for {@code texts}, one {@code inference_results} entry per text. */
    private Map<String, Object> embedLocal(String modelId, List<String> texts, Map<String, Object> body) {
        List<Object> results = new ArrayList<>(texts.size());
        for (String text : texts) {
            List<Map<String, Object>> tensors = new ArrayList<>();
            for (LocalModels.Tensor t : local.embed(modelId, List.of(text))) {
                Map<String, Object> tensor = tensorMap("sentence_embedding", t);
                tensors.add(tensor);
            }
            results.add(outputs(filter(tensors, body)));
        }
        return Map.of("inference_results", results);
    }

    private static Map<String, Object> tensorMap(String name, LocalModels.Tensor t) {
        Map<String, Object> tensor = new LinkedHashMap<>();
        tensor.put("name", name);
        tensor.put("data_type", "FLOAT32");
        tensor.put("shape", Arrays.stream(t.shape()).boxed().toList());
        List<Float> data = new ArrayList<>(t.data().length);
        for (float f : t.data()) {
            data.add(f);
        }
        tensor.put("data", data);
        return tensor;
    }

    /**
     * {@code TextSimilarityCrossEncoderModel} with its {@code TextSimilarityTranslator}: per document a
     * {@code similarity} tensor of the model's output, with its bytes (little endian, as the native
     * array gives them); this function applies no result filter.
     */
    private Map<String, Object> similarity(String modelId, Map.Entry<String, List<String>> input) {
        List<List<LocalModels.Tensor>> scores;
        try {
            scores = local.similarity(modelId, input.getKey(), input.getValue());
        } catch (LocalModels.InferenceException e) {
            throw new MLException("Failed to inference TEXT_SIMILARITY model: " + modelId, e.getCause());
        }
        List<Object> results = new ArrayList<>(scores.size());
        for (List<LocalModels.Tensor> outputs : scores) {
            List<Object> output = new ArrayList<>(outputs.size());
            for (LocalModels.Tensor t : outputs) {
                Map<String, Object> tensor = tensorMap("similarity", t);
                java.nio.ByteBuffer bytes = java.nio.ByteBuffer.allocate(t.data().length * Float.BYTES).order(java.nio.ByteOrder.LITTLE_ENDIAN);
                for (float f : t.data()) {
                    bytes.putFloat(f);
                }
                Map<String, Object> buffer = new LinkedHashMap<>();
                buffer.put("array", Base64.getEncoder().encodeToString(bytes.array()));
                buffer.put("order", "LITTLE_ENDIAN");
                tensor.put("byte_buffer", buffer);
                output.add(tensor);
            }
            results.add(Map.of("output", output));
        }
        return Map.of("inference_results", results);
    }

    /** The embeddings of {@code texts}, for the neural query, processors and semantic field. */
    public void inferenceSentences(String modelId, List<String> texts, ActionListener<List<List<Float>>> listener) {
        inferenceSentences(modelId, null, texts, listener);
    }

    /**
     * The embeddings of {@code texts} as {@code contentType} ({@code QUERY} or {@code PASSAGE}, for a
     * model registered with prefixes; {@code null} for one without).
     */
    @SuppressWarnings("unchecked")
    public void inferenceSentences(String modelId, String contentType, List<String> texts, ActionListener<List<List<Float>>> listener) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("text_docs", texts);
        if (contentType != null) {
            body.put("parameters", Map.of("content_type", contentType));
        }
        predict("text_embedding", modelId, body, ActionListener.wrap(r -> {
            List<List<Float>> vectors = new ArrayList<>(texts.size());
            for (Object result : (List<Object>) r.get("inference_results")) {
                for (Object tensor : (List<Object>) ((Map<String, Object>) result).get("output")) {
                    List<Float> data = new ArrayList<>();
                    for (Object v : (List<Object>) ((Map<String, Object>) tensor).get("data")) {
                        data.add(((Number) v).floatValue());
                    }
                    vectors.add(data);
                }
            }
            listener.onResponse(vectors);
        }, listener::onFailure));
    }

    /**
     * The sparse vectors of {@code texts} (token to weight), for the neural_sparse query and the
     * sparse_encoding processor, as {@code MLCommonsClientAccessor.inferenceSentencesWithMapResult}
     * returns them.
     */
    @SuppressWarnings("unchecked")
    public void inferenceSentencesWithMapResult(String modelId, List<String> texts, ActionListener<List<Map<String, ?>>> listener) {
        predict("sparse_encoding", modelId, Map.of("text_docs", texts), ActionListener.wrap(r -> {
            List<Map<String, ?>> maps = new ArrayList<>(texts.size());
            for (Object result : (List<Object>) r.get("inference_results")) {
                for (Object tensor : (List<Object>) ((Map<String, Object>) result).get("output")) {
                    Object data = ((Map<String, Object>) tensor).get("dataAsMap");
                    if (data instanceof Map<?, ?> m) {
                        maps.add((Map<String, ?>) m);
                    }
                }
            }
            listener.onResponse(maps);
        }, listener::onFailure));
    }

    /**
     * The relevance score of each of {@code docs} against {@code query} from a cross-encoder, for the
     * rerank processor, as {@code MLCommonsClientAccessor.inferenceSimilarity} returns them.
     */
    @SuppressWarnings("unchecked")
    public void inferenceSimilarity(String modelId, String query, List<String> docs, ActionListener<List<Float>> listener) {
        predict("text_similarity", modelId, Map.of("query_text", query, "text_docs", docs), ActionListener.wrap(r -> {
            List<Float> scores = new ArrayList<>(docs.size());
            for (Object result : (List<Object>) r.get("inference_results")) {
                List<Object> output = (List<Object>) ((Map<String, Object>) result).get("output");
                List<Object> data = (List<Object>) ((Map<String, Object>) output.get(0)).get("data");
                scores.add(((Number) data.get(0)).floatValue());
            }
            listener.onResponse(scores);
        }, listener::onFailure));
    }

    // ---- Node side -------------------------------------------------------------------------

    /** Runs {@code op} for {@code modelId} on this node; the answer is JSON (or a state name). */
    byte[] onNode(MlNodes.Op op, String modelId, byte[] payload) throws Exception {
        switch (op) {
            case DEPLOY -> {
                PlainActionFuture<Void> f = PlainActionFuture.newFuture();
                local.deploy(modelId, f);
                f.actionGet(TimeValue.timeValueMinutes(10));
                return new byte[0];
            }
            case UNDEPLOY -> {
                return (local.undeploy(modelId) ? "undeployed" : "not_found").getBytes(java.nio.charset.StandardCharsets.UTF_8);
            }
            case PREDICT -> {
                Map<String, Object> call = parse(payload);
                @SuppressWarnings("unchecked")
                Map<String, Object> body = (Map<String, Object>) call.get("body");
                Map<String, Object> result = predictLocal((String) call.get("route"), modelId, body);
                return json(result).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            }
            case PROFILE -> {
                return json(Map.of("models", local.deployedIds())).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            }
        }
        throw new IllegalArgumentException("unknown operation " + op);
    }

    /** Deploys here, once the model index is readable, every model the cluster says runs on this node. */
    private ClusterStateListener redeployListener() {
        return new ClusterStateListener() {
            @Override
            public void clusterChanged(ClusterChangedEvent event) {
                if (redeployed.get() || event.state().metadata().hasIndex(MlIndices.MODEL) == false) {
                    return;
                }
                var routing = event.state().routingTable().index(MlIndices.MODEL);
                if (routing == null || routing.allPrimaryShardsActive() == false) {
                    return;
                }
                if (redeployed.compareAndSet(false, true)) {
                    threadPool.executor(DEPLOY_THREAD_POOL).execute(() -> redeployLocal());
                }
            }
        };
    }

    @SuppressWarnings("unchecked")
    private void redeployLocal() {
        String self = clusterService.localNode().getId();
        SearchSourceBuilder source = new SearchSourceBuilder().size(1000)
            .query(
                QueryBuilders.boolQuery()
                    .filter(QueryBuilders.termsQuery("model_state", "DEPLOYED", "PARTIALLY_DEPLOYED"))
                    .mustNot(QueryBuilders.existsQuery("chunk_number"))
            )
            .fetchSource(new String[] { "planning_worker_nodes", "deploy_to_all_nodes", "algorithm", "model_format" }, null);
        searchDocs(new SearchRequest(MlIndices.MODEL).source(source), ActionListener.wrap(r -> {
            for (var hit : r.getHits().getHits()) {
                Map<String, Object> m = hit.getSourceAsMap();
                List<String> planned = (List<String>) m.getOrDefault("planning_worker_nodes", List.of());
                if (planned.contains(self) || Boolean.TRUE.equals(m.get("deploy_to_all_nodes")) && eligibleHere()) {
                    local.deploy(hit.getId(), logged("redeploy of " + hit.getId()));
                }
            }
        }, e -> LOGGER.warn("lucene-rust ml: cannot list deployed models", e)));
    }

    private boolean eligibleHere() {
        try {
            String self = clusterService.localNode().getId();
            return eligibleNodes().stream().anyMatch(n -> n.getId().equals(self));
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    // ---- JSON ------------------------------------------------------------------------------

    static String json(Object value) {
        try (XContentBuilder b = JsonXContent.contentBuilder()) {
            b.value(value);
            return b.toString();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static Map<String, Object> parse(byte[] json) {
        return XContentHelper.convertToMap(new BytesArray(json), true, MediaTypeRegistry.JSON).v2();
    }

    static boolean created(DocWriteResponse r) {
        return r.getResult() == DocWriteResponse.Result.CREATED;
    }
}
