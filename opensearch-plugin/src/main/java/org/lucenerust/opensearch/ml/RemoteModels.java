/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lucenerust.opensearch.NativeBridge;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.DocWriteRequest;
import org.opensearch.action.delete.DeleteRequest;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.IndicesOptions;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Setting;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.index.engine.VersionConflictEngineException;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.script.Script;
import org.opensearch.script.ScriptService;
import org.opensearch.script.ScriptType;
import org.opensearch.script.TemplateScript;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.transport.client.Client;

import java.io.IOException;
import java.net.Inet6Address;
import java.net.InetAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.regex.Pattern;

/**
 * Remote models, as ml-commons runs them: connectors in {@code .plugins-ml-connector} with their
 * credentials encrypted under the cluster's master key ({@code .plugins-ml-config/master_key}, the
 * AWS Encryption SDK message stock writes), and each predict call's requests made, signed and read
 * back in Rust ({@code ml_remote.rs}). This class sends the requests, runs pre- and post-process
 * Painless scripts as template scripts, and retries throttled calls as the connector's {@code
 * client_config} asks.
 */
final class RemoteModels {
    private static final Logger LOGGER = LogManager.getLogger(RemoteModels.class);
    static final List<String> PROTOCOLS = List.of("aws_sigv4", "http", "google_cloud", "mcp_sse", "mcp_streamable_http");
    private static final Set<String> ACTION_TYPES = Set.of("PREDICT", "EXECUTE", "BATCH_PREDICT", "CANCEL_BATCH_PREDICT", "BATCH_PREDICT_STATUS");
    private static final List<String> FIELDS = List.of("name", "version", "description", "protocol", "parameters", "credential", "actions", "client_config");
    private static final List<String> ACTION_FIELDS = List.of(
        "action_type",
        "name",
        "method",
        "url",
        "headers",
        "request_body",
        "pre_process_function",
        "post_process_function"
    );
    private static final DateTimeFormatter AMZ_DATE = DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'", Locale.ROOT);
    static final String DISABLED = "Remote Inference is currently disabled. To enable it, update the setting \"plugins.ml_commons.remote_inference_enabled\" to true.";

    /**
     * A deployed remote model's connection: its HTTP client, how it retries, and for a {@code
     * google_cloud} connector the access token it holds.
     */
    private static final class Connection {
        final HttpClient http;
        /** {@code null}: no read timeout. */
        final Duration readTimeout;
        final int maxRetries;
        final long retryTimeoutMillis;
        final long backoffMillis;
        final String backoffPolicy;
        final boolean privateIp;
        final boolean google;
        /** Guarded by this. */
        String token;
        /** When {@link #token} expires, epoch millis; guarded by this. */
        long expires;

        Connection(
            HttpClient http,
            Duration readTimeout,
            int maxRetries,
            long retryTimeoutMillis,
            long backoffMillis,
            String backoffPolicy,
            boolean privateIp,
            boolean google
        ) {
            this.http = http;
            this.readTimeout = readTimeout;
            this.maxRetries = maxRetries;
            this.retryTimeoutMillis = retryTimeoutMillis;
            this.backoffMillis = backoffMillis;
            this.backoffPolicy = backoffPolicy;
            this.privateIp = privateIp;
            this.google = google;
        }

        HttpClient http() {
            return http;
        }

        /** {@code b} with the read timeout, if there is one. */
        HttpRequest.Builder timed(HttpRequest.Builder b) {
            return readTimeout == null ? b : b.timeout(readTimeout);
        }

        int maxRetries() {
            return maxRetries;
        }

        long backoffMillis() {
            return backoffMillis;
        }

        String backoffPolicy() {
            return backoffPolicy;
        }

        boolean privateIp() {
            return privateIp;
        }
    }

    /** Stock's exception for AWS's throttling header; its name is the error type a caller sees. */
    static final class RemoteConnectorThrottlingException extends OpenSearchStatusException {
        RemoteConnectorThrottlingException(String message, RestStatus status) {
            super(message, status);
        }
    }

    private final Client client;
    private final ClusterService clusterService;
    private final MlIndices indices;
    private final ScriptService scripts;
    private final Map<Long, Connection> connections = new ConcurrentHashMap<>();
    private volatile String masterKey;

    RemoteModels(Client client, ClusterService clusterService, MlIndices indices, ScriptService scripts) {
        this.client = client;
        this.clusterService = clusterService;
        this.indices = indices;
        this.scripts = scripts;
    }

    private <T> T setting(Setting<T> s) {
        return clusterService.getClusterSettings().get(s);
    }

    boolean enabled() {
        return setting(MlSettings.ML_COMMONS_REMOTE_INFERENCE_ENABLED);
    }

    // ---- Master key ------------------------------------------------------------------------

    /** The cluster's master key, created on first use as {@code MLEngine} does. */
    void masterKey(ActionListener<String> listener) {
        String key = masterKey;
        if (key != null) {
            listener.onResponse(key);
            return;
        }
        indices.ensure(MlIndices.CONFIG, ActionListener.wrap(v -> readOrCreateKey(listener, 3), listener::onFailure));
    }

    private void readOrCreateKey(ActionListener<String> listener, int attempts) {
        GetRequest get = new GetRequest(MlIndices.CONFIG, "master_key");
        MlIndices.system(client, l -> client.get(get, l), ActionListener.<GetResponse>wrap(r -> {
            if (r.isExists() && r.getSourceAsMap().get("master_key") instanceof String k) {
                masterKey = k;
                listener.onResponse(k);
                return;
            }
            String generated = NativeBridge.mlMasterKey();
            if (generated == null) {
                throw new IllegalStateException("cannot make a master key: " + NativeBridge.lastError());
            }
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put("master_key", generated);
            doc.put("create_time", System.currentTimeMillis());
            IndexRequest index = new IndexRequest(MlIndices.CONFIG).id("master_key")
                .source(doc)
                .opType(DocWriteRequest.OpType.CREATE)
                .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            MlIndices.system(client, l -> client.index(index, l), ActionListener.<IndexResponse>wrap(w -> {
                masterKey = generated;
                listener.onResponse(generated);
            }, e -> {
                // Another node wrote the key first: read that one.
                if (org.opensearch.ExceptionsHelper.unwrapCause(e) instanceof VersionConflictEngineException && attempts > 1) {
                    readOrCreateKey(listener, attempts - 1);
                } else {
                    listener.onFailure(e);
                }
            }));
        }, listener::onFailure));
    }

    // ---- Connectors ------------------------------------------------------------------------

    /**
     * The connector document {@code body} describes, checked as {@code Connector.createConnector}
     * and the create action check it, credentials still in clear; {@code standalone} for a
     * connector of its own (which must have credentials), not one inside a model.
     */
    @SuppressWarnings("unchecked")
    Map<String, Object> connectorDoc(Map<String, Object> body, boolean standalone) {
        Object protocol = body.get("protocol");
        if (protocol == null) {
            throw new IllegalArgumentException("Connector protocol is null");
        }
        if (PROTOCOLS.contains(String.valueOf(protocol)) == false) {
            throw new IllegalArgumentException("Unsupported connector protocol. Please use one of " + PROTOCOLS);
        }
        boolean google = "google_cloud".equals(protocol);
        if (google && setting(MlSettings.ML_COMMONS_VERTEXAI_CONNECTOR_ENABLED) == false) {
            throw new OpenSearchStatusException(MlSettings.ML_COMMONS_VERTEXAI_CONNECTOR_DISABLED_MESSAGE, RestStatus.FORBIDDEN);
        }
        if (String.valueOf(protocol).startsWith("mcp_")) {
            return mcpDoc(body);
        }
        List<Map<String, Object>> actions = new ArrayList<>();
        if (body.get("actions") instanceof List<?> list) {
            for (Object a : list) {
                if (a instanceof Map<?, ?> m) {
                    actions.add(action((Map<String, Object>) m));
                }
            }
        }
        if (body.get("name") == null) {
            throw new IllegalArgumentException("Connector name is null");
        }
        if (body.get("version") == null) {
            throw new IllegalArgumentException("Connector version is null");
        }
        Object credential = body.get("credential");
        if (standalone && (credential instanceof Map<?, ?> c ? c.isEmpty() : true)) {
            throw new IllegalArgumentException("Connector credential is null or empty list");
        }
        if (actions.isEmpty()) {
            throw new IllegalArgumentException("No actions configured for this connector");
        }
        Map<String, String> parameters = strings(body.get("parameters"));
        if (google) {
            // GoogleCloudConnector.validate: a service-account key, or application default credentials.
            Map<?, ?> c = credential instanceof Map<?, ?> m ? m : null;
            boolean key = c != null && (c.containsKey("private_key") || c.containsKey("client_email"));
            if ("adc".equalsIgnoreCase(parameters.get("auth_mode"))) {
                if (key) {
                    throw new IllegalArgumentException("auth_mode=adc must not include service-account credentials (private_key/client_email)");
                }
            } else if (c == null || c.containsKey("private_key") == false || c.containsKey("client_email") == false) {
                throw new IllegalArgumentException("Missing credential");
            }
        }
        List<Pattern> trusted = setting(MlSettings.ML_COMMONS_TRUSTED_CONNECTOR_ENDPOINTS_REGEX).stream().map(Pattern::compile).toList();
        for (Map<String, Object> a : actions) {
            String url = fill(String.valueOf(a.get("url")), parameters);
            if (trusted.stream().noneMatch(p -> p.matcher(url).matches())) {
                throw new IllegalArgumentException("Connector URL is not matching the trusted connector endpoint regex");
            }
        }
        Map<String, Object> doc = new LinkedHashMap<>();
        for (String f : FIELDS) {
            Object v = switch (f) {
                case "parameters" -> body.get(f) == null ? null : parameters;
                case "credential" -> credential == null ? null : strings(credential);
                case "actions" -> actions;
                case "version", "name", "description", "protocol" -> body.get(f) == null ? null : String.valueOf(body.get(f));
                default -> body.get(f);
            };
            if (v != null) {
                doc.put(f, v);
            }
        }
        return doc;
    }

    /** A connector action as {@code ConnectorAction.parse} reads it. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> action(Map<String, Object> a) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (String f : ACTION_FIELDS) {
            Object v = a.get(f);
            if (v == null) {
                continue;
            }
            if (f.equals("action_type")) {
                String type = String.valueOf(v).toUpperCase(Locale.ROOT);
                if (ACTION_TYPES.contains(type) == false) {
                    throw new IllegalArgumentException("No enum constant org.opensearch.ml.common.connector.ConnectorAction.ActionType." + type);
                }
                v = type;
            } else if (f.equals("headers")) {
                v = strings(v);
            } else if (v instanceof Map<?, ?> || v instanceof List<?>) {
                throw new IllegalStateException("Can't get text on a " + (v instanceof Map<?, ?> ? "START_OBJECT" : "START_ARRAY"));
            } else {
                v = String.valueOf(v);
            }
            out.put(f, v);
        }
        if (out.get("method") == null) {
            throw new IllegalArgumentException("method can't be null");
        }
        if (out.get("url") == null) {
            throw new IllegalArgumentException("url can't be null");
        }
        return out;
    }

    /** The document of an MCP connector: a url, headers and parameters, no actions. */
    private Map<String, Object> mcpDoc(Map<String, Object> body) {
        if (body.get("name") == null) {
            throw new IllegalArgumentException("Connector name is null");
        }
        if (body.get("version") == null) {
            throw new IllegalArgumentException("Connector version is null");
        }
        String url = body.get("url") == null ? null : String.valueOf(body.get("url"));
        if (url == null || url.isBlank()) {
            throw new IllegalArgumentException("MCP Connector url is null or blank");
        }
        if (setting(MlSettings.ML_COMMONS_TRUSTED_CONNECTOR_ENDPOINTS_REGEX).stream().noneMatch(r -> Pattern.compile(r).matcher(url).matches())) {
            throw new IllegalArgumentException("Connector URL is not matching the trusted connector endpoint regex");
        }
        Map<String, Object> doc = new LinkedHashMap<>();
        for (String f : List.of("name", "version", "description", "protocol")) {
            if (body.get(f) != null) {
                doc.put(f, String.valueOf(body.get(f)));
            }
        }
        if (body.get("credential") instanceof Map<?, ?> c && c.isEmpty() == false) {
            doc.put("credential", strings(c));
        }
        if (body.get("client_config") instanceof Map<?, ?> given) {
            Map<String, Object> config = new LinkedHashMap<>();
            config.put("max_connection", 30);
            config.put("connection_timeout", 30);
            config.put("read_timeout", 30);
            config.put("retry_backoff_millis", 200);
            config.put("retry_timeout_seconds", 30);
            config.put("max_retry_times", 0);
            config.put("retry_backoff_policy", "constant");
            config.put("skip_ssl_verification", false);
            config.put("mutual_tls_enabled", false);
            for (Map.Entry<?, ?> e : given.entrySet()) {
                if (config.containsKey(String.valueOf(e.getKey())) && e.getValue() != null) {
                    config.put(String.valueOf(e.getKey()), e.getValue());
                }
            }
            doc.put("client_config", config);
        }
        doc.put("url", url);
        if (body.get("headers") instanceof Map<?, ?> h) {
            doc.put("headers", strings(h));
        }
        doc.put("parameters", strings(body.get("parameters")));
        return doc;
    }

    /** {@code ${parameters.x}} in {@code url} filled from {@code parameters}. */
    private static String fill(String url, Map<String, String> parameters) {
        String out = url;
        for (Map.Entry<String, String> e : parameters.entrySet()) {
            out = out.replace("${parameters." + e.getKey() + "}", e.getValue());
        }
        return out;
    }

    /** A map of strings, a value that is not a string as its JSON, as {@code getParameterMap} keeps it. */
    private static Map<String, String> strings(Object value) {
        Map<String, String> out = new LinkedHashMap<>();
        if (value instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                Object v = e.getValue();
                out.put(String.valueOf(e.getKey()), v == null ? null : v instanceof String s ? s : MlCommons.json(v));
            }
        }
        return out;
    }

    /** {@code doc} with each credential value encrypted under the master key. */
    @SuppressWarnings("unchecked")
    void encrypt(Map<String, Object> doc, ActionListener<Map<String, Object>> listener) {
        if (doc.get("credential") instanceof Map<?, ?> c && c.isEmpty() == false) {
            masterKey(ActionListener.wrap(key -> {
                Map<String, String> encrypted = new LinkedHashMap<>();
                for (Map.Entry<String, String> e : ((Map<String, String>) c).entrySet()) {
                    String v = NativeBridge.mlCredentialEncrypt(key, e.getValue() == null ? "" : e.getValue());
                    if (v == null) {
                        throw new IllegalStateException("cannot encrypt a credential: " + NativeBridge.lastError());
                    }
                    encrypted.put(e.getKey(), v);
                }
                doc.put("credential", encrypted);
                listener.onResponse(doc);
            }, listener::onFailure));
        } else {
            listener.onResponse(doc);
        }
    }

    private static OpenSearchStatusException missing(String id) {
        return new OpenSearchStatusException("Failed to find connector with the provided connector id: " + id, RestStatus.NOT_FOUND);
    }

    /** {@code POST _plugins/_ml/connectors/_create}. */
    void create(Map<String, Object> body, ActionListener<Map<String, Object>> listener) {
        if (enabled() == false) {
            throw new IllegalArgumentException(DISABLED);
        }
        Map<String, Object> doc = connectorDoc(body, true);
        encrypt(doc, ActionListener.wrap(d -> {
            long t = System.currentTimeMillis();
            d.put("created_time", t);
            d.put("last_updated_time", t);
            IndexRequest request = new IndexRequest(MlIndices.CONNECTOR).source(d).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            indices.ensure(MlIndices.CONNECTOR, ActionListener.wrap(v -> MlIndices.system(client, l -> client.index(request, l), ActionListener.<IndexResponse>wrap(r -> {
                listener.onResponse(Map.of("connector_id", r.getId()));
            }, listener::onFailure)), listener::onFailure));
        }, listener::onFailure));
    }

    /** The stored connector {@code id}, credentials as stored. */
    void stored(String id, ActionListener<GetResponse> listener) {
        MlIndices.system(client, l -> client.get(new GetRequest(MlIndices.CONNECTOR, id), l), ActionListener.<GetResponse>wrap(r -> {
            if (r.isExists() == false) {
                throw missing(id);
            }
            listener.onResponse(r);
        }, e -> listener.onFailure(e instanceof org.opensearch.index.IndexNotFoundException ? missing(id) : e)));
    }

    /** Whether MCP connectors may be used. */
    boolean mcpEnabled() {
        return setting(MlSettings.ML_COMMONS_MCP_CONNECTOR_ENABLED);
    }

    /**
     * MCP connector {@code id} ready to connect to: its protocol, url, headers with the
     * credentials filled in, and parameters.
     */
    @SuppressWarnings("unchecked")
    void mcpConnector(String id, ActionListener<Map<String, Object>> listener) {
        stored(id, ActionListener.wrap(r -> masterKey(ActionListener.wrap(key -> {
            Map<String, Object> doc = r.getSourceAsMap();
            Map<String, String> clear = new LinkedHashMap<>();
            if (doc.get("credential") instanceof Map<?, ?> c) {
                for (Map.Entry<?, ?> e : c.entrySet()) {
                    String v = NativeBridge.mlCredentialDecrypt(key, String.valueOf(e.getValue()));
                    if (v == null) {
                        throw new IllegalStateException("cannot decrypt a credential: " + NativeBridge.lastError());
                    }
                    clear.put(String.valueOf(e.getKey()), v);
                }
            }
            Map<String, String> parameters = doc.get("parameters") instanceof Map<?, ?> p ? (Map<String, String>) p : Map.of();
            Map<String, String> headers = new LinkedHashMap<>();
            if (doc.get("headers") instanceof Map<?, ?> h) {
                for (Map.Entry<?, ?> e : h.entrySet()) {
                    String v = String.valueOf(e.getValue());
                    for (Map.Entry<String, String> c : clear.entrySet()) {
                        v = v.replace("${credential." + c.getKey() + "}", c.getValue());
                    }
                    headers.put(String.valueOf(e.getKey()), fill(v, parameters));
                }
            }
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("protocol", doc.get("protocol"));
            out.put("url", doc.get("url"));
            out.put("headers", headers);
            out.put("parameters", parameters);
            out.put("client_config", doc.get("client_config"));
            listener.onResponse(out);
        }, listener::onFailure)), listener::onFailure));
    }

    /** {@code GET _plugins/_ml/connectors/<id>}: the connector without its credentials. */
    void get(String id, ActionListener<Map<String, Object>> listener) {
        stored(id, ActionListener.wrap(r -> {
            Map<String, Object> doc = r.getSourceAsMap();
            doc.remove("credential");
            listener.onResponse(doc);
        }, listener::onFailure));
    }

    /** The ids of the models that use connector {@code id}, deployed ones only when {@code deployed}. */
    private void users(String id, boolean deployed, ActionListener<List<String>> listener) {
        if (clusterService.state().metadata().hasIndex(MlIndices.MODEL) == false) {
            listener.onResponse(List.of());
            return;
        }
        var query = QueryBuilders.boolQuery().filter(QueryBuilders.termQuery("connector_id.keyword", id));
        if (deployed) {
            query.filter(QueryBuilders.termsQuery("model_state", "DEPLOYED", "PARTIALLY_DEPLOYED", "DEPLOYING"));
        }
        SearchRequest request = new SearchRequest(MlIndices.MODEL).source(new SearchSourceBuilder().query(query).fetchSource(false).size(10_000))
            .indicesOptions(IndicesOptions.lenientExpandOpen());
        MlIndices.system(client, l -> client.search(request, l), ActionListener.<SearchResponse>wrap(r -> {
            List<String> ids = new ArrayList<>();
            for (SearchHit h : r.getHits().getHits()) {
                ids.add(h.getId());
            }
            listener.onResponse(ids);
        }, listener::onFailure));
    }

    /** {@code PUT _plugins/_ml/connectors/<id>}: refused while a model using it is deployed. */
    void update(String id, Map<String, Object> body, ActionListener<UpdateResponse> listener) {
        stored(id, ActionListener.wrap(r -> users(id, true, ActionListener.wrap(ids -> {
            if (ids.isEmpty() == false) {
                throw new OpenSearchStatusException(
                    ids.size() + " models are still using this connector, please undeploy the models first: " + ids,
                    RestStatus.BAD_REQUEST
                );
            }
            Map<String, Object> current = r.getSourceAsMap();
            Map<String, Object> merged = new LinkedHashMap<>(current);
            // Without a new credential the stored (encrypted) one stays, and is what a google_cloud
            // connector's key check sees.
            merged.putAll(body);
            Map<String, Object> doc = connectorDoc(merged, false);
            ActionListener<Map<String, Object>> write = ActionListener.wrap(d -> {
                if (body.containsKey("credential") == false && current.get("credential") != null) {
                    d.put("credential", current.get("credential"));
                }
                d.put("created_time", current.get("created_time"));
                d.put("last_updated_time", System.currentTimeMillis());
                UpdateRequest request = new UpdateRequest(MlIndices.CONNECTOR, id).doc(d).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                MlIndices.system(client, l -> client.update(request, l), listener);
            }, listener::onFailure);
            if (body.containsKey("credential")) {
                encrypt(doc, write);
            } else {
                write.onResponse(doc);
            }
        }, listener::onFailure)), listener::onFailure));
    }

    /** {@code DELETE _plugins/_ml/connectors/<id>}: refused while any model uses it. */
    void delete(String id, ActionListener<DeleteResponse> listener) {
        users(id, false, ActionListener.wrap(ids -> {
            if (ids.isEmpty() == false) {
                throw new OpenSearchStatusException(
                    ids.size() + " models are still using this connector, please delete or update the models first: " + ids,
                    RestStatus.CONFLICT
                );
            }
            DeleteRequest request = new DeleteRequest(MlIndices.CONNECTOR, id).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            MlIndices.system(client, l -> client.delete(request, l), ActionListener.<DeleteResponse>wrap(listener::onResponse, e -> {
                listener.onFailure(e instanceof org.opensearch.index.IndexNotFoundException ? missing(id) : e);
            }));
        }, listener::onFailure));
    }

    // ---- Deployed models -------------------------------------------------------------------

    /** Opens the connector of remote model document {@code model} in Rust; the handle. */
    @SuppressWarnings("unchecked")
    void open(Map<String, Object> model, ActionListener<Long> listener) {
        ActionListener<Map<String, Object>> withConnector = ActionListener.wrap(connector -> masterKey(ActionListener.wrap(key -> {
            Connection connection = connection(connector.get("client_config"), "google_cloud".equals(connector.get("protocol")));
            long[] handle = new long[1];
            if (NativeBridge.mlRemoteOpen(MlCommons.json(connector), key, handle) != NativeBridge.OK) {
                connection.http().shutdownNow();
                throw new IllegalArgumentException(NativeBridge.lastError());
            }
            connections.put(handle[0], connection);
            listener.onResponse(handle[0]);
        }, listener::onFailure)), listener::onFailure);
        if (model.get("connector") instanceof Map<?, ?> inline) {
            withConnector.onResponse((Map<String, Object>) inline);
        } else if (model.get("connector_id") instanceof String id) {
            stored(id, ActionListener.wrap(r -> withConnector.onResponse(r.getSourceAsMap()), e -> {
                listener.onFailure(e instanceof OpenSearchStatusException ? new OpenSearchStatusException("Failed to find connector:" + id, RestStatus.NOT_FOUND) : e);
            }));
        } else {
            listener.onFailure(new IllegalArgumentException("Model has no connector"));
        }
    }

    /**
     * {@code ConnectorClientConfig}'s settings, with its defaults: timeouts in seconds (one of 0
     * or less is none), retries until {@code max_retry_times} ({@code -1}: no limit) or {@code
     * retry_timeout_seconds} has passed.
     */
    private Connection connection(Object config, boolean google) {
        Map<?, ?> c = config instanceof Map<?, ?> m ? m : Map.of();
        long connect = number(c.get("connection_timeout"), 30);
        long read = number(c.get("read_timeout"), 30);
        int retries = (int) number(c.get("max_retry_times"), 0);
        long retryTimeout = number(c.get("retry_timeout_seconds"), 30);
        long backoff = number(c.get("retry_backoff_millis"), 200);
        String policy = c.get("retry_backoff_policy") == null ? "constant" : String.valueOf(c.get("retry_backoff_policy")).toLowerCase(Locale.ROOT);
        HttpClient.Builder http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).followRedirects(HttpClient.Redirect.NEVER);
        if (connect > 0) {
            http.connectTimeout(Duration.ofSeconds(connect));
        }
        return new Connection(http.build(), read > 0 ? Duration.ofSeconds(read) : null, retries < 0 ? Integer.MAX_VALUE : retries,
            Math.max(0, retryTimeout) * 1000, backoff, policy, setting(MlSettings.ML_COMMONS_CONNECTOR_PRIVATE_IP_ENABLED), google);
    }

    private static long number(Object v, long otherwise) {
        if (v instanceof Number n) {
            return n.longValue();
        }
        if (v instanceof String s && s.isBlank() == false) {
            return Long.parseLong(s.trim());
        }
        return otherwise;
    }

    /** Forgets remote model {@code handle}, closed. */
    void closed(long handle) {
        Connection c = connections.remove(handle);
        if (c != null) {
            // Calls in flight fail now rather than holding the undeploy.
            c.http().shutdownNow();
        }
    }

    /**
     * Runs a predict call of remote model {@code handle}: {@code route} is the path's algorithm
     * ({@code null} on {@code models/<id>/_predict}, a remote model's own parameters), each split
     * input one request; the answer as {@code MLPredictionOutput} writes it.
     */
    Map<String, Object> predict(long handle, String route, Map<String, Object> body) throws Exception {
        return predict(handle, route, "predict", body);
    }

    /** {@link #predict(long, String, Map)} for the connector action named {@code action}. */
    @SuppressWarnings("unchecked")
    Map<String, Object> predict(long handle, String route, String action, Map<String, Object> body) throws Exception {
        Connection connection = connections.get(handle);
        if (connection == null) {
            throw new IllegalStateException("the remote model is not deployed here");
        }
        String algorithm = route == null ? "REMOTE" : route;
        Map<String, Object> input = switch (algorithm) {
            case "REMOTE" -> Map.of("parameters", body.get("parameters") instanceof Map<?, ?> p ? p : Map.of());
            case "TEXT_SIMILARITY" -> {
                Map.Entry<String, List<String>> s = MlCommons.similarityInput(body);
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("query_text", s.getKey());
                m.put("text_docs", s.getValue());
                yield m;
            }
            default -> Map.of("text_docs", MlCommons.textDocs(body));
        };
        Map<String, Object> split = call(handle, Map.of("op", "split", "action", action, "input", input));
        List<Object> parts = (List<Object>) split.get("inputs");
        List<Object> results = new ArrayList<>();
        if (parts.size() == 1) {
            results.add(request(handle, connection, algorithm, action, (Map<String, Object>) parts.get(0)));
            return Map.of("inference_results", results);
        }
        // Each part is its own request, sent at once as stock does; virtual threads, since each waits on HTTP.
        try (java.util.concurrent.ExecutorService parallel = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()) {
            List<java.util.concurrent.Future<Map<String, Object>>> pending = new ArrayList<>();
            for (Object part : parts) {
                pending.add(parallel.submit(() -> request(handle, connection, algorithm, action, (Map<String, Object>) part)));
            }
            for (java.util.concurrent.Future<Map<String, Object>> f : pending) {
                try {
                    results.add(f.get());
                } catch (java.util.concurrent.ExecutionException e) {
                    throw e.getCause() instanceof Exception cause ? cause : new IllegalStateException(e.getCause());
                }
            }
        }
        return Map.of("inference_results", results);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> request(long handle, Connection connection, String algorithm, String action, Map<String, Object> input) throws Exception {
        Map<String, Object> prepare = new LinkedHashMap<>();
        prepare.put("op", "prepare");
        prepare.put("action", action);
        prepare.put("algorithm", algorithm);
        prepare.put("input", input);
        prepare.put("amz_date", ZonedDateTime.now(ZoneOffset.UTC).format(AMZ_DATE));
        Map<String, Object> prepared = call(handle, prepare);
        if (prepared.get("script") instanceof Map<?, ?> script) {
            prepare.put("script_output", run((Map<String, Object>) script));
            prepared = call(handle, prepare);
        }
        Map<String, Object> request = (Map<String, Object>) prepared.get("request");
        if (connection.google) {
            authorize(request, "Bearer " + googleToken(handle, connection));
        }
        HttpResponse<String> response = send(connection, request);
        int status = response.statusCode();
        if (status < 200 || status > 300) {
            RestStatus rest = RestStatus.fromCode(status);
            if (rest == null) {
                rest = RestStatus.INTERNAL_SERVER_ERROR;
            }
            if (throttled(response)) {
                throw new RemoteConnectorThrottlingException(
                    "Error from remote service: The request was denied due to remote server throttling. "
                        + "To change the retry policy and behavior, please update the connector client_config.",
                    rest
                );
            }
            throw new OpenSearchStatusException("Error from remote service: " + response.body(), rest);
        }
        Map<String, Object> parse = new LinkedHashMap<>();
        parse.put("op", "parse");
        parse.put("action", action);
        parse.put("parameters", prepared.get("parameters"));
        parse.put("body", response.body());
        Map<String, Object> parsed = call(handle, parse);
        if (parsed.get("script") instanceof Map<?, ?> script) {
            parse.put("script_output", run((Map<String, Object>) script));
            parsed = call(handle, parse);
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("output", parsed.get("output"));
        result.put("status_code", status);
        return result;
    }

    /** {@code request}'s {@code Authorization} header set to {@code value}, as {@code GoogleCloudConnector} sets it. */
    @SuppressWarnings("unchecked")
    private static void authorize(Map<String, Object> request, String value) {
        List<Object> headers = new ArrayList<>();
        for (Object h : (List<Object>) request.get("headers")) {
            if ("authorization".equalsIgnoreCase(String.valueOf(((List<Object>) h).get(0))) == false) {
                headers.add(h);
            }
        }
        headers.add(List.of("Authorization", value));
        request.put("headers", headers);
    }

    private static final String NO_ADC = "Your default credentials were not found. To set up Application Default Credentials for your "
        + "environment, see https://cloud.google.com/docs/authentication/external/set-up-adc.";

    /**
     * The access token of a {@code google_cloud} connector, signed in again once it is within five
     * minutes of expiring: with the connector's service-account key, or with application default
     * credentials (a {@code GOOGLE_APPLICATION_CREDENTIALS} key file, else the metadata server).
     */
    @SuppressWarnings("unchecked")
    private String googleToken(long handle, Connection c) throws Exception {
        synchronized (c) {
            long now = System.currentTimeMillis();
            if (c.token != null && now < c.expires - 300_000) {
                return c.token;
            }
            Map<String, Object> ask = new LinkedHashMap<>();
            ask.put("op", "google_token_request");
            ask.put("now", now / 1000);
            Map<String, Object> answer = signIn(handle, ask, "service-account key");
            if (Boolean.TRUE.equals(answer.get("adc"))) {
                String file = System.getenv("GOOGLE_APPLICATION_CREDENTIALS");
                if (file != null && file.isBlank() == false) {
                    Map<String, Object> key;
                    try {
                        key = MlCommons.parse(java.nio.file.Files.readAllBytes(java.nio.file.Path.of(file)));
                    } catch (IOException e) {
                        throw new MLException("Failed to initialize Google credentials (ADC/Workload Identity): " + e.getMessage(), e);
                    }
                    ask.put("key", key);
                    answer = signIn(handle, ask, "ADC/Workload Identity");
                } else {
                    answer = metadataToken(c, String.valueOf(answer.get("scope")));
                }
            }
            if (answer.get("body") != null) {
                answer = exchange(c, answer);
            }
            c.token = String.valueOf(answer.get("access_token"));
            c.expires = now + ((Number) answer.getOrDefault("expires_in", 3600)).longValue() * 1000;
            return c.token;
        }
    }

    private static Map<String, Object> signIn(long handle, Map<String, Object> ask, String mode) {
        try {
            return call(handle, ask);
        } catch (MLException e) {
            throw new MLException("Failed to initialize Google credentials (" + mode + "): " + e.getMessage(), e);
        }
    }

    /** Trades a signed assertion for an access token at the token endpoint. */
    private static Map<String, Object> exchange(Connection c, Map<String, Object> signed) throws Exception {
        HttpRequest r = c.timed(HttpRequest.newBuilder(URI.create(String.valueOf(signed.get("token_uri")))))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .POST(HttpRequest.BodyPublishers.ofString(String.valueOf(signed.get("body"))))
            .build();
        // A service account's failures name it; an authorized user's (no client_email) do not.
        Object email = signed.get("client_email");
        String failed = email == null
            ? "Failed to obtain Google OAuth2 access token: "
            : "Failed to obtain Google OAuth2 access token: Error getting access token for service account: ";
        String iss = email == null ? "" : ", iss: " + email;
        HttpResponse<String> response;
        try {
            response = c.http().send(r, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new MLException(failed + e.getMessage() + iss, e);
        }
        if (response.statusCode() != 200) {
            throw new MLException(failed + response.statusCode() + " " + response.body() + iss);
        }
        return token(response.body(), failed);
    }

    /** An access token from the metadata server, for application default credentials on Google Cloud. */
    private static Map<String, Object> metadataToken(Connection c, String scope) throws Exception {
        String host = System.getenv("GCE_METADATA_HOST");
        URI uri = URI.create(
            "http://" + (host == null || host.isBlank() ? "metadata.google.internal" : host)
                + "/computeMetadata/v1/instance/service-accounts/default/token?scopes="
                + java.net.URLEncoder.encode(scope, StandardCharsets.UTF_8)
        );
        HttpRequest r = HttpRequest.newBuilder(uri).timeout(Duration.ofSeconds(2)).header("Metadata-Flavor", "Google").GET().build();
        HttpResponse<String> response;
        try {
            response = c.http().send(r, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new MLException("Failed to initialize Google credentials (ADC/Workload Identity): " + NO_ADC, e);
        }
        String failed = "Failed to obtain Google OAuth2 access token: ";
        if (response.statusCode() != 200) {
            throw new MLException(failed + "Unexpected Error code " + response.statusCode() + " trying to get security access token from Compute Engine metadata for the default service account: " + response.body());
        }
        return token(response.body(), failed);
    }

    private static Map<String, Object> token(String body, String failed) {
        Map<String, Object> t;
        try {
            t = MlCommons.parse(body.getBytes(StandardCharsets.UTF_8));
        } catch (RuntimeException e) {
            throw new MLException(failed + "the token response is not JSON", e);
        }
        if (t.get("access_token") instanceof String == false) {
            throw new MLException(failed + "the token response has no access_token");
        }
        return t;
    }

    /** One step of a remote call in Rust. */
    private static Map<String, Object> call(long handle, Map<String, Object> request) {
        int[] status = new int[1];
        String answer = NativeBridge.mlRemoteCall(handle, MlCommons.json(request), status);
        if (answer == null) {
            String why = NativeBridge.lastError();
            throw status[0] == NativeBridge.INVALID_ARGUMENT ? new IllegalArgumentException(why) : new MLException(why);
        }
        return MlCommons.parse(answer.getBytes(StandardCharsets.UTF_8));
    }

    /** Runs a pre- or post-process script as {@code MLConnectorUtils.executeScript} does. */
    @SuppressWarnings("unchecked")
    private String run(Map<String, Object> script) {
        Map<String, Object> params = (Map<String, Object>) doubles(MlCommons.parse(String.valueOf(script.get("params")).getBytes(StandardCharsets.UTF_8)));
        Script s = new Script(ScriptType.INLINE, "painless", String.valueOf(script.get("source")), Collections.emptyMap());
        TemplateScript t = scripts.compile(s, TemplateScript.CONTEXT).newInstance(params);
        return t.execute();
    }

    /** {@code value} with every number a double, as Gson reads JSON into a map. */
    private static Object doubles(Object value) {
        if (value instanceof Number n) {
            return n.doubleValue();
        }
        if (value instanceof Map<?, ?> m) {
            Map<String, Object> out = new LinkedHashMap<>();
            m.forEach((k, v) -> out.put(String.valueOf(k), doubles(v)));
            return out;
        }
        if (value instanceof List<?> l) {
            List<Object> out = new ArrayList<>(l.size());
            for (Object o : l) {
                out.add(doubles(o));
            }
            return out;
        }
        return value;
    }

    /** Headers {@code java.net.http} sets itself. */
    private static final Set<String> RESTRICTED = Set.of("content-length", "host", "connection", "expect", "upgrade");

    @SuppressWarnings("unchecked")
    private HttpResponse<String> send(Connection connection, Map<String, Object> request) throws Exception {
        URI uri = URI.create(String.valueOf(request.get("url")));
        if (connection.privateIp() == false) {
            checkPublic(uri.getHost());
        }
        String method = String.valueOf(request.get("method"));
        String body = String.valueOf(request.getOrDefault("body", ""));
        HttpRequest.Builder b = connection.timed(HttpRequest.newBuilder(uri));
        for (Object h : (List<Object>) request.get("headers")) {
            List<Object> kv = (List<Object>) h;
            String name = String.valueOf(kv.get(0));
            if (RESTRICTED.contains(name.toLowerCase(Locale.ROOT)) == false) {
                b.header(name, String.valueOf(kv.get(1)));
            }
        }
        HttpRequest.BodyPublisher publisher = method.equals("GET") ? HttpRequest.BodyPublishers.noBody() : HttpRequest.BodyPublishers.ofString(body);
        HttpRequest r = b.method(method, publisher).build();
        long start = System.nanoTime();
        for (int attempt = 0;; attempt++) {
            HttpResponse<String> response;
            try {
                response = connection.http().send(r, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            } catch (java.net.http.HttpTimeoutException e) {
                throw new OpenSearchStatusException("Error communicating with remote model: Read timed out", RestStatus.INTERNAL_SERVER_ERROR, e);
            } catch (IOException e) {
                throw new OpenSearchStatusException("Error communicating with remote model: " + e.getMessage(), RestStatus.INTERNAL_SERVER_ERROR, e);
            }
            long elapsed = (System.nanoTime() - start) / 1_000_000;
            if (retryable(response) == false || attempt >= connection.maxRetries() || elapsed >= connection.retryTimeoutMillis) {
                return response;
            }
            Thread.sleep(backoff(connection, attempt));
        }
    }

    /** {@code RetryableActionExtension.isRetryable}: 429, any 5xx, or AWS's throttling header. */
    private static boolean retryable(HttpResponse<String> response) {
        int status = response.statusCode();
        return status == 429 || status >= 500 && RestStatus.fromCode(status) != null || throttled(response);
    }

    /** Whether a failed answer carries AWS's {@code x-amzn-ErrorType: ThrottlingException...}. */
    private static boolean throttled(HttpResponse<String> response) {
        int status = response.statusCode();
        return (status < 200 || status > 300)
            && response.headers().allValues("x-amzn-ErrorType").stream().anyMatch(v -> v.startsWith("ThrottlingException"));
    }

    /** The wait before retry {@code attempt + 1}, by the connector's backoff policy. */
    private static long backoff(Connection c, int attempt) {
        long base = c.backoffMillis();
        long exponential = base * (1L << Math.min(attempt, 20));
        return switch (c.backoffPolicy()) {
            case "exponential_equal_jitter" -> exponential / 2 + ThreadLocalRandom.current().nextLong(exponential / 2 + 1);
            case "exponential_full_jitter" -> ThreadLocalRandom.current().nextLong(exponential + 1);
            default -> base;
        };
    }

    /** Refuses a host that resolves to a private address, as {@code MLHttpClientFactory} does. */
    private static void checkPublic(String host) throws IOException {
        for (InetAddress a : InetAddress.getAllByName(host)) {
            boolean uniqueLocal = a instanceof Inet6Address && (a.getAddress()[0] & 0xfe) == 0xfc;
            if (a.isLoopbackAddress() || a.isSiteLocalAddress() || a.isLinkLocalAddress() || a.isAnyLocalAddress() || uniqueLocal) {
                throw new IllegalArgumentException("Remote inference host name has private ip address: " + host);
            }
        }
    }
}
