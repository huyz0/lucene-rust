package org.lucenerust.opensearch.ml;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.OpenSearchException;
import org.opensearch.core.action.ActionListener;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * MCP servers as agents use them: the tools of the connectors named by an agent's {@code
 * mcp_connectors} parameter, listed and called over the connector's protocol (streamable HTTP or
 * SSE).
 */
final class MlMcp {
    static final String HTTP_TOOL = "McpStreamableHttpTool";
    static final String SSE_TOOL = "McpSseTool";

    private static final Logger LOG = LogManager.getLogger(MlMcp.class);
    private static final String PROTOCOL_VERSION = "2025-11-25";
    private static final String SSE_PROTOCOL_VERSION = "2024-11-05";
    private static final Duration REPLY_TIMEOUT = Duration.ofSeconds(30);
    private static final String HTTP_DESCRIPTION = "A tool from MCP Streamable HTTP Server";
    private static final String SSE_DESCRIPTION = "A tool from MCP Server";

    private MlMcp() {}

    /** Whether {@code type} is one of the MCP tool types. */
    static boolean isTool(String type) {
        return HTTP_TOOL.equals(type) || SSE_TOOL.equals(type);
    }

    /** A tool of an MCP server, with the session it is called through. */
    record Tool(String name, String type, String description, String schema, Session session) {
        /** The tool as {@code GET _plugins/_ml/connectors/<id>/tools} lists it. */
        Map<String, Object> listing() {
            Map<String, Object> out = new LinkedHashMap<>();
            out.put("name", name);
            out.put("type", type);
            out.put("description", description);
            out.put("input_schema", schema);
            return out;
        }

        /** Calls the tool with {@code arguments}: the JSON of the content it answers. */
        String call(Map<String, Object> arguments) throws Exception {
            Map<String, Object> params = new LinkedHashMap<>();
            params.put("name", name);
            params.put("arguments", arguments);
            Object result = session.request("tools/call", params);
            List<Object> content = new ArrayList<>();
            if (result instanceof Map<?, ?> r && r.get("content") instanceof List<?> items) {
                for (Object item : items) {
                    if (item instanceof Map<?, ?> m) {
                        Map<String, Object> kept = new LinkedHashMap<>();
                        m.forEach((k, v) -> {
                            if ("type".equals(k) == false && v != null) {
                                kept.put(String.valueOf(k), v);
                            }
                        });
                        content.add(kept);
                    } else {
                        content.add(item);
                    }
                }
            }
            return MlCommons.json(content);
        }
    }

    /** A connection to one MCP server: JSON-RPC requests and notifications. */
    abstract static class Session implements AutoCloseable {
        private final AtomicLong next = new AtomicLong();
        private final String prefix = UUID.randomUUID().toString().substring(0, 8);

        final String id() {
            return prefix + "-" + next.getAndIncrement();
        }

        /** The {@code result} of the answer to {@code method}. */
        abstract Object request(String method, Object params) throws Exception;

        abstract void notify(String method) throws Exception;

        @Override
        public void close() {}

        /** The handshake every session starts with. */
        final void initialize(String version) throws Exception {
            Map<String, Object> roots = new LinkedHashMap<>();
            roots.put("listChanged", false);
            Map<String, Object> capabilities = new LinkedHashMap<>();
            capabilities.put("roots", roots);
            Map<String, Object> client = new LinkedHashMap<>();
            client.put("name", "Java SDK MCP Client");
            client.put("version", "0.15.0");
            Map<String, Object> params = new LinkedHashMap<>();
            params.put("protocolVersion", version);
            params.put("capabilities", capabilities);
            params.put("clientInfo", client);
            request("initialize", params);
            notify("notifications/initialized");
        }

        final List<Tool> tools(String type) throws Exception {
            Object result = request("tools/list", Map.of());
            List<Tool> out = new ArrayList<>();
            if (result instanceof Map<?, ?> r && r.get("tools") instanceof List<?> list) {
                for (Object o : list) {
                    if (o instanceof Map<?, ?> t) {
                        Object description = t.get("description");
                        Object schema = t.get("inputSchema");
                        out.add(
                            new Tool(
                                String.valueOf(t.get("name")),
                                type,
                                description == null ? (HTTP_TOOL.equals(type) ? HTTP_DESCRIPTION : SSE_DESCRIPTION) : String.valueOf(description),
                                schema == null ? "{}" : MlCommons.json(schema),
                                this
                            )
                        );
                    }
                }
            }
            return out;
        }

        static Object answer(Object message) {
            if (message instanceof Map<?, ?> m && m.get("error") instanceof Map<?, ?> e) {
                throw new MLException(e.get("message") == null ? MlCommons.json(e) : String.valueOf(e.get("message")));
            }
            return message instanceof Map<?, ?> m ? m.get("result") : null;
        }
    }

    private static Map<String, Object> message(String id, String method, Object params) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("jsonrpc", "2.0");
        body.put("method", method);
        if (id != null) {
            body.put("id", id);
        }
        if (params != null) {
            body.put("params", params);
        }
        return body;
    }

    private static HttpRequest.Builder builder(URI uri, Map<String, String> headers) {
        HttpRequest.Builder b = HttpRequest.newBuilder(uri).timeout(REPLY_TIMEOUT);
        headers.forEach(b::header);
        return b;
    }

    /** The refusal of text the JSON parser cannot read; named as the parser's own exception, as the answer shows it. */
    static final class StreamReadException extends IllegalArgumentException {
        StreamReadException(String message) {
            super(message);
        }
    }

    /** The tool arguments in {@code input}, a JSON object; text that is not JSON is refused as the JSON parser refuses it. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> arguments(String input) {
        Object parsed;
        try {
            parsed = MlCommons.parse(input.getBytes(StandardCharsets.UTF_8));
        } catch (RuntimeException e) {
            java.util.regex.Matcher token = Pattern.compile("^\\s*([A-Za-z0-9_]+)").matcher(input);
            throw new StreamReadException(
                "Unrecognized token '" + (token.find() ? token.group(1) : input.strip())
                    + "': was expecting (JSON String, Number, Array, Object or token 'null', 'true' or 'false')\n"
                    + " at [Source: REDACTED (`StreamReadFeature.INCLUDE_SOURCE_IN_LOCATION` disabled); byte offset: #UNKNOWN]"
            );
        }
        return parsed instanceof Map<?, ?> m ? (Map<String, Object>) m : new LinkedHashMap<>();
    }

    // ---- Streamable HTTP ----------------------------------------------------------------------

    /** The streamable HTTP transport: each request a POST, answered as JSON or as an event stream. */
    static final class Http extends Session {
        private final HttpClient http;
        private final URI uri;
        private final Map<String, String> headers;
        private volatile String session;

        Http(HttpClient http, URI uri, Map<String, String> headers) {
            this.http = http;
            this.uri = uri;
            this.headers = headers;
        }

        private HttpResponse<String> post(Map<String, Object> body) throws IOException, InterruptedException {
            HttpRequest.Builder b = builder(uri, headers).header("Accept", "application/json, text/event-stream")
                .header("Content-Type", "application/json");
            String s = session;
            if (s != null) {
                b.header("Mcp-Session-Id", s).header("MCP-Protocol-Version", PROTOCOL_VERSION);
            }
            HttpResponse<String> r = http.send(
                b.POST(HttpRequest.BodyPublishers.ofString(MlCommons.json(body))).build(),
                HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
            );
            if (r.statusCode() >= 400) {
                throw new MLException("MCP server answered HTTP " + r.statusCode());
            }
            r.headers().firstValue("Mcp-Session-Id").ifPresent(v -> session = v);
            return r;
        }

        @Override
        Object request(String method, Object params) throws Exception {
            String id = id();
            HttpResponse<String> r = post(message(id, method, params));
            String type = r.headers().firstValue("Content-Type").orElse("");
            if (type.contains("text/event-stream")) {
                for (String data : events(r.body())) {
                    Object parsed = MlCommons.parse(data.getBytes(StandardCharsets.UTF_8));
                    if (parsed instanceof Map<?, ?> m && id.equals(m.get("id"))) {
                        return answer(parsed);
                    }
                }
                throw new MLException("MCP server sent no answer to " + method);
            }
            return answer(MlCommons.parse(r.body().getBytes(StandardCharsets.UTF_8)));
        }

        @Override
        void notify(String method) throws Exception {
            post(message(null, method, null));
        }

        @Override
        public void close() {
            http.shutdownNow();
        }
    }

    /** The data of each event of server-sent {@code body}. */
    static List<String> events(String body) {
        List<String> out = new ArrayList<>();
        StringBuilder data = new StringBuilder();
        for (String line : body.split("\r?\n", -1)) {
            if (line.isEmpty()) {
                if (data.length() > 0) {
                    out.add(data.toString());
                    data.setLength(0);
                }
            } else if (line.startsWith("data:")) {
                if (data.length() > 0) {
                    data.append('\n');
                }
                data.append(line.substring(5).stripLeading());
            }
        }
        if (data.length() > 0) {
            out.add(data.toString());
        }
        return out;
    }

    // ---- SSE ----------------------------------------------------------------------------------

    /** The SSE transport: answers come on the event stream the session opened, requests go by POST. */
    static final class Sse extends Session {
        private final HttpClient http;
        private final Map<String, String> headers;
        private final Map<String, CompletableFuture<Object>> pending = new ConcurrentHashMap<>();
        private final CompletableFuture<URI> endpoint = new CompletableFuture<>();
        private volatile HttpResponse<Stream<String>> stream;

        Sse(HttpClient http, URI uri, Map<String, String> headers) {
            this.http = http;
            this.headers = headers;
            HttpRequest.Builder b = HttpRequest.newBuilder(uri).header("Accept", "text/event-stream").header("Cache-Control", "no-cache");
            headers.forEach(b::header);
            http.sendAsync(b.GET().build(), HttpResponse.BodyHandlers.ofLines()).whenComplete((r, e) -> {
                if (e != null) {
                    endpoint.completeExceptionally(e);
                    pending.values().forEach(f -> f.completeExceptionally(e));
                    return;
                }
                stream = r;
                if (r.statusCode() >= 400) {
                    endpoint.completeExceptionally(new MLException("MCP server answered HTTP " + r.statusCode()));
                    return;
                }
                read(r.body(), uri);
            });
        }

        private void read(Stream<String> lines, URI base) {
            String[] event = { "message" };
            StringBuilder data = new StringBuilder();
            try {
                lines.forEach(line -> {
                    if (line.isEmpty()) {
                        if (data.length() > 0) {
                            dispatch(event[0], data.toString(), base);
                        }
                        event[0] = "message";
                        data.setLength(0);
                    } else if (line.startsWith("event:")) {
                        event[0] = line.substring(6).strip();
                    } else if (line.startsWith("data:")) {
                        if (data.length() > 0) {
                            data.append('\n');
                        }
                        data.append(line.substring(5).stripLeading());
                    }
                });
            } catch (RuntimeException e) {
                LOG.debug("MCP event stream ended", e);
            } finally {
                MLException ended = new MLException("The MCP event stream ended");
                endpoint.completeExceptionally(ended);
                pending.values().forEach(f -> f.completeExceptionally(ended));
            }
        }

        private void dispatch(String event, String data, URI base) {
            if (event.equals("endpoint")) {
                endpoint.complete(base.resolve(data));
                return;
            }
            Object parsed = MlCommons.parse(data.getBytes(StandardCharsets.UTF_8));
            if (parsed instanceof Map<?, ?> m && m.get("id") != null) {
                CompletableFuture<Object> f = pending.remove(String.valueOf(m.get("id")));
                if (f != null) {
                    f.complete(parsed);
                }
            }
        }

        private void post(Map<String, Object> body) throws Exception {
            URI target = endpoint.get(REPLY_TIMEOUT.toSeconds(), TimeUnit.SECONDS);
            HttpRequest.Builder b = builder(target, headers).header("Content-Type", "application/json");
            HttpResponse<String> r = http.send(
                b.POST(HttpRequest.BodyPublishers.ofString(MlCommons.json(body))).build(),
                HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
            );
            if (r.statusCode() >= 400) {
                throw new MLException("MCP server answered HTTP " + r.statusCode());
            }
        }

        @Override
        Object request(String method, Object params) throws Exception {
            String id = id();
            CompletableFuture<Object> reply = new CompletableFuture<>();
            pending.put(id, reply);
            try {
                post(message(id, method, params));
                return answer(reply.get(REPLY_TIMEOUT.toSeconds(), TimeUnit.SECONDS));
            } finally {
                pending.remove(id);
            }
        }

        @Override
        void notify(String method) throws Exception {
            post(message(null, method, null));
        }

        @Override
        public void close() {
            HttpResponse<Stream<String>> s = stream;
            if (s != null) {
                s.body().close();
            }
            http.shutdownNow();
        }
    }

    // ---- An agent's connectors ----------------------------------------------------------------

    /** The tools an agent can use from its MCP connectors, and the sessions to close after its run. */
    static final class Tools implements AutoCloseable {
        private final Map<String, Tool> byName = new LinkedHashMap<>();
        private final List<Session> sessions = new ArrayList<>();

        Tool get(String name) {
            return byName.get(name);
        }

        boolean isEmpty() {
            return byName.isEmpty();
        }

        Iterable<Tool> all() {
            return byName.values();
        }

        @Override
        public void close() {
            sessions.forEach(Session::close);
        }
    }

    /**
     * The tools of the connectors in {@code agent}'s {@code mcp_connectors} parameter, filtered by
     * each entry's {@code tool_filters} (regular expressions on the whole name) and with its {@code
     * tool_descriptions} applied. A connector that cannot be reached adds no tools.
     */
    @SuppressWarnings("unchecked")
    static void resolve(MlCommons ml, Map<String, Object> agent, ActionListener<Tools> listener) {
        Object raw = agent.get("parameters") instanceof Map<?, ?> p ? p.get("mcp_connectors") : null;
        Tools found = new Tools();
        if (raw == null) {
            listener.onResponse(found);
            return;
        }
        if (ml.remote().mcpEnabled() == false) {
            listener.onFailure(new OpenSearchException(MlSettings.ML_COMMONS_MCP_CONNECTOR_DISABLED_MESSAGE));
            return;
        }
        List<Object> entries = MlAgentRunner.json(MlAgentRunner.text(raw)) instanceof List<?> l ? (List<Object>) l : List.of();
        next(ml, entries, 0, found, listener);
    }

    @SuppressWarnings("unchecked")
    private static void next(MlCommons ml, List<Object> entries, int at, Tools found, ActionListener<Tools> listener) {
        if (at == entries.size()) {
            listener.onResponse(found);
            return;
        }
        Map<String, Object> entry = entries.get(at) instanceof Map<?, ?> m ? (Map<String, Object>) m : Map.of();
        Runnable more = () -> next(ml, entries, at + 1, found, listener);
        if (entry.get("mcp_connector_id") == null) {
            more.run();
            return;
        }
        String id = String.valueOf(entry.get("mcp_connector_id"));
        ml.remote().mcpConnector(id, ActionListener.wrap(connector -> ml.remoteWork(() -> {
            try {
                Session session = open(connector);
                found.sessions.add(session);
                String type = "mcp_sse".equals(connector.get("protocol")) ? SSE_TOOL : HTTP_TOOL;
                List<Tool> tools = session.tools(type);
                List<Pattern> filters = new ArrayList<>();
                if (entry.get("tool_filters") instanceof List<?> l) {
                    l.forEach(f -> filters.add(Pattern.compile(String.valueOf(f))));
                }
                Map<String, String> descriptions = new LinkedHashMap<>();
                if (entry.get("tool_descriptions") instanceof List<?> l) {
                    for (Object d : l) {
                        if (d instanceof Map<?, ?> m) {
                            m.forEach((k, v) -> {
                                if (k != null && v instanceof String s) {
                                    descriptions.put(String.valueOf(k), s);
                                }
                            });
                        }
                    }
                }
                for (Tool t : tools) {
                    if (filters.isEmpty() == false && filters.stream().noneMatch(p -> p.matcher(t.name()).matches())) {
                        continue;
                    }
                    String description = descriptions.get(t.name());
                    Tool kept = description == null || description.isEmpty()
                        ? t
                        : new Tool(t.name(), t.type(), description, t.schema(), t.session());
                    found.byName.putIfAbsent(kept.name(), kept);
                }
            } catch (Exception e) {
                LOG.error("Failed to get MCP tools from connector " + id, e);
            }
            more.run();
        }), e -> {
            LOG.error("Failed to get MCP tools from connector " + id, e);
            more.run();
        }));
    }

    /** {@code GET _plugins/_ml/connectors/<id>/tools}: the tools of the MCP server of connector {@code id}. */
    static void listTools(MlCommons ml, String id, ActionListener<Map<String, Object>> listener) {
        if (ml.remote().mcpEnabled() == false) {
            listener.onFailure(new OpenSearchException(MlSettings.ML_COMMONS_MCP_CONNECTOR_DISABLED_MESSAGE));
            return;
        }
        ml.remote().mcpConnector(id, ActionListener.wrap(connector -> ml.remoteWork(() -> {
            try (Session session = open(connector)) {
                List<Object> out = new ArrayList<>();
                for (Tool t : session.tools("mcp_sse".equals(connector.get("protocol")) ? SSE_TOOL : HTTP_TOOL)) {
                    out.add(t.listing());
                }
                listener.onResponse(Map.of("tools", out));
            } catch (RuntimeException e) {
                listener.onFailure(e);
            } catch (Exception e) {
                listener.onFailure(new MLException(e.getMessage() == null ? e.toString() : e.getMessage(), e));
            }
        }), e -> listener.onFailure(
            e instanceof org.opensearch.OpenSearchStatusException s && s.status() == org.opensearch.core.rest.RestStatus.NOT_FOUND
                ? new org.opensearch.OpenSearchStatusException("Failed to find connector:" + id, org.opensearch.core.rest.RestStatus.NOT_FOUND)
                : e
        )));
    }

    /** A session on the MCP server of {@code connector} (as {@code RemoteModels.mcpConnector} gives it), initialised. */
    @SuppressWarnings("unchecked")
    static Session open(Map<String, Object> connector) throws Exception {
        Map<String, String> parameters = (Map<String, String>) connector.get("parameters");
        Map<String, String> headers = (Map<String, String>) connector.get("headers");
        boolean sse = "mcp_sse".equals(connector.get("protocol"));
        String url = String.valueOf(connector.get("url"));
        String path = sse ? parameters.getOrDefault("sse_endpoint", "/sse") : parameters.getOrDefault("endpoint", "/mcp/");
        long connect = 30;
        if (connector.get("client_config") instanceof Map<?, ?> c && c.get("connection_timeout") instanceof Number n) {
            connect = n.longValue();
        }
        HttpClient.Builder http = HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1).followRedirects(HttpClient.Redirect.NEVER);
        if (connect > 0) {
            http.connectTimeout(Duration.ofSeconds(connect));
        }
        URI uri = URI.create(url).resolve(path);
        Session session = sse ? new Sse(http.build(), uri, headers) : new Http(http.build(), uri, headers);
        try {
            session.initialize(sse ? SSE_PROTOCOL_VERSION : PROTOCOL_VERSION);
        } catch (Exception e) {
            session.close();
            throw e;
        }
        return session;
    }
}
