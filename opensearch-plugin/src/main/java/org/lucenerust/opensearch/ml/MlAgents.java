/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.delete.DeleteRequest;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * ml-commons' agents: the {@code agents} registry ({@code MLAgent} with its tool, model and memory
 * specs, as {@code TransportRegisterAgentAction}, {@code GetAgentTransportAction}, {@code
 * UpdateAgentTransportAction} and {@code DeleteAgentTransportAction} keep it in {@code
 * .plugins-ml-agent}) and the tool catalogue.
 */
final class MlAgents {
    static final Set<String> TYPES = Set.of("flow", "conversational", "conversational_flow", "conversational_v2", "plan_execute_and_reflect", "ag_ui");
    private static final int MAX_NAME = 128;

    private final MlCommons ml;
    private final MlAgentRunner runner;

    MlAgents(MlCommons ml) {
        this.ml = ml;
        this.runner = new MlAgentRunner(ml);
    }

    // ---- Execution ----------------------------------------------------------------------------

    /** The registry the search tool parses queries with. */
    void registry(org.opensearch.core.xcontent.NamedXContentRegistry registry) {
        runner.registry = registry;
    }

    /** {@code POST _plugins/_ml/agents/<id>/_execute}: the agent's answer. */
    void execute(String id, Map<String, Object> body, ActionListener<Map<String, Object>> listener) {
        execute(id, body, List.of(), listener);
    }

    /** {@code chain}: the agents that called this one, to stop an agent calling itself. */
    void execute(String id, Map<String, Object> body, List<String> chain, ActionListener<Map<String, Object>> listener) {
        load(id, body, ActionListener.wrap(agent -> {
            List<String> inner = new ArrayList<>(chain);
            inner.add(id);
            runner.run(agent, body, inner, listener);
        }, listener::onFailure));
    }

    /** {@code ?async=true}: a task now, the answer in the task when the agent is done. */
    void executeAsync(String id, Map<String, Object> body, ActionListener<Map<String, Object>> listener) {
        load(id, body, ActionListener.wrap(agent -> ml.createTask("AGENT_EXECUTION", "AGENT", null, List.of(ml.localNodeId()), true, ActionListener.wrap(task -> {
            Map<String, Object> started = new LinkedHashMap<>();
            started.put("task_id", task);
            started.put("status", "RUNNING");
            ActionListener<Map<String, Object>> done = ActionListener.wrap(
                answer -> ml.finishTask(task, "COMPLETED", Map.of("response", answer)),
                e -> ml.finishTask(task, "FAILED", Map.of("response", Map.of("error_message", e.getMessage() == null ? "" : e.getMessage())))
            );
            try {
                runner.run(agent, body, List.of(id), done);
            } catch (RuntimeException e) {
                done.onFailure(e);
            }
            listener.onResponse(started);
        }, listener::onFailure)), listener::onFailure));
    }

    /** An agent's document by id, as {@code _execute} finds it. */
    void agent(String id, ActionListener<Map<String, Object>> listener) {
        load(id, Map.of("parameters", Map.of()), listener);
    }

    private void load(String id, Map<String, Object> body, ActionListener<Map<String, Object>> listener) {
        if (body.containsKey("parameters") == false) {
            listener.onFailure(new IllegalArgumentException("Agent input data can not be empty."));
            return;
        }
        ml.getDoc(MlIndices.AGENT, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw new OpenSearchStatusException("Failed to find agent with the provided agent id: " + id, RestStatus.NOT_FOUND);
            }
            listener.onResponse(r.getSourceAsMap());
        }, e -> listener.onFailure(MlCommons.missingIndex(e, "Failed to find agent with the provided agent id: " + id))));
    }

    // ---- Tools --------------------------------------------------------------------------------

    /** The tools ml-commons registers, as {@code GET _plugins/_ml/tools} lists them. */
    @SuppressWarnings("unchecked")
    static List<Map<String, Object>> tools() {
        return TOOLS;
    }

    private static final List<Map<String, Object>> TOOLS = loadTools();

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> loadTools() {
        try (InputStream in = MlAgents.class.getResourceAsStream("tools.json")) {
            if (in == null) {
                throw new IllegalStateException("no tools.json resource");
            }
            return (List<Map<String, Object>>) (List<?>) org.opensearch.common.xcontent.XContentHelper.convertToMap(
                org.opensearch.common.xcontent.XContentType.JSON.xContent(),
                new String(in.readAllBytes(), StandardCharsets.UTF_8),
                false
            ).get("tools");
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** {@code GET _plugins/_ml/tools/<name>}. */
    static Map<String, Object> tool(String name) {
        for (Map<String, Object> t : tools()) {
            if (name.equals(t.get("name"))) {
                Map<String, Object> one = new LinkedHashMap<>(t);
                one.remove("attributes");
                return one;
            }
        }
        throw new OpenSearchStatusException("Failed to find tool information with the provided tool name: " + name, RestStatus.NOT_FOUND);
    }

    // ---- Registry -----------------------------------------------------------------------------

    /** {@code POST _plugins/_ml/agents/_register}: the new agent's id. */
    void register(Map<String, Object> body, String raw, ActionListener<String> listener) {
        Map<String, Object> agent;
        try {
            agent = parse(body, raw);
        } catch (RuntimeException e) {
            listener.onFailure(e);
            return;
        }
        long t = MlCommons.now();
        agent.put("created_time", t);
        agent.put("last_updated_time", t);
        agent.put("is_hidden", false);
        if ("plan_execute_and_reflect".equalsIgnoreCase(String.valueOf(agent.get("type")))) {
            // The planner runs its steps through a conversational agent of its own, registered with it.
            Map<String, Object> executor = new LinkedHashMap<>();
            executor.put("name", agent.get("name") + " (ReAct)");
            executor.put("type", "CONVERSATIONAL");
            executor.put("description", "Execution Agent for Plan Execute Reflect - " + agent.get("name"));
            for (String k : List.of("llm", "tools", "parameters", "memory")) {
                if (agent.get(k) != null) {
                    executor.put(k, agent.get(k));
                }
            }
            executor.put("created_time", t);
            executor.put("last_updated_time", t);
            executor.put("is_hidden", false);
            ml.indexDoc(MlIndices.AGENT, null, executor, ActionListener.wrap(e -> {
                @SuppressWarnings("unchecked")
                Map<String, Object> parameters = new TreeMap<>((Map<String, Object>) agent.get("parameters"));
                parameters.put("executor_agent_id", e.getId());
                agent.put("parameters", parameters);
                ml.indexDoc(MlIndices.AGENT, null, agent, ActionListener.wrap(r -> listener.onResponse(r.getId()), listener::onFailure));
            }, listener::onFailure));
            return;
        }
        ml.indexDoc(MlIndices.AGENT, null, agent, ActionListener.wrap(r -> listener.onResponse(r.getId()), listener::onFailure));
    }

    /** The document of a registration: the fields {@code MLAgent} reads, checked as its parser and constructor check them. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> parse(Map<String, Object> body, String raw) {
        Map<String, Object> agent = new LinkedHashMap<>();
        for (String key : List.of("name", "type", "description", "app_type")) {
            if (body.containsKey(key)) {
                Object v = body.get(key);
                String token = v == null ? "VALUE_NULL" : v instanceof Map<?, ?> ? "START_OBJECT" : v instanceof List<?> ? "START_ARRAY" : null;
                if (token != null) {
                    throw new OpenSearchStatusException(
                        "Can't get text on a " + token + " at " + location(raw, key),
                        RestStatus.INTERNAL_SERVER_ERROR
                    );
                }
            }
        }
        expect(body, "llm", Map.class, "START_OBJECT");
        expect(body, "memory", Map.class, "START_OBJECT");
        expect(body, "tools", List.class, "START_ARRAY");
        Object name = body.get("name");
        if (name == null) {
            throw new IllegalArgumentException("Agent name can't be null");
        }
        Map<String, Object> llm = null;
        if (body.get("llm") == null && body.get("model") instanceof Map<?, ?>) {
            throw new IllegalArgumentException("model_provider must be provided");
        }
        if (body.get("llm") instanceof Map<?, ?> l) {
            llm = llmSpec((Map<String, Object>) l);
        }
        List<Map<String, Object>> tools = null;
        if (body.get("tools") instanceof List<?> list) {
            tools = new ArrayList<>();
            for (Object o : list) {
                tools.add(toolSpec(o instanceof Map<?, ?> m ? (Map<String, Object>) m : Map.of()));
            }
        }
        Map<String, Object> memory = null;
        if (body.get("memory") instanceof Map<?, ?> m) {
            memory = memorySpec((Map<String, Object>) m);
        }
        String n = String.valueOf(name);
        if (n.isBlank() || n.length() > MAX_NAME) {
            throw new IllegalArgumentException("Agent name cannot be empty or exceed max length of " + MAX_NAME + " characters");
        }
        Object type = body.get("type");
        if (type == null) {
            throw new IllegalArgumentException("Agent type can't be null");
        }
        String t = String.valueOf(type);
        if (TYPES.contains(t.toLowerCase(Locale.ROOT)) == false) {
            throw new IllegalArgumentException("Invalid Agent Type");
        }
        if ("conversational".equalsIgnoreCase(t) && llm == null) {
            throw new IllegalArgumentException("We need model information for the conversational agent type");
        }
        if ("conversational_v2".equalsIgnoreCase(t)) {
            Object kind = memory == null ? null : memory.get("type");
            if ("agentic_memory".equals(kind) == false && "remote_agentic_memory".equals(kind) == false) {
                throw new IllegalArgumentException(
                    "V2 agents (CONVERSATIONAL_V2) require memory configuration. Please configure 'memory' with type 'agentic_memory' or 'remote_agentic_memory'."
                );
            }
        }
        if ("plan_execute_and_reflect".equalsIgnoreCase(t) && body.get("parameters") instanceof Map<?, ?> == false) {
            throw new OpenSearchStatusException(
                "Cannot invoke \"java.util.Map.containsKey(Object)\" because the return value of \"org.opensearch.ml.common.agent.MLAgent.getParameters()\" is null",
                RestStatus.INTERNAL_SERVER_ERROR
            );
        }
        if (tools != null) {
            Set<String> seen = new HashSet<>();
            for (Map<String, Object> tool : tools) {
                String key = String.valueOf(tool.getOrDefault("name", tool.get("type")));
                if (seen.add(key) == false) {
                    throw new IllegalArgumentException("Duplicate tool defined in agent configuration");
                }
            }
        }
        agent.put("name", n);
        agent.put("type", t);
        if (body.get("description") != null) {
            agent.put("description", String.valueOf(body.get("description")));
        }
        if (llm != null) {
            agent.put("llm", llm);
        }
        if (tools != null) {
            agent.put("tools", tools);
        }
        Map<String, String> parameters = strings(body.get("parameters"));
        if (parameters != null) {
            agent.put("parameters", parameters);
        }
        if (memory != null) {
            agent.put("memory", memory);
        }
        if (body.get("app_type") != null) {
            agent.put("app_type", String.valueOf(body.get("app_type")));
        }
        return agent;
    }

    /** Refuses a {@code key} that is neither absent, null nor a {@code wanted}, as a parser expecting {@code token} does. */
    private static void expect(Map<String, Object> body, String key, Class<?> wanted, String token) {
        Object v = body.get(key);
        if (v == null || wanted.isInstance(v)) {
            return;
        }
        String found = v instanceof String ? "VALUE_STRING"
            : v instanceof Number ? "VALUE_NUMBER"
            : v instanceof Boolean b ? (b ? "VALUE_TRUE" : "VALUE_FALSE")
            : v instanceof Map<?, ?> ? "START_OBJECT" : "START_ARRAY";
        throw new IllegalArgumentException("Failed to parse object: expecting token of type [" + token + "] but found [" + found + "]");
    }

    /** {@code line:column} where the value of {@code key} starts in the request's JSON text, as the parser reports it. */
    private static String location(String raw, String key) {
        int at = raw.indexOf("\"" + key + "\"");
        if (at < 0) {
            return "1:1";
        }
        int i = raw.indexOf(':', at + key.length() + 2) + 1;
        while (i < raw.length() && Character.isWhitespace(raw.charAt(i))) {
            i++;
        }
        int line = 1;
        int lineStart = 0;
        for (int k = 0; k < i; k++) {
            if (raw.charAt(k) == '\n') {
                line++;
                lineStart = k + 1;
            }
        }
        return line + ":" + (i - lineStart + 1);
    }

    private static Map<String, Object> llmSpec(Map<String, Object> llm) {
        if (llm.get("model_id") == null) {
            throw new IllegalArgumentException("model id is null");
        }
        Map<String, Object> spec = new LinkedHashMap<>();
        spec.put("model_id", String.valueOf(llm.get("model_id")));
        Map<String, String> parameters = strings(llm.get("parameters"));
        if (parameters != null) {
            spec.put("parameters", parameters);
        }
        return spec;
    }

    private static Map<String, Object> toolSpec(Map<String, Object> tool) {
        if (tool.get("type") == null) {
            throw new IllegalArgumentException("tool type is null");
        }
        Map<String, Object> spec = new LinkedHashMap<>();
        spec.put("type", String.valueOf(tool.get("type")));
        if (tool.get("name") != null) {
            spec.put("name", String.valueOf(tool.get("name")));
        }
        if (tool.get("description") != null) {
            spec.put("description", String.valueOf(tool.get("description")));
        }
        Map<String, String> attributes = strings(tool.get("attributes"));
        if (attributes != null) {
            spec.put("attributes", attributes);
        }
        Map<String, String> parameters = strings(tool.get("parameters"));
        if (parameters != null) {
            spec.put("parameters", parameters);
        }
        spec.put("include_output_in_agent_response", Boolean.TRUE.equals(tool.get("include_output_in_agent_response")));
        Map<String, String> config = strings(tool.get("config"));
        if (config != null) {
            spec.put("config", config);
        }
        return spec;
    }

    private static Map<String, Object> memorySpec(Map<String, Object> memory) {
        Map<String, Object> spec = new LinkedHashMap<>();
        if (memory.get("type") != null) {
            spec.put("type", String.valueOf(memory.get("type")));
        }
        if (memory.get("session_id") != null) {
            spec.put("session_id", String.valueOf(memory.get("session_id")));
        }
        if (memory.get("window_size") instanceof Number w) {
            spec.put("window_size", w.intValue());
        } else if (memory.get("window_size") instanceof String w && w.matches("-?\\d+")) {
            spec.put("window_size", Integer.parseInt(w));
        }
        return spec;
    }

    /** A {@code Map<String, String>} of {@code o}: text stays, any other value is its JSON text. */
    private static Map<String, String> strings(Object o) {
        if (o instanceof Map<?, ?> m == false) {
            return null;
        }
        Map<String, String> out = new TreeMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) o).entrySet()) {
            Object v = e.getValue();
            out.put(String.valueOf(e.getKey()), v instanceof String s ? s : v == null ? null : json(v));
        }
        return out;
    }

    private static String json(Object v) {
        if (v instanceof Number || v instanceof Boolean) {
            return String.valueOf(v);
        }
        try (XContentBuilder b = XContentFactory.jsonBuilder()) {
            b.value(v);
            return b.toString();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** {@code GET _plugins/_ml/agents/<id>}. */
    void get(String id, ActionListener<Map<String, Object>> listener) {
        String missing = "Failed to find agent with the provided agent id: " + id;
        ml.getDoc(MlIndices.AGENT, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw new OpenSearchStatusException(missing, RestStatus.NOT_FOUND);
            }
            listener.onResponse(ordered(r.getSourceAsMap()));
        }, e -> listener.onFailure(MlCommons.missingIndex(e, missing))));
    }

    /** The fields of {@code MLAgent#toXContent}, in its order. */
    private static Map<String, Object> ordered(Map<String, Object> doc) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (String k : List.of("name", "type", "description", "llm", "tools", "parameters", "memory", "created_time", "last_updated_time", "app_type", "is_hidden")) {
            if (doc.get(k) != null) {
                out.put(k, doc.get(k));
            }
        }
        return out;
    }

    /** {@code PUT _plugins/_ml/agents/<id>}: the given fields merged into the agent. */
    @SuppressWarnings("unchecked")
    void update(String id, Map<String, Object> body, ActionListener<UpdateResponse> listener) {
        ml.getDoc(MlIndices.AGENT, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw new OpenSearchStatusException("Failed to get agent with ID " + id, RestStatus.NOT_FOUND);
            }
            Map<String, Object> agent = r.getSourceAsMap();
            Map<String, Object> fields = new LinkedHashMap<>();
            if (body.get("type") != null && String.valueOf(body.get("type")).equals(agent.get("type")) == false) {
                throw new IllegalArgumentException("Agent type cannot be updated");
            }
            if (body.get("name") != null) {
                String n = String.valueOf(body.get("name"));
                if (n.isBlank() || n.length() > MAX_NAME) {
                    throw new IllegalArgumentException("Agent name cannot be empty or exceed max length of " + MAX_NAME + " characters");
                }
                fields.put("name", n);
            }
            if (body.get("description") != null) {
                fields.put("description", String.valueOf(body.get("description")));
            }
            if (body.get("llm") instanceof Map<?, ?> l) {
                Map<String, Object> given = (Map<String, Object>) l;
                Map<String, Object> merged = agent.get("llm") instanceof Map<?, ?> old ? new LinkedHashMap<>((Map<String, Object>) old) : new LinkedHashMap<>();
                if (given.get("model_id") != null) {
                    merged.put("model_id", String.valueOf(given.get("model_id")));
                }
                Map<String, String> parameters = strings(given.get("parameters"));
                if (parameters != null) {
                    Map<String, Object> all = merged.get("parameters") instanceof Map<?, ?> p ? new TreeMap<>((Map<String, Object>) p) : new TreeMap<>();
                    all.putAll(parameters);
                    merged.put("parameters", all);
                }
                fields.put("llm", merged);
            }
            if (body.get("tools") instanceof List<?> list && list.isEmpty() == false) {
                List<Map<String, Object>> tools = new ArrayList<>();
                for (Object o : list) {
                    tools.add(toolSpec(o instanceof Map<?, ?> m ? (Map<String, Object>) m : Map.of()));
                }
                fields.put("tools", tools);
            }
            Map<String, String> parameters = strings(body.get("parameters"));
            if (parameters != null) {
                Map<String, Object> all = agent.get("parameters") instanceof Map<?, ?> p ? new TreeMap<>((Map<String, Object>) p) : new TreeMap<>();
                all.putAll(parameters);
                fields.put("parameters", all);
            }
            if (body.get("memory") instanceof Map<?, ?> m) {
                Map<String, Object> given = (Map<String, Object>) m;
                if (given.get("type") != null && "conversation_index".equals(String.valueOf(given.get("type"))) == false) {
                    throw new IllegalArgumentException("Invalid memory type: " + given.get("type"));
                }
                if (agent.get("memory") instanceof Map<?, ?> old == false) {
                    throw new OpenSearchStatusException(
                        "Cannot invoke \"org.opensearch.ml.common.agent.MLMemorySpec.getSessionId()\" because \"originalMemory\" is null",
                        RestStatus.INTERNAL_SERVER_ERROR
                    );
                }
                Map<String, Object> merged = new LinkedHashMap<>((Map<String, Object>) agent.get("memory"));
                merged.putAll(memorySpec(given));
                fields.put("memory", merged);
            }
            if (body.get("app_type") != null) {
                fields.put("app_type", String.valueOf(body.get("app_type")));
            }
            fields.put("last_updated_time", MlCommons.now());
            ml.updateDoc(MlIndices.AGENT, id, fields, listener);
        }, e -> listener.onFailure(MlCommons.missingIndex(e, "Failed to get agent with ID " + id))));
    }

    /** {@code DELETE _plugins/_ml/agents/<id>}. */
    void delete(String id, ActionListener<DeleteResponse> listener) {
        DeleteRequest request = new DeleteRequest(MlIndices.AGENT, id).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
        MlIndices.system(ml.client(), l -> ml.client().delete(request, l), ActionListener.<DeleteResponse>wrap(r -> {
            if (r.getResult() == org.opensearch.action.DocWriteResponse.Result.NOT_FOUND) {
                throw new OpenSearchStatusException("Fail to find ml agent", RestStatus.NOT_FOUND);
            }
            listener.onResponse(r);
        }, e -> listener.onFailure(MlCommons.missingIndex(e, "Fail to find ml agent"))));
    }
}
