/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.admin.cluster.health.ClusterHealthRequest;
import org.opensearch.action.admin.cluster.health.ClusterHealthResponse;
import org.opensearch.action.admin.cluster.state.ClusterStateResponse;
import org.opensearch.action.admin.indices.get.GetIndexResponse;
import org.opensearch.action.admin.indices.stats.IndicesStatsResponse;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.admin.cluster.state.ClusterStateRequest;
import org.opensearch.action.admin.indices.get.GetIndexRequest;
import org.opensearch.action.admin.indices.stats.CommonStats;
import org.opensearch.action.admin.indices.stats.IndicesStatsRequest;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.support.IndicesOptions;
import org.opensearch.cluster.health.ClusterHealthStatus;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;

/**
 * Runs the agents ml-commons runs without a language model ({@code flow} and {@code
 * conversational_flow}, as {@code MLFlowAgentRunner} and {@code MLChatAgentRunner}'s flow sibling
 * do): each tool in turn, its answer handed to the next as {@code <name>.output}. The tools here
 * are the ones that need none: {@code ListIndexTool}, {@code IndexMappingTool}, {@code
 * SearchIndexTool}, {@code MLModelTool}, {@code ConnectorTool}, {@code AgentTool} and the
 * scratch pad tools.
 */
final class MlAgentRunner {
    private static final String LIST_HEADER = "row,health,status,index,uuid,pri(number of primary shards),rep(number of replica shards),"
        + "docs.count(number of available documents),docs.deleted(number of deleted documents),"
        + "store.size(store size of primary and replica shards),pri.store.size(store size of primary shards)\n";

    private static final int MAX_DEPTH = 10;

    private final MlCommons ml;
    volatile NamedXContentRegistry registry = NamedXContentRegistry.EMPTY;

    private static final String LAST = "\u0000last";
    private static final String LAST_OUTPUT = "\u0000last.output";
    private final MlChatRunner chat;

    MlAgentRunner(MlCommons ml) {
        this.ml = ml;
        this.chat = new MlChatRunner(ml, this);
    }

    /** What a tool answered: text, or the tensors of a model's answer. */
    record Answer(String text, Object tensors) {
        static Answer of(String text) {
            return new Answer(text, null);
        }
    }

    // ---- Agents -------------------------------------------------------------------------------

    /** The answer of {@code agent} to {@code request}: the {@code inference_results} document. */
    @SuppressWarnings("unchecked")
    void run(Map<String, Object> agent, Map<String, Object> request, List<String> chain, ActionListener<Map<String, Object>> listener) {
        String type = String.valueOf(agent.get("type"));
        if (type.equals("conversational")) {
            chat.run(agent, request, chain, listener);
            return;
        }
        if (type.equals("flow") == false && type.equals("conversational_flow") == false) {
            throw new OpenSearchStatusException("Running " + type + " agents is not supported", RestStatus.NOT_IMPLEMENTED);
        }
        List<Map<String, Object>> tools = agent.get("tools") instanceof List<?> l ? (List<Map<String, Object>>) l : List.of();
        if (tools.isEmpty()) {
            throw new IllegalArgumentException("no tool configured");
        }
        Map<String, String> given = new LinkedHashMap<>();
        if (request.get("parameters") instanceof Map<?, ?> p) {
            p.forEach((k, v) -> given.put(String.valueOf(k), text(v)));
        }
        Map<String, String> shared = new LinkedHashMap<>();
        boolean named = type.equals("conversational_flow") || tools.size() > 1;
        List<Map<String, Object>> outputs = new ArrayList<>();
        if (type.equals("conversational_flow") && agent.get("memory") instanceof Map<?, ?>) {
            String question = given.get("question");
            if (question == null) {
                throw new NullPointerException();
            }
            String memoryId = given.get("memory_id");
            if (memoryId != null) {
                ml.memory().get(memoryId, ActionListener.wrap(found -> step(tools, 0, shared, given, named, outputs, chain, ActionListener.wrap(answer -> {
                    Map<String, Object> message = new LinkedHashMap<>();
                    message.put("input", question);
                    message.put("additional_info", Map.of(shared.get(LAST) + ".output", shared.get(LAST_OUTPUT)));
                    ml.memory().addMessage(memoryId, message, ActionListener.wrap(id -> listener.onResponse(answer), listener::onFailure));
                }, listener::onFailure)), listener::onFailure));
                return;
            }
        }
        step(tools, 0, shared, given, named, outputs, chain, listener);
    }

    @SuppressWarnings("unchecked")
    private void step(
        List<Map<String, Object>> tools,
        int at,
        Map<String, String> shared,
        Map<String, String> given,
        boolean named,
        List<Map<String, Object>> outputs,
        List<String> chain,
        ActionListener<Map<String, Object>> listener
    ) {
        Map<String, Object> tool = tools.get(at);
        String type = String.valueOf(tool.get("type"));
        String name = tool.get("name") == null ? type : String.valueOf(tool.get("name"));
        Map<String, String> params = new LinkedHashMap<>(shared);
        params.remove(LAST);
        params.remove(LAST_OUTPUT);
        if (tool.get("parameters") instanceof Map<?, ?> p) {
            p.forEach((k, v) -> params.put(String.valueOf(k), String.valueOf(v)));
        }
        params.putAll(given);
        for (String prefix : new String[] { type + ".", name + "." }) {
            for (Map.Entry<String, String> e : new ArrayList<>(params.entrySet())) {
                if (e.getKey().startsWith(prefix) && e.getKey().length() > prefix.length() && e.getKey().endsWith(".output") == false) {
                    params.put(e.getKey().substring(prefix.length()), e.getValue());
                }
            }
        }
        boolean last = at == tools.size() - 1;
        ActionListener<Answer> next = ActionListener.wrap(answer -> {
            String label = named ? name : "response";
            if (last || Boolean.TRUE.equals(tool.get("include_output_in_agent_response"))) {
                if (answer.tensors() instanceof List<?> tensors) {
                    for (Object t : tensors) {
                        outputs.add((Map<String, Object>) t);
                    }
                } else {
                    Map<String, Object> item = new LinkedHashMap<>();
                    item.put("name", label);
                    item.put("result", answer.text());
                    outputs.add(item);
                }
            }
            if (last) {
                shared.put(LAST, name);
                shared.put(LAST_OUTPUT, answer.text() != null ? answer.text() : MlCommons.json(answer.tensors()));
                listener.onResponse(Map.of("inference_results", List.of(Map.of("output", outputs))));
                return;
            }
            String out = answer.text() != null ? answer.text() : MlCommons.json(answer.tensors());
            shared.put(name + ".output", escape(out));
            step(tools, at + 1, shared, given, named, outputs, chain, listener);
        }, listener::onFailure);
        try {
            tool(type, params, chain, next);
        } catch (RuntimeException e) {
            listener.onFailure(e);
        }
    }

    // ---- Tools --------------------------------------------------------------------------------

    void tool(String type, Map<String, String> params, List<String> chain, ActionListener<Answer> listener) {
        switch (type) {
            case "ListIndexTool" -> listIndices(params, listener);
            case "IndexMappingTool" -> indexMapping(params, listener);
            case "SearchIndexTool" -> searchIndex(params, listener);
            case "MLModelTool" -> model(params, listener);
            case "ConnectorTool" -> connector(params, listener);
            case "AgentTool" -> agent(params, chain, listener);
            case "WriteToScratchPadTool" -> {
                String notes = params.get("notes");
                if (notes == null) {
                    throw new IllegalArgumentException("Parameter 'notes' is required for WriteToScratchPadTool.");
                }
                listener.onResponse(Answer.of("Wrote to scratchpad: " + notes));
            }
            case "ReadFromScratchPadTool" -> listener.onResponse(Answer.of("Scratchpad is empty."));
            default -> throw new OpenSearchStatusException("Tool " + type + " is not supported", RestStatus.NOT_IMPLEMENTED);
        }
    }

    private void listIndices(Map<String, String> params, ActionListener<Answer> listener) {
        List<String> given = list(params.get("indices"));
        String[] names = given == null ? new String[0] : given.toArray(String[]::new);
        ClusterStateRequest state = new ClusterStateRequest().clear()
            .metadata(true)
            .indices(names)
            .indicesOptions(IndicesOptions.strictExpandHidden());
        ml.client().admin().cluster().state(state, ActionListener.wrap(s -> {
            List<IndexMetadata> found = new ArrayList<>(s.getState().metadata().indices().values());
            String[] concrete = found.stream().map(m -> m.getIndex().getName()).toArray(String[]::new);
            if (concrete.length == 0) {
                listener.onResponse(Answer.of(names.length == 0 ? LIST_HEADER : "There were no results searching the indices parameter [" + params.get("indices") + "]."));
                return;
            }
            ClusterHealthRequest health = new ClusterHealthRequest(concrete);
            health.local(true);
            health.level(ClusterHealthRequest.Level.INDICES);
            IndicesStatsRequest stats = new IndicesStatsRequest().indices(concrete).indicesOptions(IndicesOptions.strictExpandHidden()).docs(true).store(true);
            ml.client().admin().cluster().health(health, ActionListener.wrap(h -> {
                ml.client().admin().indices().stats(stats, ActionListener.wrap(st -> {
                    StringBuilder out = new StringBuilder(LIST_HEADER);
                    int row = 0;
                    for (IndexMetadata m : found) {
                        String name = m.getIndex().getName();
                        var index = h.getIndices().get(name);
                        var one = st.getIndices().get(name);
                        CommonStats total = one == null ? null : one.getTotal();
                        CommonStats primaries = one == null ? null : one.getPrimaries();
                        boolean open = m.getState() == IndexMetadata.State.OPEN;
                        out.append(++row).append(',')
                            .append(index == null ? "" : status(index.getStatus())).append(',')
                            .append(open ? "open" : "close").append(',')
                            .append(name).append(',')
                            .append(m.getIndexUUID()).append(',')
                            .append(m.getNumberOfShards()).append(',')
                            .append(m.getNumberOfReplicas()).append(',')
                            .append(total == null || total.getDocs() == null ? "" : total.getDocs().getCount()).append(',')
                            .append(total == null || total.getDocs() == null ? "" : total.getDocs().getDeleted()).append(',')
                            .append(total == null || total.getStore() == null ? "" : total.getStore().size()).append(',')
                            .append(primaries == null || primaries.getStore() == null ? "" : primaries.getStore().size())
                            .append('\n');
                    }
                    listener.onResponse(Answer.of(out.toString()));
                }, listener::onFailure));
            }, listener::onFailure));
        }, listener::onFailure));
    }

    private static String status(ClusterHealthStatus s) {
        return s.name().toLowerCase(Locale.ROOT);
    }

    private void indexMapping(Map<String, String> params, ActionListener<Answer> listener) {
        String raw = params.get("index");
        List<String> given = raw != null && raw.strip().startsWith("[") ? list(raw) : raw == null ? null : List.of(raw);
        if (given == null || given.isEmpty()) {
            listener.onResponse(Answer.of("There were no results searching the index parameter [" + params.get("index") + "]."));
            return;
        }
        GetIndexRequest request = new GetIndexRequest().indices(given.toArray(String[]::new))
            .addFeatures(GetIndexRequest.Feature.MAPPINGS, GetIndexRequest.Feature.SETTINGS);
        ml.client().admin().indices().getIndex(request, ActionListener.wrap(r -> {
            StringBuilder out = new StringBuilder();
            for (String index : r.indices()) {
                out.append("index: ").append(index).append("\n\nmappings:\n");
                var mapping = r.mappings().get(index);
                if (mapping != null) {
                    for (Map.Entry<String, Object> e : mapping.getSourceAsMap().entrySet()) {
                        out.append(e.getKey()).append('=').append(e.getValue()).append('\n');
                    }
                }
                out.append("\n\nsettings:\n");
                Settings settings = r.settings().get(index);
                if (settings != null) {
                    for (String key : new TreeSet<>(settings.keySet())) {
                        out.append(key).append('=').append(settings.get(key)).append('\n');
                    }
                }
                out.append("\n\n");
            }
            listener.onResponse(Answer.of(out.toString()));
        }, listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    private void searchIndex(Map<String, String> params, ActionListener<Answer> listener) throws IllegalArgumentException {
        String input = params.get("input");
        Object parsed;
        if (input == null) {
            parsed = Map.of();
        } else {
            parsed = json(input);
            if (parsed instanceof Map<?, ?> == false) {
                String kind = parsed instanceof List<?> ? "JsonArray" : parsed == null ? "JsonNull" : "JsonPrimitive";
                throw new IllegalArgumentException(
                    "Invalid JSON format in input parameter. Expected valid JSON with 'index' and 'query' fields. Error: Expected a com.google.gson.JsonObject but was com.google.gson."
                        + kind + "; at path $"
                );
            }
        }
        Map<String, Object> both = (Map<String, Object>) parsed;
        boolean noIndex = both.get("index") == null;
        boolean noQuery = both.get("query") == null;
        if (noIndex || noQuery) {
            throw new IllegalArgumentException(
                "SearchIndexTool's two parameters: index and query are required and should be in valid format. Missing: "
                    + (noIndex && noQuery ? "both 'index' and 'query'" : noIndex ? "'index'" : "'query'")
            );
        }
        String index = String.valueOf(both.get("index"));
        String query = both.get("query") instanceof String q ? q : MlCommons.json(both.get("query"));
        SearchSourceBuilder source;
        try (XContentParser p = JsonXContent.jsonXContent.createParser(registry, DeprecationHandler.IGNORE_DEPRECATIONS, query)) {
            source = SearchSourceBuilder.fromXContent(p);
        } catch (org.opensearch.core.common.ParsingException e) {
            throw new org.opensearch.core.common.ParsingException(
                e.getLineNumber(),
                e.getColumnNumber(),
                "ParsingException[Invalid query format]: " + e.getMessage(),
                null
            );
        } catch (IOException e) {
            throw new IllegalArgumentException(e.getMessage(), e);
        }
        SearchRequest request = new SearchRequest(index).source(source);
        ml.client().search(request, ActionListener.wrap(r -> {
            StringBuilder out = new StringBuilder();
            for (SearchHit hit : r.getHits().getHits()) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("_index", hit.getIndex());
                m.put("_source", hit.getSourceAsMap());
                m.put("_id", hit.getId());
                String body = gson(m);
                float score = hit.getScore();
                out.append(body, 0, body.length() - 1).append(",\"_score\":").append(Float.isNaN(score) ? "NaN" : Float.toString(score)).append("}\n");
            }
            listener.onResponse(Answer.of(out.toString()));
        }, listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    private void model(Map<String, String> params, ActionListener<Answer> listener) {
        String id = params.get("model_id");
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("Model ID can't be null or empty");
        }
        String field = params.getOrDefault("response_field", "response");
        ml.predict(null, id, Map.of("parameters", new LinkedHashMap<String, Object>(params)), ActionListener.wrap(answer -> {
            Map<String, Object> first = (Map<String, Object>) ((List<?>) ((Map<String, Object>) ((List<?>) answer.get("inference_results")).get(0)).get("output")).get(0);
            Object data = first.get("dataAsMap");
            if (data instanceof Map<?, ?> m && m.containsKey(field)) {
                Object v = m.get(field);
                listener.onResponse(Answer.of(v instanceof String s ? s : MlCommons.json(v)));
            } else {
                listener.onResponse(Answer.of(MlCommons.json(data)));
            }
        }, listener::onFailure));
    }

    private void connector(Map<String, String> params, ActionListener<Answer> listener) {
        String id = params.get("connector_id");
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("Connector ID can't be null or empty");
        }
        String action = params.getOrDefault("connector_action", "EXECUTE");
        ml.executeConnector(id, action, new LinkedHashMap<String, Object>(params), ActionListener.wrap(tensors -> listener.onResponse(new Answer(null, tensors)), listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    private void agent(Map<String, String> params, List<String> chain, ActionListener<Answer> listener) {
        String id = params.get("agent_id");
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("Agent ID cannot be null or empty");
        }
        if (chain.contains(id) || chain.size() >= MAX_DEPTH) {
            throw new IllegalArgumentException("Agent [" + id + "] is called by itself or through more than " + MAX_DEPTH + " agents");
        }
        Map<String, Object> input = new LinkedHashMap<>();
        input.put("parameters", new LinkedHashMap<String, Object>(params));
        ml.agents().execute(id, input, chain, ActionListener.wrap(answer -> {
            Map<String, Object> first = (Map<String, Object>) ((List<?>) ((Map<String, Object>) ((List<?>) answer.get("inference_results")).get(0)).get("output")).get(0);
            listener.onResponse(first.get("result") instanceof String s ? Answer.of(s) : new Answer(null, first.get("dataAsMap")));
        }, listener::onFailure));
    }

    // ---- Values -------------------------------------------------------------------------------

    /** A parameter value as the agent's string parameters carry it. */
    static String text(Object v) {
        if (v instanceof String s) {
            return s;
        }
        if (v instanceof Map<?, ?> || v instanceof List<?>) {
            return gson(v);
        }
        return String.valueOf(v);
    }

    /** {@code v} as JSON, with the characters gson escapes for HTML as it does. */
    static String gson(Object v) {
        return MlCommons.json(v).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("=", "\\u003d").replace("'", "\\u0027");
    }

    /** {@code s} as the inside of a JSON string, as the agent passes a tool's answer on. */
    static String escape(String s) {
        String quoted = MlCommons.json(s);
        return quoted.substring(1, quoted.length() - 1);
    }

    /** The JSON in {@code text}, or a bare word as the string it is (as gson reads leniently). */
    @SuppressWarnings("unchecked")
    private static Object json(String text) {
        String t = text.strip();
        try {
            Map<String, Object> wrapped = MlCommons.parse(("{\"v\":" + t + "}").getBytes(java.nio.charset.StandardCharsets.UTF_8));
            return wrapped.get("v");
        } catch (RuntimeException e) {
            return t;
        }
    }

    /** A JSON array of strings, as the tools read their index lists; {@code null} when absent. */
    private static List<String> list(String text) {
        if (text == null) {
            return null;
        }
        String t = text.strip();
        Object parsed = json(t);
        if (parsed instanceof List<?> l) {
            List<String> out = new ArrayList<>();
            for (Object o : l) {
                out.add(String.valueOf(o));
            }
            return out;
        }
        String kind = parsed instanceof Map<?, ?> ? "BEGIN_OBJECT" : parsed instanceof Boolean ? "BOOLEAN" : parsed instanceof Number ? "NUMBER" : parsed == null ? "NULL" : "STRING";
        int column = switch (kind) {
            case "BEGIN_OBJECT" -> 2;
            case "NUMBER", "BOOLEAN" -> t.length() + 1;
            default -> 1;
        };
        throw new IllegalStateException(
            "Expected BEGIN_ARRAY but was " + kind + " at line 1 column " + column + " path $\nSee https://github.com/google/gson/blob/main/Troubleshooting.md#unexpected-json-structure"
        );
    }
}
