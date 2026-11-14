/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The {@code conversational} agent of ml-commons ({@code MLChatAgentRunner}): the model is asked, it names a tool, the
 * tool runs, its answer goes back to the model, until the model answers in words or {@code max_iteration} calls
 * have been made. The turns and every step are written to the conversation memory.
 *
 * <p>Only the {@code openai/v1/chat/completions} function-calling interface is ported; the other interfaces and the
 * text ReAct prompt answer 501.
 */
final class MlChatRunner {
    /** Stock's default for {@code max_iteration}. */
    private static final int DEFAULT_MAX_ITERATIONS = 10;
    private static final int DEFAULT_HISTORY = 10;
    private static final String SUMMARY_PROMPT = "You are a concise and reliable Summary Agent.\nYour job is to read the user's message and return a clear, accurate, and compact summary of the content.\n\n"
        + "Instructions:\n\n1. Produce a single coherent summary capturing the essential meaning of the input message.\n2. Do not omit important context or alter the original intent.\n"
        + "3. Do not add new information or assumptions.\n4. Use neutral tone and simple phrasing.\n5. If the message contains multiple points, combine them into a unified summary.\n"
        + "6. If the message is unclear or incomplete, summarize only what is present.\n7. Output only the summary text, with no explanations, no preambles, and no questions.\n\n"
        + "Your only task is to summarize the message.";
    private static final String NO_ESCAPE = "_chat_history,_tools,_interactions,tool_configs";
    private static final String OPENAI = "openai/v1/chat/completions";

    /** The error ml-commons' JSON path reader raises for a model answer without the expected field. */
    static final class PathNotFoundException extends RuntimeException {
        PathNotFoundException(String message) {
            super(message);
        }
    }

    private final MlCommons ml;
    private final MlAgentRunner tools;

    MlChatRunner(MlCommons ml, MlAgentRunner tools) {
        this.ml = ml;
        this.tools = tools;
    }

    /** One agent run: its parameters, tools, and where its steps are written. */
    private static final class Run {
        Map<String, String> params;
        Map<String, Map<String, Object>> specs = new HashMap<>();
        String modelId;
        String memoryId;
        String parent;
        int trace;
        int max;
        String question;
        final StringBuilder interactions = new StringBuilder();
        List<String> chain;
        boolean memory;
    }

    @SuppressWarnings("unchecked")
    void run(Map<String, Object> agent, Map<String, Object> request, List<String> chain, ActionListener<Map<String, Object>> listener) {
        Run run = new Run();
        run.chain = chain;
        run.params = new LinkedHashMap<>();
        if (agent.get("parameters") instanceof Map<?, ?> p) {
            p.forEach((k, v) -> run.params.put(String.valueOf(k), MlAgentRunner.text(v)));
        }
        Map<String, Object> llm = agent.get("llm") instanceof Map<?, ?> m ? (Map<String, Object>) m : Map.of();
        if (llm.get("parameters") instanceof Map<?, ?> p) {
            p.forEach((k, v) -> run.params.put(String.valueOf(k), MlAgentRunner.text(v)));
        }
        run.modelId = llm.get("model_id") == null ? null : String.valueOf(llm.get("model_id"));
        if (request.get("parameters") instanceof Map<?, ?> p) {
            p.forEach((k, v) -> run.params.put(String.valueOf(k), MlAgentRunner.text(v)));
        }
        String iface = run.params.get("_llm_interface");
        if (OPENAI.equals(iface) == false) {
            throw new OpenSearchStatusException(
                "Running conversational agents is only supported with _llm_interface " + OPENAI,
                RestStatus.NOT_IMPLEMENTED
            );
        }
        run.question = run.params.get("question");
        if (run.question == null) {
            throw new NullPointerException();
        }
        run.max = Integer.parseInt(run.params.getOrDefault("max_iteration", String.valueOf(DEFAULT_MAX_ITERATIONS)));
        List<Map<String, Object>> list = agent.get("tools") instanceof List<?> l ? (List<Map<String, Object>>) l : List.of();
        for (Map<String, Object> spec : list) {
            String type = String.valueOf(spec.get("type"));
            String name = spec.get("name") == null ? type : String.valueOf(spec.get("name"));
            run.specs.put(name, spec);
        }
        run.memory = agent.get("memory") instanceof Map<?, ?>;
        String given = run.params.get("memory_id");
        if (run.memory == false) {
            begin(run, "", listener);
            return;
        }
        if (given == null) {
            ml.memory().create(Map.of("name", run.question), ActionListener.wrap(id -> {
                run.memoryId = id;
                begin(run, "", listener);
            }, listener::onFailure));
        } else {
            run.memoryId = given;
            int limit = Integer.parseInt(run.params.getOrDefault("message_history_limit", String.valueOf(DEFAULT_HISTORY)));
            ml.memory().recent(given, limit, ActionListener.wrap(history -> {
                StringBuilder out = new StringBuilder();
                for (Map<String, Object> turn : history) {
                    if (turn.get("input") != null && turn.get("response") != null) {
                        out.append("{\"role\":\"user\",\"content\":\"").append(MlAgentRunner.escape(String.valueOf(turn.get("input")))).append("\"}, ");
                        out.append("{\"role\":\"assistant\",\"content\":\"").append(MlAgentRunner.escape(String.valueOf(turn.get("response")))).append("\"}, ");
                    }
                }
                begin(run, out.toString(), listener);
            }, listener::onFailure));
        }
    }

    private void begin(Run run, String history, ActionListener<Map<String, Object>> listener) {
        run.params.put("_chat_history", history);
        if (run.memory == false) {
            iterate(run, 1, listener);
            return;
        }
        Map<String, Object> parent = new LinkedHashMap<>();
        parent.put("input", run.question);
        parent.put("additional_info", Map.of());
        ml.memory().addMessage(run.memoryId, parent, ActionListener.wrap(id -> {
            run.parent = id;
            iterate(run, 1, listener);
        }, listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    private void iterate(Run run, int round, ActionListener<Map<String, Object>> listener) {
        call(run, run.params, ActionListener.wrap(data -> handle(run, data, round, listener), listener::onFailure));
    }

    /** One call of the model with the steps so far; its answer as the connector's response map. */
    @SuppressWarnings("unchecked")
    private void call(Run run, Map<String, String> base, ActionListener<Map<String, Object>> listener) {
        Map<String, Object> call = new LinkedHashMap<>(base);
        call.putIfAbsent("system_prompt", "You are a helpful assistant.");
        call.putIfAbsent("no_escape_params", NO_ESCAPE);
        call.put("_interactions", run.interactions.toString());
        call.put("tool_configs", toolConfigs(run));
        ml.predict(null, run.modelId, Map.of("parameters", call), ActionListener.wrap(answer -> {
            Map<String, Object> first = (Map<String, Object>) ((List<?>) ((Map<String, Object>) ((List<?>) answer.get("inference_results")).get(0)).get("output")).get(0);
            listener.onResponse((Map<String, Object>) first.get("dataAsMap"));
        }, listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    private void handle(Run run, Map<String, Object> data, int round, ActionListener<Map<String, Object>> listener) {
        {
            String raw = MlCommons.json(data);
            if (data == null || data.get("choices") instanceof List<?> == false) {
                throw new PathNotFoundException("Missing property in path $['choices']");
            }
            Map<String, Object> choice = (Map<String, Object>) ((List<?>) data.get("choices")).get(0);
            Map<String, Object> message = (Map<String, Object>) choice.get("message");
            Map<String, Object> toolCall = firstToolCall(message);
            if (toolCall == null) {
                Object content = message.get("content");
                if (content == null) {
                    throw new NullPointerException("Cannot invoke \"Object.toString()\" because \"response\" is null");
                }
                String text = content.toString();
                trace(run, run.question, text, "LLM", ActionListener.wrap(v -> finish(run, text, listener), listener::onFailure));
                return;
            }
            Map<String, Object> function = (Map<String, Object>) toolCall.get("function");
            String name = String.valueOf(function.get("name"));
            trace(run, run.question, raw, "LLM", ActionListener.wrap(v -> {
                if (run.specs.containsKey(name) == false) {
                    finish(run, raw, listener);
                } else if (round >= run.max - 1) {
                    exhausted(run, listener);
                } else {
                    runTool(run, name, String.valueOf(toolCall.get("id")), String.valueOf(function.get("arguments")), message, round, listener);
                }
            }, listener::onFailure));
        }
    }

    /** The model kept asking for tools: it is asked to summarise the steps, and that is the answer. */
    private void exhausted(Run run, ActionListener<Map<String, Object>> listener) {
        String plain = "Agent reached maximum iterations (" + run.max + ") without completing the task";
        Map<String, String> base = new LinkedHashMap<>(run.params);
        base.put("system_prompt", SUMMARY_PROMPT);
        ActionListener<String> done = ActionListener.wrap(
            text -> trace(run, run.question, text, "LLM", ActionListener.wrap(v -> finish(run, text, listener), listener::onFailure)),
            listener::onFailure
        );
        call(run, base, ActionListener.wrap(data -> done.onResponse(plain), e -> done.onResponse(plain)));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> firstToolCall(Map<String, Object> message) {
        if (message.get("tool_calls") instanceof List<?> l && l.isEmpty() == false && l.get(0) instanceof Map<?, ?> m) {
            return (Map<String, Object>) m;
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private void runTool(Run run, String name, String id, String arguments, Map<String, Object> message, int round, ActionListener<Map<String, Object>> listener) {
        Map<String, Object> spec = run.specs.get(name);
        String type = String.valueOf(spec.get("type"));
        Map<String, String> params = new LinkedHashMap<>();
        if (spec.get("parameters") instanceof Map<?, ?> p) {
            p.forEach((k, v) -> params.put(String.valueOf(k), MlAgentRunner.text(v)));
        }
        Object parsed = null;
        try {
            parsed = MlCommons.parse(arguments.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        } catch (RuntimeException e) {
            parsed = null;
        }
        if (parsed instanceof Map<?, ?> m) {
            m.forEach((k, v) -> params.put(String.valueOf(k), MlAgentRunner.text(v)));
        }
        String assistant = MlCommons.json(message);
        if (type.equals("IndexMappingTool") && params.containsKey("index") == false) {
            trace(run, arguments, "Failed to run the tool " + name + " due to wrong input " + arguments + ".", name, ActionListener.wrap(v -> {
                run.interactions.append(", ").append(assistant);
                iterate(run, round + 1, listener);
            }, listener::onFailure));
            return;
        }
        ActionListener<MlAgentRunner.Answer> done = ActionListener.wrap(answer -> {
            String out = answer.text() != null ? answer.text() : MlCommons.json(answer.tensors());
            trace(run, arguments, out, name, ActionListener.wrap(v -> {
                run.interactions.append(", ").append(assistant).append(", {\"tool_call_id\":\"").append(MlAgentRunner.escape(id)).append("\",\"content\":\"").append(MlAgentRunner.escape(out)).append("\",\"role\":\"tool\"}");
                iterate(run, round + 1, listener);
            }, listener::onFailure));
        }, e -> {
            String text = "Tool " + name + " failed: " + e.getMessage();
            trace(run, arguments, "Failed to run the tool " + name + " with the error message " + e.getMessage() + ".", name, ActionListener.wrap(v -> {
                run.interactions.append(", ").append(assistant).append(", {\"role\":\"tool\",\"tool_call_id\":\"").append(MlAgentRunner.escape(id)).append("\",\"content\":\"").append(MlAgentRunner.escape(text)).append("\"}");
                iterate(run, round + 1, listener);
            }, listener::onFailure));
        });
        try {
            tools.tool(type, params, run.chain, done);
        } catch (RuntimeException e) {
            done.onFailure(e);
        }
    }

    /** The {@code tool_configs} parameter: every tool as the model's function list wants it. */
    private String toolConfigs(Run run) {
        List<String> entries = new ArrayList<>();
        for (Map.Entry<String, Map<String, Object>> e : run.specs.entrySet()) {
            Map<String, Object> spec = e.getValue();
            String type = String.valueOf(spec.get("type"));
            Map<String, Object> entry = null;
            for (Map<String, Object> t : MlAgents.tools()) {
                if (type.equals(t.get("name"))) {
                    entry = t;
                }
            }
            String description = spec.get("description") != null ? String.valueOf(spec.get("description")) : entry == null ? "" : String.valueOf(entry.get("description"));
            Map<?, ?> attributes = entry != null && entry.get("attributes") instanceof Map<?, ?> a ? a : Map.of();
            String schema = attributes.get("input_schema") == null ? "${tool.attributes.input_schema}" : String.valueOf(attributes.get("input_schema"));
            boolean strict = attributes.get("strict") instanceof Boolean b && b;
            entries.add(
                "{\"type\": \"function\", \"function\": { \"name\": \"" + MlAgentRunner.escape(e.getKey()) + "\", \"description\": \"" + MlAgentRunner.escape(description)
                    + "\", \"parameters\": " + schema + ", \"strict\": " + strict + " } }"
            );
        }
        return ", \"tools\": [" + String.join(", ", entries) + "], \"parallel_tool_calls\": false";
    }

    private void trace(Run run, String input, String response, String origin, ActionListener<Void> next) {
        if (run.memory == false) {
            next.onResponse(null);
            return;
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("input", input);
        body.put("response", response);
        body.put("origin", origin);
        body.put("parent_message_id", run.parent);
        body.put("trace_number", ++run.trace);
        ml.memory().addMessage(run.memoryId, body, ActionListener.wrap(id -> next.onResponse(null), next::onFailure));
    }

    private void finish(Run run, String text, ActionListener<Map<String, Object>> listener) {
        Runnable reply = () -> {
            List<Map<String, Object>> output = new ArrayList<>();
            if (run.memory) {
                output.add(Map.of("name", "memory_id", "result", run.memoryId));
                output.add(Map.of("name", "parent_interaction_id", "result", run.parent));
            }
            Map<String, Object> data = new LinkedHashMap<>();
            data.put("additional_info", Map.of());
            data.put("response", text);
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("name", "response");
            response.put("dataAsMap", data);
            output.add(response);
            listener.onResponse(Map.of("inference_results", List.of(Map.of("output", output))));
        };
        if (run.memory == false) {
            reply.run();
            return;
        }
        Map<String, Object> update = new LinkedHashMap<>();
        update.put("response", text);
        update.put("additional_info", Map.of());
        ml.memory().updateMessage(run.parent, update, ActionListener.wrap(r -> reply.run(), listener::onFailure));
    }
}
