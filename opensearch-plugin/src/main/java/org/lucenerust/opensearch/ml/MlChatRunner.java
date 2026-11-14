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
 * <p>The openai, bedrock converse and gemini function-calling interfaces are ported; the other interfaces and the
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
    private static final String BEDROCK = "bedrock/converse/claude";
    private static final String GEMINI = "gemini/v1beta/generatecontent";

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
        final Map<String, MlMcp.Tool> mcp = new HashMap<>();
        String modelId;
        String memoryId;
        String parent;
        int trace;
        int max;
        String question;
        final StringBuilder interactions = new StringBuilder();
        List<String> chain;
        boolean memory;
        boolean agui;
        String style;
    }

    /** What the model answered: words, or a tool to run. */
    private record Reply(String text, String id, String name, String arguments, Object message) {}

    @SuppressWarnings("unchecked")
    void run(Map<String, Object> agent, Map<String, Object> request, List<String> chain, ActionListener<Map<String, Object>> listener) {
        Run run = new Run();
        run.agui = "ag_ui".equals(agent.get("type"));
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
        run.style = switch (iface == null ? "" : iface) {
            case OPENAI -> "openai";
            case BEDROCK -> "bedrock";
            case GEMINI -> "gemini";
            default -> throw new OpenSearchStatusException(
                "Running conversational agents is only supported with _llm_interface " + OPENAI + ", " + BEDROCK + " or " + GEMINI,
                RestStatus.NOT_IMPLEMENTED
            );
        };
        run.question = run.params.get("question");
        if (run.question == null) {
            throw new NullPointerException();
        }
        run.max = Integer.parseInt(run.params.getOrDefault("max_iteration", String.valueOf(DEFAULT_MAX_ITERATIONS)));
        List<Map<String, Object>> list = agent.get("tools") instanceof List<?> l ? (List<Map<String, Object>>) l : List.of();
        MlMcp.Tools mcp = agent.get(MlAgentRunner.MCP) instanceof MlMcp.Tools t ? t : new MlMcp.Tools();
        for (Map<String, Object> spec : list) {
            String type = String.valueOf(spec.get("type"));
            String name = spec.get("name") == null ? type : String.valueOf(spec.get("name"));
            run.specs.put(name, spec);
        }
        for (MlMcp.Tool t : mcp.all()) {
            Map<String, Object> spec = new LinkedHashMap<>();
            spec.put("type", t.type());
            spec.put("name", t.name());
            spec.put("description", t.description());
            spec.put("attributes", Map.of("input_schema", t.schema()));
            run.specs.put(t.name(), spec);
            run.mcp.put(t.name(), t);
        }
        if (mcp.isEmpty() == false) {
            // With MCP tools the agent's tools sit in a map sized for them and one more.
            Map<String, Map<String, Object>> sized = new HashMap<>(run.specs.size() + 1);
            sized.putAll(run.specs);
            run.specs = sized;
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
                        String q = MlAgentRunner.escape(String.valueOf(turn.get("input")));
                        String a = MlAgentRunner.escape(String.valueOf(turn.get("response")));
                        switch (run.style) {
                            case "bedrock" -> out.append("{\"role\":\"user\",\"content\":[{\"text\":\"").append(q).append("\"}]}, {\"role\":\"assistant\",\"content\":[{\"text\":\"").append(a).append("\"}]}, ");
                            case "gemini" -> out.append("{\"role\":\"user\",\"parts\":[{\"text\":\"").append(q).append("\"}]}, {\"role\":\"model\",\"parts\":[{\"text\":\"").append(a).append("\"}]}, ");
                            default -> out.append("{\"role\":\"user\",\"content\":\"").append(q).append("\"}, {\"role\":\"assistant\",\"content\":\"").append(a).append("\"}, ");
                        }
                    }
                }
                begin(run, out.toString(), listener);
            }, listener::onFailure));
        }
    }

    private void begin(Run run, String history, ActionListener<Map<String, Object>> listener) {
        if (run.agui == false) {
            run.params.put("_chat_history", history);
        }
        if (run.memory == false || run.agui) {
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

    /** {@code Current date and time: <now in UTC>}, formatted with {@code format} or ISO seconds when it is blank or not a pattern. */
    static String currentDateTime(String format) {
        java.time.format.DateTimeFormatter formatter;
        try {
            formatter = format == null || format.isBlank() ? null : java.time.format.DateTimeFormatter.ofPattern(format).withZone(java.time.ZoneOffset.UTC);
        } catch (IllegalArgumentException e) {
            formatter = null;
        }
        if (formatter == null) {
            formatter = java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss'Z'").withZone(java.time.ZoneOffset.UTC);
        }
        return "Current date and time: " + formatter.format(java.time.Instant.now());
    }

    @SuppressWarnings("unchecked")
    private void iterate(Run run, int round, ActionListener<Map<String, Object>> listener) {
        call(run, run.params, ActionListener.wrap(data -> handle(run, data, round, listener), listener::onFailure));
    }

    /** One call of the model with the steps so far; its answer as the connector's response map. */
    @SuppressWarnings("unchecked")
    private void call(Run run, Map<String, String> base, ActionListener<Map<String, Object>> listener) {
        Map<String, Object> call = new LinkedHashMap<>(base);
        boolean inject = Boolean.parseBoolean(String.valueOf(call.getOrDefault("inject_datetime", "false")));
        String datetime = inject ? currentDateTime(call.get("datetime_format") == null ? null : String.valueOf(call.get("datetime_format"))) : null;
        if (inject && call.containsKey("system_prompt")) {
            call.put("system_prompt", call.get("system_prompt") + "\n\n" + datetime);
        }
        call.putIfAbsent("system_prompt", inject ? "You are a helpful assistant.\n\n" + datetime : "You are a helpful assistant.");
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
            Reply reply = parse(run, data);
            if (reply.text() != null) {
                String text = reply.text();
                if (run.agui) {
                    finish(run, text, listener);
                } else {
                    trace(run, run.question, text, "LLM", ActionListener.wrap(v -> finish(run, text, listener), listener::onFailure));
                }
                return;
            }
            String name = reply.name();
            ActionListener<Void> traced = ActionListener.wrap(v -> {
                if (run.specs.containsKey(name) == false) {
                    finish(run, raw, listener);
                } else if (round >= run.max - 1) {
                    exhausted(run, listener);
                } else {
                    runTool(run, name, reply.id(), reply.arguments(), reply.message(), round, listener);
                }
            }, listener::onFailure);
            if (run.agui && run.specs.containsKey(name) == false) {
                traced.onResponse(null);
            } else {
                trace(run, run.question, raw, "LLM", traced);
            }
        }
    }

    /** The model kept asking for tools: it is asked to summarise the steps, and that is the answer. */
    private void exhausted(Run run, ActionListener<Map<String, Object>> listener) {
        String plain = "Agent reached maximum iterations (" + run.max + ") without completing the task";
        Map<String, String> base = new LinkedHashMap<>(run.params);
        base.put("system_prompt", SUMMARY_PROMPT);
        ActionListener<String> done = ActionListener.wrap(
            text -> {
                if (run.agui) {
                    finish(run, text, listener);
                } else {
                    trace(run, run.question, text, "LLM", ActionListener.wrap(v -> finish(run, text, listener), listener::onFailure));
                }
            },
            listener::onFailure
        );
        call(run, base, ActionListener.wrap(data -> done.onResponse(plain), e -> done.onResponse(plain)));
    }

    @SuppressWarnings("unchecked")
    private void runTool(Run run, String name, String id, String arguments, Object message, int round, ActionListener<Map<String, Object>> listener) {
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
                run.interactions.append(", ").append(assistant).append(", ").append(toolResult(run, id, name, out, false));
                iterate(run, round + 1, listener);
            }, listener::onFailure));
        }, e -> {
            String text = "Tool " + name + " failed: " + e.getMessage();
            trace(run, arguments, "Failed to run the tool " + name + " with the error message " + e.getMessage() + ".", name, ActionListener.wrap(v -> {
                run.interactions.append(", ").append(assistant).append(", ").append(toolResult(run, id, name, text, true));
                iterate(run, round + 1, listener);
            }, listener::onFailure));
        });
        try {
            if (MlMcp.isTool(type)) {
                Map<String, Object> given;
                try {
                    given = MlMcp.arguments(arguments);
                } catch (IllegalArgumentException e) {
                    done.onFailure(new IllegalArgumentException("Invalid JSON format: " + e.getMessage(), e));
                    return;
                }
                tools.callMcp(run.mcp.get(name), given, done);
            } else {
                tools.tool(type, params, run.chain, done);
            }
        } catch (RuntimeException e) {
            done.onFailure(e);
        }
    }

    @SuppressWarnings("unchecked")
    private static Reply parse(Run run, Map<String, Object> data) {
        switch (run.style) {
            case "bedrock" -> {
                if (data == null || data.get("output") instanceof Map<?, ?> == false) {
                    throw new PathNotFoundException("Missing property in path $['output']");
                }
                Map<String, Object> message = (Map<String, Object>) ((Map<String, Object>) data.get("output")).get("message");
                List<Object> content = (List<Object>) message.get("content");
                if ("tool_use".equals(data.get("stopReason"))) {
                    for (Object part : content) {
                        if (part instanceof Map<?, ?> m && m.get("toolUse") instanceof Map<?, ?> use) {
                            return new Reply(null, String.valueOf(use.get("toolUseId")), String.valueOf(use.get("name")), MlCommons.json(use.get("input")), firstCall(message, "content", "toolUse"));
                        }
                    }
                }
                return new Reply(answerText(((Map<String, Object>) content.get(0)).get("text")), null, null, null, message);
            }
            case "gemini" -> {
                if (data == null || data.get("candidates") instanceof List<?> == false) {
                    throw new PathNotFoundException("Missing property in path $['candidates']");
                }
                Map<String, Object> content = (Map<String, Object>) ((Map<String, Object>) ((List<?>) data.get("candidates")).get(0)).get("content");
                List<Object> parts = (List<Object>) content.get("parts");
                Map<String, Object> first = (Map<String, Object>) parts.get(0);
                if (first.get("text") != null) {
                    return new Reply(first.get("text").toString(), null, null, null, content);
                }
                for (Object part : parts) {
                    if (part instanceof Map<?, ?> m && m.get("functionCall") instanceof Map<?, ?> call) {
                        return new Reply(null, String.valueOf(call.get("name")), String.valueOf(call.get("name")), MlCommons.json(call.get("args")), firstCall(content, "parts", "functionCall"));
                    }
                }
                return new Reply(answerText(null), null, null, null, content);
            }
            default -> {
                if (data == null || data.get("choices") instanceof List<?> == false) {
                    throw new PathNotFoundException("Missing property in path $['choices']");
                }
                Map<String, Object> choice = (Map<String, Object>) ((List<?>) data.get("choices")).get(0);
                Map<String, Object> message = (Map<String, Object>) choice.get("message");
                if (message.get("tool_calls") instanceof List<?> l && l.isEmpty() == false && l.get(0) instanceof Map<?, ?> m) {
                    Map<String, Object> function = (Map<String, Object>) m.get("function");
                    return new Reply(null, String.valueOf(m.get("id")), String.valueOf(function.get("name")), String.valueOf(function.get("arguments")), message);
                }
                return new Reply(answerText(message.get("content")), null, null, null, message);
            }
        }
    }

    /** {@code message} keeping only the first tool call among the parts under {@code key}. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> firstCall(Map<String, Object> message, String key, String call) {
        List<Object> kept = new ArrayList<>();
        boolean seen = false;
        for (Object part : (List<Object>) message.get(key)) {
            boolean isCall = part instanceof Map<?, ?> m && m.get(call) != null;
            if (isCall && seen) {
                continue;
            }
            seen |= isCall;
            kept.add(part);
        }
        Map<String, Object> out = new LinkedHashMap<>(message);
        out.put(key, kept);
        return out;
    }

    /** The words in a model's answer on the given interface style; used by agents that ask for plain answers. */
    static String answer(String style, Map<String, Object> data) {
        Run run = new Run();
        run.style = style;
        return parse(run, data).text();
    }

    static String style(String interfaceName) {
        return switch (interfaceName == null ? "" : interfaceName) {
            case OPENAI -> "openai";
            case BEDROCK -> "bedrock";
            case GEMINI -> "gemini";
            default -> null;
        };
    }

    private static String answerText(Object content) {
        if (content == null) {
            throw new NullPointerException("Cannot invoke \"Object.toString()\" because \"response\" is null");
        }
        return content.toString();
    }

    private static String toolResult(Run run, String id, String name, String text, boolean error) {
        String e = MlAgentRunner.escape(text);
        return switch (run.style) {
            case "bedrock" -> "{\"role\":\"user\",\"content\":[{\"toolResult\":{\"content\":[{\"text\":\"" + e + "\"}]," + (error ? "" : "\"status\":null,") + "\"toolUseId\":\"" + MlAgentRunner.escape(id) + "\"}}]}";
            case "gemini" -> "{\"role\":\"user\",\"parts\":[{\"functionResponse\":{\"name\":\"" + MlAgentRunner.escape(name) + "\",\"response\":{\"text\":\"" + e + "\"}}}]}";
            default -> error
                ? "{\"role\":\"tool\",\"tool_call_id\":\"" + MlAgentRunner.escape(id) + "\",\"content\":\"" + e + "\"}"
                : "{\"tool_call_id\":\"" + MlAgentRunner.escape(id) + "\",\"content\":\"" + e + "\",\"role\":\"tool\"}";
        };
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
            String description = MlAgentRunner.escape(spec.get("description") != null ? String.valueOf(spec.get("description")) : entry == null ? "" : String.valueOf(entry.get("description")));
            String name = MlAgentRunner.escape(e.getKey());
            Map<?, ?> attributes = spec.get("attributes") instanceof Map<?, ?> own ? own
                : entry != null && entry.get("attributes") instanceof Map<?, ?> a ? a : Map.of();
            Object given = attributes.get("input_schema");
            switch (run.style) {
                case "bedrock" -> entries.add(
                    "{\"toolSpec\":{\"name\":\"" + name + "\",\"description\":\"" + description + "\",\"inputSchema\": {\"json\": "
                        + (given == null ? "${tool.attributes.input_schema}" : given) + " } }}"
                );
                case "gemini" -> entries.add(
                    "{\"name\":\"" + name + "\",\"description\":\"" + description + "\",\"parameters\":"
                        + (given == null ? "${tool.attributes.input_schema_cleaned}" : cleaned(String.valueOf(given))) + "}"
                );
                default -> entries.add(
                    "{\"type\": \"function\", \"function\": { \"name\": \"" + name + "\", \"description\": \"" + description + "\", \"parameters\": "
                        + (given == null ? "${tool.attributes.input_schema}" : given) + ", \"strict\": " + (attributes.get("strict") instanceof Boolean b && b) + " } }"
                );
            }
        }
        return switch (run.style) {
            case "bedrock" -> ", \"toolConfig\": {\"tools\": [" + String.join(", ", entries) + "]}";
            case "gemini" -> ", \"tools\": [{\"functionDeclarations\": [" + String.join(", ", entries) + "]}], \"toolConfig\": {\"functionCallingConfig\": {\"mode\": \"AUTO\"}}";
            default -> ", \"tools\": [" + String.join(", ", entries) + "], \"parallel_tool_calls\": false";
        };
    }

    /** A tool's input schema without the keys Gemini's API refuses. */
    private static String cleaned(String schema) {
        return MlCommons.json(strip(MlCommons.parse(schema.getBytes(java.nio.charset.StandardCharsets.UTF_8))));
    }

    private static Object strip(Object v) {
        if (v instanceof Map<?, ?> m) {
            Map<String, Object> out = new LinkedHashMap<>();
            m.forEach((k, x) -> {
                if ("additionalProperties".equals(k) == false) {
                    out.put(String.valueOf(k), strip(x));
                }
            });
            return out;
        }
        if (v instanceof List<?> l) {
            List<Object> out = new ArrayList<>();
            l.forEach(x -> out.add(strip(x)));
            return out;
        }
        return v;
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
        if (run.agui == false) {
            body.put("parent_message_id", run.parent);
            body.put("trace_number", ++run.trace);
        }
        ml.memory().addMessage(run.memoryId, body, ActionListener.wrap(id -> next.onResponse(null), next::onFailure));
    }

    private void finish(Run run, String text, ActionListener<Map<String, Object>> listener) {
        Runnable reply = () -> {
            List<Map<String, Object>> output = new ArrayList<>();
            if (run.memory) {
                output.add(Map.of("name", "memory_id", "result", run.memoryId));
                if (run.agui == false) {
                    output.add(Map.of("name", "parent_interaction_id", "result", run.parent));
                }
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
        if (run.agui) {
            Map<String, Object> last = new LinkedHashMap<>();
            last.put("input", run.question);
            last.put("response", text);
            ml.memory().addMessage(run.memoryId, last, ActionListener.wrap(id -> reply.run(), listener::onFailure));
            return;
        }
        Map<String, Object> update = new LinkedHashMap<>();
        update.put("response", text);
        update.put("additional_info", Map.of());
        ml.memory().updateMessage(run.parent, update, ActionListener.wrap(r -> reply.run(), listener::onFailure));
    }
}
