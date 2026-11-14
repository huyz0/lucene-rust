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

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The {@code plan_execute_and_reflect} agent of ml-commons ({@code MLPlanExecuteAndReflectAgentRunner}): a planner
 * model writes a plan as JSON, the executor agent runs its first step, the planner is shown the result and
 * revises the plan, until it returns a result or {@code max_steps} steps have run.
 */
final class MlPlanRunner {
    private static final int DEFAULT_MAX_STEPS = 20;
    private static final String DEFAULT_EXECUTOR_ITERATIONS = "20";
    private static final String DEFAULT_HISTORY = "10";
    private static final Map<String, String> PROMPTS = prompts();

    private final MlCommons ml;
    private final MlChatRunner chat;

    MlPlanRunner(MlCommons ml, MlChatRunner chat) {
        this.ml = ml;
        this.chat = chat;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> prompts() {
        try (InputStream in = MlPlanRunner.class.getResourceAsStream("plan_execute_prompts.json")) {
            if (in == null) {
                throw new IllegalStateException("no plan_execute_prompts.json resource");
            }
            Map<String, Object> raw = MlCommons.parse(in.readAllBytes());
            Map<String, String> out = new LinkedHashMap<>();
            raw.forEach((k, v) -> out.put(k, String.valueOf(v)));
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** One run: parameters, the steps so far and where they are written. */
    private static final class Run {
        Map<String, String> params;
        Map<String, Object> agent;
        Map<String, Object> executor;
        String style;
        String modelId;
        String memoryId;
        String parent;
        int trace;
        int max;
        int executed;
        String executorMemory;
        String executorParent;
        List<String> plan = new ArrayList<>();
        final List<String> completed = new ArrayList<>();
        String lastStep;
        String lastResult;
        List<String> chain;
    }

    @SuppressWarnings("unchecked")
    void run(Map<String, Object> agent, Map<String, Object> request, List<String> chain, ActionListener<Map<String, Object>> listener) {
        Run run = new Run();
        run.agent = agent;
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
        run.style = MlChatRunner.style(run.params.get("_llm_interface"));
        if (run.style == null) {
            throw new OpenSearchStatusException("Running plan_execute_and_reflect agents needs a supported _llm_interface", RestStatus.NOT_IMPLEMENTED);
        }
        String question = run.params.get("question");
        if (question == null) {
            throw new NullPointerException();
        }
        run.max = Integer.parseInt(run.params.getOrDefault("max_steps", String.valueOf(DEFAULT_MAX_STEPS)));
        String executorId = run.params.get("executor_agent_id");
        ml.agents().agent(executorId, ActionListener.wrap(executor -> {
            run.executor = executor;
            String given = run.params.get("memory_id");
            if (given == null) {
                ml.memory().create(Map.of("name", question), ActionListener.wrap(id -> {
                    run.memoryId = id;
                    begin(run, question, listener);
                }, listener::onFailure));
            } else {
                ml.memory().get(given, ActionListener.wrap(found -> {
                    run.memoryId = given;
                    begin(run, question, listener);
                }, listener::onFailure));
            }
        }, listener::onFailure));
    }

    private void begin(Run run, String question, ActionListener<Map<String, Object>> listener) {
        ml.memory().addMessage(run.memoryId, Map.of("input", question), ActionListener.wrap(id -> {
            run.parent = id;
            plan(run, question, listener);
        }, listener::onFailure));
    }

    /** Asks the planner: the first plan when nothing has run, a revision after each step. */
    private void plan(Run run, String question, ActionListener<Map<String, Object>> listener) {
        Map<String, String> params = new LinkedHashMap<>(run.params);
        params.put("system_prompt", PROMPTS.get("planner_system_prompt"));
        params.putIfAbsent("planner_prompt", PROMPTS.get("planner_prompt"));
        params.putIfAbsent("reflect_prompt", PROMPTS.get("reflect_prompt"));
        params.put("user_prompt", question);
        params.put("tools_prompt", toolsPrompt(run));
        String template;
        if (run.completed.isEmpty()) {
            template = params.getOrDefault("planner_prompt_template", PROMPTS.get("planner_prompt_template"));
        } else {
            params.put("steps", String.join(", ", run.plan));
            params.put("completed_steps", String.join(", ", run.completed));
            template = params.getOrDefault("planner_with_history_template", PROMPTS.get("planner_with_history_template"));
        }
        params.put("prompt", substitute(template, params));
        params.putIfAbsent("no_escape_params", "tool_configs,_tools");
        model(run, params, ActionListener.wrap(text -> {
            Map<String, Object> parsed = parse(text);
            Object result = parsed.get("result");
            Object steps = parsed.get("steps");
            if (result != null && result.toString().isEmpty() == false) {
                finish(run, result.toString(), listener);
                return;
            }
            run.plan = new ArrayList<>();
            if (steps instanceof List<?> l) {
                l.forEach(s -> run.plan.add(String.valueOf(s)));
            }
            if (run.plan.isEmpty()) {
                throw new java.util.NoSuchElementException();
            }
            if (run.executed >= run.max) {
                exhausted(run, question, listener);
                return;
            }
            execute(run, question, listener);
        }, listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    private void model(Run run, Map<String, String> params, ActionListener<String> listener) {
        ml.predict(null, run.modelId, Map.of("parameters", new LinkedHashMap<String, Object>(params)), ActionListener.wrap(answer -> {
            Map<String, Object> first = (Map<String, Object>) ((List<?>) ((Map<String, Object>) ((List<?>) answer.get("inference_results")).get(0)).get("output")).get(0);
            listener.onResponse(MlChatRunner.answer(run.style, (Map<String, Object>) first.get("dataAsMap")));
        }, listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    private void execute(Run run, String question, ActionListener<Map<String, Object>> listener) {
        String step = run.plan.get(0);
        int n = ++run.executed;
        Map<String, String> params = new LinkedHashMap<>();
        params.put("question", step);
        params.put("prompt", step);
        params.put("system_prompt", run.params.getOrDefault("executor_system_prompt", PROMPTS.get("executor_system_prompt")));
        params.put("max_iteration", run.params.getOrDefault("executor_max_iterations", DEFAULT_EXECUTOR_ITERATIONS));
        params.put("message_history_limit", run.params.getOrDefault("executor_message_history_limit", DEFAULT_HISTORY));
        if (run.executorMemory != null) {
            params.put("memory_id", run.executorMemory);
        }
        Map<String, Object> request = new LinkedHashMap<>();
        request.put("parameters", new LinkedHashMap<String, Object>(params));
        chat.run(run.executor, request, run.chain, ActionListener.wrap(answer -> {
            List<Map<String, Object>> output = (List<Map<String, Object>>) ((Map<String, Object>) ((List<?>) answer.get("inference_results")).get(0)).get("output");
            String result = "";
            for (Map<String, Object> o : output) {
                switch (String.valueOf(o.get("name"))) {
                    case "memory_id" -> run.executorMemory = String.valueOf(o.get("result"));
                    case "parent_interaction_id" -> run.executorParent = String.valueOf(o.get("result"));
                    case "response" -> result = String.valueOf(((Map<String, Object>) o.get("dataAsMap")).get("response"));
                    default -> { }
                }
            }
            String stepBlock = "\n<step-" + n + ">\n" + step + "\n</step-" + n + ">\n";
            String resultText = result + "\n<step-traces>\n" + "" + "\n</step-traces>\n";
            String resultBlock = "\n<step-" + n + "-result>\n" + resultText + "\n</step-" + n + "-result>\n";
            run.completed.add(stepBlock);
            run.completed.add(resultBlock);
            run.lastStep = stepBlock;
            run.lastResult = resultBlock;
            Map<String, Object> trace = new LinkedHashMap<>();
            trace.put("input", step);
            trace.put("response", resultText);
            trace.put("origin", "PlanExecuteReflect Agent");
            trace.put("parent_message_id", run.parent);
            trace.put("trace_number", ++run.trace);
            ml.memory().addMessage(run.memoryId, trace, ActionListener.wrap(id -> {
                if (run.executed >= run.max) {
                    exhausted(run, question, listener);
                } else {
                    plan(run, question, listener);
                }
            }, listener::onFailure));
        }, listener::onFailure));
    }

    /** The step budget is spent: the model is asked for a summary, and the answer says where the run stopped. */
    private void exhausted(Run run, String question, ActionListener<Map<String, Object>> listener) {
        Map<String, String> params = new LinkedHashMap<>(run.params);
        params.put("system_prompt", PROMPTS.get("summary_system_prompt"));
        params.put("prompt", "Objective: " + question + "\n\nCompleted Steps:\n" + String.join("\n", run.completed));
        params.remove("tool_configs");
        params.putIfAbsent("no_escape_params", "tool_configs,_tools");
        String text = "Max Steps Limit (" + run.max + ") Reached. Use memory_id with same task to restart. \n Last executed step: " + run.lastStep
            + ", \n Last executed step result: " + run.lastResult;
        ActionListener<String> done = ActionListener.wrap(t -> finish(run, text, listener), listener::onFailure);
        model(run, params, ActionListener.wrap(summary -> done.onResponse(text), e -> done.onResponse(text)));
    }

    private void finish(Run run, String result, ActionListener<Map<String, Object>> listener) {
        Map<String, Object> update = new LinkedHashMap<>();
        update.put("response", result);
        ml.memory().updateMessage(run.parent, update, ActionListener.wrap(r -> {
            List<Map<String, Object>> output = new ArrayList<>();
            output.add(Map.of("name", "memory_id", "result", run.memoryId));
            output.add(Map.of("name", "parent_interaction_id", "result", run.parent));
            if (run.executorMemory != null) {
                output.add(Map.of("name", "executor_agent_memory_id", "result", run.executorMemory));
                output.add(Map.of("name", "executor_agent_parent_interaction_id", "result", run.executorParent));
            }
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("name", "response");
            response.put("dataAsMap", Map.of("response", result));
            output.add(response);
            listener.onResponse(Map.of("inference_results", List.of(Map.of("output", output))));
        }, listener::onFailure));
    }

    /** The planner's JSON, possibly inside a markdown fence. */
    private static Map<String, Object> parse(String text) {
        String t = text.strip();
        int fence = t.indexOf("```json");
        if (fence >= 0) {
            int from = fence + "```json".length();
            int end = t.indexOf("```", from);
            t = (end >= 0 ? t.substring(from, end) : t.substring(from)).strip();
        }
        Map<String, Object> parsed;
        try {
            parsed = MlCommons.parse(t.getBytes(StandardCharsets.UTF_8));
        } catch (RuntimeException e) {
            throw new IllegalStateException("Failed to parse LLM output due to invalid JSON");
        }
        if (parsed.containsKey("steps") == false && parsed.containsKey("result") == false) {
            throw new IllegalArgumentException("Missing required fields 'steps' and 'result' in JSON response");
        }
        return parsed;
    }

    @SuppressWarnings("unchecked")
    private static String toolsPrompt(Run run) {
        StringBuilder out = new StringBuilder("In this environment, you have access to the tools listed below. Use these tools to create your plan, and do not reference or use any tools not listed here.\n");
        List<Map<String, Object>> tools = run.agent.get("tools") instanceof List<?> l ? (List<Map<String, Object>>) l : List.of();
        int n = 0;
        for (Map<String, Object> spec : tools) {
            String type = String.valueOf(spec.get("type"));
            String name = spec.get("name") == null ? type : String.valueOf(spec.get("name"));
            String description = spec.get("description") == null ? null : String.valueOf(spec.get("description"));
            if (description == null) {
                for (Map<String, Object> t : MlAgents.tools()) {
                    if (type.equals(t.get("name"))) {
                        description = String.valueOf(t.get("description"));
                    }
                }
            }
            out.append("Tool ").append(++n).append(" - ").append(name).append(": ").append(description).append("\n\n");
        }
        return out.append("No other tools are available. Do not invent tools. Only use tools to create the plan.\n\n").toString();
    }

    private static String substitute(String template, Map<String, String> params) {
        String out = template;
        for (Map.Entry<String, String> e : params.entrySet()) {
            out = out.replace("${parameters." + e.getKey() + "}", e.getValue());
        }
        return out;
    }
}
