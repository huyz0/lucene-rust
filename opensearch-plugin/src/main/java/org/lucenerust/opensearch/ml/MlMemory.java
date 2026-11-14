/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.OpenSearchParseException;
import org.opensearch.ResourceNotFoundException;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.action.bulk.BulkRequest;
import org.opensearch.action.delete.DeleteRequest;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.core.action.ActionListener;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.sort.SortOrder;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * ml-commons' conversation memory ({@code memory} REST routes over {@code .plugins-ml-memory-meta}
 * and {@code .plugins-ml-memory-message}): a memory is a conversation, its messages the turns, and
 * a message's traces the steps an agent took to answer it.
 */
final class MlMemory {
    private static final int DEFAULT_MAX = 10;

    private final MlCommons ml;

    MlMemory(MlCommons ml) {
        this.ml = ml;
    }

    // ---- Memories -----------------------------------------------------------------------------

    /** {@code POST memory}: the new memory's id. */
    void create(Map<String, Object> body, ActionListener<String> listener) {
        Map<String, Object> doc = new LinkedHashMap<>();
        String now = Instant.now().toString();
        doc.put("updated_time", now);
        doc.put("create_time", now);
        put(doc, "application_type", string(body.get("application_type")));
        put(doc, "additional_info", map(body.get("additional_info")));
        put(doc, "name", string(body.get("name")));
        ml.indexDoc(MlIndices.MEMORY_META, null, doc, ActionListener.wrap(r -> listener.onResponse(r.getId()), listener::onFailure));
    }

    /** {@code GET memory/<id>}. */
    void get(String id, ActionListener<Map<String, Object>> listener) {
        memory(id, ActionListener.wrap(doc -> listener.onResponse(memoryView(id, doc)), listener::onFailure));
    }

    private void memory(String id, ActionListener<Map<String, Object>> listener) {
        ml.getDoc(MlIndices.MEMORY_META, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw notFound("Memory [" + id + "] not found");
            }
            listener.onResponse(r.getSourceAsMap());
        }, e -> listener.onFailure(e instanceof org.opensearch.index.IndexNotFoundException ? notFound("Memory [" + id + "] not found") : e)));
    }

    private static Map<String, Object> memoryView(String id, Map<String, Object> doc) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("memory_id", id);
        out.put("create_time", doc.get("create_time"));
        out.put("updated_time", doc.get("updated_time"));
        out.put("name", (String) doc.get("name"));
        put(out, "additional_info", castMap(doc.get("additional_info")));
        put(out, "user", (String) doc.get("user"));
        return out;
    }

    /** {@code GET memory}: newest first. */
    void list(String maxResults, String nextToken, ActionListener<Map<String, Object>> listener) {
        int max = integer(maxResults, DEFAULT_MAX);
        int from = integer(nextToken, 0);
        if (max <= 0) {
            ActionRequestValidationException e = new ActionRequestValidationException();
            e.addValidationError("Can't list 0 or negative conversations");
            throw e;
        }
        SearchSourceBuilder source = new SearchSourceBuilder().query(QueryBuilders.matchAllQuery())
            .sort("updated_time", SortOrder.DESC)
            .from(from)
            .size(max);
        ml.searchDocs(new SearchRequest(MlIndices.MEMORY_META).source(source), ActionListener.wrap(r -> {
            List<Object> memories = new ArrayList<>();
            for (SearchHit hit : r.getHits().getHits()) {
                memories.add(memoryView(hit.getId(), hit.getSourceAsMap()));
            }
            listener.onResponse(page("memories", memories, max, from));
        }, e -> listener.onFailure(e instanceof org.opensearch.index.IndexNotFoundException ? notFoundEmpty() : e)));
    }

    /** {@code PUT memory/<id>}. */
    @SuppressWarnings("unchecked")
    void update(String id, Map<String, Object> body, ActionListener<UpdateResponse> listener) {
        if (body == null) {
            throw new OpenSearchParseException("Failed to update conversation: Request body is empty");
        }
        memory(id, ActionListener.wrap(doc -> {
            Map<String, Object> fields = new LinkedHashMap<>();
            put(fields, "name", body.get("name"));
            if (body.get("additional_info") instanceof Map<?, ?> given) {
                Map<String, Object> merged = doc.get("additional_info") instanceof Map<?, ?> old ? new LinkedHashMap<>((Map<String, Object>) old) : new LinkedHashMap<>();
                merged.putAll((Map<String, Object>) given);
                fields.put("additional_info", merged);
            }
            fields.put("updated_time", Instant.now().toString());
            ml.updateDoc(MlIndices.MEMORY_META, id, fields, listener);
        }, listener::onFailure));
    }

    /** {@code DELETE memory/<id>}: the memory and its messages. */
    void delete(String id, ActionListener<Map<String, Object>> listener) {
        memory(id, ActionListener.wrap(doc -> deleteMessages(id, ActionListener.wrap(v -> {
            DeleteRequest request = new DeleteRequest(MlIndices.MEMORY_META, id).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            MlIndices.<org.opensearch.action.delete.DeleteResponse>system(ml.client(), l -> ml.client().delete(request, l), ActionListener.wrap(r -> listener.onResponse(Map.of("success", true)), listener::onFailure));
        }, listener::onFailure)), listener::onFailure));
    }

    private void deleteMessages(String memoryId, ActionListener<Void> done) {
        SearchSourceBuilder source = new SearchSourceBuilder().query(QueryBuilders.boolQuery().filter(QueryBuilders.termQuery("memory_id", memoryId)).mustNot(QueryBuilders.existsQuery("trace_number"))).size(1000);
        ml.searchDocs(new SearchRequest(MlIndices.MEMORY_MESSAGE).source(source), ActionListener.wrap(r -> {
            if (r.getHits().getHits().length == 0) {
                done.onResponse(null);
                return;
            }
            BulkRequest bulk = new BulkRequest().setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            for (SearchHit hit : r.getHits().getHits()) {
                messageView(hit.getId(), hit.getSourceAsMap(), true);
            }
            for (SearchHit hit : r.getHits().getHits()) {
                bulk.add(new DeleteRequest(MlIndices.MEMORY_MESSAGE, hit.getId()));
            }
            MlIndices.<org.opensearch.action.bulk.BulkResponse>system(ml.client(), l -> ml.client().bulk(bulk, l), ActionListener.wrap(b -> {
                if (b.hasFailures()) {
                    done.onFailure(new org.opensearch.OpenSearchException(b.buildFailureMessage()));
                } else {
                    deleteMessages(memoryId, done);
                }
            }, done::onFailure));
        }, e -> {
            if (e instanceof org.opensearch.index.IndexNotFoundException) {
                done.onResponse(null);
            } else {
                done.onFailure(e);
            }
        }));
    }

    // ---- Messages -----------------------------------------------------------------------------

    /** {@code POST memory/<id>/messages}: the new message's id. */
    void addMessage(String memoryId, Map<String, Object> body, ActionListener<String> listener) {
        if (body == null) {
            throw new OpenSearchParseException("request body is required");
        }
        String input = text(body.get("input"));
        String template = text(body.get("prompt_template"));
        String response = text(body.get("response"));
        String origin = text(body.get("origin"));
        Map<String, Object> info = body.get("additional_info") instanceof Map<?, ?> m ? strings(m) : null;
        Integer trace = traceNumber(body.get("trace_number"));
        String parent = text(body.get("parent_message_id"));
        if (body.get("input") instanceof Map) {
            throw new IllegalStateException("Can't get text on a START_OBJECT at 1:1");
        }
        if (empty(input) && empty(template) && empty(response) && empty(origin) && (info == null || info.isEmpty())) {
            throw new IllegalArgumentException(
                "At least one of the following parameters must be non-empty: input, prompt_template, response, origin, additional_info"
            );
        }
        memory(memoryId, ActionListener.wrap(doc -> {
            Map<String, Object> message = new LinkedHashMap<>();
            String now = Instant.now().toString();
            message.put("updated_time", now);
            put(message, "input", input);
            message.put("memory_id", memoryId);
            message.put("trace_number", trace);
            message.put("create_time", now);
            put(message, "response", response);
            message.put("parent_message_id", trace == null ? null : parent);
            put(message, "prompt_template", template);
            put(message, "origin", origin);
            put(message, "additional_info", info);
            ml.indexDoc(MlIndices.MEMORY_MESSAGE, null, message, ActionListener.wrap(r -> ml.updateDoc(
                MlIndices.MEMORY_META,
                memoryId,
                Map.of("updated_time", now),
                ActionListener.wrap(u -> listener.onResponse(r.getId()), listener::onFailure)
            ), listener::onFailure));
        }, listener::onFailure));
    }

    /** {@code GET memory/<id>/messages}: the turns, oldest first (their traces are listed per message). */
    void messages(String memoryId, String maxResults, String nextToken, ActionListener<Map<String, Object>> listener) {
        int max = integer(maxResults, DEFAULT_MAX);
        int from = integer(nextToken, 0);
        check(max, from, "interactions");
        memory(memoryId, ActionListener.wrap(doc -> {
            BoolQueryBuilder query = QueryBuilders.boolQuery()
                .filter(QueryBuilders.termQuery("memory_id", memoryId))
                .mustNot(QueryBuilders.existsQuery("trace_number"));
            page(query, "create_time", from, max, ActionListener.wrap(hits -> {
                List<Object> out = new ArrayList<>();
                for (SearchHit hit : hits) {
                    out.add(messageView(hit.getId(), hit.getSourceAsMap(), false));
                }
                listener.onResponse(page("messages", out, max, from));
            }, listener::onFailure));
        }, listener::onFailure));
    }

    /** The last {@code limit} turns of a memory, oldest first, for an agent's chat history. */
    void recent(String memoryId, int limit, ActionListener<List<Map<String, Object>>> listener) {
        memory(memoryId, ActionListener.wrap(doc -> {
            BoolQueryBuilder query = QueryBuilders.boolQuery()
                .filter(QueryBuilders.termQuery("memory_id", memoryId))
                .mustNot(QueryBuilders.existsQuery("trace_number"));
            SearchSourceBuilder source = new SearchSourceBuilder().query(query).sort("create_time", SortOrder.DESC).size(limit);
            ml.searchDocs(new SearchRequest(MlIndices.MEMORY_MESSAGE).source(source), ActionListener.wrap(r -> {
                List<Map<String, Object>> out = new ArrayList<>();
                for (SearchHit hit : r.getHits().getHits()) {
                    out.add(0, hit.getSourceAsMap());
                }
                listener.onResponse(out);
            }, e -> {
                if (e instanceof org.opensearch.index.IndexNotFoundException) {
                    listener.onResponse(List.of());
                } else {
                    listener.onFailure(e);
                }
            }));
        }, listener::onFailure));
    }

    /** {@code GET memory/message/<id>}. */
    void message(String id, ActionListener<Map<String, Object>> listener) {
        message(id, "Message [" + id + "] not found", ActionListener.wrap(doc -> listener.onResponse(messageView(id, doc, true)), listener::onFailure));
    }

    private void message(String id, String missing, ActionListener<Map<String, Object>> listener) {
        ml.getDoc(MlIndices.MEMORY_MESSAGE, id, ActionListener.wrap(r -> {
            if (r.isExists() == false) {
                throw notFound(missing);
            }
            listener.onResponse(r.getSourceAsMap());
        }, e -> listener.onFailure(e instanceof org.opensearch.index.IndexNotFoundException ? notFound(missing) : e)));
    }

    /** {@code PUT memory/message/<id>}. */
    void updateMessage(String id, Map<String, Object> body, ActionListener<UpdateResponse> listener) {
        if (body == null) {
            throw new OpenSearchParseException("Failed to update interaction: Request body is empty");
        }
        message(id, "message [" + id + "] not found", ActionListener.wrap(doc -> {
            messageView(id, doc, true);
            Map<String, Object> fields = new LinkedHashMap<>();
            put(fields, "response", body.get("response"));
            if (body.get("additional_info") instanceof Map<?, ?> given) {
                Map<String, Object> merged = doc.get("additional_info") instanceof Map<?, ?> old ? new LinkedHashMap<>(castMap(old)) : new LinkedHashMap<>();
                merged.putAll(castMap(given));
                fields.put("additional_info", merged);
            }
            fields.put("updated_time", Instant.now().toString());
            ml.updateDoc(MlIndices.MEMORY_MESSAGE, id, fields, listener);
        }, listener::onFailure));
    }

    /** {@code GET memory/message/<id>/traces}: the steps taken to answer a message, in order. */
    void traces(String id, String maxResults, String nextToken, ActionListener<Map<String, Object>> listener) {
        int max = integer(maxResults, DEFAULT_MAX);
        int from = integer(nextToken, 0);
        check(max, from, "traces");
        BoolQueryBuilder query = QueryBuilders.boolQuery().filter(QueryBuilders.termQuery("parent_message_id", id)).filter(QueryBuilders.existsQuery("trace_number"));
        message(id, "Message [" + id + "] not found", ActionListener.wrap(doc -> page(query, "trace_number", from, max, ActionListener.wrap(hits -> {
            List<Object> out = new ArrayList<>();
            for (SearchHit hit : hits) {
                out.add(messageView(hit.getId(), hit.getSourceAsMap(), true));
            }
            listener.onResponse(page("traces", out, max, from));
        }, listener::onFailure)), listener::onFailure));
    }

    /** {@code POST memory/<id>/_search}: a search of that memory's messages. */
    void searchMessages(String memoryId, SearchSourceBuilder source, ActionListener<SearchResponse> listener) {
        memory(memoryId, ActionListener.wrap(doc -> {
            BoolQueryBuilder query = QueryBuilders.boolQuery().must(QueryBuilders.termQuery("memory_id", memoryId));
            query.must(source.query());
            source.query(query);
            ml.searchDocs(new SearchRequest(MlIndices.MEMORY_MESSAGE).source(source), listener);
        }, listener::onFailure));
    }

    // ---- Helpers ------------------------------------------------------------------------------

    private void page(BoolQueryBuilder query, String sort, int from, int size, ActionListener<SearchHit[]> listener) {
        SearchSourceBuilder source = new SearchSourceBuilder().query(query).sort(sort, SortOrder.ASC).from(from).size(size);
        ml.searchDocs(new SearchRequest(MlIndices.MEMORY_MESSAGE).source(source), ActionListener.wrap(r -> listener.onResponse(r.getHits().getHits()), e -> {
            if (e instanceof org.opensearch.index.IndexNotFoundException) {
                listener.onResponse(new SearchHit[0]);
            } else {
                listener.onFailure(e);
            }
        }));
    }

    private static Map<String, Object> messageView(String id, Map<String, Object> doc, boolean steps) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("memory_id", doc.get("memory_id"));
        out.put("message_id", id);
        out.put("create_time", doc.get("create_time"));
        out.put("updated_time", doc.get("updated_time"));
        put(out, "input", (String) doc.get("input"));
        put(out, "prompt_template", (String) doc.get("prompt_template"));
        put(out, "response", (String) doc.get("response"));
        put(out, "origin", (String) doc.get("origin"));
        put(out, "additional_info", castMap(doc.get("additional_info")));
        if (steps) {
            put(out, "parent_message_id", (String) doc.get("parent_message_id"));
            put(out, "trace_number", doc.get("trace_number"));
        }
        return out;
    }

    /** A page of results; {@code next_token} when the page is full, the position to continue from. */
    private static Map<String, Object> page(String key, List<Object> items, int max, int from) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(key, items);
        if (items.size() == max) {
            out.put("next_token", from + max);
        }
        return out;
    }

    private static void check(int max, int from, String what) {
        List<String> problems = new ArrayList<>();
        if (max <= 0) {
            problems.add("The number of " + what + " to retrieve must be positive");
        }
        if (from < 0) {
            problems.add("The starting position must be nonnegative");
        }
        if (problems.isEmpty() == false) {
            ActionRequestValidationException e = new ActionRequestValidationException();
            problems.forEach(e::addValidationError);
            throw e;
        }
    }

    private static int integer(String v, int otherwise) {
        return v == null ? otherwise : Integer.parseInt(v);
    }

    private static boolean empty(String s) {
        return s == null || s.isEmpty();
    }

    private static void put(Map<String, Object> m, String key, Object value) {
        if (value != null) {
            m.put(key, value);
        }
    }

    private static Exception notFoundEmpty() {
        return new org.opensearch.index.IndexNotFoundException(MlIndices.MEMORY_META);
    }

    private static ResourceNotFoundException notFound(String message) {
        return new ResourceNotFoundException(message);
    }

    private static String string(Object v) {
        if (v == null || v instanceof String) {
            return (String) v;
        }
        throw cast(v, "java.lang.String");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object v) {
        if (v == null || v instanceof Map) {
            return (Map<String, Object>) v;
        }
        throw cast(v, "java.util.Map");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castMap(Object v) {
        return (Map<String, Object>) v;
    }

    /** A scalar as the request parser's {@code text()} reads it. */
    private static String text(Object v) {
        return v == null || v instanceof Map || v instanceof List ? null : String.valueOf(v);
    }

    /** The values of an {@code additional_info} object as the request parser reads them: text. */
    private static Map<String, Object> strings(Map<?, ?> m) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : m.entrySet()) {
            Object v = e.getValue();
            if (v instanceof Map || v instanceof List) {
                throw new IllegalStateException("Can't get text on a START_OBJECT at 1:1");
            }
            out.put(String.valueOf(e.getKey()), v == null ? null : String.valueOf(v));
        }
        return out;
    }

    private static Integer traceNumber(Object v) {
        if (v == null) {
            return null;
        }
        if (v instanceof String) {
            throw new IllegalArgumentException("Integer value passed as String");
        }
        if (v instanceof Number n) {
            if (v instanceof Double || v instanceof Float) {
                double d = n.doubleValue();
                if (d != Math.rint(d) || d > Integer.MAX_VALUE || d < Integer.MIN_VALUE) {
                    throw new IllegalArgumentException(n + " cannot be converted to Integer without data loss");
                }
            }
            return n.intValue();
        }
        return null;
    }

    /** The parse error a wrongly typed field makes in ml-commons: Java's own cast message. */
    private static OpenSearchParseException cast(Object v, String wanted) {
        String have = v instanceof Map ? "java.util.HashMap" : v.getClass().getName();
        return new OpenSearchParseException(
            "class " + have + " cannot be cast to class " + wanted + " (" + have + " and " + wanted + " are in module java.base of loader 'bootstrap')"
        );
    }
}
