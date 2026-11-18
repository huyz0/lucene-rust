/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.ml;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.core.rest.RestStatus;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** The inputs of a predict request, parsed and refused as stock's REST layer does. */
final class MlInputs {
    private MlInputs() {}

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
     * {@code QuestionAnsweringInput}: the question and context of a {@code question_answering} predict
     * input. A value that is not text, a number or a boolean leaves stock's parser with no input at all; an absent
     * or empty one is refused as not provided.
     */
    static Map.Entry<String, String> questionAnsweringInput(Map<String, Object> body) {
        for (String key : List.of("question", "context")) {
            if (body.containsKey(key) && (body.get(key) instanceof String || body.get(key) instanceof Number || body.get(key) instanceof Boolean) == false) {
                throw new OpenSearchStatusException(
                    "Cannot invoke \"org.opensearch.ml.common.input.MLInput.setAlgorithm(org.opensearch.ml.common.FunctionName)\" because \"mlInput\" is null",
                    RestStatus.INTERNAL_SERVER_ERROR
                );
            }
        }
        String question = body.get("question") == null ? "" : String.valueOf(body.get("question"));
        String context = body.get("context") == null ? "" : String.valueOf(body.get("context"));
        if (question.isEmpty()) {
            throw new IllegalArgumentException("Question is not provided");
        }
        if (context.isEmpty()) {
            throw new IllegalArgumentException("Context is not provided");
        }
        return Map.entry(question, context);
    }
}
