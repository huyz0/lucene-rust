/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.mapper.MapperParsingException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * A {@code sparse_vector} field's {@code method}: neural-search's {@code SparseMethodContext} and
 * {@code MethodComponentContext}, with {@code Seismic.validateMethod}'s checks. Only the
 * {@code lucene} engine is served: the Rust engine writes and searches its clustered postings.
 */
public final class SparseMethod {
    static final String NAME_FIELD = "name";
    static final String ENGINE_FIELD = "engine";
    static final String PARAMETERS_FIELD = "parameters";
    static final String SEISMIC = "seismic";
    static final String ENGINE_LUCENE = "lucene";
    static final String ENGINE_NATIVE = "native";

    static final String N_POSTINGS = "n_postings";
    static final String SUMMARY_PRUNE_RATIO = "summary_prune_ratio";
    static final String CLUSTER_RATIO = "cluster_ratio";
    static final String APPROXIMATE_THRESHOLD = "approximate_threshold";
    static final String QUANTIZATION_CEILING_INGEST = "quantization_ceiling_ingest";
    static final String QUANTIZATION_CEILING_SEARCH = "quantization_ceiling_search";
    static final String FORWARD_INDEX = "forward_index";
    static final String CLUSTERING_BATCH_SIZE = "clustering_batch_size";

    static final int DEFAULT_N_POSTINGS = -1;
    static final float DEFAULT_SUMMARY_PRUNE_RATIO = 0.4f;
    static final float DEFAULT_CLUSTER_RATIO = 0.1f;
    static final int DEFAULT_APPROXIMATE_THRESHOLD = 1000000;
    static final float DEFAULT_QUANTIZATION_CEILING_INGEST = 3.0f;
    static final float DEFAULT_QUANTIZATION_CEILING_SEARCH = 16.0f;
    static final String DEFAULT_FORWARD_INDEX = "shared";
    static final int DEFAULT_CLUSTERING_BATCH_SIZE = 1;
    private static final int MIN_CLUSTERING_BATCH_SIZE = 1;
    private static final int MAX_CLUSTERING_BATCH_SIZE = 10000;

    final String name;
    final String engine;
    /** {@code null} when the mapping set {@code "parameters": null}. */
    final Map<String, Object> parameters;

    private SparseMethod(String name, String engine, Map<String, Object> parameters) {
        this.name = name;
        this.engine = engine;
        this.parameters = parameters;
    }

    /** {@code SparseMethodContext.parse}. */
    @SuppressWarnings("unchecked")
    static SparseMethod parse(Object in) {
        if (in instanceof Map<?, ?> == false) {
            throw new MapperParsingException("Unable to parse mapping into SparseMethodContext. Object not of type \"Map\"");
        }
        String name = "";
        String engine = ENGINE_LUCENE;
        Map<String, Object> parameters = new HashMap<>();
        for (Map.Entry<String, Object> e : ((Map<String, Object>) in).entrySet()) {
            Object value = e.getValue();
            switch (e.getKey()) {
                case NAME_FIELD -> name = (String) value;
                case ENGINE_FIELD -> engine = (String) value;
                case PARAMETERS_FIELD -> {
                    if (value == null) {
                        parameters = null;
                        continue;
                    }
                    if (value instanceof Map<?, ?> == false) {
                        throw new MapperParsingException("Unable to parse parameters for main method component");
                    }
                    parameters = new HashMap<>();
                    for (Map.Entry<String, Object> p : ((Map<String, Object>) value).entrySet()) {
                        if (p.getValue() instanceof Map) {
                            throw new IllegalArgumentException("Value should not be map");
                        }
                        parameters.put(p.getKey(), p.getValue());
                    }
                }
                default -> throw new MapperParsingException("Invalid parameter: " + e.getKey());
            }
        }
        if (name.isEmpty()) {
            throw new MapperParsingException(NAME_FIELD + " needs to be set");
        }
        if (ENGINE_LUCENE.equalsIgnoreCase(engine) == false && ENGINE_NATIVE.equalsIgnoreCase(engine) == false) {
            throw new MapperParsingException(ENGINE_FIELD + " needs to be valid engine");
        }
        return new SparseMethod(name, engine, parameters);
    }

    /** {@code SparseTypeParser.parse}'s checks after the method parsed. */
    void validate() {
        if (SEISMIC.equals(name) == false) {
            throw new MapperParsingException("[method.name]: " + name + " is not supported");
        }
        List<String> errors = new ArrayList<>();
        Map<String, Object> left = new HashMap<>(parameters());
        number(left, errors, SUMMARY_PRUNE_RATIO, false, v -> v <= 0 || v > 1, "must be in (0, 1]");
        number(left, errors, N_POSTINGS, true, v -> v <= 0, "must be a positive integer");
        number(left, errors, CLUSTER_RATIO, false, v -> v <= 0 || v >= 1, "must be in (0, 1)");
        number(left, errors, APPROXIMATE_THRESHOLD, true, v -> v < 0, "must be a non-negative integer");
        number(left, errors, QUANTIZATION_CEILING_INGEST, false, v -> v <= 0, "must be a positive float number");
        number(left, errors, QUANTIZATION_CEILING_SEARCH, false, v -> v <= 0, "must be a positive float number");
        if (left.containsKey(FORWARD_INDEX)) {
            String v = String.valueOf(left.remove(FORWARD_INDEX));
            if ("shared".equalsIgnoreCase(v) == false && "per_block".equalsIgnoreCase(v) == false) {
                errors.add(String.format(Locale.ROOT, "Parameter [%s] must be one of [shared, per_block]", FORWARD_INDEX));
            }
        }
        number(
            left,
            errors,
            CLUSTERING_BATCH_SIZE,
            true,
            v -> v < MIN_CLUSTERING_BATCH_SIZE || v > MAX_CLUSTERING_BATCH_SIZE,
            String.format(Locale.ROOT, "must be in [%d, %d]", MIN_CLUSTERING_BATCH_SIZE, MAX_CLUSTERING_BATCH_SIZE)
        );
        for (String key : left.keySet()) {
            errors.add(String.format(Locale.ROOT, "Unknown parameter '%s' found", key));
        }
        if (errors.isEmpty() == false) {
            // ValidationException's message.
            StringBuilder sb = new StringBuilder("Validation Failed: ");
            for (int i = 0; i < errors.size(); i++) {
                sb.append(i + 1).append(": ").append(errors.get(i)).append(';');
            }
            throw new MapperParsingException(sb.toString());
        }
        if (ENGINE_NATIVE.equalsIgnoreCase(engine)) {
            throw new MapperParsingException("[" + ENGINE_FIELD + "]: the native engine is not supported by the lucene-rust engine");
        }
        if (DEFAULT_FORWARD_INDEX.equals(forwardIndex()) == false) {
            throw new MapperParsingException("[" + FORWARD_INDEX + "]: " + forwardIndex() + " is only supported with the native engine");
        }
        if (String.valueOf(DEFAULT_CLUSTERING_BATCH_SIZE).equals(clusteringBatchSize()) == false) {
            throw new MapperParsingException(
                "[" + CLUSTERING_BATCH_SIZE + "]: " + clusteringBatchSize() + " is only supported with the native engine"
            );
        }
    }

    private interface Check {
        boolean bad(double v);
    }

    /** One of {@code Seismic.validate*}: a number, parsed from its string form, in range. */
    private static void number(Map<String, Object> left, List<String> errors, String key, boolean integer, Check check, String range) {
        if (left.containsKey(key) == false) {
            return;
        }
        String text = String.valueOf(left.remove(key));
        try {
            double v = integer ? Integer.decode(text.trim()) : Float.parseFloat(text);
            if (check.bad(v)) {
                errors.add(String.format(Locale.ROOT, "Parameter [%s] %s", key, range));
            }
        } catch (NumberFormatException e) {
            errors.add(
                String.format(
                    Locale.ROOT,
                    "Parameter [%s] must be of %s type",
                    key,
                    integer ? Integer.class.getName() : Float.class.getName()
                )
            );
        }
    }

    Map<String, Object> parameters() {
        return parameters == null ? Map.of() : parameters;
    }

    private Object parameter(String key, Object defaultValue) {
        return parameters().getOrDefault(key, defaultValue);
    }

    /** {@code MethodComponentContext.getFloatParameter}: a non-number falls back to the default. */
    private float floatParameter(String key, float defaultValue) {
        Object v = parameter(key, defaultValue);
        return v instanceof Number n ? n.floatValue() : defaultValue;
    }

    private String forwardIndex() {
        return String.valueOf(parameter(FORWARD_INDEX, DEFAULT_FORWARD_INDEX));
    }

    private String clusteringBatchSize() {
        return String.valueOf(parameter(CLUSTERING_BATCH_SIZE, DEFAULT_CLUSTERING_BATCH_SIZE));
    }

    /**
     * {@code SparseVectorFieldMapper.setFieldTypeAttributes}: what the field's {@code FieldInfo}
     * carries, which the Rust writer clusters by and neural-search reads back.
     */
    Map<String, String> attributes() {
        Map<String, String> a = new LinkedHashMap<>();
        a.put(N_POSTINGS, String.valueOf(parameter(N_POSTINGS, DEFAULT_N_POSTINGS)));
        a.put(SUMMARY_PRUNE_RATIO, String.valueOf(floatParameter(SUMMARY_PRUNE_RATIO, DEFAULT_SUMMARY_PRUNE_RATIO)));
        a.put(CLUSTER_RATIO, String.valueOf(floatParameter(CLUSTER_RATIO, DEFAULT_CLUSTER_RATIO)));
        a.put(APPROXIMATE_THRESHOLD, String.valueOf(parameter(APPROXIMATE_THRESHOLD, DEFAULT_APPROXIMATE_THRESHOLD)));
        a.put(QUANTIZATION_CEILING_INGEST, String.valueOf(floatParameter(QUANTIZATION_CEILING_INGEST, DEFAULT_QUANTIZATION_CEILING_INGEST)));
        a.put(QUANTIZATION_CEILING_SEARCH, String.valueOf(floatParameter(QUANTIZATION_CEILING_SEARCH, DEFAULT_QUANTIZATION_CEILING_SEARCH)));
        a.put(ENGINE_FIELD, engine);
        a.put(FORWARD_INDEX, forwardIndex());
        a.put(CLUSTERING_BATCH_SIZE, clusteringBatchSize());
        return a;
    }

    /** {@code SparseMethodContext.toXContent}: a default engine is left out. */
    void toXContent(XContentBuilder b, ToXContent.Params params) throws IOException {
        if (engine != null && ENGINE_LUCENE.equalsIgnoreCase(engine) == false) {
            b.field(ENGINE_FIELD, engine);
        }
        b.field(NAME_FIELD, name);
        if (parameters == null) {
            b.field(PARAMETERS_FIELD, (String) null);
        } else {
            b.startObject(PARAMETERS_FIELD);
            for (Map.Entry<String, Object> e : parameters.entrySet()) {
                b.field(e.getKey(), e.getValue());
            }
            b.endObject();
        }
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof SparseMethod m
            && name.equals(m.name)
            && Objects.equals(engine, m.engine)
            && Objects.equals(parameters, m.parameters);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, engine, parameters);
    }

    @Override
    public String toString() {
        return name + "/" + engine + parameters;
    }
}
