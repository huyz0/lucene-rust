/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.search.Query;
import org.lucenerust.opensearch.sparse.SparseVectorQuery;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * The feature and sparse nodes of the query tree ({@code queries.rs}):
 *
 * <ul>
 *   <li>{@code FeatureField}'s {@code FeatureQuery} ({@code rank_feature}, {@code neural_sparse} on
 *       a {@code rank_features} field, a sparse search's fallback), node 37. Lucene's class is
 *       package-private and read through {@link Reflect}, as {@link DocValuesEncoder} reads its.
 *   <li>{@link SparseVectorQuery} ({@code neural_sparse} on a {@code sparse_vector} field), node 38.
 * </ul>
 */
final class SparseEncoder {
    static final byte NODE_FEATURE = 37;
    static final byte NODE_SPARSE_ANN = 38;

    /** The feature node's functions. */
    static final byte FEATURE_LINEAR = 0;
    static final byte FEATURE_LOG = 1;
    static final byte FEATURE_SATURATION = 2;
    static final byte FEATURE_SIGMOID = 3;

    /** What {@link #node} returns for a query that is neither. */
    static final String NOT_SPARSE = "not_sparse";

    private static final String FEATURE_QUERY = "org.apache.lucene.document.FeatureQuery";
    private static final String FUNCTION = "org.apache.lucene.document.FeatureField$";

    private SparseEncoder() {}

    /**
     * Appends {@code q} when it is one of these queries: null when it was written, a fallback reason
     * when it cannot be, {@link #NOT_SPARSE} when it is neither.
     */
    static String node(Query q, ByteArrayOutputStream out, ExtendedEncoder.Child child) {
        if (q instanceof SparseVectorQuery s) {
            return sparse(s, out, child);
        }
        if (q.getClass().getName().equals(FEATURE_QUERY) == false) {
            return NOT_SPARSE;
        }
        try {
            Object function = ExtendedEncoder.field(q, "function");
            ByteArrayOutputStream f = new ByteArrayOutputStream();
            switch (function.getClass().getName()) {
                case FUNCTION + "LinearFunction" -> f.write(FEATURE_LINEAR);
                case FUNCTION + "LogFunction" -> {
                    f.write(FEATURE_LOG);
                    writeFloat(f, (float) ExtendedEncoder.field(function, "scalingFactor"));
                }
                case FUNCTION + "SaturationFunction" -> {
                    // Set by FeatureQuery.rewrite, which runs before encoding.
                    Float pivot = (Float) ExtendedEncoder.field(function, "pivot");
                    if (pivot == null) {
                        return "clause_FeatureQuery_unrewritten";
                    }
                    f.write(FEATURE_SATURATION);
                    writeFloat(f, pivot);
                }
                case FUNCTION + "SigmoidFunction" -> {
                    f.write(FEATURE_SIGMOID);
                    writeFloat(f, (float) ExtendedEncoder.field(function, "pivot"));
                    writeFloat(f, (float) ExtendedEncoder.field(function, "a"));
                }
                default -> {
                    return "clause_FeatureQuery";
                }
            }
            out.write(NODE_FEATURE);
            writeString(out, (String) ExtendedEncoder.field(q, "fieldName"));
            writeString(out, (String) ExtendedEncoder.field(q, "featureName"));
            out.writeBytes(f.toByteArray());
            return null;
        } catch (ReflectiveOperationException | ClassCastException e) {
            return "clause_FeatureQuery";
        }
    }

    /**
     * {@code NODE_SPARSE_ANN}: the field, the tokens with their weights, {@code top_n}, {@code k},
     * {@code heap_factor}, the filter when there is one, then the fallback.
     */
    private static String sparse(SparseVectorQuery q, ByteArrayOutputStream out, ExtendedEncoder.Child child) {
        out.write(NODE_SPARSE_ANN);
        writeString(out, q.field());
        writeInt(out, q.tokens().size());
        for (Map.Entry<Integer, Float> t : q.tokens().entrySet()) {
            writeInt(out, t.getKey());
            writeFloat(out, t.getValue());
        }
        writeInt(out, q.topN());
        writeInt(out, q.k());
        writeFloat(out, q.heapFactor());
        if (q.filter() == null) {
            out.write(0);
        } else {
            out.write(1);
            String reason = child.node(q.filter());
            if (reason != null) {
                return reason;
            }
        }
        return child.node(q.fallback());
    }

    private static void writeString(ByteArrayOutputStream out, String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        writeInt(out, b.length);
        out.writeBytes(b);
    }

    private static void writeFloat(ByteArrayOutputStream out, float v) {
        writeInt(out, Float.floatToIntBits(v));
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }
}
