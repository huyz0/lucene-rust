/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.script;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.lucenerust.opensearch.knn.KnnScriptDocValues;

import java.math.BigInteger;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * k-NN's {@code KNNScoringUtil}: the distances its {@code knn_score} spaces and its Painless
 * functions ({@code l2Squared}, {@code l1Norm}, {@code lInfNorm}, {@code innerProduct}, {@code
 * cosineSimilarity}, {@code hamming}, and {@code KNNPainlessScriptUtils.lateInteractionScore})
 * compute, with the same checks and messages.
 */
public final class KnnScoring {
    private KnnScoring() {}

    private static void requireEqualDimension(int query, int input) {
        if (query != input) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "query vector dimension mismatch. Expected: %d, Given: %d", input, query)
            );
        }
    }

    private static void requireNonBinary(String space, KnnScriptDocValues values) {
        if (values.dataType().equals("binary")) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Incompatible field_type for %s space. The data type should be either float or byte but got %s",
                    space,
                    values.dataType()
                )
            );
        }
    }

    private static void requireBinary(String space, KnnScriptDocValues values) {
        if (values.dataType().equals("binary") == false) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Incompatible field_type for %s space. The data type should be binary but got %s", space, values.dataType())
            );
        }
    }

    /** {@code KNNValidationUtil.validateByteVectorValue}; {@code dataType} as k-NN spells it, {@code byte} or {@code binary}. */
    static void checkByte(float value, String dataType) {
        if (Float.isNaN(value)) {
            throw new IllegalArgumentException("KNN vector values cannot be NaN");
        }
        if (Float.isInfinite(value)) {
            throw new IllegalArgumentException("KNN vector values cannot be infinity");
        }
        if (value % 1 != 0) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[data_type] field was set as [%s] in index mapping. But, KNN vector values are floats instead of byte integers",
                    dataType
                )
            );
        }
        if ((int) value < Byte.MIN_VALUE || (int) value > Byte.MAX_VALUE) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "[data_type] field was set as [%s] in index mapping. But, KNN vector values are not within in the byte range [%d, %d]",
                    dataType,
                    Byte.MIN_VALUE,
                    Byte.MAX_VALUE
                )
            );
        }
    }

    static float[] toFloats(List<?> vector, String dataType) {
        Objects.requireNonNull(vector);
        float[] out = new float[vector.size()];
        for (int i = 0; i < out.length; i++) {
            float v = ((Number) vector.get(i)).floatValue();
            if (dataType.equals("byte") || dataType.equals("binary")) {
                checkByte(v, dataType);
            }
            out[i] = v;
        }
        return out;
    }

    static byte[] toBytes(List<?> vector, String dataType) {
        Objects.requireNonNull(vector);
        byte[] out = new byte[vector.size()];
        for (int i = 0; i < out.length; i++) {
            Number n = (Number) vector.get(i);
            if (dataType.equals("byte") || dataType.equals("binary")) {
                checkByte(n.floatValue(), dataType);
            }
            out[i] = n.byteValue();
        }
        return out;
    }

    /** {@code SpaceType.COSINESIMIL.validateVector}. */
    static void checkNotZero(float[] v) {
        for (float f : v) {
            if (f != 0) {
                return;
            }
        }
        throw new IllegalArgumentException("zero vector is not supported when space type is [cosinesimil]");
    }

    static void checkNotZero(byte[] v) {
        for (byte b : v) {
            if (b != 0) {
                return;
            }
        }
        throw new IllegalArgumentException("zero vector is not supported when space type is [cosinesimil]");
    }

    public static float l2Squared(float[] q, float[] v) {
        return VectorUtil.squareDistance(q, v);
    }

    public static float l2Squared(byte[] q, byte[] v) {
        return VectorUtil.squareDistance(q, v);
    }

    public static float cosinesimil(float[] q, float[] v) {
        requireEqualDimension(q.length, v.length);
        try {
            return VectorUtil.cosine(q, v);
        } catch (IllegalArgumentException | AssertionError e) {
            return 0.0f;
        }
    }

    public static float cosinesimil(byte[] q, byte[] v) {
        requireEqualDimension(q.length, v.length);
        try {
            return VectorUtil.cosine(q, v);
        } catch (IllegalArgumentException | AssertionError e) {
            return 0.0f;
        }
    }

    /** Cosine with the query's squared magnitude given. */
    public static float cosinesimilOptimized(float[] q, float[] v, float queryNormSquared) {
        requireEqualDimension(q.length, v.length);
        float dot = 0.0f;
        float norm = 0.0f;
        for (int i = 0; i < q.length; i++) {
            dot += q[i] * v[i];
            norm += v[i] * v[i];
        }
        float product = queryNormSquared * norm;
        if (product == 0) {
            return 0.0f;
        }
        return (float) (dot / Math.sqrt(product));
    }

    public static float hammingBit(BigInteger q, BigInteger v) {
        return v.xor(q).bitCount();
    }

    public static float hammingBit(long q, long v) {
        return Long.bitCount(q ^ v);
    }

    public static float hammingBit(byte[] q, byte[] v) {
        requireEqualDimension(q.length, v.length);
        return VectorUtil.xorBitCount(q, v);
    }

    public static float l1Norm(float[] q, float[] v) {
        requireEqualDimension(q.length, v.length);
        float d = 0;
        for (int i = 0; i < v.length; i++) {
            d += Math.abs(q[i] - v[i]);
        }
        return d;
    }

    public static float l1Norm(byte[] q, byte[] v) {
        requireEqualDimension(q.length, v.length);
        float d = 0;
        for (int i = 0; i < v.length; i++) {
            float diff = q[i] - v[i];
            d += Math.abs(diff);
        }
        return d;
    }

    public static float lInfNorm(float[] q, float[] v) {
        requireEqualDimension(q.length, v.length);
        float d = 0;
        for (int i = 0; i < v.length; i++) {
            d = Math.max(Math.abs(q[i] - v[i]), d);
        }
        return d;
    }

    public static float lInfNorm(byte[] q, byte[] v) {
        requireEqualDimension(q.length, v.length);
        float d = 0;
        for (int i = 0; i < v.length; i++) {
            float diff = q[i] - v[i];
            d = Math.max(Math.abs(diff), d);
        }
        return d;
    }

    public static float innerProduct(float[] q, float[] v) {
        requireEqualDimension(q.length, v.length);
        return VectorUtil.dotProduct(q, v);
    }

    public static float innerProduct(byte[] q, byte[] v) {
        requireEqualDimension(q.length, v.length);
        return VectorUtil.dotProduct(q, v);
    }

    // The Painless functions: the query as a list of numbers, the document's vector as doc['field'].

    public static float l2Squared(List<?> query, KnnScriptDocValues doc) {
        requireNonBinary("l2Squared", doc);
        if (doc.floats()) {
            return l2Squared(toFloats(query, doc.dataType()), (float[]) doc.getValue());
        }
        return l2Squared(toBytes(query, doc.dataType()), (byte[]) doc.getValue());
    }

    public static float lInfNorm(List<?> query, KnnScriptDocValues doc) {
        requireNonBinary("lInfNorm", doc);
        if (doc.floats()) {
            return lInfNorm(toFloats(query, doc.dataType()), (float[]) doc.getValue());
        }
        return lInfNorm(toBytes(query, doc.dataType()), (byte[]) doc.getValue());
    }

    public static float l1Norm(List<?> query, KnnScriptDocValues doc) {
        requireNonBinary("l1Norm", doc);
        if (doc.floats()) {
            return l1Norm(toFloats(query, doc.dataType()), (float[]) doc.getValue());
        }
        return l1Norm(toBytes(query, doc.dataType()), (byte[]) doc.getValue());
    }

    public static float innerProduct(List<?> query, KnnScriptDocValues doc) {
        requireNonBinary("innerProduct", doc);
        if (doc.floats()) {
            return innerProduct(toFloats(query, doc.dataType()), (float[]) doc.getValue());
        }
        return innerProduct(toBytes(query, doc.dataType()), (byte[]) doc.getValue());
    }

    public static float cosineSimilarity(List<?> query, KnnScriptDocValues doc) {
        requireNonBinary("cosineSimilarity", doc);
        if (doc.floats()) {
            float[] q = toFloats(query, doc.dataType());
            checkNotZero(q);
            return cosinesimil(q, (float[]) doc.getValue());
        }
        byte[] q = toBytes(query, doc.dataType());
        checkNotZero(q);
        return cosinesimil(q, (byte[]) doc.getValue());
    }

    public static float cosineSimilarity(List<?> query, KnnScriptDocValues doc, Number queryNormSquared) {
        requireNonBinary("cosineSimilarity", doc);
        float[] q = toFloats(query, doc.dataType());
        checkNotZero(q);
        if (doc.floats()) {
            return cosinesimilOptimized(q, (float[]) doc.getValue(), queryNormSquared.floatValue());
        }
        byte[] b = (byte[]) doc.getValue();
        float[] v = new float[b.length];
        for (int i = 0; i < b.length; i++) {
            v[i] = b[i];
        }
        return cosinesimilOptimized(q, v, queryNormSquared.floatValue());
    }

    public static float hamming(List<?> query, KnnScriptDocValues doc) {
        requireBinary("hamming", doc);
        return hammingBit(toBytes(query, doc.dataType()), (byte[]) doc.getValue());
    }

    private static final List<String> SPACES = List.of("l2", "cosinesimil", "l1", "linf", "innerproduct", "hamming");

    /** {@code lateInteractionScore} in the default space, {@code l2}. */
    public static float lateInteractionScore(List<?> queryVectors, String docFieldName, Map<?, ?> doc) {
        return lateInteractionScore(queryVectors, docFieldName, doc, "l2");
    }

    /**
     * ColBERT-style late interaction: for each query vector, the best similarity to any of the
     * document's vectors (a list of lists in {@code doc[docFieldName]}, usually {@code params._source}),
     * summed. Similarity is the space's Lucene function ({@code l2}, {@code cosinesimil}, {@code
     * innerproduct} or {@code hamming}, which refuses float vectors).
     */
    public static float lateInteractionScore(List<?> queryVectors, String docFieldName, Map<?, ?> doc, String spaceType) {
        if (queryVectors == null) {
            throw new IllegalArgumentException("Query vectors cannot be null");
        }
        if (queryVectors.isEmpty()) {
            throw new IllegalArgumentException("Query vectors cannot be empty");
        }
        if (docFieldName == null || docFieldName.isEmpty()) {
            throw new IllegalArgumentException("Document field name cannot be null or empty");
        }
        if (doc == null) {
            throw new IllegalArgumentException("Document cannot be null");
        }
        if (spaceType == null || spaceType.isEmpty()) {
            throw new IllegalArgumentException("Space type cannot be null or empty");
        }
        List<?> docVectors;
        try {
            docVectors = (List<?>) doc.get(docFieldName);
        } catch (ClassCastException e) {
            throw new IllegalArgumentException("Field " + docFieldName + " must contain a list of vector lists", e);
        }
        if (docVectors == null || docVectors.isEmpty()) {
            throw new IllegalArgumentException("Document vectors cannot be null or empty");
        }
        String space = spaceType.toLowerCase(Locale.ROOT);
        if (space.equals("undefined") == false && SPACES.contains(space) == false) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Unable to find space: %s . Valid values are: %s", spaceType, SPACES)
            );
        }
        VectorSimilarityFunction function = switch (space) {
            case "l2" -> VectorSimilarityFunction.EUCLIDEAN;
            case "cosinesimil" -> VectorSimilarityFunction.COSINE;
            case "innerproduct" -> VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
            case "hamming" -> throw new IllegalStateException("Hamming space is not supported with float vectors");
            default -> throw new IllegalArgumentException("Space type " + spaceType + " does not support vector similarity function");
        };
        float[][] docs = new float[docVectors.size()][];
        int n = 0;
        for (Object d : docVectors) {
            List<?> v = (List<?>) d;
            if (v != null && v.isEmpty() == false) {
                docs[n++] = numbers(v);
            }
        }
        float total = 0.0f;
        for (Object q : queryVectors) {
            List<?> v = (List<?>) q;
            if (v == null || v.isEmpty()) {
                throw new IllegalArgumentException("Every single vector within query vectors cannot be empty or null");
            }
            float[] qv = numbers(v);
            float best = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                best = Math.max(best, function.compare(qv, docs[i]));
            }
            if (best != Float.NEGATIVE_INFINITY) {
                total += best;
            }
        }
        return total;
    }

    private static float[] numbers(List<?> v) {
        float[] out = new float[v.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) v.get(i)).floatValue();
        }
        return out;
    }
}
