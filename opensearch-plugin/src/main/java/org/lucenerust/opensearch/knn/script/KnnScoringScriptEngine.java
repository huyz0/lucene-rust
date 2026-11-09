/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.script;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.knn.KnnScriptDocValues;
import org.lucenerust.opensearch.knn.KnnVectorFieldMapper.KnnVectorFieldType;
import org.opensearch.Version;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.index.mapper.BinaryFieldMapper;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.NumberFieldMapper;
import org.opensearch.script.ScoreScript;
import org.opensearch.script.ScriptContext;
import org.opensearch.script.ScriptEngine;
import org.opensearch.script.ScriptFactory;
import org.opensearch.search.lookup.SearchLookup;

import java.math.BigInteger;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * k-NN's {@code KNNScoringScriptEngine}: the {@code knn} script language, whose one script, {@code
 * knn_score}, scores each document by the distance between its vector and {@code
 * params.query_value} in {@code params.space_type} ({@code l2}, {@code cosinesimil}, {@code
 * innerproduct}, {@code l1}, {@code linf} or {@code hamming} over a {@code knn_vector} field, {@code
 * hammingbit} over a {@code long} or {@code binary} one). A document without a value scores 0.
 */
public final class KnnScoringScriptEngine implements ScriptEngine {
    public static final String NAME = "knn";
    public static final String SCRIPT_SOURCE = "knn_score";
    public static final String HAMMING_BIT = "hammingbit";

    @Override
    public String getType() {
        return NAME;
    }

    @Override
    public <T> T compile(String name, String code, ScriptContext<T> context, Map<String, String> params) {
        if (ScoreScript.CONTEXT.equals(context) == false) {
            throw new IllegalArgumentException(getType() + " KNN scoring scripts cannot be used for context [" + context.name + "]");
        }
        if (SCRIPT_SOURCE.equals(code) == false) {
            throw new IllegalArgumentException("Unknown script name " + code);
        }
        return context.factoryClazz.cast(new Factory());
    }

    @Override
    public Set<ScriptContext<?>> getSupportedContexts() {
        return Set.of(ScoreScript.CONTEXT);
    }

    /** {@code KNNScoreScriptFactory}. */
    static final class Factory implements ScoreScript.Factory, ScriptFactory {
        @Override
        public boolean isResultDeterministic() {
            return true;
        }

        @Override
        public ScoreScript.LeafFactory newFactory(Map<String, Object> params, SearchLookup lookup, IndexSearcher searcher) {
            String field = param(params, "field").toString();
            String space = param(params, "space_type").toString();
            Object query = param(params, "query_value");
            Prepared prepared = scorer(space, query, lookup.doc().mapperService().fieldType(field));
            return new KnnLeafFactory(params, lookup, searcher, field, prepared);
        }
    }

    /**
     * What a native search needs to score a {@code knn_vector} space as this script does: the
     * space ({@link #SPACES}), the field's data type ({@code 0} float, {@code 1} byte or binary,
     * {@code 2} half_float), whether cosine is {@code (1 + cos) / 2}, and the query as floats or
     * bytes.
     */
    public record Native(String field, byte space, byte dataType, boolean halfCosine, float[] floats, byte[] bytes) {}

    /** The {@code knn_vector} spaces, by wire number. */
    public static final List<String> SPACES = List.of("l2", "cosinesimil", "innerproduct", "l1", "linf", "hamming");

    private record Prepared(Scorer scorer, Native nativeSpec) {}

    /** {@code KNNScoreScriptLeafFactory}. */
    public static final class KnnLeafFactory implements ScoreScript.LeafFactory {
        private final Map<String, Object> params;
        private final SearchLookup lookup;
        private final IndexSearcher searcher;
        private final String field;
        private final Prepared prepared;

        KnnLeafFactory(Map<String, Object> params, SearchLookup lookup, IndexSearcher searcher, String field, Prepared prepared) {
            this.params = params;
            this.lookup = lookup;
            this.searcher = searcher;
            this.field = field;
            this.prepared = prepared;
        }

        /** The script as a native search scores it, or {@code null} for {@code hammingbit}. */
        public Native nativeSpec() {
            return prepared.nativeSpec();
        }

        @Override
        public boolean needs_score() {
            return false;
        }

        @Override
        public ScoreScript newInstance(LeafReaderContext ctx) {
            Scorer scorer = prepared.scorer();
            return new ScoreScript(params, lookup, searcher, ctx) {
                @Override
                public double execute(ExplanationHolder explanation) {
                    ScriptDocValues<?> values = (ScriptDocValues<?>) getDoc().get(field);
                    if (values.isEmpty()) {
                        return 0.0;
                    }
                    return scorer.score(values);
                }
            };
        }
    }

    /** One document's score from its doc values. */
    private interface Scorer {
        float score(ScriptDocValues<?> values);
    }

    private static Object param(Map<String, Object> params, String name) {
        Object v = params.get(name);
        if (v == null) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "Missing parameter [%s]", name));
        }
        return v;
    }

    /** {@code KNNScoringSpaceFactory.create} and the space's scoring method. */
    static Prepared scorer(String space, Object query, MappedFieldType fieldType) {
        String s = space.toLowerCase(Locale.ROOT);
        if (s.equals(HAMMING_BIT)) {
            return new Prepared(hammingBit(query, fieldType), null);
        }
        return switch (s) {
            case "hamming" -> vectorSpace(query, fieldType, "hamming", Set.of("binary"), null, (q, v) -> 1 / (1 + KnnScoring.hammingBit(q, v)));
            case "l2" -> vectorSpace(
                query,
                fieldType,
                "l2",
                DEFAULT_TYPES,
                (q, v) -> 1 / (1 + KnnScoring.l2Squared(q, v)),
                (q, v) -> 1 / (1 + KnnScoring.l2Squared(q, v))
            );
            case "l1" -> vectorSpace(
                query,
                fieldType,
                "l1",
                DEFAULT_TYPES,
                (q, v) -> 1 / (1 + KnnScoring.l1Norm(q, v)),
                (q, v) -> 1 / (1 + KnnScoring.l1Norm(q, v))
            );
            case "linf" -> vectorSpace(
                query,
                fieldType,
                "l-inf",
                DEFAULT_TYPES,
                (q, v) -> 1 / (1 + KnnScoring.lInfNorm(q, v)),
                (q, v) -> 1 / (1 + KnnScoring.lInfNorm(q, v))
            );
            case "innerproduct" -> vectorSpace(
                query,
                fieldType,
                "innerproduct",
                DEFAULT_TYPES,
                (q, v) -> normalize(-KnnScoring.innerProduct(q, v)),
                (q, v) -> normalize(-KnnScoring.innerProduct(q, v))
            );
            case "cosinesimil" -> cosine(query, fieldType);
            default -> throw new IllegalArgumentException("Invalid space type. Please refer to the available space types.");
        };
    }

    private static final Set<String> DEFAULT_TYPES = Set.of("float", "byte", "half_float");

    /** {@code KNNWeight.normalizeScore}. */
    private static float normalize(float score) {
        return score >= 0 ? 1 / (1 + score) : -score + 1;
    }

    private interface FloatScore {
        float score(float[] q, float[] v);
    }

    private interface ByteScore {
        float score(byte[] q, byte[] v);
    }

    private static KnnVectorFieldType vectorField(MappedFieldType fieldType, String space, Set<String> types) {
        if (fieldType instanceof KnnVectorFieldType == false) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Incompatible field_type for %s space. The field type must be knn_vector.", space)
            );
        }
        KnnVectorFieldType ft = (KnnVectorFieldType) fieldType;
        String type = ft.config().dataTypeName();
        if (types.contains(type) == false) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "Incompatible field_type for %s space. The data type should be %s but got %s",
                    space,
                    types.stream().map(t -> t.toUpperCase(Locale.ROOT)).sorted().toList(),
                    type.toUpperCase(Locale.ROOT)
                )
            );
        }
        return ft;
    }

    /** A space over a {@code knn_vector} field: float queries scored by {@code floats}, byte ones by {@code bytes}. */
    private static Prepared vectorSpace(
        Object query,
        MappedFieldType fieldType,
        String space,
        Set<String> types,
        FloatScore floats,
        ByteScore bytes
    ) {
        KnnVectorFieldType ft = vectorField(fieldType, space, types);
        String type = ft.config().dataTypeName();
        int length = ft.config().vectorDimension();
        byte code = (byte) SPACES.indexOf(space.equals("l-inf") ? "linf" : space);
        if (type.equals("float") || type.equals("half_float")) {
            float[] q = floatQuery(query, length, type);
            return new Prepared(
                values -> floats.score(q, (float[]) ((KnnScriptDocValues) values).getValue()),
                new Native(ft.name(), code, dataCode(type), false, q, null)
            );
        }
        byte[] q = byteQuery(query, length, type);
        return new Prepared(
            values -> bytes.score(q, (byte[]) ((KnnScriptDocValues) values).getValue()),
            new Native(ft.name(), code, dataCode(type), false, null, q)
        );
    }

    private static byte dataCode(String type) {
        return switch (type) {
            case "float" -> 0;
            case "half_float" -> 2;
            default -> 1;
        };
    }

    private static Prepared cosine(Object query, MappedFieldType fieldType) {
        KnnVectorFieldType ft = vectorField(fieldType, "cosine", DEFAULT_TYPES);
        String type = ft.config().dataTypeName();
        int length = ft.config().vectorDimension();
        if (type.equals("float") || type.equals("half_float")) {
            float[] q = floatQuery(query, length, type);
            KnnScoring.checkNotZero(q);
            float norm = 0.0f;
            for (float x : q) {
                norm += x * x;
            }
            float qNorm = norm;
            // Indexes created on 2.19 or later score cosine as Lucene does, (1 + cos) / 2.
            boolean half = ft.indexCreated().onOrAfter(Version.V_2_19_0);
            Native spec = new Native(ft.name(), (byte) 1, dataCode(type), half, q, null);
            if (half) {
                return new Prepared(
                    values -> Math.max((1 + KnnScoring.cosinesimilOptimized(q, (float[]) ((KnnScriptDocValues) values).getValue(), qNorm)) / 2.0F, 0),
                    spec
                );
            }
            return new Prepared(values -> 1 + KnnScoring.cosinesimilOptimized(q, (float[]) ((KnnScriptDocValues) values).getValue(), qNorm), spec);
        }
        byte[] q = byteQuery(query, length, type);
        KnnScoring.checkNotZero(q);
        return new Prepared(
            values -> 1 + KnnScoring.cosinesimil(q, (byte[]) ((KnnScriptDocValues) values).getValue()),
            new Native(ft.name(), (byte) 1, dataCode(type), false, null, q)
        );
    }

    private static float[] floatQuery(Object query, int length, String type) {
        float[] q = KnnScoring.toFloats((List<?>) query, type);
        if (q.length != length) {
            throw new IllegalStateException("Object's length=" + q.length + " does not match the expected length=" + length + ".");
        }
        return q;
    }

    private static byte[] byteQuery(Object query, int length, String type) {
        byte[] q = KnnScoring.toBytes((List<?>) query, type);
        if (q.length != length) {
            throw new IllegalStateException("Object's length=" + q.length + " does not match the expected length=" + length + ".");
        }
        return q;
    }

    /** {@code KNNScoringSpace.HammingBit}: a {@code long} field, or a {@code binary} one with a base64 query. */
    private static Scorer hammingBit(Object query, MappedFieldType fieldType) {
        if (fieldType instanceof NumberFieldMapper.NumberFieldType n && n.numericType() == NumberFieldMapper.NumberType.LONG.numericType()) {
            long q;
            if (query instanceof Integer i) {
                q = i;
            } else if (query instanceof Long l) {
                q = l;
            } else {
                throw new IllegalArgumentException("Object cannot be parsed as a Long.");
            }
            return values -> 1.0f / (1 + KnnScoring.hammingBit(q, ((ScriptDocValues.Longs) values).getValue()));
        }
        if (fieldType instanceof BinaryFieldMapper.BinaryFieldType) {
            BigInteger q = new BigInteger(1, Base64.getDecoder().decode((String) query));
            return values -> {
                BytesRef b = ((ScriptDocValues.BytesRefs) values).getValue();
                return 1.0f / (1 + KnnScoring.hammingBit(q, new BigInteger(1, Arrays.copyOfRange(b.bytes, b.offset, b.offset + b.length))));
            };
        }
        throw new IllegalArgumentException("Incompatible field_type for hammingbit space. The field type must of type long or binary.");
    }
}
