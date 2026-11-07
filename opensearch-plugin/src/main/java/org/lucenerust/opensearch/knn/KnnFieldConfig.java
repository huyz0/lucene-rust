/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.index.VectorEncoding;
import org.lucenerust.opensearch.engine.VectorsFormatSpec;
import org.opensearch.index.mapper.MapperParsingException;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A {@code knn_vector} mapping resolved as the stock k-NN plugin resolves it ({@code
 * KNNVectorFieldMapper.Builder} with its engine and mode/compression resolvers), into the vector
 * format the Rust writer builds: Lucene's HNSW graph over float or byte vectors, quantized with a
 * Lucene104 scalar encoding when the method's encoder or the compression asks for one, or k-NN's
 * {@code KNN990HnswBinaryVectorsFormat} graph over packed bit vectors ({@code data_type: binary},
 * {@code hamming} space), or its {@code KNN1040HnswHalfFloatVectorsFormat} graph over half floats
 * ({@code data_type: half_float}).
 *
 * <p>A mapping without an engine uses {@code lucene}; {@code faiss} is refused. The search reads the
 * raw vectors through the graph, so a quantized field's scores are the exact ones the stock plugin
 * rescores to.
 *
 * <p>The {@code lucene_rust} engine is this node's own, opt-in: {@code method: ivf} is an IVF index
 * with 1-bit (BBQ) codes beside the raw vectors ({@code LuceneRustIvfVectorsFormat}), no training
 * step; {@code parameters.target_cluster_size} (16 to 65536, default 384) sizes its clusters and
 * {@code parameters.rerank_encoding} ({@code float}, {@code half_float} by default, {@code bfloat16}
 * or {@code int8}) stores the copy it reranks from. Float fields only. A stock node does not know
 * the engine, so it refuses the mapping rather than reading the field another way.
 *
 * @param dimension the mapping's dimension: bits for a {@link #binary} field
 * @param binary packed bit vectors, {@code dimension / 8} bytes each
 * @param halfFloat float vectors stored as half floats
 * @param encoding a {@link VectorsFormatSpec} scalar encoding wire number, {@code -1} for full precision
 * @param clusterSize an IVF field's target cluster size, {@code 0} for an HNSW one
 * @param rerankEncoding an IVF field's rerank encoding wire number ({@link #RERANK_ENCODINGS})
 */
public record KnnFieldConfig(
    int dimension,
    VectorEncoding dataType,
    boolean binary,
    boolean halfFloat,
    SpaceType spaceType,
    String engine,
    int m,
    int efConstruction,
    int encoding,
    String mode,
    String compression,
    int clusterSize,
    int rerankEncoding
) {
    public static final String FAISS = "faiss";
    public static final String LUCENE = "lucene";
    public static final String LUCENE_RUST = "lucene_rust";
    public static final String IVF = "ivf";
    public static final int DEFAULT_CLUSTER_SIZE = 384;
    /** {@code LuceneRustIvfVectorsFormat}'s rerank encodings, by wire number. */
    public static final List<String> RERANK_ENCODINGS = List.of("float", "half_float", "bfloat16", "int8");
    static final int MIN_CLUSTER_SIZE = 16;
    static final int MAX_CLUSTER_SIZE = 65_536;
    public static final int MAX_DIMENSION = 16_000;
    public static final int DEFAULT_M = 16;
    public static final int DEFAULT_EF_CONSTRUCTION = 100;
    /** Lucene's {@code HnswGraphBuilder} limits. */
    static final int MAX_M = 512;
    static final int MAX_EF_CONSTRUCTION = 3200;

    /** Elements per vector: the dimension, or its bytes for a {@link #binary} field. */
    public int vectorDimension() {
        return binary ? dimension / 8 : dimension;
    }

    /** An IVF field ({@code lucene_rust} engine, {@code ivf} method). */
    public boolean ivf() {
        return clusterSize > 0;
    }

    /** The format the Rust writer builds the field with. */
    public VectorsFormatSpec format() {
        if (ivf()) {
            return VectorsFormatSpec.ivf(clusterSize, rerankEncoding);
        }
        if (binary) {
            return VectorsFormatSpec.hnswHamming(m, efConstruction);
        }
        if (halfFloat) {
            return VectorsFormatSpec.hnswHalfFloat(m, efConstruction);
        }
        return encoding < 0 ? VectorsFormatSpec.hnsw(m, efConstruction) : VectorsFormatSpec.hnswQuantized(encoding, m, efConstruction);
    }

    /**
     * The mapping's parameters resolved: {@code dimension}, {@code data_type}, the top-level {@code
     * space_type} and {@code engine}, {@code method}, {@code mode} and {@code compression_level}.
     */
    static KnnFieldConfig resolve(
        String field,
        int dimension,
        String dataType,
        String topSpace,
        String topEngine,
        Map<String, Object> method,
        String mode,
        String compression,
        String indexSpace
    ) {
        if (dimension == -1) {
            throw new MapperParsingException(String.format(Locale.ROOT, "Dimension value missing for vector: %s", field));
        }
        if (dimension <= 0) {
            throw new MapperParsingException(String.format(Locale.ROOT, "Dimension value must be greater than 0 for vector: %s", field));
        }
        if (dimension > MAX_DIMENSION) {
            throw new MapperParsingException(
                String.format(Locale.ROOT, "Dimension value cannot be greater than %d for vector: %s", MAX_DIMENSION, field)
            );
        }
        // Stock parses data_type and engine names in any case, and persists data_type upper case.
        VectorEncoding encodingType = switch (dataType.toLowerCase(Locale.ROOT)) {
            case "float", "half_float" -> VectorEncoding.FLOAT32;
            case "byte", "binary" -> VectorEncoding.BYTE;
            default -> throw new MapperParsingException(
                String.format(Locale.ROOT, "Invalid value provided for [data_type] field: [%s]. Supported values are [float, half_float, byte, binary]", dataType)
            );
        };
        boolean binary = dataType.equalsIgnoreCase("binary");
        boolean halfFloat = dataType.equalsIgnoreCase("half_float");

        String methodName = "hnsw";
        String methodSpace = null;
        String methodEngine = null;
        Map<String, Object> params = Map.of();
        if (method != null) {
            Object name = method.get("name");
            if (name == null) {
                throw new MapperParsingException("name needs to be set");
            }
            methodName = name.toString();
            methodSpace = str(method.get("space_type"));
            methodEngine = str(method.get("engine"));
            params = map(method.get("parameters"), "parameters");
            for (String k : method.keySet()) {
                if (k.equals("name") == false && k.equals("space_type") == false && k.equals("engine") == false && k.equals("parameters") == false) {
                    throw new MapperParsingException(String.format(Locale.ROOT, "Invalid parameter: %s", k));
                }
            }
        }

        String space = pick(topSpace, methodSpace, "Space type", SpaceType.UNDEFINED);
        SpaceType spaceType = SpaceType.of(
            space == null ? (indexSpace == null ? (binary ? SpaceType.HAMMING.value() : SpaceType.DEFAULT) : indexSpace) : space
        );
        // SpaceType.validateVectorDataType: hamming is for binary vectors only, and binary vectors only for hamming.
        if (binary != (spaceType == SpaceType.HAMMING)) {
            throw new IllegalArgumentException(
                String.format(Locale.ROOT, "Space type [%s] is not supported with [%s] data type", spaceType.value(), dataType.toLowerCase(Locale.ROOT))
            );
        }
        if (binary && dimension % 8 != 0) {
            throw new MapperParsingException("Validation Failed: 1: Dimension should be multiply of 8 for binary vector data type;");
        }

        String resolvedMode = mode;
        if (resolvedMode != null && resolvedMode.equals("in_memory") == false && resolvedMode.equals("on_disk") == false) {
            throw new MapperParsingException(String.format(Locale.ROOT, "Invalid mode: %s, valid modes are [in_memory, on_disk]", mode));
        }
        String level = compression;
        if (level != null && compressionEncoding(level) == -2) {
            throw new MapperParsingException(
                String.format(Locale.ROOT, "Invalid compression level: %s, valid values are [1x, 2x, 4x, 8x, 16x, 32x]", compression)
            );
        }
        if (level == null && "on_disk".equals(resolvedMode)) {
            level = encodingType == VectorEncoding.BYTE ? "1x" : halfFloat ? "16x" : "32x";
        }

        // Stock parses engine names in any case; the lucene engine is the default here, and faiss
        // (whose .faiss files this plugin neither reads nor writes) is refused.
        String engine = pick(lower(topEngine), lower(methodEngine), "Engine", "undefined");
        if (engine == null) {
            engine = LUCENE;
        }
        switch (engine) {
            case LUCENE -> {}
            case LUCENE_RUST -> {
                return ivf(dimension, encodingType, dataType, spaceType, methodName, params, resolvedMode, level);
            }
            case FAISS -> throw new MapperParsingException(
                "Engine [faiss] is not supported: this node's knn_vector fields use the [lucene] engine"
            );
            case "nmslib" -> throw new MapperParsingException(
                "nmslib engine is deprecated in OpenSearch and cannot be used for new index creation starting from version 3.0.0."
            );
            default -> throw new MapperParsingException(
                String.format(Locale.ROOT, "Invalid engine type: %s", topEngine != null && topEngine.equalsIgnoreCase(engine) ? topEngine : methodEngine)
            );
        }

        if (methodName.equals("hnsw") == false) {
            throw new MapperParsingException(
                methodName.equals(IVF)
                    ? "[method] [ivf] of the [lucene] engine needs a trained model, which this node does not support; "
                        + "use the [lucene_rust] engine's [ivf], which needs none"
                    : String.format(Locale.ROOT, "Invalid method name: %s", methodName)
            );
        }

        int m = DEFAULT_M;
        int efConstruction = DEFAULT_EF_CONSTRUCTION;
        int encoding = -1;
        boolean explicitEncoder = false;
        for (Map.Entry<String, Object> p : params.entrySet()) {
            switch (p.getKey()) {
                case "m" -> m = positiveInt(p.getValue(), "m", MAX_M);
                case "ef_construction" -> efConstruction = positiveInt(p.getValue(), "ef_construction", MAX_EF_CONSTRUCTION);
                case "ef_search" -> throw new MapperParsingException("Invalid parameter: ef_search for the lucene engine; set it in the query");
                case "encoder" -> {
                    explicitEncoder = true;
                    encoding = encoder(map(p.getValue(), "encoder"));
                }
                default -> throw new MapperParsingException(String.format(Locale.ROOT, "Invalid parameter for hnsw: %s", p.getKey()));
            }
        }

        if (halfFloat) {
            // LuceneHNSWMethodResolver.resolveHalfFloatMethod: x1 or x16, set by compression_level only.
            List<String> errors = new ArrayList<>();
            if (level != null && level.equals("1x") == false && level.equals("16x") == false) {
                errors.add(String.format(Locale.ROOT, "\"lucene\" with \"half_float\" data type does not support \"%s\" compression", level));
            }
            if (explicitEncoder) {
                errors.add("\"encoder\" parameter is not supported for \"half_float\" data type; use \"compression_level\" instead.");
            }
            if (errors.isEmpty() == false) {
                StringBuilder message = new StringBuilder("Validation Failed: ");
                for (int i = 0; i < errors.size(); i++) {
                    message.append(i + 1).append(": ").append(errors.get(i)).append(';');
                }
                throw new MapperParsingException(message.toString());
            }
            if ("16x".equals(level)) {
                throw new MapperParsingException("Compression level [16x] for [data_type] [half_float] is not supported by this node");
            }
            return new KnnFieldConfig(dimension, encodingType, false, true, spaceType, engine, m, efConstruction, -1, resolvedMode, level, 0, 0);
        }

        if (level != null) {
            int fromLevel = compressionEncoding(level);
            if (encodingType == VectorEncoding.BYTE && level.equals("1x") == false) {
                throw new MapperParsingException("Compression is only supported for [data_type] [float]");
            }
            if (explicitEncoder) {
                if (fromLevel != encoding && level.equals("1x") == false) {
                    throw new MapperParsingException(
                        String.format(Locale.ROOT, "Compression level [%s] and the method's encoder do not agree", level)
                    );
                }
            } else {
                encoding = fromLevel;
            }
        }
        if (encoding >= 0 && encodingType == VectorEncoding.BYTE) {
            throw new MapperParsingException("Quantization is only supported for [data_type] [float]");
        }
        return new KnnFieldConfig(dimension, encodingType, binary, false, spaceType, engine, m, efConstruction, encoding, resolvedMode, level, 0, 0);
    }

    /** A {@code lucene_rust} engine mapping: {@code method: ivf} over float vectors. */
    private static KnnFieldConfig ivf(
        int dimension,
        VectorEncoding encodingType,
        String dataType,
        SpaceType spaceType,
        String methodName,
        Map<String, Object> params,
        String mode,
        String level
    ) {
        if (methodName.equals(IVF) == false) {
            throw new MapperParsingException(String.format(Locale.ROOT, "Invalid method name for the [lucene_rust] engine: %s; supported: [ivf]", methodName));
        }
        if (dataType.equalsIgnoreCase("float") == false) {
            throw new MapperParsingException(
                String.format(Locale.ROOT, "[data_type] [%s] is not supported by the [lucene_rust] engine's [ivf]; use [float]", dataType.toLowerCase(Locale.ROOT))
            );
        }
        if (mode != null || level != null) {
            throw new MapperParsingException("[mode] and [compression_level] are not supported by the [lucene_rust] engine's [ivf]; set [parameters.rerank_encoding]");
        }
        int clusterSize = DEFAULT_CLUSTER_SIZE;
        int rerank = 1;
        for (Map.Entry<String, Object> p : params.entrySet()) {
            switch (p.getKey()) {
                case "target_cluster_size" -> {
                    clusterSize = positiveInt(p.getValue(), "target_cluster_size", MAX_CLUSTER_SIZE);
                    if (clusterSize < MIN_CLUSTER_SIZE) {
                        throw new MapperParsingException(
                            String.format(Locale.ROOT, "[target_cluster_size] must be in [%d, %d]; got %d", MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE, clusterSize)
                        );
                    }
                }
                case "rerank_encoding" -> {
                    rerank = RERANK_ENCODINGS.indexOf(String.valueOf(p.getValue()).toLowerCase(Locale.ROOT));
                    if (rerank < 0) {
                        throw new MapperParsingException(
                            String.format(Locale.ROOT, "Invalid [rerank_encoding]: %s; supported: %s", p.getValue(), RERANK_ENCODINGS)
                        );
                    }
                }
                case "quantizer" -> {
                    if ("bbq".equals(String.valueOf(p.getValue()).toLowerCase(Locale.ROOT)) == false) {
                        throw new MapperParsingException(String.format(Locale.ROOT, "Invalid [quantizer]: %s; supported: [bbq]", p.getValue()));
                    }
                }
                default -> throw new MapperParsingException(String.format(Locale.ROOT, "Invalid parameter for ivf: %s", p.getKey()));
            }
        }
        return new KnnFieldConfig(dimension, encodingType, false, false, spaceType, LUCENE_RUST, 0, 0, -1, null, null, clusterSize, rerank);
    }

    /** The encoder's scalar encoding, {@code -1} for full precision. */
    private static int encoder(Map<String, Object> encoder) {
        Object name = encoder.get("name");
        if (name == null) {
            throw new MapperParsingException("encoder name needs to be set");
        }
        Map<String, Object> p = map(encoder.get("parameters"), "encoder parameters");
        String n = name.toString();
        if (n.equals("sq") == false) {
            throw new MapperParsingException(String.format(Locale.ROOT, "Invalid encoder for the lucene engine: %s", n));
        }
        for (String k : p.keySet()) {
            if (k.equals("bits") == false && k.equals("confidence_interval") == false) {
                throw new MapperParsingException(String.format(Locale.ROOT, "Invalid parameter for sq: %s", k));
            }
        }
        if (p.containsKey("bits") == false) {
            throw new MapperParsingException(
                "Validation Failed: 1: Parameter [bits] is required for encoder [sq] on indices created with version 3.6.0 or later. Supported values: [7, 1, 2, 4];"
            );
        }
        int bits = positiveInt(p.get("bits"), "bits", 8);
        return switch (bits) {
            case 7 -> VectorsFormatSpec.SEVEN_BIT;
            case 4 -> VectorsFormatSpec.PACKED_NIBBLE;
            case 2 -> VectorsFormatSpec.DIBIT;
            case 1 -> VectorsFormatSpec.SINGLE_BIT;
            default -> throw new MapperParsingException("Validation Failed: 1: Unsupported [bits] " + bits + " for encoder [sq]. Supported values: [7, 1, 2, 4];");
        };
    }

    /** {@code CompressionLevel}'s scalar encoding: {@code -1} full precision, {@code -2} not a level. */
    static int compressionEncoding(String level) {
        return switch (level) {
            case "1x", "2x" -> -1;
            case "4x" -> VectorsFormatSpec.SEVEN_BIT;
            case "8x" -> VectorsFormatSpec.PACKED_NIBBLE;
            case "16x" -> VectorsFormatSpec.DIBIT;
            case "32x" -> VectorsFormatSpec.SINGLE_BIT;
            default -> -2;
        };
    }

    private static String pick(String top, String inMethod, String what, String undefined) {
        String t = top == null || top.equals(undefined) ? null : top;
        String m = inMethod == null || inMethod.equals(undefined) ? null : inMethod;
        if (t != null && m != null && t.equals(m) == false) {
            throw new MapperParsingException(
                String.format(Locale.ROOT, "%s in \"method\" and top level %s should be same or one of them should be defined", what, what.toLowerCase(Locale.ROOT))
            );
        }
        return t != null ? t : m;
    }

    private static String lower(String s) {
        return s == null ? null : s.toLowerCase(Locale.ROOT);
    }

    private static String str(Object o) {
        return o == null ? null : o.toString();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object o, String what) {
        if (o == null) {
            return Map.of();
        }
        if (o instanceof Map<?, ?> m) {
            return (Map<String, Object>) m;
        }
        throw new MapperParsingException(String.format(Locale.ROOT, "Unable to parse [%s]: expected an object", what));
    }

    private static int positiveInt(Object o, String name, int max) {
        int v;
        if (o instanceof Number n && n.doubleValue() == Math.rint(n.doubleValue())) {
            v = n.intValue();
        } else {
            try {
                v = Integer.parseInt(String.valueOf(o));
            } catch (NumberFormatException e) {
                throw new MapperParsingException(String.format(Locale.ROOT, "Unable to parse [%s]: %s is not an integer", name, o));
            }
        }
        if (v <= 0 || v > max) {
            throw new MapperParsingException(String.format(Locale.ROOT, "[%s] must be in (0, %d]; got %d", name, max, v));
        }
        return v;
    }
}
