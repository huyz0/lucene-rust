/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.index.VectorEncoding;
import org.lucenerust.opensearch.engine.VectorsFormatSpec;
import org.opensearch.index.mapper.MapperParsingException;

import java.util.Locale;
import java.util.Map;

/**
 * A {@code knn_vector} mapping resolved as the stock k-NN plugin resolves it ({@code
 * KNNVectorFieldMapper.Builder} with its engine and mode/compression resolvers), into the vector
 * format the Rust writer builds: Lucene's HNSW graph over float or byte vectors, quantized with a
 * Lucene104 scalar encoding when the method's encoder or the compression asks for one, or k-NN's
 * {@code KNN990HnswBinaryVectorsFormat} graph over packed bit vectors ({@code data_type: binary},
 * {@code hamming} space).
 *
 * <p>A mapping without an engine uses {@code lucene}; {@code faiss} is refused. The search reads the
 * raw vectors through the graph, so a quantized field's scores are the exact ones the stock plugin
 * rescores to.
 *
 * @param dimension the mapping's dimension: bits for a {@link #binary} field
 * @param binary packed bit vectors, {@code dimension / 8} bytes each
 * @param encoding a {@link VectorsFormatSpec} scalar encoding wire number, {@code -1} for full precision
 */
public record KnnFieldConfig(
    int dimension,
    VectorEncoding dataType,
    boolean binary,
    SpaceType spaceType,
    String engine,
    int m,
    int efConstruction,
    int encoding,
    String mode,
    String compression
) {
    public static final String FAISS = "faiss";
    public static final String LUCENE = "lucene";
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

    /** The format the Rust writer builds the field with. */
    public VectorsFormatSpec format() {
        if (binary) {
            return VectorsFormatSpec.hnswHamming(m, efConstruction);
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
            case "float" -> VectorEncoding.FLOAT32;
            case "byte", "binary" -> VectorEncoding.BYTE;
            default -> throw new MapperParsingException(
                String.format(Locale.ROOT, "Invalid value provided for [data_type] field: [%s]. Supported values are [float, byte, binary]", dataType)
            );
        };
        boolean binary = dataType.equalsIgnoreCase("binary");

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
            level = encodingType == VectorEncoding.BYTE ? "1x" : "32x";
        }

        // Stock parses engine names in any case; the lucene engine is the default here, and faiss
        // (whose .faiss files this plugin neither reads nor writes) is refused.
        String engine = pick(lower(topEngine), lower(methodEngine), "Engine", "undefined");
        if (engine == null) {
            engine = LUCENE;
        }
        switch (engine) {
            case LUCENE -> {}
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
                methodName.equals("ivf")
                    ? "[method] [ivf] needs a trained model, which this node does not support"
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
        return new KnnFieldConfig(dimension, encodingType, binary, spaceType, engine, m, efConstruction, encoding, resolvedMode, level);
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
