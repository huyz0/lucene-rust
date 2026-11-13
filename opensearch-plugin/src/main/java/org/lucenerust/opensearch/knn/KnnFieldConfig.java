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
 * <p>The {@code extended} engine is this node's own, opt-in: {@code method: ivf} is an IVF index
 * with quantized codes beside the raw vectors ({@code IvfVectorsFormat}), no training
 * step; {@code parameters.target_cluster_size} (16 to 65536, default 384) sizes its clusters,
 * {@code parameters.quantizer} ({@link #IVF_QUANTIZERS}: 1-bit {@code bbq} by default, {@code
 * osq2}/{@code osq4}/{@code osq7} for Lucene's optimized scalar quantization at 2, 4 or 7 bits,
 * {@code erq2} to {@code erq8} for Extended RaBitQ at 2 to 8 bits, {@code rabitq} for 1-bit RaBitQ, {@code auto}
 * to pick among bbq, osq2 and erq2 to erq4 per segment, at each flush and merge, by what the
 * field's own vectors reach)
 * codes the clusters and {@code parameters.rerank_encoding} ({@code float}, {@code half_float} by
 * default, {@code bfloat16} or {@code int8}) stores the copy it reranks from. Its {@code method: hnsw}
 * is Lucene's graph (built on the raw vectors, {@code parameters.m} and {@code
 * parameters.ef_construction} as for {@code lucene}) walked on quantized codes, the candidates
 * reranked on the raw vectors ({@code HnswQuantizedVectorsFormat}); {@code
 * parameters.quantizer} ({@link #HNSW_QUANTIZERS}: {@code erq2} by default) picks the codes. Lucene's
 * own scalar quantization stays the {@code lucene} engine's ({@code encoder}, {@code
 * compression_level}). Its {@code method: disk_hnsw} is the same graph with a page-aligned copy of every
 * node (its neighbours and vector in one record, {@code DiskHnswVectorsFormat}), searched by
 * ranking candidates on {@code parameters.quantizer} ({@link #DISK_HNSW_QUANTIZERS}: {@code bbq} by
 * default) codes and reading one record per expanded node: for an index larger than memory, where it
 * reads far fewer pages than {@code hnsw}, at the cost of a second copy of the vectors on disk. All three
 * methods are for float fields only, in an index on the Rust engine
 * ({@code index.engine.accelerated}), which builds the index. A stock node does not know the engine,
 * so it refuses the mapping rather than reading the field another way.
 *
 * @param dimension the mapping's dimension: bits for a {@link #binary} field
 * @param binary packed bit vectors, {@code dimension / 8} bytes each
 * @param halfFloat float vectors stored as half floats
 * @param encoding a {@link VectorsFormatSpec} scalar encoding wire number, {@code -1} for full precision
 * @param clusterSize an IVF field's target cluster size, {@code 0} for an HNSW one
 * @param rerankEncoding an IVF field's rerank encoding wire number ({@link #RERANK_ENCODINGS})
 * @param quantizer a {@code extended} field's quantizer wire number ({@link #IVF_QUANTIZERS})
 * @param disk a {@code disk_hnsw} field of the {@code extended} engine
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
    int rerankEncoding,
    int quantizer,
    boolean disk
) {
    public static final String FAISS = "faiss";
    public static final String LUCENE = "lucene";
    public static final String EXTENDED = "extended";
    public static final String IVF = "ivf";
    public static final String DISK_HNSW = "disk_hnsw";
    public static final int DEFAULT_CLUSTER_SIZE = 384;
    /** {@code IvfVectorsFormat}'s rerank encodings, by wire number. */
    public static final List<String> RERANK_ENCODINGS = List.of("float", "half_float", "bfloat16", "int8");
    /** {@code IvfVectorsFormat}'s quantizers, by wire number ({@code per_field_knn_vectors.rs}'s {@code IVF_QUANTIZERS}). */
    public static final List<String> IVF_QUANTIZERS = List.of(
        "bbq",
        "osq2",
        "osq4",
        "osq7",
        "erq2",
        "erq3",
        "erq4",
        "erq5",
        "erq6",
        "erq7",
        "erq8",
        "rabitq",
        "auto"
    );
    /** The quantizers of the {@code extended} engine's {@code hnsw}: 1-bit RaBitQ and Extended RaBitQ at 2 to 8 bits. */
    public static final List<String> HNSW_QUANTIZERS = List.of("rabitq", "erq2", "erq3", "erq4", "erq5", "erq6", "erq7", "erq8");
    static final String DEFAULT_HNSW_QUANTIZER = "erq2";
    /** The codes a {@code disk_hnsw} ranks candidates on: every IVF quantizer but {@code auto}. */
    public static final List<String> DISK_HNSW_QUANTIZERS = IVF_QUANTIZERS.subList(0, IVF_QUANTIZERS.size() - 1);
    static final String DEFAULT_DISK_HNSW_QUANTIZER = "bbq";
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

    /** The mapping's {@code data_type}: {@code float}, {@code byte}, {@code binary} or {@code half_float}. */
    public String dataTypeName() {
        return binary ? "binary" : halfFloat ? "half_float" : dataType == VectorEncoding.BYTE ? "byte" : "float";
    }

    /** An IVF field ({@code extended} engine, {@code ivf} method). */
    public boolean ivf() {
        return EXTENDED.equals(engine) && clusterSize > 0;
    }

    /** A field of the Rust engine's own: {@code extended} engine, {@code ivf}, {@code hnsw} or {@code disk_hnsw}. */
    public boolean rustEngine() {
        return EXTENDED.equals(engine);
    }

    /** The format the Rust writer builds the field with. */
    public VectorsFormatSpec format() {
        if (ivf()) {
            return VectorsFormatSpec.ivf(clusterSize, rerankEncoding, quantizer);
        }
        if (disk) {
            return VectorsFormatSpec.hnswDisk(quantizer, m, efConstruction);
        }
        if (rustEngine()) {
            return VectorsFormatSpec.hnswRq(quantizer, m, efConstruction);
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
            case EXTENDED -> {
                return rust(dimension, encodingType, dataType, spaceType, methodName, params, resolvedMode, level);
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
                        + "use the [extended] engine's [ivf], which needs none"
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
            return new KnnFieldConfig(dimension, encodingType, false, true, spaceType, engine, m, efConstruction, -1, resolvedMode, level, 0, 0, 0);
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
        return new KnnFieldConfig(dimension, encodingType, binary, false, spaceType, engine, m, efConstruction, encoding, resolvedMode, level, 0, 0, 0, false);
    }

    /** A {@code extended} engine mapping: {@code method: ivf} or {@code hnsw} over float vectors. */
    private static KnnFieldConfig rust(
        int dimension,
        VectorEncoding encodingType,
        String dataType,
        SpaceType spaceType,
        String methodName,
        Map<String, Object> params,
        String mode,
        String level
    ) {
        if (methodName.equals(IVF) == false && methodName.equals("hnsw") == false && methodName.equals(DISK_HNSW) == false) {
            throw new MapperParsingException(
                String.format(Locale.ROOT, "Invalid method name for the [extended] engine: %s; supported: [ivf, hnsw, disk_hnsw]", methodName)
            );
        }
        if (dataType.equalsIgnoreCase("float") == false) {
            throw new MapperParsingException(
                String.format(
                    Locale.ROOT,
                    "[data_type] [%s] is not supported by the [extended] engine's [%s]; use [float]",
                    dataType.toLowerCase(Locale.ROOT),
                    methodName
                )
            );
        }
        if (mode != null || level != null) {
            throw new MapperParsingException(
                methodName.equals(IVF)
                    ? "[mode] and [compression_level] are not supported by the [extended] engine's [ivf]; set [parameters.rerank_encoding]"
                    : "[mode] and [compression_level] are not supported by the [extended] engine's ["
                        + methodName
                        + "]; set [parameters.quantizer]"
            );
        }
        if (methodName.equals("hnsw") || methodName.equals(DISK_HNSW)) {
            return rustHnsw(dimension, encodingType, spaceType, params, methodName.equals(DISK_HNSW));
        }
        int clusterSize = DEFAULT_CLUSTER_SIZE;
        int rerank = 1;
        int quantizer = 0;
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
                    quantizer = IVF_QUANTIZERS.indexOf(String.valueOf(p.getValue()).toLowerCase(Locale.ROOT));
                    if (quantizer < 0) {
                        throw new MapperParsingException(
                            String.format(Locale.ROOT, "Invalid [quantizer]: %s; supported: %s", p.getValue(), IVF_QUANTIZERS)
                        );
                    }
                }
                default -> throw new MapperParsingException(String.format(Locale.ROOT, "Invalid parameter for ivf: %s", p.getKey()));
            }
        }
        return new KnnFieldConfig(dimension, encodingType, false, false, spaceType, EXTENDED, 0, 0, -1, null, null, clusterSize, rerank, quantizer, false);
    }

    /** A {@code extended} engine {@code hnsw} mapping: the graph walked on {@link #HNSW_QUANTIZERS} codes. */
    private static KnnFieldConfig rustHnsw(
        int dimension,
        VectorEncoding encodingType,
        SpaceType spaceType,
        Map<String, Object> params,
        boolean disk
    ) {
        int m = DEFAULT_M;
        int efConstruction = DEFAULT_EF_CONSTRUCTION;
        String quantizer = disk ? DEFAULT_DISK_HNSW_QUANTIZER : DEFAULT_HNSW_QUANTIZER;
        String method = disk ? DISK_HNSW : "hnsw";
        List<String> allowed = disk ? DISK_HNSW_QUANTIZERS : HNSW_QUANTIZERS;
        for (Map.Entry<String, Object> p : params.entrySet()) {
            switch (p.getKey()) {
                case "m" -> m = positiveInt(p.getValue(), "m", MAX_M);
                case "ef_construction" -> efConstruction = positiveInt(p.getValue(), "ef_construction", MAX_EF_CONSTRUCTION);
                case "ef_search" -> throw new MapperParsingException("Invalid parameter: ef_search for the extended engine; set it in the query");
                case "quantizer" -> {
                    quantizer = String.valueOf(p.getValue()).toLowerCase(Locale.ROOT);
                    if (allowed.contains(quantizer) == false) {
                        throw new MapperParsingException(
                            "auto".equals(quantizer)
                                ? String.format(
                                    Locale.ROOT,
                                    "[quantizer] [auto] is for the [extended] engine's [ivf] only; its [%s] supports %s",
                                    method,
                                    allowed
                                )
                                : disk == false && IVF_QUANTIZERS.contains(quantizer)
                                ? String.format(
                                    Locale.ROOT,
                                    "[quantizer] [%s] is Lucene's scalar quantization: use the [lucene] engine's [encoder] or [compression_level]; "
                                        + "the [extended] engine's [hnsw] supports %s",
                                    quantizer,
                                    allowed
                                )
                                : String.format(Locale.ROOT, "Invalid [quantizer]: %s; supported: %s", p.getValue(), allowed)
                        );
                    }
                }
                default -> throw new MapperParsingException(String.format(Locale.ROOT, "Invalid parameter for %s: %s", method, p.getKey()));
            }
        }
        return new KnnFieldConfig(
            dimension,
            encodingType,
            false,
            false,
            spaceType,
            EXTENDED,
            m,
            efConstruction,
            -1,
            null,
            null,
            0,
            0,
            IVF_QUANTIZERS.indexOf(quantizer),
            disk
        );
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
