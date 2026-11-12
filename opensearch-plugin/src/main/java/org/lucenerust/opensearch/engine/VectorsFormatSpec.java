/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

/**
 * The {@code KnnVectorsFormat} a vector field is written with, as the {@code knn_vector} mapper
 * names it in the field attribute {@link #ATTRIBUTE}: {@code hnsw:<m>:<beam>} for {@code
 * Lucene99HnswVectorsFormat}, {@code hnsw_sq:<encoding>:<m>:<beam>} for {@code
 * Lucene104HnswScalarQuantizedVectorsFormat}, {@code sq:<encoding>} for {@code
 * Lucene104ScalarQuantizedVectorsFormat}, {@code hnsw_hamming:<m>:<beam>} for k-NN's {@code
 * KNN990HnswBinaryVectorsFormat} and {@code hnsw_half:<m>:<beam>} for its {@code
 * KNN1040HnswHalfFloatVectorsFormat}, and {@code ivf:<target cluster size>:<rerank encoding>} (1-bit
 * codes) or {@code ivf:<target cluster size>:<rerank encoding>:<quantizer>} for this engine's {@code
 * IvfVectorsFormat}, and {@code hnsw_rq:<quantizer>:<m>:<beam>} for its {@code
 * HnswQuantizedVectorsFormat} ({@code encoding} is {@code ScalarEncoding}'s wire number, the
 * rerank encoding's for {@code ivf} and the quantizer's for {@code hnsw_rq}; an IVF field's cluster
 * size rides in {@code maxConn}, its quantizer in {@code beamWidth}).
 * The attribute stays in the segments' field infos, so a reopened writer merges the field in the
 * same format ({@code engine_writer.rs}'s {@code parse_vectors_format}).
 */
public record VectorsFormatSpec(int kind, int maxConn, int beamWidth, int encoding) {
    public static final String ATTRIBUTE = "accelerated.vectors_format";

    /** The codec's default, {@code Lucene99HnswVectorsFormat()}. */
    public static final VectorsFormatSpec DEFAULT = new VectorsFormatSpec(0, 0, 0, 0);

    /** {@code ScalarEncoding} wire numbers. */
    public static final int UNSIGNED_BYTE = 0;
    public static final int PACKED_NIBBLE = 1;
    public static final int SEVEN_BIT = 2;
    public static final int SINGLE_BIT = 3;
    public static final int DIBIT = 4;

    /** How many IVF quantizers there are: {@code KnnFieldConfig.IVF_QUANTIZERS}'s size. */
    static final int IVF_QUANTIZERS = 13;
    /** The wire number of {@code auto}: IVF only, an HNSW walk has no such codes. */
    static final int IVF_AUTO_QUANTIZER = 12;

    public static VectorsFormatSpec hnsw(int m, int beam) {
        return new VectorsFormatSpec(1, m, beam, 0);
    }

    public static VectorsFormatSpec hnswQuantized(int encoding, int m, int beam) {
        return new VectorsFormatSpec(2, m, beam, encoding);
    }

    public static VectorsFormatSpec quantized(int encoding) {
        return new VectorsFormatSpec(3, 0, 0, encoding);
    }

    public static VectorsFormatSpec hnswHamming(int m, int beam) {
        return new VectorsFormatSpec(4, m, beam, 0);
    }

    public static VectorsFormatSpec hnswHalfFloat(int m, int beam) {
        return new VectorsFormatSpec(5, m, beam, 0);
    }

    public static VectorsFormatSpec ivf(int targetClusterSize, int rerankEncoding, int quantizer) {
        return new VectorsFormatSpec(6, targetClusterSize, quantizer, rerankEncoding);
    }

    /** A quantized HNSW field: the graph walked on {@code quantizer}'s codes ({@code IVF_QUANTIZERS} wire number). */
    public static VectorsFormatSpec hnswRq(int quantizer, int m, int beam) {
        return new VectorsFormatSpec(7, m, beam, quantizer);
    }

    /** The attribute value. */
    public String attribute() {
        return switch (kind) {
            case 1 -> "hnsw:" + maxConn + ":" + beamWidth;
            case 2 -> "hnsw_sq:" + encoding + ":" + maxConn + ":" + beamWidth;
            case 3 -> "sq:" + encoding;
            case 4 -> "hnsw_hamming:" + maxConn + ":" + beamWidth;
            case 5 -> "hnsw_half:" + maxConn + ":" + beamWidth;
            case 6 -> "ivf:" + maxConn + ":" + encoding + (beamWidth == 0 ? "" : ":" + beamWidth);
            case 7 -> "hnsw_rq:" + encoding + ":" + maxConn + ":" + beamWidth;
            default -> "default";
        };
    }

    /** The spec an attribute value names; {@link #DEFAULT} for none. */
    public static VectorsFormatSpec of(String attribute) {
        if (attribute == null) {
            return DEFAULT;
        }
        String[] p = attribute.split(":");
        try {
            return switch (p[0]) {
                case "hnsw" -> p.length == 3 ? hnsw(Integer.parseInt(p[1]), Integer.parseInt(p[2])) : DEFAULT;
                case "hnsw_sq" -> p.length == 4
                    ? hnswQuantized(Integer.parseInt(p[1]), Integer.parseInt(p[2]), Integer.parseInt(p[3]))
                    : DEFAULT;
                case "sq" -> p.length == 2 ? quantized(Integer.parseInt(p[1])) : DEFAULT;
                case "hnsw_hamming" -> p.length == 3 ? hnswHamming(Integer.parseInt(p[1]), Integer.parseInt(p[2])) : DEFAULT;
                case "hnsw_half" -> p.length == 3 ? hnswHalfFloat(Integer.parseInt(p[1]), Integer.parseInt(p[2])) : DEFAULT;
                case "hnsw_rq" -> {
                    int quantizer = p.length == 4 ? Integer.parseInt(p[1]) : -1;
                    yield quantizer >= 0 && quantizer < IVF_QUANTIZERS && quantizer != IVF_AUTO_QUANTIZER
                        ? hnswRq(quantizer, Integer.parseInt(p[2]), Integer.parseInt(p[3]))
                        : DEFAULT;
                }
                case "ivf" -> {
                    int quantizer = p.length == 4 ? Integer.parseInt(p[3]) : 0;
                    yield (p.length == 3 || p.length == 4) && quantizer >= 0 && quantizer < IVF_QUANTIZERS
                        ? ivf(Integer.parseInt(p[1]), Integer.parseInt(p[2]), quantizer)
                        : DEFAULT;
                }
                default -> DEFAULT;
            };
        } catch (NumberFormatException e) {
            return DEFAULT;
        }
    }

    /** {@code engine_writer.rs}'s {@code decode_vector_schema} format part. */
    void encode(Blob b) {
        b.u8(kind);
        if (kind == 1 || kind == 2 || kind == 4 || kind == 5) {
            b.i32(maxConn).i32(beamWidth);
        }
        if (kind == 2 || kind == 3) {
            b.u8(encoding);
        }
        if (kind == 6) {
            b.i32(maxConn).u8(encoding).u8(beamWidth);
        }
        if (kind == 7) {
            b.i32(maxConn).i32(beamWidth).u8(encoding);
        }
    }
}
