/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.lucenerust.opensearch.engine.VectorsFormatSpec;
import org.lucenerust.opensearch.knn.KnnFieldConfig;
import org.lucenerust.opensearch.knn.KnnVectorFieldMapper;
import org.opensearch.index.codec.CodecService;
import org.opensearch.index.codec.CodecServiceConfig;
import org.opensearch.index.mapper.MappedFieldType;
import org.opensearch.index.mapper.MapperService;

/**
 * The codecs OpenSearch's own engine writes a k-NN index with ({@code KNNCodecService}): each
 * codec of the index's service with its {@code knn_vector} fields routed to the vector format
 * their mapping resolves to ({@link KnnFieldConfig#format()}), the format the Rust writer builds
 * for the same mapping. Every other field keeps the codec's format.
 */
public final class KnnCodecService extends CodecService {
    private final MapperService mapperService;

    public KnnCodecService(CodecServiceConfig config) {
        super(config.getMapperService(), config.getIndexSettings(), config.getLogger(), config.getAdditionalCodecs());
        this.mapperService = config.getMapperService();
    }

    @Override
    public Codec codec(String name) {
        Codec delegate = super.codec(name);
        return mapperService == null ? delegate : new KnnFieldsCodec(delegate, mapperService);
    }

    /** {@code delegate} under its own name, so a reader opens the segments with it. */
    private static final class KnnFieldsCodec extends FilterCodec {
        private final KnnVectorsFormat vectors;

        KnnFieldsCodec(Codec delegate, MapperService mapperService) {
            super(delegate.getName(), delegate);
            KnnVectorsFormat fallback = delegate.knnVectorsFormat();
            this.vectors = new PerFieldKnnVectorsFormat() {
                @Override
                public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                    MappedFieldType ft = mapperService.fieldType(field);
                    if (ft instanceof KnnVectorFieldMapper.KnnVectorFieldType knn) {
                        return format(knn.config().format());
                    }
                    return fallback instanceof PerFieldKnnVectorsFormat p ? p.getKnnVectorsFormatForField(field) : fallback;
                }

                /** The mapping's limit for a knn_vector field, as k-NN's lucene engine allows. */
                @Override
                public int getMaxDimensions(String field) {
                    return mapperService.fieldType(field) instanceof KnnVectorFieldMapper.KnnVectorFieldType
                        ? KnnFieldConfig.MAX_DIMENSION
                        : super.getMaxDimensions(field);
                }
            };
        }

        @Override
        public KnnVectorsFormat knnVectorsFormat() {
            return vectors;
        }
    }

    /**
     * The format a {@link VectorsFormatSpec} names, under the names the Rust writer records: Lucene's
     * own for the scalar-quantized formats (k-NN's {@code KNN1040} names read only some encodings).
     */
    static KnnVectorsFormat format(VectorsFormatSpec spec) {
        return switch (spec.kind()) {
            case 1 -> new Lucene99HnswVectorsFormat(spec.maxConn(), spec.beamWidth());
            case 2 -> new Lucene104HnswScalarQuantizedVectorsFormat(encoding(spec), spec.maxConn(), spec.beamWidth());
            case 3 -> new Lucene104ScalarQuantizedVectorsFormat(encoding(spec));
            case 4 -> new KNN990HnswBinaryVectorsFormat(spec.maxConn(), spec.beamWidth());
            case 5 -> new KNN1040HnswHalfFloatVectorsFormat(spec.maxConn(), spec.beamWidth());
            default -> new Lucene99HnswVectorsFormat();
        };
    }

    private static ScalarEncoding encoding(VectorsFormatSpec spec) {
        return ScalarEncoding.fromWireNumber(spec.encoding())
            .orElseThrow(() -> new IllegalArgumentException("unknown scalar encoding " + spec.encoding()));
    }
}
