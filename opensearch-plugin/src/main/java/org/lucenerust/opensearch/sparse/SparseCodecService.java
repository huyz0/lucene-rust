/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.apache.lucene.codecs.Codec;
import org.opensearch.index.codec.CodecService;
import org.opensearch.index.codec.CodecServiceConfig;

/**
 * neural-search's {@code SparseCodecService}: OpenSearch's engine writes an {@code index.sparse}
 * index's segments with {@link SparseCodec} over the codec the index names.
 */
public final class SparseCodecService extends CodecService {
    public SparseCodecService(CodecServiceConfig config) {
        super(config.getMapperService(), config.getIndexSettings(), config.getLogger(), config.getAdditionalCodecs());
    }

    @Override
    public Codec codec(String name) {
        return new SparseCodec(super.codec(name));
    }
}
