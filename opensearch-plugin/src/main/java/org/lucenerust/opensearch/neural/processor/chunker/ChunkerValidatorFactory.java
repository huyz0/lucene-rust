/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.chunker;

import java.util.Map;

/**
 * A factory to create chunker validator.
 */
public class ChunkerValidatorFactory {
    private final Map<String, Validator> ALGORITHM_TO_VALIDATOR = Map.of(FixedTokenLengthChunker.ALGORITHM_NAME, new FixedTokenLengthChunker(), DelimiterChunker.ALGORITHM_NAME, new DelimiterChunker(), FixedCharLengthChunker.ALGORITHM_NAME, new FixedCharLengthChunker());

    public Validator getValidator(final String algorithm) {
        java.util.Objects.requireNonNull(algorithm, "algorithm is marked non-null but is null");
        return ALGORITHM_TO_VALIDATOR.get(algorithm);
    }
}
