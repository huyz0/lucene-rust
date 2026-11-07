/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.rest;

import java.util.List;

/**
 * k-NN's {@code KNNInvalidIndicesException}, under its class name, which the error's {@code type}
 * ({@code k_n_n_invalid_indices_exception}) is made from.
 */
public final class KNNInvalidIndicesException extends RuntimeException {
    private final List<String> invalidIndices;

    public KNNInvalidIndicesException(List<String> invalidIndices, String message) {
        super(message);
        this.invalidIndices = invalidIndices;
    }

    public List<String> getInvalidIndices() {
        return invalidIndices;
    }

    @Override
    public String toString() {
        return "[KNN] " + String.join(",", invalidIndices) + ' ' + super.toString();
    }
}
