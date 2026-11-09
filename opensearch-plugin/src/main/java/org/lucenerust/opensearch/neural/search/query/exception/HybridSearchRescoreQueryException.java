/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.search.query.exception;

import org.opensearch.OpenSearchException;

/**
 * Exception thrown when there is an issue with the hybrid search rescore query.
 */
public class HybridSearchRescoreQueryException extends OpenSearchException {

    public HybridSearchRescoreQueryException(Throwable cause) {
        super("rescore failed for hybrid query", cause);
    }
}
