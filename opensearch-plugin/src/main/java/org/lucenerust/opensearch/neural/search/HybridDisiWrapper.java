/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.search;

import org.apache.lucene.search.DisiWrapper;
import org.apache.lucene.search.Scorer;

/**
 * Wrapper for DisiWrapper, saves state of sub-queries for performance reasons
 */
public class HybridDisiWrapper extends DisiWrapper {
    // index of disi wrapper sub-query object when its part of the hybrid query
    private final int subQueryIndex;

    public HybridDisiWrapper(Scorer scorer, int subQueryIndex) {
        super(scorer, false);
        this.subQueryIndex = subQueryIndex;
    }

    @java.lang.SuppressWarnings("all")
    public int getSubQueryIndex() {
        return this.subQueryIndex;
    }
}
