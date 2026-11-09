/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.util;

import java.util.List;
import java.util.Objects;

import org.lucenerust.opensearch.neural.processor.CompoundTopDocs;

/** Helpers the normalization techniques share. */
public final class ProcessorUtils {
    private ProcessorUtils() {}

    /** The number of sub-queries, from the first shard result that has any. */
    public static int getNumOfSubqueries(final List<CompoundTopDocs> queryTopDocs) {
        return queryTopDocs.stream()
            .filter(Objects::nonNull)
            .filter(topDocs -> !topDocs.getTopDocs().isEmpty())
            .findAny()
            .get()
            .getTopDocs()
            .size();
    }
}
