/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.explain;

import java.util.Map;
import java.util.List;

/**
 * DTO class to store value and description for explain details.
 * Used in {@link org.lucenerust.opensearch.neural.processor.NormalizationProcessorWorkflow} to normalize scores across shards.
 */
public final class ExplanationDetails {
    private final int docId;
    private final List<Map.Entry<Float, String>> scoreDetails;

    public ExplanationDetails(List<Map.Entry<Float, String>> scoreDetails) {
        // pass docId as -1 to match docId in SearchHit
        // https://github.com/opensearch-project/OpenSearch/blob/main/server/src/main/java/org/opensearch/search/SearchHit.java#L170
        this(-1, scoreDetails);
    }

    @java.lang.SuppressWarnings("all")
    public int getDocId() {
        return this.docId;
    }

    @java.lang.SuppressWarnings("all")
    public List<Map.Entry<Float, String>> getScoreDetails() {
        return this.scoreDetails;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof ExplanationDetails)) return false;
        final ExplanationDetails other = (ExplanationDetails) o;
        if (this.getDocId() != other.getDocId()) return false;
        final java.lang.Object this$scoreDetails = this.getScoreDetails();
        final java.lang.Object other$scoreDetails = other.getScoreDetails();
        if (this$scoreDetails == null ? other$scoreDetails != null : !this$scoreDetails.equals(other$scoreDetails)) return false;
        return true;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        result = result * PRIME + this.getDocId();
        final java.lang.Object $scoreDetails = this.getScoreDetails();
        result = result * PRIME + ($scoreDetails == null ? 43 : $scoreDetails.hashCode());
        return result;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public java.lang.String toString() {
        return "ExplanationDetails(docId=" + this.getDocId() + ", scoreDetails=" + this.getScoreDetails() + ")";
    }

    @java.lang.SuppressWarnings("all")
    public ExplanationDetails(final int docId, final List<Map.Entry<Float, String>> scoreDetails) {
        this.docId = docId;
        this.scoreDetails = scoreDetails;
    }
}
