/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.explain;

import org.lucenerust.opensearch.neural.processor.SearchShard;

/**
 * DTO class to store docId and search shard for a query.
 * Used in {@link org.lucenerust.opensearch.neural.processor.NormalizationProcessorWorkflow} to normalize scores across shards.
 */
public final class DocIdAtSearchShard {
    private final int docId;
    private final SearchShard searchShard;

    @java.lang.SuppressWarnings("all")
    public DocIdAtSearchShard(final int docId, final SearchShard searchShard) {
        this.docId = docId;
        this.searchShard = searchShard;
    }

    @java.lang.SuppressWarnings("all")
    public int getDocId() {
        return this.docId;
    }

    @java.lang.SuppressWarnings("all")
    public SearchShard getSearchShard() {
        return this.searchShard;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof DocIdAtSearchShard)) return false;
        final DocIdAtSearchShard other = (DocIdAtSearchShard) o;
        if (this.getDocId() != other.getDocId()) return false;
        final java.lang.Object this$searchShard = this.getSearchShard();
        final java.lang.Object other$searchShard = other.getSearchShard();
        if (this$searchShard == null ? other$searchShard != null : !this$searchShard.equals(other$searchShard)) return false;
        return true;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        result = result * PRIME + this.getDocId();
        final java.lang.Object $searchShard = this.getSearchShard();
        result = result * PRIME + ($searchShard == null ? 43 : $searchShard.hashCode());
        return result;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public java.lang.String toString() {
        return "DocIdAtSearchShard(docId=" + this.getDocId() + ", searchShard=" + this.getSearchShard() + ")";
    }
}
