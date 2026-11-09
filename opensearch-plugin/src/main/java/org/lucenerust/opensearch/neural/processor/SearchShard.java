/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor;

import org.opensearch.search.SearchShardTarget;

/**
 * DTO class to store index, shardId and nodeId for a search shard.
 */
public final class SearchShard {
    private final String index;
    private final int shardId;
    private final String nodeId;

    /**
     * Create SearchShard from SearchShardTarget
     * @param searchShardTarget
     * @return SearchShard
     */
    public static SearchShard createSearchShard(final SearchShardTarget searchShardTarget) {
        return new SearchShard(searchShardTarget.getIndex(), searchShardTarget.getShardId().id(), searchShardTarget.getNodeId());
    }

    @java.lang.SuppressWarnings("all")
    public String getIndex() {
        return this.index;
    }

    @java.lang.SuppressWarnings("all")
    public int getShardId() {
        return this.shardId;
    }

    @java.lang.SuppressWarnings("all")
    public String getNodeId() {
        return this.nodeId;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof SearchShard)) return false;
        final SearchShard other = (SearchShard) o;
        if (this.getShardId() != other.getShardId()) return false;
        final java.lang.Object this$index = this.getIndex();
        final java.lang.Object other$index = other.getIndex();
        if (this$index == null ? other$index != null : !this$index.equals(other$index)) return false;
        final java.lang.Object this$nodeId = this.getNodeId();
        final java.lang.Object other$nodeId = other.getNodeId();
        if (this$nodeId == null ? other$nodeId != null : !this$nodeId.equals(other$nodeId)) return false;
        return true;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        result = result * PRIME + this.getShardId();
        final java.lang.Object $index = this.getIndex();
        result = result * PRIME + ($index == null ? 43 : $index.hashCode());
        final java.lang.Object $nodeId = this.getNodeId();
        result = result * PRIME + ($nodeId == null ? 43 : $nodeId.hashCode());
        return result;
    }

    @java.lang.Override
    @java.lang.SuppressWarnings("all")
    public java.lang.String toString() {
        return "SearchShard(index=" + this.getIndex() + ", shardId=" + this.getShardId() + ", nodeId=" + this.getNodeId() + ")";
    }

    @java.lang.SuppressWarnings("all")
    public SearchShard(final String index, final int shardId, final String nodeId) {
        this.index = index;
        this.shardId = shardId;
        this.nodeId = nodeId;
    }
}
