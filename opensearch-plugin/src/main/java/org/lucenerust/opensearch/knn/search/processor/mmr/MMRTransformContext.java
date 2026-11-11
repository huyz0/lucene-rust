/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.knn.search.processor.mmr;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.lucenerust.opensearch.knn.index.SpaceType;
import org.lucenerust.opensearch.knn.index.VectorDataType;
import org.opensearch.transport.client.Client;
import java.util.List;

/**
 * A DTO to hold the context for MMR query transformer to transform the request
 */
public class MMRTransformContext {
    private final Integer candidates;
    // During transform, we will also collect some info for MMR rerank processor to use later
    private final MMRRerankContext mmrRerankContext;
    // During transform, we need to figure out the knn_vector space type based on the index metadata
    private final List<IndexMetadata> localIndexMetadataList;
    private final List<String> remoteIndices;
    private final SpaceType userProvidedSpaceType;
    private final String userProvidedVectorFieldPath;
    private final VectorDataType userProvidedVectorDataType;
    private final Client client;
    private boolean isVectorFieldInfoResolved;

    
    
    public Integer getCandidates() {
        return this.candidates;
    }

    
    
    public MMRRerankContext getMmrRerankContext() {
        return this.mmrRerankContext;
    }

    
    
    public List<IndexMetadata> getLocalIndexMetadataList() {
        return this.localIndexMetadataList;
    }

    
    
    public List<String> getRemoteIndices() {
        return this.remoteIndices;
    }

    
    
    public SpaceType getUserProvidedSpaceType() {
        return this.userProvidedSpaceType;
    }

    
    
    public String getUserProvidedVectorFieldPath() {
        return this.userProvidedVectorFieldPath;
    }

    
    
    public VectorDataType getUserProvidedVectorDataType() {
        return this.userProvidedVectorDataType;
    }

    
    
    public Client getClient() {
        return this.client;
    }

    
    
    public boolean isVectorFieldInfoResolved() {
        return this.isVectorFieldInfoResolved;
    }

    
    
    public void setVectorFieldInfoResolved(final boolean isVectorFieldInfoResolved) {
        this.isVectorFieldInfoResolved = isVectorFieldInfoResolved;
    }

    @java.lang.Override
    
    
    public boolean equals(final java.lang.Object o) {
        if (o == this) return true;
        if (!(o instanceof MMRTransformContext)) return false;
        final MMRTransformContext other = (MMRTransformContext) o;
        if (!other.canEqual((java.lang.Object) this)) return false;
        if (this.isVectorFieldInfoResolved() != other.isVectorFieldInfoResolved()) return false;
        final java.lang.Object this$candidates = this.getCandidates();
        final java.lang.Object other$candidates = other.getCandidates();
        if (this$candidates == null ? other$candidates != null : !this$candidates.equals(other$candidates)) return false;
        final java.lang.Object this$mmrRerankContext = this.getMmrRerankContext();
        final java.lang.Object other$mmrRerankContext = other.getMmrRerankContext();
        if (this$mmrRerankContext == null ? other$mmrRerankContext != null : !this$mmrRerankContext.equals(other$mmrRerankContext)) return false;
        final java.lang.Object this$localIndexMetadataList = this.getLocalIndexMetadataList();
        final java.lang.Object other$localIndexMetadataList = other.getLocalIndexMetadataList();
        if (this$localIndexMetadataList == null ? other$localIndexMetadataList != null : !this$localIndexMetadataList.equals(other$localIndexMetadataList)) return false;
        final java.lang.Object this$remoteIndices = this.getRemoteIndices();
        final java.lang.Object other$remoteIndices = other.getRemoteIndices();
        if (this$remoteIndices == null ? other$remoteIndices != null : !this$remoteIndices.equals(other$remoteIndices)) return false;
        final java.lang.Object this$userProvidedSpaceType = this.getUserProvidedSpaceType();
        final java.lang.Object other$userProvidedSpaceType = other.getUserProvidedSpaceType();
        if (this$userProvidedSpaceType == null ? other$userProvidedSpaceType != null : !this$userProvidedSpaceType.equals(other$userProvidedSpaceType)) return false;
        final java.lang.Object this$userProvidedVectorFieldPath = this.getUserProvidedVectorFieldPath();
        final java.lang.Object other$userProvidedVectorFieldPath = other.getUserProvidedVectorFieldPath();
        if (this$userProvidedVectorFieldPath == null ? other$userProvidedVectorFieldPath != null : !this$userProvidedVectorFieldPath.equals(other$userProvidedVectorFieldPath)) return false;
        final java.lang.Object this$userProvidedVectorDataType = this.getUserProvidedVectorDataType();
        final java.lang.Object other$userProvidedVectorDataType = other.getUserProvidedVectorDataType();
        if (this$userProvidedVectorDataType == null ? other$userProvidedVectorDataType != null : !this$userProvidedVectorDataType.equals(other$userProvidedVectorDataType)) return false;
        final java.lang.Object this$client = this.getClient();
        final java.lang.Object other$client = other.getClient();
        if (this$client == null ? other$client != null : !this$client.equals(other$client)) return false;
        return true;
    }

    
    
    protected boolean canEqual(final java.lang.Object other) {
        return other instanceof MMRTransformContext;
    }

    @java.lang.Override
    
    
    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        result = result * PRIME + (this.isVectorFieldInfoResolved() ? 79 : 97);
        final java.lang.Object $candidates = this.getCandidates();
        result = result * PRIME + ($candidates == null ? 43 : $candidates.hashCode());
        final java.lang.Object $mmrRerankContext = this.getMmrRerankContext();
        result = result * PRIME + ($mmrRerankContext == null ? 43 : $mmrRerankContext.hashCode());
        final java.lang.Object $localIndexMetadataList = this.getLocalIndexMetadataList();
        result = result * PRIME + ($localIndexMetadataList == null ? 43 : $localIndexMetadataList.hashCode());
        final java.lang.Object $remoteIndices = this.getRemoteIndices();
        result = result * PRIME + ($remoteIndices == null ? 43 : $remoteIndices.hashCode());
        final java.lang.Object $userProvidedSpaceType = this.getUserProvidedSpaceType();
        result = result * PRIME + ($userProvidedSpaceType == null ? 43 : $userProvidedSpaceType.hashCode());
        final java.lang.Object $userProvidedVectorFieldPath = this.getUserProvidedVectorFieldPath();
        result = result * PRIME + ($userProvidedVectorFieldPath == null ? 43 : $userProvidedVectorFieldPath.hashCode());
        final java.lang.Object $userProvidedVectorDataType = this.getUserProvidedVectorDataType();
        result = result * PRIME + ($userProvidedVectorDataType == null ? 43 : $userProvidedVectorDataType.hashCode());
        final java.lang.Object $client = this.getClient();
        result = result * PRIME + ($client == null ? 43 : $client.hashCode());
        return result;
    }

    @java.lang.Override
    
    
    public java.lang.String toString() {
        return "MMRTransformContext(candidates=" + this.getCandidates() + ", mmrRerankContext=" + this.getMmrRerankContext() + ", localIndexMetadataList=" + this.getLocalIndexMetadataList() + ", remoteIndices=" + this.getRemoteIndices() + ", userProvidedSpaceType=" + this.getUserProvidedSpaceType() + ", userProvidedVectorFieldPath=" + this.getUserProvidedVectorFieldPath() + ", userProvidedVectorDataType=" + this.getUserProvidedVectorDataType() + ", client=" + this.getClient() + ", isVectorFieldInfoResolved=" + this.isVectorFieldInfoResolved() + ")";
    }

    
    
    public MMRTransformContext(final Integer candidates, final MMRRerankContext mmrRerankContext, final List<IndexMetadata> localIndexMetadataList, final List<String> remoteIndices, final SpaceType userProvidedSpaceType, final String userProvidedVectorFieldPath, final VectorDataType userProvidedVectorDataType, final Client client, final boolean isVectorFieldInfoResolved) {
        if (candidates == null) {
            throw new java.lang.NullPointerException("candidates is marked non-null but is null");
        }
        if (mmrRerankContext == null) {
            throw new java.lang.NullPointerException("mmrRerankContext is marked non-null but is null");
        }
        if (localIndexMetadataList == null) {
            throw new java.lang.NullPointerException("localIndexMetadataList is marked non-null but is null");
        }
        if (remoteIndices == null) {
            throw new java.lang.NullPointerException("remoteIndices is marked non-null but is null");
        }
        this.candidates = candidates;
        this.mmrRerankContext = mmrRerankContext;
        this.localIndexMetadataList = localIndexMetadataList;
        this.remoteIndices = remoteIndices;
        this.userProvidedSpaceType = userProvidedSpaceType;
        this.userProvidedVectorFieldPath = userProvidedVectorFieldPath;
        this.userProvidedVectorDataType = userProvidedVectorDataType;
        this.client = client;
        this.isVectorFieldInfoResolved = isVectorFieldInfoResolved;
    }
}
