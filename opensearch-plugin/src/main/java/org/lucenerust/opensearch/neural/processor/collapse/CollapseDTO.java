/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.collapse;

import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Sort;
import org.lucenerust.opensearch.neural.processor.CompoundTopDocs;
import org.lucenerust.opensearch.neural.processor.combination.CombineScoresDto;
import org.opensearch.search.query.QuerySearchResult;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Data Transfer Object (DTO) for managing collapse-related search data across shards.
 * Contains both base collapse information and shard-specific details for search operations.
 */
public class CollapseDTO {
    // Base collapse fields
    private final List<CompoundTopDocs> collapseQueryTopDocs;
    private final List<QuerySearchResult> collapseQuerySearchResults;
    private final Sort collapseSort;
    private final boolean isFetchPhaseExecuted;
    private final CombineScoresDto collapseCombineScoresDTO;
    // Shard-specific fields
    private List<? extends Map.Entry<?, FieldDoc>> relevantCollapseEntries;
    private String collapseField;
    private CompoundTopDocs updatedCollapseTopDocs;
    private int collapseShardIndex;
    private Class<?> collapseFieldType;

    /**
     * Constructor to create a new CollapseDTO instance with initial collapse parameters.
     *
     * @param collapseQueryTopDocs List of compound top documents from collapse query
     * @param collapseQuerySearchResults List of search results from collapse query
     * @param collapseSort Sort criteria for collapse operation
     * @param isFetchPhaseExecuted Flag indicating if  fetch phase is complete
     * @param collapseCombineScoresDTO DTO containing score combination parameters
     * @param collapseFieldType Contains class type of collapse field
     */
    public CollapseDTO(List<CompoundTopDocs> collapseQueryTopDocs, List<QuerySearchResult> collapseQuerySearchResults, Sort collapseSort, boolean isFetchPhaseExecuted, CombineScoresDto collapseCombineScoresDTO, Class<?> collapseFieldType) {
        this.collapseQueryTopDocs = Collections.unmodifiableList(collapseQueryTopDocs);
        this.collapseQuerySearchResults = collapseQuerySearchResults;
        this.collapseSort = collapseSort;
        this.isFetchPhaseExecuted = isFetchPhaseExecuted;
        this.collapseCombineScoresDTO = collapseCombineScoresDTO;
        this.collapseFieldType = collapseFieldType;
    }

    /**
     * Updates the DTO with shard-specific collapse information.
     *
     * @param <T> The type of the collapse field value
     * @param relevantCollapseEntries List of collapse entries relevant for the shard
     * @param collapseField Name of the field being collapsed on
     * @param updatedCollapseTopDocs Updated compound top documents for the shard
     * @param collapseShardIndex Index of the current shard being processed
     */
    public <T> void updateForShard(List<Map.Entry<T, FieldDoc>> relevantCollapseEntries, String collapseField, CompoundTopDocs updatedCollapseTopDocs, int collapseShardIndex) {
        this.relevantCollapseEntries = relevantCollapseEntries;
        this.collapseField = collapseField;
        this.updatedCollapseTopDocs = updatedCollapseTopDocs;
        this.collapseShardIndex = collapseShardIndex;
    }

    @java.lang.SuppressWarnings("all")
    public List<CompoundTopDocs> getCollapseQueryTopDocs() {
        return this.collapseQueryTopDocs;
    }

    @java.lang.SuppressWarnings("all")
    public List<QuerySearchResult> getCollapseQuerySearchResults() {
        return this.collapseQuerySearchResults;
    }

    @java.lang.SuppressWarnings("all")
    public Sort getCollapseSort() {
        return this.collapseSort;
    }

    @java.lang.SuppressWarnings("all")
    public boolean isFetchPhaseExecuted() {
        return this.isFetchPhaseExecuted;
    }

    @java.lang.SuppressWarnings("all")
    public CombineScoresDto getCollapseCombineScoresDTO() {
        return this.collapseCombineScoresDTO;
    }

    @java.lang.SuppressWarnings("all")
    public List<? extends Map.Entry<?, FieldDoc>> getRelevantCollapseEntries() {
        return this.relevantCollapseEntries;
    }

    @java.lang.SuppressWarnings("all")
    public String getCollapseField() {
        return this.collapseField;
    }

    @java.lang.SuppressWarnings("all")
    public CompoundTopDocs getUpdatedCollapseTopDocs() {
        return this.updatedCollapseTopDocs;
    }

    @java.lang.SuppressWarnings("all")
    public int getCollapseShardIndex() {
        return this.collapseShardIndex;
    }

    @java.lang.SuppressWarnings("all")
    public Class<?> getCollapseFieldType() {
        return this.collapseFieldType;
    }
}
