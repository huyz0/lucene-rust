/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.processor.dto;

import org.lucenerust.opensearch.neural.processor.CompoundTopDocs;
import org.lucenerust.opensearch.neural.processor.explain.ExplainableTechnique;
import java.util.List;

/**
 * DTO object to hold data required for explain.
 */
public class ExplainDTO {
    
    private List<CompoundTopDocs> queryTopDocs;
    
    private ExplainableTechnique explainableTechnique;
    private boolean singleShard;


    @java.lang.SuppressWarnings("all")
    public static class ExplainDTOBuilder {
        @java.lang.SuppressWarnings("all")
        private List<CompoundTopDocs> queryTopDocs;
        @java.lang.SuppressWarnings("all")
        private ExplainableTechnique explainableTechnique;
        @java.lang.SuppressWarnings("all")
        private boolean singleShard;

        @java.lang.SuppressWarnings("all")
        ExplainDTOBuilder() {
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public ExplainDTO.ExplainDTOBuilder queryTopDocs(final List<CompoundTopDocs> queryTopDocs) {
            if (queryTopDocs == null) {
                throw new java.lang.NullPointerException("queryTopDocs is marked non-null but is null");
            }
            this.queryTopDocs = queryTopDocs;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public ExplainDTO.ExplainDTOBuilder explainableTechnique(final ExplainableTechnique explainableTechnique) {
            if (explainableTechnique == null) {
                throw new java.lang.NullPointerException("explainableTechnique is marked non-null but is null");
            }
            this.explainableTechnique = explainableTechnique;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public ExplainDTO.ExplainDTOBuilder singleShard(final boolean singleShard) {
            this.singleShard = singleShard;
            return this;
        }

        @java.lang.SuppressWarnings("all")
        public ExplainDTO build() {
            return new ExplainDTO(this.queryTopDocs, this.explainableTechnique, this.singleShard);
        }

        @java.lang.Override
        @java.lang.SuppressWarnings("all")
        public java.lang.String toString() {
            return "ExplainDTO.ExplainDTOBuilder(queryTopDocs=" + this.queryTopDocs + ", explainableTechnique=" + this.explainableTechnique + ", singleShard=" + this.singleShard + ")";
        }
    }

    @java.lang.SuppressWarnings("all")
    public static ExplainDTO.ExplainDTOBuilder builder() {
        return new ExplainDTO.ExplainDTOBuilder();
    }

    @java.lang.SuppressWarnings("all")
    public ExplainDTO(final List<CompoundTopDocs> queryTopDocs, final ExplainableTechnique explainableTechnique, final boolean singleShard) {
        if (queryTopDocs == null) {
            throw new java.lang.NullPointerException("queryTopDocs is marked non-null but is null");
        }
        if (explainableTechnique == null) {
            throw new java.lang.NullPointerException("explainableTechnique is marked non-null but is null");
        }
        this.queryTopDocs = queryTopDocs;
        this.explainableTechnique = explainableTechnique;
        this.singleShard = singleShard;
    }

    
    @java.lang.SuppressWarnings("all")
    public List<CompoundTopDocs> getQueryTopDocs() {
        return this.queryTopDocs;
    }

    
    @java.lang.SuppressWarnings("all")
    public ExplainableTechnique getExplainableTechnique() {
        return this.explainableTechnique;
    }

    @java.lang.SuppressWarnings("all")
    public boolean isSingleShard() {
        return this.singleShard;
    }
}
