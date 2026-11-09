/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.query;

/**
 * Class that holds the low level information of hybrid query in the form of context
 */
public class HybridQueryContext {
    private Integer paginationDepth;

    @java.lang.SuppressWarnings("all")
    HybridQueryContext(final Integer paginationDepth) {
        this.paginationDepth = paginationDepth;
    }


    @java.lang.SuppressWarnings("all")
    public static class HybridQueryContextBuilder {
        @java.lang.SuppressWarnings("all")
        private Integer paginationDepth;

        @java.lang.SuppressWarnings("all")
        HybridQueryContextBuilder() {
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public HybridQueryContext.HybridQueryContextBuilder paginationDepth(final Integer paginationDepth) {
            this.paginationDepth = paginationDepth;
            return this;
        }

        @java.lang.SuppressWarnings("all")
        public HybridQueryContext build() {
            return new HybridQueryContext(this.paginationDepth);
        }

        @java.lang.Override
        @java.lang.SuppressWarnings("all")
        public java.lang.String toString() {
            return "HybridQueryContext.HybridQueryContextBuilder(paginationDepth=" + this.paginationDepth + ")";
        }
    }

    @java.lang.SuppressWarnings("all")
    public static HybridQueryContext.HybridQueryContextBuilder builder() {
        return new HybridQueryContext.HybridQueryContextBuilder();
    }

    @java.lang.SuppressWarnings("all")
    public Integer getPaginationDepth() {
        return this.paginationDepth;
    }
}
