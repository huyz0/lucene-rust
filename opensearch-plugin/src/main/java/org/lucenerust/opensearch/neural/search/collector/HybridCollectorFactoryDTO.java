/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.search.collector;

import org.apache.lucene.search.FieldDoc;
import org.lucenerust.opensearch.neural.search.HitsThresholdChecker;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.sort.SortAndFormats;

public class HybridCollectorFactoryDTO {
    private final SortAndFormats sortAndFormats;
    private final SearchContext searchContext;
    private final HitsThresholdChecker hitsThresholdChecker;
    private final int numHits;
    private final FieldDoc after;


    @java.lang.SuppressWarnings("all")
    public static class HybridCollectorFactoryDTOBuilder {
        @java.lang.SuppressWarnings("all")
        private SortAndFormats sortAndFormats;
        @java.lang.SuppressWarnings("all")
        private SearchContext searchContext;
        @java.lang.SuppressWarnings("all")
        private HitsThresholdChecker hitsThresholdChecker;
        @java.lang.SuppressWarnings("all")
        private int numHits;
        @java.lang.SuppressWarnings("all")
        private FieldDoc after;

        @java.lang.SuppressWarnings("all")
        HybridCollectorFactoryDTOBuilder() {
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public HybridCollectorFactoryDTO.HybridCollectorFactoryDTOBuilder sortAndFormats(final SortAndFormats sortAndFormats) {
            this.sortAndFormats = sortAndFormats;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public HybridCollectorFactoryDTO.HybridCollectorFactoryDTOBuilder searchContext(final SearchContext searchContext) {
            this.searchContext = searchContext;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public HybridCollectorFactoryDTO.HybridCollectorFactoryDTOBuilder hitsThresholdChecker(final HitsThresholdChecker hitsThresholdChecker) {
            this.hitsThresholdChecker = hitsThresholdChecker;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public HybridCollectorFactoryDTO.HybridCollectorFactoryDTOBuilder numHits(final int numHits) {
            this.numHits = numHits;
            return this;
        }

        /**
         * @return {@code this}.
         */
        @java.lang.SuppressWarnings("all")
        public HybridCollectorFactoryDTO.HybridCollectorFactoryDTOBuilder after(final FieldDoc after) {
            this.after = after;
            return this;
        }

        @java.lang.SuppressWarnings("all")
        public HybridCollectorFactoryDTO build() {
            return new HybridCollectorFactoryDTO(this.sortAndFormats, this.searchContext, this.hitsThresholdChecker, this.numHits, this.after);
        }

        @java.lang.Override
        @java.lang.SuppressWarnings("all")
        public java.lang.String toString() {
            return "HybridCollectorFactoryDTO.HybridCollectorFactoryDTOBuilder(sortAndFormats=" + this.sortAndFormats + ", searchContext=" + this.searchContext + ", hitsThresholdChecker=" + this.hitsThresholdChecker + ", numHits=" + this.numHits + ", after=" + this.after + ")";
        }
    }

    @java.lang.SuppressWarnings("all")
    public static HybridCollectorFactoryDTO.HybridCollectorFactoryDTOBuilder builder() {
        return new HybridCollectorFactoryDTO.HybridCollectorFactoryDTOBuilder();
    }

    @java.lang.SuppressWarnings("all")
    public HybridCollectorFactoryDTO(final SortAndFormats sortAndFormats, final SearchContext searchContext, final HitsThresholdChecker hitsThresholdChecker, final int numHits, final FieldDoc after) {
        this.sortAndFormats = sortAndFormats;
        this.searchContext = searchContext;
        this.hitsThresholdChecker = hitsThresholdChecker;
        this.numHits = numHits;
        this.after = after;
    }

    @java.lang.SuppressWarnings("all")
    public SortAndFormats getSortAndFormats() {
        return this.sortAndFormats;
    }

    @java.lang.SuppressWarnings("all")
    public SearchContext getSearchContext() {
        return this.searchContext;
    }

    @java.lang.SuppressWarnings("all")
    public HitsThresholdChecker getHitsThresholdChecker() {
        return this.hitsThresholdChecker;
    }

    @java.lang.SuppressWarnings("all")
    public int getNumHits() {
        return this.numHits;
    }

    @java.lang.SuppressWarnings("all")
    public FieldDoc getAfter() {
        return this.after;
    }
}
