/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.lucenerust.opensearch.neural.search.query;

import org.apache.lucene.search.SortField;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.internal.SearchContext;
import org.opensearch.search.rescore.RescoreContext;
import org.opensearch.search.sort.SortAndFormats;
import java.util.List;

/**
 * Class that holds all parameters required for processing collector results in HybridCollectorResultsUtil
 */
public class HybridCollectorResultsUtilParams {
    private final SearchContext searchContext;
    private final TopDocsMerger topDocsMerger;
    private final int trackTotalHitsUpTo;
    private final boolean isSortEnabled;
    private final boolean isCollapseEnabled;
    private SortField[] sortFields = null;
    private final List<RescoreContext> rescoreContexts;
    private final boolean rescoredNatively;
    private SortAndFormats sortAndFormats = null;
    private DocValueFormat[] docValueFormats = null;
    private static final SortField[] DEFAULT_SORT_FIELDS = new SortField[] {new SortField(null, SortField.Type.SCORE)};

    private HybridCollectorResultsUtilParams(Builder builder) {
        this.searchContext = builder.searchContext;
        this.topDocsMerger = new TopDocsMerger(searchContext.sort(), searchContext.collapse());
        this.trackTotalHitsUpTo = searchContext.trackTotalHitsUpTo();
        this.isSortEnabled = searchContext.sort() != null;
        if (isSortEnabled) {
            setSortEnabledParameters();
        }
        this.isCollapseEnabled = searchContext.collapse() != null;
        if (isCollapseEnabled && isSortEnabled == false) {
            this.docValueFormats = new DocValueFormat[] {DocValueFormat.RAW};
        }
        if ((isSortEnabled || isCollapseEnabled) && sortFields == null) {
            this.sortFields = DEFAULT_SORT_FIELDS;
        }
        this.rescoreContexts = searchContext.rescore();
        this.rescoredNatively = builder.rescoredNatively;
    }

    private void setSortEnabledParameters() {
        this.sortAndFormats = searchContext.sort();
        this.sortFields = sortAndFormats.sort.getSort();
        this.docValueFormats = sortAndFormats.formats;
    }


    /**
     * Builder class
     */
    public static class Builder {
        private SearchContext searchContext;
        private boolean rescoredNatively;

        /** The sub-queries' hits were rescored before they reached the result util. */
        public Builder rescoredNatively(boolean rescoredNatively) {
            this.rescoredNatively = rescoredNatively;
            return this;
        }

        public Builder searchContext(SearchContext searchContext) {
            this.searchContext = searchContext;
            return this;
        }

        public HybridCollectorResultsUtilParams build() {
            return new HybridCollectorResultsUtilParams(this);
        }
    }

    @java.lang.SuppressWarnings("all")
    public SearchContext getSearchContext() {
        return this.searchContext;
    }

    @java.lang.SuppressWarnings("all")
    public TopDocsMerger getTopDocsMerger() {
        return this.topDocsMerger;
    }

    @java.lang.SuppressWarnings("all")
    public int getTrackTotalHitsUpTo() {
        return this.trackTotalHitsUpTo;
    }

    @java.lang.SuppressWarnings("all")
    public boolean isSortEnabled() {
        return this.isSortEnabled;
    }

    @java.lang.SuppressWarnings("all")
    public boolean isCollapseEnabled() {
        return this.isCollapseEnabled;
    }

    @java.lang.SuppressWarnings("all")
    public SortField[] getSortFields() {
        return this.sortFields;
    }

    @java.lang.SuppressWarnings("all")
    public List<RescoreContext> getRescoreContexts() {
        return this.rescoreContexts;
    }

    /** Whether the hits arrive rescored already, so the result util must not rescore them. */
    public boolean isRescoredNatively() {
        return this.rescoredNatively;
    }

    @java.lang.SuppressWarnings("all")
    public SortAndFormats getSortAndFormats() {
        return this.sortAndFormats;
    }

    @java.lang.SuppressWarnings("all")
    public DocValueFormat[] getDocValueFormats() {
        return this.docValueFormats;
    }
}
