/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.opensearch.index.IndexSettings;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.search.deciders.ConcurrentSearchDecision;
import org.opensearch.search.deciders.ConcurrentSearchRequestDecider;

import java.util.Optional;

/**
 * k-NN's {@code KNNConcurrentSearchRequestDecider}: in concurrent search's {@code auto} mode, a
 * request whose query is a {@code knn} query on a k-NN index searches its segments concurrently.
 * One instance per request.
 */
public final class KnnConcurrentSearchDecider extends ConcurrentSearchRequestDecider {
    private static final ConcurrentSearchDecision NO_OP = new ConcurrentSearchDecision(
        ConcurrentSearchDecision.DecisionStatus.NO_OP,
        "Default decision"
    );
    private static final ConcurrentSearchDecision YES = new ConcurrentSearchDecision(
        ConcurrentSearchDecision.DecisionStatus.YES,
        "Enable concurrent search for knn as Query has k-NN query in it and index is k-nn index"
    );

    private ConcurrentSearchDecision decision = NO_OP;

    @Override
    public void evaluateForQuery(QueryBuilder queryBuilder, IndexSettings indexSettings) {
        decision = queryBuilder instanceof KnnQueryBuilder && indexSettings.getValue(KnnSettings.KNN_INDEX) ? YES : NO_OP;
    }

    @Override
    public ConcurrentSearchDecision getConcurrentSearchDecision() {
        return decision;
    }

    /** A decider for each request on a k-NN index. */
    public static final class Factory implements ConcurrentSearchRequestDecider.Factory {
        @Override
        public Optional<ConcurrentSearchRequestDecider> create(IndexSettings indexSettings) {
            return indexSettings.getValue(KnnSettings.KNN_INDEX) ? Optional.of(new KnnConcurrentSearchDecider()) : Optional.empty();
        }
    }
}
