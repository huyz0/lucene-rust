/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * How much traffic ran native, and why the rest did not (M2 T2.4: fallback must be observable).
 * Served by {@code GET /_plugins/lucene_rust/stats}.
 */
public final class SearchStats {
    private final LongAdder nativeQueries = new LongAdder();
    private final LongAdder nativeErrors = new LongAdder();
    private final ConcurrentHashMap<String, LongAdder> fallbacks = new ConcurrentHashMap<>();

    public void nativeQuery() {
        nativeQueries.increment();
    }

    /** A native search that failed and was re-run on Lucene; also counted as a fallback. */
    public void nativeError() {
        nativeErrors.increment();
    }

    public void fallback(String reason) {
        fallbacks.computeIfAbsent(reason, r -> new LongAdder()).increment();
    }

    private final LongAdder nativeNanos = new LongAdder();
    private final LongAdder luceneNanos = new LongAdder();
    private final LongAdder luceneSearches = new LongAdder();

    /**
     * Time spent in {@code QueryPhaseSearcher.searchWith}, by the path that answered: the native
     * one, or Lucene's (a fallback, the plugin's checks included). Comparing the two per search is
     * how a request's shard-side cost is measured without the REST round trip's noise; Lucene's
     * aggregations finish in {@code postProcess}, outside it, so an aggregation's Lucene time is
     * understated.
     */
    public void nativeTime(long nanos) {
        nativeNanos.add(nanos);
    }

    public void luceneTime(long nanos) {
        luceneNanos.add(nanos);
        luceneSearches.increment();
    }

    public long nativeNanos() {
        return nativeNanos.sum();
    }

    public long luceneNanos() {
        return luceneNanos.sum();
    }

    public long luceneCount() {
        return luceneSearches.sum();
    }

    public long nativeCount() {
        return nativeQueries.sum();
    }

    public long errorCount() {
        return nativeErrors.sum();
    }

    public Map<String, Long> fallbackCounts() {
        Map<String, Long> out = new TreeMap<>();
        fallbacks.forEach((k, v) -> out.put(k, v.sum()));
        return out;
    }
}
