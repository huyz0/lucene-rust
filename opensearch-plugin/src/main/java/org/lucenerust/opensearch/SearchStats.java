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
