/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * How much traffic ran native, and why the rest did not, so that every fallback is observable.
 * Served by {@code GET /_plugins/lucene_rust/stats}.
 */
public final class SearchStats {
    private final LongAdder nativeQueries = new LongAdder();
    private final LongAdder nativeErrors = new LongAdder();
    private final LongAdder nativeFetches = new LongAdder();
    private final ConcurrentHashMap<String, LongAdder> fallbacks = new ConcurrentHashMap<>();

    public void nativeQuery() {
        nativeQueries.increment();
    }

    private final LongAdder liveQueries = new LongAdder();

    /** A native search that read the writer's buffered documents too; also counted as native. */
    public void liveQuery() {
        liveQueries.increment();
    }

    public long liveCount() {
        return liveQueries.sum();
    }

    /** A native search that failed and was re-run on Lucene; also counted as a fallback. */
    public void nativeError() {
        nativeErrors.increment();
    }

    private final LongAdder scriptErrors = new LongAdder();

    /**
     * A native search that a Painless script's own exception ended, failing as the script does in
     * Java and never re-run; {@code counted} when it was first counted as a {@link #nativeError}.
     */
    public void scriptError(boolean counted) {
        if (counted) {
            nativeErrors.decrement();
        }
        scriptErrors.increment();
    }

    public long scriptErrorCount() {
        return scriptErrors.sum();
    }

    private final LongAdder nativeSequentialFetches = new LongAdder();

    /**
     * A document whose stored fields the native reader served; {@code sequential}
     * when through the fetch phase's sequential reader (a run of adjacent hits).
     */
    public void nativeFetch(boolean sequential) {
        nativeFetches.increment();
        if (sequential) {
            nativeSequentialFetches.increment();
        }
    }

    private final LongAdder sourcesWithoutVectors = new LongAdder();

    /** A derived {@code _source} read without its vectors, for a fetch whose filter drops them. */
    public void sourceWithoutVectors() {
        sourcesWithoutVectors.increment();
    }

    public long sourcesWithoutVectorsCount() {
        return sourcesWithoutVectors.sum();
    }

    private final LongAdder nativeDocValues = new LongAdder();

    /** A document's doc values read natively ({@link NativeDocValues}). */
    public void nativeDocValues() {
        nativeDocValues.increment();
    }

    public long docValuesCount() {
        return nativeDocValues.sum();
    }

    private final LongAdder nativeHighlights = new LongAdder();

    /** A field of a search highlighted natively ({@code NativeUnifiedHighlighter}), counted once per search. */
    public void nativeHighlight() {
        nativeHighlights.increment();
    }

    public long highlightCount() {
        return nativeHighlights.sum();
    }

    public long sequentialFetchCount() {
        return nativeSequentialFetches.sum();
    }

    public long fetchCount() {
        return nativeFetches.sum();
    }

    private final LongAdder fetchNativeNanos = new LongAdder();
    private final LongAdder fetchLuceneNanos = new LongAdder();
    private final LongAdder fetchLuceneDocs = new LongAdder();

    /**
     * Time spent in {@code StoredFields.document} by the fetch phase and get API, by the path that
     * read the document: per document, without the REST round trip, as {@link #nativeTime} is for
     * the query phase.
     */
    public void fetchTime(boolean nativePath, long nanos) {
        if (nativePath) {
            fetchNativeNanos.add(nanos);
        } else {
            fetchLuceneNanos.add(nanos);
            fetchLuceneDocs.increment();
        }
    }

    public long fetchNativeNanos() {
        return fetchNativeNanos.sum();
    }

    public long fetchLuceneNanos() {
        return fetchLuceneNanos.sum();
    }

    public long fetchLuceneCount() {
        return fetchLuceneDocs.sum();
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

    private final java.util.concurrent.atomic.LongAccumulator requestPeak = new java.util.concurrent.atomic.LongAccumulator(Math::max, 0);
    private final LongAdder requestBytes = new LongAdder();
    private final LongAdder requestScopes = new LongAdder();

    /** The most a native query phase held ({@link NativeMemory.Scope#peak}). */
    public void nativeMemory(long peak) {
        requestPeak.accumulate(peak);
        requestBytes.add(Math.max(0, peak));
        requestScopes.increment();
    }

    /** {most one request held, sum of requests' peaks, requests measured}. */
    public long[] requestMemory() {
        return new long[] { requestPeak.get(), requestBytes.sum(), requestScopes.sum() };
    }

    public void luceneTime(long nanos) {
        luceneNanos.add(nanos);
        luceneSearches.increment();
    }

    public long nativeNanos() {
        return nativeNanos.sum();
    }

    private final LongAdder liveReaderNanos = new LongAdder();
    private final LongAdder liveReaders = new LongAdder();

    /** Time spent building a realtime view's reader (its buffers written as segments). */
    public void liveReaderTime(long nanos) {
        liveReaderNanos.add(nanos);
        liveReaders.increment();
    }

    public long liveReaderNanos() {
        return liveReaderNanos.sum();
    }

    public long liveReaderCount() {
        return liveReaders.sum();
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
