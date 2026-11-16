/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.lucenerust.opensearch.NativeBridge.*;
import static org.lucenerust.opensearch.NativeCall.*;
import static org.lucenerust.opensearch.NativeStore.*;
import static org.lucenerust.opensearch.NativeAnalysis.*;
import static org.lucenerust.opensearch.NativeMl.*;
import static org.lucenerust.opensearch.NativeSparse.*;
import static org.lucenerust.opensearch.NativeZstd.*;
import static org.lucenerust.opensearch.NativeLimits.*;
import static org.lucenerust.opensearch.NativeWriter.*;
import static org.lucenerust.opensearch.NativeLive.*;

/** The reader, search, aggregation and fetch downcalls; see {@link NativeBridge}. */
public final class NativeSearch {
    private NativeSearch() {}

    private static final class H {
        static final MethodHandle OPEN_READER = bind(
            "ffi_open_jvm_reader",
            FunctionDescriptor.of(
                JAVA_INT,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle OPEN_READER_VIEW = bind(
            "ffi_open_jvm_reader_view",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle SEARCH = bind(
            "ffi_jvm_reader_search",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle READER_KNN = bind(
            "ffi_jvm_reader_knn",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle LIVE_VIEW_KNN = bind(
            "ffi_live_view_knn",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle DOC_FREQ = bind(
            "ffi_jvm_reader_doc_freq",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle SEARCH_SORTED = bind(
            "ffi_jvm_reader_search_sorted_alloc",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle AGGREGATE = bind(
            "ffi_jvm_reader_aggregate_alloc",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle AGGREGATE_TREE = bind(
            "ffi_jvm_reader_aggregate_tree_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle AGGREGATE_TREE_CACHED = bind(
            "ffi_jvm_reader_aggregate_tree_cached_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle SEARCH_HYBRID = bind(
            "ffi_jvm_reader_search_hybrid_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle PROFILE = bind(
            "ffi_jvm_reader_profile_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle SEARCH_COLLAPSED = bind(
            "ffi_jvm_reader_search_collapsed_alloc",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_LONG,
                ADDRESS,
                ADDRESS
            )
        );
        static final MethodHandle RESCORE = bind(
            "ffi_jvm_reader_rescore",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                JAVA_INT,
                JAVA_FLOAT,
                JAVA_FLOAT,
                JAVA_INT,
                ADDRESS,
                ADDRESS,
                JAVA_LONG
            )
        );
        static final MethodHandle DOCUMENT = bind(
            "ffi_jvm_reader_document_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle DOC_VALUES = bind(
            "ffi_jvm_reader_doc_values_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle DERIVED_SOURCE = bind(
            "ffi_jvm_reader_derived_source_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle PREFETCH_DOCUMENTS = bind(
            "ffi_jvm_reader_prefetch_documents",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG)
        );
        static final MethodHandle COUNT_TERMINATES = bind(
            "ffi_jvm_reader_count_terminates",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle CLOSE_READER = bind("ffi_close_jvm_reader", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
    }

    /**
     * Opens a reader over the segments listed in {@code segmentInfos} (bytes written by {@code
     * SegmentInfos.write(IndexOutput)} at {@code generation}), checking that segment {@code i} has
     * {@code maxDocs[i]} documents and masking it with {@code liveDocs[i]} ({@code
     * FixedBitSet.getBits()} words, exactly {@code ceil(maxDoc / 64)} of them; null for no
     * deletions; a null array for none anywhere). {@code previous} is an open handle whose unchanged
     * segments are reused, or 0. The new handle is written to {@code outHandle[0]}; it is immutable,
     * so searches on it never wait on each other or on opens and closes.
     */
    public static int openReader(
        byte[] indexPathUtf8,
        byte[] segmentInfos,
        long generation,
        long previous,
        int[] maxDocs,
        long[][] liveDocs,
        long[] outHandle
    ) {
        if (indexPathUtf8 == null) {
            return nullArgument("path");
        }
        if (segmentInfos == null) {
            return nullArgument("infos");
        }
        if (maxDocs == null) {
            return nullArgument("maxDocs");
        }
        int segments = maxDocs.length;
        if (liveDocs != null && liveDocs.length != segments) {
            return fail(INVALID_ARGUMENT, "liveDocs has " + liveDocs.length + " entries for " + segments + " segments");
        }
        try (Scratch s = Scratch.open()) {
            // `liveDocs[i]`'s words, one after another, and how many each segment has.
            long[] counts = new long[segments];
            long words = 0;
            if (liveDocs != null) {
                for (int i = 0; i < segments; i++) {
                    if (liveDocs[i] != null) {
                        counts[i] = liveDocs[i].length;
                        words += liveDocs[i].length;
                    }
                }
            }
            MemorySegment wordsSeg = s.room(JAVA_LONG, words);
            long at = 0;
            if (liveDocs != null) {
                for (long[] w : liveDocs) {
                    if (w != null) {
                        MemorySegment.copy(w, 0, wordsSeg, JAVA_LONG, at * Long.BYTES, w.length);
                        at += w.length;
                    }
                }
            }
            MemorySegment out = s.alloc(Long.BYTES);
            int rc = (int) H.OPEN_READER.invokeExact(
                s.bytes(indexPathUtf8),
                (long) indexPathUtf8.length,
                s.bytes(segmentInfos),
                (long) segmentInfos.length,
                generation,
                previous,
                s.ints(maxDocs),
                (long) segments,
                wordsSeg,
                s.longs(counts),
                out
            );
            if (rc == OK) {
                long handle = out.get(JAVA_LONG, 0);
                if (outHandle == null || outHandle.length < 1) {
                    // The caller will never see this handle, so nothing would close it.
                    closeReader(handle);
                    return fail(INVALID_ARGUMENT, "outHandle holds no slot");
                }
                outHandle[0] = handle;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@code IndexReader.docFreq(new Term(field, term))} over the reader's segments, into
     * {@code out[0]}: the total-hits shortcut of a term query, counted by the native term
     * dictionaries the search reads next.
     */
    public static int docFreq(long handle, byte[] field, byte[] term, long[] out) {
        if (field == null) {
            return nullArgument("field");
        }
        if (term == null) {
            return nullArgument("term");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment n = s.alloc(Long.BYTES);
            int rc = docFreq(s, handle, field, term, n);
            if (rc == OK) {
                out[0] = n.get(JAVA_LONG, 0);
            }
            return rc;
        }
    }

    private static int docFreq(Scratch s, long handle, byte[] field, byte[] term, MemorySegment out) {
        try {
            return (int) H.DOC_FREQ.invokeExact(handle, s.bytes(field), (long) field.length, s.bytes(term), (long) term.length, out);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob. {@code outCounts[0]} receives the number of hits written; {@code
     * outCounts[1]} the total hits, exact below {@code countLimit} and otherwise a lower bound of at
     * least {@code countLimit}, with {@code outCounts[2]} 1 in that case -- Lucene's {@code
     * totalHitsThreshold}. A {@code countLimit} of 0 counts nothing ({@code outCounts[1]} is -1);
     * {@link Long#MAX_VALUE} counts exactly.
     */
    public static int search(
        long handle,
        byte[] query,
        int topN,
        long countLimit,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    ) {
        int refused = checkSearch(query, topN, outDocs, outScores, outCounts, 3);
        if (refused != OK) {
            return refused;
        }
        try (Scratch s = Scratch.open()) {
            return searchInto(s, handle, query, topN, countLimit, outDocs, outScores, outCounts);
        }
    }

    /** {@link #search}'s argument checks, in the order the native code made them. */
    static int checkSearch(byte[] query, int topN, int[] outDocs, float[] outScores, long[] outCounts, int countSlots) {
        if (topN < 0) {
            return fail(INVALID_ARGUMENT, "topN " + topN + " is negative");
        }
        if (query == null) {
            return nullArgument("query");
        }
        if (topN > 0) {
            if (outDocs == null) {
                return nullArgument("outDocs");
            }
            if (outScores == null) {
                return nullArgument("outScores");
            }
            int room = Math.min(outDocs.length, outScores.length);
            if (room < topN) {
                return fail(BUFFER_TOO_SMALL, "output arrays hold " + room + " hits, topN is " + topN);
            }
        }
        if (outCounts == null || outCounts.length < countSlots) {
            return fail(INVALID_ARGUMENT, "outCounts holds fewer than " + countSlots + " slots");
        }
        return OK;
    }

    private static int searchInto(
        Scratch s,
        long handle,
        byte[] query,
        int topN,
        long countLimit,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    ) {
        MemorySegment docs = s.room(JAVA_INT, topN);
        MemorySegment scores = s.room(JAVA_FLOAT, topN);
        MemorySegment counts = s.alloc(3 * Long.BYTES);
        int rc;
        try {
            rc = (int) H.SEARCH.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                (long) topN,
                countLimit,
                docs,
                scores,
                (long) topN,
                counts,
                counts.asSlice(Long.BYTES),
                counts.asSlice(2 * Long.BYTES)
            );
        } catch (Throwable t) {
            throw rethrow(t);
        }
        if (rc != OK) {
            return rc;
        }
        int n = Math.toIntExact(counts.get(JAVA_LONG, 0));
        MemorySegment.copy(docs, JAVA_INT, 0, outDocs, 0, n);
        MemorySegment.copy(scores, JAVA_FLOAT, 0, outScores, 0, n);
        outCounts[0] = n;
        outCounts[1] = counts.get(JAVA_LONG, Long.BYTES);
        outCounts[2] = counts.get(JAVA_BOOLEAN, 2 * Long.BYTES) ? 1 : 0;
        return OK;
    }

    /**
     * The reader's top hits for a KNN request ({@code knn.rs}): global doc ids into {@code
     * outDocs}, scores into {@code outScores}, best first, at most {@code room} of them; returns the
     * status, the hit count in {@code outCount[0]}. An expanded nested search can find more than
     * {@code room}: the status is then {@link #BUFFER_TOO_SMALL} and {@code outCount[0]} the room it
     * needs.
     */
    public static int readerKnn(long handle, byte[] request, int room, int[] outDocs, float[] outScores, int[] outCount) {
        return knn(handle, 0, false, request, room, outDocs, outScores, outCount);
    }

    /**
     * {@link #readerKnn} over the reader at {@code handle} and the writer's buffers of {@code view}
     * ({@code ffi_live_view_knn}): doc ids numbered as {@link #liveViewReader}'s handles number them.
     */
    public static int liveViewKnn(long handle, long view, byte[] request, int room, int[] outDocs, float[] outScores, int[] outCount) {
        return knn(handle, view, true, request, room, outDocs, outScores, outCount);
    }

    private static int knn(
        long handle,
        long view,
        boolean live,
        byte[] request,
        int room,
        int[] outDocs,
        float[] outScores,
        int[] outCount
    ) {
        if (request == null) {
            return nullArgument("request");
        }
        if (room <= 0 || outDocs == null || outScores == null || Math.min(outDocs.length, outScores.length) < room) {
            return fail(BUFFER_TOO_SMALL, "output arrays hold fewer than " + room + " hits");
        }
        if (outCount == null || outCount.length < 1) {
            return fail(INVALID_ARGUMENT, "outCount holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment docs = s.room(JAVA_INT, room);
            MemorySegment scores = s.room(JAVA_FLOAT, room);
            MemorySegment count = s.alloc(Long.BYTES);
            count.set(JAVA_LONG, 0, 0L);
            int rc;
            try {
                rc = live
                    ? (int) H.LIVE_VIEW_KNN.invokeExact(handle, view, s.bytes(request), (long) request.length, docs, scores, (long) room, count)
                    : (int) H.READER_KNN.invokeExact(handle, s.bytes(request), (long) request.length, docs, scores, (long) room, count);
            } catch (Throwable t) {
                throw rethrow(t);
            }
            long n = count.get(JAVA_LONG, 0);
            if (rc == BUFFER_TOO_SMALL) {
                outCount[0] = n > room && n <= Integer.MAX_VALUE - 8 ? (int) n : 0;
                return rc;
            }
            if (rc != OK) {
                return rc;
            }
            int hits = Math.toIntExact(n);
            MemorySegment.copy(docs, JAVA_INT, 0, outDocs, 0, hits);
            MemorySegment.copy(scores, JAVA_FLOAT, 0, outScores, 0, hits);
            outCount[0] = hits;
            return OK;
        }
    }

    /**
     * {@link #search} and {@link #docFreq} of {@code field:term} together -- a term query's
     * total-hits shortcut and its hits: {@code outCounts} as for {@link #search} (all 0 when {@code
     * topN} is 0, which searches nothing), with a fourth slot receiving the document frequency.
     */
    public static int searchDocFreq(
        long handle,
        byte[] query,
        int topN,
        long countLimit,
        byte[] field,
        byte[] term,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    ) {
        if (field == null) {
            return nullArgument("field");
        }
        if (term == null) {
            return nullArgument("term");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment df = s.alloc(Long.BYTES);
            int rc = docFreq(s, handle, field, term, df);
            if (rc != OK) {
                return rc;
            }
            if (topN == 0) {
                if (outCounts == null || outCounts.length < 4) {
                    return fail(INVALID_ARGUMENT, "outCounts holds fewer than 4 slots");
                }
                outCounts[0] = 0;
                outCounts[1] = 0;
                outCounts[2] = 0;
            } else {
                int refused = checkSearch(query, topN, outDocs, outScores, outCounts, 4);
                if (refused != OK) {
                    return refused;
                }
                rc = searchInto(s, handle, query, topN, countLimit, outDocs, outScores, outCounts);
                if (rc != OK) {
                    return rc;
                }
            }
            outCounts[3] = df.get(JAVA_LONG, 0);
            return OK;
        }
    }

    /**
     * Whether a concurrent {@code size: 0} search's count stops early: {@code spec} is the limit
     * ({@code int}), each segment's iterate flag ({@code int} count, one byte each: 1 where Lucene's
     * {@code Weight.count} gave -1) and the slices ({@link NativeAggregations#writeSlices});
     * {@code out[0]} receives 1 or 0.
     */
    public static int countTerminates(long handle, byte[] query, byte[] spec, long[] out) {
        if (query == null) {
            return nullArgument("query");
        }
        if (spec == null) {
            return nullArgument("spec");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment terminated = s.alloc(1);
            int rc = (int) H.COUNT_TERMINATES.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(spec),
                (long) spec.length,
                terminated
            );
            if (rc == OK) {
                out[0] = terminated.get(JAVA_BOOLEAN, 0) ? 1 : 0;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob sorted by a sort blob ({@link SortEncoder}): {@code outDocs} receives the
     * hits' global doc ids, best first, and {@code outValues} their sort values, one {@code long}
     * per key per hit in key order ({@link SortEncoder#value} turns them back). A keyword key's
     * terms come back in a new array stored in {@code outTerms[0]} (see {@link SortEncoder#hits});
     * {@code outCounts} as for {@link #search}, a fourth slot holding the tracked max score's
     * float bits ({@code NaN} untracked) and a fifth 1 when the sort blob's {@code terminate_after}
     * ended the search early; {@code topN} must be at least 1.
     */
    public static int searchSorted(
        long handle,
        byte[] query,
        byte[] sort,
        int topN,
        long countLimit,
        int[] outDocs,
        long[] outValues,
        long[] outCounts,
        byte[][] outTerms
    ) {
        if (topN < 0) {
            return fail(INVALID_ARGUMENT, "topN " + topN + " is negative");
        }
        if (query == null) {
            return nullArgument("query");
        }
        if (sort == null) {
            return nullArgument("sort");
        }
        if (outDocs == null) {
            return nullArgument("outDocs");
        }
        if (outValues == null) {
            return nullArgument("outValues");
        }
        int keys = sort.length == 0 ? 0 : sort[0] & 0xff;
        long wantValues = (long) topN * keys;
        if (outDocs.length < topN || outValues.length < wantValues) {
            return fail(
                BUFFER_TOO_SMALL,
                "output arrays hold " + outDocs.length + " hits and " + outValues.length + " values, topN is " + topN + " with " + keys + " keys"
            );
        }
        if (topN == 0) {
            return fail(INVALID_ARGUMENT, "searchSorted: topN must be at least 1");
        }
        if (outCounts == null || outCounts.length < 5) {
            return fail(INVALID_ARGUMENT, "outCounts holds fewer than 5 slots");
        }
        if (outTerms == null || outTerms.length < 1) {
            return fail(INVALID_ARGUMENT, "outTerms holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment docs = s.room(JAVA_INT, topN);
            MemorySegment values = s.room(JAVA_LONG, wantValues);
            MemorySegment counts = s.alloc(6 * Long.BYTES);
            MemorySegment terms = ownedSlot(s);
            int rc = (int) H.SEARCH_SORTED.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(sort),
                (long) sort.length,
                (long) topN,
                countLimit,
                docs,
                (long) topN,
                values,
                wantValues,
                counts,
                ptrOf(terms),
                lenOf(terms)
            );
            if (rc != OK) {
                return rc;
            }
            int n = Math.toIntExact(counts.get(JAVA_LONG, 0));
            MemorySegment.copy(docs, JAVA_INT, 0, outDocs, 0, n);
            MemorySegment.copy(values, JAVA_LONG, 0, outValues, 0, n * keys);
            MemorySegment.copy(counts, JAVA_LONG, 0, outCounts, 0, 5);
            byte[] termBytes = take(terms);
            if (termBytes == null) {
                return tooLarge("the sort terms");
            }
            if (counts.get(JAVA_LONG, 5 * Long.BYTES) != 0) {
                outTerms[0] = termBytes;
            }
            return OK;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob's matches through the numeric metrics of a metrics blob ({@link
     * NativeAggregations.Plan#blob}): per slice (one when the blob names none), per field, {@code
     * outCounts} receives its value count and {@code outValues} {@link NativeAggregations#VALUES}
     * doubles -- the compensated sum and its delta, the minimum and maximum over every value, and
     * over each document's first and last; and in {@code outTerms[0]} the {@code terms} results:
     * per slice, per terms aggregation, the other-doc count ({@code long}), the bucket count ({@code
     * int}) and per bucket, by term, its doc count ({@code long}) and term ({@code int} length,
     * bytes), little-endian. With a positive {@code countLimit}, {@code outTotal} receives the {@code
     * size: 0} search's total and whether it is a lower bound, as {@link #search} counts them, from
     * the matches the aggregations visited -- left alone when some segment's matches were not
     * visited behind a {@code min_score} (its aggregations answered from points). When counted,
     * every segment's matches were visited and {@code outSegCounts} (may be null) holds a slot per
     * segment, those get each segment's live match count -- what the count collector iterates
     * there -- and {@code outTotal[2]} (when {@code outTotal} has a third slot) becomes 1, else 0.
     */
    public static int aggregate(
        long handle,
        byte[] query,
        byte[] aggs,
        long[] outCounts,
        double[] outValues,
        byte[][] outTerms,
        long countLimit,
        long[] outTotal,
        long[] outSegCounts
    ) {
        if (query == null) {
            return nullArgument("query");
        }
        if (aggs == null) {
            return nullArgument("aggs");
        }
        if (outCounts == null) {
            return nullArgument("outCounts");
        }
        if (outValues == null) {
            return nullArgument("outValues");
        }
        if (outTerms == null || outTerms.length < 1) {
            return fail(INVALID_ARGUMENT, "outTerms holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            // The output arrays' current contents go in and all of them come back out, so the
            // slots past the fields the blob names keep what they held.
            MemorySegment counts = s.longs(outCounts);
            MemorySegment values = s.room(JAVA_DOUBLE, outValues.length);
            if (outValues.length > 0) {
                MemorySegment.copy(outValues, 0, values, JAVA_DOUBLE, 0, outValues.length);
            }
            MemorySegment total = s.alloc(4 * Long.BYTES);
            total.set(JAVA_LONG, 0, 0);
            int segCap = outSegCounts == null ? 0 : outSegCounts.length;
            MemorySegment segCounts = s.room(JAVA_LONG, segCap);
            MemorySegment terms = ownedSlot(s);
            int rc = (int) H.AGGREGATE.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(aggs),
                (long) aggs.length,
                countLimit,
                counts,
                (long) outCounts.length,
                values,
                (long) outValues.length,
                total,
                segCounts,
                (long) segCap,
                ptrOf(terms),
                lenOf(terms)
            );
            // Taken (and so freed) first: every return below would otherwise leak it.
            byte[] termBytes = take(terms);
            if (total.get(JAVA_LONG, 0) != 0) {
                if (outTotal == null || outTotal.length < 2) {
                    return fail(INVALID_ARGUMENT, "outTotal holds fewer than 2 slots");
                }
                outTotal[0] = total.get(JAVA_LONG, Long.BYTES);
                outTotal[1] = total.get(JAVA_LONG, 2 * Long.BYTES);
                boolean perSegment = total.get(JAVA_LONG, 3 * Long.BYTES) != 0;
                if (outTotal.length > 2) {
                    outTotal[2] = perSegment ? 1 : 0;
                }
                if (perSegment) {
                    MemorySegment.copy(segCounts, JAVA_LONG, 0, outSegCounts, 0, segCap);
                }
            }
            if (rc != OK) {
                return rc;
            }
            MemorySegment.copy(counts, JAVA_LONG, 0, outCounts, 0, outCounts.length);
            MemorySegment.copy(values, JAVA_DOUBLE, 0, outValues, 0, outValues.length);
            if (termBytes == null) {
                return tooLarge("the terms aggregations' results");
            }
            outTerms[0] = termBytes;
            return OK;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob's matches through an aggregation tree blob ({@link
     * NativeAggregationTree.Tree#blob}): the encoded shard results in {@code out[0]} (the layout
     * {@code agg_tree.rs} documents), which {@link NativeAggregationTree.Tree#build} reads.
     */
    public static int aggregateTree(long handle, byte[] query, byte[] tree, byte[][] out) {
        return aggregateTree(handle, query, tree, false, out);
    }

    /**
     * {@link #aggregateTree(long, byte[], byte[], byte[][])}; with {@code cached}, each committed
     * segment's collected part is read from and kept in the node's aggregation segment cache
     * ({@link #aggCacheSetBudget}): the same results.
     */
    public static int aggregateTree(long handle, byte[] query, byte[] tree, boolean cached, byte[][] out) {
        if (query == null) {
            return nullArgument("query");
        }
        if (tree == null) {
            return nullArgument("tree");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            MethodHandle call = cached ? H.AGGREGATE_TREE_CACHED : H.AGGREGATE_TREE;
            int rc = (int) call.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(tree),
                (long) tree.length,
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the aggregation results");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a hybrid query's shard phase ({@link NativeHybrid}): the encoded slices' answers
     * ({@code search_hybrid_blobs} in {@code reader_search.rs}) in {@code out[0]}.
     */
    public static int searchHybrid(long handle, byte[] blob, byte[][] out) {
        if (blob == null) {
            return nullArgument("blob");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.SEARCH_HYBRID.invokeExact(handle, s.bytes(blob), (long) blob.length, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the hybrid results");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a request's query blob once more with a timer around every clause of its scorer tree
     * ({@link NativeProfile}): the timers ({@code profile_blob} in {@code profile.rs} of the
     * engine crate) in {@code out[0]}. {@code topScores} runs it into the top-hits collector
     * ({@code topN}, counting under {@code countLimit}), otherwise into a plain count.
     */
    public static int profile(long handle, byte[] blob, boolean topScores, long topN, long countLimit, byte[][] out) {
        if (blob == null) {
            return nullArgument("blob");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.PROFILE.invokeExact(
                handle,
                s.bytes(blob),
                (long) blob.length,
                topScores ? 1 : 0,
                topN,
                countLimit,
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the profile timers");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Runs a query blob collapsed ({@link NativeCollapse}): the sort blob's keys ({@link
     * SortEncoder}) on the collapse blob's field, {@code topN} groups per slice; the encoded
     * groups ({@code search_collapsed_blobs} in {@code reader_search.rs}) in {@code out[0]}.
     */
    public static int searchCollapsed(long handle, byte[] query, byte[] sort, byte[] collapse, int topN, byte[][] out) {
        if (query == null) {
            return nullArgument("query");
        }
        if (sort == null) {
            return nullArgument("sort");
        }
        if (collapse == null) {
            return nullArgument("collapse");
        }
        if (topN < 1) {
            return fail(INVALID_ARGUMENT, "searchCollapsed: topN must be at least 1");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.SEARCH_COLLAPSED.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                s.bytes(sort),
                (long) sort.length,
                s.bytes(collapse),
                (long) collapse.length,
                (long) topN,
                ptrOf(result),
                lenOf(result)
            );
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the collapsed hits");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Rescores {@code docs}/{@code scores} (a search's hits, best first, global ids) in place as
     * OpenSearch's {@code QueryRescorer} does: the query blob's query over the top {@code window},
     * {@code mode} the {@code QueryRescoreMode} ordinal ({@code rescore_hits} in {@code
     * reader_search.rs}).
     */
    public static int rescore(
        long handle,
        byte[] query,
        int window,
        float queryWeight,
        float rescoreQueryWeight,
        int mode,
        int[] docs,
        float[] scores
    ) {
        if (query == null) {
            return nullArgument("query");
        }
        if (docs == null || scores == null || docs.length != scores.length) {
            return fail(INVALID_ARGUMENT, "rescore: as many scores as documents");
        }
        int n = docs.length;
        try (Scratch s = Scratch.open()) {
            MemorySegment d = s.room(JAVA_INT, Math.max(1, n));
            MemorySegment sc = s.room(JAVA_FLOAT, Math.max(1, n));
            MemorySegment.copy(docs, 0, d, JAVA_INT, 0, n);
            MemorySegment.copy(scores, 0, sc, JAVA_FLOAT, 0, n);
            int rc = (int) H.RESCORE.invokeExact(
                handle,
                s.bytes(query),
                (long) query.length,
                window,
                queryWeight,
                rescoreQueryWeight,
                mode,
                d,
                sc,
                (long) n
            );
            if (rc == OK) {
                MemorySegment.copy(d, JAVA_INT, 0, docs, 0, n);
                MemorySegment.copy(sc, JAVA_FLOAT, 0, scores, 0, n);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Segment {@code segment}'s document {@code doc} (segment-local): its stored fields, in stored
     * order, encoded as {@code fetch.rs} documents ({@link NativeStoredFieldsReader} reads them),
     * lent to {@code use} in native memory and freed when it returns.
     */
    public static int document(long handle, int segment, int doc, Borrowed use) throws IOException {
        return lendOwned(H.DOCUMENT, handle, segment, doc, use);
    }

    /**
     * Segment {@code segment}'s document {@code doc} (segment-local): its values of the fields
     * numbered {@code fields}, encoded as {@code fetch.rs}'s doc-values blob, in {@code out[0]}
     * ({@link NativeDocValues} reads them).
     */
    public static int docValues(long handle, int segment, int doc, int[] fields, byte[][] out) {
        if (fields == null) {
            return nullArgument("fields");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.DOC_VALUES.invokeExact(handle, segment, doc, s.ints(fields), (long) fields.length, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the doc values");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The {@code _source} of segment {@code segment}'s document {@code doc} with its k-NN vectors put
     * back, lent to {@code use}: the value a {@code DERIVED_SOURCE} field of {@link #document} leaves out.
     */
    public static int derivedSource(long handle, int segment, int doc, Borrowed use) throws IOException {
        return lendOwned(H.DERIVED_SOURCE, handle, segment, doc, use);
    }

    /**
     * {@code StoredFields.prefetch} for the documents {@code docs} (segment-local) of segment {@code
     * segment}: their stored-fields chunks fetched together where the segment is read by range, so
     * the fetch phase's reads one at a time find them local. Nothing to do for files on local disk.
     */
    public static int prefetchDocuments(long handle, int segment, int[] docs) {
        if (docs == null) {
            return nullArgument("docs");
        }
        try (Scratch s = Scratch.open()) {
            return (int) H.PREFETCH_DOCUMENTS.invokeExact(handle, segment, s.ints(docs), (long) docs.length);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Opens a view of the reader {@code base} names: its segments, segment {@code i} hiding the
     * fields {@code hidden[i]} lists, masked with {@code liveDocs} ({@link #openReader}'s contract).
     * What a search sees through a reader wrapper that filters fields and documents -- the security
     * plugin's field- and document-level security. Nothing is read from disk, and the view stands
     * on its own once opened: closing {@code base} leaves it searchable.
     */
    public static int openReaderView(long base, String[][] hidden, int[] maxDocs, long[][] liveDocs, long[] outHandle) {
        return openReaderView(base, hidden, null, maxDocs, liveDocs, outHandle);
    }

    /** What a masked field's values are shown as: the security plugin's field masking, as a function of the stored bytes. */
    public interface FieldMasker {
        /** The bytes shown in place of {@code raw} for {@code field}. */
        byte[] apply(String field, byte[] raw);
    }

    /** The fields a view masks, and how. */
    public record Masking(String[] fields, FieldMasker masker) {}

    /** The arenas holding the mask upcall of each open view, by handle: closed with the handle. */
    private static final java.util.concurrent.ConcurrentHashMap<Long, Arena> MASK_ARENAS = new java.util.concurrent.ConcurrentHashMap<>();

    private static final MethodHandle MASK_CALL;

    static {
        try {
            MASK_CALL = java.lang.invoke.MethodHandles.lookup()
                .findStatic(
                    NativeBridge.class,
                    "maskCall",
                    java.lang.invoke.MethodType.methodType(
                        long.class,
                        FieldMasker.class,
                        MemorySegment.class,
                        long.class,
                        MemorySegment.class,
                        long.class,
                        MemorySegment.class,
                        long.class
                    )
                );
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /**
     * The native side's call for a masked value: writes the masked bytes into {@code out} when
     * they fit and answers their length, or -1 when the mask threw. Nothing may escape to native
     * code, so every throwable is an answer.
     */
    @SuppressWarnings("restricted") // the native side passes exactly nameLen, inputLen and outCap bytes
    private static long maskCall(
        FieldMasker masker,
        MemorySegment name,
        long nameLen,
        MemorySegment input,
        long inputLen,
        MemorySegment out,
        long outCap
    ) {
        try {
            String field = new String(name.reinterpret(nameLen).toArray(JAVA_BYTE), java.nio.charset.StandardCharsets.UTF_8);
            byte[] masked = masker.apply(field, input.reinterpret(inputLen).toArray(JAVA_BYTE));
            if (masked == null) {
                return -1;
            }
            if (masked.length <= outCap) {
                MemorySegment.copy(masked, 0, out.reinterpret(outCap), JAVA_BYTE, 0, masked.length);
            }
            return masked.length;
        } catch (Throwable t) {
            return -1;
        }
    }

    /**
     * {@link #openReaderView(long, String[][], int[], long[][], long[])}, also showing the fields
     * {@code masking} names through its masker: the native side calls the masker, from the
     * searching thread, whenever it shows one of their values. The masker stays reachable until
     * the returned handle is closed ({@link #closeReader}).
     */
    @SuppressWarnings("restricted") // upcall stub lifetime is the arena closed with the handle
    public static int openReaderView(
        long base,
        String[][] hidden,
        Masking masking,
        int[] maxDocs,
        long[][] liveDocs,
        long[] outHandle
    ) {
        if (hidden == null) {
            return nullArgument("hidden");
        }
        if (maxDocs == null) {
            return nullArgument("maxDocs");
        }
        int segments = maxDocs.length;
        if (hidden.length != segments || (liveDocs != null && liveDocs.length != segments)) {
            return fail(INVALID_ARGUMENT, "hidden and liveDocs must have an entry for each of the " + segments + " segments");
        }
        java.io.ByteArrayOutputStream blob = new java.io.ByteArrayOutputStream();
        java.nio.ByteBuffer word = java.nio.ByteBuffer.allocate(Integer.BYTES).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (String[] names : hidden) {
            writeInt(blob, word, names.length);
            for (String name : names) {
                byte[] utf8 = name.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                writeInt(blob, word, utf8.length);
                blob.writeBytes(utf8);
            }
        }
        byte[] hiddenBlob = blob.toByteArray();
        byte[] maskedBlob = new byte[0];
        if (masking != null && masking.fields().length > 0) {
            java.io.ByteArrayOutputStream masked = new java.io.ByteArrayOutputStream();
            writeInt(masked, word, masking.fields().length);
            for (String name : masking.fields()) {
                byte[] utf8 = name.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                writeInt(masked, word, utf8.length);
                masked.writeBytes(utf8);
            }
            maskedBlob = masked.toByteArray();
        }
        Arena maskArena = maskedBlob.length == 0 ? null : Arena.ofShared();
        boolean opened = false;
        try (Scratch s = Scratch.open()) {
            long[] counts = new long[segments];
            long words = 0;
            if (liveDocs != null) {
                for (int i = 0; i < segments; i++) {
                    if (liveDocs[i] != null) {
                        counts[i] = liveDocs[i].length;
                        words += liveDocs[i].length;
                    }
                }
            }
            MemorySegment wordsSeg = s.room(JAVA_LONG, words);
            long at = 0;
            if (liveDocs != null) {
                for (long[] w : liveDocs) {
                    if (w != null) {
                        MemorySegment.copy(w, 0, wordsSeg, JAVA_LONG, at * Long.BYTES, w.length);
                        at += w.length;
                    }
                }
            }
            MemorySegment stub = MemorySegment.NULL;
            if (maskArena != null) {
                stub = Linker.nativeLinker()
                    .upcallStub(
                        java.lang.invoke.MethodHandles.insertArguments(MASK_CALL, 0, masking.masker()),
                        FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG),
                        maskArena
                    );
            }
            MemorySegment out = s.alloc(Long.BYTES);
            int rc = (int) H.OPEN_READER_VIEW.invokeExact(
                base,
                s.bytes(hiddenBlob),
                (long) hiddenBlob.length,
                s.bytes(maskedBlob),
                (long) maskedBlob.length,
                stub,
                s.ints(maxDocs),
                (long) segments,
                wordsSeg,
                s.longs(counts),
                out
            );
            if (rc == OK) {
                long handle = out.get(JAVA_LONG, 0);
                if (outHandle == null || outHandle.length < 1) {
                    closeReader(handle);
                    return fail(INVALID_ARGUMENT, "outHandle holds no slot");
                }
                if (maskArena != null) {
                    MASK_ARENAS.put(handle, maskArena);
                }
                opened = true;
                outHandle[0] = handle;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        } finally {
            if (maskArena != null && opened == false) {
                maskArena.close();
            }
        }
    }

    private static void writeInt(java.io.ByteArrayOutputStream out, java.nio.ByteBuffer word, int value) {
        word.clear();
        word.putInt(value);
        out.write(word.array(), 0, Integer.BYTES);
    }

    public static int closeReader(long handle) {
        try {
            int rc = (int) H.CLOSE_READER.invokeExact(handle);
            // The view's mask upcall goes after its handle: nothing calls it once it is closed.
            Arena maskArena = MASK_ARENAS.remove(handle);
            if (maskArena != null) {
                maskArena.close();
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
