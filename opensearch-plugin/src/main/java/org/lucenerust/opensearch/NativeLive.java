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
import static org.lucenerust.opensearch.NativeSearch.*;
import static org.lucenerust.opensearch.NativeStore.*;
import static org.lucenerust.opensearch.NativeAnalysis.*;
import static org.lucenerust.opensearch.NativeMl.*;
import static org.lucenerust.opensearch.NativeSparse.*;
import static org.lucenerust.opensearch.NativeZstd.*;
import static org.lucenerust.opensearch.NativeLimits.*;
import static org.lucenerust.opensearch.NativeWriter.*;

/** The live search downcalls over the writer's buffers; see {@link NativeBridge}. */
public final class NativeLive {
    private NativeLive() {}

    private static final class H {
        static final MethodHandle LIVE_VIEW_OPEN = bind(
            "ffi_engine_writer_live_view_open",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS, ADDRESS)
        );
        static final MethodHandle LIVE_VIEW_CLOSE = bind("ffi_live_view_close", FunctionDescriptor.of(JAVA_INT, JAVA_LONG));
        static final MethodHandle LIVE_VIEW_READER = bind(
            "ffi_live_view_reader",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS)
        );
        static final MethodHandle LIVE_VIEW_LUCENE = bind(
            "ffi_live_view_lucene",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle WRITER_SET_LIVE_STALENESS = bind(
            "ffi_engine_writer_set_live_staleness",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle WRITER_LIVE_REFRESH_SHARED = bind(
            "ffi_engine_writer_live_refresh_shared",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG)
        );
        static final MethodHandle WRITER_RELEASE_LIVE = bind(
            "ffi_engine_writer_release_live",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG)
        );
        static final MethodHandle LIVE_DOC_VALUES = bind(
            "ffi_live_view_doc_values_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS)
        );
        static final MethodHandle LIVE_DOCUMENT = bind(
            "ffi_live_view_document_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle LIVE_POINT_BOUNDS = bind(
            "ffi_live_view_point_bounds_alloc",
            FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS)
        );
        static final MethodHandle SEARCH_LIVE = bind(
            "ffi_jvm_reader_search_live",
            FunctionDescriptor.of(
                JAVA_INT,
                JAVA_LONG,
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
                ADDRESS,
                ADDRESS
            )
        );
    }

    // ---- Live search of the writer's buffers (live.rs). ----

    /**
     * Opens a view of the documents buffered in writer {@code writer} that a reader of the segment
     * list at version {@code readerVersion} (a commit's or an NRT snapshot's) lacks: its handle into {@code outView[0]}, its blob ({@code published:
     * i64}, {@code count: i32}, then per buffer {@code maxDoc: i32}, {@code deleted: i32} and that many
     * deleted documents, little-endian) into {@code outBlob[0]}. {@link #INVALID_ARGUMENT} when the
     * buffers cannot be read for that list; the caller then searches the reader alone.
     */
    public static int liveViewOpen(long writer, long readerVersion, long[] outView, byte[][] outBlob) {
        if (outView == null || outView.length < 1 || outBlob == null || outBlob.length < 1) {
            return fail(INVALID_ARGUMENT, "outView and outBlob need a slot each");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment view = s.alloc(Long.BYTES);
            MemorySegment blob = ownedSlot(s);
            int rc = (int) H.LIVE_VIEW_OPEN.invokeExact(writer, readerVersion, view, ptrOf(blob), lenOf(blob));
            if (rc == OK) {
                long id = view.get(JAVA_LONG, 0);
                byte[] bytes = take(blob);
                if (bytes == null) {
                    liveViewClose(id);
                    return tooLarge("the live view");
                }
                outView[0] = id;
                outBlob[0] = bytes;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * What a search Lucene answers reads {@code view}'s buffers through, over the reader at {@code
     * handle} (the view's base), into {@code out[0]}: the reader's live documents as of the view, and
     * each leaf of the view as a segment with its postings ({@code ffi_live_view_lucene} gives the
     * layout). A buffer flushed since the view opened is waited for up to {@code waitMillis};
     * {@link #INVALID_ARGUMENT} when it has not published by then.
     */
    public static int liveViewLucene(long handle, long view, long waitMillis, byte[][] out) {
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out needs a slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment blob = ownedSlot(s);
            int rc = (int) H.LIVE_VIEW_LUCENE.invokeExact(handle, view, waitMillis, ptrOf(blob), lenOf(blob));
            if (rc == OK) {
                byte[] bytes = take(blob);
                if (bytes == null) {
                    return tooLarge("the buffered segments");
                }
                out[0] = bytes;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * A reader handle over the reader at {@code handle} and {@code view}'s buffers as segments after
     * its own, one per buffer, into {@code out[0]}: every native search of it reads the buffered
     * documents as the leaves of the view number them. It belongs to the view, which closes it; the
     * caller must not. {@link #INVALID_ARGUMENT} when the buffers cannot be written (one was taken
     * for a flush since the view opened); the caller then searches the reader alone.
     * <p>
     * With {@code query} (a query blob), {@code tree} (an aggregation tree blob, or null) and
     * {@code extra} (the other queries the search runs, a rescore's, or null), the buffers are
     * written without their postings and the handle holds only the terms that request reads: it
     * serves searches of that request alone, much cheaper to build. With a null {@code query}, or a
     * request whose terms cannot be named up front, it holds every term.
     */
    public static int liveViewReader(long handle, long view, byte[] query, byte[] tree, byte[][] extra, long[] out) {
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out needs a slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment id = s.alloc(Long.BYTES);
            MemorySegment q = query == null || query.length == 0 ? MemorySegment.NULL : s.bytes(query);
            MemorySegment t = query == null || tree == null || tree.length == 0 ? MemorySegment.NULL : s.bytes(tree);
            byte[] listed = query == null ? null : lengthPrefixed(extra);
            MemorySegment e = listed == null ? MemorySegment.NULL : s.bytes(listed);
            int rc = (int) H.LIVE_VIEW_READER.invokeExact(
                handle,
                view,
                q,
                q == MemorySegment.NULL ? 0L : (long) query.length,
                t,
                t == MemorySegment.NULL ? 0L : (long) tree.length,
                e,
                e == MemorySegment.NULL ? 0L : (long) listed.length,
                id
            );
            if (rc == OK) {
                out[0] = id.get(JAVA_LONG, 0);
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** {@code blobs} one after another, each after its length as a little-endian int; null when none. */
    private static byte[] lengthPrefixed(byte[][] blobs) {
        if (blobs == null || blobs.length == 0) {
            return null;
        }
        int size = 0;
        for (byte[] b : blobs) {
            size = Math.addExact(size, Math.addExact(Integer.BYTES, b.length));
        }
        java.nio.ByteBuffer out = java.nio.ByteBuffer.allocate(size).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (byte[] b : blobs) {
            out.putInt(b.length).put(b);
        }
        return out.array();
    }

    /** Closes a view: its buffers may go once nothing else reads them. */
    public static int liveViewClose(long view) {
        try {
            return (int) H.LIVE_VIEW_CLOSE.invokeExact(view);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Lets searches of the writer's buffers share a view up to {@code millis} milliseconds old,
     * which the writer opens ahead of them; 0 gives each search a view of every write before it.
     */
    public static int writerSetLiveStaleness(long writer, long millis) {
        try {
            return (int) H.WRITER_SET_LIVE_STALENESS.invokeExact(writer, millis);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Makes every later search of the writer's buffers read a view opened from now on. */
    public static int writerLiveRefreshShared(long writer) {
        try {
            return (int) H.WRITER_LIVE_REFRESH_SHARED.invokeExact(writer);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Lets go of the buffers a reader of the segment list at {@code version} already has, once no view reads them. */
    public static int writerReleaseLive(long writer, long version) {
        try {
            return (int) H.WRITER_RELEASE_LIVE.invokeExact(writer, version);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The bounds of field number {@code field}'s points in buffer {@code buffer} of {@code view}
     * into {@code out[0]}: the point count (little-endian long), then the packed least values, then
     * the packed greatest; empty when the buffer holds no point of the field.
     */
    public static int livePointBounds(long view, int buffer, int field, byte[][] out) {
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.LIVE_POINT_BOUNDS.invokeExact(view, buffer, field, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the bounds");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Document {@code doc} of buffer {@code buffer} of view {@code view}: its doc values of the
     * fields numbered {@code fields}, encoded as {@link #docValues} encodes a segment's, the terms
     * of each field numbered {@code 0..n} for that document alone, in {@code out[0]}.
     */
    public static int liveDocValues(long view, int buffer, int doc, int[] fields, byte[][] out) {
        if (fields == null) {
            return nullArgument("fields");
        }
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.LIVE_DOC_VALUES.invokeExact(view, buffer, doc, s.ints(fields), (long) fields.length, ptrOf(result), lenOf(result));
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
     * Document {@code doc} of buffer {@code buffer} of {@code view}: its stored fields encoded as
     * {@link #document} encodes them, into {@code out[0]}.
     */
    public static int liveDocument(long view, int buffer, int doc, byte[][] out) {
        if (out == null || out.length < 1) {
            return fail(INVALID_ARGUMENT, "out holds no slot");
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment result = ownedSlot(s);
            int rc = (int) H.LIVE_DOCUMENT.invokeExact(view, buffer, doc, ptrOf(result), lenOf(result));
            if (rc == OK) {
                byte[] encoded = take(result);
                if (encoded == null) {
                    return tooLarge("the document");
                }
                out[0] = encoded;
            }
            return rc;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * {@link #search} over reader {@code handle} and the buffers of {@code view} together, a
     * buffered hit numbered after the reader's {@code maxDoc}. {@code outCounts} as for {@link
     * #search}, with a fourth slot 1 when the buffers could answer the query and 0 when they could
     * not, in which case nothing else is written.
     */
    public static int searchLive(
        long handle,
        long view,
        byte[] query,
        int topN,
        long countLimit,
        int[] outDocs,
        float[] outScores,
        long[] outCounts
    ) {
        int refused = NativeSearch.checkSearch(query, topN, outDocs, outScores, outCounts, 4);
        if (refused != OK) {
            return refused;
        }
        try (Scratch s = Scratch.open()) {
            MemorySegment docs = s.room(JAVA_INT, topN);
            MemorySegment scores = s.room(JAVA_FLOAT, topN);
            MemorySegment counts = s.alloc(4 * Long.BYTES);
            int rc = (int) H.SEARCH_LIVE.invokeExact(
                handle,
                view,
                s.bytes(query),
                (long) query.length,
                (long) topN,
                countLimit,
                docs,
                scores,
                (long) topN,
                counts,
                counts.asSlice(Long.BYTES),
                counts.asSlice(2 * Long.BYTES),
                counts.asSlice(3 * Long.BYTES)
            );
            if (rc != OK) {
                return rc;
            }
            boolean supported = counts.get(JAVA_BOOLEAN, 3 * Long.BYTES);
            outCounts[3] = supported ? 1 : 0;
            if (supported == false) {
                return OK;
            }
            int n = Math.toIntExact(counts.get(JAVA_LONG, 0));
            MemorySegment.copy(docs, JAVA_INT, 0, outDocs, 0, n);
            MemorySegment.copy(scores, JAVA_FLOAT, 0, outScores, 0, n);
            outCounts[0] = n;
            outCounts[1] = counts.get(JAVA_LONG, Long.BYTES);
            outCounts[2] = counts.get(JAVA_BOOLEAN, 2 * Long.BYTES) ? 1 : 0;
            return OK;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
