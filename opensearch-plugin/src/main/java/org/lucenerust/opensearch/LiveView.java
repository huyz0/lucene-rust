/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One searcher's view of the documents still in the Rust writer's buffers ({@link
 * NativeBridge#liveViewOpen}): how many documents each buffer held when the view opened. Its
 * documents are numbered buffer after buffer; each buffer is one or more chunks of consecutive
 * documents, as the native reader of the view holds it as segments, and a {@link LiveLeafReader} per
 * chunk gives them to the searcher after the reader's own. Closing it lets its buffers go.
 */
public final class LiveView implements AutoCloseable {
    private final long id;
    /** Buffer {@code i}'s first document, and the total last. */
    private final int[] starts;
    /** Per chunk: its buffer, its first document in the buffer, and its documents. */
    private final int[] chunkBuffer;
    private final int[] chunkStart;
    private final int[] chunkDocs;
    private final AtomicBoolean closed = new AtomicBoolean();
    /** By buffer and field: {@link #pointBounds}, once read. */
    private final Map<Long, byte[]> bounds = new ConcurrentHashMap<>();

    private LiveView(long id, int[] starts, int[] chunkBuffer, int[] chunkStart, int[] chunkDocs) {
        this.id = id;
        this.starts = starts;
        this.chunkBuffer = chunkBuffer;
        this.chunkStart = chunkStart;
        this.chunkDocs = chunkDocs;
    }

    /** The view {@code id} its blob describes; the view is closed if the blob cannot be read. */
    public static LiveView of(long id, byte[] blob) {
        try {
            ByteBuffer b = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
            b.getLong();
            int count = b.getInt();
            int[] starts = new int[count + 1];
            IntArrayList buffers = new IntArrayList();
            IntArrayList chunkStarts = new IntArrayList();
            IntArrayList docs = new IntArrayList();
            for (int i = 0; i < count; i++) {
                int maxDoc = b.getInt();
                starts[i + 1] = Math.addExact(starts[i], maxDoc);
                // The native search applies the deletes itself.
                int deleted = b.getInt();
                b.position(Math.addExact(b.position(), Math.multiplyExact(deleted, Integer.BYTES)));
                int chunks = b.getInt();
                int from = 0;
                for (int c = 0; c < chunks; c++) {
                    int end = b.getInt();
                    if (end < from || end > maxDoc) {
                        throw new IllegalArgumentException("live view: chunk end " + end + " of a buffer of " + maxDoc);
                    }
                    buffers.add(i);
                    chunkStarts.add(from);
                    docs.add(end - from);
                    from = end;
                }
                if (from != maxDoc) {
                    throw new IllegalArgumentException("live view: chunks of " + from + " of a buffer of " + maxDoc);
                }
            }
            return new LiveView(id, starts, buffers.toArray(), chunkStarts.toArray(), docs.toArray());
        } catch (RuntimeException e) {
            NativeBridge.liveViewClose(id);
            throw e;
        }
    }

    /**
     * Field number {@code field}'s points' bounds in buffer {@code buffer} ({@link
     * NativeBridge#livePointBounds}), read once per view: empty when it holds none.
     */
    public byte[] pointBounds(int buffer, int field) throws IOException {
        long key = ((long) buffer << 32) | (field & 0xFFFFFFFFL);
        byte[] got = bounds.get(key);
        if (got == null) {
            byte[][] out = new byte[1][];
            int rc = NativeBridge.livePointBounds(id, buffer, field, out);
            if (rc != NativeBridge.OK) {
                throw new IOException("buffered points of field " + field + ": " + NativeBridge.lastError());
            }
            got = out[0];
            bounds.put(key, got);
        }
        return got;
    }

    /** The native handle. */
    public long id() {
        return id;
    }

    /** Every buffered document the view holds, deleted ones included. */
    public int maxDoc() {
        return starts[starts.length - 1];
    }

    /** How many buffers the view reads. */
    public int buffers() {
        return starts.length - 1;
    }

    /** Buffer {@code i}'s documents, deleted ones included. */
    public int bufferMaxDoc(int i) {
        return starts[i + 1] - starts[i];
    }

    /** How many chunks the view's buffers are, together. */
    public int chunks() {
        return chunkDocs.length;
    }

    /** The buffer chunk {@code c} is part of. */
    public int chunkBuffer(int c) {
        return chunkBuffer[c];
    }

    /** Chunk {@code c}'s first document, in its buffer. */
    public int chunkStart(int c) {
        return chunkStart[c];
    }

    /** Chunk {@code c}'s documents, deleted ones included. */
    public int chunkMaxDoc(int c) {
        return chunkDocs[c];
    }

    /** A growable list of ints. */
    private static final class IntArrayList {
        private int[] values = new int[4];
        private int size;

        void add(int v) {
            if (size == values.length) {
                values = java.util.Arrays.copyOf(values, size * 2);
            }
            values[size++] = v;
        }

        int[] toArray() {
            return java.util.Arrays.copyOf(values, size);
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            NativeBridge.liveViewClose(id);
        }
    }
}
