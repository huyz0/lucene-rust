/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One searcher's view of the documents still in the Rust writer's buffers ({@link
 * NativeBridge#liveViewOpen}): how many documents each buffer held when the view opened. Its
 * documents are numbered buffer after buffer; a {@link LiveLeafReader} per buffer gives them to the
 * searcher after the reader's own. Closing it lets its buffers go.
 */
public final class LiveView implements AutoCloseable {
    private final long id;
    /** Buffer {@code i}'s first document, and the total last. */
    private final int[] starts;
    private final AtomicBoolean closed = new AtomicBoolean();

    private LiveView(long id, int[] starts) {
        this.id = id;
        this.starts = starts;
    }

    /** The view {@code id} its blob describes; the view is closed if the blob cannot be read. */
    public static LiveView of(long id, byte[] blob) {
        try {
            ByteBuffer b = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
            b.getLong();
            int count = b.getInt();
            int[] starts = new int[count + 1];
            for (int i = 0; i < count; i++) {
                starts[i + 1] = Math.addExact(starts[i], b.getInt());
                // The native search applies the deletes itself.
                int deleted = b.getInt();
                b.position(Math.addExact(b.position(), Math.multiplyExact(deleted, Integer.BYTES)));
            }
            return new LiveView(id, starts);
        } catch (RuntimeException e) {
            NativeBridge.liveViewClose(id);
            throw e;
        }
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

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            NativeBridge.liveViewClose(id);
        }
    }
}
