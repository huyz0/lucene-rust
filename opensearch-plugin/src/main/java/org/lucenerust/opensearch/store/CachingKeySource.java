/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.store;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.LongSupplier;

/**
 * Bounds KMS round trips. A generated data key is reused for new files until {@code generateTtlNanos}
 * passes (each chunk still gets its own random nonce, and one key seals far fewer than the 2^32
 * chunks NIST allows in that time), so a flush that writes a dozen files calls the KMS at most once.
 * Unwrapped keys are kept for {@code unwrapTtlNanos}, at most {@code maxEntries} of them, so a
 * revoked key stops working within that time.
 */
public final class CachingKeySource implements KeySource {
    private final KeySource inner;
    private final long generateTtlNanos;
    private final long unwrapTtlNanos;
    private final int maxEntries;
    private final LongSupplier clock;

    private DataKey current;
    private long currentAt;
    private final LinkedHashMap<Entry, Cached> unwrapped = new LinkedHashMap<>(16, 0.75f, true);

    public CachingKeySource(KeySource inner, long generateTtlNanos, long unwrapTtlNanos, int maxEntries) {
        this(inner, generateTtlNanos, unwrapTtlNanos, maxEntries, System::nanoTime);
    }

    CachingKeySource(KeySource inner, long generateTtlNanos, long unwrapTtlNanos, int maxEntries, LongSupplier clock) {
        this.inner = inner;
        this.generateTtlNanos = generateTtlNanos;
        this.unwrapTtlNanos = unwrapTtlNanos;
        this.maxEntries = maxEntries;
        this.clock = clock;
    }

    @Override
    public synchronized DataKey generate() throws IOException {
        long now = clock.getAsLong();
        if (current == null || now - currentAt >= generateTtlNanos) {
            current = inner.generate();
            currentAt = now;
            remember(new Entry(current.keyId(), current.wrapped()), current.raw(), now);
        }
        return current;
    }

    @Override
    public byte[] unwrap(String keyId, byte[] wrapped) throws IOException {
        Entry e = new Entry(keyId, wrapped);
        long now = clock.getAsLong();
        synchronized (this) {
            Cached c = unwrapped.get(e);
            if (c != null && now - c.at < unwrapTtlNanos) {
                return c.raw;
            }
        }
        byte[] raw = inner.unwrap(keyId, wrapped);
        if (raw == null || raw.length != Lrse.DATA_KEY_LEN) {
            throw new IOException("key provider returned a data key that is not " + Lrse.DATA_KEY_LEN + " bytes");
        }
        synchronized (this) {
            remember(e, raw, now);
        }
        return raw;
    }

    private void remember(Entry e, byte[] raw, long now) {
        if (maxEntries <= 0) {
            return;
        }
        for (Iterator<Map.Entry<Entry, Cached>> it = unwrapped.entrySet().iterator(); it.hasNext();) {
            if (now - it.next().getValue().at >= unwrapTtlNanos) {
                it.remove();
            }
        }
        while (unwrapped.size() >= maxEntries) {
            Iterator<Entry> it = unwrapped.keySet().iterator();
            it.next();
            it.remove();
        }
        unwrapped.put(e, new Cached(raw, now));
    }

    /** Forgets every cached key, generated and unwrapped. */
    public synchronized void clear() {
        current = null;
        unwrapped.clear();
    }

    @Override
    public void close() throws IOException {
        clear();
        inner.close();
    }

    private record Entry(String keyId, byte[] wrapped) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Entry e && keyId.equals(e.keyId) && Arrays.equals(wrapped, e.wrapped);
        }

        @Override
        public int hashCode() {
            return keyId.hashCode() * 31 + Arrays.hashCode(wrapped);
        }
    }

    private record Cached(byte[] raw, long at) {}
}
