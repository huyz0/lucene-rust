/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;

/**
 * A match-all {@code terms} aggregation on a low-cardinality keyword field, answered from each
 * segment's term document frequencies -- what OpenSearch's own aggregator does for it -- and those
 * kept per segment (segments never change), so a repeat costs the merge of a few dozen counts and
 * neither the query encoding, the native call nor the memory scope. Only where document
 * frequencies are exact: no deletions anywhere, a field of at most {@link #MAX_TERMS} terms per
 * segment, a segment that offers a cache key; anything else is the caller's normal path.
 */
final class TermsDocFreqShortcut {
    /** Most terms of a field in one segment that are kept and summed. */
    static final int MAX_TERMS = 1024;
    private static final int MAX_ENTRIES = 512;

    private record Written(int shardSize, byte[] bytes) {}

    /** One segment's terms with their frequencies, and the last result written for it alone. */
    private static final class Freqs {
        final BytesRef[] terms;
        final long[] counts;
        private volatile Written written;

        Freqs(BytesRef[] terms, long[] counts) {
            this.terms = terms;
            this.counts = counts;
        }
    }

    private record Key(Object segment, String field) {}

    private static final Map<Key, Freqs> CACHE = new LinkedHashMap<>(64, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<Key, Freqs> eldest) {
            return size() > MAX_ENTRIES;
        }
    };

    private TermsDocFreqShortcut() {}

    /**
     * The native-format terms result (what {@code Terms.read} reads) of {@code slices}' segments,
     * one block per slice (one for all leaves when {@code slices} is empty), or null when a
     * segment cannot be answered this way.
     */
    static byte[] result(IndexReader reader, String field, int shardSize, int[][] slices) throws IOException {
        if (reader.hasDeletions()) {
            return null;
        }
        List<LeafReaderContext> leaves = reader.leaves();
        Freqs[] entries = new Freqs[leaves.size()];
        for (int i = 0; i < entries.length; i++) {
            entries[i] = entry(leaves.get(i).reader(), field);
            if (entries[i] == null) {
                return null;
            }
        }
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        if (slices.length == 0) {
            int[] all = new int[entries.length];
            Arrays.setAll(all, i -> i);
            write(out, entries, all, shardSize);
        } else {
            for (int[] slice : slices) {
                write(out, entries, slice, shardSize);
            }
        }
        return out.toByteArray();
    }

    private static void write(java.io.ByteArrayOutputStream out, Freqs[] entries, int[] segments, int shardSize) {
        if (segments.length == 1) {
            Freqs only = entries[segments[0]];
            Written kept = only.written;
            if (kept == null || kept.shardSize != shardSize) {
                java.io.ByteArrayOutputStream one = new java.io.ByteArrayOutputStream();
                writeMerged(one, entries, segments, shardSize);
                kept = new Written(shardSize, one.toByteArray());
                only.written = kept;
            }
            out.writeBytes(kept.bytes);
            return;
        }
        writeMerged(out, entries, segments, shardSize);
    }

    private static void writeMerged(java.io.ByteArrayOutputStream out, Freqs[] entries, int[] segments, int shardSize) {
        TreeMap<BytesRef, long[]> merged = new TreeMap<>();
        for (int s : segments) {
            Freqs e = entries[s];
            for (int t = 0; t < e.terms.length; t++) {
                merged.computeIfAbsent(e.terms[t], k -> new long[1])[0] += e.counts[t];
            }
        }
        List<Map.Entry<BytesRef, long[]>> ranked = new ArrayList<>(merged.entrySet());
        long total = 0;
        for (Map.Entry<BytesRef, long[]> e : ranked) {
            total += e.getValue()[0];
        }
        // The shard_size most frequent, ties by term: the native pass's selection, emitted by term.
        ranked.sort((a, b) -> {
            int c = Long.compare(b.getValue()[0], a.getValue()[0]);
            return c != 0 ? c : a.getKey().compareTo(b.getKey());
        });
        List<Map.Entry<BytesRef, long[]>> kept = ranked.subList(0, Math.min(Math.max(shardSize, 0), ranked.size()));
        long selected = 0;
        for (Map.Entry<BytesRef, long[]> e : kept) {
            selected += e.getValue()[0];
        }
        List<Map.Entry<BytesRef, long[]>> byTerm = new ArrayList<>(kept);
        byTerm.sort(Map.Entry.comparingByKey());
        int size = 12;
        for (Map.Entry<BytesRef, long[]> e : byTerm) {
            size += 12 + e.getKey().length;
        }
        ByteBuffer b = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
        b.putLong(total - selected);
        b.putInt(byTerm.size());
        for (Map.Entry<BytesRef, long[]> e : byTerm) {
            b.putLong(e.getValue()[0]);
            b.putInt(e.getKey().length);
            b.put(e.getKey().bytes, e.getKey().offset, e.getKey().length);
        }
        out.writeBytes(b.array());
    }

    private static Freqs entry(LeafReader leaf, String field) throws IOException {
        LeafReader.CacheHelper helper = leaf.getCoreCacheHelper();
        if (helper == null) {
            return null;
        }
        Key key = new Key(helper.getKey(), field);
        synchronized (CACHE) {
            Freqs hit = CACHE.get(key);
            if (hit != null) {
                return hit;
            }
        }
        Terms terms = leaf.terms(field);
        if (terms == null) {
            return null;
        }
        long n = terms.size();
        if (n < 0 || n > MAX_TERMS) {
            return null;
        }
        BytesRef[] names = new BytesRef[(int) n];
        long[] counts = new long[(int) n];
        TermsEnum te = terms.iterator();
        int i = 0;
        for (BytesRef t = te.next(); t != null; t = te.next()) {
            if (i == names.length) {
                return null;
            }
            names[i] = BytesRef.deepCopyOf(t);
            counts[i++] = te.docFreq();
        }
        if (i != names.length) {
            return null;
        }
        Freqs made = new Freqs(names, counts);
        synchronized (CACHE) {
            if (CACHE.put(key, made) == null) {
                helper.addClosedListener(k -> {
                    synchronized (CACHE) {
                        CACHE.remove(key);
                    }
                });
            }
        }
        return made;
    }
}
