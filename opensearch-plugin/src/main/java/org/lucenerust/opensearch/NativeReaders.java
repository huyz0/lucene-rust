/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.StandardDirectoryReader;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.store.ByteBuffersIndexOutput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * One native reader per Java searcher reader, opened on first use and closed when the Java reader
 * closes.
 *
 * <p>The native reader is built from exactly what the Java reader sees -- its leaves' own {@link
 * SegmentCommitInfo}s, in leaf order, and each leaf's live docs -- so a native doc ID names the same
 * document as a Java one. An OpenSearch searcher is an NRT reader: its newest segments are not in
 * any commit, and its deletions (hard and soft) are in memory, so opening the latest {@code
 * segments_N} instead would be wrong in both.
 *
 * <p>Lifecycle (M2 T2.6): the native handle is closed from the Java reader's {@link
 * IndexReader.CacheHelper} closed-listener -- when OpenSearch releases a searcher's reader, the
 * native side releases its mapped segment files with it, so a merged-away segment's files can be
 * deleted. A refresh reuses the previous native reader's unchanged segments.
 */
public final class NativeReaders {
    private static final Logger logger = LogManager.getLogger(NativeReaders.class);

    /** A native handle, or the reason there is none (cached, so a failing reader fails once). */
    public record Acquired(long handle, String fallbackReason) {}

    /**
     * The one postings format the Rust reader decodes: Lucene 10.5.0's default. A field written in
     * any other -- OpenSearch's {@code completion} fields use {@code Completion104} -- makes the whole
     * reader fall back, rather than failing (and falling back) on every query that opens it.
     */
    static final String SUPPORTED_POSTINGS_FORMAT = "Lucene104";

    private static final String POSTINGS_FORMAT_ATTRIBUTE = "PerFieldPostingsFormat.format";

    private final ConcurrentHashMap<IndexReader.CacheKey, Acquired> byReader = new ConcurrentHashMap<>();
    /** The most recent handle per index directory: the reuse candidate for the next refresh. */
    private final ConcurrentHashMap<Path, Long> latestByDir = new ConcurrentHashMap<>();
    private final AtomicLong open = new AtomicLong();

    /** Native readers currently open. */
    public long openCount() {
        return open.get();
    }

    /** Readers with a cached outcome, native or fallback: must shrink as Java readers close. */
    public int cachedCount() {
        return byReader.size();
    }

    public Acquired acquire(IndexReader reader) {
        IndexReader.CacheHelper helper = reader.getReaderCacheHelper();
        if (helper == null) {
            return new Acquired(0, "reader_uncacheable");
        }
        IndexReader.CacheKey key = helper.getKey();
        Acquired cached = byReader.get(key);
        if (cached != null) {
            return cached;
        }
        return byReader.computeIfAbsent(key, k -> {
            Acquired a = openFor(reader);
            // Failures are cached for the reader's life too -- and so must be evicted with it, or an
            // index that always falls back (a completion field) leaks one entry per refresh.
            try {
                helper.addClosedListener(this::onClosed);
            } catch (RuntimeException e) {
                // The reader closed between our search starting and now: nothing will call us back.
                if (a.handle() != 0) {
                    close(a.handle());
                }
                return new Acquired(0, "reader_closed");
            }
            return a;
        });
    }

    private void onClosed(IndexReader.CacheKey key) {
        Acquired a = byReader.remove(key);
        if (a != null && a.handle() != 0) {
            latestByDir.values().remove(a.handle());
            close(a.handle());
        }
    }

    private void close(long handle) {
        int rc = NativeBridge.closeReader(handle);
        if (rc == NativeBridge.OK) {
            open.decrementAndGet();
        } else {
            logger.warn("lucene-rust: closing native reader failed ({}): {}", rc, NativeBridge.lastError());
        }
    }

    /** Closes every native reader; for node shutdown. */
    public void closeAll() {
        for (IndexReader.CacheKey key : List.copyOf(byReader.keySet())) {
            onClosed(key);
        }
    }

    private Acquired openFor(IndexReader reader) {
        if (!(reader instanceof DirectoryReader dr)) {
            return new Acquired(0, "reader_not_directory");
        }
        DirectoryReader unwrapped = FilterDirectoryReader.unwrap(dr);
        if (!(unwrapped instanceof StandardDirectoryReader sdr)) {
            return new Acquired(0, "reader_not_standard");
        }
        List<LeafReaderContext> leaves = reader.leaves();
        SegmentInfos infos = sdr.getSegmentInfos().clone();
        infos.clear();
        int[] maxDocs = new int[leaves.size()];
        Path path = null;
        for (int i = 0; i < leaves.size(); i++) {
            LeafReader leaf = leaves.get(i).reader();
            for (FieldInfo fi : leaf.getFieldInfos()) {
                String format = fi.getAttribute(POSTINGS_FORMAT_ATTRIBUTE);
                if (format != null && format.equals(SUPPORTED_POSTINGS_FORMAT) == false) {
                    return new Acquired(0, "postings_format");
                }
            }
            SegmentCommitInfo sci = org.opensearch.common.lucene.Lucene.segmentReader(leaf).getSegmentInfo();
            infos.add(sci);
            maxDocs[i] = leaf.maxDoc();
            Directory d = FilterDirectory.unwrap(sci.info.dir);
            if (!(d instanceof FSDirectory fs)) {
                return new Acquired(0, "directory_not_fs");
            }
            Path p = fs.getDirectory();
            if (path != null && path.equals(p) == false) {
                return new Acquired(0, "directory_mixed");
            }
            path = p;
        }
        if (path == null) {
            Directory d = FilterDirectory.unwrap(sdr.directory());
            if (!(d instanceof FSDirectory fs)) {
                return new Acquired(0, "directory_not_fs");
            }
            path = fs.getDirectory();
        }
        byte[] infoBytes;
        try {
            ByteBuffersDataOutput buf = new ByteBuffersDataOutput();
            try (ByteBuffersIndexOutput out = new ByteBuffersIndexOutput(buf, "segments", "segments")) {
                infos.write(out);
            }
            infoBytes = buf.toArrayCopy();
        } catch (IOException e) {
            logger.warn("lucene-rust: serializing SegmentInfos failed", e);
            return new Acquired(0, "segment_infos_write");
        }
        byte[] pathBytes = path.toAbsolutePath().toString().getBytes(StandardCharsets.UTF_8);
        long[][] live = new long[leaves.size()][];
        for (int i = 0; i < leaves.size(); i++) {
            Bits bits = leaves.get(i).reader().getLiveDocs();
            live[i] = bits == null ? null : words(bits, maxDocs[i]);
        }
        long[] out = new long[1];
        Long previous = latestByDir.get(path);
        int rc = NativeBridge.openReader(pathBytes, infoBytes, infos.getGeneration(), previous == null ? 0 : previous, maxDocs, live, out);
        if (rc == NativeBridge.INVALID_HANDLE && previous != null) {
            // The reuse candidate closed under us; open from scratch.
            rc = NativeBridge.openReader(pathBytes, infoBytes, infos.getGeneration(), 0, maxDocs, live, out);
        }
        if (rc != NativeBridge.OK) {
            logger.warn("lucene-rust: native reader open failed for [{}] ({}): {}", path, rc, NativeBridge.lastError());
            return new Acquired(0, "native_open_failed");
        }
        long handle = out[0];
        open.incrementAndGet();
        latestByDir.put(path, handle);
        return new Acquired(handle, null);
    }

    /** {@code live} as exactly {@code ceil(maxDoc / 64)} words, no bits set past {@code maxDoc}. */
    static long[] words(Bits live, int maxDoc) {
        int n = (maxDoc + 63) >>> 6;
        long[] words;
        if (live instanceof FixedBitSet fbs) {
            words = java.util.Arrays.copyOf(fbs.getBits(), n);
        } else {
            words = new long[n];
            for (int d = 0; d < maxDoc; d++) {
                if (live.get(d)) {
                    words[d >>> 6] |= 1L << d;
                }
            }
        }
        int tail = maxDoc & 63;
        if (tail != 0) {
            words[n - 1] &= (1L << tail) - 1;
        }
        return words;
    }
}
