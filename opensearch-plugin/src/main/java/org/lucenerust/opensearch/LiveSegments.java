/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SlowCodecReaderWrapper;
import org.apache.lucene.index.StandardDirectoryReader;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * A view's buffers as segments Lucene reads ({@link NativeBridge#liveViewLucene}): what a search the
 * native engine cannot run reads the buffered documents through.
 */
final class LiveSegments {
    /** Per segment of the view's base reader: its live documents as of the view, or null when unchanged. */
    final FixedBitSet[] baseLive;
    /** Per leaf of the view: a segment of its documents, numbered as the leaf numbers them. */
    final LeafReader[] leaves;
    /** What holds each of {@link #leaves} open. */
    final Closeable[] owners;

    private LiveSegments(FixedBitSet[] baseLive, LeafReader[] leaves, Closeable[] owners) {
        this.baseLive = baseLive;
        this.leaves = leaves;
        this.owners = owners;
    }

    /**
     * The segments {@code blob} describes, for leaves of {@code docs[i]} documents each; {@code
     * baseMaxDocs} are the base reader's segments' sizes, and {@code store} the index's directory,
     * where a flushed buffer's segment is.
     */
    static LiveSegments open(byte[] blob, int[] baseMaxDocs, int[] docs, Directory store) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        int segments = b.getInt();
        if (segments != baseMaxDocs.length) {
            throw new IOException("buffered segments: " + segments + " base segments for a reader of " + baseMaxDocs.length);
        }
        FixedBitSet[] baseLive = new FixedBitSet[segments];
        for (int s = 0; s < segments; s++) {
            int words = b.getInt();
            if (words < 0) {
                continue;
            }
            if (words != FixedBitSet.bits2words(baseMaxDocs[s])) {
                throw new IOException("buffered segments: " + words + " words of live docs for a segment of " + baseMaxDocs[s]);
            }
            long[] bits = new long[words];
            for (int w = 0; w < words; w++) {
                bits[w] = b.getLong();
            }
            baseLive[s] = new FixedBitSet(bits, baseMaxDocs[s]);
        }
        int count = b.getInt();
        if (count != docs.length) {
            throw new IOException("buffered segments: " + count + " for a view of " + docs.length + " leaves");
        }
        LeafReader[] leaves = new LeafReader[count];
        Closeable[] owners = new Closeable[count];
        boolean success = false;
        try {
            for (int i = 0; i < count; i++) {
                int kind = b.get();
                String name = string(b);
                byte[] id = new byte[StringHelper.ID_LENGTH];
                b.get(id);
                String codec = string(b);
                int offset = b.getInt();
                Directory dir;
                if (kind == 0) {
                    ByteBuffersDirectory memory = new ByteBuffersDirectory();
                    int files = b.getInt();
                    for (int f = 0; f < files; f++) {
                        String file = string(b);
                        long len = b.getLong();
                        if (len < 0 || len > b.remaining()) {
                            throw new IOException("buffered segments: file " + file + " of " + len + " bytes");
                        }
                        try (IndexOutput out = memory.createOutput(file, IOContext.DEFAULT)) {
                            out.writeBytes(blob, b.position(), (int) len);
                        }
                        b.position(b.position() + (int) len);
                    }
                    dir = memory;
                } else if (kind == 1) {
                    dir = store;
                } else {
                    throw new IOException("buffered segments: kind " + kind);
                }
                DirectoryReader reader = openSegment(dir, name, id, codec);
                owners[i] = reader;
                LeafReader leaf = reader.leaves().get(0).reader();
                if (offset != 0 || leaf.maxDoc() != docs[i]) {
                    // Part of a segment: its documents copied out, renumbered from 0.
                    DirectoryReader part = copyRange(leaf, offset, docs[i]);
                    IOUtils.close(reader);
                    owners[i] = part;
                    leaf = part.leaves().isEmpty() ? null : part.leaves().get(0).reader();
                    if (leaf == null || leaf.maxDoc() != docs[i]) {
                        throw new IOException("buffered segments: a part of " + docs[i] + " documents of " + name);
                    }
                }
                leaves[i] = leaf;
            }
            if (b.hasRemaining()) {
                throw new IOException("buffered segments: " + b.remaining() + " trailing bytes");
            }
            success = true;
            return new LiveSegments(baseLive, leaves, owners);
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(owners);
            }
        }
    }

    private static String string(ByteBuffer b) throws IOException {
        int len = b.getInt();
        if (len < 0 || len > b.remaining()) {
            throw new IOException("buffered segments: a name of " + len + " bytes");
        }
        byte[] bytes = new byte[len];
        b.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /** Segment {@code name} of {@code dir} alone, written by codec {@code codec}, without its deletes. */
    private static DirectoryReader openSegment(Directory dir, String name, byte[] id, String codec) throws IOException {
        Codec c = Codec.forName(codec);
        SegmentInfo si = c.segmentInfoFormat().read(dir, name, id, IOContext.READONCE);
        si.setCodec(c);
        SegmentInfos infos = new SegmentInfos(Version.LATEST.major);
        infos.add(new SegmentCommitInfo(si, 0, 0, -1, -1, -1, StringHelper.randomId()));
        return StandardDirectoryReader.open(dir, infos, null, null, null);
    }

    /** Documents {@code from} to {@code from + docs} of {@code leaf}, as a segment of their own in memory. */
    private static DirectoryReader copyRange(LeafReader leaf, int from, int docs) throws IOException {
        if (from < 0 || docs < 0 || from + docs > leaf.maxDoc()) {
            throw new IOException("buffered segments: documents " + from + "+" + docs + " of a segment of " + leaf.maxDoc());
        }
        FieldInfos infos = leaf.getFieldInfos();
        IndexWriterConfig config = new IndexWriterConfig(null).setMergePolicy(NoMergePolicy.INSTANCE).setUseCompoundFile(false);
        if (infos.getSoftDeletesField() != null) {
            config.setSoftDeletesField(infos.getSoftDeletesField());
        }
        if (infos.getParentField() != null) {
            config.setParentField(infos.getParentField());
        }
        ByteBuffersDirectory out = new ByteBuffersDirectory();
        Range range = new Range(leaf, from, docs);
        try (IndexWriter w = new IndexWriter(out, config)) {
            w.addIndexes(SlowCodecReaderWrapper.wrap(range));
            w.commit();
        }
        return DirectoryReader.open(out);
    }

    /** {@code in} with only documents {@code from} to {@code from + docs} live. */
    private static final class Range extends FilterLeafReader {
        private final Bits live;
        private final int docs;

        Range(LeafReader in, int from, int docs) {
            super(in);
            FixedBitSet bits = new FixedBitSet(in.maxDoc());
            bits.set(from, from + docs);
            this.live = bits;
            this.docs = docs;
        }

        @Override
        public Bits getLiveDocs() {
            return live;
        }

        @Override
        public int numDocs() {
            return docs;
        }

        @Override
        public CacheHelper getCoreCacheHelper() {
            return null;
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            return null;
        }
    }
}
