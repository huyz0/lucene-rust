/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.ImpactsEnum;
import org.apache.lucene.index.LeafMetaData;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SlowImpactsEnum;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.TermState;
import org.apache.lucene.index.TermVectors;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.lang.ref.Cleaner;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;

/**
 * SPIKE (real-time study, not for merge): one snapshot of the Rust writer's live buffer -- the
 * documents indexed since the last refresh -- as a Lucene leaf, so OpenSearch's searcher sees them
 * without a refresh. Terms are looked up exactly ({@code seekExact}); postings carry docs and freqs
 * (no positions); norms, numeric doc values and stored fields come from the buffer.
 */
public final class LiveLeafReader extends LeafReader {
    private static final Cleaner CLEANER = Cleaner.create();


    private final CacheKey cacheKey = newKey();

    private static CacheKey newKey() {
        try {
            var c = CacheKey.class.getDeclaredConstructor();
            c.setAccessible(true);
            return c.newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }
    private final java.util.List<ClosedListener> closedListeners = new java.util.concurrent.CopyOnWriteArrayList<>();
    private final CacheHelper cacheHelper = new CacheHelper() {
        @Override
        public CacheKey getKey() {
            return cacheKey;
        }

        @Override
        public void addClosedListener(ClosedListener listener) {
            closedListeners.add(listener);
        }
    };

    private void fireClosed() throws java.io.IOException {
        for (ClosedListener l : closedListeners) {
            l.onClose(cacheKey);
        }
    }

    private final long id;
    private final int maxDoc;
    private final FieldInfos infos;
    private final FixedBitSet live;
    private final int numDocs;
    private final Map<String, NumericDocValues> unused = new HashMap<>();

    /** Pins {@code bufferId}'s buffer as a snapshot of its first {@code maxDoc} documents. */
    public LiveLeafReader(long bufferId, int maxDoc, FieldInfos infos) {
        long pinned = NativeBridge.livePin(bufferId);
        this.id = pinned;
        this.maxDoc = maxDoc;
        this.infos = infos;
        CLEANER.register(this, () -> NativeBridge.liveRelease(pinned));
        byte[] del = NativeBridge.liveDeleted(id, maxDoc);
        FixedBitSet bits = new FixedBitSet(Math.max(1, maxDoc));
        bits.set(0, maxDoc);
        int deleted = 0;
        if (del != null) {
            ByteBuffer b = ByteBuffer.wrap(del).order(ByteOrder.LITTLE_ENDIAN);
            while (b.remaining() >= 4) {
                int d = b.getInt();
                if (bits.get(d)) {
                    bits.clear(d);
                    deleted++;
                }
            }
        }
        this.live = deleted == 0 ? null : bits;
        this.numDocs = maxDoc - deleted;
    }

    private long[] column(int kind, String field) {
        FieldInfo fi = infos.fieldInfo(field);
        if (fi == null) {
            return null;
        }
        byte[] col = NativeBridge.liveColumn(id, kind, fi.number, maxDoc);
        if (col == null || col.length == 0) {
            return null;
        }
        ByteBuffer b = ByteBuffer.wrap(col).order(ByteOrder.LITTLE_ENDIAN);
        int n = col.length / 12;
        long[] out = new long[n * 2];
        for (int i = 0; i < n; i++) {
            out[2 * i] = b.getInt() & 0xffffffffL;
            out[2 * i + 1] = b.getLong();
        }
        return out;
    }

    private static NumericDocValues numeric(long[] pairs) {
        if (pairs == null) {
            return null;
        }
        return new NumericDocValues() {
            int i = -1;
            int doc = -1;

            @Override
            public long longValue() {
                return pairs[2 * i + 1];
            }

            @Override
            public boolean advanceExact(int target) {
                while (i + 1 < pairs.length / 2 && pairs[2 * (i + 1)] <= target) {
                    i++;
                }
                doc = target;
                return i >= 0 && pairs[2 * i] == target;
            }

            @Override
            public int docID() {
                return doc;
            }

            @Override
            public int nextDoc() {
                i++;
                doc = i < pairs.length / 2 ? (int) pairs[2 * i] : NO_MORE_DOCS;
                return doc;
            }

            @Override
            public int advance(int target) {
                while (i + 1 < pairs.length / 2 && pairs[2 * (i + 1)] < target) {
                    i++;
                }
                return nextDoc();
            }

            @Override
            public long cost() {
                return pairs.length / 2;
            }
        };
    }

    @Override
    public Terms terms(String field) {
        FieldInfo fi = infos.fieldInfo(field);
        if (fi == null || fi.getIndexOptions() == org.apache.lucene.index.IndexOptions.NONE) {
            return null;
        }
        long[] stats = NativeBridge.liveFieldStats(id, fi.number);
        if (stats == null || stats[2] == 0) {
            return null;
        }
        return new LiveTerms(fi, stats);
    }

    private final class LiveTerms extends Terms {
        private final FieldInfo fi;
        private final long[] stats;

        LiveTerms(FieldInfo fi, long[] stats) {
            this.fi = fi;
            this.stats = stats;
        }

        @Override
        public TermsEnum iterator() {
            return new LiveTermsEnum(fi);
        }

        @Override
        public long size() {
            return -1;
        }

        @Override
        public long getSumTotalTermFreq() {
            return stats[0];
        }

        @Override
        public long getSumDocFreq() {
            return stats[1];
        }

        @Override
        public int getDocCount() {
            return (int) Math.min(stats[2], maxDoc);
        }

        @Override
        public boolean hasFreqs() {
            return fi.getIndexOptions().compareTo(org.apache.lucene.index.IndexOptions.DOCS_AND_FREQS) >= 0;
        }

        @Override
        public boolean hasOffsets() {
            return false;
        }

        @Override
        public boolean hasPositions() {
            return fi.getIndexOptions().compareTo(org.apache.lucene.index.IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) >= 0;
        }

        @Override
        public boolean hasPayloads() {
            return false;
        }
    }

    /** Exact lookups only: a live buffer's term dictionary is a hash, not sorted. */
    private final class LiveTermsEnum extends org.apache.lucene.index.BaseTermsEnum {
        private final FieldInfo fi;
        private BytesRef term;
        private int[] docs = new int[0];
        private int[] freqs = new int[0];

        LiveTermsEnum(FieldInfo fi) {
            this.fi = fi;
        }

        @Override
        public boolean seekExact(BytesRef text) {
            byte[] p = NativeBridge.livePostings(id, fi.number, text.bytes, text.offset, text.length, maxDoc);
            if (p == null || p.length == 0) {
                term = null;
                return false;
            }
            int n = p.length / 8;
            docs = new int[n];
            freqs = new int[n];
            ByteBuffer b = ByteBuffer.wrap(p).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) {
                docs[i] = b.getInt();
                freqs[i] = b.getInt();
            }
            term = BytesRef.deepCopyOf(text);
            return true;
        }

        @Override
        public void seekExact(BytesRef text, TermState state) {
            seekExact(text);
        }

        @Override
        public TermState termState() {
            BytesRef t = term;
            return new TermState() {
                @Override
                public void copyFrom(TermState other) {}

                @Override
                public String toString() {
                    return "live:" + t;
                }
            };
        }

        @Override
        public SeekStatus seekCeil(BytesRef text) {
            return seekExact(text) ? SeekStatus.FOUND : SeekStatus.END;
        }

        @Override
        public void seekExact(long ord) {
            throw new UnsupportedOperationException();
        }

        @Override
        public BytesRef term() {
            return term;
        }

        @Override
        public long ord() {
            throw new UnsupportedOperationException();
        }

        @Override
        public int docFreq() {
            return docs.length;
        }

        @Override
        public long totalTermFreq() {
            long t = 0;
            for (int f : freqs) {
                t += f;
            }
            return t;
        }

        @Override
        public PostingsEnum postings(PostingsEnum reuse, int flags) {
            int[] d = docs;
            int[] f = freqs;
            return new PostingsEnum() {
                int i = -1;

                @Override
                public int freq() {
                    return f[i];
                }

                @Override
                public int nextPosition() {
                    return 0;
                }

                @Override
                public int startOffset() {
                    return -1;
                }

                @Override
                public int endOffset() {
                    return -1;
                }

                @Override
                public BytesRef getPayload() {
                    return null;
                }

                @Override
                public int docID() {
                    return i < 0 ? -1 : i < d.length ? d[i] : NO_MORE_DOCS;
                }

                @Override
                public int nextDoc() {
                    i++;
                    return docID();
                }

                @Override
                public int advance(int target) {
                    do {
                        i++;
                    } while (i < d.length && d[i] < target);
                    return docID();
                }

                @Override
                public long cost() {
                    return d.length;
                }
            };
        }

        @Override
        public ImpactsEnum impacts(int flags) {
            return new SlowImpactsEnum(postings(null, flags));
        }

        @Override
        public BytesRef next() {
            // No sorted iteration in the spike: multi-term queries see no live terms.
            return null;
        }
    }

    @Override
    public NumericDocValues getNumericDocValues(String field) {
        return numeric(column(1, field));
    }

    @Override
    public BinaryDocValues getBinaryDocValues(String field) {
        return null;
    }

    @Override
    public SortedDocValues getSortedDocValues(String field) {
        return null;
    }

    @Override
    public SortedNumericDocValues getSortedNumericDocValues(String field) {
        NumericDocValues n = numeric(column(1, field));
        if (n == null) {
            return null;
        }
        return new SortedNumericDocValues() {
            @Override
            public long nextValue() throws IOException {
                return n.longValue();
            }

            @Override
            public int docValueCount() {
                return 1;
            }

            @Override
            public boolean advanceExact(int target) throws IOException {
                return n.advanceExact(target);
            }

            @Override
            public int docID() {
                return n.docID();
            }

            @Override
            public int nextDoc() throws IOException {
                return n.nextDoc();
            }

            @Override
            public int advance(int target) throws IOException {
                return n.advance(target);
            }

            @Override
            public long cost() {
                return n.cost();
            }
        };
    }

    @Override
    public SortedSetDocValues getSortedSetDocValues(String field) {
        return null;
    }

    @Override
    public NumericDocValues getNormValues(String field) {
        return numeric(column(0, field));
    }

    @Override
    public DocValuesSkipper getDocValuesSkipper(String field) {
        return null;
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) {
        return null;
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) {
        return null;
    }

    @Override
    public void searchNearestVectors(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) {}

    @Override
    public void searchNearestVectors(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) {}

    @Override
    public FieldInfos getFieldInfos() {
        return infos;
    }

    @Override
    public Bits getLiveDocs() {
        return live;
    }

    @Override
    public PointValues getPointValues(String field) {
        return null;
    }

    @Override
    public void checkIntegrity() {}

    @Override
    public LeafMetaData getMetaData() {
        return new LeafMetaData(Version.LATEST.major, Version.LATEST, null, false);
    }

    @Override
    public TermVectors termVectors() {
        return TermVectors.EMPTY;
    }

    @Override
    public int numDocs() {
        return numDocs;
    }

    @Override
    public int maxDoc() {
        return maxDoc;
    }

    @Override
    public StoredFields storedFields() {
        return new StoredFields() {
            @Override
            public void document(int docID, StoredFieldVisitor visitor) throws IOException {
                byte[] blob = NativeBridge.liveDocument(id, docID);
                if (blob == null) {
                    throw new IOException("live document " + docID + ": " + NativeBridge.lastError());
                }
                NativeStoredFieldsReader.replay(infos, blob, visitor);
            }
        };
    }

    @Override
    protected void doClose() throws IOException {
        fireClosed();
    }

    @Override
    public CacheHelper getCoreCacheHelper() {
        return cacheHelper;
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return cacheHelper;
    }

    @Override
    public String toString() {
        return "LiveLeafReader(maxDoc=" + maxDoc + ")";
    }
}
