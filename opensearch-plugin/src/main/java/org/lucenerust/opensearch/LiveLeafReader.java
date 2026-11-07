/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafMetaData;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.TermVectors;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;
import org.opensearch.common.lucene.index.SequentialStoredFieldsLeafReader;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * One chunk of a buffer of a {@link LiveView} (consecutive documents of it) as a leaf after the reader's, so its hits have doc IDs the fetch
 * phase reads. Only the native searches find them ({@code ffi_jvm_reader_search_live}, or any search
 * of the reader handle {@code ffi_live_view_reader} makes, whose segments match these leaves): to
 * Lucene every one of them is deleted and the leaf holds no terms, points or values (only bounds that
 * take in the buffered documents', so OpenSearch never skips the shard or rewrites a query to
 * match nothing for the reader's bounds), so a search Lucene answers sees the reader alone, as it would without live
 * search. Its stored fields are the
 * buffered documents', read through the view.
 */
public final class LiveLeafReader extends SequentialStoredFieldsLeafReader {
    private final Core core;

    LiveLeafReader(LiveView view, int buffer, int start, int maxDoc, FieldInfos infos) {
        this(new Core(view, buffer, start, maxDoc, infos));
    }

    private LiveLeafReader(Core core) {
        super(core);
        this.core = core;
    }

    /** The view the leaf reads. */
    public LiveView view() {
        return core.view;
    }

    /** The view's buffer the leaf holds. */
    public int buffer() {
        return core.buffer;
    }

    /** {@code leaf} itself, or the leaf of buffered documents under its wrappers; null for any other leaf. */
    public static LiveLeafReader unwrap(LeafReader leaf) {
        while (true) {
            if (leaf instanceof LiveLeafReader live) {
                return live;
            }
            if (leaf instanceof org.apache.lucene.index.FilterLeafReader f) {
                leaf = f.getDelegate();
            } else {
                return null;
            }
        }
    }

    @Override
    public StoredFieldsReader getSequentialStoredFieldsReader() {
        return new Sequential();
    }

    @Override
    protected StoredFieldsReader doGetSequentialStoredFieldsReader(StoredFieldsReader reader) {
        return new Sequential();
    }

    @Override
    public CacheHelper getCoreCacheHelper() {
        return core.cacheHelper;
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return core.cacheHelper;
    }

    @Override
    public String toString() {
        return "LiveLeafReader(maxDoc=" + maxDoc() + ")";
    }

    /** The sequential reader: the same reads, in whatever order they come. */
    private final class Sequential extends StoredFieldsReader {
        @Override
        public void document(int docID, StoredFieldVisitor visitor) throws IOException {
            core.document(docID, visitor);
        }

        @Override
        public StoredFieldsReader clone() {
            return new Sequential();
        }

        @Override
        public void checkIntegrity() {}

        @Override
        public void close() {}
    }

    /** What the leaf holds: the view's documents, deleted to Lucene, with their stored fields. */
    private static final class Core extends LeafReader {
        private final LiveView view;
        private final int buffer;
        /** The leaf's first document, in its buffer. */
        private final int start;
        private final int maxDoc;
        private final FieldInfos infos;
        private final Bits none;
        /** By field: its {@link #getPointValues}, once read. */
        private final Map<String, Optional<PointValues>> points = new ConcurrentHashMap<>();
        private final IndexReader.CacheKey key = CacheKeys.newKey();
        private final List<ClosedListener> listeners = new CopyOnWriteArrayList<>();
        private final CacheHelper cacheHelper = new CacheHelper() {
            @Override
            public CacheKey getKey() {
                return key;
            }

            @Override
            public void addClosedListener(ClosedListener listener) {
                listeners.add(listener);
            }
        };

        Core(LiveView view, int buffer, int start, int maxDoc, FieldInfos infos) {
            this.view = view;
            this.buffer = buffer;
            this.start = start;
            this.maxDoc = maxDoc;
            this.infos = infos;
            this.none = new Bits.MatchNoBits(maxDoc);
        }

        /**
         * Whether {@code field} declares doc values of {@code type}: Lucene takes a reader that
         * declares them and has none as a broken index, so the leaf answers with empty ones.
         */
        private boolean has(String field, DocValuesType type) {
            FieldInfo fi = infos.fieldInfo(field);
            return fi != null && fi.getDocValuesType() == type;
        }

        void document(int docID, StoredFieldVisitor visitor) throws IOException {
            byte[][] out = new byte[1][];
            int rc = NativeBridge.liveDocument(view.id(), buffer, start + docID, out);
            if (rc != NativeBridge.OK) {
                throw new IOException("buffered document " + docID + ": " + NativeBridge.lastError());
            }
            NativeStoredFieldsReader.replay(infos, out[0], visitor);
        }

        @Override
        public StoredFields storedFields() {
            return new StoredFields() {
                @Override
                public void document(int docID, StoredFieldVisitor visitor) throws IOException {
                    Core.this.document(docID, visitor);
                }
            };
        }

        /**
         * No terms, bounded by the least and a greater than any term (no UTF-8 term holds byte
         * {@code 0xFF}), so OpenSearch never skips the shard by its terms' bounds either.
         */
        @Override
        public Terms terms(String field) {
            FieldInfo fi = infos.fieldInfo(field);
            if (fi == null || fi.getIndexOptions() == IndexOptions.NONE || maxDoc == 0) {
                return null;
            }
            return new AnyTerms(fi);
        }

        @Override
        public NumericDocValues getNumericDocValues(String field) {
            return has(field, DocValuesType.NUMERIC) ? DocValues.emptyNumeric() : null;
        }

        @Override
        public BinaryDocValues getBinaryDocValues(String field) {
            return has(field, DocValuesType.BINARY) ? DocValues.emptyBinary() : null;
        }

        @Override
        public SortedDocValues getSortedDocValues(String field) {
            return has(field, DocValuesType.SORTED) ? DocValues.emptySorted() : null;
        }

        @Override
        public SortedNumericDocValues getSortedNumericDocValues(String field) {
            return has(field, DocValuesType.SORTED_NUMERIC) ? DocValues.emptySortedNumeric() : null;
        }

        @Override
        public SortedSetDocValues getSortedSetDocValues(String field) {
            return has(field, DocValuesType.SORTED_SET) ? DocValues.emptySortedSet() : null;
        }

        @Override
        public NumericDocValues getNormValues(String field) {
            return null;
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
            return none;
        }

        /**
         * The buffer's points of {@code field} as bounds without points (the leaf's documents are
         * deleted): the least and greatest values of every document the buffer holds, so the
         * bounds OpenSearch reads to skip a shard (a sort's competitive range, a range query
         * rewritten to match nothing) never leave its buffered documents out.
         */
        @Override
        public PointValues getPointValues(String field) throws IOException {
            FieldInfo fi = infos.fieldInfo(field);
            if (fi == null || fi.getPointDimensionCount() == 0 || maxDoc == 0) {
                return null;
            }
            Optional<PointValues> got = points.get(field);
            if (got == null) {
                got = Optional.ofNullable(bounds(fi));
                points.put(field, got);
            }
            return got.orElse(null);
        }

        private PointValues bounds(FieldInfo fi) throws IOException {
            byte[][] out = new byte[1][];
            int rc = NativeBridge.livePointBounds(view.id(), buffer, fi.number, out);
            if (rc != NativeBridge.OK) {
                throw new IOException("buffered points of " + fi.name + ": " + NativeBridge.lastError());
            }
            byte[] b = out[0];
            int packed = fi.getPointIndexDimensionCount() * fi.getPointNumBytes();
            if (b.length == 0) {
                return null;
            }
            if (b.length != Long.BYTES + 2 * packed) {
                throw new IOException("buffered points of " + fi.name + ": bounds of " + b.length + " bytes");
            }
            long count = ByteBuffer.wrap(b, 0, Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).getLong();
            byte[] min = Arrays.copyOfRange(b, Long.BYTES, Long.BYTES + packed);
            byte[] max = Arrays.copyOfRange(b, Long.BYTES + packed, b.length);
            return new BoundsOnly(fi, min, max, count, (int) Math.min(count, maxDoc));
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
            return 0;
        }

        @Override
        public int maxDoc() {
            return maxDoc;
        }

        @Override
        protected void doClose() throws IOException {
            for (ClosedListener l : listeners) {
                l.onClose(key);
            }
        }

        @Override
        public CacheHelper getCoreCacheHelper() {
            return cacheHelper;
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            return cacheHelper;
        }
    }


    /** A field's terms: none, bounded as {@link Core#terms} describes. */
    private static final class AnyTerms extends Terms {
        private static final BytesRef MIN = new BytesRef();
        private static final BytesRef MAX = new BytesRef(new byte[] { (byte) 0xFF });
        private final FieldInfo fi;

        AnyTerms(FieldInfo fi) {
            this.fi = fi;
        }

        @Override
        public TermsEnum iterator() {
            return TermsEnum.EMPTY;
        }

        @Override
        public BytesRef getMin() {
            return MIN;
        }

        @Override
        public BytesRef getMax() {
            return MAX;
        }

        @Override
        public long size() {
            return 0;
        }

        @Override
        public long getSumTotalTermFreq() {
            return 0;
        }

        @Override
        public long getSumDocFreq() {
            return 0;
        }

        @Override
        public int getDocCount() {
            return 0;
        }

        @Override
        public boolean hasFreqs() {
            return fi.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS) >= 0;
        }

        @Override
        public boolean hasOffsets() {
            return fi.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS) >= 0;
        }

        @Override
        public boolean hasPositions() {
            return fi.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) >= 0;
        }

        @Override
        public boolean hasPayloads() {
            return fi.hasPayloads();
        }
    }
    /** A field's points as one cell between bounds, counting {@code size} points, that visits none. */
    private static final class BoundsOnly extends PointValues {
        private final int dims;
        private final int indexDims;
        private final int bytesPerDim;
        private final byte[] min;
        private final byte[] max;
        private final long size;
        private final int docs;

        BoundsOnly(FieldInfo fi, byte[] min, byte[] max, long size, int docs) {
            this.dims = fi.getPointDimensionCount();
            this.indexDims = fi.getPointIndexDimensionCount();
            this.bytesPerDim = fi.getPointNumBytes();
            this.min = min;
            this.max = max;
            this.size = size;
            this.docs = docs;
        }

        @Override
        public PointTree getPointTree() {
            return new PointTree() {
                @Override
                public PointTree clone() {
                    return this;
                }

                @Override
                public boolean moveToChild() {
                    return false;
                }

                @Override
                public boolean moveToSibling() {
                    return false;
                }

                @Override
                public boolean moveToParent() {
                    return false;
                }

                @Override
                public byte[] getMinPackedValue() {
                    return min.clone();
                }

                @Override
                public byte[] getMaxPackedValue() {
                    return max.clone();
                }

                @Override
                public long size() {
                    return size;
                }

                @Override
                public void visitDocIDs(IntersectVisitor visitor) {}

                @Override
                public void visitDocValues(IntersectVisitor visitor) {}
            };
        }

        @Override
        public byte[] getMinPackedValue() {
            return min.clone();
        }

        @Override
        public byte[] getMaxPackedValue() {
            return max.clone();
        }

        @Override
        public int getNumDimensions() {
            return dims;
        }

        @Override
        public int getNumIndexDimensions() {
            return indexDims;
        }

        @Override
        public int getBytesPerDimension() {
            return bytesPerDim;
        }

        @Override
        public long size() {
            return size;
        }

        @Override
        public int getDocCount() {
            return docs;
        }
    }
}
