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
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.Version;
import org.opensearch.common.lucene.index.SequentialStoredFieldsLeafReader;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * One chunk of a buffer of a {@link LiveView} (consecutive documents of it) as a leaf after the reader's, so its hits have doc IDs the fetch
 * phase reads. Only the native searches find them ({@code ffi_jvm_reader_search_live}, or any search
 * of the reader handle {@code ffi_live_view_reader} makes, whose segments match these leaves): to
 * Lucene every one of them is deleted and the leaf holds no terms, points or values, so a search
 * Lucene answers sees the reader alone, as it would without live search. Its stored fields are the
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

        /**
         * The buffered documents' doc values, read a document at a time natively: what the fetch
         * phase reads of a hit ({@code docvalue_fields}, a derived {@code _source}). To a scan the
         * leaf has none, as it has no documents Lucene sees.
         */
        private final NativeDocValues.Batch docValues = new NativeDocValues.Batch(this::readDocValues);

        private byte[] readDocValues(int doc, int[] fields) throws IOException {
            byte[][] out = new byte[1][];
            int rc = NativeBridge.liveDocValues(view.id(), buffer, start + doc, fields, out);
            if (rc != NativeBridge.OK) {
                throw new IOException("doc values of buffered document " + doc + ": " + NativeBridge.lastError());
            }
            return out[0];
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

        @Override
        public Terms terms(String field) {
            return null;
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
            if (has(field, DocValuesType.SORTED_NUMERIC) == false) {
                return null;
            }
            return new NativeDocValues.Numbers(docValues, infos.fieldInfo(field).number, DocValues::emptySortedNumeric, Integer.MAX_VALUE);
        }

        @Override
        public SortedSetDocValues getSortedSetDocValues(String field) {
            if (has(field, DocValuesType.SORTED_SET) == false) {
                return null;
            }
            return new NativeDocValues.Terms(docValues, infos.fieldInfo(field).number, DocValues::emptySortedSet, Integer.MAX_VALUE);
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
}
