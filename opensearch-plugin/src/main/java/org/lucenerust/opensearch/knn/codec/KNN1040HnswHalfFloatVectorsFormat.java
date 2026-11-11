/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.packed.DirectMonotonicReader;
import org.apache.lucene.search.AcceptDocs;

/**
 * The k-NN plugin's lucene-engine format for {@code half_float} fields: {@code
 * Lucene99HnswVectorsFormat}'s graph over {@link Flat}, which stores FLOAT32 vectors as
 * little-endian IEEE half floats ({@code Float.floatToFloat16}) and scores them decoded. The Rust
 * engine writes and reads the same files ({@code vectors.rs}'s {@code HALF_FLOAT_FORMAT_NAME}).
 */
public final class KNN1040HnswHalfFloatVectorsFormat extends KnnVectorsFormat {
    public static final String NAME = "KNN1040HnswHalfFloatVectorsFormat";

    private static final FlatVectorsFormat FLAT = new Flat();

    private final int maxConn;
    private final int beamWidth;

    public KNN1040HnswHalfFloatVectorsFormat() {
        this(Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN, Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH);
    }

    public KNN1040HnswHalfFloatVectorsFormat(int maxConn, int beamWidth) {
        super(NAME);
        if (maxConn <= 0 || maxConn > Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN) {
            throw new IllegalArgumentException(
                "maxConn must be positive and less than or equal to " + Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN
                    + "; maxConn=" + maxConn);
        }
        if (beamWidth <= 0 || beamWidth > Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH) {
            throw new IllegalArgumentException(
                "beamWidth must be positive and less than or equal to " + Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH
                    + "; beamWidth=" + beamWidth);
        }
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene99HnswVectorsWriter(state, maxConn, beamWidth, FLAT, FLAT.fieldsWriter(state), 1, null);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Lucene99HnswVectorsReader(state, FLAT.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return 16_000;
    }

    @Override
    public String toString() {
        return NAME + "(name=" + NAME + ", maxConn=" + maxConn + ", beamWidth=" + beamWidth + ")";
    }

    /** {@code KNN1040HalfFloatFlatVectorsFormat}: half-float flat vectors. */
    static final class Flat extends FlatVectorsFormat {
        static final String NAME = "KNN1040HalfFloatFlatVectorsFormat";
        static final String META_CODEC = "KNN1040HalfFloatFlatVectorsFormatMeta";
        static final String DATA_CODEC = "KNN1040HalfFloatFlatVectorsFormatData";
        static final String META_EXTENSION = "vemf";
        static final String DATA_EXTENSION = "vec";
        static final int VERSION = 0;
        static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;
        static final int ALIGNMENT = 64;

        Flat() {
            super(NAME);
        }

        @Override
        public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
            return new Writer(state);
        }

        @Override
        public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
            return new Reader(state);
        }

        @Override
        public int getMaxDimensions(String fieldName) {
            return 16_000;
        }

        @Override
        public String toString() {
            return NAME + "()";
        }
    }

    private static void encode(float[] vector, byte[] out, int dimension) {
        for (int i = 0; i < dimension; i++) {
            short h = Float.floatToFloat16(vector[i]);
            out[2 * i] = (byte) h;
            out[2 * i + 1] = (byte) (h >>> 8);
        }
    }

    /**
     * The similarity searches score with: k-NN normalizes half-float cosine vectors and scores them with
     * its FP16 cosine kernel, {@code (1 + dot) / 2}, the dot-product score.
     */
    static VectorSimilarityFunction scoring(VectorSimilarityFunction similarity) {
        return similarity == VectorSimilarityFunction.COSINE ? VectorSimilarityFunction.DOT_PRODUCT : similarity;
    }

    private static void checkFloat32(FieldInfo info) {
        if (info.getVectorEncoding() != VectorEncoding.FLOAT32) {
            throw new IllegalArgumentException(
                "FP16 flat format only supports FLOAT32 encoding, got [" + info.getVectorEncoding()
                    + "] for field [" + info.name + "]");
        }
    }

    /** {@code KNN1040HalfFloatFlatVectorsWriter}. */
    private static final class Writer extends FlatVectorsWriter {
        private final SegmentWriteState state;
        private final IndexOutput meta;
        private final IndexOutput data;
        private final List<FieldWriter> fields = new ArrayList<>();
        private boolean finished;

        Writer(SegmentWriteState state) throws IOException {
            super(DefaultFlatVectorScorer.INSTANCE);
            this.state = state;
            IndexOutput m = null;
            IndexOutput d = null;
            boolean success = false;
            try {
                m = state.directory.createOutput(
                    IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, Flat.META_EXTENSION),
                    state.context);
                d = state.directory.createOutput(
                    IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, Flat.DATA_EXTENSION),
                    state.context);
                CodecUtil.writeIndexHeader(m, Flat.META_CODEC, Flat.VERSION, state.segmentInfo.getId(), state.segmentSuffix);
                CodecUtil.writeIndexHeader(d, Flat.DATA_CODEC, Flat.VERSION, state.segmentInfo.getId(), state.segmentSuffix);
                success = true;
            } finally {
                if (!success) {
                    IOUtils.closeWhileHandlingException(m, d);
                }
            }
            this.meta = m;
            this.data = d;
        }

        @Override
        public FlatFieldVectorsWriter<?> addField(FieldInfo info) {
            checkFloat32(info);
            FieldWriter w = new FieldWriter(info);
            fields.add(w);
            return w;
        }

        @Override
        public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
            for (FieldWriter field : fields) {
                int dimension = field.info.getVectorDimension();
                byte[] buffer = new byte[dimension * Short.BYTES];
                long offset = data.alignFilePointer(Flat.ALIGNMENT);
                DocsWithFieldSet docs;
                if (sortMap == null) {
                    for (float[] v : field.vectors) {
                        encode(v, buffer, dimension);
                        data.writeBytes(buffer, buffer.length);
                    }
                    docs = field.docs;
                } else {
                    int[] ordMap = new int[field.docs.cardinality()];
                    docs = new DocsWithFieldSet();
                    mapOldOrdToNewOrd(field.docs, sortMap, null, ordMap, docs);
                    for (int ord : ordMap) {
                        encode(field.vectors.get(ord), buffer, dimension);
                        data.writeBytes(buffer, buffer.length);
                    }
                }
                writeMeta(field.info, maxDoc, offset, data.getFilePointer() - offset, docs);
                field.finish();
            }
        }

        @Override
        public void mergeOneFlatVectorField(FieldInfo info, MergeState mergeState) throws IOException {
            checkFloat32(info);
            FloatVectorValues merged = KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(info, mergeState);
            int dimension = info.getVectorDimension();
            byte[] buffer = new byte[dimension * Short.BYTES];
            long offset = data.alignFilePointer(Flat.ALIGNMENT);
            DocsWithFieldSet docs = new DocsWithFieldSet();
            KnnVectorValues.DocIndexIterator it = merged.iterator();
            for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                encode(merged.vectorValue(it.index()), buffer, dimension);
                data.writeBytes(buffer, buffer.length);
                docs.add(doc);
            }
            writeMeta(info, state.segmentInfo.maxDoc(), offset, data.getFilePointer() - offset, docs);
        }

        private void writeMeta(FieldInfo info, int maxDoc, long offset, long length, DocsWithFieldSet docs)
            throws IOException {
            meta.writeInt(info.number);
            meta.writeInt(info.getVectorEncoding().ordinal());
            meta.writeVLong(offset);
            meta.writeVLong(length);
            meta.writeVInt(info.getVectorDimension());
            int count = docs.cardinality();
            meta.writeInt(count);
            OrdToDocDISIReaderConfiguration.writeStoredMeta(
                Flat.DIRECT_MONOTONIC_BLOCK_SHIFT, meta, data, count, maxDoc, docs);
        }

        @Override
        public void finish() throws IOException {
            if (finished) {
                throw new IllegalStateException("already finished");
            }
            finished = true;
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
            CodecUtil.writeFooter(data);
        }

        @Override
        public long ramBytesUsed() {
            long total = 0;
            for (FieldWriter f : fields) {
                total += f.ramBytesUsed();
            }
            return total;
        }

        @Override
        public void close() throws IOException {
            IOUtils.close(meta, data);
        }
    }

    /** The vectors of one field being flushed, on heap. */
    private static final class FieldWriter extends FlatFieldVectorsWriter<float[]> {
        final FieldInfo info;
        final List<float[]> vectors = new ArrayList<>();
        final DocsWithFieldSet docs = new DocsWithFieldSet();
        private int lastDoc = -1;
        private boolean finished;

        FieldWriter(FieldInfo info) {
            this.info = info;
        }

        @Override
        public void addValue(int doc, float[] value) {
            if (finished) {
                throw new IllegalStateException("already finished");
            }
            if (doc == lastDoc) {
                throw new IllegalArgumentException(
                    "VectorValuesField \"" + info.name + "\" appears more than once in this document");
            }
            docs.add(doc);
            vectors.add(copyValue(value));
            lastDoc = doc;
        }

        @Override
        public float[] copyValue(float[] value) {
            return ArrayUtil.copyOfSubArray(value, 0, info.getVectorDimension());
        }

        @Override
        public List<float[]> getVectors() {
            return vectors;
        }

        @Override
        public DocsWithFieldSet getDocsWithFieldSet() {
            return docs;
        }

        @Override
        public void finish() {
            finished = true;
        }

        @Override
        public boolean isFinished() {
            return finished;
        }

        @Override
        public long ramBytesUsed() {
            return docs.ramBytesUsed()
                + (long) vectors.size() * (RamUsageEstimator.NUM_BYTES_OBJECT_REF + RamUsageEstimator.NUM_BYTES_ARRAY_HEADER
                    + (long) info.getVectorDimension() * Float.BYTES);
        }
    }

    private record FieldEntry(
        FieldInfo info, long offset, long length, int dimension, int size, OrdToDocDISIReaderConfiguration ordToDoc) {
    }

    /** {@code KNN1040HalfFloatFlatVectorsReader}: each field's similarity comes from its FieldInfo. */
    private static final class Reader extends FlatVectorsReader {
        private final Map<Integer, FieldEntry> fields = new HashMap<>();
        private final FieldInfos fieldInfos;
        private final IndexInput data;

        Reader(SegmentReadState state) throws IOException {
            super();
            this.fieldInfos = state.fieldInfos;
            int metaVersion;
            String metaName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, Flat.META_EXTENSION);
            try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaName)) {
                Throwable prior = null;
                metaVersion = -1;
                try {
                    metaVersion = CodecUtil.checkIndexHeader(
                        meta, Flat.META_CODEC, Flat.VERSION, Flat.VERSION, state.segmentInfo.getId(), state.segmentSuffix);
                    readFields(meta);
                } catch (Throwable t) {
                    prior = t;
                } finally {
                    CodecUtil.checkFooter(meta, prior);
                }
            }
            String dataName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, Flat.DATA_EXTENSION);
            IndexInput in = state.directory.openInput(dataName, state.context);
            boolean success = false;
            try {
                int dataVersion = CodecUtil.checkIndexHeader(
                    in, Flat.DATA_CODEC, Flat.VERSION, Flat.VERSION, state.segmentInfo.getId(), state.segmentSuffix);
                if (dataVersion != metaVersion) {
                    throw new CorruptIndexException(
                        "Format versions mismatch: meta=" + metaVersion + ", " + Flat.DATA_CODEC + "=" + dataVersion, in);
                }
                CodecUtil.retrieveChecksum(in);
                success = true;
            } finally {
                if (!success) {
                    IOUtils.closeWhileHandlingException(in);
                }
            }
            this.data = in;
        }

        private void readFields(ChecksumIndexInput meta) throws IOException {
            for (int number = meta.readInt(); number != -1; number = meta.readInt()) {
                FieldInfo info = fieldInfos.fieldInfo(number);
                if (info == null) {
                    throw new CorruptIndexException("Invalid field number: " + number, meta);
                }
                int encoding = meta.readInt();
                if (encoding != VectorEncoding.FLOAT32.ordinal()) {
                    throw new CorruptIndexException("Invalid vector encoding id: " + encoding, meta);
                }
                long offset = meta.readVLong();
                long length = meta.readVLong();
                int dimension = meta.readVInt();
                int size = meta.readInt();
                OrdToDocDISIReaderConfiguration ordToDoc = OrdToDocDISIReaderConfiguration.fromStoredMeta(meta, size);
                if (dimension != info.getVectorDimension()) {
                    throw new CorruptIndexException(
                        "Inconsistent vector dimension for field=\"" + info.name + "\"; " + info.getVectorDimension()
                            + " != " + dimension, meta);
                }
                long expected = Math.multiplyExact(Math.multiplyExact((long) dimension, Short.BYTES), size);
                if (expected != length) {
                    throw new CorruptIndexException(
                        "Vector data length " + length + " not matching size=" + size + " * dim=" + dimension
                            + " * byteSize=2 = " + expected, meta);
                }
                fields.put(number, new FieldEntry(info, offset, length, dimension, size, ordToDoc));
            }
        }

        private FieldEntry entry(String field) {
            FieldInfo info = fieldInfos.fieldInfo(field);
            FieldEntry entry = info == null ? null : fields.get(info.number);
            if (entry == null) {
                throw new IllegalArgumentException("field=\"" + field + "\" not found");
            }
            return entry;
        }

        @Override
        public FloatVectorValues getFloatVectorValues(String field) throws IOException {
            FieldEntry e = entry(field);
            IndexInput slice = data.slice("half-float-vectors", e.offset(), e.length());
            DirectMonotonicReader ordToDoc = e.ordToDoc().isDense() || e.ordToDoc().isEmpty()
                ? null
                : e.ordToDoc().getDirectMonotonicReader(data);
            return new HalfFloatValues(e.dimension(), e.size(), slice, ordToDoc, e.info().getVectorSimilarityFunction());
        }

        @Override
        public ByteVectorValues getByteVectorValues(String field) {
            throw new UnsupportedOperationException("FP16 format does not support byte vectors");
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
            FloatVectorValues values = getFloatVectorValues(field);
            return DefaultFlatVectorScorer.INSTANCE.getRandomVectorScorer(
                scoring(entry(field).info().getVectorSimilarityFunction()), values, target);
        }

        @Override
        public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) {
            throw new UnsupportedOperationException("FP16 format does not support byte vector scoring");
        }

        @Override
        public void search(String field, float[] target, KnnCollector collector, AcceptDocs acceptDocs)
            throws IOException {
            RandomVectorScorer scorer = getRandomVectorScorer(field, target);
            Bits accepted = scorer.getAcceptOrds(acceptDocs.bits());
            for (int ord = 0; ord < scorer.maxOrd(); ord++) {
                if (collector.earlyTerminated()) {
                    break;
                }
                if (accepted == null || accepted.get(ord)) {
                    collector.incVisitedCount(1);
                    collector.collect(scorer.ordToDoc(ord), scorer.score(ord));
                }
            }
        }

        @Override
        public FlatVectorsScorer getFlatVectorScorer(String field) {
            entry(field);
            return DefaultFlatVectorScorer.INSTANCE;
        }

        @Override
        public void checkIntegrity() throws IOException {
            CodecUtil.checksumEntireFile(data);
        }

        @Override
        public Map<String, Long> getOffHeapByteSize(FieldInfo info) {
            return Map.of(Flat.DATA_EXTENSION, entry(info.name).length());
        }

        @Override
        public long ramBytesUsed() {
            return RamUsageEstimator.shallowSizeOfInstance(Reader.class) + 64L * fields.size();
        }

        @Override
        public void close() throws IOException {
            IOUtils.close(data);
        }
    }

    /** One field's half-float vectors, decoded per read. */
    private static final class HalfFloatValues extends FloatVectorValues {
        private final int dimension;
        private final int size;
        private final IndexInput slice;
        private final DirectMonotonicReader ordToDoc;
        private final VectorSimilarityFunction similarity;
        private final byte[] raw;
        private final float[] value;
        private int lastOrd = -1;

        HalfFloatValues(
            int dimension, int size, IndexInput slice, DirectMonotonicReader ordToDoc, VectorSimilarityFunction similarity) {
            this.dimension = dimension;
            this.size = size;
            this.slice = slice;
            this.ordToDoc = ordToDoc;
            this.similarity = scoring(similarity);
            this.raw = new byte[dimension * Short.BYTES];
            this.value = new float[dimension];
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int getVectorByteLength() {
            return raw.length;
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            if (ord == lastOrd) {
                return value;
            }
            slice.seek((long) ord * raw.length);
            slice.readBytes(raw, 0, raw.length);
            for (int i = 0; i < dimension; i++) {
                value[i] = Float.float16ToFloat((short) ((raw[2 * i] & 0xff) | (raw[2 * i + 1] << 8)));
            }
            lastOrd = ord;
            return value;
        }

        @Override
        public int ordToDoc(int ord) {
            return ordToDoc == null ? ord : (int) ordToDoc.get(ord);
        }

        @Override
        public DocIndexIterator iterator() {
            return ordToDoc == null ? createDenseIterator() : createSparseIterator();
        }

        @Override
        public HalfFloatValues copy() {
            return new HalfFloatValues(dimension, size, slice.clone(), ordToDoc, similarity);
        }

        @Override
        public VectorScorer scorer(float[] target) throws IOException {
            if (size == 0) {
                return null;
            }
            HalfFloatValues copy = copy();
            DocIndexIterator it = copy.iterator();
            return new VectorScorer() {
                @Override
                public float score() throws IOException {
                    return similarity.compare(target, copy.vectorValue(it.index()));
                }

                @Override
                public DocIdSetIterator iterator() {
                    return it;
                }
            };
        }
    }
}
