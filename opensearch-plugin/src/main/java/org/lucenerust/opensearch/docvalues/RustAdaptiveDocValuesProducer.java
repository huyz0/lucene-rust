/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.docvalues;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.packed.DirectMonotonicReader;

/**
 * {@link RustAdaptiveDocValuesFormat}'s reader: Lucene90's producer for everything, except the
 * NUMERIC and SORTED_NUMERIC fields whose value array is adaptive blocks, which this class decodes.
 *
 * <p>Lucene90's producer parses the whole {@code .dvm} -- an adaptive entry keeps Lucene90's shape,
 * with a {@code tableSize} of {@value #ADAPTIVE_TABLE_SIZE}, or {@value #ADAPTIVE_ZONES_TABLE_SIZE} when
 * per-1024-value statistics follow the block descriptor (which only the native aggregations read)
 * -- and never touches those value arrays,
 * since every read of an adaptive field comes here. This class walks the same {@code .dvm} to find
 * them ({@link #readFields}), mirroring {@code Lucene90DocValuesProducer.readFields}.
 */
final class RustAdaptiveDocValuesProducer extends DocValuesProducer {
    /** {@code ADAPTIVE_TABLE_SIZE} in {@code doc_values.rs}. */
    static final int ADAPTIVE_TABLE_SIZE = -1000;
    /** {@code ADAPTIVE_ZONES_TABLE_SIZE} in {@code doc_values.rs}. */
    static final int ADAPTIVE_ZONES_TABLE_SIZE = -1001;

    private static final String META_CODEC = "Lucene90DocValuesMetadata";
    private static final String DATA_CODEC = "Lucene90DocValuesData";
    private static final int VERSION_START = 0;
    private static final int VERSION_SKIPPER_MAX_VALUE_COUNT = 2;
    private static final int VERSION_CURRENT = VERSION_SKIPPER_MAX_VALUE_COUNT;
    private static final byte NUMERIC = 0;
    private static final byte BINARY = 1;
    private static final byte SORTED = 2;
    private static final byte SORTED_SET = 3;
    private static final byte SORTED_NUMERIC = 4;
    private static final int TERMS_DICT_BLOCK_LZ4_SHIFT = 6;

    /** An adaptive value array, and for SORTED_NUMERIC its per-document addresses. */
    static final class Entry {
        long docsWithFieldOffset;
        long docsWithFieldLength;
        short jumpTableEntryCount;
        byte denseRankPower;
        long numValues;
        long valuesOffset;
        long valuesLength;
        long descriptorOffset;
        boolean adaptive;
        // SORTED_NUMERIC only.
        int numDocsWithField = -1;
        long addressesOffset;
        long addressesLength;
        DirectMonotonicReader.Meta addressesMeta;
        // The block addresses, from the descriptor in .dvd.
        DirectMonotonicReader.Meta blocksMeta;
        long blocksOffset;
        long blocksLength;
    }

    private final DocValuesProducer delegate;
    private final Map<Integer, Entry> numerics;
    private final Map<Integer, Entry> sortedNumerics;
    private final IndexInput data;
    private final int maxDoc;
    private final boolean owner;

    RustAdaptiveDocValuesProducer(SegmentReadState state, DocValuesProducer delegate) throws IOException {
        this.delegate = delegate;
        this.maxDoc = state.segmentInfo.maxDoc();
        this.numerics = new HashMap<>();
        this.sortedNumerics = new HashMap<>();
        this.owner = true;
        String metaName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "dvm");
        // Lucene90's producer has verified this file's checksum already.
        try (IndexInput meta = state.directory.openInput(metaName, IOContext.READONCE)) {
            int version = CodecUtil.checkIndexHeader(
                meta, META_CODEC, VERSION_START, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            readFields(meta, state, version);
        }
        if (numerics.isEmpty() && sortedNumerics.isEmpty()) {
            this.data = null;
            return;
        }
        String dataName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "dvd");
        IndexInput in = state.directory.openInput(dataName, state.context);
        try {
            CodecUtil.checkIndexHeader(
                in, DATA_CODEC, VERSION_START, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
            for (Entry e : numerics.values()) {
                readDescriptor(in, e);
            }
            for (Entry e : sortedNumerics.values()) {
                readDescriptor(in, e);
            }
        } catch (Throwable t) {
            IOUtils.closeWhileSuppressingExceptions(t, in);
            throw t;
        }
        this.data = in;
    }

    private RustAdaptiveDocValuesProducer(RustAdaptiveDocValuesProducer of, DocValuesProducer delegate) {
        this.delegate = delegate;
        this.numerics = of.numerics;
        this.sortedNumerics = of.sortedNumerics;
        this.data = of.data;
        this.maxDoc = of.maxDoc;
        this.owner = false;
    }

    private void readFields(IndexInput meta, SegmentReadState state, int version) throws IOException {
        for (int number = meta.readInt(); number != -1; number = meta.readInt()) {
            FieldInfo info = state.fieldInfos.fieldInfo(number);
            if (info == null) {
                throw new CorruptIndexException("Invalid field number: " + number, meta);
            }
            byte type = meta.readByte();
            if (info.docValuesSkipIndexType() != DocValuesSkipIndexType.NONE) {
                // offset, length, maxValue, minValue, docCount, maxDocID[, maxValueCount]
                meta.skipBytes(8 * 4 + 4 * 2 + (version >= VERSION_SKIPPER_MAX_VALUE_COUNT ? 4 : 0));
            }
            switch (type) {
                case NUMERIC -> {
                    Entry e = readNumeric(meta);
                    if (e.adaptive) {
                        numerics.put(number, e);
                    }
                }
                case BINARY -> skipBinary(meta);
                case SORTED -> skipSorted(meta);
                case SORTED_SET -> {
                    byte multiValued = meta.readByte();
                    if (multiValued == 0) {
                        skipSorted(meta);
                    } else if (multiValued == 1) {
                        if (readSortedNumeric(meta).adaptive) {
                            throw new CorruptIndexException("adaptive values for ordinals", meta);
                        }
                        skipTermDict(meta);
                    } else {
                        throw new CorruptIndexException("Invalid multiValued flag: " + multiValued, meta);
                    }
                }
                case SORTED_NUMERIC -> {
                    Entry e = readSortedNumeric(meta);
                    if (e.adaptive) {
                        sortedNumerics.put(number, e);
                    }
                }
                default -> throw new CorruptIndexException("invalid type: " + type, meta);
            }
        }
    }

    /** {@code Lucene90DocValuesProducer.readNumeric}, keeping what an adaptive entry needs. */
    private static Entry readNumeric(IndexInput meta) throws IOException {
        Entry e = new Entry();
        e.docsWithFieldOffset = meta.readLong();
        e.docsWithFieldLength = meta.readLong();
        e.jumpTableEntryCount = meta.readShort();
        e.denseRankPower = meta.readByte();
        e.numValues = meta.readLong();
        int tableSize = meta.readInt();
        if (tableSize > 256) {
            throw new CorruptIndexException("invalid table size: " + tableSize, meta);
        }
        if (tableSize > 0) {
            meta.skipBytes(8L * tableSize);
        }
        meta.readByte(); // bitsPerValue
        meta.readLong(); // minValue
        meta.readLong(); // gcd
        e.valuesOffset = meta.readLong();
        e.valuesLength = meta.readLong();
        e.descriptorOffset = meta.readLong();
        e.adaptive = tableSize == ADAPTIVE_TABLE_SIZE || tableSize == ADAPTIVE_ZONES_TABLE_SIZE;
        return e;
    }

    /** {@code Lucene90DocValuesProducer.readSortedNumeric}. */
    private static Entry readSortedNumeric(IndexInput meta) throws IOException {
        Entry e = readNumeric(meta);
        e.numDocsWithField = meta.readInt();
        if (e.numDocsWithField != e.numValues) {
            e.addressesOffset = meta.readLong();
            int blockShift = meta.readVInt();
            e.addressesMeta = DirectMonotonicReader.loadMeta(meta, e.numDocsWithField + 1L, blockShift);
            e.addressesLength = meta.readLong();
        }
        return e;
    }

    private static void skipBinary(IndexInput meta) throws IOException {
        meta.skipBytes(8 * 4 + 2 + 1); // data, docsWithField offsets/lengths, jump table, rank power
        int numDocsWithField = meta.readInt();
        int minLength = meta.readInt();
        int maxLength = meta.readInt();
        if (minLength < maxLength) {
            meta.readLong();
            int blockShift = meta.readVInt();
            DirectMonotonicReader.loadMeta(meta, numDocsWithField + 1L, blockShift);
            meta.readLong();
        }
    }

    private static void skipSorted(IndexInput meta) throws IOException {
        if (readNumeric(meta).adaptive) {
            throw new CorruptIndexException("adaptive values for ordinals", meta);
        }
        skipTermDict(meta);
    }

    private static void skipTermDict(IndexInput meta) throws IOException {
        long termsDictSize = meta.readVLong();
        int blockShift = meta.readInt();
        long addressesSize = (termsDictSize + (1L << TERMS_DICT_BLOCK_LZ4_SHIFT) - 1) >>> TERMS_DICT_BLOCK_LZ4_SHIFT;
        DirectMonotonicReader.loadMeta(meta, addressesSize, blockShift);
        meta.readInt(); // maxTermLength
        meta.readInt(); // maxBlockLength
        meta.skipBytes(8 * 4);
        int termsDictIndexShift = meta.readInt();
        long indexSize = (termsDictSize + (1L << termsDictIndexShift) - 1) >>> termsDictIndexShift;
        DirectMonotonicReader.loadMeta(meta, 1 + indexSize, blockShift);
        meta.skipBytes(8 * 4);
    }

    private static void readDescriptor(IndexInput data, Entry e) throws IOException {
        data.seek(e.descriptorOffset);
        int blockShift = data.readVInt();
        long numBlocks = (e.numValues + AdaptiveBlocks.BLOCK_SIZE - 1) >>> AdaptiveBlocks.BLOCK_SHIFT;
        e.blocksMeta = DirectMonotonicReader.loadMeta(data, numBlocks + 1, blockShift);
        e.blocksOffset = data.readLong();
        e.blocksLength = data.readLong();
    }

    private AdaptiveBlocks blocks(Entry e) throws IOException {
        RandomAccessInput values = data.randomAccessSlice(e.valuesOffset, e.valuesLength);
        RandomAccessInput addresses = data.randomAccessSlice(e.blocksOffset, e.blocksLength);
        return new AdaptiveBlocks(values, DirectMonotonicReader.getInstance(e.blocksMeta, addresses), e.numValues);
    }

    @Override
    public NumericDocValues getNumeric(FieldInfo field) throws IOException {
        Entry e = numerics.get(field.number);
        return e == null ? delegate.getNumeric(field) : numeric(e);
    }

    private NumericDocValues numeric(Entry e) throws IOException {
        if (e.docsWithFieldOffset == -2) {
            return DocValues.emptyNumeric();
        }
        AdaptiveBlocks blocks = blocks(e);
        if (e.docsWithFieldOffset == -1) {
            return new DenseNumeric(maxDoc, blocks);
        }
        IndexedDISI disi = new IndexedDISI(
            data, e.docsWithFieldOffset, e.docsWithFieldLength, e.jumpTableEntryCount, e.denseRankPower, e.numValues);
        return new SparseNumeric(disi, blocks);
    }

    @Override
    public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
        Entry e = sortedNumerics.get(field.number);
        if (e == null) {
            return delegate.getSortedNumeric(field);
        }
        if (e.numValues == e.numDocsWithField) {
            return DocValues.singleton(numeric(e));
        }
        RandomAccessInput addressesInput = data.randomAccessSlice(e.addressesOffset, e.addressesLength);
        LongValues addresses = DirectMonotonicReader.getInstance(e.addressesMeta, addressesInput);
        AdaptiveBlocks blocks = blocks(e);
        if (e.docsWithFieldOffset == -1) {
            return new DenseSortedNumeric(maxDoc, addresses, blocks);
        }
        IndexedDISI disi = new IndexedDISI(
            data, e.docsWithFieldOffset, e.docsWithFieldLength, e.jumpTableEntryCount, e.denseRankPower, e.numDocsWithField);
        return new SparseSortedNumeric(disi, addresses, blocks);
    }

    @Override
    public BinaryDocValues getBinary(FieldInfo field) throws IOException {
        return delegate.getBinary(field);
    }

    @Override
    public SortedDocValues getSorted(FieldInfo field) throws IOException {
        return delegate.getSorted(field);
    }

    @Override
    public SortedSetDocValues getSortedSet(FieldInfo field) throws IOException {
        return delegate.getSortedSet(field);
    }

    @Override
    public DocValuesSkipper getSkipper(FieldInfo field) throws IOException {
        return delegate.getSkipper(field);
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegate.checkIntegrity();
    }

    @Override
    public DocValuesProducer getMergeInstance() {
        return new RustAdaptiveDocValuesProducer(this, delegate.getMergeInstance());
    }

    @Override
    public void close() throws IOException {
        if (owner) {
            IOUtils.close(delegate, data);
        }
    }

    private static final class DenseNumeric extends NumericDocValues {
        private final int maxDoc;
        private final AdaptiveBlocks blocks;
        private int doc = -1;

        DenseNumeric(int maxDoc, AdaptiveBlocks blocks) {
            this.maxDoc = maxDoc;
            this.blocks = blocks;
        }

        @Override
        public long longValue() throws IOException {
            return blocks.get(doc);
        }

        @Override
        public boolean advanceExact(int target) {
            doc = target;
            return true;
        }

        @Override
        public int docID() {
            return doc;
        }

        @Override
        public int nextDoc() {
            return advance(doc + 1);
        }

        @Override
        public int advance(int target) {
            return doc = target >= maxDoc ? NO_MORE_DOCS : target;
        }

        @Override
        public long cost() {
            return maxDoc;
        }
    }

    private static final class SparseNumeric extends NumericDocValues {
        private final IndexedDISI disi;
        private final AdaptiveBlocks blocks;

        SparseNumeric(IndexedDISI disi, AdaptiveBlocks blocks) {
            this.disi = disi;
            this.blocks = blocks;
        }

        @Override
        public long longValue() throws IOException {
            return blocks.get(disi.index());
        }

        @Override
        public boolean advanceExact(int target) throws IOException {
            return disi.advanceExact(target);
        }

        @Override
        public int docID() {
            return disi.docID();
        }

        @Override
        public int nextDoc() throws IOException {
            return disi.nextDoc();
        }

        @Override
        public int advance(int target) throws IOException {
            return disi.advance(target);
        }

        @Override
        public long cost() {
            return disi.cost();
        }
    }

    private static final class DenseSortedNumeric extends SortedNumericDocValues {
        private final int maxDoc;
        private final LongValues addresses;
        private final AdaptiveBlocks blocks;
        private int doc = -1;
        private long start;
        private int count;

        DenseSortedNumeric(int maxDoc, LongValues addresses, AdaptiveBlocks blocks) {
            this.maxDoc = maxDoc;
            this.addresses = addresses;
            this.blocks = blocks;
        }

        @Override
        public long nextValue() throws IOException {
            return blocks.get(start++);
        }

        @Override
        public int docValueCount() {
            return count;
        }

        @Override
        public boolean advanceExact(int target) {
            start = addresses.get(target);
            count = (int) (addresses.get(target + 1L) - start);
            doc = target;
            return true;
        }

        @Override
        public int docID() {
            return doc;
        }

        @Override
        public int nextDoc() {
            return advance(doc + 1);
        }

        @Override
        public int advance(int target) {
            if (target >= maxDoc) {
                return doc = NO_MORE_DOCS;
            }
            advanceExact(target);
            return doc;
        }

        @Override
        public long cost() {
            return maxDoc;
        }
    }

    private static final class SparseSortedNumeric extends SortedNumericDocValues {
        private final IndexedDISI disi;
        private final LongValues addresses;
        private final AdaptiveBlocks blocks;
        private long start;
        private int count;
        private boolean set;

        SparseSortedNumeric(IndexedDISI disi, LongValues addresses, AdaptiveBlocks blocks) {
            this.disi = disi;
            this.addresses = addresses;
            this.blocks = blocks;
        }

        private void load() {
            if (set == false) {
                int index = disi.index();
                start = addresses.get(index);
                count = (int) (addresses.get(index + 1L) - start);
                set = true;
            }
        }

        @Override
        public long nextValue() throws IOException {
            load();
            return blocks.get(start++);
        }

        @Override
        public int docValueCount() {
            load();
            return count;
        }

        @Override
        public boolean advanceExact(int target) throws IOException {
            set = false;
            return disi.advanceExact(target);
        }

        @Override
        public int docID() {
            return disi.docID();
        }

        @Override
        public int nextDoc() throws IOException {
            set = false;
            return disi.nextDoc();
        }

        @Override
        public int advance(int target) throws IOException {
            set = false;
            return disi.advance(target);
        }

        @Override
        public long cost() {
            return disi.cost();
        }
    }
}
