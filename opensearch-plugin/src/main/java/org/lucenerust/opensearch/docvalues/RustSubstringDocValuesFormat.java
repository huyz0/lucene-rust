/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.docvalues;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.LongValues;
import org.apache.lucene.util.compress.LZ4;
import org.apache.lucene.util.packed.DirectMonotonicReader;

/**
 * The {@code Substring90} doc-values format the Rust engine writes for a {@code compact_wildcard}
 * field ({@code lucene-codecs/src/substring.rs}): Lucene90's files and entry prefix, with each BINARY
 * field's values in LZ4 blocks of {@code 2^shift} documents next to a trigram index of the blocks.
 * This side reads the values back, so every Lucene path that reads the field's doc values (a
 * search answered by Lucene, {@code CheckIndex}) sees what was indexed; the trigram index is the
 * native searches'.
 *
 * <p>Java never writes the format: a Java merge writes its own codec's formats, plain BINARY, which
 * the field's queries read alike.
 */
public final class RustSubstringDocValuesFormat extends DocValuesFormat {
    /** {@code Lucene90DocValuesFormat::RUST_SUBSTRING_NAME} in {@code doc_values.rs}. */
    public static final String NAME = "Substring90";

    public RustSubstringDocValuesFormat() {
        super(NAME);
    }

    @Override
    public DocValuesConsumer fieldsConsumer(SegmentWriteState state) {
        throw new UnsupportedOperationException("the " + NAME + " doc values format is written by the accelerated engine only");
    }

    @Override
    public DocValuesProducer fieldsProducer(SegmentReadState state) throws IOException {
        return new Producer(state);
    }

    /** {@code MIN_LENGTH_MARKER} in {@code substring.rs}: the {@code minLength} of a block entry. */
    static final int MIN_LENGTH_MARKER = -1001;
    /** {@code MAX_BLOCK_SHIFT} in {@code substring.rs}. */
    static final int MAX_BLOCK_SHIFT = 10;
    /** {@code ADDRESS_BLOCK_SHIFT} in {@code substring.rs}. */
    static final int ADDRESS_BLOCK_SHIFT = 16;

    private static final String META_CODEC = "Lucene90DocValuesMetadata";
    private static final String DATA_CODEC = "Lucene90DocValuesData";
    private static final int VERSION_START = 0;
    private static final int VERSION_SKIPPER_MAX_VALUE_COUNT = 2;
    private static final int VERSION_CURRENT = VERSION_SKIPPER_MAX_VALUE_COUNT;
    private static final byte BINARY = 1;

    /** One field's entry: Lucene90's BINARY prefix and the block layout after it. */
    static final class Entry {
        long docsWithFieldOffset;
        long docsWithFieldLength;
        short jumpTableEntryCount;
        byte denseRankPower;
        int numDocsWithField;
        int maxLength;
        int blockShift;
        long numBlocks;
        long blocksOffset;
        long blocksLength;
        long addressesOffset;
        long addressesLength;
        DirectMonotonicReader.Meta addressesMeta;
    }

    static final class Producer extends DocValuesProducer {
        private final Map<Integer, Entry> binaries = new HashMap<>();
        private final IndexInput data;
        private final int maxDoc;

        Producer(SegmentReadState state) throws IOException {
            this.maxDoc = state.segmentInfo.maxDoc();
            String metaName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "dvm");
            int version = -1;
            try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaName)) {
                Throwable prior = null;
                try {
                    version = CodecUtil.checkIndexHeader(
                        meta, META_CODEC, VERSION_START, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
                    readFields(meta, state);
                } catch (Throwable t) {
                    prior = t;
                } finally {
                    CodecUtil.checkFooter(meta, prior);
                }
            }
            String dataName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, "dvd");
            IndexInput in = state.directory.openInput(dataName, state.context);
            try {
                int dataVersion = CodecUtil.checkIndexHeader(
                    in, DATA_CODEC, VERSION_START, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);
                if (dataVersion != version) {
                    throw new CorruptIndexException("format versions mismatch: meta=" + version + ", data=" + dataVersion, in);
                }
                CodecUtil.retrieveChecksum(in);
            } catch (Throwable t) {
                IOUtils.closeWhileSuppressingExceptions(t, in);
                throw t;
            }
            this.data = in;
        }

        private void readFields(IndexInput meta, SegmentReadState state) throws IOException {
            for (int number = meta.readInt(); number != -1; number = meta.readInt()) {
                FieldInfo info = state.fieldInfos.fieldInfo(number);
                if (info == null) {
                    throw new CorruptIndexException("Invalid field number: " + number, meta);
                }
                byte type = meta.readByte();
                if (type != BINARY) {
                    throw new CorruptIndexException("a " + NAME + " field holds BINARY doc values only, got type " + type, meta);
                }
                binaries.put(number, readBinary(meta));
            }
        }

        /** {@code read_binary_entry} in {@code doc_values.rs}, then {@code substring::read_meta}. */
        private static Entry readBinary(IndexInput meta) throws IOException {
            Entry e = new Entry();
            meta.readLong(); // data offset
            meta.readLong(); // data length
            e.docsWithFieldOffset = meta.readLong();
            e.docsWithFieldLength = meta.readLong();
            e.jumpTableEntryCount = meta.readShort();
            e.denseRankPower = meta.readByte();
            e.numDocsWithField = meta.readInt();
            int minLength = meta.readInt();
            e.maxLength = meta.readInt();
            if (minLength != MIN_LENGTH_MARKER) {
                throw new CorruptIndexException("a " + NAME + " entry without blocks (minLength " + minLength + ")", meta);
            }
            e.blockShift = meta.readByte();
            if (e.blockShift < 0 || e.blockShift > MAX_BLOCK_SHIFT) {
                throw new CorruptIndexException("block shift out of range: " + e.blockShift, meta);
            }
            e.numBlocks = meta.readVLong();
            if (e.numBlocks < 0 || e.numBlocks > (((long) Integer.MAX_VALUE) >> e.blockShift) + 1) {
                throw new CorruptIndexException("block count out of range: " + e.numBlocks, meta);
            }
            e.blocksOffset = meta.readLong();
            e.blocksLength = meta.readLong();
            e.addressesOffset = meta.readLong();
            e.addressesLength = meta.readLong();
            e.addressesMeta = DirectMonotonicReader.loadMeta(meta, e.numBlocks + 1, ADDRESS_BLOCK_SHIFT);
            meta.readLong(); // grams offset
            meta.readVLong(); // gram count
            meta.readLong(); // postings offset
            meta.readLong(); // postings length
            return e;
        }

        @Override
        public BinaryDocValues getBinary(FieldInfo field) throws IOException {
            Entry e = binaries.get(field.number);
            if (e == null || e.docsWithFieldOffset == -2) {
                return DocValues.emptyBinary();
            }
            DocIdSetIterator docs = e.docsWithFieldOffset == -1
                ? DocIdSetIterator.all(maxDoc)
                : new IndexedDISI(data, e.docsWithFieldOffset, e.docsWithFieldLength, e.jumpTableEntryCount, e.denseRankPower, e.numDocsWithField);
            LongValues addresses = DirectMonotonicReader.getInstance(
                e.addressesMeta, data.randomAccessSlice(e.addressesOffset, e.addressesLength));
            IndexInput blocks = data.slice("substring blocks", e.blocksOffset, e.blocksLength);
            return new Values(e, docs, addresses, blocks);
        }

        @Override
        public NumericDocValues getNumeric(FieldInfo field) {
            throw new IllegalStateException("a " + NAME + " field holds BINARY doc values only");
        }

        @Override
        public SortedDocValues getSorted(FieldInfo field) {
            throw new IllegalStateException("a " + NAME + " field holds BINARY doc values only");
        }

        @Override
        public SortedNumericDocValues getSortedNumeric(FieldInfo field) {
            throw new IllegalStateException("a " + NAME + " field holds BINARY doc values only");
        }

        @Override
        public SortedSetDocValues getSortedSet(FieldInfo field) {
            throw new IllegalStateException("a " + NAME + " field holds BINARY doc values only");
        }

        @Override
        public DocValuesSkipper getSkipper(FieldInfo field) {
            return null;
        }

        @Override
        public void checkIntegrity() throws IOException {
            CodecUtil.checksumEntireFile(data);
        }

        @Override
        public void close() throws IOException {
            data.close();
        }
    }

    /** A field's values: the documents with one, and each value read from its decompressed block. */
    static final class Values extends BinaryDocValues {
        private final Entry entry;
        private final DocIdSetIterator docs;
        private final LongValues addresses;
        private final IndexInput blocks;
        private final BytesRef value = new BytesRef();
        private long block = -1;
        private byte[] raw = BytesRef.EMPTY_BYTES;
        private int count;
        private int[] blockDocs = new int[0];
        private int[] starts = new int[0];
        private int[] lengths = new int[0];

        Values(Entry entry, DocIdSetIterator docs, LongValues addresses, IndexInput blocks) {
            this.entry = entry;
            this.docs = docs;
            this.addresses = addresses;
            this.blocks = blocks;
        }

        @Override
        public int docID() {
            return docs.docID();
        }

        @Override
        public int nextDoc() throws IOException {
            return docs.nextDoc();
        }

        @Override
        public int advance(int target) throws IOException {
            return docs.advance(target);
        }

        @Override
        public boolean advanceExact(int target) throws IOException {
            if (docs instanceof IndexedDISI disi) {
                return disi.advanceExact(target);
            }
            // Dense: every document has a value.
            docs.advance(target);
            return true;
        }

        @Override
        public long cost() {
            return docs.cost();
        }

        @Override
        public BytesRef binaryValue() throws IOException {
            int doc = docs.docID();
            load(((long) doc) >>> entry.blockShift);
            int lo = 0;
            int hi = count - 1;
            while (lo <= hi) {
                int mid = (lo + hi) >>> 1;
                int d = blockDocs[mid];
                if (d < doc) {
                    lo = mid + 1;
                } else if (d > doc) {
                    hi = mid - 1;
                } else {
                    value.bytes = raw;
                    value.offset = starts[mid];
                    value.length = lengths[mid];
                    return value;
                }
            }
            throw new CorruptIndexException("document " + doc + " has no value in its block", blocks);
        }

        /** {@code SubstringIndex::read_block} in {@code substring.rs}. */
        private void load(long b) throws IOException {
            if (b == block) {
                return;
            }
            block = -1;
            count = 0;
            if (b < 0 || b >= entry.numBlocks) {
                throw new CorruptIndexException("block out of range: " + b, blocks);
            }
            long start = addresses.get(b);
            long end = addresses.get(b + 1);
            if (end < start || end > blocks.length()) {
                throw new CorruptIndexException("block addresses out of order", blocks);
            }
            if (end > start) {
                blocks.seek(start);
                int len = blocks.readVInt();
                // LZ4 expands a byte to at most 255.
                if (len < 0 || len > (end - start) * 255 + 16) {
                    throw new CorruptIndexException("block length past what its bytes can hold", blocks);
                }
                if (raw.length < len) {
                    raw = new byte[len];
                }
                LZ4.decompress(blocks, len, raw, 0);
                int[] pos = { 0 };
                int n = readVInt(raw, pos, len);
                int size = 1 << entry.blockShift;
                if (n < 0 || n > size) {
                    throw new CorruptIndexException("block holds more documents than its size", blocks);
                }
                if (blockDocs.length < n) {
                    blockDocs = new int[size];
                    starts = new int[size];
                    lengths = new int[size];
                }
                int base = (int) (b << entry.blockShift);
                for (int i = 0; i < n; i++) {
                    int d = readVInt(raw, pos, len);
                    int l = readVInt(raw, pos, len);
                    if (d < 0 || d >= size || l < 0) {
                        throw new CorruptIndexException("block header out of range", blocks);
                    }
                    blockDocs[i] = base + d;
                    lengths[i] = l;
                }
                int at = pos[0];
                for (int i = 0; i < n; i++) {
                    starts[i] = at;
                    at += lengths[i];
                    if (at < 0 || at > len) {
                        throw new CorruptIndexException("value past the block", blocks);
                    }
                }
                count = n;
            }
            block = b;
        }

        /** {@code read_vu32} in {@code substring.rs}. */
        private int readVInt(byte[] bytes, int[] pos, int len) throws CorruptIndexException {
            int v = 0;
            for (int shift = 0; shift <= 28; shift += 7) {
                if (pos[0] >= len) {
                    break;
                }
                byte b = bytes[pos[0]++];
                v |= (b & 0x7F) << shift;
                if ((b & 0x80) == 0) {
                    return v;
                }
            }
            throw new CorruptIndexException("block header", blocks);
        }
    }
}
