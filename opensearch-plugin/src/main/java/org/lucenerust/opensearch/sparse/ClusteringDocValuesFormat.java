/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.sparse;

import org.apache.lucene.codecs.DocValuesConsumer;
import org.apache.lucene.codecs.DocValuesFormat;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.NativeBridge;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.TreeMap;

/**
 * {@code delegate}'s doc values, plus the segment's clustered postings: neural-search's
 * {@code SparseDocValuesConsumer} and {@code SparsePostingsConsumer}. Each {@code sparse_vector}
 * field's forward index (its BINARY doc values) is handed to a Rust clusterer as it is flushed or
 * merged, and once the segment's doc values are written its {@code .sit}/{@code .sip} are encoded
 * in Rust -- the files, clusters and seeds the Rust writer gives the same segment.
 */
final class ClusteringDocValuesFormat extends DocValuesFormat {
    static final String TERMS_EXTENSION = "sit";
    static final String POSTINGS_EXTENSION = "sip";
    private static final String SPARSE_FIELD = "sparse_vector_field";
    private static final String APPROXIMATE_THRESHOLD = "approximate_threshold";
    private static final String ENGINE = "engine";

    private final DocValuesFormat delegate;

    ClusteringDocValuesFormat(DocValuesFormat delegate) {
        super(delegate.getName());
        this.delegate = delegate;
    }

    @Override
    public DocValuesConsumer fieldsConsumer(SegmentWriteState state) throws IOException {
        return new Consumer(delegate.fieldsConsumer(state), state);
    }

    @Override
    public DocValuesProducer fieldsProducer(SegmentReadState state) throws IOException {
        return delegate.fieldsProducer(state);
    }

    /**
     * {@code SparseVectorField.isSparseField} and {@code PredicateUtils.shouldRunSeisPredicate}: a
     * {@code sparse_vector} field of the lucene engine whose segment of {@code maxDoc} documents is
     * clustered.
     */
    static boolean clustered(FieldInfo field, int maxDoc) {
        if (field.getDocValuesType() != DocValuesType.BINARY || Boolean.parseBoolean(field.getAttribute(SPARSE_FIELD)) == false) {
            return false;
        }
        if ("native".equalsIgnoreCase(field.getAttribute(ENGINE))) {
            return false;
        }
        String threshold = field.getAttribute(APPROXIMATE_THRESHOLD);
        return threshold != null && maxDoc >= Integer.parseInt(threshold);
    }

    /** The attributes as {@code ffi_sparse_clusterer_open} takes them. */
    static byte[] encodeAttributes(Map<String, String> attributes) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        writeInt(out, attributes.size());
        for (Map.Entry<String, String> e : attributes.entrySet()) {
            for (String s : new String[] { e.getKey(), e.getValue() == null ? "" : e.getValue() }) {
                byte[] b = s.getBytes(StandardCharsets.UTF_8);
                writeInt(out, b.length);
                out.writeBytes(b);
            }
        }
        return out.toByteArray();
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.writeBytes(ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array());
    }

    private static IOException failure(String what) {
        return new IOException("sparse clustering: " + what + ": " + NativeBridge.lastError());
    }

    private static final class Consumer extends DocValuesConsumer {
        private final DocValuesConsumer delegate;
        private final SegmentWriteState state;
        /** Each clustered field's clusterer, by name: the order the fields are written in. */
        private final Map<String, Long> clusterers = new TreeMap<>();

        Consumer(DocValuesConsumer delegate, SegmentWriteState state) {
            this.delegate = delegate;
            this.state = state;
        }

        @Override
        public void addNumericField(FieldInfo field, DocValuesProducer values) throws IOException {
            delegate.addNumericField(field, values);
        }

        @Override
        public void addBinaryField(FieldInfo field, DocValuesProducer values) throws IOException {
            delegate.addBinaryField(field, values);
            cluster(field, values);
        }

        @Override
        public void addSortedField(FieldInfo field, DocValuesProducer values) throws IOException {
            delegate.addSortedField(field, values);
        }

        @Override
        public void addSortedNumericField(FieldInfo field, DocValuesProducer values) throws IOException {
            delegate.addSortedNumericField(field, values);
        }

        @Override
        public void addSortedSetField(FieldInfo field, DocValuesProducer values) throws IOException {
            delegate.addSortedSetField(field, values);
        }

        /** The delegate's own merge, then each clustered field's merged forward index. */
        @Override
        public void merge(MergeState mergeState) throws IOException {
            delegate.merge(mergeState);
            int maxDoc = mergeState.segmentInfo.maxDoc();
            DocValuesConsumer clustering = new DocValuesConsumer() {
                @Override
                public void addBinaryField(FieldInfo field, DocValuesProducer values) throws IOException {
                    cluster(field, values);
                }

                @Override
                public void addNumericField(FieldInfo field, DocValuesProducer values) {
                    throw new UnsupportedOperationException();
                }

                @Override
                public void addSortedField(FieldInfo field, DocValuesProducer values) {
                    throw new UnsupportedOperationException();
                }

                @Override
                public void addSortedNumericField(FieldInfo field, DocValuesProducer values) {
                    throw new UnsupportedOperationException();
                }

                @Override
                public void addSortedSetField(FieldInfo field, DocValuesProducer values) {
                    throw new UnsupportedOperationException();
                }

                @Override
                public void close() {}
            };
            for (FieldInfo field : mergeState.mergeFieldInfos) {
                if (clustered(field, maxDoc)) {
                    clustering.mergeBinaryField(field, mergeState);
                }
            }
        }

        private void cluster(FieldInfo field, DocValuesProducer values) throws IOException {
            int maxDoc = state.segmentInfo.maxDoc();
            if (clustered(field, maxDoc) == false || clusterers.containsKey(field.name)) {
                return;
            }
            long[] handle = new long[1];
            byte[] name = field.name.getBytes(StandardCharsets.UTF_8);
            if (NativeBridge.sparseClustererOpen(name, field.number, encodeAttributes(field.attributes()), maxDoc, handle) != 0) {
                throw failure("field [" + field.name + "]");
            }
            clusterers.put(field.name, handle[0]);
            BinaryDocValues docs = values.getBinary(field);
            for (int doc = docs.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = docs.nextDoc()) {
                BytesRef v = docs.binaryValue();
                if (NativeBridge.sparseClustererAdd(handle[0], doc, v.bytes, v.offset, v.length) != 0) {
                    throw failure("field [" + field.name + "] doc " + doc);
                }
            }
        }

        @Override
        public void close() throws IOException {
            try {
                delegate.close();
                writePostings();
            } finally {
                for (long h : clusterers.values()) {
                    NativeBridge.sparseClustererClose(h);
                }
                clusterers.clear();
            }
        }

        /** {@code SparsePostingsConsumer}: the segment's {@code .sit}/{@code .sip}, fields in name order. */
        private void writePostings() throws IOException {
            long[] handles = clusterers.values().stream().mapToLong(Long::longValue).toArray();
            // The write takes the handles, whatever its outcome.
            clusterers.clear();
            byte[] files = NativeBridge.sparsePostingsWrite(handles, state.segmentInfo.getId());
            if (files == null) {
                throw failure("segment [" + state.segmentInfo.name + "]");
            }
            int sitLength = Math.toIntExact(ByteBuffer.wrap(files, 0, Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).getLong());
            write(TERMS_EXTENSION, files, Long.BYTES, sitLength);
            write(POSTINGS_EXTENSION, files, Long.BYTES + sitLength, files.length - Long.BYTES - sitLength);
        }

        private void write(String extension, byte[] bytes, int offset, int length) throws IOException {
            String name = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, extension);
            try (IndexOutput out = state.directory.createOutput(name, state.context)) {
                out.writeBytes(bytes, offset, length);
            }
        }
    }
}
