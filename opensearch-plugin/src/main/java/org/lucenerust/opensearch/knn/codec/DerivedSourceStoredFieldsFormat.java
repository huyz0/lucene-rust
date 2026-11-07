/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn.codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.StoredFieldsFormat;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.StoredFieldsWriter;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.StoredFieldVisitor;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.IOUtils;
import org.opensearch.common.collect.Tuple;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.compress.NotXContentException;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * k-NN's derived source ({@code KNN10010DerivedSourceStoredFieldsFormat}): a segment whose {@code .si}
 * attribute {@link #DERIVED_VECTOR_FIELDS} (or {@link #NESTED_DERIVED_VECTOR_FIELDS}) lists vector
 * fields stores each document's {@code _source} with those fields' values masked. Reading puts each
 * field's vector, from the segment's vector values, back in place of every non-null value at its
 * path; a nested field's values are its child documents' vectors, in order. {@code KNN9120Codec}'s
 * segments mark the fields with {@link #FIELD_ATTRIBUTE} instead. The stored fields themselves are
 * those of the codec {@link #DELEGATE_CODEC} names, or of the delegate when it names none.
 *
 * <p>Writing records the delegate's name and nothing else: this node's writers keep {@code _source}
 * whole, and a merge reads every document through {@link #fieldsReader}, so the merged segment holds
 * the vectors.
 */
final class DerivedSourceStoredFieldsFormat extends StoredFieldsFormat {
    static final String DERIVED_VECTOR_FIELDS = "derived_vector_fields";
    static final String NESTED_DERIVED_VECTOR_FIELDS = "nested_derived_vector_fields";
    static final String DELEGATE_CODEC = "knn_delegate_stored_fields_codec_key";
    static final String FIELD_ATTRIBUTE = "knn-derived-source-enabled";
    private static final String SOURCE = "_source";
    /** Set on root documents only, so it finds a parent's first child. */
    private static final String PRIMARY_TERM = "_primary_term";

    private final Codec codec;
    private final String delegateName;
    private final StoredFieldsFormat delegate;

    DerivedSourceStoredFieldsFormat(Codec codec, Codec delegate) {
        this.codec = codec;
        this.delegateName = delegate.getName();
        this.delegate = delegate.storedFieldsFormat();
    }

    @Override
    public StoredFieldsReader fieldsReader(Directory directory, SegmentInfo si, FieldInfos fieldInfos, IOContext context)
        throws IOException {
        StoredFieldsFormat format = delegate;
        String name = si.getAttribute(DELEGATE_CODEC);
        if (name != null && name.equalsIgnoreCase(delegateName) == false) {
            format = Codec.forName(name).storedFieldsFormat();
        }
        List<FieldInfo> root = new ArrayList<>();
        List<FieldInfo> nested = new ArrayList<>();
        resolve(fieldInfos, si.getAttribute(DERIVED_VECTOR_FIELDS), root);
        resolve(fieldInfos, si.getAttribute(NESTED_DERIVED_VECTOR_FIELDS), nested);
        for (FieldInfo fi : fieldInfos) {
            if ("true".equals(fi.getAttribute(FIELD_ATTRIBUTE)) && root.contains(fi) == false) {
                root.add(fi);
            }
        }
        StoredFieldsReader reader = format.fieldsReader(directory, si, fieldInfos, context);
        if (root.isEmpty() && nested.isEmpty()) {
            return reader;
        }
        SegmentReadState state = new SegmentReadState(directory, si, fieldInfos, context);
        KnnVectorsReader vectors = null;
        DocValuesProducer docValues = null;
        boolean ok = false;
        try {
            vectors = codec.knnVectorsFormat().fieldsReader(state);
            if (nested.isEmpty() == false && fieldInfos.hasDocValues()) {
                docValues = codec.docValuesFormat().fieldsProducer(state);
            }
            Reader r = new Reader(reader, root, nested, fieldInfos, new Readers(vectors, docValues));
            ok = true;
            return r;
        } finally {
            if (ok == false) {
                IOUtils.closeWhileHandlingException(reader, vectors, docValues);
            }
        }
    }

    @Override
    public StoredFieldsWriter fieldsWriter(Directory directory, SegmentInfo si, IOContext context) throws IOException {
        String previous = si.putAttribute(DELEGATE_CODEC, delegateName);
        if (previous != null && previous.equals(delegateName) == false) {
            throw new IllegalStateException(
                "segment [" + si.name + "] records stored fields codec [" + previous + "], not [" + delegateName + "]"
            );
        }
        return delegate.fieldsWriter(directory, si, context);
    }

    private static void resolve(FieldInfos infos, String list, List<FieldInfo> out) {
        if (list == null || list.isEmpty()) {
            return;
        }
        for (String field : list.split(",", -1)) {
            FieldInfo fi = fieldInfo(infos, field);
            if (fi != null && out.contains(fi) == false) {
                out.add(fi);
            }
        }
    }

    /**
     * {@code KNN10010DerivedSourceStoredFieldsFormat.getFieldInfo}: the field by its name, else the one
     * matching ignoring case (one with vectors before one without; two with vectors match none).
     */
    static FieldInfo fieldInfo(FieldInfos infos, String field) {
        FieldInfo exact = infos.fieldInfo(field);
        if (exact != null) {
            return exact;
        }
        FieldInfo matched = null;
        for (FieldInfo candidate : infos) {
            if (candidate.name.equalsIgnoreCase(field) == false) {
                continue;
            }
            if (matched == null || candidate.hasVectorValues() && matched.hasVectorValues() == false) {
                matched = candidate;
            } else if (matched.hasVectorValues() && candidate.hasVectorValues()) {
                return null;
            }
        }
        return matched;
    }

    /** The segment's vectors and doc values, closed by the reader that opened them. */
    private static final class Readers implements Closeable {
        final KnnVectorsReader vectors;
        final DocValuesProducer docValues;

        Readers(KnnVectorsReader vectors, DocValuesProducer docValues) {
            this.vectors = vectors;
            this.docValues = docValues;
        }

        @Override
        public void close() throws IOException {
            IOUtils.close(vectors, docValues);
        }
    }

    private record Cursor(KnnVectorValues values, KnnVectorValues.DocIndexIterator it) {}

    private static final class Reader extends StoredFieldsReader {
        private final StoredFieldsReader delegate;
        private final List<FieldInfo> root;
        private final List<FieldInfo> nested;
        private final FieldInfos fieldInfos;
        private final Readers readers;
        private final boolean owner;
        /** Per field, the vectors read last; documents are mostly read in order. */
        private final Map<String, Cursor> cursors = new HashMap<>();

        Reader(StoredFieldsReader delegate, List<FieldInfo> root, List<FieldInfo> nested, FieldInfos fieldInfos, Readers readers) {
            this(delegate, root, nested, fieldInfos, readers, true);
        }

        private Reader(
            StoredFieldsReader delegate,
            List<FieldInfo> root,
            List<FieldInfo> nested,
            FieldInfos fieldInfos,
            Readers readers,
            boolean owner
        ) {
            this.delegate = delegate;
            this.root = root;
            this.nested = nested;
            this.fieldInfos = fieldInfos;
            this.readers = readers;
            this.owner = owner;
        }

        @Override
        public void document(int docID, StoredFieldVisitor visitor) throws IOException {
            delegate.document(docID, new StoredFieldVisitor() {
                @Override
                public void binaryField(FieldInfo fi, byte[] value) throws IOException {
                    visitor.binaryField(fi, fi.name.equals(SOURCE) ? inject(docID, value) : value);
                }

                @Override
                public void stringField(FieldInfo fi, String value) throws IOException {
                    visitor.stringField(fi, value);
                }

                @Override
                public void intField(FieldInfo fi, int value) throws IOException {
                    visitor.intField(fi, value);
                }

                @Override
                public void longField(FieldInfo fi, long value) throws IOException {
                    visitor.longField(fi, value);
                }

                @Override
                public void floatField(FieldInfo fi, float value) throws IOException {
                    visitor.floatField(fi, value);
                }

                @Override
                public void doubleField(FieldInfo fi, double value) throws IOException {
                    visitor.doubleField(fi, value);
                }

                @Override
                public Status needsField(FieldInfo fi) throws IOException {
                    return visitor.needsField(fi);
                }
            });
        }

        /** {@code DerivedSourceVectorTransformer.injectVectors}. */
        private byte[] inject(int docID, byte[] source) throws IOException {
            Tuple<? extends MediaType, Map<String, Object>> parsed;
            try {
                parsed = XContentHelper.convertToMap(new BytesArray(source), true, (MediaType) null);
            } catch (NotXContentException e) {
                // A no-op tombstone's _source is not content; it holds no vectors.
                return source;
            }
            Map<String, Function<Object, Object>> transforms = new HashMap<>();
            for (FieldInfo fi : root) {
                Cursor c = cursor(fi, docID);
                boolean present = c != null && advanceTo(c, docID) == docID;
                transforms.put(fi.name, v -> v == null || present == false ? v : vector(c.values, c.it));
            }
            if (nested.isEmpty() == false) {
                int first = firstChild(docID);
                for (FieldInfo fi : nested) {
                    Cursor c = first < 0 ? null : cursor(fi, first);
                    if (c != null) {
                        advanceTo(c, first);
                    }
                    transforms.put(fi.name, v -> {
                        if (v == null || c == null || c.it.docID() >= docID) {
                            return v;
                        }
                        Object vector = vector(c.values, c.it);
                        try {
                            c.it.nextDoc();
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                        return vector;
                    });
                }
            }
            Map<String, Object> map;
            try {
                map = XContentMapValues.transform(transforms, true).apply(parsed.v2());
            } catch (UncheckedIOException e) {
                throw e.getCause();
            }
            BytesStreamOutput out = new BytesStreamOutput(Math.max(1024, source.length * 2));
            try (XContentBuilder b = MediaTypeRegistry.contentBuilder(parsed.v1(), out)) {
                b.map(map);
            }
            return BytesReference.toBytes(out.bytes());
        }

        private static int advanceTo(Cursor c, int target) throws IOException {
            return c.it.docID() < target ? c.it.advance(target) : c.it.docID();
        }

        /**
         * Field {@code fi}'s vector cursor, positioned at or before {@code target}: the one kept from
         * the last document read, or a new one when that has moved past it. Null when the field has
         * no vectors in the segment.
         */
        private Cursor cursor(FieldInfo fi, int target) throws IOException {
            if (readers.vectors == null || fi.hasVectorValues() == false) {
                return null;
            }
            Cursor c = cursors.get(fi.name);
            if (c == null || c.it.docID() > target) {
                KnnVectorValues values = fi.getVectorEncoding() == VectorEncoding.BYTE
                    ? readers.vectors.getByteVectorValues(fi.name)
                    : readers.vectors.getFloatVectorValues(fi.name);
                if (values == null) {
                    return null;
                }
                c = new Cursor(values, values.iterator());
                cursors.put(fi.name, c);
            }
            return c;
        }

        private static Object vector(KnnVectorValues values, KnnVectorValues.DocIndexIterator it) {
            try {
                int ord = it.index();
                if (values instanceof FloatVectorValues f) {
                    return f.vectorValue(ord).clone();
                }
                byte[] b = ((ByteVectorValues) values).vectorValue(ord);
                int[] out = new int[b.length];
                for (int i = 0; i < b.length; i++) {
                    out[i] = b[i];
                }
                return out;
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        /**
         * {@code DerivedSourceLuceneHelper.getFirstChild}: the first child of the block {@code parent}
         * ends, from the previous root document ({@link #PRIMARY_TERM}); -1 when it has none.
         */
        private int firstChild(int parent) throws IOException {
            FieldInfo pt = fieldInfos.fieldInfo(PRIMARY_TERM);
            if (parent == 0 || pt == null || readers.docValues == null) {
                return -1;
            }
            int last = -1;
            for (int offset : new int[] { 150, 1500, 6000, Integer.MAX_VALUE }) {
                int start = (int) Math.max(0L, (long) parent - offset);
                if (start <= last) {
                    break;
                }
                NumericDocValues dv = readers.docValues.getNumeric(pt);
                int previous = -1;
                for (int d = dv.advance(start); d < parent; d = dv.nextDoc()) {
                    previous = d;
                }
                if (previous >= 0 || start == 0) {
                    return parent - previous <= 1 ? -1 : previous + 1;
                }
                last = start;
            }
            return -1;
        }

        @Override
        public StoredFieldsReader clone() {
            return new Reader(delegate.clone(), root, nested, fieldInfos, readers, false);
        }

        @Override
        public StoredFieldsReader getMergeInstance() {
            return new Reader(delegate.getMergeInstance(), root, nested, fieldInfos, readers, false);
        }

        @Override
        public void checkIntegrity() throws IOException {
            delegate.checkIntegrity();
        }

        @Override
        public void close() throws IOException {
            if (owner) {
                IOUtils.close(delegate, readers);
            } else {
                delegate.close();
            }
        }
    }
}
