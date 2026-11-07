/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.knn;

import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.SortField;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.util.BigArrays;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.index.fielddata.IndexFieldData;
import org.opensearch.index.fielddata.IndexFieldDataCache;
import org.opensearch.index.fielddata.LeafFieldData;
import org.opensearch.index.fielddata.ScriptDocValues;
import org.opensearch.index.fielddata.SortedBinaryDocValues;
import org.opensearch.index.mapper.DocValueFetcher;
import org.opensearch.core.indices.breaker.CircuitBreakerService;
import org.opensearch.search.DocValueFormat;
import org.opensearch.search.MultiValueMode;
import org.opensearch.search.aggregations.support.CoreValuesSourceType;
import org.opensearch.search.aggregations.support.ValuesSourceType;
import org.opensearch.search.sort.BucketedSort;
import org.opensearch.search.sort.SortOrder;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * k-NN's {@code KNNVectorIndexFieldData} and {@code KNNVectorDocValueFormat}: a {@code knn_vector}
 * field's values for {@code docvalue_fields}, read from the vector files, or from the binary doc
 * values of a field without them. Scripts, sorts and aggregations are refused.
 */
public final class KnnVectorFieldData implements IndexFieldData<KnnVectorFieldData.Leaf> {
    private final String name;
    private final KnnFieldConfig config;

    KnnVectorFieldData(String name, KnnFieldConfig config) {
        this.name = name;
        this.config = config;
    }

    /** {@code KNNVectorDocValueFormat}: {@code binary} (the default) or {@code array}. */
    public enum Format implements DocValueFormat {
        ARRAY("array", false),
        BINARY("binary", true);

        public static final String NAME = "knn_vector";
        private final String formatName;
        private final boolean binary;

        Format(String formatName, boolean binary) {
            this.formatName = formatName;
            this.binary = binary;
        }

        public static Format fromStream(StreamInput in) throws IOException {
            return in.readBoolean() ? BINARY : ARRAY;
        }

        static Format of(String format) {
            if (format == null || BINARY.formatName.equals(format)) {
                return BINARY;
            }
            if (ARRAY.formatName.equals(format)) {
                return ARRAY;
            }
            throw new IllegalArgumentException(
                "Unsupported knn_vector docvalue_fields format [" + format + "]. Supported formats are " + Arrays.toString(values())
            );
        }

        @Override
        public String getWriteableName() {
            return NAME;
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeBoolean(binary);
        }

        @Override
        public String toString() {
            return "knn_vector(" + formatName + ")";
        }
    }

    /** {@code KNNVectorFieldType.valueForDisplay}: a stored vector as {@code deserializeStoredVector} reads it. */
    static Object stored(BytesRef bytes, KnnFieldConfig config) {
        if (config.dataType() == VectorEncoding.BYTE) {
            int[] out = new int[bytes.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = bytes.bytes[bytes.offset + i];
            }
            return out;
        }
        return config.halfFloat() ? halfFloats(bytes) : floats(bytes);
    }

    /** {@code KNNVectorAsCollectionOfFloatsSerializer.byteToFloatArray}: big-endian floats. */
    private static float[] floats(BytesRef bytes) {
        ByteBuffer b = ByteBuffer.wrap(bytes.bytes, bytes.offset, bytes.length);
        float[] out = new float[bytes.length / Float.BYTES];
        b.asFloatBuffer().get(out);
        return out;
    }

    /** {@code KNNVectorAsCollectionOfHalfFloatsSerializer.byteToFloatArray}: little-endian half floats. */
    private static float[] halfFloats(BytesRef bytes) {
        float[] out = new float[bytes.length / 2];
        for (int i = 0; i < out.length; i++) {
            int at = bytes.offset + 2 * i;
            short h = (short) ((bytes.bytes[at] & 0xFF) | (bytes.bytes[at + 1] << 8));
            out[i] = Float.float16ToFloat(h);
        }
        return out;
    }

    private static byte[] littleEndian(float[] v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        b.asFloatBuffer().put(v);
        return b.array();
    }

    @Override
    public String getFieldName() {
        return name;
    }

    @Override
    public ValuesSourceType getValuesSourceType() {
        return CoreValuesSourceType.BYTES;
    }

    @Override
    public Leaf load(LeafReaderContext context) {
        return new Leaf(context.reader());
    }

    @Override
    public Leaf loadDirect(LeafReaderContext context) {
        return load(context);
    }

    @Override
    public SortField sortField(Object missingValue, MultiValueMode sortMode, XFieldComparatorSource.Nested nested, boolean reverse) {
        throw new UnsupportedOperationException("knn vector field doesn't support this operation");
    }

    @Override
    public BucketedSort newBucketedSort(
        BigArrays bigArrays,
        Object missingValue,
        MultiValueMode sortMode,
        XFieldComparatorSource.Nested nested,
        SortOrder sortOrder,
        DocValueFormat format,
        int bucketSize,
        BucketedSort.ExtraData extra
    ) {
        throw new UnsupportedOperationException("knn vector field doesn't support this operation");
    }

    /** {@code KNNVectorDVLeafFieldData}. */
    public final class Leaf implements LeafFieldData {
        private final LeafReader reader;

        Leaf(LeafReader reader) {
            this.reader = reader;
        }

        @Override
        public long ramBytesUsed() {
            return 0;
        }

        @Override
        public void close() {}

        @Override
        public ScriptDocValues<?> getScriptValues() {
            throw new IllegalArgumentException("[" + KnnVectorFieldMapper.CONTENT_TYPE + "] fields do not support scripting yet");
        }

        @Override
        public SortedBinaryDocValues getBytesValues() {
            throw new UnsupportedOperationException("knn vector field '" + name + "' doesn't support sorting");
        }

        @Override
        public DocValueFetcher.Leaf getLeafValueFetcher(DocValueFormat format) {
            if (format instanceof Format == false) {
                throw new IllegalArgumentException(
                    "Unsupported DocValueFormat [" + format + "] for knn_vector field '" + name + "'. Expected KNNVectorDocValueFormat."
                );
            }
            Format f = (Format) format;
            FieldInfo info = reader.getFieldInfos().fieldInfo(name);
            if (info == null) {
                return EMPTY;
            }
            Values values;
            try {
                values = values(info);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            boolean binary = f.binary;
            boolean floats = config.dataType() == VectorEncoding.FLOAT32;
            return new DocValueFetcher.Leaf() {
                private int count;

                @Override
                public boolean advanceExact(int doc) throws IOException {
                    count = values.advance(doc) == doc ? 1 : 0;
                    return count == 1;
                }

                @Override
                public int docValueCount() {
                    return count;
                }

                @Override
                public Object nextValue() throws IOException {
                    Object v = values.vector();
                    if (floats) {
                        return binary ? littleEndian((float[]) v) : ((float[]) v).clone();
                    }
                    // A byte field without vector files reads its doc values as floats, as k-NN does.
                    if (v instanceof float[] fv) {
                        return fv.clone();
                    }
                    byte[] b = ((byte[]) v).clone();
                    return binary ? b : DerivedSource.signed(b);
                }
            };
        }

        /**
         * {@code KNNVectorValuesFactory.getVectorValues}: the vector files, else the binary doc
         * values decoded by the field's {@code data_type} attribute (float without one).
         */
        private Values values(FieldInfo info) throws IOException {
            if (info.hasVectorValues()) {
                if (info.getVectorEncoding() == VectorEncoding.BYTE) {
                    ByteVectorValues v = reader.getByteVectorValues(name);
                    KnnVectorValues.DocIndexIterator it = v.iterator();
                    return new Values(it, () -> v.vectorValue(it.index()));
                }
                FloatVectorValues v = reader.getFloatVectorValues(name);
                KnnVectorValues.DocIndexIterator it = v.iterator();
                return new Values(it, () -> v.vectorValue(it.index()));
            }
            BinaryDocValues dv = DocValues.getBinary(reader, name);
            String type = info.getAttribute(KnnVectorFieldMapper.DATA_TYPE);
            if ("half_float".equals(type)) {
                return new Values(dv, () -> halfFloats(dv.binaryValue()));
            }
            if ("byte".equals(type) || "binary".equals(type)) {
                return new Values(dv, () -> BytesRef.deepCopyOf(dv.binaryValue()).bytes);
            }
            return new Values(dv, () -> floats(dv.binaryValue()));
        }
    }

    private interface Vector {
        Object get() throws IOException;
    }

    private record Values(DocIdSetIterator it, Vector value) {
        int advance(int doc) throws IOException {
            return it.docID() >= doc ? it.docID() : it.advance(doc);
        }

        Object vector() throws IOException {
            return value.get();
        }
    }

    private static final DocValueFetcher.Leaf EMPTY = new DocValueFetcher.Leaf() {
        @Override
        public boolean advanceExact(int doc) {
            return false;
        }

        @Override
        public int docValueCount() {
            return 0;
        }

        @Override
        public Object nextValue() {
            return null;
        }
    };

    /** The builder {@code KNNVectorFieldType.fielddataBuilder} returns. */
    record Builder(String name, KnnFieldConfig config) implements IndexFieldData.Builder {
        @Override
        public IndexFieldData<?> build(IndexFieldDataCache cache, CircuitBreakerService breakerService) {
            return new KnnVectorFieldData(name, config);
        }
    }
}
