/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterCodecReader;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.opensearch.common.lucene.index.SequentialStoredFieldsLeafReader;
import org.opensearch.index.mapper.SeqNoFieldMapper;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * A reader that shows {@code _seq_no} indexed {@link SeqNoSettings#DOC_VALUES_ONLY} as the
 * one-dimensional long points OpenSearch's {@code SeqNoFieldMapper} queries: {@code
 * LongPoint.newRangeQuery}, {@code newExactQuery} and {@code newSetQuery} on {@code _seq_no} (a
 * {@code range}, {@code term} or {@code terms} search on it) rewrite and match exactly as over the
 * stock layout. The points are one leaf cell read off the doc values, its bounds taken from their
 * skip index; the native engine runs such queries on the doc values directly ({@link
 * SeqNoQueries#docValuesOnly}). Other segments and fields are passed through untouched.
 */
public final class SeqNoPointsReader extends FilterDirectoryReader {
    SeqNoPointsReader(DirectoryReader in) throws IOException {
        super(in, new SubReaderWrapper() {
            @Override
            public LeafReader wrap(LeafReader reader) {
                try {
                    return wrapLeaf(reader);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        });
    }

    /** {@code in}, wrapped when one of its leaves has doc-values-only {@code _seq_no} not yet shown as points. */
    public static DirectoryReader wrapIfNeeded(DirectoryReader in) throws IOException {
        for (LeafReaderContext leaf : in.leaves()) {
            if (needsPoints(leaf.reader().getFieldInfos())) {
                return new SeqNoPointsReader(in);
            }
        }
        return in;
    }

    private static boolean needsPoints(FieldInfos infos) {
        FieldInfo seqNo = infos.fieldInfo(SeqNoFieldMapper.NAME);
        return seqNo != null && seqNo.getPointDimensionCount() == 0 && seqNo.getDocValuesType() == DocValuesType.NUMERIC;
    }

    /** Whether {@code reader} (a segment, possibly wrapped) shows doc-values-only {@code _seq_no} as points. */
    static boolean synthesizes(LeafReader reader) {
        return reader instanceof SyntheticCodecLeaf || reader instanceof SyntheticLeaf;
    }

    private static LeafReader wrapLeaf(LeafReader reader) throws IOException {
        if (needsPoints(reader.getFieldInfos()) == false) {
            return reader;
        }
        FieldInfos infos = withPoints(reader.getFieldInfos(), reader.getFieldInfos().fieldInfo(SeqNoFieldMapper.NAME));
        if (reader instanceof CodecReader codec) {
            return new SyntheticCodecLeaf(codec, infos);
        }
        return new SyntheticLeaf(reader, infos);
    }

    /** {@code infos} with {@code seqNo} declaring one 8-byte point dimension. */
    private static FieldInfos withPoints(FieldInfos infos, FieldInfo seqNo) {
        FieldInfo[] out = new FieldInfo[infos.size()];
        int i = 0;
        for (FieldInfo fi : infos) {
            out[i++] = fi != seqNo
                ? fi
                : new FieldInfo(
                    fi.name,
                    fi.number,
                    fi.hasTermVectors(),
                    fi.omitsNorms(),
                    fi.hasPayloads(),
                    fi.getIndexOptions(),
                    fi.getDocValuesType(),
                    fi.docValuesSkipIndexType(),
                    fi.getDocValuesGen(),
                    fi.attributes(),
                    1,
                    1,
                    Long.BYTES,
                    fi.getVectorDimension(),
                    fi.getVectorEncoding(),
                    fi.getVectorSimilarityFunction(),
                    fi.isSoftDeletesField(),
                    fi.isParentField()
                );
        }
        return new FieldInfos(out);
    }

    @Override
    protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) throws IOException {
        return new SeqNoPointsReader(in);
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return in.getReaderCacheHelper();
    }

    /** A codec segment: still a {@link CodecReader}, as OpenSearch's fetch phase needs. */
    private static final class SyntheticCodecLeaf extends FilterCodecReader {
        private final FieldInfos infos;

        SyntheticCodecLeaf(CodecReader in, FieldInfos infos) {
            super(in);
            this.infos = infos;
        }

        @Override
        public FieldInfos getFieldInfos() {
            return infos;
        }

        @Override
        public PointsReader getPointsReader() {
            PointsReader real = in.getPointsReader();
            CodecReader segment = in;
            return new PointsReader() {
                @Override
                public void checkIntegrity() throws IOException {
                    if (real != null) {
                        real.checkIntegrity();
                    }
                }

                @Override
                public PointValues getValues(String field) throws IOException {
                    if (field.equals(SeqNoFieldMapper.NAME)) {
                        return DocValuesPoints.of(segment, field);
                    }
                    return real == null ? null : real.getValues(field);
                }

                @Override
                public void close() {}
            };
        }

        @Override
        public CacheHelper getCoreCacheHelper() {
            return in.getCoreCacheHelper();
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            return in.getReaderCacheHelper();
        }
    }

    /** Any other leaf, such as one of the writer's buffers: its stored fields still read sequentially. */
    private static final class SyntheticLeaf extends SequentialStoredFieldsLeafReader {
        private final FieldInfos infos;

        SyntheticLeaf(LeafReader in, FieldInfos infos) {
            super(in);
            this.infos = infos;
        }

        @Override
        public FieldInfos getFieldInfos() {
            return infos;
        }

        @Override
        protected StoredFieldsReader doGetSequentialStoredFieldsReader(StoredFieldsReader reader) {
            return reader;
        }

        @Override
        public PointValues getPointValues(String field) throws IOException {
            if (field.equals(SeqNoFieldMapper.NAME)) {
                return DocValuesPoints.of(in, field);
            }
            return in.getPointValues(field);
        }

        @Override
        public CacheHelper getCoreCacheHelper() {
            return in.getCoreCacheHelper();
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            return in.getReaderCacheHelper();
        }
    }

    /** A field's NUMERIC doc values as one leaf cell of one-dimensional long points. */
    static final class DocValuesPoints extends PointValues {
        private final LeafReader reader;
        private final String field;
        private final byte[] min;
        private final byte[] max;
        /** At least the documents with a value. */
        private final int size;
        private final int docCount;

        private DocValuesPoints(LeafReader reader, String field, long min, long max, int size, int docCount) {
            this.reader = reader;
            this.field = field;
            this.min = LongPoint.pack(min).bytes;
            this.max = LongPoint.pack(max).bytes;
            this.size = size;
            this.docCount = docCount;
        }

        /**
         * Null when no document has a value, as a segment without the field's points reads. Without
         * a skip index (a leaf of the writer's buffers) the bounds are the whole long range and no
         * document is counted as having the field, so a query's rewrite keeps it whole.
         */
        static PointValues of(LeafReader reader, String field) throws IOException {
            DocValuesSkipper skipper = reader.getDocValuesSkipper(field);
            if (skipper != null) {
                return skipper.docCount() == 0 ? null : new DocValuesPoints(reader, field, skipper.minValue(), skipper.maxValue(), skipper.docCount(), skipper.docCount());
            }
            if (reader.maxDoc() == 0 || reader.getNumericDocValues(field) == null) {
                return null;
            }
            return new DocValuesPoints(reader, field, Long.MIN_VALUE, Long.MAX_VALUE, reader.maxDoc(), 0);
        }

        @Override
        public PointTree getPointTree() {
            return new Leaf();
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
            return 1;
        }

        @Override
        public int getNumIndexDimensions() {
            return 1;
        }

        @Override
        public int getBytesPerDimension() {
            return Long.BYTES;
        }

        @Override
        public long size() {
            return size;
        }

        @Override
        public int getDocCount() {
            return docCount;
        }

        /** The one cell: every document with a value. */
        private final class Leaf implements PointTree {
            @Override
            public PointTree clone() {
                return new Leaf();
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
            public void visitDocIDs(IntersectVisitor visitor) throws IOException {
                NumericDocValues values = reader.getNumericDocValues(field);
                visitor.grow(size);
                for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
                    visitor.visit(doc);
                }
            }

            @Override
            public void visitDocValues(IntersectVisitor visitor) throws IOException {
                NumericDocValues values = reader.getNumericDocValues(field);
                byte[] packed = new byte[Long.BYTES];
                visitor.grow(size);
                for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
                    LongPoint.encodeDimension(values.longValue(), packed, 0);
                    visitor.visit(doc, packed);
                }
            }
        }
    }
}
