/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesSkipper;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterCodecReader;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.search.DocIdSetIterator;
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
final class SeqNoPointsReader extends FilterDirectoryReader {
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

    /** Whether {@code reader} (a segment, possibly wrapped) shows doc-values-only {@code _seq_no} as points. */
    static boolean synthesizes(LeafReader reader) {
        return reader instanceof SyntheticCodecLeaf || reader instanceof SyntheticLeaf;
    }

    private static LeafReader wrapLeaf(LeafReader reader) throws IOException {
        FieldInfo seqNo = reader.getFieldInfos().fieldInfo(SeqNoFieldMapper.NAME);
        if (seqNo == null || seqNo.getPointDimensionCount() != 0 || seqNo.getDocValuesType() != DocValuesType.NUMERIC) {
            return reader;
        }
        FieldInfos infos = withPoints(reader.getFieldInfos(), seqNo);
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

    /** Any other leaf. */
    private static final class SyntheticLeaf extends FilterLeafReader {
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
        private final int docCount;

        private DocValuesPoints(LeafReader reader, String field, long min, long max, int docCount) {
            this.reader = reader;
            this.field = field;
            this.min = LongPoint.pack(min).bytes;
            this.max = LongPoint.pack(max).bytes;
            this.docCount = docCount;
        }

        /** Null when no document has a value, as a segment without the field's points reads. */
        static PointValues of(LeafReader reader, String field) throws IOException {
            DocValuesSkipper skipper = reader.getDocValuesSkipper(field);
            if (skipper != null) {
                return skipper.docCount() == 0 ? null : new DocValuesPoints(reader, field, skipper.minValue(), skipper.maxValue(), skipper.docCount());
            }
            NumericDocValues values = reader.getNumericDocValues(field);
            if (values == null) {
                return null;
            }
            long min = Long.MAX_VALUE;
            long max = Long.MIN_VALUE;
            int count = 0;
            for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
                long v = values.longValue();
                min = Math.min(min, v);
                max = Math.max(max, v);
                count++;
            }
            return count == 0 ? null : new DocValuesPoints(reader, field, min, max, count);
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
            return docCount;
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
                return docCount;
            }

            @Override
            public void visitDocIDs(IntersectVisitor visitor) throws IOException {
                NumericDocValues values = reader.getNumericDocValues(field);
                visitor.grow(docCount);
                for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
                    visitor.visit(doc);
                }
            }

            @Override
            public void visitDocValues(IntersectVisitor visitor) throws IOException {
                NumericDocValues values = reader.getNumericDocValues(field);
                byte[] packed = new byte[Long.BYTES];
                visitor.grow(docCount);
                for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
                    LongPoint.encodeDimension(values.longValue(), packed, 0);
                    visitor.visit(doc, packed);
                }
            }
        }
    }
}
