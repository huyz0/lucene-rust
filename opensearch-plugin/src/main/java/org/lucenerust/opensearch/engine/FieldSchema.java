/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.index.IndexableFieldType;

import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * One field's schema as a document declares it: Lucene's {@code IndexingChain.FieldSchema}, which
 * folds every instance of a field name in one document into one schema and refuses instances that
 * disagree. Two schemas for the same name must be {@link #equals} across documents, as {@code
 * FieldSchema.assertSameSchema} requires.
 */
final class FieldSchema {
    /** Set at construction or by {@link #reset}; a registered schema is a {@link #copy} and never changes. */
    String name;
    IndexOptions indexOptions = IndexOptions.NONE;
    boolean omitNorms;
    boolean storeTermVectors;
    DocValuesType docValuesType = DocValuesType.NONE;
    DocValuesSkipIndexType skipIndex = DocValuesSkipIndexType.NONE;
    int pointDimensionCount;
    int pointIndexDimensionCount;
    int pointNumBytes;
    int vectorDimension;
    VectorEncoding vectorEncoding = VectorEncoding.FLOAT32;
    VectorSimilarityFunction vectorSimilarity = VectorSimilarityFunction.EUCLIDEAN;
    boolean softDeletes;
    /** The field types' own attributes ({@code FieldType.putAttribute}), as {@code IndexingChain} collects them. */
    final Map<String, String> attributes = new TreeMap<>();

    FieldSchema(String name) {
        this.name = name;
    }

    /** Back to a new schema for {@code name}, so a thread can reuse it for the next document. */
    void reset(String name) {
        this.name = name;
        indexOptions = IndexOptions.NONE;
        omitNorms = false;
        storeTermVectors = false;
        docValuesType = DocValuesType.NONE;
        skipIndex = DocValuesSkipIndexType.NONE;
        pointDimensionCount = 0;
        pointIndexDimensionCount = 0;
        pointNumBytes = 0;
        vectorDimension = 0;
        vectorEncoding = VectorEncoding.FLOAT32;
        vectorSimilarity = VectorSimilarityFunction.EUCLIDEAN;
        softDeletes = false;
        attributes.clear();
    }

    /** An independent copy, for the registry to keep while this one is reused. */
    FieldSchema copy() {
        FieldSchema c = new FieldSchema(name);
        c.indexOptions = indexOptions;
        c.omitNorms = omitNorms;
        c.storeTermVectors = storeTermVectors;
        c.docValuesType = docValuesType;
        c.skipIndex = skipIndex;
        c.pointDimensionCount = pointDimensionCount;
        c.pointIndexDimensionCount = pointIndexDimensionCount;
        c.pointNumBytes = pointNumBytes;
        c.vectorDimension = vectorDimension;
        c.vectorEncoding = vectorEncoding;
        c.vectorSimilarity = vectorSimilarity;
        c.softDeletes = softDeletes;
        c.attributes.putAll(attributes);
        return c;
    }

    /** {@code IndexingChain.updateDocFieldSchema}: folds one instance's type into this schema. */
    void absorb(IndexableFieldType type) {
        if (type.indexOptions() != IndexOptions.NONE) {
            if (indexOptions == IndexOptions.NONE) {
                indexOptions = type.indexOptions();
                omitNorms = type.omitNorms();
                storeTermVectors = type.storeTermVectors();
            } else {
                same("index options", indexOptions, type.indexOptions());
                same("omit norms", omitNorms, type.omitNorms());
                same("store term vector", storeTermVectors, type.storeTermVectors());
            }
        }
        if (type.docValuesType() != DocValuesType.NONE) {
            if (docValuesType == DocValuesType.NONE) {
                docValuesType = type.docValuesType();
                skipIndex = type.docValuesSkipIndexType();
            } else {
                same("doc values type", docValuesType, type.docValuesType());
                same("doc values skip index type", skipIndex, type.docValuesSkipIndexType());
            }
        }
        if (type.pointDimensionCount() != 0) {
            if (pointDimensionCount == 0) {
                pointDimensionCount = type.pointDimensionCount();
                pointIndexDimensionCount = type.pointIndexDimensionCount();
                pointNumBytes = type.pointNumBytes();
            } else {
                same("point dimension", pointDimensionCount, type.pointDimensionCount());
                same("point index dimension", pointIndexDimensionCount, type.pointIndexDimensionCount());
                same("point num bytes", pointNumBytes, type.pointNumBytes());
            }
        }
        if (type.vectorDimension() != 0) {
            if (vectorDimension == 0) {
                vectorDimension = type.vectorDimension();
                vectorEncoding = type.vectorEncoding();
                vectorSimilarity = type.vectorSimilarityFunction();
            } else {
                same("vector dimension", vectorDimension, type.vectorDimension());
                same("vector encoding", vectorEncoding, type.vectorEncoding());
                same("vector similarity function", vectorSimilarity, type.vectorSimilarityFunction());
            }
        }
        Map<String, String> a = type.getAttributes();
        if (a != null) {
            attributes.putAll(a);
        }
    }

    /** {@code FieldSchema.assertSameSchema}: this document's schema against the field's registered one. */
    void assertSameAs(FieldSchema registered) {
        same("index options", registered.indexOptions, indexOptions);
        same("omit norms", registered.omitNorms, omitNorms);
        same("store term vector", registered.storeTermVectors, storeTermVectors);
        same("doc values type", registered.docValuesType, docValuesType);
        same("doc values skip index type", registered.skipIndex, skipIndex);
        same("vector dimension", registered.vectorDimension, vectorDimension);
        if (vectorDimension != 0) {
            same("vector encoding", registered.vectorEncoding, vectorEncoding);
            same("vector similarity function", registered.vectorSimilarity, vectorSimilarity);
        }
        same("point dimension", registered.pointDimensionCount, pointDimensionCount);
        same("point index dimension", registered.pointIndexDimensionCount, pointIndexDimensionCount);
        same("point num bytes", registered.pointNumBytes, pointNumBytes);
    }

    /** {@code FieldSchema.raiseNotSame}'s message, less the in-buffer doc id this side never has. */
    private void same(String label, Object expected, Object given) {
        if (Objects.equals(expected, given) == false) {
            throw new IllegalArgumentException(
                "Inconsistency of field data structures across documents for field ["
                    + name
                    + "]. "
                    + label
                    + ": expected '"
                    + expected
                    + "', but it has '"
                    + given
                    + "'."
            );
        }
    }

    /**
     * The things the Rust writer does not write, refused before any byte reaches it: a document using
     * one fails, as a document Lucene cannot index does.
     */
    void checkSupported() {
        if (storeTermVectors) {
            throw new IllegalArgumentException("field [" + name + "]: the Rust engine does not index term vectors");
        }
        if (indexOptions == IndexOptions.DOCS_AND_CUSTOM_FREQS) {
            throw new IllegalArgumentException("field [" + name + "]: the Rust engine does not index DOCS_AND_CUSTOM_FREQS");
        }
    }

    /** The field spec {@code engine_writer.rs}'s {@code decode_field} reads. */
    byte[] encode() {
        Blob b = new Blob(64).string(name);
        b.u8(switch (indexOptions) {
            case NONE -> 0;
            case DOCS -> 1;
            case DOCS_AND_FREQS -> 2;
            case DOCS_AND_FREQS_AND_POSITIONS -> 3;
            case DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS -> 4;
            case DOCS_AND_CUSTOM_FREQS -> throw new IllegalArgumentException("unsupported index options");
        });
        b.bool(omitNorms).bool(storeTermVectors).bool(false);
        b.u8(switch (docValuesType) {
            case NONE -> 0;
            case NUMERIC -> 1;
            case BINARY -> 2;
            case SORTED -> 3;
            case SORTED_SET -> 4;
            case SORTED_NUMERIC -> 5;
        });
        b.bool(skipIndex != DocValuesSkipIndexType.NONE);
        b.i32(pointDimensionCount).i32(pointIndexDimensionCount).i32(pointNumBytes).i32(vectorDimension);
        if (vectorDimension != 0) {
            b.u8(vectorEncoding == VectorEncoding.BYTE ? 1 : 0);
            b.u8(vectorSimilarity.ordinal());
            VectorsFormatSpec.of(attributes.get(VectorsFormatSpec.ATTRIBUTE)).encode(b);
        }
        b.bool(softDeletes).i32(attributes.size());
        for (Map.Entry<String, String> a : attributes.entrySet()) {
            b.string(a.getKey()).string(a.getValue());
        }
        return b.toArray();
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof FieldSchema s) {
            return name.equals(s.name)
                && indexOptions == s.indexOptions
                && omitNorms == s.omitNorms
                && storeTermVectors == s.storeTermVectors
                && docValuesType == s.docValuesType
                && skipIndex == s.skipIndex
                && pointDimensionCount == s.pointDimensionCount
                && pointIndexDimensionCount == s.pointIndexDimensionCount
                && pointNumBytes == s.pointNumBytes
                && vectorDimension == s.vectorDimension
                && vectorEncoding == s.vectorEncoding
                && vectorSimilarity == s.vectorSimilarity
                && softDeletes == s.softDeletes
                && attributes.equals(s.attributes);
        }
        return false;
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, indexOptions, docValuesType, pointDimensionCount);
    }

    @Override
    public String toString() {
        return name
            + "{index="
            + indexOptions
            + ", omitNorms="
            + omitNorms
            + ", dv="
            + docValuesType
            + ", points="
            + pointDimensionCount
            + "/"
            + pointIndexDimensionCount
            + "/"
            + pointNumBytes
            + (softDeletes ? ", soft" : "")
            + "}";
    }
}
