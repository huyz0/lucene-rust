/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PayloadAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.TermFrequencyAttribute;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInvertState;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.document.InvertableType;
import org.apache.lucene.document.StoredValue;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefHash;
import org.apache.lucene.util.IntsRefBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Turns OpenSearch's Lucene documents into the operation blobs {@code engine_writer.rs} decodes.
 *
 * <p>Everything that decides <em>what</em> gets indexed happens here, in the JVM, with the shard's
 * own analyzer and similarity, exactly as Lucene's {@code IndexingChain} would: the per-document
 * field schema ({@code FieldSchema}), inversion ({@code PerField.invert}: positions, offsets, the
 * position and offset gaps between instances of one field, overlap counting), the per-term
 * frequencies ({@code FreqProxTermsWriterPerField}) and the norm ({@code PerField.finish}, which
 * calls {@link Similarity#computeNorm} on the same {@link FieldInvertState}). The Rust side only
 * lays the result out on disk. The error messages are Lucene's, since OpenSearch shows them to users.
 *
 * <p>Stateless apart from the registry, so any number of indexing threads can share it.
 */
final class DocumentEncoder {
    static final int OP_ADD = 0;
    static final int OP_SOFT_UPDATE = 1;
    static final int OP_PANIC = 0xFF;

    /** Answers whether the Rust writer lays out this field's formats; null means yes. */
    interface FormatCheck {
        String unsupported(String field, FieldSchema schema);
    }

    private final FieldRegistry registry;
    private final Analyzer analyzer;
    private final Similarity similarity;
    private final int indexCreatedVersionMajor;
    private final String softDeletesField;
    private final FormatCheck formats;

    DocumentEncoder(
        FieldRegistry registry,
        Analyzer analyzer,
        Similarity similarity,
        int indexCreatedVersionMajor,
        String softDeletesField,
        FormatCheck formats
    ) {
        this.registry = registry;
        this.analyzer = analyzer;
        this.similarity = similarity;
        this.indexCreatedVersionMajor = indexCreatedVersionMajor;
        this.softDeletesField = softDeletesField;
        this.formats = formats;
    }

    /** An add of {@code docs} as one block. */
    Blob add(List<? extends Iterable<? extends IndexableField>> docs) throws IOException {
        Blob b = new Blob(256 * docs.size()).u8(OP_ADD);
        return documents(b, docs);
    }

    /**
     * {@code softUpdateDocuments(term, docs, softDeletes)}: live documents matching {@code term} get
     * {@code softDeletes}' value, and {@code docs} are added, atomically.
     */
    Blob softUpdate(Term term, List<? extends Iterable<? extends IndexableField>> docs, IndexableField softDeletes)
        throws IOException {
        if (softDeletes.name().equals(softDeletesField) == false || softDeletes.numericValue() == null) {
            throw new IllegalArgumentException("a soft update needs a numeric [" + softDeletesField + "] value");
        }
        Blob b = new Blob(256 * docs.size() + 64).u8(OP_SOFT_UPDATE);
        b.string(term.field()).bytes(term.bytes());
        b.string(softDeletes.name()).i64(softDeletes.numericValue().longValue());
        return documents(b, docs);
    }

    private Blob documents(Blob b, List<? extends Iterable<? extends IndexableField>> docs) throws IOException {
        if (docs.isEmpty()) {
            throw new IllegalArgumentException("no documents");
        }
        b.i32(docs.size());
        for (Iterable<? extends IndexableField> doc : docs) {
            document(b, doc);
        }
        return b;
    }

    /** One field name's state within one document. */
    private static final class PerField {
        final FieldSchema schema;
        int number;
        final List<IndexableField> instances = new ArrayList<>(1);
        Inverted inverted;

        PerField(String name) {
            schema = new FieldSchema(name);
        }
    }

    private void document(Blob b, Iterable<? extends IndexableField> doc) throws IOException {
        // Pass 1: the document's schema per field name, checked against (or registered as) the
        // shard's -- IndexingChain.processDocument's first loop.
        Map<String, PerField> fields = new LinkedHashMap<>();
        for (IndexableField field : doc) {
            PerField pf = fields.computeIfAbsent(field.name(), PerField::new);
            pf.schema.absorb(field.fieldType());
            pf.instances.add(field);
        }
        for (PerField pf : fields.values()) {
            if (pf.schema.name.equals(softDeletesField)) {
                pf.schema.softDeletes = true;
            }
            String unsupported = formats == null ? null : formats.unsupported(pf.schema.name, pf.schema);
            if (unsupported != null) {
                throw new IllegalArgumentException(unsupported);
            }
            pf.number = registry.resolve(pf.schema).number();
        }

        // Pass 2, in document order: stored values, inversion, doc values, points.
        int storedAt = b.reserveI32();
        int stored = 0;
        List<long[]> numericDv = new ArrayList<>();
        List<Object[]> binaryDv = new ArrayList<>();
        List<Object[]> points = new ArrayList<>();
        for (IndexableField field : doc) {
            PerField pf = fields.get(field.name());
            IndexableFieldType type = field.fieldType();
            if (type.indexOptions() != IndexOptions.NONE) {
                if (pf.inverted == null) {
                    pf.inverted = new Inverted(pf.schema.name, type.indexOptions());
                }
                invert(pf.inverted, field);
            }
            if (type.stored()) {
                storedValue(b, pf.number, field);
                stored++;
            }
            DocValuesType dv = type.docValuesType();
            if (dv != DocValuesType.NONE) {
                switch (dv) {
                    case NUMERIC, SORTED_NUMERIC -> {
                        Number n = field.numericValue();
                        if (n == null) {
                            throw new IllegalArgumentException("field=\"" + pf.schema.name + "\": null value not allowed");
                        }
                        numericDv.add(new long[] { pf.number, n.longValue() });
                    }
                    default -> binaryDv.add(new Object[] { pf.number, BytesRef.deepCopyOf(field.binaryValue()) });
                }
            }
            if (type.pointDimensionCount() != 0) {
                points.add(new Object[] { pf.number, BytesRef.deepCopyOf(field.binaryValue()) });
            }
        }
        b.patchI32(storedAt, stored);

        int invertedCount = 0;
        for (PerField pf : fields.values()) {
            if (pf.inverted != null) {
                invertedCount++;
            }
        }
        b.i32(invertedCount);
        for (PerField pf : fields.values()) {
            if (pf.inverted != null) {
                pf.inverted.write(b, pf.number, pf.schema.omitNorms ? null : similarity, indexCreatedVersionMajor);
            }
        }

        b.i32(numericDv.size() + binaryDv.size());
        for (long[] v : numericDv) {
            b.i32((int) v[0]).u8(0).i64(v[1]);
        }
        for (Object[] v : binaryDv) {
            b.i32((Integer) v[0]).u8(1).bytes((BytesRef) v[1]);
        }
        b.i32(points.size());
        for (Object[] v : points) {
            b.i32((Integer) v[0]).bytes((BytesRef) v[1]);
        }
    }

    private static void storedValue(Blob b, int number, IndexableField field) throws IOException {
        StoredValue v = field.storedValue();
        if (v == null) {
            throw new IllegalArgumentException("Cannot store a null value");
        }
        b.i32(number);
        switch (v.getType()) {
            case STRING -> {
                String s = v.getStringValue();
                if (s.length() > IndexWriter.MAX_STORED_STRING_LENGTH) {
                    throw new IllegalArgumentException(
                        "stored field \"" + field.name() + "\" is too large (" + s.length() + " characters) to store"
                    );
                }
                b.u8(0).string(s);
            }
            case BINARY -> b.u8(1).bytes(v.getBinaryValue());
            case INTEGER -> b.u8(2).i32(4).i32(v.getIntValue());
            case LONG -> b.u8(3).i32(8).i64(v.getLongValue());
            case FLOAT -> b.u8(4).i32(4).i32(Float.floatToRawIntBits(v.getFloatValue()));
            case DOUBLE -> b.u8(5).i32(8).i64(Double.doubleToRawLongBits(v.getDoubleValue()));
            case DATA_INPUT -> {
                var in = v.getDataInputValue();
                byte[] bytes = new byte[in.getLength()];
                in.getDataInput().readBytes(bytes, 0, bytes.length);
                b.u8(1).bytes(bytes, 0, bytes.length);
            }
        }
    }

    /**
     * {@code PerField.invert}: one instance of an indexed field, appended to the field's state for
     * this document.
     */
    private void invert(Inverted inv, IndexableField field) throws IOException {
        if (field.invertableType() == InvertableType.BINARY) {
            invertTerm(inv, field);
            return;
        }
        IndexableFieldType type = field.fieldType();
        boolean analyzed = type.tokenized() && analyzer != null;
        try (TokenStream stream = field.tokenStream(analyzer, null)) {
            stream.reset();
            TermToBytesRefAttribute termAtt = stream.getAttribute(TermToBytesRefAttribute.class);
            PositionIncrementAttribute posIncrAtt = stream.addAttribute(PositionIncrementAttribute.class);
            OffsetAttribute offsetAtt = stream.addAttribute(OffsetAttribute.class);
            TermFrequencyAttribute termFreqAtt = stream.addAttribute(TermFrequencyAttribute.class);
            PayloadAttribute payloadAtt = stream.hasAttribute(PayloadAttribute.class) ? stream.getAttribute(PayloadAttribute.class) : null;
            while (stream.incrementToken()) {
                int posIncr = posIncrAtt.getPositionIncrement();
                inv.position += posIncr;
                if (inv.position < inv.lastPosition) {
                    if (posIncr == 0) {
                        throw new IllegalArgumentException("first position increment must be > 0 (got 0) for field '" + field.name() + "'");
                    } else if (posIncr < 0) {
                        throw new IllegalArgumentException(
                            "position increment must be >= 0 (got " + posIncr + ") for field '" + field.name() + "'"
                        );
                    } else {
                        throw new IllegalArgumentException(
                            "position overflowed Integer.MAX_VALUE (got posIncr="
                                + posIncr
                                + " lastPosition="
                                + inv.lastPosition
                                + " position="
                                + inv.position
                                + ") for field '"
                                + field.name()
                                + "'"
                        );
                    }
                } else if (inv.position > IndexWriter.MAX_POSITION) {
                    throw new IllegalArgumentException(
                        "position "
                            + inv.position
                            + " is too large for field '"
                            + field.name()
                            + "': max allowed position is "
                            + IndexWriter.MAX_POSITION
                    );
                }
                inv.lastPosition = inv.position;
                if (posIncr == 0) {
                    inv.numOverlap++;
                }
                int startOffset = inv.offset + offsetAtt.startOffset();
                int endOffset = inv.offset + offsetAtt.endOffset();
                if (startOffset < inv.lastStartOffset || endOffset < startOffset) {
                    throw new IllegalArgumentException(
                        "startOffset must be non-negative, and endOffset must be >= startOffset, and offsets must not go backwards "
                            + "startOffset="
                            + startOffset
                            + ",endOffset="
                            + endOffset
                            + ",lastStartOffset="
                            + inv.lastStartOffset
                            + " for field '"
                            + field.name()
                            + "'"
                    );
                }
                inv.lastStartOffset = startOffset;
                int termFreq = termFreqAtt.getTermFrequency();
                try {
                    inv.length = Math.addExact(inv.length, termFreq);
                } catch (ArithmeticException ae) {
                    throw new IllegalArgumentException("too many tokens for field \"" + field.name() + "\"", ae);
                }
                if (payloadAtt != null) {
                    BytesRef payload = payloadAtt.getPayload();
                    if (payload != null && payload.length > 0) {
                        throw new IllegalArgumentException("field [" + field.name() + "]: the Rust engine does not index payloads");
                    }
                }
                inv.add(termAtt.getBytesRef(), termFreq, startOffset, endOffset, field.name());
            }
            stream.end();
            inv.position += posIncrAtt.getPositionIncrement();
            inv.offset += offsetAtt.endOffset();
        }
        if (analyzed) {
            inv.position += analyzer.getPositionIncrementGap(inv.name);
            inv.offset += analyzer.getOffsetGap(inv.name);
        }
    }

    /** {@code PerField.invertTerm}, including its double {@code length} increment. */
    private void invertTerm(Inverted inv, IndexableField field) {
        BytesRef value = field.binaryValue();
        if (value == null) {
            throw new IllegalArgumentException(
                "Field " + field.name() + " returns TERM for invertableType() and null for binaryValue(), which is illegal"
            );
        }
        IndexableFieldType type = field.fieldType();
        if (type.tokenized()
            || type.indexOptions().compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) >= 0
            || type.storeTermVectorPositions()
            || type.storeTermVectorOffsets()
            || type.storeTermVectorPayloads()) {
            throw new IllegalArgumentException(
                "Fields that are tokenized or index proximity data must produce a non-null TokenStream, but "
                    + field.name()
                    + " did not"
            );
        }
        inv.position++;
        inv.length++;
        inv.length = Math.addExact(inv.length, 1);
        inv.add(value, 1, 0, 0, field.name());
    }

    /**
     * One field's inversion within one document: {@code FieldInvertState} plus the terms
     * {@code FreqProxTermsWriterPerField} would have buffered for it.
     */
    private static final class Inverted {
        final String name;
        final IndexOptions options;
        final boolean hasFreq;
        final boolean hasProx;
        final boolean hasOffsets;
        int position = -1;
        int length;
        int numOverlap;
        int offset;
        int maxTermFrequency;
        int uniqueTermCount;
        int lastStartOffset;
        int lastPosition;

        final BytesRefHash terms = new BytesRefHash();
        int[] freqs = new int[8];
        /** Per term: positions, and (start, end) offsets interleaved after them, when indexed. */
        IntsRefBuilder[] prox = new IntsRefBuilder[8];

        Inverted(String name, IndexOptions options) {
            this.name = name;
            this.options = options;
            hasFreq = options.compareTo(IndexOptions.DOCS_AND_FREQS) >= 0;
            hasProx = options.compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) >= 0;
            hasOffsets = options.compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS) >= 0;
        }

        /** {@code TermsHashPerField.add}: {@code newTerm} or {@code addTerm} for this document. */
        void add(BytesRef term, int termFreq, int startOffset, int endOffset, String fieldName) {
            if (term.length > IndexWriter.MAX_TERM_LENGTH) {
                byte[] prefix = Arrays.copyOfRange(term.bytes, term.offset, term.offset + 30);
                throw new IllegalArgumentException(
                    "Document contains at least one immense term in field=\""
                        + fieldName
                        + "\" (whose UTF8 encoding is longer than the max length "
                        + IndexWriter.MAX_TERM_LENGTH
                        + "), all of which were skipped.  Please correct the analyzer to not produce such terms.  The prefix of "
                        + "the first immense term is: '"
                        + Arrays.toString(prefix)
                        + "...'"
                );
            }
            if (termFreq != 1 && hasProx) {
                throw new IllegalStateException(
                    "field \"" + fieldName + "\": cannot index positions while using custom TermFrequencyAttribute"
                );
            }
            int id = terms.add(term);
            if (id >= 0) {
                if (id >= freqs.length) {
                    freqs = org.apache.lucene.util.ArrayUtil.grow(freqs, id + 1);
                    prox = org.apache.lucene.util.ArrayUtil.grow(prox, id + 1);
                }
                freqs[id] = hasFreq ? termFreq : 1;
                maxTermFrequency = hasFreq ? Math.max(freqs[id], maxTermFrequency) : Math.max(1, maxTermFrequency);
                uniqueTermCount++;
                prox[id] = null;
            } else {
                id = -id - 1;
                if (hasFreq == false) {
                    if (termFreq != 1) {
                        throw new IllegalStateException(
                            "field \"" + fieldName + "\": must index term freq while using custom TermFrequencyAttribute"
                        );
                    }
                } else {
                    freqs[id] = Math.addExact(freqs[id], termFreq);
                    maxTermFrequency = Math.max(maxTermFrequency, freqs[id]);
                }
            }
            if (hasProx) {
                if (prox[id] == null) {
                    prox[id] = new IntsRefBuilder();
                }
                prox[id].append(position);
                if (hasOffsets) {
                    prox[id].append(startOffset);
                    prox[id].append(endOffset);
                }
            }
        }

        /** The field's entry of the blob, norm computed as {@code PerField.finish} does. */
        void write(Blob b, int number, Similarity similarity, int indexCreatedVersionMajor) {
            b.i32(number);
            if (similarity == null) {
                b.u8(0);
            } else {
                long norm;
                if (length == 0) {
                    norm = 0;
                } else {
                    FieldInvertState state = new FieldInvertState(
                        indexCreatedVersionMajor,
                        name,
                        options,
                        position,
                        length,
                        numOverlap,
                        offset,
                        maxTermFrequency,
                        uniqueTermCount
                    );
                    norm = similarity.computeNorm(state);
                    if (norm == 0) {
                        throw new IllegalStateException("Similarity " + similarity + " return 0 for non-empty field");
                    }
                }
                b.u8(1).i64(norm);
            }
            int n = terms.size();
            b.i32(n);
            BytesRef scratch = new BytesRef();
            for (int id = 0; id < n; id++) {
                b.bytes(terms.get(id, scratch));
                int freq = freqs[id];
                b.i32(freq);
                b.u8((hasProx ? 1 : 0) | (hasOffsets ? 2 : 0));
                if (hasProx) {
                    int[] p = prox[id].ints();
                    int stride = hasOffsets ? 3 : 1;
                    for (int i = 0; i < freq; i++) {
                        b.i32(p[i * stride]);
                    }
                    if (hasOffsets) {
                        for (int i = 0; i < freq; i++) {
                            b.i32(p[i * 3 + 1]).i32(p[i * 3 + 2]);
                        }
                    }
                }
            }
        }
    }
}
