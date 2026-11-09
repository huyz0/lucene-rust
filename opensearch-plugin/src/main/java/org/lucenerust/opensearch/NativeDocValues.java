/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Doc values read a document at a time from the native reader: what OpenSearch reads per hit -- a
 * derived {@code _source}'s fields ({@code FieldValueFetcher}), {@code docvalue_fields} -- each
 * {@code advanceExact} answered from one native read of the document's values of every field the
 * leaf has been asked for ({@link Batch}), so deriving a hit's {@code _source} reads its doc
 * values in one call, as its stored fields are.
 *
 * <p>An iterator walked as a scan -- {@code nextDoc}, {@code advance}, or more than {@link
 * #POINT_READS} documents -- is a consumer reading the column (a sort or aggregation Lucene
 * answers), which the segment's own iterator serves; so are the calls with no document
 * ({@code lookupOrd} of an ordinal the current document lacks, {@code termsEnum}, {@code
 * getValueCount}), and every read the native side fails.
 */
final class NativeDocValues {
    /** {@code jvm_fetch.rs}'s kinds. */
    static final byte NONE = 0;
    static final byte SORTED_NUMERIC = 1;
    static final byte SORTED_SET = 2;

    /** Point reads an iterator answers natively before it is taken for a scan. */
    static final int POINT_READS = 128;

    private NativeDocValues() {}

    private static final VarHandle INT = MethodHandles.byteArrayViewVarHandle(int[].class, ByteOrder.LITTLE_ENDIAN);
    private static final VarHandle LONG = MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);

    /**
     * One document's values of some fields, read in place from the bytes the native side encodes:
     * per field its kind and where its values start, and nothing copied out until asked for.
     */
    static final class Values {
        final Thread thread;
        final int doc;
        final byte[] blob;
        final int[] fields;
        final byte[] kinds;
        /** Per field, the offset of its value count in {@link #blob}; -1 for {@link #NONE}. */
        final int[] offsets;

        private Values(Thread thread, int doc, byte[] blob, int[] fields, byte[] kinds, int[] offsets) {
            this.thread = thread;
            this.doc = doc;
            this.blob = blob;
            this.fields = fields;
            this.kinds = kinds;
            this.offsets = offsets;
        }

        /** {@code field}'s slot, or -1 when the read did not ask for it ({@code fields} is sorted: {@link Batch#wanted}). */
        int slot(int field) {
            int i = Arrays.binarySearch(fields, field);
            return i < 0 ? -1 : i;
        }

        /** The number of values at {@code slot}, of {@code kind}; 0 when it has none or is of another kind. */
        int count(int slot, byte kind) {
            return kinds[slot] == kind ? (int) INT.get(blob, offsets[slot]) : 0;
        }

        static Values decode(Thread thread, int doc, byte[] blob) throws IOException {
            try {
                int n = (int) INT.get(blob, 0);
                int[] fields = new int[n];
                byte[] kinds = new byte[n];
                int[] offsets = new int[n];
                int at = 4;
                for (int i = 0; i < n; i++) {
                    fields[i] = (int) INT.get(blob, at);
                    byte kind = blob[at + 4];
                    at += 5;
                    kinds[i] = kind;
                    switch (kind) {
                        case NONE -> offsets[i] = -1;
                        case SORTED_NUMERIC -> {
                            offsets[i] = at;
                            at += 4 + 8 * (int) INT.get(blob, at);
                        }
                        case SORTED_SET -> {
                            offsets[i] = at;
                            int count = (int) INT.get(blob, at);
                            at += 4;
                            for (int j = 0; j < count; j++) {
                                at += 12 + (int) INT.get(blob, at + 8);
                            }
                        }
                        default -> throw new IOException("native doc values of unknown kind " + kind);
                    }
                }
                if (at != blob.length) {
                    throw new IOException("native doc values: " + (blob.length - at) + " bytes left over");
                }
                return new Values(thread, doc, blob, fields, kinds, offsets);
            } catch (IndexOutOfBoundsException e) {
                throw new IOException("native doc values cut short", e);
            }
        }
    }

    /**
     * A document's encoded values of {@code fields}, from wherever the leaf's documents live; null
     * when they cannot be read natively, and the segment's own iterator answers instead.
     */
    @FunctionalInterface
    interface Source {
        byte[] read(int doc, int[] fields) throws IOException;
    }

    /**
     * The fields a segment's reads have asked for so far, shared by the leaves that read the
     * segment: OpenSearch wraps the reader again for every search, and a derived {@code _source}
     * asks for the same fields every hit, so once known each hit is one read.
     */
    static final class Wanted {
        private volatile int[] fields = new int[0];

        /** The fields, {@code field} among them: sorted, and never changed once returned. */
        int[] with(int field) {
            int[] w = fields;
            if (Arrays.binarySearch(w, field) >= 0) {
                return w;
            }
            synchronized (this) {
                w = fields;
                if (Arrays.binarySearch(w, field) < 0) {
                    w = Arrays.copyOf(w, w.length + 1);
                    w[w.length - 1] = field;
                    Arrays.sort(w);
                    fields = w;
                }
                return w;
            }
        }
    }

    /** One leaf's native reads: the segment's {@link Wanted} fields, and the last document read, kept for the thread that read it. */
    static final class Batch {
        private final Source source;
        private final Wanted wanted;
        private volatile Values last;

        Batch(Source source) {
            this(source, new Wanted());
        }

        Batch(Source source, Wanted wanted) {
            this.source = source;
            this.wanted = wanted;
        }

        /** Document {@code doc}'s values, {@code field}'s among them; null when the native read failed. */
        Values values(int doc, int field) throws IOException {
            Thread t = Thread.currentThread();
            Values v = last;
            if (v != null && v.doc == doc && v.thread == t && v.slot(field) >= 0) {
                return v;
            }
            byte[] blob = source.read(doc, wanted.with(field));
            if (blob == null) {
                return null;
            }
            v = Values.decode(t, doc, blob);
            last = v;
            return v;
        }
    }

    /** The segment's own iterator, made on first need; never null for a field of its type. */
    @FunctionalInterface
    interface Column<T> {
        T open() throws IOException;
    }

    /** {@code SortedNumericDocValues} read natively per document. */
    static final class Numbers extends SortedNumericDocValues {
        private final Batch batch;
        private final int field;
        private final Column<SortedNumericDocValues> column;
        /** For {@link #cost}: the segment's iterator, never positioned. */
        private SortedNumericDocValues probe;
        private SortedNumericDocValues scan;
        private int doc = -1;
        private int reads;
        /** The current document's values: {@code count} longs in {@code blob} from {@code at}. */
        private byte[] blob;
        private int at;
        private int count;
        private int upto;

        /** Point reads answered natively before handing over: {@link #POINT_READS}. */
        private final int pointReads;

        Numbers(Batch batch, int field, Column<SortedNumericDocValues> column) {
            this(batch, field, column, POINT_READS);
        }

        /** With no limit on the point reads where {@code column} holds nothing to scan. */
        Numbers(Batch batch, int field, Column<SortedNumericDocValues> column, int pointReads) {
            this.batch = batch;
            this.field = field;
            this.column = column;
            this.pointReads = pointReads;
        }

        /** The segment's iterator, from now on; not yet positioned. */
        private SortedNumericDocValues scan() throws IOException {
            if (scan == null) {
                scan = column.open();
                if (scan == null) {
                    throw new IllegalStateException("no column of doc values to scan for field number " + field);
                }
            }
            return scan;
        }

        /** {@code nextDoc} on the segment's iterator, which this one hands over to at {@link #doc}. */
        private int scanNext() throws IOException {
            boolean fresh = scan == null;
            SortedNumericDocValues s = scan();
            return fresh && doc >= 0 ? s.advance(doc + 1) : s.nextDoc();
        }

        @Override
        public boolean advanceExact(int target) throws IOException {
            Values v = scan != null || ++reads > pointReads ? null : batch.values(target, field);
            if (v == null) {
                boolean has = scan().advanceExact(target);
                doc = target;
                return has;
            }
            int slot = v.slot(field);
            doc = target;
            blob = v.blob;
            count = v.count(slot, SORTED_NUMERIC);
            at = v.offsets[slot] + 4;
            upto = 0;
            return count > 0;
        }

        @Override
        public long nextValue() throws IOException {
            if (scan != null) {
                return scan.nextValue();
            }
            if (upto >= count) {
                throw new IllegalStateException("no more values of document " + doc);
            }
            return (long) LONG.get(blob, at + 8 * upto++);
        }

        @Override
        public int docValueCount() {
            return scan != null ? scan.docValueCount() : count;
        }

        @Override
        public int docID() {
            return scan != null ? scan.docID() : doc;
        }

        @Override
        public int nextDoc() throws IOException {
            return doc = scanNext();
        }

        @Override
        public int advance(int target) throws IOException {
            return doc = scan().advance(target);
        }

        @Override
        public long cost() {
            try {
                if (scan != null) {
                    return scan.cost();
                }
                if (probe == null) {
                    probe = column.open();
                }
                return probe == null ? 0 : probe.cost();
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
    }

    /** {@code SortedSetDocValues} read natively per document. */
    static final class Terms extends SortedSetDocValues {
        private final Batch batch;
        private final int field;
        private final Column<SortedSetDocValues> column;
        /** For calls with no document, and {@link #cost}: ordinals and terms are the segment's own. */
        private SortedSetDocValues dictionary;
        private SortedSetDocValues scan;
        private int doc = -1;
        private int reads;
        /**
         * The current document's values in {@code blob}: {@code count} of them from {@code start},
         * each its ordinal, length and bytes; {@code at} is the next one {@link #nextOrd} reads.
         */
        private byte[] blob;
        private int start;
        private int count;
        private int at;
        private int upto;
        /** What {@link #lookupOrd} returns for the current document's terms, pointed at them in turn. */
        private final BytesRef term = new BytesRef();

        /** Point reads answered natively before handing over: {@link #POINT_READS}. */
        private final int pointReads;

        Terms(Batch batch, int field, Column<SortedSetDocValues> column) {
            this(batch, field, column, POINT_READS);
        }

        /** With no limit on the point reads where {@code column} holds nothing to scan. */
        Terms(Batch batch, int field, Column<SortedSetDocValues> column, int pointReads) {
            this.batch = batch;
            this.field = field;
            this.column = column;
            this.pointReads = pointReads;
        }

        private SortedSetDocValues dictionary() throws IOException {
            if (scan != null) {
                return scan;
            }
            if (dictionary == null) {
                dictionary = column.open();
                if (dictionary == null) {
                    throw new IllegalStateException("no column of doc values for field number " + field);
                }
            }
            return dictionary;
        }

        /** The segment's iterator, from now on; not yet positioned. */
        private SortedSetDocValues scan() throws IOException {
            if (scan == null) {
                scan = column.open();
                if (scan == null) {
                    throw new IllegalStateException("no column of doc values to scan for field number " + field);
                }
            }
            return scan;
        }

        /** {@code nextDoc} on the segment's iterator, which this one hands over to at {@link #doc}. */
        private int scanNext() throws IOException {
            boolean fresh = scan == null;
            SortedSetDocValues s = scan();
            return fresh && doc >= 0 ? s.advance(doc + 1) : s.nextDoc();
        }

        @Override
        public boolean advanceExact(int target) throws IOException {
            Values v = scan != null || ++reads > pointReads ? null : batch.values(target, field);
            if (v == null) {
                boolean has = scan().advanceExact(target);
                doc = target;
                return has;
            }
            int slot = v.slot(field);
            doc = target;
            blob = v.blob;
            count = v.count(slot, SORTED_SET);
            start = v.offsets[slot] + 4;
            at = start;
            upto = 0;
            return count > 0;
        }

        @Override
        public long nextOrd() throws IOException {
            if (scan != null) {
                return scan.nextOrd();
            }
            if (upto >= count) {
                throw new IllegalStateException("no more ordinals of document " + doc);
            }
            long ord = (long) LONG.get(blob, at);
            at += 12 + (int) INT.get(blob, at + 8);
            upto++;
            return ord;
        }

        @Override
        public int docValueCount() {
            return scan != null ? scan.docValueCount() : count;
        }

        @Override
        public BytesRef lookupOrd(long ord) throws IOException {
            if (scan == null) {
                // The current document's own terms come with it; any other ordinal is the segment's.
                for (int i = 0, p = start; i < count; i++) {
                    int len = (int) INT.get(blob, p + 8);
                    if ((long) LONG.get(blob, p) == ord) {
                        term.bytes = blob;
                        term.offset = p + 12;
                        term.length = len;
                        return term;
                    }
                    p += 12 + len;
                }
            }
            return dictionary().lookupOrd(ord);
        }

        @Override
        public long getValueCount() {
            try {
                return dictionary().getValueCount();
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }

        @Override
        public TermsEnum termsEnum() throws IOException {
            return dictionary().termsEnum();
        }

        @Override
        public long lookupTerm(BytesRef key) throws IOException {
            return dictionary().lookupTerm(key);
        }

        @Override
        public int docID() {
            return scan != null ? scan.docID() : doc;
        }

        @Override
        public int nextDoc() throws IOException {
            return doc = scanNext();
        }

        @Override
        public int advance(int target) throws IOException {
            return doc = scan().advance(target);
        }

        @Override
        public long cost() {
            try {
                return dictionary().cost();
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
    }
}
