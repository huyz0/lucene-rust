/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.ByteBuffer;
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
 * getValueCount}). A leaf with no such iterator (the writer's buffered documents) answers them as
 * a column that holds only the documents read.
 */
final class NativeDocValues {
    /** {@code jvm_fetch.rs}'s kinds. */
    static final byte NONE = 0;
    static final byte SORTED_NUMERIC = 1;
    static final byte SORTED_SET = 2;

    /** Point reads an iterator answers natively before it is taken for a scan. */
    static final int POINT_READS = 128;

    private NativeDocValues() {}

    /** One document's values of some fields, as the native side encodes them. */
    static final class Values {
        final Thread thread;
        final int doc;
        final int[] fields;
        final long[][] numbers;
        final long[][] ords;
        final BytesRef[][] terms;

        private Values(Thread thread, int doc, int[] fields, long[][] numbers, long[][] ords, BytesRef[][] terms) {
            this.thread = thread;
            this.doc = doc;
            this.fields = fields;
            this.numbers = numbers;
            this.ords = ords;
            this.terms = terms;
        }

        /** {@code field}'s slot, or -1 when the read did not ask for it. */
        int slot(int field) {
            for (int i = 0; i < fields.length; i++) {
                if (fields[i] == field) {
                    return i;
                }
            }
            return -1;
        }

        static Values decode(Thread thread, int doc, byte[] blob) throws IOException {
            ByteBuffer b = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
            int n = b.getInt();
            int[] fields = new int[n];
            long[][] numbers = new long[n][];
            long[][] ords = new long[n][];
            BytesRef[][] terms = new BytesRef[n][];
            for (int i = 0; i < n; i++) {
                fields[i] = b.getInt();
                byte kind = b.get();
                switch (kind) {
                    case NONE -> {}
                    case SORTED_NUMERIC -> {
                        long[] v = new long[b.getInt()];
                        for (int j = 0; j < v.length; j++) {
                            v[j] = b.getLong();
                        }
                        numbers[i] = v;
                    }
                    case SORTED_SET -> {
                        int count = b.getInt();
                        long[] o = new long[count];
                        BytesRef[] t = new BytesRef[count];
                        for (int j = 0; j < count; j++) {
                            o[j] = b.getLong();
                            int len = b.getInt();
                            t[j] = new BytesRef(blob, b.position(), len);
                            b.position(b.position() + len);
                        }
                        ords[i] = o;
                        terms[i] = t;
                    }
                    default -> throw new IOException("native doc values of unknown kind " + kind);
                }
            }
            return new Values(thread, doc, fields, numbers, ords, terms);
        }
    }

    /** A document's encoded values of {@code fields}, from wherever the leaf's documents live. */
    @FunctionalInterface
    interface Source {
        byte[] read(int doc, int[] fields) throws IOException;
    }

    /**
     * One leaf's native reads: the fields asked for so far (a derived {@code _source} asks for the
     * same ones every hit, so from the second hit on each is one read), and the last document read,
     * kept for the thread that read it.
     */
    static final class Batch {
        private final Source source;
        private volatile int[] wanted = new int[0];
        private volatile Values last;

        Batch(Source source) {
            this.source = source;
        }

        /** Document {@code doc}'s values, {@code field}'s among them. */
        Values values(int doc, int field) throws IOException {
            Thread t = Thread.currentThread();
            Values v = last;
            if (v != null && v.doc == doc && v.thread == t && v.slot(field) >= 0) {
                return v;
            }
            int[] w = wanted;
            if (Arrays.binarySearch(w, field) < 0) {
                synchronized (this) {
                    w = wanted;
                    if (Arrays.binarySearch(w, field) < 0) {
                        w = Arrays.copyOf(w, w.length + 1);
                        w[w.length - 1] = field;
                        Arrays.sort(w);
                        wanted = w;
                    }
                }
            }
            v = Values.decode(t, doc, source.read(doc, w));
            last = v;
            return v;
        }
    }

    /** The segment's own iterator, made on first need. */
    @FunctionalInterface
    interface Column<T> {
        T open() throws IOException;
    }

    /** {@code SortedNumericDocValues} read natively per document. */
    static final class Numbers extends SortedNumericDocValues {
        private final Batch batch;
        private final int field;
        private final Column<SortedNumericDocValues> column;
        private SortedNumericDocValues scan;
        private int doc = -1;
        private int reads;
        private long[] values;
        private int upto;

        Numbers(Batch batch, int field, Column<SortedNumericDocValues> column) {
            this.batch = batch;
            this.field = field;
            this.column = column;
        }

        /** The segment's iterator, from now on; not yet positioned. */
        private SortedNumericDocValues scan() throws IOException {
            if (scan == null) {
                scan = column == null ? null : column.open();
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

        /** The segment's iterator's cost, without handing over to it. */
        private long costOf() {
            try {
                if (scan != null) {
                    return scan.cost();
                }
                SortedNumericDocValues c = column == null ? null : column.open();
                return c == null ? 0 : c.cost();
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }

        @Override
        public boolean advanceExact(int target) throws IOException {
            if (scan != null || (++reads > POINT_READS && column != null)) {
                boolean has = scan().advanceExact(target);
                doc = target;
                return has;
            }
            Values v = batch.values(target, field);
            doc = target;
            values = v.numbers[v.slot(field)];
            upto = 0;
            return values != null;
        }

        @Override
        public long nextValue() throws IOException {
            return scan != null ? scan.nextValue() : values[upto++];
        }

        @Override
        public int docValueCount() {
            return scan != null ? scan.docValueCount() : values.length;
        }

        @Override
        public int docID() {
            return scan != null ? scan.docID() : doc;
        }

        @Override
        public int nextDoc() throws IOException {
            if (column == null) {
                return doc = NO_MORE_DOCS;
            }
            return doc = scanNext();
        }

        @Override
        public int advance(int target) throws IOException {
            if (column == null) {
                return doc = NO_MORE_DOCS;
            }
            return doc = scan().advance(target);
        }

        @Override
        public long cost() {
            return costOf();
        }
    }

    /** {@code SortedSetDocValues} read natively per document. */
    static final class Terms extends SortedSetDocValues {
        private final Batch batch;
        private final int field;
        private final Column<SortedSetDocValues> column;
        /** For calls with no document: ordinals and terms are the segment's own. */
        private SortedSetDocValues dictionary;
        private SortedSetDocValues scan;
        private int doc = -1;
        private int reads;
        private long[] ords;
        private BytesRef[] terms;
        private int upto;

        Terms(Batch batch, int field, Column<SortedSetDocValues> column) {
            this.batch = batch;
            this.field = field;
            this.column = column;
        }

        private SortedSetDocValues dictionary() throws IOException {
            if (scan != null) {
                return scan;
            }
            if (dictionary == null && column != null) {
                dictionary = column.open();
            }
            return dictionary;
        }

        /** The segment's iterator, from now on; not yet positioned. */
        private SortedSetDocValues scan() throws IOException {
            if (scan == null) {
                scan = column == null ? null : column.open();
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

        /** The segment's iterator's cost, without handing over to it. */
        private long costOf() {
            try {
                if (scan != null) {
                    return scan.cost();
                }
                SortedSetDocValues c = column == null ? null : column.open();
                return c == null ? 0 : c.cost();
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }

        @Override
        public boolean advanceExact(int target) throws IOException {
            if (scan != null || (++reads > POINT_READS && column != null)) {
                boolean has = scan().advanceExact(target);
                doc = target;
                return has;
            }
            Values v = batch.values(target, field);
            int slot = v.slot(field);
            doc = target;
            ords = v.ords[slot];
            terms = v.terms[slot];
            upto = 0;
            return ords != null;
        }

        @Override
        public long nextOrd() throws IOException {
            return scan != null ? scan.nextOrd() : ords[upto++];
        }

        @Override
        public int docValueCount() {
            return scan != null ? scan.docValueCount() : ords.length;
        }

        @Override
        public BytesRef lookupOrd(long ord) throws IOException {
            if (scan == null && ords != null) {
                for (int i = 0; i < ords.length; i++) {
                    if (ords[i] == ord) {
                        return terms[i];
                    }
                }
            }
            SortedSetDocValues d = dictionary();
            if (d == null) {
                throw new IllegalArgumentException("ordinal " + ord + " is not the current document's");
            }
            return d.lookupOrd(ord);
        }

        @Override
        public long getValueCount() {
            try {
                SortedSetDocValues d = dictionary();
                return d == null ? (ords == null ? 0 : ords[ords.length - 1] + 1) : d.getValueCount();
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }

        @Override
        public TermsEnum termsEnum() throws IOException {
            SortedSetDocValues d = dictionary();
            return d == null ? TermsEnum.EMPTY : d.termsEnum();
        }

        @Override
        public long lookupTerm(BytesRef key) throws IOException {
            SortedSetDocValues d = dictionary();
            return d == null ? -1 : d.lookupTerm(key);
        }

        @Override
        public int docID() {
            return scan != null ? scan.docID() : doc;
        }

        @Override
        public int nextDoc() throws IOException {
            if (column == null) {
                return doc = DocIdSetIterator.NO_MORE_DOCS;
            }
            return doc = scanNext();
        }

        @Override
        public int advance(int target) throws IOException {
            if (column == null) {
                return doc = DocIdSetIterator.NO_MORE_DOCS;
            }
            return doc = scan().advance(target);
        }

        @Override
        public long cost() {
            return costOf();
        }
    }
}
