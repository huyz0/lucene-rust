/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;

/**
 * A reader over a commit with soft deletes applied the way {@code IndexWriter}'s NRT readers apply
 * them: a document carrying the soft-deletes field is not live, and a segment all of whose
 * documents are deleted <em>stays in the reader</em>.
 *
 * <p>Lucene's {@code SoftDeletesDirectoryReaderWrapper} drops such a segment. That is harmless for
 * search, but a segment the retention policy keeps is kept for its history -- a lone delete
 * tombstone, a batch of superseded versions -- and {@code LuceneChangesSnapshot}, which reads every
 * document through {@code Lucene.wrapAllDocsLive}, must still find it there; peer recovery replays
 * those operations to replicas.
 */
final class SoftDeletesReader extends FilterDirectoryReader {
    private final String field;

    SoftDeletesReader(DirectoryReader in, String field) throws IOException {
        super(in, new SubReaderWrapper() {
            @Override
            public LeafReader wrap(LeafReader reader) {
                try {
                    return applySoftDeletes(reader, field);
                } catch (IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
            }
        });
        this.field = field;
    }

    private static LeafReader applySoftDeletes(LeafReader reader, String field) throws IOException {
        NumericDocValues soft = DocValues.getNumeric(reader, field);
        if (soft.nextDoc() == DocIdSetIterator.NO_MORE_DOCS) {
            return reader;
        }
        Bits hard = reader.getLiveDocs();
        FixedBitSet live;
        if (hard instanceof FixedBitSet bits) {
            live = bits.clone();
        } else {
            live = new FixedBitSet(reader.maxDoc());
            if (hard == null) {
                live.set(0, reader.maxDoc());
            } else {
                for (int doc = 0; doc < reader.maxDoc(); doc++) {
                    if (hard.get(doc)) {
                        live.set(doc);
                    }
                }
            }
        }
        int softDeleted = 0;
        for (int doc = soft.docID(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = soft.nextDoc()) {
            if (live.get(doc)) {
                live.clear(doc);
                softDeleted++;
            }
        }
        return new SoftDeletedLeaf(reader, live, reader.numDocs() - softDeleted);
    }

    @Override
    protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) throws IOException {
        return new SoftDeletesReader(in, field);
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return in.getReaderCacheHelper();
    }

    /** A leaf whose live documents exclude the soft-deleted ones. */
    private static final class SoftDeletedLeaf extends FilterLeafReader {
        private final FixedBitSet live;
        private final int numDocs;

        SoftDeletedLeaf(LeafReader in, FixedBitSet live, int numDocs) {
            super(in);
            this.live = live;
            this.numDocs = numDocs;
        }

        @Override
        public Bits getLiveDocs() {
            return live;
        }

        @Override
        public int numDocs() {
            return numDocs;
        }

        @Override
        public CacheHelper getCoreCacheHelper() {
            return in.getCoreCacheHelper();
        }

        /**
         * The wrapped segment reader's key: these live documents are a function of that reader
         * alone (its deletions and doc-values generation), and Lucene's query cache keeps matches
         * independent of deletions anyway.
         */
        @Override
        public CacheHelper getReaderCacheHelper() {
            return in.getReaderCacheHelper();
        }
    }
}
