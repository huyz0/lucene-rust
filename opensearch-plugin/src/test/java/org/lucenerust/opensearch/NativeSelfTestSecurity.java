/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterDirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.BiConsumer;

/**
 * Readers wrapped the way the security plugin wraps them: a leaf that hides fields (no terms,
 * values, points or norms, no field info) and, apart from that, documents. The native path must
 * open a view of the base reader that hides the same, answer like Lucene over the wrapper, and give
 * two wrappers of one reader two handles.
 */
final class NativeSelfTestSecurity {
    private NativeSelfTestSecurity() {}

    static int compared;

    /** The rule a leaf masks fields with: this one masks nothing. */
    static final class Rule {
        boolean isAllowAll() {
            return true;
        }
    }

    /** Stands in for the security plugin's leaf: its class name is the one the readers recognise. */
    static final class HidingLeaf extends FilterLeafReader {
        // Found by name, as the security plugin's field is.
        @SuppressWarnings("unused")
        private final Rule fmRule = new Rule();
        private final Set<String> hidden;
        private final FieldInfos infos;
        private final Bits live;
        private final int numDocs;

        HidingLeaf(LeafReader in, Set<String> hidden, int modulus) {
            super(in);
            this.hidden = hidden;
            List<FieldInfo> keep = new ArrayList<>();
            for (FieldInfo fi : in.getFieldInfos()) {
                if (hidden.contains(fi.name) == false) {
                    keep.add(fi);
                }
            }
            this.infos = new FieldInfos(keep.toArray(new FieldInfo[0]));
            Bits base = in.getLiveDocs();
            FixedBitSet bits = new FixedBitSet(in.maxDoc());
            int n = 0;
            for (int d = 0; d < in.maxDoc(); d++) {
                boolean alive = (base == null || base.get(d)) && (modulus == 0 || d % modulus != 0);
                if (alive) {
                    bits.set(d);
                    n++;
                }
            }
            this.live = bits;
            this.numDocs = n;
        }

        @Override
        public FieldInfos getFieldInfos() {
            return infos;
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
        public Terms terms(String field) throws IOException {
            return hidden.contains(field) ? null : in.terms(field);
        }

        @Override
        public NumericDocValues getNumericDocValues(String field) throws IOException {
            return hidden.contains(field) ? null : in.getNumericDocValues(field);
        }

        @Override
        public SortedDocValues getSortedDocValues(String field) throws IOException {
            return hidden.contains(field) ? null : in.getSortedDocValues(field);
        }

        @Override
        public SortedNumericDocValues getSortedNumericDocValues(String field) throws IOException {
            return hidden.contains(field) ? null : in.getSortedNumericDocValues(field);
        }

        @Override
        public SortedSetDocValues getSortedSetDocValues(String field) throws IOException {
            return hidden.contains(field) ? null : in.getSortedSetDocValues(field);
        }

        @Override
        public NumericDocValues getNormValues(String field) throws IOException {
            return hidden.contains(field) ? null : in.getNormValues(field);
        }

        @Override
        public PointValues getPointValues(String field) throws IOException {
            return hidden.contains(field) ? null : in.getPointValues(field);
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

    static final class HidingReader extends FilterDirectoryReader {
        private final Set<String> hidden;
        private final int modulus;

        HidingReader(DirectoryReader in, Set<String> hidden, int modulus) throws IOException {
            super(in, new SubReaderWrapper() {
                @Override
                public LeafReader wrap(LeafReader reader) {
                    return new HidingLeaf(reader, hidden, modulus);
                }
            });
            this.hidden = hidden;
            this.modulus = modulus;
        }

        @Override
        protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) throws IOException {
            return new HidingReader(in, hidden, modulus);
        }

        @Override
        public CacheHelper getReaderCacheHelper() {
            return in.getReaderCacheHelper();
        }
    }

    static void run(Random r, BiConsumer<Boolean, String> check, NativeSelfTestExtended.Compare compare) throws Exception {
        Path dir = Files.createTempDirectory("lucene-rust-selftest-security");
        NativeReaders readers = new NativeReaders(c -> c.getClassLoader() != NativeSelfTestSecurity.class.getClassLoader()
            || c.getName().startsWith("org.apache.lucene"), Set.of(HidingLeaf.class.getName()));
        try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(new StandardAnalyzer()))) {
            for (int i = 0; i < 1500; i++) {
                w.addDocument(NativeSelfTest.doc(r, i));
                if (i == 700) {
                    w.commit();
                }
            }
            for (int i = 0; i < 60; i++) {
                w.deleteDocuments(new Term("id", Integer.toString(r.nextInt(1500))));
            }
            DirectoryReader base = DirectoryReader.open(w);
            check.accept(base.leaves().size() >= 2, "security: several segments");
            List<Query> queries = new ArrayList<>();
            for (int i = 0; i < 40; i++) {
                String[] words = { "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "sigma" };
                queries.add(new TermQuery(new Term("body", words[r.nextInt(words.length)])));
                queries.add(new TermQuery(new Term("tag", words[r.nextInt(words.length)])));
                queries.add(new TermQuery(new Term("kt", words[r.nextInt(words.length)])));
                queries.add(LongPoint.newRangeQuery("n", r.nextInt(700), 700 + r.nextInt(700)));
                queries.add(LongPoint.newRangeQuery("sl", -50, r.nextInt(100)));
                queries.add(
                    new BooleanQuery.Builder().add(new TermQuery(new Term("body", words[r.nextInt(words.length)])), Occur.SHOULD)
                        .add(new TermQuery(new Term("tag", words[r.nextInt(words.length)])), Occur.SHOULD)
                        .build()
                );
            }
            queries.add(org.apache.lucene.search.MatchAllDocsQuery.INSTANCE);
            Set<String> hidden = new HashSet<>(Set.of("tag", "sl", "kw"));
            DirectoryReader plain = new HidingReader(base, Set.of(), 3);
            DirectoryReader hiding = new HidingReader(base, hidden, 0);
            DirectoryReader both = new HidingReader(base, hidden, 5);
            compare.run("security: live docs only", plain, readers, queries);
            compare.run("security: fields hidden", hiding, readers, queries);
            compare.run("security: fields and live docs hidden", both, readers, queries);
            NativeReaders.Acquired a = readers.acquire(plain);
            NativeReaders.Acquired b = readers.acquire(hiding);
            NativeReaders.Acquired c = readers.acquire(both);
            NativeReaders.Acquired again = readers.acquire(new HidingReader(base, hidden, 5));
            check.accept(a.handle() != 0 && b.handle() != 0 && c.handle() != 0, "security: views open");
            check.accept(a.handle() != b.handle() && b.handle() != c.handle() && a.handle() != c.handle(), "security: views do not share a handle");
            check.accept(again.handle() == c.handle(), "security: an equal wrapper reuses its view");
            compared += queries.size() * 3;
            // A leaf the readers do not recognise stays on Lucene.
            NativeReaders strict = new NativeReaders(c2 -> c2.getName().startsWith("org.apache.lucene"), Set.of());
            NativeReaders.Acquired refused = strict.acquire(hiding);
            check.accept(refused.handle() == 0 && "reader_not_standard".equals(refused.fallbackReason()), "security: unknown wrapper refused, was " + refused.fallbackReason());
            base.close();
        }
    }
}
