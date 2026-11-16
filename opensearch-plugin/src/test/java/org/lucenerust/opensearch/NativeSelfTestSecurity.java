/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FilterBinaryDocValues;
import org.apache.lucene.index.FilterSortedDocValues;
import org.apache.lucene.index.FilterSortedSetDocValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.util.BytesRef;
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
import java.nio.charset.StandardCharsets;
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

    /** What the security plugin's masking function of one field does to a value: a salted digest, in hex. */
    public static final class Mask {
        private final String salt;

        Mask(String salt) {
            this.salt = salt;
        }

        public BytesRef apply(BytesRef value) {
            if (value == null) {
                return null;
            }
            try {
                java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
                digest.update(salt.getBytes(StandardCharsets.UTF_8));
                digest.update(value.bytes, value.offset, value.length);
                return new BytesRef(java.util.HexFormat.of().formatHex(digest.digest()));
            } catch (java.security.NoSuchAlgorithmException e) {
                throw new IllegalStateException(e);
            }
        }

        @Override
        public String toString() {
            return "sha256:" + salt;
        }
    }

    /** The rule a leaf masks fields with. */
    static final class Rule {
        private final Set<String> masked;
        private final Mask mask;

        Rule(Set<String> masked, String salt) {
            this.masked = masked;
            this.mask = new Mask(salt);
        }

        boolean isAllowAll() {
            return masked.isEmpty();
        }

        boolean isMasked(String field) {
            return masked.contains(field);
        }

        Mask get(String field) {
            return masked.contains(field) ? mask : null;
        }
    }

    /** Stands in for the security plugin's leaf: its class name is the one the readers recognise. */
    static final class HidingLeaf extends FilterLeafReader {
        // Found by name, as the security plugin's field is.
        @SuppressWarnings("unused")
        private final Rule fmRule;
        @SuppressWarnings("unused")
        private final Set<String> metaFields = Set.of("_id");
        /** Fields that answer null for everything: hidden by the role, or masked without values a mask could rewrite. */
        private final Set<String> hidden;
        /** Masked fields whose keyword or binary values the leaf shows through the mask. */
        private final Set<String> shownMasked = new HashSet<>();
        /** Every masked field: none of them has postings, points, norms or numeric values. */
        private final Set<String> masked;
        private final FieldInfos infos;
        private final Bits live;
        private final int numDocs;

        HidingLeaf(LeafReader in, Set<String> hidden, Set<String> masked, int modulus) {
            this(in, hidden, masked, modulus, "salt");
        }

        HidingLeaf(LeafReader in, Set<String> hidden, Set<String> masked, int modulus, String salt) {
            super(in);
            this.fmRule = new Rule(masked, salt);
            this.masked = masked;
            Set<String> gone = new HashSet<>(hidden);
            List<FieldInfo> keep = new ArrayList<>();
            for (FieldInfo fi : in.getFieldInfos()) {
                if (hidden.contains(fi.name)) {
                    continue;
                }
                if (masked.contains(fi.name)) {
                    switch (fi.getDocValuesType()) {
                        case BINARY, SORTED, SORTED_SET -> {
                            shownMasked.add(fi.name);
                            // The plugin's field info for it: its doc values, and nothing that indexes it.
                            keep.add(
                                new FieldInfo(
                                    fi.name,
                                    fi.number,
                                    false,
                                    false,
                                    false,
                                    IndexOptions.NONE,
                                    fi.getDocValuesType(),
                                    DocValuesSkipIndexType.NONE,
                                    fi.getDocValuesGen(),
                                    fi.attributes(),
                                    0,
                                    0,
                                    0,
                                    0,
                                    fi.getVectorEncoding(),
                                    fi.getVectorSimilarityFunction(),
                                    fi.isSoftDeletesField(),
                                    fi.isParentField()
                                )
                            );
                        }
                        default -> gone.add(fi.name);
                    }
                    continue;
                }
                keep.add(fi);
            }
            this.hidden = gone;
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

        /** Whether the leaf answers nothing of {@code field} but, for a masked one, its doc values. */
        private boolean unindexed(String field) {
            return hidden.contains(field) || masked.contains(field);
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
            return unindexed(field) ? null : in.terms(field);
        }

        @Override
        public NumericDocValues getNumericDocValues(String field) throws IOException {
            return unindexed(field) ? null : in.getNumericDocValues(field);
        }

        @Override
        public BinaryDocValues getBinaryDocValues(String field) throws IOException {
            if (hidden.contains(field)) {
                return null;
            }
            BinaryDocValues values = in.getBinaryDocValues(field);
            Mask mask = shownMasked.contains(field) ? fmRule.get(field) : null;
            if (values == null || mask == null) {
                return values;
            }
            return new FilterBinaryDocValues(values) {
                @Override
                public BytesRef binaryValue() throws IOException {
                    return mask.apply(BytesRef.deepCopyOf(values.binaryValue()));
                }
            };
        }

        @Override
        public SortedDocValues getSortedDocValues(String field) throws IOException {
            if (hidden.contains(field)) {
                return null;
            }
            SortedDocValues values = in.getSortedDocValues(field);
            Mask mask = shownMasked.contains(field) ? fmRule.get(field) : null;
            if (values == null || mask == null) {
                return values;
            }
            return new FilterSortedDocValues(values) {
                @Override
                public BytesRef lookupOrd(int ord) throws IOException {
                    return mask.apply(BytesRef.deepCopyOf(values.lookupOrd(ord)));
                }
            };
        }

        @Override
        public SortedNumericDocValues getSortedNumericDocValues(String field) throws IOException {
            return unindexed(field) ? null : in.getSortedNumericDocValues(field);
        }

        @Override
        public SortedSetDocValues getSortedSetDocValues(String field) throws IOException {
            if (hidden.contains(field)) {
                return null;
            }
            SortedSetDocValues values = in.getSortedSetDocValues(field);
            Mask mask = shownMasked.contains(field) ? fmRule.get(field) : null;
            if (values == null || mask == null) {
                return values;
            }
            return new FilterSortedSetDocValues(values) {
                @Override
                public BytesRef lookupOrd(long ord) throws IOException {
                    return mask.apply(BytesRef.deepCopyOf(values.lookupOrd(ord)));
                }
            };
        }

        @Override
        public NumericDocValues getNormValues(String field) throws IOException {
            return unindexed(field) ? null : in.getNormValues(field);
        }

        @Override
        public PointValues getPointValues(String field) throws IOException {
            return unindexed(field) ? null : in.getPointValues(field);
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
        private final Set<String> masked;
        private final int modulus;
        private final String salt;

        HidingReader(DirectoryReader in, Set<String> hidden, int modulus) throws IOException {
            this(in, hidden, Set.of(), modulus);
        }

        HidingReader(DirectoryReader in, Set<String> hidden, Set<String> masked, int modulus) throws IOException {
            this(in, hidden, masked, modulus, "salt");
        }

        HidingReader(DirectoryReader in, Set<String> hidden, Set<String> masked, int modulus, String salt) throws IOException {
            super(in, new SubReaderWrapper() {
                @Override
                public LeafReader wrap(LeafReader reader) {
                    return new HidingLeaf(reader, hidden, masked, modulus, salt);
                }
            });
            this.hidden = hidden;
            this.masked = masked;
            this.modulus = modulus;
            this.salt = salt;
        }

        @Override
        protected DirectoryReader doWrapDirectoryReader(DirectoryReader in) throws IOException {
            return new HidingReader(in, hidden, masked, modulus, salt);
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
            // A masked field with no sorted values reads as null everywhere: hidden, native.
            DirectoryReader maskedPoints = new HidingReader(base, Set.of(), Set.of("n"), 0);
            compare.run("security: masked field without values", maskedPoints, readers, queries);
            NativeReaders.Acquired mp = readers.acquire(maskedPoints);
            NativeSelfTest.check(mp.handle() != 0, "security: a masked field without sorted values opens natively, was " + mp.fallbackReason());
            // One whose keyword values survive is shown through the plugin's masking function: the
            // native side calls it wherever it shows a value, and every search, sort and aggregation
            // over the masked fields must answer what Lucene answers over the plugin's own wrapper.
            Set<String> keywords = Set.of("kt", "kw", "kx");
            DirectoryReader maskedKeyword = new HidingReader(base, Set.of(), keywords, 0);
            NativeReaders.Acquired mk = readers.acquire(maskedKeyword);
            NativeSelfTest.check(mk.handle() != 0, "security: masked keyword values open natively, was " + mk.fallbackReason());
            compare.run("security: masked keyword fields", maskedKeyword, readers, queries);
            DirectoryReader maskedAndHidden = new HidingReader(base, Set.of("tag", "sl"), keywords, 7);
            compare.run("security: masked keyword fields, others hidden, live docs filtered", maskedAndHidden, readers, queries);
            compared += queries.size() * 3;
            // Equal wrappers share one native view; another salt is another mask and another view.
            NativeSelfTest.check(readers.acquire(new HidingReader(base, Set.of(), keywords, 0)).handle() == mk.handle(), "security: an equal masking wrapper reuses its view");
            NativeReaders.Acquired resalted = readers.acquire(new HidingReader(base, Set.of(), keywords, 0, "other salt"));
            NativeSelfTest.check(resalted.handle() != 0 && resalted.handle() != mk.handle(), "security: another mask is another view");
            NativeSelfTest.check(readers.acquire(new HidingReader(base, Set.of(), Set.of("kt"), 0)).handle() != mk.handle(), "security: another set of masked fields is another view");
            NativeSelfTest.check(readers.acquire(maskedKeyword).handle() != readers.acquire(base).handle(), "security: a masked view is never the base's handle");
            leaks(check, base, readers, mk, resalted);
            // One user's reader is never served to another: the base, a role that hides `tag`, and
            // a role that hides nothing share segments and cache key but not a handle or an answer.
            NativeReaders.Acquired baseHandle = readers.acquire(base);
            long[] seen = new long[1];
            long[] hiddenSeen = new long[1];
            check.accept(NativeBridge.docFreq(baseHandle.handle(), "tag".getBytes(StandardCharsets.UTF_8), "alpha".getBytes(StandardCharsets.UTF_8), seen) == NativeBridge.OK && seen[0] > 0, "security: the unwrapped reader sees tag");
            check.accept(NativeBridge.docFreq(b.handle(), "tag".getBytes(StandardCharsets.UTF_8), "alpha".getBytes(StandardCharsets.UTF_8), hiddenSeen) == NativeBridge.OK && hiddenSeen[0] == 0, "security: the role hiding tag does not");
            check.accept(readers.acquire(plain).handle() != baseHandle.handle() && b.handle() != baseHandle.handle(), "security: a wrapped reader never takes the base's handle");
            check.accept(readers.acquire(base).handle() == baseHandle.handle() && readers.acquire(hiding).handle() == b.handle(), "security: acquiring in another order returns each its own");
            // A leaf the readers do not recognise stays on Lucene.
            NativeReaders strict = new NativeReaders(c2 -> c2.getName().startsWith("org.apache.lucene"), Set.of());
            NativeReaders.Acquired refused = strict.acquire(hiding);
            check.accept(refused.handle() == 0 && "reader_not_standard".equals(refused.fallbackReason()), "security: unknown wrapper refused, was " + refused.fallbackReason());
            base.close();
        }
    }

    /** The keys of a native terms aggregation of {@code field} over every document of {@code handle}. */
    private static List<String> nativeKeys(long handle, String field, int shardSize) {
        QueryEncoder.Encoded all = QueryEncoder.encode(org.apache.lucene.search.MatchAllDocsQuery.INSTANCE, f -> true);
        NativeAggregations.Terms spec = new NativeAggregations.Terms(
            "t",
            field,
            org.opensearch.search.DocValueFormat.RAW,
            null,
            org.opensearch.search.aggregations.BucketOrder.compound(
                org.opensearch.search.aggregations.BucketOrder.count(false),
                org.opensearch.search.aggregations.BucketOrder.key(true)
            ),
            new org.opensearch.search.aggregations.bucket.terms.TermsAggregator.BucketCountThresholds(1, 0, shardSize, shardSize),
            false
        );
        byte[][] out = new byte[1][];
        NativeAggregations.Plan plan = new NativeAggregations.Plan(List.of(spec));
        int rc = NativeBridge.aggregate(handle, all.blob(), plan.blob(new int[0][]), new long[0], new double[0], out, -1, new long[2], null);
        if (rc != NativeBridge.OK) {
            throw new IllegalStateException("terms aggregation failed: " + rc + " " + NativeBridge.lastError());
        }
        List<String> keys = new ArrayList<>();
        for (var b : spec.read(java.nio.ByteBuffer.wrap(out[0]).order(java.nio.ByteOrder.LITTLE_ENDIAN)).getBuckets()) {
            keys.add(b.getKeyAsString());
        }
        return keys;
    }

    /**
     * No raw value of a masked field leaves the native view: the values it shows -- aggregation
     * keys, here -- are the mask's, for every mask, while the unmasked reader still shows its own.
     */
    private static void leaks(
        BiConsumer<Boolean, String> check,
        DirectoryReader base,
        NativeReaders readers,
        NativeReaders.Acquired masked,
        NativeReaders.Acquired resalted
    ) throws IOException {
        long baseHandle = readers.acquire(base).handle();
        Mask first = new Mask("salt");
        Mask second = new Mask("other salt");
        for (String field : new String[] { "kt", "kw", "kx" }) {
            Set<String> raw = new HashSet<>(nativeKeys(baseHandle, field, 1000));
            check.accept(raw.size() > 3 && raw.stream().noneMatch(k -> k.length() == 64), "security: " + field + " has raw values to leak, " + raw.size());
            Set<String> shownFirst = new HashSet<>();
            Set<String> shownSecond = new HashSet<>();
            for (String k : raw) {
                shownFirst.add(first.apply(new BytesRef(k)).utf8ToString());
                shownSecond.add(second.apply(new BytesRef(k)).utf8ToString());
            }
            for (int shardSize : new int[] { 1, 3, 1000 }) {
                List<String> keys = nativeKeys(masked.handle(), field, shardSize);
                check.accept(keys.isEmpty() == false && keys.stream().noneMatch(raw::contains), "security: no raw " + field + " value in a masked terms aggregation (shard_size " + shardSize + "): " + keys);
                check.accept(shownFirst.containsAll(keys), "security: masked " + field + " keys are the mask's, shard_size " + shardSize);
                List<String> other = nativeKeys(resalted.handle(), field, shardSize);
                check.accept(shownSecond.containsAll(other) && other.stream().noneMatch(shownFirst::contains), "security: another salt shows another mask, " + field);
            }
        }
    }
}
