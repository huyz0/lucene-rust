/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.FeatureField;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.KeepOnlyLastCommitDeletionPolicy;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SoftDeletesDirectoryReaderWrapper;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.lucenerust.opensearch.NativeLibrary;
import org.opensearch.index.mapper.ParseContext;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.stream.Stream;

/**
 * The writer's differential test: the same documents, operation for operation, through Lucene's
 * {@link IndexWriter} and through {@link RustIndexWriter}, then every live document compared field
 * by field through Lucene's own readers -- stored values, postings with frequencies, positions and
 * offsets, norms, all five doc-values types, points -- and the Rust index through Lucene's {@link
 * CheckIndex}.
 *
 * <p>The documents are built the way OpenSearch's mappers build them (a field repeated as indexed
 * and doc-valued instances, several instances of one text field, empty text, custom term
 * frequencies) and analyzed by an analyzer with stop words, injected synonyms at position
 * increment 0, and non-zero position and offset gaps, so that everything {@code FieldInvertState}
 * carries into a norm is exercised. Soft updates, delete tombstones and stale operations run as
 * OpenSearch issues them.
 */
public final class EngineWriterDiffTest {
    private static int checks;
    private static int failures;

    private static final String SOFT = "__soft_deletes";
    private static final String[] WORDS = ("the quick brown fox jumps over a lazy dog and cats run to their houses while birds sing "
        + "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda").split(" ");

    public static void main(String[] args) throws Exception {
        NativeLibrary.load(Path.of("."));
        for (long seed : new long[] { 1, 2, 3 }) {
            run(seed, 1500);
        }
        refusals();
        System.out.printf("EngineWriterDiffTest: %d checks, %d failures%n", checks, failures);
        if (failures > 0) {
            System.exit(1);
        }
    }

    private static void check(boolean ok, String what) {
        checks++;
        if (ok == false) {
            failures++;
            if (failures <= 40) {
                System.out.println("FAIL: " + what);
            }
        }
    }

    /** Stop words, a synonym at position increment 0 for words ending in "s", and gaps. */
    static final class TestAnalyzer extends Analyzer {
        @Override
        protected TokenStreamComponents createComponents(String fieldName) {
            Tokenizer t = new StandardTokenizer();
            TokenStream s = new LowerCaseFilter(t);
            s = new StopFilter(s, EnglishAnalyzer.ENGLISH_STOP_WORDS_SET);
            s = new SynonymS(s);
            return new TokenStreamComponents(t, s);
        }

        @Override
        public int getPositionIncrementGap(String fieldName) {
            return 100;
        }

        @Override
        public int getOffsetGap(String fieldName) {
            return 7;
        }
    }

    /** Emits "syn_" + term at the same position after every term ending in "s". */
    static final class SynonymS extends TokenFilter {
        private final CharTermAttribute term = addAttribute(CharTermAttribute.class);
        private final PositionIncrementAttribute posInc = addAttribute(PositionIncrementAttribute.class);
        private State pending;

        SynonymS(TokenStream in) {
            super(in);
        }

        @Override
        public boolean incrementToken() throws IOException {
            if (pending != null) {
                restoreState(pending);
                pending = null;
                String t = "syn_" + term;
                term.setEmpty().append(t);
                posInc.setPositionIncrement(0);
                return true;
            }
            if (input.incrementToken() == false) {
                return false;
            }
            if (term.length() > 1 && term.charAt(term.length() - 1) == 's') {
                pending = captureState();
            }
            return true;
        }

        @Override
        public void reset() throws IOException {
            super.reset();
            pending = null;
        }
    }

    private static final FieldType OFFSETS = new FieldType(TextField.TYPE_NOT_STORED);
    private static final FieldType FREQS_ONLY = new FieldType(TextField.TYPE_NOT_STORED);
    static {
        OFFSETS.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
        OFFSETS.freeze();
        FREQS_ONLY.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
        FREQS_ONLY.freeze();
    }

    private static String words(Random r, int n) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < n; i++) {
            b.append(WORDS[(int) (Math.pow(r.nextDouble(), 2) * WORDS.length)]).append(i % 7 == 6 ? ". " : " ");
        }
        return b.toString();
    }

    /** One document as an OpenSearch mapper would build it. */
    static ParseContext.Document doc(Random r, String id, long seqNo, boolean tombstone) {
        ParseContext.Document d = new ParseContext.Document();
        d.add(new StringField("_id", new BytesRef(id), Field.Store.YES));
        d.add(new NumericDocValuesField("_seq_no", seqNo));
        d.add(new LongPoint("_seq_no", seqNo));
        d.add(new NumericDocValuesField("_version", 1));
        if (tombstone) {
            d.add(new NumericDocValuesField("_tombstone", 1));
            return d;
        }
        d.add(new StoredField("_source", ("{\"id\":\"" + id + "\"}").getBytes(StandardCharsets.UTF_8)));
        int shape = r.nextInt(10);
        d.add(new TextField("title", shape == 0 ? "" : words(r, 1 + r.nextInt(6)), Field.Store.YES));
        if (shape != 1) {
            for (int i = 0, n = 1 + r.nextInt(3); i < n; i++) {
                d.add(new Field("body", words(r, r.nextInt(30)), OFFSETS));
            }
        }
        for (int i = 0, n = r.nextInt(3); i < n; i++) {
            String tag = "t" + r.nextInt(12);
            d.add(new Field("tag", new BytesRef(tag), tagType()));
            d.add(new SortedSetDocValuesField("tag", new BytesRef(tag)));
        }
        if (shape % 2 == 0) {
            // Sparse and single-valued: Lucene writes the SORTED shape for it.
            d.add(new SortedSetDocValuesField("one", new BytesRef("o" + r.nextInt(3))));
        }
        if (shape != 2) {
            long num = r.nextLong() % 100_000;
            d.add(new LongPoint("num", num));
            d.add(new NumericDocValuesField("num", num));
            d.add(new StoredField("num", num));
        }
        if (shape != 3) {
            int a = r.nextInt(1000);
            int b = r.nextInt(1000);
            d.add(new IntPoint("price", a));
            d.add(new IntPoint("price", b));
            d.add(new SortedNumericDocValuesField("price", a));
            d.add(new SortedNumericDocValuesField("price", b));
            d.add(new StoredField("price", a));
        }
        if (shape > 5) {
            d.add(new Field("freqs", words(r, 5 + r.nextInt(10)), FREQS_ONLY));
            d.add(new FeatureField("feature", "f" + r.nextInt(4), 1 + r.nextFloat() * 50));
            d.add(new BinaryDocValuesField("bin", new BytesRef(new byte[] { (byte) r.nextInt(), 0, 7 })));
            d.add(new SortedDocValuesField("kind", new BytesRef("k" + r.nextInt(5))));
            d.add(new StoredField("f32", r.nextFloat()));
            d.add(new StoredField("f64", r.nextDouble()));
            d.add(new StoredField("i32", r.nextInt()));
        }
        return d;
    }

    private static FieldType tagType() {
        FieldType t = new FieldType(StringField.TYPE_NOT_STORED);
        t.freeze();
        return t;
    }

    private static void run(long seed, int ops) throws Exception {
        Path root = Files.createTempDirectory("engine-writer-diff");
        Path javaPath = root.resolve("java");
        Path rustPath = root.resolve("rust");
        Analyzer analyzer = new TestAnalyzer();
        BM25Similarity similarity = new BM25Similarity();
        try (FSDirectory javaDir = FSDirectory.open(javaPath); FSDirectory rustDir = FSDirectory.open(rustPath)) {
            // The Rust shard starts from an empty commit, as Store.createEmpty leaves it.
            try (IndexWriter empty = new IndexWriter(rustDir, new IndexWriterConfig(analyzer))) {
                empty.commit();
            }
            IndexWriterConfig cfg = new IndexWriterConfig(analyzer).setSimilarity(similarity)
                .setSoftDeletesField(SOFT)
                .setMaxBufferedDocs(97);
            RustIndexWriter rust = RustIndexWriter.open(
                rustDir,
                rustPath,
                16,
                false,
                analyzer,
                similarity,
                10,
                SOFT,
                null,
                new KeepOnlyLastCommitDeletionPolicy(),
                null,
                null,
                null
            );
            rust.setLiveCommitData(Map.of("k", "v").entrySet());
            Random r = new Random(seed);
            Random rj = new Random(seed ^ 0x5eed);
            try (IndexWriter java = new IndexWriter(javaDir, cfg)) {
                java.setLiveCommitData(Map.of("k", "v").entrySet());
                // A soft update before any document has carried the soft-deletes field: it must
                // still land (IndexWriter knows the field from its config).
                long firstSeed = r.nextLong();
                java.addDocument(doc(new Random(firstSeed), "first", -2, false));
                rust.addDocument(doc(new Random(firstSeed), "first", -2, false), null);
                Field firstSoft = new NumericDocValuesField(SOFT, 1);
                java.softUpdateDocument(new Term("_id", "first"), doc(new Random(firstSeed), "first", -1, false), firstSoft);
                rust.softUpdateDocument(new Term("_id", "first"), doc(new Random(firstSeed), "first", -1, false), 1, -1, 1, firstSoft);
                int nextId = 0;
                for (int op = 0; op < ops; op++) {
                    int kind = r.nextInt(20);
                    long docSeed = r.nextLong();
                    if (kind < 12 || nextId == 0) {
                        String id = "d" + nextId++;
                        java.addDocument(doc(new Random(docSeed), id, op, false));
                        rust.addDocument(doc(new Random(docSeed), id, op, false), null);
                    } else if (kind < 16) {
                        String id = "d" + rj.nextInt(nextId);
                        Term uid = new Term("_id", id);
                        Field soft = new NumericDocValuesField(SOFT, 1);
                        java.softUpdateDocument(uid, doc(new Random(docSeed), id, op, false), soft);
                        rust.softUpdateDocument(uid, doc(new Random(docSeed), id, op, false), 1, op, 1, soft);
                    } else if (kind < 18) {
                        // A delete: the tombstone replaces the document, born soft-deleted.
                        String id = "d" + rj.nextInt(nextId);
                        Term uid = new Term("_id", id);
                        Field soft = new NumericDocValuesField(SOFT, 1);
                        ParseContext.Document jt = doc(new Random(docSeed), id, op, true);
                        jt.add(soft);
                        ParseContext.Document rt = doc(new Random(docSeed), id, op, true);
                        rt.add(soft);
                        java.softUpdateDocument(uid, jt, soft);
                        rust.deleteDocument(uid, false, rt, 1, op, 1, soft);
                    } else if (kind < 19) {
                        // A stale operation: indexed already soft-deleted.
                        String id = "stale" + op;
                        ParseContext.Document jd = doc(new Random(docSeed), id, op, false);
                        jd.add(new NumericDocValuesField(SOFT, 1));
                        ParseContext.Document rd = doc(new Random(docSeed), id, op, false);
                        rd.add(new NumericDocValuesField(SOFT, 1));
                        java.addDocument(jd);
                        rust.addDocument(rd, null);
                    } else {
                        java.commit();
                        rust.commit();
                    }
                }
                java.commit();
                rust.commit();
            }
            check(rust.hasUncommittedChanges() == false, "seed " + seed + ": nothing left uncommitted");
            rust.close();
            compare("seed " + seed, javaDir, rustDir);
            checkIndex("seed " + seed, rustDir);
        } finally {
            try (Stream<Path> s = Files.walk(root)) {
                s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }

    private static void checkIndex(String where, FSDirectory dir) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (CheckIndex ci = new CheckIndex(dir)) {
            ci.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
            ci.setInfoStream(new PrintStream(out, true, StandardCharsets.UTF_8));
            CheckIndex.Status status = ci.checkIndex();
            check(status.clean, where + ": CheckIndex on the Rust index\n" + out.toString(StandardCharsets.UTF_8));
        }
    }

    /** Every live document's full content, keyed by field, as Lucene reads it. */
    private static Map<String, TreeMap<String, List<String>>> content(DirectoryReader reader) throws IOException {
        Map<String, TreeMap<String, List<String>>> byId = new TreeMap<>();
        for (LeafReaderContext ctx : reader.leaves()) {
            LeafReader leaf = ctx.reader();
            Bits live = leaf.getLiveDocs();
            StoredFields stored = leaf.storedFields();
            String[] ids = new String[leaf.maxDoc()];
            for (int doc = 0; doc < leaf.maxDoc(); doc++) {
                if (live != null && live.get(doc) == false) {
                    continue;
                }
                TreeMap<String, List<String>> c = new TreeMap<>();
                for (IndexableField f : stored.document(doc)) {
                    Object v = f.numericValue() != null ? f.numericValue() + ":" + f.numericValue().getClass().getSimpleName()
                        : f.binaryValue() != null ? hex(f.binaryValue())
                        : f.stringValue();
                    c.computeIfAbsent("stored." + f.name(), k -> new ArrayList<>()).add(String.valueOf(v));
                }
                String id = c.get("stored._id").get(0);
                ids[doc] = id;
                byId.put(id, c);
            }
            for (FieldInfo fi : leaf.getFieldInfos()) {
                String schema = fi.getIndexOptions()
                    + "/"
                    + fi.omitsNorms()
                    + "/"
                    + fi.getDocValuesType()
                    + "/"
                    + fi.getPointDimensionCount()
                    + "x"
                    + fi.getPointNumBytes()
                    + "/"
                    + fi.isSoftDeletesField()
                    + "/"
                    + fi.hasPayloads();
                for (String id : ids) {
                    if (id != null) {
                        byId.get(id).put("schema." + fi.name, List.of(schema));
                    }
                }
                if (fi.getIndexOptions() != IndexOptions.NONE) {
                    postings(leaf, fi, ids, byId);
                    if (fi.omitsNorms() == false) {
                        NumericDocValues norms = leaf.getNormValues(fi.name);
                        if (norms != null) {
                            for (int doc = norms.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = norms.nextDoc()) {
                                if (ids[doc] != null) {
                                    byId.get(ids[doc]).put("norm." + fi.name, List.of(Long.toString(norms.longValue())));
                                }
                            }
                        }
                    }
                }
                if (fi.getDocValuesType() != DocValuesType.NONE) {
                    docValues(leaf, fi, ids, byId);
                }
                if (fi.getPointDimensionCount() > 0) {
                    PointValues pv = leaf.getPointValues(fi.name);
                    if (pv != null) {
                        pv.intersect(new PointValues.IntersectVisitor() {
                            @Override
                            public void visit(int docID) {
                                throw new AssertionError("unreachable: every cell is crossed");
                            }

                            @Override
                            public void visit(int docID, byte[] packedValue) {
                                if (ids[docID] != null) {
                                    byId.get(ids[docID])
                                        .computeIfAbsent("points." + fi.name, k -> new ArrayList<>())
                                        .add(HexFormat.of().formatHex(packedValue));
                                }
                            }

                            @Override
                            public PointValues.Relation compare(byte[] min, byte[] max) {
                                return PointValues.Relation.CELL_CROSSES_QUERY;
                            }
                        });
                    }
                }
            }
        }
        for (TreeMap<String, List<String>> c : byId.values()) {
            // A segment's field list depends on what else happened to land in it (merges union
            // them), so only the schema of a field this document carries is compared.
            c.keySet().removeIf(k -> k.startsWith("schema.") && c.keySet().stream().noneMatch(o -> o.startsWith("schema.") == false && o.endsWith("." + k.substring(7))));
            for (Map.Entry<String, List<String>> e : c.entrySet()) {
                if (e.getKey().startsWith("points.") || e.getKey().startsWith("postings.")) {
                    e.getValue().sort(null);
                }
            }
        }
        return byId;
    }

    private static void postings(LeafReader leaf, FieldInfo fi, String[] ids, Map<String, TreeMap<String, List<String>>> byId)
        throws IOException {
        Terms terms = leaf.terms(fi.name);
        if (terms == null) {
            return;
        }
        TermsEnum te = terms.iterator();
        PostingsEnum pe = null;
        for (BytesRef term = te.next(); term != null; term = te.next()) {
            String t = term.utf8ToString();
            pe = te.postings(pe, PostingsEnum.ALL);
            for (int doc = pe.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = pe.nextDoc()) {
                if (ids[doc] == null) {
                    continue;
                }
                StringBuilder b = new StringBuilder(t).append(" freq=").append(pe.freq());
                if (fi.getIndexOptions().compareTo(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS) >= 0) {
                    for (int i = 0; i < pe.freq(); i++) {
                        b.append(' ').append(pe.nextPosition());
                        if (fi.getIndexOptions() == IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS) {
                            b.append('[').append(pe.startOffset()).append(',').append(pe.endOffset()).append(']');
                        }
                    }
                }
                byId.get(ids[doc]).computeIfAbsent("postings." + fi.name, k -> new ArrayList<>()).add(b.toString());
            }
        }
    }

    private static void docValues(LeafReader leaf, FieldInfo fi, String[] ids, Map<String, TreeMap<String, List<String>>> byId)
        throws IOException {
        String key = "dv." + fi.name;
        switch (fi.getDocValuesType()) {
            case NUMERIC -> {
                NumericDocValues v = leaf.getNumericDocValues(fi.name);
                for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
                    if (ids[doc] != null) {
                        byId.get(ids[doc]).put(key, List.of(Long.toString(v.longValue())));
                    }
                }
            }
            case BINARY -> {
                BinaryDocValues v = leaf.getBinaryDocValues(fi.name);
                for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
                    if (ids[doc] != null) {
                        byId.get(ids[doc]).put(key, List.of(hex(v.binaryValue())));
                    }
                }
            }
            case SORTED -> {
                SortedDocValues v = leaf.getSortedDocValues(fi.name);
                for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
                    if (ids[doc] != null) {
                        byId.get(ids[doc]).put(key, List.of(v.lookupOrd(v.ordValue()).utf8ToString()));
                    }
                }
            }
            case SORTED_NUMERIC -> {
                SortedNumericDocValues v = leaf.getSortedNumericDocValues(fi.name);
                for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
                    if (ids[doc] != null) {
                        List<String> vals = new ArrayList<>();
                        for (int i = 0; i < v.docValueCount(); i++) {
                            vals.add(Long.toString(v.nextValue()));
                        }
                        byId.get(ids[doc]).put(key, vals);
                    }
                }
            }
            case SORTED_SET -> {
                SortedSetDocValues v = leaf.getSortedSetDocValues(fi.name);
                for (int doc = v.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = v.nextDoc()) {
                    if (ids[doc] != null) {
                        List<String> vals = new ArrayList<>();
                        for (int i = 0; i < v.docValueCount(); i++) {
                            vals.add(v.lookupOrd(v.nextOrd()).utf8ToString());
                        }
                        byId.get(ids[doc]).put(key, vals);
                    }
                }
            }
            default -> {}
        }
    }

    private static String hex(BytesRef b) {
        return HexFormat.of().formatHex(Arrays.copyOfRange(b.bytes, b.offset, b.offset + b.length));
    }

    private static void compare(String where, FSDirectory javaDir, FSDirectory rustDir) throws IOException {
        try (
            DirectoryReader java = new SoftDeletesDirectoryReaderWrapper(DirectoryReader.open(javaDir), SOFT);
            DirectoryReader rust = new SoftDeletesDirectoryReaderWrapper(DirectoryReader.open(rustDir), SOFT)
        ) {
            check(java.numDocs() == rust.numDocs(), where + ": live docs " + java.numDocs() + " vs " + rust.numDocs());
            check(
                DirectoryReader.listCommits(rustDir).get(0).getUserData().equals(Map.of("k", "v")),
                where + ": commit user data"
            );
            Map<String, TreeMap<String, List<String>>> expected = content(java);
            Map<String, TreeMap<String, List<String>>> actual = content(rust);
            check(expected.keySet().equals(actual.keySet()), where + ": the same live ids");
            int shown = 0;
            for (Map.Entry<String, TreeMap<String, List<String>>> e : expected.entrySet()) {
                TreeMap<String, List<String>> a = actual.get(e.getKey());
                boolean same = e.getValue().equals(a);
                check(same, where + ": document " + e.getKey() + (same || shown++ > 3 ? "" : "\n  java " + e.getValue() + "\n  rust " + a));
            }
        }
    }

    /** What the Rust engine refuses, it refuses per document, and the writer carries on. */
    private static void refusals() throws Exception {
        Path root = Files.createTempDirectory("engine-writer-refusals");
        try (FSDirectory dir = FSDirectory.open(root)) {
            try (IndexWriter empty = new IndexWriter(dir, new IndexWriterConfig())) {
                empty.commit();
            }
            RustIndexWriter w = RustIndexWriter.open(
                dir,
                root,
                16,
                false,
                new TestAnalyzer(),
                new BM25Similarity(),
                10,
                SOFT,
                null,
                new KeepOnlyLastCommitDeletionPolicy(),
                null,
                null,
                null
            );
            w.setLiveCommitData(Map.<String, String>of().entrySet());
            FieldType tv = new FieldType(TextField.TYPE_NOT_STORED);
            tv.setStoreTermVectors(true);
            ParseContext.Document d = new ParseContext.Document();
            d.add(new Field("tv", "a b", tv));
            check(refused(w, d, "term vectors"), "term vectors are refused");
            ParseContext.Document inconsistent = new ParseContext.Document();
            inconsistent.add(new StringField("x", "a", Field.Store.NO));
            w.addDocument(inconsistent, null);
            ParseContext.Document other = new ParseContext.Document();
            other.add(new TextField("x", "a", Field.Store.NO));
            check(refused(w, other, "Inconsistency of field data structures"), "a changed schema is refused");
            ParseContext.Document big = new ParseContext.Document();
            big.add(new StringField("y", new BytesRef(new byte[IndexWriter.MAX_TERM_LENGTH + 1]), Field.Store.NO));
            check(refused(w, big, "immense term"), "an immense term is refused");
            w.addDocument(inconsistent, null);
            w.commit();
            check(w.getTragicException() == null, "refusals are not tragic");
            check(w.getPendingNumDocs() == 2, "two documents indexed, got " + w.getPendingNumDocs());
            w.close();
            checkIndex("refusals", dir);
        } finally {
            try (Stream<Path> s = Files.walk(root)) {
                s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }

    private static boolean refused(RustIndexWriter w, ParseContext.Document d, String message) {
        try {
            w.addDocument(d, null);
            return false;
        } catch (IllegalArgumentException e) {
            return e.getMessage().contains(message);
        } catch (IOException e) {
            return false;
        }
    }
}
