/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

/**
 * The JVM half of the plugin's tests: loads the real {@code liblucene_ffi.so} through {@link
 * NativeLibrary} and checks the whole native path -- {@link QueryEncoder}, {@link NativeReaders},
 * {@link NativeBridge}'s JNI marshalling -- against Lucene's own {@link IndexSearcher} on the same
 * reader.
 *
 * <ul>
 *   <li>NRT readers straight from an {@link IndexWriter}, with deletions that exist only in memory
 *       -- the reader OpenSearch actually searches -- across refreshes, merges and reuse;
 *   <li>every Java-written multi-segment fixture index under {@code fixtures/data};
 *   <li>the JNI error paths: bad handles, bad blobs, short and null arrays;
 *   <li>lifecycle: closing a Java reader closes its native reader.
 * </ul>
 *
 * <p>Top hits must agree on doc IDs and on scores to within 1e-5 (a doc-ID swap is accepted only
 * between two hits whose scores tie to that tolerance); counts must agree exactly.
 */
public final class NativeSelfTest {
    private static int checks;
    private static int failures;
    private static int bitExact;
    private static int scored;

    public static void main(String[] args) throws Exception {
        NativeLibrary.load(Path.of("."));
        check(NativeBridge.abiVersion() == NativeBridge.EXPECTED_ABI_VERSION, "ABI handshake");

        jniErrorPaths();
        encoderMatrix();
        nrtReaders(new Random(42));
        for (Path fixture : fixtureIndexes(Path.of(args[0]))) {
            fixture(fixture);
        }
        System.out.printf(
            "NativeSelfTest: %d checks, %d failures; %d of %d compared scores bit-exact%n",
            checks,
            failures,
            bitExact,
            scored
        );
        if (failures > 0) {
            System.exit(1);
        }
    }

    private static void check(boolean ok, String what) {
        checks++;
        if (ok == false) {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }

    // --- JNI error paths ---------------------------------------------------------------

    private static void jniErrorPaths() {
        int[] docs = new int[4];
        float[] scores = new float[4];
        long[] counts = new long[2];
        byte[] blob = QueryEncoder.encode(new TermQuery(new Term("f", "t")), f -> true).blob();
        int rc = NativeBridge.search(12345L, blob, 4, true, docs, scores, counts);
        check(rc == NativeBridge.INVALID_HANDLE, "search on a fabricated handle -> INVALID_HANDLE, got " + rc);
        check(NativeBridge.lastError().contains("unknown"), "lastError names the bad handle: " + NativeBridge.lastError());
        check(NativeBridge.closeReader(0) == NativeBridge.INVALID_HANDLE, "closing handle 0");
        check(NativeBridge.setLiveDocs(99, 0, null) == NativeBridge.INVALID_HANDLE, "live docs on a fabricated handle");
        // Marshalling failures are status codes, never exceptions.
        check(NativeBridge.search(1, null, 4, true, docs, scores, counts) == 10, "null query blob -> InvalidArgument");
        check(NativeBridge.search(1, blob, -1, true, docs, scores, counts) == 10, "negative topN -> InvalidArgument");
        check(NativeBridge.search(1, blob, 8, true, docs, scores, counts) == 8, "short output arrays -> BufferTooSmall");
        check(NativeBridge.search(1, blob, 4, true, null, scores, counts) == 10, "null output array -> InvalidArgument");
        check(NativeBridge.openReader(null, new byte[0], 1, 0, new int[0], new long[1]) == 10, "null path -> InvalidArgument");
        check(NativeBridge.setLiveDocs(1, -1, null) == 7, "negative segment -> IndexOutOfBounds");
    }

    // --- QueryEncoder --------------------------------------------------------------------

    private static void encoderMatrix() {
        Query t = new TermQuery(new Term("body", "a"));
        check(QueryEncoder.encode(t, f -> true).blob() != null, "TermQuery encodes");
        check(QueryEncoder.encode(new BoostQuery(t, 1f), f -> true).blob() != null, "unit BoostQuery encodes");
        check("query_BoostQuery".equals(QueryEncoder.encode(new BoostQuery(t, 2f), f -> true).fallbackReason()), "boosted falls back");
        check("field_similarity".equals(QueryEncoder.encode(t, f -> false).fallbackReason()), "rejected field falls back");
        check(
            "query_MatchAllDocsQuery".equals(QueryEncoder.encode(MatchAllDocsQuery.INSTANCE, f -> true).fallbackReason()),
            "match_all falls back"
        );
        Query phrase = new BooleanQuery.Builder().add(t, Occur.MUST).add(new PhraseQuery("body", "a", "b"), Occur.SHOULD).build();
        check("clause_PhraseQuery".equals(QueryEncoder.encode(phrase, f -> true).fallbackReason()), "phrase clause falls back");
        Query negative = new BooleanQuery.Builder().add(t, Occur.MUST_NOT).build();
        check("boolean_pure_negative".equals(QueryEncoder.encode(negative, f -> true).fallbackReason()), "pure negative falls back");
        check("boolean_empty".equals(QueryEncoder.encode(new BooleanQuery.Builder().build(), f -> true).fallbackReason()), "empty");
    }

    // --- differential ---------------------------------------------------------------------

    private static final String[] WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
        + "pi rho sigma tau upsilon phi chi psi omega").split(" ");

    private static String word(Random r) {
        // Zipf-ish: low indices are common, so both rare and dense postings occur.
        return WORDS[(int) Math.min(WORDS.length - 1, Math.floor(Math.pow(r.nextDouble(), 2.2) * WORDS.length))];
    }

    private static Query randomQuery(Random r, List<String> fields, int depth) {
        String field = fields.get(r.nextInt(fields.size()));
        if (depth > 1 || r.nextInt(4) == 0) {
            return new TermQuery(new Term(field, word(r)));
        }
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        int n = 1 + r.nextInt(4);
        int shoulds = 0;
        for (int i = 0; i < n; i++) {
            Occur occur = Occur.values()[r.nextInt(8) < 5 ? 2 : r.nextInt(4)];
            shoulds += occur == Occur.SHOULD ? 1 : 0;
            b.add(randomQuery(r, fields, depth + 1), occur);
        }
        if (shoulds > 1 && r.nextInt(3) == 0) {
            b.setMinimumNumberShouldMatch(1 + r.nextInt(shoulds));
        }
        return b.build();
    }

    /** Compares the native path with Lucene for {@code queries} on {@code reader}. */
    private static void compare(String where, DirectoryReader reader, NativeReaders readers, List<Query> queries) throws Exception {
        IndexSearcher searcher = new IndexSearcher(reader);
        NativeReaders.Acquired acquired = readers.acquire(reader);
        check(acquired.handle() != 0, where + ": native reader opens (" + acquired.fallbackReason() + ")");
        if (acquired.handle() == 0) {
            return;
        }
        for (Query query : queries) {
            Query rewritten = searcher.rewrite(query);
            QueryEncoder.Encoded enc = QueryEncoder.encode(rewritten, f -> true);
            if (enc.blob() == null) {
                continue;
            }
            for (int topN : new int[] { 10, 3 }) {
                TopDocs want = searcher.search(rewritten, topN);
                int[] docs = new int[topN];
                float[] scores = new float[topN];
                long[] counts = new long[2];
                int rc = NativeBridge.search(acquired.handle(), enc.blob(), topN, true, docs, scores, counts);
                String what = where + ": " + rewritten + " top" + topN;
                check(rc == NativeBridge.OK, what + ": native search status " + rc + " " + NativeBridge.lastError());
                if (rc != NativeBridge.OK) {
                    continue;
                }
                check(counts[1] == searcher.count(rewritten), what + ": count " + counts[1] + " vs " + searcher.count(rewritten));
                check(counts[0] == want.scoreDocs.length, what + ": hits " + counts[0] + " vs " + want.scoreDocs.length);
                for (int i = 0; i < Math.min(counts[0], want.scoreDocs.length); i++) {
                    float w = want.scoreDocs[i].score;
                    boolean close = Math.abs(scores[i] - w) <= 1e-5f * Math.max(1f, Math.abs(w));
                    boolean sameDoc = docs[i] == want.scoreDocs[i].doc;
                    scored++;
                    bitExact += sameDoc && Float.floatToIntBits(scores[i]) == Float.floatToIntBits(w) ? 1 : 0;
                    boolean tieSwap = sameDoc == false && close && isTie(want, i);
                    check(
                        close && (sameDoc || tieSwap),
                        what + ": hit " + i + " native (" + docs[i] + ", " + scores[i] + ") lucene (" + want.scoreDocs[i].doc + ", " + w + ")"
                    );
                }
            }
        }
    }

    private static boolean isTie(TopDocs td, int i) {
        float s = td.scoreDocs[i].score;
        boolean prev = i > 0 && Math.abs(td.scoreDocs[i - 1].score - s) <= 1e-5f * Math.max(1f, s);
        boolean next = i + 1 < td.scoreDocs.length && Math.abs(td.scoreDocs[i + 1].score - s) <= 1e-5f * Math.max(1f, s);
        return prev || next;
    }

    /**
     * The reader OpenSearch searches: NRT from the writer, deletions applied but not written. Checked
     * after each of several rounds of adds, deletes and updates, with each refresh reusing the
     * previous native reader, and a forced merge at the end.
     */
    private static void nrtReaders(Random r) throws Exception {
        Path dir = Files.createTempDirectory("lucene-rust-selftest");
        NativeReaders readers = new NativeReaders();
        List<String> fields = List.of("body", "tag");
        try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(new StandardAnalyzer()))) {
            DirectoryReader reader = null;
            int id = 0;
            for (int round = 0; round < 6; round++) {
                for (int i = 0; i < 400 + r.nextInt(400); i++, id++) {
                    w.addDocument(doc(r, id));
                }
                for (int i = 0; i < 40; i++) {
                    w.deleteDocuments(new Term("id", Integer.toString(r.nextInt(id))));
                }
                w.updateDocument(new Term("id", Integer.toString(r.nextInt(id))), doc(r, r.nextInt(id)));
                if (round == 5) {
                    w.forceMerge(1);
                }
                DirectoryReader next = reader == null ? DirectoryReader.open(w) : DirectoryReader.openIfChanged(reader, w);
                if (next != null) {
                    if (reader != null) {
                        reader.close();
                    }
                    reader = next;
                }
                List<Query> queries = new ArrayList<>();
                for (int q = 0; q < 60; q++) {
                    queries.add(randomQuery(r, fields, 0));
                }
                compare("nrt round " + round + " (" + reader.leaves().size() + " segments, " + reader.numDeletedDocs() + " deleted)", reader, readers, queries);
            }
            long openBefore = readers.openCount();
            reader.close();
            check(readers.openCount() == openBefore - 1, "closing the Java reader closes its native reader");
            check(readers.openCount() == 0, "no native readers left open: " + readers.openCount());
        }
        try (Stream<Path> s = Files.walk(dir)) {
            s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    private static Document doc(Random r, int id) {
        Document d = new Document();
        d.add(new StringField("id", Integer.toString(id), Field.Store.YES));
        StringBuilder body = new StringBuilder();
        for (int i = 0, n = 1 + r.nextInt(40); i < n; i++) {
            body.append(word(r)).append(' ');
        }
        d.add(new TextField("body", body.toString(), Field.Store.NO));
        d.add(new StringField("tag", word(r), Field.Store.NO));
        d.add(new LongPoint("n", id));
        return d;
    }

    /** Every fixture directory holding a {@code segments_N}. */
    private static List<Path> fixtureIndexes(Path root) throws Exception {
        List<Path> out = new ArrayList<>();
        try (Stream<Path> s = Files.list(root)) {
            for (Path p : s.sorted().toList()) {
                if (Files.isDirectory(p)) {
                    try (Stream<Path> files = Files.list(p)) {
                        if (files.anyMatch(f -> f.getFileName().toString().startsWith("segments_"))) {
                            out.add(p);
                        }
                    }
                }
            }
        }
        return out;
    }

    /** A Java-written fixture: term and boolean queries over its own terms. */
    private static void fixture(Path path) throws Exception {
        NativeReaders readers = new NativeReaders();
        try (FSDirectory d = FSDirectory.open(path); DirectoryReader reader = DirectoryReader.open(d)) {
            NativeReaders.Acquired a = readers.acquire(reader);
            if (a.handle() == 0) {
                // Not every fixture is a searchable index (some hold only vectors, points, or
                // deliberately corrupt files); an unopenable one must fail cleanly, which it did.
                System.out.println("fixture " + path.getFileName() + ": not opened natively (" + a.fallbackReason() + ")");
                return;
            }
            List<Term> terms = new ArrayList<>();
            for (var leaf : reader.leaves()) {
                for (var fi : leaf.reader().getFieldInfos()) {
                    if (fi.getIndexOptions() == org.apache.lucene.index.IndexOptions.NONE) {
                        continue;
                    }
                    var t = leaf.reader().terms(fi.name);
                    if (t == null) {
                        continue;
                    }
                    var it = t.iterator();
                    for (int i = 0; i < 12 && it.next() != null; i++) {
                        terms.add(new Term(fi.name, org.apache.lucene.util.BytesRef.deepCopyOf(it.term())));
                    }
                }
            }
            if (terms.isEmpty()) {
                return;
            }
            Random r = new Random(path.getFileName().toString().hashCode());
            List<Query> queries = new ArrayList<>();
            for (Term t : terms) {
                queries.add(new TermQuery(t));
            }
            for (int i = 0; i < 20; i++) {
                BooleanQuery.Builder b = new BooleanQuery.Builder();
                for (int c = 0; c < 2 + r.nextInt(3); c++) {
                    b.add(new TermQuery(terms.get(r.nextInt(terms.size()))), r.nextBoolean() ? Occur.SHOULD : Occur.MUST);
                }
                queries.add(b.build());
            }
            compare("fixture " + path.getFileName(), reader, readers, queries);
        }
        check(readers.openCount() == 0, "fixture " + path.getFileName() + ": native reader closed with the Java reader");
    }
}
