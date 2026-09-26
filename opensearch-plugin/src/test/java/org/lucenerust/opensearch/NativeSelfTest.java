/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
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
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopFieldCollectorManager;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.TermInSetQuery;
import org.apache.lucene.index.MultiReader;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.store.FSDirectory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
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
    private static int sortedChecks;
    private static int trackedPages;
    private static int aggChecks;

    public static void main(String[] args) throws Exception {
        NativeLibrary.load(Path.of("."));
        check(NativeBridge.abiVersion() == NativeBridge.EXPECTED_ABI_VERSION, "ABI handshake");

        jniErrorPaths();
        encoderMatrix();
        nrtReaders(new Random(42));
        softDeletes(new Random(7));
        int compared = 0;
        for (Path fixture : fixtureIndexes(Path.of(args[0]))) {
            compared += fixture(fixture) ? 1 : 0;
        }
        // A regression that stopped fixtures opening natively would otherwise pass silently.
        check(compared >= 20, "fixtures compared natively: " + compared);
        check(trackedPages >= 20, "sorted pages tracking the max score: " + trackedPages);
        check(aggChecks >= 100, "queries aggregated natively: " + aggChecks);
        System.out.printf(
            "NativeSelfTest: %d checks, %d failures; %d of %d compared scores bit-exact; %d sorted pages compared (%d tracking the max score); %d aggregations%n",
            checks,
            failures,
            bitExact,
            scored,
            sortedChecks,
            trackedPages,
            aggChecks
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
        long[] counts = new long[3];
        byte[] blob = QueryEncoder.encode(new TermQuery(new Term("f", "t")), f -> true).blob();
        int rc = NativeBridge.search(12345L, blob, 4, Long.MAX_VALUE, docs, scores, counts);
        check(rc == NativeBridge.INVALID_HANDLE, "search on a fabricated handle -> INVALID_HANDLE, got " + rc);
        check(NativeBridge.lastError().contains("unknown"), "lastError names the bad handle: " + NativeBridge.lastError());
        check(NativeBridge.closeReader(0) == NativeBridge.INVALID_HANDLE, "closing handle 0");
        // Marshalling failures are status codes, never exceptions.
        check(NativeBridge.search(1, null, 4, Long.MAX_VALUE, docs, scores, counts) == 10, "null query blob -> InvalidArgument");
        check(NativeBridge.search(1, blob, -1, Long.MAX_VALUE, docs, scores, counts) == 10, "negative topN -> InvalidArgument");
        check(NativeBridge.search(1, blob, 8, Long.MAX_VALUE, docs, scores, counts) == 8, "short output arrays -> BufferTooSmall");
        check(NativeBridge.search(1, blob, 4, Long.MAX_VALUE, null, scores, counts) == 10, "null output array -> InvalidArgument");
        check(NativeBridge.openReader(null, new byte[0], 1, 0, new int[0], null, new long[1]) == 10, "null path -> InvalidArgument");
        check(
            NativeBridge.openReader(new byte[] { '/' }, new byte[0], 1, 0, new int[2], new long[1][], new long[1]) == 10,
            "liveDocs shorter than maxDocs -> InvalidArgument"
        );
    }

    // --- QueryEncoder --------------------------------------------------------------------

    private static void encoderMatrix() throws Exception {
        Query t = new TermQuery(new Term("body", "a"));
        check(QueryEncoder.encode(t, f -> true).blob() != null, "TermQuery encodes");
        check(QueryEncoder.encode(new BoostQuery(t, 1f), f -> true).blob() != null, "unit BoostQuery encodes");
        check(QueryEncoder.encode(new BoostQuery(t, 2f), f -> true).blob() != null, "boosted encodes");
        check(QueryEncoder.encode(new ConstantScoreQuery(t), f -> true).blob() != null, "constant score encodes");
        Query wrappedWildcard = new ConstantScoreQuery(new WildcardQuery(new Term("body", "a*b")));
        check(
            "clause_WildcardQuery".equals(QueryEncoder.encode(wrappedWildcard, f -> true).fallbackReason()),
            "wrapped wildcard falls back"
        );
        check("field_similarity".equals(QueryEncoder.encode(t, f -> false).fallbackReason()), "rejected field falls back");
        check(QueryEncoder.encode(MatchAllDocsQuery.INSTANCE, f -> true).blob() != null, "match_all encodes");
        Query phrase = new BooleanQuery.Builder().add(t, Occur.MUST).add(new PhraseQuery("body", "a", "b"), Occur.SHOULD).build();
        check(QueryEncoder.encode(phrase, f -> true).blob() != null, "a phrase clause encodes");
        Query gap = new PhraseQuery.Builder().add(new Term("body", "a"), 0).add(new Term("body", "b"), 2).build();
        check("phrase_positions".equals(QueryEncoder.encode(gap, f -> true).fallbackReason()), "a phrase with a gap falls back");
        Query negative = new BooleanQuery.Builder().add(t, Occur.MUST_NOT).build();
        check(QueryEncoder.encode(negative, f -> true).blob() != null, "pure negative encodes (and matches nothing)");
        Query t2 = new TermQuery(new Term("body", "b"));
        check(QueryEncoder.isFast(t) && QueryEncoder.isFast(new ConstantScoreQuery(t)), "terms are fast");
        check(QueryEncoder.isFast(new BooleanQuery.Builder().add(t, Occur.SHOULD).add(t2, Occur.SHOULD).build()), "disjunction is fast");
        check(QueryEncoder.isFast(new BooleanQuery.Builder().add(t, Occur.MUST).add(t2, Occur.FILTER).build()), "conjunction is fast");
        // Since read path R1 every encodable shape is measured at least as fast as Lucene.
        check(QueryEncoder.isFast(new BooleanQuery.Builder().add(t, Occur.MUST).add(t2, Occur.SHOULD).build()), "mixed is fast");
        check(QueryEncoder.isFast(new BooleanQuery.Builder().add(t, Occur.MUST).add(t2, Occur.MUST_NOT).build()), "must_not is fast");
        check(QueryEncoder.isFast(new BoostQuery(t, 2f)), "boost is fast");
        check(
            QueryEncoder.isFast(new BooleanQuery.Builder().add(t, Occur.SHOULD).add(t2, Occur.SHOULD).setMinimumNumberShouldMatch(2).build()),
            "msm 2 is fast"
        );
        check(QueryEncoder.isFast(new DisjunctionMaxQuery(List.of(t, t2), 0.3f)), "dismax is fast");
        check(QueryEncoder.isFast(wrappedWildcard) == false, "a wildcard is not encodable yet");
        check(QueryEncoder.encode(new BooleanQuery.Builder().build(), f -> true).blob() != null, "empty encodes (and matches nothing)");
        check(
            "query_WildcardQuery".equals(QueryEncoder.encode(new WildcardQuery(new Term("body", "a*b")), f -> true).fallbackReason()),
            "an unsupported root reports query_, not clause_"
        );
        check(
            "wildcard_escape".equals(
                QueryEncoder.encode(new IndexSearcher(new MultiReader()).rewrite(new WildcardQuery(new Term("body", "a\\*b"))), f -> true)
                    .fallbackReason()
            ),
            "an escaped wildcard falls back"
        );
        check(
            "points_width".equals(QueryEncoder.encode(IntPoint.newRangeQuery("i", 1, 5), f -> true).fallbackReason()),
            "a 4-byte points range falls back"
        );
        check(QueryEncoder.encode(LongPoint.newRangeQuery("n", 1, 5), f -> true).blob() != null, "a long range encodes");
        check(SortEncoder.encode(new Sort(SortField.FIELD_SCORE, SortField.FIELD_DOC), null).blob() != null, "score, doc sort encodes");
        check(
            SortEncoder.encode(new Sort(new org.apache.lucene.search.SortedSetSortField("k", false)), null).blob() != null,
            "a keyword sort encodes"
        );
        check(
            SortEncoder.encode(
                new Sort(new org.apache.lucene.search.SortedSetSortField("k", false, org.apache.lucene.search.SortedSetSelector.Type.MIDDLE_MIN)),
                null
            ).fallbackReason().startsWith("sort_"),
            "a keyword sort by a middle value falls back"
        );
        check(
            "search_after_type".equals(
                SortEncoder.encode(new Sort(new org.apache.lucene.search.SortedSetSortField("k", false)), new FieldDoc(1, 0f, new Object[] { "x" }))
                    .fallbackReason()
            ),
            "a keyword search_after that is not a BytesRef falls back"
        );
        check(
            SortEncoder.encode(new Sort(new org.apache.lucene.search.SortedSetSortField("k", false)), new FieldDoc(1, 0f, new Object[] { null }))
                .blob() != null,
            "a missing keyword search_after value encodes"
        );
        check(
            SortEncoder.encode(new Sort(new SortField("n", SortField.Type.LONG)), null).fallbackReason().startsWith("sort_"),
            "a plain numeric SortField falls back"
        );
        check(
            "search_after_fields".equals(SortEncoder.encode(new Sort(SortField.FIELD_DOC), new FieldDoc(1, 0f, new Object[0])).fallbackReason()),
            "search_after with the wrong arity falls back"
        );
        // The native decoder's node cap, 1024 nodes: a boolean of 1023 terms is 1024.
        BooleanQuery.Builder atCap = new BooleanQuery.Builder();
        for (int i = 0; i < 1023; i++) {
            atCap.add(new TermQuery(new Term("body", "t" + i)), Occur.SHOULD);
        }
        check(QueryEncoder.encode(atCap.build(), f -> true).blob() != null, "1024 nodes encode");
        atCap.add(new TermQuery(new Term("body", "t1023")), Occur.SHOULD);
        check(
            "query_too_large".equals(QueryEncoder.encode(atCap.build(), f -> true).fallbackReason()),
            "1025 nodes fall back as query_too_large, not a native error"
        );
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
            Query t = new TermQuery(new Term(field, word(r)));
            return switch (r.nextInt(10)) {
                case 0 -> new ConstantScoreQuery(t);
                case 1 -> new BoostQuery(t, r.nextInt(4) * 0.75f);
                case 2 -> new DisjunctionMaxQuery(
                    List.of(t, new TermQuery(new Term(field, word(r))), new BoostQuery(new TermQuery(new Term(field, word(r))), 1.5f)),
                    r.nextInt(3) * 0.25f
                );
                case 3 -> r.nextInt(3) == 0 ? MatchAllDocsQuery.INSTANCE : t;
                // Phrases need positions: `body` only (`tag` is a keyword).
                case 4 -> new PhraseQuery(r.nextInt(3) == 0 ? r.nextInt(3) : 0, "body", word(r), word(r));
                case 5 -> r.nextInt(2) == 0 ? new PhraseQuery("body", word(r), word(r), word(r)) : t;
                // The multi-term family, rewritten by the searcher to its constant-score wrapper.
                case 8 -> {
                    long a = r.nextInt(400), b = a + r.nextInt(200);
                    yield LongPoint.newRangeQuery("n", a, b);
                }
                case 6 -> new PrefixQuery(new Term(field, word(r).substring(0, 1 + r.nextInt(2))));
                case 7 -> r.nextInt(2) == 0
                    ? new WildcardQuery(new Term(field, word(r).charAt(0) + "?" + "*"))
                    : new TermInSetQuery(field, List.of(new BytesRef(word(r)), new BytesRef(word(r)), new BytesRef(word(r))));
                default -> t;
            };
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
        Query q = b.build();
        return r.nextInt(8) == 0 ? new BoostQuery(new ConstantScoreQuery(q), r.nextInt(3)) : q;
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
            // size: 0 -- no collector, so the count takes its own path (bounds, then counting).
            long[] zero = new long[3];
            long exactCount = searcher.count(rewritten);
            for (long limit : new long[] { Long.MAX_VALUE, 1, Math.max(1, exactCount / 2), Math.max(1, exactCount) }) {
                check(NativeBridge.search(acquired.handle(), enc.blob(), 0, limit, new int[0], new float[0], zero) == NativeBridge.OK, where + ": size 0");
                boolean ok0 = zero[2] == 0 ? zero[1] == exactCount : exactCount > limit && zero[1] > limit && zero[1] <= exactCount;
                check(ok0 && zero[0] == 0, where + ": " + rewritten + " size 0 limit " + limit + " gave " + zero[1] + (zero[2] == 1 ? "+" : "") + ", exact " + exactCount);
            }
            compareSorted(where, searcher, acquired.handle(), rewritten, enc.blob(), new Random(where.hashCode() * 31L + rewritten.hashCode()));
            compareAggs(where + ": " + rewritten, searcher, acquired.handle(), rewritten, enc.blob());
            for (int topN : new int[] { 10, 3 }) {
                TopDocs want = searcher.search(rewritten, topN);
                int[] docs = new int[topN];
                float[] scores = new float[topN];
                long[] counts = new long[3];
                int rc = NativeBridge.search(acquired.handle(), enc.blob(), topN, Long.MAX_VALUE, docs, scores, counts);
                String what = where + ": " + rewritten + " top" + topN;
                check(rc == NativeBridge.OK, what + ": native search status " + rc + " " + NativeBridge.lastError());
                if (rc != NativeBridge.OK) {
                    continue;
                }
                long exact = searcher.count(rewritten);
                check(counts[1] == exact && counts[2] == 0, what + ": count " + counts[1] + " vs " + exact);
                // Lucene's totalHitsThreshold: exact below the limit, a lower bound >= limit at it.
                long limit = exact > 0 ? 1 + new Random(exact).nextLong(exact) : 1;
                long[] limited = new long[3];
                check(NativeBridge.search(acquired.handle(), enc.blob(), topN, limit, docs, scores, limited) == NativeBridge.OK, what + ": limited");
                // Exact up to the limit (Lucene switches to GTE only past it); above it either exact, or
                // a lower bound that itself exceeds the limit.
                boolean ok = limited[2] == 0 ? limited[1] == exact : exact > limit && limited[1] > limit && limited[1] <= exact;
                check(ok, what + ": limit " + limit + " gave " + limited[1] + (limited[2] == 1 ? "+" : "") + ", exact " + exact);
                check(counts[0] == want.scoreDocs.length, what + ": hits " + counts[0] + " vs " + want.scoreDocs.length);
                for (int i = 0; i < Math.min(counts[0], want.scoreDocs.length); i++) {
                    float w = want.scoreDocs[i].score;
                    boolean close = Math.abs(scores[i] - w) <= 1e-5f * Math.max(1f, Math.abs(w));
                    boolean sameDoc = docs[i] == want.scoreDocs[i].doc;
                    scored++;
                    bitExact += sameDoc && Float.floatToIntBits(scores[i]) == Float.floatToIntBits(w) ? 1 : 0;
                    boolean tieSwap = sameDoc == false && close && isTie(want, i, docs[i]);
                    check(
                        close && (sameDoc || tieSwap),
                        what + ": hit " + i + " native (" + docs[i] + ", " + scores[i] + ") lucene (" + want.scoreDocs[i].doc + ", " + w + ")"
                    );
                }
            }
        }
    }

    /** The metric fields and how each reads its stored longs ({@link NativeAggregations}' kinds). */
    private static final String[] AGG_FIELDS = { "sl", "si", "sd", "sf", "missing" };
    private static final byte[] AGG_KINDS = {
        NativeAggregations.LONG,
        NativeAggregations.LONG,
        NativeAggregations.DOUBLE,
        NativeAggregations.FLOAT,
        NativeAggregations.LONG };

    /**
     * {@code NativeBridge.aggregate} against the loop OpenSearch's metric aggregators run (as in
     * {@code fixtures/src/GenMetricAggs.java}): every value of every live match, summed with {@code
     * CompensatedSum}, {@code Math.min}/{@code Math.max} over the values and over each document's
     * first and last. Bit for bit.
     */
    private static void compareAggs(String what, IndexSearcher searcher, long handle, Query query, byte[] blob) throws Exception {
        // Only where the fields are numeric (a fixture may use the names for other doc values).
        for (org.apache.lucene.index.LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
            for (String f : AGG_FIELDS) {
                org.apache.lucene.index.FieldInfo info = leaf.reader().getFieldInfos().fieldInfo(f);
                if (info != null
                    && info.getDocValuesType() != org.apache.lucene.index.DocValuesType.NUMERIC
                    && info.getDocValuesType() != org.apache.lucene.index.DocValuesType.SORTED_NUMERIC) {
                    return;
                }
            }
        }
        NativeAggregations.Plan plan = new NativeAggregations.Plan(
            java.util.stream.IntStream.range(0, AGG_FIELDS.length)
                .mapToObj(i -> new NativeAggregations.Metric("m" + i, NativeAggregations.Kind.STATS, AGG_FIELDS[i], AGG_KINDS[i], null, null, NativeAggregations.DOC_VALUES))
                .toList()
        );
        long[] counts = new long[AGG_FIELDS.length];
        double[] values = new double[AGG_FIELDS.length * NativeAggregations.VALUES];
        int rc = NativeBridge.aggregate(handle, blob, plan.blob(), counts, values);
        check(rc == NativeBridge.OK, what + ": aggregate status " + rc + " " + NativeBridge.lastError());
        if (rc != NativeBridge.OK) {
            return;
        }
        long[] wantCounts = new long[AGG_FIELDS.length];
        double[] want = new double[values.length];
        for (int k = 0; k < AGG_FIELDS.length; k++) {
            want[k * NativeAggregations.VALUES + 2] = Double.POSITIVE_INFINITY;
            want[k * NativeAggregations.VALUES + 3] = Double.NEGATIVE_INFINITY;
            want[k * NativeAggregations.VALUES + 4] = Double.POSITIVE_INFINITY;
            want[k * NativeAggregations.VALUES + 5] = Double.NEGATIVE_INFINITY;
        }
        searcher.search(query, new org.apache.lucene.search.CollectorManager<org.apache.lucene.search.SimpleCollector, Void>() {
            @Override
            public org.apache.lucene.search.SimpleCollector newCollector() {
                return new org.apache.lucene.search.SimpleCollector() {
                    final org.apache.lucene.index.SortedNumericDocValues[] dvs = new org.apache.lucene.index.SortedNumericDocValues[AGG_FIELDS.length];

                    @Override
                    protected void doSetNextReader(org.apache.lucene.index.LeafReaderContext context) throws java.io.IOException {
                        for (int k = 0; k < AGG_FIELDS.length; k++) {
                            dvs[k] = org.apache.lucene.index.DocValues.getSortedNumeric(context.reader(), AGG_FIELDS[k]);
                        }
                    }

                    @Override
                    public void collect(int doc) throws java.io.IOException {
                        for (int k = 0; k < AGG_FIELDS.length; k++) {
                            if (dvs[k].advanceExact(doc) == false) {
                                continue;
                            }
                            int at = k * NativeAggregations.VALUES;
                            int n = dvs[k].docValueCount();
                            wantCounts[k] += n;
                            double first = 0, last = 0;
                            for (int j = 0; j < n; j++) {
                                long raw = dvs[k].nextValue();
                                double v = switch (AGG_KINDS[k]) {
                                    case NativeAggregations.DOUBLE -> org.apache.lucene.util.NumericUtils.sortableLongToDouble(raw);
                                    case NativeAggregations.FLOAT -> org.apache.lucene.util.NumericUtils.sortableIntToFloat((int) raw);
                                    default -> (double) raw;
                                };
                                first = j == 0 ? v : first;
                                last = v;
                                // CompensatedSum.add
                                if (Double.isFinite(v) == false) {
                                    want[at] = v + want[at];
                                }
                                if (Double.isFinite(want[at])) {
                                    double corrected = v + want[at + 1];
                                    double updated = want[at] + corrected;
                                    want[at + 1] = corrected - (updated - want[at]);
                                    want[at] = updated;
                                }
                                want[at + 2] = Math.min(want[at + 2], v);
                                want[at + 3] = Math.max(want[at + 3], v);
                            }
                            want[at + 4] = Math.min(want[at + 4], first);
                            want[at + 5] = Math.max(want[at + 5], last);
                        }
                    }

                    @Override
                    public org.apache.lucene.search.ScoreMode scoreMode() {
                        return org.apache.lucene.search.ScoreMode.COMPLETE_NO_SCORES;
                    }
                };
            }

            @Override
            public Void reduce(java.util.Collection<org.apache.lucene.search.SimpleCollector> collectors) {
                return null;
            }
        });
        boolean same = Arrays.equals(counts, wantCounts);
        for (int i = 0; i < values.length && same; i++) {
            same = Double.doubleToRawLongBits(values[i]) == Double.doubleToRawLongBits(want[i]);
        }
        check(same, what + ": aggregations native " + Arrays.toString(counts) + Arrays.toString(values) + " lucene " + Arrays.toString(wantCounts) + Arrays.toString(want));
        aggChecks++;
    }

    /** A random sort of one to three keys over the self-test documents' sort fields. */
    private static Sort randomSort(Random r) {
        List<SortField> keys = new ArrayList<>();
        for (int i = 0, n = 1 + r.nextInt(3); i < n; i++) {
            boolean reverse = r.nextBoolean();
            var sel = r.nextBoolean() ? org.apache.lucene.search.SortedNumericSelector.Type.MIN : org.apache.lucene.search.SortedNumericSelector.Type.MAX;
            var ssel = r.nextBoolean() ? org.apache.lucene.search.SortedSetSelector.Type.MIN : org.apache.lucene.search.SortedSetSelector.Type.MAX;
            SortField f = switch (r.nextInt(9)) {
                case 6 -> new org.apache.lucene.search.SortedSetSortField("kt", reverse, ssel);
                case 7 -> new org.apache.lucene.search.SortedSetSortField("kw", reverse, ssel);
                case 8 -> new org.apache.lucene.search.SortedSetSortField("kx", reverse, ssel);
                case 0 -> reverse ? new SortField(null, SortField.Type.SCORE, true) : SortField.FIELD_SCORE;
                case 1 -> SortField.FIELD_DOC;
                case 2 -> new org.apache.lucene.search.SortedNumericSortField("sl", SortField.Type.LONG, reverse, sel);
                case 3 -> new org.apache.lucene.search.SortedNumericSortField("si", SortField.Type.INT, reverse, sel);
                case 4 -> new org.apache.lucene.search.SortedNumericSortField("sd", SortField.Type.DOUBLE, reverse, sel);
                default -> new org.apache.lucene.search.SortedNumericSortField("sf", SortField.Type.FLOAT, reverse, sel);
            };
            if (f instanceof org.apache.lucene.search.SortedNumericSortField sn && r.nextBoolean()) {
                boolean high = r.nextBoolean();
                sn.setMissingValue(switch (sn.getNumericType()) {
                    case LONG -> high ? Long.MAX_VALUE : Long.MIN_VALUE;
                    case INT -> high ? Integer.MAX_VALUE : Integer.MIN_VALUE;
                    case DOUBLE -> high ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
                    default -> high ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
                });
            }
            if (f instanceof org.apache.lucene.search.SortedSetSortField ss && r.nextInt(3) != 0) {
                ss.setMissingValue(r.nextBoolean() ? SortField.STRING_LAST : SortField.STRING_FIRST);
            }
            keys.add(f);
        }
        return new Sort(keys.toArray(new SortField[0]));
    }

    /**
     * The native sorted search against {@code TopFieldCollectorManager}: hits and their sort
     * values exactly, the total exactly when Lucene's is exact (and past the threshold when it is
     * a lower bound), then the next page after the last hit.
     */
    private static void compareSorted(String where, IndexSearcher searcher, long handle, Query query, byte[] blob, Random r) throws Exception {
        for (int s = 0; s < 2; s++) {
            Sort sort = randomSort(r);
            int topN = r.nextBoolean() ? 10 : 3;
            int threshold = r.nextBoolean() ? Integer.MAX_VALUE : 1 + r.nextInt(50);
            FieldDoc after = null;
            // track_scores behind another key: Lucene's collector beside a max-score collector.
            boolean track = sort.getSort()[0].getType() != SortField.Type.SCORE && r.nextInt(3) == 0;
            for (int page = 0; page < 2; page++) {
                float[] wantMax = {Float.NEGATIVE_INFINITY};
                TopFieldDocs want = track
                    ? searchTracked(searcher, query, sort, topN, after, threshold, wantMax)
                    : searcher.search(query, new TopFieldCollectorManager(sort, topN, after, threshold));
                SortEncoder.Encoded enc = SortEncoder.encode(sort, after, track);
                String what = where + ": " + query + " sorted " + sort + " top" + topN + " threshold " + threshold + " page " + page
                    + (track ? " tracking the max score" : "");
                check(enc.blob() != null, what + ": sort encodes (" + enc.fallbackReason() + ")");
                if (enc.blob() == null) {
                    return;
                }
                SortField[] keys = sort.getSort();
                int[] docs = new int[topN];
                long[] values = new long[topN * keys.length];
                long[] counts = new long[4];
                byte[][] terms = new byte[1][];
                long limit = threshold == Integer.MAX_VALUE ? Long.MAX_VALUE : threshold;
                int rc = NativeBridge.searchSorted(handle, blob, enc.blob(), topN, limit, docs, values, counts, terms);
                check(rc == NativeBridge.OK, what + ": status " + rc + " " + NativeBridge.lastError());
                if (rc != NativeBridge.OK) {
                    return;
                }
                boolean same = counts[0] == want.scoreDocs.length;
                FieldDoc[] got = SortEncoder.hits(keys, (int) counts[0], docs, values, terms[0]);
                check(SortEncoder.hasTerms(keys) == (terms[0] != null), what + ": terms come back exactly for keyword keys");
                for (int i = 0; same && i < counts[0]; i++) {
                    FieldDoc w = (FieldDoc) want.scoreDocs[i];
                    same = got[i].doc == w.doc && Arrays.equals(got[i].fields, w.fields);
                }
                sortedChecks++;
                check(same, what + ": native " + Arrays.toString(Arrays.copyOf(docs, (int) counts[0])) + " lucene " + Arrays.toString(Arrays.stream(want.scoreDocs).mapToInt(d -> d.doc).toArray()));
                boolean exact = want.totalHits.relation() == TotalHits.Relation.EQUAL_TO;
                check(
                    // Tracked, nothing is skipped on either side: the same count, bound or not.
                    track ? counts[1] == want.totalHits.value() && (counts[2] == 1) == !exact
                        : exact ? counts[2] == 0 && counts[1] == want.totalHits.value() : counts[2] == 1 && counts[1] > threshold,
                    what + ": total " + counts[1] + (counts[2] == 1 ? "+" : "") + " vs " + want.totalHits
                );
                if (track) {
                    float gotMax = Float.intBitsToFloat((int) counts[3]);
                    float luceneMax = Float.isInfinite(wantMax[0]) ? Float.NaN : wantMax[0];
                    check(Float.floatToIntBits(gotMax) == Float.floatToIntBits(luceneMax), what + ": max score " + gotMax + " vs " + luceneMax);
                    trackedPages++;
                }
                if (want.scoreDocs.length == 0) {
                    break;
                }
                after = (FieldDoc) want.scoreDocs[want.scoreDocs.length - 1];
            }
        }
    }

    /** OpenSearch's MaxScoreCollector. */
    private static final class MaxScore extends org.apache.lucene.search.SimpleCollector {
        private final float[] max;
        private org.apache.lucene.search.Scorable scorer;

        MaxScore(float[] max) {
            this.max = max;
        }

        @Override
        public void setScorer(org.apache.lucene.search.Scorable scorer) {
            this.scorer = scorer;
        }

        @Override
        public void collect(int doc) throws java.io.IOException {
            max[0] = Math.max(max[0], scorer.score());
        }

        @Override
        public org.apache.lucene.search.ScoreMode scoreMode() {
            return org.apache.lucene.search.ScoreMode.COMPLETE;
        }
    }

    /** What OpenSearch runs for track_scores behind another key: a MultiCollector of the two. */
    private static TopFieldDocs searchTracked(IndexSearcher searcher, Query query, Sort sort, int topN, FieldDoc after, int threshold, float[] max)
        throws Exception {
        TopFieldCollectorManager tfcm = new TopFieldCollectorManager(sort, topN, after, threshold);
        return searcher.search(query, new org.apache.lucene.search.CollectorManager<org.apache.lucene.search.Collector, TopFieldDocs>() {
            @Override
            public org.apache.lucene.search.Collector newCollector() throws java.io.IOException {
                return org.apache.lucene.search.MultiCollector.wrap(tfcm.newCollector(), new MaxScore(max));
            }

            @Override
            public TopFieldDocs reduce(java.util.Collection<org.apache.lucene.search.Collector> cs) throws java.io.IOException {
                List<org.apache.lucene.search.TopFieldCollector> tops = new ArrayList<>();
                for (org.apache.lucene.search.Collector c : cs) {
                    for (org.apache.lucene.search.Collector sub : ((org.apache.lucene.search.MultiCollector) c).getCollectors()) {
                        if (sub instanceof org.apache.lucene.search.TopFieldCollector t) {
                            tops.add(t);
                        }
                    }
                }
                return tfcm.reduce(tops);
            }
        });
    }

    /** True when {@code doc} is one of the Lucene hits whose score ties hit {@code i}'s. */
    private static boolean isTie(TopDocs td, int i, int doc) {
        float s = td.scoreDocs[i].score;
        for (var hit : td.scoreDocs) {
            if (hit.doc == doc && Math.abs(hit.score - s) <= 1e-5f * Math.max(1f, s)) {
                return true;
            }
        }
        return false;
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

    /**
     * OpenSearch's reader shape: soft deletes through {@link SoftDeletesDirectoryReaderWrapper}, with
     * one middle segment soft-deleted entirely -- the wrapper drops such a leaf, so the native reader
     * must be built from the leaves the searcher actually sees, not from the writer's segment list.
     */
    private static void softDeletes(Random r) throws Exception {
        Path dir = Files.createTempDirectory("lucene-rust-softdeletes");
        NativeReaders readers = new NativeReaders();
        String soft = "__soft_deletes";
        IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer()).setSoftDeletesField(soft)
            .setMergePolicy(org.apache.lucene.index.NoMergePolicy.INSTANCE);
        try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, cfg)) {
            int id = 0;
            for (int seg = 0; seg < 3; seg++) {
                for (int i = 0; i < 300; i++, id++) {
                    Document doc = doc(r, id);
                    doc.add(new StringField("seg", Integer.toString(seg), Field.Store.NO));
                    w.addDocument(doc);
                }
                w.flush();
            }
            // Soft-delete every document of the middle segment, and a scattering elsewhere.
            w.softUpdateDocuments(new Term("seg", "1"), List.of(), new org.apache.lucene.document.NumericDocValuesField(soft, 1));
            for (int i = 0; i < 30; i++) {
                w.softUpdateDocument(new Term("id", Integer.toString(r.nextInt(300))), doc(r, 10_000 + i), new org.apache.lucene.document.NumericDocValuesField(soft, 1));
            }
            try (DirectoryReader reader = new org.apache.lucene.index.SoftDeletesDirectoryReaderWrapper(DirectoryReader.open(w), soft)) {
                check(reader.leaves().size() == 3, "soft deletes: a fully deleted middle segment is dropped (" + reader.leaves().size() + " leaves of 4)");
                List<Query> queries = new ArrayList<>();
                for (int q = 0; q < 60; q++) {
                    queries.add(randomQuery(r, List.of("body", "tag"), 0));
                }
                compare("soft deletes (" + reader.leaves().size() + " leaves, " + reader.numDeletedDocs() + " deleted)", reader, readers, queries);
            }
            check(readers.openCount() == 0 && readers.cachedCount() == 0, "soft deletes: native reader released");
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
        String tag = word(r);
        d.add(new StringField("tag", tag, Field.Store.NO));
        // Keyword sort fields: dense with postings (`kt`), multi-valued and sparse with postings
        // (`kw`), and doc values alone (`kx`).
        d.add(new StringField("kt", tag, Field.Store.NO));
        d.add(new org.apache.lucene.document.SortedSetDocValuesField("kt", new org.apache.lucene.util.BytesRef(tag)));
        for (int i = 0, n = r.nextInt(3); i < n; i++) {
            String k = word(r) + r.nextInt(20);
            d.add(new StringField("kw", k, Field.Store.NO));
            d.add(new org.apache.lucene.document.SortedSetDocValuesField("kw", new org.apache.lucene.util.BytesRef(k)));
        }
        if (r.nextInt(4) != 0) {
            d.add(new org.apache.lucene.document.SortedSetDocValuesField("kx", new org.apache.lucene.util.BytesRef(word(r))));
        }
        d.add(new LongPoint("n", id));
        // Sort fields: doc values and points, with ties, gaps and several values per document.
        if (r.nextInt(8) != 0) {
            long l = r.nextInt(200) - 100;
            d.add(new org.apache.lucene.document.SortedNumericDocValuesField("sl", l));
            d.add(new LongPoint("sl", l));
        }
        for (int i = 0, n = r.nextInt(3); i < n; i++) {
            int v = r.nextInt(50);
            d.add(new org.apache.lucene.document.SortedNumericDocValuesField("si", v));
            d.add(new org.apache.lucene.document.IntPoint("si", v));
        }
        double dv = r.nextInt(10) == 0 ? -0.0 : r.nextGaussian() * 1e3;
        d.add(new org.apache.lucene.document.SortedNumericDocValuesField("sd", org.apache.lucene.util.NumericUtils.doubleToSortableLong(dv)));
        d.add(new org.apache.lucene.document.DoublePoint("sd", dv));
        float fv = (float) r.nextGaussian();
        d.add(new org.apache.lucene.document.SortedNumericDocValuesField("sf", org.apache.lucene.util.NumericUtils.floatToSortableInt(fv)));
        d.add(new org.apache.lucene.document.FloatPoint("sf", fv));
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

    /** A Java-written fixture: term and boolean queries over its own terms; true if it was compared. */
    private static boolean fixture(Path path) throws Exception {
        NativeReaders readers = new NativeReaders();
        try (FSDirectory d = FSDirectory.open(path); DirectoryReader reader = DirectoryReader.open(d)) {
            NativeReaders.Acquired a = readers.acquire(reader);
            if (a.handle() == 0) {
                // Not every fixture is a searchable index (some hold only vectors, points, or
                // deliberately corrupt files); an unopenable one must fail cleanly, which it did.
                System.out.println("fixture " + path.getFileName() + ": not opened natively (" + a.fallbackReason() + ")");
                return false;
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
                return false;
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
        check(readers.cachedCount() == 0, "fixture " + path.getFileName() + ": cache entry evicted with the Java reader");
        return true;
    }
}
