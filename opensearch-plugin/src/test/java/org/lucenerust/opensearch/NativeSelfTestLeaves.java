/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.InetAddressPoint;
import org.apache.lucene.document.LatLonDocValuesField;
import org.apache.lucene.document.LatLonPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermStates;
import org.apache.lucene.sandbox.search.CoveringQuery;
import org.apache.lucene.sandbox.search.DocValuesMultiRangeQuery;
import org.apache.lucene.search.AutomatonQuery;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.ConstantScoreQuery;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.MultiTermQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.RegexpQuery;
import org.apache.lucene.search.TermInSetQuery;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TermRangeQuery;
import org.apache.lucene.search.WildcardQuery;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.automaton.Automata;
import org.apache.lucene.util.automaton.Operations;
import org.apache.lucene.util.automaton.RegExp;
import org.opensearch.index.fielddata.IndexNumericFieldData;
import org.opensearch.index.fielddata.plain.SortedNumericIndexFieldData;

import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

/**
 * The Lucene leaves that used to run on Lucene over the bridge, in {@link NativeSelfTest}: any
 * {@code AutomatonQuery} and {@code FuzzyQuery} under the constant-score and doc-values rewrites
 * (regexp flags and {@code case_insensitive}, wildcard escapes), the doc-values-only queries,
 * {@code CoveringQuery} with OpenSearch's {@code minimum_should_match_field} source, the two
 * {@code distance_feature} queries, and term queries carrying {@code TermStates} -- each built as
 * OpenSearch builds it, every one encoded (the test fails on any fallback) and compared by {@code
 * NativeSelfTest.compare}: top hits and scores, counts, sorted pages, {@code terminate_after},
 * {@code min_score} and the aggregations.
 */
final class NativeSelfTestLeaves {
    private NativeSelfTestLeaves() {}

    /** Queries encoded and compared, by shape. */
    static final Map<String, int[]> ENCODED = new java.util.TreeMap<>();

    /** The shapes {@link #run} must have compared. */
    static final int SHAPES = 8;

    private static final String[] WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
        + "pi rho sigma tau upsilon phi chi psi omega").split(" ");

    private static String word(Random r) {
        return WORDS[(int) Math.min(WORDS.length - 1, Math.floor(Math.pow(r.nextDouble(), 2.2) * WORDS.length))];
    }

    private static byte[] ip(int v) {
        try {
            return InetAddressPoint.encode(java.net.InetAddress.getByAddress(new byte[] { 10, 0, (byte) (v >> 8), (byte) v }));
        } catch (java.net.UnknownHostException e) {
            throw new AssertionError(e);
        }
    }

    /** {@code NativeSelfTest.doc} plus fields with doc values only, a date, a point and the terms_set minimum. */
    private static Document doc(Random r, int id) {
        Document d = NativeSelfTest.doc(r, id);
        // Doc values only, as OpenSearch indexes a field mapped with "index": false.
        for (int i = 0, n = r.nextInt(3); i < n; i++) {
            d.add(new SortedNumericDocValuesField("dl", r.nextInt(400) - 200));
        }
        if (r.nextInt(5) != 0) {
            d.add(new NumericDocValuesField("dn", r.nextInt(1000)));
        }
        for (int i = 0, n = r.nextInt(3); i < n; i++) {
            d.add(new SortedSetDocValuesField("kd", new BytesRef(word(r) + r.nextInt(5))));
        }
        if (r.nextInt(6) != 0) {
            d.add(new SortedDocValuesField("ks", new BytesRef(word(r))));
        }
        if (r.nextInt(4) != 0) {
            d.add(new SortedSetDocValuesField("ipd", new BytesRef(ip(r.nextInt(300)))));
        }
        // A date (points and doc values), a geo_point, and the terms_set minimum.
        if (r.nextInt(7) != 0) {
            long t = 1_600_000_000_000L + r.nextInt(1_000_000) * 1000L;
            d.add(new LongPoint("date", t));
            d.add(new SortedNumericDocValuesField("date", t));
        }
        if (r.nextInt(6) != 0) {
            double lat = r.nextDouble() * 20 + 40;
            double lon = r.nextDouble() * 20 - 10;
            d.add(new LatLonPoint("loc", lat, lon));
            d.add(new LatLonDocValuesField("loc", lat, lon));
        }
        if (r.nextInt(5) != 0) {
            d.add(new SortedNumericDocValuesField("msm", r.nextInt(4)));
        }
        d.add(new StringField("leaf", "yes", Field.Store.NO));
        return d;
    }

    static void run(Random r, BiConsumer<Boolean, String> check, NativeSelfTestM10.Compare compare) throws Exception {
        Path dir = Files.createTempDirectory("lucene-rust-selftest-leaves");
        NativeReaders readers = new NativeReaders();
        try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(new StandardAnalyzer()))) {
            DirectoryReader reader = null;
            int id = 0;
            for (int round = 0; round < 3; round++) {
                for (int i = 0; i < 700 + r.nextInt(400); i++, id++) {
                    w.addDocument(doc(r, id));
                }
                for (int i = 0; i < 40; i++) {
                    w.deleteDocuments(new Term("id", Integer.toString(r.nextInt(id))));
                }
                if (round == 2) {
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
                List<String> shapes = new ArrayList<>();
                for (int q = 0; q < 25; q++) {
                    add(queries, shapes, "automaton", automaton(r));
                    add(queries, shapes, "fuzzy", fuzzy(r));
                    add(queries, shapes, "doc_values_rewrite", docValuesRewrite(r));
                    add(queries, shapes, "doc_values_numeric", docValuesNumeric(r));
                    add(queries, shapes, "doc_values_ords", docValuesOrds(r));
                    add(queries, shapes, "covering", covering(r));
                    add(queries, shapes, "distance_feature", distanceFeature(r));
                }
                // Each inside a boosted boolean beside a scored clause, and as a filter.
                int base = queries.size();
                for (int q = 0; q < base; q += 2) {
                    add(
                        queries,
                        shapes,
                        shapes.get(q),
                        new BoostQuery(
                            new BooleanQuery.Builder().add(queries.get(q), Occur.MUST)
                                .add(new TermQuery(new Term("body", word(r))), Occur.SHOULD)
                                .build(),
                            0.5f + r.nextInt(4)
                        )
                    );
                }
                for (int q = 0; q < base; q += 3) {
                    add(
                        queries,
                        shapes,
                        shapes.get(q),
                        new BooleanQuery.Builder().add(new TermQuery(new Term("body", word(r))), Occur.SHOULD)
                            .add(queries.get(q), r.nextBoolean() ? Occur.MUST : Occur.FILTER)
                            .build()
                    );
                }
                IndexSearcher searcher = new IndexSearcher(reader);
                for (int q = 0; q < queries.size(); q++) {
                    QueryEncoder.Encoded enc = QueryEncoder.encode(searcher.rewrite(queries.get(q)), f -> true);
                    check.accept(enc.blob() != null, "leaf query does not encode: " + queries.get(q) + " (" + enc.fallbackReason() + ")");
                    if (enc.blob() != null) {
                        ENCODED.computeIfAbsent(shapes.get(q), k -> new int[1])[0]++;
                    }
                }
                compare.run("leaves round " + round + " (" + reader.leaves().size() + " segments, " + reader.numDeletedDocs() + " deleted)", reader, readers, queries);
                termStates(r, searcher, reader, readers, check);
            }
            reader.close();
            check.accept(readers.openCount() == 0, "leaves: no native readers left open");
        }
        try (Stream<Path> s = Files.walk(dir)) {
            s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    private static void add(List<Query> queries, List<String> shapes, String shape, Query q) {
        queries.add(q);
        shapes.add(shape);
    }

    /** {@code case_insensitive}'s match flag as OpenSearch's {@code RegexpQueryBuilder} sets it. */
    @SuppressWarnings("deprecation")
    private static final int CASE_INSENSITIVE = RegExp.ASCII_CASE_INSENSITIVE;

    private static MultiTermQuery.RewriteMethod constantRewrite(Random r) {
        return r.nextBoolean() ? MultiTermQuery.CONSTANT_SCORE_BLENDED_REWRITE : MultiTermQuery.CONSTANT_SCORE_REWRITE;
    }

    /**
     * AutomatonQuery as OpenSearch reaches it: a regexp with flags or {@code case_insensitive}
     * (ASCII_CASE_INSENSITIVE), a wildcard with escapes, an automaton of its own (a keyword {@code
     * prefix} with {@code case_insensitive}), a prefix under the plain constant-score rewrite.
     */
    private static Query automaton(Random r) {
        String w = word(r);
        String field = r.nextBoolean() ? "body" : "tag";
        return switch (r.nextInt(6)) {
            case 0 -> new RegexpQuery(new Term(field, w.substring(0, 2).toUpperCase(java.util.Locale.ROOT) + ".*"), RegExp.ALL, CASE_INSENSITIVE, RegexpQuery.DEFAULT_PROVIDER, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT, constantRewrite(r));
            case 1 -> new RegexpQuery(new Term(field, w.charAt(0) + "[a-m]+"), RegExp.NONE, 0, RegexpQuery.DEFAULT_PROVIDER, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT, constantRewrite(r));
            case 2 -> new WildcardQuery(new Term(field, "\\" + w.charAt(0) + w.charAt(1) + "*"), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT, constantRewrite(r));
            case 3 -> new AutomatonQuery(
                new Term(field, w),
                Operations.determinize(Operations.union(List.of(Automata.makeString(w), Automata.makeString(word(r)), Automata.makeString("nomatch"))), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT),
                false,
                constantRewrite(r)
            );
            case 4 -> new PrefixQuery(new Term(field, w.substring(0, 1 + r.nextInt(2))), MultiTermQuery.CONSTANT_SCORE_REWRITE);
            default -> new TermRangeQuery(field, new BytesRef(w), new BytesRef(word(r)), r.nextBoolean(), r.nextBoolean(), MultiTermQuery.CONSTANT_SCORE_REWRITE);
        };
    }

    /** FuzzyQuery under the constant-score rewrites: edits 0-2, prefixes, an empty term. */
    private static Query fuzzy(Random r) {
        String w = word(r);
        String text = switch (r.nextInt(5)) {
            case 0 -> "";
            case 1 -> w.substring(0, Math.max(1, w.length() - 1)) + "q";
            default -> w;
        };
        int edits = r.nextInt(3);
        int prefix = r.nextInt(4) == 0 ? 10 : r.nextInt(2);
        return new FuzzyQuery(new Term(r.nextBoolean() ? "body" : "tag", text), edits, prefix, 50, r.nextBoolean(), constantRewrite(r));
    }

    /** Multi-term queries on a doc-values-only keyword under DOC_VALUES_REWRITE. */
    private static Query docValuesRewrite(Random r) {
        String field = r.nextBoolean() ? "kd" : "ks";
        String w = word(r);
        MultiTermQuery.RewriteMethod dv = MultiTermQuery.DOC_VALUES_REWRITE;
        return switch (r.nextInt(7)) {
            case 0 -> new PrefixQuery(new Term(field, w.substring(0, 1 + r.nextInt(2))), dv);
            case 1 -> new WildcardQuery(new Term(field, w.charAt(0) + "*" + r.nextInt(5)), Operations.DEFAULT_DETERMINIZE_WORK_LIMIT, dv);
            case 2 -> new RegexpQuery(new Term(field, w.substring(0, 2).toUpperCase(java.util.Locale.ROOT) + ".*"), RegExp.ALL, CASE_INSENSITIVE, RegexpQuery.DEFAULT_PROVIDER, Operations.DEFAULT_DETERMINIZE_WORK_LIMIT, dv);
            case 3 -> new FuzzyQuery(new Term(field, w + "1"), 1 + r.nextInt(2), 0, 50, true, dv);
            case 4 -> new TermRangeQuery(field, r.nextBoolean() ? new BytesRef(w) : null, new BytesRef(word(r)), r.nextBoolean(), r.nextBoolean(), dv);
            default -> {
                List<BytesRef> terms = new ArrayList<>();
                for (int i = 0, n = 1 + r.nextInt(30); i < n; i++) {
                    terms.add(new BytesRef(field.equals("kd") ? word(r) + r.nextInt(5) : word(r)));
                }
                yield new TermInSetQuery(dv, field, terms);
            }
        };
    }

    /** {@code SortedNumericDocValuesField.newSlow*Query} on doc-values-only and point fields. */
    private static Query docValuesNumeric(Random r) {
        String field = switch (r.nextInt(3)) {
            case 0 -> "dl";
            case 1 -> "dn";
            default -> "sl";
        };
        long a = r.nextInt(600) - 200;
        long b = a + r.nextInt(300);
        return switch (r.nextInt(5)) {
            case 0 -> SortedNumericDocValuesField.newSlowExactQuery(field, a / 4);
            case 1 -> {
                long[] values = new long[1 + r.nextInt(20)];
                for (int i = 0; i < values.length; i++) {
                    values[i] = r.nextInt(400) - 200;
                }
                if (r.nextInt(4) == 0) {
                    values[0] = Long.MIN_VALUE;
                }
                yield SortedNumericDocValuesField.newSlowSetQuery(field, values);
            }
            case 2 -> SortedNumericDocValuesField.newSlowRangeQuery(field, a, Long.MAX_VALUE - r.nextInt(2));
            default -> SortedNumericDocValuesField.newSlowRangeQuery(field, a, b);
        };
    }

    /** {@code SortedSetDocValuesField.newSlow*Query}, and the sandbox's ip multi-range. */
    private static Query docValuesOrds(Random r) {
        String w = word(r);
        return switch (r.nextInt(4)) {
            case 0 -> SortedSetDocValuesField.newSlowExactQuery(r.nextBoolean() ? "kd" : "kx", new BytesRef(w + (r.nextBoolean() ? "" + r.nextInt(5) : "")));
            case 1 -> SortedSetDocValuesField.newSlowRangeQuery(
                r.nextBoolean() ? "kd" : "ks",
                r.nextInt(3) == 0 ? null : new BytesRef(w),
                r.nextInt(3) == 0 ? null : new BytesRef(word(r)),
                r.nextBoolean(),
                r.nextBoolean()
            );
            case 2 -> {
                int lo = r.nextInt(300);
                yield SortedSetDocValuesField.newSlowRangeQuery("ipd", new BytesRef(ip(lo)), new BytesRef(ip(lo + r.nextInt(80))), true, true);
            }
            default -> {
                DocValuesMultiRangeQuery.SortedSetStabbingBuilder b = new DocValuesMultiRangeQuery.SortedSetStabbingBuilder("ipd");
                for (int i = 0, n = 2 + r.nextInt(6); i < n; i++) {
                    int lo = r.nextInt(300);
                    if (r.nextBoolean()) {
                        b.add(new BytesRef(ip(lo)));
                    } else {
                        b.add(new BytesRef(ip(lo)), new BytesRef(ip(lo + r.nextInt(40))));
                    }
                }
                yield b.build();
            }
        };
    }

    /**
     * {@code terms_set}: OpenSearch's {@code CoveringQuery} over term queries, its minimum from
     * {@code minimum_should_match_field} ({@code TermsSetQueryBuilder.FieldValuesSource}, built by
     * reflection: it is package-private) or a constant.
     */
    private static Query covering(Random r) throws Exception {
        List<Query> terms = new ArrayList<>();
        String field = r.nextBoolean() ? "body" : "kw";
        for (int i = 0, n = 1 + r.nextInt(5); i < n; i++) {
            String w = word(r);
            terms.add(new TermQuery(new Term(field, field.equals("kw") ? w + r.nextInt(20) : w)));
        }
        if (r.nextInt(4) == 0) {
            return new CoveringQuery(terms, LongValuesSource.constant(r.nextInt(3)));
        }
        Class<?> c = Class.forName("org.opensearch.index.query.TermsSetQueryBuilder$FieldValuesSource");
        Constructor<?> k = c.getDeclaredConstructor(IndexNumericFieldData.class);
        k.setAccessible(true);
        LongValuesSource min = (LongValuesSource) k.newInstance(new SortedNumericIndexFieldData("msm", IndexNumericFieldData.NumericType.INT));
        return new CoveringQuery(terms, min);
    }

    /** {@code distance_feature} on a date and on a geo_point, boosted as OpenSearch boosts it. */
    private static Query distanceFeature(Random r) {
        float boost = r.nextBoolean() ? 1f : 0.5f + r.nextInt(4);
        if (r.nextBoolean()) {
            long origin = 1_600_000_000_000L + r.nextInt(1_000_000) * 1000L;
            return org.apache.lucene.document.LongField.newDistanceFeatureQuery("date", boost, origin, 1 + r.nextInt(100_000_000));
        }
        return LatLonPoint.newDistanceFeatureQuery("loc", boost, r.nextDouble() * 20 + 40, r.nextDouble() * 20 - 10, 1 + r.nextInt(500_000));
    }

    /**
     * Term queries carrying {@code TermStates}: states built for another reader (TermQuery ignores
     * them and reads its own, so they encode as the plain term) and empty states built for this one
     * (no statistics, so nothing matches).
     */
    private static void termStates(Random r, IndexSearcher searcher, DirectoryReader reader, NativeReaders readers, BiConsumer<Boolean, String> check)
        throws Exception {
        Term t = new Term("body", word(r));
        Path other = Files.createTempDirectory("lucene-rust-selftest-states");
        try (FSDirectory d = FSDirectory.open(other); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(new StandardAnalyzer()))) {
            w.addDocument(doc(r, 0));
            try (DirectoryReader o = DirectoryReader.open(w)) {
                TermStates foreign = TermStates.build(new IndexSearcher(o), t, true);
                QueryEncoder.Encoded got = QueryEncoder.encode(must(new TermQuery(t, foreign)), f -> true, reader.getContext());
                QueryEncoder.Encoded want = QueryEncoder.encode(must(new TermQuery(t)), f -> true, reader.getContext());
                check.accept(
                    got.blob() != null && java.util.Arrays.equals(got.blob(), want.blob()),
                    "term states of another reader encode as the plain term (" + got.fallbackReason() + ")"
                );
            }
        }
        try (Stream<Path> s = Files.walk(other)) {
            s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
        // Empty states built for this reader: Lucene's TermWeight finds no term state on any leaf.
        TermStates empty = new TermStates(reader.getContext());
        Query emptyTerm = new BooleanQuery.Builder().add(new TermQuery(t, empty), Occur.SHOULD)
            .add(new TermQuery(new Term("body", word(r))), Occur.SHOULD)
            .build();
        QueryEncoder.Encoded none = QueryEncoder.encode(emptyTerm, f -> true, reader.getContext());
        check.accept(none.blob() != null, "empty term states encode (" + none.fallbackReason() + ")");
        if (none.blob() == null) {
            return;
        }
        ENCODED.computeIfAbsent("term_states", k -> new int[1])[0] += 2;
        NativeReaders.Acquired a = readers.acquire(reader);
        int[] docs = new int[10];
        float[] scores = new float[10];
        long[] counts = new long[3];
        int rc = NativeBridge.search(a.handle(), none.blob(), 10, Long.MAX_VALUE, docs, scores, counts);
        // Counted exactly, as the native call with no limit counts.
        var want = searcher.search(emptyTerm, new org.apache.lucene.search.TopScoreDocCollectorManager(10, null, Integer.MAX_VALUE));
        boolean same = rc == NativeBridge.OK && counts[1] == want.totalHits.value() && counts[0] == want.scoreDocs.length;
        for (int i = 0; same && i < counts[0]; i++) {
            same = docs[i] == want.scoreDocs[i].doc && Float.floatToIntBits(scores[i]) == Float.floatToIntBits(want.scoreDocs[i].score);
        }
        check.accept(same, "empty term states search as Lucene's: " + counts[1] + " vs " + want.totalHits.value());
    }

    private static Query must(Query q) {
        return new BooleanQuery.Builder().add(q, Occur.MUST).build();
    }
}
