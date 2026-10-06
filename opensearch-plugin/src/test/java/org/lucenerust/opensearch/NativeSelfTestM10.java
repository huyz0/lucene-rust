/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.intervals.IntervalQuery;
import org.apache.lucene.queries.intervals.Intervals;
import org.apache.lucene.queries.intervals.IntervalsSource;
import org.apache.lucene.queries.spans.FieldMaskingSpanQuery;
import org.apache.lucene.queries.spans.SpanContainingQuery;
import org.apache.lucene.queries.spans.SpanFirstQuery;
import org.apache.lucene.queries.spans.SpanMultiTermQueryWrapper;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanNotQuery;
import org.apache.lucene.queries.spans.SpanOrQuery;
import org.apache.lucene.queries.spans.SpanPositionRangeQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.queries.spans.SpanWithinQuery;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.CombinedFieldQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.opensearch.common.lucene.search.function.CombineFunction;
import org.opensearch.common.lucene.search.function.FieldValueFactorFunction;
import org.opensearch.common.lucene.search.function.FunctionScoreQuery;
import org.opensearch.common.lucene.search.function.RandomScoreFunction;
import org.opensearch.common.lucene.search.function.ScoreFunction;
import org.opensearch.common.lucene.search.function.WeightFactorFunction;
import org.opensearch.index.fielddata.IndexNumericFieldData;
import org.opensearch.index.fielddata.plain.SortedNumericIndexFieldData;
import org.opensearch.index.query.functionscore.DecayFunction;
import org.opensearch.index.query.functionscore.ExponentialDecayFunctionBuilder;
import org.opensearch.index.query.functionscore.GaussDecayFunctionBuilder;
import org.opensearch.index.query.functionscore.LinearDecayFunctionBuilder;
import org.opensearch.index.search.OpenSearchToParentBlockJoinQuery;
import org.opensearch.search.MultiValueMode;

import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * M10 T10.7's shapes in {@link NativeSelfTest}: {@code nested} (block joins over OpenSearch's
 * nested layout), the span queries, {@code intervals}, {@code combined_fields} and OpenSearch's
 * {@code function_score}, built as OpenSearch builds them, over an NRT block index with deletions --
 * every one encoded (the test fails on any fallback) and compared by {@code NativeSelfTest.compare}:
 * top hits and scores, counts, sorted pages, {@code terminate_after}, {@code min_score} and the
 * aggregations.
 */
final class NativeSelfTestM10 {
    private NativeSelfTestM10() {}

    /** Queries encoded and compared, by shape. */
    static final Map<String, int[]> ENCODED = new java.util.TreeMap<>();

    private static final String[] WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu").split(" ");

    private static String word(Random r) {
        return WORDS[Math.min(WORDS.length - 1, (int) Math.abs(r.nextGaussian() * 4))];
    }

    private static String text(Random r, int max) {
        StringBuilder b = new StringBuilder();
        for (int i = 0, n = 1 + r.nextInt(max); i < n; i++) {
            b.append(word(r)).append(' ');
        }
        return b.toString();
    }

    /**
     * One block: a root document after its nested {@code comments}, as OpenSearch indexes them --
     * each nested document carries the root's {@code id} too, so a delete removes the whole block.
     */
    private static List<Document> block(Random r, int id, Document root) {
        List<Document> docs = new ArrayList<>();
        for (int i = 0, n = r.nextInt(5); i < n; i++) {
            Document c = new Document();
            c.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            c.add(new StringField("_nested_path", "comments", Field.Store.NO));
            c.add(new TextField("c_body", text(r, 8), Field.Store.NO));
            long v = r.nextInt(100);
            c.add(new LongPoint("c_n", v));
            c.add(new SortedNumericDocValuesField("c_n", v));
            docs.add(c);
        }
        root.add(new NumericDocValuesField("_primary_term", 1));
        root.add(new NumericDocValuesField("_seq_no", id));
        root.add(new TextField("title", text(r, 6), Field.Store.NO));
        root.add(new SortedNumericDocValuesField("pos", 1 + r.nextInt(1000)));
        docs.add(root);
        return docs;
    }

    static void run(Random r, BiConsumer<Boolean, String> check, Compare compare) throws Exception {
        Path dir = Files.createTempDirectory("lucene-rust-selftest-m10");
        NativeReaders readers = new NativeReaders();
        try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(new StandardAnalyzer()))) {
            DirectoryReader reader = null;
            int id = 0;
            for (int round = 0; round < 4; round++) {
                for (int i = 0; i < 500 + r.nextInt(300); i++, id++) {
                    w.addDocuments(block(r, id, NativeSelfTest.doc(r, id)));
                }
                for (int i = 0; i < 30; i++) {
                    w.deleteDocuments(new Term("id", Integer.toString(r.nextInt(id))));
                }
                if (round == 3) {
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
                for (int q = 0; q < 40; q++) {
                    add(queries, shapes, "nested", nested(r));
                    add(queries, shapes, "span", roots(span(r, 0)));
                    add(queries, shapes, "intervals", roots(interval(r)));
                    add(queries, shapes, "combined_fields", roots(combined(r)));
                    add(queries, shapes, "function_score", roots(functionScore(r)));
                }
                // Each beside a scored clause and as a filter.
                int n = queries.size();
                for (int q = 0; q < n; q += 3) {
                    Query inner = queries.get(q);
                    add(
                        queries,
                        shapes,
                        shapes.get(q),
                        new BooleanQuery.Builder().add(new TermQuery(new Term("body", word(r))), Occur.SHOULD)
                            .add(inner, r.nextBoolean() ? Occur.MUST : Occur.FILTER)
                            .build()
                    );
                }
                IndexSearcher searcher = new IndexSearcher(reader);
                for (int q = 0; q < queries.size(); q++) {
                    QueryEncoder.Encoded enc = QueryEncoder.encode(searcher.rewrite(queries.get(q)), f -> true);
                    check.accept(enc.blob() != null, "M10 query does not encode: " + queries.get(q) + " (" + enc.fallbackReason() + ")");
                    if (enc.blob() != null) {
                        ENCODED.computeIfAbsent(shapes.get(q), k -> new int[1])[0]++;
                    }
                }
                compare.run("m10 round " + round + " (" + reader.leaves().size() + " segments, " + reader.numDeletedDocs() + " deleted)", reader, readers, queries);
            }
            reader.close();
            check.accept(readers.openCount() == 0, "m10: no native readers left open");
        }
        try (Stream<Path> s = Files.walk(dir)) {
            s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
        // The shapes left to Lucene fall back by name.
        Predicate<String> any = f -> true;
        SpanNearQuery gap = SpanNearQuery.newOrderedNearQuery("body")
            .addClause(new SpanTermQuery(new Term("body", "alpha")))
            .addGap(1)
            .addClause(new SpanTermQuery(new Term("body", "beta")))
            .build();
        check.accept("span_gap".equals(QueryEncoder.encode(gap, any).fallbackReason()), "a span_near with a gap falls back");
        Query minScore = new FunctionScoreQuery(MatchAllDocsQuery.INSTANCE, new WeightFactorFunction(2f), CombineFunction.MULTIPLY, 1f, Float.MAX_VALUE);
        check.accept("function_score_min_score".equals(QueryEncoder.encode(minScore, any).fallbackReason()), "a function_score min_score falls back");
    }

    /** {@code NativeSelfTest.compare}. */
    interface Compare {
        void run(String where, DirectoryReader reader, NativeReaders readers, List<Query> queries) throws Exception;
    }

    /**
     * {@code DefaultSearchContext.buildFilteredQuery} on an index with nested fields: a query that
     * might match nested documents, restricted to the roots. ({@code nested} matches roots only, and
     * is left alone, as {@code NestedHelper} leaves it.)
     */
    private static Query roots(Query q) {
        return new BooleanQuery.Builder().add(q, Occur.MUST).add(new FieldExistsQuery("_primary_term"), Occur.FILTER).build();
    }

    private static void add(List<Query> queries, List<String> shapes, String shape, Query q) {
        queries.add(q);
        shapes.add(shape);
    }

    // ---------------------------------------------------------------------------------------------

    private static final ScoreMode[] SCORE_MODES = ScoreMode.values();

    /** {@code NestedQueryBuilder.doToQuery}: the child query filtered to the path, the root filter. */
    private static Query nested(Random r) {
        Query child = switch (r.nextInt(5)) {
            case 0 -> new TermQuery(new Term("c_body", word(r)));
            case 1 -> new BooleanQuery.Builder().add(new TermQuery(new Term("c_body", word(r))), Occur.SHOULD)
                .add(new TermQuery(new Term("c_body", word(r))), Occur.SHOULD)
                .build();
            case 2 -> LongPoint.newRangeQuery("c_n", r.nextInt(50), 50 + r.nextInt(50));
            case 3 -> new BooleanQuery.Builder().add(new TermQuery(new Term("c_body", word(r))), Occur.MUST)
                .add(LongPoint.newRangeQuery("c_n", 0, r.nextInt(100)), Occur.FILTER)
                .build();
            default -> MatchAllDocsQuery.INSTANCE;
        };
        Query filtered = new BooleanQuery.Builder().add(child, Occur.MUST)
            .add(new TermQuery(new Term("_nested_path", "comments")), Occur.FILTER)
            .build();
        BitSetProducer parents = new QueryBitSetProducer(new FieldExistsQuery("_primary_term"));
        Query q = new OpenSearchToParentBlockJoinQuery(filtered, parents, SCORE_MODES[r.nextInt(SCORE_MODES.length)], null);
        return r.nextInt(4) == 0 ? new BoostQuery(q, 1.5f) : q;
    }

    // ---------------------------------------------------------------------------------------------

    private static SpanQuery spanTerm(Random r) {
        return new SpanTermQuery(new Term("body", word(r)));
    }

    private static SpanQuery span(Random r, int depth) {
        if (depth >= 2) {
            return spanTerm(r);
        }
        return switch (r.nextInt(10)) {
            case 0 -> spanTerm(r);
            case 1 -> new SpanNearQuery(new SpanQuery[] { span(r, depth + 1), span(r, depth + 1) }, r.nextInt(4), r.nextBoolean());
            case 2 -> new SpanOrQuery(span(r, depth + 1), span(r, depth + 1));
            case 3 -> new SpanFirstQuery(span(r, depth + 1), 1 + r.nextInt(8));
            case 4 -> new SpanPositionRangeQuery(span(r, depth + 1), r.nextInt(3), 3 + r.nextInt(10));
            case 5 -> new SpanNotQuery(span(r, depth + 1), spanTerm(r), r.nextInt(3), r.nextInt(3));
            case 6 -> new SpanContainingQuery(
                new SpanNearQuery(new SpanQuery[] { spanTerm(r), spanTerm(r) }, 3 + r.nextInt(4), false),
                spanTerm(r)
            );
            case 7 -> new SpanWithinQuery(new SpanNearQuery(new SpanQuery[] { spanTerm(r), spanTerm(r) }, 3 + r.nextInt(4), true), spanTerm(r));
            case 8 -> new SpanNearQuery(
                new SpanQuery[] { spanTerm(r), new FieldMaskingSpanQuery(new SpanTermQuery(new Term("title", word(r))), "body") },
                2,
                false
            );
            // span_multi: the searcher rewrites it to a span disjunction of the terms.
            default -> new SpanMultiTermQueryWrapper<>(new PrefixQuery(new Term("body", WORDS[r.nextInt(WORDS.length)].substring(0, 2))));
        };
    }

    // ---------------------------------------------------------------------------------------------

    private static IntervalsSource source(Random r, int depth) {
        if (depth >= 2) {
            return Intervals.term(word(r));
        }
        return switch (r.nextInt(16)) {
            case 0 -> Intervals.term(word(r));
            case 1 -> Intervals.ordered(source(r, depth + 1), source(r, depth + 1));
            case 2 -> Intervals.unordered(source(r, depth + 1), source(r, depth + 1));
            case 3 -> Intervals.phrase(word(r), word(r));
            case 4 -> Intervals.or(source(r, depth + 1), source(r, depth + 1));
            case 5 -> Intervals.maxgaps(r.nextInt(3), Intervals.ordered(source(r, depth + 1), source(r, depth + 1)));
            case 6 -> Intervals.maxwidth(2 + r.nextInt(4), Intervals.unordered(source(r, depth + 1), source(r, depth + 1)));
            case 7 -> Intervals.containing(Intervals.unordered(source(r, depth + 1), source(r, depth + 1)), Intervals.term(word(r)));
            case 8 -> Intervals.containedBy(Intervals.term(word(r)), Intervals.ordered(source(r, depth + 1), source(r, depth + 1)));
            case 9 -> Intervals.notContaining(source(r, depth + 1), Intervals.term(word(r)));
            case 10 -> Intervals.overlapping(source(r, depth + 1), source(r, depth + 1));
            case 11 -> Intervals.nonOverlapping(source(r, depth + 1), source(r, depth + 1));
            case 12 -> Intervals.before(Intervals.term(word(r)), Intervals.term(word(r)));
            case 13 -> Intervals.atLeast(2, Intervals.term(word(r)), Intervals.term(word(r)), Intervals.term(word(r)));
            case 14 -> Intervals.prefix(new BytesRef(WORDS[r.nextInt(WORDS.length)].substring(0, 2)), 128);
            default -> Intervals.extend(Intervals.term(word(r)), r.nextInt(2), r.nextInt(2));
        };
    }

    private static Query interval(Random r) {
        IntervalsSource s = r.nextInt(8) == 0 ? Intervals.wildcard(new BytesRef("*a"), 128) : source(r, 0);
        return switch (r.nextInt(3)) {
            case 0 -> new IntervalQuery("body", s);
            case 1 -> new IntervalQuery("body", s, 2f);
            default -> new IntervalQuery("body", s, 1.5f, 0.7f);
        };
    }

    private static Query combined(Random r) {
        CombinedFieldQuery.Builder b = new CombinedFieldQuery.Builder(word(r)).addField("body", 1f + r.nextInt(3)).addField("title", 1f + r.nextInt(3));
        return b.build();
    }

    // ---------------------------------------------------------------------------------------------

    private static final DecayFunction[] DECAYS = {
        new GaussDecayFunctionBuilder("pos", 0, 10, null).getDecayFunction(),
        new ExponentialDecayFunctionBuilder("pos", 0, 10, null).getDecayFunction(),
        new LinearDecayFunctionBuilder("pos", 0, 10, null).getDecayFunction() };

    /** {@code DecayFunctionBuilder.NumericFieldDataScoreFunction} (package-private). */
    private static ScoreFunction decay(Random r) throws Exception {
        Class<?> c = Class.forName("org.opensearch.index.query.functionscore.DecayFunctionBuilder$NumericFieldDataScoreFunction");
        Constructor<?> k = c.getDeclaredConstructor(
            double.class,
            double.class,
            double.class,
            double.class,
            DecayFunction.class,
            IndexNumericFieldData.class,
            MultiValueMode.class,
            String.class
        );
        k.setAccessible(true);
        boolean multi = r.nextBoolean();
        IndexNumericFieldData data = multi
            ? new SortedNumericIndexFieldData("si", IndexNumericFieldData.NumericType.INT)
            : new SortedNumericIndexFieldData("sd", IndexNumericFieldData.NumericType.DOUBLE);
        double origin = multi ? r.nextInt(50) : r.nextGaussian() * 500;
        double scale = multi ? 1 + r.nextInt(20) : 100 + r.nextInt(900);
        MultiValueMode mode = MultiValueMode.values()[r.nextInt(MultiValueMode.values().length)];
        return (ScoreFunction) k.newInstance(origin, scale, 0.1 + 0.8 * r.nextDouble(), (double) r.nextInt(3), DECAYS[r.nextInt(3)], data, mode, null);
    }

    private static final FieldValueFactorFunction.Modifier[] SAFE = {
        FieldValueFactorFunction.Modifier.NONE,
        FieldValueFactorFunction.Modifier.LOG1P,
        FieldValueFactorFunction.Modifier.LOG2P,
        FieldValueFactorFunction.Modifier.LN1P,
        FieldValueFactorFunction.Modifier.LN2P,
        FieldValueFactorFunction.Modifier.SQUARE,
        FieldValueFactorFunction.Modifier.SQRT };

    private static ScoreFunction function(Random r) throws Exception {
        return switch (r.nextInt(7)) {
            case 0 -> new WeightFactorFunction(0.5f + r.nextInt(4));
            case 1 -> new FieldValueFactorFunction(
                "si",
                0.5f + r.nextInt(3),
                SAFE[r.nextInt(SAFE.length)],
                1.0,
                new SortedNumericIndexFieldData("si", IndexNumericFieldData.NumericType.INT)
            );
            case 2 -> new FieldValueFactorFunction(
                "pos",
                1f,
                r.nextBoolean() ? FieldValueFactorFunction.Modifier.RECIPROCAL : FieldValueFactorFunction.Modifier.LOG,
                null,
                new SortedNumericIndexFieldData("pos", IndexNumericFieldData.NumericType.LONG)
            );
            case 3 -> new RandomScoreFunction(r.nextInt(), r.nextInt(), null);
            case 4 -> new RandomScoreFunction(r.nextInt(), r.nextInt(), new SortedNumericIndexFieldData("_seq_no", IndexNumericFieldData.NumericType.LONG));
            case 5 -> new WeightFactorFunction(
                2f,
                new FieldValueFactorFunction(
                    "sf",
                    1f,
                    FieldValueFactorFunction.Modifier.SQUARE,
                    null,
                    new SortedNumericIndexFieldData("sf", IndexNumericFieldData.NumericType.FLOAT)
                )
            );
            default -> decay(r);
        };
    }

    private static Query functionScore(Random r) {
        try {
            Query sub = switch (r.nextInt(3)) {
                case 0 -> new TermQuery(new Term("body", word(r)));
                case 1 -> new BooleanQuery.Builder().add(new TermQuery(new Term("body", word(r))), Occur.SHOULD)
                    .add(new TermQuery(new Term("body", word(r))), Occur.SHOULD)
                    .build();
                default -> MatchAllDocsQuery.INSTANCE;
            };
            int n = 1 + r.nextInt(3);
            ScoreFunction[] functions = new ScoreFunction[n];
            for (int i = 0; i < n; i++) {
                ScoreFunction f = function(r);
                functions[i] = r.nextInt(3) == 0 ? new FunctionScoreQuery.FilterScoreFunction(new TermQuery(new Term("tag", word(r))), f) : f;
            }
            FunctionScoreQuery.ScoreMode mode = FunctionScoreQuery.ScoreMode.values()[r.nextInt(6)];
            CombineFunction combine = CombineFunction.values()[r.nextInt(CombineFunction.values().length)];
            float maxBoost = r.nextBoolean() ? Float.MAX_VALUE : 0.5f + r.nextInt(4);
            Query q = new FunctionScoreQuery(sub, mode, functions, combine, null, maxBoost);
            return r.nextInt(4) == 0 ? new BoostQuery(q, 2.5f) : q;
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }
}
