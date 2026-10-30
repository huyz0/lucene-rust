/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.similarities.AfterEffectB;
import org.apache.lucene.search.similarities.AfterEffectL;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.BasicModelG;
import org.apache.lucene.search.similarities.BasicModelIF;
import org.apache.lucene.search.similarities.BasicModelIn;
import org.apache.lucene.search.similarities.BasicModelIne;
import org.apache.lucene.search.similarities.BooleanSimilarity;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.DFISimilarity;
import org.apache.lucene.search.similarities.DFRSimilarity;
import org.apache.lucene.search.similarities.DistributionLL;
import org.apache.lucene.search.similarities.DistributionSPL;
import org.apache.lucene.search.similarities.IBSimilarity;
import org.apache.lucene.search.similarities.IndependenceChiSquared;
import org.apache.lucene.search.similarities.IndependenceSaturated;
import org.apache.lucene.search.similarities.IndependenceStandardized;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.LambdaDF;
import org.apache.lucene.search.similarities.LambdaTTF;
import org.apache.lucene.search.similarities.Normalization;
import org.apache.lucene.search.similarities.NormalizationH1;
import org.apache.lucene.search.similarities.NormalizationH2;
import org.apache.lucene.search.similarities.NormalizationH3;
import org.apache.lucene.search.similarities.NormalizationZ;
import org.apache.lucene.search.similarities.PerFieldSimilarityWrapper;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;
import org.opensearch.lucene.similarity.LegacyBM25Similarity;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

/**
 * Per-field similarities (the {@code field_similarity} fallback, now native): the searcher's {@link
 * PerFieldSimilarityWrapper} gives {@code body} and {@code note} each a similarity OpenSearch's
 * {@code SimilarityProviders} builds, and every query shape that scores through it -- terms,
 * booleans, phrases, spans, fuzzy's blended terms, dis_max -- runs through {@link NativeSelfTest}'s
 * comparisons with the table {@link SimilarityEncoder} sends.
 */
final class NativeSelfTestSimilarity {
    private NativeSelfTestSimilarity() {}

    /** Similarities compared, and the queries each compared with. */
    static final java.util.Map<String, int[]> COMPARED = new java.util.TreeMap<>();

    private static final String[] WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split(" ");

    private static String word(Random r) {
        return WORDS[(int) Math.min(WORDS.length - 1, Math.floor(Math.pow(r.nextDouble(), 1.8) * WORDS.length))];
    }

    private static Normalization normalization(Random r) {
        return switch (r.nextInt(5)) {
            case 0 -> new Normalization.NoNormalization();
            case 1 -> new NormalizationH1(0.5f + r.nextInt(4));
            case 2 -> new NormalizationH2(0.5f + r.nextInt(4));
            case 3 -> new NormalizationH3(100f + r.nextInt(1000));
            default -> new NormalizationZ(0.1f + r.nextInt(5) / 10f);
        };
    }

    /** One of each kind the encoder sends, with random parameters. */
    @SuppressWarnings("deprecation") // LegacyBM25Similarity
    private static List<Similarity> similarities(Random r) {
        List<Similarity> out = new ArrayList<>();
        out.add(new BM25Similarity(0.5f + r.nextInt(30) / 10f, r.nextInt(11) / 10f, r.nextBoolean()));
        out.add(new BooleanSimilarity());
        out.add(
            new DFRSimilarity(
                switch (r.nextInt(4)) {
                    case 0 -> new BasicModelG();
                    case 1 -> new BasicModelIF();
                    case 2 -> new BasicModelIn();
                    default -> new BasicModelIne();
                },
                r.nextBoolean() ? new AfterEffectB() : new AfterEffectL(),
                normalization(r)
            )
        );
        out.add(
            new IBSimilarity(
                r.nextBoolean() ? new DistributionLL() : new DistributionSPL(),
                r.nextBoolean() ? new LambdaDF() : new LambdaTTF(),
                normalization(r)
            )
        );
        out.add(
            new DFISimilarity(
                switch (r.nextInt(3)) {
                    case 0 -> new IndependenceStandardized();
                    case 1 -> new IndependenceSaturated();
                    default -> new IndependenceChiSquared();
                }
            )
        );
        out.add(new LMDirichletSimilarity(100f + r.nextInt(3000)));
        out.add(new LMJelinekMercerSimilarity(0.05f + r.nextInt(19) / 20f));
        out.add(new ClassicSimilarity(r.nextBoolean()));
        out.add(new LegacyBM25Similarity(0.5f + r.nextInt(30) / 10f, r.nextInt(11) / 10f, r.nextBoolean()));
        return out;
    }

    private static Query query(Random r) {
        String f = r.nextInt(3) == 0 ? "note" : "body";
        return switch (r.nextInt(7)) {
            case 0 -> new TermQuery(new Term(f, word(r)));
            case 1 -> {
                BooleanQuery.Builder b = new BooleanQuery.Builder();
                for (int i = 0, n = 2 + r.nextInt(3); i < n; i++) {
                    b.add(new TermQuery(new Term(r.nextBoolean() ? "body" : "note", word(r))), r.nextInt(4) == 0 ? Occur.MUST : Occur.SHOULD);
                }
                yield b.build();
            }
            case 2 -> new PhraseQuery(r.nextInt(3), f, word(r), word(r));
            case 3 -> new SpanNearQuery(new SpanQuery[] { new SpanTermQuery(new Term(f, word(r))), new SpanTermQuery(new Term(f, word(r))) }, 1 + r.nextInt(3), r.nextBoolean());
            case 4 -> new FuzzyQuery(new Term(f, word(r) + "x"), 1 + r.nextInt(2));
            case 5 -> new DisjunctionMaxQuery(List.of(new TermQuery(new Term("body", word(r))), new TermQuery(new Term("note", word(r)))), r.nextInt(5) / 10f);
            default -> new BoostQuery(
                new BooleanQuery.Builder().add(new TermQuery(new Term(f, word(r))), Occur.MUST)
                    .add(new PhraseQuery(f, word(r), word(r)), Occur.SHOULD)
                    .build(),
                0.5f + r.nextInt(4)
            );
        };
    }

    static void run(Random r, BiConsumer<Boolean, String> check, Compare withSimilarity) throws Exception {
        Path dir = Files.createTempDirectory("lucene-rust-selftest-similarity");
        NativeReaders readers = new NativeReaders();
        try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(new StandardAnalyzer()))) {
            DirectoryReader reader = null;
            int id = 0;
            for (int round = 0; round < 3; round++) {
                for (int i = 0; i < 600 + r.nextInt(300); i++, id++) {
                    Document doc = NativeSelfTest.doc(r, id);
                    StringBuilder note = new StringBuilder();
                    for (int k = 0, n = r.nextInt(12); k < n; k++) {
                        note.append(word(r)).append(' ');
                    }
                    doc.add(new TextField("note", note.toString(), Field.Store.NO));
                    w.addDocument(doc);
                }
                for (int i = 0; i < 30; i++) {
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
                List<Similarity> sims = similarities(r);
                for (int s = 0; s < sims.size(); s++) {
                    Similarity body = sims.get(s);
                    Similarity note = sims.get((s + 1 + r.nextInt(sims.size() - 1)) % sims.size());
                    Similarity perField = new PerFieldSimilarityWrapper() {
                        @Override
                        public Similarity get(String name) {
                            return name.equals("body") ? body : name.equals("note") ? note : new BM25Similarity();
                        }
                    };
                    List<Query> queries = new ArrayList<>();
                    for (int q = 0; q < 24; q++) {
                        queries.add(query(r));
                    }
                    String name = body.getClass().getSimpleName();
                    COMPARED.computeIfAbsent(name, k -> new int[1])[0] += queries.size();
                    withSimilarity.run("similarity round " + round + " body " + body + " note " + note, reader, readers, queries, perField);
                }
                // The encoder refuses a similarity it does not know (a subclass may score differently),
                // and sends nothing for the default.
                SimilarityEncoder unknown = new SimilarityEncoder(new ClassicSimilarity() {});
                check.accept(unknown.test("body") == false, "a ClassicSimilarity subclass is not sent");
                SimilarityEncoder plain = new SimilarityEncoder(new BM25Similarity());
                byte[] blob = { 1, 2, 3 };
                check.accept(plain.test("body") && plain.prefix(blob) == blob, "the default BM25 sends no table");
            }
            reader.close();
            check.accept(readers.openCount() == 0, "similarity: no native readers left open");
        }
        try (Stream<Path> s = Files.walk(dir)) {
            s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    interface Compare {
        void run(String where, DirectoryReader reader, NativeReaders readers, List<Query> queries, Similarity similarity) throws Exception;
    }
}
