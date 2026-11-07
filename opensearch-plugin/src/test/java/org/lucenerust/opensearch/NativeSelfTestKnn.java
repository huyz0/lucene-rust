/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.FSDirectory;
import org.lucenerust.opensearch.knn.KnnHitsQuery;
import org.lucenerust.opensearch.knn.KnnLeafVectorQuery;
import org.lucenerust.opensearch.knn.KnnTarget;
import org.lucenerust.opensearch.knn.KnnTopVectorQuery;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.BiConsumer;
import java.util.stream.Stream;

/**
 * The KNN shapes in {@link NativeSelfTest}, over an NRT index of float (three similarities) and byte
 * vectors with deletions: {@link KnnLeafVectorQuery} (top {@code k} per segment and radial,
 * filtered or not) and the {@link KnnHitsQuery} a {@link KnnTopVectorQuery} rewrites to, each
 * encoded and compared by {@code NativeSelfTest.compare}; and the native whole-reader search
 * ({@link KnnEncoder#topHits}) against Lucene's {@code KnnFloatVectorQuery}.
 */
final class NativeSelfTestKnn {
    private NativeSelfTestKnn() {}

    static final int DIM = 12;
    static final String[] FIELDS = { "v_l2", "v_cos", "v_mip", "v_byte" };
    private static final VectorSimilarityFunction[] SIMS = {
        VectorSimilarityFunction.EUCLIDEAN,
        VectorSimilarityFunction.COSINE,
        VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT };

    /** Queries encoded and compared. */
    static int encoded;
    /** Whole-reader searches compared with Lucene's. */
    static int topCompared;

    private static float[] floats(Random r) {
        float[] v = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            v[i] = r.nextFloat() * 2 - 1;
        }
        return v;
    }

    private static byte[] bytes(Random r) {
        byte[] v = new byte[DIM];
        for (int i = 0; i < DIM; i++) {
            v[i] = (byte) (r.nextInt(41) - 20);
        }
        return v;
    }

    private static KnnTarget target(Random r, String field) {
        return field.equals("v_byte") ? KnnTarget.of(bytes(r)) : KnnTarget.of(floats(r));
    }

    static void run(Random r, BiConsumer<Boolean, String> check, NativeSelfTestExtended.Compare compare) throws Exception {
        Path dir = Files.createTempDirectory("lucene-rust-selftest-knn");
        NativeReaders readers = new NativeReaders();
        try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(new StandardAnalyzer()))) {
            DirectoryReader reader = null;
            int id = 0;
            for (int round = 0; round < 3; round++) {
                for (int i = 0; i < 300 + r.nextInt(200); i++, id++) {
                    Document doc = new Document();
                    doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
                    doc.add(new StringField("tag", id % 3 == 0 ? "a" : "b", Field.Store.NO));
                    doc.add(new NumericDocValuesField("_seq_no", id));
                    // Some documents have no vector.
                    if (r.nextInt(10) != 0) {
                        for (int f = 0; f < SIMS.length; f++) {
                            doc.add(new KnnFloatVectorField(FIELDS[f], floats(r), SIMS[f]));
                        }
                        doc.add(new KnnByteVectorField("v_byte", bytes(r), VectorSimilarityFunction.EUCLIDEAN));
                    }
                    w.addDocument(doc);
                }
                for (int i = 0; i < 20; i++) {
                    w.deleteDocuments(new Term("id", Integer.toString(r.nextInt(id))));
                }
                w.commit();
                DirectoryReader next = reader == null ? DirectoryReader.open(w) : DirectoryReader.openIfChanged(reader, w);
                if (next != null) {
                    if (reader != null) {
                        reader.close();
                    }
                    reader = next;
                }
                IndexSearcher searcher = new IndexSearcher(reader);
                Query tagA = new TermQuery(new Term("tag", "a"));
                List<Query> queries = new ArrayList<>();
                for (String field : FIELDS) {
                    for (int q = 0; q < 6; q++) {
                        KnnTarget t = target(r, field);
                        int k = 1 + r.nextInt(20);
                        queries.add(KnnLeafVectorQuery.topK(field, t, k, 50 + r.nextInt(100), null));
                        queries.add(KnnLeafVectorQuery.topK(field, t, k, 100, tagA));
                        float floor = field.equals("v_mip") ? 1.0f + r.nextFloat() : field.equals("v_byte") ? 0.0005f : 0.3f + r.nextFloat() * 0.5f;
                        queries.add(KnnLeafVectorQuery.radial(field, t, floor, r.nextBoolean() ? tagA : null));
                        Query top = searcher.rewrite(new KnnTopVectorQuery(field, t, k, 100, r.nextBoolean() ? tagA : null));
                        queries.add(top);
                        queries.add(new BooleanQuery.Builder().add(new BoostQuery(top, 2f), Occur.MUST).add(tagA, Occur.SHOULD).build());
                    }
                }
                for (Query q : queries) {
                    QueryEncoder.Encoded enc = QueryEncoder.encode(searcher.rewrite(q), f -> true);
                    check.accept(enc.blob() != null, "knn query does not encode: " + q + " (" + enc.fallbackReason() + ")");
                    encoded += enc.blob() != null ? 1 : 0;
                }
                compare.run("knn round " + round + " (" + reader.leaves().size() + " segments, " + reader.numDeletedDocs() + " deleted)", reader, readers, queries);

                // The native whole-reader search finds Lucene's hits with Lucene's scores.
                for (String field : FIELDS) {
                    for (int q = 0; q < 8; q++) {
                        KnnTarget t = target(r, field);
                        int k = 1 + r.nextInt(30);
                        KnnTopVectorQuery top = new KnnTopVectorQuery(field, t, k, 100, q % 2 == 0 ? null : tagA);
                        KnnHitsQuery got = KnnEncoder.topHits(readers, searcher, top);
                        check.accept(got != null, "knn: the native whole-reader search runs");
                        KnnTopVectorQuery.setNative(null);
                        KnnHitsQuery want = (KnnHitsQuery) top.rewrite(searcher);
                        KnnTopVectorQuery.setNative((s, x) -> KnnEncoder.topHits(readers, s, x));
                        if (got != null) {
                            boolean same = Arrays.equals(got.docs(), want.docs());
                            for (int i = 0; same && i < got.docs().length; i++) {
                                same = Math.abs(got.scores()[i] - want.scores()[i]) <= 1e-5f * Math.max(1, Math.abs(want.scores()[i]));
                            }
                            check.accept(same, "knn top " + k + " on " + field + ": native " + Arrays.toString(got.docs()) + " vs Lucene " + Arrays.toString(want.docs()));
                            topCompared++;
                        }
                    }
                }
            }
            KnnTopVectorQuery.setNative(null);
            reader.close();
            check.accept(readers.openCount() == 0, "knn: no native readers left open");
        }
        try (Stream<Path> s = Files.walk(dir)) {
            s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
        check.accept(
            "clause_KnnTopVectorQuery_unrewritten".equals(QueryEncoder.encode(new KnnTopVectorQuery("v", KnnTarget.of(new float[] { 1 }), 1, 1, null), f -> true).fallbackReason()),
            "an unrewritten whole-reader KNN falls back"
        );
    }
}
