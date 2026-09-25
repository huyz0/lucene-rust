/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.store.FSDirectory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * M2 T2.7, with the JVM in the loop: the cost of one JNI crossing, and per-query latency of the
 * native path against Lucene's {@link IndexSearcher} on the same NRT reader, in one process.
 *
 * <p>Lucene runs with {@code TopScoreDocCollectorManager(10, 10_000)} -- OpenSearch's default
 * {@code track_total_hits} -- so both sides must produce an exact count up to 10,000. Each figure is
 * the median of {@code ROUNDS} timed batches, alternating the two engines batch by batch so that
 * drift falls on both.
 *
 * <p>Usage: {@code gradle -p opensearch-plugin nativeBench [-Pdocs=N]}.
 */
public final class NativeBench {
    private static final int ROUNDS = 15;

    public static void main(String[] args) throws Exception {
        NativeLibrary.load(Path.of("."));
        int docs = args.length > 0 ? Integer.parseInt(args[0]) : 500_000;

        // An empty crossing: the floor every native call pays.
        long sink = 0;
        for (int i = 0; i < 5_000_000; i++) {
            sink += NativeBridge.abiVersion();
        }
        double[] crossing = new double[ROUNDS];
        for (int r = 0; r < ROUNDS; r++) {
            long t = System.nanoTime();
            for (int i = 0; i < 1_000_000; i++) {
                sink += NativeBridge.abiVersion();
            }
            crossing[r] = (System.nanoTime() - t) / 1_000_000.0;
        }
        System.out.printf("jni_crossing_ns %.1f%n", median(crossing));

        Path dir = Files.createTempDirectory("lucene-rust-bench");
        Random rnd = new Random(1);
        String[] words = new String[2000];
        for (int i = 0; i < words.length; i++) {
            words[i] = "w" + i;
        }
        try (FSDirectory d = FSDirectory.open(dir); IndexWriter w = new IndexWriter(d, new IndexWriterConfig(new StandardAnalyzer()))) {
            for (int i = 0; i < docs; i++) {
                Document doc = new Document();
                StringBuilder b = new StringBuilder();
                for (int k = 0, n = 5 + rnd.nextInt(40); k < n; k++) {
                    // Zipf-ish over 2000 words: a few very dense postings, a long rare tail.
                    b.append(words[(int) (Math.pow(rnd.nextDouble(), 3) * words.length)]).append(' ');
                }
                doc.add(new TextField("body", b.toString(), Field.Store.NO));
                doc.add(new StringField("id", Integer.toString(i), Field.Store.NO));
                w.addDocument(doc);
            }
            for (int i = 0; i < docs / 50; i++) {
                w.deleteDocuments(new Term("id", Integer.toString(rnd.nextInt(docs))));
            }
            try (DirectoryReader reader = DirectoryReader.open(w)) {
                NativeReaders readers = new NativeReaders();
                long handle = readers.acquire(reader).handle();
                IndexSearcher searcher = new IndexSearcher(reader);
                System.out.printf("index: %d docs, %d segments, %d deleted%n", reader.maxDoc(), reader.leaves().size(), reader.numDeletedDocs());
                Map<String, Query> queries = new LinkedHashMap<>();
                queries.put("term dense", tq("w0"));
                queries.put("term mid", tq("w40"));
                queries.put("term rare", tq("w1500"));
                queries.put("or 2", bool(Occur.SHOULD, "w3", "w90"));
                queries.put("or 4", bool(Occur.SHOULD, "w1", "w30", "w300", "w1200"));
                queries.put("and 2", bool(Occur.MUST, "w2", "w20"));
                queries.put("and 3", bool(Occur.MUST, "w5", "w25", "w60"));
                System.out.println("query | lucene_us | native_us | lucene/native | native_no_count_us | lucene/native_no_count");
                for (Map.Entry<String, Query> e : queries.entrySet()) {
                    Query q = searcher.rewrite(e.getValue());
                    byte[] blob = QueryEncoder.encode(q, f -> true).blob();
                    int per = 200;
                    double[] lucene = new double[ROUNDS];
                    double[] nat = new double[ROUNDS];
                    double[] natNoCount = new double[ROUNDS];
                    int[] docsOut = new int[10];
                    float[] scoresOut = new float[10];
                    long[] counts = new long[3];
                    for (int warm = 0; warm < 3; warm++) {
                        for (int i = 0; i < per; i++) {
                            sink += searcher.search(q, new TopScoreDocCollectorManager(10, 10_000)).totalHits.value();
                            NativeBridge.search(handle, blob, 10, 10_000, docsOut, scoresOut, counts);
                        }
                    }
                    for (int r = 0; r < ROUNDS; r++) {
                        long t = System.nanoTime();
                        for (int i = 0; i < per; i++) {
                            sink += searcher.search(q, new TopScoreDocCollectorManager(10, 10_000)).totalHits.value();
                        }
                        lucene[r] = (System.nanoTime() - t) / 1000.0 / per;
                        t = System.nanoTime();
                        for (int i = 0; i < per; i++) {
                            NativeBridge.search(handle, blob, 10, 10_000, docsOut, scoresOut, counts);
                            sink += counts[1];
                        }
                        nat[r] = (System.nanoTime() - t) / 1000.0 / per;
                        t = System.nanoTime();
                        for (int i = 0; i < per; i++) {
                            NativeBridge.search(handle, blob, 10, 0, docsOut, scoresOut, counts);
                            sink += counts[0];
                        }
                        natNoCount[r] = (System.nanoTime() - t) / 1000.0 / per;
                    }
                    double l = median(lucene), n = median(nat), nn = median(natNoCount);
                    System.out.printf("%s | %.1f | %.1f | %.2f | %.1f | %.2f%n", e.getKey(), l, n, l / n, nn, l / nn);
                }
            }
        } finally {
            try (var s = Files.walk(dir)) {
                s.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
        if (sink == 42) {
            System.out.println();
        }
    }

    private static Query tq(String w) {
        return new TermQuery(new Term("body", w));
    }

    private static Query bool(Occur occur, String... ws) {
        BooleanQuery.Builder b = new BooleanQuery.Builder();
        for (String w : ws) {
            b.add(tq(w), occur);
        }
        return b.build();
    }

    private static double median(double[] xs) {
        double[] s = xs.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }
}
