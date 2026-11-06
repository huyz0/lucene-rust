/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.engine;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.KeepOnlyLastCommitDeletionPolicy;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.NumericUtils;
import org.lucenerust.opensearch.NativeBridge;
import org.lucenerust.opensearch.NativeLibrary;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The plugin's indexing path in one JVM, split into its stages, against Lucene's {@link IndexWriter}
 * on the same documents: {@code encode} ({@link DocumentEncoder} alone), {@code prepare} (encode plus
 * the native decode and analysis, outside the writer's lock), {@code rust} ({@link RustIndexWriter},
 * end to end with a final commit) and {@code lucene}; {@code rust-refresh} and {@code lucene-refresh}
 * also open a near-real-time view every {@code -Drefresh.ms} (default 1000, OpenSearch's default
 * {@code refresh_interval}) while indexing and print the refresh times. Both writers are set up as
 * an OpenSearch 3.9 shard sets its writer up; {@code -Dmerge_on_flush.ms} sets merge-on-flush's
 * wait (default 10000, {@code 0} off) and {@code -Dforce_merge=true} times a final force merge.
 *
 * <p>The documents are shaped as OpenSearch's mappers build them for the 1M-document REST benchmark:
 * {@code _id}, {@code _source}, the sequence-number fields, two text fields, two keywords with doc
 * values, a long, a double, a date and {@code _field_names} for the text fields.
 *
 * <p>Usage: {@code gradle -p opensearch-plugin indexingBench [-Pdocs=N] [-Pthreads=T] [-Pmodes=a,b] [-PrefreshMs=M]
 * [-PmergeOnFlushMs=M] [-PforceMerge=true]}.
 */
public final class IndexingBench {
    private static final String SOFT = "__soft_deletes";
    private static final FieldType ID = type(IndexOptions.DOCS, true);
    private static final FieldType KEYWORD = type(IndexOptions.DOCS, false);

    private static FieldType type(IndexOptions options, boolean stored) {
        FieldType t = new FieldType();
        t.setIndexOptions(options);
        t.setTokenized(false);
        t.setOmitNorms(true);
        t.setStored(stored);
        t.freeze();
        return t;
    }

    /** One pre-generated document's values. */
    record Values(String id, String body, String title, String tag, String category, long num, double price, long date, byte[] source) {}

    public static void main(String[] args) throws Exception {
        NativeLibrary.load(Path.of("."));
        int docs = args.length > 0 ? Integer.parseInt(args[0]) : 200_000;
        int threads = args.length > 1 ? Integer.parseInt(args[1]) : 4;
        String[] modes = (args.length > 2 ? args[2] : "encode,prepare,rust,lucene").split(",");
        int rounds = args.length > 3 ? Integer.parseInt(args[3]) : 2;
        List<Values> corpus = corpus(docs, 42);
        System.out.printf("docs %d threads %d%n", docs, threads);
        for (int r = 0; r < rounds; r++) {
            for (String mode : modes) {
                run(mode.trim(), corpus, threads);
            }
        }
    }

    static List<Values> corpus(int n, long seed) {
        Random rnd = new Random(seed);
        double[] cdf = zipf(200_000, 1.0);
        double[] tagCdf = zipf(1_000, 1.0);
        String[] vocab = new String[200_000];
        for (int i = 0; i < vocab.length; i++) {
            vocab[i] = "t" + Integer.toString(i, 36);
        }
        List<Values> out = new ArrayList<>(n);
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < n; i++) {
            b.setLength(0);
            for (int k = 0, len = 20 + rnd.nextInt(101); k < len; k++) {
                b.append(k == 0 ? "" : " ").append(vocab[sample(cdf, rnd)]);
            }
            String body = b.toString();
            b.setLength(0);
            for (int k = 0, len = 2 + rnd.nextInt(7); k < len; k++) {
                b.append(k == 0 ? "" : " ").append(vocab[sample(cdf, rnd)]);
            }
            String title = b.toString();
            String tag = "tag" + sample(tagCdf, rnd);
            String category = "cat" + rnd.nextInt(50);
            long num = rnd.nextInt(1_000_000);
            double price = rnd.nextDouble() * 1000;
            long date = 1_600_000_000_000L + (long) rnd.nextInt(1_000_000_000) * 100;
            String id = Integer.toString(i);
            String json = "{\"body\":\"" + body + "\",\"title\":\"" + title + "\",\"tag\":\"" + tag + "\",\"category\":\"" + category
                + "\",\"num\":" + num + ",\"price\":" + price + ",\"date\":" + date + "}";
            out.add(new Values(id, body, title, tag, category, num, price, date, json.getBytes(StandardCharsets.UTF_8)));
        }
        return out;
    }

    private static double[] zipf(int n, double s) {
        double[] cdf = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += 1 / Math.pow(i + 1, s);
            cdf[i] = sum;
        }
        for (int i = 0; i < n; i++) {
            cdf[i] /= sum;
        }
        return cdf;
    }

    private static int sample(double[] cdf, Random rnd) {
        int i = java.util.Arrays.binarySearch(cdf, rnd.nextDouble());
        return Math.min(cdf.length - 1, i < 0 ? -i - 1 : i);
    }

    /** The fields OpenSearch's mappers produce for {@code v}, with sequence number {@code seqNo}. */
    static List<IndexableField> document(Values v, long seqNo) {
        List<IndexableField> d = new ArrayList<>(24);
        d.add(new Field("_id", new BytesRef(v.id), ID));
        d.add(new TextField("body", v.body, Field.Store.NO));
        d.add(new TextField("title", v.title, Field.Store.NO));
        d.add(new Field("tag", new BytesRef(v.tag), KEYWORD));
        d.add(new SortedSetDocValuesField("tag", new BytesRef(v.tag)));
        d.add(new Field("category", new BytesRef(v.category), KEYWORD));
        d.add(new SortedSetDocValuesField("category", new BytesRef(v.category)));
        d.add(new LongPoint("num", v.num));
        d.add(new SortedNumericDocValuesField("num", v.num));
        d.add(new DoublePoint("price", v.price));
        d.add(new SortedNumericDocValuesField("price", NumericUtils.doubleToSortableLong(v.price)));
        d.add(new LongPoint("date", v.date));
        d.add(new SortedNumericDocValuesField("date", v.date));
        d.add(new Field("_field_names", "body", KEYWORD));
        d.add(new Field("_field_names", "title", KEYWORD));
        d.add(new LongPoint("_seq_no", seqNo));
        d.add(new NumericDocValuesField("_seq_no", seqNo));
        d.add(new NumericDocValuesField("_primary_term", 1));
        d.add(new NumericDocValuesField("_version", 1));
        d.add(new StoredField("_source", v.source));
        return d;
    }

    static Analyzer analyzer() throws java.io.IOException {
        return CustomAnalyzer.builder().withTokenizer("standard").addTokenFilter("lowercase").build();
    }

    private static void run(String mode, List<Values> corpus, int threads) throws Exception {
        Path root = Files.createTempDirectory("indexing-bench");
        Analyzer analyzer = analyzer();
        BM25Similarity similarity = new BM25Similarity();
        try (FSDirectory dir = FSDirectory.open(root)) {
            try (IndexWriter empty = new IndexWriter(dir, new IndexWriterConfig(analyzer))) {
                empty.commit();
            }
            RustIndexWriter rust = null;
            IndexWriter lucene = null;
            DocumentEncoder encoder = null;
            boolean refresh = mode.endsWith("-refresh");
            String base = refresh ? mode.substring(0, mode.length() - "-refresh".length()) : mode;
            // As an OpenSearch 3.9 shard sets its writer up: its tiered merge policy, two merge
            // threads, and merge-on-flush's 10 s wait at refresh.
            long mergeOnFlush = Long.getLong("merge_on_flush.ms", 10_000);
            org.apache.lucene.index.TieredMergePolicy tiered = new org.apache.lucene.index.TieredMergePolicy();
            tiered.setSegmentsPerTier(10).setMaxMergeAtOnce(30).setFloorSegmentMB(16).setMaxMergedSegmentMB(5 * 1024);
            if (base.equals("lucene")) {
                org.apache.lucene.index.ConcurrentMergeScheduler cms = new org.apache.lucene.index.ConcurrentMergeScheduler();
                cms.setMaxMergesAndThreads(7, 2);
                lucene = new IndexWriter(
                    dir,
                    new IndexWriterConfig(analyzer).setSimilarity(similarity)
                        .setSoftDeletesField(SOFT)
                        .setRAMBufferSizeMB(256)
                        .setMergePolicy(tiered)
                        .setMergeScheduler(cms)
                        .setMaxFullFlushMergeWaitMillis(mergeOnFlush)
                );
            } else {
                rust = RustIndexWriter.open(
                    dir,
                    root,
                    256,
                    false,
                    false,
                    false,
                    false,
                    null,
                    0,
                    IndexWriter.MAX_DOCS,
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
                rust.setMergePolicy(tiered);
                rust.setMergeScheduler(7, 2, true, Double.POSITIVE_INFINITY);
                rust.setMergeOnFlush(mergeOnFlush, false);
                var f = RustIndexWriter.class.getDeclaredField("encoder");
                f.setAccessible(true);
                encoder = (DocumentEncoder) f.get(rust);
            }
            AtomicInteger next = new AtomicInteger();
            Throwable[] failure = new Throwable[1];
            final RustIndexWriter rw = rust;
            final IndexWriter lw = lucene;
            final DocumentEncoder enc = encoder;
            long t0 = System.nanoTime();
            java.util.concurrent.atomic.AtomicBoolean adding = new java.util.concurrent.atomic.AtomicBoolean(true);
            int[] refreshes = new int[1];
            long refreshEvery = Long.getLong("refresh.ms", 1000);
            List<Long> took = new ArrayList<>();
            Thread refresher = new Thread(() -> {
                try {
                    org.apache.lucene.index.DirectoryReader nrt = lw == null ? null : org.apache.lucene.index.DirectoryReader.open(lw);
                    while (adding.get()) {
                        Thread.sleep(refreshEvery);
                        if (adding.get() == false) {
                            break;
                        }
                        long r0 = System.nanoTime();
                        if (rw != null) {
                            long[] nrt2 = new long[2];
                            rw.acquireNrt(nrt2);
                            rw.releaseHold(nrt2[0]);
                        } else {
                            var newer = org.apache.lucene.index.DirectoryReader.openIfChanged(nrt);
                            if (newer != null) {
                                nrt.close();
                                nrt = newer;
                            }
                        }
                        took.add(System.nanoTime() - r0);
                        refreshes[0]++;
                    }
                    if (nrt != null) {
                        nrt.close();
                    }
                } catch (Throwable e) {
                    failure[0] = e;
                }
            });
            if (refresh) {
                refresher.start();
            }
            Thread[] ts = new Thread[threads];
            for (int t = 0; t < threads; t++) {
                ts[t] = new Thread(() -> {
                    try {
                        long[] id = new long[1];
                        for (int i; (i = next.getAndIncrement()) < corpus.size();) {
                            List<IndexableField> doc = document(corpus.get(i), i);
                            switch (base) {
                                case "encode" -> enc.add(List.of(doc));
                                case "prepare" -> {
                                    Blob b = enc.add(List.of(doc));
                                    if (NativeBridge.writerPrepare(b.array(), b.length(), id) != NativeBridge.OK) {
                                        throw new IllegalStateException(NativeBridge.lastError());
                                    }
                                    NativeBridge.preparedFree(id[0]);
                                }
                                case "rust" -> rw.addDocument(doc);
                                case "lucene" -> lw.addDocument(doc);
                                default -> throw new IllegalArgumentException(mode);
                            }
                        }
                    } catch (Throwable e) {
                        failure[0] = e;
                    }
                });
                ts[t].start();
            }
            for (Thread t : ts) {
                t.join();
            }
            long added = System.nanoTime();
            adding.set(false);
            if (refresh) {
                refresher.join();
            }
            if (failure[0] != null) {
                throw new RuntimeException(failure[0]);
            }
            long fm = 0;
            if (rust != null) {
                if (base.equals("rust")) {
                    rust.commit();
                }
                if (Boolean.getBoolean("force_merge")) {
                    rust.commit();
                    long f0 = System.nanoTime();
                    rust.forceMerge(1, true);
                    fm = System.nanoTime() - f0;
                }
                rust.close();
            } else {
                lucene.commit();
                if (Boolean.getBoolean("force_merge")) {
                    long f0 = System.nanoTime();
                    lucene.forceMerge(1, true);
                    fm = System.nanoTime() - f0;
                }
                lucene.close();
            }
            if (fm > 0) {
                long bytes = 0;
                for (String f : dir.listAll()) {
                    bytes += dir.fileLength(f);
                }
                System.out.printf("%-14s force merge %.2fs, index %.1f MB%n", mode, fm / 1e9, bytes / 1048576.0);
            }
            long done = System.nanoTime();
            double secs = (done - t0) / 1e9;
            System.out.printf(
                "%-14s %8.0f docs/s  add %.2fs  commit %.2fs  refreshes %d%n",
                mode,
                corpus.size() / secs,
                (added - t0) / 1e9,
                (done - added) / 1e9,
                refreshes[0]
            );
            if (took.isEmpty() == false) {
                List<Long> sorted = new ArrayList<>(took);
                sorted.sort(null);
                long total = 0;
                for (long t : sorted) {
                    total += t;
                }
                System.out.printf(
                    "%-14s refresh total %.2fs  mean %.1fms  p50 %.1fms  p90 %.1fms  max %.1fms%n",
                    mode,
                    total / 1e9,
                    total / 1e6 / sorted.size(),
                    sorted.get(sorted.size() / 2) / 1e6,
                    sorted.get(sorted.size() * 9 / 10) / 1e6,
                    sorted.get(sorted.size() - 1) / 1e6
                );
            }
        } finally {
            try (var walk = Files.walk(root)) {
                walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }
}
