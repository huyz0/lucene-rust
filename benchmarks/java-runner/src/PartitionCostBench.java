import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.*;

/**
 * What a partition layout costs on the write side, on stock Lucene: the Java half of
 * {@code scripts/bench-partition-cost.sh}.
 *
 * <pre>
 * corpus &lt;corpus.tsv&gt; &lt;queries.tsv&gt; &lt;docs&gt; &lt;tenants&gt;   PartitionBench's documents, one per line
 * index &lt;corpus.tsv&gt; &lt;dir&gt; plain|tenant|composite &lt;threads&gt; &lt;refresh ms&gt; [vectors.fbin]
 * forcemerge &lt;dir&gt; plain|tenant|composite &lt;segments&gt;
 * </pre>
 *
 * {@code index} adds the corpus from {@code threads} threads in arrival order, with a reader
 * refreshed every {@code refresh ms} and natural merges on the default {@code
 * ConcurrentMergeScheduler} and {@code TieredMergePolicy}, then commits; it prints one CSV row:
 * docs/s, refresh times, merge count and time, segments and size. With vectors each document
 * carries its vector (HNSW, the codec's default) and no body.
 */
public final class PartitionCostBench {

    static Sort sort(String layout) {
        return switch (layout) {
            case "plain" -> null;
            case "tenant" -> new Sort(new SortedSetSortField("tenant", false));
            case "composite" -> new Sort(new SortedSetSortField("tenant", false), new SortedNumericSortField("ts", SortField.Type.LONG, true));
            default -> throw new IllegalArgumentException(layout);
        };
    }

    static void corpus(Path out, Path queries, int docs, int tenants) throws IOException {
        int[] sizes = PartitionBench.tenantSizes(tenants, docs);
        int total = Arrays.stream(sizes).sum();
        int[] owner = new int[total];
        int at = 0;
        for (int t = 0; t < tenants; t++) {
            for (int i = 0; i < sizes[t]; i++) owner[at++] = t;
        }
        PartitionBench.rng = 0x1234_5678_9abc_def1L;
        for (int i = total - 1; i > 0; i--) {
            int j = (int) (PartitionBench.next() * (i + 1));
            int x = owner[i];
            owner[i] = owner[j];
            owner[j] = x;
        }
        PartitionBench.rng = 0x9e3779b97f4a7c15L;
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            for (int i = 0; i < total; i++) {
                long ts = (long) (PartitionBench.next() * PartitionBench.SPAN);
                w.write(i + "\tt" + owner[i] + "\t" + ts + "\t" + PartitionBench.body(40) + "\n");
            }
        }
        PartitionBench.writeQueries(queries, sizes);
    }

    static final class TimedCms extends ConcurrentMergeScheduler {
        final AtomicLong nanos = new AtomicLong();
        final AtomicLong merges = new AtomicLong();

        @Override
        protected void doMerge(MergeScheduler.MergeSource source, MergePolicy.OneMerge merge) throws IOException {
            long t = System.nanoTime();
            super.doMerge(source, merge);
            nanos.addAndGet(System.nanoTime() - t);
            merges.incrementAndGet();
        }
    }

    static float[][] vectors(Path p) throws IOException {
        try (FileChannel ch = FileChannel.open(p)) {
            ByteBuffer b = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()).order(ByteOrder.LITTLE_ENDIAN);
            int n = b.getInt(), d = b.getInt();
            float[][] out = new float[n][d];
            for (float[] v : out) {
                for (int j = 0; j < d; j++) v[j] = b.getFloat();
            }
            return out;
        }
    }

    static double pct(long[] v, double p) {
        if (v.length == 0) return 0;
        return v[Math.min(v.length - 1, (int) (v.length * p))] / 1e6;
    }

    static void index(Path corpus, Path path, String layout, int threads, int refreshMs, Path vecs) throws Exception {
        List<String[]> lines = new ArrayList<>();
        try (BufferedReader r = Files.newBufferedReader(corpus, StandardCharsets.UTF_8)) {
            for (String l; (l = r.readLine()) != null; ) lines.add(l.split("\t", 4));
        }
        float[][] v = vecs == null ? null : vectors(vecs);
        IndexWriterConfig cfg = new IndexWriterConfig(new org.apache.lucene.analysis.core.WhitespaceAnalyzer());
        cfg.setRAMBufferSizeMB(Double.parseDouble(System.getenv().getOrDefault("RAM_MB", "256")));
        cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        TimedCms cms = new TimedCms();
        cfg.setMergeScheduler(cms);
        Sort s = sort(layout);
        if (s != null) cfg.setIndexSort(s);
        AtomicInteger next = new AtomicInteger();
        List<Long> refreshes = Collections.synchronizedList(new ArrayList<>());
        long start = System.nanoTime();
        try (Directory dir = MMapDirectory.open(path); IndexWriter w = new IndexWriter(dir, cfg)) {
            SearcherManager sm = new SearcherManager(w, null);
            AtomicBoolean done = new AtomicBoolean();
            Thread refresher = new Thread(() -> {
                try {
                    while (!done.get()) {
                        Thread.sleep(refreshMs);
                        long t = System.nanoTime();
                        sm.maybeRefreshBlocking();
                        refreshes.add(System.nanoTime() - t);
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            refresher.start();
            Thread[] ts = new Thread[threads];
            for (int k = 0; k < threads; k++) {
                ts[k] = new Thread(() -> {
                    try {
                        for (int i; (i = next.getAndIncrement()) < lines.size(); ) {
                            String[] f = lines.get(i);
                            Document d = new Document();
                            d.add(new StoredField("id", Integer.parseInt(f[0])));
                            d.add(new StringField("tenant", f[1], Field.Store.NO));
                            d.add(new SortedSetDocValuesField("tenant", new BytesRef(f[1])));
                            long t = Long.parseLong(f[2]);
                            d.add(new LongPoint("ts", t));
                            d.add(new SortedNumericDocValuesField("ts", t));
                            if (v == null) {
                                d.add(new TextField("body", f[3], Field.Store.NO));
                            } else {
                                d.add(new KnnFloatVectorField("v", v[i], VectorSimilarityFunction.COSINE));
                            }
                            w.addDocument(d);
                        }
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                });
                ts[k].start();
            }
            for (Thread t : ts) t.join();
            double added = (System.nanoTime() - start) / 1e9;
            done.set(true);
            refresher.join();
            sm.close();
            long c0 = System.nanoTime();
            w.commit();
            double commit = (System.nanoTime() - c0) / 1e9;
            // Wait for running merges, as a shard's last flush does not.
            cms.sync();
            double all = (System.nanoTime() - start) / 1e9;
            int segs;
            try (DirectoryReader r = DirectoryReader.open(w)) {
                segs = r.leaves().size();
            }
            w.commit();
            long[] rf = refreshes.stream().mapToLong(Long::longValue).sorted().toArray();
            System.out.printf(Locale.ROOT, "java,%s,%s,%d,%.1f,%.0f,%.1f,%d,%.1f,%.1f,%.1f,%d,%.1f,%d,%.1f,%.1f%n",
                    layout, v == null ? "fts" : "hnsw", lines.size(), added, lines.size() / added, commit,
                    rf.length, Arrays.stream(rf).sum() / 1e6 / Math.max(1, rf.length), pct(rf, 0.5), pct(rf, 0.99),
                    cms.merges.get(), cms.nanos.get() / 1e9, segs, all, size(path) / 1e6);
        }
    }

    static long size(Path p) throws IOException {
        try (var s = Files.list(p)) {
            return s.mapToLong(f -> f.toFile().length()).sum();
        }
    }

    static void forceMerge(Path path, String layout, int segments) throws IOException {
        IndexWriterConfig cfg = new IndexWriterConfig(new org.apache.lucene.analysis.core.WhitespaceAnalyzer());
        cfg.setOpenMode(IndexWriterConfig.OpenMode.APPEND);
        Sort s = sort(layout);
        if (s != null) cfg.setIndexSort(s);
        try (Directory dir = MMapDirectory.open(path); IndexWriter w = new IndexWriter(dir, cfg)) {
            long t = System.nanoTime();
            w.forceMerge(segments);
            w.commit();
            System.out.printf(Locale.ROOT, "java,%s,forcemerge,%d,%.1f,%.1f%n", layout, segments, (System.nanoTime() - t) / 1e9, size(path) / 1e6);
        }
    }

    public static void main(String[] a) throws Exception {
        switch (a[0]) {
            case "corpus" -> corpus(Paths.get(a[1]), Paths.get(a[2]), Integer.parseInt(a[3]), Integer.parseInt(a[4]));
            case "index" -> index(Paths.get(a[1]), Paths.get(a[2]), a[3], Integer.parseInt(a[4]), Integer.parseInt(a[5]), a.length > 6 ? Paths.get(a[6]) : null);
            case "forcemerge" -> forceMerge(Paths.get(a[1]), a[2], Integer.parseInt(a[3]));
            default -> throw new IllegalArgumentException(a[0]);
        }
    }
}
