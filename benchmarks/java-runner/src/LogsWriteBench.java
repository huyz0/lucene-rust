import java.io.BufferedReader;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.SearcherManager;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * The stock-Lucene half of {@code scripts/bench-logs-write.sh}: the documents
 * {@code crates/lucene-search/examples/logs_write.rs} indexes, over its corpus.
 *
 * <pre>
 * index &lt;corpus.tsv&gt; &lt;dir&gt; &lt;threads&gt; &lt;refresh ms&gt;
 * forcemerge &lt;dir&gt; &lt;segments&gt;
 * </pre>
 *
 * Each indexing thread builds its documents as it adds them, as on the Rust side; the
 * {@code message} field is analyzed at add time ({@code WhitespaceAnalyzer}), which the
 * Rust side receives already inverted.
 */
public final class LogsWriteBench {
    static final String[] KEYWORDS = {"host", "status", "method", "service", "region", "agent"};
    static final String[] LONGS = {"ts", "bytes", "latency", "count"};

    static double cpuSeconds() {
        return ((com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean())
                .getProcessCpuTime() / 1e9;
    }

    static String source(String[] c) {
        StringBuilder s = new StringBuilder(400).append('{');
        for (int i = 0; i < KEYWORDS.length; i++) {
            s.append('"').append(KEYWORDS[i]).append("\":\"").append(c[1 + i]).append("\",");
        }
        for (int i = 0; i < LONGS.length; i++) {
            s.append('"').append(LONGS[i]).append("\":").append(c[7 + i]).append(',');
        }
        return s.append("\"message\":\"").append(c[11]).append("\"}").toString();
    }

    static Document document(long seq, String[] c) {
        Document d = new Document();
        BytesRef id = new BytesRef(c[0]);
        d.add(new StoredField("_id", id));
        d.add(new StringField("_id", id, Field.Store.NO));
        d.add(new StoredField("_source", new BytesRef(source(c).getBytes(StandardCharsets.UTF_8))));
        for (int i = 0; i < KEYWORDS.length; i++) {
            BytesRef v = new BytesRef(c[1 + i]);
            d.add(new StringField(KEYWORDS[i], v, Field.Store.NO));
            d.add(new SortedSetDocValuesField(KEYWORDS[i], v));
        }
        for (int i = 0; i < LONGS.length; i++) {
            long v = Long.parseLong(c[7 + i]);
            d.add(new LongPoint(LONGS[i], v));
            d.add(new SortedNumericDocValuesField(LONGS[i], v));
        }
        d.add(new LongPoint("_seq_no", seq));
        d.add(new NumericDocValuesField("_seq_no", seq));
        d.add(new NumericDocValuesField("_version", 1));
        d.add(new NumericDocValuesField("_primary_term", 1));
        d.add(new TextField("message", c[11], Field.Store.NO));
        return d;
    }

    static IndexWriterConfig config(IndexWriterConfig.OpenMode mode) {
        IndexWriterConfig cfg = new IndexWriterConfig(new WhitespaceAnalyzer());
        cfg.setRAMBufferSizeMB(Double.parseDouble(System.getenv().getOrDefault("RAM_MB", "256")));
        cfg.setOpenMode(mode);
        cfg.setUseCompoundFile(false);
        return cfg;
    }

    static long dirBytes(Path p) throws Exception {
        try (var s = Files.list(p)) {
            return s.mapToLong(f -> f.toFile().length()).sum();
        }
    }

    static void index(Path corpus, Path path, int threads, int refreshMs) throws Exception {
        List<String[]> docs = new ArrayList<>();
        try (BufferedReader r = Files.newBufferedReader(corpus)) {
            String l;
            while ((l = r.readLine()) != null) {
                docs.add(l.split("\t"));
            }
        }
        IndexWriterConfig cfg = config(IndexWriterConfig.OpenMode.CREATE);
        AtomicBoolean done = new AtomicBoolean();
        List<Long> refreshes = new ArrayList<>();
        try (Directory dir = MMapDirectory.open(path); IndexWriter w = new IndexWriter(dir, cfg)) {
            SearcherManager sm = new SearcherManager(w, null);
            double cpu0 = cpuSeconds();
            long start = System.nanoTime();
            Thread refresher = new Thread(() -> {
                try {
                    while (!done.get()) {
                        Thread.sleep(refreshMs);
                        long t = System.nanoTime();
                        sm.maybeRefreshBlocking();
                        synchronized (refreshes) {
                            refreshes.add(System.nanoTime() - t);
                        }
                    }
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
            refresher.start();
            AtomicInteger next = new AtomicInteger();
            Thread[] adders = new Thread[threads];
            for (int t = 0; t < threads; t++) {
                adders[t] = new Thread(() -> {
                    try {
                        int i;
                        while ((i = next.getAndIncrement()) < docs.size()) {
                            w.addDocument(document(i, docs.get(i)));
                        }
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });
                adders[t].start();
            }
            for (Thread a : adders) {
                a.join();
            }
            double added = (System.nanoTime() - start) / 1e9;
            done.set(true);
            refresher.join();
            w.commit();
            ((org.apache.lucene.index.ConcurrentMergeScheduler) w.getConfig().getMergeScheduler()).sync();
            double all = (System.nanoTime() - start) / 1e9;
            double cpu = cpuSeconds() - cpu0;
            w.commit();
            sm.close();
            double mean = refreshes.stream().mapToLong(Long::longValue).average().orElse(0) / 1e6;
            System.out.printf("java,%d,%.1f,%.0f,%.1f,%.1f,%d,%.1f,%.1f%n", docs.size(), added,
                    docs.size() / added, all, cpu, refreshes.size(), mean, dirBytes(path) / 1e6);
        }
    }

    static void forceMerge(Path path, int segments) throws Exception {
        try (Directory dir = MMapDirectory.open(path);
                IndexWriter w = new IndexWriter(dir, config(IndexWriterConfig.OpenMode.APPEND))) {
            double cpu0 = cpuSeconds();
            long t = System.nanoTime();
            w.forceMerge(segments);
            w.commit();
            double took = (System.nanoTime() - t) / 1e9;
            System.out.printf("java,forcemerge,%.1f,%.1f,%.1f%n", took, cpuSeconds() - cpu0, dirBytes(path) / 1e6);
        }
    }

    public static void main(String[] a) throws Exception {
        switch (a[0]) {
            case "index" -> index(Path.of(a[1]), Path.of(a[2]), Integer.parseInt(a[3]), Integer.parseInt(a[4]));
            case "forcemerge" -> forceMerge(Path.of(a[1]), Integer.parseInt(a[2]));
            default -> throw new IllegalArgumentException(a[0]);
        }
    }
}
