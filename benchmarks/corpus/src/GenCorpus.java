import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

/**
 * Generates the benchmark corpus: a real Lucene 10.5.0 index that both engines
 * read byte-identically.
 *
 * <p>The text is synthetic rather than a Wikipedia extract, deliberately. What
 * this benchmark measures is decode and scoring cost, and the property that
 * drives it is the term-frequency distribution, not English semantics. A seeded
 * Zipfian generator is fully deterministic, needs no network, and can be
 * regenerated anywhere from a checked-in file -- which is what makes a
 * published benchmark number reproducible. The limitation is recorded in the
 * verdict: synthetic text has no phrase-level co-occurrence structure, so
 * phrase-query selectivity is less realistic than real prose.
 *
 * <p>Fields, chosen to cover the query shapes in the query set:
 * <ul>
 *   <li>{@code body} -- analysed text, positions + offsets (phrase, term)
 *   <li>{@code title} -- analysed text, shorter (boolean across fields)
 *   <li>{@code keyword} -- {@link StringField}, one token (term, prefix)
 *   <li>{@code num} -- {@link LongPoint} + {@link NumericDocValuesField}
 *       (points range, doc-values range, sort-by-field)
 *   <li>{@code cat} -- {@link SortedSetDocValuesField} (facet counting)
 * </ul>
 *
 * <p>Usage: {@code GenCorpus <outDir> <numDocs> <seed> [--force-merge]}
 */
public final class GenCorpus {

    /** Zipf-distributed vocabulary: rank r has probability proportional to 1/r^s. */
    private static final double ZIPF_S = 1.07;   // close to natural-language exponent
    private static final int VOCAB = 200_000;
    private static final int CATEGORIES = 64;

    public static void main(String[] args) throws IOException {
        if (args.length < 3) {
            System.err.println("usage: GenCorpus <outDir> <numDocs> <seed> [--force-merge]");
            System.exit(2);
        }
        Path out = Paths.get(args[0]);
        int numDocs = Integer.parseInt(args[1]);
        long seed = Long.parseLong(args[2]);
        boolean forceMerge = args.length > 3 && args[3].equals("--force-merge");

        if (Files.exists(out)) deleteRecursively(out);
        Files.createDirectories(out);

        String[] vocab = buildVocab();
        double[] cdf = zipfCdf(VOCAB);

        IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
        cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        // A fixed RAM buffer makes the segment count a deterministic function of
        // the document count, so the "many small segments" variant is reproducible.
        cfg.setRAMBufferSizeMB(64.0);
        cfg.setUseCompoundFile(false);

        FieldType bodyType = new FieldType(TextField.TYPE_NOT_STORED);
        bodyType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
        bodyType.freeze();

        long t0 = System.currentTimeMillis();
        Random rnd = new Random(seed);
        try (FSDirectory dir = FSDirectory.open(out);
             IndexWriter w = new IndexWriter(dir, cfg)) {
            for (int i = 0; i < numDocs; i++) {
                Document d = new Document();
                d.add(new Field("body", sentence(rnd, vocab, cdf, 40 + rnd.nextInt(120)), bodyType));
                d.add(new TextField("title", sentence(rnd, vocab, cdf, 3 + rnd.nextInt(8)), Field.Store.NO));
                d.add(new StringField("keyword", vocab[zipfSample(rnd, cdf)], Field.Store.NO));
                long n = rnd.nextInt(1_000_000);
                d.add(new LongPoint("num", n));
                d.add(new NumericDocValuesField("num", n));
                d.add(new SortedSetDocValuesField("cat",
                        new org.apache.lucene.util.BytesRef("cat" + rnd.nextInt(CATEGORIES))));
                w.addDocument(d);
                if ((i + 1) % 500_000 == 0) {
                    System.err.printf("  %,d / %,d docs%n", i + 1, numDocs);
                }
            }
            if (forceMerge) {
                System.err.println("  force-merging to 1 segment...");
                w.forceMerge(1);
            }
            w.commit();
        }
        long elapsed = System.currentTimeMillis() - t0;

        // Manifest: what a result was measured against, so a number is reproducible.
        int segments;
        int liveDocs;
        try (FSDirectory dir = FSDirectory.open(out);
             DirectoryReader r = DirectoryReader.open(dir)) {
            segments = r.leaves().size();
            liveDocs = r.numDocs();
        }
        long bytes = 0;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(out)) {
            for (Path p : ds) bytes += Files.size(p);
        }
        Properties m = new Properties();
        m.setProperty("docs", Integer.toString(liveDocs));
        m.setProperty("segments", Integer.toString(segments));
        m.setProperty("bytes", Long.toString(bytes));
        m.setProperty("seed", Long.toString(seed));
        m.setProperty("vocab", Integer.toString(VOCAB));
        m.setProperty("zipfS", Double.toString(ZIPF_S));
        m.setProperty("forceMerge", Boolean.toString(forceMerge));
        m.setProperty("luceneVersion", org.apache.lucene.util.Version.LATEST.toString());
        m.setProperty("indexMillis", Long.toString(elapsed));
        try (var os = Files.newOutputStream(out.resolve("corpus.manifest.properties"))) {
            m.store(os, "generated by GenCorpus");
        }
        System.err.printf("done: %,d docs, %d segments, %,d bytes, %.1fs%n",
                liveDocs, segments, bytes, elapsed / 1000.0);
    }

    private static String[] buildVocab() {
        String[] v = new String[VOCAB];
        for (int i = 0; i < VOCAB; i++) v[i] = "t" + Integer.toString(i, 36);
        return v;
    }

    /** Cumulative Zipf distribution over `n` ranks. */
    private static double[] zipfCdf(int n) {
        double[] cdf = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) { sum += 1.0 / Math.pow(i + 1, ZIPF_S); cdf[i] = sum; }
        for (int i = 0; i < n; i++) cdf[i] /= sum;
        return cdf;
    }

    private static int zipfSample(Random rnd, double[] cdf) {
        double u = rnd.nextDouble();
        int lo = 0, hi = cdf.length - 1;
        while (lo < hi) { int mid = (lo + hi) >>> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
        return lo;
    }

    private static String sentence(Random rnd, String[] vocab, double[] cdf, int words) {
        StringBuilder sb = new StringBuilder(words * 6);
        for (int i = 0; i < words; i++) {
            if (i > 0) sb.append(' ');
            sb.append(vocab[zipfSample(rnd, cdf)]);
        }
        return sb.toString();
    }

    private static void deleteRecursively(Path p) throws IOException {
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(p)) {
            for (Path c : ds) { if (Files.isDirectory(c)) deleteRecursively(c); else Files.delete(c); }
        }
        Files.delete(p);
    }
}
