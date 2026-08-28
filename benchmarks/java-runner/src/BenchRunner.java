import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.BytesRef;

import java.nio.file.*;
import java.util.*;

/**
 * Java Lucene side of the M1 performance gate. Emits the identical TSV schema
 * as {@code benchmarks/rust-runner} so the two can be joined on query id.
 *
 * <p>Two methodology points that decide whether the numbers mean anything:
 * <ul>
 *   <li>The JVM must reach steady state before timing starts, or Java loses by
 *       5-10x on interpreter overhead alone. Warmup iterations are a required
 *       argument, and the count is echoed into the output header.
 *   <li>This must run with {@code --add-modules jdk.incubator.vector}. Lucene
 *       uses the Panama Vector API for its hot loops; without it Java is
 *       handicapped and the comparison flatters the Rust side. The runner
 *       refuses to start if the module is absent rather than silently
 *       producing a misleading number.
 * </ul>
 */
public final class BenchRunner {

    private static final int TOP_N = 50;

    /**
     * Documents the scorer actually produced for the query being measured.
     *
     * Answers a question timing cannot: when this port's per-document costs are
     * *lower* than Lucene's but its queries are slower, the two engines must be
     * visiting different numbers of documents. A count is immune to measurement
     * noise, and it separates "we do more work" from "we are slower" -- very
     * different defects with very different fixes.
     */
    private static long scoredDocs;

    /**
     * Counts {@code collect(doc)} per leaf while changing nothing else.
     *
     * Extends {@link FilterLeafCollector} rather than implementing
     * {@link LeafCollector} directly, specifically so {@code setScorer} is
     * forwarded: that is what carries {@code setMinCompetitiveScore} down to
     * the scorer, and a wrapper that dropped it would silently disable block-max
     * pruning and count a completely different (much larger) number.
     */
    private static final class CountingCollector extends FilterCollector {
        CountingCollector(Collector in) { super(in); }

        @Override
        public LeafCollector getLeafCollector(LeafReaderContext context) throws java.io.IOException {
            return new FilterLeafCollector(super.getLeafCollector(context)) {
                @Override
                public void collect(int doc) throws java.io.IOException {
                    scoredDocs++;
                    in.collect(doc);
                }
            };
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("usage: BenchRunner <index-dir> <queries.tsv> <warmup_ms> <measure_ms>");
            System.exit(2);
        }
        requireVectorApi();

        Path index = Paths.get(args[0]);
        List<String[]> queries = loadQueries(Paths.get(args[1]));
        // Time-boxed, not count-based -- see the Rust runner for why.
        long warmupMs = Long.parseLong(args[2]);
        long measureMs = Long.parseLong(args[3]);

        try (org.apache.lucene.store.Directory dir = MMapDirectory.open(index);
             DirectoryReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);   // single-threaded, no executor
            searcher.setSimilarity(new org.apache.lucene.search.similarities.BM25Similarity());

            System.out.println("id\thits\ttop1doc\ttop1score\ttopset\tqps\tp50_us\tp95_us\tp99_us\tscored");
            for (String[] q : queries) {
                Query query = build(q);

                long w = System.nanoTime();
                do {
                    if (q[1].equals("dv_sort")) {
                        searcher.search(query, TOP_N, new Sort(new SortField(q[2], SortField.Type.LONG)));
                    } else {
                        searcher.search(query, TOP_N);
                    }
                }
                while ((System.nanoTime() - w) / 1_000_000 < warmupMs);

                List<Long> sampleList = new ArrayList<>();
                TopDocs last = null;
                long scoredPerIter = 0;
                long t0 = System.nanoTime();
                do {
                    long s = System.nanoTime();
                    last = q[1].equals("dv_sort")
                            ? searcher.search(
                                    query,
                                    TOP_N,
                                    new Sort(new SortField(q[2], SortField.Type.LONG)))
                            : searcher.search(query, TOP_N);
                    sampleList.add((System.nanoTime() - s) / 1000);
                } while ((System.nanoTime() - t0) / 1_000_000 < measureMs || sampleList.size() < 5);
                // One extra, untimed, instrumented run: the counting wrapper is
                // kept out of the timed loop so it cannot bias the timings.
                {
                    // 1000 is IndexSearcher.TOTAL_HITS_THRESHOLD, what
                    // search(query, n) uses. Integer.MAX_VALUE here would ask
                    // for exact hit counts, which switches OFF block-max
                    // pruning -- the counted run would then execute a
                    // different query from the timed one, and Lucene's count
                    // came out at maxDoc for every term query, which is what
                    // exposed it.
                    TopScoreDocCollectorManager m =
                            new TopScoreDocCollectorManager(TOP_N, 1000);
                    TopScoreDocCollector c = m.newCollector();
                    scoredDocs = 0;
                    searcher.search(query, new CountingCollector(c));
                    scoredPerIter = scoredDocs;
                }
                double wallSec = (System.nanoTime() - t0) / 1e9;
                int iters = sampleList.size();

                long[] samples = new long[sampleList.size()];
                for (int i = 0; i < samples.length; i++) samples[i] = sampleList.get(i);
                Arrays.sort(samples);
                // Set comparison, not ordered: ties may break differently.
                int[] ids = new int[last.scoreDocs.length];
                for (int i = 0; i < ids.length; i++) ids[i] = last.scoreDocs[i].doc;
                Arrays.sort(ids);
                StringJoiner top = new StringJoiner(",");
                for (int id : ids) top.add(Integer.toString(id));
                int top1doc = last.scoreDocs.length > 0 ? last.scoreDocs[0].doc : -1;
                float top1score = last.scoreDocs.length > 0 ? last.scoreDocs[0].score : 0f;
                System.out.printf("%s\t%d\t%d\t%.6f\t%s\t%.1f\t%d\t%d\t%d\t%d%n",
                        q[0], last.scoreDocs.length, top1doc, top1score, top.toString(),
                        iters / wallSec,
                        samples[(int) ((samples.length - 1) * 0.50)],
                        samples[(int) ((samples.length - 1) * 0.95)],
                        samples[(int) ((samples.length - 1) * 0.99)],
                        scoredPerIter);
            }
        }
    }

    private static Query build(String[] f) {
        String kind = f[1], field = f[2];
        switch (kind) {
            case "term":
                return new TermQuery(new Term(field, f[3]));
            case "and":
            case "or":
            case "or_maxscore": {
                BooleanClause.Occur occur =
                        kind.equals("and") ? BooleanClause.Occur.MUST : BooleanClause.Occur.SHOULD;
                BooleanQuery.Builder b = new BooleanQuery.Builder();
                for (int i = 3; i < f.length; i++) b.add(new TermQuery(new Term(field, f[i])), occur);
                return b.build();
            }
            // Query kinds the M1.6 sweep never measured.
            case "dv_sort":
                // Same shape as the Rust side: a numeric doc-values range,
                // ranked by that field ascending.
                return org.apache.lucene.document.NumericDocValuesField.newSlowRangeQuery(
                        field, Long.parseLong(f[3]), Long.parseLong(f[4]));
            case "fuzzy":
                // maxEdits 2, prefixLength 0, transpositions on -- the Rust
                // side's settings, and Lucene's own defaults except maxEdits.
                return new org.apache.lucene.search.FuzzyQuery(
                        new Term(field, f[3]), 2, 0, 1024, true);
            case "regexp":
                return new org.apache.lucene.search.RegexpQuery(new Term(field, f[3]));
            case "prefix":
                return new org.apache.lucene.search.PrefixQuery(new Term(field, f[3]));
            case "wildcard":
                return new org.apache.lucene.search.WildcardQuery(new Term(field, f[3]));
            case "phrase": {
                PhraseQuery.Builder b = new PhraseQuery.Builder();
                for (int i = 3; i < f.length; i++) b.add(new Term(field, f[i]));
                return b.build();
            }
            default:
                throw new IllegalArgumentException("unknown query kind: " + kind);
        }
    }

    private static List<String[]> loadQueries(Path p) throws Exception {
        List<String[]> out = new ArrayList<>();
        for (String line : Files.readAllLines(p)) {
            if (line.isBlank() || line.startsWith("#")) continue;
            out.add(line.split("\t"));
        }
        return out;
    }

    /** Fail loudly rather than quietly benchmarking a handicapped Lucene. */
    private static void requireVectorApi() {
        if (ModuleLayer.boot().findModule("jdk.incubator.vector").isEmpty()) {
            System.err.println("BenchRunner: jdk.incubator.vector is not present. Lucene's hot "
                    + "loops use the Panama Vector API; without it this measures a handicapped "
                    + "Lucene and overstates the Rust side. Re-run with "
                    + "--add-modules jdk.incubator.vector.");
            System.exit(3);
        }
    }
}
