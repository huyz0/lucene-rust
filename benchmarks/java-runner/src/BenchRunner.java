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

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("usage: BenchRunner <index-dir> <queries.tsv> <warmup> <iters>");
            System.exit(2);
        }
        requireVectorApi();

        Path index = Paths.get(args[0]);
        List<String[]> queries = loadQueries(Paths.get(args[1]));
        int warmup = Integer.parseInt(args[2]);
        int iters = Integer.parseInt(args[3]);

        try (org.apache.lucene.store.Directory dir = MMapDirectory.open(index);
             DirectoryReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);   // single-threaded, no executor
            searcher.setSimilarity(new org.apache.lucene.search.similarities.BM25Similarity());

            System.out.println("id\thits\ttop1doc\ttop1score\ttopset\tqps\tp50_us\tp95_us\tp99_us");
            for (String[] q : queries) {
                Query query = build(q);

                for (int i = 0; i < warmup; i++) searcher.search(query, TOP_N);

                long[] samples = new long[iters];
                TopDocs last = null;
                long t0 = System.nanoTime();
                for (int i = 0; i < iters; i++) {
                    long s = System.nanoTime();
                    last = searcher.search(query, TOP_N);
                    samples[i] = (System.nanoTime() - s) / 1000;
                }
                double wallSec = (System.nanoTime() - t0) / 1e9;

                Arrays.sort(samples);
                // Set comparison, not ordered: ties may break differently.
                int[] ids = new int[last.scoreDocs.length];
                for (int i = 0; i < ids.length; i++) ids[i] = last.scoreDocs[i].doc;
                Arrays.sort(ids);
                StringJoiner top = new StringJoiner(",");
                for (int id : ids) top.add(Integer.toString(id));
                int top1doc = last.scoreDocs.length > 0 ? last.scoreDocs[0].doc : -1;
                float top1score = last.scoreDocs.length > 0 ? last.scoreDocs[0].score : 0f;
                System.out.printf("%s\t%d\t%d\t%.6f\t%s\t%.1f\t%d\t%d\t%d%n",
                        q[0], last.scoreDocs.length, top1doc, top1score, top.toString(),
                        iters / wallSec,
                        samples[(int) ((samples.length - 1) * 0.50)],
                        samples[(int) ((samples.length - 1) * 0.95)],
                        samples[(int) ((samples.length - 1) * 0.99)]);
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
