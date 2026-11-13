import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Full-text search over one shard shared by many tenants, on stock Lucene: the
 * index builder and the Java side of {@code scripts/bench-fts-partition.sh}.
 *
 * <pre>
 * build &lt;dir&gt; &lt;docs&gt; &lt;tenants&gt; plain|composite|blocks   the index and, beside it, queries.tsv
 * query &lt;dir&gt; &lt;queries.tsv&gt; &lt;passes&gt;                     latency per query class, as CSV
 * hits  &lt;dir&gt; &lt;queries.tsv&gt;                              each query's top hits: id and score
 * </pre>
 *
 * {@code RANGE=iodv} sends each time range as OpenSearch builds one on a field with doc values, an
 * {@code IndexOrDocValuesQuery}.
 *
 * Every document has a tenant (a keyword: indexed term and SORTED_SET doc values), a time
 * {@code ts} (a long: points and SORTED_NUMERIC doc values) and a text {@code body}.
 * {@code composite} sorts the index by tenant then time, newest first; {@code blocks} does too, each
 * document the parent of up to two child documents (OpenSearch nested objects: their own
 * text field, none of the parent's), the parent marked as OpenSearch marks a root document.
 * Tenants' sizes are Zipf over the tenant count.
 */
public final class PartitionBench {

    static final int TOP_N = 10;
    static final int THRESHOLD = 1000;

    static long rng = 0x9e3779b97f4a7c15L;

    static double next() {
        rng ^= rng << 13;
        rng ^= rng >>> 7;
        rng ^= rng << 17;
        return (rng >>> 11) / (double) (1L << 53);
    }

    /** A Zipf-ish rank of a {@code n}-word vocabulary. */
    static long zipf(int n) {
        return (long) Math.pow(n, next());
    }

    static int[] tenantSizes(int tenants, int docs) {
        double[] w = new double[tenants];
        double sum = 0;
        for (int r = 0; r < tenants; r++) {
            w[r] = 1.0 / Math.pow(r + 1, 1.1);
            sum += w[r];
        }
        int[] out = new int[tenants];
        for (int r = 0; r < tenants; r++) {
            out[r] = Math.max(1, (int) Math.round(w[r] / sum * docs));
        }
        return out;
    }

    static String body(int words) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < words; i++) {
            if (i > 0) b.append(' ');
            if (next() < 0.02) {
                // A planted phrase.
                long k = zipf(400);
                b.append('p').append(k).append(" q").append(k);
            } else {
                b.append('w').append(zipf(50_000));
            }
        }
        return b.toString();
    }

    static final long SPAN = 1_000_000_000L;

    static void build(Path path, int docs, int tenants, String mode) throws IOException {
        IndexWriterConfig cfg = new IndexWriterConfig(new org.apache.lucene.analysis.core.WhitespaceAnalyzer());
        cfg.setRAMBufferSizeMB(256);
        cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        boolean blocks = mode.equals("blocks");
        if (!mode.equals("plain")) {
            cfg.setIndexSort(new Sort(new SortedSetSortField("tenant", false), new SortedNumericSortField("ts", SortField.Type.LONG, true)));
        }
        if (blocks) {
            cfg.setParentField("_parent");
        }
        int[] sizes = tenantSizes(tenants, docs);
        // Documents arrive interleaved across tenants, as a shared index gets them.
        int total = Arrays.stream(sizes).sum();
        int[] owner = new int[total];
        int at = 0;
        for (int t = 0; t < tenants; t++) {
            for (int i = 0; i < sizes[t]; i++) owner[at++] = t;
        }
        rng = 0x1234_5678_9abc_def1L;
        for (int i = total - 1; i > 0; i--) {
            int j = (int) (next() * (i + 1));
            int x = owner[i];
            owner[i] = owner[j];
            owner[j] = x;
        }
        rng = 0x9e3779b97f4a7c15L;
        long start = System.nanoTime();
        try (Directory dir = MMapDirectory.open(path); IndexWriter w = new IndexWriter(dir, cfg)) {
            for (int i = 0; i < total; i++) {
                Document d = new Document();
                d.add(new StoredField("id", i));
                d.add(new StringField("tenant", "t" + owner[i], Field.Store.NO));
                d.add(new SortedSetDocValuesField("tenant", new BytesRef("t" + owner[i])));
                long ts = (long) (next() * SPAN);
                d.add(new LongPoint("ts", ts));
                d.add(new SortedNumericDocValuesField("ts", ts));
                d.add(new TextField("body", body(40), Field.Store.NO));
                if (blocks) {
                    d.add(new NumericDocValuesField("_primary_term", 1));
                    List<Document> block = new ArrayList<>();
                    int children = (int) (next() * 3);
                    for (int c = 0; c < children; c++) {
                        Document child = new Document();
                        child.add(new TextField("comments.body", body(10), Field.Store.NO));
                        block.add(child);
                    }
                    block.add(d);
                    w.addDocuments(block);
                } else {
                    w.addDocument(d);
                }
            }
            w.forceMerge(4);
            w.commit();
        }
        System.err.printf("built %s: %d docs, %d tenants in %.1f s%n", mode, total, tenants, (System.nanoTime() - start) / 1e9);
        writeQueries(path.resolve("queries.tsv"), sizes);
    }

    /**
     * Queries of three tenant classes: large (ranks 1-5), mid (20-60), small (300+). Each is a
     * tenant filter and a body term, a conjunction or disjunction of two terms, or a phrase; a
     * third also bound the time to a tenth of the span; then the tenant's newest documents,
     * alone or with a term (sorted by time).
     */
    static void writeQueries(Path out, int[] sizes) throws IOException {
        rng = 0x5151_2323_7777_abcdL;
        List<String> lines = new ArrayList<>();
        String[] kinds = { "term", "and", "or", "phrase", "newest", "newest_term" };
        int id = 0;
        for (String cls : new String[] { "large", "mid", "small" }) {
            for (int i = 0; i < 120; i++) {
                int t = switch (cls) {
                    case "large" -> (int) (next() * 5);
                    case "mid" -> 20 + (int) (next() * 40);
                    default -> 300 + (int) (next() * Math.max(1, sizes.length - 300));
                };
                String kind = kinds[i % kinds.length];
                String words = switch (kind) {
                    case "phrase" -> {
                        long k = 1 + (long) (next() * 30);
                        yield "p" + k + " q" + k;
                    }
                    case "and", "or" -> "w" + (20 + (long) (next() * 400)) + " w" + (2 + (long) (next() * 40));
                    case "newest" -> "";
                    default -> "w" + (20 + (long) (next() * 2000));
                };
                long lo = -1, hi = -1;
                if (!kind.startsWith("newest") && i % 3 == 0) {
                    lo = (long) (next() * SPAN * 0.9);
                    hi = lo + SPAN / 10;
                }
                lines.add(String.join("\t", Integer.toString(id++), cls, kind, "t" + t, Long.toString(lo), Long.toString(hi), words));
            }
        }
        Files.write(out, lines, StandardCharsets.UTF_8);
    }

    record Q(int id, String cls, String kind, Query query, Sort sort) {}

    static List<Q> loadQueries(Path p, boolean blocks) throws IOException {
        List<Q> out = new ArrayList<>();
        for (String line : Files.readAllLines(p, StandardCharsets.UTF_8)) {
            String[] f = line.split("\t", -1);
            BooleanQuery.Builder b = new BooleanQuery.Builder();
            String kind = f[2];
            String[] words = f[6].isEmpty() ? new String[0] : f[6].split(" ");
            switch (kind) {
                case "term", "newest_term" -> b.add(new TermQuery(new Term("body", words[0])), BooleanClause.Occur.MUST);
                case "and" -> {
                    b.add(new TermQuery(new Term("body", words[0])), BooleanClause.Occur.MUST);
                    b.add(new TermQuery(new Term("body", words[1])), BooleanClause.Occur.MUST);
                }
                case "or" -> {
                    BooleanQuery.Builder or = new BooleanQuery.Builder();
                    or.add(new TermQuery(new Term("body", words[0])), BooleanClause.Occur.SHOULD);
                    or.add(new TermQuery(new Term("body", words[1])), BooleanClause.Occur.SHOULD);
                    b.add(or.build(), BooleanClause.Occur.MUST);
                }
                case "phrase" -> b.add(new PhraseQuery("body", words), BooleanClause.Occur.MUST);
                case "newest" -> {}
                default -> throw new IllegalArgumentException(kind);
            }
            b.add(new TermQuery(new Term("tenant", f[3])), BooleanClause.Occur.FILTER);
            long lo = Long.parseLong(f[4]), hi = Long.parseLong(f[5]);
            if (lo >= 0) {
                Query range = LongPoint.newRangeQuery("ts", lo, hi);
                if ("iodv".equals(System.getenv("RANGE"))) {
                    // As OpenSearch sends a range on a field with doc values.
                    range = new IndexOrDocValuesQuery(range, SortedNumericDocValuesField.newSlowRangeQuery("ts", lo, hi));
                }
                b.add(range, BooleanClause.Occur.FILTER);
            }
            if (blocks) {
                // OpenSearch's non-nested filter on an index with nested fields.
                b.add(new FieldExistsQuery("_primary_term"), BooleanClause.Occur.FILTER);
            }
            Sort sort = kind.startsWith("newest") ? new Sort(new SortedNumericSortField("ts", SortField.Type.LONG, true)) : null;
            out.add(new Q(Integer.parseInt(f[0]), f[1], kind, b.build(), sort));
        }
        return out;
    }

    static TopDocs run(IndexSearcher s, Q q) throws IOException {
        if (q.sort() != null) {
            return s.search(q.query(), new TopFieldCollectorManager(q.sort(), TOP_N, null, THRESHOLD));
        }
        return s.search(q.query(), new TopScoreDocCollectorManager(TOP_N, null, THRESHOLD));
    }

    static long readBytes() throws IOException {
        for (String l : Files.readAllLines(Paths.get("/proc/self/io"))) {
            if (l.startsWith("read_bytes: ")) return Long.parseLong(l.substring(12).trim());
        }
        return 0;
    }

    static void query(Path path, Path queries, int passes) throws IOException {
        try (Directory dir = MMapDirectory.open(path); DirectoryReader r = DirectoryReader.open(dir)) {
            boolean blocks = r.leaves().get(0).reader().getMetaData().hasBlocks();
            List<Q> qs = loadQueries(queries, blocks);
            IndexSearcher s = new IndexSearcher(r);
            s.setQueryCache(null);
            // The cold pass first: what a query set reads from disk on an empty page cache.
            long before = readBytes();
            long t0 = System.nanoTime();
            for (Q q : qs) run(s, q);
            System.out.printf("cold,all,all,%.0f,%d%n", (System.nanoTime() - t0) / 1e3 / qs.size(), readBytes() - before);
            for (int w = 0; w < 20; w++) {
                for (Q q : qs) run(s, q);
            }
            Map<String, List<Long>> lat = new TreeMap<>();
            for (int p = 0; p < passes; p++) {
                for (Q q : qs) {
                    long t = System.nanoTime();
                    run(s, q);
                    long us = (System.nanoTime() - t) / 1000;
                    lat.computeIfAbsent(q.cls() + "," + q.kind(), k -> new ArrayList<>()).add(us);
                }
            }
            for (Map.Entry<String, List<Long>> e : lat.entrySet()) {
                List<Long> v = e.getValue();
                Collections.sort(v);
                double mean = v.stream().mapToLong(Long::longValue).average().orElse(0);
                System.out.printf("hot,%s,%.0f,%d%n", e.getKey(), mean, v.get(v.size() / 2));
            }
        }
    }

    static void hits(Path path, Path queries) throws IOException {
        try (Directory dir = MMapDirectory.open(path); DirectoryReader r = DirectoryReader.open(dir)) {
            boolean blocks = r.leaves().get(0).reader().getMetaData().hasBlocks();
            IndexSearcher s = new IndexSearcher(r);
            s.setQueryCache(null);
            StoredFields stored = r.storedFields();
            for (Q q : loadQueries(queries, blocks)) {
                TopDocs top = run(s, q);
                StringBuilder b = new StringBuilder();
                b.append(q.id()).append('\t').append(top.totalHits.value()).append(top.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "" : "+");
                for (ScoreDoc d : top.scoreDocs) {
                    int id = stored.document(d.doc).getField("id").numericValue().intValue();
                    String v = q.sort() != null ? Long.toString((Long) ((FieldDoc) d).fields[0]) : Integer.toHexString(Float.floatToIntBits(d.score));
                    b.append('\t').append(id).append(':').append(v);
                }
                System.out.println(b);
            }
        }
    }

    public static void main(String[] a) throws IOException {
        switch (a[0]) {
            case "build" -> build(Paths.get(a[1]), Integer.parseInt(a[2]), Integer.parseInt(a[3]), a[4]);
            case "query" -> query(Paths.get(a[1]), Paths.get(a[2]), Integer.parseInt(a[3]));
            case "hits" -> hits(Paths.get(a[1]), Paths.get(a[2]));
            default -> throw new IllegalArgumentException(a[0]);
        }
    }
}
