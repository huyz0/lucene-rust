import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LongValuesSource;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.grouping.DistinctValuesCollector;
import org.apache.lucene.search.grouping.DistinctValuesCollectorManager;
import org.apache.lucene.search.grouping.FirstPassGroupingCollectorManager;
import org.apache.lucene.search.grouping.GroupDocs;
import org.apache.lucene.search.grouping.GroupFacetCollector;
import org.apache.lucene.search.grouping.GroupSelector;
import org.apache.lucene.search.grouping.GroupingSearch;
import org.apache.lucene.search.grouping.LongRangeFactory;
import org.apache.lucene.search.grouping.LongRangeGroupSelector;
import org.apache.lucene.search.grouping.SearchGroup;
import org.apache.lucene.search.grouping.TermGroupFacetCollector;
import org.apache.lucene.search.grouping.TermGroupSelector;
import org.apache.lucene.search.grouping.TopGroups;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Java side of the grouping benchmark pair (M10 T10.4); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_grouping.rs}, with the same case names. {@code build <dir>}
 * writes the corpus once -- 200 000 documents in four segments, in blocks of one to eight closed
 * by an {@code end:x} document, each with a text {@code body}, a {@code SORTED} group {@code g} of
 * 2 000 values (one in ten missing), a {@code NUMERIC} {@code n}, a {@code SORTED} {@code s}, a
 * {@code SORTED} {@code v} of 50 values and the facet fields {@code fsv}/{@code fmv} -- and the
 * word list both engines search ({@code group-words.tsv}). Each case runs, per word, one grouping
 * search over {@code body:word}: {@code GroupingSearch} by term (by relevance; by a field with a
 * within-group sort; with all groups and group heads; cached), by long range and by blocks, the
 * first-pass and distinct-values managers, and the grouped facets. Each prints a {@code #check}
 * digest of its result the report compares first.
 */
public final class GroupingMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int DOCS = 200_000;
  static final int GROUPS = 2_000;
  static final String[] WORDS = {
    "red", "blue", "green", "fast", "slow", "big", "small", "shiny", "old", "new", "cheap", "rare",
    "warm", "cold", "soft", "hard"
  };

  interface Op {
    long run() throws IOException;
  }

  static void measure(String name, Op op) throws IOException {
    loop(op, warmupMs);
    long start = System.nanoTime();
    long units = loop(op, measureMs);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / units, units);
    System.out.flush();
  }

  static long loop(Op op, long budgetMs) throws IOException {
    long budgetNs = budgetMs * 1_000_000L;
    long units = 0;
    long start = System.nanoTime();
    do {
      units += op.run();
    } while (System.nanoTime() - start < budgetNs);
    if (sink == 0xDEADBEEFL) System.err.print("");
    return units;
  }

  static final class Fnv {
    long h = 0xcbf29ce484222325L;

    void add(long x) {
      h ^= x;
      h *= 0x100000001b3L;
    }

    void bytes(BytesRef b) {
      if (b == null) {
        add(-1);
        return;
      }
      for (int i = 0; i < b.length; i++) add(b.bytes[b.offset + i]);
    }
  }

  static String word(Random r) {
    int i = (int) (Math.pow(r.nextDouble(), 1.6) * WORDS.length);
    return WORDS[Math.min(i, WORDS.length - 1)];
  }

  static String body(Random r) {
    int n = 2 + r.nextInt(8);
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < n; i++) {
      if (i > 0) b.append(' ');
      b.append(word(r));
    }
    return b.toString();
  }

  static void build(Path dir) throws IOException {
    Random r = new Random(0x10_4_2026_1005L);
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setRAMBufferSizeMB(512);
    cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
    cfg.setMergePolicy(NoMergePolicy.INSTANCE);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      int i = 0;
      int nextCommit = DOCS / 4;
      while (i < DOCS) {
        int n = 1 + r.nextInt(8);
        List<Document> block = new ArrayList<>();
        for (int j = 0; j < n; j++, i++) {
          Document doc = new Document();
          doc.add(new TextField("body", body(r), Field.Store.NO));
          if (r.nextInt(10) != 0) {
            doc.add(new SortedDocValuesField("g", new BytesRef("g" + r.nextInt(GROUPS))));
          }
          doc.add(new NumericDocValuesField("n", r.nextInt(10_000)));
          doc.add(new SortedDocValuesField("s", new BytesRef("s" + r.nextInt(500))));
          doc.add(new SortedDocValuesField("v", new BytesRef("v" + r.nextInt(50))));
          doc.add(new SortedDocValuesField("fsv", new BytesRef("f" + r.nextInt(100))));
          int m = r.nextInt(4);
          for (int k = 0; k < m; k++) {
            doc.add(new SortedSetDocValuesField("fmv", new BytesRef("f" + r.nextInt(100))));
          }
          if (j == n - 1) doc.add(new StringField("end", "x", Field.Store.NO));
          block.add(doc);
        }
        w.addDocuments(block);
        if (i >= nextCommit) {
          w.commit();
          nextCommit += DOCS / 4;
        }
      }
      w.commit();
    }
    StringBuilder q = new StringBuilder();
    for (int k = 0; k < 16; k++) q.append(word(r)).append('\n');
    // Last: its presence is what marks the corpus built.
    Files.writeString(dir.resolve("group-words.tsv"), q.toString());
  }

  static void topGroups(Fnv f, TopGroups<?> tg) {
    f.add(tg.totalHitCount);
    f.add(tg.totalGroupedHitCount);
    f.add(tg.totalGroupCount == null ? -1 : tg.totalGroupCount);
    for (GroupDocs<?> g : tg.groups) {
      f.add(g.totalHits().value());
      for (ScoreDoc sd : g.scoreDocs()) {
        f.add(sd.doc);
        f.add(Float.floatToIntBits(sd.score));
      }
    }
  }

  interface Case {
    void run(String word, Fnv f) throws IOException;
  }

  static void cases(String name, List<String> words, Case c) throws IOException {
    Fnv f = new Fnv();
    for (String w : words) c.run(w, f);
    System.out.printf("#check\t%s\t%016x\t%d%n", name, f.h, words.size());
    measure(
        name,
        () -> {
          Fnv g = new Fnv();
          for (String w : words) c.run(w, g);
          sink += g.h;
          return words.size();
        });
  }

  static Query q(String word) {
    return new TermQuery(new Term("body", word));
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  static void run(Path dir) throws IOException {
    List<String> words = Files.readAllLines(dir.resolve("group-words.tsv"));
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      Sort byN = new Sort(new SortField("n", SortField.Type.LONG, true));
      Sort byS = new Sort(new SortField("s", SortField.Type.STRING));
      cases(
          "grp_term_rel",
          words,
          (w, f) -> {
            GroupingSearch g = new GroupingSearch("g");
            g.setGroupDocsLimit(3);
            topGroups(f, g.search(s, q(w), 0, 10));
          });
      cases(
          "grp_term_sorted",
          words,
          (w, f) -> {
            GroupingSearch g = new GroupingSearch("g");
            g.setGroupSort(byN);
            g.setSortWithinGroup(byS);
            g.setGroupDocsLimit(3);
            topGroups(f, g.search(s, q(w), 5, 10));
          });
      cases(
          "grp_term_all",
          words,
          (w, f) -> {
            GroupingSearch g = new GroupingSearch("g");
            g.setAllGroups(true);
            g.setAllGroupHeads(true);
            topGroups(f, g.search(s, q(w), 0, 10));
            f.add(g.getAllMatchingGroups().size());
          });
      cases(
          "grp_term_cached",
          words,
          (w, f) -> {
            GroupingSearch g = new GroupingSearch("g");
            g.setCaching(1_000_000, true);
            g.setGroupDocsLimit(3);
            topGroups(f, g.search(s, q(w), 0, 10));
          });
      cases(
          "grp_long_range",
          words,
          (w, f) -> {
            GroupingSearch g =
                new GroupingSearch(
                    new LongRangeGroupSelector(
                        LongValuesSource.fromLongField("n"), new LongRangeFactory(0, 250, 10_000)));
            g.setGroupDocsLimit(2);
            topGroups(f, g.search(s, q(w), 0, 10));
          });
      cases(
          "grp_blocks",
          words,
          (w, f) -> {
            GroupingSearch g = new GroupingSearch(new TermQuery(new Term("end", "x")));
            g.setGroupDocsLimit(2);
            topGroups(f, g.search(s, q(w), 0, 10));
          });
      cases(
          "grp_distinct",
          words,
          (w, f) -> {
            Collection<SearchGroup> first =
                (Collection<SearchGroup>)
                    s.search(
                        q(w),
                        new FirstPassGroupingCollectorManager(
                            () -> new TermGroupSelector("g"), Sort.RELEVANCE, 0, 20));
            f.add(first.size());
            if (first.isEmpty()) return;
            List<DistinctValuesCollector.GroupCount> counts =
                (List)
                    s.search(
                        q(w),
                        new DistinctValuesCollectorManager(
                            () -> new TermGroupSelector("g"),
                            first,
                            () -> new TermGroupSelector("v")));
            for (DistinctValuesCollector.GroupCount c : counts) {
              f.bytes((BytesRef) c.groupValue);
              f.add(c.uniqueValues.size());
            }
          });
      for (boolean mv : new boolean[] {false, true}) {
        cases(
            mv ? "grp_facet_mv" : "grp_facet_sv",
            words,
            (w, f) -> {
              TermGroupFacetCollector c =
                  TermGroupFacetCollector.createTermGroupFacetCollector(
                      "g", mv ? "fmv" : "fsv", mv, null, 128);
              s.search(q(w), c);
              GroupFacetCollector.GroupedFacetResult res = c.mergeSegmentResults(10, 0, true);
              f.add(res.getTotalCount());
              f.add(res.getTotalMissingCount());
              for (GroupFacetCollector.FacetEntry e : res.getFacetEntries(0, 10)) {
                f.bytes(e.value());
                f.add(e.count());
              }
            });
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      if (Files.exists(dir.resolve("group-words.tsv"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
