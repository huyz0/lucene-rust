import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.MultiCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TopFieldCollectorManager;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.TotalHitCountCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.stream.Stream;

/**
 * OpenSearch's {@code terminate_after} on a sequential search, recorded from Lucene for
 * crates/lucene-search/tests/terminate_after_fixtures.rs.
 *
 * <p>OpenSearch puts an {@code EarlyTerminatingCollector(EMPTY_COLLECTOR, n, true)} in a {@code
 * MultiCollector} beside the top-docs collector (QueryCollectorContext
 * .createEarlyTerminationCollectorContext): it counts each document collected and throws, ending
 * the whole search, at the {@code n + 1}th, or when asked for a later segment's leaf collector
 * once {@code n} are in. That collector is OpenSearch's, not Lucene's, so it is reproduced here
 * ({@link Early}); the top-docs collectors beside it are Lucene's own ({@code
 * TopScoreDocCollector}, {@code TopFieldCollector}, counting exactly), with OpenSearch's
 * MaxScoreCollector for {@code track_scores}.
 *
 * <p>Four segments of 3,000 documents, the first and third with deletions; the {@code r} points
 * are the document ids, so a range confines a query's matches to the first segments (and a later
 * segment with no match still ends the search). Per query, {@code n} runs over fixed values and
 * the query's match count and its neighbours. Each run records the hits ({@code doc:v1:...}, a
 * score as its float bits; {@code score-docs} is the unsorted {@code TopScoreDocCollector}, its
 * hits in the score sort's encoding), the total and its relation, the count let through and
 * whether the search ended early.
 */
public class GenTerminateAfter {
  static final int DOCS_PER_SEGMENT = 3_000;
  static final int SEGMENTS = 4;

  static final String[] QUERIES = {
    "(all)",
    "(t w0)",
    "(t w30)",
    "(b 0 (+ (t w1)) (- (t w2)))",
    "(b 0 (? (t w3)) (? (t w4)) (? (t w5)))",
    "(b 0 (# (t w0)) (# (r 0 5999)))",
    // Scored, matching only in the first two segments: the next segment ends the search.
    "(b 0 (+ (t w0)) (# (r 0 5999)))",
    "(b 0 (+ (t w0)) (? (p w1 w2)))",
    "(t nosuchterm)",
  };

  /** Sorts, in GenSortedSearch's spec; {@code score-docs} is no sort (TopScoreDocCollector). */
  static final String[] SORTS = {
    "score-docs",
    "l:long:min:asc:last",
    "doc",
    "i:int:min:desc:last,score",
    "score,i:int:min:asc:last",
  };

  static final int[] NS = {1, 2, 9, 100, 2_999, 3_000, 3_001, 50_000};

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("terminate_after_index");
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);

    StringBuilder m = new StringBuilder();
    try (Directory dir = FSDirectory.open(out)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setRAMBufferSizeMB(256);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      Random random = new Random(20260926L);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
            int id = seg * DOCS_PER_SEGMENT + i;
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            doc.add(new TextField("body", GenMixedBooleanScoring.body(random), Field.Store.NO));
            doc.add(new LongPoint("r", id));
            if (random.nextInt(10) != 0) {
              long l = random.nextInt(500) - 250L;
              doc.add(new SortedNumericDocValuesField("l", l));
              doc.add(new LongPoint("l", l));
            }
            int iv = random.nextInt(100);
            doc.add(new SortedNumericDocValuesField("i", iv));
            doc.add(new IntPoint("i", iv));
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int id = 0; id < DOCS_PER_SEGMENT; id += 7) {
          w.deleteDocuments(new Term("id", Integer.toString(id)));
          w.deleteDocuments(new Term("id", Integer.toString(2 * DOCS_PER_SEGMENT + id + 3)));
        }
        w.commit();
      }

      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      if (sis.size() != SEGMENTS) {
        throw new AssertionError("expected " + SEGMENTS + " segments, got " + sis.size());
      }
      int run = 0;
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setQueryCache(null);
        for (String qs : QUERIES) {
          Query q = GenSortedSearch.parse(new GenMixedBooleanScoring.Tokens(qs));
          int matches = searcher.count(q);
          TreeSet<Integer> ns = new TreeSet<>();
          for (int n : NS) {
            ns.add(n);
          }
          for (int n : new int[] {matches - 1, matches, matches + 1}) {
            if (n > 0) {
              ns.add(n);
            }
          }
          for (int n : ns) {
            for (String ss : SORTS) {
              for (boolean track : new boolean[] {false, true}) {
                if (track && (ss.equals("score-docs") || ss.startsWith("score"))) {
                  continue;
                }
                run = record(m, run, searcher, q, qs, ss, n, 10, track);
              }
            }
            run = recordCount(m, run, searcher, q, qs, n);
          }
        }
        // A concurrent size-0 count: per slice an EarlyTerminatingCollector(TotalHitCountCollector,
        // n) that is not forced (EmptyTopDocsCollectorContext.createManager).
        int counts = 0;
        Sliced sliced = new Sliced(reader);
        int[][][] sliceSets = {{{0, 2}, {1, 3}}, {{0, 1}, {2, 3}}, {{0, 1, 2, 3}}, {{3}, {2}, {1}, {0}}};
        for (String qs : QUERIES) {
          Query q = GenSortedSearch.parse(new GenMixedBooleanScoring.Tokens(qs));
          org.apache.lucene.search.Weight w =
              sliced.createWeight(sliced.rewrite(q), ScoreMode.COMPLETE_NO_SCORES, 1f);
          StringBuilder iterate = new StringBuilder();
          for (LeafReaderContext leaf : reader.leaves()) {
            iterate.append(w.count(leaf) == -1 ? '1' : '0');
          }
          TreeSet<Integer> ns = new TreeSet<>(List.of(1, 5, 100, 1_000, 2_000, 2_571, 2_572, 3_000, 5_000, 10_000));
          ns.add(Math.max(1, sliced.count(q) / 2));
          // Each segment's live matches: a limit reached exactly at a segment's end.
          for (LeafReaderContext leaf : reader.leaves()) {
            org.apache.lucene.search.Scorer sc = w.scorer(leaf);
            if (sc == null) {
              continue;
            }
            org.apache.lucene.util.Bits live = leaf.reader().getLiveDocs();
            int here = 0;
            var it = sc.iterator();
            for (int d = it.nextDoc(); d != org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS; d = it.nextDoc()) {
              if (live == null || live.get(d)) {
                here++;
              }
            }
            if (here > 0) {
              ns.add(here);
            }
          }
          for (int[][] slices : sliceSets) {
            for (int n : ns) {
              boolean terminated = false;
              for (int[] slice : slices) {
                CountEarly c = new CountEarly(new TotalHitCountCollector(), n);
                org.apache.lucene.search.IndexSearcher.LeafReaderContextPartition[] parts =
                    new org.apache.lucene.search.IndexSearcher.LeafReaderContextPartition[slice.length];
                for (int i = 0; i < slice.length; i++) {
                  parts[i] = org.apache.lucene.search.IndexSearcher.LeafReaderContextPartition
                      .createForEntireSegment(reader.leaves().get(slice[i]));
                }
                java.util.Arrays.sort(parts, Comparator.comparingInt(p -> p.ctx.docBase));
                sliced.run(parts, w, c);
                terminated |= c.terminated;
              }
              String k = "count." + counts;
              m.append(k).append(".query=").append(qs).append('\n');
              m.append(k).append(".n=").append(n).append('\n');
              m.append(k).append(".iterate=").append(iterate).append('\n');
              StringBuilder sl = new StringBuilder();
              for (int[] slice : slices) {
                if (sl.length() > 0) {
                  sl.append('|');
                }
                for (int i = 0; i < slice.length; i++) {
                  sl.append(i == 0 ? "" : ",").append(slice[i]);
                }
              }
              m.append(k).append(".slices=").append(sl).append('\n');
              m.append(k).append(".terminated=").append(terminated).append('\n');
              counts++;
            }
          }
        }
        m.insert(0, "count_count=" + counts + "\n");
      }
      m.insert(0, "run_count=" + run + "\n");
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote " + out);
  }

  /** An IndexSearcher that searches the leaves it is given, as a slice's collector sees them. */
  static final class Sliced extends IndexSearcher {
    Sliced(DirectoryReader reader) {
      super(reader);
      setQueryCache(null);
    }

    void run(LeafReaderContextPartition[] parts, org.apache.lucene.search.Weight w, Collector c) throws IOException {
      search(parts, w, c);
    }
  }

  /** OpenSearch's EarlyTerminatingCollector(in, max, false): stops a slice, not the search. */
  static final class CountEarly extends org.apache.lucene.search.FilterCollector {
    final int max;
    int collected;
    boolean terminated;

    CountEarly(Collector in, int max) {
      super(in);
      this.max = max;
    }

    @Override
    public LeafCollector getLeafCollector(LeafReaderContext context) throws IOException {
      if (collected >= max) {
        terminated = true;
        throw new org.apache.lucene.search.CollectionTerminatedException();
      }
      return new org.apache.lucene.search.FilterLeafCollector(super.getLeafCollector(context)) {
        @Override
        public void collect(int doc) throws IOException {
          if (++collected > max) {
            terminated = true;
            throw new org.apache.lucene.search.CollectionTerminatedException();
          }
          super.collect(doc);
        }
      };
    }
  }

  /** OpenSearch's EarlyTerminatingCollector with forceTermination, around nothing. */
  static final class Early implements Collector {
    static final class Stop extends RuntimeException {
      Stop() {
        super(null, null, false, false);
      }
    }

    final int max;
    int collected;
    boolean terminated;

    Early(int max) {
      this.max = max;
    }

    @Override
    public LeafCollector getLeafCollector(LeafReaderContext context) {
      if (collected >= max) {
        terminated = true;
        throw new Stop();
      }
      return new LeafCollector() {
        @Override
        public void setScorer(Scorable scorer) {}

        @Override
        public void collect(int doc) {
          if (++collected > max) {
            terminated = true;
            throw new Stop();
          }
        }
      };
    }

    @Override
    public ScoreMode scoreMode() {
      return ScoreMode.COMPLETE_NO_SCORES;
    }
  }

  static int record(StringBuilder m, int run, IndexSearcher searcher, Query q, String qs, String ss, int n, int topN, boolean track)
      throws IOException {
    Sort sort = ss.equals("score-docs") ? new Sort(SortField.FIELD_SCORE) : GenSortedSearch.sort(ss);
    TopDocsCollector<?> top = ss.equals("score-docs")
        ? new TopScoreDocCollectorManager(topN, null, Integer.MAX_VALUE).newCollector()
        : new TopFieldCollectorManager(sort, topN, null, Integer.MAX_VALUE).newCollector();
    float[] max = {Float.NEGATIVE_INFINITY};
    Early early = new Early(n);
    Collector c = track
        ? MultiCollector.wrap(early, top, new GenSortedSearch.MaxScore(max))
        : MultiCollector.wrap(early, top);
    try {
      searcher.search(q, c);
    } catch (Early.Stop e) {
      // QueryPhase: terminatedEarly(true).
    }
    TopDocs td = top.topDocs();
    String k = "run." + run;
    m.append(k).append(".query=").append(qs).append('\n');
    m.append(k).append(".sort=").append(GenSortedSearch.spec(sort)).append('\n');
    m.append(k).append(".score_docs=").append(ss.equals("score-docs")).append('\n');
    m.append(k).append(".n=").append(n).append('\n');
    m.append(k).append(".top_n=").append(topN).append('\n');
    StringBuilder hits = new StringBuilder();
    for (ScoreDoc sd : td.scoreDocs) {
      if (hits.length() > 0) {
        hits.append(',');
      }
      if (sd instanceof FieldDoc fd) {
        hits.append(GenSortedSearch.hit(sort, fd));
      } else {
        hits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
      }
    }
    m.append(k).append(".hits=").append(hits).append('\n');
    m.append(k).append(".total=").append(td.totalHits.value()).append('\n');
    m.append(k).append(".relation=")
        .append(td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte")
        .append('\n');
    m.append(k).append(".collected=").append(Math.min(early.collected, n)).append('\n');
    m.append(k).append(".terminated=").append(early.terminated).append('\n');
    if (track) {
      m.append(k).append(".track=true\n");
      m.append(k).append(".max_score=")
          .append(Float.isInfinite(max[0]) ? "nan" : Integer.toString(Float.floatToIntBits(max[0])))
          .append('\n');
    }
    return run + 1;
  }

  /** {@code size: 0}: the count alone (EmptyTopDocsCollectorContext's TotalHitCountCollector). */
  static int recordCount(StringBuilder m, int run, IndexSearcher searcher, Query q, String qs, int n) throws IOException {
    TotalHitCountCollector count = new TotalHitCountCollector();
    Early early = new Early(n);
    try {
      searcher.search(q, MultiCollector.wrap(early, count));
    } catch (Early.Stop e) {
      // terminated
    }
    String k = "run." + run;
    m.append(k).append(".query=").append(qs).append('\n');
    m.append(k).append(".sort=count\n");
    m.append(k).append(".n=").append(n).append('\n');
    m.append(k).append(".total=").append(count.getTotalHits()).append('\n');
    m.append(k).append(".collected=").append(Math.min(early.collected, n)).append('\n');
    m.append(k).append(".terminated=").append(early.terminated).append('\n');
    return run + 1;
  }
}
