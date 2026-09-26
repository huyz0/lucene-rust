import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
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
import org.apache.lucene.search.FilterCollector;
import org.apache.lucene.search.FilterLeafCollector;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.SimpleCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Random;
import java.util.TreeSet;
import java.util.stream.Stream;

/**
 * OpenSearch's {@code min_score}, recorded from Lucene for
 * crates/lucene-search/tests/min_score_fixtures.rs.
 *
 * <p>OpenSearch puts a {@code MinimumScoreCollector} outside every other collector of the query
 * phase (QueryPhase: "apply the minimum score after multi collector so we filter aggs as well"): a
 * document reaches the top-docs collector, the count and the aggregations only when its score is
 * at least the minimum, and the search runs {@code TOP_SCORES} when the top-docs collector alone
 * prunes, {@code COMPLETE} otherwise. That collector is OpenSearch's, not Lucene's, so it is
 * reproduced here ({@link MinScore}); the top-docs collector is Lucene's own.
 *
 * <p>Three segments of 3,000 documents, the first and last with deletions, each with {@code v},
 * its id as a numeric doc value. Per query, minimums at fractions of its top score and at its
 * fifth hit's score exactly (a tie at the boundary passes). Each run records the top 10 under
 * two total-hits thresholds (hits as {@code doc:scoreBits}, the total and its relation), and, over
 * a {@code COMPLETE} search, the passing documents' count and the sum of their {@code v}.
 */
public class GenMinScore {
  static final int DOCS_PER_SEGMENT = 3_000;
  static final int SEGMENTS = 3;

  static final String[] QUERIES = {
    "(all)",
    "(t w0)",
    "(t w30)",
    "(b 0 (+ (t w1)) (- (t w2)))",
    "(b 0 (? (t w3)) (? (t w4)) (? (t w5)))",
    "(b 0 (+ (t w0)) (# (r 0 5999)))",
    "(b 0 (+ (t w0)) (? (p w1 w2)))",
    "(b 2 (? (t w1)) (? (t w6)) (? (t w7)))",
    "(t nosuchterm)",
  };

  static final double[] FRACTIONS = {0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1};

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("min_score_index");
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
            doc.add(new NumericDocValuesField("v", id));
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int id = 0; id < DOCS_PER_SEGMENT; id += 11) {
          w.deleteDocuments(new Term("id", Integer.toString(id)));
          w.deleteDocuments(new Term("id", Integer.toString(2 * DOCS_PER_SEGMENT + id + 5)));
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
          TopDocs top = searcher.search(q, 5);
          TreeSet<Float> mins = new TreeSet<>();
          float best = top.scoreDocs.length == 0 ? 1f : top.scoreDocs[0].score;
          for (double f : FRACTIONS) {
            mins.add((float) (best * f));
          }
          if (top.scoreDocs.length == 5) {
            mins.add(top.scoreDocs[4].score);
          }
          for (float min : mins) {
            for (int threshold : new int[] {Integer.MAX_VALUE, 20}) {
              TopScoreDocCollector tc = new TopScoreDocCollectorManager(10, null, threshold).newCollector();
              searcher.search(q, new MinScore(tc, min));
              TopDocs td = tc.topDocs();
              String k = "run." + run;
              m.append(k).append(".query=").append(qs).append('\n');
              m.append(k).append(".min=").append(Float.floatToIntBits(min)).append('\n');
              m.append(k).append(".threshold=").append(threshold == Integer.MAX_VALUE ? "max" : threshold).append('\n');
              StringBuilder hits = new StringBuilder();
              for (ScoreDoc sd : td.scoreDocs) {
                if (hits.length() > 0) {
                  hits.append(',');
                }
                hits.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
              }
              m.append(k).append(".hits=").append(hits).append('\n');
              m.append(k).append(".total=").append(td.totalHits.value()).append('\n');
              m.append(k).append(".relation=")
                  .append(td.totalHits.relation() == TotalHits.Relation.EQUAL_TO ? "eq" : "gte")
                  .append('\n');
              // Every passing document, scored COMPLETE: the count and the aggregations' view.
              Passing passing = new Passing();
              searcher.search(q, new MinScore(passing, min));
              m.append(k).append(".passing=").append(passing.count).append('\n');
              m.append(k).append(".passing_v_sum=").append(passing.sum).append('\n');
              run++;
            }
          }
        }
      }
      m.insert(0, "run_count=" + run + "\n");
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote " + out);
  }

  /** OpenSearch's MinimumScoreCollector. */
  static final class MinScore extends FilterCollector {
    final float min;

    MinScore(Collector in, float min) {
      super(in);
      this.min = min;
    }

    @Override
    public LeafCollector getLeafCollector(LeafReaderContext context) throws IOException {
      return new FilterLeafCollector(super.getLeafCollector(context)) {
        Scorable scorer;

        @Override
        public void setScorer(Scorable scorer) throws IOException {
          this.scorer = scorer;
          in.setScorer(scorer);
        }

        @Override
        public void collect(int doc) throws IOException {
          if (scorer.score() >= min) {
            in.collect(doc);
          }
        }
      };
    }

    @Override
    public void setWeight(org.apache.lucene.search.Weight weight) {
      // MinimumScoreCollector passes no weight on.
    }

    @Override
    public ScoreMode scoreMode() {
      return in.scoreMode() == ScoreMode.TOP_SCORES ? ScoreMode.TOP_SCORES : ScoreMode.COMPLETE;
    }
  }

  /** Counts the documents that reach it and sums their {@code v} (their global id). */
  static final class Passing extends SimpleCollector {
    long count;
    long sum;
    int docBase;

    @Override
    protected void doSetNextReader(LeafReaderContext context) {
      docBase = context.docBase;
    }

    @Override
    public void collect(int doc) {
      count++;
      sum += docBase + doc;
    }

    @Override
    public ScoreMode scoreMode() {
      return ScoreMode.COMPLETE_NO_SCORES;
    }
  }
}
