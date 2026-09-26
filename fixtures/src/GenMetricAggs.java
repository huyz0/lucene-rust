import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FloatPoint;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.SimpleCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.NumericUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.Stream;

/**
 * OpenSearch's numeric metric aggregations ({@code min}, {@code max}, {@code sum}, {@code avg},
 * {@code value_count}, {@code stats}) recorded for crates/lucene-search/tests/metric_aggs_fixtures.rs.
 *
 * <p>The expected state is computed here the way OpenSearch's aggregators compute it -- this
 * generator runs on Lucene's jars alone, so their loops are reproduced: every live match in
 * document order, each value read as {@code SortedNumericDoubleValues} reads it, summed through
 * {@code CompensatedSum} (copied below), {@code Math.min}/{@code Math.max} over every value
 * ({@code StatsAggregator}) and over each document's first/last value ({@code MinAggregator}'s
 * and {@code MaxAggregator}'s {@code MultiValueMode}).
 *
 * <p>Three segments of 20,000 documents, two with deletions:
 *
 * <pre>
 *   l    long, 20% missing
 *   ml   long, 0-3 values, some near +-2^60 (rounded when widened to double)
 *   d    double, specials now and then (NaN, +-0.0, +-inf, tiny and huge magnitudes)
 *   md   double, 0-3 values, a NaN now and then
 *   f    float
 *   i    int, every document
 * </pre>
 *
 * <p>Each run records, per field, {@code count:sum:delta:min:max:minOfMins:maxOfMaxes} with the
 * doubles as {@code doubleToRawLongBits} in hex.
 */
public class GenMetricAggs {
  static final int DOCS_PER_SEGMENT = 20_000;
  static final int SEGMENTS = 3;
  static final String[] FIELDS = {"l", "ml", "d", "md", "f", "i"};

  static final String[] QUERIES = {
    "(all)",
    "(t w0)",
    "(t w30)",
    "(b 0 (+ (t w1)) (- (t w2)))",
    "(b 0 (# (t w0)) (# (r 5000 45000)))",
    "(b 0 (? (t w3)) (? (t w4)))",
    "(t nosuchterm)",
  };

  /** OpenSearch's CompensatedSum. */
  static final class CompensatedSum {
    double value;
    double delta;

    void add(double v) {
      if (Double.isFinite(v) == false) {
        value = v + value;
      }
      if (Double.isFinite(value)) {
        double corrected = v + delta;
        double updated = value + corrected;
        delta = corrected - (updated - value);
        value = updated;
      }
    }
  }

  static final class State {
    long count;
    final CompensatedSum sum = new CompensatedSum();
    double min = Double.POSITIVE_INFINITY;
    double max = Double.NEGATIVE_INFINITY;
    double minOfMins = Double.POSITIVE_INFINITY;
    double maxOfMaxes = Double.NEGATIVE_INFINITY;

    String record() {
      return count + ":" + hex(sum.value) + ":" + hex(sum.delta) + ":" + hex(min) + ":" + hex(max) + ":"
          + hex(minOfMins) + ":" + hex(maxOfMaxes);
    }
  }

  static String hex(double d) {
    return Long.toHexString(Double.doubleToRawLongBits(d));
  }

  static double read(String field, long v) {
    return switch (field) {
      case "d", "md" -> NumericUtils.sortableLongToDouble(v);
      case "f" -> NumericUtils.sortableIntToFloat((int) v);
      default -> (double) v;
    };
  }

  static double special(Random r) {
    return switch (r.nextInt(8)) {
      case 0 -> Double.NaN;
      case 1 -> -0.0;
      case 2 -> 0.0;
      case 3 -> r.nextBoolean() ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
      case 4 -> r.nextDouble() * 1e-300;
      case 5 -> (r.nextDouble() - 0.5) * 1e300;
      default -> r.nextGaussian() * 1e3;
    };
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("metric_aggs_index");
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
      Random random = new Random(20260928L);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = 0; i < DOCS_PER_SEGMENT; i++) {
            int id = seg * DOCS_PER_SEGMENT + i;
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            doc.add(new TextField("body", GenMixedBooleanScoring.body(random), Field.Store.NO));
            doc.add(new LongPoint("r", id));
            if (random.nextInt(5) != 0) {
              long l = random.nextInt(2_000_001) - 1_000_000L;
              doc.add(new SortedNumericDocValuesField("l", l));
              doc.add(new LongPoint("l", l));
            }
            for (int v = 0, n = random.nextInt(4); v < n; v++) {
              long l = random.nextInt(10) == 0 ? (random.nextBoolean() ? 1L : -1L) * ((1L << 60) + random.nextInt(1 << 20))
                  : random.nextInt(1000);
              doc.add(new SortedNumericDocValuesField("ml", l));
            }
            double d = random.nextInt(50) == 0 ? special(random) : random.nextGaussian() * 1e3;
            doc.add(new SortedNumericDocValuesField("d", NumericUtils.doubleToSortableLong(d)));
            doc.add(new DoublePoint("d", d));
            for (int v = 0, n = random.nextInt(4); v < n; v++) {
              double x = random.nextInt(100) == 0 ? Double.NaN : random.nextDouble() * 100 - 50;
              doc.add(new SortedNumericDocValuesField("md", NumericUtils.doubleToSortableLong(x)));
            }
            float f = (float) (random.nextGaussian() * 10);
            doc.add(new SortedNumericDocValuesField("f", NumericUtils.floatToSortableInt(f)));
            doc.add(new FloatPoint("f", f));
            int iv = random.nextInt(1000) - 500;
            doc.add(new SortedNumericDocValuesField("i", iv));
            doc.add(new IntPoint("i", iv));
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int id = 0; id < DOCS_PER_SEGMENT; id += 23) {
          w.deleteDocuments(new Term("id", Integer.toString(id)));
          w.deleteDocuments(new Term("id", Integer.toString(2 * DOCS_PER_SEGMENT + id + 7)));
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
          State[] states = new State[FIELDS.length];
          for (int k = 0; k < FIELDS.length; k++) {
            states[k] = new State();
          }
          searcher.search(q, new org.apache.lucene.search.CollectorManager<SimpleCollector, Void>() {
            @Override
            public SimpleCollector newCollector() {
              return new SimpleCollector() {
                final SortedNumericDocValues[] dvs = new SortedNumericDocValues[FIELDS.length];

                @Override
                protected void doSetNextReader(LeafReaderContext context) throws IOException {
                  for (int k = 0; k < FIELDS.length; k++) {
                    dvs[k] = DocValues.getSortedNumeric(context.reader(), FIELDS[k]);
                  }
                }

                @Override
                public void collect(int doc) throws IOException {
                  for (int k = 0; k < FIELDS.length; k++) {
                    if (dvs[k].advanceExact(doc) == false) {
                      continue;
                    }
                    State s = states[k];
                    int n = dvs[k].docValueCount();
                    s.count += n;
                    double first = 0, last = 0;
                    for (int j = 0; j < n; j++) {
                      double v = read(FIELDS[k], dvs[k].nextValue());
                      if (j == 0) {
                        first = v;
                      }
                      last = v;
                      s.sum.add(v);
                      s.min = Math.min(s.min, v);
                      s.max = Math.max(s.max, v);
                    }
                    s.minOfMins = Math.min(s.minOfMins, first);
                    s.maxOfMaxes = Math.max(s.maxOfMaxes, last);
                  }
                }

                @Override
                public ScoreMode scoreMode() {
                  return ScoreMode.COMPLETE_NO_SCORES;
                }
              };
            }

            @Override
            public Void reduce(java.util.Collection<SimpleCollector> cs) {
              return null;
            }
          });
          m.append("run.").append(run).append(".query=").append(qs).append('\n');
          for (int k = 0; k < FIELDS.length; k++) {
            m.append("run.").append(run).append('.').append(FIELDS[k]).append('=').append(states[k].record()).append('\n');
          }
          run++;
        }
      }
      m.insert(0, "run_count=" + run + "\n");
    }
    Files.writeString(out.resolve("manifest.properties"), m.toString());
    System.out.println("wrote " + out);
  }
}
