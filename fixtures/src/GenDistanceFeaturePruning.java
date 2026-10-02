import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LatLonDocValuesField;
import org.apache.lucene.document.LatLonPoint;
import org.apache.lucene.document.LongField;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * The distance feature queries' pruning across segments, recorded from Lucene for
 * crates/lucene-search/tests/distance_feature_pruning_fixtures.rs: what {@code totalHits} (value
 * and relation) a top-n search reports once {@code TopScoreDocCollector} starts handing the scorer
 * a minimum competitive score -- {@code Math.nextUp} of its worst kept score, pushed after a
 * competitive hit and again at every leaf's start ({@code setScorer}).
 *
 * <p>Written to {@code distance_feature_pruning/}:
 *
 * <pre>
 *   index/       {@value #SEGMENTS} segments of {@value #DOCS_PER_SEGMENT} documents, a few deleted.
 *                Fields: l (LongField: points + sorted numeric doc values, some documents with two
 *                values), n (LongPoint + NumericDocValuesField, single-valued), p (LatLonPoint +
 *                LatLonDocValuesField). A sixth of the documents sit exactly at {@value #ORIGIN}
 *                (and at the geo origin), so a top-n queue fills with hits scoring exactly the
 *                boost, which nothing can beat. Segments are large enough that an eighth of a
 *                leaf's cost exceeds the estimate of a crossing BKD leaf, so the scorer's iterator
 *                narrowing actually happens.
 *   queries.tsv  kind, field, weight, origin, pivot, n, then "=>" and S, totalHits value, relation,
 *                and the hits as doc:scoreBits in rank order -- more than 100 of them as '#' and
 *                that list's String.hashCode
 * </pre>
 *
 * <p>The geo queries include a NaN pivot: the constructor's {@code pivotDistance <= 0} check lets
 * it through, every score is NaN, and {@code updateMinCompetitiveScore}'s {@code localMinScore >
 * minCompetitiveScore} never holds, so nothing is pruned.
 */
public class GenDistanceFeaturePruning {
  static final int SEGMENTS = 4;
  static final int DOCS_PER_SEGMENT = 6000;
  static final long ORIGIN = 1_000_000L;
  static final double LAT = 40.5;
  static final double LON = -73.25;

  static void clean(Path out) throws IOException {
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
  }

  static String scored(IndexSearcher s, Query q, int n) throws IOException {
    TopDocs td = s.search(q, n);
    StringBuilder sb =
        new StringBuilder("S\t")
            .append(td.totalHits.value())
            .append('\t')
            .append(td.totalHits.relation())
            .append('\t');
    StringBuilder hits = new StringBuilder();
    for (int i = 0; i < td.scoreDocs.length; i++) {
      if (i > 0) hits.append(',');
      hits.append(td.scoreDocs[i].doc)
          .append(':')
          .append(Integer.toHexString(Float.floatToRawIntBits(td.scoreDocs[i].score)));
    }
    // A long hit list as its String.hashCode, to keep the file small.
    if (td.scoreDocs.length > 100) {
      return sb.append('#').append(hits.toString().hashCode()).toString();
    }
    return sb.append(hits).toString();
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("distance_feature_pruning");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(20261002L);
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int id = 0;
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int i = 0; i < DOCS_PER_SEGMENT; i++, id++) {
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.NO));
            boolean atOrigin = r.nextInt(6) == 0;
            if (r.nextInt(20) != 0) {
              long v = atOrigin ? ORIGIN : ORIGIN + (long) (r.nextGaussian() * 50_000);
              doc.add(new LongField("l", v, Field.Store.NO));
              if (r.nextInt(8) == 0) {
                doc.add(new LongField("l", v + 1 + r.nextInt(100_000), Field.Store.NO));
              }
            }
            if (r.nextInt(20) != 0) {
              long v = atOrigin ? ORIGIN : ORIGIN - 200_000 + (long) (r.nextDouble() * 400_000);
              doc.add(new LongPoint("n", v));
              doc.add(new NumericDocValuesField("n", v));
            }
            if (r.nextInt(20) != 0) {
              double lat = atOrigin ? LAT : LAT + r.nextGaussian() * 2;
              double lon = atOrigin ? LON : LON + r.nextGaussian() * 2;
              lat = Math.max(-90, Math.min(90, lat));
              doc.add(new LatLonPoint("p", lat, lon));
              doc.add(new LatLonDocValuesField("p", lat, lon));
            }
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int d = 0; d < SEGMENTS * DOCS_PER_SEGMENT; d += 1 + r.nextInt(60)) {
          w.deleteDocuments(new Term("id", Integer.toString(d)));
        }
        w.commit();
      }
    }

    StringBuilder q = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != SEGMENTS) throw new AssertionError("segments");
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      int[] ns = {1, 10, 100, 1200};
      float[] weights = {1f, 2.5f};
      for (String field : new String[] {"l", "n"}) {
        for (float weight : weights) {
          for (long origin : new long[] {ORIGIN, ORIGIN + 7_777, ORIGIN - 150_000}) {
            for (long pivot : new long[] {1, 1_000, 100_000}) {
              for (int n : ns) {
                Query fq = LongField.newDistanceFeatureQuery(field, weight, origin, pivot);
                q.append("long\t").append(field).append('\t').append(weight).append('\t')
                    .append(origin).append('\t').append(pivot).append('\t').append(n)
                    .append("\t=>\t").append(scored(s, fq, n)).append('\n');
              }
            }
          }
        }
      }
      for (float weight : weights) {
        for (double[] o : new double[][] {{LAT, LON}, {LAT + 1.5, LON - 0.5}}) {
          for (double pivot : new double[] {Double.NaN, 1, 10_000, 1_000_000}) {
            for (int n : ns) {
              Query fq = LatLonPoint.newDistanceFeatureQuery("p", weight, o[0], o[1], pivot);
              q.append("geo\tp\t").append(weight).append('\t').append(o[0]).append(',')
                  .append(o[1]).append('\t').append(pivot).append('\t').append(n)
                  .append("\t=>\t").append(scored(s, fq, n)).append('\n');
            }
          }
        }
      }
    }
    Files.writeString(root.resolve("queries.tsv"), q.toString());
  }
}
