import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.LatLonDocValuesField;
import org.apache.lucene.document.LatLonPoint;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Java side of the geo point query benchmark pair (M9 T9.2); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_geo_points.rs}, with the same case names. {@code build <dir>}
 * writes the corpus once -- one million documents with a {@code LatLonPoint} and a {@code
 * LatLonDocValuesField}, half uniform over the globe and half in twenty clusters, force-merged to
 * one segment -- and the query set ({@code geo-queries.tsv}) both engines replay: boxes (two across
 * the dateline), distance queries, polygons, distance sorts (top 10 of every document) and
 * {@code LatLonPoint.nearest} (10). Every hit is collected by a plain counting collector, so
 * neither engine can answer a query from a count shortcut; each case prints a {@code #check}
 * digest of its hit counts (or hit ids) that the report compares before it shows a ratio.
 */
public final class GeoPointsMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int DOCS = 1_000_000;

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
  }

  static void check(String c, Fnv d, long n) {
    System.out.printf("#check\t%s\t%016x\t%d%n", c, d.h, n);
  }

  // --- the corpus ------------------------------------------------------------------------

  static double clampLat(double v) {
    return Math.max(-90, Math.min(90, v));
  }

  static double wrapLon(double v) {
    while (v > 180) v -= 360;
    while (v < -180) v += 360;
    return v;
  }

  static void build(Path dir) throws IOException {
    Random r = new Random(0x6E0_9013L);
    double[][] clusters = new double[20][];
    for (int i = 0; i < clusters.length; i++) {
      clusters[i] = new double[] {r.nextDouble() * 140 - 70, r.nextDouble() * 360 - 180};
    }
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setRAMBufferSizeMB(512);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        double lat;
        double lon;
        if (r.nextBoolean()) {
          lat = r.nextDouble() * 180 - 90;
          lon = r.nextDouble() * 360 - 180;
        } else {
          double[] c = clusters[r.nextInt(clusters.length)];
          double s = 0.5 + r.nextInt(3);
          lat = clampLat(c[0] + r.nextGaussian() * s);
          lon = wrapLon(c[1] + r.nextGaussian() * s);
        }
        Document doc = new Document();
        doc.add(new LatLonPoint("p", lat, lon));
        doc.add(new LatLonDocValuesField("p", lat, lon));
        w.addDocument(doc);
      }
      w.forceMerge(1);
    }
    StringBuilder q = new StringBuilder();
    for (int i = 0; i < 40; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      double h = 0.05 + r.nextDouble() * 5;
      double w = 0.05 + r.nextDouble() * 5;
      double minLon = wrapLon(c[1] - w);
      double maxLon = wrapLon(c[1] + w);
      if (i % 20 == 0) {
        minLon = 175;
        maxLon = -175;
      }
      q.append("box\t").append(clampLat(c[0] - h)).append('\t').append(clampLat(c[0] + h)).append('\t')
          .append(minLon).append('\t').append(maxLon).append('\n');
    }
    for (int i = 0; i < 40; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      q.append("dist\t").append(c[0] + r.nextGaussian()).append('\t').append(wrapLon(c[1] + r.nextGaussian())).append('\t')
          .append(Math.pow(10, 3 + r.nextDouble() * 2.7)).append('\n');
    }
    for (int i = 0; i < 20; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      double[][] ring = GeoStar.ring(r, c[0], c[1], 0.1 + r.nextDouble() * 4, 8 + r.nextInt(50));
      StringBuilder sb = new StringBuilder();
      for (int k = 0; k < ring[0].length; k++) {
        if (k > 0) sb.append(';');
        sb.append(ring[0][k]).append(' ').append(ring[1][k]);
      }
      q.append("poly\t").append(sb).append('\n');
    }
    for (int i = 0; i < 8; i++) {
      q.append("sort\t").append(r.nextDouble() * 160 - 80).append('\t').append(r.nextDouble() * 360 - 180).append('\n');
    }
    for (int i = 0; i < 20; i++) {
      q.append("nearest\t").append(r.nextDouble() * 160 - 80).append('\t').append(r.nextDouble() * 360 - 180).append('\n');
    }
    Files.writeString(dir.resolve("geo-queries.tsv"), q.toString());
  }

  /** A star-shaped ring around a centre, closed; simple by construction. */
  static final class GeoStar {
    static double[][] ring(Random r, double clat, double clon, double radius, int n) {
      double[] angles = new double[n];
      for (int i = 0; i < n; i++) angles[i] = r.nextDouble() * 2 * Math.PI;
      java.util.Arrays.sort(angles);
      double[] lats = new double[n + 1];
      double[] lons = new double[n + 1];
      for (int i = 0; i < n; i++) {
        double rad = radius * (0.4 + 0.6 * r.nextDouble());
        lats[i] = clampLat(clat + rad * Math.sin(angles[i]));
        lons[i] = Math.max(-180, Math.min(180, clon + rad * Math.cos(angles[i])));
      }
      lats[n] = lats[0];
      lons[n] = lons[0];
      return new double[][] {lats, lons};
    }
  }

  // --- the cases ---------------------------------------------------------------------------

  static final class Count implements Collector {
    long count;

    @Override
    public LeafCollector getLeafCollector(LeafReaderContext context) {
      return new LeafCollector() {
        @Override
        public void setScorer(Scorable scorer) {}

        @Override
        public void collect(int doc) {
          count++;
        }
      };
    }

    @Override
    public ScoreMode scoreMode() {
      return ScoreMode.COMPLETE_NO_SCORES;
    }
  }

  static long count(IndexSearcher s, Query q) throws IOException {
    return s.search(
        q,
        new CollectorManager<Count, Long>() {
          @Override
          public Count newCollector() {
            return new Count();
          }

          @Override
          public Long reduce(java.util.Collection<Count> cs) {
            long n = 0;
            for (Count c : cs) n += c.count;
            return n;
          }
        });
  }

  static double d(String s) {
    return Double.parseDouble(s);
  }

  public static void main(String[] args) throws IOException {
    if (args[0].equals("build")) {
      Path dir = Path.of(args[1]);
      if (Files.exists(dir.resolve("geo-queries.tsv"))) return;
      Files.createDirectories(dir);
      build(dir);
      return;
    }
    Path dir = Path.of(args[1]);
    List<String[]> lines = new ArrayList<>();
    for (String l : Files.readAllLines(dir.resolve("geo-queries.tsv"))) lines.add(l.split("\t"));
    List<Query> boxes = new ArrayList<>();
    List<Query> dists = new ArrayList<>();
    List<Query> polys = new ArrayList<>();
    List<double[]> sorts = new ArrayList<>();
    List<double[]> nearest = new ArrayList<>();
    for (String[] a : lines) {
      switch (a[0]) {
        case "box" -> boxes.add(LatLonPoint.newBoxQuery("p", d(a[1]), d(a[2]), d(a[3]), d(a[4])));
        case "dist" -> dists.add(LatLonPoint.newDistanceQuery("p", d(a[1]), d(a[2]), d(a[3])));
        case "poly" -> {
          String[] pts = a[1].split(";");
          double[] lats = new double[pts.length];
          double[] lons = new double[pts.length];
          for (int i = 0; i < pts.length; i++) {
            String[] p = pts[i].split(" ");
            lats[i] = d(p[0]);
            lons[i] = d(p[1]);
          }
          polys.add(LatLonPoint.newPolygonQuery("p", new Polygon(lats, lons)));
        }
        case "sort" -> sorts.add(new double[] {d(a[1]), d(a[2])});
        case "nearest" -> nearest.add(new double[] {d(a[1]), d(a[2])});
        default -> throw new IllegalStateException(a[0]);
      }
    }
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      String[] names = {"geo_box", "geo_distance", "geo_polygon"};
      List<List<Query>> sets = List.of(boxes, dists, polys);
      for (int k = 0; k < names.length; k++) {
        List<Query> qs = sets.get(k);
        Fnv f = new Fnv();
        for (Query q : qs) f.add(count(s, q));
        check(names[k], f, qs.size());
        measure(names[k], () -> {
          long n = 0;
          for (Query q : qs) n += count(s, q);
          sink += n;
          return qs.size();
        });
      }
      Fnv f = new Fnv();
      for (double[] o : sorts) {
        TopFieldDocs td = s.search(new MatchAllDocsQuery(), 10, new Sort(LatLonDocValuesField.newDistanceSort("p", o[0], o[1])));
        for (var sd : td.scoreDocs) {
          f.add(sd.doc);
          f.add(Double.doubleToRawLongBits((Double) ((FieldDoc) sd).fields[0]));
        }
      }
      check("geo_distance_sort", f, sorts.size());
      measure("geo_distance_sort", () -> {
        for (double[] o : sorts) {
          TopFieldDocs td = s.search(new MatchAllDocsQuery(), 10, new Sort(LatLonDocValuesField.newDistanceSort("p", o[0], o[1])));
          sink += td.scoreDocs[0].doc;
        }
        return sorts.size();
      });
      f = new Fnv();
      for (double[] o : nearest) {
        TopFieldDocs td = LatLonPoint.nearest(s, "p", o[0], o[1], 10);
        for (var sd : td.scoreDocs) {
          f.add(sd.doc);
          f.add(Double.doubleToRawLongBits((Double) ((FieldDoc) sd).fields[0]));
        }
      }
      check("geo_nearest", f, nearest.size());
      measure("geo_nearest", () -> {
        for (double[] o : nearest) sink += LatLonPoint.nearest(s, "p", o[0], o[1], 10).scoreDocs[0].doc;
        return nearest.size();
      });
    }
  }
}
