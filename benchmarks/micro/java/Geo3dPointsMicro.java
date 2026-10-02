import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.spatial3d.Geo3DDocValuesField;
import org.apache.lucene.spatial3d.Geo3DPoint;
import org.apache.lucene.spatial3d.geom.GeoPoint;
import org.apache.lucene.spatial3d.geom.PlanetModel;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * Java side of the spatial3d query benchmark pair (M9 T9.4); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_geo3d_points.rs}, with the same case names. {@code build <dir>}
 * writes the corpus once -- 300 000 documents with a WGS84 {@code Geo3DPoint} and {@code
 * Geo3DDocValuesField}, half uniform and half in twenty clusters, force-merged to one segment --
 * and the query set ({@code geo3d-queries.tsv}) both engines replay: {@code
 * PointInGeo3DShapeQuery} over circles, boxes, polygons and paths (every hit through a counting
 * collector), and the distance and outside-distance sorts (top 10 of every document). Each case
 * prints a {@code #check} digest of its hit counts (or hits) that the report compares first.
 */
public final class Geo3dPointsMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int DOCS = 300_000;
  static final double RADIANS_PER_DEGREE = Math.PI / 180.0;
  static final PlanetModel PM = PlanetModel.WGS84;

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

  static double clampLat(double v) {
    return Math.max(-90, Math.min(90, v));
  }

  static double wrapLon(double v) {
    while (v > 180) v -= 360;
    while (v < -180) v += 360;
    return v;
  }

  static String ring(Random r, double lat, double lon, double radius, int n) {
    StringBuilder sb = new StringBuilder();
    double[] angles = new double[n];
    for (int i = 0; i < n; i++) angles[i] = r.nextDouble() * 2 * Math.PI;
    java.util.Arrays.sort(angles);
    for (int i = 0; i <= n; i++) {
      double a = angles[i % n];
      double rad = radius * (0.5 + 0.5 * r.nextDouble());
      if (i == n) {
        // close the ring on its first point
        sb.append(';').append(sb.substring(0, sb.indexOf(";")));
        break;
      }
      if (i > 0) sb.append(';');
      sb.append(clampLat(lat + rad * StrictMath.sin(a))).append(' ')
          .append(Math.max(-180, Math.min(180, lon + rad * StrictMath.cos(a))));
    }
    return sb.toString();
  }

  static void build(Path dir) throws IOException {
    Random r = new Random(0x3D_9013L);
    double[][] clusters = new double[20][];
    for (int i = 0; i < clusters.length; i++) {
      clusters[i] = new double[] {r.nextDouble() * 140 - 70, r.nextDouble() * 340 - 170};
    }
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setRAMBufferSizeMB(512);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
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
        doc.add(new Geo3DPoint("p", PM, lat, lon));
        doc.add(new Geo3DDocValuesField("p",
            new GeoPoint(PM, lat * RADIANS_PER_DEGREE, lon * RADIANS_PER_DEGREE), PM));
        w.addDocument(doc);
      }
      w.forceMerge(1);
    }
    StringBuilder q = new StringBuilder();
    for (int i = 0; i < 30; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      q.append("dist\t").append(c[0] + r.nextGaussian()).append('\t').append(wrapLon(c[1] + r.nextGaussian()))
          .append('\t').append(Math.pow(10, 3 + r.nextDouble() * 2.7)).append('\n');
    }
    for (int i = 0; i < 30; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      double h = 0.05 + r.nextDouble() * 5;
      double w = 0.05 + r.nextDouble() * 5;
      q.append("box\t").append(clampLat(c[0] - h)).append('\t').append(clampLat(c[0] + h)).append('\t')
          .append(wrapLon(c[1] - w)).append('\t').append(wrapLon(c[1] + w)).append('\n');
    }
    for (int i = 0; i < 15; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      q.append("poly\t").append(ring(r, c[0], c[1], 0.1 + r.nextDouble() * 4, 8 + r.nextInt(30))).append('\n');
    }
    for (int i = 0; i < 15; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      int n = 2 + r.nextInt(6);
      StringBuilder lats = new StringBuilder(), lons = new StringBuilder();
      double la = c[0], lo = c[1];
      for (int k = 0; k < n; k++) {
        if (k > 0) {
          lats.append(';');
          lons.append(';');
        }
        lats.append(la);
        lons.append(lo);
        la = clampLat(la + r.nextGaussian());
        lo = wrapLon(lo + r.nextGaussian());
      }
      q.append("path\t").append(lats).append('\t').append(lons).append('\t')
          .append(Math.pow(10, 3 + r.nextDouble() * 2)).append('\n');
    }
    for (int i = 0; i < 8; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      q.append("sort\t").append(c[0] + r.nextGaussian()).append('\t').append(wrapLon(c[1] + r.nextGaussian()))
          .append('\t').append(2e6).append('\n');
    }
    for (int i = 0; i < 8; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      q.append("osort\t").append(c[0] + r.nextGaussian()).append('\t').append(wrapLon(c[1] + r.nextGaussian()))
          .append('\t').append(1e5).append('\n');
    }
    Files.writeString(dir.resolve("geo3d-queries.tsv"), q.toString());
  }

  static final class Count implements org.apache.lucene.search.Collector {
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

  static double[] list(String s) {
    String[] p = s.split(";");
    double[] out = new double[p.length];
    for (int i = 0; i < p.length; i++) out[i] = Double.parseDouble(p[i]);
    return out;
  }

  static void run(Path dir) throws IOException {
    List<Query> dists = new ArrayList<>(), boxes = new ArrayList<>(), polys = new ArrayList<>(), paths = new ArrayList<>();
    List<SortField> sorts = new ArrayList<>(), osorts = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("geo3d-queries.tsv"))) {
      String[] a = line.split("\t");
      switch (a[0]) {
        case "dist" -> dists.add(Geo3DPoint.newDistanceQuery("p", PM, Double.parseDouble(a[1]), Double.parseDouble(a[2]), Double.parseDouble(a[3])));
        case "box" -> boxes.add(Geo3DPoint.newBoxQuery("p", PM, Double.parseDouble(a[1]), Double.parseDouble(a[2]), Double.parseDouble(a[3]), Double.parseDouble(a[4])));
        case "poly" -> {
          String[] pts = a[1].split(";");
          double[] lats = new double[pts.length], lons = new double[pts.length];
          for (int i = 0; i < pts.length; i++) {
            String[] xy = pts[i].split(" ");
            lats[i] = Double.parseDouble(xy[0]);
            lons[i] = Double.parseDouble(xy[1]);
          }
          polys.add(Geo3DPoint.newPolygonQuery("p", PM, new Polygon(lats, lons)));
        }
        case "path" -> paths.add(Geo3DPoint.newPathQuery("p", list(a[1]), list(a[2]), Double.parseDouble(a[3]), PM));
        case "sort" -> sorts.add(Geo3DDocValuesField.newDistanceSort("p", Double.parseDouble(a[1]), Double.parseDouble(a[2]), Double.parseDouble(a[3]), PM));
        case "osort" -> osorts.add(Geo3DDocValuesField.newOutsideDistanceSort("p", Double.parseDouble(a[1]), Double.parseDouble(a[2]), Double.parseDouble(a[3]), PM));
        default -> throw new IllegalStateException(a[0]);
      }
    }
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      String[] names = {"geo3d_query_distance", "geo3d_query_box", "geo3d_query_polygon", "geo3d_query_path"};
      List<List<Query>> sets = List.of(dists, boxes, polys, paths);
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
      String[] sortNames = {"geo3d_distance_sort", "geo3d_outside_sort"};
      List<List<SortField>> sortSets = List.of(sorts, osorts);
      for (int k = 0; k < sortNames.length; k++) {
        List<SortField> ss = sortSets.get(k);
        Fnv f = new Fnv();
        for (SortField sf : ss) {
          TopFieldDocs td = s.search(new MatchAllDocsQuery(), 10, new Sort(sf));
          for (var sd : td.scoreDocs) {
            f.add(sd.doc);
            f.add(Math.round((Double) ((FieldDoc) sd).fields[0]));
          }
        }
        check(sortNames[k], f, ss.size());
        measure(sortNames[k], () -> {
          for (SortField sf : ss) {
            sink += s.search(new MatchAllDocsQuery(), 10, new Sort(sf)).scoreDocs[0].doc;
          }
          return ss.size();
        });
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[args.length - 1]);
    if (args[0].equals("build")) {
      if (Files.exists(dir.resolve("geo3d-queries.tsv"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
