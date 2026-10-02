import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LatLonShape;
import org.apache.lucene.document.LatLonShapeDocValuesField;
import org.apache.lucene.document.ShapeField;
import org.apache.lucene.geo.Line;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * Java side of the shape benchmark pair (M9 T9.3); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_geo_shapes.rs}, with the same case names. {@code build <dir>}
 * writes the corpus once -- 200 000 documents, each one {@code LatLonShape} (half polygons, a
 * quarter lines, a quarter points; half of them in twenty clusters) with its shape doc value under
 * the same name, force-merged to one segment -- plus the polygons the indexing cases encode ({@code
 * geo-shapes.tsv}) and the query set both engines replay ({@code geo-shape-queries.tsv}).
 *
 * <p>Cases: {@code shape_index_fields} (tessellate and encode a polygon's triangle fields, per
 * polygon), {@code shape_index_doc_value} (tessellate and build its doc value), {@code
 * shape_intersects_polygon}, {@code shape_within}, {@code shape_contains_point} and {@code
 * shape_doc_values_box} (per query, every hit through a counting collector). Each prints a {@code
 * #check} digest the report compares before it shows a ratio.
 */
public final class GeoShapesMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int DOCS = 200_000;

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

  static double clampLon(double v) {
    return Math.max(-180, Math.min(180, v));
  }

  /** A star-shaped ring around a centre, closed; simple by construction. */
  static double[][] ring(Random r, double clat, double clon, double radius, int n) {
    double[] angles = new double[n];
    for (int i = 0; i < n; i++) angles[i] = r.nextDouble() * 2 * Math.PI;
    java.util.Arrays.sort(angles);
    double[] lats = new double[n + 1];
    double[] lons = new double[n + 1];
    for (int i = 0; i < n; i++) {
      double rad = radius * (0.4 + 0.6 * r.nextDouble());
      lats[i] = clampLat(clat + rad * Math.sin(angles[i]));
      lons[i] = clampLon(clon + rad * Math.cos(angles[i]));
    }
    lats[n] = lats[0];
    lons[n] = lons[0];
    return new double[][] {lats, lons};
  }

  static String ringSpec(double[][] ring) {
    StringBuilder sb = new StringBuilder();
    for (int k = 0; k < ring[0].length; k++) {
      if (k > 0) sb.append(';');
      sb.append(ring[0][k]).append(' ').append(ring[1][k]);
    }
    return sb.toString();
  }

  static Polygon polygon(String spec) {
    String[] rings = spec.split("\\|");
    Polygon[] holes = new Polygon[rings.length - 1];
    for (int i = 1; i < rings.length; i++) {
      double[][] h = parse(rings[i]);
      holes[i - 1] = new Polygon(h[0], h[1]);
    }
    double[][] shell = parse(rings[0]);
    return new Polygon(shell[0], shell[1], holes);
  }

  static double[][] parse(String ring) {
    String[] pts = ring.split(";");
    double[] a = new double[pts.length];
    double[] b = new double[pts.length];
    for (int i = 0; i < pts.length; i++) {
      String[] p = pts[i].split(" ");
      a[i] = Double.parseDouble(p[0]);
      b[i] = Double.parseDouble(p[1]);
    }
    return new double[][] {a, b};
  }

  /** The polygon's triangle fields, or null if Lucene cannot tessellate it. */
  static Field[] tessellates(Polygon p) {
    try {
      return LatLonShape.createIndexableFields("s", p);
    } catch (IllegalArgumentException e) {
      return null;
    }
  }

  static double[] center(Random r, double[][] clusters) {
    if (r.nextBoolean()) return new double[] {r.nextDouble() * 160 - 80, r.nextDouble() * 340 - 170};
    double[] c = clusters[r.nextInt(clusters.length)];
    return new double[] {clampLat(c[0] + r.nextGaussian()), clampLon(c[1] + r.nextGaussian())};
  }

  static void build(Path dir) throws IOException {
    Random r = new Random(0x5A4_9E3L);
    double[][] clusters = new double[20][];
    for (int i = 0; i < clusters.length; i++) {
      clusters[i] = new double[] {r.nextDouble() * 140 - 70, r.nextDouble() * 340 - 170};
    }
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setRAMBufferSizeMB(512);
    try (Directory d = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(d, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        double[] c = center(r, clusters);
        Document doc = new Document();
        int kind = r.nextInt(4);
        if (kind < 2) {
          Field[] fs;
          Polygon p;
          do {
            double[][] rg = ring(r, c[0], c[1], 0.01 + r.nextDouble() * 0.3, 5 + r.nextInt(25));
            p = new Polygon(rg[0], rg[1]);
            fs = tessellates(p);
          } while (fs == null);
          for (Field f : fs) doc.add(f);
          doc.add(LatLonShape.createDocValueField("s", p));
        } else if (kind == 2) {
          int n = 2 + r.nextInt(5);
          double[] lats = new double[n];
          double[] lons = new double[n];
          for (int k = 0; k < n; k++) {
            lats[k] = clampLat(c[0] + r.nextGaussian() * 0.1);
            lons[k] = clampLon(c[1] + r.nextGaussian() * 0.1);
          }
          Line l = new Line(lats, lons);
          for (Field f : LatLonShape.createIndexableFields("s", l)) doc.add(f);
          doc.add(LatLonShape.createDocValueField("s", l));
        } else {
          for (Field f : LatLonShape.createIndexableFields("s", c[0], c[1])) doc.add(f);
          doc.add(LatLonShape.createDocValueField("s", c[0], c[1]));
        }
        w.addDocument(doc);
      }
      w.forceMerge(1);
    }
    StringBuilder shapes = new StringBuilder();
    for (int i = 0; i < 500; i++) {
      double[] c = center(r, clusters);
      double rad = 0.05 + r.nextDouble() * 2;
      String spec = ringSpec(ring(r, c[0], c[1], rad, 8 + r.nextInt(80)));
      if (i % 4 == 0) spec += "|" + ringSpec(ring(r, c[0], c[1], rad * 0.2, 4 + r.nextInt(10)));
      if (tessellates(polygon(spec)) == null) continue;
      shapes.append(spec).append('\n');
    }
    Files.writeString(dir.resolve("geo-shapes.tsv"), shapes.toString());
    StringBuilder q = new StringBuilder();
    for (int i = 0; i < 20; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      q.append("intersects\t").append(ringSpec(ring(r, c[0], c[1], 0.2 + r.nextDouble() * 2, 8 + r.nextInt(40)))).append('\n');
    }
    for (int i = 0; i < 20; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      q.append("within\t").append(ringSpec(ring(r, c[0], c[1], 0.2 + r.nextDouble() * 2, 8 + r.nextInt(40)))).append('\n');
    }
    for (int i = 0; i < 40; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      q.append("contains\t").append(clampLat(c[0] + r.nextGaussian() * 0.5)).append('\t')
          .append(clampLon(c[1] + r.nextGaussian() * 0.5)).append('\n');
    }
    for (int i = 0; i < 20; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      double h = 0.05 + r.nextDouble();
      double w = 0.05 + r.nextDouble();
      q.append("dvbox\t").append(clampLat(c[0] - h)).append('\t').append(clampLat(c[0] + h)).append('\t')
          .append(clampLon(c[1] - w)).append('\t').append(clampLon(c[1] + w)).append('\n');
    }
    Files.writeString(dir.resolve("geo-shape-queries.tsv"), q.toString());
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
      if (Files.exists(dir.resolve("geo-shape-queries.tsv"))) return;
      Files.createDirectories(dir);
      build(dir);
      return;
    }
    Path dir = Path.of(args[1]);
    List<Polygon> polys = new ArrayList<>();
    for (String l : Files.readAllLines(dir.resolve("geo-shapes.tsv"))) polys.add(polygon(l));
    Fnv f = new Fnv();
    for (Polygon p : polys) {
      for (Field t : LatLonShape.createIndexableFields("s", p)) {
        BytesRef b = t.binaryValue();
        for (int i = 0; i < b.length; i++) f.add(b.bytes[b.offset + i]);
      }
    }
    check("shape_index_fields", f, polys.size());
    measure("shape_index_fields", () -> {
      long n = 0;
      for (Polygon p : polys) n += LatLonShape.createIndexableFields("s", p).length;
      sink += n;
      return polys.size();
    });
    f = new Fnv();
    for (Polygon p : polys) {
      BytesRef b = LatLonShape.createDocValueField("s", p).binaryValue();
      for (int i = 0; i < b.length; i++) f.add(b.bytes[b.offset + i]);
    }
    check("shape_index_doc_value", f, polys.size());
    measure("shape_index_doc_value", () -> {
      long n = 0;
      for (Polygon p : polys) n += LatLonShape.createDocValueField("s", p).binaryValue().length;
      sink += n;
      return polys.size();
    });

    List<Query> intersects = new ArrayList<>();
    List<Query> within = new ArrayList<>();
    List<Query> contains = new ArrayList<>();
    List<Query> dvbox = new ArrayList<>();
    for (String l : Files.readAllLines(dir.resolve("geo-shape-queries.tsv"))) {
      String[] a = l.split("\t");
      switch (a[0]) {
        case "intersects" -> intersects.add(LatLonShape.newPolygonQuery("s", ShapeField.QueryRelation.INTERSECTS, polygon(a[1])));
        case "within" -> within.add(LatLonShape.newPolygonQuery("s", ShapeField.QueryRelation.WITHIN, polygon(a[1])));
        case "contains" -> contains.add(LatLonShape.newPointQuery("s", ShapeField.QueryRelation.CONTAINS, new double[] {d(a[1]), d(a[2])}));
        case "dvbox" -> dvbox.add(LatLonShape.newSlowDocValuesBoxQuery("s", ShapeField.QueryRelation.INTERSECTS, d(a[1]), d(a[2]), d(a[3]), d(a[4])));
        default -> throw new IllegalStateException(a[0]);
      }
    }
    try (Directory d = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(d)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      String[] names = {"shape_intersects_polygon", "shape_within", "shape_contains_point", "shape_doc_values_box"};
      List<List<Query>> sets = List.of(intersects, within, contains, dvbox);
      for (int k = 0; k < names.length; k++) {
        List<Query> qs = sets.get(k);
        Fnv g = new Fnv();
        for (Query q : qs) g.add(count(s, q));
        check(names[k], g, qs.size());
        measure(names[k], () -> {
          long n = 0;
          for (Query q : qs) n += count(s, q);
          sink += n;
          return qs.size();
        });
      }
    }
  }
}
