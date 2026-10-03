import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.spatial.bbox.BBoxStrategy;
import org.apache.lucene.spatial.prefix.HeatmapFacetCounter;
import org.apache.lucene.spatial.prefix.NumberRangePrefixTreeStrategy;
import org.apache.lucene.spatial.prefix.RecursivePrefixTreeStrategy;
import org.apache.lucene.spatial.prefix.tree.DateRangePrefixTree;
import org.apache.lucene.spatial.prefix.tree.QuadPrefixTree;
import org.apache.lucene.spatial.query.SpatialArgs;
import org.apache.lucene.spatial.query.SpatialOperation;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.locationtech.spatial4j.context.SpatialContext;
import org.locationtech.spatial4j.context.SpatialContextFactory;
import org.locationtech.spatial4j.shape.Rectangle;
import org.locationtech.spatial4j.shape.Shape;

/**
 * Java side of the spatial-extras benchmark pair (M9 T9.5); the Rust side is {@code
 * benchmarks/rust-runner/src/micro_spatial_extras.rs}, with the same case names. {@code build
 * <dir>} writes the corpus once -- 100 000 documents, each with an RPT (quad, 11 levels) shape
 * (mostly clustered points, some small boxes), its BBox, and a date or date range -- force-merged to
 * one segment, and the inputs both engines replay ({@code spatial-queries.tsv}). The cases: RPT
 * indexing a Geo3D polygon (every token consumed), RPT intersects of boxes and circles, BBox
 * intersects with the overlap-ratio similarity of every hit, heatmaps, and date-range queries.
 * Each case prints a {@code #check} digest the report compares first.
 */
public final class SpatialExtrasMicro {
  static long warmupMs = Long.getLong("warmupMs", 1500);
  static long measureMs = Long.getLong("measureMs", 2000);
  static long sink;
  static final int DOCS = 100_000;
  static final SpatialContext GEO = SpatialContext.GEO;
  static final SpatialContext G3 = g3();

  static SpatialContext g3() {
    Map<String, String> m = new LinkedHashMap<>();
    m.put("spatialContextFactory", "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory");
    return SpatialContextFactory.makeSpatialContext(m, SpatialExtrasMicro.class.getClassLoader());
  }

  static final RecursivePrefixTreeStrategy RPT =
      new RecursivePrefixTreeStrategy(new QuadPrefixTree(GEO, 11), "rpt");
  static final RecursivePrefixTreeStrategy RPT3 =
      new RecursivePrefixTreeStrategy(new QuadPrefixTree(G3, 11), "rpt3");
  static final BBoxStrategy BB = BBoxStrategy.newInstance(GEO, "bb");
  static final DateRangePrefixTree DATES = new DateRangePrefixTree(DateRangePrefixTree.DEFAULT_CAL);
  static final NumberRangePrefixTreeStrategy DR = new NumberRangePrefixTreeStrategy(DATES, "dr");

  interface Op {
    long run() throws Exception;
  }

  static void measure(String name, Op op) throws Exception {
    loop(op, warmupMs);
    long start = System.nanoTime();
    long units = loop(op, measureMs);
    long elapsed = System.nanoTime() - start;
    System.out.printf("%s\t%.3f\t%d%n", name, (double) elapsed / units, units);
    System.out.flush();
  }

  static long loop(Op op, long budgetMs) throws Exception {
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

  static String d(double v) {
    return Double.toString(v);
  }

  static double wrapLon(double v) {
    while (v > 180) v -= 360;
    while (v < -180) v += 360;
    return v;
  }

  static double clampLat(double v) {
    return Math.max(-89, Math.min(89, v));
  }

  static String date(Random r) {
    int y = 2000 + r.nextInt(20);
    switch (r.nextInt(3)) {
      case 0:
        return String.format(Locale.ROOT, "%d-%02d", y, 1 + r.nextInt(12));
      case 1:
        return String.format(Locale.ROOT, "%d-%02d-%02d", y, 1 + r.nextInt(12), 1 + r.nextInt(28));
      default:
        return String.format(Locale.ROOT, "%d-%02d-%02dT%02d:%02d", y, 1 + r.nextInt(12), 1 + r.nextInt(28), r.nextInt(24), r.nextInt(60));
    }
  }

  static String range(Random r) {
    String a = date(r), b = date(r);
    return a.compareTo(b) <= 0 ? "[" + a + " TO " + b + "]" : "[" + b + " TO " + a + "]";
  }

  static String rect(double minX, double maxX, double minY, double maxY) {
    return "ENVELOPE(" + d(minX) + ", " + d(maxX) + ", " + d(maxY) + ", " + d(minY) + ")";
  }

  static void build(Path dir) throws Exception {
    Random r = new Random(0x5E_9013L);
    double[][] clusters = new double[20][];
    for (int i = 0; i < clusters.length; i++) {
      clusters[i] = new double[] {r.nextDouble() * 300 - 150, r.nextDouble() * 120 - 60};
    }
    IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
    cfg.setRAMBufferSizeMB(512);
    cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
    try (Directory dd = FSDirectory.open(dir);
        IndexWriter w = new IndexWriter(dd, cfg)) {
      for (int i = 0; i < DOCS; i++) {
        double[] c = clusters[r.nextInt(clusters.length)];
        double x = wrapLon(c[0] + r.nextGaussian() * 3);
        double y = clampLat(c[1] + r.nextGaussian() * 3);
        String wkt =
            r.nextInt(5) == 0
                ? rect(x, Math.min(180, x + 0.01 + r.nextDouble() * 0.5), y, y + 0.01 + r.nextDouble() * 0.5)
                : "POINT(" + d(x) + " " + d(y) + ")";
        Shape s = GEO.getFormats().getWktReader().read(wkt);
        Document doc = new Document();
        for (Field f : RPT.createIndexableFields(s)) doc.add(f);
        for (Field f : BB.createIndexableFields(s)) doc.add(f);
        String dr = r.nextInt(3) == 0 ? range(r) : date(r);
        for (Field f : DR.createIndexableFields(DATES.parseShape(dr))) doc.add(f);
        w.addDocument(doc);
      }
      w.forceMerge(1);
    }
    StringBuilder q = new StringBuilder();
    for (int i = 0; i < 20; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      double cx = c[0] + r.nextGaussian(), cy = clampLat(c[1] + r.nextGaussian());
      int n = 5 + r.nextInt(20);
      double rad = 0.5 + r.nextDouble() * 4;
      StringBuilder p = new StringBuilder("POLYGON((");
      String first = null;
      for (int k = 0; k < n; k++) {
        double a = 2 * Math.PI * k / n;
        double rr = rad * (0.6 + 0.4 * r.nextDouble());
        String pt = d(cx + rr * Math.cos(a)) + " " + d(cy + rr * Math.sin(a));
        if (first == null) first = pt;
        p.append(pt).append(", ");
      }
      q.append("poly\t").append(p.append(first).append("))")).append('\n');
    }
    for (int i = 0; i < 30; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      double w = 0.1 + r.nextDouble() * 6, h = 0.1 + r.nextDouble() * 6;
      double x = c[0] + r.nextGaussian(), y = clampLat(c[1] + r.nextGaussian());
      q.append("rect\t").append(rect(x - w, x + w, clampLat(y - h), clampLat(y + h))).append('\n');
    }
    for (int i = 0; i < 20; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      q.append("circle\tBUFFER(POINT(").append(d(c[0] + r.nextGaussian())).append(' ')
          .append(d(clampLat(c[1] + r.nextGaussian()))).append("), ").append(d(0.1 + r.nextDouble() * 4)).append(")\n");
    }
    for (int i = 0; i < 30; i++) {
      double[] c = clusters[r.nextInt(clusters.length)];
      double w = 0.1 + r.nextDouble() * 5, h = 0.1 + r.nextDouble() * 5;
      double x = c[0] + r.nextGaussian(), y = clampLat(c[1] + r.nextGaussian());
      q.append("bbox\t").append(rect(x - w, x + w, clampLat(y - h), clampLat(y + h))).append('\n');
    }
    q.append("heat\t-\t3\n");
    q.append("heat\t").append(rect(-60, 60, -40, 40)).append("\t6\n");
    q.append("heat\t").append(rect(-10, 30, 0, 30)).append("\t8\n");
    for (int i = 0; i < 30; i++) q.append("date\t").append(range(r)).append('\n');
    Files.writeString(dir.resolve("spatial-queries.tsv"), q.toString());
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

  /** Sums a value source over a query's hits, as a similarity-ranked search reads it. */
  static final class Sum implements org.apache.lucene.search.Collector {
    final DoubleValuesSource src;
    double sum;
    long n;

    Sum(DoubleValuesSource src) {
      this.src = src;
    }

    @Override
    public LeafCollector getLeafCollector(LeafReaderContext context) throws IOException {
      DoubleValues v = src.getValues(context, null);
      return new LeafCollector() {
        @Override
        public void setScorer(Scorable scorer) {}

        @Override
        public void collect(int doc) throws IOException {
          if (v.advanceExact(doc)) sum += v.doubleValue();
          n++;
        }
      };
    }

    @Override
    public ScoreMode scoreMode() {
      return ScoreMode.COMPLETE_NO_SCORES;
    }
  }

  static void run(Path dir) throws Exception {
    List<Shape> polys = new ArrayList<>(), rects = new ArrayList<>(), circles = new ArrayList<>(), boxes = new ArrayList<>();
    List<Shape> heatShapes = new ArrayList<>();
    List<Integer> heatLevels = new ArrayList<>();
    List<Shape> dates = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("spatial-queries.tsv"))) {
      String[] a = line.split("\t");
      switch (a[0]) {
        case "poly" -> polys.add(G3.getFormats().getWktReader().read(a[1]));
        case "rect" -> rects.add(GEO.getFormats().getWktReader().read(a[1]));
        case "circle" -> circles.add(GEO.getFormats().getWktReader().read(a[1]));
        case "bbox" -> boxes.add(GEO.getFormats().getWktReader().read(a[1]));
        case "heat" -> {
          heatShapes.add(a[1].equals("-") ? null : GEO.getFormats().getWktReader().read(a[1]));
          heatLevels.add(Integer.parseInt(a[2]));
        }
        case "date" -> dates.add(DATES.parseShape(a[1]));
        default -> throw new IllegalArgumentException(a[0]);
      }
    }

    // RPT indexing a polygon: every token of its field
    Fnv f = new Fnv();
    for (Shape p : polys) {
      f.add(tokens(p));
    }
    check("spx_rpt_index_polygon", f, polys.size());
    measure("spx_rpt_index_polygon", () -> {
      long n = 0;
      for (Shape p : polys) n += tokens(p);
      sink += n;
      return polys.size();
    });

    try (Directory dd = FSDirectory.open(dir);
        DirectoryReader reader = DirectoryReader.open(dd)) {
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      for (Object[] c : new Object[][] {{"spx_rpt_intersects_rect", rects}, {"spx_rpt_intersects_circle", circles}}) {
        String name = (String) c[0];
        @SuppressWarnings("unchecked")
        List<Shape> shapes = (List<Shape>) c[1];
        List<Query> qs = new ArrayList<>();
        for (Shape sh : shapes) qs.add(RPT.makeQuery(new SpatialArgs(SpatialOperation.Intersects, sh)));
        Fnv g = new Fnv();
        for (Query q : qs) g.add(count(s, q));
        check(name, g, qs.size());
        measure(name, () -> {
          long n = 0;
          for (Query q : qs) n += count(s, q);
          sink += n;
          return qs.size();
        });
      }

      // BBox intersects, every hit's overlap ratio
      List<Query> bq = new ArrayList<>();
      List<DoubleValuesSource> bs = new ArrayList<>();
      for (Shape b : boxes) {
        bq.add(BB.makeQuery(new SpatialArgs(SpatialOperation.BBoxIntersects, b)));
        bs.add(BB.makeOverlapRatioValueSource((Rectangle) b, 0.25));
      }
      Fnv g = new Fnv();
      for (int i = 0; i < bq.size(); i++) {
        Sum sum = new Sum(bs.get(i));
        s.search(bq.get(i), collectorManager(sum));
        g.add(sum.n);
        g.add(Math.round(sum.sum * 1e6));
      }
      check("spx_bbox_similarity", g, bq.size());
      measure("spx_bbox_similarity", () -> {
        for (int i = 0; i < bq.size(); i++) {
          Sum sum = new Sum(bs.get(i));
          s.search(bq.get(i), collectorManager(sum));
          sink += sum.n;
        }
        return bq.size();
      });

      // heatmaps
      Fnv hf = new Fnv();
      for (int i = 0; i < heatShapes.size(); i++) {
        HeatmapFacetCounter.Heatmap hm =
            HeatmapFacetCounter.calcFacets(RPT, reader.getContext(), null, heatShapes.get(i), heatLevels.get(i), 1_000_000);
        hf.add(hm.columns);
        hf.add(hm.rows);
        long total = 0;
        for (int c : hm.counts) total += c;
        hf.add(total);
      }
      check("spx_heatmap", hf, heatShapes.size());
      measure("spx_heatmap", () -> {
        for (int i = 0; i < heatShapes.size(); i++) {
          sink += HeatmapFacetCounter.calcFacets(RPT, reader.getContext(), null, heatShapes.get(i), heatLevels.get(i), 1_000_000).counts.length;
        }
        return heatShapes.size();
      });

      // date ranges
      List<Query> dq = new ArrayList<>();
      for (Shape sh : dates) dq.add(DR.makeQuery(new SpatialArgs(SpatialOperation.Intersects, sh)));
      Fnv df = new Fnv();
      for (Query q : dq) df.add(count(s, q));
      check("spx_date_range", df, dq.size());
      measure("spx_date_range", () -> {
        long n = 0;
        for (Query q : dq) n += count(s, q);
        sink += n;
        return dq.size();
      });
    }
  }

  static CollectorManager<Sum, Void> collectorManager(Sum sum) {
    return new CollectorManager<Sum, Void>() {
      @Override
      public Sum newCollector() {
        return sum;
      }

      @Override
      public Void reduce(java.util.Collection<Sum> cs) {
        return null;
      }
    };
  }

  /** Every token of a polygon's RPT field; the count and a digest of the bytes. */
  static long tokens(Shape p) throws IOException {
    long n = 0;
    long h = 0;
    for (Field f : RPT3.createIndexableFields(p)) {
      try (TokenStream ts = f.tokenStreamValue()) {
        TermToBytesRefAttribute term = ts.addAttribute(TermToBytesRefAttribute.class);
        ts.reset();
        while (ts.incrementToken()) {
          BytesRef b = term.getBytesRef();
          n++;
          h = h * 31 + b.length + (b.length > 0 ? b.bytes[b.offset + b.length - 1] : 0);
        }
        ts.end();
      }
    }
    return n * 1_000_003L + (h & 0xffff);
  }

  public static void main(String[] args) throws Exception {
    Path dir = Path.of(args[1]);
    if (args[0].equals("build")) {
      // once: the queries file is written last, after the force-merge
      if (Files.exists(dir.resolve("spatial-queries.tsv"))) return;
      Files.createDirectories(dir);
      build(dir);
    } else {
      run(dir);
    }
  }
}
