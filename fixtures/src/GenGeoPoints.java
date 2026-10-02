import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LatLonDocValuesField;
import org.apache.lucene.document.LatLonPoint;
import org.apache.lucene.document.ShapeField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.XYDocValuesField;
import org.apache.lucene.document.XYPointField;
import org.apache.lucene.geo.Circle;
import org.apache.lucene.geo.GeoEncodingUtils;
import org.apache.lucene.geo.LatLonGeometry;
import org.apache.lucene.geo.Line;
import org.apache.lucene.geo.Point;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.Rectangle;
import org.apache.lucene.geo.XYCircle;
import org.apache.lucene.geo.XYGeometry;
import org.apache.lucene.geo.XYPoint;
import org.apache.lucene.geo.XYPolygon;
import org.apache.lucene.geo.XYRectangle;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.FieldDoc;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopFieldDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * The geo point fields, queries, sorts and nearest-neighbour search (M9 T9.2), differentially:
 * indexes a seeded random corpus of {@link LatLonPoint}/{@link LatLonDocValuesField} and {@link
 * XYPointField}/{@link XYDocValuesField} points -- the poles, the dateline, exact duplicates,
 * dense clusters, multi-valued documents, documents without the field, several segments with
 * deletions -- and records, for many random queries of every kind, Lucene's exact answer.
 *
 * <p>Fields: {@code ll} (0-3 lat/lon points per document, point and doc values), {@code one}
 * (exactly one lat/lon point in every document of the first three segments, so the dense and
 * inverse scorers run), {@code xy} (0-2 cartesian points); {@code id} is the delete key.
 *
 * <p>Outputs, under {@code geo_points/}: {@code index/} (Lucene's index), {@code docs.tsv} ({@code
 * id} then one {@code field:a,b} spec per point, numbers in {@code Double.toString} /
 * {@code Float.toString}), {@code deletes.tsv} (ids deleted after the last commit), and {@code
 * queries.tsv}: the query's tokens, a {@code =>} token, then the answer --
 *
 * <pre>
 *   C total hexbits          constant-score hits: bit d of the hex byte string is global doc d
 *   S total relation hits    scored top-n: doc:floatbits,...
 *   T total hits             sorted top-n: doc:doublebits,...
 *   N total hits             LatLonPoint.nearest: doc:doublebits,...
 *   E message                the query threw IllegalArgumentException
 * </pre>
 *
 * Geometries are {@link GeoCorpus} specs.
 */
public class GenGeoPoints {
  static final int SEGMENTS = 3;
  static final int DOCS_PER_SEGMENT = 2000;
  static final int SMALL_SEGMENT = 60;

  static final double[][] CLUSTERS = {
    {0, 0}, {89.5, 10}, {-89.7, -170}, {10, 179.9}, {-30, -179.95}, {45.5, 7.25}, {40.7, -74.0},
  };

  static String d(double v) {
    return Double.toString(v);
  }

  static String f(float v) {
    return Float.toString(v);
  }

  static double clampLat(double v) {
    return Math.max(-90, Math.min(90, v));
  }

  static double wrapLon(double v) {
    while (v > 180) v -= 360;
    while (v < -180) v += 360;
    return v;
  }

  static final List<double[]> SEEN = new ArrayList<>();

  static double[] latLon(Random r) {
    double[] p;
    switch (r.nextInt(14)) {
      case 0:
        p = new double[] {r.nextBoolean() ? 90 : -90, r.nextDouble() * 360 - 180};
        break;
      case 1:
        p = new double[] {r.nextDouble() * 180 - 90, r.nextBoolean() ? 180 : -180};
        break;
      case 2:
        p = new double[] {r.nextDouble() * 160 - 80, r.nextBoolean() ? 179 + r.nextDouble() : -179 - r.nextDouble()};
        break;
      case 3:
      case 4:
      case 5:
      case 6:
        {
          double[] c = CLUSTERS[r.nextInt(CLUSTERS.length)];
          double s = r.nextInt(3) == 0 ? 0.01 : 0.5;
          p = new double[] {clampLat(c[0] + r.nextGaussian() * s), wrapLon(c[1] + r.nextGaussian() * s)};
          break;
        }
      case 7:
        if (!SEEN.isEmpty()) {
          p = SEEN.get(r.nextInt(SEEN.size())).clone();
          break;
        }
        // fall through
      case 8:
        {
          // exactly on an encoded value
          double lat = GeoEncodingUtils.decodeLatitude(r.nextInt());
          double lon = GeoEncodingUtils.decodeLongitude(r.nextInt());
          p = new double[] {lat, lon};
          break;
        }
      default:
        p = new double[] {r.nextDouble() * 180 - 90, r.nextDouble() * 360 - 180};
    }
    SEEN.add(p);
    return p;
  }

  static final float[][] XY_CLUSTERS = {{0, 0}, {1000, -1000}, {-3.5e6f, 2e6f}, {1e-3f, 1e-3f}};

  static float[] xy(Random r) {
    switch (r.nextInt(6)) {
      case 0:
      case 1:
        {
          float[] c = XY_CLUSTERS[r.nextInt(XY_CLUSTERS.length)];
          double s = Math.max(1e-4, Math.abs(c[0]) * 1e-3 + 1);
          return new float[] {(float) (c[0] + r.nextGaussian() * s), (float) (c[1] + r.nextGaussian() * s)};
        }
      case 2:
        return new float[] {(float) (r.nextDouble() * 2e7 - 1e7), (float) (r.nextDouble() * 2e7 - 1e7)};
      case 3:
        return new float[] {r.nextInt(21) - 10, r.nextInt(21) - 10};
      default:
        return new float[] {(float) (r.nextDouble() * 2000 - 1000), (float) (r.nextDouble() * 2000 - 1000)};
    }
  }

  static List<String> docSpecs(Random r, int id, boolean small) {
    List<String> specs = new ArrayList<>();
    int nll = r.nextInt(5) == 0 ? 0 : 1 + r.nextInt(r.nextInt(4) == 0 ? 3 : 1);
    for (int i = 0; i < nll; i++) {
      double[] p = latLon(r);
      specs.add("ll:" + d(p[0]) + "," + d(p[1]));
      // a doc-local duplicate now and then
      if (r.nextInt(25) == 0) specs.add("ll:" + d(p[0]) + "," + d(p[1]));
    }
    if (!small) {
      double[] p = latLon(r);
      specs.add("one:" + d(p[0]) + "," + d(p[1]));
      int nxy = r.nextInt(4) == 0 ? 0 : 1 + r.nextInt(2);
      for (int i = 0; i < nxy; i++) {
        float[] q = xy(r);
        specs.add("xy:" + f(q[0]) + "," + f(q[1]));
      }
    }
    return specs;
  }

  static void addFields(Document doc, String spec) {
    String[] kv = spec.split(":", 2);
    String[] ab = kv[1].split(",");
    if (kv[0].equals("xy")) {
      float x = Float.parseFloat(ab[0]);
      float y = Float.parseFloat(ab[1]);
      doc.add(new XYPointField("xy", x, y));
      doc.add(new XYDocValuesField("xy", x, y));
    } else {
      double lat = Double.parseDouble(ab[0]);
      double lon = Double.parseDouble(ab[1]);
      doc.add(new LatLonPoint(kv[0], lat, lon));
      doc.add(new LatLonDocValuesField(kv[0], lat, lon));
    }
  }

  // --- answers ------------------------------------------------------------------------------

  static String hexBits(int maxDoc, ScoreDoc[] docs) {
    byte[] b = new byte[(maxDoc + 7) / 8];
    for (ScoreDoc sd : docs) b[sd.doc >> 3] |= (byte) (1 << (sd.doc & 7));
    int end = b.length;
    while (end > 0 && b[end - 1] == 0) end--;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < end; i++) sb.append(String.format("%02x", b[i] & 0xff));
    return sb.toString();
  }

  interface QuerySupplier {
    Query get();
  }

  static String constant(IndexSearcher s, QuerySupplier q) throws IOException {
    Query query;
    try {
      query = q.get();
    } catch (IllegalArgumentException e) {
      return "E\t" + e.getMessage();
    }
    int maxDoc = s.getIndexReader().maxDoc();
    TopDocs td = s.search(query, maxDoc);
    for (ScoreDoc sd : td.scoreDocs) {
      if (sd.score != 1f) throw new AssertionError("not constant: " + query + " " + sd.score);
    }
    return "C\t" + td.scoreDocs.length + "\t" + hexBits(maxDoc, td.scoreDocs);
  }

  static String scored(IndexSearcher s, Query q, int n) throws IOException {
    TopDocs td = s.search(q, n);
    StringBuilder sb = new StringBuilder("S\t").append(td.totalHits.value()).append('\t').append(td.totalHits.relation()).append('\t');
    for (int i = 0; i < td.scoreDocs.length; i++) {
      if (i > 0) sb.append(',');
      sb.append(td.scoreDocs[i].doc).append(':').append(Integer.toHexString(Float.floatToRawIntBits(td.scoreDocs[i].score)));
    }
    return sb.toString();
  }

  static String sorted(IndexSearcher s, Query q, SortField sf, int n) throws IOException {
    TopFieldDocs td = s.search(q, n, new Sort(sf));
    StringBuilder sb = new StringBuilder("T\t").append(td.totalHits.value()).append('\t');
    for (int i = 0; i < td.scoreDocs.length; i++) {
      FieldDoc fd = (FieldDoc) td.scoreDocs[i];
      if (i > 0) sb.append(',');
      sb.append(fd.doc).append(':').append(Long.toHexString(Double.doubleToRawLongBits((Double) fd.fields[0])));
    }
    return sb.toString();
  }

  static String nearest(IndexSearcher s, String field, double lat, double lon, int n) throws IOException {
    TopFieldDocs td = LatLonPoint.nearest(s, field, lat, lon, n);
    StringBuilder sb = new StringBuilder("N\t").append(td.totalHits.value()).append('\t');
    for (int i = 0; i < td.scoreDocs.length; i++) {
      FieldDoc fd = (FieldDoc) td.scoreDocs[i];
      if (i > 0) sb.append(',');
      sb.append(fd.doc).append(':').append(Long.toHexString(Double.doubleToRawLongBits((Double) fd.fields[0])));
    }
    return sb.toString();
  }

  // --- random queries ---------------------------------------------------------------------------

  static double[] center(Random r) {
    if (r.nextInt(3) > 0) {
      double[] c = CLUSTERS[r.nextInt(CLUSTERS.length)];
      return new double[] {clampLat(c[0] + r.nextGaussian() * 0.3), wrapLon(c[1] + r.nextGaussian() * 0.3)};
    }
    return GeoCorpus.center(r);
  }

  static double radius(Random r) {
    switch (r.nextInt(8)) {
      case 0:
        return 0;
      case 1:
        return r.nextDouble() * 10;
      case 2:
        return 2.1e7 + r.nextDouble() * 1e6; // past half the globe
      default:
        return Math.pow(10, 2 + r.nextDouble() * 5.5);
    }
  }

  static double[] box(Random r) {
    double[] c = center(r);
    double h = Math.pow(10, -2 + r.nextDouble() * 3);
    double w = Math.pow(10, -2 + r.nextDouble() * 3);
    double minLat = clampLat(c[0] - h), maxLat = clampLat(c[0] + h);
    double minLon = wrapLon(c[1] - w), maxLon = wrapLon(c[1] + w);
    switch (r.nextInt(10)) {
      case 0:
        return new double[] {-90, 90, -180, 180};
      case 1:
        return new double[] {minLat, maxLat, 170 + r.nextDouble() * 10, -180 + r.nextDouble() * 10};
      case 2:
        return new double[] {90, 90, minLon, maxLon};
      case 3:
        return new double[] {minLat, maxLat, 180, r.nextBoolean() ? 180 : -170};
      case 4:
        return new double[] {80 + r.nextDouble() * 10, 90, -180, 180};
      default:
        return new double[] {minLat, maxLat, minLon, maxLon};
    }
  }

  static Polygon polygon(Random r) {
    for (int attempt = 0; ; attempt++) {
      try {
        if (r.nextInt(3) == 0) {
          List<Polygon> ps = GeoCorpus.polygons(r, 1);
          return ps.get(r.nextInt(ps.size()));
        }
        double[] c = center(r);
        double rad = Math.pow(10, -2 + r.nextDouble() * 2.5);
        double[][] ring = GeoCorpus.starRing(r, c[0], c[1], rad, 3 + r.nextInt(30), 0.4, false);
        if (r.nextInt(3) == 0) {
          double[][] hole = GeoCorpus.starRing(r, c[0], c[1], rad * 0.3, 3 + r.nextInt(8), 0.5, false);
          return new Polygon(ring[0], ring[1], new Polygon(hole[0], hole[1]));
        }
        return new Polygon(ring[0], ring[1]);
      } catch (IllegalArgumentException e) {
        if (attempt > 50) throw e;
      }
    }
  }

  static LatLonGeometry geometry(Random r, ShapeField.QueryRelation rel) {
    if (rel == ShapeField.QueryRelation.CONTAINS) {
      if (r.nextInt(5) == 0) return polygon(r); // not a point: no hits / an error
      if (r.nextBoolean() && !SEEN.isEmpty()) {
        double[] p = SEEN.get(r.nextInt(SEEN.size()));
        return new Point(GeoEncodingUtils.decodeLatitude(GeoEncodingUtils.encodeLatitude(p[0])), GeoEncodingUtils.decodeLongitude(GeoEncodingUtils.encodeLongitude(p[1])));
      }
      double[] c = center(r);
      return new Point(c[0], c[1]);
    }
    switch (r.nextInt(6)) {
      case 0:
        {
          double[] c = center(r);
          return new Circle(c[0], c[1], radius(r));
        }
      case 1:
        {
          double[] b = box(r);
          if (b[0] > b[1] || b[0] == 90 || (b[2] == 180 && b[3] == 180)) {
            b = new double[] {-10, 10, -10, 10};
          }
          return new Rectangle(b[0], b[1], b[2], b[3]);
        }
      case 2:
        {
          if (rel == ShapeField.QueryRelation.WITHIN && r.nextInt(4) != 0) return polygon(r);
          List<Line> ls = GeoCorpus.lines(r, 1);
          return ls.get(0);
        }
      case 3:
        {
          double[] c = center(r);
          return new Point(c[0], c[1]);
        }
      default:
        return polygon(r);
    }
  }

  static XYGeometry xyGeometry(Random r) {
    float[] c = XY_CLUSTERS[r.nextInt(XY_CLUSTERS.length)];
    double s = Math.pow(10, r.nextDouble() * 4 - 2) * (Math.abs(c[0]) * 1e-3 + 1);
    switch (r.nextInt(4)) {
      case 0:
        return new XYCircle(c[0] + (float) r.nextGaussian(), c[1] + (float) r.nextGaussian(), (float) (s * 2));
      case 1:
        return new XYRectangle((float) (c[0] - s), (float) (c[0] + s * r.nextDouble()), (float) (c[1] - s * r.nextDouble()), (float) (c[1] + s));
      case 2:
        return new XYPoint(c[0], c[1]);
      default:
        {
          double[][] ring = GeoCorpus.starRing(r, 0, 0, 1, 3 + r.nextInt(20), 0.4, false);
          float[] x = new float[ring[0].length];
          float[] y = new float[ring[0].length];
          for (int i = 0; i < x.length; i++) {
            x[i] = (float) (c[0] + ring[1][i] * s);
            y[i] = (float) (c[1] + ring[0][i] * s);
          }
          return new XYPolygon(x, y);
        }
    }
  }

  static LatLonGeometry[] geometries(Random r, ShapeField.QueryRelation rel) {
    int n = r.nextInt(4) == 0 ? 2 + r.nextInt(3) : 1;
    LatLonGeometry[] g = new LatLonGeometry[n];
    for (int i = 0; i < n; i++) g[i] = geometry(r, rel);
    return g;
  }

  static XYGeometry[] xyGeometries(Random r) {
    int n = r.nextInt(4) == 0 ? 2 + r.nextInt(3) : 1;
    XYGeometry[] g = new XYGeometry[n];
    for (int i = 0; i < n; i++) g[i] = xyGeometry(r);
    return g;
  }

  static final ShapeField.QueryRelation[] RELATIONS = ShapeField.QueryRelation.values();

  static void clean(Path out) throws IOException {
    if (Files.exists(out)) {
      try (Stream<Path> s = Files.walk(out)) {
        s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
    Files.createDirectories(out);
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("geo_points");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(20261002L);
    StringBuilder docsOut = new StringBuilder();
    StringBuilder deletesOut = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        int id = 0;
        for (int seg = 0; seg <= SEGMENTS; seg++) {
          boolean small = seg == SEGMENTS;
          int n = small ? SMALL_SEGMENT : DOCS_PER_SEGMENT;
          for (int i = 0; i < n; i++, id++) {
            List<String> specs = docSpecs(r, id, small);
            Document doc = new Document();
            doc.add(new StringField("id", Integer.toString(id), Field.Store.YES));
            docsOut.append(id);
            for (String s : specs) {
              docsOut.append('\t').append(s);
              addFields(doc, s);
            }
            docsOut.append('\n');
            w.addDocument(doc);
          }
          w.commit();
        }
        for (int d = 0; d < 2 * DOCS_PER_SEGMENT; d += 1 + r.nextInt(25)) {
          w.deleteDocuments(new Term("id", Integer.toString(d)));
          deletesOut.append(d).append('\n');
        }
        w.commit();
      }
    }
    Files.writeString(root.resolve("docs.tsv"), docsOut.toString());
    Files.writeString(root.resolve("deletes.tsv"), deletesOut.toString());

    StringBuilder q = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != SEGMENTS + 1) throw new AssertionError("segments");
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      String[] llFields = {"ll", "one"};

      for (int i = 0; i < 70; i++) {
        String field = llFields[i % 2];
        double[] b = box(r);
        String qa = field + "\t" + d(b[0]) + "\t" + d(b[1]) + "\t" + d(b[2]) + "\t" + d(b[3]);
        q.append("box\t").append(qa).append("\t=>\t")
            .append(constant(s, () -> LatLonPoint.newBoxQuery(field, b[0], b[1], b[2], b[3]))).append('\n');
        q.append("dvbox\t").append(qa).append("\t=>\t")
            .append(constant(s, () -> LatLonDocValuesField.newSlowBoxQuery(field, b[0], b[1], b[2], b[3]))).append('\n');
      }
      for (int i = 0; i < 90; i++) {
        String field = llFields[i % 2];
        double[] c = center(r);
        double rad = radius(r);
        if (i == 0) rad = -1;
        double radius = rad;
        String qa = field + "\t" + d(c[0]) + "\t" + d(c[1]) + "\t" + d(radius);
        q.append("dist\t").append(qa).append("\t=>\t")
            .append(constant(s, () -> LatLonPoint.newDistanceQuery(field, c[0], c[1], radius))).append('\n');
        q.append("dvdist\t").append(qa).append("\t=>\t")
            .append(constant(s, () -> LatLonDocValuesField.newSlowDistanceQuery(field, c[0], c[1], radius))).append('\n');
      }
      for (int i = 0; i < 60; i++) {
        String field = llFields[i % 2];
        int np = r.nextInt(4) == 0 ? 2 : 1;
        Polygon[] ps = new Polygon[np];
        for (int k = 0; k < np; k++) ps[k] = polygon(r);
        String spec = GeoCorpus.spec(ps);
        q.append("poly\t").append(field).append('\t').append(spec).append("\t=>\t")
            .append(constant(s, () -> LatLonPoint.newPolygonQuery(field, ps))).append('\n');
        q.append("dvpoly\t").append(field).append('\t').append(spec).append("\t=>\t")
            .append(constant(s, () -> LatLonDocValuesField.newSlowPolygonQuery(field, ps))).append('\n');
      }
      for (int i = 0; i < 160; i++) {
        String field = llFields[i % 2];
        ShapeField.QueryRelation rel = RELATIONS[(i / 2) % RELATIONS.length];
        LatLonGeometry[] gs = geometries(r, rel);
        String spec = GeoCorpus.spec(gs);
        q.append("geom\t").append(field).append('\t').append(rel).append('\t').append(spec).append("\t=>\t")
            .append(constant(s, () -> LatLonPoint.newGeometryQuery(field, rel, gs))).append('\n');
        q.append("dvgeom\t").append(field).append('\t').append(rel).append('\t').append(spec).append("\t=>\t")
            .append(constant(s, () -> LatLonDocValuesField.newSlowGeometryQuery(field, rel, gs))).append('\n');
      }
      for (int i = 0; i < 80; i++) {
        XYGeometry[] gs = xyGeometries(r);
        String spec = GeoCorpus.spec(gs);
        q.append("xygeom\txy\t").append(spec).append("\t=>\t")
            .append(constant(s, () -> XYPointField.newGeometryQuery("xy", gs))).append('\n');
        q.append("xydvgeom\txy\t").append(spec).append("\t=>\t")
            .append(constant(s, () -> XYDocValuesField.newSlowGeometryQuery("xy", gs))).append('\n');
      }
      for (int i = 0; i < 20; i++) {
        XYGeometry g = xyGeometry(r);
        if (g instanceof XYRectangle rect) {
          String qa = "xy\t" + f(rect.minX) + "\t" + f(rect.maxX) + "\t" + f(rect.minY) + "\t" + f(rect.maxY);
          q.append("xybox\t").append(qa).append("\t=>\t")
              .append(constant(s, () -> XYPointField.newBoxQuery("xy", rect.minX, rect.maxX, rect.minY, rect.maxY))).append('\n');
          q.append("xydvbox\t").append(qa).append("\t=>\t")
              .append(constant(s, () -> XYDocValuesField.newSlowBoxQuery("xy", rect.minX, rect.maxX, rect.minY, rect.maxY))).append('\n');
        } else if (g instanceof XYCircle c) {
          String qa = "xy\t" + f(c.getX()) + "\t" + f(c.getY()) + "\t" + f(c.getRadius());
          q.append("xydist\t").append(qa).append("\t=>\t")
              .append(constant(s, () -> XYPointField.newDistanceQuery("xy", c.getX(), c.getY(), c.getRadius()))).append('\n');
          q.append("xydvdist\t").append(qa).append("\t=>\t")
              .append(constant(s, () -> XYDocValuesField.newSlowDistanceQuery("xy", c.getX(), c.getY(), c.getRadius()))).append('\n');
        }
      }
      // distance feature: scores, with and without pruning
      for (int i = 0; i < 40; i++) {
        String field = llFields[i % 2];
        double[] c = center(r);
        float weight = i % 3 == 0 ? 1f : (float) (0.5 + r.nextDouble() * 3);
        double pivot = Math.pow(10, 1 + r.nextDouble() * 6);
        int n = new int[] {1, 10, 100, 3000}[i % 4];
        Query fq = LatLonPoint.newDistanceFeatureQuery(field, weight, c[0], c[1], pivot);
        q.append("feature\t").append(field).append('\t').append(f(weight)).append('\t').append(d(c[0])).append('\t')
            .append(d(c[1])).append('\t').append(d(pivot)).append('\t').append(n).append("\t=>\t")
            .append(scored(s, fq, n)).append('\n');
      }
      // distance sorts, over every document and under a filter
      for (int i = 0; i < 50; i++) {
        String field = llFields[i % 2];
        double[] c = center(r);
        int n = new int[] {1, 7, 50, 400, 1500}[i % 5];
        boolean filtered = i % 3 == 2;
        double[] b = box(r);
        Query base = filtered ? LatLonPoint.newBoxQuery("ll", b[0], b[1], b[2], b[3]) : new MatchAllDocsQuery();
        String filter = filtered ? "box:" + d(b[0]) + "," + d(b[1]) + "," + d(b[2]) + "," + d(b[3]) : "all";
        q.append("sort\t").append(field).append('\t').append(d(c[0])).append('\t').append(d(c[1])).append('\t')
            .append(n).append('\t').append(filter).append("\t=>\t")
            .append(sorted(s, base, LatLonDocValuesField.newDistanceSort(field, c[0], c[1]), n)).append('\n');
      }
      for (int i = 0; i < 20; i++) {
        float[] c = xy(r);
        int n = new int[] {1, 9, 120, 1500}[i % 4];
        q.append("xysort\txy\t").append(f(c[0])).append('\t').append(f(c[1])).append('\t').append(n).append("\tall\t=>\t")
            .append(sorted(s, new MatchAllDocsQuery(), XYDocValuesField.newDistanceSort("xy", c[0], c[1]), n)).append('\n');
      }
      for (int i = 0; i < 50; i++) {
        String field = llFields[i % 2];
        double[] c = center(r);
        int n = new int[] {1, 3, 20, 300, 1200}[i % 5];
        q.append("nearest\t").append(field).append('\t').append(d(c[0])).append('\t').append(d(c[1])).append('\t')
            .append(n).append("\t=>\t").append(nearest(s, field, c[0], c[1], n)).append('\n');
      }
    }
    Files.writeString(root.resolve("queries.tsv"), q.toString());
    big(root.resolve("big"), r);
  }

  static final int BIG_DOCS = 24_000;

  /**
   * {@code geo_points/big/}: one segment of {@value #BIG_DOCS} documents with one {@code p} point
   * each, latitude rising with the doc id (so a sort from the north pole replaces its bottom on
   * almost every document, past the comparator's 1024-update sampling), half of them in a few
   * clusters. Enough leaves for the distance feature query's iterator narrowing and a deep tree for
   * {@code nearest}. {@code big.tsv} is {@code lat,lon} per document; {@code queries.tsv} has the
   * same answer format as the main fixture's.
   */
  static void big(Path dir, Random r) throws IOException {
    Files.createDirectories(dir.resolve("index"));
    double[][] pts = new double[BIG_DOCS][];
    for (int i = 0; i < BIG_DOCS; i++) {
      double lon;
      if (r.nextBoolean()) {
        double[] c = CLUSTERS[r.nextInt(CLUSTERS.length)];
        lon = wrapLon(c[1] + r.nextGaussian() * 0.5);
      } else {
        lon = r.nextDouble() * 360 - 180;
      }
      double lat = -89.5 + 179.0 * i / BIG_DOCS + r.nextDouble() * 0.01;
      pts[i] = new double[] {lat, lon};
    }
    StringBuilder out = new StringBuilder();
    try (Directory d = FSDirectory.open(dir.resolve("index"))) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(d, cfg)) {
        for (double[] p : pts) {
          Document doc = new Document();
          doc.add(new LatLonPoint("p", p[0], p[1]));
          doc.add(new LatLonDocValuesField("p", p[0], p[1]));
          w.addDocument(doc);
          out.append(d(p[0])).append(',').append(d(p[1])).append('\n');
        }
        w.commit();
      }
    }
    Files.writeString(dir.resolve("big.tsv"), out.toString());
    StringBuilder q = new StringBuilder();
    try (Directory d = FSDirectory.open(dir.resolve("index"));
        DirectoryReader reader = DirectoryReader.open(d)) {
      if (reader.leaves().size() != 1) throw new AssertionError("one segment");
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      for (int i = 0; i < 30; i++) {
        double[] c = i < 3 ? new double[] {89.9, 0} : center(r);
        float weight = i % 3 == 0 ? 1f : (float) (0.5 + r.nextDouble() * 3);
        double pivot = Math.pow(10, 2 + r.nextDouble() * 5);
        int n = new int[] {1, 10, 100}[i % 3];
        Query fq = LatLonPoint.newDistanceFeatureQuery("p", weight, c[0], c[1], pivot);
        q.append("feature\tp\t").append(f(weight)).append('\t').append(d(c[0])).append('\t')
            .append(d(c[1])).append('\t').append(d(pivot)).append('\t').append(n).append("\t=>\t")
            .append(scored(s, fq, n)).append('\n');
      }
      for (int i = 0; i < 12; i++) {
        double[] c = i < 4 ? new double[] {90, r.nextDouble() * 360 - 180} : center(r);
        int n = new int[] {1, 3, 40}[i % 3];
        q.append("sort\tp\t").append(d(c[0])).append('\t').append(d(c[1])).append('\t').append(n)
            .append("\tall\t=>\t")
            .append(sorted(s, new MatchAllDocsQuery(), LatLonDocValuesField.newDistanceSort("p", c[0], c[1]), n))
            .append('\n');
      }
      for (int i = 0; i < 20; i++) {
        double[] c = i < 2 ? new double[] {-90, 0} : center(r);
        int n = new int[] {1, 7, 60, 500}[i % 4];
        q.append("nearest\tp\t").append(d(c[0])).append('\t').append(d(c[1])).append('\t').append(n)
            .append("\t=>\t").append(nearest(s, "p", c[0], c[1], n)).append('\n');
      }
      for (int i = 0; i < 10; i++) {
        double[] c = center(r);
        double radius = Math.pow(10, 5 + r.nextDouble() * 2.5);
        q.append("dist\tp\t").append(d(c[0])).append('\t').append(d(c[1])).append('\t').append(d(radius))
            .append("\t=>\t").append(constant(s, () -> LatLonPoint.newDistanceQuery("p", c[0], c[1], radius)))
            .append('\n');
      }
      for (int i = 0; i < 8; i++) {
        ShapeField.QueryRelation rel = RELATIONS[i % 3];
        Polygon[] ps = {polygon(r)};
        LatLonGeometry[] gs = ps;
        q.append("geom\tp\t").append(rel).append('\t').append(GeoCorpus.spec(gs)).append("\t=>\t")
            .append(constant(s, () -> LatLonPoint.newGeometryQuery("p", rel, gs))).append('\n');
      }
    }
    Files.writeString(dir.resolve("queries.tsv"), q.toString());
  }
}
