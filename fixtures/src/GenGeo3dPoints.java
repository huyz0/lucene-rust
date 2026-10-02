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
import org.apache.lucene.document.StringField;
import org.apache.lucene.geo.Polygon;
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
import org.apache.lucene.spatial3d.Geo3DDocValuesField;
import org.apache.lucene.spatial3d.Geo3DPoint;
import org.apache.lucene.spatial3d.geom.GeoPoint;
import org.apache.lucene.spatial3d.geom.PlanetModel;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * spatial3d's field, query and sorts (M9 T9.4), differentially: indexes a seeded corpus of {@link
 * Geo3DPoint}/{@link Geo3DDocValuesField} points -- the poles, the dateline, exact duplicates,
 * clusters, multi-valued documents, documents without the field, several segments with deletions,
 * on WGS84 ({@code p}, with the first point's doc value in {@code pd}) and on the sphere ({@code
 * s}, one point and doc value in every document of the first three segments) -- and records Lucene's answer to many queries of every factory ({@code
 * newDistanceQuery}, {@code newBoxQuery}, {@code newPolygonQuery}, {@code newLargePolygonQuery},
 * {@code newPathQuery}) and every sort ({@code newDistanceSort}, {@code newPathSort} and the five
 * {@code newOutside*Sort}s), plain and filtered.
 *
 * <p>Outputs, under {@code geo3d_points/}: {@code index/}, {@code docs.tsv} ({@code id} then {@code
 * field:lat,lon} per point, {@code Double.toString}), {@code deletes.tsv}, and {@code queries.tsv}:
 * the query's tokens, {@code =>}, then {@code C total hexbits} (constant-score hits; bit d is global
 * doc d), {@code T total doc:doublebits,...} (sorted top-n, the sort value in meters) or {@code E
 * class message}. {@code big/}: one segment of {@value #BIG_DOCS} points, latitude rising with the
 * doc id, so a sort from a pole moves its bottom on almost every document -- past the comparator's
 * 1024-update sampling of its bounds. Polygons are {@link GeoCorpus} specs; paths are {@code
 * lat;lat;...} and {@code lon;lon;...}.
 *
 * <p>Run with the trig intrinsics off, like {@code GenGeo3d} ({@code gen-fixtures.sh}).
 */
public class GenGeo3dPoints {
  static final int SEGMENTS = 3;
  static final int DOCS_PER_SEGMENT = 1500;
  static final int SMALL_SEGMENT = 50;
  static final int BIG_DOCS = 12_000;
  static final double RADIANS_PER_DEGREE = Math.PI / 180.0;

  static final double[][] CLUSTERS = {
    {0, 0}, {89.5, 10}, {-89.7, -170}, {10, 179.9}, {-30, -179.95}, {45.5, 7.25}, {40.7, -74.0},
  };

  static PlanetModel model(String field) {
    return field.equals("s") ? PlanetModel.SPHERE : PlanetModel.WGS84;
  }

  static String d(double v) {
    return Double.toString(v);
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
    switch (r.nextInt(12)) {
      case 0:
        p = new double[] {r.nextBoolean() ? 90 : -90, r.nextDouble() * 360 - 180};
        break;
      case 1:
        p = new double[] {r.nextDouble() * 180 - 90, r.nextBoolean() ? 180 : -180};
        break;
      case 2:
      case 3:
      case 4:
      case 5:
        {
          double[] c = CLUSTERS[r.nextInt(CLUSTERS.length)];
          double s = r.nextInt(3) == 0 ? 0.01 : 0.5;
          p = new double[] {clampLat(c[0] + r.nextGaussian() * s), wrapLon(c[1] + r.nextGaussian() * s)};
          break;
        }
      case 6:
        if (!SEEN.isEmpty()) {
          p = SEEN.get(r.nextInt(SEEN.size())).clone();
          break;
        }
        // fall through
      default:
        p = new double[] {r.nextDouble() * 180 - 90, r.nextDouble() * 360 - 180};
    }
    SEEN.add(p);
    return p;
  }

  static List<String> docSpecs(Random r, boolean small) {
    List<String> specs = new ArrayList<>();
    int np = r.nextInt(5) == 0 ? 0 : 1 + r.nextInt(r.nextInt(4) == 0 ? 3 : 1);
    for (int i = 0; i < np; i++) {
      double[] p = latLon(r);
      specs.add("p:" + d(p[0]) + "," + d(p[1]));
      if (i == 0) {
        // The doc value of the first point only: Lucene 10.5's geo3d
        // comparators re-read a multi-valued document's values in copy()
        // after compareBottom() consumed them (they lack
        // LatLonPointDistanceComparator's valuesDocID cache), reading the
        // next documents' values or past the segment's end. The sorts run
        // on single-valued fields, where that cannot happen.
        specs.add("pd:" + d(p[0]) + "," + d(p[1]));
      }
    }
    if (!small) {
      double[] p = latLon(r);
      specs.add("s:" + d(p[0]) + "," + d(p[1]));
    }
    return specs;
  }

  /** {@code p}: a point; {@code pd}: a doc value; {@code s}: both, on the sphere. */
  static void addFields(Document doc, String spec) {
    String[] kv = spec.split(":", 2);
    String[] ab = kv[1].split(",");
    double lat = Double.parseDouble(ab[0]);
    double lon = Double.parseDouble(ab[1]);
    PlanetModel pm = model(kv[0]);
    if (!kv[0].equals("pd")) {
      doc.add(new Geo3DPoint(kv[0], pm, lat, lon));
    }
    if (!kv[0].equals("p")) {
      GeoPoint g = new GeoPoint(pm, lat * RADIANS_PER_DEGREE, lon * RADIANS_PER_DEGREE);
      doc.add(new Geo3DDocValuesField(kv[0], g, pm));
    }
  }

  /** The point field a sort's filter queries: {@code pd}'s points are {@code p}. */
  static String pointField(String field) {
    return field.equals("pd") ? "p" : field;
  }

  /** A distance in meters on the planet: the sphere's unit radius scales it. */
  static double meters(PlanetModel pm, double m) {
    return m * pm.getMeanRadius() / PlanetModel.WGS84.getMeanRadius();
  }

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

  interface SortSupplier {
    SortField get();
  }

  static String err(Exception e) {
    return "E\t" + e.getClass().getName() + "\t" + GeoCorpus.esc(String.valueOf(e.getMessage()));
  }

  static String constant(IndexSearcher s, QuerySupplier q) throws IOException {
    Query query;
    try {
      query = q.get();
    } catch (RuntimeException e) {
      return err(e);
    }
    int maxDoc = s.getIndexReader().maxDoc();
    TopDocs td;
    try {
      td = s.search(query, maxDoc);
    } catch (RuntimeException e) {
      return err(e);
    }
    for (ScoreDoc sd : td.scoreDocs) {
      if (sd.score != 1f) throw new AssertionError("not constant: " + query + " " + sd.score);
    }
    return "C\t" + td.scoreDocs.length + "\t" + hexBits(maxDoc, td.scoreDocs);
  }

  static String sorted(IndexSearcher s, Query q, SortSupplier sf, int n) throws IOException {
    SortField field;
    try {
      field = sf.get();
    } catch (RuntimeException e) {
      return err(e);
    }
    TopFieldDocs td;
    try {
      td = s.search(q, n, new Sort(field));
    } catch (RuntimeException e) {
      return err(e);
    }
    StringBuilder sb = new StringBuilder("T\t").append(td.totalHits.value()).append('\t');
    for (int i = 0; i < td.scoreDocs.length; i++) {
      FieldDoc fd = (FieldDoc) td.scoreDocs[i];
      if (i > 0) sb.append(',');
      sb.append(fd.doc).append(':').append(Long.toHexString(Double.doubleToRawLongBits((Double) fd.fields[0])));
    }
    return sb.toString();
  }

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
        return 1.9e7 + r.nextDouble() * 1e6; // most of the globe
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
    switch (r.nextInt(9)) {
      case 0:
        return new double[] {-90, 90, -180, 180};
      case 1:
        return new double[] {minLat, maxLat, 170 + r.nextDouble() * 10, -180 + r.nextDouble() * 10};
      case 2:
        return new double[] {80 + r.nextDouble() * 10, 90, -180, 180};
      case 3:
        return new double[] {minLat, minLat, minLon, maxLon};
      default:
        return new double[] {minLat, maxLat, minLon, maxLon};
    }
  }

  static Polygon polygon(Random r) {
    for (int attempt = 0; ; attempt++) {
      try {
        if (r.nextInt(4) == 0) {
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

  static String polySpec(Polygon[] ps) {
    StringBuilder sb = new StringBuilder();
    for (Polygon p : ps) {
      if (sb.length() > 0) sb.append(" + ");
      sb.append(GeoCorpus.spec(p));
    }
    return sb.toString();
  }

  static double[][] path(Random r) {
    double[] c = center(r);
    int n = 1 + r.nextInt(6);
    double[] lats = new double[n], lons = new double[n];
    double la = c[0], lo = c[1];
    double step = Math.pow(10, -2 + r.nextDouble() * 2);
    for (int i = 0; i < n; i++) {
      lats[i] = la;
      lons[i] = lo;
      la = clampLat(la + (r.nextDouble() - 0.5) * step * 2);
      lo = wrapLon(lo + (r.nextDouble() - 0.5) * step * 2);
    }
    return new double[][] {lats, lons};
  }

  static String list(double[] a) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < a.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(d(a[i]));
    }
    return sb.toString();
  }

  static void clean(Path root) throws IOException {
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) Files.delete(p);
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]).resolve("geo3d_points");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(0x3D_2026_1002L);
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
            List<String> specs = docSpecs(r, small);
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
      queries(r, s, q, new String[] {"p", "s"}, new String[] {"pd", "s"}, 1);
    }
    Files.writeString(root.resolve("queries.tsv"), q.toString());
    big(root.resolve("big"), r);
  }

  /** The query and sort mix, {@code scale} times over. */
  static void queries(
      Random r, IndexSearcher s, StringBuilder q, String[] fields, String[] sortFields, int scale)
      throws IOException {
    for (int i = 0; i < 60 * scale; i++) {
      String field = fields[i % fields.length];
      PlanetModel pm = model(field);
      double[] c = center(r);
      double radius = i == 0 ? -1 : meters(pm, radius(r));
      q.append("dist\t").append(field).append('\t').append(d(c[0])).append('\t').append(d(c[1]))
          .append('\t').append(d(radius)).append("\t=>\t")
          .append(constant(s, () -> Geo3DPoint.newDistanceQuery(field, pm, c[0], c[1], radius)))
          .append('\n');
    }
    for (int i = 0; i < 50 * scale; i++) {
      String field = fields[i % fields.length];
      PlanetModel pm = model(field);
      double[] b = box(r);
      q.append("box\t").append(field).append('\t').append(d(b[0])).append('\t').append(d(b[1]))
          .append('\t').append(d(b[2])).append('\t').append(d(b[3])).append("\t=>\t")
          .append(constant(s, () -> Geo3DPoint.newBoxQuery(field, pm, b[0], b[1], b[2], b[3])))
          .append('\n');
    }
    for (int i = 0; i < 50 * scale; i++) {
      String field = fields[i % fields.length];
      PlanetModel pm = model(field);
      int np = r.nextInt(4) == 0 ? 2 : 1;
      Polygon[] ps = new Polygon[np];
      for (int k = 0; k < np; k++) ps[k] = polygon(r);
      String kind = i % 3 == 2 ? "lpoly" : "poly";
      q.append(kind).append('\t').append(field).append('\t').append(polySpec(ps)).append("\t=>\t")
          .append(constant(s, () -> kind.equals("poly")
              ? Geo3DPoint.newPolygonQuery(field, pm, ps)
              : Geo3DPoint.newLargePolygonQuery(field, pm, ps)))
          .append('\n');
    }
    for (int i = 0; i < 30 * scale; i++) {
      String field = fields[i % fields.length];
      PlanetModel pm = model(field);
      double[][] p = path(r);
      double width = i % 7 == 0 ? 0 : meters(pm, Math.pow(10, 1 + r.nextDouble() * 5));
      q.append("path\t").append(field).append('\t').append(list(p[0])).append('\t').append(list(p[1]))
          .append('\t').append(d(width)).append("\t=>\t")
          .append(constant(s, () -> Geo3DPoint.newPathQuery(field, p[0], p[1], width, pm)))
          .append('\n');
    }
    // Sorts: every factory, over everything or a box.
    for (int i = 0; i < 70 * scale; i++) {
      String field = sortFields[i % sortFields.length];
      PlanetModel pm = model(field);
      int n = new int[] {1, 7, 50, 400, 1500}[i % 5];
      boolean filtered = i % 4 == 3;
      double[] fb = box(r);
      Query base =
          filtered
              ? Geo3DPoint.newBoxQuery(pointField(field), pm, fb[0], fb[1], fb[2], fb[3])
              : new MatchAllDocsQuery();
      String filter = filtered ? "box:" + d(fb[0]) + "," + d(fb[1]) + "," + d(fb[2]) + "," + d(fb[3]) : "all";
      String spec;
      SortSupplier sf;
      switch (i % 7) {
        case 0: {
          double[] c = center(r);
          double rad = meters(pm, radius(r));
          spec = "dist\t" + d(c[0]) + "\t" + d(c[1]) + "\t" + d(rad);
          sf = () -> Geo3DDocValuesField.newDistanceSort(field, c[0], c[1], rad, pm);
          break;
        }
        case 1: {
          double[][] p = path(r);
          double width = meters(pm, Math.pow(10, 2 + r.nextDouble() * 5));
          spec = "path\t" + list(p[0]) + "\t" + list(p[1]) + "\t" + d(width);
          sf = () -> Geo3DDocValuesField.newPathSort(field, p[0], p[1], width, pm);
          break;
        }
        case 2: {
          double[] c = center(r);
          double rad = meters(pm, radius(r));
          spec = "odist\t" + d(c[0]) + "\t" + d(c[1]) + "\t" + d(rad);
          sf = () -> Geo3DDocValuesField.newOutsideDistanceSort(field, c[0], c[1], rad, pm);
          break;
        }
        case 3: {
          double[] b = box(r);
          spec = "obox\t" + d(b[0]) + "\t" + d(b[1]) + "\t" + d(b[2]) + "\t" + d(b[3]);
          sf = () -> Geo3DDocValuesField.newOutsideBoxSort(field, b[0], b[1], b[2], b[3], pm);
          break;
        }
        case 4: {
          Polygon[] ps = {polygon(r)};
          spec = "opoly\t" + polySpec(ps);
          sf = () -> Geo3DDocValuesField.newOutsidePolygonSort(field, pm, ps);
          break;
        }
        case 5: {
          Polygon[] ps = {polygon(r)};
          spec = "olpoly\t" + polySpec(ps);
          sf = () -> Geo3DDocValuesField.newOutsideLargePolygonSort(field, pm, ps);
          break;
        }
        default: {
          double[][] p = path(r);
          double width = meters(pm, Math.pow(10, 2 + r.nextDouble() * 5));
          spec = "opath\t" + list(p[0]) + "\t" + list(p[1]) + "\t" + d(width);
          sf = () -> Geo3DDocValuesField.newOutsidePathSort(field, p[0], p[1], width, pm);
        }
      }
      q.append("sort\t").append(field).append('\t').append(n).append('\t').append(filter).append('\t')
          .append(spec).append("\t=>\t").append(sorted(s, base, sf, n)).append('\n');
    }
  }

  /** {@code geo3d_points/big/}: one segment of points with rising latitude. */
  static void big(Path dir, Random r) throws IOException {
    Files.createDirectories(dir.resolve("index"));
    StringBuilder out = new StringBuilder();
    try (Directory d = FSDirectory.open(dir.resolve("index"))) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(d, cfg)) {
        for (int i = 0; i < BIG_DOCS; i++) {
          double lon;
          if (r.nextBoolean()) {
            double[] c = CLUSTERS[r.nextInt(CLUSTERS.length)];
            lon = wrapLon(c[1] + r.nextGaussian() * 0.5);
          } else {
            lon = r.nextDouble() * 360 - 180;
          }
          double lat = -89.5 + 179.0 * i / BIG_DOCS + r.nextDouble() * 0.01;
          Document doc = new Document();
          addFields(doc, "p:" + d(lat) + "," + d(lon));
          // Single-valued, so the sorts run on the same field.
          doc.add(new Geo3DDocValuesField("p",
              new GeoPoint(PlanetModel.WGS84, lat * RADIANS_PER_DEGREE, lon * RADIANS_PER_DEGREE),
              PlanetModel.WGS84));
          w.addDocument(doc);
          out.append(d(lat)).append(',').append(d(lon)).append('\n');
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
      PlanetModel pm = PlanetModel.WGS84;
      for (int i = 0; i < 8; i++) {
        double lat = i < 4 ? 90 : -90;
        double lon = r.nextDouble() * 360 - 180;
        int n = new int[] {1, 3, 40, 500}[i % 4];
        double rad = 2.1e7;
        q.append("sort\tp\t").append(n).append("\tall\tdist\t").append(d(lat)).append('\t').append(d(lon))
            .append('\t').append(d(rad)).append("\t=>\t")
            .append(sorted(s, new MatchAllDocsQuery(),
                () -> Geo3DDocValuesField.newDistanceSort("p", lat, lon, rad, pm), n))
            .append('\n');
      }
      queries(r, s, q, new String[] {"p"}, new String[] {"p"}, 1);
    }
    Files.writeString(dir.resolve("queries.tsv"), q.toString());
  }
}
