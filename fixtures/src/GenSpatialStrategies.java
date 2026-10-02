import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.stream.Stream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
 * spatial-extras' strategies (M9 T9.5), differentially: indexes a seeded corpus through every
 * {@code SpatialStrategy} of {@link SpatialExtrasCorpus} -- RPT over geohash, quad (pruned and not),
 * packed quad, S2 (Geo3D) and a planar quad, RPT points-only, the term-query strategy, BBox
 * (geodetic and planar, stored), point-vector, serialized doc values (Spatial4j's and Geo3D's
 * codec), composite (optimized and not) and the date-range strategy -- across four segments with
 * deletions, and records Lucene's answers.
 *
 * <p>Outputs, under {@code spatial_strategies/}: {@code index/}; {@code docs.tsv} ({@code id} then
 * {@code strategy=spec} per indexed shape -- several for a multi-valued field); {@code
 * deletes.tsv}; {@code fields.tsv} (per document line, every field each spec makes, as {@link
 * SpatialExtrasCorpus#describe}: tokens, values, types); {@code queries.tsv}: the question's
 * tokens, {@code =>}, the answer -- {@code C total hexbits} for a query's constant-score hits (bit d
 * is global doc d), {@code V v,v,...} for a value source over every document, {@code H cols rows
 * region counts} for a heatmap, {@code F facets} for date facets, a strategy's {@code toString},
 * or {@code E class message}.
 *
 * <p>Run with the trig intrinsics off, like {@code GenSpatial4j} ({@code gen-fixtures.sh}).
 */
public class GenSpatialStrategies {
  static final int DOCS = 400;
  static final int[] COMMITS = {120, 240, 360};

  static String d(double v) {
    return Double.toString(v);
  }

  static double lon(Random r) {
    switch (r.nextInt(4)) {
      case 0:
        return -180 + 360 * r.nextDouble();
      case 1:
        // near the dateline, either side
        return r.nextBoolean() ? 175 + 5 * r.nextDouble() : -180 + 5 * r.nextDouble();
      default:
        return -20 + 60 * r.nextDouble();
    }
  }

  static double lat(Random r) {
    return r.nextInt(4) == 0 ? -90 + 180 * r.nextDouble() : -10 + 60 * r.nextDouble();
  }

  static double wrap(double x) {
    while (x > 180) x -= 360;
    while (x < -180) x += 360;
    return x;
  }

  static String point(Random r) {
    return "POINT(" + d(lon(r)) + " " + d(lat(r)) + ")";
  }

  /** A geodetic shape's WKT: points, boxes (some across the dateline), circles, collections. */
  static String geo(Random r) {
    int k = r.nextInt(20);
    if (k < 8) return point(r);
    if (k < 13) {
      double minX = lon(r);
      double w = r.nextInt(3) == 0 ? 60 * r.nextDouble() : 8 * r.nextDouble();
      double maxX = wrap(minX + w);
      double minY = Math.max(-90, lat(r) - 5);
      double maxY = Math.min(90, minY + 30 * r.nextDouble());
      return "ENVELOPE(" + d(minX) + ", " + d(maxX) + ", " + d(maxY) + ", " + d(minY) + ")";
    }
    if (k < 17) {
      return "BUFFER(" + point(r) + ", " + d(r.nextInt(3) == 0 ? 15 * r.nextDouble() : 2 * r.nextDouble()) + ")";
    }
    if (k < 19) return "GEOMETRYCOLLECTION(" + point(r) + ", " + point(r) + ")";
    return "ENVELOPE(-180, 180, 90, -90)";
  }

  /** A Geo3D shape's WKT: points, boxes, circles, convex polygons. */
  static String g3(Random r) {
    int k = r.nextInt(10);
    if (k < 3) return point(r);
    if (k < 5) {
      double minX = -20 + 50 * r.nextDouble();
      double minY = -10 + 50 * r.nextDouble();
      return "ENVELOPE(" + d(minX) + ", " + d(minX + 1 + 10 * r.nextDouble()) + ", " + d(minY + 1 + 10 * r.nextDouble()) + ", " + d(minY) + ")";
    }
    if (k < 7) return "BUFFER(" + point(r) + ", " + d(0.2 + 3 * r.nextDouble()) + ")";
    double cx = -20 + 50 * r.nextDouble();
    double cy = -10 + 50 * r.nextDouble();
    int n = 3 + r.nextInt(4);
    double[] angles = new double[n];
    for (int i = 0; i < n; i++) angles[i] = 2 * Math.PI * (i + 0.2 + 0.6 * r.nextDouble()) / n;
    StringBuilder sb = new StringBuilder("POLYGON((");
    String first = null;
    for (int i = 0; i < n; i++) {
      double rad = 0.5 + 4 * r.nextDouble();
      String p = d(cx + rad * Math.cos(angles[i])) + " " + d(cy + rad * Math.sin(angles[i]));
      if (first == null) first = p;
      sb.append(p).append(", ");
    }
    return sb.append(first).append("))").toString();
  }

  /** A planar shape's WKT within +-1000. */
  static String flat(Random r) {
    int k = r.nextInt(10);
    double x = -800 + 1600 * r.nextDouble();
    double y = -800 + 1600 * r.nextDouble();
    if (k < 4) return "POINT(" + d(x) + " " + d(y) + ")";
    if (k < 7) return "ENVELOPE(" + d(x) + ", " + d(x + 150 * r.nextDouble()) + ", " + d(y + 150 * r.nextDouble()) + ", " + d(y) + ")";
    return "BUFFER(POINT(" + d(x) + " " + d(y) + "), " + d(1 + 100 * r.nextDouble()) + ")";
  }

  static String date(Random r) {
    int y = 1995 + r.nextInt(30);
    switch (r.nextInt(5)) {
      case 0:
        return Integer.toString(y);
      case 1:
        return String.format(Locale.ROOT, "%d-%02d", y, 1 + r.nextInt(12));
      case 2:
        return String.format(Locale.ROOT, "%d-%02d-%02d", y, 1 + r.nextInt(12), 1 + r.nextInt(28));
      case 3:
        return String.format(Locale.ROOT, "%d-%02d-%02dT%02d", y, 1 + r.nextInt(12), 1 + r.nextInt(28), r.nextInt(24));
      default:
        return String.format(Locale.ROOT, "%d-%02d-%02dT%02d:%02d:%02d.%03d", y, 1 + r.nextInt(12), 1 + r.nextInt(28), r.nextInt(24), r.nextInt(60), r.nextInt(60), r.nextInt(1000));
    }
  }

  static String dateShape(Random r) {
    if (r.nextInt(3) > 0) return date(r);
    String a = date(r);
    String b = date(r);
    // the tree's order (a prefix first) is the strings' order here
    return a.compareTo(b) <= 0 ? "[" + a + " TO " + b + "]" : "[" + b + " TO " + a + "]";
  }

  static String spec(Random r, String family) {
    switch (family) {
      case "g":
        return "g:" + geo(r);
      case "t":
        return "t:" + g3(r);
      case "f":
        return "f:" + flat(r);
      case "p":
        return "p:" + point(r);
      default:
        return "d:" + dateShape(r);
    }
  }

  static void clean(Path root) throws IOException {
    if (Files.exists(root)) {
      try (Stream<Path> walk = Files.walk(root)) {
        walk.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
      }
    }
  }

  /** One document's line: each strategy's specs (none for about one in eight). */
  static String docLine(Random r, int id) {
    StringBuilder sb = new StringBuilder(Integer.toString(id));
    String geoSpec = spec(r, "g");
    String g3Spec = spec(r, "t");
    String flatSpec = spec(r, "f");
    String pointSpec = spec(r, "p");
    String dateSpec = spec(r, "d");
    for (String[] st : SpatialExtrasCorpus.STRATEGIES) {
      String family = st[1];
      if (family.equals("-") || r.nextInt(8) == 0) continue;
      String sp =
          switch (family) {
            case "g" -> geoSpec;
            case "t" -> g3Spec;
            case "f" -> flatSpec;
            case "p" -> pointSpec;
            default -> dateSpec;
          };
      sb.append('\t').append(st[0]).append('=').append(sp);
      // multi-valued: a second shape now and then (points only where points-only)
      if (r.nextInt(10) == 0 && !st[0].equals("pv") && !st[0].equals("bb") && !st[0].equals("bbf")
          && !st[0].startsWith("sd") && !st[0].equals("cmp")) {
        sb.append('\t').append(st[0]).append('=').append(spec(r, family));
      }
    }
    return sb.toString();
  }

  static String queryShape(Random r, String family) {
    switch (family) {
      case "g":
      case "p":
        {
          int k = r.nextInt(10);
          if (k < 2) return "g:" + point(r);
          if (k < 6) {
            double minX = lon(r);
            double maxX = wrap(minX + 5 + 40 * r.nextDouble());
            double minY = Math.max(-90, lat(r) - 10);
            double maxY = Math.min(90, minY + 5 + 30 * r.nextDouble());
            return "g:ENVELOPE(" + d(minX) + ", " + d(maxX) + ", " + d(maxY) + ", " + d(minY) + ")";
          }
          if (k < 9) return "g:BUFFER(" + point(r) + ", " + d(1 + 20 * r.nextDouble()) + ")";
          return "g:ENVELOPE(-180, 180, 90, -90)";
        }
      default:
        return spec(r, family);
    }
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]).resolve("spatial_strategies");
    clean(root);
    Path indexDir = root.resolve("index");
    Files.createDirectories(indexDir);
    Random r = new Random(0x5E_2026_1002L);
    StringBuilder docsOut = new StringBuilder();
    StringBuilder fieldsOut = new StringBuilder();
    StringBuilder deletesOut = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir)) {
      IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
      cfg.setUseCompoundFile(false);
      cfg.setMergePolicy(NoMergePolicy.INSTANCE);
      cfg.setMaxBufferedDocs(IndexWriterConfig.DISABLE_AUTO_FLUSH);
      cfg.setRAMBufferSizeMB(256);
      try (IndexWriter w = new IndexWriter(dir, cfg)) {
        for (int id = 0; id < DOCS; id++) {
          String line = docLine(r, id);
          for (String f : line.split("\t")) {
            if (f.startsWith("pts=") && indexedPoints.size() < 6 && !indexedPoints.contains(f.substring(4))) {
              indexedPoints.add(f.substring(4));
            }
          }
          docsOut.append(line).append('\n');
          String[] p = line.split("\t");
          StringBuilder fl = new StringBuilder(p[0]);
          for (int i = 1; i < p.length; i++) {
            int eq = p[i].indexOf('=');
            for (Field f : SpatialExtrasCorpus.fields(p[i].substring(0, eq), p[i].substring(eq + 1))) {
              fl.append('\t').append(SpatialExtrasCorpus.describe(f));
            }
          }
          fieldsOut.append(fl).append('\n');
          w.addDocument(SpatialExtrasCorpus.document(line));
          for (int c : COMMITS) if (id + 1 == c) w.commit();
        }
        for (int d = 0; d < DOCS; d += 1 + r.nextInt(15)) {
          w.deleteDocuments(new Term("id", Integer.toString(d)));
          deletesOut.append(d).append('\n');
        }
        w.commit();
      }
    }
    Files.writeString(root.resolve("docs.tsv"), docsOut.toString());
    Files.writeString(root.resolve("fields.tsv"), fieldsOut.toString());
    Files.writeString(root.resolve("deletes.tsv"), deletesOut.toString());

    List<String[]> questions = questions(r);
    StringBuilder q = new StringBuilder();
    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != COMMITS.length + 1) throw new AssertionError("segments");
      IndexSearcher s = new IndexSearcher(reader);
      s.setQueryCache(null);
      List<String> answers = SpatialExtrasCorpus.answerAll(s, questions);
      for (int i = 0; i < questions.size(); i++) {
        q.append(String.join("\t", questions.get(i))).append("\t=>\t").append(answers.get(i)).append('\n');
      }
    }
    Files.writeString(root.resolve("queries.tsv"), q.toString());
  }

  static final String[] OPS = {
    "BBoxIntersects", "BBoxWithin", "Contains", "Intersects", "IsEqualTo", "IsDisjointTo", "IsWithin", "Overlaps"
  };

  /** The first indexed points (their {@code p:} specs), probed exactly by the queries. */
  static final List<String> indexedPoints = new ArrayList<>();

  static List<String[]> questions(Random r) {
    List<String[]> out = new ArrayList<>();
    for (String[] st : SpatialExtrasCorpus.STRATEGIES) {
      String name = st[0];
      String family = st[1].equals("-") ? "g" : st[1];
      out.add(new String[] {"s", name});
      int shapes = name.equals("dr") ? 14 : 8;
      for (int i = 0; i < shapes; i++) {
        String shape = queryShape(r, family);
        for (String op : OPS) {
          String pct = r.nextInt(4) == 0 ? (r.nextBoolean() ? "0.0" : "0.15") : "-";
          out.add(new String[] {"q", name, op, shape, pct});
        }
      }
    }
    // the query shape is a point exactly on an indexed point (RPT's grid-aligned path: for a
    // points-only field, a scored TermQuery)
    for (String p : indexedPoints) {
      for (String name : new String[] {"pts", "rq", "pv", "tq", "bb", "sdv"}) {
        for (String op : new String[] {"Intersects", "IsWithin", "Contains"}) {
          out.add(new String[] {"q", name, op, "g:" + p.substring(2), "-"});
        }
      }
    }
    // boxes at and across the dateline, lines, points and the world: every
    // branch of BBox's query building, and the point-vector's crossing box
    String[] edges = {
      "g:ENVELOPE(170, -170, 10, -10)", "g:ENVELOPE(150, -100, 60, -60)", "g:ENVELOPE(-180, 10, 50, -10)",
      "g:ENVELOPE(10, 180, 50, -10)", "g:ENVELOPE(180, 180, 10, -10)", "g:ENVELOPE(-180, -180, 10, -10)",
      "g:ENVELOPE(-180, 180, 30, -30)", "g:ENVELOPE(10, 10, 40, 0)", "g:ENVELOPE(0, 20, 10, 10)",
      "g:ENVELOPE(5, 5, 20, 20)", "g:ENVELOPE(-180, 180, 90, -90)",
    };
    for (String e : edges) {
      for (String name : new String[] {"bb", "pv", "rq", "sdv", "cmp", "cmpn"}) {
        for (String op : OPS) {
          out.add(new String[] {"q", name, op, e, "-"});
        }
      }
    }
    for (String e : new String[] {"f:ENVELOPE(-300, 300, 300, -300)", "f:ENVELOPE(100, 100, 300, -300)", "f:POINT(0 0)"}) {
      for (String op : OPS) {
        out.add(new String[] {"q", "bbf", op, e, "-"});
        out.add(new String[] {"q", "rfl", op, e, "-"});
      }
    }
    // value sources
    for (String name : new String[] {"pts", "bb", "bbf", "pv", "sdv", "sd3", "rq", "cmp", "dr"}) {
      String family = name.equals("bbf") ? "f" : name.equals("sd3") ? "t" : "g";
      for (int i = 0; i < 3; i++) {
        String p = family.equals("f") ? "f:POINT(" + d(-500 + 1000 * r.nextDouble()) + " " + d(-500 + 1000 * r.nextDouble()) + ")" : family + ":" + point(r);
        out.add(new String[] {"v", name, "dist", p, i == 0 ? "1.0" : "111.19492664455873"});
      }
      out.add(new String[] {"v", name, "recip", queryShape(r, family.equals("f") ? "f" : "g").replace("g:", family + ":")});
      out.add(new String[] {"v", name, "cached", family.equals("f") ? "f:POINT(1.5 -2.5)" : family + ":POINT(1.5 -2.5)", "2.0"});
    }
    for (String name : new String[] {"bb", "bbf"}) {
      String fam = name.equals("bb") ? "g" : "f";
      String[] rects =
          fam.equals("g")
              ? new String[] {"g:ENVELOPE(-10, 30, 40, 0)", "g:ENVELOPE(170, -170, 10, -10)", "g:ENVELOPE(5, 5, 20, 20)", "g:ENVELOPE(0, 20, 10, 10)", "g:ENVELOPE(-180, 180, 90, -90)", "g:ENVELOPE(10, 10, 40, 0)", "g:ENVELOPE(175, 180, 10, -10)"}
              : new String[] {"f:ENVELOPE(-300, 300, 300, -300)", "f:ENVELOPE(0, 0, 0, 0)", "f:ENVELOPE(-100, 400, 0, 0)"};
      for (String rect : rects) {
        out.add(new String[] {"v", name, "overlap", rect, "0.25", "0.0"});
        out.add(new String[] {"v", name, "overlap", rect, "0.5", "0.5"});
        out.add(new String[] {"v", name, "overlap2", rect, "0.75"});
      }
      out.add(new String[] {"v", name, "area", "true", "1.0"});
      out.add(new String[] {"v", name, "area", "false", "2.0"});
    }
    out.add(new String[] {"v", "sdv", "area", "true", "1.0"});
    out.add(new String[] {"v", "sdv", "area", "false", "1.0"});
    out.add(new String[] {"v", "sd3", "area", "true", "1.0"});
    // heatmaps
    String[][] heat = {
      {"rgh", "-", "1", "1000"}, {"rgh", "g:ENVELOPE(-20, 40, 50, -10)", "3", "100000"},
      {"rgh", "g:ENVELOPE(170, -170, 30, -30)", "2", "100000"},
      {"rq", "-", "2", "1000"}, {"rq", "g:ENVELOPE(-20, 40, 50, -10)", "5", "100000"},
      {"rq", "g:ENVELOPE(150, -150, 60, -60)", "4", "100000"}, {"rq", "g:POINT(10 20)", "6", "100000"},
      {"rq", "g:ENVELOPE(-20, 40, 50, -10)", "8", "10"},
      {"rqn", "g:ENVELOPE(-20, 40, 50, -10)", "6", "100000"},
      {"rpq", "g:ENVELOPE(-20, 40, 50, -10)", "6", "100000"}, {"rpq", "-", "3", "100000"},
      {"rfl", "-", "3", "100000"}, {"rfl", "f:ENVELOPE(-500, 200, 300, -100)", "5", "100000"},
      {"pts", "g:ENVELOPE(-20, 40, 50, -10)", "6", "100000"},
      {"cmp", "-", "2", "1000"}, {"rs2", "-", "2", "1000"}, {"dr", "-", "2", "1000"},
    };
    for (String[] h : heat) {
      String name = h[0].equals("cmp") ? "rq" : h[0];
      out.add(new String[] {"h", name, h[1], h[2], h[3], "all"});
      out.add(new String[] {"h", name, h[1], h[2], h[3], "mask"});
    }
    // date facets
    String[][] facets = {
      {"d:2000", "d:2010"}, {"d:2014-03", "d:2014-10"}, {"d:2005-01-01", "d:2005-12-31"},
      {"d:1995", "d:2025"}, {"d:2010-06-15T00", "d:2010-06-20T00"},
    };
    for (String[] f : facets) {
      out.add(new String[] {"f", "dr", f[0], f[1], "all"});
      out.add(new String[] {"f", "dr", f[0], f[1], "mask"});
    }
    out.add(new String[] {"fr", "dr", "d:[2000 TO 2020]", "3"});
    out.add(new String[] {"fr", "dr", "d:[2010-01 TO 2010-06]", "5"});
    out.add(new String[] {"fr", "dr", "d:*", "1"});
    return out;
  }
}
