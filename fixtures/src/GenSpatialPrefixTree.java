import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import org.apache.lucene.spatial.prefix.tree.Cell;
import org.apache.lucene.spatial.prefix.tree.CellIterator;
import org.apache.lucene.spatial.prefix.tree.DateRangePrefixTree;
import org.apache.lucene.spatial.prefix.tree.NumberRangePrefixTree;
import org.apache.lucene.spatial.prefix.tree.NumberRangePrefixTree.NRShape;
import org.apache.lucene.spatial.prefix.tree.NumberRangePrefixTree.UnitNRShape;
import org.apache.lucene.spatial.prefix.tree.PackedQuadPrefixTree;
import org.apache.lucene.spatial.prefix.tree.S2PrefixTree;
import org.apache.lucene.spatial.prefix.tree.SpatialPrefixTree;
import org.apache.lucene.spatial.prefix.tree.SpatialPrefixTreeFactory;
import org.apache.lucene.spatial.query.SpatialArgs;
import org.apache.lucene.spatial.query.SpatialArgsParser;
import org.apache.lucene.spatial.query.SpatialOperation;
import org.apache.lucene.util.BytesRef;
import org.locationtech.spatial4j.context.SpatialContext;
import org.locationtech.spatial4j.context.SpatialContextFactory;
import org.locationtech.spatial4j.shape.Rectangle;
import org.locationtech.spatial4j.shape.Shape;

/**
 * Lucene's spatial prefix trees -- geohash, quad, packed quad, S2, and the number-range / date
 * range tree -- and the query arguments, called directly on seeded random inputs for {@code
 * crates/lucene-util/tests/spatial_prefix_tree_fixtures.rs}.
 *
 * <p>Each line is {@code op TAB inputs TAB => TAB result}. A tree is an index into {@link #TREES}
 * (built by {@code SpatialPrefixTreeFactory.makeSPT} over a context of {@link #CTX_ARGS}); shapes
 * are WKT. A cell is printed as its token with the leaf byte (hex), its level, leaf flag and
 * relation. Generated with the trig intrinsics off, like {@code GenGeo3d}.
 */
public class GenSpatialPrefixTree {
  static final Random R = new Random(Long.getLong("spatial.seed", 0x5_9EF1L));
  static final StringBuilder OUT = new StringBuilder();

  static final String[][] CTX_ARGS = {
    {},
    {"geo", "false", "worldBounds", "ENVELOPE(-1000, 1000, 1000, -1000)"},
    {"spatialContextFactory", "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory"},
    {
      "spatialContextFactory", "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory",
      "planetModel", "wgs84"
    },
  };

  /** context index, then makeSPT args. */
  static final String[][] TREES = {
    {"0", "prefixTree", "geohash", "maxLevels", "6"},
    {"0", "prefixTree", "quad", "maxLevels", "12"},
    {"1", "prefixTree", "quad", "maxLevels", "10"},
    {"0", "prefixTree", "packedQuad", "maxLevels", "12"},
    {"0", "prefixTree", "packedQuad", "maxLevels", "10", "noprune", "1"},
    {"2", "prefixTree", "s2", "maxLevels", "5"},
    {"3", "prefixTree", "s2", "maxLevels", "4"},
    {"2", "prefixTree", "quad", "maxLevels", "8"},
    {"0"},
    {"1", "maxDistErr", "0.5"},
    {"0", "prefixTree", "packedQuad"},
    {"2", "prefixTree", "s2", "maxDistErr", "0.01"},
    // the factory only builds arity 1; these use the constructor
    {"2", "s2arity", "2", "maxLevels", "6"},
    {"3", "s2arity", "3", "maxLevels", "3"},
  };

  static SpatialContext[] CTX = new SpatialContext[CTX_ARGS.length];
  static SpatialPrefixTree[] GRID = new SpatialPrefixTree[TREES.length];
  static int[] GRID_CTX = new int[TREES.length];

  static String h(double v) {
    return Long.toHexString(Double.doubleToRawLongBits(v));
  }

  static String esc(String s) {
    return s == null ? "null" : s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  static String hex(BytesRef b) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < b.length; i++) sb.append(String.format("%02x", b.bytes[b.offset + i]));
    return sb.length() == 0 ? "-" : sb.toString();
  }

  static byte[] unhex(String s) {
    if (s.equals("-")) return new byte[0];
    byte[] b = new byte[s.length() / 2];
    for (int i = 0; i < b.length; i++) b[i] = (byte) Integer.parseInt(s.substring(2 * i, 2 * i + 2), 16);
    return b;
  }

  interface Op {
    Object run() throws Exception;
  }

  static String err(Throwable e) {
    String s = "ERR " + e.getClass().getName() + " " + esc(e.getMessage());
    if (e instanceof ParseException) s += " @" + ((ParseException) e).getErrorOffset();
    return s;
  }

  static String safe(Op f) {
    try {
      return String.valueOf(f.run());
    } catch (Exception e) {
      return err(e);
    }
  }

  static void rec(String op, String inputs, Op f) {
    OUT.append(op).append('\t').append(inputs).append("\t=>\t").append(safe(f)).append('\n');
  }

  static String cell(Cell c) {
    return hex(c.getTokenBytesWithLeaf(null)) + "/" + c.getLevel() + "/" + (c.isLeaf() ? "L" : "-")
        + "/" + c.getShapeRel();
  }

  /** Every cell an iterator yields (at most {@code max}), with an order-sensitive hash. */
  static String cells(CellIterator it, int max) {
    StringBuilder sb = new StringBuilder();
    long hash = 0;
    int n = 0;
    while (it.hasNext()) {
      Cell c = it.next();
      String s = cell(c);
      for (int i = 0; i < s.length(); i++) hash = hash * 31 + s.charAt(i);
      if (n < 12) sb.append(' ').append(s);
      if (++n >= max) {
        sb.append(" TRUNC");
        break;
      }
    }
    return n + " " + Long.toHexString(hash) + sb;
  }

  static String bbox(Shape s) {
    Rectangle r = s.getBoundingBox();
    return h(r.getMinX()) + "," + h(r.getMaxX()) + "," + h(r.getMinY()) + "," + h(r.getMaxY());
  }

  static String d(double v) {
    return Double.toString(v);
  }

  static double lon() {
    return R.nextInt(10) == 0 ? (R.nextBoolean() ? 180 : -180) : R.nextDouble() * 360 - 180;
  }

  static double lat() {
    return R.nextInt(10) == 0 ? (R.nextBoolean() ? 90 : -90) : R.nextDouble() * 180 - 90;
  }

  /** A random WKT shape for a context. */
  static String wkt(int c) {
    boolean geo = CTX[c].isGeo();
    double scale = geo ? 1 : 5;
    switch (R.nextInt(c >= 2 ? 6 : 5)) {
      case 0:
        return "POINT(" + d(geo ? lon() : R.nextDouble() * 2000 - 1000) + " " + d(geo ? lat() : R.nextDouble() * 2000 - 1000) + ")";
      case 1:
        {
          double x = geo ? lon() : R.nextDouble() * 1800 - 900;
          double y = geo ? R.nextDouble() * 150 - 80 : R.nextDouble() * 1800 - 900;
          double w = Math.pow(2, -R.nextInt(8)) * 40 * scale;
          double x2 = geo ? x + w : Math.min(1000, x + w);
          if (geo && x2 > 180) x2 -= 360;
          return "ENVELOPE(" + d(x) + ", " + d(x2) + ", " + d(Math.min(geo ? 90 : 1000, y + w / 2)) + ", " + d(y) + ")";
        }
      case 2:
        return "BUFFER(POINT(" + d(geo ? lon() : R.nextDouble() * 1600 - 800) + " " + d(geo ? R.nextDouble() * 160 - 80 : R.nextDouble() * 1600 - 800) + "), "
            + d(Math.pow(2, -R.nextInt(8)) * 20 * scale) + ")";
      case 3:
        {
          double x = geo ? R.nextDouble() * 300 - 150 : R.nextDouble() * 1600 - 800;
          double y = geo ? R.nextDouble() * 120 - 60 : R.nextDouble() * 1600 - 800;
          return "BUFFER(LINESTRING(" + d(x) + " " + d(y) + ", " + d(x + R.nextDouble() * 20 * scale) + " " + d(y + R.nextDouble() * 10 * scale) + "), "
              + d(c >= 2 ? R.nextDouble() * 0.05 : R.nextDouble() * 2 * scale) + ")";
        }
      case 4:
        return "GEOMETRYCOLLECTION(" + "POINT(" + d(geo ? lon() : 10) + " " + d(geo ? lat() : 20) + "), ENVELOPE(" + d(geo ? 10 : 100) + ", " + d(geo ? 12 : 120) + ", " + d(geo ? 5 : 50) + ", " + d(geo ? 3 : 30) + "))";
      default:
        {
          double x = R.nextDouble() * 300 - 150, y = R.nextDouble() * 120 - 60;
          double w = Math.pow(2, -R.nextInt(6)) * 20;
          return "POLYGON((" + d(x) + " " + d(y) + ", " + d(x + w) + " " + d(y) + ", " + d(x + w / 2) + " " + d(y + w) + ", " + d(x) + " " + d(y) + "))";
        }
    }
  }

  static void trees() throws Exception {
    for (int t = 0; t < TREES.length; t++) {
      final int tt = t;
      SpatialPrefixTree g = GRID[t];
      rec("tree", String.valueOf(t), () -> {
        StringBuilder sb = new StringBuilder();
        sb.append(esc(g.toString().replaceAll("@[0-9a-f]+", ""))).append(" | ").append(g.getMaxLevels()).append(" |");
        for (double dd : new double[] {0, 1e-6, 1e-3, 0.01, 0.1, 1, 5, 20, 90, 200, 1000}) {
          sb.append(' ').append(g.getLevelForDistance(dd));
        }
        sb.append(" |");
        for (int l = 0; l <= g.getMaxLevels() + 1; l++) {
          final int ll = l;
          sb.append(' ').append(safe(() -> h(g.getDistanceForLevel(ll))));
        }
        return sb.toString();
      });
      int c = GRID_CTX[t];
      for (int i = 0; i < 70; i++) {
        String w = wkt(c);
        double pct = new double[] {0.025, 0.1, 0.25, 0.5, 0}[R.nextInt(5)];
        int dl = R.nextInt(g.getMaxLevels()) + 1;
        rec("cells", t + "\t" + esc(w) + "\t" + pct + "\t" + dl, () -> {
          Shape s = CTX[c].getFormats().getWktReader().read(w);
          int detail = pct == 0 ? dl : g.getLevelForDistance(SpatialArgs.calcDistanceFromErrPct(s, pct, CTX[c]));
          return detail + " " + cells(g.getTreeCellIterator(s, detail), 3000);
        });
      }
      // reading cells back, their shapes, children and relations
      List<String> tokens = new ArrayList<>();
      for (int i = 0; i < 25; i++) {
        String w = wkt(c);
        try {
          Shape s = CTX[c].getFormats().getWktReader().read(w);
          CellIterator it = g.getTreeCellIterator(s, Math.min(g.getMaxLevels(), 1 + R.nextInt(4)));
          while (it.hasNext()) {
            Cell cc = it.next();
            if (R.nextInt(4) == 0) tokens.add(hex(cc.getTokenBytesWithLeaf(null)));
          }
        } catch (Exception e) {
          // skip
        }
      }
      tokens.add("-");
      for (String tok : tokens) {
        String w = wkt(c);
        String other = tokens.get(R.nextInt(tokens.size()));
        rec("read", t + "\t" + tok + "\t" + other + "\t" + esc(w), () -> {
          Cell cell = g.readCell(new BytesRef(unhex(tok)), null);
          Cell o = g.readCell(new BytesRef(unhex(other)), null);
          StringBuilder sb = new StringBuilder(cell(cell));
          sb.append(" | ").append(safe(() -> bbox(cell.getShape())));
          sb.append(" | ").append(cell.isPrefixOf(o)).append(' ').append(Integer.signum(cell.compareToNoLeaf(o)));
          Shape s = CTX[c].getFormats().getWktReader().read(w);
          sb.append(" | ").append(safe(() -> cell.getShape().relate(s)));
          if (cell.getLevel() < g.getMaxLevels()) {
            sb.append(" | ").append(safe(() -> cells(cell.getNextLevelCells(s), 100)));
            sb.append(" | ").append(safe(() -> cells(cell.getNextLevelCells(null), 100)));
          }
          return sb.toString();
        });
      }
    }
  }

  static final DateRangePrefixTree[] DRT = {
    new DateRangePrefixTree(DateRangePrefixTree.DEFAULT_CAL),
    new DateRangePrefixTree(DateRangePrefixTree.JAVA_UTIL_TIME_COMPAT_CAL),
  };

  static String raw(UnitNRShape u) {
    StringBuilder sb = new StringBuilder("[");
    for (int l = 1; l <= u.getLevel(); l++) sb.append(l > 1 ? "," : "").append(u.getValAtLevel(l));
    return sb.append(']').toString();
  }

  static String token(UnitNRShape u) {
    return hex(((Cell) u).getTokenBytesNoLeaf(null));
  }

  static final String[] DATES = {
    "*", "2014", "2014-10", "2014-10-23", "2014-10-23T21", "2014-10-23T21:22", "2014-10-23T21:22:33",
    "2014-10-23T21:22:33.159", "2014-10-23T21:22:33.159Z", "2014-10-23T21:22:33.1", "2014-10-23T21:22:33.15",
    "2014-10-23T21:22:33.1599", "+2014-10-23", "0000", "-0001", "-0002-03", "0001-01-01", "1582-10-04",
    "1582-10-05", "1582-10-14", "1582-10-15", "1582-10", "1500-02-29", "1600-02-29", "1900-02-29",
    "2000-02-29", "2014-04-31", "2014-10-23T24", "99999", "-99999", "+10000-01", "292278994", "-292269054",
    "2014-13", "2014-00", "2014-10-32", "2014-10-23T25", "2014-10-23T21:60", "2014-1", "2014-10-2",
    "2014-10x23", "2014-10-23x21", "2014-10-23T21x22", "2014-10-23T21:22x33", "2014-10-23T21:22:33x1",
    "abc", "", "2014-10-23T21:22:33.", "1970-01-01T00:00:00.000", "-0-01", "2014-10-23T21:22:33.12345678901",
  };

  static void dates() throws Exception {
    for (int k = 0; k < DRT.length; k++) {
      DateRangePrefixTree tree = DRT[k];
      final int kk = k;
      rec("drt", String.valueOf(k), () -> {
        StringBuilder sb = new StringBuilder(tree.toString()).append(' ').append(tree.getMaxLevels());
        UnitNRShape min = tree.toUnitShape(new Date(Long.MIN_VALUE));
        UnitNRShape max = tree.toUnitShape(new Date(Long.MAX_VALUE));
        sb.append(' ').append(raw(min)).append(' ').append(raw(max)).append(' ').append(esc(min.toString())).append(' ').append(esc(max.toString()));
        for (int f : new int[] {Calendar.ERA, Calendar.YEAR, Calendar.MONTH, Calendar.WEEK_OF_YEAR, Calendar.DAY_OF_MONTH, Calendar.HOUR_OF_DAY, Calendar.MINUTE, Calendar.SECOND, Calendar.MILLISECOND, Calendar.DAY_OF_YEAR}) {
          sb.append(' ').append(safe(() -> tree.getTreeLevelForCalendarField(f)));
        }
        return sb.toString();
      });
      for (String s : DATES) {
        rec("date", k + "\t" + esc(s), () -> {
          Calendar cal = tree.parseCalendar(s);
          String str = tree.toString(cal);
          UnitNRShape u = tree.toShape(cal);
          String back = tree.toString(tree.toCalendar(u));
          return esc(str) + " " + raw(u) + " " + token(u) + " " + esc(u.toString()) + " " + esc(back) + " " + tree.getCalPrecisionField(cal)
              + " " + safe(() -> tree.toCalendar(u).getTimeInMillis());
        });
      }
      // random instants, at random precisions
      for (int i = 0; i < 300; i++) {
        long ms = R.nextInt(3) == 0 ? R.nextLong() : (long) ((R.nextDouble() - 0.5) * 2e14);
        int level = 1 + R.nextInt(tree.getMaxLevels());
        rec("datems", k + "\t" + ms + "\t" + level, () -> {
          UnitNRShape u = tree.toUnitShape(new Date(ms));
          UnitNRShape r = u.roundToLevel(level);
          StringBuilder sb = new StringBuilder(raw(u)).append(' ').append(esc(u.toString())).append(' ').append(raw(r)).append(' ').append(esc(r.toString()));
          sb.append(' ').append(token(r));
          for (int l = 0; l < r.getLevel(); l++) sb.append(l == 0 ? " " : ",").append(tree.getNumSubCells(r.getShapeAtLevel(l)));
          return sb.toString();
        });
      }
      // months' sub-cell counts across calendars and eras
      for (int i = 0; i < 200; i++) {
        int year = R.nextInt(5) == 0 ? R.nextInt(4000) - 2000 : 1500 + R.nextInt(200);
        int month = R.nextInt(12);
        rec("subcells", k + "\t" + year + "\t" + month, () -> {
          Calendar cal = tree.newCal();
          cal.set(Calendar.ERA, year <= 0 ? 0 : 1);
          cal.set(Calendar.YEAR, year <= 0 ? 1 - year : year);
          cal.set(Calendar.MONTH, month);
          UnitNRShape u = tree.toShape(cal);
          return raw(u) + " " + tree.getNumSubCells(u) + " " + esc(u.toString());
        });
      }
      // ranges: parse, normalise, relate, and the cells they index at full detail
      List<String> shapes = new ArrayList<>();
      for (int i = 0; i < 120; i++) {
        String a = randomDate(), b = randomDate();
        String s = R.nextInt(4) == 0 ? a : "[" + a + " TO " + b + "]";
        shapes.add(s);
      }
      shapes.add("[* TO *]");
      shapes.add("[2014 TO 2014-01]");
      shapes.add("[2014-04 TO 2014-04-30]");
      shapes.add("[2014-04-01 TO 2014-04]");
      shapes.add("[2014-12-31 TO 2014]");
      shapes.add("[2014 TO 2013]");
      shapes.add("{2014 TO 2015}");
      shapes.add("[2014 TO 2015");
      shapes.add("[2014 2015]");
      shapes.add("[2014-02-01T00:00:00 TO 2014-02-28T23:59:59.999]");
      for (String s : shapes) {
        String o = shapes.get(R.nextInt(shapes.size()));
        int lvl = 1 + R.nextInt(tree.getMaxLevels());
        rec("nrshape", k + "\t" + esc(s) + "\t" + esc(o) + "\t" + lvl, () -> {
          NRShape a = tree.parseShape(s);
          StringBuilder sb = new StringBuilder(esc(a.toString()));
          sb.append(" | ").append(safe(() -> esc(a.roundToLevel(lvl).toString())));
          sb.append(" | ").append(safe(() -> a.relate(tree.parseShape(o))));
          sb.append(" | ").append(safe(() -> a.equals(tree.parseShape(o))));
          sb.append(" | ").append(safe(() -> cells(tree.getTreeCellIterator(a, tree.getMaxLevels()), 2000)));
          sb.append(" | ").append(safe(() -> cells(tree.getTreeCellIterator(a, lvl), 2000)));
          return sb.toString();
        });
      }
      // relations between units and spans meeting at their edges
      String[] rel = {
        "2014", "2014-01", "2014-01-01", "2014-01-02", "2014-01-30", "2014-01-31",
        "2014-01-01T05", "[2014-01 TO 2014-01-01T05]", "[2014-01 TO 2014-01-02T05]",
        "[2014-01-30T05 TO 2014-01]", "[2014-01-31T05 TO 2014-01]", "[2014 TO 2014-06-15]",
        "[2013-12-31 TO 2014-01-01]", "[2014-01-01T05 TO 2014-01-01T07]", "[2015 TO 2016]",
        "[* TO 2014-01-01]", "*", "-0001", "[1582-10 TO 1582-10-15]", "1582-10"
      };
      for (String a : rel) {
        for (String b : rel) {
          rec("nrrel", k + "\t" + esc(a) + "\t" + esc(b), () -> {
            NRShape x = tree.parseShape(a), y = tree.parseShape(b);
            String cmp =
                x instanceof UnitNRShape ux && y instanceof UnitNRShape uy
                    ? String.valueOf(Integer.signum(ux.compareTo(uy)))
                    : "-";
            return x.relate(y) + " " + cmp;
          });
        }
      }
      // reading cell terms back
      for (int i = 0; i < 60; i++) {
        long ms = (long) ((R.nextDouble() - 0.5) * 2e14);
        int lvl = 1 + R.nextInt(tree.getMaxLevels());
        boolean leaf = R.nextBoolean();
        rec("nrread", k + "\t" + ms + "\t" + lvl + "\t" + leaf, () -> {
          UnitNRShape u = tree.toUnitShape(new Date(ms)).roundToLevel(lvl);
          Cell c = (Cell) u;
          BytesRef term = c.getTokenBytesNoLeaf(null);
          BytesRef t2 = BytesRef.deepCopyOf(term);
          if (leaf) {
            byte[] b = new byte[t2.length + 1];
            System.arraycopy(t2.bytes, t2.offset, b, 0, t2.length);
            t2 = new BytesRef(b);
          }
          Cell back = tree.readCell(t2, null);
          return hex(t2) + " " + cell(back) + " " + esc(back.getShape().toString()) + " " + safe(() -> cells(back.getNextLevelCells(null), 50));
        });
      }
    }
  }

  static String randomDate() {
    switch (R.nextInt(8)) {
      case 0:
        return "*";
      case 1:
        return String.valueOf(R.nextInt(4000) - 1000);
      case 2:
        return String.format(Locale.ROOT, "%04d-%02d", 1500 + R.nextInt(600), 1 + R.nextInt(12));
      case 3:
        return String.format(Locale.ROOT, "%04d-%02d-%02d", 1500 + R.nextInt(600), 1 + R.nextInt(12), 1 + R.nextInt(28));
      case 4:
        return String.format(Locale.ROOT, "%04d-%02d-%02dT%02d", 2000 + R.nextInt(30), 1 + R.nextInt(12), 1 + R.nextInt(28), R.nextInt(24));
      case 5:
        return String.format(Locale.ROOT, "%04d-%02d-%02dT%02d:%02d:%02d", 2014, 1 + R.nextInt(12), 1 + R.nextInt(28), R.nextInt(24), R.nextInt(60), R.nextInt(60));
      case 6:
        return String.format(Locale.ROOT, "%04d-%02d-%02dT%02d:%02d:%02d.%03d", 2014, 10, 1 + R.nextInt(28), R.nextInt(24), R.nextInt(60), R.nextInt(60), R.nextInt(1000));
      default:
        return String.format(Locale.ROOT, "%04d", 2014 + R.nextInt(3));
    }
  }

  static final String[] ARGS = {
    "Intersects(ENVELOPE(-10,-8,22,20)) distErrPct=0.025",
    "Intersects(ENVELOPE(-10,-8,22,20)) distErr=0.5",
    "IsWithin(POINT(1 2))",
    "Contains( BUFFER(POINT(1 2), 3) )",
    "Disjoint(ENVELOPE(0,10,10,0))  distErrPct=0.1 ",
    "intersects(POINT(1 2))",
    "COVEREDBY(POINT(1 2))",
    "Nope(POINT(1 2))",
    "Intersects POINT(1 2)",
    "Intersects)POINT(1 2)(",
    "Intersects()",
    "Intersects(  )",
    "Intersects(POINT(1 2)) foo=bar",
    "Intersects(POINT(1 2)) distErr=1 distErrPct=0.1",
    "Intersects(POINT(1 2)) distErr=abc",
    "Intersects(FOO(1 2))",
    "Intersects(POINT(1 2) x)",
    "Overlaps(ENVELOPE(-180,180,90,-90)) distErrPct=0.6",
    "BBoxWithin(ENVELOPE(170,-170,10,-10)) distErrPct=0.3",
    "Equals(BUFFER(LINESTRING(0 0, 10 10), 1)) distErrPct=0.025",
    "Within(GEOMETRYCOLLECTION(POINT(1 2), ENVELOPE(0,5,5,0))) distErrPct=0.025",
    "Intersects(POINT(1 2)) distErrPct",
  };

  static void args() {
    SpatialArgsParser parser = new SpatialArgsParser();
    for (int c = 0; c < CTX.length; c++) {
      final int cc = c;
      SpatialContext ctx = CTX[c];
      for (String a : ARGS) {
        rec("args", c + "\t" + esc(a), () -> {
          SpatialArgs args = parser.parse(a, ctx);
          return (cc >= 2 ? args.getOperation().getName() : esc(args.toString())) + " | " + safe(() -> h(args.resolveDistErr(ctx, 0.025))) + " | " + args.getDistErrPct() + " " + args.getDistErr();
        });
      }
    }
    // the operations against random shape pairs
    for (int i = 0; i < 300; i++) {
      final int c = R.nextInt(2) == 0 ? 0 : 1;
      String a = wkt(c), b = wkt(c);
      rec("op", c + "\t" + esc(a) + "\t" + esc(b), () -> {
        Shape sa = CTX[c].getFormats().getWktReader().read(a);
        Shape sb = CTX[c].getFormats().getWktReader().read(b);
        StringBuilder out = new StringBuilder();
        for (SpatialOperation op : SpatialOperation.values()) {
          out.append(op.getName()).append('=').append(safe(() -> op.evaluate(sa, sb))).append(' ');
        }
        return out.toString().trim();
      });
    }
  }

  public static void main(String[] args) throws Exception {
    for (int c = 0; c < CTX_ARGS.length; c++) {
      Map<String, String> m = new HashMap<>();
      for (int i = 0; i < CTX_ARGS[c].length; i += 2) m.put(CTX_ARGS[c][i], CTX_ARGS[c][i + 1]);
      CTX[c] = SpatialContextFactory.makeSpatialContext(m, GenSpatialPrefixTree.class.getClassLoader());
    }
    for (int t = 0; t < TREES.length; t++) {
      Map<String, String> m = new HashMap<>();
      GRID_CTX[t] = Integer.parseInt(TREES[t][0]);
      boolean noprune = false;
      int arity = 0;
      for (int i = 1; i < TREES[t].length; i += 2) {
        if (TREES[t][i].equals("noprune")) noprune = true;
        else if (TREES[t][i].equals("s2arity")) arity = Integer.parseInt(TREES[t][i + 1]);
        else m.put(TREES[t][i], TREES[t][i + 1]);
      }
      GRID[t] =
          arity > 0
              ? new S2PrefixTree(CTX[GRID_CTX[t]], Integer.parseInt(m.get("maxLevels")), arity)
              : SpatialPrefixTreeFactory.makeSPT(m, GenSpatialPrefixTree.class.getClassLoader(), CTX[GRID_CTX[t]]);
      if (noprune) ((PackedQuadPrefixTree) GRID[t]).setPruneLeafyBranches(false);
    }
    trees();
    dates();
    args();
    Path dir = Path.of(args[0], "spatial_prefix_tree");
    Files.createDirectories(dir);
    Files.writeString(dir.resolve("trees.tsv"), OUT.toString(), StandardCharsets.UTF_8);
  }
}
