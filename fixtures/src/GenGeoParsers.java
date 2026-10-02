import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.lucene.geo.Line;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.Rectangle;
import org.apache.lucene.geo.SimpleWKTShapeParser;

/**
 * Cross-engine ground truth for {@code lucene-util}'s {@code geo::simple_wkt_shape_parser} and
 * {@code geo::simple_geojson_polygon_parser}: what real {@link SimpleWKTShapeParser} and {@code
 * Polygon.fromGeoJSON} (its package-private {@code SimpleGeoJSONPolygonParser}) make of hand-written
 * inputs, of {@link GeoCorpus} geometries, and of random mutations of both (truncations, deletions,
 * substitutions) -- which is what exercises the error messages and offsets.
 *
 * <p>{@code geo/wkt.tsv} and {@code geo/geojson.tsv}: {@code input<TAB>result}, input escaped
 * ({@code \t \n \r \\}); result is a canonical dump of the parsed geometry (coordinates as raw
 * double bits in hex) or {@code ERR<TAB>class<TAB>message<TAB>offset} (offset only for {@code
 * ParseException}).
 */
public class GenGeoParsers {
  static final Random R = new Random(0x9A25E5L);

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("geo");
    Files.createDirectories(out);
    Files.writeString(out.resolve("wkt.tsv"), wkt());
    Files.writeString(out.resolve("geojson.tsv"), geojson());
  }

  static String h(double v) {
    return Long.toHexString(Double.doubleToRawLongBits(v));
  }

  static String ring(double[] lats, double[] lons) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < lats.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(h(lats[i])).append(' ').append(h(lons[i]));
    }
    return sb.toString();
  }

  static String dump(Object o) {
    if (o == null) return "null";
    if (o instanceof double[] p) return "P(" + h(p[0]) + "," + h(p[1]) + ")";
    if (o instanceof double[][] mp) {
      StringBuilder sb = new StringBuilder("MP[");
      for (int i = 0; i < mp.length; i++) sb.append(i > 0 ? "," : "").append(dump(mp[i]));
      return sb.append(']').toString();
    }
    if (o instanceof Line l) return "L(" + ring(l.getLats(), l.getLons()) + ")";
    if (o instanceof Polygon p) {
      StringBuilder sb = new StringBuilder("G(").append(ring(p.getPolyLats(), p.getPolyLons()));
      for (Polygon hole : p.getHoles()) sb.append('|').append(ring(hole.getPolyLats(), hole.getPolyLons()));
      return sb.append(')').toString();
    }
    if (o instanceof Rectangle r) return "R(" + h(r.minLat) + "," + h(r.maxLat) + "," + h(r.minLon) + "," + h(r.maxLon) + ")";
    if (o instanceof Object[] arr) {
      String tag = o instanceof Line[] ? "ML" : o instanceof Polygon[] ? "MG" : "GC";
      StringBuilder sb = new StringBuilder(tag).append('[');
      for (int i = 0; i < arr.length; i++) sb.append(i > 0 ? "," : "").append(dump(arr[i]));
      return sb.append(']').toString();
    }
    throw new IllegalStateException(o.getClass().toString());
  }

  static String err(Throwable t) {
    String s = "ERR\t" + t.getClass().getName() + "\t" + GeoCorpus.esc(String.valueOf(t.getMessage()));
    if (t instanceof java.text.ParseException pe) s += "\t" + pe.getErrorOffset();
    return s;
  }

  static String mutate(String s) {
    if (s.isEmpty()) return s;
    String alphabet = " ,()[]{}\"-.+0123456789eExXpPaZ#\n\r\t:\\u";
    int i = R.nextInt(s.length());
    switch (R.nextInt(4)) {
      case 0:
        return s.substring(0, i);
      case 1:
        return s.substring(0, i) + s.substring(i + 1);
      case 2:
        return s.substring(0, i) + alphabet.charAt(R.nextInt(alphabet.length())) + s.substring(i + 1);
      default:
        return s.substring(0, i) + alphabet.charAt(R.nextInt(alphabet.length())) + s.substring(i);
    }
  }

  // ------------------------------------------------------------------ WKT

  static String num(double v) {
    switch (R.nextInt(12)) {
      case 0:
        return Double.toHexString(v);
      case 1:
        return v + "d";
      case 2:
        return (v >= 0 ? "+" : "") + v;
      default:
        return Double.toString(v);
    }
  }

  static String wktRing(double[] lats, double[] lons) {
    StringBuilder sb = new StringBuilder("(");
    for (int i = 0; i < lats.length; i++) {
      if (i > 0) sb.append(R.nextBoolean() ? ", " : ",");
      sb.append(num(lons[i])).append(' ').append(num(lats[i]));
      if (R.nextInt(15) == 0) sb.append(' ').append(num(R.nextDouble()));
    }
    return sb.append(')').toString();
  }

  static String wktPolygon(Polygon p) {
    StringBuilder sb = new StringBuilder("(").append(wktRing(p.getPolyLats(), p.getPolyLons()));
    for (Polygon h : p.getHoles()) sb.append(", ").append(wktRing(h.getPolyLats(), h.getPolyLons()));
    return sb.append(')').toString();
  }

  static String wkt() {
    List<String> inputs = new ArrayList<>();
    String[] fixed = {
      "POINT (30 10)", "point(30 10)", "POINT EMPTY", "POINT (30 10 5)", "POINT (NaN 10)", "POINT (nan 10)",
      "POINT(1e2 -0.5e-1)", "POINT (0x1.8p1 2)", "POINT (1d 2F)", "POINT (- 1)", "POINT (1..2 3)", "POINT (30)",
      "POINT 30 10", "POINT (30 10", "POINT (30 10))", "POINT (30, 10)", "POINT (Infinity 0)", "POINT (+Infinity 0)",
      "MULTIPOINT (10 40, 40 30, 20 20, 30 10)", "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))", "MULTIPOINT ((10 40))",
      "MULTIPOINT ((10 40) (40 30))", "MULTIPOINT EMPTY", "MULTIPOINT (10 40 1, 40 30 2)", "MULTIPOINT ()",
      "LINESTRING (30 10, 10 30, 40 40)", "LINESTRING EMPTY", "LINESTRING (30 10)", "LINESTRING (30 100, 10 30)",
      "LINESTRING ((30 10), (10 30))", "LINESTRING (30 10, (10 30))",
      "MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))", "MULTILINESTRING (EMPTY, (1 1, 2 2))",
      "MULTILINESTRING EMPTY", "MULTILINESTRING ((10 10, 20 20) (1 1, 2 2))",
      "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))", "POLYGON EMPTY",
      "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))",
      "POLYGON ((30 10, 40 40, 20 40, 10 20))", "POLYGON ((30 10, 40 40, 30 10))", "POLYGON (30 10, 40 40, 20 40, 30 10)",
      "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10)", "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10), EMPTY)",
      "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))",
      "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (30 20, 20 15, 20 25, 30 20)))",
      "MULTIPOLYGON EMPTY", "MULTIPOLYGON (EMPTY, ((1 1, 2 1, 2 2, 1 1)))",
      "ENVELOPE (10, 40, 40, 10)", "BBOX (10, 40, 40, 10)", "bbox(-180, 180, 90, -90)", "ENVELOPE EMPTY",
      "ENVELOPE (10 40 40 10)", "ENVELOPE (10, 40, 40)", "ENVELOPE (10, 400, 40, 10)",
      "GEOMETRYCOLLECTION (POINT (40 10), LINESTRING (10 10, 20 20, 10 40), POLYGON ((40 40, 20 45, 45 30, 40 40)))",
      "GEOMETRYCOLLECTION EMPTY", "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (1 2)), POINT EMPTY)",
      "GEOMETRYCOLLECTION (POINT (1 2) POINT (3 4))", "GEOMETRYCOLLECTION (BBOX (1, 2, 4, 3), MULTIPOINT (1 2))",
      "CIRCLE (30 10, 5)", "", "   ", "(30 10)", "POINT", "POINT (30 10) extra", "POINT (30 10)\n# trailing comment",
      "# comment\nPOINT (30 10)", "POINT\n(\n30\r\n10\r)\nx", "POINT\r\r(30 10) x", "POINT (30 10)\n\n\n,",
      "POINT\t(30\t10)", "PoInT (30 10)", "POINT (30 10) )", "POINT [30 10]", "POINT (30 'x')", "POINT (30 é)",
      "POINT (30 ١٠)", "POINT (30 10 20 40)", "LINESTRING (30 10, 10 30,)", "LINESTRING (30 10,, 10 30)",
      "MULTIPOINT (10 40, , 40 30)", "EMPTY", "POINT (1e400 0)", "POINT (0x1p-1080 0)", "POINT (.5 .5)", "POINT (5. 5.)",
      "POINT (1e 2)", "POINT (1e+ 2)", "POINT (1-2 3)", "POINT(1 2)#c", "POINT (1 2\u00a0)", "POINT (1 2)\u0000",
    };
    for (String s : fixed) inputs.add(s);
    List<Polygon> polys = GeoCorpus.polygons(R, 40);
    for (Polygon p : polys) {
      if (p.numPoints() > 60) continue;
      inputs.add("POLYGON " + wktPolygon(p));
    }
    for (int i = 0; i + 1 < polys.size(); i += 2) {
      if (polys.get(i).numPoints() + polys.get(i + 1).numPoints() > 60) continue;
      inputs.add("MULTIPOLYGON (" + wktPolygon(polys.get(i)) + ", " + wktPolygon(polys.get(i + 1)) + ")");
    }
    for (Line l : GeoCorpus.lines(R, 20)) {
      if (l.numPoints() > 30) continue;
      inputs.add("LINESTRING " + wktRing(l.getLats(), l.getLons()));
      inputs.add("MULTIPOINT " + wktRing(l.getLats(), l.getLons()));
    }
    for (int i = 0; i < 20; i++) {
      double[] c = GeoCorpus.center(R);
      inputs.add("POINT (" + num(c[1]) + " " + num(c[0]) + ")");
      inputs.add("ENVELOPE (" + num(c[1]) + ", " + num(Math.min(180, c[1] + 1)) + ", " + num(Math.min(90, c[0] + 1)) + ", " + num(c[0]) + ")");
    }
    int base = inputs.size();
    for (int i = 0; i < base; i++) {
      String s = inputs.get(i);
      if (s.length() > 400) continue;
      for (int k = 0; k < 3; k++) inputs.add(mutate(s));
    }
    StringBuilder sb = new StringBuilder();
    for (String s : inputs) {
      sb.append(GeoCorpus.esc(s)).append('\t');
      try {
        sb.append(dump(SimpleWKTShapeParser.parse(s)));
      } catch (Exception e) {
        sb.append(err(e));
      }
      sb.append('\n');
    }
    // parseExpectedType
    String[][] typed = {
      {"POINT (1 2)", "POINT"}, {"POINT (1 2)", "POLYGON"}, {"BBOX (1, 2, 4, 3)", "ENVELOPE"},
      {"ENVELOPE (1, 2, 4, 3)", "POLYGON"}, {"POINT (1 2)", "GEOMETRYCOLLECTION"},
      {"GEOMETRYCOLLECTION (POINT (1 2))", "GEOMETRYCOLLECTION"}, {"LINESTRING (1 2, 3 4)", "MULTILINESTRING"},
    };
    for (String[] t : typed) {
      sb.append("@").append(t[1]).append(' ').append(GeoCorpus.esc(t[0])).append('\t');
      try {
        sb.append(dump(SimpleWKTShapeParser.parseExpectedType(t[0], SimpleWKTShapeParser.ShapeType.valueOf(t[1]))));
      } catch (Exception e) {
        sb.append(err(e));
      }
      sb.append('\n');
    }
    return sb.toString();
  }

  // ------------------------------------------------------------------ GeoJSON

  static String geojson() {
    List<String> inputs = new ArrayList<>();
    String square = "[[[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]]";
    String[] fixed = {
      "{\"type\": \"Polygon\", \"coordinates\": " + square + "}",
      "{\"coordinates\": " + square + ", \"type\": \"Polygon\"}",
      "{\"type\": \"Feature\", \"geometry\": {\"type\": \"Polygon\", \"coordinates\": " + square + "}, \"properties\": {\"prop0\": \"value0\", \"prop1\": {\"this\": \"that\"}}}",
      "{\"type\": \"FeatureCollection\", \"features\": [{\"type\": \"Feature\", \"geometry\": {\"type\": \"Polygon\", \"coordinates\": " + square + "}}]}",
      "{\"type\": \"FeatureCollection\", \"features\": [{\"type\": \"Feature\", \"geometry\": {\"type\": \"Polygon\", \"coordinates\": " + square + "}}, {\"type\": \"Feature\", \"geometry\": {\"type\": \"Polygon\", \"coordinates\": " + square + "}}]}",
      "{\"type\": \"MultiPolygon\", \"coordinates\": [" + square + ", [[[0,0],[1,0],[1,1],[0,0]], [[0.1,0.1],[0.2,0.1],[0.2,0.2],[0.1,0.1]]]]}",
      "{\"type\": \"MultiPolygon\", \"coordinates\": [5]}", "{\"type\": \"MultiPolygon\", \"coordinates\": [\"x\"]}",
      "{\"type\": \"MultiPolygon\", \"coordinates\": [{}]}", "{\"type\": \"Polygon\", \"coordinates\": []}",
      "{\"type\": \"Polygon\", \"coordinates\": null}", "{\"type\": \"Polygon\", \"coordinates\": {}}",
      "{\"type\": \"Polygon\", \"coordinates\": true}", "{\"type\": \"Polygon\", \"coordinates\": \"x\"}",
      "{\"type\": \"Polygon\", \"coordinates\": [[[0,0],[1,0],[1,1],[0,0]], 5]}", "{\"type\": \"Polygon\", \"coordinates\": [5]}",
      "{\"type\": \"Polygon\", \"coordinates\": [[5]]}", "{\"type\": \"Polygon\", \"coordinates\": [[[1, 2, 3]]]}",
      "{\"type\": \"Polygon\", \"coordinates\": [[[\"a\", 2]]]}", "{\"type\": \"Polygon\", \"coordinates\": [[[1, \"b\"]]]}",
      "{\"type\": \"Polygon\", \"coordinates\": [[[1, [2]]]]}", "{\"type\": \"Polygon\", \"coordinates\": [[[0,0],[1,0],[1,1],[0,1]]]}",
      "{\"type\": \"Polygon\", \"coordinates\": [[[0,0],[1,0],[0,0]]]}", "{\"type\": \"Polygon\", \"coordinates\": [[[0,0],[1,0],[1,100],[0,0]]]}",
      "{\"type\": \"Polygon\"}", "{\"coordinates\": " + square + "}", "{\"type\": \"Point\", \"coordinates\": [1, 2]}",
      "{\"type\": 5}", "{\"type\": null}", "{\"type\": [1, \"a\", null]}", "{\"type\": true}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"coordinates\": " + square + "}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"crs\": {\"type\": \"name\", \"properties\": {\"name\": \"urn:ogc:def:crs:OGC:1.3:CRS84\"}}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"crs\": {\"type\": \"name\", \"properties\": {\"name\": \"EPSG:4326\"}}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"crs\": {\"properties\": {\"name\": 5}}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"crs\": {\"properties\": {\"href\": \"http://x\"}}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": {\"s\": \"a\\\\b\"}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": {\"s\": \"\\u0041\"}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": {\"s\": \"\\u-001\"}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": {\"s\": \"\\uzz12\"}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": {\"s\": \"\\n\"}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": {\"s\": \"\\u12\"}}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": [1, \"x\", [true]]}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": tru}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": nul}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": -.5e3}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": -}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": 1e}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ",}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + "} x",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + "}\n\t ",
      "{\"type\": \"Polygon\" \"coordinates\": " + square + "}",
      "{\"type\": \"Polygon\", \"coordinates\": [[[0,0] [1,0]]]}",
      "{\"type\": \"Polygon\", \"coordinates\": [[[0,0],",
      "{\"type\": \"Polygon\", \"coordinates\": [[[0,0]",
      "{\"type\": \"Polygon\", \"coordinates\": [x]}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": x}",
      "{\"type\": \"Polygon\", \"coordinates\": " + square + ", \"p\": }",
      "", "[]", "{", "{\"", "{\"a", "{\"a\\", "{\"a\": 1,", "{}", "{\"a\": \"é\u4e2d\"} x",
      "{\"type\": \"Feature\", \"geometry\": {\"type\": \"Feature\"}}",
      "{\"type\": \"FeatureCollection\", \"features\": [{\"type\": \"FeatureCollection\"}]}",
      "{\"geometry\": {\"type\": \"MultiPolygon\", \"coordinates\": [" + square + "]}}",
      "{\"features\": [{\"geometry\": {\"type\": \"Polygon\", \"coordinates\": " + square + "}}]}",
      "{\"other\": {\"type\": \"Polygon\", \"coordinates\": " + square + "}}",
    };
    for (String s : fixed) inputs.add(s);
    List<Polygon> polys = GeoCorpus.polygons(R, 30);
    for (Polygon p : polys) {
      if (p.numPoints() > 60) continue;
      inputs.add("{\"type\": \"Polygon\", \"coordinates\": " + p.toGeoJSON() + "}");
    }
    for (int i = 0; i + 1 < polys.size(); i += 3) {
      if (polys.get(i).numPoints() + polys.get(i + 1).numPoints() > 60) continue;
      inputs.add("{\"type\": \"Feature\", \"geometry\": {\"type\": \"MultiPolygon\", \"coordinates\": [" + polys.get(i).toGeoJSON() + "," + polys.get(i + 1).toGeoJSON() + "]}}");
    }
    int base = inputs.size();
    for (int i = 0; i < base; i++) {
      String s = inputs.get(i);
      if (s.length() > 600) continue;
      for (int k = 0; k < 4; k++) inputs.add(mutate(s));
    }
    StringBuilder sb = new StringBuilder();
    for (String s : inputs) {
      sb.append(GeoCorpus.esc(s)).append('\t');
      try {
        sb.append(dump(Polygon.fromGeoJSON(s)));
      } catch (Exception e) {
        sb.append(err(e));
      }
      sb.append('\n');
    }
    return sb.toString();
  }
}
