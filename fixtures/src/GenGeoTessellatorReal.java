import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.InflaterInputStream;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.SimpleWKTShapeParser;
import org.apache.lucene.geo.Tessellator;
import org.apache.lucene.geo.XYPolygon;

/**
 * Cross-engine ground truth for {@code lucene-util}'s {@code geo::tessellator} over real-world
 * polygons: {@code fixtures/corpus/real_polygons.z} (built by {@code scripts/gen-tessellator-corpus.py} from
 * Lucene's own {@code TestTessellator} shapes and Natural Earth countries, provinces and lakes),
 * each shape parsed by Lucene's {@code SimpleWKTShapeParser} or {@code Polygon.fromGeoJSON} and
 * every polygon of it tessellated with and without {@code checkSelfIntersections}; the Lucene
 * shapes also as cartesian polygons (their lon/lat as float x/y).
 *
 * <p>{@code geo/tessellator_real.tsv}: per shape a {@code shape<TAB>source<TAB>name<TAB>polygons}
 * line (or {@code shape<TAB>source<TAB>name<TAB>ERR<TAB>class<TAB>message} when the parser
 * throws), then per polygon and mode a {@code poly<TAB>index<TAB>latlon|xy<TAB>check<TAB>
 * triangles<TAB>hash} line -- {@code hash} is FNV-1a 64 over every triangle's encoded x,y of each
 * vertex (4 bytes little-endian each) and its three edge-from-polygon flags (one byte, bit v) --
 * or {@code poly<TAB>index<TAB>latlon|xy<TAB>check<TAB>ERR<TAB>class<TAB>message}. The shapes of
 * {@code TestTessellator} itself (small) also list their triangles in full, after the poly line,
 * in {@code GenGeoTessellator}'s {@code tri} format.
 */
public class GenGeoTessellatorReal {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("geo");
    Files.createDirectories(out);
    String dir = System.getenv("FIXTURES_CORPUS");
    if (dir == null) {
      throw new IOException("FIXTURES_CORPUS is not set (run scripts/gen-fixtures.sh)");
    }
    Path in = Path.of(dir).resolve("real_polygons.z");
    String corpus;
    try (InputStream s = new InflaterInputStream(Files.newInputStream(in))) {
      corpus = new String(s.readAllBytes(), StandardCharsets.UTF_8);
    }
    StringBuilder sb = new StringBuilder();
    for (String line : corpus.split("\n")) {
      String[] f = line.split("\t", 4);
      String source = f[0];
      boolean lucene = !source.startsWith("ne_");
      boolean full = source.equals("TestTessellator");
      sb.append("shape\t").append(source).append('\t').append(f[1]).append('\t');
      Polygon[] polygons;
      try {
        polygons = parse(f[2], f[3]);
      } catch (Exception e) {
        sb.append("ERR\t").append(e.getClass().getName()).append('\t').append(GeoCorpus.esc(e.getMessage())).append('\n');
        continue;
      }
      sb.append(polygons.length).append('\n');
      for (int i = 0; i < polygons.length; i++) {
        for (boolean check : new boolean[] {true, false}) {
          sb.append("poly\t").append(i).append("\tlatlon\t").append(check ? 1 : 0).append('\t');
          try {
            result(sb, Tessellator.tessellate(polygons[i], check), full);
          } catch (IllegalArgumentException e) {
            err(sb, e);
          }
        }
        if (lucene) {
          XYPolygon xy;
          try {
            xy = toXY(polygons[i]);
          } catch (IllegalArgumentException e) {
            sb.append("poly\t").append(i).append("\txy\t-\t");
            err(sb, e);
            continue;
          }
          for (boolean check : new boolean[] {true, false}) {
            sb.append("poly\t").append(i).append("\txy\t").append(check ? 1 : 0).append('\t');
            try {
              result(sb, Tessellator.tessellate(xy, check), full);
            } catch (IllegalArgumentException e) {
              err(sb, e);
            }
          }
        }
      }
    }
    Files.writeString(out.resolve("tessellator_real.tsv"), sb.toString());
  }

  static Polygon[] parse(String format, String text) throws Exception {
    if (format.equals("geojson")) {
      return Polygon.fromGeoJSON(text);
    }
    Object g = SimpleWKTShapeParser.parse(text);
    if (g instanceof Polygon p) {
      return new Polygon[] {p};
    }
    return (Polygon[]) g;
  }

  /** The polygon's lon/lat as float x/y, holes included. */
  static XYPolygon toXY(Polygon p) {
    Polygon[] src = p.getHoles();
    XYPolygon[] holes = new XYPolygon[src.length];
    for (int h = 0; h < holes.length; h++) {
      holes[h] = toXY(src[h]);
    }
    double[] lats = p.getPolyLats();
    double[] lons = p.getPolyLons();
    float[] x = new float[lats.length];
    float[] y = new float[lats.length];
    for (int v = 0; v < lats.length; v++) {
      x[v] = (float) lons[v];
      y[v] = (float) lats[v];
    }
    return new XYPolygon(x, y, holes);
  }

  static void err(StringBuilder sb, Exception e) {
    sb.append("ERR\t").append(e.getClass().getName()).append('\t').append(GeoCorpus.esc(e.getMessage())).append('\n');
  }

  static void result(StringBuilder sb, List<Tessellator.Triangle> tris, boolean full) {
    long h = 0xcbf29ce484222325L;
    for (Tessellator.Triangle t : tris) {
      int flags = 0;
      for (int v = 0; v < 3; v++) {
        h = fnv(h, t.getEncodedX(v));
        h = fnv(h, t.getEncodedY(v));
        flags |= t.isEdgefromPolygon(v) ? 1 << v : 0;
      }
      h = (h ^ flags) * 0x100000001b3L;
    }
    sb.append(tris.size()).append('\t').append(Long.toHexString(h)).append('\n');
    if (full) {
      GenGeoTessellator.emit(sb, tris);
    }
  }

  static long fnv(long h, int v) {
    for (int b = 0; b < 4; b++) {
      h = (h ^ ((v >>> (8 * b)) & 0xff)) * 0x100000001b3L;
    }
    return h;
  }
}
