import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.apache.lucene.geo.Polygon;
import org.apache.lucene.geo.Tessellator;
import org.apache.lucene.geo.XYPolygon;

/**
 * Cross-engine ground truth for {@code lucene-util}'s {@code geo::tessellator}: real {@link
 * Tessellator} output, triangle for triangle and in order, for the seeded {@link GeoCorpus}
 * polygons (holes, poles, the dateline, self-intersections, collinear and duplicate points, the
 * morton path past 80 vertices) as lat/lon and as cartesian polygons, with and without {@code
 * checkSelfIntersections}.
 *
 * <p>{@code geo/tessellator.tsv}: a {@code poly<TAB>id<TAB>latlon|xy<TAB>check<TAB>spec} line,
 * then one {@code tri<TAB>ax,ay,bx,by,cx,cy,ab,bc,ca} line per triangle (encoded coordinates,
 * edge-from-polygon flags as 0/1) or a single {@code ERR<TAB>class<TAB>message} line.
 */
public class GenGeoTessellator {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("geo");
    Files.createDirectories(out);
    Random r = new Random(0x7E55L);
    List<Polygon> polys = GeoCorpus.polygons(r, 120);
    StringBuilder sb = new StringBuilder();
    int id = 0;
    for (Polygon p : polys) {
      for (boolean check : new boolean[] {true, false}) {
        sb.append("poly\t").append(id).append("\tlatlon\t").append(check ? 1 : 0).append('\t').append(GeoCorpus.esc(GeoCorpus.spec(p))).append('\n');
        try {
          emit(sb, Tessellator.tessellate(p, check));
        } catch (IllegalArgumentException e) {
          sb.append("ERR\t").append(e.getClass().getName()).append('\t').append(GeoCorpus.esc(e.getMessage())).append('\n');
        }
      }
      id++;
    }
    for (int i = 0; i < polys.size(); i += 2) {
      XYPolygon p;
      try {
        p = GeoCorpus.xyPolygon(polys.get(i), GeoCorpus.randomScale(r), (r.nextDouble() - 0.5) * 1e3);
      } catch (IllegalArgumentException e) {
        continue;
      }
      for (boolean check : new boolean[] {true, false}) {
        sb.append("poly\t").append(id).append("\txy\t").append(check ? 1 : 0).append('\t').append(GeoCorpus.esc(GeoCorpus.spec(p))).append('\n');
        try {
          emit(sb, Tessellator.tessellate(p, check));
        } catch (IllegalArgumentException e) {
          sb.append("ERR\t").append(e.getClass().getName()).append('\t').append(GeoCorpus.esc(e.getMessage())).append('\n');
        }
      }
      id++;
    }
    Files.writeString(out.resolve("tessellator.tsv"), sb.toString());
  }

  static void emit(StringBuilder sb, List<Tessellator.Triangle> tris) {
    for (Tessellator.Triangle t : tris) {
      sb.append("tri\t");
      for (int v = 0; v < 3; v++) {
        sb.append(t.getEncodedX(v)).append(',').append(t.getEncodedY(v)).append(',');
      }
      for (int v = 0; v < 3; v++) {
        sb.append(t.isEdgefromPolygon(v) ? 1 : 0).append(v < 2 ? "," : "");
      }
      sb.append('\n');
    }
  }
}
