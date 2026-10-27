import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.search.Query;
import org.apache.lucene.spatial3d.Geo3DPoint;
import org.apache.lucene.spatial3d.geom.GeoShape;
import org.apache.lucene.spatial3d.geom.PlanetObject;
import org.apache.lucene.spatial3d.geom.SerializableObject;
import org.apache.lucene.util.Accountable;

/**
 * Java's {@code equals}, {@code hashCode} and {@code toString} of every geo3d shape, and of the
 * {@code PointInGeo3DShapeQuery} over it (plus its {@code ramBytesUsed}), for
 * crates/lucene-search/tests/geo3d_objects_fixtures.rs.
 *
 * <p>The shapes are {@link GenGeo3d}'s corpus, read back from the serialized bytes its {@code SH}
 * records hold ({@code geo3d/shapes.tsv} in the output directory, which GenGeo3d writes first), so
 * both sides compare the same objects. Records:
 *
 * <pre>
 * OBJ  id  class  hashCode  ids-equal-to-it(,)  toString
 * Q    id  hashCode  ramBytesUsed(field "f")  ramBytesUsed(field LONG)  [toString("f")  toString("g")]
 * </pre>
 *
 * The equal ids list every shape of the corpus {@code equals} returns true for (itself
 * included); every other pair is unequal. A query whose constructor throws has no {@code Q}
 * record; only the first {@link #QUERY_STRINGS} carry the query's strings (the rest would repeat
 * the shape's).
 */
public class GenGeo3dObjects {
  static final String LONG = "a_rather_longer_geo3d_field_name";
  static final int QUERY_STRINGS = 40;

  static String esc(String s) {
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n");
  }

  static byte[] unhex(String s) {
    byte[] b = new byte[s.length() / 2];
    for (int i = 0; i < b.length; i++) {
      b[i] = (byte) Integer.parseInt(s.substring(2 * i, 2 * i + 2), 16);
    }
    return b;
  }

  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]).resolve("geo3d");
    List<Integer> ids = new ArrayList<>();
    List<GeoShape> shapes = new ArrayList<>();
    for (String line : Files.readAllLines(dir.resolve("shapes.tsv"))) {
      if (!line.startsWith("SH\t")) {
        continue;
      }
      String[] f = line.split("\t");
      int arrow = 0;
      while (!f[arrow].equals("=>")) {
        arrow++;
      }
      if (!f[arrow + 1].equals("OK")) {
        continue;
      }
      PlanetObject o =
          SerializableObject.readPlanetObject(new ByteArrayInputStream(unhex(f[arrow + 3])));
      if (o instanceof GeoShape) {
        ids.add(Integer.parseInt(f[1]));
        shapes.add((GeoShape) o);
      }
    }
    StringBuilder out = new StringBuilder();
    for (int i = 0; i < shapes.size(); i++) {
      GeoShape s = shapes.get(i);
      StringBuilder equal = new StringBuilder();
      for (int j = 0; j < shapes.size(); j++) {
        if (s.equals(shapes.get(j))) {
          equal.append(equal.length() == 0 ? "" : ",").append(ids.get(j));
        }
      }
      out.append("OBJ\t")
          .append(ids.get(i))
          .append('\t')
          .append(s.getClass().getSimpleName())
          .append('\t')
          .append(s.hashCode())
          .append('\t')
          .append(equal)
          .append('\t')
          .append(esc(s.toString()))
          .append('\n');
      Query q;
      Query longQ;
      try {
        q = Geo3DPoint.newShapeQuery("f", s);
        longQ = Geo3DPoint.newShapeQuery(LONG, s);
      } catch (RuntimeException e) {
        continue;
      }
      out.append("Q\t")
          .append(ids.get(i))
          .append('\t')
          .append(q.hashCode())
          .append('\t')
          .append(((Accountable) q).ramBytesUsed())
          .append('\t')
          .append(((Accountable) longQ).ramBytesUsed());
      if (i < QUERY_STRINGS) {
        out.append('\t').append(esc(q.toString("f"))).append('\t').append(esc(q.toString("g")));
      }
      out.append('\n');
    }
    Files.writeString(dir.resolve("objects.tsv"), out.toString());
  }
}
