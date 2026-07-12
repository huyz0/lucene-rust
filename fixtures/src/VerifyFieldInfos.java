import org.apache.lucene.codecs.FieldInfosFormat;
import org.apache.lucene.codecs.lucene94.Lucene94FieldInfosFormat;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;

/**
 * Reverse-direction verifier (Rust writes, Java reads): opens a `.fnm` file
 * written by this port's `field_infos::write` (see
 * `crates/lucene-codecs/examples/write_field_infos_fixture.rs`) directly
 * through real Lucene's {@link Lucene94FieldInfosFormat}, using a hand-built
 * {@link SegmentInfo} rather than also requiring Rust to write {@code .si} --
 * this keeps this write-path slice scoped to exactly the field-infos format
 * itself, the same way {@code VerifyStoredFields.java} scopes to stored
 * fields alone.
 *
 * <p>Usage: {@code java VerifyFieldInfos <fixture-dir>}, where
 * {@code <fixture-dir>} contains {@code _0.fnm} and a
 * {@code manifest.properties} (see the Rust example above for the exact
 * format) describing the expected per-field properties. Exits nonzero and
 * prints a diff on any mismatch.
 */
public class VerifyFieldInfos {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = readManifest(dir.resolve("manifest.properties"));

    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));
    int fieldCount = Integer.parseInt(manifest.get("field_count"));
    String[] fieldOrder = manifest.get("field_order").split(",", -1);

    try (Directory directory = FSDirectory.open(dir)) {
      SegmentInfo si =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              "_0",
              1,
              false,
              false,
              null,
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);

      FieldInfosFormat format = new Lucene94FieldInfosFormat();
      FieldInfos fis = format.read(directory, si, "", IOContext.DEFAULT);

      int failures = 0;
      if (fis.size() != fieldCount) {
        System.out.println(
            "MISMATCH field_count: expected=" + fieldCount + " got=" + fis.size());
        failures++;
      }

      for (String name : fieldOrder) {
        String prefix = "field." + name + ".";
        FieldInfo fi = fis.fieldInfo(name);
        if (fi == null) {
          System.out.println("MISMATCH field " + name + ": missing from read-back FieldInfos");
          failures++;
          continue;
        }

        StringBuilder mismatches = new StringBuilder();
        checkInt(mismatches, "number", Integer.parseInt(manifest.get(prefix + "number")), fi.number);
        checkStr(
            mismatches,
            "index_options",
            manifest.get(prefix + "index_options"),
            fi.getIndexOptions().toString());
        checkStr(
            mismatches,
            "doc_values_type",
            manifest.get(prefix + "doc_values_type"),
            fi.getDocValuesType().toString());
        checkStr(
            mismatches,
            "doc_values_skip_index_type",
            manifest.get(prefix + "doc_values_skip_index_type"),
            fi.docValuesSkipIndexType().toString());
        checkLong(
            mismatches,
            "doc_values_gen",
            Long.parseLong(manifest.get(prefix + "doc_values_gen")),
            fi.getDocValuesGen());
        checkBool(
            mismatches,
            "has_term_vectors",
            Boolean.parseBoolean(manifest.get(prefix + "has_term_vectors")),
            fi.hasTermVectors());
        checkBool(
            mismatches,
            "omit_norms",
            Boolean.parseBoolean(manifest.get(prefix + "omit_norms")),
            fi.omitsNorms());
        checkBool(
            mismatches,
            "store_payloads",
            Boolean.parseBoolean(manifest.get(prefix + "store_payloads")),
            fi.hasPayloads());
        checkBool(
            mismatches,
            "is_soft_deletes",
            Boolean.parseBoolean(manifest.get(prefix + "is_soft_deletes")),
            fi.isSoftDeletesField());
        checkBool(
            mismatches,
            "is_parent_field",
            Boolean.parseBoolean(manifest.get(prefix + "is_parent_field")),
            fi.isParentField());
        checkInt(
            mismatches,
            "point_dimension_count",
            Integer.parseInt(manifest.get(prefix + "point_dimension_count")),
            fi.getPointDimensionCount());
        checkInt(
            mismatches,
            "point_index_dimension_count",
            Integer.parseInt(manifest.get(prefix + "point_index_dimension_count")),
            fi.getPointIndexDimensionCount());
        checkInt(
            mismatches,
            "point_num_bytes",
            Integer.parseInt(manifest.get(prefix + "point_num_bytes")),
            fi.getPointNumBytes());
        checkInt(
            mismatches,
            "vector_dimension",
            Integer.parseInt(manifest.get(prefix + "vector_dimension")),
            fi.getVectorDimension());
        checkStr(
            mismatches,
            "vector_similarity",
            manifest.get(prefix + "vector_similarity"),
            fi.getVectorSimilarityFunction().toString());

        String expectedAttrs = manifest.getOrDefault(prefix + "attributes", "");
        String gotAttrs = renderAttributes(fi.attributes());
        if (!expectedAttrs.equals(gotAttrs)) {
          mismatches.append("attributes: expected=[").append(expectedAttrs)
              .append("] got=[").append(gotAttrs).append("]; ");
        }

        if (mismatches.length() > 0) {
          System.out.println("MISMATCH field " + name + ": " + mismatches);
          failures++;
        } else {
          System.out.println("field " + name + " OK");
        }
      }

      if (failures > 0) {
        System.out.println(failures + " mismatch(es)");
        System.exit(1);
      }
      System.out.println("All " + fieldCount + " fields verified against real Lucene. PASS");
    } catch (CorruptIndexException e) {
      System.out.println("FAILED TO OPEN: " + e);
      System.exit(1);
    }
  }

  static void checkInt(StringBuilder sb, String label, int expected, int got) {
    if (expected != got) {
      sb.append(label).append(": expected=").append(expected).append(" got=").append(got).append("; ");
    }
  }

  static void checkLong(StringBuilder sb, String label, long expected, long got) {
    if (expected != got) {
      sb.append(label).append(": expected=").append(expected).append(" got=").append(got).append("; ");
    }
  }

  static void checkBool(StringBuilder sb, String label, boolean expected, boolean got) {
    if (expected != got) {
      sb.append(label).append(": expected=").append(expected).append(" got=").append(got).append("; ");
    }
  }

  /**
   * Compares Rust's {@code Debug}-formatted enum name (e.g. {@code
   * "SortedSet"}) against Java's {@code toString()} (e.g. {@code
   * "SORTED_SET"}) case- and underscore-insensitively, since the two enums
   * use different naming conventions for the same wire value.
   */
  static void checkStr(StringBuilder sb, String label, String expected, String got) {
    if (!normalize(expected).equals(normalize(got))) {
      sb.append(label).append(": expected=").append(expected).append(" got=").append(got).append("; ");
    }
  }

  static String normalize(String s) {
    return s.replace("_", "").toLowerCase(java.util.Locale.ROOT);
  }

  static String renderAttributes(Map<String, String> attrs) {
    if (attrs.isEmpty()) return "";
    StringBuilder sb = new StringBuilder();
    for (Map.Entry<String, String> e : attrs.entrySet()) {
      if (sb.length() > 0) sb.append(';');
      sb.append(e.getKey()).append('=').append(e.getValue());
    }
    return sb.toString();
  }

  static Map<String, String> readManifest(Path path) throws IOException {
    Map<String, String> m = new HashMap<>();
    for (String line : Files.readAllLines(path)) {
      int eq = line.indexOf('=');
      if (eq < 0) continue;
      m.put(line.substring(0, eq), line.substring(eq + 1));
    }
    return m;
  }
}
