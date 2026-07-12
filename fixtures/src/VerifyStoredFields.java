import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.StoredFieldVisitor;
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
 * Reverse-direction verifier (Rust writes, Java reads): opens a
 * `.fdt`/`.fdx`/`.fdm` triple written by this port's
 * `stored_fields::write_best_speed` (see
 * `crates/lucene-codecs/examples/write_stored_fields_fixture.rs`) directly
 * through real Lucene's {@code Lucene90StoredFieldsFormat}, using a
 * hand-built {@link SegmentInfo}/{@link FieldInfos} rather than also
 * requiring Rust to write {@code .si}/{@code .fnm} -- this keeps the first
 * write-path slice scoped to exactly the stored-fields format itself.
 *
 * <p>Usage: {@code java VerifyStoredFields <fixture-dir>}, where
 * {@code <fixture-dir>} contains {@code _0.fdt}/{@code _0.fdx}/
 * {@code _0.fdm} and a {@code manifest.properties} (see the Rust example
 * above for the exact format) describing the expected per-doc field values.
 * Exits nonzero and prints a diff on any mismatch.
 */
public class VerifyStoredFields {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = readManifest(dir.resolve("manifest.properties"));

    int maxDoc = Integer.parseInt(manifest.get("max_doc"));
    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));
    int numFields = Integer.parseInt(manifest.get("num_fields"));

    FieldInfo[] fieldInfos = new FieldInfo[numFields];
    for (int i = 0; i < numFields; i++) {
      fieldInfos[i] =
          new FieldInfo(
              "field" + i,
              i,
              false,
              false,
              false,
              IndexOptions.NONE,
              DocValuesType.NONE,
              DocValuesSkipIndexType.NONE,
              -1,
              new HashMap<>(),
              0,
              0,
              0,
              0,
              VectorEncoding.FLOAT32,
              VectorSimilarityFunction.EUCLIDEAN,
              false,
              false);
    }
    FieldInfos fis = new FieldInfos(fieldInfos);

    try (Directory directory = FSDirectory.open(dir)) {
      Map<String, String> attributes = new HashMap<>();
      // Lucene90StoredFieldsFormat reads this attribute at open time to pick
      // BEST_SPEED vs BEST_COMPRESSION -- it's not derived from the .fdt
      // codec name by this wrapper class (unlike this port's own reader,
      // which peeks the codec name directly).
      attributes.put(Lucene90StoredFieldsFormat.MODE_KEY, "BEST_SPEED");

      SegmentInfo si =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              "_0",
              maxDoc,
              false,
              false,
              new Lucene104Codec(),
              Collections.emptyMap(),
              id,
              attributes,
              null);

      Lucene90StoredFieldsFormat format = new Lucene90StoredFieldsFormat();
      org.apache.lucene.codecs.StoredFieldsReader reader =
          format.fieldsReader(directory, si, fis, IOContext.DEFAULT);

      int failures = 0;
      for (int doc = 0; doc < maxDoc; doc++) {
        String expectedLine = manifest.getOrDefault("doc." + doc + ".fields", "");
        DumpVisitor visitor = new DumpVisitor();
        reader.document(doc, visitor);
        String got = visitor.render();
        if (!got.equals(expectedLine)) {
          System.out.println("MISMATCH doc " + doc + ": expected=[" + expectedLine + "] got=[" + got + "]");
          failures++;
        } else {
          System.out.println("doc " + doc + " OK: " + got);
        }
      }
      reader.close();

      if (failures > 0) {
        System.out.println(failures + " document(s) mismatched");
        System.exit(1);
      }
      System.out.println("All " + maxDoc + " documents verified against real Lucene. PASS");
    } catch (CorruptIndexException e) {
      System.out.println("FAILED TO OPEN: " + e);
      System.exit(1);
    }
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

  /** Renders each visited field as `number:type:value`, joined by ';' -- same shape the Rust example writes to its manifest. */
  static class DumpVisitor extends StoredFieldVisitor {
    private final StringBuilder sb = new StringBuilder();

    private void sep() {
      if (sb.length() > 0) sb.append(';');
    }

    String render() {
      return sb.toString();
    }

    @Override
    public Status needsField(FieldInfo fieldInfo) {
      return Status.YES;
    }

    @Override
    public void stringField(FieldInfo fieldInfo, String value) {
      sep();
      sb.append(fieldInfo.number).append(":string:").append(value);
    }

    @Override
    public void binaryField(FieldInfo fieldInfo, byte[] value) {
      sep();
      sb.append(fieldInfo.number).append(":binary:").append(HexFormat.of().formatHex(value));
    }

    @Override
    public void intField(FieldInfo fieldInfo, int value) {
      sep();
      sb.append(fieldInfo.number).append(":int:").append(value);
    }

    @Override
    public void longField(FieldInfo fieldInfo, long value) {
      sep();
      sb.append(fieldInfo.number).append(":long:").append(value);
    }

    @Override
    public void floatField(FieldInfo fieldInfo, float value) {
      sep();
      sb.append(fieldInfo.number).append(":float:").append(value);
    }

    @Override
    public void doubleField(FieldInfo fieldInfo, double value) {
      sep();
      sb.append(fieldInfo.number).append(":double:").append(value);
    }
  }
}
