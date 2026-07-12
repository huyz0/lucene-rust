import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.FieldInfosFormat;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.lucene90.Lucene90CompoundFormat;
import org.apache.lucene.codecs.lucene90.Lucene90StoredFieldsFormat;
import org.apache.lucene.codecs.lucene94.Lucene94FieldInfosFormat;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.StoredFieldVisitor;
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
import java.util.TreeSet;

/**
 * Reverse-direction verifier (Rust writes, Java reads): opens a `_0.cfs`/
 * `_0.cfe` pair written by this port's `compound_format::write` (see
 * `crates/lucene-codecs/examples/write_compound_format_fixture.rs`) directly
 * through real Lucene's {@link Lucene90CompoundFormat#getCompoundReader},
 * using a hand-built {@link SegmentInfo} -- same division of labor as the
 * other write-path verifiers in this directory.
 *
 * <p>Two checks, in order: (1) the compound reader's sub-file list and
 * lengths match the manifest exactly -- a directory-listing-level check; (2)
 * the packed `.fnm` and `.fdt`/`.fdx`/`.fdm` sub-files are independently
 * re-decoded through their OWN real Lucene formats (
 * {@link Lucene94FieldInfosFormat} and {@link Lucene90StoredFieldsFormat}
 * respectively), reading *through* the compound reader as the
 * {@link Directory}, not the raw sub-file bytes directly -- this is the
 * check that would catch a byte-offset bug that still left the entries
 * table "looking right".
 *
 * <p>Usage: {@code java VerifyCompoundFormat <fixture-dir>}, where
 * {@code <fixture-dir>} contains {@code _0.cfs}/{@code _0.cfe} and a
 * {@code manifest.properties} (see the Rust example above for the exact
 * format). Exits nonzero and prints a diff on any mismatch.
 */
public class VerifyCompoundFormat {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = readManifest(dir.resolve("manifest.properties"));

    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));
    String segmentName = manifest.get("segment_name");
    int maxDoc = Integer.parseInt(manifest.get("max_doc"));
    int numFields = Integer.parseInt(manifest.get("num_fields"));

    int failures = 0;

    try (Directory directory = FSDirectory.open(dir)) {
      Map<String, String> attributes = new HashMap<>();
      // Lucene90StoredFieldsFormat reads this attribute at open time to pick
      // BEST_SPEED vs BEST_COMPRESSION -- same requirement as
      // VerifyStoredFields.java.
      attributes.put(Lucene90StoredFieldsFormat.MODE_KEY, "BEST_SPEED");

      SegmentInfo si =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              segmentName,
              maxDoc,
              true,
              false,
              null,
              Collections.emptyMap(),
              id,
              attributes,
              null);

      CompoundDirectory cfsDir = new Lucene90CompoundFormat().getCompoundReader(directory, si);

      // Check 1: sub-file list/lengths, independent of how the bytes decode.
      TreeSet<String> expectedSubFiles = new TreeSet<>();
      Map<String, Long> expectedLengths = new HashMap<>();
      for (String entry : manifest.get("sub_files").split(",")) {
        int colon = entry.lastIndexOf(':');
        String name = entry.substring(0, colon);
        long length = Long.parseLong(entry.substring(colon + 1));
        expectedSubFiles.add(name);
        expectedLengths.put(name, length);
      }

      TreeSet<String> actualSubFiles = new TreeSet<>();
      for (String name : cfsDir.listAll()) {
        actualSubFiles.add(org.apache.lucene.index.IndexFileNames.stripSegmentName(name));
      }

      if (!expectedSubFiles.equals(actualSubFiles)) {
        System.out.println(
            "MISMATCH sub-file list: expected=" + expectedSubFiles + " got=" + actualSubFiles);
        failures++;
      }
      for (String name : cfsDir.listAll()) {
        String id2 = org.apache.lucene.index.IndexFileNames.stripSegmentName(name);
        Long expectedLength = expectedLengths.get(id2);
        long actualLength = cfsDir.fileLength(name);
        if (expectedLength == null) {
          continue; // already reported above
        }
        if (expectedLength != actualLength) {
          System.out.println(
              "MISMATCH sub-file length "
                  + id2
                  + ": expected="
                  + expectedLength
                  + " got="
                  + actualLength);
          failures++;
        }
      }

      // Check 2a: re-decode the packed .fnm through real
      // Lucene94FieldInfosFormat, reading through the compound directory.
      FieldInfosFormat fieldInfosFormat = new Lucene94FieldInfosFormat();
      FieldInfos fis = fieldInfosFormat.read(cfsDir, si, "", IOContext.DEFAULT);
      if (fis.size() != numFields) {
        System.out.println(
            "MISMATCH num_fields: expected=" + numFields + " got=" + fis.size());
        failures++;
      }

      // Check 2b: re-decode the packed .fdt/.fdx/.fdm through real
      // Lucene90StoredFieldsFormat, also through the compound directory.
      Lucene90StoredFieldsFormat storedFieldsFormat = new Lucene90StoredFieldsFormat();
      StoredFieldsReader reader =
          storedFieldsFormat.fieldsReader(cfsDir, si, fis, IOContext.DEFAULT);

      for (int doc = 0; doc < maxDoc; doc++) {
        String expectedLine = manifest.getOrDefault("doc." + doc + ".fields", "");
        DumpVisitor visitor = new DumpVisitor();
        reader.document(doc, visitor);
        String got = visitor.render();
        if (!got.equals(expectedLine)) {
          System.out.println(
              "MISMATCH doc " + doc + ": expected=[" + expectedLine + "] got=[" + got + "]");
          failures++;
        } else {
          System.out.println("doc " + doc + " OK: " + got);
        }
      }
      reader.close();
      cfsDir.close();

      if (failures > 0) {
        System.out.println(failures + " mismatch(es)");
        System.exit(1);
      }
      System.out.println("Compound format sub-files and packed contents verified against real Lucene. PASS");
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
