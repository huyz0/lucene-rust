import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Reverse-direction verifier (Rust writes, Java reads) for the write side of
 * {@code SegmentInfos} -- the {@code segments_N} commit file format -- and,
 * transitively, everything it references: {@code .si}, {@code .fnm}, and
 * {@code .fdt}/{@code .fdx}/{@code .fdm}, all written by
 * {@code crates/lucene-index/examples/write_segment_infos_fixture.rs}.
 *
 * <p>Unlike {@code VerifyStoredFields}/{@code VerifyFieldInfos}/
 * {@code VerifySegmentInfo}, which each open exactly one format directly
 * through its own codec-level reader (with hand-built {@code SegmentInfo}/
 * {@code FieldInfos} scaffolding for whichever pieces weren't the thing
 * being tested), this verifier is the first to open the fixture through
 * real, high-level {@code DirectoryReader.open(FSDirectory.open(path))} --
 * ordinary application code, no codec internals in this file at all -- and
 * asserts document count and stored field values via {@code IndexReader}/
 * {@code StoredFields}, the normal way a real application reads an index.
 * That this succeeds is the actual milestone: proof that a Rust-written
 * index is a real Lucene index, not merely bytes that individually parse.
 *
 * <p>Usage: {@code java VerifySegmentInfos <fixture-dir>}, where
 * {@code <fixture-dir>} is the directory passed to
 * {@code write_segment_infos_fixture} (containing {@code segments_N},
 * {@code _0.si}, {@code _0.fnm}, {@code _0.fdt}/{@code .fdx}/{@code .fdm},
 * and {@code manifest.properties}). Exits nonzero and prints a diff on any
 * mismatch.
 */
public class VerifySegmentInfos {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = readManifest(dir.resolve("manifest.properties"));

    int maxDoc = Integer.parseInt(manifest.get("max_doc"));
    StringBuilder mismatches = new StringBuilder();

    try (Directory directory = FSDirectory.open(dir);
        IndexReader reader = DirectoryReader.open(directory)) {
      checkInt(mismatches, "numDocs", maxDoc, reader.numDocs());
      checkInt(mismatches, "maxDoc", maxDoc, reader.maxDoc());

      StoredFields storedFields = reader.storedFields();
      for (int docId = 0; docId < maxDoc; docId++) {
        Document doc = storedFields.document(docId);
        String expectedId = manifest.get("doc." + docId + ".id");
        String expectedBody = manifest.get("doc." + docId + ".body");
        String gotId = doc.get("id");
        String gotBody = doc.get("body");
        if (!expectedId.equals(gotId)) {
          mismatches.append("doc ").append(docId).append(" id: expected=").append(expectedId)
              .append(" got=").append(gotId).append("; ");
        }
        if (!expectedBody.equals(gotBody)) {
          mismatches.append("doc ").append(docId).append(" body: expected=").append(expectedBody)
              .append(" got=").append(gotBody).append("; ");
        }
      }
    } catch (Exception e) {
      System.out.println("FAILED TO OPEN via DirectoryReader.open: " + e);
      e.printStackTrace();
      System.exit(1);
      return;
    }

    if (mismatches.length() > 0) {
      System.out.println("MISMATCH: " + mismatches);
      System.exit(1);
    }
    System.out.println(
        "DirectoryReader.open succeeded; " + maxDoc + " doc(s) verified against real Lucene. PASS");
  }

  static void checkInt(StringBuilder sb, String label, int expected, int got) {
    if (expected != got) {
      sb.append(label).append(": expected=").append(expected).append(" got=").append(got).append("; ");
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
}
