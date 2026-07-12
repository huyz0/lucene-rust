import org.apache.lucene.codecs.lucene90.Lucene90LiveDocsFormat;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;

/**
 * Reverse-direction verifier (Rust writes, Java reads): opens `.liv` files
 * written by this port's `live_docs::write` directly through real Lucene's
 * {@link Lucene90LiveDocsFormat}, using a hand-built {@link SegmentInfo}/
 * {@link SegmentCommitInfo} (no `.si`/`segments_N` writer needed -- deletions
 * aren't per-field, so no {@code FieldInfos} is needed either).
 *
 * <p>Iterates every doc id via real {@link Bits#get(int)} (the same API
 * {@code LiveDocsFormat.readLiveDocs} callers use) and confirms live/deleted
 * status matches the manifest.
 *
 * <p>Usage: {@code java VerifyLiveDocs <fixture-dir>}, where
 * {@code <fixture-dir>} contains one {@code <segment>_<gen36>.liv} file per
 * segment named in the manifest's {@code segments} key, and a {@code
 * manifest.properties} describing each segment's expected shape under
 * {@code <segment>.liv_file_name}/{@code <segment>.max_doc}/{@code
 * <segment>.del_gen}/{@code <segment>.del_count}/{@code
 * <segment>.deleted_doc_ids}. Exits nonzero and prints a diff on any
 * mismatch.
 */
public class VerifyLiveDocs {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = readManifest(dir.resolve("manifest.properties"));
    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));

    int failures = 0;
    for (String segment : manifest.get("segments").split(",")) {
      failures += verifySegment(dir, id, segment, manifest);
    }

    if (failures > 0) {
      System.out.println(failures + " mismatch(es) overall");
      System.exit(1);
    }
    System.out.println("All segments verified against real Lucene. PASS");
  }

  static int verifySegment(Path dir, byte[] id, String segment, Map<String, String> manifest)
      throws IOException {
    int maxDoc = Integer.parseInt(manifest.get(segment + ".max_doc"));
    long delGen = Long.parseLong(manifest.get(segment + ".del_gen"));
    int expectedDelCount = Integer.parseInt(manifest.get(segment + ".del_count"));
    String deletedSpec = manifest.getOrDefault(segment + ".deleted_doc_ids", "");

    List<Integer> deleted = new ArrayList<>();
    if (!deletedSpec.isEmpty()) {
      for (String v : deletedSpec.split(",")) {
        deleted.add(Integer.parseInt(v));
      }
    }

    try (Directory directory = FSDirectory.open(dir)) {
      SegmentInfo si =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              segment,
              maxDoc,
              false,
              false,
              null,
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);

      SegmentCommitInfo sci =
          new SegmentCommitInfo(si, expectedDelCount, 0, delGen, -1, -1, id);

      Lucene90LiveDocsFormat format = new Lucene90LiveDocsFormat();
      Bits liveDocs = format.readLiveDocs(directory, sci, IOContext.DEFAULT);

      int failures = 0;
      for (int doc = 0; doc < maxDoc; doc++) {
        boolean expectedLive = !deleted.contains(doc);
        boolean actualLive = liveDocs.get(doc);
        if (expectedLive != actualLive) {
          System.out.println(
              "MISMATCH "
                  + segment
                  + " doc "
                  + doc
                  + ": expected live="
                  + expectedLive
                  + " got live="
                  + actualLive);
          failures++;
        }
      }

      if (failures == 0) {
        System.out.println(
            segment + ": all " + maxDoc + " doc live/deleted states verified against real Lucene");
      }
      return failures;
    }
  }

  static Map<String, String> readManifest(Path path) throws IOException {
    Map<String, String> m = new HashMap<>();
    for (String line : Files.readAllLines(path)) {
      if (line.isBlank()) continue;
      int idx = line.indexOf('=');
      m.put(line.substring(0, idx), line.substring(idx + 1));
    }
    return m;
  }
}
