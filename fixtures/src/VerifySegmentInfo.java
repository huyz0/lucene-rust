import org.apache.lucene.codecs.lucene99.Lucene99SegmentInfoFormat;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;

/**
 * Reverse-direction verifier (Rust writes, Java reads): opens a {@code .si}
 * file written by this port's {@code segment_info::write} (see
 * {@code crates/lucene-index/examples/write_segment_info_fixture.rs}) directly
 * through real Lucene's {@link Lucene99SegmentInfoFormat#read}, then checks
 * every property against a per-segment {@code <name>.manifest.properties}.
 *
 * <p>Usage: {@code java VerifySegmentInfo <fixture-dir>}, where
 * {@code <fixture-dir>} contains one or more {@code <name>.si} +
 * {@code <name>.manifest.properties} pairs (see the Rust example above for
 * the exact manifest format and the segment names it writes). Exits nonzero
 * and prints a diff on any mismatch.
 */
public class VerifySegmentInfo {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    int totalFailures = 0;
    int segmentsChecked = 0;

    try (Directory directory = FSDirectory.open(dir)) {
      for (Path manifestPath : findManifests(dir)) {
        String fileName = manifestPath.getFileName().toString();
        String segmentName = fileName.substring(0, fileName.length() - ".manifest.properties".length());
        totalFailures += verifySegment(directory, segmentName, manifestPath);
        segmentsChecked++;
      }
    }

    if (segmentsChecked == 0) {
      System.out.println("no fixtures found under " + dir);
      System.exit(1);
    }
    if (totalFailures > 0) {
      System.out.println(totalFailures + " mismatch(es) across " + segmentsChecked + " segment(s)");
      System.exit(1);
    }
    System.out.println("All " + segmentsChecked + " segment(s) verified against real Lucene. PASS");
  }

  static int verifySegment(Directory directory, String segmentName, Path manifestPath) throws IOException {
    Map<String, String> manifest = readManifest(manifestPath);
    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));

    int failures = 0;
    SegmentInfo si;
    try {
      Lucene99SegmentInfoFormat format = new Lucene99SegmentInfoFormat();
      si = format.read(directory, segmentName, id, IOContext.DEFAULT);
    } catch (CorruptIndexException e) {
      System.out.println("FAILED TO OPEN " + segmentName + ": " + e);
      return 1;
    }

    StringBuilder mismatches = new StringBuilder();

    checkInt(mismatches, "version_major", Integer.parseInt(manifest.get("version_major")), si.getVersion().major);
    checkInt(mismatches, "version_minor", Integer.parseInt(manifest.get("version_minor")), si.getVersion().minor);
    checkInt(mismatches, "version_bugfix", Integer.parseInt(manifest.get("version_bugfix")), si.getVersion().bugfix);

    boolean expectMinVersion = Integer.parseInt(manifest.get("has_min_version")) != 0;
    boolean gotMinVersion = si.getMinVersion() != null;
    checkBool(mismatches, "has_min_version", expectMinVersion, gotMinVersion);
    if (expectMinVersion && gotMinVersion) {
      checkInt(
          mismatches,
          "min_version_major",
          Integer.parseInt(manifest.get("min_version_major")),
          si.getMinVersion().major);
      checkInt(
          mismatches,
          "min_version_minor",
          Integer.parseInt(manifest.get("min_version_minor")),
          si.getMinVersion().minor);
      checkInt(
          mismatches,
          "min_version_bugfix",
          Integer.parseInt(manifest.get("min_version_bugfix")),
          si.getMinVersion().bugfix);
    }

    checkInt(mismatches, "doc_count", Integer.parseInt(manifest.get("doc_count")), si.maxDoc());
    checkBool(
        mismatches,
        "is_compound_file",
        Integer.parseInt(manifest.get("is_compound_file")) != 0,
        si.getUseCompoundFile());

    String expectedDiagnostics = sortEntries(manifest.getOrDefault("diagnostics", ""));
    String gotDiagnostics = sortEntries(renderMap(si.getDiagnostics()));
    if (!expectedDiagnostics.equals(gotDiagnostics)) {
      mismatches.append("diagnostics: expected=[").append(expectedDiagnostics)
          .append("] got=[").append(gotDiagnostics).append("]; ");
    }

    String expectedAttributes = sortEntries(manifest.getOrDefault("attributes", ""));
    String gotAttributes = sortEntries(renderMap(si.getAttributes()));
    if (!expectedAttributes.equals(gotAttributes)) {
      mismatches.append("attributes: expected=[").append(expectedAttributes)
          .append("] got=[").append(gotAttributes).append("]; ");
    }

    String expectedFiles = manifest.getOrDefault("files", "");
    String gotFiles = String.join(",", new java.util.TreeSet<>(si.files()));
    String expectedFilesSorted = expectedFiles.isEmpty()
        ? ""
        : String.join(",", new java.util.TreeSet<>(java.util.List.of(expectedFiles.split(","))));
    if (!expectedFilesSorted.equals(gotFiles)) {
      mismatches.append("files: expected=[").append(expectedFilesSorted)
          .append("] got=[").append(gotFiles).append("]; ");
    }

    if (mismatches.length() > 0) {
      System.out.println("MISMATCH segment " + segmentName + ": " + mismatches);
      failures++;
    } else {
      System.out.println("segment " + segmentName + " OK");
    }
    return failures;
  }

  static java.util.List<Path> findManifests(Path dir) throws IOException {
    try (var stream = Files.list(dir)) {
      return stream
          .filter(p -> p.getFileName().toString().endsWith(".manifest.properties"))
          .sorted()
          .toList();
    }
  }

  static void checkInt(StringBuilder sb, String label, int expected, int got) {
    if (expected != got) {
      sb.append(label).append(": expected=").append(expected).append(" got=").append(got).append("; ");
    }
  }

  static void checkBool(StringBuilder sb, String label, boolean expected, boolean got) {
    if (expected != got) {
      sb.append(label).append(": expected=").append(expected).append(" got=").append(got).append("; ");
    }
  }

  static String sortEntries(String joined) {
    if (joined.isEmpty()) return "";
    String[] parts = joined.split(";");
    java.util.Arrays.sort(parts);
    return String.join(";", parts);
  }

  static String renderMap(Map<String, String> m) {
    if (m == null || m.isEmpty()) return "";
    StringBuilder sb = new StringBuilder();
    for (Map.Entry<String, String> e : m.entrySet()) {
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
