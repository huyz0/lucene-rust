import org.apache.lucene.document.Document;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.text.ParseException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Reverse-direction verifier (Rust merges, Java reads) for the two facts
 * {@code SegmentMerger} derives from its <b>readers</b> rather than from the
 * merging writer: {@code SegmentInfo.minVersion} and
 * {@code SegmentInfo.hasBlocks}.
 *
 * <p>The sources come from {@code fixtures/data/merge_metadata}, three real
 * Lucene segments whose {@code .si} files record differing {@code minVersion}s
 * (10.2.0, 10.0.0 and 10.1.0 as this is written) with {@code hasBlocks} set on
 * the middle one only. The expected answers are read from that fixture's
 * {@code manifest.properties}, which {@code GenMergeMetadata} computes from
 * the versions it actually stamped using Java's own {@code Version.onOrAfter}
 * -- so this class holds no constant that can go stale against the fixture.
 * Java's rules are:
 *
 * <ul>
 *   <li>{@code SegmentMerger}'s constructor: {@code minVersion} is the
 *       <b>smallest</b> across the readers ({@code null} if any reader has
 *       none) -- so 10.0.0 here.
 *   <li>{@code IndexWriter.mergeMiddle}: {@code hasBlocks} is the
 *       <b>disjunction</b> across the merged segments -- so true here.
 * </ul>
 *
 * <p>Both are read back through {@link org.apache.lucene.index.LeafMetaData},
 * which is the only place either surfaces. Neither affects a checksum, a
 * document or a query, so a merge that stamps the merging writer's own version
 * (what this port did before c36) produces an index that opens cleanly, reads
 * every document correctly and passes {@link CheckIndex} -- while claiming it
 * was never touched by the older Lucene whose bytes it is still carrying, and
 * while silently invalidating every parent/child join query if the block flag
 * is the one that was lost.
 *
 * <p>Usage: {@code java VerifyMergedMetadata <index-dir>}. Exits nonzero with a
 * diagnosis on any mismatch.
 */
public class VerifyMergedMetadata {
  public static void main(String[] args) throws IOException, ParseException {
    Path path = Path.of(args[0]);
    if (args.length < 2) {
      System.out.println(
          "usage: VerifyMergedMetadata <merged-index-dir> <source-fixture-dir>"
              + " (the second is fixtures/data/merge_metadata, whose manifest.properties"
              + " carries the expected answers)");
      System.exit(2);
    }
    // The expected answers are *not* restated here. `GenMergeMetadata` computes
    // them from the very versions it stamped into the source `.si` files, using
    // Java's own `Version.onOrAfter`, and writes them to the manifest -- so
    // editing the fixture's version list cannot leave this verifier checking a
    // stale constant. Ten of this suite's verifiers already work this way.
    Properties manifest = new Properties();
    try (var in = Files.newInputStream(Path.of(args[1]).resolve("manifest.properties"))) {
      manifest.load(in);
    }
    Version expectedMinVersion =
        Version.parse(require(manifest, "expected_merged_min_version"));
    boolean expectedHasBlocks =
        Boolean.parseBoolean(require(manifest, "expected_merged_has_blocks"));
    String[] expectedIds = require(manifest, "doc_ids").split(",");
    String sourceMinVersions = require(manifest, "segment_min_versions");

    int failures = 0;

    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.leaves().size() != 1) {
        System.out.println(
            "MISMATCH leaf count: " + reader.leaves().size() + " expected 1 (a single merged segment)");
        failures++;
      }
      if (reader.maxDoc() != expectedIds.length) {
        System.out.println(
            "MISMATCH maxDoc: " + reader.maxDoc() + " expected " + expectedIds.length);
        failures++;
      }

      for (LeafReaderContext ctx : reader.leaves()) {
        Version min = ctx.reader().getMetaData().minVersion();
        if (min == null || !min.equals(expectedMinVersion)) {
          System.out.println(
              "MISMATCH minVersion: merged segment reports "
                  + min
                  + " expected "
                  + expectedMinVersion
                  + " (the oldest of its sources: "
                  + sourceMinVersions
                  + ")");
          failures++;
        }
        boolean hasBlocks = ctx.reader().getMetaData().hasBlocks();
        if (hasBlocks != expectedHasBlocks) {
          System.out.println(
              "MISMATCH hasBlocks: merged segment reports "
                  + hasBlocks
                  + " expected "
                  + expectedHasBlocks
                  + " (one source was built with addDocuments)");
          failures++;
        }
      }

      // The metadata is only interesting if the documents actually survived
      // the merge: a merge that dropped or reordered them would otherwise
      // still report the right two flags.
      StoredFields stored = reader.storedFields();
      List<String> seen = new ArrayList<>();
      for (int doc = 0; doc < reader.maxDoc(); doc++) {
        Document d = stored.document(doc);
        seen.add(d.get("id"));
      }
      List<String> expected = List.of(expectedIds);
      if (!seen.equals(expected)) {
        System.out.println("MISMATCH merged documents: " + seen + " expected " + expected);
        failures++;
      }
    }

    try (Directory dir = FSDirectory.open(path);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8), false);
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      CheckIndex.Status status = checker.checkIndex();
      if (!status.clean) {
        System.out.println("CheckIndex reported problems:");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
        failures++;
      }
    }

    if (failures == 0) {
      System.out.println(
          "VerifyMergedMetadata: ok (minVersion="
              + expectedMinVersion
              + ", hasBlocks="
              + expectedHasBlocks
              + ", "
              + expectedIds.length
              + " documents)");
    }
    System.exit(failures == 0 ? 0 : 1);
  }

  /** A manifest key the fixture must carry -- a missing one is a broken fixture, not a pass. */
  private static String require(Properties manifest, String key) {
    String value = manifest.getProperty(key);
    if (value == null) {
      throw new IllegalStateException("manifest.properties has no " + key);
    }
    return value;
  }
}
