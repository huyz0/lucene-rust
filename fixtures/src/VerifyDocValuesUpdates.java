import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CheckIndex;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Reverse-direction verifier (Rust writes, Java reads) for <b>doc-values field
 * updates</b> -- {@code IndexWriter.updateNumericDocValue} /
 * {@code updateBinaryDocValue}.
 *
 * <p>This port used to write a delta file of its own invention here, so an
 * index carrying a doc-values update was one real Lucene could not open at
 * all. The format is now Lucene's: the updated field's <b>whole column</b>
 * rewritten into a new generation of ordinary {@code Lucene90DocValuesFormat}
 * files ({@code _0_1_Lucene90_0.dvm/.dvd/.dvs}), a new {@code FieldInfos}
 * generation ({@code _0_1.fnm}) recording {@code FieldInfo.docValuesGen}, and
 * {@code segments_N} entries for {@code docValuesGen} and the per-field
 * {@code dvUpdatesFiles} map.
 *
 * <p>What is checked, in the order the failures would show up:
 *
 * <ol>
 *   <li>Every document's value reads back as the value the <b>last</b> update
 *       round set for it -- through {@code DirectoryReader}, i.e. through
 *       {@code SegmentDocValuesProducer}'s per-field generation resolution.
 *       A wrong generation suffix, an unrecorded {@code fieldInfosGen} or a
 *       {@code docValuesGen} left at -1 all land here.
 *   <li>Documents the update rounds did <b>not</b> touch keep the value the
 *       base column gave them. A full-column rewrite that only wrote the
 *       updated docs reads back as "no value" for everything else, which no
 *       structural check would notice.
 *   <li>{@code SegmentCommitInfo} agrees with the directory: the recorded
 *       {@code docValuesGen}/{@code fieldInfosGen} files exist, and every
 *       <b>superseded</b> generation is gone -- a generation is a complete
 *       column, so keeping the previous one referenced leaks files forever.
 *   <li>{@code CheckIndex} at {@code MIN_LEVEL_FOR_SLOW_CHECKS}, which
 *       cross-checks every file listed for the segment (base column included)
 *       against its checksum and walks the doc-values themselves.
 * </ol>
 *
 * <p>Usage: {@code java VerifyDocValuesUpdates <output-dir>}, where the
 * directory holds the {@code numeric/} and {@code binary/} indices
 * {@code write_doc_values_updates_fixture.rs} writes. Exits nonzero with a
 * diagnosis on any mismatch.
 */
public class VerifyDocValuesUpdates {
  /** Must match `write_doc_values_updates_fixture.rs`. */
  private static final int NUM_DOCS = 260;

  private static final long EVEN_FINAL = 7_000L;
  private static final long ODD_FINAL = 9_000L;
  private static final String BINARY_UPDATED = "updated-payload";

  /**
   * Documents {@code 0..NUM_RESET} had their {@code val} <b>removed</b> in the
   * last round ({@code updateDocValues} with a null value ==
   * {@code DocValuesFieldUpdates.reset}). Those documents must read back as
   * having <b>no value</b> -- which is a different claim from reading back a
   * wrong one, and the only thing that exercises the sparse/empty
   * {@code IndexedDISI} shape of a rewritten column.
   */
  private static final int NUM_RESET = 40;

  public static void main(String[] args) throws IOException {
    Path root = Path.of(args[0]);
    int failures = 0;

    failures += verifyNumeric(root.resolve("numeric"));
    failures += verifyBinary(root.resolve("binary"));

    if (failures == 0) {
      System.out.println("OK: doc-values updates read back through real Lucene");
      return;
    }
    System.out.println("FAILURES: " + failures);
    System.exit(1);
  }

  private static int verifyNumeric(Path path) throws IOException {
    int failures = 0;
    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      if (reader.maxDoc() != NUM_DOCS) {
        System.out.println("MISMATCH numeric maxDoc=" + reader.maxDoc() + " expected " + NUM_DOCS);
        failures++;
      }
      for (LeafReaderContext ctx : reader.leaves()) {
        NumericDocValues values = ctx.reader().getNumericDocValues("val");
        if (values == null) {
          System.out.println(
              "MISSING numeric doc values for field `val` -- the update generation was written"
                  + " but nothing resolves to it (FieldInfo.docValuesGen / fieldInfosGen)");
          failures++;
          continue;
        }
        int seen = 0;
        for (int doc = values.nextDoc();
            doc != DocIdSetIterator.NO_MORE_DOCS;
            doc = values.nextDoc()) {
          int global = ctx.docBase + doc;
          if (global < NUM_RESET) {
            System.out.println(
                "MISMATCH val for doc "
                    + global
                    + ": has a value ("
                    + values.longValue()
                    + ") but the last update round removed it");
            failures++;
            if (failures > 5) {
              return failures;
            }
            continue;
          }
          long expected = (global % 2 == 0) ? EVEN_FINAL : ODD_FINAL;
          long actual = values.longValue();
          if (actual != expected) {
            System.out.println(
                "MISMATCH val for doc " + global + ": " + actual + " expected " + expected);
            failures++;
            if (failures > 5) {
              return failures;
            }
          }
          seen++;
        }
        int expectedPresent = ctx.reader().maxDoc() - NUM_RESET;
        if (seen != expectedPresent) {
          System.out.println(
              "MISMATCH numeric column density: "
                  + seen
                  + " documents have a value, expected "
                  + expectedPresent
                  + " -- a full-column rewrite must carry forward the documents the"
                  + " update did not touch, and must drop the ones it reset");
          failures++;
        }
      }

      // The `FieldInfo` a reader actually sees must name a generation.
      for (LeafReaderContext ctx : reader.leaves()) {
        FieldInfo fi = ctx.reader().getFieldInfos().fieldInfo("val");
        if (fi == null || fi.getDocValuesGen() == -1) {
          System.out.println(
              "MISMATCH FieldInfo.docValuesGen for `val` is -1: the reader is on the base"
                  + " FieldInfos, not the generational one");
          failures++;
        }
      }
    }

    failures += verifyGenerationBookkeeping(path, "val", 4);
    failures += checkIndex(path);
    return failures;
  }

  private static int verifyBinary(Path path) throws IOException {
    int failures = 0;
    try (Directory dir = FSDirectory.open(path);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      for (LeafReaderContext ctx : reader.leaves()) {
        BinaryDocValues values = ctx.reader().getBinaryDocValues("tag");
        if (values == null) {
          System.out.println("MISSING binary doc values for field `tag`");
          failures++;
          continue;
        }
        int seen = 0;
        for (int doc = values.nextDoc();
            doc != DocIdSetIterator.NO_MORE_DOCS;
            doc = values.nextDoc()) {
          int global = ctx.docBase + doc;
          String expected = (global % 2 == 0) ? BINARY_UPDATED : "base-" + global;
          BytesRef v = values.binaryValue();
          String actual = new String(v.bytes, v.offset, v.length, StandardCharsets.UTF_8);
          if (!actual.equals(expected)) {
            System.out.println(
                "MISMATCH tag for doc " + global + ": `" + actual + "` expected `" + expected + "`");
            failures++;
            if (failures > 5) {
              return failures;
            }
          }
          seen++;
        }
        if (seen != ctx.reader().maxDoc()) {
          System.out.println(
              "MISMATCH binary column density: " + seen + " expected " + ctx.reader().maxDoc());
          failures++;
        }
      }
    }

    failures += verifyGenerationBookkeeping(path, "tag", 1);
    failures += checkIndex(path);
    return failures;
  }

  /**
   * `segments_N` must record the generation the files on disk actually are, and
   * nothing may still reference a superseded one.
   */
  private static int verifyGenerationBookkeeping(Path path, String field, long expectedGen)
      throws IOException {
    int failures = 0;
    try (Directory dir = FSDirectory.open(path)) {
      SegmentInfos sis = SegmentInfos.readLatestCommit(dir);
      for (SegmentCommitInfo sci : sis) {
        if (sci.getDocValuesGen() != expectedGen) {
          System.out.println(
              "MISMATCH docValuesGen="
                  + sci.getDocValuesGen()
                  + " expected "
                  + expectedGen
                  + " for segment "
                  + sci.info.name);
          failures++;
        }
        if (sci.getFieldInfosGen() == -1 || sci.getFieldInfosFiles().isEmpty()) {
          System.out.println("MISMATCH no FieldInfos generation recorded for " + sci.info.name);
          failures++;
        }
        List<String> recorded = new ArrayList<>();
        for (Set<String> files : sci.getDocValuesUpdatesFiles().values()) {
          recorded.addAll(files);
        }
        recorded.addAll(sci.getFieldInfosFiles());
        if (recorded.isEmpty()) {
          System.out.println("MISMATCH no doc-values update files recorded for " + field);
          failures++;
        }
        for (String name : recorded) {
          if (!Files.exists(path.resolve(name))) {
            System.out.println("MISSING recorded file " + name);
            failures++;
          }
        }
        // Every superseded generation must be gone from the directory.
        for (long gen = 1; gen < expectedGen; gen++) {
          String stale = sci.info.name + "_" + Long.toString(gen, Character.MAX_RADIX) + "_";
          for (String name : dir.listAll()) {
            if (name.startsWith(stale)) {
              System.out.println(
                  "LEAKED superseded generation file " + name + " (generation " + gen + ")");
              failures++;
            }
          }
        }
      }
    }
    return failures;
  }

  private static int checkIndex(Path path) throws IOException {
    try (Directory dir = FSDirectory.open(path);
        CheckIndex checker = new CheckIndex(dir)) {
      ByteArrayOutputStream captured = new ByteArrayOutputStream();
      checker.setInfoStream(new PrintStream(captured, true, StandardCharsets.UTF_8), false);
      checker.setLevel(CheckIndex.Level.MIN_LEVEL_FOR_SLOW_CHECKS);
      CheckIndex.Status status = checker.checkIndex();
      if (!status.clean) {
        System.out.println("CheckIndex reported problems for " + path + ":");
        System.out.println(captured.toString(StandardCharsets.UTF_8));
        return 1;
      }
    }
    return 0;
  }
}
