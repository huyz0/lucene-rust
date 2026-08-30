import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.lucene90.Lucene90NormsFormat;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;

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
 * Reverse-direction verifier (Rust writes, Java reads): opens `.nvm`/`.nvd`
 * pairs written by this port's `norms::write_fields` (see
 * `crates/lucene-codecs/examples/write_norms_fixture.rs`) directly through
 * real Lucene's {@link Lucene90NormsFormat}, using a hand-built
 * {@link SegmentInfo}/{@link FieldInfos} the same way
 * {@code VerifyDocValues.java} does -- this keeps the slice scoped to
 * exactly the norms format itself, no `.si`/`.fnm` writer needed.
 *
 * <p>Iterates each field via real {@link NumericDocValues} (the same API
 * {@link NormsProducer#getNorms} returns, and the production-facing way
 * scoring reads norms), and confirms every doc's value matches the
 * manifest.
 *
 * <p>Usage: {@code java VerifyNorms <fixture-dir>}, where
 * {@code <fixture-dir>} contains one {@code <segment>.nvm}/{@code
 * <segment>.nvd} pair per segment named in the manifest's {@code segments}
 * key, and a {@code manifest.properties} describing each segment under
 * {@code <segment>.max_doc}, {@code <segment>.field_numbers} (comma
 * separated) and, per field, {@code <segment>.<number>.values} -- a
 * {@code ;}-separated positional list where {@code -} means the doc has no
 * norm. Exits nonzero and prints a diff on any mismatch.
 *
 * <p>The segments cover every shape {@code Lucene90NormsConsumer} can write:
 * all five {@code numBytesPerValue} widths (constant, 1, 2, 4, 8 bytes per
 * doc), the sparse {@code IndexedDISI} docs-with-field structure, and several
 * fields interleaved into one {@code .nvm}/{@code .nvd} pair.
 */
public class VerifyNorms {
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

  /**
   * Opens one Rust-written `.nvm`/`.nvd` segment (named {@code segment},
   * e.g. {@code "_0"}) through real Lucene and checks every field listed in
   * {@code segment + ".field_numbers"} against its expected per-doc values
   * at {@code segment + "." + number + ".values"} ({@code -} for a doc with
   * no norm). Returns the number of mismatches (0 on full success).
   */
  static int verifySegment(Path dir, byte[] id, String segment, Map<String, String> manifest)
      throws IOException {
    int maxDoc = Integer.parseInt(manifest.get(segment + ".max_doc"));
    String[] fieldNumbers = manifest.get(segment + ".field_numbers").split(",");

    List<FieldInfo> fieldInfoList = new ArrayList<>();
    Map<Integer, List<Long>> expectedByField = new HashMap<>();
    for (String numberText : fieldNumbers) {
      int fieldNumber = Integer.parseInt(numberText.trim());
      fieldInfoList.add(normedField("field" + fieldNumber, fieldNumber));

      List<Long> expected = new ArrayList<>();
      String valuesSpec = manifest.getOrDefault(segment + "." + fieldNumber + ".values", "");
      if (!valuesSpec.isEmpty()) {
        for (String v : valuesSpec.split(";")) {
          // `-` means the doc has no norm at all (the sparse shape).
          expected.add(v.equals("-") ? null : Long.parseLong(v));
        }
      }
      expectedByField.put(fieldNumber, expected);
    }
    FieldInfos fis = new FieldInfos(fieldInfoList.toArray(new FieldInfo[0]));

    try (Directory directory = FSDirectory.open(dir)) {
      SegmentInfo si =
          new SegmentInfo(
              directory,
              org.apache.lucene.util.Version.LATEST,
              org.apache.lucene.util.Version.LATEST,
              segment,
              maxDoc,
              false,
              false,
              null,
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);

      Lucene90NormsFormat format = new Lucene90NormsFormat();
      SegmentReadState readState = new SegmentReadState(directory, si, fis, IOContext.DEFAULT);
      NormsProducer producer = format.normsProducer(readState);

      int failures = 0;
      for (FieldInfo fieldInfo : fieldInfoList) {
        List<Long> expected = expectedByField.get(fieldInfo.number);
        NumericDocValues values = producer.getNorms(fieldInfo);
        int seenDocs = 0;
        int expectedDocs = 0;
        for (Long v : expected) {
          if (v != null) {
            expectedDocs++;
          }
        }
        for (int doc = values.nextDoc();
            doc != DocIdSetIterator.NO_MORE_DOCS;
            doc = values.nextDoc()) {
          seenDocs++;
          long got = values.longValue();
          if (doc >= expected.size() || expected.get(doc) == null) {
            System.out.println(
                "MISMATCH "
                    + segment
                    + " field "
                    + fieldInfo.number
                    + ": doc "
                    + doc
                    + " should have no norm, got "
                    + got);
            failures++;
            continue;
          }
          long want = expected.get(doc);
          if (want != got) {
            System.out.println(
                "MISMATCH "
                    + segment
                    + " field "
                    + fieldInfo.number
                    + " doc "
                    + doc
                    + ": expected="
                    + want
                    + " got="
                    + got);
            failures++;
          }
        }

        if (seenDocs != expectedDocs) {
          System.out.println(
              "MISMATCH "
                  + segment
                  + " field "
                  + fieldInfo.number
                  + " doc count: expected="
                  + expectedDocs
                  + " got="
                  + seenDocs);
          failures++;
        } else if (failures == 0) {
          System.out.println(
              segment
                  + " field "
                  + fieldInfo.number
                  + ": all "
                  + expectedDocs
                  + " doc norms verified against real Lucene");
        }
      }

      producer.close();
      return failures;
    } catch (CorruptIndexException e) {
      System.out.println(segment + " FAILED TO OPEN: " + e);
      return 1;
    }
  }

  /** A {@link FieldInfo} for an indexed field that has norms (omitNorms == false). */
  static FieldInfo normedField(String name, int number) {
    return new FieldInfo(
        name,
        number,
        false, // storeTermVector
        false, // omitNorms == false -> field DOES have norms
        false, // storePayloads
        IndexOptions.DOCS,
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
