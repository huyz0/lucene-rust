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
 * Reverse-direction verifier (Rust writes, Java reads): opens a
 * `.nvm`/`.nvd` pair written by this port's `norms::write_single_dense_field`
 * (a single norms field, dense, at most 1 byte per doc -- see
 * `crates/lucene-codecs/examples/write_norms_fixture.rs`) directly through
 * real Lucene's {@link Lucene90NormsFormat}, using a hand-built
 * {@link SegmentInfo}/{@link FieldInfos} the same way
 * {@code VerifyDocValues.java} does -- this keeps the slice scoped to
 * exactly the norms format itself, no `.si`/`.fnm` writer needed.
 *
 * <p>Iterates the field via real {@link NumericDocValues} (the same API
 * {@link NormsProducer#getNorms} returns, and the production-facing way
 * scoring reads norms), and confirms every doc's value matches the
 * manifest.
 *
 * <p>Usage: {@code java VerifyNorms <fixture-dir>}, where
 * {@code <fixture-dir>} contains one {@code <segment>.nvm}/{@code
 * <segment>.nvd} pair per segment named in the manifest's {@code segments}
 * key, and a {@code manifest.properties} describing each segment's expected
 * per-doc values under {@code <segment>.max_doc}/{@code
 * <segment>.field_number}/{@code <segment>.values}. Exits nonzero and prints
 * a diff on any mismatch.
 *
 * <p>Two segments are verified: {@code _0} (varying small signed values,
 * exercising the real {@code bytesPerNorm == 1} path) and {@code _1}
 * (all-equal values, the {@code bytesPerNorm == 0} constant encoding -- a
 * regression case for a branch the doc-values write-side review found was
 * previously verified only against this port's own reader, not real
 * Lucene).
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
   * e.g. {@code "_0"}) through real Lucene and checks every doc's norm
   * value against the manifest under {@code segment + ".max_doc"} /
   * {@code segment + ".field_number"} / {@code segment + ".values"}.
   * Returns the number of mismatches (0 on full success).
   */
  static int verifySegment(Path dir, byte[] id, String segment, Map<String, String> manifest)
      throws IOException {
    int maxDoc = Integer.parseInt(manifest.get(segment + ".max_doc"));
    int fieldNumber = Integer.parseInt(manifest.get(segment + ".field_number"));
    String valuesSpec = manifest.getOrDefault(segment + ".values", "");

    List<Long> expected = new ArrayList<>();
    if (!valuesSpec.isEmpty()) {
      for (String v : valuesSpec.split(";")) {
        expected.add(Long.parseLong(v));
      }
    }

    FieldInfo fieldInfo =
        new FieldInfo(
            "body",
            fieldNumber,
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
    FieldInfos fis = new FieldInfos(new FieldInfo[] {fieldInfo});

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

      NumericDocValues values = producer.getNorms(fieldInfo);
      int failures = 0;
      int seenDocs = 0;
      for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
        seenDocs++;
        long got = values.longValue();
        if (doc >= expected.size()) {
          System.out.println(
              "MISMATCH "
                  + segment
                  + ": unexpected doc "
                  + doc
                  + " (expected only "
                  + expected.size()
                  + " docs)");
          failures++;
          continue;
        }
        long want = expected.get(doc);
        if (want != got) {
          System.out.println(
              "MISMATCH " + segment + " doc " + doc + ": expected=" + want + " got=" + got);
          failures++;
        }
      }

      if (seenDocs != expected.size()) {
        System.out.println(
            "MISMATCH "
                + segment
                + " doc count: expected="
                + expected.size()
                + " got="
                + seenDocs);
        failures++;
      }

      producer.close();

      if (failures == 0) {
        System.out.println(
            segment + ": all " + expected.size() + " doc norms verified against real Lucene");
      }
      return failures;
    } catch (CorruptIndexException e) {
      System.out.println(segment + " FAILED TO OPEN: " + e);
      return 1;
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
