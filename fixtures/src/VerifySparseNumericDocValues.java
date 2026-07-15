import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
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
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;
import java.util.TreeMap;

/**
 * Reverse-direction verifier (Rust writes, Java reads): opens the
 * `.dvm`/`.dvd`/`.dvs` triple written by this port's
 * `doc_values::write_single_sparse_numeric_field`
 * (`crates/lucene-codecs/src/doc_values.rs`, exercised by
 * `crates/lucene-codecs/examples/write_sparse_numeric_doc_values_fixture.rs`)
 * directly through real Lucene's {@link Lucene90DocValuesFormat}, using a
 * hand-built {@link SegmentInfo}/{@link FieldInfos} the same way {@code
 * VerifyDocValues.java} does for the dense writers -- this keeps the slice
 * scoped to exactly the sparse NUMERIC doc-values format, no `.si`/`.fnm`
 * writer needed.
 *
 * <p>Unlike the dense verifier, this one does not just iterate every
 * present doc via {@code nextDoc()} -- it walks every doc id from 0 to
 * {@code max_doc - 1} and calls real {@link NumericDocValues#advanceExact}
 * for each, confirming docs with a value return {@code true} and the
 * correct value via {@code longValue()}, and docs without a value in the
 * manifest correctly report {@code false} -- this is the property that
 * actually matters for a sparse field (real Lucene's {@code IndexedDISI}
 * must correctly distinguish "no value" from "value present" for every doc,
 * not just skip past them during forward iteration).
 *
 * <p>Two segments are verified, both previously checked only against this
 * port's own reader (`write_single_sparse_numeric_field_round_trips_through_
 * own_reader` in `doc_values.rs`), never against real Lucene: {@code _0}
 * (20 docs, values missing on docs interspersed throughout -- not just
 * trailing) and {@code _1} (200,000 docs, 1 of every 3 present, forcing
 * {@code IndexedDISI}'s DENSE-bitset per-block shape).
 *
 * <p>Usage: {@code java VerifySparseNumericDocValues <fixture-dir>}, where
 * {@code <fixture-dir>} contains one {@code <segment>.dvm}/{@code
 * <segment>.dvd}/{@code <segment>.dvs} triple per segment named in the
 * manifest's {@code segments} key, and a {@code manifest.properties}
 * describing each segment's {@code <segment>.max_doc}/{@code
 * <segment>.field_number}/{@code <segment>.values} (a `;`-separated list of
 * {@code doc:value} pairs -- only present docs are listed; every other doc
 * id up to {@code max_doc} is expected absent). Exits nonzero and prints a
 * diff on any mismatch.
 */
public class VerifySparseNumericDocValues {
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
    int fieldNumber = Integer.parseInt(manifest.get(segment + ".field_number"));
    String valuesSpec = manifest.getOrDefault(segment + ".values", "");
    String stepSpec = manifest.get(segment + ".step");

    Map<Integer, Long> expected = new TreeMap<>();
    if (stepSpec != null) {
      // Formula-encoded (avoids a ~900KB manifest for a 200,000-doc, 1-in-3
      // present segment): present docs are every `step`th doc starting at
      // 0, with value = doc * value_mul - value_sub.
      int step = Integer.parseInt(stepSpec);
      long valueMul = Long.parseLong(manifest.get(segment + ".value_mul"));
      long valueSub = Long.parseLong(manifest.get(segment + ".value_sub"));
      for (int doc = 0; doc < maxDoc; doc += step) {
        expected.put(doc, doc * valueMul - valueSub);
      }
    } else if (!valuesSpec.isEmpty()) {
      for (String pair : valuesSpec.split(";")) {
        int idx = pair.indexOf(':');
        int doc = Integer.parseInt(pair.substring(0, idx));
        long value = Long.parseLong(pair.substring(idx + 1));
        expected.put(doc, value);
      }
    }

    FieldInfo fieldInfo =
        new FieldInfo(
            "f",
            fieldNumber,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NUMERIC,
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

      Lucene90DocValuesFormat format = new Lucene90DocValuesFormat();
      SegmentReadState readState = new SegmentReadState(directory, si, fis, IOContext.DEFAULT);
      DocValuesProducer producer = format.fieldsProducer(readState);

      int failures = 0;
      NumericDocValues values = producer.getNumeric(fieldInfo);
      for (int doc = 0; doc < maxDoc; doc++) {
        boolean hasValue = values.advanceExact(doc);
        Long want = expected.get(doc);
        if (want == null) {
          if (hasValue) {
            System.out.println(
                "MISMATCH " + segment + " doc " + doc + ": expected absent, got present ("
                    + values.longValue() + ")");
            failures++;
          }
        } else {
          if (!hasValue) {
            System.out.println(
                "MISMATCH " + segment + " doc " + doc + ": expected present (" + want
                    + "), got absent");
            failures++;
          } else {
            long got = values.longValue();
            if (got != want) {
              System.out.println(
                  "MISMATCH " + segment + " doc " + doc + ": expected=" + want + " got=" + got);
              failures++;
            }
          }
        }
      }

      producer.close();
      if (failures == 0) {
        System.out.println(
            segment
                + ": all "
                + maxDoc
                + " docs verified against real Lucene ("
                + expected.size()
                + " present, "
                + (maxDoc - expected.size())
                + " absent)");
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
