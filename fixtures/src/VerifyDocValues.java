import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.lucene90.Lucene90DocValuesFormat;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.index.SortedSetDocValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;

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
 * Reverse-direction verifier (Rust writes, Java reads): opens every
 * `.dvm`/`.dvd`/`.dvs` triple written by this port's five
 * `doc_values::write_single_dense_*_field` functions (NUMERIC, BINARY,
 * SORTED_NUMERIC, SORTED, SORTED_SET -- all dense, single-field-per-triple --
 * see `crates/lucene-codecs/examples/write_doc_values_fixture.rs`) directly
 * through real Lucene's {@link Lucene90DocValuesFormat}, using a hand-built
 * {@link SegmentInfo}/{@link FieldInfos} the same way
 * {@code VerifyPoints.java}/{@code VerifyStoredFields.java} do -- this keeps
 * the slice scoped to exactly the doc-values format itself, no
 * `.si`/`.fnm` writer needed.
 *
 * <p>Each segment's {@code <segment>.type} manifest key selects the matching
 * production-facing read API -- {@link NumericDocValues}, {@link
 * BinaryDocValues}, {@link SortedNumericDocValues}, {@link SortedDocValues},
 * or {@link SortedSetDocValues} -- never a codec-internal decode, and
 * confirms every doc's value(s) match the manifest.
 *
 * <p>Usage: {@code java VerifyDocValues <fixture-dir>}, where
 * {@code <fixture-dir>} contains one {@code <segment>.dvm}/{@code
 * <segment>.dvd}/{@code <segment>.dvs} triple per segment named in the
 * manifest's {@code segments} key, and a {@code manifest.properties}
 * describing each segment's {@code <segment>.type}, expected per-doc
 * value(s) under {@code <segment>.max_doc}/{@code
 * <segment>.field_number}/{@code <segment>.values}. Exits nonzero and prints
 * a diff on any mismatch.
 *
 * <p>Ten segments are verified. NUMERIC: {@code _0} (mixed small/medium/negative
 * values, exercising a real {@code bitsPerValue > 0} with {@code min <= 0}),
 * {@code _1} (every value has {@code min > 0} and {@code
 * unsignedBitsRequired(max) == unsignedBitsRequired(max-min)}, a regression
 * case for the min-shift-drop optimization a review pass found was
 * previously unverified against real Lucene), and {@code _2} (all-equal
 * values, the {@code bitsPerValue == 0} constant encoding). BINARY: {@code
 * _3} (fixed-length values, direct addressing) and {@code _4} (variable
 * length, including an empty value, {@code DirectMonotonicReader} address
 * block). SORTED_NUMERIC: {@code _5} (every doc exactly one value, the
 * address-array collapse case) and {@code _6} (1-3 values per doc, real
 * address-range array). SORTED: {@code _7} (repeated terms over a 3-term
 * dictionary). SORTED_SET: {@code _8} (every doc one distinct value, the
 * {@code multiValued = false} collapse case) and {@code _9} (1-2 distinct
 * values per doc sharing a dictionary, including a doc whose raw values
 * repeat and dedup to one ordinal). All ten were previously verified only
 * against this port's own reader, never against real Lucene.
 */
public class VerifyDocValues {
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
   * Opens one Rust-written `.dvm`/`.dvd`/`.dvs` segment (named {@code
   * segment}, e.g. {@code "_0"}) through real Lucene and checks every doc's
   * value(s) against the manifest under {@code segment + ".max_doc"} /
   * {@code segment + ".field_number"} / {@code segment + ".values"}, using
   * whichever real doc-values iteration API matches {@code segment +
   * ".type"} (NUMERIC/BINARY/SORTED_NUMERIC/SORTED/SORTED_SET). Returns the
   * number of mismatches (0 on full success).
   */
  static int verifySegment(Path dir, byte[] id, String segment, Map<String, String> manifest)
      throws IOException {
    int maxDoc = Integer.parseInt(manifest.get(segment + ".max_doc"));
    int fieldNumber = Integer.parseInt(manifest.get(segment + ".field_number"));
    String valuesSpec = manifest.getOrDefault(segment + ".values", "");
    DocValuesType type = DocValuesType.valueOf(manifest.get(segment + ".type"));

    FieldInfo fieldInfo =
        new FieldInfo(
            "f",
            fieldNumber,
            false,
            false,
            false,
            IndexOptions.NONE,
            type,
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

      int failures;
      switch (type) {
        case NUMERIC:
          failures = verifyNumeric(producer, fieldInfo, segment, maxDoc, valuesSpec);
          break;
        case BINARY:
          failures = verifyBinary(producer, fieldInfo, segment, maxDoc, valuesSpec);
          break;
        case SORTED_NUMERIC:
          failures = verifySortedNumeric(producer, fieldInfo, segment, maxDoc, valuesSpec);
          break;
        case SORTED:
          failures = verifySorted(producer, fieldInfo, segment, maxDoc, valuesSpec);
          break;
        case SORTED_SET:
          failures = verifySortedSet(producer, fieldInfo, segment, maxDoc, valuesSpec);
          break;
        default:
          throw new IllegalStateException("unexpected type " + type);
      }

      producer.close();
      return failures;
    } catch (CorruptIndexException e) {
      System.out.println(segment + " FAILED TO OPEN: " + e);
      return 1;
    }
  }

  static List<Long> parseLongs(String valuesSpec) {
    List<Long> expected = new ArrayList<>();
    if (!valuesSpec.isEmpty()) {
      for (String v : valuesSpec.split(";")) {
        expected.add(Long.parseLong(v));
      }
    }
    return expected;
  }

  static int verifyNumeric(
      DocValuesProducer producer, FieldInfo fieldInfo, String segment, int maxDoc, String valuesSpec)
      throws IOException {
    List<Long> expected = parseLongs(valuesSpec);
    NumericDocValues values = producer.getNumeric(fieldInfo);
    int failures = 0;
    int seenDocs = 0;
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      seenDocs++;
      long got = values.longValue();
      long want = expected.get(doc);
      if (want != got) {
        System.out.println(
            "MISMATCH " + segment + " doc " + doc + ": expected=" + want + " got=" + got);
        failures++;
      }
    }
    failures += checkDocCount(segment, expected.size(), seenDocs);
    if (failures == 0) {
      System.out.println(
          segment + ": all " + expected.size() + " doc values verified against real Lucene");
    }
    return failures;
  }

  static int verifyBinary(
      DocValuesProducer producer, FieldInfo fieldInfo, String segment, int maxDoc, String valuesSpec)
      throws IOException {
    HexFormat hex = HexFormat.of();
    List<byte[]> expected = new ArrayList<>();
    if (!valuesSpec.isEmpty()) {
      for (String v : valuesSpec.split(";", -1)) {
        expected.add(hex.parseHex(v));
      }
    }
    BinaryDocValues values = producer.getBinary(fieldInfo);
    int failures = 0;
    int seenDocs = 0;
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      seenDocs++;
      BytesRef got = values.binaryValue();
      byte[] want = expected.get(doc);
      byte[] gotBytes =
          java.util.Arrays.copyOfRange(got.bytes, got.offset, got.offset + got.length);
      if (!java.util.Arrays.equals(want, gotBytes)) {
        System.out.println(
            "MISMATCH "
                + segment
                + " doc "
                + doc
                + ": expected="
                + hex.formatHex(want)
                + " got="
                + hex.formatHex(gotBytes));
        failures++;
      }
    }
    failures += checkDocCount(segment, expected.size(), seenDocs);
    if (failures == 0) {
      System.out.println(
          segment + ": all " + expected.size() + " doc values verified against real Lucene");
    }
    return failures;
  }

  static int verifySortedNumeric(
      DocValuesProducer producer, FieldInfo fieldInfo, String segment, int maxDoc, String valuesSpec)
      throws IOException {
    List<List<Long>> expected = new ArrayList<>();
    if (!valuesSpec.isEmpty()) {
      for (String perDoc : valuesSpec.split(";", -1)) {
        List<Long> vals = new ArrayList<>();
        if (!perDoc.isEmpty()) {
          for (String v : perDoc.split(",")) {
            vals.add(Long.parseLong(v));
          }
        }
        expected.add(vals);
      }
    }
    SortedNumericDocValues values = producer.getSortedNumeric(fieldInfo);
    int failures = 0;
    int seenDocs = 0;
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      seenDocs++;
      List<Long> want = expected.get(doc);
      int count = values.docValueCount();
      if (count != want.size()) {
        System.out.println(
            "MISMATCH "
                + segment
                + " doc "
                + doc
                + ": expected "
                + want.size()
                + " values, got "
                + count);
        failures++;
      }
      for (int i = 0; i < count; i++) {
        long got = values.nextValue();
        if (i < want.size() && want.get(i) != got) {
          System.out.println(
              "MISMATCH "
                  + segment
                  + " doc "
                  + doc
                  + " value "
                  + i
                  + ": expected="
                  + want.get(i)
                  + " got="
                  + got);
          failures++;
        }
      }
    }
    failures += checkDocCount(segment, expected.size(), seenDocs);
    if (failures == 0) {
      System.out.println(
          segment + ": all " + expected.size() + " doc values verified against real Lucene");
    }
    return failures;
  }

  static int verifySorted(
      DocValuesProducer producer, FieldInfo fieldInfo, String segment, int maxDoc, String valuesSpec)
      throws IOException {
    HexFormat hex = HexFormat.of();
    List<byte[]> expected = new ArrayList<>();
    if (!valuesSpec.isEmpty()) {
      for (String v : valuesSpec.split(";", -1)) {
        expected.add(hex.parseHex(v));
      }
    }
    SortedDocValues values = producer.getSorted(fieldInfo);
    int failures = 0;
    int seenDocs = 0;
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      seenDocs++;
      int ord = values.ordValue();
      BytesRef term = values.lookupOrd(ord);
      byte[] got = java.util.Arrays.copyOfRange(term.bytes, term.offset, term.offset + term.length);
      byte[] want = expected.get(doc);
      if (!java.util.Arrays.equals(want, got)) {
        System.out.println(
            "MISMATCH "
                + segment
                + " doc "
                + doc
                + ": expected="
                + hex.formatHex(want)
                + " got="
                + hex.formatHex(got));
        failures++;
      }
    }
    failures += checkDocCount(segment, expected.size(), seenDocs);
    if (failures == 0) {
      System.out.println(
          segment + ": all " + expected.size() + " doc values verified against real Lucene");
    }
    return failures;
  }

  static int verifySortedSet(
      DocValuesProducer producer, FieldInfo fieldInfo, String segment, int maxDoc, String valuesSpec)
      throws IOException {
    HexFormat hex = HexFormat.of();
    List<List<byte[]>> expected = new ArrayList<>();
    if (!valuesSpec.isEmpty()) {
      for (String perDoc : valuesSpec.split(";", -1)) {
        List<byte[]> vals = new ArrayList<>();
        if (!perDoc.isEmpty()) {
          for (String v : perDoc.split(",")) {
            vals.add(hex.parseHex(v));
          }
        }
        expected.add(vals);
      }
    }
    SortedSetDocValues values = producer.getSortedSet(fieldInfo);
    int failures = 0;
    int seenDocs = 0;
    for (int doc = values.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = values.nextDoc()) {
      seenDocs++;
      List<byte[]> want = expected.get(doc);
      int count = (int) values.docValueCount();
      if (count != want.size()) {
        System.out.println(
            "MISMATCH "
                + segment
                + " doc "
                + doc
                + ": expected "
                + want.size()
                + " values, got "
                + count);
        failures++;
      }
      for (int i = 0; i < count; i++) {
        long ord = values.nextOrd();
        BytesRef term = values.lookupOrd(ord);
        byte[] got =
            java.util.Arrays.copyOfRange(term.bytes, term.offset, term.offset + term.length);
        if (i < want.size() && !java.util.Arrays.equals(want.get(i), got)) {
          System.out.println(
              "MISMATCH "
                  + segment
                  + " doc "
                  + doc
                  + " value "
                  + i
                  + ": expected="
                  + hex.formatHex(want.get(i))
                  + " got="
                  + hex.formatHex(got));
          failures++;
        }
      }
    }
    failures += checkDocCount(segment, expected.size(), seenDocs);
    if (failures == 0) {
      System.out.println(
          segment + ": all " + expected.size() + " doc values verified against real Lucene");
    }
    return failures;
  }

  static int checkDocCount(String segment, int expectedCount, int seenDocs) {
    if (seenDocs != expectedCount) {
      System.out.println(
          "MISMATCH "
              + segment
              + " doc count: expected="
              + expectedCount
              + " got="
              + seenDocs);
      return 1;
    }
    return 0;
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
