import org.apache.lucene.codecs.TermVectorsReader;
import org.apache.lucene.codecs.lucene90.Lucene90TermVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Fields;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;

/**
 * Reverse-direction verifier (Rust writes, Java reads): opens a
 * `.tvd`/`.tvx`/`.tvm` triple written by this port's
 * `term_vectors::write_best_speed` (see
 * `crates/lucene-codecs/examples/write_term_vectors_fixture.rs`) directly
 * through real Lucene's {@link Lucene90TermVectorsFormat}, using a
 * hand-built {@link SegmentInfo}/{@link FieldInfos} rather than also
 * requiring Rust to write {@code .si}/{@code .fnm} -- same pattern as
 * {@code VerifyStoredFields.java}.
 *
 * <p>The Rust writer only supports positions (no offsets/payloads, no
 * prefix sharing, single chunk -- see its doc comment), so this verifier
 * only checks term text, freq, and positions.
 *
 * <p>Usage: {@code java VerifyTermVectors <fixture-dir>}, where
 * {@code <fixture-dir>} contains {@code _0.tvd}/{@code _0.tvx}/
 * {@code _0.tvm} and a {@code manifest.properties} (see the Rust example
 * above for the exact format) describing the expected per-doc field/term
 * values. Exits nonzero and prints a diff on any mismatch.
 */
public class VerifyTermVectors {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> manifest = readManifest(dir.resolve("manifest.properties"));
    byte[] id = HexFormat.of().parseHex(manifest.get("id_hex"));

    // "_0": the primary multi-field-number fixture (positions, an empty
    // doc, a no-position field) -- keys are unprefixed ("max_doc",
    // "doc.N.fields", ...).
    int failures = verifySegment(dir, id, "_0", manifest, "");

    // "_1": every field across every doc has field_number == 0
    // (max_field_num == 0 for the whole chunk), a regression case for a
    // real Lucene-incompatible bits_per_field_num == 0 encoding that "_0"
    // can never hit since it always mixes field numbers 0 and 1 -- keys
    // are prefixed with "all_zero." (e.g. "all_zero.max_doc").
    failures += verifySegment(dir, id, "_1", manifest, "all_zero.");

    if (failures > 0) {
      System.out.println(failures + " document(s) mismatched overall");
      System.exit(1);
    }
    System.out.println("All segments verified against real Lucene. PASS");
  }

  /**
   * Opens one Rust-written `.tvd`/`.tvx`/`.tvm` segment (named {@code
   * segmentName}, e.g. {@code "_0"}) through real Lucene and checks every
   * doc's rendered fields against the manifest under {@code
   * keyPrefix + "max_doc"} / {@code keyPrefix + "num_fields"} / {@code
   * keyPrefix + "doc." + doc + ".fields"}. Returns the number of mismatched
   * docs (0 on full success).
   */
  static int verifySegment(
      Path dir, byte[] id, String segmentName, Map<String, String> manifest, String keyPrefix)
      throws IOException {
    int maxDoc = Integer.parseInt(manifest.get(keyPrefix + "max_doc"));
    int numFields = Integer.parseInt(manifest.get(keyPrefix + "num_fields"));

    FieldInfo[] fieldInfos = new FieldInfo[numFields];
    for (int i = 0; i < numFields; i++) {
      fieldInfos[i] =
          new FieldInfo(
              "field" + i,
              i,
              true, // storeTermVector
              false,
              false,
              IndexOptions.DOCS_AND_FREQS_AND_POSITIONS,
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
    FieldInfos fis = new FieldInfos(fieldInfos);

    try (Directory directory = FSDirectory.open(dir)) {
      SegmentInfo si =
          new SegmentInfo(
              directory,
              Version.LATEST,
              Version.LATEST,
              segmentName,
              maxDoc,
              false,
              false,
              new Lucene104Codec(),
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);

      Lucene90TermVectorsFormat format = new Lucene90TermVectorsFormat();
      TermVectorsReader reader = format.vectorsReader(directory, si, fis, IOContext.DEFAULT);

      int failures = 0;
      for (int doc = 0; doc < maxDoc; doc++) {
        String expectedLine = manifest.getOrDefault(keyPrefix + "doc." + doc + ".fields", "");
        String got = renderDoc(reader, doc, fis);
        if (!got.equals(expectedLine)) {
          System.out.println(
              "MISMATCH "
                  + segmentName
                  + " doc "
                  + doc
                  + ": expected=["
                  + expectedLine
                  + "] got=["
                  + got
                  + "]");
          failures++;
        } else {
          System.out.println(segmentName + " doc " + doc + " OK: " + got);
        }
      }
      reader.close();
      return failures;
    } catch (CorruptIndexException e) {
      System.out.println(segmentName + " FAILED TO OPEN: " + e);
      return 1;
    }
  }

  /** Renders a doc's fields/terms in the same `num[terms]` shape the Rust example writes. */
  static String renderDoc(TermVectorsReader reader, int doc, FieldInfos fis) throws IOException {
    Fields fields = reader.get(doc);
    if (fields == null) {
      return "";
    }
    StringBuilder sb = new StringBuilder();
    for (String fieldName : fields) {
      FieldInfo fi = fis.fieldInfo(fieldName);
      Terms terms = fields.terms(fieldName);
      if (sb.length() > 0) sb.append(';');
      sb.append(fi.number).append('[');
      TermsEnum te = terms.iterator();
      boolean firstTerm = true;
      PostingsEnum pe = null;
      BytesRef term;
      while ((term = te.next()) != null) {
        if (!firstTerm) sb.append(',');
        firstTerm = false;
        pe = te.postings(pe, PostingsEnum.POSITIONS);
        pe.nextDoc();
        int freq = pe.freq();
        StringBuilder positions = new StringBuilder();
        for (int i = 0; i < freq; i++) {
          int pos = pe.nextPosition();
          if (pos >= 0) {
            if (i > 0) positions.append(',');
            positions.append(pos);
          }
        }
        sb.append(term.utf8ToString()).append(':').append(freq).append(':').append(positions);
      }
      sb.append(']');
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
