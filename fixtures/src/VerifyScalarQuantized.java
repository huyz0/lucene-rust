import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene94.Lucene94FieldInfosFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;

/**
 * Reverse-direction verifier (Rust writes, Java reads) for
 * `Lucene104HnswScalarQuantizedVectorsFormat`: opens the six vector files per suffix written by
 * {@code crates/lucene-codecs/examples/write_scalar_quantized_fixture.rs} through real Lucene's
 * own format (with the {@code .fnm} read back through {@code Lucene94FieldInfosFormat}, so the
 * format's field-entry cross-checks run against this port's bytes) and requires, for every
 * recorded query, exactly the top-k -- documents and score bits -- the port computed over the
 * same files. Quantized scores are pure scalar float arithmetic on the stored codes and
 * corrective terms, so any byte Lucene reads differently from the port shows up as a different
 * score or a different hit.
 *
 * <p>Usage: {@code java VerifyScalarQuantized <fixture-dir>}.
 */
public class VerifyScalarQuantized {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0]);
    Map<String, String> m = new HashMap<>();
    for (String line : Files.readAllLines(dir.resolve("manifest.properties"))) {
      int eq = line.indexOf('=');
      if (eq > 0) m.put(line.substring(0, eq), line.substring(eq + 1));
    }
    int failures = 0;
    int checked = 0;
    try (Directory directory = FSDirectory.open(dir)) {
      String name = m.get("segment_name");
      byte[] id = HexFormat.of().parseHex(m.get("id_hex"));
      int maxDoc = Integer.parseInt(m.get("max_doc"));
      int k = Integer.parseInt(m.get("k"));
      int fieldCount = Integer.parseInt(m.get("field_count"));
      SegmentInfo si =
          new SegmentInfo(
              directory,
              org.apache.lucene.util.Version.LATEST,
              org.apache.lucene.util.Version.LATEST,
              name,
              maxDoc,
              false,
              false,
              null,
              Collections.emptyMap(),
              id,
              new HashMap<>(),
              null);
      FieldInfos fis = new Lucene94FieldInfosFormat().read(directory, si, "", IOContext.DEFAULT);
      Map<String, KnnVectorsReader> readers = new HashMap<>();
      for (int f = 0; f < fieldCount; f++) {
        String key = "f" + f;
        String field = m.get(key + ".name");
        String suffix = m.get(key + ".suffix");
        FieldInfo fi = fis.fieldInfo(field);
        if (fi == null
            || !Lucene104HnswScalarQuantizedVectorsFormat.NAME.equals(
                fi.getAttribute("PerFieldKnnVectorsFormat.format"))) {
          System.out.println(field + ": MISMATCH missing field or per-field format attribute");
          failures++;
          continue;
        }
        KnnVectorsReader reader = readers.get(suffix);
        if (reader == null) {
          SegmentReadState state = new SegmentReadState(directory, si, fis, IOContext.DEFAULT, suffix);
          reader = new Lucene104HnswScalarQuantizedVectorsFormat().fieldsReader(state);
          reader.checkIntegrity();
          readers.put(suffix, reader);
        }
        FloatVectorValues raw = reader.getFloatVectorValues(field);
        int count = Integer.parseInt(m.get(key + ".count"));
        if (raw.size() != count || raw.dimension() != Integer.parseInt(m.get(key + ".dim"))) {
          System.out.println(field + ": MISMATCH size/dimension " + raw.size() + "/" + raw.dimension());
          failures++;
        }
        int queries = Integer.parseInt(m.get(key + ".queries"));
        for (int q = 0; q < queries; q++) {
          String[] bits = m.get(key + ".q" + q + ".vec").split(",");
          float[] target = new float[bits.length];
          for (int i = 0; i < bits.length; i++) target[i] = Float.intBitsToFloat(Integer.parseInt(bits[i]));
          TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE, null);
          reader.search(field, target, collector, AcceptDocs.fromLiveDocs(null, maxDoc));
          TopDocs td = collector.topDocs();
          StringBuilder got = new StringBuilder();
          for (ScoreDoc sd : td.scoreDocs) {
            if (got.length() > 0) got.append(',');
            got.append(sd.doc).append(':').append(Float.floatToIntBits(sd.score));
          }
          String want = m.get(key + ".q" + q + ".hits");
          if (!got.toString().equals(want)) {
            System.out.println(field + " q" + q + ": MISMATCH\n  lucene=" + got + "\n  rust  =" + want);
            failures++;
          }
          checked++;
        }
      }
      for (KnnVectorsReader r : readers.values()) r.close();
    } catch (Throwable t) {
      System.out.println("FAILED TO OPEN: " + t);
      t.printStackTrace(System.out);
      System.exit(1);
    }
    if (failures > 0 || checked == 0) {
      System.out.println(failures + " mismatch(es), " + checked + " queries");
      System.exit(1);
    }
    System.out.println(
        "All " + checked + " queries over Rust-written scalar-quantized vectors match real Lucene. PASS");
  }
}
