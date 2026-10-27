import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexableFieldType;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.BytesRef;

/**
 * Differential fixture for {@code LuceneTestCase.newField}/{@code newStringField}/{@code
 * newTextField} (M14.h): the field type Lucene 10.5.0's test framework upgrades each base type to
 * for a seeded {@code java.util.Random}, and the type it reuses for a name it has seen.
 *
 * <p>Writes {@code new_field.txt}: for each seed and case, {@code <seed> <case> <type> <next>},
 * the resulting type as {@code stored,tv,tvPositions,tvPayloads,tvOffsets} bits and the random's
 * next {@code nextInt()}. Every case uses a fresh field name, then {@code reuse} asks again for the
 * first case's name with another base type and must get the first type back, drawing nothing.
 *
 * <p>Usage: java GenTestFrameworkNewField <outdir>
 */
public class GenTestFrameworkNewField {
  static final int SEEDS = 200;

  static String bits(IndexableFieldType t) {
    return (t.stored() ? "1" : "0")
        + (t.storeTermVectors() ? "1" : "0")
        + (t.storeTermVectorPositions() ? "1" : "0")
        + (t.storeTermVectorPayloads() ? "1" : "0")
        + (t.storeTermVectorOffsets() ? "1" : "0");
  }

  static void line(StringBuilder out, int seed, String name, Field f, Random r) {
    out.append(seed).append(' ').append(name).append(' ').append(bits(f.fieldType()));
    out.append(' ').append(r.nextInt()).append('\n');
  }

  public static void main(String[] args) throws Exception {
    FieldType customFreqs = new FieldType(TextField.TYPE_NOT_STORED);
    customFreqs.setIndexOptions(IndexOptions.DOCS_AND_CUSTOM_FREQS);
    customFreqs.freeze();
    FieldType storedOnly = new FieldType();
    storedOnly.setStored(true);
    storedOnly.freeze();
    FieldType vectors = new FieldType(TextField.TYPE_NOT_STORED);
    vectors.setStoreTermVectors(true);
    vectors.freeze();

    StringBuilder out = new StringBuilder();
    for (int seed = 0; seed < SEEDS; seed++) {
      Random r = new Random(seed);
      String p = "s" + seed + "_";
      line(out, seed, "string", LuceneTestCase.newStringField(r, p + "a", "v", Field.Store.NO), r);
      line(out, seed, "string_stored", LuceneTestCase.newStringField(r, p + "b", "v", Field.Store.YES), r);
      line(out, seed, "string_bytes", LuceneTestCase.newStringField(r, p + "c", new BytesRef("v"), Field.Store.NO), r);
      line(out, seed, "text", LuceneTestCase.newTextField(r, p + "d", "v w", Field.Store.NO), r);
      line(out, seed, "text_stored", LuceneTestCase.newTextField(r, p + "e", "v w", Field.Store.YES), r);
      line(out, seed, "custom_freqs", LuceneTestCase.newField(r, p + "f", "v", customFreqs), r);
      line(out, seed, "stored_only", LuceneTestCase.newField(r, p + "g", "v", storedOnly), r);
      line(out, seed, "vectors", LuceneTestCase.newField(r, p + "h", "v", vectors), r);
      line(out, seed, "reuse", LuceneTestCase.newTextField(r, p + "a", "v w", Field.Store.YES), r);
    }
    Path dir = Path.of(args[0]);
    Files.createDirectories(dir);
    Files.writeString(dir.resolve("new_field.txt"), out.toString());
  }
}
