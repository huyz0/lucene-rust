import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;

/**
 * Writes the unpacked DFA tables of a JFlex 1.8.2 scanner class Lucene 10.5.0 ships, for {@code
 * crates/lucene-analysis/src/util/jflex.rs}: {@code email/tables.bin.z} from {@code
 * org.apache.lucene.analysis.email.UAX29URLEmailTokenizerImpl} and {@code
 * charfilter/html_strip_tables.bin.z} from {@code
 * org.apache.lucene.analysis.charfilter.HTMLStripCharFilter}, {@code classic/tables.bin.z} from
 * {@code org.apache.lucene.analysis.classic.ClassicTokenizerImpl} and {@code
 * wikipedia/tables.bin.z} from {@code org.apache.lucene.analysis.wikipedia.WikipediaTokenizerImpl}
 * (all under {@code crates/lucene-analysis/src/}).
 *
 * <p>As {@code ExtractStandardTokenizerTables} does for {@code StandardTokenizerImpl}, the {@code
 * private static final int[]} tables are read back by reflection after the class's static
 * initialiser has unpacked them, never transcribed. They are too large for Rust source (the email
 * scanner's {@code ZZ_TRANS} has 1,095,920 entries), so the file is binary: {@code "JFLX"}, then per
 * table its name ({@code u8} length + ASCII), element count ({@code u32} LE) and elements ({@code
 * i32} LE), the whole zlib-compressed at the best level.
 *
 * <p>Run (the jars are what {@code scripts/lib-lucene-jars.sh} resolves):
 *
 * <pre>
 *   javac -d /tmp/x crates/lucene-analysis/tools/ExtractJFlexTables.java
 *   java -cp /tmp/x:lucene-core-10.5.0.jar:lucene-analysis-common-10.5.0.jar ExtractJFlexTables \
 *       org.apache.lucene.analysis.email.UAX29URLEmailTokenizerImpl \
 *       crates/lucene-analysis/src/email/tables.bin.z
 * </pre>
 */
public class ExtractJFlexTables {
  private static final String[] TABLES = {
    "ZZ_LEXSTATE", "ZZ_CMAP_TOP", "ZZ_CMAP_BLOCKS", "ZZ_ACTION", "ZZ_ROWMAP", "ZZ_TRANS", "ZZ_ATTRIBUTE"
  };

  public static void main(String[] args) throws Exception {
    Class<?> impl = Class.forName(args[0]);
    String version = org.apache.lucene.util.Version.LATEST.toString();
    if (!version.equals("10.5.0")) {
      throw new IllegalStateException("expected Lucene 10.5.0 on the classpath, got " + version);
    }
    ByteArrayOutputStream raw = new ByteArrayOutputStream();
    raw.write("JFLX".getBytes(StandardCharsets.US_ASCII));
    for (String name : TABLES) {
      Field f = impl.getDeclaredField(name);
      f.setAccessible(true);
      int[] v = (int[]) f.get(null);
      raw.write(name.length());
      raw.write(name.getBytes(StandardCharsets.US_ASCII));
      writeIntLE(raw, v.length);
      for (int x : v) {
        writeIntLE(raw, x);
      }
    }
    ByteArrayOutputStream z = new ByteArrayOutputStream();
    try (DeflaterOutputStream out = new DeflaterOutputStream(z, new Deflater(Deflater.BEST_COMPRESSION))) {
      raw.writeTo(out);
    }
    Files.write(Path.of(args[1]), z.toByteArray());
  }

  private static void writeIntLE(ByteArrayOutputStream out, int v) {
    out.write(v);
    out.write(v >>> 8);
    out.write(v >>> 16);
    out.write(v >>> 24);
  }
}
