import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.apache.lucene.tests.util.TestUtil;

/**
 * Differential fixture for {@code TestUtil.unzip} and {@code VocabularyAssert}'s zip overloads
 * (M14.h): a vocabulary zip as {@code java.util.zip.ZipOutputStream} writes it, and the files
 * Lucene 10.5.0's {@code TestUtil.unzip} extracts from it.
 *
 * <p>Writes {@code vocabulary.zip} (a stored entry, deflated entries with data descriptors, a
 * directory entry and a UTF-8 name) and {@code unzip.txt}: for each path {@code unzip} created,
 * sorted, {@code <path>/} for a directory or {@code <path>\t<length>\t<crc32 hex>} for a file.
 *
 * <p>Deflated entries use {@code Deflater.NO_COMPRESSION}, whose stored deflate blocks do not
 * depend on the JDK's zlib build, so {@code --check} is stable across JDKs.
 *
 * <p>Usage: java GenTestFrameworkZip <outdir>
 */
public class GenTestFrameworkZip {
  static final long TIME = 1_000_000_000_000L;

  static void deflated(ZipOutputStream zip, String name, String text) throws Exception {
    ZipEntry e = new ZipEntry(name);
    e.setTime(TIME);
    zip.putNextEntry(e);
    zip.write(text.getBytes(StandardCharsets.UTF_8));
    zip.closeEntry();
  }

  static void stored(ZipOutputStream zip, String name, byte[] data) throws Exception {
    ZipEntry e = new ZipEntry(name);
    e.setTime(TIME);
    e.setMethod(ZipEntry.STORED);
    e.setSize(data.length);
    e.setCompressedSize(data.length);
    CRC32 crc = new CRC32();
    crc.update(data);
    e.setCrc(crc.getValue());
    zip.putNextEntry(e);
    zip.write(data);
    zip.closeEntry();
  }

  public static void main(String[] args) throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (ZipOutputStream zip = new ZipOutputStream(bytes)) {
      zip.setLevel(Deflater.NO_COMPRESSION);
      deflated(zip, "voc.txt", "Dogs\r\nCATS\nÉté\n");
      stored(zip, "out.txt", "dogs\ncats\nété".getBytes(StandardCharsets.UTF_8));
      stored(zip, "dir/", new byte[0]);
      deflated(zip, "dir/voc_out.txt", "# comment\n\nDogs\tdogs\nÉTÉ\tété\n");
      deflated(zip, "dir/sub/big.txt", "lucene ".repeat(300));
    }
    Path dir = Path.of(args[0]);
    Files.createDirectories(dir);
    Files.write(dir.resolve("vocabulary.zip"), bytes.toByteArray());

    Path tmp = Files.createTempDirectory("unzip");
    TestUtil.unzip(new ByteArrayInputStream(bytes.toByteArray()), tmp);
    List<String> lines = new ArrayList<>();
    try (Stream<Path> walk = Files.walk(tmp)) {
      for (Path p : (Iterable<Path>) walk::iterator) {
        if (p.equals(tmp)) continue;
        String rel = tmp.relativize(p).toString().replace('\\', '/');
        if (Files.isDirectory(p)) {
          lines.add(rel + "/");
        } else {
          byte[] data = Files.readAllBytes(p);
          CRC32 crc = new CRC32();
          crc.update(data);
          lines.add(rel + "\t" + data.length + "\t" + Long.toHexString(crc.getValue()));
        }
      }
    }
    Collections.sort(lines);
    Files.writeString(dir.resolve("unzip.txt"), String.join("\n", lines) + "\n");
  }
}
