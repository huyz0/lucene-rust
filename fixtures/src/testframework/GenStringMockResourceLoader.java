import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;
import org.apache.lucene.tests.util.StringMockResourceLoader;

/** Lucene 10.5.0's UTF-8 bytes and independent streams for every resource name. */
public class GenStringMockResourceLoader {
  private static String units(String text) {
    if (text.isEmpty()) return "-";
    StringBuilder result = new StringBuilder();
    for (int i = 0; i < text.length(); i++) {
      result.append(String.format(java.util.Locale.ROOT, "%04x", (int) text.charAt(i)));
    }
    return result.toString();
  }

  public static void main(String[] args) throws Exception {
    String[] texts = {
      "", "plain ascii", "\u0000\t\r\n", "caf\u00e9 e\u0301",
      "\ufeff# comment\nalpha\r\nbeta", "\u65e5\u672c\u8a9e \ud83d\ude00 \udbff\udfff",
      "repeated UTF-8 \u00e9 \ud83d\ude00\n".repeat(16)
    };
    String[] resources = {"", "missing.txt", "../outside", "\u0000\ud83d\ude00"};
    StringBuilder out = new StringBuilder();
    for (String text : texts) {
      StringMockResourceLoader loader = new StringMockResourceLoader(text);
      for (String resource : resources) {
        try (InputStream first = loader.openResource(resource);
            InputStream second = loader.openResource(resource)) {
          // Reading the first stream must leave the next stream untouched.
          first.readNBytes(3);
          byte[] bytes = second.readAllBytes();
          out.append(units(text)).append('\t').append(units(resource)).append('\t')
              .append(bytes.length == 0 ? "-" : HexFormat.of().formatHex(bytes)).append('\n');
        }
      }
    }
    Path dir = Path.of(args[0]);
    Files.createDirectories(dir);
    Files.writeString(dir.resolve("string_mock_resource_loader.tsv"), out, StandardCharsets.UTF_8);
  }
}
