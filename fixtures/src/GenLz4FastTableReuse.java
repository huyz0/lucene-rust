import org.apache.lucene.store.ByteArrayDataOutput;
import org.apache.lucene.util.compress.LZ4;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * `LZ4.compress` through ONE reused `FastCompressionHashTable` over a run of
 * inputs whose sizes cross the 64 KiB line both ways, the way a stored-fields
 * or term-vectors writer reuses its table chunk after chunk: the hash log
 * changes at each crossing (13 below, 12 above), and Java reallocates the
 * table whenever its offset width changes, so every stale entry a reused
 * table carries must resolve to the same candidates Java's would.
 *
 * `lz4_fast_table_reuse.bin`: per case, BE int input length, BE int
 * compressed length, the compressed bytes. The inputs are not stored: the
 * reader regenerates them with the same xorshift. Read by
 * `crates/lucene-codecs/tests/lz4_fixtures.rs`.
 */
public class GenLz4FastTableReuse {
  static final String[] WORDS = {
    "lucene", "segment", "posting", "rust", "codec", "merge", "vector", "term",
    "field", "doc", "values", "index", "reader", "writer", "block", "skip"
  };

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("lz4_fast_table_reuse.bin");
    int[] sizes = {20_000, 70_000, 10_000, 66_000, 200, 70_000, 65_000};
    LZ4.FastCompressionHashTable ht = new LZ4.FastCompressionHashTable();
    long seed = 0x9E3779B97F4A7C15L;
    try (DataOutputStream os = new DataOutputStream(Files.newOutputStream(out))) {
      for (int c = 0; c < sizes.length; c++) {
        StringBuilder sb = new StringBuilder();
        while (sb.length() < sizes[c]) {
          seed ^= seed << 13;
          seed ^= seed >>> 7;
          seed ^= seed << 17;
          sb.append(WORDS[(int) ((seed >>> 33) % WORDS.length)]).append(' ');
          if (((seed >>> 20) & 7) == 0) {
            sb.append(seed & 0xFFFF).append(' ');
          }
        }
        byte[] input = sb.substring(0, sizes[c]).getBytes(StandardCharsets.UTF_8);
        byte[] buf = new byte[input.length * 2 + 64];
        ByteArrayDataOutput bout = new ByteArrayDataOutput(buf);
        LZ4.compress(input, 0, input.length, bout, ht);
        os.writeInt(input.length);
        os.writeInt(bout.getPosition());
        os.write(buf, 0, bout.getPosition());
      }
    }
  }
}
