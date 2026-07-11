import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.StringHelper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Generates fixtures for CodecUtil header/footer framing (magic/version/id/suffix/CRC),
 * pinned to the Lucene version on the classpath. Produces:
 *   - plain_header_footer.bin: writeHeader + some payload + writeFooter
 *   - index_header_footer.bin: writeIndexHeader (with id+suffix) + payload + writeFooter
 *   - corrupt_checksum.bin: same as plain, with one payload byte flipped (checksum must fail)
 *   - manifest.properties: codec name/version/id/suffix/payload used, so Rust can assert without parsing Java
 */
public class GenCodecUtil {
  static final String CODEC = "LuceneRustFixture";
  static final int VERSION = 3;
  static final String SUFFIX = "seg1";

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);

    byte[] payload = "hello lucene-rust codec header/footer fixture payload".getBytes("UTF-8");
    byte[] id = StringHelper.randomId();

    try (Directory dir = new ByteBuffersDirectory()) {
      writePlain(dir, payload);
      copyOut(dir, "plain.bin", out);
      corrupt(dir, "plain.bin", out, "corrupt_checksum.bin");

      writeIndexHeader(dir, payload, id);
      copyOut(dir, "indexed.bin", out);

      StringBuilder m = new StringBuilder();
      m.append("codec=").append(CODEC).append('\n');
      m.append("version=").append(VERSION).append('\n');
      m.append("suffix=").append(SUFFIX).append('\n');
      m.append("id_hex=").append(hex(id)).append('\n');
      m.append("payload_len=").append(payload.length).append('\n');
      Files.writeString(out.resolve("manifest.properties"), m.toString());
    }

    System.out.println("codec fixtures written to " + out);
  }

  static void writePlain(Directory dir, byte[] payload) throws IOException {
    try (IndexOutput o = dir.createOutput("plain.bin", IOContext.DEFAULT)) {
      CodecUtil.writeHeader(o, CODEC, VERSION);
      o.writeBytes(payload, payload.length);
      CodecUtil.writeFooter(o);
    }
    // sanity round-trip with Lucene itself
    try (ChecksumIndexInput in = dir.openChecksumInput("plain.bin")) {
      CodecUtil.checkHeader(in, CODEC, VERSION, VERSION);
      byte[] back = new byte[payload.length];
      in.readBytes(back, 0, back.length);
      CodecUtil.checkFooter(in);
    }
  }

  static void writeIndexHeader(Directory dir, byte[] payload, byte[] id) throws IOException {
    try (IndexOutput o = dir.createOutput("indexed.bin", IOContext.DEFAULT)) {
      CodecUtil.writeIndexHeader(o, CODEC, VERSION, id, SUFFIX);
      o.writeBytes(payload, payload.length);
      CodecUtil.writeFooter(o);
    }
    try (ChecksumIndexInput in = dir.openChecksumInput("indexed.bin")) {
      CodecUtil.checkIndexHeader(in, CODEC, VERSION, VERSION, id, SUFFIX);
      byte[] back = new byte[payload.length];
      in.readBytes(back, 0, back.length);
      CodecUtil.checkFooter(in);
    }
  }

  static void copyOut(Directory dir, String name, Path outDir) throws IOException {
    try (IndexInput in = dir.openInput(name, IOContext.DEFAULT)) {
      byte[] b = new byte[(int) in.length()];
      in.readBytes(b, 0, b.length);
      Files.write(outDir.resolve(name), b);
    }
  }

  static void corrupt(Directory dir, String name, Path outDir, String outName) throws IOException {
    byte[] b = Files.readAllBytes(outDir.resolve(name));
    // flip a byte in the middle of the payload (well clear of header/footer)
    b[b.length / 2] ^= 0x01;
    Files.write(outDir.resolve(outName), b);
  }

  static String hex(byte[] b) {
    StringBuilder sb = new StringBuilder();
    for (byte x : b) sb.append(String.format("%02x", x));
    return sb.toString();
  }
}
