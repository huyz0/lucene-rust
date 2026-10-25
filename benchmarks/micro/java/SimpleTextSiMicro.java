import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import org.apache.lucene.codecs.simpletext.SimpleTextSegmentInfoFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;

/** SimpleText .si read/write, on exactly the typed-sort fixtures Rust uses. */
public final class SimpleTextSiMicro {
  static volatile Object sink;
  interface Op { long run() throws IOException; }
  record Case(String name, byte[] id, byte[] bytes, SegmentInfo info) {}

  static long loop(Op op, long millis) throws IOException {
    long start = System.nanoTime(), operations = 0;
    do { operations += op.run(); } while (System.nanoTime() - start < millis * 1_000_000L);
    return operations;
  }

  static void measure(String name, Op operation) throws IOException {
    loop(operation, Long.getLong("warmupMs", 1500));
    long start = System.nanoTime();
    long operations = loop(operation, Long.getLong("measureMs", 2000));
    System.out.printf("%s\t%.3f\t%d%n", name,
        (double) (System.nanoTime() - start) / operations, operations);
  }

  static byte[] bytes(Directory dir, String name) throws IOException {
    try (var input = dir.openInput(name + ".si", IOContext.DEFAULT)) {
      byte[] bytes = new byte[Math.toIntExact(input.length())];
      input.readBytes(bytes, 0, bytes.length);
      return bytes;
    }
  }

  static long hash(long hash, byte[] bytes) {
    hash = (hash ^ bytes.length) * 0x100000001b3L;
    for (byte value : bytes) hash = (hash ^ (value & 255)) * 0x100000001b3L;
    return hash;
  }

  public static void main(String[] args) throws Exception {
    Path root = Path.of(args[0]);
    var format = new SimpleTextSegmentInfoFormat();
    try (var input = new ByteBuffersDirectory()) {
      var cases = new ArrayList<Case>();
      long hash = 0xcbf29ce484222325L;
      for (String line : Files.readAllLines(root.resolve("cases.tsv"))) {
        String[] row = line.split("\t");
        byte[] id = HexFormat.of().parseHex(row[1]);
        byte[] data = Files.readAllBytes(root.resolve(row[0] + ".si"));
        try (var output = input.createOutput(row[0] + ".si", IOContext.DEFAULT)) {
          output.writeBytes(data, data.length);
        }
        SegmentInfo info = format.read(input, row[0], id, IOContext.DEFAULT);
        if (!info.getVersion().toString().equals(row[3])
            || !String.valueOf(info.getMinVersion()).equals(row[4]))
          throw new AssertionError("version mismatch");
        String description = info.getIndexSort() == null ? "<none>" : info.getIndexSort().toString();
        if (!description.equals(new String(HexFormat.of().parseHex(row[2]),
            java.nio.charset.StandardCharsets.UTF_8))) throw new AssertionError("typed sort");
        try (var output = new ByteBuffersDirectory()) {
          format.write(output, info, IOContext.DEFAULT);
          byte[] encoded = bytes(output, row[0]);
          if (!Arrays.equals(data, encoded)) throw new AssertionError("Java fixture mismatch");
          hash = hash(hash, encoded);
        }
        cases.add(new Case(row[0], id, data, info));
      }
      System.out.printf("#check\tcx_st_si_read\t%016x\t1%n", hash);
      System.out.printf("#check\tcx_st_si_write\t%016x\t1%n", hash);
      measure("cx_st_si_read", () -> {
        for (Case sample : cases)
          sink = format.read(input, sample.name(), sample.id(), IOContext.DEFAULT);
        return cases.size();
      });
      measure("cx_st_si_write", () -> {
        for (Case sample : cases) {
          try (var output = new ByteBuffersDirectory()) {
            format.write(output, sample.info(), IOContext.DEFAULT);
            sink = output.fileLength(sample.name() + ".si");
          }
        }
        return cases.size();
      });
    }
  }
}
