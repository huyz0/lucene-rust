import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.lucene90.Lucene90CompoundFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.Version;

/** Actual cranky compound lifecycle, identical archive/input bytes to Rust. */
public class CrankyCompoundMicro {
  static final byte[] ID = "0123456789abcdef".getBytes(StandardCharsets.US_ASCII);
  static final String[] NAMES = {"_base.a", "_base.bb", "_base_Codec_0.cc"};
  static final byte[][] INPUTS = new byte[3][];
  static volatile long sink;
  static long hash(long digest, long value) { return (digest ^ value) * 0x100000001b3L; }
  static SegmentInfo info(Directory dir) {
    var info = new SegmentInfo(dir, Version.fromBits(10, 5, 0), null, "_base", 0,
        false, false, null, Map.of(), ID.clone(), Map.of(), null);
    info.setFiles(Set.of(NAMES));
    return info;
  }
  static void put(Directory dir, String name, byte[] bytes) throws IOException {
    try (var output = dir.createOutput(name, IOContext.DEFAULT)) { output.writeBytes(bytes, bytes.length); }
  }
  static long consume(long digest, Directory dir, String name) throws IOException {
    try (var input = dir.openInput(name, IOContext.DEFAULT)) {
      digest = hash(digest, input.length());
      while (input.getFilePointer() < input.length()) digest = hash(digest, input.readByte() & 255);
    }
    return digest;
  }
  static long batch(CompoundFormat format, Random random, Directory base, SegmentInfo baseInfo, boolean write) throws Exception {
    random.setSeed(42);
    long digest = 0xcbf29ce484222325L;
    try (var directory = new ByteBuffersDirectory()) {
      if (write) for (int i = 0; i < NAMES.length; i++) put(directory, NAMES[i], INPUTS[i]);
      var metadata = info(directory);
      for (int op = 0; op < 256; op++) {
        if (write) {
          boolean injected = false;
          try { format.write(directory, metadata, IOContext.DEFAULT); }
          catch (IOException e) {
            if (!e.getMessage().equals("Fake IOException from CompoundFormat.write()")) throw e;
            injected = true;
          }
          if (injected) {
            for (String name : directory.listAll())
              if (name.endsWith(".cfs") || name.endsWith(".cfe")) throw new AssertionError("injection created output");
            digest = hash(digest, -1L);
          } else {
            digest = hash(digest, metadata.files().size());
            for (String name : new String[] {"_base.cfs", "_base.cfe"}) {
              digest = consume(digest, directory, name);
              directory.deleteFile(name);
            }
          }
        } else {
          try (var reader = format.getCompoundReader(base, baseInfo)) {
            String[] names = reader.listAll();
            Arrays.sort(names);
            for (String name : names) {
              for (byte value : name.getBytes(StandardCharsets.UTF_8)) digest = hash(digest, value & 255);
              digest = consume(digest, reader, name);
            }
            reader.checkIntegrity();
          }
        }
      }
    }
    sink = digest;
    return digest;
  }
  public static void main(String[] args) throws Exception {
    long warmup = Long.parseLong(args[0]), measure = Long.parseLong(args[1]);
    Path fixtures = Path.of("fixtures/testframework/data");
    for (int i = 0; i < INPUTS.length; i++) INPUTS[i] = Files.readAllBytes(fixtures.resolve("cranky_compound_input_" + i + ".bin"));
    var constructor = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyCompoundFormat")
        .getDeclaredConstructor(CompoundFormat.class, Random.class);
    constructor.setAccessible(true);
    Random random = new Random(42);
    var delegate = new Lucene90CompoundFormat();
    var format = (CompoundFormat) constructor.newInstance(delegate, random);
    long ownedBytes = RamUsageEstimator.shallowSizeOf(format) + RamUsageEstimator.shallowSizeOf(delegate)
        + RamUsageEstimator.shallowSizeOf(random) + RamUsageEstimator.shallowSizeOf(new java.util.concurrent.atomic.AtomicLong());
    System.out.printf("cranky_compound_owned_bytes\t%d%n", ownedBytes);
    try (var base = new ByteBuffersDirectory()) {
      put(base, "_base.cfs", Files.readAllBytes(fixtures.resolve("cranky_compound.cfs")));
      put(base, "_base.cfe", Files.readAllBytes(fixtures.resolve("cranky_compound.cfe")));
      var baseInfo = info(base);
      for (boolean write : new boolean[] {false, true}) {
        String name = write ? "cranky_compound_write" : "cranky_compound_read";
        long expected = batch(format, random, base, baseInfo, write);
        for (int phase = 0; phase < 2; phase++) {
          long budget = phase == 0 ? warmup : measure;
          long start = System.nanoTime(), operations = 0;
          do {
            if (batch(format, random, base, baseInfo, write) != expected) throw new AssertionError("unstable digest");
            operations += 256;
          } while (System.nanoTime() - start < budget * 1000000);
          if (phase == 1) System.out.printf(java.util.Locale.ROOT, "%s\t%.3f\t%d%n", name, (double)(System.nanoTime() - start) / operations, operations);
        }
        System.out.printf("#check\t%s\t%s\t256%n", name, Long.toUnsignedString(expected));
      }
    }
  }
}
