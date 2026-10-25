import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.CompoundFormat;
import org.apache.lucene.codecs.lucene90.Lucene90CompoundFormat;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.Version;

/** Actual cranky compound wrapper with shared Random and real Lucene90 archives. */
public class GenCrankyCompound {
  static final byte[] ID = "0123456789abcdef".getBytes(StandardCharsets.US_ASCII);
  static final String[] IDS = {".a", ".bb", "_Codec_0.cc"};
  static final int[] SIZES = {0, 17, 131};
  static final long OFFSET = 0xcbf29ce484222325L;
  static long hash(long digest, int value) { return (digest ^ value) * 0x100000001b3L; }
  static CompoundFormat wrap(CompoundFormat delegate, Random random) throws Exception {
    var constructor = Class.forName("org.apache.lucene.tests.codecs.cranky.CrankyCompoundFormat")
        .getDeclaredConstructor(CompoundFormat.class, Random.class);
    constructor.setAccessible(true);
    return (CompoundFormat) constructor.newInstance(delegate, random);
  }
  static SegmentInfo info(Directory dir, String name, Set<String> files) {
    var info = new SegmentInfo(dir, Version.fromBits(10, 5, 0), null, name, 0,
        false, false, null, Map.of(), ID.clone(), Map.of(), null);
    info.setFiles(files);
    return info;
  }
  static SegmentInfo inputs(Directory dir, String name) throws IOException {
    var files = new java.util.HashSet<String>();
    for (int f = 0; f < IDS.length; f++) {
      String file = name + IDS[f];
      files.add(file);
      try (var output = dir.createOutput(file, IOContext.DEFAULT)) {
        CodecUtil.writeIndexHeader(output, "CrankyCompoundInput", 0, ID, "");
        for (int i = 0; i < SIZES[f]; i++) output.writeByte((byte)(i * 31 + f));
        CodecUtil.writeFooter(output);
      }
    }
    return info(dir, name, files);
  }
  static byte[] bytes(Directory dir, String file) throws IOException {
    try (var input = dir.openInput(file, IOContext.DEFAULT)) {
      byte[] bytes = new byte[Math.toIntExact(input.length())];
      input.readBytes(bytes, 0, bytes.length);
      return bytes;
    }
  }
  static void put(Directory dir, String file, byte[] bytes) throws IOException {
    try (var output = dir.createOutput(file, IOContext.DEFAULT)) { output.writeBytes(bytes, bytes.length); }
  }
  static long digest(CompoundDirectory reader) throws IOException {
    String[] names = reader.listAll();
    Arrays.sort(names);
    long digest = OFFSET;
    for (String name : names) {
      for (byte value : name.getBytes(StandardCharsets.UTF_8)) digest = hash(digest, value & 255);
      byte[] bytes = bytes(reader, name);
      digest = hash(digest, bytes.length);
      for (byte value : bytes) digest = hash(digest, value & 255);
    }
    reader.checkIntegrity();
    return digest;
  }
  static class State { int reads, writes; boolean failRead, failWrite; }
  static class Recording extends CompoundFormat {
    final State state;
    final CompoundFormat delegate = new Lucene90CompoundFormat();
    Recording(State state) { this.state = state; }
    @Override public CompoundDirectory getCompoundReader(Directory dir, SegmentInfo info) throws IOException {
      state.reads++;
      if (state.failRead) throw new IOException("delegate read failure");
      return delegate.getCompoundReader(dir, info);
    }
    @Override public void write(Directory dir, SegmentInfo info, IOContext context) throws IOException {
      state.writes++;
      if (state.failWrite) throw new IOException("delegate write failure");
      delegate.write(dir, info, context);
    }
  }
  static String sequence(long seed, boolean failures) throws Exception {
    Random random = new Random(seed);
    State state = new State();
    CompoundFormat[] formats = {wrap(new Recording(state), random), wrap(new Recording(state), random)};
    StringBuilder trace = new StringBuilder();
    try (var base = new ByteBuffersDirectory()) {
      SegmentInfo baseInfo = inputs(base, "_base");
      new Lucene90CompoundFormat().write(base, baseInfo, IOContext.DEFAULT);
      for (int op = 0; op < 256; op++) {
        boolean read = op % 3 == 0;
        state.failRead = failures && op % 21 == 0;
        state.failWrite = failures && op % 19 == 0;
        String result = "ok", digest = "-";
        boolean cfs = false, cfe = false, unchanged = true;
        try (var target = new ByteBuffersDirectory()) {
          SegmentInfo targetInfo = inputs(target, "_write");
          Set<String> original = Set.copyOf(targetInfo.files());
          try {
            if (read) {
              try (var reader = formats[op & 1].getCompoundReader(base, baseInfo)) {
                digest = Long.toUnsignedString(digest(reader));
              }
            } else formats[op & 1].write(target, targetInfo, IOContext.DEFAULT);
          } catch (IOException error) { result = error.getMessage(); }
          cfs = Arrays.asList(target.listAll()).contains("_write.cfs");
          cfe = Arrays.asList(target.listAll()).contains("_write.cfe");
          unchanged = targetInfo.files().equals(original);
        }
        trace.append("op\t").append(seed).append('\t').append(failures).append('\t').append(op)
            .append('\t').append(read ? "read" : "write").append('\t').append(result)
            .append('\t').append(state.reads).append('\t').append(state.writes)
            .append('\t').append(cfs).append('\t').append(cfe).append('\t').append(unchanged)
            .append('\t').append(digest).append('\n');
      }
    }
    trace.append("rng\t").append(seed).append('\t').append(failures).append('\t').append(random.nextInt()).append('\n');
    return trace.toString();
  }
  static String negative(String kind) throws Exception {
    try (var dir = new ByteBuffersDirectory()) {
      Random random = new Random(42);
      CompoundFormat format = wrap(new Lucene90CompoundFormat(), random);
      SegmentInfo info = inputs(dir, "_bad");
      if (kind.startsWith("reader")) {
        new Lucene90CompoundFormat().write(dir, info, IOContext.DEFAULT);
        String file = kind.equals("reader_bad_cfe") ? "_bad.cfe" : "_bad.cfs";
        byte[] damaged = bytes(dir, file);
        if (kind.equals("reader_bad_cfe")) damaged[0] ^= 1;
        else damaged = Arrays.copyOf(damaged, damaged.length - 1);
        dir.deleteFile(file);
        put(dir, file, damaged);
      } else if (kind.equals("writer_missing_input")) {
        dir.deleteFile("_bad.a");
      } else {
        byte[] damaged = bytes(dir, "_bad.a");
        if (kind.equals("writer_bad_footer")) damaged[damaged.length - 1] ^= 1;
        else damaged[4 + 1 + "CrankyCompoundInput".length() + 4] ^= 1;
        dir.deleteFile("_bad.a");
        put(dir, "_bad.a", damaged);
      }
      boolean failed = false;
      try {
        if (kind.startsWith("reader")) {
          try (var ignored = format.getCompoundReader(dir, info)) { }
        } else format.write(dir, info, IOContext.DEFAULT);
      } catch (IOException expected) { failed = true; }
      if (!failed) throw new AssertionError(kind);
      var names = Arrays.asList(dir.listAll());
      return "negative\t" + kind + "\t" + random.nextInt() + "\t"
          + names.contains("_bad.cfs") + "\t" + names.contains("_bad.cfe") + "\n";
    }
  }
  static String failureOrder(String kind) throws Exception {
    try (var base = new ByteBuffersDirectory()) {
      SegmentInfo info = inputs(base, "_bad");
      if (!kind.equals("valid")) {
        byte[] damaged = bytes(base, "_bad.a");
        int offset = kind.equals("checksum") ? damaged.length - 1
            : 4 + 1 + "CrankyCompoundInput".length() + 4;
        damaged[offset] ^= 1;
        base.deleteFile("_bad.a");
        put(base, "_bad.a", damaged);
      }
      var opened = new java.util.ArrayList<String>();
      int[] lengths = {0};
      var directory = new org.apache.lucene.store.FilterDirectory(base) {
        @Override public long fileLength(String name) throws IOException {
          lengths[0]++;
          return super.fileLength(name);
        }
        @Override public org.apache.lucene.store.IndexInput openInput(String name, IOContext context) throws IOException {
          opened.add(name);
          if (name.equals("_bad.bb")) throw new IOException("later open");
          return super.openInput(name, context);
        }
      };
      Random random = new Random(42);
      String result;
      try {
        wrap(new Lucene90CompoundFormat(), random).write(directory, info, IOContext.DEFAULT);
        throw new AssertionError("expected combined failure");
      } catch (org.apache.lucene.index.CorruptIndexException expected) { result = "corruption"; }
      catch (IOException expected) {
        if (!expected.getMessage().equals("later open")) throw expected;
        result = "later open";
      }
      if (!result.equals(kind.equals("valid") ? "later open" : "corruption"))
        throw new AssertionError("wrong precedence: " + kind);
      return kind + "\t" + result + "\t" + lengths[0] + "\t"
          + String.join(",", opened) + "\t" + random.nextInt() + "\n";
    }
  }
  public static void main(String[] args) throws Exception {
    Path out = Path.of(args[0]);
    Files.createDirectories(out);
    try (var dir = new ByteBuffersDirectory()) {
      SegmentInfo info = inputs(dir, "_base");
      wrap(new Lucene90CompoundFormat(), new Random(42)).write(dir, info, IOContext.DEFAULT);
      try (var reader = wrap(new Lucene90CompoundFormat(), new Random(42)).getCompoundReader(dir, info)) {
        digest(reader);
        for (int f = 0; f < IDS.length; f++)
          Files.write(out.resolve("cranky_compound_input_" + f + ".bin"), bytes(reader, "_base" + IDS[f]));
      }
      Files.write(out.resolve("cranky_compound.cfs"), bytes(dir, "_base.cfs"));
      Files.write(out.resolve("cranky_compound.cfe"), bytes(dir, "_base.cfe"));
    }
    try (var dir = new ByteBuffersDirectory()) {
      SegmentInfo info = info(dir, "_empty", Set.of());
      wrap(new Lucene90CompoundFormat(), new Random(42)).write(dir, info, IOContext.DEFAULT);
      try (var reader = wrap(new Lucene90CompoundFormat(), new Random(42)).getCompoundReader(dir, info)) {
        if (reader.listAll().length != 0) throw new AssertionError("empty archive");
        reader.checkIntegrity();
      }
      Files.write(out.resolve("cranky_compound_empty.cfs"), bytes(dir, "_empty.cfs"));
      Files.write(out.resolve("cranky_compound_empty.cfe"), bytes(dir, "_empty.cfe"));
    }
    StringBuilder trace = new StringBuilder();
    for (long seed : new long[] {0, 1, 42, -1, Long.MIN_VALUE, Long.MAX_VALUE})
      for (boolean failures : new boolean[] {false, true}) trace.append(sequence(seed, failures));
    for (String kind : new String[] {"reader_bad_cfe", "reader_short_cfs", "writer_missing_input", "writer_bad_footer", "writer_bad_id"})
      trace.append(negative(kind));
Files.writeString(out.resolve("cranky_compound.tsv"), trace, StandardCharsets.UTF_8);
    StringBuilder order = new StringBuilder();
    for (String kind : new String[] {"checksum", "id", "valid"}) order.append(failureOrder(kind));
    Files.writeString(out.resolve("cranky_compound_failure_order.tsv"), order, StandardCharsets.UTF_8);
  }
}
